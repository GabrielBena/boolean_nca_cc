"""
Training loop implementation for GNN-based boolean circuit optimization.

This module provides functions for training GNN models to optimize
boolean circuits over multiple epochs.
"""

import jax
import jax.numpy as jp
import optax
from flax import nnx
import jraph
from typing import List, Tuple, Dict, Any, Optional
from functools import partial
from tqdm.auto import tqdm
import os
import logging
from datetime import datetime


from boolean_nca_cc.circuits.model import gen_circuit
from boolean_nca_cc.models import CircuitSelfAttention
from boolean_nca_cc.training.checkpointing import (
    save_checkpoint,
    setup_checkpoint_dir,
    save_periodic_checkpoint,
    save_best_checkpoint,
    save_stable_state,
    get_metric_value,
    check_early_stopping,
)

from boolean_nca_cc.training.schedulers import (
    should_reset_pool,
    get_learning_rate_schedule,
)

from boolean_nca_cc.training.pool.pool import GraphPool, initialize_graph_pool
from boolean_nca_cc.training.evaluation import (
    evaluate_model_stepwise_batched,
    get_loss_and_update_graph, evaluate_circuits_in_chunks,
    get_loss_from_wires_logits,
)
from boolean_nca_cc.training.preconfigure import preconfigure_circuit_logits
# Removed unused knockout dataset imports since we now use vocabulary-based evaluation
import wandb

from boolean_nca_cc.utils.graph_builder import build_graph
from boolean_nca_cc.training.pool.structural_perturbation import    (
    create_reproducible_knockout_pattern, 
    create_knockout_vocabulary,
    create_strip_knockout_pattern,
    create_group_greedy_pattern,
)
from functools import partial
from boolean_nca_cc.circuits.train import create_gate_mask_from_knockout_pattern
from boolean_nca_cc.circuits.data_split import split_input_combinations
from boolean_nca_cc.analysis.hamming_utils import (
    _hard_truth_tables_from_logits,
    _active_gate_mask_from_knockout,
    _hamming_distance_tables,
)
from boolean_nca_cc.analysis.visualization import plot_combined_bp_sa_stepwise_performance

# Optional imports for visualization - imported lazily when needed
# These are only used conditionally and may not be available in all environments
import json
import os
import pandas as pd
import numpy as np

# Type alias for PyTree
PyTree = Any

# Setup logging
log = logging.getLogger(__name__)


def _init_wandb(wandb_logging: bool, wandb_run_config: dict | None = None) -> Any | None:
    """Initialize wandb if enabled and return the run object."""
    if not wandb_logging:
        return None

    try:
        import wandb

        if not wandb.run:
            # Only initialize wandb if not already initialized
            wandb.init(
                config=wandb_run_config,
                resume="allow",
            )

        # Define metrics for sweeps and summaries
        try:
            # Attach step metrics so panels align and summaries compute properly
            wandb.define_metric("training/*", step_metric="training/epoch")
            wandb.define_metric("eval_ko_in/*", step_metric="eval_ko_in/epoch")
            wandb.define_metric("eval_ko_out/*", step_metric="eval_ko_out/epoch")
            wandb.define_metric("eval_no_damage/*", step_metric="eval_no_damage/epoch")
            # eval_no_damage_steps/* intentionally has no metric definition (like eval_ko_in_steps/*)
            # This ensures WandB uses its default step counter, giving each log call its own x-axis position
            # Note: Raw metrics are no longer logged to WandB - they are saved locally instead
            # See boolean_nca_cc.utils.metrics_storage for local metrics storage
            wandb.define_metric("model/*", step_metric="training/epoch")
            wandb.define_metric("scheduler/*", step_metric="training/epoch")

            # Make the run summary of this metric be its max over the run
            # so sweeps can optimize it even if it's logged periodically
            wandb.define_metric("eval_ko_in/final_hard_accuracy", summary="max")
        except Exception as e:
            log.warning(f"Error defining wandb metrics: {e}")

        # Get the unique run ID for checkpointing
        log.info(f"WandB run ID: {wandb.run.id}")
        return wandb
    except ImportError:
        log.warning("wandb not installed. Running without wandb logging.")
        return None
    except Exception as e:
        log.warning(f"Error initializing wandb: {e}. Running without wandb logging.")
        return None



def _log_to_wandb(
    wandb_run, metrics_dict: Dict, epoch: int, log_interval: int = 1
) -> None:
    """Log metrics to wandb if enabled and interval allows."""
    if wandb_run is None or epoch % log_interval != 0:
        return

    try:
        wandb_run.log(metrics_dict)
    except Exception as e:
        log.warning(f"Error logging to wandb: {e}")



def _log_final_wandb_metrics(wandb_run, results: Dict, epochs: int) -> None:
    """Log final metrics and plots to wandb."""
    if wandb_run is None:
        return

    try:
        # Log final metrics
        wandb_run.log(
            {
                "final/loss": results["losses"][-1],
                "final/hard_loss": results["hard_losses"][-1],
                "final/accuracy": results["accuracies"][-1],
                "final/hard_accuracy": results["hard_accuracies"][-1],
                "final/epoch": epochs,
                f"best/{results.get('best_metric', 'metric')}": results.get(
                    "best_metric_value", 0
                ),
            }
        )

    except Exception as e:
        log.warning(f"Error logging final metrics to wandb: {e}")



def run_knockout_periodic_evaluation(
    model,
    knockout_vocabulary: Optional[jp.ndarray],
    base_wires: PyTree,
    base_logits: PyTree,
    knockout_config: Dict,
    periodic_eval_test_seed: int,
    x_data,
    y_data,
    input_n,
    arity,
    circuit_hidden_dim,
    n_message_steps,
    loss_type,
    epoch,
    wandb_run,
    eval_batch_size: int,
    accumulated_pattern_data: List,  # Add this parameter
    training_mode: str = "growth",  # Add training mode parameter
    log_stepwise=False,
    layer_sizes: Optional[List[Tuple[int, int]]] = None,
    use_scan: bool = False,
    knockout_diversity: Optional[int] = None,  # Add diversity parameter for color coding
    hamming_analysis_dir: Optional[str] = None,  # Directory for hamming analysis plots
    bp_hamming_summary: Optional[List[Dict]] = None,
    layer_neighbors: bool = False,
    metric_suffix: str = "",  # Suffix for metric names (e.g., "_train" for train data eval)
    blind_mode: bool = False,  # If True, zero out loss feedback (ablation study)
    input_split_enabled: bool = False,  # Whether input split is enabled (for input_split metadata)
) -> Tuple[Dict, List]:  # Return both results and updated accumulated data
    """
    Run periodic evaluation on circuits with persistent knockouts using vocabulary-based sampling.
    
    Args:
        model: The trained model to evaluate
        knockout_vocabulary: Optional vocabulary of knockout patterns
        base_wires: Base circuit wires
        base_logits: Base circuit logits
        knockout_config: Configuration for knockout evaluation
        periodic_eval_test_seed: Seed for evaluation
        x_data: Input data
        y_data: Target data
        input_n: Number of inputs
        arity: Circuit arity
        circuit_hidden_dim: Hidden dimension for circuit
        n_message_steps: Number of message passing steps
        loss_type: Type of loss function
        epoch: Current training epoch
        wandb_run: WandB run object for logging
        eval_batch_size: Batch size for evaluation
        accumulated_pattern_data: List to accumulate pattern performance data
        training_mode: Training mode ("growth" or "repair") for mode-aware visualization
        log_stepwise: Whether to log step-by-step metrics
        layer_sizes: Layer sizes for the circuit
        use_scan: Whether to use scan for message passing
        knockout_diversity: Diversity parameter for color coding scatter plots
        hamming_analysis_dir: Directory for hamming analysis plots
        bp_hamming_summary: Backpropagation hamming summary for comparison
        
    Returns:
        Tuple of (evaluation results, updated accumulated pattern data)
    """
    try:
        
        # Use the layer_sizes parameter that's already passed to the function
        # This is in the correct format (total_gates, group_size) from generate_layer_sizes
        true_layer_sizes = layer_sizes if layer_sizes is not None else []
        
        # 1. Sample IN-distribution knockout patterns from vocabulary
        if knockout_vocabulary is not None:
            log.info(f"Running IN-distribution Knockout evaluation using vocabulary ({eval_batch_size} patterns)...")
            
            # Sample patterns from vocabulary with replacement
            id_rng = jax.random.PRNGKey(periodic_eval_test_seed)
            pattern_indices = jax.random.choice(
                id_rng, len(knockout_vocabulary), shape=(eval_batch_size,), replace=True
            )
            in_knockout_patterns = knockout_vocabulary[pattern_indices]
        else:
            log.info(f"Running IN-distribution Knockout evaluation with fresh patterns ({eval_batch_size} patterns)...")
            
            # Generate patterns using same logic as vocabulary but with eval seed
            pattern_creator_fn = partial(
                create_reproducible_knockout_pattern,
                layer_sizes=true_layer_sizes,
                damage_prob=knockout_config["damage_prob"],
            )
            
            id_rng = jax.random.PRNGKey(periodic_eval_test_seed)
            in_pattern_keys = jax.random.split(id_rng, eval_batch_size)
            in_knockout_patterns = jax.vmap(pattern_creator_fn)(in_pattern_keys)
        
        # Replicate base circuit for the batch
        in_wires = jax.tree.map(
            lambda x: jp.repeat(x[None, ...], eval_batch_size, axis=0), base_wires
        )
        in_logits = jax.tree.map(
            lambda x: jp.repeat(x[None, ...], eval_batch_size, axis=0), base_logits
        )
        
        step_metrics_in = evaluate_circuits_in_chunks(
            eval_fn=evaluate_model_stepwise_batched,
            wires=in_wires,
            logits=in_logits,
            knockout_patterns=in_knockout_patterns,
            target_chunk_size=eval_batch_size,
            model=model,
            x_data=x_data,
            y_data=y_data,
            input_n=input_n,
            arity=arity,
            circuit_hidden_dim=circuit_hidden_dim,
            n_message_steps=n_message_steps,
            loss_type=loss_type,
            layer_sizes=layer_sizes,
            return_per_pattern=True,  # Enable per-pattern analysis
            layer_neighbors=layer_neighbors,
            # use_scan=use_scan,
            # Unified damage control system
            damage_mode=knockout_config.get("damage_mode", "greedy"),
            damage_injection_mode=knockout_config.get("damage_injection_mode", "multi"),
            max_damage_per_circuit=int(knockout_config.get("max_damage_per_circuit", 10)),
            greedy_ordered_indices=knockout_config.get("greedy_ordered_indices", None),
            greedy_window_size=int(knockout_config.get("greedy_window_size", 1)),
            greedy_injection_recover_steps=int(knockout_config.get("greedy_injection_recover_steps", 10)),
            # Damage start offset parameters
            damage_start_offset=int(knockout_config.get("damage_start_offset", 0)),
            damage_start_offset_random=knockout_config.get("damage_start_offset_random", False),
            damage_start_offset_seed=int(knockout_config.get("damage_start_offset_seed", 42)),
            # Vocabulary-based evaluation parameters
            knockout_vocabulary=knockout_vocabulary,  # If provided => seen; else => unseen
            blind_mode=blind_mode,  # Ablation study: disable loss feedback
        )

        final_metrics_in = {
            f"eval_ko_in{metric_suffix}/final_loss": step_metrics_in["soft_loss"][-1],
            f"eval_ko_in{metric_suffix}/final_hard_loss": step_metrics_in["hard_loss"][-1],
            f"eval_ko_in{metric_suffix}/final_accuracy": step_metrics_in["soft_accuracy"][-1],
            f"eval_ko_in{metric_suffix}/final_hard_accuracy": step_metrics_in["hard_accuracy"][-1],
            f"eval_ko_in{metric_suffix}/final_full_map_accuracy": step_metrics_in["full_map_accuracy"][-1],
            f"eval_ko_in{metric_suffix}/epoch": epoch,
        }
        
        # Calculate and add standard deviation for eval_ko_in/hard_accuracy
        if "per_pattern" in step_metrics_in:
            final_hard_accuracies_in = step_metrics_in["per_pattern"]["pattern_hard_accuracies"][-1]
            hard_acc_std = float(jp.std(final_hard_accuracies_in))
            final_metrics_in[f"eval_ko_in{metric_suffix}/final_hard_accuracy_std"] = hard_acc_std

        
        # Replicate base circuit for the batch
        out_wires = jax.tree.map(
            lambda x: jp.repeat(x[None, ...], eval_batch_size, axis=0), base_wires
        )
        out_logits = jax.tree.map(
            lambda x: jp.repeat(x[None, ...], eval_batch_size, axis=0), base_logits
        )
        
        step_metrics_out = evaluate_circuits_in_chunks(
            eval_fn=evaluate_model_stepwise_batched,
            wires=out_wires,
            logits=out_logits,
            knockout_patterns=None,  # No pre-generated patterns for OUT evaluation
            target_chunk_size=eval_batch_size,
            model=model,
            x_data=x_data,
            y_data=y_data,
            input_n=input_n,
            arity=arity,
            circuit_hidden_dim=circuit_hidden_dim,
            n_message_steps=n_message_steps,
            loss_type=loss_type,
            layer_sizes=layer_sizes,
            return_per_pattern=True,  # Enable per-pattern analysis
            layer_neighbors=layer_neighbors,
            # Unified damage control for OUT (unseen) evaluation
            damage_mode=knockout_config.get("damage_mode", "greedy"),
            damage_injection_mode=knockout_config.get("damage_injection_mode", "multi"),
            max_damage_per_circuit=int(knockout_config.get("max_damage_per_circuit", 10)),
            greedy_ordered_indices=knockout_config.get("greedy_ordered_indices", None),
            greedy_window_size=int(knockout_config.get("greedy_window_size", 1)),
            greedy_injection_recover_steps=int(knockout_config.get("greedy_injection_recover_steps", 10)),
            # Damage start offset parameters
            damage_start_offset=int(knockout_config.get("damage_start_offset", 0)),
            damage_start_offset_random=knockout_config.get("damage_start_offset_random", False),
            damage_start_offset_seed=int(knockout_config.get("damage_start_offset_seed", 42)),
            # For OUT, force unseen by not providing a vocabulary
            knockout_vocabulary=None,  # Force unseen patterns for OUT evaluation
            blind_mode=blind_mode,  # Ablation study: disable loss feedback
        )

        final_metrics_out = {
            f"eval_ko_out{metric_suffix}/final_loss": step_metrics_out["soft_loss"][-1],
            f"eval_ko_out{metric_suffix}/final_hard_loss": step_metrics_out["hard_loss"][-1],
            f"eval_ko_out{metric_suffix}/final_accuracy": step_metrics_out["soft_accuracy"][-1],
            f"eval_ko_out{metric_suffix}/final_hard_accuracy": step_metrics_out["hard_accuracy"][-1],
            f"eval_ko_out{metric_suffix}/final_full_map_accuracy": step_metrics_out["full_map_accuracy"][-1],
            f"eval_ko_out{metric_suffix}/epoch": epoch,
        }
        
        # Calculate and add standard deviation for eval_ko_out/hard_accuracy
        if "per_pattern" in step_metrics_out:
            final_hard_accuracies_out = step_metrics_out["per_pattern"]["pattern_hard_accuracies"][-1]
            hard_acc_std = float(jp.std(final_hard_accuracies_out))
            final_metrics_out[f"eval_ko_out{metric_suffix}/final_hard_accuracy_std"] = hard_acc_std

            
                    # Add new pattern data to accumulated data for persistent scatter plot
        if "per_pattern" in step_metrics_in:
            final_hard_accuracies_in = step_metrics_in["per_pattern"]["pattern_hard_accuracies"][-1]
            for pattern_idx, hard_acc in enumerate(final_hard_accuracies_in):
                # Include knockout_diversity in data point for color coding
                diversity_value = knockout_diversity if knockout_diversity is not None else 0
                data_point = [int(epoch), pattern_idx, float(hard_acc), diversity_value]
                accumulated_pattern_data.append(data_point)
        
        # Log persistent scatter plot with all accumulated data
        if accumulated_pattern_data:
            
            pattern_table = wandb_run.Table(
                data=accumulated_pattern_data,
                columns=["epoch", "pattern_id", "hard_accuracy", "knockout_diversity"]
            )
            
            # Create scatter plot with all accumulated points
            # Note: Color coding by diversity will be handled by wandb's automatic grouping
            # based on the knockout_diversity column in the table
            scatter_plot = wandb_run.plot.scatter(
                pattern_table, 
                "epoch", 
                "hard_accuracy",
                title="In-Distribution Pattern Performance by Epoch (Colored by Knockout Diversity)"
            )
            
            wandb_run.log({"pattern_performance_scatter": scatter_plot})
        
        # Perform hamming distance analysis and create plots if directory is provided
        if hamming_analysis_dir and "per_pattern" in step_metrics_in:
            try:
                # Extract per-pattern logits and metrics for GNN
                final_logits_per_pattern = step_metrics_in["per_pattern"]["pattern_logits"][-1]
                final_hard_accuracies_in = step_metrics_in["per_pattern"]["pattern_hard_accuracies"][-1]
                
                # Analyze hamming distances for GNN
                gnn_hamming_summary = _analyze_knockout_hamming_distances(
                    base_logits=base_logits,
                    knockout_patterns=in_knockout_patterns,
                    final_logits_per_pattern=final_logits_per_pattern,
                    layer_sizes=layer_sizes,
                    final_hard_accuracies=final_hard_accuracies_in,
                )
                for row in gnn_hamming_summary:
                    row['method'] = 'gnn'

                # Combine with BP results if available for joint plotting
                if bp_hamming_summary:
                    combined_summary_data = gnn_hamming_summary + bp_hamming_summary
                else:
                    combined_summary_data = gnn_hamming_summary
                
                # Create and save accuracy vs distance plot
                if combined_summary_data:
                    # Ensure hamming analysis directory exists
                    os.makedirs(hamming_analysis_dir, exist_ok=True)
                    
                    df = pd.DataFrame(combined_summary_data)
                    
                    plot_filename = f"accuracy_vs_distance_epoch_{epoch:04d}.png"
                    plot_path = os.path.join(hamming_analysis_dir, plot_filename)
                    
                    # Import visualization function locally (may not be available in all environments)
                    try:
                        # Try direct import first (works when project root is in path)
                        from experiments.visualization.plot_accuracy_vs_distance import plot_accuracy_vs_distance  # type: ignore
                        
                        plot_accuracy_vs_distance(
                            summary_df=df,
                            output_path=plot_path,
                            color_by_method=True
                        )
                    except ImportError:
                        # Fallback: add project root to path if direct import fails
                        try:
                            import sys
                            import os as os_module
                            project_root = os_module.path.dirname(os_module.path.dirname(os_module.path.dirname(__file__)))
                            if project_root not in sys.path:
                                sys.path.insert(0, project_root)
                            from experiments.visualization.plot_accuracy_vs_distance import plot_accuracy_vs_distance  # type: ignore
                            
                            plot_accuracy_vs_distance(
                                summary_df=df,
                                output_path=plot_path,
                                color_by_method=True
                            )
                        except (ImportError, Exception) as e:
                            log.warning(f"Could not generate plot: {e}. Skipping plot generation.")
                            plot_path = None
                    except Exception as e:
                        log.warning(f"Error generating plot: {e}. Skipping plot generation.")
                        plot_path = None
                
                    if plot_path:
                        log.info(f"Hamming distance analysis plot saved to: {plot_path}")
                    
                    # Log plot to wandb if enabled
                    if wandb_run and plot_path:
                        try:
                            import wandb
                            
                            # Calculate mean and std hamming distance from GNN results
                            if gnn_hamming_summary:
                                hamming_values = [row['per_gate_mean_hamming'] for row in gnn_hamming_summary]
                                gnn_mean_hamming = np.mean(hamming_values)
                                gnn_std_hamming = np.std(hamming_values)
                                log.info(f"Mean hamming distance (epoch {epoch}): {gnn_mean_hamming:.4f} ± {gnn_std_hamming:.4f}")
                                
                                wandb_run.log({
                                    "hamming_analysis/joint_plot": wandb.Image(plot_path),
                                    "hamming_analysis/epoch": epoch,
                                    "hamming_analysis/mean_hamming_distance": gnn_mean_hamming,
                                    "hamming_analysis/std_hamming_distance": gnn_std_hamming,
                                })
                            else:
                                log.warning(f"No GNN hamming summary data available for epoch {epoch}")
                                wandb_run.log({
                                    "hamming_analysis/joint_plot": wandb.Image(plot_path),
                                    "hamming_analysis/epoch": epoch,
                                })
                        except Exception as e:
                            log.warning(f"Error logging hamming analysis plot to wandb: {e}")
                
                # Save summary data to CSV for this epoch
                if combined_summary_data:
                    df = pd.DataFrame(combined_summary_data)
                    csv_filename = f"hamming_analysis_epoch_{epoch:04d}.csv"
                    csv_path = os.path.join(hamming_analysis_dir, csv_filename)
                    df.to_csv(csv_path, index=False)
                    log.info(f"Hamming distance analysis CSV saved to: {csv_path}")
                    
            except Exception as e:
                log.warning(f"Error during hamming distance analysis at epoch {epoch}: {e}")

            if log_stepwise:
                for step_idx in range(len(step_metrics_in["step"])):
                    log_dict = {
                        f"eval_ko_in{metric_suffix}_steps/step": step_metrics_in["step"][step_idx],
                        f"eval_ko_in{metric_suffix}_steps/loss": step_metrics_in["soft_loss"][step_idx],
                        f"eval_ko_in{metric_suffix}_steps/hard_loss": step_metrics_in["hard_loss"][step_idx],
                        f"eval_ko_in{metric_suffix}_steps/accuracy": step_metrics_in["soft_accuracy"][step_idx],
                        f"eval_ko_in{metric_suffix}_steps/hard_accuracy": step_metrics_in["hard_accuracy"][step_idx],
                        f"eval_ko_in{metric_suffix}_steps/full_map_accuracy": step_metrics_in["full_map_accuracy"][step_idx],
                        f"eval_ko_in{metric_suffix}_steps/epoch": epoch,
                    }
                    
                    # Calculate and add standard deviation for eval_ko_in/hard_accuracy at this step
                    if "per_pattern" in step_metrics_in:
                        if step_idx < len(step_metrics_in["per_pattern"]["pattern_hard_accuracies"]):
                            step_hard_accuracies_in = step_metrics_in["per_pattern"]["pattern_hard_accuracies"][step_idx]
                            hard_acc_std = float(jp.std(step_hard_accuracies_in))
                            log_dict[f"eval_ko_in{metric_suffix}_steps/hard_accuracy_std"] = hard_acc_std
                    
                    # DEBUG: Add damage validation metrics if available
                    if "damaged_node_avg_change" in step_metrics_in and step_idx < len(step_metrics_in["damaged_node_avg_change"]):
                        log_dict[f"eval_ko_in{metric_suffix}_steps/damaged_node_avg_change"] = step_metrics_in["damaged_node_avg_change"][step_idx]
                    if "healthy_node_avg_change" in step_metrics_in and step_idx < len(step_metrics_in["healthy_node_avg_change"]):
                        log_dict[f"eval_ko_in{metric_suffix}_steps/healthy_node_avg_change"] = step_metrics_in["healthy_node_avg_change"][step_idx]
                    wandb_run.log(log_dict)
                for step_idx in range(len(step_metrics_out["step"])):
                    log_dict = {
                        f"eval_ko_out{metric_suffix}_steps/step": step_metrics_out["step"][step_idx],
                        f"eval_ko_out{metric_suffix}_steps/loss": step_metrics_out["soft_loss"][step_idx],
                        f"eval_ko_out{metric_suffix}_steps/hard_loss": step_metrics_out["hard_loss"][step_idx],
                        f"eval_ko_out{metric_suffix}_steps/accuracy": step_metrics_out["soft_accuracy"][step_idx],
                        f"eval_ko_out{metric_suffix}_steps/hard_accuracy": step_metrics_out["hard_accuracy"][step_idx],
                        f"eval_ko_out{metric_suffix}_steps/full_map_accuracy": step_metrics_out["full_map_accuracy"][step_idx],
                        f"eval_ko_out{metric_suffix}_steps/epoch": epoch,
                    }
                    
                    # Calculate and add standard deviation for eval_ko_out/hard_accuracy at this step
                    if "per_pattern" in step_metrics_out:
                        if step_idx < len(step_metrics_out["per_pattern"]["pattern_hard_accuracies"]):
                            step_hard_accuracies_out = step_metrics_out["per_pattern"]["pattern_hard_accuracies"][step_idx]
                            hard_acc_std = float(jp.std(step_hard_accuracies_out))
                            log_dict[f"eval_ko_out{metric_suffix}_steps/hard_accuracy_std"] = hard_acc_std
                    
                    # DEBUG: Add damage validation metrics if available
                    if "damaged_node_avg_change" in step_metrics_out and step_idx < len(step_metrics_out["damaged_node_avg_change"]):
                        log_dict[f"eval_ko_out{metric_suffix}_steps/damaged_node_avg_change"] = step_metrics_out["damaged_node_avg_change"][step_idx]
                    if "healthy_node_avg_change" in step_metrics_out and step_idx < len(step_metrics_out["healthy_node_avg_change"]):
                        log_dict[f"eval_ko_out{metric_suffix}_steps/healthy_node_avg_change"] = step_metrics_out["healthy_node_avg_change"][step_idx]
                    wandb_run.log(log_dict)

        suffix_label = f" ({metric_suffix.strip('_')})" if metric_suffix else ""
        log.info(
            f"Knockout Eval{suffix_label} (epoch {epoch}):\n"
            f"  IN-distribution KO: Loss={final_metrics_in[f'eval_ko_in{metric_suffix}/final_loss']:.4f}, "
            f"Acc={final_metrics_in[f'eval_ko_in{metric_suffix}/final_accuracy']:.4f}, "
            f"Hard Acc={final_metrics_in[f'eval_ko_in{metric_suffix}/final_hard_accuracy']:.4f}\n"
            f"  OUT-of-distribution KO: Loss={final_metrics_out[f'eval_ko_out{metric_suffix}/final_loss']:.4f}, "
            f"Acc={final_metrics_out[f'eval_ko_out{metric_suffix}/final_accuracy']:.4f}, "
            f"Hard Acc={final_metrics_out[f'eval_ko_out{metric_suffix}/final_hard_accuracy']:.4f}"
        )

        # Save metrics locally with metadata
        try:
            from boolean_nca_cc.utils.metrics_storage import save_eval_metrics_locally, save_stepwise_metrics_locally
            import time
            
            # Get run ID and sweep_id
            sweep_id = None
            if wandb_run and hasattr(wandb_run, 'run') and wandb_run.run:
                run_id = wandb_run.run.id
                if hasattr(wandb_run.run, 'sweep_id') and wandb_run.run.sweep_id:
                    sweep_id = str(wandb_run.run.sweep_id)
            else:
                run_id = f"local_{int(time.time())}"
            
            # Extract task and other config from wandb_run if available
            task = None
            if wandb_run:
                try:
                    if hasattr(wandb_run, 'config') and wandb_run.config:
                        config = wandb_run.config if isinstance(wandb_run.config, dict) else dict(wandb_run.config)
                        circuit_config = config.get("circuit", {})
                        if isinstance(circuit_config, dict):
                            task = circuit_config.get("task")
                except Exception:
                    pass  # Fallback to None if config access fails
            
            # Get damage metadata from knockout_config
            damage_prob = knockout_config.get("damage_prob", None)
            damage_mode = knockout_config.get("damage_mode", None)
            damage_injection_mode = knockout_config.get("damage_injection_mode", None)
            
            # Get damage_type from model (defaults to "permanent" if not set)
            damage_type = getattr(model, "damage_behavior", "permanent")
            
            # Determine split (test vs train) from metric_suffix
            split = "test" if metric_suffix == "" else "train"
            
            # Determine input_split: when input_split_enabled, test=unseen, train=seen
            input_split_val = None
            if input_split_enabled:
                input_split_val = "unseen" if metric_suffix == "" else "seen"
            
            # Save IN-distribution (seen damage patterns) metrics
            save_eval_metrics_locally(
                metrics=final_metrics_in,
                epoch=epoch,
                split=split,
                run_id=run_id,
                sweep_id=sweep_id,
                metrics_dir="results/metrics",
                format="jsonl",
                task=task,
                training_mode=training_mode,
                damage_prob=damage_prob,
                damage_type=damage_type,
                damage_mode=damage_mode,
                damage_injection_mode=damage_injection_mode,
                                    eval_type="ko_in",  # Seen damage patterns
                                    input_split=input_split_val,
            )
            
            # Save OUT-of-distribution (unseen damage patterns) metrics
            save_eval_metrics_locally(
                metrics=final_metrics_out,
                epoch=epoch,
                split=split,
                run_id=run_id,
                sweep_id=sweep_id,
                metrics_dir="results/metrics",
                format="jsonl",
                task=task,
                training_mode=training_mode,
                damage_prob=damage_prob,
                damage_type=damage_type,
                damage_mode=damage_mode,
                damage_injection_mode=damage_injection_mode,
                eval_type="ko_out",  # Unseen damage patterns
                input_split=input_split_val,
            )
            
            # Save stepwise metrics if enabled
            if log_stepwise and step_metrics_in and step_metrics_out:
                # Save stepwise metrics for IN-distribution
                save_stepwise_metrics_locally(
                    step_metrics=step_metrics_in,
                    epoch=epoch,
                    split=split,
                    run_id=run_id,
                    sweep_id=sweep_id,
                    metrics_dir="results/metrics",
                    format="jsonl",
                    task=task,
                    training_mode=training_mode,
                    damage_prob=damage_prob,
                    damage_type=damage_type,
                    damage_mode=damage_mode,
                    damage_injection_mode=damage_injection_mode,
                    eval_type="ko_in",
                    input_split=input_split_val,
                )
                
                # Save stepwise metrics for OUT-of-distribution
                save_stepwise_metrics_locally(
                    step_metrics=step_metrics_out,
                    epoch=epoch,
                    split=split,
                    run_id=run_id,
                    sweep_id=sweep_id,
                    metrics_dir="results/metrics",
                    format="jsonl",
                    task=task,
                    training_mode=training_mode,
                    damage_prob=damage_prob,
                    damage_type=damage_type,
                    damage_mode=damage_mode,
                    damage_injection_mode=damage_injection_mode,
                    eval_type="ko_out",
                    input_split=input_split_val,
                )
        except Exception as e:
            log.warning(f"Failed to save knockout metrics locally: {e}")

        return {
            "final_metrics_in": final_metrics_in,
            "final_metrics_out": final_metrics_out,
        }, accumulated_pattern_data  # Return updated accumulated data

    except Exception as e:
        log.warning(f"Error during knockout periodic evaluation at epoch {epoch}: {e}")
        return {}, accumulated_pattern_data





def _analyze_knockout_hamming_distances(
    base_logits: PyTree,
    knockout_patterns: jp.ndarray,
    final_logits_per_pattern: List[PyTree],
    layer_sizes: List[Tuple[int, int]],
    final_hard_accuracies: jp.ndarray,
) -> List[Dict]:
    """
    Analyze hamming distances between baseline and knockout patterns.
    
    Args:
        base_logits: Baseline circuit logits
        knockout_patterns: Array of knockout patterns
        final_logits_per_pattern: List of final logits for each pattern (batched PyTree)
        layer_sizes: Circuit layer sizes
        final_hard_accuracies: Final hard accuracies for each pattern
    
    Returns:
        List of dictionaries with analysis results
    """
    # Convert baseline logits to hard truth tables
    baseline_tables = _hard_truth_tables_from_logits(base_logits)
    
    # Unbatch the per-pattern logits from a batched PyTree to a list of PyTrees
    batch_size = final_hard_accuracies.shape[0]
    final_logits_unbatched = [
        jax.tree.map(lambda x: x[i], final_logits_per_pattern)
        for i in range(batch_size)
    ]
    
    summary_data = []
    
    for idx, (pattern, final_logits, hard_acc) in enumerate(zip(
        knockout_patterns, final_logits_unbatched, final_hard_accuracies
    )):
        # Convert final logits to hard truth tables
        pert_tables = _hard_truth_tables_from_logits(final_logits)
        
        # Get active gate masks for this pattern
        active_masks = _active_gate_mask_from_knockout(layer_sizes, pattern)
        
        # Calculate hamming distances
        metrics = _hamming_distance_tables(baseline_tables, pert_tables, active_masks)
        
        # Create summary row
        row = {
            "pattern_idx": int(idx),
            "overall_bitwise_fraction_diff": metrics["overall_bitwise_fraction_diff"],
            "per_gate_mean_hamming": metrics["per_gate_mean_hamming"],
            "counted_bits_total": metrics["counted_bits_total"],
            "counted_gates_total": metrics["counted_gates_total"],
            "final_hard_accuracy": float(hard_acc),  # Using consistent naming convention
            "per_layer_bitwise_fraction_diff": json.dumps(metrics["per_layer_bitwise_fraction_diff"])
        }
        
        summary_data.append(row)
    
    return summary_data

def train_model(
    # Data parameters
    x_data: jp.ndarray,
    y_data: jp.ndarray,
    layer_sizes: List[Tuple[int, int]],
    # Model architecture parameters
    arity: int = 2,
    circuit_hidden_dim: int = 16,
    # node_mlp_features: List[int] = [64, 32],
    # edge_mlp_features: List[int] = [64, 32],
    use_attention: bool = False,
    # Training hyperparameters
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    epochs: int = 100,
    n_message_steps: int = 1,
    layer_neighbors: bool = False,
    use_scan: bool = False,
    # Loss parameters
    loss_type: str = "l4",  # Options: 'l4' or 'bce'
    meta_batch_size: int = 64,
    # Pool parameters
    pool_size: int = 1024,
    reset_pool_fraction: float = 0.05,
    reset_pool_interval: int = 128,
    reset_strategy: str = "uniform",  # Options: "uniform", "steps_biased", "loss_biased", or "combined"
    combined_weights: Tuple[float, float] = (
        0.5,
        0.5,
    ),  # Weights for [loss, steps] in combined strategy
    # Damage (sample-based knockout) parameters
    damage_pool_enabled: bool = False,
    damage_pool_interval: int = 0,
    damage_pool_fraction: float = 0.0,
    damage_strategy: str = "uniform",
    damage_combined_weights: Tuple[float, float] = (0.5, 0.5),
    damage_mode: str = "shotgun",  # Options: "shotgun", "strip", "greedy", or "greedy_vocabulary"
    # Perturbation configurations
    persistent_knockout_config: Optional[Dict] = None,
    knockout_diversity: Optional[int] = None,  # Size of knockout pattern vocabulary for shared training/evaluation patterns
    # Damage-specific generation control (decoupled from legacy persistent_knockout)
    damage_pool_damage_prob: float = 0.0,
    greedy_ordered_indices: Optional[List[int]] = None,
    greedy_window_size: int = 1,
    damage_eval_steps: int = 50,  # Number of message passing steps for damage evaluation
    # Unified damage control parameters
    damage_injection_mode: str = "multi",  # "single" (one damage per circuit) or "multi" (multiple damages)
    max_damage_per_circuit: int = 10,  # Maximum damage events per circuit
    damage_seed: int = 481,  # Independent seed for damage pattern generation
    knockout_eval: Optional[Dict] = None,
    # Learning rate scheduling
    lr_scheduler: str = "constant",  # Options: "constant", "exponential", "cosine", "linear_warmup"
    lr_scheduler_params: Optional[Dict] = None,
    # Initialization parameters
    key: int = 0,
    wiring_fixed_key: jax.random.PRNGKey = jax.random.PRNGKey(
        42
    ),  # Fixed key for generating wirings when wiring_mode='fixed'
    init_model: Optional[CircuitSelfAttention] = None,
    init_optimizer: Optional[nnx.Optimizer] = None,
    initial_metrics: Optional[Dict] = None,
    # Checkpointing parameters
    checkpoint_enabled: bool = False,
    checkpoint_dir: str | None = None,
    checkpoint_interval: int = 1024,
    save_best: bool = True,
    best_metric: str = "final_hard_accuracy",
    best_metric_source: str = "eval",
    save_stable_states: bool = True,
    # Periodic evaluation parameters
    periodic_eval_inner_steps: int = 100,
    periodic_eval_interval: int = 1024,
    periodic_eval_test_seed: int = 42,
    periodic_eval_log_stepwise: bool = False,
    periodic_eval_batch_size: int = 16,  # Batch size for random wiring evaluation
    # Input combination train/test split parameters
    input_split_enabled: bool = False,  # Enable train/test split on input combinations
    input_train_fraction: float = 0.8,  # Fraction of combinations for training (0.8 = 80% train, 20% test)
    input_split_seed: int = 42,  # Seed for reproducible split
    # Wandb parameters
    wandb_logging: bool = False,
    log_interval: int = 1,
    wandb_run_config: Optional[Dict] = None,
    # Repair mode preconfig params
    training_mode: str = "growth",
    preconfig_steps: int = 200,
    preconfig_lr: float = 1e-2,
    # Early stopping parameters
    stop_accuracy_enabled: bool = False,
    stop_accuracy_threshold: float = 0.95,
    stop_accuracy_metric: str = "final_hard_accuracy",
    stop_accuracy_source: str = "eval_ko_in",
    stop_accuracy_patience: int = 10,
    stop_accuracy_min_epochs: int = 100,
    # Hamming distance analysis parameters
    hamming_analysis_dir: Optional[str] = None,
    backprop_config: Optional[Dict] = None,
    # DEBUG Param: Permanent damage validation parameters
    track_pool_damage_validation: bool = False,  # Enable validation of permanent damage in pool
):
    """
    Train a GNN to optimize boolean circuit parameters.

    Args:
        layer_sizes: List of tuples (nodes, group_size) for each layer
        x_data: Input data for training [batch, input_bits]
        y_data: Target output data [batch, output_bits]
        arity: Number of inputs per gate
        circuit_hidden_dim: Dimension of hidden features
        # node_mlp_features: Hidden layer sizes for the node MLP
        # edge_mlp_features: Hidden layer sizes for the edge MLP
        # use_attention: Whether to use attention-based message aggregation
        learning_rate: Learning rate for optimization
        epochs: Number of training epochs
        n_message_steps: Number of message passing steps per pool batch
        loss_type: Type of loss to use ('l4' for L4 norm or 'bce' for binary cross-entropy)
        meta_batch_size: Batch size for training
        pool_size: Size of the graph pool
        reset_pool_fraction: Fraction of pool to reset periodically
        reset_pool_interval: Number of epochs between pool resets
        reset_strategy: Strategy for selecting graphs to reset ("uniform", "steps_biased", "loss_biased", or "combined")
        combined_weights: Tuple of weights (loss_weight, steps_weight) for combining factors in "combined" strategy
        damage_pool_enabled: Whether to enable sample-based damage to pool entries.
        damage_pool_interval: Number of epochs between damage applications.
        damage_pool_fraction: Fraction of pool to damage when damage is applied.
        damage_strategy: Strategy for selecting circuits to damage.
        damage_combined_weights: Weights for combined damage selection strategy.
        damage_mode: Type of damage pattern ("shotgun" for random, "strip" for localized, 
            "greedy" for rolling window, "greedy_vocabulary" for vocabulary sampling).
        persistent_knockout_config: Configuration for persistent knockout perturbations.
        knockout_diversity: Size of knockout pattern vocabulary for shared training/evaluation patterns
        damage_pool_damage_prob: Damage probability per node when generating fresh patterns.
        damage_eval_steps: Number of message passing steps for damage evaluation (baseline and recovery).
        damage_injection_mode: "single" for one damage per circuit, "multi" for multiple damages.
        max_damage_per_circuit: Maximum damage events per circuit (unified control).
        knockout_eval: Configuration for knockout evaluation during training.
        key: Random seed
        wiring_fixed_key: Fixed key for generating wirings when wiring_mode='fixed'
        init_model: Optional pre-trained GNN model to continue training
        init_optimizer: Optional pre-trained optimizer to continue training
        initial_metrics: Optional dictionary of metrics from previous training
        lr_scheduler: Learning rate scheduler type
        lr_scheduler_params: Dictionary of parameters for the scheduler
        periodic_eval_inner_steps: Number of inner steps for periodic evaluation
        periodic_eval_interval: Interval for periodic evaluation
        periodic_eval_test_seed: Seed for periodic evaluation test circuit generation
        periodic_eval_log_stepwise: Whether to log step-by-step evaluation metrics
        periodic_eval_batch_size: Batch size for random wiring evaluation
        input_split_enabled: Whether to enable train/test split on input combinations
        input_train_fraction: Fraction of input combinations for training (default 0.8 = 80% train, 20% test)
        input_split_seed: Seed for reproducible input combination split
        use_scan: Whether to use scan for message passing
        wandb_logging: Whether to log metrics to wandb
        log_interval: Interval for logging metrics
        wandb_run_config: Configuration to pass to wandb
        training_mode: Training mode ("growth" or "repair")
        preconfig_steps: Number of preconfiguration steps for repair mode
        preconfig_lr: Learning rate for preconfiguration
        checkpoint_enabled: Whether to enable checkpointing
        checkpoint_dir: Directory for checkpoints
        checkpoint_interval: Interval for periodic checkpoints
        save_best: Whether to save best model
        best_metric: Metric to track for best model
        best_metric_source: Source for best metric
        save_stable_states: Whether to save stable states
        stop_accuracy_enabled: Whether to enable early stopping
        stop_accuracy_threshold: Accuracy threshold for early stopping
        stop_accuracy_metric: Metric to monitor for early stopping
        stop_accuracy_source: Source for early stopping metric
        stop_accuracy_patience: Patience for early stopping
        stop_accuracy_min_epochs: Minimum epochs before early stopping
        hamming_analysis_dir: Directory for hamming analysis plots and CSVs
        backprop_config: Configuration for one-time backprop evaluation
        track_pool_damage_validation: DEBUG - If True, enable validation of permanent damage in pool.
                                      Validates that damaged nodes remain at expected values
                                      (logits=-10.0, hidden=0.0) across pool updates and resets.
    """
       # Validate unified damage control parameters
    if damage_injection_mode not in ["single", "multi"]:
        raise ValueError(f"damage_injection_mode must be 'single' or 'multi', got '{damage_injection_mode}'")
    # Automatically set max_damage_per_circuit=1 when mode is 'single'
    if damage_injection_mode == "single":
        max_damage_per_circuit = 1
    if max_damage_per_circuit < 1:
        raise ValueError(f"max_damage_per_circuit must be >= 1, got {max_damage_per_circuit}")

    # Initialize random key
    rng = jax.random.PRNGKey(key)

    # Get dimension from layer sizes
    input_n = layer_sizes[0][0]

    # Initialize metrics storage
    if initial_metrics is None:
        # Start with empty lists
        losses = []
        accuracies = []
        hard_losses = []
        hard_accuracies = []
        reset_steps = []
    else:
        # Continue from previous metrics
        losses = list(initial_metrics.get("losses", []))
        accuracies = list(initial_metrics.get("accuracies", []))
        hard_losses = list(initial_metrics.get("hard_losses", []))
        hard_accuracies = list(initial_metrics.get("hard_accuracies", []))
        reset_steps = list(initial_metrics.get("reset_steps", []))

    # Initialize or reuse GNN
    if init_model is None:
        raise ValueError("init_model must be provided")
    else:
        # Use the provided GNN
        model = init_model

    # Create optimizer or reuse existing optimizer
    if init_optimizer is None:
        # Create the learning rate schedule using our scheduler module
        schedule = get_learning_rate_schedule(
            lr_scheduler, learning_rate, epochs, lr_scheduler_params
        )

        # Create a new optimizer with the schedule
        opt_fn = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.zero_nans(),
            optax.adamw(learning_rate=schedule, weight_decay=weight_decay),
        )
        # As of Flax 0.11.0, nnx.Optimizer requires wrt argument
        # Since nnx.value_and_grad optimizes all parameters by default, use Everything()
        optimizer = nnx.Optimizer(model, opt_fn, wrt=nnx.Everything())
    else:
        # Use the provided optimizer
        optimizer = init_optimizer
        schedule = None

    # Split input combinations early if enabled
    if input_split_enabled:
        log.info(f"Splitting input combinations: {input_train_fraction*100:.0f}% train, {(1-input_train_fraction)*100:.0f}% test (seed={input_split_seed})")
        x_train, y_train, x_test, y_test = split_input_combinations(
            x_data=x_data,
            y_data=y_data,
            train_fraction=input_train_fraction,
            seed=input_split_seed,
            shuffle=True,
        )
        log.info(f"Train set: {x_train.shape[0]} combinations, Test set: {x_test.shape[0]} combinations")
    else:
        # When split is disabled, use all data for both training and "test" (backward compatible)
        x_train, y_train = x_data, y_data
        x_test, y_test = x_data, y_data
        log.debug("Input split disabled - using all combinations for training and evaluation")

    # Initialize Graph Pool for training
    # Use consistent key generation: wiring_fixed_key for fixed/genetic modes, dynamic for random
    training_pool_key = wiring_fixed_key

    # Setup wandb logging if enabled (moved earlier to avoid UnboundLocalError)
    wandb_run = _init_wandb(wandb_logging, wandb_run_config)
    wandb_id = wandb_run.run.id if wandb_run else None

    # Preconfigure if in repair mode
    base_wires_preconfig = None
    base_logits_preconfig = None
    if training_mode == "repair":
        log.info("Repair mode: running preconfiguration of fixed wiring circuit")
        # Pull optimizer hyperparameters from backprop_config if provided to match test_preconfigure
        pre_opt = backprop_config.get("optimizer", "adam") if backprop_config else "adam"
        pre_wd = backprop_config.get("weight_decay", 0.0) if backprop_config else 0.0
        pre_b1 = backprop_config.get("beta1", 0.9) if backprop_config else 0.9
        pre_b2 = backprop_config.get("beta2", 0.999) if backprop_config else 0.999
        
        # DEBUG: Print full debug info matching GUI and test script format
        wiring_seed = int(wiring_fixed_key[1]) if len(wiring_fixed_key) >= 2 else int(wiring_fixed_key[0])
        log.info(f"Preconfiguring circuit with params (matching train_loop.py):")
        log.info(f"  wiring_seed={wiring_seed} (from wiring_fixed_key)")
        log.info(f"  wiring_key={wiring_fixed_key}")  # DEBUG: Show actual key value
        log.info(f"  layer_sizes={layer_sizes}")
        log.info(f"  arity={arity}, loss_type={loss_type}")
        log.info(f"  steps={preconfig_steps}, lr={preconfig_lr}, optimizer={pre_opt}")
        log.info(f"  weight_decay={pre_wd}, beta1={pre_b1}, beta2={pre_b2}")
        log.info(f"  x_data shape={x_data.shape}, y_data shape={y_data.shape}")  # DEBUG: Show data shapes
        
        base_wires_preconfig, base_logits_preconfig = preconfigure_circuit_logits(
            wiring_key=wiring_fixed_key,
            layer_sizes=layer_sizes,
            arity=arity,
            x_data=x_data,
            y_data=y_data,
            loss_type=loss_type,
            steps=preconfig_steps,
            lr=preconfig_lr,
            optimizer=pre_opt,
            weight_decay=pre_wd,
            beta1=pre_b1,
            beta2=pre_b2,
        )
        
        # DEBUG: Log logits statistics after preconfiguration
        logits_stats = []
        for i, logit_layer in enumerate(base_logits_preconfig):
            logit_np = np.array(logit_layer) if hasattr(logit_layer, '__array__') else logit_layer
            logits_stats.append(f"layer_{i}: mean={logit_np.mean():.6f}, std={logit_np.std():.6f}, min={logit_np.min():.6f}, max={logit_np.max():.6f}")
        log.info(f"  Post-preconfig logits stats: {'; '.join(logits_stats)}")
        
        # Log preconfigured circuit metrics
        preconfig_loss, preconfig_aux = get_loss_from_wires_logits(
            base_logits_preconfig, base_wires_preconfig, x_data, y_data, loss_type
        )
        preconfig_hard_loss, _, _, preconfig_accuracy, preconfig_hard_accuracy, _, _, _ = preconfig_aux
        log.info(
            f"Preconfigured circuit metrics: "
            f"loss={float(preconfig_loss):.6f}, "
            f"hard_loss={float(preconfig_hard_loss):.4f}, "
            f"accuracy={float(preconfig_accuracy):.4f}, "
            f"hard_accuracy={float(preconfig_hard_accuracy):.4f}"
        )
        
        if wandb_logging and wandb_run:
            wandb_run.log({
                "preconfig/steps": preconfig_steps,
            })

    circuit_pool = initialize_graph_pool(
        rng=training_pool_key,
        layer_sizes=layer_sizes,
        pool_size=pool_size,
        input_n=input_n,
        arity=arity,
        circuit_hidden_dim=circuit_hidden_dim,
        loss_value=0.0,  # Initial loss will be calculated properly in first step
        base_wires=base_wires_preconfig,
        base_logits=base_logits_preconfig,
    )

    # Define pool-based training step
    @partial(
        nnx.jit,
        static_argnames=(
            "layer_sizes",
            "n_message_steps",
            "loss_type",
        ),
    )

    def pool_train_step(
        model: CircuitSelfAttention,
        optimizer: nnx.Optimizer,
        pool: GraphPool,
        idxs: jp.ndarray,
        graphs: jraph.GraphsTuple,
        wires: PyTree,
        logits: PyTree,
        x: jp.ndarray,
        y_target: jp.ndarray,
        layer_sizes: Tuple[Tuple[int, int], ...],
        n_message_steps: int,
        loss_type: str,
        loss_key: jax.random.PRNGKey,
        epoch: int,
        knockout_patterns: Optional[jp.ndarray] = None,
    ):
        """
        Single training step using graphs from the pool.

        Args:
            model: CircuitGNN model
            optimizer: nnx Optimizer
            pool: GraphPool containing all circuits
            idxs: Indices of sampled graphs in the pool
            graphs: Batch of graphs from the pool
            wires: Corresponding wires for the graphs
            logits: Corresponding logits for the graphs
            x: Input data
            y_target: Target output data
            layer_sizes: Tuple of (nodes, group_size) tuples for each layer
            n_message_steps: Number of message passing steps
            loss_type: Type of loss function to use
            loss_key: Random key for loss computation
            knockout_patterns: Optional knockout pattern for structural perturbations.
        Returns:
            Tuple of (loss, auxiliary outputs, updated pool)
        """

        def get_loss_step(loss_key):
            return n_message_steps - 1

        def loss_fn_no_scan(model, graph, logits, wires, loss_key, knockout_pattern):
            # Store original shapes for reconstruction
            logits_original_shapes = [logit.shape for logit in logits]
            loss_step = get_loss_step(loss_key)

            all_results = []

            for i in range(n_message_steps):
                graph = model(
                    graph,
                    knockout_pattern=knockout_pattern,
                    layer_neighbors=layer_neighbors,
                    layer_sizes=layer_sizes,
                )

                graph, loss, logits, aux = get_loss_and_update_graph(
                    graph=graph,
                    logits_original_shapes=logits_original_shapes,
                    wires=wires,
                    x_data=x,
                    y_data=y_target,
                    loss_type=loss_type,
                    layer_sizes=layer_sizes,
                )
                # Update graph globals with current update steps
                current_update_steps = (
                    graph.globals[..., 1] if graph.globals is not None else 0
                )
                graph = graph._replace(
                    globals=jp.array([loss, current_update_steps + 1], dtype=jp.float32)
                )
                all_results.append((loss, aux, graph, logits))

            # Stack all results using jax.tree_map
            stacked_results = jax.tree.map(lambda *args: jp.stack(args), *all_results)

            # Index at n_loss_step
            final_loss, final_aux, final_graph, final_logits = jax.tree.map(
                lambda x: x[loss_step], stacked_results
            )

            return final_loss, (final_aux, final_graph, final_logits, loss_step)

        def batch_loss_fn(model, graphs, logits, wires, loss_key, knockout_patterns):
            loss_fn = loss_fn_no_scan # TODO: add scan as an option

            loss_keys = jax.random.split(loss_key, graphs.n_node.shape[0])
            
            loss, (aux, updated_graphs, updated_logits, loss_steps) = nnx.vmap(
                loss_fn, in_axes=(None, 0, 0, 0, 0, 0)
            )(model, graphs, logits, wires, loss_keys, knockout_patterns)
            return jp.mean(loss), (
                jax.tree.map(lambda x: jp.mean(x, axis=0), aux),
                updated_graphs,
                updated_logits,
                jp.mean(loss_steps),
            )

        # Compute loss and gradients
        (loss, (aux, updated_graphs, updated_logits, loss_steps)), grads = (
            nnx.value_and_grad(batch_loss_fn, has_aux=True)(
                model=model,
                graphs=graphs,
                logits=logits,
                wires=wires,
                loss_key=loss_key,
                knockout_patterns=knockout_patterns,
            )
        )

        # Update GNN parameters
        # As of Flax 0.11.0, update() requires both model and grads arguments
        optimizer.update(model, grads)

        # Update pool with the updated graphs and logits (wires stay the same)
        updated_pool = pool.update(idxs, updated_graphs, batch_of_logits=updated_logits)

        return loss, (aux, updated_pool, loss_steps)


    # Setup checkpointing directory
    checkpoint_path = setup_checkpoint_dir(checkpoint_dir, wandb_id)

    # Track best model
    best_metric_value = float("-inf") if "accuracy" in best_metric else float("inf")

    # Early stopping variables
    early_stop_triggered = False
    epochs_above_threshold = 0
    first_threshold_epoch = None

    # Create progress bar for training
    pbar = tqdm(range(epochs), desc="Training GNN")
    avg_steps_reset = 0

    # Track last reset epoch for scheduling
    last_reset_epoch = -1  # Initialize to -1 so first check works correctly

    # Initialize knockout evaluation base circuit (needed for both knockout eval and no-damage eval)
    knockout_eval_base_circuit = None
    if knockout_eval and knockout_eval.get("enabled"):
        # Store base circuit for on-demand evaluation (no pre-created datasets)
        log.info("Knockout evaluation enabled - will use vocabulary-based evaluation")
        if training_mode == "repair" and base_wires_preconfig is not None:
            knockout_eval_base_circuit = (base_wires_preconfig, base_logits_preconfig)
        else:
            knockout_eval_base_circuit = gen_circuit(wiring_fixed_key, layer_sizes, arity=arity)
        log.info("Base circuit created for knockout evaluation")
    elif periodic_eval_inner_steps > 0:
        # Even if knockout eval is disabled, we need base circuit for no-damage evaluation
        log.info("Knockout evaluation disabled - will use no-damage evaluation")
        if training_mode == "repair" and base_wires_preconfig is not None:
            knockout_eval_base_circuit = (base_wires_preconfig, base_logits_preconfig)
        else:
            knockout_eval_base_circuit = gen_circuit(wiring_fixed_key, layer_sizes, arity=arity)
        log.info("Base circuit created for no-damage evaluation")

    # Initialize knockout vocabulary if knockout_diversity is configured
    knockout_vocabulary = None
    if knockout_diversity is not None and knockout_diversity > 0:
        
        # Use dedicated damage seed for knockout pattern generation
        vocab_rng = jax.random.PRNGKey(damage_seed)
        
        # Use the layer_sizes parameter directly - it's already in the correct format
        # from generate_layer_sizes: (total_gates, group_size)
        true_layer_sizes = layer_sizes
        
        # Generate the shared knockout vocabulary
        knockout_vocabulary = create_knockout_vocabulary(
            rng=vocab_rng,
            vocabulary_size=knockout_diversity,
            layer_sizes=true_layer_sizes,
            damage_prob=damage_pool_damage_prob,
            damage_mode=damage_mode,
            ordered_indices=greedy_ordered_indices,
        )
        
    else:
        log.warning(f"VOCAB CREATION DEBUG: VOCABULARY NOT CREATED - knockout_diversity={knockout_diversity}")

    # Save initial stable state if needed
    last_stable_state = {
        "model": model,
        "optimizer": optimizer,
        "pool": circuit_pool,
        "metrics": {
            "losses": losses,
            "hard_losses": hard_losses,
            "accuracies": accuracies,
            "hard_accuracies": hard_accuracies,
            "reset_steps": reset_steps,
        },
        "epoch": 0,
    }

    # Initialize accumulated pattern data for persistent scatter plot
    # Each data point will be: [epoch, pattern_id, hard_accuracy, knockout_diversity]
    accumulated_pattern_data = []
    bp_hamming_summary = None
    bp_results = None

    # # Counter for event-driven damage evaluations (for WandB filtering)
    # damage_event_id = 0
    result = {}
    # Training loop
    try:
        for epoch in pbar:
            # Pool-based training
            # Sample a batch from the pool using the current (potentially dynamic) batch size
            rng, sample_key, loss_key = jax.random.split(rng, 3)
            idxs, graphs, wires, logits, knockout_patterns = circuit_pool.sample(
                sample_key, meta_batch_size
            )

            # Perform pool training step
            (
                loss,
                (aux, circuit_pool, loss_steps),
            ) = pool_train_step(
                model,
                optimizer,
                circuit_pool,
                idxs,
                graphs,
                wires,
                logits,
                x_train,
                y_train,
                tuple(layer_sizes),  # Convert list to tuple for JAX static arguments
                n_message_steps,
                loss_type=loss_type,
                loss_key=loss_key,
                epoch=epoch,
                knockout_patterns=knockout_patterns,
            )

            hard_loss, _, _, accuracy, hard_accuracy, full_map_accuracy, _, _ = aux

            # DEBUG BLOCK: PERMANENT DAMAGE VALIDATION (Vectorized)
            # Validate permanent damage after pool update (if enabled)
            if track_pool_damage_validation:
                damage_behavior = getattr(model, "damage_behavior", "permanent")
                validation_result = circuit_pool.validate_permanent_damage(
                    damage_behavior=damage_behavior,
                    tolerance=1e-6,
                )
                skipped_pending = validation_result.get("skipped_pending", 0)
                if validation_result["total_checks"] > 0:
                    if validation_result["num_failures"] > 0:
                        log.warning(
                            f"POOL DAMAGE VALIDATION FAILED (epoch {epoch}): "
                            f"{validation_result['passed_checks']}/{validation_result['total_checks']} checks passed "
                            f"({100*validation_result['pass_rate']:.1f}%), "
                            f"{skipped_pending} circuits with pending damage skipped"
                        )
                        for f in validation_result["failures"][:3]:  # Show first 3
                            log.warning(
                                f"  Circuit {f['circuit_idx']}: logits={f['logits_range']}, "
                                f"hidden={f['hidden_range']}, n_damaged={f['n_damaged_nodes']}"
                            )
                    elif epoch % 100 == 0:  # Log success periodically
                        log.info(
                            f"POOL DAMAGE VALIDATION PASSED (epoch {epoch}): "
                            f"{validation_result['passed_checks']}/{validation_result['total_checks']} checks passed, "
                            f"{skipped_pending} circuits with pending damage skipped"
                        )
                    
                    # Log to wandb if enabled
                    if wandb_run and epoch % log_interval == 0:
                        wandb_run.log({
                            "pool_damage_validation/total_checks": validation_result["total_checks"],
                            "pool_damage_validation/passed_checks": validation_result["passed_checks"],
                            "pool_damage_validation/pass_rate": validation_result["pass_rate"],
                            "pool_damage_validation/num_failures": validation_result["num_failures"],
                            "pool_damage_validation/skipped_pending": skipped_pending,
                            "training/epoch": epoch,
                        })

            # Reset a fraction of the pool using scheduled intervals
            if should_reset_pool(epoch, reset_pool_interval, last_reset_epoch):
                rng, reset_key, fresh_key = jax.random.split(rng, 3)

                # Original logic for fixed and random wiring modes
                # Generate fresh circuits for resetting, potentially with knockouts
                num_to_reset = max(1, round(pool_size * reset_pool_fraction))

                # Use consistent key generation for pool resets
                reset_pool_key = wiring_fixed_key

                # Create a pool of fresh circuits, applying knockout patterns
                # In repair mode, reset with preconfigured base; otherwise default
                reset_pool = initialize_graph_pool(
                    rng=reset_pool_key,
                    layer_sizes=layer_sizes,
                    pool_size=num_to_reset,  # Only create circuits we need
                    input_n=input_n,
                    arity=arity,
                    circuit_hidden_dim=circuit_hidden_dim,
                    knockout_config=None, # resets no longer introduce damage
                    knockout_patterns=None,
                    base_wires=base_wires_preconfig if training_mode == "repair" else None,
                    base_logits=base_logits_preconfig if training_mode == "repair" else None,
                )

                # Reset a fraction of the pool and get avg steps of reset graphs
                # Pass reset_pool.knockout_patterns (zeros) to clear accumulated damage on reset
                circuit_pool, avg_steps_reset, reset_idxs = circuit_pool.reset_fraction(
                    reset_key,
                    reset_pool_fraction,
                    reset_pool.graphs,
                    reset_pool.wires,
                    reset_pool.logits,
                    reset_pool.knockout_patterns,  # Clear damage patterns on reset
                    reset_strategy=reset_strategy,
                    combined_weights=combined_weights,
                )

                # DEBUG BLOCK: RESET DAMAGE CLEARING VALIDATION
                # Validate that reset cleared damage patterns (if enabled)
                if track_pool_damage_validation:
                    reset_validation = circuit_pool.validate_reset_cleared_damage(
                        reset_indices=reset_idxs
                    )
                    if reset_validation["total_checks"] > 0:
                        if reset_validation["num_failures"] > 0:
                            log.warning(
                                f"RESET DAMAGE CLEARING VALIDATION FAILED (epoch {epoch}): "
                                f"{reset_validation['passed_checks']}/{reset_validation['total_checks']} checks passed"
                            )
                            for f in reset_validation["failures"][:3]:  # Show first 3
                                log.warning(
                                    f"  Circuit {f['circuit_idx']}: still has {f['n_damaged_nodes']} damaged nodes"
                                )
                        else:
                            log.info(
                                f"RESET DAMAGE CLEARING VALIDATION PASSED (epoch {epoch}): "
                                f"All {reset_validation['total_checks']} reset circuits have cleared damage patterns"
                            )
                        
                        # Log to wandb if enabled
                        if wandb_run:
                            wandb_run.log({
                                "reset_validation/total_checks": reset_validation["total_checks"],
                                "reset_validation/passed_checks": reset_validation["passed_checks"],
                                "reset_validation/pass_rate": reset_validation["pass_rate"],
                                "reset_validation/num_failures": reset_validation["num_failures"],
                                "training/epoch": epoch,
                            })

                # Update last reset epoch
                last_reset_epoch = epoch
                # diversity = circuit_pool.get_wiring_diversity(layer_sizes)
            
            # Apply sample-based damage to a fraction of the pool (independent of resets)
            if (
                damage_pool_enabled
                and damage_pool_interval > 0
                and damage_pool_fraction > 0.0
                and (damage_pool_damage_prob > 0.0 or knockout_vocabulary is not None)
                and (epoch % damage_pool_interval == 0)
            ):
                rng, damage_key = jax.random.split(rng)

                # Select which circuits to damage using unified damage control
                damaged_idxs, _ = circuit_pool.get_reset_indices(
                    damage_key,
                    fraction=damage_pool_fraction,
                    reset_strategy=damage_strategy,
                    combined_weights=damage_combined_weights,
                    invert_loss=True,
                    # Unified damage control parameters
                    damage_injection_mode=damage_injection_mode,
                    max_damage_per_circuit=max_damage_per_circuit,
                )

                # Build damage patterns based on mode: greedy rolling-window, vocabulary sampling, or random
                new_patterns = None
                if (greedy_ordered_indices is not None and len(greedy_ordered_indices) > 0 
                    and damage_mode == "greedy"):
                    # Legacy rolling window mode: deterministic pattern progression
                    n = len(greedy_ordered_indices)
                    window = max(1, int(greedy_window_size))

                    def build_pattern(idx):
                        count = circuit_pool.perturb_counter[idx]
                        start = (count * window) % n
                        return create_group_greedy_pattern(
                            greedy_ordered_indices, tuple(layer_sizes), start, window
                        )

                    new_patterns = jax.vmap(build_pattern, in_axes=(0,))(damaged_idxs)
                elif knockout_vocabulary is not None:
                    # Sample vocabulary patterns for damaged indices
                    vocab_size = knockout_vocabulary.shape[0]
                    vocab_indices = jax.random.choice(
                        damage_key, vocab_size, shape=(damaged_idxs.shape[0],), replace=True
                    )
                    new_patterns = knockout_vocabulary[vocab_indices]
                else:
                    # Fresh random patterns as last resort
                    log.warning(f"TRAINING DAMAGE DEBUG: FALLBACK to RANDOM SHOTGUN patterns! (knockout_vocabulary=None, damage_mode={damage_mode})")
                    vmapped_pattern_creator = jax.vmap(
                        lambda k: create_reproducible_knockout_pattern(
                            key=k,
                            layer_sizes=tuple(layer_sizes),
                            damage_prob=float(damage_pool_damage_prob),
                        )
                    )
                    pattern_keys = jax.random.split(damage_key, damaged_idxs.shape[0])
                    new_patterns = vmapped_pattern_creator(pattern_keys)

                # Apply into pool (increments perturb_counter internally)
                # Determine accumulation behavior based on model's damage mode
                damage_behavior = getattr(model, "damage_behavior", "permanent")
                should_accumulate = (damage_behavior == "permanent")
                
                # DEBUG BLOCK: CUMULATIVE DAMAGE VALIDATION
                # Capture old patterns BEFORE application (for cumulative damage validation)
                old_patterns = None
                if track_pool_damage_validation and damage_behavior == "permanent" and circuit_pool.knockout_patterns is not None:
                    old_patterns = circuit_pool.knockout_patterns[damaged_idxs].copy()
                
                circuit_pool = circuit_pool.apply_knockouts(
                    damaged_idxs, 
                    new_patterns, 
                    accumulate=should_accumulate
                )

                # DEBUG BLOCK: CUMULATIVE DAMAGE VALIDATION
                # Validate cumulative damage (if enabled and we have old patterns)
                if track_pool_damage_validation and damage_behavior == "permanent" and old_patterns is not None:
                    cumulative_validation = circuit_pool.validate_cumulative_damage(
                        damaged_indices=damaged_idxs,
                        new_patterns=new_patterns,
                        old_patterns=old_patterns,
                        damage_behavior=damage_behavior,
                    )
                    if cumulative_validation["total_checks"] > 0:
                        if cumulative_validation["num_failures"] > 0:
                            log.warning(
                                f"CUMULATIVE DAMAGE VALIDATION FAILED (epoch {epoch}): "
                                f"{cumulative_validation['passed_checks']}/{cumulative_validation['total_checks']} checks passed "
                                f"({100*cumulative_validation['pass_rate']:.1f}%)"
                            )
                            log.warning(
                                "  This indicates damage patterns are NOT being accumulated (OR'd) correctly!"
                            )
                            for f in cumulative_validation["failures"][:3]:  # Show first 3
                                log.warning(
                                    f"  Circuit {f['circuit_idx']}: "
                                    f"old={f['old_damage_count']}, new={f['new_damage_count']}, "
                                    f"current={f['current_damage_count']}, expected={f['expected_damage_count']}, "
                                    f"previously_damaged_remain={f['previously_damaged_remain']}, "
                                    f"new_damage_added={f['new_damage_added']}, "
                                    f"patterns_match={f['patterns_match']}"
                                )
                        else:
                            log.info(
                                f"CUMULATIVE DAMAGE VALIDATION PASSED (epoch {epoch}): "
                                f"All {cumulative_validation['total_checks']} circuits with previous damage "
                                f"correctly accumulated new damage (OR operation)"
                            )
                        
                        # Log to wandb if enabled
                        if wandb_run:
                            wandb_run.log({
                                "cumulative_damage_validation/total_checks": cumulative_validation["total_checks"],
                                "cumulative_damage_validation/passed_checks": cumulative_validation["passed_checks"],
                                "cumulative_damage_validation/pass_rate": cumulative_validation["pass_rate"],
                                "cumulative_damage_validation/num_failures": cumulative_validation["num_failures"],
                                "training/epoch": epoch,
                            })

                # DEBUG BLOCK: DAMAGE APPLICATION VALIDATION
                # Validate damage application (if enabled)
                if track_pool_damage_validation and damage_behavior == "permanent":
                    # Check that damaged nodes have expected values immediately after application
                    # Note: This validates the pool state, but the actual graph features will be
                    # updated on the next training step when the model is applied
                    validation_result = circuit_pool.validate_permanent_damage(
                        damage_behavior=damage_behavior,
                        tolerance=1e-6,
                    )
                    if validation_result["total_checks"] > 0 and validation_result["num_failures"] > 0:
                        log.warning(
                            f"DAMAGE APPLICATION VALIDATION FAILED (epoch {epoch}): "
                            f"{validation_result['passed_checks']}/{validation_result['total_checks']} checks passed "
                            f"({100*validation_result['pass_rate']:.1f}%)"
                        )
                        log.warning(
                            "  Note: Graph features will be updated on next training step. "
                            "This validates pool state after pattern application."
                        )

                damaged_count = int(damaged_idxs.shape[0]) if hasattr(damaged_idxs, "shape") else 0
                damaged_frac = damaged_count / float(pool_size) if pool_size > 0 else 0.0
                
                # Log damage application results with perturb_counter info
                if circuit_pool.perturb_counter is not None:
                    max_perturb_count = int(jp.max(circuit_pool.perturb_counter))
                    circuits_with_damage = int(jp.sum(circuit_pool.perturb_counter > 0))
                    log.info(
                        f"Damage applied at epoch {epoch}: count={damaged_count}, fraction={damaged_frac:.4f}, "
                        f"max_perturb_count={max_perturb_count}, circuits_with_damage={circuits_with_damage}"
                    )
                else:
                    log.info(
                        f"Damage applied at epoch {epoch}: count={damaged_count}, fraction={damaged_frac:.4f}"
                    )
                if wandb_run:
                    wandb_run.log(
                        {
                            "pool/damaged_count": damaged_count,
                            "pool/damaged_fraction": damaged_frac,
                            "training/epoch": epoch,
                        }
                    )

            if jp.isnan(loss):
                log.warning(f"Loss is NaN at epoch {epoch}, returning last stable state")
                # Save the last stable state if enabled
                save_stable_state(
                    checkpoint_path,
                    save_stable_states,
                    last_stable_state,
                    epoch,
                    wandb_run,
                )
                return last_stable_state
            else:
                # Update last stable state
                last_stable_state = {
                    "model": model,
                    "optimizer": optimizer,
                    "pool": circuit_pool,
                    "metrics": {
                        "losses": losses.copy(),
                        "hard_losses": hard_losses.copy(),
                        "accuracies": accuracies.copy(),
                        "hard_accuracies": hard_accuracies.copy(),
                        "reset_steps": reset_steps.copy(),
                    },
                    "epoch": epoch,
                }

                # Record metrics
                # Ensure metrics are scalars (handle batched case)
                loss_scalar = float(jp.mean(loss) if hasattr(loss, 'ndim') and loss.ndim > 0 else loss)
                hard_loss_scalar = float(jp.mean(hard_loss) if hasattr(hard_loss, 'ndim') and hard_loss.ndim > 0 else hard_loss)
                accuracy_scalar = float(jp.mean(accuracy) if hasattr(accuracy, 'ndim') and accuracy.ndim > 0 else accuracy)
                hard_accuracy_scalar = float(jp.mean(hard_accuracy) if hasattr(hard_accuracy, 'ndim') and hard_accuracy.ndim > 0 else hard_accuracy)
                
                losses.append(loss_scalar)
                hard_losses.append(hard_loss_scalar)
                accuracies.append(accuracy_scalar)
                hard_accuracies.append(hard_accuracy_scalar)
                reset_steps.append(float(avg_steps_reset))

                # Prepare training metrics for best model tracking
                training_metrics = {
                    "loss": loss_scalar,
                    "hard_loss": hard_loss_scalar,
                    "accuracy": accuracy_scalar,
                    "hard_accuracy": hard_accuracy_scalar,
                }

                # Initialize evaluation metrics as None (will be set if periodic eval runs)
                current_eval_metrics = None

                avg_steps = circuit_pool.get_average_update_steps()
                # Log to wandb if enabled
                metrics_dict = {
                    "training/epoch": epoch,
                    "training/loss": float(loss),
                    "training/hard_loss": float(hard_loss),
                    "training/accuracy": float(accuracy),
                    "training/hard_accuracy": float(hard_accuracy),
                    "training/mode": training_mode,
                    # "pool/wiring_diversity": float(diversity),
                    "pool/reset_steps": float(avg_steps_reset),
                    # "pool/avg_update_steps": float(avg_steps),
                    "pool/loss_steps": loss_steps,
                }

                # Add learning rate if available
                schedule_value = schedule(epoch) if schedule is not None else learning_rate
                metrics_dict["scheduler/learning_rate"] = schedule_value

                # Add early stopping metrics if enabled
                if stop_accuracy_enabled:
                    metrics_dict["early_stop/enabled"] = True
                    metrics_dict["early_stop/epochs_above_threshold"] = epochs_above_threshold
                    metrics_dict["early_stop/threshold"] = stop_accuracy_threshold
                    if first_threshold_epoch is not None:
                        metrics_dict["early_stop/first_threshold_epoch"] = first_threshold_epoch

                _log_to_wandb(wandb_run, metrics_dict, epoch, log_interval)

                # Update progress bar with current metrics
                postfix_dict = {
                    "Loss": f"{loss:.4f}",
                    "Accuracy": f"{accuracy:.4f}",
                    "Hard Acc": f"{hard_accuracy:.4f}",
                    # "Diversity": f"{diversity:.3f}",
                    "Reset Steps": f"{avg_steps_reset:.2f}",
                    "Loss Steps": f"{loss_steps:.2f}",
                }

                pbar.set_postfix(postfix_dict)

                # Step 2: Run knockout evaluation if enabled (BEFORE best model tracking)
                all_eval_metrics = {}
                if (
                    knockout_eval
                    and knockout_eval.get("enabled")
                    and knockout_eval_base_circuit is not None
                    and epoch % periodic_eval_interval == 0
                ):
                    base_wires, base_logits = knockout_eval_base_circuit

                    # Run one-time backprop evaluation for comparison
                    if bp_results is None and backprop_config is not None:
                        log.info("Running one-time backprop evaluation for reuse (hamming + plotting)...")
                        from types import SimpleNamespace
                        from boolean_nca_cc.training.backprop import _run_backpropagation_training_with_knockouts

                        mock_cfg = SimpleNamespace(
                            backprop=SimpleNamespace(**backprop_config),
                            circuit=SimpleNamespace(layer_sizes=layer_sizes, arity=arity),
                            test_seed=periodic_eval_test_seed,
                            logging=SimpleNamespace(log_interval=log_interval),
                        )

                        # Use train data only to avoid data leakage (test data should never be seen during training)
                        bp_results = _run_backpropagation_training_with_knockouts(
                            mock_cfg, x_train, y_train, loss_type, knockout_vocabulary,
                            parallel=backprop_config.get("parallel", True),
                            batch_size=backprop_config.get("batch_size", None)
                        )

                        # Derive hamming summary from backprop results for comparison
                        bp_final_logits_list = [p['params'] for p in bp_results['patterns_performance']]
                        bp_final_logits = jax.tree.map(lambda *xs: jp.stack(xs, axis=0), *bp_final_logits_list)
                        bp_final_hard_accuracies = jp.array([p['final_hard_accuracy'] for p in bp_results['patterns_performance']])

                        bp_summary_rows = _analyze_knockout_hamming_distances(
                            base_logits=base_logits,
                            knockout_patterns=knockout_vocabulary,
                            final_logits_per_pattern=bp_final_logits,
                            layer_sizes=layer_sizes,
                            final_hard_accuracies=bp_final_hard_accuracies,
                        )
                        
                        for row in bp_summary_rows:
                            row['method'] = 'bp'
                        bp_hamming_summary = bp_summary_rows
                        log.info(f"Backprop evaluation complete. Found {len(bp_hamming_summary)} results.")

                    # Pass vocabulary and wandb_run for detailed checks
                    # Pass accumulated data and get updated version back
                    # Ensure greedy indices are available to eval schedule
                    local_knockout_eval = dict(knockout_eval) if knockout_eval is not None else {}
                    if greedy_ordered_indices is not None:
                        local_knockout_eval.setdefault("greedy_ordered_indices", greedy_ordered_indices)

                    # Run knockout evaluation on TEST data (default metrics: eval_ko_in/*, eval_ko_out/*)
                    ko_eval_results, accumulated_pattern_data = run_knockout_periodic_evaluation(
                        model=model,
                        knockout_vocabulary=knockout_vocabulary,
                        base_wires=base_wires,
                        base_logits=base_logits,
                        knockout_config=local_knockout_eval,
                        periodic_eval_test_seed=periodic_eval_test_seed,
                        x_data=x_test,
                        y_data=y_test,
                        input_n=input_n,
                        arity=arity,
                        circuit_hidden_dim=circuit_hidden_dim,
                        n_message_steps=periodic_eval_inner_steps,
                        loss_type=loss_type,
                        epoch=epoch,
                        wandb_run=wandb_run,
                        eval_batch_size=periodic_eval_batch_size,
                        accumulated_pattern_data=accumulated_pattern_data,  # Pass accumulated data
                        training_mode=training_mode,  # Pass training mode
                        log_stepwise=periodic_eval_log_stepwise,
                        layer_sizes=layer_sizes,
                        use_scan=use_scan,
                        knockout_diversity=knockout_diversity,  # Pass diversity for color coding
                        hamming_analysis_dir=hamming_analysis_dir,  # Pass hamming analysis directory
                        bp_hamming_summary=bp_hamming_summary,
                        layer_neighbors=layer_neighbors,
                        metric_suffix="",  # Default metrics for test data
                        input_split_enabled=input_split_enabled,  # Pass input split flag
                    )
                    # Extract final metrics for best model tracking
                    if ko_eval_results and "final_metrics_in" in ko_eval_results:
                        all_eval_metrics.update(ko_eval_results["final_metrics_in"])
                        all_eval_metrics.update(ko_eval_results["final_metrics_out"])
                    
                    # Set current eval metrics to the combined dictionary if any evals ran
                    current_eval_metrics = ko_eval_results.get("final_metrics_in", None)
                    
                    # When input split is enabled, also run evaluation on TRAIN data for comparison
                    if input_split_enabled:
                        ko_eval_results_train, _ = run_knockout_periodic_evaluation(
                            model=model,
                            knockout_vocabulary=knockout_vocabulary,
                            base_wires=base_wires,
                            base_logits=base_logits,
                            knockout_config=local_knockout_eval,
                            periodic_eval_test_seed=periodic_eval_test_seed,
                            x_data=x_train,
                            y_data=y_train,
                            input_n=input_n,
                            arity=arity,
                            circuit_hidden_dim=circuit_hidden_dim,
                            n_message_steps=periodic_eval_inner_steps,
                            loss_type=loss_type,
                            epoch=epoch,
                            wandb_run=wandb_run,
                            eval_batch_size=periodic_eval_batch_size,
                            accumulated_pattern_data=[],  # Don't accumulate for train eval
                            training_mode=training_mode,
                            log_stepwise=False,  # Skip stepwise for train eval to reduce noise
                            layer_sizes=layer_sizes,
                            use_scan=use_scan,
                            knockout_diversity=knockout_diversity,
                            hamming_analysis_dir=None,  # Skip hamming analysis for train eval
                            bp_hamming_summary=None,
                            layer_neighbors=layer_neighbors,
                            metric_suffix="_train",  # Train data metrics: eval_ko_in_train/*, eval_ko_out_train/*
                            input_split_enabled=input_split_enabled,  # Pass input split flag
                        )
                        # Also add train metrics to all_eval_metrics
                        if ko_eval_results_train and "final_metrics_in" in ko_eval_results_train:
                            all_eval_metrics.update(ko_eval_results_train["final_metrics_in"])
                            all_eval_metrics.update(ko_eval_results_train["final_metrics_out"])
                
                # Simple no-damage evaluation: just evaluate model on base circuit
                # This tests if the model degrades well-configured circuits even when trained without damage
                # Only run when knockout eval is disabled (otherwise knockout eval handles evaluation)
                if (
                    (not knockout_eval or not knockout_eval.get("enabled"))
                    and knockout_eval_base_circuit is not None
                    and epoch % periodic_eval_interval == 0
                ):
                    base_wires, base_logits = knockout_eval_base_circuit
                    
                    # Replicate base circuit for batch
                    eval_batch_size = 16  # Small batch for simple eval
                    eval_wires = jax.tree.map(
                        lambda x: jp.repeat(x[None, ...], eval_batch_size, axis=0), base_wires
                    )
                    eval_logits = jax.tree.map(
                        lambda x: jp.repeat(x[None, ...], eval_batch_size, axis=0), base_logits
                    )
                    
                    # Helper function to run no-damage eval with configurable data and suffix
                    def _run_no_damage_eval(x_eval, y_eval, metric_suffix=""):
                        step_metrics = evaluate_circuits_in_chunks(
                            eval_fn=evaluate_model_stepwise_batched,
                            wires=eval_wires,
                            logits=eval_logits,
                            knockout_patterns=None,  # No damage!
                            target_chunk_size=eval_batch_size,
                            model=model,
                            x_data=x_eval,
                            y_data=y_eval,
                            input_n=input_n,
                            arity=arity,
                            circuit_hidden_dim=circuit_hidden_dim,
                            n_message_steps=periodic_eval_inner_steps,
                            loss_type=loss_type,
                            layer_sizes=layer_sizes,
                            return_per_pattern=False,
                            layer_neighbors=layer_neighbors,
                            # Disable damage injection completely
                            damage_mode="greedy",  # Won't matter since no patterns
                            damage_injection_mode="single",
                            max_damage_per_circuit=1,
                            greedy_ordered_indices=None,  # Explicitly disable periodic injections
                            greedy_window_size=1,
                            greedy_injection_recover_steps=10,
                            damage_start_offset=0,
                            knockout_vocabulary=None,  # Also disable vocabulary
                        )
                        
                        # Log metrics
                        final_metrics = {
                            f"eval_no_damage{metric_suffix}/final_loss": step_metrics["soft_loss"][-1],
                            f"eval_no_damage{metric_suffix}/final_hard_loss": step_metrics["hard_loss"][-1],
                            f"eval_no_damage{metric_suffix}/final_accuracy": step_metrics["soft_accuracy"][-1],
                            f"eval_no_damage{metric_suffix}/final_hard_accuracy": step_metrics["hard_accuracy"][-1],
                            f"eval_no_damage{metric_suffix}/final_full_map_accuracy": step_metrics["full_map_accuracy"][-1],
                            f"eval_no_damage{metric_suffix}/epoch": epoch,
                        }
                        
                        # Add to all_eval_metrics for best model tracking
                        all_eval_metrics.update(final_metrics)
                        
                        if wandb_run:
                            wandb_run.log(final_metrics)
                        
                        # Save metrics locally for reliable data access (avoids WandB step grouping issues)
                        # This replaces the need for raw metrics in WandB
                        try:
                            from boolean_nca_cc.utils.metrics_storage import save_eval_metrics_locally, save_stepwise_metrics_locally
                            import time
                            # Get run ID and sweep_id: wandb_run is the wandb module, so use wandb.run.id
                            sweep_id = None
                            if wandb_run and hasattr(wandb_run, 'run') and wandb_run.run:
                                run_id = wandb_run.run.id
                                # Get sweep_id if available
                                if hasattr(wandb_run.run, 'sweep_id') and wandb_run.run.sweep_id:
                                    sweep_id = str(wandb_run.run.sweep_id)
                            else:
                                run_id = f"local_{int(time.time())}"
                            
                            # Extract task and config from wandb_run_config or wandb_run
                            task = None
                            if wandb_run_config and isinstance(wandb_run_config, dict):
                                circuit_config = wandb_run_config.get("circuit", {})
                                if isinstance(circuit_config, dict):
                                    task = circuit_config.get("task")
                            elif wandb_run:
                                try:
                                    if hasattr(wandb_run, 'config') and wandb_run.config:
                                        config = wandb_run.config if isinstance(wandb_run.config, dict) else dict(wandb_run.config)
                                        circuit_config = config.get("circuit", {})
                                        if isinstance(circuit_config, dict):
                                            task = circuit_config.get("task")
                                except Exception:
                                    pass
                            
                            # Get damage_type from model (defaults to "permanent" if not set)
                            damage_type = getattr(model, "damage_behavior", "permanent")
                            
                            # Determine input_split: when input_split_enabled, test=unseen, train=seen
                            # metric_suffix="" means test data, "_train" means train data
                            input_split_val = None
                            if input_split_enabled:
                                input_split_val = "unseen" if metric_suffix == "" else "seen"
                            
                            # Save final metrics
                            save_eval_metrics_locally(
                                metrics=final_metrics,
                                epoch=epoch,
                                split="test" if metric_suffix == "" else "train",
                                run_id=run_id,
                                sweep_id=sweep_id,
                                metrics_dir="results/metrics",
                                format="jsonl",
                                task=task,
                                training_mode=training_mode,
                                damage_type=damage_type,
                                eval_type="no_damage",
                                input_split=input_split_val,
                            )
                            
                            # Save stepwise metrics if enabled
                            if periodic_eval_log_stepwise and step_metrics:
                                save_stepwise_metrics_locally(
                                    step_metrics=step_metrics,
                                    epoch=epoch,
                                    split="test" if metric_suffix == "" else "train",
                                    run_id=run_id,
                                    sweep_id=sweep_id,
                                    metrics_dir="results/metrics",
                                    format="jsonl",
                                    task=task,
                                    training_mode=training_mode,
                                    damage_type=damage_type,
                                    eval_type="no_damage",
                                    input_split=input_split_val,
                                )
                        except Exception as e:
                            log.warning(f"Failed to save metrics locally: {e}")
                        
                        # Log stepwise if enabled (for both test and train sets)
                        if periodic_eval_log_stepwise and wandb_run:
                            for step_idx in range(len(step_metrics["step"])):
                                wandb_run.log({
                                    f"eval_no_damage{metric_suffix}_steps/step": step_metrics["step"][step_idx],
                                    f"eval_no_damage{metric_suffix}_steps/loss": step_metrics["soft_loss"][step_idx],
                                    f"eval_no_damage{metric_suffix}_steps/hard_loss": step_metrics["hard_loss"][step_idx],
                                    f"eval_no_damage{metric_suffix}_steps/accuracy": step_metrics["soft_accuracy"][step_idx],
                                    f"eval_no_damage{metric_suffix}_steps/hard_accuracy": step_metrics["hard_accuracy"][step_idx],
                                    f"eval_no_damage{metric_suffix}_steps/full_map_accuracy": step_metrics["full_map_accuracy"][step_idx],
                                    f"eval_no_damage{metric_suffix}_steps/epoch": epoch,
                                })
                        
                        suffix_label = f" ({metric_suffix.strip('_')})" if metric_suffix else ""
                        log.info(
                            f"No-Damage Eval{suffix_label} (epoch {epoch}): "
                            f"Loss={final_metrics[f'eval_no_damage{metric_suffix}/final_loss']:.4f}, "
                            f"Acc={final_metrics[f'eval_no_damage{metric_suffix}/final_accuracy']:.4f}, "
                            f"Hard Acc={final_metrics[f'eval_no_damage{metric_suffix}/final_hard_accuracy']:.4f}"
                        )
                        
                        return final_metrics
                    
                    # Run no-damage evaluation on TEST data (default metrics)
                    final_metrics = _run_no_damage_eval(x_test, y_test, metric_suffix="")
                    
                    # When input split is enabled, also run on TRAIN data for comparison
                    if input_split_enabled:
                        _run_no_damage_eval(x_train, y_train, metric_suffix="_train")
                    
                    # Update current_eval_metrics (merge if already set, otherwise set)
                    if current_eval_metrics is None:
                        current_eval_metrics = final_metrics
                    else:
                        current_eval_metrics.update(final_metrics)

                # Step 3: Get current metric value for best model tracking using modular approach
                try:
                    current_metric_value = get_metric_value(
                        best_metric,
                        best_metric_source,
                        training_metrics,
                        current_eval_metrics,
                    )
                except (ValueError, KeyError) as e:
                    if best_metric_source == "eval" and not knockout_eval:
                        log.warning(
                            f"Best metric source is 'eval' but periodic evaluation is disabled. "
                            f"Falling back to training metrics for {best_metric}."
                        )
                        current_metric_value = get_metric_value(
                            best_metric,
                            "training",
                            training_metrics,
                            current_eval_metrics,
                        )
                    elif (best_metric_source == "eval" or best_metric_source == "eval_ko_in" or best_metric_source == "eval_no_damage") and current_eval_metrics is None:
                        # Evaluation is enabled but hasn't run yet this epoch, skip best model check
                        current_metric_value = None
                    else:
                        raise e

                # Check if this is the best model based on the specified metric
                is_best = False
                if current_metric_value is not None:
                    if "accuracy" in best_metric:  # For accuracy metrics, higher is better
                        if current_metric_value > best_metric_value:
                            best_metric_value = current_metric_value
                            is_best = True
                    else:  # For loss metrics, lower is better
                        if current_metric_value < best_metric_value:
                            best_metric_value = current_metric_value
                            is_best = True

                # Step 4: Save checkpoints (periodic always, best if improvement detected)
                if checkpoint_enabled:
                    save_periodic_checkpoint(
                        checkpoint_path,
                        model,
                        optimizer,
                        {
                            "losses": losses,
                            "hard_losses": hard_losses,
                            "accuracies": accuracies,
                            "hard_accuracies": hard_accuracies,
                            "reset_steps": reset_steps,
                        },
                        epoch,
                        checkpoint_interval,
                        wandb_run,
                    )

                    # Save best model if enabled and is best
                    save_best_checkpoint(
                        checkpoint_path,
                        is_best,
                        save_best,
                        model,
                        optimizer,
                        {
                            "losses": losses,
                            "hard_losses": hard_losses,
                            "accuracies": accuracies,
                            "hard_accuracies": hard_accuracies,
                            "reset_steps": reset_steps,
                        },
                        epoch,
                        f"{best_metric_source}_{best_metric}",  # Include source in metric name
                        current_metric_value
                        if current_metric_value is not None
                        else best_metric_value,
                        wandb_run,
                    )

                # Step 5: Check for early stopping based on accuracy
                (
                    should_break,
                    early_stop_triggered,
                    epochs_above_threshold,
                    first_threshold_epoch,
                    current_eval_metrics,
                    rng,
                ) = check_early_stopping(
                    stop_accuracy_enabled=stop_accuracy_enabled,
                    epoch=epoch,
                    stop_accuracy_min_epochs=stop_accuracy_min_epochs,
                    early_stop_triggered=early_stop_triggered,
                    stop_accuracy_metric=stop_accuracy_metric,
                    stop_accuracy_source=stop_accuracy_source,
                    training_metrics=training_metrics,
                    current_eval_metrics=current_eval_metrics,
                    stop_accuracy_threshold=stop_accuracy_threshold,
                    first_threshold_epoch=first_threshold_epoch,
                    epochs_above_threshold=epochs_above_threshold,
                    stop_accuracy_patience=stop_accuracy_patience,
                    rng=rng,
                )

                if should_break:
                    break

                # # Set current eval metrics to the combined dictionary if any evals ran
                # current_eval_metrics = all_eval_metrics if all_eval_metrics else None

                # Return the trained GNN model and metrics
                result = {
                    "model": model,
                    "optimizer": optimizer,
                    "losses": losses,
                    "hard_losses": hard_losses,
                    "accuracies": accuracies,
                    "hard_accuracies": hard_accuracies,
                    "reset_steps": reset_steps,
                    "best_metric_value": best_metric_value,
                    "best_metric": best_metric,
                    "early_stopped": early_stop_triggered,
                    "early_stop_epoch": epoch if early_stop_triggered else None,
                    "first_threshold_epoch": first_threshold_epoch,
                }

                # Add pool to result if used
                result["pool"] = circuit_pool
    except KeyboardInterrupt:
        log.info(f"Training interrupted by user at epoch {epoch}/{epochs}")
        # Ensure progress bar is properly closed
        pbar.close()
    # Log final results to wandb
    _log_final_wandb_metrics(wandb_run, result, epochs)
    
    # Create final Figure 1: Training progress tracking (complete dataset)
    # NOTE: create_eval_plot_prog was commented out in the original visualization.py
    # and has not been moved to the new location. This functionality is disabled.
    if False and accumulated_pattern_data:  # Disabled until create_eval_plot_prog is reimplemented
        try:
            # from experiments.visualization.plot_trajectory import create_eval_plot_prog  # Not yet implemented
            pass
            
            # Compute backpropagation results for reference line (reuse across figures)
            bp_results = None
            if knockout_vocabulary is not None:
                try:
                    from types import SimpleNamespace
                    from boolean_nca_cc.training.backprop import _run_backpropagation_training_with_knockouts
                    
                    # Create config object for backprop evaluation
                    if backprop_config is not None:
                        mock_cfg = SimpleNamespace(
                            test_seed=periodic_eval_test_seed,
                            circuit=SimpleNamespace(
                                layer_sizes=layer_sizes,
                                arity=arity
                            ),
                            backprop=SimpleNamespace(**backprop_config),
                            logging=SimpleNamespace(log_interval=log_interval),
                        )
                    else:
                        # Create default backprop config
                        mock_cfg = SimpleNamespace(
                            test_seed=periodic_eval_test_seed,
                            circuit=SimpleNamespace(
                                layer_sizes=layer_sizes,
                                arity=arity
                            ),
                            backprop=SimpleNamespace(
                                epochs=50,  # Reduced for faster evaluation
                                learning_rate=1e-2,
                                weight_decay=1e-4,
                                optimizer="adam",
                                beta1=0.9,
                                beta2=0.999,
                            ),
                            logging=SimpleNamespace(log_interval=log_interval),
                        )
                    
                    # Run backpropagation training once for reuse
                    # Use train data only to avoid data leakage (test data should never be seen during training)
                    log.info("Computing backpropagation results for Figure 1 reference line...")
                    bp_results = _run_backpropagation_training_with_knockouts(
                        mock_cfg, x_train, y_train, loss_type, knockout_vocabulary,
                        parallel=backprop_config.get("parallel", True) if backprop_config else True,
                        batch_size=backprop_config.get("batch_size", None) if backprop_config else None
                    )
                    log.info(f"Backpropagation results computed. Mean final accuracy: {bp_results['aggregate_metrics']['mean_final_hard_accuracy']:.3f}")
                    
                except Exception as e:
                    log.warning(f"Error computing backpropagation results for Figure 1: {e}")
                    bp_results = None
            
            # Create the final plot with all accumulated data and BP reference
            final_plot_path = create_eval_plot_prog(
                pattern_data=accumulated_pattern_data,
                training_mode=training_mode,
                wandb_run=wandb_run,
                output_dir="reports/figures",
                filename=f"training_progress_final_{training_mode}.png",
                bp_results=bp_results,
                show_bp_reference=True
            )
            
            # Log the final plot to wandb if enabled
            if wandb_run:
                import wandb as wandb_module
                wandb_run.log({"training_progress_final": wandb_module.Image(final_plot_path)})
                log.info(f"Final Figure 1 (training progress) saved and logged to wandb: {final_plot_path}")
            else:
                log.info(f"Final Figure 1 (training progress) saved to: {final_plot_path}")
            
        except Exception as e:
            log.warning(f"Error creating final training progress plot: {e}")

    # Create Figure 3: Damage recovery and growth trajectories (stepwise)
    if (knockout_vocabulary is not None and 
        knockout_eval_base_circuit is not None and 
        result.get("model") is not None):
        try:
            import os
            import wandb
            import matplotlib.pyplot as plt
            
            # Create reports/figures directory if it doesn't exist
            os.makedirs("reports/figures", exist_ok=True)
            
            # Validate knockout_vocabulary before proceeding
            log.info(f"Figure 3 creation: knockout_vocabulary type={type(knockout_vocabulary)}, bp_results is {'not None' if bp_results else 'None'}")
            
            # Create a proper config object for backprop evaluation (needed for SA evaluation)
            from types import SimpleNamespace
            
            # Use backprop_config if available, otherwise create default config
            if backprop_config is not None:
                mock_cfg = SimpleNamespace(
                    test_seed=periodic_eval_test_seed,
                    circuit=SimpleNamespace(
                        layer_sizes=layer_sizes,
                        arity=arity
                    ),
                    backprop=SimpleNamespace(**backprop_config),
                    logging=SimpleNamespace(log_interval=log_interval),
                )
            else:
                # Create default backprop config
                mock_cfg = SimpleNamespace(
                    test_seed=periodic_eval_test_seed,
                    circuit=SimpleNamespace(
                        layer_sizes=layer_sizes,
                        arity=arity
                    ),
                    backprop=SimpleNamespace(
                        epochs=50,  # Reduced for faster evaluation
                        learning_rate=1e-2,
                        weight_decay=1e-4,
                        optimizer="adam",
                        beta1=0.9,
                        beta2=0.999,
                    ),
                    logging=SimpleNamespace(log_interval=log_interval),
                )
            
            # Create Figure 3 with BP reference line (show_bp_trajectory=False)
            # Reuse bp_results from Figure 1 if available, otherwise compute new ones
            # Use test data for evaluation/plotting (training is complete)
            fig3 = plot_combined_bp_sa_stepwise_performance(
                cfg=mock_cfg,
                x_data=x_test,
                y_data=y_test,
                loss_type=loss_type,
                knockout_patterns=knockout_vocabulary,
                model=result["model"],
                base_circuit=knockout_eval_base_circuit,
                n_message_steps=periodic_eval_inner_steps,
                layer_sizes=layer_sizes,
                input_n=input_n,
                arity=arity,
                circuit_hidden_dim=circuit_hidden_dim,
                bp_results=bp_results,  # Reuse BP results from Figure 1
                show_bp_trajectory=False,  # Figure 3 mode: BP as reference line
                periodic_eval_test_seed=periodic_eval_test_seed,
                knockout_config=knockout_eval,
                show_ood_trajectory=True,  # Enable OOD trajectory plotting
                # Multi-damage support parameters
                damage_mode=knockout_eval.get("damage_mode", "greedy"),
                damage_injection_mode=knockout_eval.get("damage_injection_mode", "multi"),
                max_damage_per_circuit=int(knockout_eval.get("max_damage_per_circuit", 10)),
                greedy_ordered_indices=greedy_ordered_indices,
                greedy_window_size=int(knockout_eval.get("greedy_window_size", 1)),
                greedy_injection_recover_steps=int(knockout_eval.get("greedy_injection_recover_steps", 10)),
                damage_start_offset=int(knockout_eval.get("damage_start_offset", 0)),
                damage_start_offset_random=knockout_eval.get("damage_start_offset_random", False),
                damage_start_offset_seed=int(knockout_eval.get("damage_start_offset_seed", 42)),
                knockout_vocabulary=knockout_vocabulary,
                training_mode=training_mode,  # Add this line
            )
            
            # Save Figure 3 locally
            fig3_path = f"reports/figures/damage_recovery_trajectories_{training_mode}.png"
            fig3.savefig(fig3_path, dpi=300, bbox_inches='tight')
            plt.close(fig3)
            
            # Log Figure 3 to wandb if enabled
            if wandb_run:
                wandb_run.log({"figure3_damage_recovery_trajectories": wandb.Image(fig3_path)})
                log.info(f"Figure 3 (damage recovery trajectories) saved and logged to wandb: {fig3_path}")
            else:
                log.info(f"Figure 3 (damage recovery trajectories) saved to: {fig3_path}")
            
        except Exception as e:
            log.warning(f"Error creating Figure 3 (damage recovery trajectories): {e}")

    return result