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
    get_loss_and_update_graph, evaluate_circuits_in_chunks
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
from boolean_nca_cc.analysis.hamming_distance import (
    _hard_truth_tables_from_logits,
    _active_gate_mask_from_knockout,
    _hamming_distance_tables,
)
from boolean_nca_cc.analysis.visualization import plot_accuracy_vs_distance
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
            # Vocabulary-based evaluation parameters
            knockout_vocabulary=knockout_vocabulary,  # If provided => seen; else => unseen
        )

        final_metrics_in = {
            "eval_ko_in/final_loss": step_metrics_in["soft_loss"][-1],
            "eval_ko_in/final_hard_loss": step_metrics_in["hard_loss"][-1],
            "eval_ko_in/final_accuracy": step_metrics_in["soft_accuracy"][-1],
            "eval_ko_in/final_hard_accuracy": step_metrics_in["hard_accuracy"][-1],
            "eval_ko_in/epoch": epoch,
        }
        
        # Calculate and add standard deviation for eval_ko_in/hard_accuracy
        if "per_pattern" in step_metrics_in:
            final_hard_accuracies_in = step_metrics_in["per_pattern"]["pattern_hard_accuracies"][-1]
            hard_acc_std = float(jp.std(final_hard_accuracies_in))
            final_metrics_in["eval_ko_in/final_hard_accuracy_std"] = hard_acc_std

        
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
            # For OUT, force unseen by not providing a vocabulary
            knockout_vocabulary=None,  # Force unseen patterns for OUT evaluation
        )

        final_metrics_out = {
            "eval_ko_out/final_loss": step_metrics_out["soft_loss"][-1],
            "eval_ko_out/final_hard_loss": step_metrics_out["hard_loss"][-1],
            "eval_ko_out/final_accuracy": step_metrics_out["soft_accuracy"][-1],
            "eval_ko_out/final_hard_accuracy": step_metrics_out["hard_accuracy"][-1],
            "eval_ko_out/epoch": epoch,
        }
        

        # Log main metrics normally (these will create panels)
        main_metrics = {**final_metrics_in, **final_metrics_out}
        
        if wandb_run:
            wandb_run.log(main_metrics)
            
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
                    
                    plot_accuracy_vs_distance(
                        summary_df=df,
                        output_path=plot_path,
                        color_by_method=True
                    )
                
                    log.info(f"Hamming distance analysis plot saved to: {plot_path}")
                    
                    # Log plot to wandb if enabled
                    if wandb_run:
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
                    wandb_run.log({
                        "eval_ko_in_steps/step": step_metrics_in["step"][step_idx],
                        "eval_ko_in_steps/loss": step_metrics_in["soft_loss"][step_idx],
                        "eval_ko_in_steps/hard_loss": step_metrics_in["hard_loss"][step_idx],
                        "eval_ko_in_steps/accuracy": step_metrics_in["soft_accuracy"][step_idx],
                        "eval_ko_in_steps/hard_accuracy": step_metrics_in["hard_accuracy"][step_idx],
                        "eval_ko_in_steps/epoch": epoch,
                    })
                for step_idx in range(len(step_metrics_out["step"])):
                    wandb_run.log({
                        "eval_ko_out_steps/step": step_metrics_out["step"][step_idx],
                        "eval_ko_out_steps/loss": step_metrics_out["soft_loss"][step_idx],
                        "eval_ko_out_steps/hard_loss": step_metrics_out["hard_loss"][step_idx],
                        "eval_ko_out_steps/accuracy": step_metrics_out["soft_accuracy"][step_idx],
                        "eval_ko_out_steps/hard_accuracy": step_metrics_out["hard_accuracy"][step_idx],
                        "eval_ko_out_steps/epoch": epoch,
                    })

        log.info(
            f"Knockout Eval (epoch {epoch}):\n"
            f"  IN-distribution KO: Loss={final_metrics_in['eval_ko_in/final_loss']:.4f}, "
            f"Acc={final_metrics_in['eval_ko_in/final_accuracy']:.4f}, "
            f"Hard Acc={final_metrics_in['eval_ko_in/final_hard_accuracy']:.4f}"
            f"  OUT-of-distribution KO: Loss={final_metrics_out['eval_ko_out/final_loss']:.4f}, "
            f"Acc={final_metrics_out['eval_ko_out/final_accuracy']:.4f}, "
            f"Hard Acc={final_metrics_out['eval_ko_out/final_hard_accuracy']:.4f}"
        )

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

def plot_combined_bp_sa_stepwise_performance(
    cfg, 
    x_data, 
    y_data, 
    loss_type, 
    knockout_patterns,
    model,
    base_circuit,
    n_message_steps=100,
    layer_sizes=None,
    input_n=None,
    arity=2,
    circuit_hidden_dim=16,
    bp_results=None,
    show_bp_trajectory=True,
    periodic_eval_test_seed=42,
    knockout_config=None,
    show_ood_trajectory=True,
    layer_neighbors=False,
    # Multi-damage support parameters
    damage_mode: str = "greedy",
    damage_injection_mode: str = "multi",
    max_damage_per_circuit: int = 10,
    greedy_ordered_indices: Optional[List[int]] = None,
    greedy_window_size: int = 1,
    greedy_injection_recover_steps: int = 10,
    knockout_vocabulary: Optional[jp.ndarray] = None,
):
    """
    Create a combined plot showing backpropagation and SA stepwise performance on the same axes.
    
    Args:
        cfg: Configuration object
        x_data: Input data
        y_data: Target data  
        loss_type: Type of loss function ('l4' or 'bce')
        knockout_patterns: Array of IN-distribution knockout patterns to evaluate (used for static damage mode)
        model: Trained SA model
        base_circuit: Base circuit (wires, logits) for SA evaluation
        n_message_steps: Number of message passing steps for SA
        layer_sizes: Circuit layer sizes
        input_n: Number of inputs
        arity: Circuit arity
        circuit_hidden_dim: Circuit hidden dimension
        bp_results: Pre-computed backpropagation results (optional)
        show_bp_trajectory: If True, show full BP trajectory; if False, show only final BP accuracy as reference line
        periodic_eval_test_seed: Seed for generating OOD patterns (should match training evaluation)
        knockout_config: Configuration for knockout evaluation (needed for OOD pattern generation)
        show_ood_trajectory: If True, show OOD SA trajectory; if False, show only IN-distribution
        layer_neighbors: Whether to use layer neighbors in model
        # Multi-damage support parameters
        damage_mode: Pattern type ("greedy", "greedy_vocabulary", "shotgun", "strip")
        damage_injection_mode: "single" (one damage per circuit) or "multi" (multiple damages)
        max_damage_per_circuit: Maximum damage events per circuit
        greedy_ordered_indices: Ordered indices for greedy damage patterns
        greedy_window_size: Window size for greedy patterns
        greedy_injection_recover_steps: Recovery steps between damage injections
        knockout_vocabulary: Vocabulary of patterns for seen evaluation (None for unseen)
        
    Returns:
        matplotlib figure with the combined performance plot
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from boolean_nca_cc.training.evaluation import evaluate_circuits_in_chunks, evaluate_model_stepwise_batched
    # Import backpropagation training function
    if bp_results is None:
        from boolean_nca_cc.training.backprop import _run_backpropagation_training_with_knockouts
        # Run backpropagation training
        results = _run_backpropagation_training_with_knockouts(
            cfg, x_data, y_data, loss_type, knockout_patterns, 
            parallel=cfg.backprop.get("parallel", True),
            batch_size=cfg.backprop.get("batch_size", None)
        )
    else:
        results = bp_results
    
    # Run SA evaluation on the same patterns
    base_wires, base_logits = base_circuit
    
    # Determine batch size and evaluation approach based on damage mode
    if damage_mode in ["greedy", "greedy_vocabulary"] and damage_injection_mode == "multi":
        # Multi-damage mode: Use dynamic evaluation with periodic injections
        # Use a reasonable batch size for statistical robustness
        eval_batch_size = max(10, len(knockout_patterns) if knockout_patterns is not None else 10)
        
        # Replicate base circuit for the batch
        batch_wires = jax.tree.map(
            lambda x: jp.repeat(x[None, ...], eval_batch_size, axis=0), base_wires
        )
        batch_logits = jax.tree.map(
            lambda x: jp.repeat(x[None, ...], eval_batch_size, axis=0), base_logits
        )
        
        # Run SA evaluation with multi-damage support (no pre-generated patterns)
        sa_step_metrics_in = evaluate_circuits_in_chunks(
            eval_fn=evaluate_model_stepwise_batched,
            wires=batch_wires,
            logits=batch_logits,
            knockout_patterns=None,  # Let evaluation system handle dynamic patterns
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
            return_per_pattern=True,
            layer_neighbors=layer_neighbors,
            # Multi-damage parameters
            damage_mode=damage_mode,
            damage_injection_mode=damage_injection_mode,
            max_damage_per_circuit=max_damage_per_circuit,
            greedy_ordered_indices=greedy_ordered_indices,
            greedy_window_size=greedy_window_size,
            greedy_injection_recover_steps=greedy_injection_recover_steps,
            knockout_vocabulary=knockout_vocabulary,
        )
    else:
        # Static damage mode: Use pre-generated patterns (backward compatible)
        if knockout_patterns is None:
            raise ValueError("knockout_patterns must be provided for static damage modes")
            
        # Replicate base circuit for the batch
        batch_wires = jax.tree.map(
            lambda x: jp.repeat(x[None, ...], len(knockout_patterns), axis=0), base_wires
        )
        batch_logits = jax.tree.map(
            lambda x: jp.repeat(x[None, ...], len(knockout_patterns), axis=0), base_logits
        )
        
        # Run SA evaluation with stepwise metrics on IN-distribution patterns
        sa_step_metrics_in = evaluate_circuits_in_chunks(
            eval_fn=evaluate_model_stepwise_batched,
            wires=batch_wires,
            logits=batch_logits,
            knockout_patterns=knockout_patterns,
            target_chunk_size=len(knockout_patterns),
            model=model,
            x_data=x_data,
            y_data=y_data,
            input_n=input_n,
            arity=arity,
            circuit_hidden_dim=circuit_hidden_dim,
            n_message_steps=n_message_steps,
            loss_type=loss_type,
            layer_sizes=layer_sizes,
            return_per_pattern=True,
            layer_neighbors=layer_neighbors,
        )
    
    # Generate OUT-of-distribution knockout patterns if requested and config provided
    sa_step_metrics_out = None
    if show_ood_trajectory and knockout_config is not None:
        if damage_mode in ["greedy", "greedy_vocabulary"] and damage_injection_mode == "multi":
            # Multi-damage mode: Use dynamic evaluation with periodic injections for OOD
            log.info(f"Running OOD SA evaluation with multi-damage support (batch size: {eval_batch_size})...")
            
            # Replicate base circuit for the OOD batch
            out_batch_wires = jax.tree.map(
                lambda x: jp.repeat(x[None, ...], eval_batch_size, axis=0), base_wires
            )
            out_batch_logits = jax.tree.map(
                lambda x: jp.repeat(x[None, ...], eval_batch_size, axis=0), base_logits
            )
            
            # Run SA evaluation on OOD patterns with multi-damage support
            sa_step_metrics_out = evaluate_circuits_in_chunks(
                eval_fn=evaluate_model_stepwise_batched,
                wires=out_batch_wires,
                logits=out_batch_logits,
                knockout_patterns=None,  # Let evaluation system handle dynamic patterns
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
                return_per_pattern=True,
                layer_neighbors=layer_neighbors,
                # Multi-damage parameters for OOD (force unseen by not providing vocabulary)
                damage_mode=damage_mode,
                damage_injection_mode=damage_injection_mode,
                max_damage_per_circuit=max_damage_per_circuit,
                greedy_ordered_indices=greedy_ordered_indices,
                greedy_window_size=greedy_window_size,
                greedy_injection_recover_steps=greedy_injection_recover_steps,
                knockout_vocabulary=None,  # Force unseen patterns for OOD evaluation
            )
        else:
            # Static damage mode: Use pre-generated OOD patterns (backward compatible)
            if knockout_patterns is None:
                raise ValueError("knockout_patterns must be provided for static damage modes")
                
            log.info(f"Generating OOD knockout patterns for SA evaluation ({len(knockout_patterns)} patterns)...")
            
            # Use the same logic as in run_knockout_periodic_evaluation
            from boolean_nca_cc.training.pool.structural_perturbation import create_reproducible_knockout_pattern
            from functools import partial
            
            pattern_creator_fn = partial(
                create_reproducible_knockout_pattern,
                layer_sizes=layer_sizes,
                damage_prob=knockout_config["damage_prob"],
            )
            
            # Use different seed for OOD patterns (same as training evaluation)
            ood_rng = jax.random.PRNGKey(periodic_eval_test_seed + 1)
            out_pattern_keys = jax.random.split(ood_rng, len(knockout_patterns))
            out_knockout_patterns = jax.vmap(pattern_creator_fn)(out_pattern_keys)
            
            # Replicate base circuit for the OOD batch
            out_batch_wires = jax.tree.map(
                lambda x: jp.repeat(x[None, ...], len(out_knockout_patterns), axis=0), base_wires
            )
            out_batch_logits = jax.tree.map(
                lambda x: jp.repeat(x[None, ...], len(out_knockout_patterns), axis=0), base_logits
            )
            
            # Run SA evaluation on OOD patterns
            sa_step_metrics_out = evaluate_circuits_in_chunks(
                eval_fn=evaluate_model_stepwise_batched,
                wires=out_batch_wires,
                logits=out_batch_logits,
                knockout_patterns=out_knockout_patterns,
                target_chunk_size=len(out_knockout_patterns),
                model=model,
                x_data=x_data,
                y_data=y_data,
                input_n=input_n,
                arity=arity,
                circuit_hidden_dim=circuit_hidden_dim,
                n_message_steps=n_message_steps,
                loss_type=loss_type,
                layer_sizes=layer_sizes,
                return_per_pattern=True,
                layer_neighbors=layer_neighbors,
            )
        log.info("OOD SA evaluation completed")
    
    # Use IN-distribution metrics as the primary SA metrics for backward compatibility
    sa_step_metrics = sa_step_metrics_in
    
    # Create single figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Plot: Hard Accuracy over steps - BP vs SA comparison
    sa_steps = sa_step_metrics["step"]
    sa_hard_accuracies = sa_step_metrics["hard_accuracy"]
    
    # Evaluate the base circuit (preconfigured) without knockout patterns to get true pre-damage performance
    from boolean_nca_cc.training.evaluation import get_loss_from_wires_logits
    base_wires, base_logits = base_circuit
    _, base_aux = get_loss_from_wires_logits(base_logits, base_wires, x_data, y_data, loss_type)
    pre_damage_accuracy = float(base_aux[4])
    
    # Normalize all accuracy values so that pre-damage performance = 1.0
    normalization_factor = pre_damage_accuracy
    
    # Use SA data as-is (no artificial concatenations)
    # SA already includes step 0 (pre-damage) and steps 1 to n_message_steps
    sa_steps = np.array(sa_steps)
    sa_hard_accuracies = np.array(sa_hard_accuracies) / normalization_factor
    
    # Debug logging to understand what's happening with the data
    log.info(f"SA step_metrics keys: {list(sa_step_metrics.keys())}")
    log.info(f"SA steps shape: {sa_steps.shape}, SA accuracies shape: {sa_hard_accuracies.shape}")
    log.info(f"SA steps: {sa_steps}")
    log.info(f"SA accuracies (first 5): {sa_hard_accuracies[:5]}")
    log.info(f"SA accuracies (last 5): {sa_hard_accuracies[-5:]}")
    log.info(f"Damage mode: {damage_mode}, greedy_ordered_indices: {greedy_ordered_indices}")
    if 'per_pattern' in sa_step_metrics:
        log.info(f"SA per_pattern keys: {list(sa_step_metrics['per_pattern'].keys())}")
        if 'pattern_hard_accuracies' in sa_step_metrics['per_pattern']:
            per_pattern_shape = sa_step_metrics['per_pattern']['pattern_hard_accuracies'].shape
            log.info(f"SA per_pattern shape: {per_pattern_shape}")
    
    # Plot SA performance with error bands
    try:
        # Check if we have per-pattern data for error bands
        if 'per_pattern' in sa_step_metrics and 'pattern_hard_accuracies' in sa_step_metrics['per_pattern']:
            # Extract per-pattern hard accuracies
            per_pattern_accuracies = sa_step_metrics['per_pattern']['pattern_hard_accuracies']
            # Convert to numpy array for easier manipulation
            per_pattern_accuracies = np.array(per_pattern_accuracies)  # Shape: [n_steps, n_patterns]
            
            # Normalize per-pattern accuracies (no artificial concatenations)
            normalized_per_pattern_accuracies = per_pattern_accuracies / normalization_factor
            
            # Calculate mean and std across patterns at each step
            sa_mean_accuracies = np.mean(normalized_per_pattern_accuracies, axis=1)
            sa_std_accuracies = np.std(normalized_per_pattern_accuracies, axis=1)
            
            # Use SA steps as-is (should match per-pattern data)
            sa_steps_for_plot = sa_steps
            
            # Debug logging for per-pattern plotting
            log.info(f"Plotting SA per-pattern: steps shape={sa_steps_for_plot.shape}, mean shape={sa_mean_accuracies.shape}")
            log.info(f"SA mean accuracies (first 5): {sa_mean_accuracies[:5]}")
            log.info(f"SA mean accuracies (last 5): {sa_mean_accuracies[-5:]}")
            
            # Plot mean line
            ax.plot(sa_steps_for_plot, sa_mean_accuracies, 
                    color='black',
                    linewidth=1.5, 
                    alpha=0.9,
                    label='Trajectory (Seen)')
            
            # Plot error bands
            ax.fill_between(sa_steps_for_plot, 
                           sa_mean_accuracies - sa_std_accuracies,
                           sa_mean_accuracies + sa_std_accuracies,
                           color='black',
                           alpha=0.2)
                        #    label='Trajectory (±1σ)')
        else:
            # Fallback to averaged data without error bands
            log.info(f"Using SA fallback plotting: steps shape={sa_steps.shape}, accuracies shape={sa_hard_accuracies.shape}")
            log.info(f"SA fallback accuracies (first 5): {sa_hard_accuracies[:5]}")
            log.info(f"SA fallback accuracies (last 5): {sa_hard_accuracies[-5:]}")
            
            ax.plot(sa_steps, sa_hard_accuracies, 
                    color='black',
                    linewidth=1.5, 
                    alpha=0.7,
                    label='Trajectory (Seen)')
    except Exception as e:
        log.error(f"Error plotting SA data: {e}")
        log.error(f"SA steps shape: {len(sa_steps)}, SA accuracies shape: {len(sa_hard_accuracies)}")
        if 'per_pattern' in sa_step_metrics and 'pattern_hard_accuracies' in sa_step_metrics['per_pattern']:
            per_pattern_accuracies = np.array(sa_step_metrics['per_pattern']['pattern_hard_accuracies'])
            log.error(f"Per-pattern accuracies shape: {per_pattern_accuracies.shape}")
        raise
    
    # Plot OOD SA trajectory if available
    if sa_step_metrics_out is not None:
        try:
            # Process OOD data similar to IN-distribution data (no artificial concatenations)
            ood_sa_steps = sa_step_metrics_out["step"]
            ood_sa_hard_accuracies = sa_step_metrics_out["hard_accuracy"]
            
            # Use OOD SA data as-is (no artificial concatenations)
            # OOD SA already includes step 0 (pre-damage) and steps 1 to n_message_steps
            ood_sa_steps = np.array(ood_sa_steps)
            ood_sa_hard_accuracies = np.array(ood_sa_hard_accuracies) / normalization_factor
            
            # Check if we have per-pattern data for error bands
            if 'per_pattern' in sa_step_metrics_out and 'pattern_hard_accuracies' in sa_step_metrics_out['per_pattern']:
                # Extract per-pattern hard accuracies
                ood_per_pattern_accuracies = sa_step_metrics_out['per_pattern']['pattern_hard_accuracies']
                # Convert to numpy array for easier manipulation
                ood_per_pattern_accuracies = np.array(ood_per_pattern_accuracies)  # Shape: [n_steps, n_patterns]
                
                # Normalize OOD per-pattern accuracies (no artificial concatenations)
                normalized_ood_per_pattern_accuracies = ood_per_pattern_accuracies / normalization_factor
                
                # Calculate mean and std across patterns at each step
                ood_sa_mean_accuracies = np.mean(normalized_ood_per_pattern_accuracies, axis=1)
                ood_sa_std_accuracies = np.std(normalized_ood_per_pattern_accuracies, axis=1)
                
                # Use OOD SA steps as-is (should match per-pattern data)
                ood_sa_steps_for_plot = ood_sa_steps
                
                # Plot OOD mean line
                ax.plot(ood_sa_steps_for_plot, ood_sa_mean_accuracies, 
                        color='purple',
                        linewidth=1.5, 
                        alpha=0.9,
                        label='Trajectory (Unseen)')
                
                # Plot OOD error bands
                ax.fill_between(ood_sa_steps_for_plot, 
                               ood_sa_mean_accuracies - ood_sa_std_accuracies,
                               ood_sa_mean_accuracies + ood_sa_std_accuracies,
                               color='purple',
                               alpha=0.2)
            else:
                # Fallback to averaged data without error bands
                ax.plot(ood_sa_steps, ood_sa_hard_accuracies, 
                        color='purple',
                        linewidth=1.5, 
                        alpha=0.7,
                        label='Trajectory (Unseen)')
                        
        except Exception as e:
            log.warning(f"Error plotting OOD SA data: {e}")
            log.warning(f"OOD SA steps shape: {len(ood_sa_steps) if 'ood_sa_steps' in locals() else 'N/A'}, OOD SA accuracies shape: {len(ood_sa_hard_accuracies) if 'ood_sa_hard_accuracies' in locals() else 'N/A'}")
    
    # Aggregate BP performance across all patterns
    bp_accuracies_list = [pattern_results["hard_accuracies"] for pattern_results in results["patterns_performance"]]
    bp_accuracies_array = np.array(bp_accuracies_list)  # Shape: [n_patterns, n_steps]
    
    # Rebuild BP trajectory to match SA structure:
    # SA: [step0_pre_damage, step1, step2, ..., stepN] (n_message_steps + 1 total)
    # BP: [step0, step1, step2, ..., stepN-1] (epochs total)
    # We need to add the pre-damage state to BP to match SA structure
    
    # Get the initial (pre-damage) accuracy for BP - this should be the same as SA's step 0
    bp_initial_accuracy = pre_damage_accuracy  # Use the same pre-damage accuracy as SA
    
    # Add pre-damage state to BP trajectory
    bp_accuracies_with_initial = np.column_stack([
        np.full((bp_accuracies_array.shape[0], 1), bp_initial_accuracy),  # Add initial state
        bp_accuracies_array  # Add training steps
    ])  # Shape: [n_patterns, n_steps + 1]
    
    # Normalize BP accuracies by pre-damage performance
    bp_accuracies_array_normalized = bp_accuracies_with_initial / normalization_factor
    bp_mean_accuracies = np.mean(bp_accuracies_array_normalized, axis=0)  # Average across patterns
    bp_std_accuracies = np.std(bp_accuracies_array_normalized, axis=0)  # Standard deviation across patterns
    
    # BP steps now match SA structure: [0, 1, 2, ..., n_steps]
    bp_steps = range(len(bp_mean_accuracies))
    
    # Log step count information for debugging
    log.info(f"SA steps: {len(sa_steps)}, BP steps: {len(bp_steps)}")
    log.info(f"SA step range: {sa_steps[0]} to {sa_steps[-1]}")
    log.info(f"BP step range: {bp_steps[0]} to {bp_steps[-1]}")
    
    # Both SA and BP now have the same structure: [0, 1, 2, ..., n_steps]
    # SA: n_message_steps + 1 steps (includes pre-damage step 0)
    # BP: epochs + 1 steps (includes pre-damage step 0)
    
    # Handle step count differences with fallback truncation
    if len(sa_steps) != len(bp_steps):
        log.warning(f"Step count mismatch: SA={len(sa_steps)}, BP={len(bp_steps)}")
        # Use the shorter length to avoid dimension mismatch
        min_steps = min(len(sa_steps), len(bp_steps))
        sa_steps = sa_steps[:min_steps]
        sa_hard_accuracies = sa_hard_accuracies[:min_steps]
        bp_steps = range(min_steps)
        bp_mean_accuracies = bp_mean_accuracies[:min_steps]
        bp_std_accuracies = bp_std_accuracies[:min_steps]
        log.info(f"Truncated both datasets to {min_steps} steps")
    
    # Final verification that dimensions match
    if len(sa_steps) != len(sa_hard_accuracies):
        raise ValueError(f"SA steps and accuracies have different lengths: {len(sa_steps)} vs {len(sa_hard_accuracies)}")
    
    log.info(f"Final data shape: SA steps={len(sa_steps)}, SA accuracies={len(sa_hard_accuracies)}, BP steps={len(bp_steps)}, BP accuracies={len(bp_mean_accuracies)}")
    
    try:
        if show_bp_trajectory:
            # Plot full BP trajectory with error bands
            ax.plot(bp_steps, bp_mean_accuracies, 
                    color='blue',
                    linewidth=1.5,
                    alpha=0.7,
                    label='Backpropagation')
            
            # Add error bands for BP trajectory
            ax.fill_between(bp_steps, 
                           bp_mean_accuracies - bp_std_accuracies,
                           bp_mean_accuracies + bp_std_accuracies,
                           color='blue',
                           alpha=0.2)
        else:
            # Plot pre-damage circuit accuracy as horizontal reference line (normalized to 1.0)
            ax.axhline(y=1.0, 
                      color='#377eb8',
                      linestyle='--',
                      linewidth=2.0,
                      alpha=1.0,
                      label=f'Pre-damage Performance')
    except Exception as e:
        log.error(f"Error plotting BP data: {e}")
        log.error(f"BP steps shape: {len(bp_steps)}, BP accuracies shape: {len(bp_mean_accuracies)}")
        raise
    
    # Set font sizes - much larger increase for Figure 3
    ax.set_xlabel('Message Steps', fontsize=22)
    ax.set_ylabel('Hard Accuracy', fontsize=22)
    
    # Update title based on mode
    if show_bp_trajectory:
        ax.set_title('Hard Accuracy Over Steps: SA vs Backpropagation', fontsize=26)
    else:
        ax.set_title('Reconfiguration Trajectory', fontsize=26)
    
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.grid(True, alpha=0.3)
    # ax.set_ylim(0.6, 1.05)  # Adjusted for normalized values, with pre-damage at 1.0
    
    # Set y-axis ticks to show 0.05 steps
    import numpy as np
    y_ticks = np.arange(0.6, 1.03, 0.05)
    ax.set_yticks(y_ticks)
    
    # Set xlim based on damage mode and whether BP trajectory is shown
    if damage_injection_mode == "multi":
        # Multi-damage mode: extend xlim to accommodate multiple damage injections
        max_x = max_damage_per_circuit * greedy_injection_recover_steps + 20
        if show_bp_trajectory:
            ax.set_xlim(0, max(60, max_x))  # Use the larger of BP range or multi-damage range
        else:
            ax.set_xlim(0, max_x)  # Extended range for multi-damage SA-only plot
    else:
        # Single damage mode: use original ranges
        if show_bp_trajectory:
            ax.set_xlim(0, 60)  # Extended range for full BP trajectory
        else:
            ax.set_xlim(0, 17)  # Standard range for SA-only plot
    
    # Add color-coded regions and damage indicator only for single damage mode
    if damage_injection_mode == "single":
        ax.axvspan(0, 2, alpha=0.1, color='#377eb8')  # Pastel blue for pre-damage region
        ax.axvspan(2, 3, alpha=0.1, color='#ff7f00')  # Pastel orange for damage region
        ax.axvspan(3, 17, alpha=0.1, color='#4daf4a')  # Pastel green for recovery region
        ax.axvline(x=2, color='#ff7f00', linestyle='--', linewidth=2)  # Vertical line at damage point
    
    # Get existing legend handles and labels (only trajectories and pre-damage line)
    existing_handles, existing_labels = ax.get_legend_handles_labels()
    
    # Update legend with solid background (only trajectories and pre-damage line)
    legend = ax.legend(handles=existing_handles, labels=existing_labels, loc='lower right', fontsize=16)
    legend.get_frame().set_alpha(1.0)  # Make legend box solid
    
    plt.tight_layout()
    
    return fig


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
    """
       # Validate unified damage control parameters
    if damage_injection_mode not in ["single", "multi"]:
        raise ValueError(f"damage_injection_mode must be 'single' or 'multi', got '{damage_injection_mode}'")
    if max_damage_per_circuit < 1:
        raise ValueError(f"max_damage_per_circuit must be >= 1, got {max_damage_per_circuit}")
    if damage_injection_mode == "single" and max_damage_per_circuit != 1:
        raise ValueError(f"damage_injection_mode='single' requires max_damage_per_circuit=1, got {max_damage_per_circuit}")

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
        optimizer = nnx.Optimizer(model, opt_fn)
    else:
        # Use the provided optimizer
        optimizer = init_optimizer
        schedule = None

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
        optimizer.update(grads)

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

    # Initialize knockout evaluation if enabled
    knockout_eval_base_circuit = None
    if knockout_eval and knockout_eval.get("enabled"):
        # Store base circuit for on-demand evaluation (no pre-created datasets)
        log.info("Knockout evaluation enabled - will use vocabulary-based evaluation")
        if training_mode == "repair" and base_wires_preconfig is not None:
            knockout_eval_base_circuit = (base_wires_preconfig, base_logits_preconfig)
        else:
            knockout_eval_base_circuit = gen_circuit(wiring_fixed_key, layer_sizes, arity=arity)
        log.info("Base circuit created for knockout evaluation")

    # Initialize knockout vocabulary if knockout_diversity is configured
    knockout_vocabulary = None
    log.info(f"VOCAB CREATION DEBUG: knockout_diversity={knockout_diversity}, damage_pool_damage_prob={damage_pool_damage_prob}, damage_mode={damage_mode}")
    if knockout_diversity is not None and knockout_diversity > 0:
        log.info(f"VOCAB CREATION DEBUG: Creating knockout pattern vocabulary with {knockout_diversity} patterns")
        
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
        
        log.info(f"VOCAB CREATION DEBUG: SUCCESS - Generated vocabulary with shape: {knockout_vocabulary.shape}")
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
                x_data,
                y_data,
                tuple(layer_sizes),  # Convert list to tuple for JAX static arguments
                n_message_steps,
                loss_type=loss_type,
                loss_key=loss_key,
                epoch=epoch,
                knockout_patterns=knockout_patterns,
            )

            *_, hard_loss, _, _, accuracy, hard_accuracy, _, _ = aux

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
                circuit_pool, avg_steps_reset = circuit_pool.reset_fraction(
                    reset_key,
                    reset_pool_fraction,
                    reset_pool.graphs,
                    reset_pool.wires,
                    reset_pool.logits,
                    None,
                    reset_strategy=reset_strategy,
                    combined_weights=combined_weights,
                )

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
                log.info(f"DAMAGE TRIGGER DEBUG: Damage applied at epoch {epoch} (vocabulary_exists={knockout_vocabulary is not None})")
                log.info(f"UNIFIED DAMAGE CONTROL: mode={damage_injection_mode}, max_damage_per_circuit={max_damage_per_circuit}")
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
                    log.info(f"TRAINING DAMAGE DEBUG: Using ROLLING WINDOW mode (damage_mode={damage_mode})")
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
                    log.info(f"TRAINING DAMAGE DEBUG: Using VOCABULARY mode (vocab_size={knockout_vocabulary.shape[0]}, damage_mode={damage_mode})")
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
                circuit_pool = circuit_pool.apply_knockouts(damaged_idxs, new_patterns)

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
                losses.append(float(loss))
                hard_losses.append(float(hard_loss))
                accuracies.append(float(accuracy))
                hard_accuracies.append(float(hard_accuracy))
                reset_steps.append(float(avg_steps_reset))

                # Prepare training metrics for best model tracking
                training_metrics = {
                    "loss": float(loss),
                    "hard_loss": float(hard_loss),
                    "accuracy": float(accuracy),
                    "hard_accuracy": float(hard_accuracy),
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

                        bp_results = _run_backpropagation_training_with_knockouts(
                            mock_cfg, x_data, y_data, loss_type, knockout_vocabulary,
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

                    ko_eval_results, accumulated_pattern_data = run_knockout_periodic_evaluation(
                        model=model,
                        knockout_vocabulary=knockout_vocabulary,
                        base_wires=base_wires,
                        base_logits=base_logits,
                        knockout_config=local_knockout_eval,
                        periodic_eval_test_seed=periodic_eval_test_seed,
                        x_data=x_data,
                        y_data=y_data,
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
                    )
                    # Extract final metrics for best model tracking
                    if ko_eval_results and "final_metrics_in" in ko_eval_results:
                        all_eval_metrics.update(ko_eval_results["final_metrics_in"])
                        all_eval_metrics.update(ko_eval_results["final_metrics_out"])
                    
                    # Set current eval metrics to the combined dictionary if any evals ran
                    current_eval_metrics = ko_eval_results.get("final_metrics_in", None)
                

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
                    elif (best_metric_source == "eval" or best_metric_source == "eval_ko_in") and current_eval_metrics is None:
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
    if accumulated_pattern_data:
        try:
            from boolean_nca_cc.analysis.visualization import create_eval_plot_prog
            
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
                    log.info("Computing backpropagation results for Figure 1 reference line...")
                    bp_results = _run_backpropagation_training_with_knockouts(
                        mock_cfg, x_data, y_data, loss_type, knockout_vocabulary,
                        parallel=backprop_config.get("parallel", True),
                        batch_size=backprop_config.get("batch_size", None)
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
                wandb_run.log({"training_progress_final": wandb.Image(final_plot_path)})
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
            fig3 = plot_combined_bp_sa_stepwise_performance(
                cfg=mock_cfg,
                x_data=x_data,
                y_data=y_data,
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
                knockout_vocabulary=knockout_vocabulary,
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