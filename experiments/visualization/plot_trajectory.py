"""
Inner loop trajectory visualization.

This module provides unified plotting functions for inner loop trajectories,
handling both boolean discovery (no damage) and damage response trajectories.

Supports both:
1. Pure plotting from pre-computed metrics (plot_inner_loop_trajectory)
2. Checkpoint loading + evaluation + plotting (plot_trajectory_from_checkpoint)
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List, Dict, Tuple, Union
import logging
import jax
import jax.numpy as jp
import os
from omegaconf import OmegaConf

from .figure_config import (
    setup_style,
    create_figure,
    format_axis,
    format_suptitle,
    save_figure,
    COLORS,
    LINE_STYLES,
    FONT_SIZES,
)

log = logging.getLogger(__name__)


# ============================================================================
# Helper Functions
# ============================================================================

def replicate_circuit(base_wires, base_logits, batch_size):
    """
    Replicate a single circuit to create a batch of identical circuits.
    
    Args:
        base_wires: Single circuit wires (PyTree)
        base_logits: Single circuit logits (PyTree)
        batch_size: Number of copies to create
    
    Returns:
        Tuple of (batch_wires, batch_logits) with batch dimension added
    """
    batch_wires = jax.tree.map(
        lambda x: jp.repeat(x[None, ...], batch_size, axis=0), base_wires
    )
    batch_logits = jax.tree.map(
        lambda x: jp.repeat(x[None, ...], batch_size, axis=0), base_logits
    )
    return batch_wires, batch_logits


def generate_ood_patterns(
    knockout_patterns: Optional[jp.ndarray],
    layer_sizes: List[Tuple[int, int]],
    damage_prob: float,
    periodic_eval_test_seed: int,
    batch_size: Optional[int] = None,
) -> jp.ndarray:
    """
    Generate out-of-distribution knockout patterns for evaluation.
    
    Args:
        knockout_patterns: IN-distribution patterns (used to determine count if batch_size is None)
        layer_sizes: Circuit layer sizes
        damage_prob: Damage probability for pattern generation
        periodic_eval_test_seed: Seed for reproducible pattern generation
        batch_size: Number of patterns to generate (uses len(knockout_patterns) if None)
    
    Returns:
        Array of OOD knockout patterns
    """
    from functools import partial
    from boolean_nca_cc.training.pool.structural_perturbation import create_reproducible_knockout_pattern
    
    pattern_creator_fn = partial(
        create_reproducible_knockout_pattern,
        layer_sizes=layer_sizes,
        damage_prob=damage_prob,
    )
    
    num_patterns = batch_size if batch_size is not None else len(knockout_patterns)
    ood_rng = jax.random.PRNGKey(periodic_eval_test_seed + 1)
    out_pattern_keys = jax.random.split(ood_rng, num_patterns)
    out_knockout_patterns = jax.vmap(pattern_creator_fn)(out_pattern_keys)
    
    return out_knockout_patterns


def build_evaluation_params(
    model,
    x_data: jp.ndarray,
    y_data: jp.ndarray,
    input_n: int,
    arity: int,
    circuit_hidden_dim: int,
    n_message_steps: int,
    loss_type: str,
    layer_sizes: List[Tuple[int, int]],
    layer_neighbors: bool = False,
    return_per_pattern: bool = True,
    damage_mode: str = "greedy",
    damage_injection_mode: str = "single",
    max_damage_per_circuit: int = 10,
    greedy_ordered_indices: Optional[List[int]] = None,
    greedy_window_size: int = 1,
    greedy_injection_recover_steps: int = 10,
    damage_start_offset: int = 0,
    damage_start_offset_random: bool = False,
    damage_start_offset_seed: int = 42,
    knockout_vocabulary: Optional[jp.ndarray] = None,
) -> Dict:
    """
    Build common evaluation parameters dictionary to reduce repetition.
    
    Returns:
        Dictionary of evaluation parameters ready to be unpacked into evaluate_circuits_in_chunks
    """
    return {
        'model': model,
        'x_data': x_data,
        'y_data': y_data,
        'input_n': input_n,
        'arity': arity,
        'circuit_hidden_dim': circuit_hidden_dim,
        'n_message_steps': n_message_steps,
        'loss_type': loss_type,
        'layer_sizes': layer_sizes,
        'return_per_pattern': return_per_pattern,
        'layer_neighbors': layer_neighbors,
        'damage_mode': damage_mode,
        'damage_injection_mode': damage_injection_mode,
        'max_damage_per_circuit': max_damage_per_circuit,
        'greedy_ordered_indices': greedy_ordered_indices,
        'greedy_window_size': greedy_window_size,
        'greedy_injection_recover_steps': greedy_injection_recover_steps,
        'damage_start_offset': damage_start_offset,
        'damage_start_offset_random': damage_start_offset_random,
        'damage_start_offset_seed': damage_start_offset_seed,
        'knockout_vocabulary': knockout_vocabulary,
    }


def run_sa_evaluation(
    eval_fn,
    base_wires,
    base_logits,
    knockout_patterns: Optional[jp.ndarray],
    target_chunk_size: int,
    eval_params: Dict,
) -> Dict:
    """
    Unified SA evaluation function that handles both static and multi-damage modes.
    
    Args:
        eval_fn: Evaluation function (e.g., evaluate_model_stepwise_batched)
        base_wires: Single circuit wires
        base_logits: Single circuit logits
        knockout_patterns: Pre-generated patterns (None for multi-damage mode)
        target_chunk_size: Batch size for evaluation
        eval_params: Evaluation parameters dict (from build_evaluation_params)
    
    Returns:
        Step metrics dictionary
    """
    from boolean_nca_cc.training.evaluation import evaluate_circuits_in_chunks
    
    # Determine batch size
    if knockout_patterns is not None:
        batch_size = len(knockout_patterns)
    else:
        batch_size = target_chunk_size
    
    # Replicate circuit
    batch_wires, batch_logits = replicate_circuit(base_wires, base_logits, batch_size)
    
    # Run evaluation
    return evaluate_circuits_in_chunks(
        eval_fn=eval_fn,
        wires=batch_wires,
        logits=batch_logits,
        knockout_patterns=knockout_patterns,
        target_chunk_size=target_chunk_size,
        **eval_params,
    )


def get_bp_results(
    cfg,
    x_data: jp.ndarray,
    y_data: jp.ndarray,
    loss_type: str,
    knockout_patterns: jp.ndarray,
    bp_results: Optional[Dict] = None,
) -> Optional[Dict]:
    """
    Get backpropagation results, either from provided results or by running training.
    
    Args:
        cfg: Configuration object
        x_data: Input data
        y_data: Target data
        loss_type: Loss type ('l4' or 'bce')
        knockout_patterns: Knockout patterns for BP training
        bp_results: Pre-computed BP results (optional)
    
    Returns:
        BP results dictionary or None if not needed
    """
    if bp_results is not None:
        log.info("Using pre-computed bp_results")
        return bp_results
    
    from boolean_nca_cc.training.backprop import _run_backpropagation_training_with_knockouts
    
    if knockout_patterns is None:
        raise ValueError(
            "knockout_patterns is None but bp_results is also None. "
            "Either provide pre-computed bp_results or knockout_patterns for BP training."
        )
    
    log.info(f"Running BP training with knockout_patterns length: {len(knockout_patterns)}")
    return _run_backpropagation_training_with_knockouts(
        cfg, x_data, y_data, loss_type, knockout_patterns,
        parallel=getattr(cfg.backprop, 'parallel', True) if hasattr(cfg.backprop, 'parallel') else cfg.backprop.get("parallel", True) if hasattr(cfg.backprop, 'get') else True,
        batch_size=getattr(cfg.backprop, 'batch_size', None) if hasattr(cfg.backprop, 'batch_size') else cfg.backprop.get("batch_size", None) if hasattr(cfg.backprop, 'get') else None
    )


# ============================================================================
# Main Plotting Functions
# ============================================================================

def plot_inner_loop_trajectory(
    trajectory_type: str,  # "boolean_discovery" or "damage_response"
    # Data for boolean_discovery mode
    train_metrics: Optional[Dict] = None,
    test_metrics: Optional[Dict] = None,
    # Data for damage_response mode
    sa_step_metrics_in: Optional[Dict] = None,
    sa_step_metrics_out: Optional[Dict] = None,
    bp_results: Optional[Dict] = None,
    show_bp_trajectory: bool = False,
    show_ood_trajectory: bool = True,
    # Damage response specific parameters
    damage_injection_mode: str = "single",
    damage_start_offset: int = 0,
    max_damage_per_circuit: int = 10,
    greedy_injection_recover_steps: int = 10,
    training_mode: str = "growth",
    pre_damage_accuracy: Optional[float] = None,
    # Common parameters
    output_path: Optional[str] = None,
    title: Optional[str] = None,
    figsize: Optional[Tuple[float, float]] = None,
    dpi: int = 300,
) -> plt.Figure:
    """
    Unified inner loop trajectory visualization.
    
    Handles two types of trajectories:
    1. "boolean_discovery": No damage, shows train/test input generalization
       - Restructured to split by metric (like Figure 2)
       - Shows train/test curves within each metric subplot
    2. "damage_response": With damage, shows seen/unseen damage pattern generalization
       - Shows SA trajectories with optional BP comparison
       - Includes damage region shading for visualization
    
    Args:
        trajectory_type: "boolean_discovery" or "damage_response"
        train_metrics: For boolean_discovery - training split stepwise metrics
        test_metrics: For boolean_discovery - test split stepwise metrics
        sa_step_metrics_in: For damage_response - seen damage pattern metrics
        sa_step_metrics_out: For damage_response - unseen damage pattern metrics
        bp_results: For damage_response - backpropagation results
        show_bp_trajectory: For damage_response - show BP comparison
        show_ood_trajectory: For damage_response - show unseen patterns
        damage_injection_mode: "single" or "multi" for damage shading
        damage_start_offset: Step offset before first damage
        max_damage_per_circuit: Max damage events (for multi mode)
        greedy_injection_recover_steps: Recovery steps between damages
        training_mode: "growth" or "repair" (affects pre-damage line)
        pre_damage_accuracy: Pre-damage accuracy for reference line
        output_path: Path to save figure
        title: Figure title
        figsize: Custom figure size
        dpi: Image resolution
    
    Returns:
        matplotlib Figure object
    """
    # Setup standardized style
    setup_style()
    
    if trajectory_type == "boolean_discovery":
        # Boolean discovery: Split by metric, show train/test within each
        if train_metrics is None or test_metrics is None:
            raise ValueError("train_metrics and test_metrics required for boolean_discovery mode")
        
        # Extract step data
        steps_train = np.array(train_metrics["step"])
        steps_test = np.array(test_metrics["step"])
        
        # Extract metrics
        metrics_config = [
            {
                'key': 'full_map_accuracy',
                'ylabel': 'Full Map Accuracy',
                'title': 'Full Map Accuracy',
            },
            {
                'key': 'hard_accuracy',
                'ylabel': 'Hard Accuracy',
                'title': 'Hard Accuracy',
            },
        ]
        
        # Create figure with 2 subplots
        fig, axes = create_figure(
            width_type='double',
            nrows=1,
            ncols=2,
            figsize=figsize,
        )
        
        # Plot each metric in its subplot
        for ax, metric_cfg in zip(axes, metrics_config):
            metric_key = metric_cfg['key']
            train_values = np.array(train_metrics[metric_key])
            test_values = np.array(test_metrics[metric_key])
            
            # Plot train and test curves
            ax.plot(
                steps_train,
                train_values,
                color=COLORS['train'],
                linewidth=LINE_STYLES['linewidth_thick'],
                label='Train',
                marker='o',
                markersize=LINE_STYLES['markersize'],
                alpha=LINE_STYLES['alpha_mean'],
                markevery=max(1, len(steps_train) // 20),
            )
            ax.plot(
                steps_test,
                test_values,
                color=COLORS['test'],
                linewidth=LINE_STYLES['linewidth_thick'],
                label='Test',
                marker='s',
                markersize=LINE_STYLES['markersize'],
                alpha=LINE_STYLES['alpha_mean'],
                markevery=max(1, len(steps_test) // 20),
            )
            # ax.axhline(y=1.0, color=COLORS['reference'], linestyle=':', alpha=0.5, linewidth=1.5)
            
            format_axis(
                ax,
                xlabel='Evaluation Step',
                ylabel=metric_cfg['ylabel'],
                title=metric_cfg['title'],
                ylim=[0, 1.05],
                grid=True,
                legend=True,
                legend_loc='best',
            )
        
        # Overall title
        if title is None:
            title = "Boolean Function Discovery"
        format_suptitle(fig, title, y=1.02)
        
    elif trajectory_type == "damage_response":
        # Damage response: Single subplot with damage shading
        if sa_step_metrics_in is None:
            raise ValueError("sa_step_metrics_in required for damage_response mode")
        
        # Create single subplot figure
        fig, ax = create_figure(
            width_type='double',
            nrows=1,
            ncols=1,
            figsize=figsize,
        )
        
        # Extract SA metrics
        sa_steps = np.array(sa_step_metrics_in["step"])
        sa_hard_accuracies = sa_step_metrics_in["hard_accuracy"]
        
        # Plot SA IN-distribution trajectory with error bands
        if 'per_pattern' in sa_step_metrics_in and 'pattern_hard_accuracies' in sa_step_metrics_in['per_pattern']:
            per_pattern_accuracies = np.array(sa_step_metrics_in['per_pattern']['pattern_hard_accuracies'])
            sa_mean_accuracies = np.mean(per_pattern_accuracies, axis=1)
            sa_std_accuracies = np.std(per_pattern_accuracies, axis=1)
            
            ax.plot(
                sa_steps,
                sa_mean_accuracies,
                color=COLORS['train'],  # Seen patterns = train color
                linewidth=LINE_STYLES['linewidth_thick'],
                alpha=LINE_STYLES['alpha_mean'],
                label='Trajectory (Seen)',
            )
            ax.fill_between(
                sa_steps,
                sa_mean_accuracies - sa_std_accuracies,
                sa_mean_accuracies + sa_std_accuracies,
                color=COLORS['train'],
                alpha=LINE_STYLES['alpha_std'],
            )
        else:
            ax.plot(
                sa_steps,
                sa_hard_accuracies,
                color=COLORS['train'],
                linewidth=LINE_STYLES['linewidth_thick'],
                alpha=LINE_STYLES['alpha_mean'],
                label='Trajectory (Seen)',
            )
        
        # Plot SA OUT-of-distribution trajectory if available
        if show_ood_trajectory and sa_step_metrics_out is not None:
            ood_sa_steps = np.array(sa_step_metrics_out["step"])
            ood_sa_hard_accuracies = sa_step_metrics_out["hard_accuracy"]
            
            if 'per_pattern' in sa_step_metrics_out and 'pattern_hard_accuracies' in sa_step_metrics_out['per_pattern']:
                ood_per_pattern_accuracies = np.array(sa_step_metrics_out['per_pattern']['pattern_hard_accuracies'])
                ood_sa_mean_accuracies = np.mean(ood_per_pattern_accuracies, axis=1)
                ood_sa_std_accuracies = np.std(ood_per_pattern_accuracies, axis=1)
                
                ax.plot(
                    ood_sa_steps,
                    ood_sa_mean_accuracies,
                    color=COLORS['test'],  # Unseen patterns = test color
                    linewidth=LINE_STYLES['linewidth_thick'],
                    alpha=LINE_STYLES['alpha_mean'],
                    label='Trajectory (Unseen)',
                )
                ax.fill_between(
                    ood_sa_steps,
                    ood_sa_mean_accuracies - ood_sa_std_accuracies,
                    ood_sa_mean_accuracies + ood_sa_std_accuracies,
                    color=COLORS['test'],
                    alpha=LINE_STYLES['alpha_std'],
                )
            else:
                ax.plot(
                    ood_sa_steps,
                    ood_sa_hard_accuracies,
                    color=COLORS['test'],
                    linewidth=LINE_STYLES['linewidth_thick'],
                    alpha=LINE_STYLES['alpha_mean'],
                    label='Trajectory (Unseen)',
                )
        
        # Plot BP trajectory if requested
        if show_bp_trajectory and bp_results is not None:
            bp_accuracies_list = [pattern_results["hard_accuracies"] for pattern_results in bp_results["patterns_performance"]]
            bp_accuracies_array = np.array(bp_accuracies_list)
            
            # Add pre-damage state to BP trajectory
            if pre_damage_accuracy is not None:
                bp_initial_accuracy = pre_damage_accuracy
            else:
                bp_initial_accuracy = sa_hard_accuracies[0] if len(sa_hard_accuracies) > 0 else 0.0
            
            bp_accuracies_with_initial = np.column_stack([
                np.full((bp_accuracies_array.shape[0], 1), bp_initial_accuracy),
                bp_accuracies_array
            ])
            
            bp_mean_accuracies = np.mean(bp_accuracies_with_initial, axis=0)
            bp_std_accuracies = np.std(bp_accuracies_with_initial, axis=0)
            bp_steps = range(len(bp_mean_accuracies))
            
            # Truncate to match SA length if needed
            if len(sa_steps) != len(bp_steps):
                min_steps = min(len(sa_steps), len(bp_steps))
                bp_steps = range(min_steps)
                bp_mean_accuracies = bp_mean_accuracies[:min_steps]
                bp_std_accuracies = bp_std_accuracies[:min_steps]
            
            ax.plot(
                bp_steps,
                bp_mean_accuracies,
                color=COLORS['full_map'],  # Blue for BP
                linewidth=LINE_STYLES['linewidth_thick'],
                alpha=LINE_STYLES['alpha_mean'],
                label='Backpropagation',
            )
            ax.fill_between(
                bp_steps,
                bp_mean_accuracies - bp_std_accuracies,
                bp_mean_accuracies + bp_std_accuracies,
                color=COLORS['full_map'],
                alpha=LINE_STYLES['alpha_std'],
            )
        # elif pre_damage_accuracy is not None and training_mode == "repair":
        #     # Show pre-damage reference line in repair mode
        #     ax.axhline(
        #         y=pre_damage_accuracy,
        #         color=COLORS['reference'],
        #         linestyle='--',
        #         linewidth=LINE_STYLES['linewidth_thick'],
        #         alpha=0.8,
        #         label='Pre-damage Performance',
        #     )
        
        # Add damage region shading
        if damage_injection_mode == "single":
            first_damage_step = damage_start_offset + 1
            pre_damage_end = first_damage_step
            damage_start = first_damage_step
            damage_end = first_damage_step + 1
            
            # Pre-damage region
            ax.axvspan(0, pre_damage_end, alpha=0.1, color='#377eb8')
            # Damage region
            ax.axvspan(damage_start, damage_end, alpha=0.1, color='#ff7f00')
            # Recovery region
            recovery_start = damage_end
            recovery_end = max(sa_steps[-1], ax.get_xlim()[1]) if len(sa_steps) > 0 else ax.get_xlim()[1]
            ax.axvspan(recovery_start, recovery_end, alpha=0.1, color='#4daf4a')
        
        # Format axis
        if title is None:
            if show_bp_trajectory:
                title = "Hard Accuracy Over Steps: SA vs Backpropagation"
            else:
                title = "Reconfiguration Trajectory"
        
        # Set xlim based on damage mode
        if damage_injection_mode == "multi":
            max_x = damage_start_offset + max_damage_per_circuit * (greedy_injection_recover_steps + 1) + 20
            if show_bp_trajectory:
                ax.set_xlim(0, max(60, max_x))
            else:
                ax.set_xlim(0, max_x)
        else:
            if show_bp_trajectory:
                ax.set_xlim(0, 60)
            else:
                first_damage_step = damage_start_offset + 1
                max_x = max(len(sa_steps) - 1, first_damage_step + 15) if len(sa_steps) > 0 else first_damage_step + 15
                ax.set_xlim(0, max_x)
        
        format_axis(
            ax,
            xlabel='Message Steps',
            ylabel='Hard Accuracy',
            title=title,
            ylim=[0.6, 1.05],
            grid=True,
            legend=True,
            legend_loc='lower right',
        )
        
        # Set y-axis ticks
        y_ticks = np.arange(0.6, 1.03, 0.05)
        ax.set_yticks(y_ticks)
        
    else:
        raise ValueError(f"Unknown trajectory_type: {trajectory_type}. Must be 'boolean_discovery' or 'damage_response'")
    
    plt.tight_layout()
    
    # Save if output path provided
    if output_path:
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        save_figure(output_path, fig=fig)
    
    return fig


def plot_trajectory_from_checkpoint(
    run_id: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
    use_best_model: bool = True,
    trajectory_type: str = "damage_response",  # "boolean_discovery" or "damage_response"
    # Boolean discovery mode parameters
    eval_on_train: bool = True,  # If True, also evaluate on train split for boolean_discovery
    # Damage response mode parameters
    show_bp_trajectory: bool = False,
    show_ood_trajectory: bool = True,
    damage_injection_mode: str = "single",  # "single" or "multi"
    damage_mode: str = "greedy",  # "greedy", "greedy_vocabulary", "shotgun", "strip"
    damage_start_offset: int = 0,
    max_damage_per_circuit: int = 10,
    greedy_injection_recover_steps: int = 10,
    greedy_ordered_indices: Optional[List[int]] = None,
    greedy_window_size: int = 1,
    knockout_vocabulary: Optional[jp.ndarray] = None,
    knockout_patterns: Optional[jp.ndarray] = None,  # For static damage modes
    knockout_config: Optional[Dict] = None,  # For OOD pattern generation
    # Evaluation parameters
    n_message_steps: Optional[int] = None,  # Override config default
    eval_batch_size: Optional[int] = None,  # Override config default
    periodic_eval_test_seed: int = 42,
    # Output parameters
    output_path: Optional[str] = None,
    title: Optional[str] = None,
    figsize: Optional[Tuple[float, float]] = None,
    dpi: int = 300,
    # WandB parameters
    project: str = "boolean-nca-cc",
    entity: str = "marcello-barylli-growai",
    # Damage behavior override
    force_damage_behavior: Optional[str] = None,  # "reversible" or "permanent" to override model's default
) -> plt.Figure:
    """
    Load model from checkpoint, run inner loop evaluation, and plot trajectory.
    
    This function provides a complete workflow: checkpoint loading → evaluation → plotting.
    Supports all trajectory modulations (reversible/permanent, single/multi damage, etc.).
    
    Args:
        run_id: WandB run ID to load model from (mutually exclusive with checkpoint_path)
        checkpoint_path: Local checkpoint path (.pkl file) (mutually exclusive with run_id)
        use_best_model: If True, load best model based on config checkpoint settings (WandB only)
        trajectory_type: "boolean_discovery" (no damage) or "damage_response" (with damage)
        eval_on_train: For boolean_discovery, also evaluate on train split
        show_bp_trajectory: For damage_response, show backpropagation comparison
        show_ood_trajectory: For damage_response, show out-of-distribution (unseen) patterns
        damage_injection_mode: "single" (one damage per circuit) or "multi" (multiple damages)
        damage_mode: Pattern type ("greedy", "greedy_vocabulary", "shotgun", "strip")
        damage_start_offset: Number of steps before first damage injection
        max_damage_per_circuit: Maximum damage events per circuit (for multi mode)
        greedy_injection_recover_steps: Recovery steps between damage injections
        greedy_ordered_indices: Ordered indices for greedy damage patterns (required for damage_mode="greedy")
        greedy_window_size: Window size for greedy patterns
        knockout_vocabulary: Vocabulary of patterns for seen evaluation (None for unseen)
        knockout_patterns: Pre-generated patterns for static damage modes
        knockout_config: Configuration for knockout evaluation (needed for OOD pattern generation)
        n_message_steps: Override number of message passing steps (uses config default if None)
        eval_batch_size: Override evaluation batch size (uses config default if None)
        periodic_eval_test_seed: Seed for generating OOD patterns
        output_path: Path to save figure
        title: Figure title
        figsize: Custom figure size
        dpi: Image resolution
        project: WandB project name
        entity: WandB entity/username
    
    Returns:
        matplotlib Figure object
    """
    from boolean_nca_cc.training.checkpointing import (
        load_config_from_wandb,
        load_model_from_config_and_checkpoint,
        derive_checkpoint_metric_from_config,
        load_checkpoint,
        instantiate_model_from_config,
    )
    from boolean_nca_cc.circuits.tasks import get_task_data
    from boolean_nca_cc.circuits.data_split import split_input_combinations
    from boolean_nca_cc.circuits.model import gen_circuit, generate_layer_sizes
    from boolean_nca_cc.training.preconfigure import preconfigure_circuit_logits
    from boolean_nca_cc.training.evaluation import (
        evaluate_model_stepwise_batched,
        get_loss_from_wires_logits,
    )
    
    # Validate inputs
    if run_id is None and checkpoint_path is None:
        raise ValueError("Either run_id or checkpoint_path must be provided")
    if run_id is not None and checkpoint_path is not None:
        raise ValueError("Cannot specify both run_id and checkpoint_path")
    
    # Load config and model
    if checkpoint_path is not None:
        # Load from local checkpoint
        log.info(f"Loading model from local checkpoint: {checkpoint_path}")
        loaded = load_checkpoint(checkpoint_path)
        config = OmegaConf.create(loaded.get("config", {}))
        
        # Instantiate and load model
        model = instantiate_model_from_config(config, seed=config.get("seed", 0))
        from flax import nnx as _nnx
        _nnx.update(model, loaded["model"])
        
        log.info(f"Loaded model from checkpoint: {checkpoint_path}")
    else:
        # Load from WandB
        log.info(f"Loading model from WandB run_id: {run_id}")
        
        metric_name = None
        prefer_metric = None
        
        if use_best_model:
            # First, load config to get checkpoint settings
            temp_config, _, _ = load_config_from_wandb(
                run_id=run_id,
                filename="latest_checkpoint",
                select_by_best_metric=False,
                project=project,
                entity=entity,
            )
            
            # Derive metric name from config's checkpoint settings
            metric_name, prefer_metric = derive_checkpoint_metric_from_config(temp_config)
            log.info(f"Using checkpoint metric from config: {metric_name} (prefer: {prefer_metric})")
            
            # Load the actual best model
            config, checkpoint_path, run_id = load_config_from_wandb(
                run_id=run_id,
                filename="best_model",
                select_by_best_metric=True,
                metric_name=metric_name,
                prefer_metric=prefer_metric,
                project=project,
                entity=entity,
            )
        else:
            config, checkpoint_path, run_id = load_config_from_wandb(
                run_id=run_id,
                filename="latest_checkpoint",
                project=project,
                entity=entity,
            )
        
        log.info(f"Loaded config from checkpoint: {checkpoint_path}")
        
        # Load model from config and checkpoint
        model, loaded_dict = load_model_from_config_and_checkpoint(
            config=config,
            checkpoint_path=checkpoint_path,
            run_id=run_id,
            seed=0,
        )
        
        log.info("Model loaded successfully")
    
    # Get model's damage_behavior if available (for reversible/permanent)
    damage_behavior = getattr(model, "damage_behavior", "permanent")
    log.info(f"Model damage_behavior: {damage_behavior}")
    
    # Force damage behavior if requested (for compatibility with models trained with different behavior)
    if force_damage_behavior is not None:
        if force_damage_behavior not in ["reversible", "permanent"]:
            raise ValueError(f"force_damage_behavior must be 'reversible' or 'permanent', got '{force_damage_behavior}'")
        log.info(f"Force setting damage_behavior to: {force_damage_behavior}")
        setattr(model, "damage_behavior", force_damage_behavior)
        damage_behavior = force_damage_behavior
    
    # Generate data and splits
    log.info(f"Generating task data: {config.circuit.task}")
    x_data, y_data = get_task_data(
        task_name=config.circuit.task,
        case_n=2**config.circuit.input_bits,
        input_bits=config.circuit.input_bits,
        output_bits=config.circuit.output_bits,
    )
    
    # Split if enabled
    if config.eval.input_split_enabled:
        log.info(f"Splitting data: {config.eval.input_train_fraction*100:.0f}% train, "
                f"{(1-config.eval.input_train_fraction)*100:.0f}% test")
        x_train, y_train, x_test, y_test = split_input_combinations(
            x_data=x_data,
            y_data=y_data,
            train_fraction=config.eval.input_train_fraction,
            seed=config.eval.input_split_seed,
            shuffle=True,
        )
    else:
        x_train, y_train = x_data, y_data
        x_test, y_test = x_data, y_data
        log.info("Input split disabled - using all combinations")
    
    # Generate layer sizes
    if config.circuit.layer_sizes is None:
        layer_sizes = generate_layer_sizes(
            input_n=config.circuit.input_bits,
            output_n=config.circuit.output_bits,
            arity=config.circuit.arity,
            layer_n=config.circuit.num_layers,
        )
    else:
        layer_sizes = config.circuit.layer_sizes
    
    # Generate base circuit
    wiring_seed = OmegaConf.select(config, "wiring_fixed_key", default=None)
    if wiring_seed is None:
        wiring_seed = OmegaConf.select(config, "test_seed", default=42)
    wiring_key = jax.random.PRNGKey(wiring_seed)
    
    if config.training.training_mode == "growth":
        log.info("Growth mode: generating random circuit")
        base_wires, base_logits = gen_circuit(
            wiring_key,
            layer_sizes,
            arity=config.circuit.arity
        )
    elif config.training.training_mode == "repair":
        log.info("Repair mode: preconfiguring circuit")
        backprop_config = OmegaConf.select(config, "backprop", default={})
        base_wires, base_logits = preconfigure_circuit_logits(
            wiring_key=wiring_key,
            layer_sizes=layer_sizes,
            arity=config.circuit.arity,
            x_data=x_data,
            y_data=y_data,
            loss_type=config.training.loss_type,
            steps=OmegaConf.select(config.training, "preconfig_steps", default=200),
            lr=OmegaConf.select(config.training, "preconfig_lr", default=1e-2),
            optimizer=OmegaConf.select(backprop_config, "optimizer", default="adam"),
            weight_decay=OmegaConf.select(backprop_config, "weight_decay", default=0.0),
            beta1=OmegaConf.select(backprop_config, "beta1", default=0.9),
            beta2=OmegaConf.select(backprop_config, "beta2", default=0.999),
        )
    else:
        raise ValueError(f"Unknown training_mode: {config.training.training_mode}")
    
    log.info("Base circuit generated")
    
    # Get evaluation parameters (use overrides if provided, otherwise from config)
    n_steps = n_message_steps if n_message_steps is not None else config.eval.periodic_eval_inner_steps
    batch_size = eval_batch_size if eval_batch_size is not None else config.eval.periodic_eval_batch_size
    
    # Determine trajectory type and run appropriate evaluation
    if trajectory_type == "boolean_discovery":
        # Boolean discovery: No damage, evaluate on train/test splits
        log.info("Running boolean discovery evaluation (no damage)")
        
        # Build evaluation parameters (no damage)
        eval_params = build_evaluation_params(
            model=model,
            x_data=x_test,  # Will override per split
            y_data=y_test,
            input_n=config.circuit.input_bits,
            arity=config.circuit.arity,
            circuit_hidden_dim=config.circuit.circuit_hidden_dim,
            n_message_steps=n_steps,
            loss_type=config.training.loss_type,
            layer_sizes=layer_sizes,
            layer_neighbors=config.training.layer_neighbors,
            return_per_pattern=False,
            damage_mode="greedy",  # Won't matter (no damage)
            damage_injection_mode="single",
            max_damage_per_circuit=1,
        )
        
        # Evaluate on test split
        test_eval_params = eval_params.copy()
        test_eval_params['x_data'] = x_test
        test_eval_params['y_data'] = y_test
        step_metrics_test = run_sa_evaluation(
            eval_fn=evaluate_model_stepwise_batched,
            base_wires=base_wires,
            base_logits=base_logits,
            knockout_patterns=None,  # No damage
            target_chunk_size=batch_size,
            eval_params=test_eval_params,
        )
        
        # Evaluate on train split if requested
        if eval_on_train and config.eval.input_split_enabled:
            log.info("Running evaluation on train split")
            train_eval_params = eval_params.copy()
            train_eval_params['x_data'] = x_train
            train_eval_params['y_data'] = y_train
            step_metrics_train = run_sa_evaluation(
                eval_fn=evaluate_model_stepwise_batched,
                base_wires=base_wires,
                base_logits=base_logits,
                knockout_patterns=None,  # No damage
                target_chunk_size=batch_size,
                eval_params=train_eval_params,
            )
        else:
            # Use test metrics for both if no train split
            step_metrics_train = step_metrics_test
        
        # Plot boolean discovery trajectory
        fig = plot_inner_loop_trajectory(
            trajectory_type="boolean_discovery",
            train_metrics=step_metrics_train,
            test_metrics=step_metrics_test,
            output_path=output_path,
            title=title,
            figsize=figsize,
            dpi=dpi,
        )
        
    elif trajectory_type == "damage_response":
        # Damage response: With damage, evaluate seen/unseen patterns
        log.info("Running damage response evaluation")
        
        # Get pre-damage accuracy
        _, base_aux = get_loss_from_wires_logits(base_logits, base_wires, x_data, y_data, config.training.loss_type)
        pre_damage_accuracy = float(base_aux[4])  # hard_accuracy is index 4
        
        # Determine batch size and evaluation approach based on damage mode
        if damage_mode in ["greedy", "greedy_vocabulary"] and damage_injection_mode == "multi":
            # Multi-damage mode: Use dynamic evaluation
            if damage_mode == "greedy" and greedy_ordered_indices is None:
                raise ValueError(
                    "greedy_ordered_indices is None but required for damage_mode='greedy'. "
                    "Either provide greedy_ordered_indices or use a different damage_mode."
                )
            eval_batch = max(10, len(knockout_patterns) if knockout_patterns is not None else batch_size)
            knockout_patterns_for_eval = None  # Let evaluation system handle dynamic patterns
        else:
            # Static damage mode: Use pre-generated patterns
            if knockout_patterns is None:
                raise ValueError("knockout_patterns must be provided for static damage modes")
            eval_batch = len(knockout_patterns)
            knockout_patterns_for_eval = knockout_patterns
        
        # Build evaluation parameters
        eval_params = build_evaluation_params(
            model=model,
            x_data=x_data,
            y_data=y_data,
            input_n=config.circuit.input_bits,
            arity=config.circuit.arity,
            circuit_hidden_dim=config.circuit.circuit_hidden_dim,
            n_message_steps=n_steps,
            loss_type=config.training.loss_type,
            layer_sizes=layer_sizes,
            layer_neighbors=config.training.layer_neighbors,
            return_per_pattern=True,
            damage_mode=damage_mode,
            damage_injection_mode=damage_injection_mode,
            max_damage_per_circuit=max_damage_per_circuit,
            greedy_ordered_indices=greedy_ordered_indices,
            greedy_window_size=greedy_window_size,
            greedy_injection_recover_steps=greedy_injection_recover_steps,
            damage_start_offset=damage_start_offset,
            damage_start_offset_random=False,
            damage_start_offset_seed=42,
            knockout_vocabulary=knockout_vocabulary,
        )
        
        # Run SA evaluation on IN-distribution patterns
        log.info("Running SA evaluation on IN-distribution patterns")
        sa_step_metrics_in = run_sa_evaluation(
            eval_fn=evaluate_model_stepwise_batched,
            base_wires=base_wires,
            base_logits=base_logits,
            knockout_patterns=knockout_patterns_for_eval,
            target_chunk_size=eval_batch,
            eval_params=eval_params,
        )
        
        # Run SA evaluation on OUT-of-distribution patterns if requested
        sa_step_metrics_out = None
        if show_ood_trajectory and knockout_config is not None:
            log.info("Running SA evaluation on OUT-of-distribution patterns")
            
            # Generate OOD patterns
            if knockout_patterns is not None:
                ood_batch_size = len(knockout_patterns)
            else:
                ood_batch_size = eval_batch
            
            out_knockout_patterns = generate_ood_patterns(
                knockout_patterns=knockout_patterns,
                layer_sizes=layer_sizes,
                damage_prob=knockout_config["damage_prob"],
                periodic_eval_test_seed=periodic_eval_test_seed,
                batch_size=ood_batch_size,
            )
            
            # Build OOD evaluation parameters (force unseen by not providing vocabulary)
            ood_eval_params = eval_params.copy()
            ood_eval_params['knockout_vocabulary'] = None  # Force unseen patterns
            
            # Run SA evaluation on OOD patterns
            sa_step_metrics_out = run_sa_evaluation(
                eval_fn=evaluate_model_stepwise_batched,
                base_wires=base_wires,
                base_logits=base_logits,
                knockout_patterns=out_knockout_patterns,
                target_chunk_size=len(out_knockout_patterns),
                eval_params=ood_eval_params,
            )
        
        # Run BP evaluation if requested
        bp_results = None
        if show_bp_trajectory:
            log.info("Running backpropagation training for comparison")
            if knockout_patterns is None:
                raise ValueError("knockout_patterns required for BP trajectory")
            bp_results = get_bp_results(
                config, x_data, y_data, config.training.loss_type, knockout_patterns, None
            )
        
        # Plot damage response trajectory
        fig = plot_inner_loop_trajectory(
            trajectory_type="damage_response",
            sa_step_metrics_in=sa_step_metrics_in,
            sa_step_metrics_out=sa_step_metrics_out,
            bp_results=bp_results,
            show_bp_trajectory=show_bp_trajectory,
            show_ood_trajectory=show_ood_trajectory,
            damage_injection_mode=damage_injection_mode,
            damage_start_offset=damage_start_offset,
            max_damage_per_circuit=max_damage_per_circuit,
            greedy_injection_recover_steps=greedy_injection_recover_steps,
            training_mode=config.training.training_mode,
            pre_damage_accuracy=pre_damage_accuracy,
            output_path=output_path,
            title=title,
            figsize=figsize,
            dpi=dpi,
        )
    else:
        raise ValueError(f"Unknown trajectory_type: {trajectory_type}. Must be 'boolean_discovery' or 'damage_response'")
    
    return fig

