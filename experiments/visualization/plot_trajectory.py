"""
Inner loop trajectory visualization.

This module provides unified plotting functions for inner loop trajectories,
handling both boolean discovery (no damage) and damage response trajectories.

Supports both:
1. Pure plotting from pre-computed metrics (plot_inner_loop_trajectory)
2. Checkpoint loading + evaluation + plotting (plot_trajectory_from_checkpoint)

The evaluation wrappers are designed to wrap evaluate_with_loop (via evaluate_model_stepwise_batched),
allowing checkpoint loading and different evaluation modes (reversible/permanent, growth/repair, single/multi-damage).
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List, Dict, Tuple, Union
import logging
import jax
import jax.numpy as jp
import os
from omegaconf import OmegaConf
from functools import partial

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


def generate_knockout_vocabulary(
    layer_sizes: List[Tuple[int, int]],
    damage_prob: int,
    vocabulary_size: int,
    seed: int,
) -> jp.ndarray:
    """
    Generate a vocabulary of knockout patterns matching training configuration.
    
    This generates patterns using the same logic as training, ensuring ID/OOD
    evaluation uses patterns consistent with how the model was trained.
    
    Args:
        layer_sizes: Circuit layer sizes
        damage_prob: Number of gates to knock out per pattern
        vocabulary_size: Number of patterns in vocabulary
        seed: Seed for reproducible pattern generation
    
    Returns:
        Array of knockout patterns [vocabulary_size, total_nodes]
    """
    from boolean_nca_cc.training.pool.structural_perturbation import create_reproducible_knockout_pattern
    
    pattern_creator_fn = partial(
        create_reproducible_knockout_pattern,
        layer_sizes=layer_sizes,
        damage_prob=damage_prob,
    )
    
    vocab_rng = jax.random.PRNGKey(seed)
    vocab_pattern_keys = jax.random.split(vocab_rng, vocabulary_size)
    vocabulary = jax.vmap(pattern_creator_fn)(vocab_pattern_keys)
    
    return vocabulary


def sample_id_patterns(
    knockout_vocabulary: Optional[jp.ndarray],
    layer_sizes: List[Tuple[int, int]],
    damage_prob: int,
    batch_size: int,
    periodic_eval_test_seed: int,
) -> jp.ndarray:
    """
    Sample IN-distribution knockout patterns.
    
    If vocabulary is provided, samples from it (seen patterns).
    Otherwise, generates fresh patterns using the training seed (unseen but same distribution).
    
    Args:
        knockout_vocabulary: Optional vocabulary of patterns (if None, generates fresh)
        layer_sizes: Circuit layer sizes
        damage_prob: Number of gates to knock out per pattern
        batch_size: Number of patterns to sample/generate
        periodic_eval_test_seed: Seed for evaluation
    
    Returns:
        Array of IN-distribution knockout patterns [batch_size, total_nodes]
    """
    if knockout_vocabulary is not None:
        # Sample from vocabulary (seen patterns)
        id_rng = jax.random.PRNGKey(periodic_eval_test_seed)
        pattern_indices = jax.random.choice(
            id_rng, len(knockout_vocabulary), shape=(batch_size,), replace=True
        )
        return knockout_vocabulary[pattern_indices]
    else:
        # Generate fresh patterns (unseen but same distribution as training)
        from boolean_nca_cc.training.pool.structural_perturbation import create_reproducible_knockout_pattern
        
        pattern_creator_fn = partial(
            create_reproducible_knockout_pattern,
            layer_sizes=layer_sizes,
            damage_prob=damage_prob,
        )
        
        id_rng = jax.random.PRNGKey(periodic_eval_test_seed)
        in_pattern_keys = jax.random.split(id_rng, batch_size)
        return jax.vmap(pattern_creator_fn)(in_pattern_keys)


def generate_ood_patterns(
    layer_sizes: List[Tuple[int, int]],
    damage_prob: int,
    batch_size: int,
    periodic_eval_test_seed: int,
) -> jp.ndarray:
    """
    Generate OUT-of-distribution knockout patterns.
    
    Always generates fresh patterns using test_seed + 1 to ensure they are
    different from any training patterns.
    
    Args:
        layer_sizes: Circuit layer sizes
        damage_prob: Number of gates to knock out per pattern
        batch_size: Number of patterns to generate
        periodic_eval_test_seed: Seed for evaluation (OOD uses seed + 1)
    
    Returns:
        Array of OOD knockout patterns [batch_size, total_nodes]
    """
    from boolean_nca_cc.training.pool.structural_perturbation import create_reproducible_knockout_pattern
    
    pattern_creator_fn = partial(
        create_reproducible_knockout_pattern,
        layer_sizes=layer_sizes,
        damage_prob=damage_prob,
    )
    
    ood_rng = jax.random.PRNGKey(periodic_eval_test_seed + 1)
    out_pattern_keys = jax.random.split(ood_rng, batch_size)
    return jax.vmap(pattern_creator_fn)(out_pattern_keys)


def evaluate_with_damage(
    model,
    base_wires,
    base_logits,
    x_data: jp.ndarray,
    y_data: jp.ndarray,
    input_n: int,
    arity: int,
    circuit_hidden_dim: int,
    n_message_steps: int,
    loss_type: str,
    layer_sizes: List[Tuple[int, int]],
    layer_neighbors: bool = False,
    knockout_patterns: Optional[jp.ndarray] = None,
    knockout_vocabulary: Optional[jp.ndarray] = None,
    eval_batch_size: int = 256,
    return_per_pattern: bool = True,
    # Damage control parameters
    damage_mode: str = "greedy",
    damage_injection_mode: str = "single",
    max_damage_per_circuit: int = 10,
    greedy_ordered_indices: Optional[List[int]] = None,
    greedy_window_size: int = 1,
    greedy_injection_recover_steps: int = 10,
    damage_start_offset: int = 0,
    damage_start_offset_random: bool = False,
    damage_start_offset_seed: int = 42,
    blind_mode: bool = False,
) -> Dict:
    """
    Wrapper around evaluate_with_loop (via evaluate_model_stepwise_batched) for damage evaluation.
    
    This function handles checkpoint loading, circuit replication, and calls the evaluation
    function with appropriate parameters. It supports all damage modes and injection strategies.
    
    Args:
        model: Trained model to evaluate
        base_wires: Single circuit wires
        base_logits: Single circuit logits
        x_data: Input data
        y_data: Target data
        input_n: Number of inputs
        arity: Circuit arity
        circuit_hidden_dim: Circuit hidden dimension
        n_message_steps: Number of message passing steps
        loss_type: Loss function type
        layer_sizes: Circuit layer sizes
        layer_neighbors: Whether to use layer neighbors
        knockout_patterns: Pre-generated patterns (None for dynamic generation)
        knockout_vocabulary: Vocabulary for seen patterns (None for unseen)
        eval_batch_size: Batch size for evaluation
        return_per_pattern: Whether to return per-pattern metrics
        damage_mode: Pattern type ("greedy", "greedy_vocabulary", "shotgun", "strip")
        damage_injection_mode: "single" or "multi"
        max_damage_per_circuit: Maximum damage events per circuit
        greedy_ordered_indices: Ordered indices for greedy damage
        greedy_window_size: Window size for greedy patterns (damage injection timing only)
        greedy_injection_recover_steps: Recovery steps between injections
        damage_start_offset: Steps before first damage
        damage_start_offset_random: Randomize offset per circuit
        damage_start_offset_seed: Seed for random offset
        blind_mode: Zero out loss feedback (ablation)
    
    Returns:
        Dictionary with step-wise metrics
    """
    from boolean_nca_cc.training.evaluation import (
        evaluate_model_stepwise_batched,
        evaluate_circuits_in_chunks,
    )
    
    # Determine batch size
    if knockout_patterns is not None:
        batch_size = len(knockout_patterns)
    else:
        batch_size = eval_batch_size
    
    # Replicate circuit for batch
    batch_wires, batch_logits = replicate_circuit(base_wires, base_logits, batch_size)
    
    # Run evaluation
    return evaluate_circuits_in_chunks(
        eval_fn=evaluate_model_stepwise_batched,
        wires=batch_wires,
        logits=batch_logits,
        knockout_patterns=knockout_patterns,
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
        return_per_pattern=return_per_pattern,
        layer_neighbors=layer_neighbors,
        damage_mode=damage_mode,
        damage_injection_mode=damage_injection_mode,
        max_damage_per_circuit=max_damage_per_circuit,
        greedy_ordered_indices=greedy_ordered_indices,
        greedy_window_size=greedy_window_size,
        greedy_injection_recover_steps=greedy_injection_recover_steps,
        damage_start_offset=damage_start_offset,
        damage_start_offset_random=damage_start_offset_random,
        damage_start_offset_seed=damage_start_offset_seed,
        knockout_vocabulary=knockout_vocabulary,
        blind_mode=blind_mode,
    )


def evaluate_id_ood(
    model,
    base_wires,
    base_logits,
    x_data: jp.ndarray,
    y_data: jp.ndarray,
    input_n: int,
    arity: int,
    circuit_hidden_dim: int,
    n_message_steps: int,
    loss_type: str,
    layer_sizes: List[Tuple[int, int]],
    layer_neighbors: bool = False,
    knockout_vocabulary: Optional[jp.ndarray] = None,
    damage_prob: int = 40,
    eval_batch_size: int = 256,
    periodic_eval_test_seed: int = 42,
    return_per_pattern: bool = True,
    # Damage control parameters (from knockout_config)
    damage_mode: str = "greedy",
    damage_injection_mode: str = "single",
    max_damage_per_circuit: int = 10,
    greedy_ordered_indices: Optional[List[int]] = None,
    greedy_window_size: int = 1,
    greedy_injection_recover_steps: int = 10,
    damage_start_offset: int = 0,
    damage_start_offset_random: bool = False,
    damage_start_offset_seed: int = 42,
    blind_mode: bool = False,
) -> Tuple[Dict, Dict]:
    """
    Evaluate model on IN-distribution and OUT-of-distribution damage patterns.
    
    This follows the exact pattern from run_knockout_periodic_evaluation:
    - IN: Uses vocabulary if provided (seen), otherwise generates fresh with training seed
    - OUT: Always generates fresh patterns with test_seed + 1 (unseen)
    
    Args:
        model: Trained model to evaluate
        base_wires: Single circuit wires
        base_logits: Single circuit logits
        x_data: Input data
        y_data: Target data
        input_n: Number of inputs
        arity: Circuit arity
        circuit_hidden_dim: Circuit hidden dimension
        n_message_steps: Number of message passing steps
        loss_type: Loss function type
        layer_sizes: Circuit layer sizes
        layer_neighbors: Whether to use layer neighbors
        knockout_vocabulary: Vocabulary of seen patterns (None for unseen)
        damage_prob: Number of gates to knock out per pattern
        eval_batch_size: Batch size for evaluation
        periodic_eval_test_seed: Seed for evaluation
        return_per_pattern: Whether to return per-pattern metrics
        damage_mode: Pattern type
        damage_injection_mode: "single" or "multi"
        max_damage_per_circuit: Maximum damage events per circuit
        greedy_ordered_indices: Ordered indices for greedy damage
        greedy_window_size: Window size for greedy patterns
        greedy_injection_recover_steps: Recovery steps between injections
        damage_start_offset: Steps before first damage
        damage_start_offset_random: Randomize offset per circuit
        damage_start_offset_seed: Seed for random offset
        blind_mode: Zero out loss feedback (ablation)
    
    Returns:
        Tuple of (id_metrics, ood_metrics) dictionaries
    """
    # Generate IN-distribution patterns
    log.info(f"Generating IN-distribution patterns (batch_size={eval_batch_size})...")
    id_patterns = sample_id_patterns(
        knockout_vocabulary=knockout_vocabulary,
        layer_sizes=layer_sizes,
        damage_prob=damage_prob,
        batch_size=eval_batch_size,
        periodic_eval_test_seed=periodic_eval_test_seed,
    )
    
    # Evaluate IN-distribution
    log.info("Evaluating IN-distribution patterns...")
    id_metrics = evaluate_with_damage(
        model=model,
        base_wires=base_wires,
        base_logits=base_logits,
        x_data=x_data,
        y_data=y_data,
        input_n=input_n,
        arity=arity,
        circuit_hidden_dim=circuit_hidden_dim,
        n_message_steps=n_message_steps,
        loss_type=loss_type,
        layer_sizes=layer_sizes,
        layer_neighbors=layer_neighbors,
        knockout_patterns=id_patterns,
        knockout_vocabulary=knockout_vocabulary,  # Pass vocabulary for seen patterns
        eval_batch_size=eval_batch_size,
        return_per_pattern=return_per_pattern,
        damage_mode=damage_mode,
        damage_injection_mode=damage_injection_mode,
        max_damage_per_circuit=max_damage_per_circuit,
        greedy_ordered_indices=greedy_ordered_indices,
        greedy_window_size=greedy_window_size,
        greedy_injection_recover_steps=greedy_injection_recover_steps,
        damage_start_offset=damage_start_offset,
        damage_start_offset_random=damage_start_offset_random,
        damage_start_offset_seed=damage_start_offset_seed,
        blind_mode=blind_mode,
    )
    
    # Generate OUT-of-distribution patterns
    log.info(f"Generating OUT-of-distribution patterns (batch_size={eval_batch_size})...")
    ood_patterns = generate_ood_patterns(
        layer_sizes=layer_sizes,
        damage_prob=damage_prob,
        batch_size=eval_batch_size,
        periodic_eval_test_seed=periodic_eval_test_seed,
    )
    
    # Evaluate OUT-of-distribution (force unseen by not providing vocabulary)
    log.info("Evaluating OUT-of-distribution patterns...")
    ood_metrics = evaluate_with_damage(
        model=model,
        base_wires=base_wires,
        base_logits=base_logits,
        x_data=x_data,
        y_data=y_data,
        input_n=input_n,
        arity=arity,
        circuit_hidden_dim=circuit_hidden_dim,
        n_message_steps=n_message_steps,
        loss_type=loss_type,
        layer_sizes=layer_sizes,
        layer_neighbors=layer_neighbors,
        knockout_patterns=ood_patterns,
        knockout_vocabulary=None,  # Force unseen patterns
        eval_batch_size=eval_batch_size,
        return_per_pattern=return_per_pattern,
        damage_mode=damage_mode,
        damage_injection_mode=damage_injection_mode,
        max_damage_per_circuit=max_damage_per_circuit,
        greedy_ordered_indices=greedy_ordered_indices,
        greedy_window_size=greedy_window_size,
        greedy_injection_recover_steps=greedy_injection_recover_steps,
        damage_start_offset=damage_start_offset,
        damage_start_offset_random=damage_start_offset_random,
        damage_start_offset_seed=damage_start_offset_seed,
        blind_mode=blind_mode,
    )
    
    return id_metrics, ood_metrics


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
    # Evaluation parameters
    n_message_steps: Optional[int] = None,  # Override config default
    eval_batch_size: Optional[int] = None,  # Override config default
    periodic_eval_test_seed: Optional[int] = None,  # Override config default
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
    
    For ID/OOD evaluation, this follows the pattern from run_knockout_periodic_evaluation:
    - Generates vocabulary matching training configuration (damage_prob from config, seeds from config)
    - IN: Uses vocabulary (seen patterns)
    - OUT: Generates fresh patterns with test_seed + 1 (unseen patterns)
    
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
        greedy_window_size: Window size for greedy patterns (damage injection timing only)
        n_message_steps: Override number of message passing steps (uses config default if None)
        eval_batch_size: Override evaluation batch size (uses config default if None)
        periodic_eval_test_seed: Override evaluation seed (uses config default if None)
        output_path: Path to save figure
        title: Figure title
        figsize: Custom figure size
        dpi: Image resolution
        project: WandB project name
        entity: WandB entity/username
        force_damage_behavior: Override model's damage_behavior ("reversible" or "permanent")
    
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
    from boolean_nca_cc.training.evaluation import get_loss_from_wires_logits
    
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
        
        # Extract and log checkpoint metadata
        step = loaded.get("step")
        epoch = loaded.get("epoch")
        if epoch is None:
            # Try to get from config
            if isinstance(config, dict):
                epoch = config.get("epoch")
            else:
                epoch = getattr(config, "epoch", None)
        
        if step is not None:
            log.info(f"DEBUG: Loaded checkpoint at step: {step}")
        if epoch is not None:
            log.info(f"DEBUG: Loaded checkpoint at epoch: {epoch}")
        if step is None and epoch is None:
            log.warning("DEBUG: Could not determine checkpoint epoch/step from loaded data")
        
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
        
        # Extract and log checkpoint metadata (load_model_from_config_and_checkpoint already logs this,
        # but we'll also log it here for consistency and visibility)
        step = loaded_dict.get("step")
        epoch = loaded_dict.get("epoch")
        if epoch is None:
            checkpoint_config = loaded_dict.get("config", {})
            if isinstance(checkpoint_config, dict):
                epoch = checkpoint_config.get("epoch")
            else:
                epoch = getattr(checkpoint_config, "epoch", None)
        
        if step is not None:
            log.info(f"DEBUG: Loaded checkpoint at step: {step}")
        if epoch is not None:
            log.info(f"DEBUG: Loaded checkpoint at epoch: {epoch}")
        if step is None and epoch is None:
            log.warning("DEBUG: Could not determine checkpoint epoch/step from loaded data")
        
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
    test_seed = periodic_eval_test_seed if periodic_eval_test_seed is not None else config.eval.periodic_eval_test_seed
    
    # Determine trajectory type and run appropriate evaluation
    if trajectory_type == "boolean_discovery":
        # Boolean discovery: No damage, evaluate on train/test splits
        log.info("Running boolean discovery evaluation (no damage)")
        
        # Evaluate on test split
        step_metrics_test = evaluate_with_damage(
            model=model,
            base_wires=base_wires,
            base_logits=base_logits,
            x_data=x_test,
            y_data=y_test,
            input_n=config.circuit.input_bits,
            arity=config.circuit.arity,
            circuit_hidden_dim=config.circuit.circuit_hidden_dim,
            n_message_steps=n_steps,
            loss_type=config.training.loss_type,
            layer_sizes=layer_sizes,
            layer_neighbors=config.training.layer_neighbors,
            knockout_patterns=None,  # No damage
            knockout_vocabulary=None,
            eval_batch_size=batch_size,
            return_per_pattern=False,
            damage_mode="greedy",  # Won't matter (no damage)
            damage_injection_mode="single",
            max_damage_per_circuit=1,
        )
        
        # Evaluate on train split if requested
        if eval_on_train and config.eval.input_split_enabled:
            log.info("Running evaluation on train split")
            step_metrics_train = evaluate_with_damage(
                model=model,
                base_wires=base_wires,
                base_logits=base_logits,
                x_data=x_train,
                y_data=y_train,
                input_n=config.circuit.input_bits,
                arity=config.circuit.arity,
                circuit_hidden_dim=config.circuit.circuit_hidden_dim,
                n_message_steps=n_steps,
                loss_type=config.training.loss_type,
                layer_sizes=layer_sizes,
                layer_neighbors=config.training.layer_neighbors,
                knockout_patterns=None,  # No damage
                knockout_vocabulary=None,
                eval_batch_size=batch_size,
                return_per_pattern=False,
                damage_mode="greedy",  # Won't matter (no damage)
                damage_injection_mode="single",
                max_damage_per_circuit=1,
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
        
        # Get knockout configuration from config
        knockout_config = config.eval.knockout_eval if hasattr(config.eval, 'knockout_eval') else {}
        damage_prob = knockout_config.get("damage_prob", config.pool.damage_prob)
        
        # Generate vocabulary matching training configuration
        # This ensures ID evaluation uses patterns consistent with training
        knockout_vocabulary = None
        if hasattr(config.pool, 'damage_knockout_diversity') and config.pool.damage_knockout_diversity > 0:
            vocab_size = config.pool.damage_knockout_diversity
            # Use damage_seed from config for vocabulary generation (matches training)
            vocab_seed = config.get("damage_seed", config.get("test_seed", 42))
            log.info(f"Generating knockout vocabulary (size={vocab_size}, damage_prob={damage_prob}, seed={vocab_seed})")
            knockout_vocabulary = generate_knockout_vocabulary(
                layer_sizes=layer_sizes,
                damage_prob=damage_prob,
                vocabulary_size=vocab_size,
                seed=vocab_seed,
            )
        
        # Get damage control parameters from config or use defaults
        knockout_config_params = {
            "damage_mode": knockout_config.get("damage_mode", config.pool.get("damage_mode", "greedy")),
            "damage_injection_mode": knockout_config.get("damage_injection_mode", "single"),
            "max_damage_per_circuit": knockout_config.get("max_damage_per_circuit", 1),
            "greedy_ordered_indices": config.pool.get("greedy_ordered_indices", None),
            "greedy_window_size": knockout_config.get("greedy_window_size", config.pool.get("greedy_window_size", 1)),
            "greedy_injection_recover_steps": knockout_config.get("greedy_injection_recover_steps", 30),
            "damage_start_offset": knockout_config.get("damage_start_offset", 5),
            "damage_start_offset_random": knockout_config.get("damage_start_offset_random", False),
            "damage_start_offset_seed": knockout_config.get("damage_start_offset_seed", test_seed),
        }
        
        # Override with user-provided parameters if specified
        if damage_mode != "greedy" or damage_injection_mode != "single":
            knockout_config_params["damage_mode"] = damage_mode
            knockout_config_params["damage_injection_mode"] = damage_injection_mode
        if max_damage_per_circuit != 10:
            knockout_config_params["max_damage_per_circuit"] = max_damage_per_circuit
        if greedy_ordered_indices is not None:
            knockout_config_params["greedy_ordered_indices"] = greedy_ordered_indices
        if greedy_window_size != 1:
            knockout_config_params["greedy_window_size"] = greedy_window_size
        if greedy_injection_recover_steps != 10:
            knockout_config_params["greedy_injection_recover_steps"] = greedy_injection_recover_steps
        if damage_start_offset != 0:
            knockout_config_params["damage_start_offset"] = damage_start_offset
        
        # Evaluate ID and OOD patterns
        if show_ood_trajectory:
            log.info("Running ID/OOD evaluation...")
            sa_step_metrics_in, sa_step_metrics_out = evaluate_id_ood(
                model=model,
                base_wires=base_wires,
                base_logits=base_logits,
                x_data=x_data,
                y_data=y_data,
                input_n=config.circuit.input_bits,
                arity=config.circuit.arity,
                circuit_hidden_dim=config.circuit.circuit_hidden_dim,
                n_message_steps=n_steps,
                loss_type=config.training.loss_type,
                layer_sizes=layer_sizes,
                layer_neighbors=config.training.layer_neighbors,
                knockout_vocabulary=knockout_vocabulary,
                damage_prob=damage_prob,
                eval_batch_size=batch_size,
                periodic_eval_test_seed=test_seed,
                return_per_pattern=True,
                blind_mode=config.eval.get("blind_mode", False),
                **knockout_config_params,
            )
        else:
            # Only evaluate ID patterns
            log.info("Running ID evaluation only...")
            id_patterns = sample_id_patterns(
                knockout_vocabulary=knockout_vocabulary,
                layer_sizes=layer_sizes,
                damage_prob=damage_prob,
                batch_size=batch_size,
                periodic_eval_test_seed=test_seed,
            )
            sa_step_metrics_in = evaluate_with_damage(
                model=model,
                base_wires=base_wires,
                base_logits=base_logits,
                x_data=x_data,
                y_data=y_data,
                input_n=config.circuit.input_bits,
                arity=config.circuit.arity,
                circuit_hidden_dim=config.circuit.circuit_hidden_dim,
                n_message_steps=n_steps,
                loss_type=config.training.loss_type,
                layer_sizes=layer_sizes,
                layer_neighbors=config.training.layer_neighbors,
                knockout_patterns=id_patterns,
                knockout_vocabulary=knockout_vocabulary,
                eval_batch_size=batch_size,
                return_per_pattern=True,
                blind_mode=config.eval.get("blind_mode", False),
                **knockout_config_params,
            )
            sa_step_metrics_out = None
        
        # Run BP evaluation if requested
        bp_results = None
        if show_bp_trajectory:
            log.info("Running backpropagation training for comparison")
            # Use ID patterns for BP training
            id_patterns = sample_id_patterns(
                knockout_vocabulary=knockout_vocabulary,
                layer_sizes=layer_sizes,
                damage_prob=damage_prob,
                batch_size=batch_size,
                periodic_eval_test_seed=test_seed,
            )
            bp_results = get_bp_results(
                config, x_data, y_data, config.training.loss_type, id_patterns, None
            )
        
        # Plot damage response trajectory
        fig = plot_inner_loop_trajectory(
            trajectory_type="damage_response",
            sa_step_metrics_in=sa_step_metrics_in,
            sa_step_metrics_out=sa_step_metrics_out,
            bp_results=bp_results,
            show_bp_trajectory=show_bp_trajectory,
            show_ood_trajectory=show_ood_trajectory,
            damage_injection_mode=knockout_config_params["damage_injection_mode"],
            damage_start_offset=knockout_config_params["damage_start_offset"],
            max_damage_per_circuit=knockout_config_params["max_damage_per_circuit"],
            greedy_injection_recover_steps=knockout_config_params["greedy_injection_recover_steps"],
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
