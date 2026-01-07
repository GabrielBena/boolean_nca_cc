"""
Inner loop trajectory visualization.

This module provides unified plotting functions for inner loop trajectories,
handling both boolean discovery (no damage) and damage response trajectories.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List, Dict, Tuple
import logging
import jax
import jax.numpy as jp
import os

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
       - Reuses damage shading logic from plot_combined_bp_sa_stepwise_performance
       - Shows SA trajectories with optional BP comparison
    
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
        full_map_acc_train = np.array(train_metrics["full_map_accuracy"])
        hard_acc_train = np.array(train_metrics["hard_accuracy"])
        full_map_acc_test = np.array(test_metrics["full_map_accuracy"])
        hard_acc_test = np.array(test_metrics["hard_accuracy"])
        
        # Create figure with 2 subplots (Full Map and Hard Accuracy)
        fig, (ax1, ax2) = create_figure(
            width_type='double',
            nrows=1,
            ncols=2,
            figsize=figsize,
        )
        
        # Left subplot: Full Map Accuracy
        ax1.plot(
            steps_train,
            full_map_acc_train,
            color=COLORS['train'],
            linewidth=LINE_STYLES['linewidth_thick'],
            label='Train',
            marker='o',
            markersize=LINE_STYLES['markersize'],
            alpha=LINE_STYLES['alpha_mean'],
            markevery=max(1, len(steps_train) // 20),
        )
        ax1.plot(
            steps_test,
            full_map_acc_test,
            color=COLORS['test'],
            linewidth=LINE_STYLES['linewidth_thick'],
            label='Test',
            marker='s',
            markersize=LINE_STYLES['markersize'],
            alpha=LINE_STYLES['alpha_mean'],
            markevery=max(1, len(steps_test) // 20),
        )
        ax1.axhline(y=1.0, color=COLORS['reference'], linestyle=':', alpha=0.5, linewidth=1.5)
        
        format_axis(
            ax1,
            xlabel='Evaluation Step',
            ylabel='Full Map Accuracy',
            title='Full Map Accuracy',
            ylim=[0, 1.05],
            grid=True,
            legend=True,
            legend_loc='best',
        )
        
        # Right subplot: Hard Accuracy
        ax2.plot(
            steps_train,
            hard_acc_train,
            color=COLORS['train'],
            linewidth=LINE_STYLES['linewidth_thick'],
            label='Train',
            marker='o',
            markersize=LINE_STYLES['markersize'],
            alpha=LINE_STYLES['alpha_mean'],
            markevery=max(1, len(steps_train) // 20),
        )
        ax2.plot(
            steps_test,
            hard_acc_test,
            color=COLORS['test'],
            linewidth=LINE_STYLES['linewidth_thick'],
            label='Test',
            marker='s',
            markersize=LINE_STYLES['markersize'],
            alpha=LINE_STYLES['alpha_mean'],
            markevery=max(1, len(steps_test) // 20),
        )
        ax2.axhline(y=1.0, color=COLORS['reference'], linestyle=':', alpha=0.5, linewidth=1.5)
        
        format_axis(
            ax2,
            xlabel='Evaluation Step',
            ylabel='Hard Accuracy',
            title='Hard Accuracy',
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
        elif pre_damage_accuracy is not None and training_mode == "repair":
            # Show pre-damage reference line in repair mode
            ax.axhline(
                y=pre_damage_accuracy,
                color=COLORS['reference'],
                linestyle='--',
                linewidth=LINE_STYLES['linewidth_thick'],
                alpha=0.8,
                label='Pre-damage Performance',
            )
        
        # Add damage region shading (reused from plot_combined_bp_sa_stepwise_performance)
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
    # Damage start offset (warm-up period before first damage)
    damage_start_offset: int = 0,  # Number of steps to run before first damage injection
    damage_start_offset_random: bool = False,  # If True, randomize offset per circuit
    damage_start_offset_seed: int = 42,  # Seed for random offset generation
    knockout_vocabulary: Optional[jp.ndarray] = None,
    training_mode: str = "growth",  # Add this parameter
):
    """
    Create a plot showing SA stepwise performance, optionally with backpropagation comparison.
    
    NOTE: This function is maintained for backward compatibility. New code should use
    plot_inner_loop_trajectory(trajectory_type="damage_response", ...) instead, which
    provides standardized styling and unified handling of both boolean discovery and
    damage response trajectories.
    
    When show_bp_trajectory=True: Shows both SA and BP trajectories for comparison.
    When show_bp_trajectory=False: Focuses solely on SA performance, comparing ID vs OOD patterns.
                                  BP computation is completely skipped in this mode.
    
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
        bp_results: Pre-computed backpropagation results (optional, only used if show_bp_trajectory=True)
        show_bp_trajectory: If True, compute and show full BP trajectory; if False, completely skip BP computation and focus on SA ID vs OOD comparison only
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
        training_mode: Training mode ("growth" or "repair") - determines if pre-damage line is shown
        
    Returns:
        matplotlib figure with the combined performance plot
    """
    from boolean_nca_cc.training.evaluation import evaluate_circuits_in_chunks, evaluate_model_stepwise_batched, get_loss_from_wires_logits
    from functools import partial

    # Only compute/use BP results if show_bp_trajectory is True
    # When False, we focus solely on SA ID vs OOD comparison
    results = None
    if show_bp_trajectory:
        # Import backpropagation training function
        if bp_results is None:
            # Validate that knockout_patterns is available for BP training
            if knockout_patterns is None:
                raise ValueError(
                    "knockout_patterns is None but bp_results is also None and show_bp_trajectory=True. "
                    "Either provide pre-computed bp_results, knockout_patterns for BP training, or set show_bp_trajectory=False."
                )
            from boolean_nca_cc.training.backprop import _run_backpropagation_training_with_knockouts
            # Run backpropagation training
            log.info(f"Running BP training with knockout_patterns length: {len(knockout_patterns) if knockout_patterns is not None else 'None'}")
            results = _run_backpropagation_training_with_knockouts(
                cfg, x_data, y_data, loss_type, knockout_patterns, 
                parallel=getattr(cfg.backprop, 'parallel', True) if hasattr(cfg.backprop, 'parallel') else cfg.backprop.get("parallel", True) if hasattr(cfg.backprop, 'get') else True,
                batch_size=getattr(cfg.backprop, 'batch_size', None) if hasattr(cfg.backprop, 'batch_size') else cfg.backprop.get("batch_size", None) if hasattr(cfg.backprop, 'get') else None
            )
        else:
            log.info(f"Using pre-computed bp_results")
            results = bp_results
    else:
        log.info(f"show_bp_trajectory=False: Skipping BP computation entirely, focusing on SA ID vs OOD comparison")
    
    # Run SA evaluation on the same patterns
    base_wires, base_logits = base_circuit
    
    # Determine batch size and evaluation approach based on damage mode
    if damage_mode in ["greedy", "greedy_vocabulary"] and damage_injection_mode == "multi":
        # Multi-damage mode: Use dynamic evaluation with periodic injections
        # Validate greedy_ordered_indices for greedy modes
        if damage_mode == "greedy" and greedy_ordered_indices is None:
            raise ValueError(
                "greedy_ordered_indices is None but required for damage_mode='greedy'. "
                "Either provide greedy_ordered_indices or use a different damage_mode."
            )
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
        try:
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
                damage_start_offset=damage_start_offset,
                damage_start_offset_random=damage_start_offset_random,
                damage_start_offset_seed=damage_start_offset_seed,
                knockout_vocabulary=knockout_vocabulary,
            )
        except Exception as e:
            log.error(f"Error during SA evaluation (multi-damage mode): {e}")
            log.error(f"  damage_mode={damage_mode}, damage_injection_mode={damage_injection_mode}")
            log.error(f"  greedy_ordered_indices type={type(greedy_ordered_indices)}, is_none={greedy_ordered_indices is None}")
            log.error(f"  knockout_vocabulary type={type(knockout_vocabulary)}, is_none={knockout_vocabulary is None}")
            raise
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
        try:
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
                # Damage control parameters (required by evaluate_model_stepwise_batched)
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
            )
        except Exception as e:
            log.error(f"Error during SA evaluation (static damage mode): {e}")
            log.error(f"  knockout_patterns type={type(knockout_patterns)}, length={len(knockout_patterns) if knockout_patterns is not None else 'None'}")
            raise
    
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
                damage_start_offset=damage_start_offset,
                damage_start_offset_random=damage_start_offset_random,
                damage_start_offset_seed=damage_start_offset_seed,
                knockout_vocabulary=None,  # Force unseen patterns for OOD evaluation
            )
        else:
            # Static damage mode: Use pre-generated OOD patterns (backward compatible)
            if knockout_patterns is None:
                raise ValueError("knockout_patterns must be provided for static damage modes")
                
            log.info(f"Generating OOD knockout patterns for SA evaluation ({len(knockout_patterns)} patterns)...")
            
            # Use the same logic as in run_knockout_periodic_evaluation
            from boolean_nca_cc.training.pool.structural_perturbation import create_reproducible_knockout_pattern
            
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
                # Damage control parameters (required by evaluate_model_stepwise_batched)
                damage_mode=damage_mode,
                damage_injection_mode=damage_injection_mode,
                max_damage_per_circuit=max_damage_per_circuit,
                greedy_ordered_indices=greedy_ordered_indices,
                greedy_window_size=greedy_window_size,
                greedy_injection_recover_steps=greedy_injection_recover_steps,
                damage_start_offset=damage_start_offset,
                damage_start_offset_random=damage_start_offset_random,
                damage_start_offset_seed=damage_start_offset_seed,
                knockout_vocabulary=None,  # Force unseen patterns for OOD evaluation
            )
        log.info("OOD SA evaluation completed")
    
    # Use IN-distribution metrics as the primary SA metrics for backward compatibility
    sa_step_metrics = sa_step_metrics_in
    
    # Validate that we have valid step metrics
    if sa_step_metrics is None:
        raise ValueError("SA step metrics are None - evaluation may have failed. Check evaluation logs for errors.")
    if not isinstance(sa_step_metrics, dict):
        raise ValueError(f"SA step metrics is not a dict, got type: {type(sa_step_metrics)}")
    if "step" not in sa_step_metrics:
        raise ValueError(f"SA step metrics missing 'step' key. Available keys: {list(sa_step_metrics.keys())}")
    if "hard_accuracy" not in sa_step_metrics:
        raise ValueError(f"SA step metrics missing 'hard_accuracy' key. Available keys: {list(sa_step_metrics.keys())}")
    
    # Evaluate the base circuit (preconfigured) without knockout patterns to get true pre-damage performance
    base_wires, base_logits = base_circuit
    _, base_aux = get_loss_from_wires_logits(base_logits, base_wires, x_data, y_data, loss_type)
    pre_damage_accuracy = float(base_aux[4])
    
    # Use the unified plotting function instead of manual plotting
    # This provides standardized styling and reuses the damage shading logic
    fig = plot_inner_loop_trajectory(
        trajectory_type="damage_response",
        sa_step_metrics_in=sa_step_metrics_in,
        sa_step_metrics_out=sa_step_metrics_out if show_ood_trajectory else None,
        bp_results=results if show_bp_trajectory else None,
        show_bp_trajectory=show_bp_trajectory,
        show_ood_trajectory=show_ood_trajectory,
        damage_injection_mode=damage_injection_mode,
        damage_start_offset=damage_start_offset,
        max_damage_per_circuit=max_damage_per_circuit,
        greedy_injection_recover_steps=greedy_injection_recover_steps,
        training_mode=training_mode,
        pre_damage_accuracy=pre_damage_accuracy,
        output_path=None,  # Don't save here, let caller handle it
        title=None,  # Will use default from unified function
        figsize=None,  # Use standardized sizing
    )
    
    return fig

