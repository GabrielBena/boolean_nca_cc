"""
Visualization utilities for circuit analysis.

This module provides plotting functions for circuit performance analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Optional, List, Dict, Tuple
import logging
import jax
import jax.numpy as jp

try:
    import wandb
except ImportError:
    wandb = None

log = logging.getLogger(__name__)


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
    
    # Create single figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Plot: Hard Accuracy over steps - BP vs SA comparison
    sa_steps = sa_step_metrics["step"]
    sa_hard_accuracies = sa_step_metrics["hard_accuracy"]
    
    # Validate extracted values
    if sa_steps is None:
        raise ValueError("sa_steps is None after extraction from sa_step_metrics")
    if sa_hard_accuracies is None:
        raise ValueError("sa_hard_accuracies is None after extraction from sa_step_metrics")
    
    # Evaluate the base circuit (preconfigured) without knockout patterns to get true pre-damage performance
    base_wires, base_logits = base_circuit
    _, base_aux = get_loss_from_wires_logits(base_logits, base_wires, x_data, y_data, loss_type)
    pre_damage_accuracy = float(base_aux[4])
    
    # Use SA data as-is (no artificial concatenations)
    # SA already includes step 0 (pre-damage) and steps 1 to n_message_steps
    sa_steps = np.array(sa_steps)
    sa_hard_accuracies = np.array(sa_hard_accuracies)
    
    
    # Plot SA performance with error bands
    try:
        # Check if we have per-pattern data for error bands
        if 'per_pattern' in sa_step_metrics and 'pattern_hard_accuracies' in sa_step_metrics['per_pattern']:
            # Extract per-pattern hard accuracies
            per_pattern_accuracies = sa_step_metrics['per_pattern']['pattern_hard_accuracies']
            # Convert to numpy array for easier manipulation
            per_pattern_accuracies = np.array(per_pattern_accuracies)  # Shape: [n_steps, n_patterns]
            
            # Calculate mean and std across patterns at each step
            sa_mean_accuracies = np.mean(per_pattern_accuracies, axis=1)
            sa_std_accuracies = np.std(per_pattern_accuracies, axis=1)
            
            # Use SA steps as-is (should match per-pattern data)
            sa_steps_for_plot = sa_steps
            
            
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
            ood_sa_hard_accuracies = np.array(ood_sa_hard_accuracies)
            
            # Check if we have per-pattern data for error bands
            if 'per_pattern' in sa_step_metrics_out and 'pattern_hard_accuracies' in sa_step_metrics_out['per_pattern']:
                # Extract per-pattern hard accuracies
                ood_per_pattern_accuracies = sa_step_metrics_out['per_pattern']['pattern_hard_accuracies']
                # Convert to numpy array for easier manipulation
                ood_per_pattern_accuracies = np.array(ood_per_pattern_accuracies)  # Shape: [n_steps, n_patterns]
                
                # Calculate mean and std across patterns at each step
                ood_sa_mean_accuracies = np.mean(ood_per_pattern_accuracies, axis=1)
                ood_sa_std_accuracies = np.std(ood_per_pattern_accuracies, axis=1)
                
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
    
    # Only process BP data if show_bp_trajectory is True
    if show_bp_trajectory and results is not None:
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
        
        # Calculate mean and std across patterns
        bp_mean_accuracies = np.mean(bp_accuracies_with_initial, axis=0)  # Average across patterns
        bp_std_accuracies = np.std(bp_accuracies_with_initial, axis=0)  # Standard deviation across patterns
        
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
        
        log.info(f"Final data shape: SA steps={len(sa_steps)}, SA accuracies={len(sa_hard_accuracies)}, BP steps={len(bp_steps)}, BP accuracies={len(bp_mean_accuracies)}")
    else:
        # When show_bp_trajectory=False, we only work with SA data
        log.info(f"SA steps: {len(sa_steps)}")
        log.info(f"SA step range: {sa_steps[0]} to {sa_steps[-1]}")
        log.info(f"Final data shape: SA steps={len(sa_steps)}, SA accuracies={len(sa_hard_accuracies)}")
    
    # Final verification that dimensions match
    if len(sa_steps) != len(sa_hard_accuracies):
        raise ValueError(f"SA steps and accuracies have different lengths: {len(sa_steps)} vs {len(sa_hard_accuracies)}")
    
    # Plot BP trajectory or reference line
    try:
        if show_bp_trajectory and results is not None:
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
            # Plot pre-damage circuit accuracy as horizontal reference line
            # This comes from SA step 0, so no BP computation needed
            # Only show in repair mode
            if training_mode == "repair":
                ax.axhline(y=pre_damage_accuracy, 
                          color='#377eb8',
                          linestyle='--',
                          linewidth=2.0,
                          alpha=1.0,
                          label=f'Pre-damage Performance')
    except Exception as e:
        if show_bp_trajectory:
            log.error(f"Error plotting BP data: {e}")
            if 'bp_steps' in locals() and 'bp_mean_accuracies' in locals():
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
    y_ticks = np.arange(0.6, 1.03, 0.05)
    ax.set_yticks(y_ticks)
    
    # Calculate actual damage injection step based on offset (matching evaluation logic)
    # First damage occurs at step: damage_start_offset + 1
    first_damage_step = damage_start_offset + 1
    
    # Set xlim based on damage mode and whether BP trajectory is shown
    if damage_injection_mode == "multi":
        # Multi-damage mode: extend xlim to accommodate multiple damage injections
        # Account for offset + multiple damage events
        max_x = damage_start_offset + max_damage_per_circuit * (greedy_injection_recover_steps + 1) + 20
        if show_bp_trajectory:
            ax.set_xlim(0, max(60, max_x))  # Use the larger of BP range or multi-damage range
        else:
            ax.set_xlim(0, max_x)  # Extended range for multi-damage SA-only plot
    else:
        # Single damage mode: use data-driven range or reasonable default
        if show_bp_trajectory:
            ax.set_xlim(0, 60)  # Extended range for full BP trajectory
        else:
            # Use actual data range or calculate based on offset + recovery period
            max_x = max(len(sa_steps) - 1, first_damage_step + 15) if len(sa_steps) > 0 else first_damage_step + 15
            ax.set_xlim(0, max_x)
    
    # Add color-coded regions and damage indicator only for single damage mode
    # Regions are now aligned with actual damage injection timing from evaluation
    if damage_injection_mode == "single":
        # Pre-damage region: from step 0 to first damage injection
        pre_damage_end = first_damage_step
        ax.axvspan(0, pre_damage_end, alpha=0.1, color='#377eb8')  # Pastel blue for pre-damage region
        
        # Damage region: single step where damage is injected
        damage_start = first_damage_step
        damage_end = first_damage_step + 1
        ax.axvspan(damage_start, damage_end, alpha=0.1, color='#ff7f00')  # Pastel orange for damage region
        
        # Recovery region: from after damage injection to end of plot
        # Use the xlim that was just set (or actual data max if available)
        recovery_start = damage_end
        if len(sa_steps) > 0:
            recovery_end = max(sa_steps[-1], ax.get_xlim()[1])  # Use max of data or xlim
        else:
            recovery_end = ax.get_xlim()[1]  # Fallback to xlim
        ax.axvspan(recovery_start, recovery_end, alpha=0.1, color='#4daf4a')  # Pastel green for recovery region
    
    # Get existing legend handles and labels (only trajectories and pre-damage line)
    existing_handles, existing_labels = ax.get_legend_handles_labels()
    
    # Update legend with solid background (only trajectories and pre-damage line)
    legend = ax.legend(handles=existing_handles, labels=existing_labels, loc='lower right', fontsize=16)
    legend.get_frame().set_alpha(1.0)  # Make legend box solid
    
    plt.tight_layout()
    
    return fig


def main():
    """
    Standalone CLI for generating trajectory plots from model checkpoints.
    
    Matches the approach used in hamming_distance_plot.py for model/config loading,
    and extracts parameters from config matching train_loop.py.
    """
    import argparse
    import os
    from types import SimpleNamespace
    from omegaconf import OmegaConf
    
    from boolean_nca_cc.circuits.model import gen_circuit, generate_layer_sizes
    from boolean_nca_cc.circuits.tasks import get_task_data
    from boolean_nca_cc.circuits.data_split import split_input_combinations
    from boolean_nca_cc.training.checkpointing import (
        load_best_model_from_wandb,
        load_checkpoint,
        instantiate_model_from_config,
        load_config_from_wandb,
    )
    from boolean_nca_cc.training.preconfigure import preconfigure_circuit_logits
    from boolean_nca_cc.training.pool.structural_perturbation import create_knockout_vocabulary
    from boolean_nca_cc.training.backprop import _run_backpropagation_training_with_knockouts
    
    parser = argparse.ArgumentParser(
        description="Generate trajectory plots from model checkpoints",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Checkpoint loading - match hamming_distance_plot.py
    parser.add_argument("--run-id", type=str, default="nypyrbwh", 
                        help="WandB run id for GNN model load (default: nypyrbwh)")
    parser.add_argument("--checkpoint", type=str, default=None, 
                        help="Local checkpoint path .pkl for GNN model (optional)")
    
    # Output parameters
    parser.add_argument("--output", type=str, default=None,
                        help="Output path for figure (default: auto-generated)")
    parser.add_argument("--dpi", type=int, default=300,
                        help="Image resolution")
    
    # Optional overrides
    parser.add_argument("--show-bp", action="store_true",
                        help="Show backpropagation comparison trajectory")
    parser.add_argument("--no-ood", action="store_true",
                        help="Don't show out-of-distribution (unseen) patterns")
    parser.add_argument("--n-message-steps", type=int, default=None,
                        help="Number of message passing steps (default: from config)")
    
    args = parser.parse_args()
    
    # Load config first (always needed) - match hamming_distance_plot.py
    if args.checkpoint is not None:
        loaded = load_checkpoint(args.checkpoint)
        cfg = OmegaConf.create(loaded.get("config", {}))
    else:
        # Load config from WandB (without loading the model yet)
        # Use standard artifact filename to get config
        filename_to_load = "best_model_eval_ko_hard_accuracy"
        cfg, _, _ = load_config_from_wandb(
            run_id=args.run_id,
            filename=filename_to_load,
            select_by_best_metric=False,  # Just get config, not necessarily best model
        )
    
    # Load model - match hamming_distance_plot.py
    if args.checkpoint is not None:
        loaded = load_checkpoint(args.checkpoint)
        model = instantiate_model_from_config(cfg, seed=cfg.get("seed", 0))
        from flax import nnx as _nnx
        _nnx.update(model, loaded["model"])
        
        # Extract and display epoch/step information
        step = loaded.get("step")
        checkpoint_config = loaded.get("config", {})
        if isinstance(checkpoint_config, dict):
            epoch = checkpoint_config.get("epoch")
        else:
            epoch = getattr(checkpoint_config, "epoch", None)
        
        if step is not None:
            print(f"Loaded checkpoint at step: {step}")
        if epoch is not None:
            print(f"Loaded checkpoint at epoch: {epoch}")
    else:
        # Load the actual model now
        model, loaded_dict, _ = load_best_model_from_wandb(
            run_id=args.run_id,
            seed=0,  # Use default seed, will be overridden by GNN config
            filename=filename_to_load,
        )
        
        # Extract and display epoch/step information
        step = loaded_dict.get("step")
        checkpoint_config = loaded_dict.get("config", {})
        if isinstance(checkpoint_config, dict):
            epoch = checkpoint_config.get("epoch")
        else:
            epoch = getattr(checkpoint_config, "epoch", None)
        
        if step is not None:
            print(f"Loaded checkpoint at step: {step}")
        if epoch is not None:
            print(f"Loaded checkpoint at epoch: {epoch}")
    
    # Extract circuit parameters - match train_loop.py and hamming_distance_plot.py
    circuit_cfg = cfg.get("circuit", {})
    input_n = circuit_cfg.get("input_bits", 12)
    output_n = circuit_cfg.get("output_bits", 12)
    arity = circuit_cfg.get("arity", 4)
    num_layers = circuit_cfg.get("num_layers", 3)
    
    layer_sizes = circuit_cfg.get("layer_sizes", None)
    if layer_sizes is None:
        layer_sizes = generate_layer_sizes(input_n, output_n, arity, layer_n=num_layers)
    
    circuit_hidden_dim = int(cfg.model.get("circuit_hidden_dim", 16))
    loss_type = cfg.get("training", {}).get("loss_type", "l4")
    training_mode = cfg.training.get("training_mode", "growth")  # Fix: use cfg.training.training_mode
    
    # Extract periodic eval parameters - match train_loop.py
    periodic_eval_inner_steps = cfg.eval.get("periodic_eval_inner_steps", 100)
    periodic_eval_test_seed = cfg.eval.get("periodic_eval_test_seed", 42)
    periodic_eval_batch_size = cfg.eval.get("periodic_eval_batch_size", 16)
    
    # Override n_message_steps if provided
    n_message_steps = args.n_message_steps if args.n_message_steps is not None else periodic_eval_inner_steps
    
    # Extract knockout eval config - match train_loop.py
    knockout_eval = cfg.eval.get("knockout_eval", None)
    if knockout_eval is None:
        knockout_eval = {}
    
    # Extract damage parameters - match train_loop.py
    pool_cfg = cfg.get("pool", {})
    damage_mode = pool_cfg.get("damage_mode", "greedy")
    damage_injection_mode = pool_cfg.get("damage_injection_mode", "multi")
    max_damage_per_circuit = pool_cfg.get("max_damage_per_circuit", 10)
    damage_pool_damage_prob = pool_cfg.get("damage_prob", 10)
    damage_seed = cfg.get("damage_seed", 481)
    knockout_diversity = pool_cfg.get("damage_knockout_diversity", pool_cfg.get("persistent_knockout", {}).get("knockout_diversity", 0))
    greedy_ordered_indices = pool_cfg.get("greedy_ordered_indices", None)
    greedy_window_size = pool_cfg.get("greedy_window_size", 1)
    
    # Extract from knockout_eval config if available
    if knockout_eval:
        damage_mode = knockout_eval.get("damage_mode", damage_mode)
        damage_injection_mode = knockout_eval.get("damage_injection_mode", damage_injection_mode)
        max_damage_per_circuit = knockout_eval.get("max_damage_per_circuit", max_damage_per_circuit)
        greedy_ordered_indices = knockout_eval.get("greedy_ordered_indices", greedy_ordered_indices)
        greedy_window_size = knockout_eval.get("greedy_window_size", greedy_window_size)
    
    greedy_injection_recover_steps = knockout_eval.get("greedy_injection_recover_steps", 10)
    damage_start_offset = knockout_eval.get("damage_start_offset", 0)
    damage_start_offset_random = knockout_eval.get("damage_start_offset_random", False)
    damage_start_offset_seed = knockout_eval.get("damage_start_offset_seed", 42)
    
    # Extract damage_behavior from model config and set it on the model
    damage_behavior = cfg.model.get("damage_behavior", "permanent")
    setattr(model, "damage_behavior", damage_behavior)
    log.info(f"Set model damage_behavior to: {damage_behavior}")
    
    # Extract layer_neighbors
    layer_neighbors = cfg.model.get("layer_neighbors", False)
    
    # Load data - match train_loop.py and hamming_distance_plot.py
    case_n = 1 << input_n
    x, y0 = get_task_data(
        cfg.circuit.task, 
        case_n, 
        max_samples=circuit_cfg.get("max_task_samples", 100000),
        sample_seed=cfg.get("test_seed", 42),
        input_bits=input_n, 
        output_bits=output_n
    )
    
    # Handle input split if enabled - match train_loop.py
    input_split_enabled = cfg.eval.get("input_split_enabled", False)
    if input_split_enabled:
        input_train_fraction = cfg.eval.get("input_train_fraction", 0.8)
        input_split_seed = cfg.eval.get("input_split_seed", 42)
        x_train, x_test, y_train, y_test = split_input_combinations(
            x, y0, train_fraction=input_train_fraction, seed=input_split_seed
        )
        # Use test data for evaluation/plotting (training is complete)
        x_data = x_test
        y_data = y_test
    else:
        x_data = x
        y_data = y0
    
    # Create base circuit - match train_loop.py
    wiring_fixed_key = cfg.get("wiring_fixed_key", cfg.get("test_seed", 42))
    if isinstance(wiring_fixed_key, int):
        wiring_fixed_key = jax.random.PRNGKey(wiring_fixed_key)
    
    if training_mode == "repair":
        log.info("Model was trained in repair mode - recreating exact preconfigured baseline circuit")
        
        # Extract preconfiguration parameters from training config
        preconfig_steps = cfg.get("preconfig_steps", cfg.backprop.get("epochs", 200))
        preconfig_lr = cfg.get("preconfig_lr", cfg.backprop.get("learning_rate", 1e-2))
        
        # Recreate the exact same preconfigured circuit used during training
        base_wires, base_logits = preconfigure_circuit_logits(
            wiring_key=wiring_fixed_key,
            layer_sizes=layer_sizes,
            arity=arity,
            x_data=x_data,
            y_data=y_data,
            loss_type=loss_type,
            steps=preconfig_steps,
            lr=preconfig_lr,
        )
        log.info(f"Recreated preconfigured circuit with {preconfig_steps} steps, lr={preconfig_lr}")
    else:
        log.info("Model was trained in growth mode - generating base circuit")
        base_wires, base_logits = gen_circuit(wiring_fixed_key, layer_sizes, arity=arity)
    
    base_circuit = (base_wires, base_logits)
    
    # Create knockout vocabulary - match train_loop.py
    knockout_vocabulary = None
    if knockout_diversity is not None and knockout_diversity > 0:
        log.info(f"Creating knockout vocabulary with size {knockout_diversity}")
        vocab_rng = jax.random.PRNGKey(damage_seed)
        
        knockout_vocabulary = create_knockout_vocabulary(
            rng=vocab_rng,
            vocabulary_size=knockout_diversity,
            layer_sizes=layer_sizes,
            damage_prob=damage_pool_damage_prob,
            damage_mode=damage_mode,
            ordered_indices=greedy_ordered_indices,
        )
        log.info(f"Created knockout vocabulary with {len(knockout_vocabulary)} patterns")
    else:
        log.warning("No knockout vocabulary created - knockout_diversity is None or 0")
    
    # Create config object for backprop evaluation (if needed) - match train_loop.py
    bp_results = None
    mock_cfg = None
    if args.show_bp and knockout_vocabulary is not None:
        backprop_config = cfg.get("backprop", {})
        mock_cfg = SimpleNamespace(
            test_seed=periodic_eval_test_seed,
            circuit=SimpleNamespace(
                layer_sizes=layer_sizes,
                arity=arity
            ),
            backprop=SimpleNamespace(
                epochs=backprop_config.get("epochs", 50),
                learning_rate=backprop_config.get("learning_rate", 1e-2),
                weight_decay=backprop_config.get("weight_decay", 1e-4),
                optimizer=backprop_config.get("optimizer", "adam"),
                beta1=backprop_config.get("beta1", 0.9),
                beta2=backprop_config.get("beta2", 0.999),
            ),
            logging=SimpleNamespace(log_interval=1),
        )
        
        # Run backpropagation training - match train_loop.py
        log.info("Computing backpropagation results for comparison...")
        try:
            bp_results = _run_backpropagation_training_with_knockouts(
                mock_cfg, x_data, y_data, loss_type, knockout_vocabulary,
                parallel=backprop_config.get("parallel", True),
                batch_size=backprop_config.get("batch_size", None)
            )
            log.info(f"Backpropagation results computed. Mean final accuracy: {bp_results['aggregate_metrics']['mean_final_hard_accuracy']:.3f}")
        except Exception as e:
            log.warning(f"Error computing backpropagation results: {e}")
            bp_results = None
    elif args.show_bp:
        log.warning("Cannot compute BP results: knockout_vocabulary is None")
        bp_results = None
    
    # Use mock_cfg for plot function if not created yet
    if mock_cfg is None:
        backprop_config = cfg.get("backprop", {})
        mock_cfg = SimpleNamespace(
            test_seed=periodic_eval_test_seed,
            circuit=SimpleNamespace(
                layer_sizes=layer_sizes,
                arity=arity
            ),
            backprop=SimpleNamespace(**backprop_config),
            logging=SimpleNamespace(log_interval=1),
        )
    
    # Determine output path
    output_path = args.output
    if output_path is None:
        if args.run_id:
            base_name = f"trajectory_{args.run_id}"
        else:
            checkpoint_name = os.path.splitext(os.path.basename(args.checkpoint))[0] if args.checkpoint else "checkpoint"
            base_name = f"trajectory_{checkpoint_name}"
        
        output_path = f"reports/figures/{base_name}.png"
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    log.info(f"Generating trajectory plot...")
    log.info(f"  Training mode: {training_mode}")
    log.info(f"  Damage behavior: {damage_behavior}")
    log.info(f"  Damage mode: {damage_mode}")
    log.info(f"  Damage injection mode: {damage_injection_mode}")
    log.info(f"  N message steps: {n_message_steps}")
    log.info(f"  Show BP: {args.show_bp}")
    log.info(f"  Show OOD: {not args.no_ood}")
    log.info(f"  Output: {output_path}")
    
    # Call the plotting function - match train_loop.py call
    fig = plot_combined_bp_sa_stepwise_performance(
        cfg=mock_cfg,
        x_data=x_data,
        y_data=y_data,
        loss_type=loss_type,
        knockout_patterns=knockout_vocabulary,
        model=model,
        base_circuit=base_circuit,
        n_message_steps=n_message_steps,
        layer_sizes=layer_sizes,
        input_n=input_n,
        arity=arity,
        circuit_hidden_dim=circuit_hidden_dim,
        bp_results=bp_results,
        show_bp_trajectory=args.show_bp,
        periodic_eval_test_seed=periodic_eval_test_seed,
        knockout_config=knockout_eval,
        show_ood_trajectory=not args.no_ood,
        layer_neighbors=layer_neighbors,
        # Multi-damage support parameters
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
        training_mode=training_mode,
    )
    
    # Save figure
    fig.savefig(output_path, dpi=args.dpi, bbox_inches='tight')
    plt.close(fig)
    
    print(f"\n✓ Figure saved to: {output_path}")


if __name__ == "__main__":
    main()
