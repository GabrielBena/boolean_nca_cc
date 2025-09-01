"""
Backpropagation training utilities for boolean circuit optimization.

This module provides functions for running backpropagation training
on boolean circuits with knockout patterns.
"""

import jax
import jax.numpy as jp
import optax
from tqdm.auto import tqdm
from functools import partial
import logging
from typing import Dict, List, Any

from boolean_nca_cc.circuits.model import gen_circuit
from boolean_nca_cc.circuits.train import TrainState, loss_f_l4, loss_f_bce, train_step
from boolean_nca_cc.circuits.train import create_gate_mask_from_knockout_pattern

# Setup logging
log = logging.getLogger(__name__)


def _run_backpropagation_training_with_knockouts(cfg, x_data, y_data, loss_type, knockout_patterns):
    """
    Run backpropagation training for each knockout pattern in the vocabulary.
    """
    # Generate circuit
    key = jax.random.PRNGKey(cfg.test_seed)
    wires, logits = gen_circuit(key, cfg.circuit.layer_sizes, arity=cfg.circuit.arity)

    # Setup optimizer
    if cfg.backprop.optimizer == "adamw":
        opt = optax.adamw(
            cfg.backprop.learning_rate,
            b1=cfg.backprop.beta1,
            b2=cfg.backprop.beta2,
            weight_decay=cfg.backprop.weight_decay,
        )
    else:
        opt = optax.adam(cfg.backprop.learning_rate)

    patterns_performance = []
    
    # Train on each knockout pattern
    for pattern_idx, knockout_pattern in enumerate(knockout_patterns):
        log.info(f"Training on knockout pattern {pattern_idx + 1}/{len(knockout_patterns)}")
        
        # Initialize fresh state for each pattern
        state = TrainState(params=logits, opt_state=opt.init(logits))
        
        # Training loop for this pattern
        losses = []
        hard_losses = []
        accuracies = []
        hard_accuracies = []

        # Partial function for train_step with knockout pattern
        _train_step_fn = partial(
            train_step,
            opt=opt,
            wires=wires,
            x=x_data,
            y0=y_data,
            loss_type=loss_type,
            do_train=True,
            knockout_pattern=knockout_pattern,
            layer_sizes=cfg.circuit.layer_sizes,
        )

        pbar = tqdm(range(cfg.backprop.epochs), desc=f"BP pattern {pattern_idx + 1}")
        for i in pbar:
            loss, aux_metrics, new_state = _train_step_fn(state=state)
            state = new_state

            accuracy = float(aux_metrics["accuracy"])
            hard_accuracy = float(aux_metrics["hard_accuracy"])
            hard_loss = float(aux_metrics["hard_loss"])

            # Log metrics
            if i % cfg.logging.log_interval == 0:
                log.info(
                    f"BP Pattern {pattern_idx + 1} Epoch {i}: Loss={loss:.4f}, Acc={accuracy:.4f}, Hard Acc={hard_accuracy:.4f}"
                )

            # Store metrics
            losses.append(float(loss))
            hard_losses.append(hard_loss)
            accuracies.append(accuracy)
            hard_accuracies.append(hard_accuracy)

            # Update tqdm postfix
            pbar.set_postfix(
                loss=loss,
                acc=accuracy,
                hard_acc=hard_accuracy,
                hard_loss=hard_loss,
            )

        # Final evaluation for this pattern
        loss_fn = loss_f_l4 if loss_type == "l4" else loss_f_bce
        final_loss, final_aux_metrics = loss_fn(
            state.params, wires, x_data, y_data, 
            gate_mask=create_gate_mask_from_knockout_pattern(knockout_pattern, cfg.circuit.layer_sizes)
        )
        final_accuracy = float(final_aux_metrics["accuracy"])
        final_hard_accuracy = float(final_aux_metrics["hard_accuracy"])
        final_hard_loss = float(final_aux_metrics["hard_loss"])

        log.info(
            f"BP Pattern {pattern_idx + 1} Final: Loss={final_loss:.4f}, Acc={final_accuracy:.4f}, Hard Acc={final_hard_accuracy:.4f}"
        )

        # Store results for this pattern
        pattern_results = {
            "pattern_idx": pattern_idx,
            "knockout_pattern": knockout_pattern,
            "losses": losses,
            "hard_losses": hard_losses,
            "accuracies": accuracies,
            "hard_accuracies": hard_accuracies,
            "final_loss": float(final_loss),
            "final_hard_loss": final_hard_loss,
            "final_accuracy": final_accuracy,
            "final_hard_accuracy": final_hard_accuracy,
            "params": state.params,
        }
        
        patterns_performance.append(pattern_results)

    # Calculate aggregate metrics
    final_losses = [p["final_loss"] for p in patterns_performance]
    final_accuracies = [p["final_accuracy"] for p in patterns_performance]
    final_hard_accuracies = [p["final_hard_accuracy"] for p in patterns_performance]
    
    best_pattern_idx = min(range(len(final_losses)), key=lambda i: final_losses[i])
    worst_pattern_idx = max(range(len(final_losses)), key=lambda i: final_losses[i])
    
    aggregate_metrics = {
        "mean_final_loss": float(jax.numpy.mean(jax.numpy.array(final_losses))),
        "std_final_loss": float(jax.numpy.std(jax.numpy.array(final_losses))),
        "mean_final_accuracy": float(jax.numpy.mean(jax.numpy.array(final_accuracies))),
        "std_final_accuracy": float(jax.numpy.std(jax.numpy.array(final_accuracies))),
        "mean_final_hard_accuracy": float(jax.numpy.mean(jax.numpy.array(final_hard_accuracies))),
        "std_final_hard_accuracy": float(jax.numpy.std(jax.numpy.array(final_hard_accuracies))),
        "best_pattern_idx": best_pattern_idx,
        "worst_pattern_idx": worst_pattern_idx,
        "best_final_loss": final_losses[best_pattern_idx],
        "worst_final_loss": final_losses[worst_pattern_idx],
    }

    log.info(f"Knockout training complete. Mean final loss: {aggregate_metrics['mean_final_loss']:.4f}")
    log.info(f"Best pattern: {best_pattern_idx}, Worst pattern: {worst_pattern_idx}")

    results = {
        "patterns_performance": patterns_performance,
        "vocabulary_patterns": knockout_patterns,
        "aggregate_metrics": aggregate_metrics,
        "wires": wires,  # Same wires for all patterns
    }

    return results 