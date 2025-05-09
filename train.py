#!/usr/bin/env python3
"""
Main training script for Boolean Circuit NCA optimization using GNNs and Self-Attention.

This script trains models that learn to simulate the inner loop of backpropagation
when training boolean circuits, using either Graph Neural Networks or Self-Attention.
"""

import os
import sys
import time
import logging
from typing import Dict, List, Tuple, Optional, Union
from functools import partial

import jax
import jax.numpy as jp
import flax
from flax import nnx
import optax
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

import hydra
from omegaconf import DictConfig, OmegaConf, open_dict
import wandb

from boolean_nca_cc.circuits.model import gen_circuit
from boolean_nca_cc.circuits.train import TrainState, loss_f_l4, loss_f_bce, train_step
from boolean_nca_cc.circuits.tasks import get_task_data
from boolean_nca_cc import generate_layer_sizes
from boolean_nca_cc.models import (
    CircuitGNN,
    CircuitSelfAttention,
)
from boolean_nca_cc.training.train_loop import train_model
from boolean_nca_cc.training.evaluation import evaluate_model_stepwise
from boolean_nca_cc.utils.graph_builder import build_graph

# Configure logging
log = logging.getLogger(__name__)


def run_backpropagation_training(cfg, x_data, y_data, loss_type="l4"):
    """
    Run standard backpropagation training for comparison.

    Args:
        cfg: Configuration object
        x_data: Input data
        y_data: Target data
        loss_type: Loss function type ('l4' or 'bce')

    Returns:
        Dictionary of training results
    """
    log.info("Running baseline backpropagation training")

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

    state = TrainState(params=logits, opt_state=opt.init(logits))

    # Training loop
    losses = []
    hard_losses = []
    accuracies = []
    hard_accuracies = []

    # Partial function for train_step to avoid passing opt and wires repeatedly
    # Note: optax optimizers are not JAX types, so they cannot be static arguments for JIT
    # if we were to JIT the loop here. train_step itself handles JITting of grad computation.
    _train_step_fn = partial(
        train_step,
        opt=opt,
        wires=wires,
        x=x_data,
        y0=y_data,
        loss_type=loss_type,
        do_train=True,
    )

    pbar = tqdm(range(cfg.backprop.epochs), desc="Backprop training")
    for i in pbar:
        loss, aux_metrics, new_state = _train_step_fn(state=state)
        state = new_state  # Update state for the next iteration

        accuracy = float(aux_metrics["accuracy"])
        hard_accuracy = float(aux_metrics["hard_accuracy"])
        hard_loss = float(aux_metrics["hard_loss"])

        # Log metrics
        if i % cfg.logging.log_interval == 0:
            log.info(
                f"BP Epoch {i}: Loss={loss:.4f}, Acc={accuracy:.4f}, Hard Acc={hard_accuracy:.4f}"
            )
            if cfg.wandb.enabled:
                wandb.log(
                    {
                        "bp/loss": float(loss),
                        "bp/hard_loss": hard_loss,
                        "bp/accuracy": accuracy,
                        "bp/hard_accuracy": hard_accuracy,
                        "bp/epoch": i,
                    }
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

    # Final evaluation (using the appropriate loss function directly for clarity)
    loss_fn = loss_f_l4 if loss_type == "l4" else loss_f_bce
    final_loss, final_aux_metrics = loss_fn(state.params, wires, x_data, y_data)
    final_accuracy = float(final_aux_metrics["accuracy"])
    final_hard_accuracy = float(final_aux_metrics["hard_accuracy"])
    final_hard_loss = float(final_aux_metrics["hard_loss"])

    log.info(
        f"BP Final: Loss={final_loss:.4f}, Acc={final_accuracy:.4f}, Hard Acc={final_hard_accuracy:.4f}"
    )

    results = {
        "losses": losses,
        "hard_losses": hard_losses,
        "accuracies": accuracies,
        "hard_accuracies": hard_accuracies,
        "final_loss": float(final_loss),
        "final_hard_loss": final_hard_loss,
        "final_accuracy": final_accuracy,
        "final_hard_accuracy": final_hard_accuracy,
        "params": state.params,
        "wires": wires,
    }

    return results


def save_checkpoint(model, optimizer, metrics, cfg, step, output_dir):
    """Save a checkpoint of the model and optimizer."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    checkpoint = {
        "model": nnx.state(model),
        "optimizer": nnx.state(optimizer),
        "metrics": metrics,
        "config": OmegaConf.to_container(cfg, resolve=True),
        "step": step,
    }

    checkpoint_path = os.path.join(output_dir, f"checkpoint_{step}.ckpt")
    with open(checkpoint_path, "wb") as f:
        checkpoint_bytes = flax.serialization.to_bytes(checkpoint)
        f.write(checkpoint_bytes)

    log.info(f"Saved checkpoint to {checkpoint_path}")
    if cfg.wandb.enabled:
        wandb.save(checkpoint_path)

    return checkpoint_path


def load_checkpoint(checkpoint_path):
    """Load a checkpoint of the model and optimizer."""
    with open(checkpoint_path, "rb") as f:
        checkpoint_bytes = f.read()
        checkpoint = flax.serialization.from_bytes(None, checkpoint_bytes)

    return checkpoint


def plot_training_curves(metrics, title, output_dir):
    """Generate and save training curve plots."""
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Plot loss curves
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(metrics["losses"], label="Soft Loss")
    ax.plot(metrics["hard_losses"], label="Hard Loss")
    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Loss")
    ax.set_yscale("log")
    ax.set_title(f"{title} - Loss Curves")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()

    loss_plot_path = os.path.join(
        output_dir, f"{title.lower().replace(' ', '_')}_loss.png"
    )
    plt.savefig(loss_plot_path)
    if wandb.run is not None:
        wandb.log({f"{title} Loss": wandb.Image(fig)})
    plt.close(fig)

    # Plot accuracy curves
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(metrics["accuracies"], label="Soft Accuracy")
    ax.plot(metrics["hard_accuracies"], label="Hard Accuracy")
    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"{title} - Accuracy Curves")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()

    acc_plot_path = os.path.join(
        output_dir, f"{title.lower().replace(' ', '_')}_accuracy.png"
    )
    plt.savefig(acc_plot_path)
    if wandb.run is not None:
        wandb.log({f"{title} Accuracy": wandb.Image(fig)})
    plt.close(fig)


def plot_inner_loop_metrics(step_metrics, title, output_dir):
    """Plot inner loop metrics over GNN message passing steps."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Plot loss curves
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(step_metrics["step"], step_metrics["soft_loss"], label="Soft Loss")
    ax.plot(step_metrics["step"], step_metrics["hard_loss"], label="Hard Loss")
    ax.set_xlabel("Message Passing Steps")
    ax.set_ylabel("Loss")
    ax.set_yscale("log")
    ax.set_title(f"{title} - Inner Loop Loss")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()

    loss_plot_path = os.path.join(
        output_dir, f"{title.lower().replace(' ', '_')}_inner_loss.png"
    )
    plt.savefig(loss_plot_path)
    if wandb.run is not None:
        wandb.log({f"{title} Inner Loop Loss": wandb.Image(fig)})
    plt.close(fig)

    # Plot accuracy curves
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(step_metrics["step"], step_metrics["soft_accuracy"], label="Soft Accuracy")
    ax.plot(step_metrics["step"], step_metrics["hard_accuracy"], label="Hard Accuracy")
    ax.set_xlabel("Message Passing Steps")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"{title} - Inner Loop Accuracy")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()

    acc_plot_path = os.path.join(
        output_dir, f"{title.lower().replace(' ', '_')}_inner_accuracy.png"
    )
    plt.savefig(acc_plot_path)
    if wandb.run is not None:
        wandb.log({f"{title} Inner Loop Accuracy": wandb.Image(fig)})
    plt.close(fig)

    # Log the final metrics
    if wandb.run is not None:
        wandb.log(
            {
                f"inner_loop/final_soft_loss": step_metrics["soft_loss"][-1],
                f"inner_loop/final_hard_loss": step_metrics["hard_loss"][-1],
                f"inner_loop/final_soft_accuracy": step_metrics["soft_accuracy"][-1],
                f"inner_loop/final_hard_accuracy": step_metrics["hard_accuracy"][-1],
            }
        )


def compare_with_backprop(gnn_metrics, bp_metrics, title, output_dir):
    """Compare GNN and backpropagation performance."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Create a figure with 2x2 subplots
    fig, axs = plt.subplots(2, 2, figsize=(16, 12), constrained_layout=True)

    # Plot soft loss comparison
    axs[0, 0].plot(gnn_metrics["losses"], label="GNN")
    axs[0, 0].plot(bp_metrics["losses"], label="Backprop")
    axs[0, 0].set_title("Soft Loss Comparison")
    axs[0, 0].set_xlabel("Training Steps")
    axs[0, 0].set_ylabel("Loss")
    axs[0, 0].set_yscale("log")
    axs[0, 0].legend()
    axs[0, 0].grid(True)

    # Plot hard loss comparison
    axs[0, 1].plot(gnn_metrics["hard_losses"], label="GNN")
    axs[0, 1].plot(bp_metrics["hard_losses"], label="Backprop")
    axs[0, 1].set_title("Hard Loss Comparison")
    axs[0, 1].set_xlabel("Training Steps")
    axs[0, 1].set_ylabel("Loss")
    axs[0, 1].set_yscale("log")
    axs[0, 1].legend()
    axs[0, 1].grid(True)

    # Plot soft accuracy comparison
    axs[1, 0].plot(gnn_metrics["accuracies"], label="GNN")
    axs[1, 0].plot(bp_metrics["accuracies"], label="Backprop")
    axs[1, 0].set_title("Soft Accuracy Comparison")
    axs[1, 0].set_xlabel("Training Steps")
    axs[1, 0].set_ylabel("Accuracy")
    axs[1, 0].legend()
    axs[1, 0].grid(True)

    # Plot hard accuracy comparison
    axs[1, 1].plot(gnn_metrics["hard_accuracies"], label="GNN")
    axs[1, 1].plot(bp_metrics["hard_accuracies"], label="Backprop")
    axs[1, 1].set_title("Hard Accuracy Comparison")
    axs[1, 1].set_xlabel("Training Steps")
    axs[1, 1].set_ylabel("Accuracy")
    axs[1, 1].legend()
    axs[1, 1].grid(True)

    plt.suptitle(f"{title} - GNN vs Backprop Comparison")

    # Save the figure
    comp_plot_path = os.path.join(
        output_dir, f"{title.lower().replace(' ', '_')}_comparison.png"
    )
    plt.savefig(comp_plot_path)
    if wandb.run is not None:
        wandb.log({f"{title} Comparison": wandb.Image(fig)})
    plt.close(fig)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Main training function using Hydra for configuration.

    Args:
        cfg: Hydra configuration object
    """
    # Print configuration
    log.info(OmegaConf.to_yaml(cfg))

    # Set random seed
    rng = jax.random.PRNGKey(cfg.seed)

    # Create output directory
    output_dir = os.getcwd()
    log.info(f"Working directory: {output_dir}")

    # Initialize wandb if enabled
    if cfg.wandb.enabled:
        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            name=cfg.wandb.run_name,
            dir=output_dir,
        )

    # Generate circuit layer sizes
    input_n, output_n = cfg.circuit.input_bits, cfg.circuit.output_bits
    arity = cfg.circuit.arity
    if cfg.circuit.layer_sizes is None:
        layer_sizes = generate_layer_sizes(
            input_n, output_n, arity, layer_n=cfg.circuit.num_layers
        )
        with open_dict(cfg):
            cfg.circuit.layer_sizes = layer_sizes
    else:
        layer_sizes = cfg.circuit.layer_sizes

    # Generate dummy circuit
    test_key = jax.random.PRNGKey(cfg.test_seed)
    wires, logits = gen_circuit(
        test_key, cfg.circuit.layer_sizes, arity=cfg.circuit.arity
    )

    # Generate dummy graph
    graph = build_graph(
        wires=wires,
        logits=logits,
        input_n=input_n,
        arity=arity,
        hidden_dim=cfg.model.hidden_dim,
    )
    n_nodes = int(graph.n_node[0])

    log.info(f"Circuit layer sizes: {layer_sizes}")
    log.info(f"Number of nodes: {n_nodes}")
    if cfg.wandb.enabled:
        wandb.config.update({"layer_sizes": layer_sizes, "n_nodes": n_nodes})

    # Get task data
    case_n = 1 << input_n
    x, y0 = get_task_data(
        cfg.circuit.task, case_n, input_bits=input_n, output_bits=output_n
    )

    # Run backpropagation training for comparison if enabled
    bp_results = None
    if cfg.backprop.enabled:
        bp_results = run_backpropagation_training(
            cfg, x, y0, loss_type=cfg.training.loss_type
        )
        plot_training_curves(
            bp_results, "Backpropagation", os.path.join(output_dir, "plots")
        )

    # Initialize model
    rng, init_rng = jax.random.split(rng)

    # Common overrides for hydra.instantiate
    # These are values computed in train.py or essential for all models
    instantiate_overrides = {"arity": arity, "rngs": nnx.Rngs(params=init_rng)}

    # Specific overrides based on model type, which should still be in the YAML.
    # Alternatively, we could inspect cfg.model._target_ if 'type' was removed.
    if cfg.model.type == "self_attention":
        instantiate_overrides["n_node"] = n_nodes
    # CircuitGNN does not require n_node in its constructor based on original setup.

    # Instantiate the model using Hydra
    try:
        model = hydra.utils.instantiate(cfg.model, **instantiate_overrides)
    except Exception as e:
        log.error(
            f"Error instantiating model {cfg.model._target_ if '_target_' in cfg.model else cfg.model.type}: {e}"
        )
        raise

    # Train model
    log.info(f"Starting {cfg.model.type.upper()} training")
    gnn_results = train_model(
        # Initialization parameters
        key=cfg.seed,
        init_model=model,
        # Data parameters
        x_data=x,
        y_data=y0,
        layer_sizes=layer_sizes,
        # Training hyperparameters
        learning_rate=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
        epochs=cfg.training.epochs,
        n_message_steps=cfg.training.n_message_steps,
        loss_type=cfg.training.loss_type,
        # Wiring mode parameters
        wiring_mode=cfg.training.wiring_mode,
        meta_batch_size=cfg.training.meta_batch_size,
        wiring_fixed_key=jax.random.PRNGKey(cfg.test_seed),
        # Pool parameters
        use_pool=cfg.pool.enabled,
        pool_size=cfg.pool.size,
        reset_pool_fraction=cfg.pool.reset_fraction,
        reset_pool_interval=cfg.pool.reset_interval,
        reset_strategy=cfg.pool.reset_strategy,
        combined_weights=cfg.pool.combined_weights,
        # Gate knockout parameters
        gate_knockout_active=cfg.pool.gate_knockout.active,
        gate_knockout_fraction=cfg.pool.gate_knockout.fraction,
        gate_knockout_interval=cfg.pool.gate_knockout.interval,
        gate_knockout_strategy=cfg.pool.gate_knockout.strategy,
        gate_knockout_damage_prob=cfg.pool.gate_knockout.damage_prob,
        gate_knockout_combined_weights=cfg.pool.gate_knockout.combined_weights,
        # Soft LUT damage parameters
        soft_lut_damage_active=cfg.pool.soft_lut_damage.active,
        soft_lut_damage_fraction=cfg.pool.soft_lut_damage.fraction,
        soft_lut_damage_interval=cfg.pool.soft_lut_damage.interval,
        soft_lut_damage_strategy=cfg.pool.soft_lut_damage.strategy,
        soft_lut_damage_damage_prob=cfg.pool.soft_lut_damage.damage_prob,
        soft_lut_damage_combined_weights=cfg.pool.soft_lut_damage.combined_weights,
        # Learning rate scheduling
        lr_scheduler=cfg.training.lr_scheduler,
    )

    # Format metrics for plotting
    metrics = {
        "losses": gnn_results["losses"],
        "hard_losses": gnn_results["hard_losses"],
        "accuracies": gnn_results["accuracies"],
        "hard_accuracies": gnn_results["hard_accuracies"],
    }

    # Plot training curves
    model_name = cfg.model.type.upper()
    plot_training_curves(
        metrics, f"{model_name} Training", os.path.join(output_dir, "plots")
    )

    # Save final model
    if cfg.checkpoint.enabled:
        save_checkpoint(
            gnn_results["model"],
            gnn_results["optimizer"],
            metrics,
            cfg,
            cfg.training.epochs,
            os.path.join(output_dir, "checkpoints"),
        )

    # Evaluate inner loop (message passing steps)
    log.info("Evaluating inner loop performance")

    # Get a representative circuit for inner loop evaluation
    test_wires, test_logits = gen_circuit(
        test_key, cfg.circuit.layer_sizes, arity=cfg.circuit.arity
    )

    # Run stepwise evaluation
    step_metrics = evaluate_model_stepwise(
        model=gnn_results["model"],
        wires=test_wires,
        logits=test_logits,
        x_data=x,
        y_data=y0,
        input_n=input_n,
        arity=arity,
        hidden_dim=cfg.model.hidden_dim,
        n_message_steps=cfg.eval.inner_steps,
        loss_type=cfg.training.loss_type,
    )

    # Plot inner loop metrics
    plot_inner_loop_metrics(
        step_metrics, f"{model_name} Inner Loop", os.path.join(output_dir, "plots")
    )

    # Compare with backpropagation if available
    if bp_results is not None:
        compare_with_backprop(
            metrics,
            bp_results,
            f"{model_name} vs Backprop",
            os.path.join(output_dir, "plots"),
        )

    # Final log
    log.info(f"Training complete. Final results:")
    log.info(f"  Meta Loss: {metrics['losses'][-1]:.4f}")
    log.info(f"  Meta Hard Loss: {metrics['hard_losses'][-1]:.4f}")
    log.info(f"  Meta Accuracy: {metrics['accuracies'][-1]:.4f}")
    log.info(f"  Meta Hard Accuracy: {metrics['hard_accuracies'][-1]:.4f}")
    log.info(f"  Inner Loop Final Loss: {step_metrics['soft_loss'][-1]:.4f}")
    log.info(f"  Inner Loop Final Hard Loss: {step_metrics['hard_loss'][-1]:.4f}")
    log.info(f"  Inner Loop Final Accuracy: {step_metrics['soft_accuracy'][-1]:.4f}")
    log.info(
        f"  Inner Loop Final Hard Accuracy: {step_metrics['hard_accuracy'][-1]:.4f}"
    )

    # Close wandb if enabled
    if cfg.wandb.enabled:
        wandb.finish()


if __name__ == "__main__":
    main()
