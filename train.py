#!/usr/bin/env python3
"""
Main training script for Boolean Circuit NCA optimization using GNNs and Self-Attention.

This script trains models that learn to simulate the inner loop of backpropagation
when training boolean circuits, using either Graph Neural Networks or Self-Attention.
"""

import os
import logging
import jax
import optax
import hydra
from omegaconf import DictConfig, OmegaConf, open_dict
import wandb
from tqdm.auto import tqdm
from functools import partial
from flax import nnx

from boolean_nca_cc.circuits.model import gen_circuit
from boolean_nca_cc.circuits.tasks import get_task_data
from boolean_nca_cc import generate_layer_sizes
from boolean_nca_cc.models import (
    CircuitGNN,
    CircuitSelfAttention,
)
from boolean_nca_cc.circuits.train import TrainState, loss_f_l4, loss_f_bce, train_step

from boolean_nca_cc.training.train_loop import train_model
from boolean_nca_cc.training.evaluation import evaluate_model_stepwise
from boolean_nca_cc.utils.graph_builder import build_graph
from boolean_nca_cc.training.utils import (
    plot_training_curves,
    save_checkpoint,
    plot_inner_loop_metrics,
    compare_with_backprop,
)

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
            config=OmegaConf.to_container(cfg, resolve=True),
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
    if cfg.model.type == "gnn":
        model = CircuitGNN(
            hidden_dim=cfg.model.hidden_dim,
            message_passing=cfg.model.message_passing,
            node_mlp_features=cfg.model.node_mlp_features,
            edge_mlp_features=cfg.model.edge_mlp_features,
            use_attention=cfg.model.use_attention,
            arity=arity,
            rngs=nnx.Rngs(params=init_rng),
        )
    elif cfg.model.type == "self_attention":
        model = CircuitSelfAttention(
            n_node=n_nodes,
            hidden_dim=cfg.model.hidden_dim,
            arity=arity,
            num_heads=cfg.model.num_heads,
            num_layers=cfg.model.num_layers,
            mlp_dim=cfg.model.mlp_dim,
            dropout_rate=cfg.model.dropout_rate,
            rngs=nnx.Rngs(params=init_rng),
        )
    else:
        raise ValueError(f"Unknown model type: {cfg.model.type}")

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
        hidden_dim=cfg.model.hidden_dim,
        arity=arity,
        # Training hyperparameters
        learning_rate=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
        epochs=cfg.training.epochs or 2**cfg.training.epochs_power_of_2,
        n_message_steps=1,
        loss_type=cfg.training.loss_type,
        # Wiring mode parameters
        wiring_mode=cfg.training.wiring_mode,
        meta_batch_size=cfg.training.meta_batch_size,
        wiring_fixed_key=jax.random.PRNGKey(cfg.test_seed),
        # Pool parameters
        pool_size=cfg.pool.size,
        reset_pool_fraction=cfg.pool.reset_fraction,
        reset_pool_interval=cfg.pool.reset_interval,
        reset_strategy=cfg.pool.reset_strategy,
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
            cfg.training.epochs or 2**cfg.training.epochs_power_of_2,
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
