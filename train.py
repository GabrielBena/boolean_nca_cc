#!/usr/bin/env python3
"""
Main training script for Boolean Circuit NCA optimization using GNNs and Self-Attention.

This script trains models that learn to simulate the inner loop of backpropagation
when training boolean circuits, using either Graph Neural Networks or Self-Attention.
"""

import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import logging
from functools import partial

import hydra
import jax
import optax
import pandas as pd
from flax import nnx
from omegaconf import DictConfig, OmegaConf, open_dict
from tqdm.auto import tqdm

import wandb
from boolean_nca_cc import generate_layer_sizes
from boolean_nca_cc.circuits.model import gen_circuit
from boolean_nca_cc.circuits.tasks import get_task_data
from boolean_nca_cc.circuits.train import TrainState, loss_f_bce, loss_f_l4, train_step
from boolean_nca_cc.training.checkpointing import save_checkpoint
from boolean_nca_cc.training.eval_datasets import (
    create_unified_evaluation_datasets,
)
from boolean_nca_cc.training.train_loop import (
    run_unified_periodic_evaluation,
    train_model,
)
from boolean_nca_cc.training.utils import (
    cleanup_redundant_wandb_artifacts,
    plot_training_curves,
)
from boolean_nca_cc.utils.graph_builder import build_graph
from boolean_nca_cc.utils.pool_stats import (
    calculate_expected_pool_updates,
    compute_pool_parameter,
)

# Configure logging
log = logging.getLogger(__name__)


def extract_track_metrics_config(cfg) -> list[str] | None:
    """
    Extract track_metrics configuration from config.

    Returns:
        List of metrics to track or None if not configured
    """
    if not cfg.checkpoint.get("track_best_metrics", {}).get("enabled", False):
        return None

    metrics_config = cfg.checkpoint.get("track_best_metrics", {}).get("metrics", [])
    if not metrics_config:
        return None

    return list(metrics_config)


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


def create_and_save_final_results(
    cfg,
    model_results,
    eval_results,
    layer_sizes,
    output_dir,
    wandb_run=None,
    total_params=None,
):
    """
    Create comprehensive final results dictionary, save as CSV, and log results.

    Args:
        cfg: Configuration object
        model_results: Results from model training
        eval_results: Results from evaluation
        layer_sizes: Circuit layer sizes
        output_dir: Directory to save results
        wandb_run: WandB run object (optional)

    Returns:
        dict: Final results dictionary
    """
    # Collect comprehensive final results
    final_results = {
        # Model and training configuration
        "model_type": cfg.model.type,
        "wiring_mode": cfg.training.wiring_mode,
        "loss_type": cfg.training.loss_type,
        "learning_rate": cfg.training.learning_rate,
        "epochs_completed": len(model_results["losses"]),
        "total_epochs_planned": cfg.training.epochs or 2**cfg.training.epochs_power_of_2,
        "early_stopped": model_results.get("early_stopped", False),
        "early_stop_epoch": model_results.get("early_stop_epoch", None),
        "first_threshold_epoch": model_results.get("first_threshold_epoch", None),
        # Meta-learning metrics (final training values)
        "meta_loss": model_results["losses"][-1],
        "meta_hard_loss": model_results["hard_losses"][-1],
        "meta_accuracy": model_results["accuracies"][-1],
        "meta_hard_accuracy": model_results["hard_accuracies"][-1],
        # Best model performance
        "best_metric": model_results.get("best_metric", None),
        "best_metric_value": model_results.get("best_metric_value", None),
    }

    # Add unified evaluation metrics (IN-distribution and OUT-of-distribution)
    if eval_results:
        # IN-distribution evaluation (matches training pattern)
        final_in_metrics = eval_results.get("final_metrics_in", {})
        if final_in_metrics is not None:
            final_results.update(
                {
                    "eval_in_final_loss": final_in_metrics.get("eval_in/final_loss", None),
                    "eval_in_final_hard_loss": final_in_metrics.get(
                        "eval_in/final_hard_loss", None
                    ),
                    "eval_in_final_accuracy": final_in_metrics.get("eval_in/final_accuracy", None),
                    "eval_in_final_hard_accuracy": final_in_metrics.get(
                        "eval_in/final_hard_accuracy", None
                    ),
                }
            )

        # OUT-of-distribution evaluation (always random)
        final_out_metrics = eval_results.get("final_metrics_out", {})
        if final_out_metrics is not None:
            final_results.update(
                {
                    "eval_out_final_loss": final_out_metrics.get("eval_out/final_loss", None),
                    "eval_out_final_hard_loss": final_out_metrics.get(
                        "eval_out/final_hard_loss", None
                    ),
                    "eval_out_final_accuracy": final_out_metrics.get(
                        "eval_out/final_accuracy", None
                    ),
                    "eval_out_final_hard_accuracy": final_out_metrics.get(
                        "eval_out/final_hard_accuracy", None
                    ),
                }
            )

    # Circuit and model architecture info
    final_results.update(
        {
            "input_bits": cfg.circuit.input_bits,
            "output_bits": cfg.circuit.output_bits,
            "num_layers": len(layer_sizes),
            "total_nodes": sum(size[0] for size in layer_sizes),
            "circuit_hidden_dim": cfg.model.circuit_hidden_dim,
            "attention_dim": cfg.model.get("attention_dim", None),
            "mlp_dim_multiplier": cfg.model.get("mlp_dim_multiplier", None),
            "mlp_dim": cfg.model.get("mlp_dim", None),
            "total_parameters": total_params,
            "message_steps": cfg.training.n_message_steps,
            "eval_target_batch_size": cfg.eval.batch_size,
            "pool_size": cfg.pool.size,
            "pool_initial_diversity": cfg.pool.initial_diversity,
        }
    )

    # Add unified evaluation specific information if available
    if eval_results:
        # Check if datasets information is available (from UnifiedEvaluationDatasets)
        eval_datasets_info = eval_results.get("datasets_info", {})
        if eval_datasets_info:
            final_results.update(
                {
                    "eval_in_actual_batch_size": eval_datasets_info.get(
                        "in_actual_batch_size", None
                    ),
                    "eval_out_actual_batch_size": eval_datasets_info.get(
                        "out_actual_batch_size", None
                    ),
                    "eval_in_used_chunking": eval_datasets_info.get("in_used_chunking", False),
                    "eval_out_used_chunking": eval_datasets_info.get("out_used_chunking", False),
                    "eval_training_wiring_mode": eval_datasets_info.get(
                        "training_wiring_mode", None
                    ),
                    "eval_training_initial_diversity": eval_datasets_info.get(
                        "training_initial_diversity", None
                    ),
                }
            )

    # Create pandas DataFrame
    results_df = pd.DataFrame([final_results])

    # Save DataFrame to CSV
    results_csv_path = os.path.join(output_dir, "final_results.csv")
    results_df.to_csv(results_csv_path, index=False)
    log.info(f"Final results saved to: {results_csv_path}")

    # Display the DataFrame
    log.info("Final Results Summary:")
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    log.info("\n" + str(results_df.T))  # Transpose for better readability

    # Log to wandb if enabled
    if cfg.wandb.enabled and wandb_run:
        # Log the DataFrame as a table
        wandb.log({"final_results_table": wandb.Table(dataframe=results_df)})
        # Also log individual final metrics for easy access
        wandb.log({f"final/{k}": v for k, v in final_results.items() if v is not None})

    # Final log (traditional format for backward compatibility)
    log.info("Training complete. Final results:")
    log.info(f"  Meta Loss: {model_results['losses'][-1]:.4f}")
    log.info(f"  Meta Hard Loss: {model_results['hard_losses'][-1]:.4f}")
    log.info(f"  Meta Accuracy: {model_results['accuracies'][-1]:.4f}")
    log.info(f"  Meta Hard Accuracy: {model_results['hard_accuracies'][-1]:.4f}")

    if eval_results:
        final_in_metrics = eval_results.get("final_metrics_in", {})
        if final_in_metrics:
            log.info(
                f"  Eval IN-distribution Final Loss: {final_in_metrics.get('eval_in/final_loss', 'N/A'):.4f}"
            )
            log.info(
                f"  Eval IN-distribution Final Hard Accuracy: {final_in_metrics.get('eval_in/final_hard_accuracy', 'N/A'):.4f}"
            )

        final_out_metrics = eval_results.get("final_metrics_out", {})
        if final_out_metrics:
            log.info(
                f"  Eval OUT-of-distribution Final Loss: {final_out_metrics.get('eval_out/final_loss', 'N/A'):.4f}"
            )
            log.info(
                f"  Eval OUT-of-distribution Final Hard Accuracy: {final_out_metrics.get('eval_out/final_hard_accuracy', 'N/A'):.4f}"
            )

    # Display best model performance if applicable
    if cfg.checkpoint.save_best and "best_metric_value" in model_results:
        log.info(
            f"  Best {model_results.get('best_metric', 'metric')}: {model_results['best_metric_value']:.4f}"
        )

    # Display early stopping and threshold information if applicable
    first_threshold_epoch = model_results.get("first_threshold_epoch", None)
    if first_threshold_epoch is not None:
        log.info(
            f"  Accuracy threshold {cfg.stop_accuracy.threshold} first reached at epoch {first_threshold_epoch}"
        )

    if model_results.get("early_stopped", False):
        log.info(
            f"  Training stopped early at epoch {model_results.get('early_stop_epoch', 'unknown')}"
        )
        log.info(
            f"  Early stopping was triggered by {cfg.stop_accuracy.source}_{cfg.stop_accuracy.metric} >= {cfg.stop_accuracy.threshold}"
        )
    else:
        log.info(
            f"  Training completed all {cfg.training.epochs or 2**cfg.training.epochs_power_of_2} epochs"
        )

    return final_results


def process_pool_configuration(cfg):
    """
    Process pool configuration to automatically compute missing parameters based on expected updates.

    Args:
        cfg: Configuration object with pool settings

    Returns:
        Updated configuration with computed parameters

    Raises:
        ValueError: If configuration is underspecified or invalid
    """
    if cfg.pool.expected_updates is None:
        # No automatic computation requested, validate current config
        log.info("Using explicit pool configuration (no expected_updates specified)")
        return cfg

    # Check which parameters need to be computed
    pool_params = {
        "pool_size": cfg.pool.size,
        "batch_size": cfg.training.meta_batch_size,
        "n_message_steps": cfg.training.n_message_steps,
        "reset_interval": cfg.pool.reset_interval,
        "reset_fraction": cfg.pool.reset_fraction,
    }

    # Count None/null parameters
    none_params = [key for key, value in pool_params.items() if value is None]

    if len(none_params) == 0:
        # All parameters specified, verify the configuration matches expected updates
        log.info(
            f"Verifying pool configuration matches target expected updates: {cfg.pool.expected_updates}"
        )
        stats = calculate_expected_pool_updates(**pool_params)
        actual_updates = stats.expected_updates

        if abs(actual_updates - cfg.pool.expected_updates) > 0.1:
            log.warning(
                f"Configuration mismatch: Expected {cfg.pool.expected_updates:.2f} updates, "
                f"but configuration yields {actual_updates:.2f} updates"
            )
        else:
            log.info(f"Configuration verified: {actual_updates:.2f} expected updates per circuit")

        return cfg

    elif len(none_params) == 1:
        # Exactly one parameter to compute
        param_to_solve = none_params[0]
        log.info(
            f"Computing {param_to_solve} for target expected updates: {cfg.pool.expected_updates}"
        )

        # Prepare arguments for computation
        compute_kwargs = {k: v for k, v in pool_params.items() if v is not None}

        try:
            computed_value = compute_pool_parameter(
                target_expected_updates=cfg.pool.expected_updates,
                solve_for=param_to_solve,
                **compute_kwargs,
            )

            # Update configuration with computed value
            with open_dict(cfg):
                if param_to_solve == "pool_size":
                    cfg.pool.size = int(computed_value)
                elif param_to_solve == "batch_size":
                    cfg.training.meta_batch_size = int(computed_value)
                elif param_to_solve == "n_message_steps":
                    cfg.training.n_message_steps = int(computed_value)
                elif param_to_solve == "reset_interval":
                    cfg.pool.reset_interval = int(computed_value)
                elif param_to_solve == "reset_fraction":
                    cfg.pool.reset_fraction = float(computed_value)

            log.info(f"Computed {param_to_solve} = {computed_value:.4f}")

            # Verify the computation
            updated_params = {
                "pool_size": cfg.pool.size,
                "batch_size": cfg.training.meta_batch_size,
                "n_message_steps": cfg.training.n_message_steps,
                "reset_interval": cfg.pool.reset_interval,
                "reset_fraction": cfg.pool.reset_fraction,
            }
            stats = calculate_expected_pool_updates(**updated_params)
            log.info(f"Verification: {stats.expected_updates:.2f} expected updates per circuit")

        except ValueError as e:
            raise ValueError(
                f"Cannot compute {param_to_solve} for target {cfg.pool.expected_updates} updates: {e}"
            )

    else:
        # Multiple parameters are None - underspecified
        raise ValueError(
            f"Pool configuration is underspecified. Cannot compute multiple parameters: {none_params}. "
            f"Please specify all but one of: pool.size, training.meta_batch_size, "
            f"training.n_message_steps, pool.reset_interval, pool.reset_fraction"
        )

    return cfg


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Main training function using Hydra for configuration.

    Args:
        cfg: Hydra configuration object
    """
    # Print configuration
    log.info(OmegaConf.to_yaml(cfg))

    # Process pool configuration for automatic parameter computation
    cfg = process_pool_configuration(cfg)

    # Log final pool configuration and expected updates
    pool_params = {
        "pool_size": cfg.pool.size,
        "batch_size": cfg.training.meta_batch_size,
        "n_message_steps": cfg.training.n_message_steps,
        "reset_interval": cfg.pool.reset_interval,
        "reset_fraction": cfg.pool.reset_fraction,
    }
    stats = calculate_expected_pool_updates(**pool_params)

    log.info("Final Pool Configuration:")
    log.info(f"  Pool Size: {cfg.pool.size}")
    log.info(f"  Batch Size: {cfg.training.meta_batch_size}")
    log.info(f"  Message Steps: {cfg.training.n_message_steps}")
    log.info(f"  Reset Interval: {cfg.pool.reset_interval}")
    log.info(f"  Reset Fraction: {cfg.pool.reset_fraction:.4f}")
    log.info(f"  Expected Updates per Circuit: {stats.expected_updates:.2f}")
    log.info(f"  Selection Probability: {stats.selection_probability:.4f}")
    log.info(f"  Expected Lifetime: {stats.expected_lifetime_epochs:.1f} epochs")

    # Set random seed
    rng = jax.random.PRNGKey(cfg.seed)

    # Create output directory
    if cfg.output.dir is not None:
        output_dir = cfg.output.dir
        os.makedirs(output_dir, exist_ok=True)
    else:
        output_dir = os.getcwd()
    log.info(f"Output directory: {output_dir}")

    # Initialize wandb if enabled
    wandb_run = None
    if cfg.wandb.enabled:
        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            name=cfg.wandb.run_name,
            dir=output_dir,
            config=OmegaConf.to_container(cfg, resolve=True),
            group=cfg.wandb.group,
        )
        wandb_run = wandb.run

    # Generate circuit layer sizes
    input_n, output_n = cfg.circuit.input_bits, cfg.circuit.output_bits
    arity = cfg.circuit.arity
    if cfg.circuit.layer_sizes is None:
        layer_sizes = generate_layer_sizes(input_n, output_n, arity, layer_n=cfg.circuit.num_layers)
        with open_dict(cfg):
            cfg.circuit.layer_sizes = layer_sizes
    else:
        layer_sizes = cfg.circuit.layer_sizes

    # Generate dummy circuit
    test_key = jax.random.PRNGKey(cfg.test_seed)
    wires, logits = gen_circuit(test_key, cfg.circuit.layer_sizes, arity=cfg.circuit.arity)

    # Generate dummy graph
    graph = build_graph(
        wires=wires,
        logits=logits,
        input_n=input_n,
        arity=arity,
        circuit_hidden_dim=cfg.model.circuit_hidden_dim,
    )
    n_nodes = int(graph.n_node[0])

    log.info(f"Circuit layer sizes: {layer_sizes}")
    log.info(f"Number of nodes: {n_nodes}")
    if cfg.wandb.enabled:
        wandb.config.update({"layer_sizes": layer_sizes, "n_nodes": n_nodes})

    # Get task data
    case_n = 1 << input_n
    x, y0 = get_task_data(
        cfg.circuit.task,
        case_n,
        input_bits=input_n,
        output_bits=output_n,
        text=cfg.circuit.get("text", None),
    )

    # Run backpropagation training for comparison if enabled
    bp_results = None
    if cfg.backprop.enabled:
        bp_results = run_backpropagation_training(cfg, x, y0, loss_type=cfg.training.loss_type)
        plot_training_curves(bp_results, "Backpropagation", os.path.join(output_dir, "plots"))

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

    # Count and log model parameters with detailed breakdown
    log.info(f"Model instantiated: {cfg.model.type}")

    params = nnx.state(model, nnx.Param)
    module_param_counts = {}

    # Count parameters per module
    for module_name, module_params in params.items():
        num_params = jax.tree.reduce(lambda x, y: x + y.size, module_params, 0)
        module_param_counts[module_name] = num_params
        log.info(f"Number of params in {module_name}: {num_params:,}")

    # Calculate total parameters
    total_params = jax.tree.reduce(lambda x, y: x + y.size, params, 0)
    log.info(f"Total number of params: {total_params:,}")

    # Log to wandb if enabled
    if cfg.wandb.enabled:
        wandb_metrics = {
            "model/parameter_count": total_params,
        }
        # Log per-module counts as well
        for module_name, count in module_param_counts.items():
            wandb_metrics[f"model/params_{module_name}"] = count

        wandb.log(wandb_metrics)

    # Prepare checkpoint directory
    if cfg.checkpoint.enabled:
        checkpoint_dir = os.path.join(output_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
    else:
        checkpoint_dir = None

    # Get track_metrics configuration for training
    track_metrics = extract_track_metrics_config(cfg)

    # Train model
    log.info(f"Starting {cfg.model.type.upper()} training")
    model_results = train_model(
        # Initialization parameters
        key=cfg.seed,
        init_model=model,
        # Data parameters
        x_data=x,
        y_data=y0,
        layer_sizes=layer_sizes,
        circuit_hidden_dim=cfg.model.circuit_hidden_dim,
        arity=arity,
        # Training hyperparameters
        learning_rate=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
        epochs=cfg.training.epochs or 2**cfg.training.epochs_power_of_2,
        n_message_steps=cfg.training.n_message_steps,
        use_scan=cfg.training.use_scan,
        # Loss parameters
        loss_type=cfg.training.loss_type,
        random_loss_step=cfg.training.random_loss_step,
        use_beta_loss_step=cfg.training.use_beta_loss_step,
        # Wiring mode parameters
        wiring_mode=cfg.training.wiring_mode,
        meta_batch_size=cfg.training.meta_batch_size,
        batch_chunk_size=cfg.training.batch_chunk_size,
        wiring_fixed_key=jax.random.PRNGKey(cfg.test_seed),
        # Pool parameters
        pool_size=cfg.pool.size,
        reset_pool_fraction=cfg.pool.reset_fraction,
        reset_strategy=cfg.pool.reset_strategy,
        reset_pool_interval=cfg.pool.reset_interval,
        # Genetic mutation parameters
        genetic_mutation_rate=cfg.pool.mutation_rate,
        genetic_swaps_per_layer=cfg.pool.n_swaps_per_layer,
        initial_diversity=cfg.pool.initial_diversity,
        # Learning rate scheduling
        lr_scheduler=cfg.training.lr_scheduler,
        lr_scheduler_params=cfg.training.lr_scheduler_params,
        # Checkpoint parameters
        checkpoint_enabled=cfg.checkpoint.enabled,
        checkpoint_dir=checkpoint_dir,
        checkpoint_interval=cfg.checkpoint.interval,
        # Periodic evaluation parameters
        periodic_eval_enabled=cfg.eval.periodic.enabled,
        periodic_eval_interval=cfg.eval.periodic.interval,
        periodic_eval_inner_steps=cfg.eval.inner_steps,
        periodic_eval_test_seed=cfg.test_seed,
        periodic_eval_log_stepwise=cfg.eval.log_stepwise,
        periodic_eval_batch_size=cfg.eval.batch_size,
        periodic_eval_log_pool_scatter=cfg.eval.log_pool_scatter,
        # WandB parameters
        wandb_logging=cfg.wandb.enabled,
        log_interval=cfg.logging.log_interval,
        wandb_run_config=OmegaConf.to_container(cfg, resolve=True),
        # Early stopping parameters
        stop_accuracy_enabled=cfg.stop_accuracy.enabled,
        stop_accuracy_threshold=cfg.stop_accuracy.threshold,
        stop_accuracy_metric=cfg.stop_accuracy.metric,
        stop_accuracy_source=cfg.stop_accuracy.source,
        stop_accuracy_patience=cfg.stop_accuracy.patience,
        stop_accuracy_min_epochs=cfg.stop_accuracy.min_epochs,
        # Best model tracking parameters
        track_metrics=track_metrics,
    )

    # Save final model if checkpointing is enabled
    if cfg.checkpoint.enabled and not cfg.wandb.enabled:
        # If wandb is enabled, checkpoints are already being saved during training
        save_checkpoint(
            model_results["model"],
            model_results["optimizer"],
            {
                "losses": model_results["losses"],
                "hard_losses": model_results["hard_losses"],
                "accuracies": model_results["accuracies"],
                "hard_accuracies": model_results["hard_accuracies"],
                "reset_steps": model_results.get("reset_steps", []),
            },
            cfg,
            cfg.training.epochs or 2**cfg.training.epochs_power_of_2,
            checkpoint_dir,
            filename="final_model.pkl",
        )

    # Create standardized evaluation datasets
    log.info("Creating standardized evaluation datasets (seed + pool + OOD)")

    # Create evaluation datasets using standardized approach
    datasets = create_unified_evaluation_datasets(
        evaluation_base_seed=cfg.test_seed,
        training_wiring_mode=cfg.training.wiring_mode,
        training_initial_diversity=cfg.pool.initial_diversity,
        layer_sizes=layer_sizes,
        arity=cfg.circuit.arity,
        eval_batch_size=cfg.eval.batch_size,
    )

    # Get track_metrics configuration
    track_metrics = extract_track_metrics_config(cfg)

    # Run comprehensive evaluation using standardized datasets
    eval_results = run_unified_periodic_evaluation(
        model=model_results["model"],
        datasets=datasets,
        pool=model_results.get("pool", None),
        x_data=x,
        y_data=y0,
        input_n=input_n,
        arity=arity,
        circuit_hidden_dim=cfg.model.circuit_hidden_dim,
        n_message_steps=cfg.eval.inner_steps,
        loss_type=cfg.training.loss_type,
        epoch=-1,  # Final evaluation marker
        wandb_run=wandb_run,
        log_stepwise=False,
        layer_sizes=layer_sizes,
        log_pool_scatter=False,
        # Best model tracking parameters (final evaluation, so no saving)
        track_metrics=track_metrics,
    )

    if "metrics" in model_results:
        model_results.update(model_results["metrics"])

    # Collect comprehensive final results
    final_results = create_and_save_final_results(
        cfg,
        model_results,
        eval_results,
        layer_sizes,
        output_dir,
        wandb_run,
        total_params,
    )

    # Close wandb if enabled
    if cfg.wandb.enabled:
        cleanup_redundant_wandb_artifacts(
            run_id=wandb_run.id,
            dry_run=False,
            verbose=True,
        )
        wandb.finish()

    return final_results


if __name__ == "__main__":
    main()
