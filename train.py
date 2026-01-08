#!/usr/bin/env python3
"""
Main training script for Boolean Circuit NCA optimization using GNNs and Self-Attention.

This script trains models that learn to simulate the inner loop of backpropagation
when training boolean circuits, using either Graph Neural Networks or Self-Attention.
"""

import os

# === Optional CPU-only run ===
# To run on CPU only, set the below environment variable before JAX import.
# You can set this via an environment variable or command line argument:
#   $ JAX_PLATFORM_NAME=cpu python train.py

# Or uncomment the following line to force CPU from script and create 8 virtual devices:
# os.environ["JAX_PLATFORM_NAME"] = "cpu"
# os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"

# Configure JAX/XLA memory allocation BEFORE importing JAX
# Use "platform" allocator - slower but actually releases memory after pool resets
# The default BFC allocator is faster but pools memory aggressively, causing OOM at pool resets
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
# os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
# os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"


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
from boolean_nca_cc.circuits.train import LOSS_L4, LossConfig, TrainState, loss_f, train_step
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
from boolean_nca_cc.utils.configured_graph_builder import (
    configure_build_graph,
    configured_build_graph,
    is_configured,
)
from boolean_nca_cc.utils.pool_stats import (
    calculate_expected_damages,
    calculate_expected_pool_updates,
    compute_damage_parameter,
    compute_pool_parameter,
)

# Configure logging
log = logging.getLogger(__name__)

if os.path.exists("/mnt/storage/gb21"):
    wandb_cache_dir = "/mnt/storage/gb21/wandb/cache"
    wandb_artifact_dir = "/mnt/storage/gb21/wandb/artifacts"
    os.environ["WANDB_CACHE_DIR"] = wandb_cache_dir
    os.environ["WANDB_ARTIFACT_DIR"] = wandb_artifact_dir
    os.makedirs(wandb_cache_dir, exist_ok=True)
    os.makedirs(wandb_artifact_dir, exist_ok=True)


def extract_track_metrics_config(cfg) -> list[str] | None:
    """
    Extract track_metrics configuration from config using filter-based approach.

    Filters (omit or empty list = include all):
    - metric: [hard_accuracy, accuracy, hard_loss, loss]
    - data: [test, train]
    - wiring: [in, out]
    - damaged: [true, false]

    Returns:
        List of metrics to track, or None to auto-detect all active evaluation metrics.
    """
    track_cfg = cfg.checkpoint.get("track_best_metrics", {})
    if not track_cfg.get("enabled", False):
        return None

    # Get filters (empty list = include all)
    metrics = track_cfg.get("metric", []) or ["hard_accuracy", "accuracy", "hard_loss", "loss"]
    data_splits = track_cfg.get("data", []) or ["test", "train"]
    wirings = track_cfg.get("wiring", []) or ["in", "out"]
    damaged_opts = track_cfg.get("damaged", [])
    if not damaged_opts:
        damaged_prefixes = ["", "damaged_"]
    else:
        damaged_prefixes = ["damaged_" if d else "" for d in damaged_opts]

    # Generate all metric keys from filter combinations
    result = []
    for dp in damaged_prefixes:
        for w in wirings:
            for d in data_splits:
                for m in metrics:
                    result.append(f"eval_{dp}{w}_{d}_{m}")

    return result if result else None


def run_backpropagation_training(cfg, x_data, y_data, loss_cfg=None):
    """
    Run standard backpropagation training for comparison.

    Args:
        cfg: Configuration object
        x_data: Input data
        y_data: Target data
        loss_cfg: Loss config dict (default: LOSS_L4)

    Returns:
        Dictionary of training results
    """
    if loss_cfg is None:
        loss_cfg = LOSS_L4

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
        loss_cfg=loss_cfg,
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

    # Final evaluation (using the unified loss function)
    final_loss, final_aux_metrics = loss_f(state.params, wires, x_data, y_data, loss_cfg=loss_cfg)
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
        eval_results: Results from evaluation (with final_metrics dict)
        layer_sizes: Circuit layer sizes
        output_dir: Directory to save results
        wandb_run: WandB run object (optional)
        total_params: Total model parameters (optional)

    Returns:
        dict: Final results dictionary
    """
    # Collect comprehensive final results
    final_results = {
        # Model and training configuration
        "model_type": cfg.model.type,
        "wiring_mode": cfg.training.wiring_mode,
        "loss_cfg": dict(cfg.loss),
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
        # Circuit and model architecture
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
        "eval_target_batch_size_in": cfg.eval.batch_size_in,
        "eval_target_batch_size_out": cfg.eval.batch_size_out,
        "pool_size": cfg.pool.size,
        "pool_initial_diversity": cfg.pool.initial_diversity,
    }

    # Add evaluation metrics from all scenarios (wiring x data)
    if eval_results:
        final_metrics = eval_results.get("final_metrics", {})

        # Extract metrics for all evaluation scenarios
        for full_key, metrics in final_metrics.items():
            if metrics is None:
                continue
            # full_key is e.g. "in_test", "out_train", "damaged_in_test"
            prefix = f"eval_{full_key}"
            for metric_name in [
                "final_loss",
                "final_hard_loss",
                "final_accuracy",
                "final_hard_accuracy",
            ]:
                wandb_key = f"{prefix}/{metric_name}"
                result_key = f"{full_key}_{metric_name}".replace("/", "_")
                final_results[result_key] = metrics.get(wandb_key)

        # Add datasets info
        datasets_info = eval_results.get("datasets_info", {})
        if datasets_info:
            final_results.update(
                {
                    "eval_in_actual_batch_size": datasets_info.get("in_actual_batch_size"),
                    "eval_out_actual_batch_size": datasets_info.get("out_actual_batch_size"),
                    "eval_in_used_chunking": datasets_info.get("in_used_chunking", False),
                    "eval_out_used_chunking": datasets_info.get("out_used_chunking", False),
                    "eval_training_wiring_mode": datasets_info.get("training_wiring_mode"),
                    "eval_training_initial_diversity": datasets_info.get(
                        "training_initial_diversity"
                    ),
                }
            )

    # Save to CSV
    results_df = pd.DataFrame([final_results])
    results_csv_path = os.path.join(output_dir, "final_results.csv")
    results_df.to_csv(results_csv_path, index=False)
    log.info(f"Final results saved to: {results_csv_path}")

    # Display summary
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    log.info("Final Results Summary:\n" + str(results_df.T))

    # Log to wandb
    if cfg.wandb.enabled and wandb_run:
        wandb.log({"final_results_table": wandb.Table(dataframe=results_df)})
        wandb.log({f"final/{k}": v for k, v in final_results.items() if v is not None})

    # Console summary
    log.info("Training complete. Final results:")
    log.info(
        f"  Meta: Loss={model_results['losses'][-1]:.4f}, "
        f"Hard Acc={model_results['hard_accuracies'][-1]:.4f}"
    )

    if eval_results:
        final_metrics = eval_results.get("final_metrics", {})
        for key in ["in_test", "out_test", "in_train", "out_train"]:
            m = final_metrics.get(key)
            if m:
                prefix = f"eval_{key}"
                desc = key.replace("_", " ").upper()
                log.info(
                    f"  {desc}: Loss={m.get(f'{prefix}/final_loss', 0):.4f}, "
                    f"Hard Acc={m.get(f'{prefix}/final_hard_accuracy', 0):.4f}"
                )

    if cfg.checkpoint.save_best and "best_metric_value" in model_results:
        log.info(
            f"  Best {model_results.get('best_metric', 'metric')}: "
            f"{model_results['best_metric_value']:.4f}"
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

    if cfg.training.random_loss_step:
        n_message_steps_effective = (
            cfg.training.n_message_steps + cfg.training.random_loss_step_min
        ) // 2
    else:
        n_message_steps_effective = cfg.training.n_message_steps

    if cfg.pool.expected_updates is None:
        # No automatic computation requested, validate current config
        log.info("Using explicit pool configuration (no expected_updates specified)")
        return cfg, n_message_steps_effective

    # Check which parameters need to be computed
    pool_params = {
        "pool_size": cfg.pool.size,
        "batch_size": cfg.training.meta_batch_size,
        "n_message_steps": n_message_steps_effective,
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

        return cfg, n_message_steps_effective

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
                "n_message_steps": n_message_steps_effective,
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

    return cfg, n_message_steps_effective


def process_damage_configuration(cfg, expected_lifetime_epochs=None):
    """
    Process damage configuration to automatically compute missing parameters.

    Uses the formula:
        expected_damages = (expected_updates / damage_interval) * damage_fraction

    Where expected_updates comes from the pool configuration (must be processed first).
    If expected_lifetime_epochs is provided, it is used to compute the expected damages.

    Args:
        cfg: Configuration object with damage and pool settings.
             Pool configuration must be processed first (pool.expected_updates resolved) or expected_lifetime_epochs must be provided.

    Returns:
        Updated configuration with computed damage parameters

    Raises:
        ValueError: If configuration is invalid or underspecified
    """

    # Compute knockouts_per_event if not specified (needed for eval)
    if cfg.damage.knockouts_per_event is None:
        assert cfg.damage.random_knockouts_per_event, (
            "random_knockouts_per_event must be set to True"
        )
        # we'll draw from normal distribution with mean = (random_knockouts_per_event_min + random_knockouts_per_event_max) / 2
        cfg.damage.knockouts_per_event = (
            cfg.damage.random_knockouts_per_event_min + cfg.damage.random_knockouts_per_event_max
        ) / 2

        log.info(f"Computed knockouts_per_event = {cfg.damage.knockouts_per_event}")

    # Skip if damage is disabled
    if not cfg.damage.enabled:
        log.info("Damage system disabled (damage.enabled=false)")
        return cfg

    if expected_lifetime_epochs is None:
        # Ensure pool configuration is processed (we need expected_updates)
        expected_updates = cfg.pool.expected_updates
        if expected_updates is None:
            raise ValueError(
                "Cannot process damage configuration: pool.expected_updates is not set. "
                "Ensure pool configuration is processed first."
            )
    else:
        expected_updates = expected_lifetime_epochs

    # Check if expected_damages is specified (triggers auto-computation)
    if cfg.damage.expected_damages is None:
        # No automatic computation - validate explicit configuration
        if cfg.damage.damage_interval is None or cfg.damage.damage_fraction is None:
            raise ValueError(
                "Damage configuration incomplete: either set damage.expected_damages for "
                "auto-computation, or explicitly set both damage_interval and damage_fraction"
            )

        # Calculate what the configuration yields
        stats = calculate_expected_damages(
            expected_updates=expected_updates,
            damage_interval=cfg.damage.damage_interval,
            damage_fraction=cfg.damage.damage_fraction,
            knockouts_per_event=cfg.damage.knockouts_per_event,
        )
        log.info(
            f"Using explicit damage configuration: "
            f"{stats.expected_damages:.2f} expected damages per circuit, "
            f"{stats.expected_knockouts:.2f} expected knockouts"
        )
        return cfg

    # Determine which parameter to compute
    damage_params = {
        "damage_interval": cfg.damage.damage_interval,
        "damage_fraction": cfg.damage.damage_fraction,
    }

    none_params = [key for key, value in damage_params.items() if value is None]

    if len(none_params) == 0:
        # All parameters specified - verify configuration
        log.info(
            f"Verifying damage configuration matches target: {cfg.damage.expected_damages} expected damages"
        )
        stats = calculate_expected_damages(
            expected_updates=expected_updates,
            damage_interval=cfg.damage.damage_interval,
            damage_fraction=cfg.damage.damage_fraction,
            knockouts_per_event=cfg.damage.knockouts_per_event,
        )

        if abs(stats.expected_damages - cfg.damage.expected_damages) > 0.1:
            log.warning(
                f"Damage configuration mismatch: Expected {cfg.damage.expected_damages:.2f} damages, "
                f"but configuration yields {stats.expected_damages:.2f} damages"
            )
        else:
            log.info(
                f"Damage configuration verified: {stats.expected_damages:.2f} expected damages"
            )

        return cfg

    elif len(none_params) == 1:
        # Exactly one parameter to compute
        param_to_solve = none_params[0]
        log.info(
            f"Computing damage.{param_to_solve} for target: {cfg.damage.expected_damages} expected damages"
        )

        # Prepare arguments
        compute_kwargs = {
            "expected_updates": expected_updates,
        }
        if cfg.damage.damage_interval is not None:
            compute_kwargs["damage_interval"] = cfg.damage.damage_interval
        if cfg.damage.damage_fraction is not None:
            compute_kwargs["damage_fraction"] = cfg.damage.damage_fraction

        try:
            computed_value = compute_damage_parameter(
                target_expected_damages=cfg.damage.expected_damages,
                solve_for=param_to_solve,
                **compute_kwargs,
            )

            # Update configuration with computed value
            with open_dict(cfg):
                if param_to_solve == "damage_interval":
                    cfg.damage.damage_interval = int(max(1, round(computed_value)))
                elif param_to_solve == "damage_fraction":
                    cfg.damage.damage_fraction = float(computed_value)

            log.info(f"Computed damage.{param_to_solve} = {computed_value:.4f}")

            # Verify the computation
            stats = calculate_expected_damages(
                expected_updates=expected_updates,
                damage_interval=cfg.damage.damage_interval,
                damage_fraction=cfg.damage.damage_fraction,
                knockouts_per_event=cfg.damage.knockouts_per_event,
            )
            log.info(
                f"Verification: {stats.expected_damages:.2f} expected damages, "
                f"{stats.expected_knockouts:.2f} expected knockouts per circuit"
            )

        except ValueError as e:
            raise ValueError(
                f"Cannot compute damage.{param_to_solve} for target "
                f"{cfg.damage.expected_damages} damages: {e}"
            )

    else:
        # Multiple parameters are None - underspecified
        raise ValueError(
            f"Damage configuration underspecified. Cannot compute multiple parameters: {none_params}. "
            f"Set damage.expected_damages and provide either damage_interval or damage_fraction."
        )

    return cfg


def process_probabilistic_damage_configuration(cfg, layer_sizes):
    """
    Process probabilistic damage configuration to compute p_fault.

    Uses the formula from compute_p_fault_from_expected:
        p_fault = 1 - (1 - k/n)^(1/L)

    Where:
        k = expected_faulty_gates_at_reset
        n = number of eligible gates (hidden layers)
        L = expected circuit lifetime in steps (pool.expected_updates * n_message_steps)

    Args:
        cfg: Configuration object with damage and pool settings
        layer_sizes: List of (gate_n, group_size) tuples for the circuit

    Returns:
        Computed p_fault value, or None if damage is disabled or mode is discrete
    """
    from boolean_nca_cc.training.pool.structural_perturbation import (
        compute_p_fault_from_expected,
        count_eligible_gates,
    )

    # Skip if damage disabled or mode is discrete
    if not cfg.damage.enabled:
        log.info("Damage system disabled, p_fault = None")
        return None

    damage_mode = cfg.damage.get("mode", "probabilistic")
    if damage_mode != "probabilistic":
        log.info(f"Damage mode is '{damage_mode}', not computing p_fault")
        return None

    # If p_fault is explicitly set, use it
    if cfg.damage.get("p_fault") is not None:
        p_fault = float(cfg.damage.p_fault)
        log.info(f"Using explicit p_fault = {p_fault:.2e}")
        return p_fault

    # Auto-compute p_fault from expected_faulty_gates_at_reset
    expected_faulty = cfg.damage.get("expected_faulty_gates_at_reset", 4)
    if expected_faulty is None or expected_faulty <= 0:
        log.info("expected_faulty_gates_at_reset not set or <= 0, p_fault = None")
        return None

    # Count eligible gates (hidden layers only)
    n_eligible = count_eligible_gates(layer_sizes)
    if n_eligible <= 0:
        log.warning("No eligible gates for damage (no hidden layers?), p_fault = None")
        return None

    # Compute p_fault
    p_fault = compute_p_fault_from_expected(
        expected_faulty_gates=expected_faulty,
        n_eligible_gates=n_eligible,
        expected_lifetime_steps=cfg.pool.expected_updates,
    )

    log.info(
        f"Computed p_fault = {p_fault:.2e} "
        f"(target {expected_faulty} faulty gates, {n_eligible} eligible gates, "
        f"{cfg.pool.expected_updates} updates lifetime)"
    )

    return p_fault


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
    cfg, effective_n_message_steps = process_pool_configuration(cfg)

    # Log final pool configuration and expected updates
    pool_params = {
        "pool_size": cfg.pool.size,
        "batch_size": cfg.training.meta_batch_size,
        "n_message_steps": effective_n_message_steps,
        "reset_interval": cfg.pool.reset_interval,
        "reset_fraction": cfg.pool.reset_fraction,
    }
    stats = calculate_expected_pool_updates(**pool_params)

    # Process damage configuration (requires pool config to be processed first)
    cfg = process_damage_configuration(cfg, expected_lifetime_epochs=stats.expected_lifetime_epochs)

    # Configure global build_graph function with settings from config
    log.info(
        f"Configuring graph builder: neighboring_connections={cfg.graph.neighboring_connections}, bidirectional_edges={cfg.graph.bidirectional_edges}"
    )
    configure_build_graph(
        neighboring_connections=cfg.graph.neighboring_connections,
        bidirectional_edges=cfg.graph.bidirectional_edges,
    )

    # Verify configuration was applied
    if is_configured():
        log.info("✅ Graph builder successfully configured with explicit settings")
    else:
        log.warning("⚠️ Graph builder configuration failed, will use defaults")

    log.info("Final Pool Configuration:")
    log.info(f"  Pool Size: {cfg.pool.size}")
    log.info(f"  Batch Size: {cfg.training.meta_batch_size}")
    log.info(f"  Message Steps: {effective_n_message_steps}")
    log.info(f"  Reset Interval: {cfg.pool.reset_interval}")
    log.info(f"  Reset Fraction: {cfg.pool.reset_fraction:.4f}")
    log.info(f"  Expected Updates per Circuit: {stats.expected_updates:.2f}")
    log.info(f"  Selection Probability: {stats.selection_probability:.4f}")
    log.info(f"  Expected Lifetime: {stats.expected_lifetime_epochs:.1f} epochs")

    # Log damage configuration if enabled
    if cfg.damage.enabled:
        damage_mode = cfg.damage.mode
        log.info("Final Damage Configuration:")
        log.info(f"  Mode: {damage_mode.upper()}")

        if damage_mode == "probabilistic":
            # p_fault will be computed later, but log the target
            expected_faulty = cfg.damage.get("expected_faulty_gates_at_reset", 4)
            log.info(f"  Expected Faulty Gates at Reset: {expected_faulty}")
            log.info(
                f"  Circuit Lifetime: {stats.expected_lifetime_epochs:.1f} epochs x {effective_n_message_steps} steps"
            )
            log.info(f"  (p_fault will be auto-computed before training)")
        else:
            # Discrete mode - log interval-based config
            damage_stats = calculate_expected_damages(
                expected_updates=stats.expected_lifetime_epochs,
                damage_interval=cfg.damage.damage_interval,
                damage_fraction=cfg.damage.damage_fraction,
                knockouts_per_event=cfg.damage.knockouts_per_event,
            )
            log.info(f"  Knockouts per Event: {cfg.damage.knockouts_per_event}")
            log.info(f"  Damage Interval: {cfg.damage.damage_interval} epochs")
            log.info(f"  Damage Fraction: {cfg.damage.damage_fraction:.4f}")
            log.info(f"  Expected Damages per Circuit: {damage_stats.expected_damages:.2f}")
            log.info(f"  Expected Total Knockouts: {damage_stats.expected_knockouts:.2f}")

        log.info(f"  Faulty Logit Value: {cfg.damage.faulty_logit_value}")
        if cfg.damage.max_damage_per_circuit is not None:
            log.info(f"  Max Damage per Circuit: {cfg.damage.max_damage_per_circuit}")
    else:
        log.info("Damage Configuration: DISABLED")

    # Set random seed
    rng = jax.random.PRNGKey(cfg.seed)

    # Create output directory
    if cfg.output.dir is not None:
        output_dir = cfg.output.dir
        os.makedirs(output_dir, exist_ok=True)
    else:
        output_dir = os.getcwd()

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
            reinit="finish_previous",
        )
        wandb_run = wandb

    log.info(f"Output directory: {output_dir}")

    # Generate circuit layer sizes
    input_n, output_n = cfg.circuit.input_bits, cfg.circuit.output_bits
    arity = cfg.circuit.arity

    if cfg.circuit.layer_sizes is None:
        layer_sizes = generate_layer_sizes(
            input_n,
            output_n,
            arity,
            layer_n=cfg.circuit.num_layers,
            width_factor=cfg.circuit.width_factor,
        )
        with open_dict(cfg):
            cfg.circuit.layer_sizes = layer_sizes
    else:
        layer_sizes = cfg.circuit.layer_sizes

    # Generate dummy circuit
    test_key = jax.random.PRNGKey(cfg.test_seed)
    wires, logits = gen_circuit(test_key, cfg.circuit.layer_sizes, arity=cfg.circuit.arity)

    # Generate dummy graph using globally configured function
    graph = configured_build_graph(
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

    if cfg.training.test_num is not None:
        if isinstance(cfg.training.test_num, float):
            assert cfg.training.test_num >= 0 and cfg.training.test_num <= 1, (
                "test_num must be a float between 0 and 1, or an integer"
            )
            test_ratio = cfg.training.test_num
        else:
            assert isinstance(cfg.training.test_num, int), "test_num must be a float or an integer"
            assert cfg.training.test_num > 0, (
                "test_num must be greater than 0 and less than the total number of cases"
            )
            if cfg.training.test_num > case_n:
                log.warning(
                    f"test_num is greater than the total number of cases, setting test_ratio to 1"
                )
                test_ratio = 1
            else:
                test_ratio = cfg.training.test_num / case_n

        # Max 1/4 of the data for testing
        test_ratio = min(test_ratio, 0.25)
        print(f"Test ratio: {test_ratio}")
    else:
        test_ratio = None

    (x_train, y_train), (x_test, y_test), (x_total, y_total) = get_task_data(
        cfg.circuit.task,
        case_n,
        input_bits=input_n,
        output_bits=output_n,
        text=cfg.circuit.get("text", None),
        train_test_split=test_ratio is not None,
        test_ratio=test_ratio,
        seed=cfg.seed,
    )
    data_dict = {
        "x_train": x_train,
        "y_train": y_train,
        "x_test": x_test,
        "y_test": y_test,
        "x_total": x_total,
        "y_total": y_total,
    }

    # Compute data fraction
    n_train = x_train.shape[0]
    print(f"Train data shape : {x_train.shape}")

    data_fraction = (
        min(cfg.training.data_per_batch / n_train, 1.0)
        if cfg.training.data_per_batch is not None
        else 1.0
    )

    log.info(f"Data Fraction: {data_fraction:.4f}")
    if test_ratio is not None:
        if x_test is not None:
            log.info(
                f"Test Ratio: {test_ratio:.4f} | Sizes = {x_train.shape[0]}, {x_test.shape[0]}, {x_total.shape[0]}"
            )
        else:
            log.info(
                f"Test Ratio: {test_ratio:.4f} | Sizes = {x_train.shape[0]}, {x_total.shape[0]}"
            )

    # Run backpropagation training for comparison if enabled
    bp_results = None
    if cfg.backprop.enabled:
        bp_results = run_backpropagation_training(
            cfg, x_train, y_train, loss_cfg=LossConfig.from_dict(dict(cfg.loss))
        )
        plot_training_curves(bp_results, "Backpropagation", os.path.join(output_dir, "plots"))

    # Initialize model
    rng, init_rng = jax.random.split(rng)

    # Common overrides for hydra.instantiate
    # These are values computed in train.py or essential for all models
    instantiate_overrides = {"arity": arity, "rngs": nnx.Rngs(params=init_rng)}

    # Specific overrides based on model type, which should still be in the YAML.
    # Alternatively, we could inspect cfg.model._target_ if 'type' was removed.
    if cfg.model.type in ["self_attention", "perceiver_attention"]:
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
        if cfg.checkpoint.save_on_mnt_storage:
            checkpoint_dir = checkpoint_dir.replace("/home/", "/mnt/storage/")
    else:
        checkpoint_dir = None

    # Get track_metrics configuration for training
    track_metrics = extract_track_metrics_config(cfg)

    # Compute p_fault for probabilistic damage mode
    p_fault = process_probabilistic_damage_configuration(cfg, layer_sizes)

    # Train model
    log.info(f"Starting {cfg.model.type.upper()} training")
    model_results = train_model(
        # Initialization parameters
        key=cfg.seed,
        init_model=model,
        # Data parameters
        data_dict=data_dict,
        data_fraction=data_fraction,
        # Model architecture parameters
        layer_sizes=layer_sizes,
        circuit_hidden_dim=cfg.model.circuit_hidden_dim,
        arity=arity,
        # Training hyperparameters
        learning_rate=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
        epochs=cfg.training.epochs or 2**cfg.training.epochs_power_of_2,
        n_message_steps=cfg.training.n_message_steps,
        use_scan=cfg.training.use_scan,
        gradient_checkpointing=cfg.training.gradient_checkpointing,
        # Loss parameters
        loss_cfg=LossConfig.from_dict(dict(cfg.loss)),
        random_loss_step=cfg.training.random_loss_step,
        random_loss_step_min=cfg.training.random_loss_step_min,
        use_beta_loss_step=cfg.training.use_beta_loss_step,
        # Wiring mode parameters
        wiring_mode=cfg.training.wiring_mode,
        meta_batch_size=cfg.training.meta_batch_size,
        # Multi-GPU data parallelism
        multi_gpu_enabled=cfg.training.multi_gpu.enabled,
        multi_gpu_num_devices=cfg.training.multi_gpu.num_devices,
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
        periodic_eval_enabled=cfg.eval.enabled,
        periodic_eval_interval=cfg.eval.interval,
        periodic_eval_inner_steps=cfg.eval.inner_steps,
        periodic_eval_test_seed=cfg.test_seed,
        periodic_eval_log_stepwise=cfg.eval.log_stepwise,
        periodic_eval_batch_size_in=cfg.eval.batch_size_in,
        periodic_eval_batch_size_out=cfg.eval.batch_size_out,
        periodic_eval_do_ood_evaluation=cfg.eval.do_ood_evaluation,
        periodic_eval_log_pool_scatter=cfg.eval.log_pool_scatter,
        periodic_eval_damage_enabled=cfg.eval.damage_enabled,
        periodic_eval_n_damage_steps=cfg.eval.n_damage_steps,
        periodic_eval_get_all_wirings=cfg.eval.get_all_wirings,
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
        save_best=cfg.checkpoint.save_best,
        track_metrics=track_metrics,
        # Damage parameters for resilience testing
        damage_enabled=cfg.damage.enabled,
        damage_mode=cfg.damage.mode,
        p_fault=p_fault,
        damage_interval=cfg.damage.damage_interval,
        damage_fraction=cfg.damage.damage_fraction,
        knockouts_per_event=cfg.damage.knockouts_per_event,
        random_knockouts_per_event=cfg.damage.random_knockouts_per_event,
        random_knockouts_per_event_min=cfg.damage.random_knockouts_per_event_min,
        random_knockouts_per_event_max=cfg.damage.random_knockouts_per_event_max,
        max_damage_per_circuit=cfg.damage.max_damage_per_circuit,
        faulty_logit_value=cfg.damage.faulty_logit_value,
        # Debugging parameters
        do_check_gradients=cfg.training.check_gradients,
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

    # Get track_metrics configuration
    track_metrics = extract_track_metrics_config(cfg)

    # Run comprehensive evaluation using standardized datasets
    if cfg.eval.enabled:
        # Create standardized evaluation datasets
        log.info("Creating standardized evaluation datasets (seed + pool + OOD)")

        # Create evaluation datasets using standardized approach
        datasets = create_unified_evaluation_datasets(
            evaluation_base_seed=cfg.test_seed,
            training_wiring_mode=cfg.training.wiring_mode,
            training_initial_diversity=cfg.pool.initial_diversity,
            layer_sizes=layer_sizes,
            arity=cfg.circuit.arity,
            eval_batch_size_in=cfg.eval.batch_size_in
            if cfg.eval.batch_size_in is not None
            else cfg.training.initial_diversity,
            eval_batch_size_out=cfg.eval.batch_size_out
            if cfg.eval.batch_size_out is not None
            else cfg.training.meta_batch_size,
            do_ood_evaluation=cfg.eval.do_ood_evaluation
            if cfg.eval.do_ood_evaluation is not None
            else cfg.training.wiring_mode == "random",
            get_all_wirings=cfg.eval.get_all_wirings,
        )
        eval_results = run_unified_periodic_evaluation(
            model=model_results["model"],
            datasets=datasets,
            pool=model_results.get("pool", None),
            data_dict={
                "x_train": x_train,
                "y_train": y_train,
                "x_test": x_test,
                "y_test": y_test,
                "x_total": x_total,
                "y_total": y_total,
            },
            input_n=input_n,
            arity=arity,
            circuit_hidden_dim=cfg.model.circuit_hidden_dim,
            n_message_steps=cfg.eval.inner_steps,
            loss_cfg=LossConfig.from_dict(dict(cfg.loss)),
            epoch=-1,  # Final evaluation marker
            wandb_run=wandb_run,
            log_stepwise=False,
            layer_sizes=layer_sizes,
            log_pool_scatter=False,
            # Best model tracking parameters (final evaluation, so no saving)
            track_metrics=track_metrics,
        )
    else:
        eval_results = None

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
            run_id=wandb_run.run.id,
            dry_run=False,
            verbose=True,
            entity=cfg.wandb.entity,
            project=cfg.wandb.project,
        )
        wandb.finish()

    return final_results


if __name__ == "__main__":
    main()
