"""
Training loop implementation for GNN-based boolean circuit optimization.

This module provides functions for training GNN models to optimize
boolean circuits over multiple epochs.
"""

import logging
from functools import partial
from typing import Any

import jax
import jax.numpy as jp
import jraph
import optax
from flax import nnx
from tqdm.auto import tqdm

import wandb
from boolean_nca_cc.circuits.train import LOSS_L4, LossConfig
from boolean_nca_cc.circuits.viz import plot_wandb_stepwise_results
from boolean_nca_cc.models import CircuitGatheredAttention, CircuitGNN, CircuitSelfAttention
from boolean_nca_cc.training.checkpointing import (
    BestModelTracker,
    EarlyStopping,
    save_periodic_checkpoint,
    setup_checkpoint_dir,
)
from boolean_nca_cc.training.eval_datasets import (
    UnifiedEvaluationDatasets,
    create_unified_evaluation_datasets,
)
from boolean_nca_cc.training.evaluation import (
    evaluate_model_stepwise_batched,
    get_fraction_damaged_gates,
)
from boolean_nca_cc.training.pool.pool import GraphPool, initialize_graph_pool
from boolean_nca_cc.training.schedulers import (
    get_learning_rate_schedule,
    get_step_beta,
    should_reset_pool,
)
from boolean_nca_cc.training.sharding import ShardingContext
from boolean_nca_cc.training.utils import check_gradients

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

        # Get the unique run ID for checkpointing
        log.info(f"WandB run ID: {wandb.run.id}")
        return wandb
    except ImportError:
        log.warning("wandb not installed. Running without wandb logging.")
        return None
    except Exception as e:
        log.warning(f"Error initializing wandb: {e}. Running without wandb logging.")
        return None


def _log_to_wandb(wandb_run, metrics_dict: dict, epoch: int, log_interval: int = 1) -> None:
    """Log metrics to wandb if enabled and interval allows."""
    if wandb_run is None or epoch % log_interval != 0:
        return

    try:
        wandb_run.log(metrics_dict)
    except Exception as e:
        log.warning(f"Error logging to wandb: {e}")


def _log_final_wandb_metrics(wandb_run, results: dict, epochs: int) -> None:
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
                f"best/{results.get('best_metric', 'metric')}": results.get("best_metric_value", 0),
            }
        )

    except Exception as e:
        log.warning(f"Error logging final metrics to wandb: {e}")


def _log_pool_scatter(pool, epoch, wandb_run):
    """Log pool scatterplots to wandb (loss vs steps, loss vs damage)."""
    if wandb_run is None:
        return

    all_loss = pool.graphs.globals.loss
    all_steps = pool.graphs.globals.update_steps

    # Loss vs Steps scatter
    data = list(zip(all_steps, all_loss, strict=False))
    table = wandb.Table(data=data, columns=["steps", "loss"])
    wandb_run.log({"pool/scatter_steps_vs_loss": wandb.plot.scatter(table, "steps", "loss")})

    # Loss vs Faulty Gates scatter (if damage tracking available)
    if pool.damage_count is not None:
        damage_data = list(zip(pool.damage_count, all_loss, strict=False))
        damage_table = wandb.Table(data=damage_data, columns=["faulty_gates", "loss"])
        wandb_run.log(
            {
                "pool/scatter_damage_vs_loss": wandb.plot.scatter(
                    damage_table, "faulty_gates", "loss"
                )
            }
        )


def run_unified_periodic_evaluation(
    model,
    datasets: UnifiedEvaluationDatasets,
    pool,
    data_dict: dict[str, jp.ndarray],
    input_n,
    arity,
    circuit_hidden_dim,
    n_message_steps,
    loss_cfg,
    epoch,
    wandb_run,
    log_stepwise=False,
    layer_sizes: list[tuple[int, int]] | None = None,
    log_pool_scatter: bool = False,
    # Best model tracking parameters
    best_model_tracker=None,
    checkpoint_path: str | None = None,
    save_best: bool = True,
    optimizer=None,
    training_metrics: dict | None = None,
    track_metrics: list[str] | None = None,
    # Probabilistic damage (for training-consistent evaluation)
    p_fault: float | None = None,
    faulty_value: float = -10.0,
    permanent_damage: bool = True,
    # Discrete damage (for visualization with fixed steps)
    damage_steps=None,
    knockout_per_damage_step=1,
    damage_key=jax.random.PRNGKey(42),
    # Delayed probabilistic damage onset
    p_fault_onset_step: int = 0,
    # No-repair baseline
    compute_no_repair_baseline: bool = False,
) -> dict:
    """
    Run unified periodic evaluation with IN/OUT distribution and train/test data splits.

    For meta-learning, we evaluate on both train and test data separately because
    models like Perceiver condition on the data during optimization. This gives us:
    - eval_in_train, eval_in_test: IN-distribution circuits on train/test data
    - eval_out_train, eval_out_test: OUT-distribution circuits on train/test data

    **Damage Modes:**
    - Main evaluation uses p_fault (probabilistic) to match training conditions
    - Visualization uses damage_steps (discrete) for clean, fixed damage markers

    Args:
        model: The model to evaluate
        datasets: UnifiedEvaluationDatasets object containing IN and OUT distribution circuits
        pool: GraphPool for logging scatter plot
        data_dict: Dictionary containing test input and target data
        input_n: Number of input nodes
        arity: Arity of gates
        circuit_hidden_dim: Hidden dimension
        n_message_steps: Number of message steps for evaluation
        loss_cfg: Loss config dict
        epoch: Current epoch number
        wandb_run: WandB run object (or None)
        log_stepwise: Whether to log step-by-step metrics
        layer_sizes: Circuit layer sizes
        log_pool_scatter: Whether to log pool scatterplot (loss vs steps)
        best_model_tracker: BestModelTracker instance for tracking best models (optional)
        checkpoint_path: Path to save checkpoints (optional)
        save_best: Whether to save best models (default: True)
        optimizer: Optimizer to save with checkpoints (optional)
        training_metrics: Training metrics dict for tracking (optional)
        track_metrics: List of specific metrics to track and save (optional)
        p_fault: Per-gate-per-step failure probability for training-consistent evaluation
        faulty_value: Value to set for failed gate logits
        permanent_damage: Whether to apply permanent damage to gates (True) or temporary damage (False)
        damage_steps: Fixed damage steps for visualization (discrete mode)
        knockout_per_damage_step: Gates to knock out at each discrete damage step
        damage_key: Random key for damage

    Returns:
        Dictionary with evaluation metrics from all evaluation scenarios
        and information about best model updates
    """
    try:
        # Define wiring scenarios: (key, wires, logits, batch_size, description)
        wiring_scenarios = [
            (
                "in",
                datasets.in_distribution_wires,
                datasets.in_distribution_logits,
                datasets.in_actual_batch_size,
                "IN-distribution",
            ),
            (
                "out",
                datasets.out_of_distribution_wires,
                datasets.out_of_distribution_logits,
                datasets.out_actual_batch_size,
                "OUT-of-distribution",
            ),
        ]

        # Define data scenarios: (suffix, x, y, description)
        # For meta-learning: models condition on data during optimization, so train/test must be separate
        data_scenarios = []

        if data_dict["x_test"] is not None and data_dict["y_test"] is not None:
            data_scenarios.append(("test", data_dict["x_test"], data_dict["y_test"]))
            print(f"test data shape: {data_dict['x_test'].shape}")
            n_test = data_dict["x_test"].shape[0]
        else:
            n_test = None

        # skip training data for now
        if data_dict["x_train"] is not None and data_dict["y_train"] is not None and False:
            # restrict data to the size of n_test
            data_scenarios.append(
                (
                    "train",
                    data_dict["x_train"][:n_test] if n_test is not None else data_dict["x_train"],
                    data_dict["y_train"][:n_test] if n_test is not None else data_dict["y_train"],
                )
            )
            print(f"x_train shape: {data_scenarios[-1][1].shape}")

        # Results storage: step_metrics[key] where key is e.g. "in_test", "out_train"
        step_metrics = {}
        final_metrics = {}

        # Helper to run evaluation and extract final metrics
        def run_eval(
            wiring_key, data_suffix, wires, logits, batch_size, x, y, wiring_desc, with_damage=False
        ):
            if wires is None:
                return None, None

            # Build full key: e.g., "in_test", "damaged_in_test"
            full_key = (
                f"damaged_{wiring_key}_{data_suffix}"
                if with_damage
                else f"{wiring_key}_{data_suffix}"
            )
            damage_suffix = "DAMAGED " if with_damage else ""

            log.info(
                f"Running {damage_suffix}{wiring_desc} ({data_suffix}) evaluation ({batch_size} circuits)..."
            )
            _, result = evaluate_model_stepwise_batched(
                model=model,
                batch_wires=wires,
                batch_logits=logits,
                x_data=x,
                y_data=y,
                input_n=input_n,
                arity=arity,
                circuit_hidden_dim=circuit_hidden_dim,
                n_message_steps=n_message_steps,
                loss_cfg=loss_cfg,
                layer_sizes=layer_sizes,
                p_fault=p_fault if with_damage else None,
                faulty_value=faulty_value,
                permanent_damage=permanent_damage,
                p_fault_onset_step=p_fault_onset_step if with_damage else 0,
                compute_no_repair_baseline=compute_no_repair_baseline if with_damage else False,
                chunk_size=datasets.target_batch_size,
                return_first_circuit_details=False,
            )

            prefix = f"eval_{full_key}"
            metrics = {
                f"{prefix}/final_loss": result["loss"][-1],
                f"{prefix}/final_hard_loss": result["hard_loss"][-1],
                f"{prefix}/final_accuracy": result["accuracy"][-1],
                f"{prefix}/final_hard_accuracy": result["hard_accuracy"][-1],
                f"{prefix}/epoch": epoch,
            }

            # Add no-repair baseline final metrics if available
            if with_damage and compute_no_repair_baseline and "no_repair_hard_accuracy" in result:
                metrics[f"{prefix}/final_no_repair_loss"] = result["no_repair_loss"][-1]
                metrics[f"{prefix}/final_no_repair_hard_loss"] = result["no_repair_hard_loss"][-1]
                metrics[f"{prefix}/final_no_repair_accuracy"] = result["no_repair_accuracy"][-1]
                metrics[f"{prefix}/final_no_repair_hard_accuracy"] = result["no_repair_hard_accuracy"][-1]

            return result, metrics

        # Run evaluations: loop over wiring (in/out) x data (train/test)
        for wiring_key, wires, logits, batch_size, wiring_desc in wiring_scenarios:
            if wires is None:
                log.info(f"No {wiring_desc} evaluation data available.")
                continue
            for data_suffix, x, y in data_scenarios:
                full_key = f"{wiring_key}_{data_suffix}"
                step_metrics[full_key], final_metrics[full_key] = run_eval(
                    wiring_key, data_suffix, wires, logits, batch_size, x, y, wiring_desc
                )

        # Run damaged evaluations (uses p_fault for training-consistent metrics)
        if p_fault is not None:
            for wiring_key, wires, logits, batch_size, wiring_desc in wiring_scenarios:
                if wires is None:
                    continue
                for data_suffix, x, y in data_scenarios:
                    full_key = f"damaged_{wiring_key}_{data_suffix}"
                    step_metrics[full_key], final_metrics[full_key] = run_eval(
                        wiring_key,
                        data_suffix,
                        wires,
                        logits,
                        batch_size,
                        x,
                        y,
                        wiring_desc,
                        with_damage=True,
                    )

        # Check if any evaluation ran
        if all(v is None for v in final_metrics.values()):
            log.info("No evaluation data available.")
            return {}

        # Combine all metrics for logging
        combined_metrics = {}
        for m in final_metrics.values():
            if m is not None:
                combined_metrics.update(m)

        # Log to wandb if enabled
        if wandb_run:
            wandb_run.log(combined_metrics)

            if log_pool_scatter:
                _log_pool_scatter(pool, epoch, wandb_run)

            # Create and log stepwise plots (averaged across batch, much less noisy)
            if log_stepwise:
                try:
                    import matplotlib.pyplot as plt

                    def _damage_frac(result_dict):
                        """Extract damage fraction from graphs if available."""
                        graphs = result_dict.get("graphs")
                        if graphs is None:
                            return None
                        try:
                            return get_fraction_damaged_gates(graphs)
                        except Exception:
                            return None

                    for wiring_key, wires, logits, batch_size, wiring_desc in wiring_scenarios:
                        if wires is None:
                            continue

                        for data_suffix, x, y in data_scenarios:
                            full_key = f"{wiring_key}_{data_suffix}"
                            prefix = f"eval_{full_key}"

                            # --- Undamaged stepwise (averaged across batch) ---
                            if step_metrics.get(full_key) is not None:
                                fig = plot_wandb_stepwise_results(
                                    step_metrics[full_key],
                                    title=f"{wiring_desc} ({data_suffix}) | {batch_size} circuits avg",
                                )
                                wandb_run.log({f"stepwise/{prefix}": wandb_run.Image(fig)})
                                plt.close(fig)

                            # --- p_fault damaged stepwise (averaged across batch) ---
                            damaged_key = f"damaged_{full_key}"
                            if step_metrics.get(damaged_key) is not None:
                                fig = plot_wandb_stepwise_results(
                                    step_metrics[damaged_key],
                                    title=f"{wiring_desc} ({data_suffix}) | p_fault={p_fault:.1e} | {batch_size} circuits avg",
                                    damage_fraction=_damage_frac(step_metrics[damaged_key]),
                                )
                                wandb_run.log(
                                    {f"stepwise/eval_{damaged_key}": wandb_run.Image(fig)}
                                )
                                plt.close(fig)

                            # --- Discrete damage stepwise (separate eval, averaged across batch) ---
                            if damage_steps is not None and step_metrics.get(full_key) is not None:
                                log.info(
                                    f"Running discrete damage eval for {wiring_desc} ({data_suffix})..."
                                )
                                _, discrete_result = evaluate_model_stepwise_batched(
                                    model=model,
                                    batch_wires=wires,
                                    batch_logits=logits,
                                    x_data=x,
                                    y_data=y,
                                    input_n=input_n,
                                    arity=arity,
                                    circuit_hidden_dim=circuit_hidden_dim,
                                    n_message_steps=n_message_steps,
                                    loss_cfg=loss_cfg,
                                    layer_sizes=layer_sizes,
                                    damage_steps=damage_steps,
                                    knockout_per_damage_step=knockout_per_damage_step,
                                    damage_key=damage_key,
                                    p_fault=None,  # Pure discrete damage, no probabilistic
                                    permanent_damage=permanent_damage,
                                    # Always compute no-repair baseline for discrete damage
                                    # (shotgun damage is where the repair benefit is most visible)
                                    compute_no_repair_baseline=compute_no_repair_baseline,
                                    chunk_size=datasets.target_batch_size,
                                    return_first_circuit_details=False,
                                )
                                fig = plot_wandb_stepwise_results(
                                    discrete_result,
                                    damage_steps=damage_steps,
                                    title=f"{wiring_desc} ({data_suffix}) | damage steps={damage_steps} | {batch_size} circuits avg",
                                    damage_fraction=_damage_frac(discrete_result),
                                )
                                wandb_run.log(
                                    {
                                        f"stepwise/eval_discrete_damaged_{full_key}": wandb_run.Image(
                                            fig
                                        )
                                    }
                                )
                                plt.close(fig)

                    plt.close("all")

                except Exception as e:
                    import traceback

                    log.warning(f"Error creating wandb stepwise plots: {e}")
                    log.warning(f"Traceback: {traceback.format_exc()}")

        # Log summary to console
        training_config = datasets.training_config
        log_message_parts = [f"Unified Eval (epoch {epoch}):"]

        # Helper for chunk info string
        def get_chunk_info(batch_size):
            if batch_size is not None and batch_size > datasets.target_batch_size:
                n_chunks = (
                    batch_size + datasets.target_batch_size - 1
                ) // datasets.target_batch_size
                return f", {n_chunks} chunks"
            return ""

        # Log results for each wiring x data combination
        for wiring_key, batch_size, wiring_desc in [
            ("in", datasets.in_actual_batch_size, "IN-distribution"),
            ("out", datasets.out_actual_batch_size, "OUT-of-distribution"),
        ]:
            for data_suffix, _, _ in data_scenarios:
                full_key = f"{wiring_key}_{data_suffix}"
                m = final_metrics.get(full_key)
                if m is not None:
                    prefix = f"eval_{full_key}"
                    extra = (
                        f", mode={training_config['wiring_mode']}, diversity={training_config['initial_diversity']}"
                        if wiring_key == "in"
                        else ", random"
                    )
                    log_message_parts.append(
                        f"  {wiring_desc} ({data_suffix}, {batch_size} circuits{get_chunk_info(batch_size)}{extra}): "
                        f"Loss={m[f'{prefix}/final_loss']:.4f}, "
                        f"Acc={m[f'{prefix}/final_accuracy']:.4f}, "
                        f"Hard Acc={m[f'{prefix}/final_hard_accuracy']:.4f}"
                    )
                elif wiring_key == "in" and data_suffix == "test":
                    log_message_parts.append(
                        f"  {wiring_desc}: Not available (training mode: {training_config['wiring_mode']})"
                    )

        log.info("\n".join(log_message_parts))

        # Prepare evaluation metrics for best model tracking (all data splits, including damaged)
        eval_metrics = {}
        for key in final_metrics:
            if final_metrics.get(key):
                eval_metrics.update(final_metrics[key])

        # Track and save best models if tracker is provided
        best_model_updates = {}
        if best_model_tracker is not None and optimizer is not None:
            from boolean_nca_cc.training.checkpointing import track_and_save_best_models

            checkpoint_metrics = {
                "eval_in_metrics": final_metrics.get("in_test") or {},
                "eval_out_metrics": final_metrics.get("out_test") or {},
                "training_metrics": training_metrics or {},
                "datasets_info": {
                    "in_actual_batch_size": datasets.in_actual_batch_size,
                    "out_actual_batch_size": datasets.out_actual_batch_size,
                    "target_batch_size": datasets.target_batch_size,
                    "training_wiring_mode": training_config["wiring_mode"],
                    "training_initial_diversity": training_config["initial_diversity"],
                    "evaluation_base_seed": training_config["evaluation_base_seed"],
                },
            }

            best_model_updates = track_and_save_best_models(
                best_model_tracker=best_model_tracker,
                checkpoint_path=checkpoint_path,
                save_best=save_best,
                model=model,
                optimizer=optimizer,
                metrics=checkpoint_metrics,
                epoch=epoch,
                training_metrics=training_metrics,
                eval_metrics=eval_metrics,
                wandb_run=wandb_run,
                track_metrics=track_metrics,
            )

            if best_model_updates and wandb_run:
                for metric_key, update_info in best_model_updates.items():
                    wandb_run.log(
                        {
                            f"best_model_updates/{metric_key}": update_info["value"],
                            f"best_model_updates/{metric_key}_epoch": update_info["epoch"],
                        }
                    )

        # Return all metrics using new dict structure
        return {
            "step_metrics": step_metrics,
            "final_metrics": final_metrics,
            # Backward compatibility aliases (use test data)
            "step_metrics_in": step_metrics.get("in_test"),
            "step_metrics_out": step_metrics.get("out_test"),
            "final_metrics_in": final_metrics.get("in_test"),
            "final_metrics_out": final_metrics.get("out_test"),
            "best_model_updates": best_model_updates,
            "datasets_info": {
                "in_actual_batch_size": datasets.in_actual_batch_size,
                "out_actual_batch_size": datasets.out_actual_batch_size,
                "target_batch_size": datasets.target_batch_size,
                "in_used_chunking": datasets.in_actual_batch_size is not None
                and datasets.in_actual_batch_size > datasets.target_batch_size,
                "out_used_chunking": datasets.out_actual_batch_size is not None
                and datasets.out_actual_batch_size > datasets.target_batch_size,
                "training_wiring_mode": training_config["wiring_mode"],
                "training_initial_diversity": training_config["initial_diversity"],
                "evaluation_base_seed": training_config["evaluation_base_seed"],
            },
        }

    except Exception as e:
        import traceback

        tb_str = traceback.format_exc()
        log.warning(f"Error during unified periodic evaluation at epoch {epoch}: {e}\n{tb_str}")
        return {}


def train_model(
    # Data parameters
    data_dict: dict[str, jp.ndarray],
    data_fraction: float = 1.0,
    # Model architecture parameters
    layer_sizes: list[tuple[int, int]] | None = None,
    arity: int = 2,
    circuit_hidden_dim: int = 16,
    # Training hyperparameters
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    epochs: int = 100,
    n_message_steps: int = 1,
    use_scan: bool = False,
    gradient_checkpointing: bool = False,  # Recompute model activations during backward pass to save memory
    # Loss parameters
    loss_cfg=None,  # Loss config dict (default: LOSS_L4)
    random_loss_step: bool = False,  # Use random message passing step for loss computation
    random_loss_step_min: int = 0,
    use_beta_loss_step: bool = False,  # Use beta distribution for random loss step (varies from early to late steps through training)
    # Wiring mode parameters
    wiring_mode: str = "random",  # Options: 'fixed', 'random', or 'genetic'
    meta_batch_size: int = 64,
    # Multi-GPU parameters
    multi_gpu_enabled: bool
    | None = None,  # None = auto (enable if >1 device), True/False = explicit
    multi_gpu_num_devices: int | None = None,  # Number of devices (None = all available)
    # Genetic mutation parameters (only used when wiring_mode='genetic')
    genetic_mutation_rate: float = 0.0,  # Fraction of connections to mutate (0.0 to 1.0)
    genetic_swaps_per_layer: int = 1,  # Number of swaps per layer for genetic mutation
    initial_diversity: int = 1,  # Number of initial wires for genetic mutation
    # Pool parameters
    pool_size: int = 1024,
    reset_pool_fraction: float = 0.05,
    reset_pool_interval: int = 128,
    reset_strategy: str = "uniform",  # Options: "uniform", "steps_biased", "loss_biased", or "combined"
    combined_weights: tuple[float, float] = (
        0.5,
        0.5,
    ),  # Weights for [loss, steps] in combined strategy
    pool_noise_scale: float = 0.0,
    # Learning rate scheduling
    lr_scheduler: str = "constant",  # Options: "constant", "exponential", "cosine", "linear_warmup"
    lr_scheduler_params: dict | None = None,
    # Initialization parameters
    key: int = 0,
    wiring_fixed_key: jax.random.PRNGKey = jax.random.PRNGKey(
        42
    ),  # Fixed key for generating wirings when wiring_mode='fixed'
    init_model: CircuitGatheredAttention | CircuitGNN | CircuitSelfAttention | None = None,
    init_optimizer: nnx.Optimizer | None = None,
    initial_metrics: dict | None = None,
    # Checkpointing parameters
    checkpoint_enabled: bool = False,
    checkpoint_dir: str | None = None,
    checkpoint_interval: int = 10,
    save_best: bool = True,
    best_metric: str = "hard_accuracy",  # Options: 'loss', 'hard_loss', 'accuracy', 'hard_accuracy'
    best_metric_source: str = "training",  # Options: 'training' or 'eval'
    # Periodic evaluation parameters
    periodic_eval_enabled: bool = False,
    periodic_eval_inner_steps: int = 100,
    periodic_eval_interval: int = 1024,
    periodic_eval_test_seed: int = 42,
    periodic_eval_log_stepwise: bool = False,
    periodic_eval_batch_size_in: int
    | None = None,  # Batch size for IN-distribution evaluation (None means use initial_diversity)
    periodic_eval_batch_size_out: int
    | None = None,  # Batch size for OUT-of-distribution evaluation (None means use meta_batch_size)
    periodic_eval_do_ood_evaluation: bool
    | None = None,  # Whether to do OUT-of-distribution evaluation (None means use True if wiring_mode is random)
    periodic_eval_log_pool_scatter: bool = False,
    periodic_eval_get_all_wirings: bool = False,
    # Wandb parameters
    wandb_logging: bool = False,
    log_interval: int = 1,
    wandb_run_config: dict | None = None,
    # Early stopping (pass None to disable)
    early_stopping: EarlyStopping | None = None,
    # Best model tracking parameters
    track_metrics: list[str] | None = None,
    # Damage parameters (derived from target_damage_fraction by caller)
    damage_enabled: bool = False,
    p_fault: float | None = None,  # p_fault for training (tuned for pool.expected_updates)
    p_fault_eval: float | None = None,  # p_fault for eval (tuned for eval.inner_steps)
    permanent_damage: bool = True,
    faulty_logit_value: float = -10.0,
    n_damage_steps: int = 0,  # Discrete damage volleys during eval
    knockouts_per_event: int = 1,  # Gates per volley (auto-computed if null in config)
    # Delayed probabilistic damage onset
    p_fault_onset_step_train: int = 0,  # Onset step for training scan
    p_fault_onset_step_eval: int = 0,  # Onset step for eval scan
    # No-repair baseline (for eval only)
    compute_no_repair_baseline: bool = False,
    # Debugging parameters
    do_check_gradients: bool = False,
):
    """
    Train a GNN to optimize boolean circuit parameters.

    Args:
        layer_sizes: List of tuples (nodes, group_size) for each layer
        data_dict: Dictionary containing training and testing data
            "x_train": Input data for training [num_train, input_bits]
            "y_train": Target output data [num_train, output_bits]
            "x_test": Input data for testing [num_test, input_bits]
            "y_test": Target output data [num_test, output_bits]
            "x_total": Input data for total [num_total, input_bits]
            "y_total": Target output data [num_total, output_bits]
        data_fraction: Fraction of boolean circuit data to use for each training batch
        arity: Number of inputs per gate
        circuit_hidden_dim: Dimension of hidden features
        message_passing: Whether to use message passing or only self-updates
        node_mlp_features: Hidden layer sizes for the node MLP
        edge_mlp_features: Hidden layer sizes for the edge MLP
        use_attention: Whether to use attention-based message aggregation
        learning_rate: Learning rate for optimization
        epochs: Number of training epochs
        n_message_steps: Number of message passing steps per pool batch
        loss_cfg: Loss config dict
        random_loss_step: Use random message passing step for loss computation
        random_loss_step_min: Minimum message passing step for random loss computation
        use_beta_loss_step: Use beta distribution for random loss step (varies from early to late steps through training)
        wiring_mode: Mode for circuit wirings ('fixed', 'random', or 'genetic')
        meta_batch_size: Batch size for training
        multi_gpu_enabled: Multi-GPU mode (None = auto-enable if >1 GPU, True/False = explicit)
        multi_gpu_num_devices: Number of devices to use (None = all available)
        genetic_mutation_rate: Fraction of connections to mutate (0.0 to 1.0)
        genetic_swaps_per_layer: Number of swaps per layer for genetic mutation
        pool_size: Size of the graph pool
        reset_pool_fraction: Fraction of pool to reset periodically
        reset_pool_interval: Number of epochs between pool resets
        reset_strategy: Strategy for selecting graphs to reset ("uniform", "steps_biased", "loss_biased", or "combined")
        combined_weights: Tuple of weights (loss_weight, steps_weight) for combining factors in "combined" strategy
        key: Random seed
        wiring_fixed_key: Fixed key for generating wirings when wiring_mode='fixed'
        init_model: Optional pre-trained GNN model to continue training
        init_optimizer: Optional pre-trained optimizer to continue training
        initial_metrics: Optional dictionary of metrics from previous training
        lr_scheduler: Learning rate scheduler type
        lr_scheduler_params: Dictionary of parameters for the scheduler
        checkpoint_dir: Directory to save checkpoints
        checkpoint_interval: How often to save periodic checkpoints
        save_best: Whether to track and save the best model
        best_metric: Metric to use for determining the best model
        best_metric_source: Source of the metric ('training' or 'eval')
        periodic_eval_enabled: Whether to enable periodic evaluation
        periodic_eval_inner_steps: Number of inner steps for periodic evaluation
        periodic_eval_interval: Interval for periodic evaluation
        periodic_eval_test_seed: Seed for periodic evaluation test circuit generation
        periodic_eval_log_stepwise: Whether to log step-by-step evaluation metrics
        periodic_eval_batch_size_in: Batch size for IN-distribution evaluation (None means use initial_diversity)
        periodic_eval_batch_size_out: Batch size for OUT-of-distribution evaluation (None means use meta_batch_size)
        periodic_eval_do_ood_evaluation: Whether to do OUT-of-distribution evaluation (None means use True if wiring_mode is random)
        periodic_eval_get_all_wirings: Whether to get all wirings (True) or a subset (False)
        wandb_logging: Whether to log metrics to wandb
        log_interval: Interval for logging metrics
        wandb_run_config: Configuration to pass to wandb
        early_stopping: EarlyStopping instance (or None to disable).
            Counts consecutive evaluations above threshold; see EarlyStopping docstring.
        track_metrics: List of specific metrics to track and save best models for (e.g.,
                      ["eval_in_hard_accuracy", "eval_out_hard_accuracy"]). If None,
                      tracks all available metrics during evaluation.
        damage_enabled: Whether to enable gate damage during training
        p_fault: Per-gate-per-step failure probability for training (tuned for pool.expected_updates)
        p_fault_eval: Per-gate-per-step failure probability for eval (tuned for eval.inner_steps)
        permanent_damage: Whether to apply permanent damage to gates (True / False / "random")
        faulty_logit_value: Value for knocked-out gate logits (large negative)
        n_damage_steps: Number of discrete damage volleys during eval
        knockouts_per_event: Gates knocked out per discrete damage volley
        p_fault_onset_step_train: Step at which probabilistic damage starts during training
        p_fault_onset_step_eval: Step at which probabilistic damage starts during eval
        compute_no_repair_baseline: Whether to compute no-repair baseline metrics during eval
        do_check_gradients: Whether to check gradients for zero values
    Returns:
        Dictionary with trained GNN model and training metrics
    """
    # Default loss config
    if loss_cfg is None:
        loss_cfg = LOSS_L4
    elif isinstance(loss_cfg, dict):
        loss_cfg = LossConfig.from_dict(loss_cfg)

    # Get data
    x_train, y_train = data_dict["x_train"], data_dict["y_train"]

    # Initialize random key
    rng = jax.random.PRNGKey(key)

    # Convert layer_sizes to tuple once for JAX static arguments
    # This avoids repeated conversions in the training loop
    layer_sizes = tuple(layer_sizes) if not isinstance(layer_sizes, tuple) else layer_sizes

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
        raise ValueError("init_model is required")

    model = init_model

    # Create optimizer or reuse existing optimizer
    adaptive_scheduler = None  # Will be set if using adaptive/reduce_on_plateau

    if init_optimizer is None:
        # Create the learning rate schedule using our scheduler module
        schedule_result = get_learning_rate_schedule(
            lr_scheduler, learning_rate, epochs, lr_scheduler_params
        )

        # Handle adaptive schedulers (return tuple) vs static schedulers
        if isinstance(schedule_result, tuple):
            schedule, adaptive_scheduler = schedule_result
            log.info(f"Using adaptive LR scheduler: {type(adaptive_scheduler).__name__}")
            # For adaptive schedulers, we use inject_hyperparams to allow dynamic LR updates
            # The learning_rate is exposed as a mutable hyperparameter
            opt_fn = optax.inject_hyperparams(optax.adamw)(
                learning_rate=adaptive_scheduler.get_lr(),
                weight_decay=weight_decay,
            )
            # Wrap with gradient clipping
            opt_fn = optax.chain(
                optax.clip_by_global_norm(1.0),
                optax.zero_nans(),
                opt_fn,
            )
        else:
            schedule = schedule_result
            # Create a new optimizer with the static schedule
            opt_fn = optax.chain(
                optax.clip_by_global_norm(1.0),
                optax.zero_nans(),
                optax.adamw(learning_rate=schedule, weight_decay=weight_decay),
            )
        optimizer = nnx.Optimizer(model, opt_fn, wrt=nnx.Param)
    else:
        # Use the provided optimizer
        optimizer = init_optimizer
        schedule = None

    # Initialize Graph Pool for training
    # Use consistent key generation: wiring_fixed_key for fixed/genetic modes, dynamic for random
    if wiring_mode in ["fixed", "genetic"]:
        training_pool_key = wiring_fixed_key
    else:
        # For random mode, use a portion of the main RNG to maintain consistency
        rng, training_pool_key = jax.random.split(rng)

    circuit_pool = initialize_graph_pool(
        rng=training_pool_key,
        layer_sizes=layer_sizes,
        pool_size=pool_size,
        input_n=input_n,
        arity=arity,
        circuit_hidden_dim=circuit_hidden_dim,
        loss_value=0.0,  # Initial loss will be calculated properly in first step
        wiring_mode=wiring_mode,
        initial_diversity=initial_diversity if wiring_mode in ["fixed", "genetic"] else pool_size,
        noise_scale=pool_noise_scale,
    )

    # =========================================================================
    # Multi-GPU sharding setup
    # =========================================================================
    sharding_ctx = ShardingContext(
        batch_size=meta_batch_size,
        num_devices=multi_gpu_num_devices,
        enabled=multi_gpu_enabled,
    )

    # Enter sharding context (validates batch size, creates mesh)
    sharding_ctx.__enter__()

    if sharding_ctx.enabled and sharding_ctx.mesh is not None:
        # Replicate model and optimizer across devices
        model, optimizer = sharding_ctx.replicate(model, optimizer)
        log.info(
            f"Multi-GPU: {sharding_ctx.num_devices} devices, "
            f"{sharding_ctx.per_device_batch_size} circuits/device"
        )

    # =========================================================================
    # Core loss and gradient computation
    # =========================================================================
    def _compute_loss_and_gradients(
        model: CircuitGNN,
        graphs: jraph.GraphsTuple,
        wires: PyTree,
        logits: PyTree,
        x: jp.ndarray,
        y_target: jp.ndarray,
        layer_sizes: tuple[tuple[int, int], ...],
        n_message_steps: int,
        loss_cfg,
        loss_key: jax.random.PRNGKey,
        epoch: int,
        data_fraction: float = 1.0,
        # Probabilistic damage parameters
        p_fault: float | None = None,
        faulty_value: float = -10.0,
        permanent_damage: bool = True,
    ):
        """
        Core loss and gradient computation logic.

        This is the shared implementation used by both:
        - pool_train_step (single batch processing)

        Args:
            model: CircuitGNN or CircuitSelfAttention model
            graphs: Batch of graphs
            wires: Corresponding wires for the graphs
            logits: Corresponding logits for the graphs
            x: Input data
            y_target: Target output data
            layer_sizes: Tuple of (nodes, group_size) tuples for each layer
            n_message_steps: Number of message passing steps
            loss_cfg: Loss config dict
            loss_key: Random key for loss computation
            epoch: Current epoch (used for beta loss step scheduling)
            data_fraction: Fraction of data to use for loss computation
            p_fault: Per-gate-per-step failure probability (None = disabled)
            faulty_value: Value to set for failed gate logits
            permanent_damage: Whether to apply permanent damage to gates (True) or temporary damage (False)
        Returns:
            Tuple of (loss, aux, updated_graphs, updated_logits, loss_steps, grads)
        """

        def get_loss_step(loss_key):
            if random_loss_step:
                if use_beta_loss_step:
                    return get_step_beta(
                        loss_key,
                        n_message_steps,
                        training_progress=epoch / (epochs - 1),
                    )
                else:
                    return jax.random.randint(
                        loss_key, (1,), random_loss_step_min, n_message_steps
                    )[0]
            else:
                return n_message_steps - 1

        def loss_fn_scan(model, graph, logits, wires, loss_key, permanent_damage):
            # Store original shapes for reconstruction
            logits_original_shapes = [logit.shape for logit in logits]

            # Use unified scan function for all model types
            from boolean_nca_cc.training.evaluation import run_model_scan_with_loss

            loss_key, scan_key = jax.random.split(loss_key)

            # Run scan for all steps, computing loss and updating graph at each step
            final_graph, step_outputs = run_model_scan_with_loss(
                model=model,
                graph=graph,
                num_steps=n_message_steps,
                logits_original_shapes=logits_original_shapes,
                wires=wires,
                x_data=x,
                y_data=y_target,
                loss_cfg=loss_cfg,
                layer_sizes=layer_sizes,
                data_fraction=data_fraction,
                scan_key=scan_key,
                gradient_checkpointing=gradient_checkpointing,
                # Probabilistic damage during training
                p_fault=p_fault,
                faulty_value=faulty_value,
                permanent_damage=permanent_damage,
                # Delayed onset (let circuit converge before damage starts)
                p_fault_onset_step=p_fault_onset_step_train,
            )

            loss_step = get_loss_step(loss_key)

            final_graph, final_loss, final_logits, final_aux = jax.tree.map(
                lambda arr: arr[loss_step], step_outputs
            )

            # # Take the mean of the losses until the loss step: more grads !
            # # Use masking instead of dynamic_slice since loss_step is a traced value
            # all_losses = step_outputs[1]  # shape: [n_message_steps]
            # indices = jp.arange(n_message_steps)
            # mask = indices <= loss_step
            # # Compute masked mean: sum of valid losses / count of valid losses
            # final_loss = jp.sum(jp.where(mask, all_losses, 0.0)) / (loss_step + 1)

            return final_loss, (final_aux, final_graph, final_logits, loss_step)

        def loss_fn_no_scan(model, graph, logits, wires, loss_key, permanent_damage):
            # Store original shapes for reconstruction
            from boolean_nca_cc.training.evaluation import (
                _prepare_model_fn,
                apply_model_and_compute_loss,
            )

            logits_original_shapes = [logit.shape for logit in logits]
            loss_step = get_loss_step(loss_key)

            # Prepare model function with precomputed masks (same as scan version)
            model_fn, _ = _prepare_model_fn(
                model, graph, gradient_checkpointing=gradient_checkpointing
            )

            all_results = []

            for _i in range(n_message_steps):
                # Use unified step function for consistency
                graph, loss, current_logits, aux = apply_model_and_compute_loss(
                    model_fn=model_fn,
                    graph=graph,
                    logits_original_shapes=logits_original_shapes,
                    wires=wires,
                    x_data=x,
                    y_data=y_target,
                    loss_cfg=loss_cfg,
                    layer_sizes=layer_sizes,
                )
                all_results.append((loss, aux, graph, current_logits))

            # Stack all results using jax.tree_map
            stacked_results = jax.tree.map(lambda *args: jp.stack(args), *all_results)

            # Index at n_loss_step
            final_loss, final_aux, final_graph, final_logits = jax.tree.map(
                lambda arr: arr[loss_step], stacked_results
            )

            return final_loss, (final_aux, final_graph, final_logits, loss_step)

        def batch_loss_fn(model, graphs, logits, wires, loss_key, permanent_damage):
            loss_fn = loss_fn_scan if use_scan else loss_fn_no_scan

            if permanent_damage == "random":
                loss_key, permanent_damage_key = jax.random.split(loss_key)
                permanent_damage = jax.random.choice(
                    permanent_damage_key, jp.array([True, False]), (graphs.n_node.shape[0],)
                )
            else:
                permanent_damage = jax.numpy.full(graphs.n_node.shape[0], permanent_damage)

            loss_keys = jax.random.split(loss_key, graphs.n_node.shape[0])
            loss, (aux, updated_graphs, updated_logits, loss_steps) = nnx.vmap(
                loss_fn, in_axes=(None, 0, 0, 0, 0, 0)
            )(model, graphs, logits, wires, loss_keys, permanent_damage)

            return jp.mean(loss), (
                jax.tree.map(lambda x: jp.mean(x, axis=0), aux),
                updated_graphs,
                updated_logits,
                jp.mean(loss_steps),
            )

        # Compute loss and gradients
        (loss, (aux, updated_graphs, updated_logits, loss_steps)), grads = nnx.value_and_grad(
            batch_loss_fn, has_aux=True
        )(
            model=model,
            graphs=graphs,
            logits=logits,
            wires=wires,
            loss_key=loss_key,
            permanent_damage=permanent_damage,
        )

        return loss, aux, updated_graphs, updated_logits, loss_steps, grads

    # =========================================================================
    # Single batch training step (processes full batch at once)
    # =========================================================================
    def _pool_train_step(
        model: CircuitGNN,
        optimizer: nnx.Optimizer,
        pool: GraphPool,
        idxs: jp.ndarray,
        graphs: jraph.GraphsTuple,
        wires: PyTree,
        logits: PyTree,
        x: jp.ndarray,
        y_target: jp.ndarray,
        layer_sizes: tuple[tuple[int, int], ...],
        n_message_steps: int,
        loss_cfg,
        loss_key: jax.random.PRNGKey,
        epoch: int,
        data_fraction: float = 1.0,
        # Probabilistic damage parameters
        p_fault: float | None = None,
        faulty_value: float = -10.0,
        permanent_damage: bool = True,
    ):
        """
        Single training step using graphs from the pool.

        Processes the full batch at once.
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
            loss_cfg: Loss config dict
            loss_key: Random key for loss computation
            epoch: Current epoch
            data_fraction: Fraction of data to use for loss computation
            p_fault: Per-gate-per-step failure probability (None = disabled)
            faulty_value: Value to set for failed gate logits
            permanent_damage: Whether to apply permanent damage to gates (True) or temporary damage (False)
        Returns:
            Tuple of (loss, (aux, updated_pool, loss_steps))
        """

        # Compute loss and gradients using shared core logic
        loss, aux, updated_graphs, updated_logits, loss_steps, grads = _compute_loss_and_gradients(
            model=model,
            graphs=graphs,
            wires=wires,
            logits=logits,
            x=x,
            y_target=y_target,
            layer_sizes=layer_sizes,
            n_message_steps=n_message_steps,
            loss_cfg=loss_cfg,
            loss_key=loss_key,
            epoch=epoch,
            data_fraction=data_fraction,
            p_fault=p_fault,
            faulty_value=faulty_value,
            permanent_damage=permanent_damage,
        )

        if do_check_gradients:
            check_gradients(grads)
        # Update GNN parameters
        optimizer.update(model, grads)

        # Extract gate_masks from updated graphs (tracks probabilistic damage)
        updated_gate_masks = updated_graphs.nodes["gate_knockout_mask"]

        # Update pool with the updated graphs, logits, and gate_masks
        updated_pool = pool.update(
            idxs,
            updated_graphs,
            batch_of_logits=updated_logits,
            batch_of_gate_masks=updated_gate_masks,
        )

        return loss, (aux, updated_pool, loss_steps)

    _pool_train_step_jit = partial(
        nnx.jit,
        static_argnames=(
            "layer_sizes",
            "n_message_steps",
            "loss_cfg",
            "data_fraction",
            "p_fault",
            "faulty_value",
            "permanent_damage",
        ),
    )(_pool_train_step)

    # We can't perfrom gradient checking on the JIT-compiled version
    pool_train_step = _pool_train_step if do_check_gradients else _pool_train_step_jit

    # Setup wandb logging if enabled
    wandb_run = _init_wandb(wandb_logging, wandb_run_config)
    wandb_id = wandb_run.run.id if wandb_run else None

    # Setup checkpointing directory
    checkpoint_path = setup_checkpoint_dir(checkpoint_dir, wandb_id)

    # Initialize best model tracker for unified tracking
    best_model_tracker = BestModelTracker()

    # Create progress bar for training
    pbar = tqdm(range(epochs), desc="Training GNN")
    avg_steps_reset = 0

    # Track last reset epoch for scheduling
    last_reset_epoch = -1  # Initialize to -1 so first check works correctly

    # Track damage application
    avg_damage_count = 0.0  # Average knockouts per circuit across pool

    # Initialize evaluation datasets for periodic evaluation if enabled
    eval_datasets = None
    if periodic_eval_enabled:
        log.info("Creating standardized evaluation datasets for periodic evaluation")

        # Create unified evaluation datasets
        eval_datasets = create_unified_evaluation_datasets(
            evaluation_base_seed=periodic_eval_test_seed,
            training_wiring_mode=wiring_mode,
            training_initial_diversity=initial_diversity,
            layer_sizes=layer_sizes,
            arity=arity,
            eval_batch_size_in=periodic_eval_batch_size_in
            if periodic_eval_batch_size_in is not None
            else initial_diversity,
            eval_batch_size_out=periodic_eval_batch_size_out
            if periodic_eval_batch_size_out is not None
            else meta_batch_size,
            do_ood_evaluation=periodic_eval_do_ood_evaluation
            if periodic_eval_do_ood_evaluation is not None
            else wiring_mode == "random",
            get_all_wirings=periodic_eval_get_all_wirings,
            pool_noise_scale=pool_noise_scale,
        )

        log.info(eval_datasets.get_summary())

    diversity = 0.0

    # Training loop
    try:
        for epoch in pbar:
            # Pool-based training
            # Sample a batch from the pool using the current (potentially dynamic) batch size
            rng, sample_key, loss_key = jax.random.split(rng, 3)
            idxs, graphs, wires, logits, _gate_masks = circuit_pool.sample(
                sample_key, meta_batch_size
            )

            # Shard batch data across devices for multi-GPU training
            if sharding_ctx.enabled and sharding_ctx.mesh is not None:
                graphs = sharding_ctx.shard(graphs)
                wires = sharding_ctx.shard(wires)
                logits = sharding_ctx.shard(logits)

            # Perform pool training step (single batch, possibly sharded across devices)
            (
                loss,
                (aux, circuit_pool, loss_steps),
            ) = pool_train_step(
                model=model,
                optimizer=optimizer,
                pool=circuit_pool,
                idxs=idxs,
                graphs=graphs,
                wires=wires,
                logits=logits,
                x=x_train,
                y_target=y_train,
                layer_sizes=layer_sizes,
                n_message_steps=n_message_steps,
                loss_cfg=loss_cfg,
                loss_key=loss_key,
                epoch=epoch,
                data_fraction=data_fraction,
                # Probabilistic damage during training
                p_fault=p_fault,
                faulty_value=faulty_logit_value,
                permanent_damage=permanent_damage,
            )

            hard_loss = aux["hard_loss"]
            accuracy = aux["accuracy"]
            hard_accuracy = aux["hard_accuracy"]

            # Reset a fraction of the pool using scheduled intervals

            if should_reset_pool(epoch, reset_pool_interval, last_reset_epoch):
                rng, reset_key, fresh_key = jax.random.split(rng, 3)

                if wiring_mode == "genetic":
                    # Use genetic mutations instead of completely fresh circuits
                    circuit_pool, avg_steps_reset = circuit_pool.reset_with_genetic_mutation(
                        key=reset_key,
                        fraction=reset_pool_fraction,
                        layer_sizes=layer_sizes,
                        input_n=input_n,
                        arity=arity,
                        circuit_hidden_dim=circuit_hidden_dim,
                        mutation_rate=genetic_mutation_rate,
                        n_swaps_per_layer=genetic_swaps_per_layer,
                        reset_strategy=reset_strategy,
                        combined_weights=combined_weights,
                    )
                else:
                    # Original logic for fixed and random wiring modes
                    # Generate fresh circuits for resetting

                    # Use consistent key generation for pool resets
                    # Note: "genetic" mode is handled above, so only "fixed" uses wiring_fixed_key here
                    reset_pool_key = wiring_fixed_key if wiring_mode == "fixed" else fresh_key

                    fresh_pool = initialize_graph_pool(
                        rng=reset_pool_key,
                        layer_sizes=layer_sizes,
                        pool_size=pool_size,  # Use same size as circuit_pool
                        input_n=input_n,
                        arity=arity,
                        circuit_hidden_dim=circuit_hidden_dim,
                        wiring_mode=wiring_mode,
                        initial_diversity=initial_diversity
                        if wiring_mode == "fixed"
                        else pool_size,
                        initialize_gate_masks=True,
                        noise_scale=pool_noise_scale,
                    )

                    # Reset a fraction of the pool and get avg steps of reset graphs
                    circuit_pool, avg_steps_reset = circuit_pool.reset_fraction(
                        key=reset_key,
                        fraction=reset_pool_fraction,
                        new_graphs=fresh_pool.graphs,
                        new_wires=fresh_pool.wires,
                        new_logits=fresh_pool.logits,
                        new_gate_masks=fresh_pool.gate_masks,
                        reset_strategy=reset_strategy,
                        combined_weights=combined_weights,
                    )

                # Update last reset epoch
                last_reset_epoch = epoch
                diversity = circuit_pool.get_wiring_diversity(layer_sizes)

            # Track average damage count (works for both modes - reads from gate_knockout_mask)
            avg_damage_count = circuit_pool.get_average_damage_count()

            # Record metrics
            losses.append(float(loss))
            hard_losses.append(float(hard_loss))
            accuracies.append(float(accuracy))
            hard_accuracies.append(float(hard_accuracy))
            reset_steps.append(float(avg_steps_reset))

            # Update adaptive scheduler if enabled (uses loss to adjust LR)
            if adaptive_scheduler is not None:
                # Update scheduler with current loss and get new LR
                new_lr = adaptive_scheduler.update(float(loss), epoch)

                # Update optimizer's learning rate hyperparameter
                # For optax.inject_hyperparams, the hyperparams are in opt_state
                try:
                    # Navigate to the injected hyperparams in the optimizer state
                    # Structure: chain -> [clip, zero_nans, inject_hyperparams(adamw)]
                    opt_state = optimizer.opt_state
                    if hasattr(opt_state, "inner_state") and len(opt_state.inner_state) >= 3:
                        adamw_state = opt_state.inner_state[2]
                        if hasattr(adamw_state, "hyperparams"):
                            adamw_state.hyperparams["learning_rate"] = jp.array(new_lr)
                except Exception as e:
                    log.debug(f"Could not update adaptive LR in optimizer state: {e}")

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

            # Determine current learning rate
            if adaptive_scheduler is not None:
                schedule_value = adaptive_scheduler.get_lr()
            elif schedule is not None:
                schedule_value = schedule(epoch)
            else:
                schedule_value = learning_rate

            # Log to wandb if enabled
            metrics_dict = {
                "training/epoch": epoch,
                "training/loss": float(loss),
                "training/hard_loss": float(hard_loss),
                "training/accuracy": float(accuracy),
                "training/hard_accuracy": float(hard_accuracy),
                "pool/wiring_diversity": float(diversity),
                "pool/reset_steps": float(avg_steps_reset),
                "pool/avg_update_steps": float(avg_steps),
                "pool/avg_knockouts": float(avg_damage_count),
                "pool/loss_steps": loss_steps,
            }

            # Add learning rate
            metrics_dict["scheduler/learning_rate"] = schedule_value

            # Add adaptive scheduler stats if enabled
            if adaptive_scheduler is not None:
                metrics_dict.update(adaptive_scheduler.get_stats())

            # Add early stopping metrics if enabled
            if early_stopping is not None:
                metrics_dict["early_stop/count"] = early_stopping.count
                metrics_dict["early_stop/threshold"] = early_stopping.threshold
                if early_stopping.first_epoch is not None:
                    metrics_dict["early_stop/first_epoch"] = early_stopping.first_epoch

            _log_to_wandb(wandb_run, metrics_dict, epoch, log_interval)

            # Update progress bar with current metrics
            postfix_dict = {
                "Loss": f"{loss:.4f}",
                "Accuracy": f"{accuracy:.4f}",
                "Hard Acc": f"{hard_accuracy:.4f}",
                "Diversity": f"{diversity:.3f}",
                "Reset Steps": f"{avg_steps_reset:.2f}",
                "Loss Steps": f"{loss_steps:.2f}",
                "LR": f"{schedule_value:.1e}",
            }

            # Add early stopping info if active
            if early_stopping is not None and early_stopping.count > 0:
                postfix_dict["ES"] = early_stopping.progress

            # Add damage info if active
            if damage_enabled and avg_damage_count > 0:
                postfix_dict["Dmg"] = f"{avg_damage_count:.1f}"

            pbar.set_postfix(postfix_dict)

            # Step 2: Run periodic evaluation if enabled (includes unified best model tracking)
            if (
                periodic_eval_enabled
                and eval_datasets is not None
                and epoch % periodic_eval_interval == 0
            ):
                # Run enhanced evaluations: fixed seed, pool sample (if diversity > 1), and OOD

                # Use the same datasets created during initialization
                # The pool evaluation circuits are recreated with the same logic as training
                current_datasets = eval_datasets

                # Build discrete damage steps (evenly spaced within eval window)
                if n_damage_steps > 0:
                    damage_steps = jp.linspace(
                        0,
                        periodic_eval_inner_steps,
                        n_damage_steps + 1,
                        endpoint=False,
                    ).astype(int)[1:]
                    damage_key = jax.random.PRNGKey(42)
                else:
                    damage_steps = None
                    damage_key = None

                eval_results = run_unified_periodic_evaluation(
                    model=model,
                    datasets=current_datasets,
                    pool=circuit_pool,
                    # Data
                    data_dict=data_dict,
                    input_n=input_n,
                    arity=arity,
                    circuit_hidden_dim=circuit_hidden_dim,
                    n_message_steps=periodic_eval_inner_steps,
                    loss_cfg=loss_cfg,
                    epoch=epoch,
                    wandb_run=wandb_run,
                    log_stepwise=periodic_eval_log_stepwise,
                    layer_sizes=layer_sizes,
                    log_pool_scatter=periodic_eval_log_pool_scatter,
                    # Best model tracking parameters
                    best_model_tracker=best_model_tracker,
                    checkpoint_path=checkpoint_path,
                    save_best=save_best,
                    optimizer=optimizer,
                    training_metrics=training_metrics,
                    track_metrics=track_metrics,
                    # Probabilistic damage (adjusted for eval step count)
                    p_fault=p_fault_eval,
                    faulty_value=faulty_logit_value,
                    permanent_damage=permanent_damage,
                    # Discrete damage (shotgun, for OOD visualization)
                    damage_steps=damage_steps,
                    knockout_per_damage_step=knockouts_per_event,
                    damage_key=damage_key,
                    # Delayed onset & no-repair baseline
                    p_fault_onset_step=p_fault_onset_step_eval,
                    compute_no_repair_baseline=compute_no_repair_baseline,
                )
                # Merge all final metrics into one dict for early stopping
                # (covers IN-dist, OOD, damaged, etc.)
                all_final = eval_results.get("final_metrics", {})
                current_eval_metrics = {}
                for sub_metrics in all_final.values():
                    if sub_metrics is not None:
                        current_eval_metrics.update(sub_metrics)
                if not current_eval_metrics:
                    current_eval_metrics = None

            # Step 3: Save periodic checkpoints (best models are now handled by unified system)
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

            # Step 4: Check for early stopping
            if early_stopping is not None and early_stopping.step(
                epoch, training_metrics, current_eval_metrics
            ):
                break

    except KeyboardInterrupt:
        log.info(f"Training interrupted by user at epoch {epoch}/{epochs}")
        # Ensure progress bar is properly closed
        pbar.close()

    # Build result dict once after training loop completes (or is interrupted)
    result = {
        "model": model,
        "optimizer": optimizer,
        "losses": losses,
        "hard_losses": hard_losses,
        "accuracies": accuracies,
        "hard_accuracies": hard_accuracies,
        "reset_steps": reset_steps,
        "early_stopped": early_stopping.triggered if early_stopping else False,
        "early_stop_epoch": epoch if (early_stopping and early_stopping.triggered) else None,
        "first_threshold_epoch": early_stopping.first_epoch if early_stopping else None,
        "best_model_tracker": best_model_tracker,
        "pool": circuit_pool,
    }

    # Log final results to wandb
    _log_final_wandb_metrics(wandb_run, result, epochs)

    return result
