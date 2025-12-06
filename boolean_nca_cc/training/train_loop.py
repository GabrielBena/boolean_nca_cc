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
from boolean_nca_cc.circuits.viz import create_wandb_visualization, plot_wandb_stepwise_results
from boolean_nca_cc.models import CircuitGNN, CircuitSelfAttention, PerceiverCircuitAttention
from boolean_nca_cc.training.checkpointing import (
    BestModelTracker,
    check_early_stopping,
    save_periodic_checkpoint,
    setup_checkpoint_dir,
)
from boolean_nca_cc.training.eval_datasets import (
    UnifiedEvaluationDatasets,
    create_unified_evaluation_datasets,
    evaluate_circuits_in_chunks,
)
from boolean_nca_cc.training.evaluation import (
    evaluate_model_stepwise_batched,
    get_loss_and_update_graph,
)
from boolean_nca_cc.training.pool.pool import GraphPool, initialize_graph_pool
from boolean_nca_cc.training.schedulers import (
    get_learning_rate_schedule,
    get_step_beta,
    should_reset_pool,
)
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
    """Log pool scatterplot to wandb."""
    if wandb_run is None:
        return

    all_loss, all_steps = pool.graphs.globals.loss, pool.graphs.globals.update_steps
    data = list(zip(all_steps, all_loss, strict=False))
    table = wandb.Table(data=data, columns=["steps", "loss"])
    wandb_run.log({"pool/scatter": wandb.plot.scatter(table, "steps", "loss")})


def _create_single_circuit_visualization(
    model,
    wires_batch,
    logits_batch,
    x_data,
    y_data,
    input_n,
    arity,
    circuit_hidden_dim,
    n_message_steps,
    loss_cfg,
    layer_sizes,
    circuit_idx=0,
    eval_type="eval_in",
    log_stepwise=False,
    damage_steps=None,
    knockout_per_damage_step=1,
    damage_key=jax.random.PRNGKey(42),
):
    """
    Create a wandb visualization for a single circuit from the evaluation batch.

    Args:
        model: The model to use for optimization
        wires_batch: Batch of wires (we'll take circuit_idx)
        logits_batch: Batch of logits (we'll take circuit_idx)
        x_data: Input data
        y_data: Target data
        input_n: Number of input nodes
        arity: Arity of gates
        circuit_hidden_dim: Hidden dimension
        n_message_steps: Number of message steps for evaluation
        loss_cfg: Loss config dict
        layer_sizes: Circuit layer sizes
        circuit_idx: Index of circuit to visualize (default 0)
        eval_type: Type of evaluation ("eval_in" or "eval_out")

    Returns:
        Dictionary with visualization results or None if failed
    """
    try:
        # Extract single circuit from batch
        single_wires = [layer_wires[circuit_idx] for layer_wires in wires_batch]
        single_logits = [layer_logits[circuit_idx] for layer_logits in logits_batch]

        # Run the model to optimize the single circuit
        from boolean_nca_cc.training.evaluation import evaluate_model_stepwise_generator

        # Create generator for single circuit
        generator = evaluate_model_stepwise_generator(
            model=model,
            wires=single_wires,
            logits=single_logits,
            x_data=x_data,
            y_data=y_data,
            input_n=input_n,
            arity=arity,
            circuit_hidden_dim=circuit_hidden_dim,
            max_steps=n_message_steps,
            loss_cfg=loss_cfg,
            bidirectional_edges=True,
            layer_sizes=layer_sizes,
        )

        # Run to completion
        results = []
        for result in generator:
            results.append(result)

        if damage_steps is not None:
            damage_generator = evaluate_model_stepwise_generator(
                model=model,
                wires=single_wires,
                logits=single_logits,
                x_data=x_data,
                y_data=y_data,
                input_n=input_n,
                arity=arity,
                circuit_hidden_dim=circuit_hidden_dim,
                max_steps=n_message_steps,
                loss_cfg=loss_cfg,
                bidirectional_edges=True,
                layer_sizes=layer_sizes,
                damage_steps=damage_steps,
                knockout_per_damage_step=knockout_per_damage_step,
                damage_key=damage_key,
            )
            damage_results = []
            for result in damage_generator:
                damage_results.append(result)

        final_result = results[-1]

        if final_result is None:
            return None

        # Create visualization with optimized logits
        viz_result = create_wandb_visualization(
            logits=final_result.logits,
            wires=single_wires,
            x=x_data,
            y0=y_data,
            title_prefix=f"{eval_type.upper()} Circuit {circuit_idx} - ",
            hard=True,
        )

        if log_stepwise:
            viz_result["stepwise_fig"] = plot_wandb_stepwise_results(results)
        else:
            viz_result["stepwise_fig"] = None

        if damage_steps is not None:
            viz_result_damaged = create_wandb_visualization(
                logits=damage_results[-1].logits,
                wires=single_wires,
                x=x_data,
                y0=y_data,
                title_prefix=f"{eval_type.upper()} Circuit {circuit_idx} - Damaged",
                hard=True,
            )
            if log_stepwise:
                viz_result_damaged["stepwise_fig"] = plot_wandb_stepwise_results(
                    damage_results, damage_steps=damage_steps
                )
            else:
                viz_result_damaged["stepwise_fig"] = None

        final_viz_result = {
            "figure": viz_result["figure"],
            "stepwise_fig": viz_result["stepwise_fig"],
            "accuracy": viz_result["accuracy"],
            "error_count": viz_result["error_count"],
            "total_bits": viz_result["total_bits"],
            "final_loss": float(final_result.loss),
            "final_hard_loss": float(final_result.hard_loss),
        }
        if damage_steps is not None:
            final_viz_result["figure_damaged"] = viz_result_damaged["figure"]
            final_viz_result["stepwise_fig_damaged"] = viz_result_damaged["stepwise_fig"]
            final_viz_result["accuracy_damaged"] = viz_result_damaged["accuracy"]
            final_viz_result["error_count_damaged"] = viz_result_damaged["error_count"]
            final_viz_result["total_bits_damaged"] = viz_result_damaged["total_bits"]
            final_viz_result["final_loss_damaged"] = float(damage_results[-1].loss)
            final_viz_result["final_hard_loss_damaged"] = float(damage_results[-1].hard_loss)

        return final_viz_result

    except Exception as e:
        log.warning(f"Error creating circuit visualization: {e}")
        return None


def run_unified_periodic_evaluation(
    model,
    datasets: UnifiedEvaluationDatasets,
    pool,
    x_data,
    y_data,
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
    x_plot: jp.ndarray | None = None,
    y_plot: jp.ndarray | None = None,
    # Damage parameters
    damage_steps=None,
    knockout_per_damage_step=1,
    damage_key=jax.random.PRNGKey(42),
) -> dict:
    """
    Run unified periodic evaluation with only IN-distribution and OUT-of-distribution testing.

    This function also handles best model tracking and saving for both distributions.

    Args:
        model: The model to evaluate
        datasets: UnifiedEvaluationDatasets object containing IN and OUT distribution circuits
        pool: GraphPool for logging scatter plot
        x_data: Input data
        y_data: Target data
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
        track_metrics: List of specific metrics to track and save (optional, e.g.,
                      ["eval_in_hard_accuracy", "eval_out_hard_accuracy"])

    Returns:
        Dictionary with evaluation metrics from IN-distribution and OUT-of-distribution evaluations
        and information about best model updates
    """
    try:
        # 1. Run IN-distribution evaluation (matches training pattern)
        # Use chunked evaluation to handle cases where diversity exceeds target batch size

        if datasets.in_distribution_wires is not None:
            log.info(
                f"Running IN-distribution evaluation ({datasets.in_actual_batch_size} circuits)..."
            )
            if (
                datasets.in_actual_batch_size is not None
                and datasets.in_actual_batch_size > datasets.target_batch_size
            ):
                log.info(f"Using chunked evaluation (chunks of {datasets.target_batch_size})")

            step_metrics_in = evaluate_circuits_in_chunks(
                eval_fn=evaluate_model_stepwise_batched,
                wires=datasets.in_distribution_wires,
                logits=datasets.in_distribution_logits,
                target_chunk_size=datasets.target_batch_size,
                model=model,
                x_data=x_data,
                y_data=y_data,
                input_n=input_n,
                arity=arity,
                circuit_hidden_dim=circuit_hidden_dim,
                n_message_steps=n_message_steps,
                loss_cfg=loss_cfg,
                layer_sizes=layer_sizes,
            )

            # Get final metrics (last step) for IN-distribution
            final_metrics_in = {
                "eval_in/final_loss": step_metrics_in["loss"][-1],
                "eval_in/final_hard_loss": step_metrics_in["hard_loss"][-1],
                "eval_in/final_accuracy": step_metrics_in["accuracy"][-1],
                "eval_in/final_hard_accuracy": step_metrics_in["hard_accuracy"][-1],
                "eval_in/epoch": epoch,
            }
        else:
            log.info("No IN-distribution evaluation data available.")
            step_metrics_in = None
            final_metrics_in = None

        if datasets.out_of_distribution_wires is not None:
            # 2. Run OUT-of-distribution evaluation (always random)
            log.info(
                f"Running OUT-of-distribution evaluation ({datasets.out_actual_batch_size} circuits)..."
            )
            if datasets.out_actual_batch_size > datasets.target_batch_size:
                log.info(f"Using chunked evaluation (chunks of {datasets.target_batch_size})")

            step_metrics_out = evaluate_circuits_in_chunks(
                eval_fn=evaluate_model_stepwise_batched,
                wires=datasets.out_of_distribution_wires,
                logits=datasets.out_of_distribution_logits,
                target_chunk_size=datasets.target_batch_size,
                model=model,
                x_data=x_data,
                y_data=y_data,
                input_n=input_n,
                arity=arity,
                circuit_hidden_dim=circuit_hidden_dim,
                n_message_steps=n_message_steps,
                loss_cfg=loss_cfg,
                layer_sizes=layer_sizes,
            )

            # Get final metrics (last step) for OUT-of-distribution
            final_metrics_out = {
                "eval_out/final_loss": step_metrics_out["loss"][-1],
                "eval_out/final_hard_loss": step_metrics_out["hard_loss"][-1],
                "eval_out/final_accuracy": step_metrics_out["accuracy"][-1],
                "eval_out/final_hard_accuracy": step_metrics_out["hard_accuracy"][-1],
                "eval_out/epoch": epoch,
            }
        else:
            log.info("No OUT-of-distribution evaluation data available.")
            step_metrics_out = None
            final_metrics_out = None

        if final_metrics_in is None and final_metrics_out is None:
            log.info("No evaluation data available.")
            return {}

        # Combine all metrics for logging
        combined_metrics = {
            **(final_metrics_in or {}),
            **(final_metrics_out or {}),
        }

        # Log to wandb if enabled
        if wandb_run:
            wandb_run.log(combined_metrics)

            if log_pool_scatter:
                _log_pool_scatter(pool, epoch, wandb_run)

            # Create and log circuit visualizations
            try:
                # Create visualization for IN-distribution circuit (if available)
                if (
                    datasets.in_distribution_wires is not None
                    and datasets.in_distribution_logits is not None
                ):
                    viz_in = _create_single_circuit_visualization(
                        model=model,
                        wires_batch=datasets.in_distribution_wires,
                        logits_batch=datasets.in_distribution_logits,
                        x_data=x_plot if x_plot is not None else x_data,
                        y_data=y_plot if y_plot is not None else y_data,
                        input_n=input_n,
                        arity=arity,
                        circuit_hidden_dim=circuit_hidden_dim,
                        n_message_steps=n_message_steps,
                        loss_cfg=loss_cfg,
                        layer_sizes=layer_sizes,
                        circuit_idx=0,
                        eval_type="eval_in",
                        log_stepwise=log_stepwise,
                        damage_steps=damage_steps,
                        knockout_per_damage_step=knockout_per_damage_step,
                        damage_key=damage_key,
                    )

                    import matplotlib.pyplot as plt

                    if viz_in is not None:
                        # Log the visualization figure
                        wandb_run.log(
                            {
                                "eval_in/circuit_visualization": wandb_run.Image(viz_in["figure"]),
                                "eval_in/viz_accuracy": viz_in["accuracy"],
                                # "eval_in/viz_error_count": viz_in["error_count"],
                                # "eval_in/viz_final_loss": viz_in["final_loss"],
                            }
                        )
                        if viz_in["stepwise_fig"] is not None:
                            wandb_run.log(
                                {
                                    "eval_in/stepwise_fig": wandb_run.Image(viz_in["stepwise_fig"]),
                                }
                            )

                        if viz_in.get("figure_damaged", None) is not None:
                            wandb_run.log(
                                {
                                    "eval_in/circuit_visualization_damaged": wandb_run.Image(
                                        viz_in["figure_damaged"]
                                    ),
                                    "eval_in/viz_accuracy_damaged": viz_in["accuracy_damaged"],
                                }
                            )

                        if viz_in.get("stepwise_fig_damaged", None) is not None:
                            wandb_run.log(
                                {
                                    "eval_in/stepwise_fig_damaged": wandb_run.Image(
                                        viz_in["stepwise_fig_damaged"]
                                    ),
                                }
                            )

                        plt.close("all")  # Close all figures to free memory

                # Create visualization for OUT-of-distribution circuit (if available)
                if (
                    datasets.out_of_distribution_wires is not None
                    and datasets.out_of_distribution_logits is not None
                ):
                    viz_out = _create_single_circuit_visualization(
                        model=model,
                        wires_batch=datasets.out_of_distribution_wires,
                        logits_batch=datasets.out_of_distribution_logits,
                        x_data=x_plot if x_plot is not None else x_data,
                        y_data=y_plot if y_plot is not None else y_data,
                        input_n=input_n,
                        arity=arity,
                        circuit_hidden_dim=circuit_hidden_dim,
                        n_message_steps=n_message_steps,
                        loss_cfg=loss_cfg,
                        layer_sizes=layer_sizes,
                        circuit_idx=0,
                        eval_type="eval_out",
                        log_stepwise=log_stepwise,
                        damage_steps=damage_steps,
                        knockout_per_damage_step=knockout_per_damage_step,
                        damage_key=damage_key,
                    )

                    import matplotlib.pyplot as plt

                    if viz_out is not None:
                        # Log the visualization figure
                        wandb_run.log(
                            {
                                "eval_out/circuit_visualization": wandb_run.Image(
                                    viz_out["figure"]
                                ),
                                "eval_out/viz_accuracy": viz_out["accuracy"],
                                # "eval_out/viz_error_count": viz_out["error_count"],
                                # "eval_out/viz_final_loss": viz_out["final_loss"],
                            }
                        )
                        if viz_out.get("stepwise_fig", None) is not None:
                            wandb_run.log(
                                {
                                    "eval_out/stepwise_fig": wandb_run.Image(
                                        viz_out["stepwise_fig"]
                                    ),
                                }
                            )

                        if viz_out.get("figure_damaged", None) is not None:
                            wandb_run.log(
                                {
                                    "eval_out/circuit_visualization_damaged": wandb_run.Image(
                                        viz_out["figure_damaged"]
                                    ),
                                    "eval_out/viz_accuracy_damaged": viz_out["accuracy_damaged"],
                                }
                            )
                            if viz_out.get("stepwise_fig_damaged", None) is not None:
                                wandb_run.log(
                                    {
                                        "eval_out/stepwise_fig_damaged": wandb_run.Image(
                                            viz_out["stepwise_fig_damaged"]
                                        ),
                                    }
                                )
                        plt.close("all")  # Close all figures to free memory

            except Exception as e:
                import traceback

                tb_str = traceback.format_exc()
                log.warning(f"Error creating wandb circuit visualizations: {e}")
                log.warning(f"Traceback: {tb_str}")

        # Log summary to console
        training_config = datasets.training_config

        # Add chunking info if used
        in_chunk_info = ""
        if (
            datasets.in_actual_batch_size is not None
            and datasets.in_actual_batch_size > datasets.target_batch_size
        ):
            num_in_chunks = (
                datasets.in_actual_batch_size + datasets.target_batch_size - 1
            ) // datasets.target_batch_size
            in_chunk_info = f", {num_in_chunks} chunks"

        out_chunk_info = ""
        if (
            datasets.out_actual_batch_size is not None
            and datasets.out_actual_batch_size > datasets.target_batch_size
        ):
            num_out_chunks = (
                datasets.out_actual_batch_size + datasets.target_batch_size - 1
            ) // datasets.target_batch_size
            out_chunk_info = f", {num_out_chunks} chunks"

        # Construct log message conditionally based on available data
        log_message_parts = [f"Unified Eval (epoch {epoch}):"]

        if final_metrics_in is not None:
            log_message_parts.append(
                f"  IN-distribution ({datasets.in_actual_batch_size} circuits{in_chunk_info}, "
                f"mode={training_config['wiring_mode']}, diversity={training_config['initial_diversity']}): "
                f"Loss={final_metrics_in['eval_in/final_loss']:.4f}, "
                f"Acc={final_metrics_in['eval_in/final_accuracy']:.4f}, "
                f"Hard Acc={final_metrics_in['eval_in/final_hard_accuracy']:.4f}"
            )
        else:
            log_message_parts.append(
                f"  IN-distribution: Not available (training mode: {training_config['wiring_mode']})"
            )

        if final_metrics_out is not None:
            log_message_parts.append(
                f"  OUT-of-distribution ({datasets.out_actual_batch_size} circuits{out_chunk_info}, random): "
                f"Loss={final_metrics_out['eval_out/final_loss']:.4f}, "
                f"Acc={final_metrics_out['eval_out/final_accuracy']:.4f}, "
                f"Hard Acc={final_metrics_out['eval_out/final_hard_accuracy']:.4f}"
            )

        log_message = "\n".join(log_message_parts)

        log.info(log_message)

        # Prepare evaluation metrics for best model tracking
        eval_metrics = {
            **(final_metrics_in or {}),
            **(final_metrics_out or {}),
        }

        # Track and save best models if tracker is provided
        best_model_updates = {}
        if best_model_tracker is not None and optimizer is not None:
            from boolean_nca_cc.training.checkpointing import track_and_save_best_models

            # Prepare metrics for saving with checkpoint
            checkpoint_metrics = {
                "eval_in_metrics": final_metrics_in or {},
                "eval_out_metrics": final_metrics_out or {},
                "training_metrics": training_metrics or {},
                "datasets_info": {
                    "in_actual_batch_size": datasets.in_actual_batch_size,
                    "out_actual_batch_size": datasets.out_actual_batch_size,
                    "target_batch_size": datasets.target_batch_size,
                    "training_wiring_mode": datasets.training_config["wiring_mode"],
                    "training_initial_diversity": datasets.training_config["initial_diversity"],
                    "evaluation_base_seed": datasets.training_config["evaluation_base_seed"],
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

            # Log best model updates to wandb if any occurred
            if best_model_updates and wandb_run:
                for metric_key, update_info in best_model_updates.items():
                    wandb_run.log(
                        {
                            f"best_model_updates/{metric_key}": update_info["value"],
                            f"best_model_updates/{metric_key}_epoch": update_info["epoch"],
                        }
                    )

        # Return all step metrics and final metrics for best model tracking
        result = {
            "step_metrics_in": step_metrics_in,
            "step_metrics_out": step_metrics_out,
            "final_metrics_in": final_metrics_in,
            "final_metrics_out": final_metrics_out,
            "best_model_updates": best_model_updates,
            # Add datasets information for comprehensive result reporting
            "datasets_info": {
                "in_actual_batch_size": datasets.in_actual_batch_size,
                "out_actual_batch_size": datasets.out_actual_batch_size,
                "target_batch_size": datasets.target_batch_size,
                "in_used_chunking": datasets.in_actual_batch_size is not None
                and datasets.in_actual_batch_size > datasets.target_batch_size,
                "out_used_chunking": datasets.out_actual_batch_size is not None
                and datasets.out_actual_batch_size > datasets.target_batch_size,
                "training_wiring_mode": datasets.training_config["wiring_mode"],
                "training_initial_diversity": datasets.training_config["initial_diversity"],
                "evaluation_base_seed": datasets.training_config["evaluation_base_seed"],
            },
        }

        return result

    except Exception as e:
        import traceback

        tb_str = traceback.format_exc()
        log.warning(f"Error during unified periodic evaluation at epoch {epoch}: {e}\n{tb_str}")
        return {}


def train_model(
    # Data parameters
    x_train: jp.ndarray,
    y_train: jp.ndarray,
    x_test: jp.ndarray,
    y_test: jp.ndarray,
    x_total: jp.ndarray,
    y_total: jp.ndarray,
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
    batch_chunk_size: int
    | None = None,  # Sequential batch processing chunk size (None means use meta_batch_size)
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
    # Learning rate scheduling
    lr_scheduler: str = "constant",  # Options: "constant", "exponential", "cosine", "linear_warmup"
    lr_scheduler_params: dict | None = None,
    # Initialization parameters
    key: int = 0,
    wiring_fixed_key: jax.random.PRNGKey = jax.random.PRNGKey(
        42
    ),  # Fixed key for generating wirings when wiring_mode='fixed'
    init_model: CircuitGNN | CircuitSelfAttention | None = None,
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
    periodic_eval_damage_enabled: bool = False,
    periodic_eval_n_damage_steps: int = 1,
    # Wandb parameters
    wandb_logging: bool = False,
    log_interval: int = 1,
    wandb_run_config: dict | None = None,
    # Early stopping parameters
    stop_accuracy_enabled: bool = False,
    stop_accuracy_threshold: float = 0.95,
    stop_accuracy_metric: str = "hard_accuracy",
    stop_accuracy_source: str = "training",
    stop_accuracy_patience: int = 10,
    stop_accuracy_min_epochs: int = 100,
    # Best model tracking parameters
    track_metrics: list[str] | None = None,
    # Damage parameters for resilience testing
    damage_enabled: bool = False,
    damage_interval: int | None = None,  # Epochs between damage applications
    damage_fraction: float = 0.1,  # Fraction of pool to damage each interval
    knockouts_per_event: int = 1,  # Gates to knock out per damage event
    random_knockouts_per_event: bool = False,
    random_knockouts_per_event_min: int = 2,
    random_knockouts_per_event_max: int = 4,
    max_damage_per_circuit: int | None = None,  # Max knockouts per circuit
    faulty_logit_value: float = -10.0,  # Value for knocked-out gate logits
    # Debugging parameters
    do_check_gradients: bool = False,
):
    """
    Train a GNN to optimize boolean circuit parameters.

    Args:
        layer_sizes: List of tuples (nodes, group_size) for each layer
        x_train: Input data for training [num_train, input_bits]
        y_train: Target output data [num_train, output_bits]
        x_test: Input data for testing [num_test, input_bits]
        y_test: Target output data [num_test, output_bits]
        data_fraction: Fraction of data to use for training
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
        batch_chunk_size: Sequential batch processing chunk size (None means use meta_batch_size)
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
        wandb_logging: Whether to log metrics to wandb
        log_interval: Interval for logging metrics
        wandb_run_config: Configuration to pass to wandb
        stop_accuracy_enabled: Whether to enable early stopping based on accuracy
        stop_accuracy_threshold: Accuracy threshold to trigger early stopping
        stop_accuracy_metric: Which accuracy metric to use ('accuracy' or 'hard_accuracy')
        stop_accuracy_source: Source of the metric ('training' or 'eval')
        stop_accuracy_patience: Number of epochs to wait after reaching threshold before stopping
        stop_accuracy_min_epochs: Minimum number of epochs before early stopping can occur
        track_metrics: List of specific metrics to track and save best models for (e.g.,
                      ["eval_in_hard_accuracy", "eval_out_hard_accuracy"]). If None,
                      tracks all available metrics during evaluation.
        damage_enabled: Whether to enable gate damage during training
        damage_interval: Epochs between damage applications
        damage_fraction: Fraction of pool to damage each interval
        knockouts_per_event: Number of gates to knock out per damage event
        max_damage_per_circuit: Maximum knockouts per circuit (None = no limit)
        faulty_logit_value: Value for knocked-out gate logits (large negative)
        do_check_gradients: Whether to check gradients for zero values
    Returns:
        Dictionary with trained GNN model and training metrics
    """
    # Default loss config
    if loss_cfg is None:
        loss_cfg = LOSS_L4
    elif isinstance(loss_cfg, dict):
        loss_cfg = LossConfig.from_dict(loss_cfg)

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
    )

    # =========================================================================
    # Core loss and gradient computation (shared by single batch and chunked)
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
    ):
        """
        Core loss and gradient computation logic.

        This is the shared implementation used by both:
        - pool_train_step (single batch processing)
        - pool_train_step_sequential (chunked processing with gradient accumulation)

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

        def loss_fn_scan(model, graph, logits, wires, loss_key):
            # Store original shapes for reconstruction
            logits_original_shapes = [logit.shape for logit in logits]

            # Determine which scan function to use based on model type
            if isinstance(model, CircuitGNN):
                from boolean_nca_cc.models.gnn.model import run_gnn_scan_with_loss

                scan_fn = run_gnn_scan_with_loss
            elif isinstance(model, CircuitSelfAttention):
                from boolean_nca_cc.models.attention.self_attention import (
                    run_self_attention_scan_with_loss,
                )

                scan_fn = run_self_attention_scan_with_loss
            elif isinstance(model, PerceiverCircuitAttention):
                from boolean_nca_cc.models.attention.perceiver_attention import (
                    run_perceiver_scan_with_loss,
                )

                scan_fn = run_perceiver_scan_with_loss
            else:
                raise ValueError(f"Unknown model type: {type(model)}")

            loss_key, scan_key = jax.random.split(loss_key)

            # Run scan for all steps, computing loss and updating graph at each step
            final_graph, step_outputs = scan_fn(
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
            )

            loss_step = get_loss_step(loss_key)

            final_graph, final_loss, final_logits, final_aux = jax.tree.map(
                lambda x: x[loss_step], step_outputs
            )

            # # Take the mean of the losses until the loss step: more grads !
            # # Use masking instead of dynamic_slice since loss_step is a traced value
            # all_losses = step_outputs[1]  # shape: [n_message_steps]
            # indices = jp.arange(n_message_steps)
            # mask = indices <= loss_step
            # # Compute masked mean: sum of valid losses / count of valid losses
            # final_loss = jp.sum(jp.where(mask, all_losses, 0.0)) / (loss_step + 1)

            return final_loss, (final_aux, final_graph, final_logits, loss_step)

        def loss_fn_no_scan(model, graph, logits, wires, loss_key):
            # Store original shapes for reconstruction
            logits_original_shapes = [logit.shape for logit in logits]
            loss_step = get_loss_step(loss_key)

            all_results = []

            for _i in range(n_message_steps):
                graph = model(graph)

                graph, loss, logits, aux = get_loss_and_update_graph(
                    graph=graph,
                    logits_original_shapes=logits_original_shapes,
                    wires=wires,
                    x_data=x,
                    y_data=y_target,
                    loss_cfg=loss_cfg,
                    layer_sizes=layer_sizes,
                )
                all_results.append((loss, aux, graph, logits))

            # Stack all results using jax.tree_map
            stacked_results = jax.tree.map(lambda *args: jp.stack(args), *all_results)

            # Index at n_loss_step
            final_loss, final_aux, final_graph, final_logits = jax.tree.map(
                lambda x: x[loss_step], stacked_results
            )

            return final_loss, (final_aux, final_graph, final_logits, loss_step)

        def batch_loss_fn(model, graphs, logits, wires, loss_key):
            loss_fn = loss_fn_scan if use_scan else loss_fn_no_scan

            loss_keys = jax.random.split(loss_key, graphs.n_node.shape[0])
            loss, (aux, updated_graphs, updated_logits, loss_steps) = nnx.vmap(
                loss_fn, in_axes=(None, 0, 0, 0, 0)
            )(model, graphs, logits, wires, loss_keys)
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
        )

        return loss, aux, updated_graphs, updated_logits, loss_steps, grads

    # JIT-compiled version of core computation (used by both single batch and chunked)
    _compute_loss_and_gradients_jit = partial(
        nnx.jit,
        static_argnames=(
            "layer_sizes",
            "n_message_steps",
            "loss_cfg",
            "data_fraction",
        ),
    )(_compute_loss_and_gradients)

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
    ):
        """
        Single training step using graphs from the pool.

        Processes the full batch at once. For memory-constrained scenarios,
        use pool_train_step_sequential instead.

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
        )

        if do_check_gradients:
            check_gradients(grads)
        # Update GNN parameters
        optimizer.update(model, grads)

        # Update pool with the updated graphs and logits (wires stay the same)
        updated_pool = pool.update(idxs, updated_graphs, batch_of_logits=updated_logits)

        return loss, (aux, updated_pool, loss_steps)

    _pool_train_step_jit = partial(
        nnx.jit,
        static_argnames=(
            "layer_sizes",
            "n_message_steps",
            "loss_cfg",
            "data_fraction",
        ),
    )(_pool_train_step)

    # We can't perfrom gradient checking on the JIT-compiled version
    pool_train_step = _pool_train_step if do_check_gradients else _pool_train_step_jit

    # =========================================================================
    # Sequential batch training step (processes batch in smaller chunks)
    # =========================================================================
    def _pool_train_step_sequential(
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
        chunk_size: int,
        data_fraction: float = 1.0,
    ):
        """
        Sequential batch processing with gradient accumulation.

        Processes the batch in smaller chunks to save memory while maintaining
        the same gradient computation as processing the full batch at once.

        Uses the JIT-compiled _compute_loss_and_gradients_jit for each chunk,
        accumulating gradients before applying them in a single optimizer update.
        """
        batch_size = graphs.n_node.shape[0]
        num_chunks = (batch_size + chunk_size - 1) // chunk_size  # Ceiling division

        # Initialize accumulated gradients and metrics
        accumulated_grads = None
        accumulated_loss = 0.0
        accumulated_aux = None
        accumulated_updated_graphs = []
        accumulated_updated_logits = []
        accumulated_loss_steps = 0.0

        # Split loss keys for each chunk
        chunk_loss_keys = jax.random.split(loss_key, num_chunks)

        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min(start_idx + chunk_size, batch_size)
            actual_chunk_size = end_idx - start_idx

            # Extract chunk data
            chunk_graphs = jax.tree.map(lambda x: x[start_idx:end_idx], graphs)
            chunk_wires = jax.tree.map(lambda x: x[start_idx:end_idx], wires)
            chunk_logits = jax.tree.map(lambda x: x[start_idx:end_idx], logits)

            # Process chunk using JIT-compiled core function
            (
                chunk_loss,
                chunk_aux,
                chunk_updated_graphs,
                chunk_updated_logits,
                chunk_loss_steps,
                chunk_grads,
            ) = _compute_loss_and_gradients_jit(
                model=model,
                graphs=chunk_graphs,
                wires=chunk_wires,
                logits=chunk_logits,
                x=x,
                y_target=y_target,
                layer_sizes=layer_sizes,
                n_message_steps=n_message_steps,
                loss_cfg=loss_cfg,
                loss_key=chunk_loss_keys[chunk_idx],
                epoch=epoch,
                data_fraction=data_fraction,
            )

            # Accumulate gradients (weighted by chunk size for proper averaging)
            chunk_weight = actual_chunk_size / batch_size
            if accumulated_grads is None:
                accumulated_grads = jax.tree.map(lambda g: g * chunk_weight, chunk_grads)
            else:
                accumulated_grads = jax.tree.map(
                    lambda acc_g, chunk_g: acc_g + chunk_g * chunk_weight,
                    accumulated_grads,
                    chunk_grads,
                )

            # Accumulate loss and metrics (weighted by chunk size)
            accumulated_loss += chunk_loss * chunk_weight
            if accumulated_aux is None:
                accumulated_aux = jax.tree.map(lambda x: x * chunk_weight, chunk_aux)
            else:
                accumulated_aux = jax.tree.map(
                    lambda acc_x, chunk_x: acc_x + chunk_x * chunk_weight,
                    accumulated_aux,
                    chunk_aux,
                )
            accumulated_loss_steps += chunk_loss_steps * chunk_weight

            # Store updated graphs and logits
            accumulated_updated_graphs.append(chunk_updated_graphs)
            accumulated_updated_logits.append(chunk_updated_logits)

        # Combine updated graphs and logits from all chunks
        combined_updated_graphs = jax.tree.map(
            lambda *chunks: jp.concatenate(chunks, axis=0), *accumulated_updated_graphs
        )
        combined_updated_logits = jax.tree.map(
            lambda *chunks: jp.concatenate(chunks, axis=0), *accumulated_updated_logits
        )

        # Update GNN parameters with accumulated gradients
        optimizer.update(model, accumulated_grads)

        # Update pool with the updated graphs and logits
        updated_pool = pool.update(
            idxs, combined_updated_graphs, batch_of_logits=combined_updated_logits
        )

        return accumulated_loss, (accumulated_aux, updated_pool, accumulated_loss_steps)

    # Setup wandb logging if enabled
    wandb_run = _init_wandb(wandb_logging, wandb_run_config)
    wandb_id = wandb_run.run.id if wandb_run else None

    # Setup checkpointing directory
    checkpoint_path = setup_checkpoint_dir(checkpoint_dir, wandb_id)

    # Initialize best model tracker for unified tracking
    best_model_tracker = BestModelTracker()

    # Early stopping variables
    early_stop_triggered = False
    epochs_above_threshold = 0
    first_threshold_epoch = None

    # Create progress bar for training
    pbar = tqdm(range(epochs), desc="Training GNN")
    avg_steps_reset = 0

    # Track last reset epoch for scheduling
    last_reset_epoch = -1  # Initialize to -1 so first check works correctly

    # Track damage application
    last_damage_epoch = -1  # Initialize to -1 so first check works correctly
    num_circuits_damaged = 0  # Track circuits damaged in most recent damage event
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
        )

        log.info(eval_datasets.get_summary())

    # Determine effective batch chunk size
    effective_batch_chunk_size = (
        batch_chunk_size if batch_chunk_size is not None else meta_batch_size
    )
    use_sequential_batching = batch_chunk_size is not None and batch_chunk_size < meta_batch_size

    if use_sequential_batching:
        log.info(
            f"Using sequential batch processing: meta_batch_size={meta_batch_size}, chunk_size={effective_batch_chunk_size}"
        )

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

            # Perform pool training step (sequential or standard)
            if use_sequential_batching:
                (
                    loss,
                    (aux, circuit_pool, loss_steps),
                ) = _pool_train_step_sequential(
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
                    chunk_size=effective_batch_chunk_size,
                    data_fraction=data_fraction,
                )
            else:
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

            # Apply damage to ongoing optimizations if enabled
            if (
                damage_enabled
                and damage_interval is not None
                and should_reset_pool(epoch, damage_interval, last_damage_epoch)
            ):
                rng, damage_key = jax.random.split(rng)

                circuit_pool, num_circuits_damaged = circuit_pool.apply_damage(
                    key=damage_key,
                    fraction=damage_fraction,
                    layer_sizes=layer_sizes,
                    num_knockouts=knockouts_per_event,
                    random_knockouts_per_event=random_knockouts_per_event,
                    random_knockouts_per_event_min=random_knockouts_per_event_min,
                    random_knockouts_per_event_max=random_knockouts_per_event_max,
                    input_n=input_n,
                    arity=arity,
                    circuit_hidden_dim=circuit_hidden_dim,
                    faulty_value=faulty_logit_value,
                    max_damage_per_circuit=max_damage_per_circuit,
                    selection_strategy="uniform",  # Use uniform for damage
                )

                last_damage_epoch = epoch
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
                "damage/circuits_damaged": int(num_circuits_damaged),
                "damage/avg_knockouts": float(avg_damage_count),
                "damage/enabled": damage_enabled,
                "pool/loss_steps": loss_steps,
            }

            # Add sequential batching metrics if enabled
            if use_sequential_batching:
                num_chunks = (
                    meta_batch_size + effective_batch_chunk_size - 1
                ) // effective_batch_chunk_size
                metrics_dict.update(
                    {
                        "training/sequential_batching": True,
                        "training/meta_batch_size": meta_batch_size,
                        "training/chunk_size": effective_batch_chunk_size,
                        "training/num_chunks": num_chunks,
                    }
                )
            else:
                metrics_dict["training/sequential_batching"] = False

            # Add learning rate
            metrics_dict["scheduler/learning_rate"] = schedule_value

            # Add adaptive scheduler stats if enabled
            if adaptive_scheduler is not None:
                metrics_dict.update(adaptive_scheduler.get_stats())

            # Add early stopping metrics if enabled
            if stop_accuracy_enabled:
                metrics_dict["early_stop/enabled"] = True
                metrics_dict["early_stop/epochs_above_threshold"] = epochs_above_threshold
                metrics_dict["early_stop/threshold"] = stop_accuracy_threshold
                if first_threshold_epoch is not None:
                    metrics_dict["early_stop/first_threshold_epoch"] = first_threshold_epoch

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

            # Add chunk info if using sequential batching
            if use_sequential_batching:
                num_chunks = (
                    meta_batch_size + effective_batch_chunk_size - 1
                ) // effective_batch_chunk_size
                postfix_dict["Chunks"] = f"{num_chunks}x{effective_batch_chunk_size}"

            # Add early stopping info if active
            if stop_accuracy_enabled and epochs_above_threshold > 0:
                postfix_dict["ES"] = f"{epochs_above_threshold}/{stop_accuracy_patience}"

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
                rng, eval_key = jax.random.split(rng)

                # Use the same datasets created during initialization
                # The pool evaluation circuits are recreated with the same logic as training
                current_datasets = eval_datasets

                if periodic_eval_damage_enabled:
                    damage_steps = jp.linspace(
                        0,
                        periodic_eval_inner_steps,
                        periodic_eval_n_damage_steps + 1,
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
                    x_data=x_test,
                    y_data=y_test,
                    input_n=input_n,
                    arity=arity,
                    circuit_hidden_dim=circuit_hidden_dim,
                    n_message_steps=periodic_eval_inner_steps,  # Use fixed message steps
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
                    x_plot=x_total[:256],
                    y_plot=y_total[:256],
                    # Damage parameters
                    damage_steps=damage_steps,
                    knockout_per_damage_step=knockouts_per_event,
                    damage_key=damage_key,
                )
                # Extract final metrics for best model tracking (use IN-distribution metrics)
                current_eval_metrics = eval_results.get("final_metrics_in", None)

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

            # Step 4: Check for early stopping based on accuracy
            (
                should_break,
                early_stop_triggered,
                epochs_above_threshold,
                first_threshold_epoch,
                current_eval_metrics,
                rng,
            ) = check_early_stopping(
                stop_accuracy_enabled=stop_accuracy_enabled,
                epoch=epoch,
                stop_accuracy_min_epochs=stop_accuracy_min_epochs,
                early_stop_triggered=early_stop_triggered,
                stop_accuracy_metric=stop_accuracy_metric,
                stop_accuracy_source=stop_accuracy_source,
                training_metrics=training_metrics,
                current_eval_metrics=current_eval_metrics,
                stop_accuracy_threshold=stop_accuracy_threshold,
                first_threshold_epoch=first_threshold_epoch,
                epochs_above_threshold=epochs_above_threshold,
                stop_accuracy_patience=stop_accuracy_patience,
                rng=rng,
            )

            if should_break:
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
        "early_stopped": early_stop_triggered,
        "early_stop_epoch": epoch if early_stop_triggered else None,
        "first_threshold_epoch": first_threshold_epoch,
        "best_model_tracker": best_model_tracker,
        "pool": circuit_pool,
    }

    # Log final results to wandb
    _log_final_wandb_metrics(wandb_run, result, epochs)

    return result
