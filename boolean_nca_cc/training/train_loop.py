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
from boolean_nca_cc.circuits.viz import create_wandb_visualization
from boolean_nca_cc.models import CircuitGNN, CircuitSelfAttention
from boolean_nca_cc.training.checkpointing import (
    BestModelTracker,
    check_early_stopping,
    save_periodic_checkpoint,
    save_stable_state,
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


# Removed _setup_checkpoint_dir - now in checkpointing.py


# Removed _save_periodic_checkpoint - now in checkpointing.py


# Removed _save_best_checkpoint - now in checkpointing.py


# Removed _save_stable_state - now in checkpointing.py


# Removed _check_early_stopping - now in checkpointing.py


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


# Removed _get_metric_value - now in checkpointing.py


def _log_pool_scatter(pool, epoch, wandb_run):
    """Log pool scatterplot to wandb."""
    if wandb_run is None:
        return

    all_loss, all_steps = pool.graphs.globals[..., 0], pool.graphs.globals[..., 1]
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
    loss_type,
    layer_sizes,
    circuit_idx=0,
    eval_type="eval_in",
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
        loss_type: Loss function type
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
            loss_type=loss_type,
            bidirectional_edges=True,
            layer_sizes=layer_sizes,
        )

        # Run to completion
        final_result = None
        for result in generator:
            final_result = result

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

        return {
            "figure": viz_result["figure"],
            "accuracy": viz_result["accuracy"],
            "error_count": viz_result["error_count"],
            "total_bits": viz_result["total_bits"],
            "final_loss": float(final_result.loss),
            "final_hard_loss": float(final_result.hard_loss),
        }

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
    loss_type,
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
        loss_type: Loss function type
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
                loss_type=loss_type,
                layer_sizes=layer_sizes,
            )

            # Get final metrics (last step) for IN-distribution
            final_metrics_in = {
                "eval_in/final_loss": step_metrics_in["soft_loss"][-1],
                "eval_in/final_hard_loss": step_metrics_in["hard_loss"][-1],
                "eval_in/final_accuracy": step_metrics_in["soft_accuracy"][-1],
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
                loss_type=loss_type,
                layer_sizes=layer_sizes,
            )

            # Get final metrics (last step) for OUT-of-distribution
            final_metrics_out = {
                "eval_out/final_loss": step_metrics_out["soft_loss"][-1],
                "eval_out/final_hard_loss": step_metrics_out["hard_loss"][-1],
                "eval_out/final_accuracy": step_metrics_out["soft_accuracy"][-1],
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
                        x_data=x_data,
                        y_data=y_data,
                        input_n=input_n,
                        arity=arity,
                        circuit_hidden_dim=circuit_hidden_dim,
                        n_message_steps=n_message_steps,
                        loss_type=loss_type,
                        layer_sizes=layer_sizes,
                        circuit_idx=0,
                        eval_type="eval_in",
                    )

                    if viz_in is not None:
                        # Log the visualization figure
                        wandb_run.log(
                            {
                                "eval_in/circuit_visualization": wandb_run.Image(viz_in["figure"]),
                                "eval_in/viz_accuracy": viz_in["accuracy"],
                                "eval_in/viz_error_count": viz_in["error_count"],
                                "eval_in/viz_final_loss": viz_in["final_loss"],
                            }
                        )
                        # Close the figure to free memory
                        import matplotlib.pyplot as plt

                        plt.close(viz_in["figure"])

                # Create visualization for OUT-of-distribution circuit (if available)
                if (
                    datasets.out_of_distribution_wires is not None
                    and datasets.out_of_distribution_logits is not None
                ):
                    viz_out = _create_single_circuit_visualization(
                        model=model,
                        wires_batch=datasets.out_of_distribution_wires,
                        logits_batch=datasets.out_of_distribution_logits,
                        x_data=x_data,
                        y_data=y_data,
                        input_n=input_n,
                        arity=arity,
                        circuit_hidden_dim=circuit_hidden_dim,
                        n_message_steps=n_message_steps,
                        loss_type=loss_type,
                        layer_sizes=layer_sizes,
                        circuit_idx=0,
                        eval_type="eval_out",
                    )

                    if viz_out is not None:
                        # Log the visualization figure
                        wandb_run.log(
                            {
                                "eval_out/circuit_visualization": wandb_run.Image(
                                    viz_out["figure"]
                                ),
                                "eval_out/viz_accuracy": viz_out["accuracy"],
                                "eval_out/viz_error_count": viz_out["error_count"],
                                "eval_out/viz_final_loss": viz_out["final_loss"],
                            }
                        )
                        # Close the figure to free memory
                        import matplotlib.pyplot as plt

                        plt.close(viz_out["figure"])

            except Exception as e:
                log.warning(f"Error creating wandb circuit visualizations: {e}")

            # Optionally log step-by-step metrics for both evaluations
            if log_stepwise:
                # IN-distribution step-wise metrics
                if step_metrics_in is not None:
                    for step_idx in range(len(step_metrics_in["step"])):
                        step_data_in = {
                            "eval_in_steps/step": step_metrics_in["step"][step_idx],
                            "eval_in_steps/loss": step_metrics_in["soft_loss"][step_idx],
                            "eval_in_steps/hard_loss": step_metrics_in["hard_loss"][step_idx],
                            "eval_in_steps/accuracy": step_metrics_in["soft_accuracy"][step_idx],
                            "eval_in_steps/hard_accuracy": step_metrics_in["hard_accuracy"][
                                step_idx
                            ],
                            "eval_in_steps/epoch": epoch,
                        }
                        wandb_run.log(step_data_in)

                # OUT-of-distribution step-wise metrics
                if step_metrics_out is not None:
                    for step_idx in range(len(step_metrics_out["step"])):
                        step_data_out = {
                            "eval_out_steps/step": step_metrics_out["step"][step_idx],
                            "eval_out_steps/loss": step_metrics_out["soft_loss"][step_idx],
                            "eval_out_steps/hard_loss": step_metrics_out["hard_loss"][step_idx],
                            "eval_out_steps/accuracy": step_metrics_out["soft_accuracy"][step_idx],
                            "eval_out_steps/hard_accuracy": step_metrics_out["hard_accuracy"][
                                step_idx
                            ],
                            "eval_out_steps/epoch": epoch,
                        }
                        wandb_run.log(step_data_out)

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
        if datasets.out_actual_batch_size > datasets.target_batch_size:
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
                "out_used_chunking": datasets.out_actual_batch_size > datasets.target_batch_size,
                "training_wiring_mode": datasets.training_config["wiring_mode"],
                "training_initial_diversity": datasets.training_config["initial_diversity"],
                "evaluation_base_seed": datasets.training_config["evaluation_base_seed"],
            },
        }

        return result

    except Exception as e:
        log.warning(f"Error during unified periodic evaluation at epoch {epoch}: {e}")
        return {}


def train_model(
    # Data parameters
    x_data: jp.ndarray,
    y_data: jp.ndarray,
    layer_sizes: list[tuple[int, int]],
    # Model architecture parameters
    arity: int = 2,
    circuit_hidden_dim: int = 16,
    # Training hyperparameters
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    epochs: int = 100,
    n_message_steps: int = 1,
    use_scan: bool = False,
    # Loss parameters
    loss_type: str = "l4",  # Options: 'l4' or 'bce'
    random_loss_step: bool = False,  # Use random message passing step for loss computation
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
    save_stable_states: bool = True,
    # Periodic evaluation parameters
    periodic_eval_enabled: bool = False,
    periodic_eval_inner_steps: int = 100,
    periodic_eval_interval: int = 1024,
    periodic_eval_test_seed: int = 42,
    periodic_eval_log_stepwise: bool = False,
    periodic_eval_batch_size: int = 16,  # Batch size for random wiring evaluation
    periodic_eval_log_pool_scatter: bool = False,
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
):
    """
    Train a GNN to optimize boolean circuit parameters.

    Args:
        layer_sizes: List of tuples (nodes, group_size) for each layer
        x_data: Input data for training [batch, input_bits]
        y_data: Target output data [batch, output_bits]
        arity: Number of inputs per gate
        circuit_hidden_dim: Dimension of hidden features
        message_passing: Whether to use message passing or only self-updates
        node_mlp_features: Hidden layer sizes for the node MLP
        edge_mlp_features: Hidden layer sizes for the edge MLP
        use_attention: Whether to use attention-based message aggregation
        learning_rate: Learning rate for optimization
        epochs: Number of training epochs
        n_message_steps: Number of message passing steps per pool batch
        loss_type: Type of loss to use ('l4' for L4 norm or 'bce' for binary cross-entropy)
        random_loss_step: Use random message passing step for loss computation
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
        save_stable_states: Whether to save stable states (before potential NaN losses)
        periodic_eval_enabled: Whether to enable periodic evaluation
        periodic_eval_inner_steps: Number of inner steps for periodic evaluation
        periodic_eval_interval: Interval for periodic evaluation
        periodic_eval_test_seed: Seed for periodic evaluation test circuit generation
        periodic_eval_log_stepwise: Whether to log step-by-step evaluation metrics
        periodic_eval_batch_size: Batch size for random wiring evaluation
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
    Returns:
        Dictionary with trained GNN model and training metrics
    """
    # Initialize random key
    rng = jax.random.PRNGKey(key)

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
    if init_optimizer is None:
        # Create the learning rate schedule using our scheduler module
        schedule = get_learning_rate_schedule(
            lr_scheduler, learning_rate, epochs, lr_scheduler_params
        )

        # Create a new optimizer with the schedule
        opt_fn = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.zero_nans(),
            optax.adamw(learning_rate=schedule, weight_decay=weight_decay),
        )
        optimizer = nnx.Optimizer(model, opt_fn)
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
        initial_diversity=initial_diversity,
    )

    # Define pool-based training step
    @partial(
        nnx.jit,
        static_argnames=(
            "layer_sizes",
            "n_message_steps",
            "loss_type",
        ),
    )
    def pool_train_step(
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
        loss_type: str,
        loss_key: jax.random.PRNGKey,
        epoch: int,
    ):
        """
        Single training step using graphs from the pool.

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
            loss_type: Type of loss function to use
            loss_key: Random key for loss computation
        Returns:
            Tuple of (loss, auxiliary outputs, updated pool)
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
                    return jax.random.randint(loss_key, (1,), 0, n_message_steps)[0]
            else:
                return n_message_steps - 1

        # Define loss function
        def loss_fn_scan(model, graph, logits, wires, loss_key):
            # Store original shapes for reconstruction
            logits_original_shapes = [logit.shape for logit in logits]

            # Determine which scan function to use based on model type
            if isinstance(model, CircuitGNN):
                from boolean_nca_cc.models.gnn import run_gnn_scan_with_loss

                scan_fn = run_gnn_scan_with_loss
            elif isinstance(model, CircuitSelfAttention):
                from boolean_nca_cc.models.self_attention import (
                    run_self_attention_scan_with_loss,
                )

                scan_fn = run_self_attention_scan_with_loss
            else:
                raise ValueError(f"Unknown model type: {type(model)}")

            # Run scan for all steps, computing loss and updating graph at each step
            final_graph, step_outputs = scan_fn(
                model=model,
                graph=graph,
                num_steps=n_message_steps,
                logits_original_shapes=logits_original_shapes,
                wires=wires,
                x_data=x,
                y_data=y_target,
                loss_type=loss_type,
                layer_sizes=layer_sizes,
            )

            loss_step = get_loss_step(loss_key)

            final_graph, final_loss, final_logits, final_aux = jax.tree.map(
                lambda x: x[loss_step], step_outputs
            )

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
                    loss_type=loss_type,
                    layer_sizes=layer_sizes,
                )
                # Update graph globals with current update steps
                current_update_steps = graph.globals[..., 1] if graph.globals is not None else 0
                graph = graph._replace(
                    globals=jp.array([loss, current_update_steps + 1], dtype=jp.float32)
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

        # Update GNN parameters
        optimizer.update(grads)

        # Update pool with the updated graphs and logits (wires stay the same)
        updated_pool = pool.update(idxs, updated_graphs, batch_of_logits=updated_logits)

        return loss, (aux, updated_pool, loss_steps)

    # Define JIT-compiled chunk processing function
    @partial(
        nnx.jit,
        static_argnames=(
            "layer_sizes",
            "n_message_steps",
            "loss_type",
        ),
    )
    def pool_train_step_chunk_only(
        model: CircuitGNN,
        graphs: jraph.GraphsTuple,
        wires: PyTree,
        logits: PyTree,
        x: jp.ndarray,
        y_target: jp.ndarray,
        layer_sizes: tuple[tuple[int, int], ...],
        n_message_steps: int,
        loss_type: str,
        loss_key: jax.random.PRNGKey,
        epoch: int,
    ):
        """
        JIT-compiled loss and gradient computation for a single chunk.
        Reuses the exact same logic as pool_train_step but without optimizer update.
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
                    return jax.random.randint(loss_key, (1,), 0, n_message_steps)[0]
            else:
                return n_message_steps - 1

        # Define loss function (same as regular batching)
        def loss_fn_scan(model, graph, logits, wires, loss_key):
            # Store original shapes for reconstruction
            logits_original_shapes = [logit.shape for logit in logits]

            # Determine which scan function to use based on model type
            if isinstance(model, CircuitGNN):
                from boolean_nca_cc.models.gnn import run_gnn_scan_with_loss

                scan_fn = run_gnn_scan_with_loss
            elif isinstance(model, CircuitSelfAttention):
                from boolean_nca_cc.models.self_attention import (
                    run_self_attention_scan_with_loss,
                )

                scan_fn = run_self_attention_scan_with_loss
            else:
                raise ValueError(f"Unknown model type: {type(model)}")

            # Run scan for all steps, computing loss and updating graph at each step
            final_graph, step_outputs = scan_fn(
                model=model,
                graph=graph,
                num_steps=n_message_steps,
                logits_original_shapes=logits_original_shapes,
                wires=wires,
                x_data=x,
                y_data=y_target,
                loss_type=loss_type,
                layer_sizes=layer_sizes,
            )

            loss_step = get_loss_step(loss_key)

            final_graph, final_loss, final_logits, final_aux = jax.tree.map(
                lambda x: x[loss_step], step_outputs
            )

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
                    loss_type=loss_type,
                    layer_sizes=layer_sizes,
                )
                # Update graph globals with current update steps
                current_update_steps = graph.globals[..., 1] if graph.globals is not None else 0
                graph = graph._replace(
                    globals=jp.array([loss, current_update_steps + 1], dtype=jp.float32)
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

        # Compute loss and gradients (no optimizer update)
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

    # Define efficient sequential batch processing function
    def pool_train_step_sequential(
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
        loss_type: str,
        loss_key: jax.random.PRNGKey,
        epoch: int,
        chunk_size: int,
    ):
        """
        Sequential batch processing with gradient accumulation.

        Processes the batch in smaller chunks to save memory while maintaining
        the same gradient computation as processing the full batch at once.

        This implementation reuses the main pool_train_step function for each chunk,
        but accumulates gradients instead of applying them immediately.
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
            chunk_graphs = jax.tree.map(lambda x: x[start_idx:end_idx], graphs)  # noqa: B023
            chunk_wires = jax.tree.map(lambda x: x[start_idx:end_idx], wires)  # noqa: B023
            chunk_logits = jax.tree.map(lambda x: x[start_idx:end_idx], logits)  # noqa: B023

            # Process chunk using JIT-compiled function
            (
                chunk_loss,
                chunk_aux,
                chunk_updated_graphs,
                chunk_updated_logits,
                chunk_loss_steps,
                chunk_grads,
            ) = pool_train_step_chunk_only(
                model=model,
                graphs=chunk_graphs,
                wires=chunk_wires,
                logits=chunk_logits,
                x=x,
                y_target=y_target,
                layer_sizes=layer_sizes,
                n_message_steps=n_message_steps,
                loss_type=loss_type,
                loss_key=chunk_loss_keys[chunk_idx],
                epoch=epoch,
            )

            # Accumulate gradients (weighted by chunk size for proper averaging)
            chunk_weight = actual_chunk_size / batch_size
            if accumulated_grads is None:
                accumulated_grads = jax.tree.map(lambda g: g * chunk_weight, chunk_grads)  # noqa: B023
            else:
                accumulated_grads = jax.tree.map(
                    lambda acc_g, chunk_g: acc_g + chunk_g * chunk_weight,  # noqa: B023
                    accumulated_grads,
                    chunk_grads,
                )

            # Accumulate loss and metrics (weighted by chunk size)
            accumulated_loss += chunk_loss * chunk_weight
            if accumulated_aux is None:
                accumulated_aux = jax.tree.map(lambda x: x * chunk_weight, chunk_aux)  # noqa: B023
            else:
                accumulated_aux = jax.tree.map(
                    lambda acc_x, chunk_x: acc_x + chunk_x * chunk_weight,  # noqa: B023
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
        optimizer.update(accumulated_grads)

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
            eval_batch_size=periodic_eval_batch_size,
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
    result = {}
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
                ) = pool_train_step_sequential(
                    model,
                    optimizer,
                    circuit_pool,
                    idxs,
                    graphs,
                    wires,
                    logits,
                    x_data,
                    y_data,
                    tuple(layer_sizes),  # Convert list to tuple for JAX static arguments
                    n_message_steps,
                    loss_type=loss_type,
                    loss_key=loss_key,
                    epoch=epoch,
                    chunk_size=effective_batch_chunk_size,
                )
            else:
                (
                    loss,
                    (aux, circuit_pool, loss_steps),
                ) = pool_train_step(
                    model,
                    optimizer,
                    circuit_pool,
                    idxs,
                    graphs,
                    wires,
                    logits,
                    x_data,
                    y_data,
                    tuple(layer_sizes),  # Convert list to tuple for JAX static arguments
                    n_message_steps,
                    loss_type=loss_type,
                    loss_key=loss_key,
                    epoch=epoch,
                )

            *_, hard_loss, _, _, accuracy, hard_accuracy, _, _ = aux

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
                    if wiring_mode in ["fixed", "genetic"]:
                        reset_pool_key = wiring_fixed_key
                    else:
                        reset_pool_key = fresh_key

                    fresh_pool = initialize_graph_pool(
                        rng=reset_pool_key,
                        layer_sizes=layer_sizes,
                        pool_size=pool_size,  # Use same size as circuit_pool
                        input_n=input_n,
                        arity=arity,
                        circuit_hidden_dim=circuit_hidden_dim,
                        wiring_mode=wiring_mode,
                        initial_diversity=initial_diversity,
                    )

                    # Reset a fraction of the pool and get avg steps of reset graphs
                    circuit_pool, avg_steps_reset = circuit_pool.reset_fraction(
                        reset_key,
                        reset_pool_fraction,
                        fresh_pool.graphs,
                        fresh_pool.wires,
                        fresh_pool.logits,
                        reset_strategy=reset_strategy,
                        combined_weights=combined_weights,
                    )

                # Update last reset epoch
                last_reset_epoch = epoch
                diversity = circuit_pool.get_wiring_diversity(layer_sizes)

            # Record metrics
            losses.append(float(loss))
            hard_losses.append(float(hard_loss))
            accuracies.append(float(accuracy))
            hard_accuracies.append(float(hard_accuracy))
            reset_steps.append(float(avg_steps_reset))

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

            # Add learning rate if available
            schedule_value = schedule(epoch) if schedule is not None else learning_rate
            metrics_dict["scheduler/learning_rate"] = schedule_value

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

                eval_results = run_unified_periodic_evaluation(
                    model=model,
                    datasets=current_datasets,
                    pool=circuit_pool,
                    x_data=x_data,
                    y_data=y_data,
                    input_n=input_n,
                    arity=arity,
                    circuit_hidden_dim=circuit_hidden_dim,
                    n_message_steps=periodic_eval_inner_steps,  # Use fixed message steps
                    loss_type=loss_type,
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

            # Return the trained GNN model and metrics
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
                "best_model_tracker": best_model_tracker,  # Include unified best model tracker
            }

            # Add pool to result if used
            result["pool"] = circuit_pool
    except KeyboardInterrupt:
        log.info(f"Training interrupted by user at epoch {epoch}/{epochs}")
        # Ensure progress bar is properly closed
        pbar.close()

    # Log final results to wandb
    _log_final_wandb_metrics(wandb_run, result, epochs)

    return result
