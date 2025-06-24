"""
Training loop implementation for GNN-based boolean circuit optimization.

This module provides functions for training GNN models to optimize
boolean circuits over multiple epochs.
"""

import jax
import jax.numpy as jp
import optax
from flax import nnx
import jraph
from typing import List, Tuple, Dict, Any, Optional
from functools import partial
from tqdm.auto import tqdm
import os
import logging
from datetime import datetime

from boolean_nca_cc.models import CircuitGNN, CircuitSelfAttention
from boolean_nca_cc.training.utils import save_checkpoint
from boolean_nca_cc.training.schedulers import (
    should_reset_pool,
    get_learning_rate_schedule,
    get_step_beta,
)

from boolean_nca_cc.training.pool.pool import GraphPool, initialize_graph_pool
from boolean_nca_cc.training.evaluation import (
    evaluate_model_stepwise_batched,
    get_loss_and_update_graph,
)
from boolean_nca_cc.training.eval_datasets import (
    create_unified_evaluation_datasets,
    evaluate_circuits_in_chunks,
    UnifiedEvaluationDatasets,
)
import wandb

# Type alias for PyTree
PyTree = Any

# Setup logging
log = logging.getLogger(__name__)


def _init_wandb(
    wandb_logging: bool, wandb_run_config: Optional[Dict] = None
) -> Optional[Any]:
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


def _log_to_wandb(
    wandb_run, metrics_dict: Dict, epoch: int, log_interval: int = 1
) -> None:
    """Log metrics to wandb if enabled and interval allows."""
    if wandb_run is None or epoch % log_interval != 0:
        return

    try:
        wandb_run.log(metrics_dict)
    except Exception as e:
        log.warning(f"Error logging to wandb: {e}")


def _setup_checkpoint_dir(
    checkpoint_dir: Optional[str], wandb_id: Optional[str]
) -> Optional[str]:
    """Setup checkpoint directory with unique identifier."""
    if checkpoint_dir is None:
        return None

    # Create unique checkpoint directory using wandb ID or timestamp
    unique_id = wandb_id if wandb_id else datetime.now().strftime("%Y%m%d_%H%M%S")

    checkpoint_path = os.path.join(checkpoint_dir, f"run_{unique_id}")
    os.makedirs(checkpoint_path, exist_ok=True)
    log.info(f"Checkpoints will be saved to: {checkpoint_path}")

    return checkpoint_path


def _save_periodic_checkpoint(
    checkpoint_path: str,
    model,
    optimizer,
    metrics: Dict,
    epoch: int,
    checkpoint_interval: int,
    wandb_run=None,
) -> None:
    """Save periodic checkpoint if interval allows."""
    if checkpoint_path is None or epoch == 0 or epoch % checkpoint_interval != 0:
        return

    ckpt_filename = "latest_checkpoint.pkl"
    # log.info(f"Saving periodic checkpoint at epoch {epoch}")

    try:
        save_checkpoint(
            model,
            optimizer,
            metrics,
            {"epoch": epoch},
            epoch,
            checkpoint_path,
            filename=ckpt_filename,
        )

        # Log to wandb if enabled
        if wandb_run:
            wandb_run.save(os.path.join(checkpoint_path, ckpt_filename))

            # Also log this as an artifact for better tracking in wandb
            try:
                artifact = wandb_run.Artifact("latest_checkpoint", type="model")
                artifact.add_file(os.path.join(checkpoint_path, ckpt_filename))
                wandb_run.log_artifact(artifact)
            except Exception as e:
                log.warning(f"Error logging checkpoint as artifact: {e}")

    except Exception as e:
        log.warning(f"Error saving checkpoint: {e}")


def _save_best_checkpoint(
    checkpoint_path: str,
    is_best: bool,
    save_best: bool,
    model,
    optimizer,
    metrics: Dict,
    epoch: int,
    best_metric: str,
    current_metric_value: float,
    wandb_run=None,
) -> None:
    """Save best checkpoint if enabled and is best."""
    if not (checkpoint_path and save_best and is_best):
        return

    # Use a fixed filename for the best model to avoid creating multiple files
    best_filename = f"best_model_{best_metric}.pkl"
    # log.info(
    #     f"Saving best model at epoch {epoch} with {best_metric}={current_metric_value:.4f}"
    # )

    try:
        save_checkpoint(
            model,
            optimizer,
            metrics,
            {"epoch": epoch, f"best_{best_metric}": current_metric_value},
            epoch,
            checkpoint_path,
            filename=best_filename,
        )

        # Log to wandb if enabled
        if wandb_run:
            wandb_run.log(
                {f"best/{best_metric}": current_metric_value, "best/epoch": epoch}
            )

            # Save the best model to wandb (will overwrite the previous best)
            wandb_run.save(os.path.join(checkpoint_path, best_filename))

            # Also log this as an artifact for better tracking in wandb
            try:
                artifact = wandb_run.Artifact(f"best_model_{best_metric}", type="model")
                artifact.add_file(os.path.join(checkpoint_path, best_filename))
                wandb_run.log_artifact(artifact)
            except Exception as e:
                log.warning(f"Error logging best model as artifact: {e}")
    except Exception as e:
        log.warning(f"Error saving best checkpoint: {e}")


def _save_stable_state(
    checkpoint_path: str,
    save_stable_states: bool,
    last_stable_state: Dict,
    epoch: int,
    wandb_run=None,
) -> None:
    """Save the last stable state before NaN loss."""
    if not (checkpoint_path and save_stable_states):
        return

    try:
        stable_path = os.path.join(
            checkpoint_path, f"stable_state_epoch_{epoch - 1}.pkl"
        )
        # log.info(f"Saving last stable state to {stable_path}")
        save_checkpoint(
            last_stable_state["model"],
            last_stable_state["optimizer"],
            last_stable_state["metrics"],
            {"epoch": epoch - 1},
            epoch - 1,
            os.path.dirname(stable_path),
            filename=os.path.basename(stable_path),
        )

        # Log to wandb if enabled
        if wandb_run:
            wandb_run.log({"training/early_stop_epoch": epoch - 1})
            wandb_run.alert(
                title="Training Stopped - NaN Loss",
                text=f"Training stopped at epoch {epoch} due to NaN loss. Last stable state saved.",
                level=wandb_run.AlertLevel.WARN,
            )
    except Exception as e:
        log.warning(f"Error saving stable state: {e}")


def _check_early_stopping(
    stop_accuracy_enabled: bool,
    epoch: int,
    stop_accuracy_min_epochs: int,
    early_stop_triggered: bool,
    stop_accuracy_metric: str,
    stop_accuracy_source: str,
    training_metrics: Dict,
    current_eval_metrics: Optional[Dict],
    stop_accuracy_threshold: float,
    first_threshold_epoch: Optional[int],
    epochs_above_threshold: int,
    stop_accuracy_patience: int,
    rng: jax.random.PRNGKey,
) -> Tuple[bool, bool, int, Optional[int], Optional[Dict], jax.random.PRNGKey]:
    """
    Check early stopping conditions and handle early stopping logic.

    Returns:
        Tuple of (should_break, early_stop_triggered, epochs_above_threshold,
                 first_threshold_epoch, updated_current_eval_metrics, updated_rng)
    """
    if not stop_accuracy_enabled or early_stop_triggered:
        return (
            False,
            early_stop_triggered,
            epochs_above_threshold,
            first_threshold_epoch,
            current_eval_metrics,
            rng,
        )

    # Get the accuracy value for early stopping
    try:
        stop_accuracy_value = _get_metric_value(
            stop_accuracy_metric,
            stop_accuracy_source,
            training_metrics,
            current_eval_metrics,
        )
    except (ValueError, KeyError):
        if stop_accuracy_source == "eval" and current_eval_metrics is None:
            # Evaluation metrics not available, skip early stopping check this epoch
            stop_accuracy_value = None
        else:
            # Fallback to training metrics if eval not available
            stop_accuracy_value = _get_metric_value(
                stop_accuracy_metric,
                "training",
                training_metrics,
                current_eval_metrics,
            )

    if stop_accuracy_value is None:
        return (
            False,
            early_stop_triggered,
            epochs_above_threshold,
            first_threshold_epoch,
            current_eval_metrics,
            rng,
        )

    if stop_accuracy_value >= stop_accuracy_threshold:
        if first_threshold_epoch is None:
            first_threshold_epoch = epoch
            log.info(
                f"Reached accuracy threshold {stop_accuracy_threshold:.4f} "
                f"({stop_accuracy_source}_{stop_accuracy_metric}={stop_accuracy_value:.4f}) "
                f"at epoch {epoch}. Starting patience countdown."
            )
        epochs_above_threshold += 1

        # Check if we should stop (only after minimum epochs requirement is met)
        if (
            epochs_above_threshold >= stop_accuracy_patience
            and epoch >= stop_accuracy_min_epochs
        ):
            early_stop_triggered = True
            log.info(
                f"Early stopping triggered! "
                f"Accuracy {stop_accuracy_source}_{stop_accuracy_metric}={stop_accuracy_value:.4f} "
                f"has been above threshold {stop_accuracy_threshold:.4f} "
                f"for {stop_accuracy_patience} epochs. "
                f"Stopping at epoch {epoch}."
            )

            return (
                True,
                early_stop_triggered,
                epochs_above_threshold,
                first_threshold_epoch,
                current_eval_metrics,
                rng,
            )
        elif (
            epochs_above_threshold >= stop_accuracy_patience
            and epoch < stop_accuracy_min_epochs
        ):
            # Log that we would stop but are waiting for minimum epochs
            log.info(
                f"Early stopping condition met "
                f"(accuracy {stop_accuracy_source}_{stop_accuracy_metric}={stop_accuracy_value:.4f} "
                f"above threshold {stop_accuracy_threshold:.4f} for {stop_accuracy_patience} epochs), "
                f"but waiting until minimum epoch {stop_accuracy_min_epochs} (currently at epoch {epoch})."
            )
    else:
        # Reset counter if accuracy drops below threshold
        if epochs_above_threshold > 0:
            log.info(
                "Accuracy dropped below threshold. Resetting early stopping counter."
            )
        epochs_above_threshold = 0
        first_threshold_epoch = None

    return (
        False,
        early_stop_triggered,
        epochs_above_threshold,
        first_threshold_epoch,
        current_eval_metrics,
        rng,
    )


def _log_final_wandb_metrics(wandb_run, results: Dict, epochs: int) -> None:
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
                f"best/{results.get('best_metric', 'metric')}": results.get(
                    "best_metric_value", 0
                ),
            }
        )

    except Exception as e:
        log.warning(f"Error logging final metrics to wandb: {e}")


def _get_metric_value(
    metric_name: str,
    metric_source: str,
    training_metrics: Dict,
    eval_metrics: Dict = None,
) -> float:
    """
    Get metric value from the appropriate source.

    Args:
        metric_name: Name of the metric ('loss', 'hard_loss', 'accuracy', 'hard_accuracy')
        metric_source: Source of the metric ('training' or 'eval')
        training_metrics: Dictionary with training metrics
        eval_metrics: Dictionary with evaluation metrics (optional)

    Returns:
        The metric value as a float
    """
    if metric_source == "training":
        return training_metrics[metric_name]
    elif metric_source == "eval":
        if eval_metrics is None:
            raise ValueError("Evaluation metrics not available for eval source")
        # Map to evaluation metric keys (use IN-distribution evaluation for consistency)
        eval_key_map = {
            "loss": "eval_in/final_loss",
            "hard_loss": "eval_in/final_hard_loss",
            "accuracy": "eval_in/final_accuracy",
            "hard_accuracy": "eval_in/final_hard_accuracy",
        }
        return eval_metrics[eval_key_map[metric_name]]
    else:
        raise ValueError(f"Unknown metric source: {metric_source}")


def _log_pool_scatter(pool, epoch, wandb_run):
    """Log pool scatterplot to wandb."""
    if wandb_run is None:
        return

    all_loss, all_steps = pool.graphs.globals[..., 0], pool.graphs.globals[..., 1]
    data = list(zip(all_steps, all_loss))
    table = wandb.Table(data=data, columns=["steps", "loss"])
    wandb_run.log({"pool/scatter": wandb.plot.scatter(table, "steps", "loss")})


def run_unified_periodic_evaluation(
    model,
    datasets: UnifiedEvaluationDatasets,
    pool,
    x_data,
    y_data,
    input_n,
    arity,
    hidden_dim,
    n_message_steps,
    loss_type,
    epoch,
    wandb_run,
    log_stepwise=False,
    layer_sizes: List[Tuple[int, int]] = None,
    log_pool_scatter: bool = False,
) -> Dict:
    """
    Run unified periodic evaluation with only IN-distribution and OUT-of-distribution testing.

    Args:
        model: The model to evaluate
        datasets: UnifiedEvaluationDatasets object containing IN and OUT distribution circuits
        pool: GraphPool for logging scatter plot
        x_data: Input data
        y_data: Target data
        input_n: Number of input nodes
        arity: Arity of gates
        hidden_dim: Hidden dimension
        n_message_steps: Number of message steps for evaluation
        loss_type: Loss function type
        epoch: Current epoch number
        wandb_run: WandB run object (or None)
        log_stepwise: Whether to log step-by-step metrics
        layer_sizes: Circuit layer sizes
        log_pool_scatter: Whether to log pool scatterplot (loss vs steps)

    Returns:
        Dictionary with evaluation metrics from IN-distribution and OUT-of-distribution evaluations
    """
    try:
        # 1. Run IN-distribution evaluation (matches training pattern)
        # Use chunked evaluation to handle cases where diversity exceeds target batch size
        log.info(
            f"Running IN-distribution evaluation ({datasets.in_actual_batch_size} circuits)..."
        )
        if datasets.in_actual_batch_size > datasets.target_batch_size:
            log.info(
                f"Using chunked evaluation (chunks of {datasets.target_batch_size})"
            )

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
            hidden_dim=hidden_dim,
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

        # 2. Run OUT-of-distribution evaluation (always random)
        log.info(
            f"Running OUT-of-distribution evaluation ({datasets.out_actual_batch_size} circuits)..."
        )
        if datasets.out_actual_batch_size > datasets.target_batch_size:
            log.info(
                f"Using chunked evaluation (chunks of {datasets.target_batch_size})"
            )

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
            hidden_dim=hidden_dim,
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

        # Combine all metrics for logging
        combined_metrics = {
            **final_metrics_in,
            **final_metrics_out,
        }

        # Log to wandb if enabled
        if wandb_run:
            wandb_run.log(combined_metrics)

            if log_pool_scatter:
                _log_pool_scatter(pool, epoch, wandb_run)

            # Optionally log step-by-step metrics for both evaluations
            if log_stepwise:
                # IN-distribution step-wise metrics
                for step_idx in range(len(step_metrics_in["step"])):
                    step_data_in = {
                        "eval_in_steps/step": step_metrics_in["step"][step_idx],
                        "eval_in_steps/loss": step_metrics_in["soft_loss"][step_idx],
                        "eval_in_steps/hard_loss": step_metrics_in["hard_loss"][
                            step_idx
                        ],
                        "eval_in_steps/accuracy": step_metrics_in["soft_accuracy"][
                            step_idx
                        ],
                        "eval_in_steps/hard_accuracy": step_metrics_in["hard_accuracy"][
                            step_idx
                        ],
                        "eval_in_steps/epoch": epoch,
                    }
                    wandb_run.log(step_data_in)

                # OUT-of-distribution step-wise metrics
                for step_idx in range(len(step_metrics_out["step"])):
                    step_data_out = {
                        "eval_out_steps/step": step_metrics_out["step"][step_idx],
                        "eval_out_steps/loss": step_metrics_out["soft_loss"][step_idx],
                        "eval_out_steps/hard_loss": step_metrics_out["hard_loss"][
                            step_idx
                        ],
                        "eval_out_steps/accuracy": step_metrics_out["soft_accuracy"][
                            step_idx
                        ],
                        "eval_out_steps/hard_accuracy": step_metrics_out[
                            "hard_accuracy"
                        ][step_idx],
                        "eval_out_steps/epoch": epoch,
                    }
                    wandb_run.log(step_data_out)

        # Log summary to console
        training_config = datasets.training_config

        # Add chunking info if used
        in_chunk_info = ""
        if datasets.in_actual_batch_size > datasets.target_batch_size:
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

        log_message = (
            f"Unified Eval (epoch {epoch}):\n"
            f"  IN-distribution ({datasets.in_actual_batch_size} circuits{in_chunk_info}, "
            f"mode={training_config['wiring_mode']}, diversity={training_config['initial_diversity']}): "
            f"Loss={final_metrics_in['eval_in/final_loss']:.4f}, "
            f"Acc={final_metrics_in['eval_in/final_accuracy']:.4f}, "
            f"Hard Acc={final_metrics_in['eval_in/final_hard_accuracy']:.4f}\n"
            f"  OUT-of-distribution ({datasets.out_actual_batch_size} circuits{out_chunk_info}, random): "
            f"Loss={final_metrics_out['eval_out/final_loss']:.4f}, "
            f"Acc={final_metrics_out['eval_out/final_accuracy']:.4f}, "
            f"Hard Acc={final_metrics_out['eval_out/final_hard_accuracy']:.4f}"
        )

        log.info(log_message)

        # Return all step metrics and final metrics for best model tracking
        result = {
            "step_metrics_in": step_metrics_in,
            "step_metrics_out": step_metrics_out,
            "final_metrics_in": final_metrics_in,
            "final_metrics_out": final_metrics_out,
            # Add datasets information for comprehensive result reporting
            "datasets_info": {
                "in_actual_batch_size": datasets.in_actual_batch_size,
                "out_actual_batch_size": datasets.out_actual_batch_size,
                "target_batch_size": datasets.target_batch_size,
                "in_used_chunking": datasets.in_actual_batch_size
                > datasets.target_batch_size,
                "out_used_chunking": datasets.out_actual_batch_size
                > datasets.target_batch_size,
                "training_wiring_mode": datasets.training_config["wiring_mode"],
                "training_initial_diversity": datasets.training_config[
                    "initial_diversity"
                ],
                "evaluation_base_seed": datasets.training_config[
                    "evaluation_base_seed"
                ],
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
    layer_sizes: List[Tuple[int, int]],
    # Model architecture parameters
    arity: int = 2,
    hidden_dim: int = 16,
    message_passing: bool = True,
    node_mlp_features: List[int] = [64, 32],
    edge_mlp_features: List[int] = [64, 32],
    use_attention: bool = False,
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
    # Genetic mutation parameters (only used when wiring_mode='genetic')
    genetic_mutation_rate: float = 0.0,  # Fraction of connections to mutate (0.0 to 1.0)
    genetic_swaps_per_layer: int = 1,  # Number of swaps per layer for genetic mutation
    initial_diversity: int = 1,  # Number of initial wires for genetic mutation
    # Pool parameters
    pool_size: int = 1024,
    reset_pool_fraction: float = 0.05,
    reset_pool_interval: int = 128,
    reset_strategy: str = "uniform",  # Options: "uniform", "steps_biased", "loss_biased", or "combined"
    combined_weights: Tuple[float, float] = (
        0.5,
        0.5,
    ),  # Weights for [loss, steps] in combined strategy
    # Learning rate scheduling
    lr_scheduler: str = "constant",  # Options: "constant", "exponential", "cosine", "linear_warmup"
    lr_scheduler_params: Dict = None,
    # Initialization parameters
    key: int = 0,
    wiring_fixed_key: jax.random.PRNGKey = jax.random.PRNGKey(
        42
    ),  # Fixed key for generating wirings when wiring_mode='fixed'
    init_model: CircuitGNN | CircuitSelfAttention = None,
    init_optimizer: nnx.Optimizer = None,
    initial_metrics: Dict = None,
    # Checkpointing parameters
    checkpoint_enabled: bool = False,
    checkpoint_dir: str = None,
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
    wandb_run_config: Dict = None,
    # Early stopping parameters
    stop_accuracy_enabled: bool = False,
    stop_accuracy_threshold: float = 0.95,
    stop_accuracy_metric: str = "hard_accuracy",
    stop_accuracy_source: str = "training",
    stop_accuracy_patience: int = 10,
    stop_accuracy_min_epochs: int = 100,
):
    """
    Train a GNN to optimize boolean circuit parameters.

    Args:
        layer_sizes: List of tuples (nodes, group_size) for each layer
        x_data: Input data for training [batch, input_bits]
        y_data: Target output data [batch, output_bits]
        arity: Number of inputs per gate
        hidden_dim: Dimension of hidden features
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
        # Create a new GNN
        rng, init_key = jax.random.split(rng)
        model = CircuitGNN(
            node_mlp_features=node_mlp_features,
            edge_mlp_features=edge_mlp_features,
            hidden_dim=hidden_dim,
            arity=arity,
            message_passing=message_passing,
            use_attention=use_attention,
            rngs=nnx.Rngs(params=init_key),
        )
    else:
        # Use the provided GNN
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
        hidden_dim=hidden_dim,
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
        layer_sizes: Tuple[Tuple[int, int], ...],
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

            for i in range(n_message_steps):
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
                current_update_steps = (
                    graph.globals[..., 1] if graph.globals is not None else 0
                )
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
            if use_scan:
                loss_fn = loss_fn_scan
            else:
                loss_fn = loss_fn_no_scan

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
        (loss, (aux, updated_graphs, updated_logits, loss_steps)), grads = (
            nnx.value_and_grad(batch_loss_fn, has_aux=True)(
                model=model,
                graphs=graphs,
                logits=logits,
                wires=wires,
                loss_key=loss_key,
            )
        )

        # Update GNN parameters
        optimizer.update(grads)

        # Update pool with the updated graphs and logits (wires stay the same)
        updated_pool = pool.update(idxs, updated_graphs, batch_of_logits=updated_logits)

        return loss, (aux, updated_pool, loss_steps)

    # Setup wandb logging if enabled
    wandb_run = _init_wandb(wandb_logging, wandb_run_config)
    wandb_id = wandb_run.run.id if wandb_run else None

    # Setup checkpointing directory
    checkpoint_path = _setup_checkpoint_dir(checkpoint_dir, wandb_id)

    # Track best model
    best_metric_value = float("-inf") if "accuracy" in best_metric else float("inf")

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

    # Save initial stable state if needed
    last_stable_state = {
        "model": model,
        "optimizer": optimizer,
        "pool": circuit_pool,
        "metrics": {
            "losses": losses,
            "hard_losses": hard_losses,
            "accuracies": accuracies,
            "hard_accuracies": hard_accuracies,
            "reset_steps": reset_steps,
        },
        "epoch": 0,
    }

    diversity = 0.0
    result = {}
    # Training loop
    try:
        for epoch in pbar:
            # Pool-based training
            # Sample a batch from the pool using the current (potentially dynamic) batch size
            rng, sample_key, loss_key = jax.random.split(rng, 3)
            idxs, graphs, wires, logits = circuit_pool.sample(
                sample_key, meta_batch_size
            )

            # Perform pool training step
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
                    circuit_pool, avg_steps_reset = (
                        circuit_pool.reset_with_genetic_mutation(
                            key=reset_key,
                            fraction=reset_pool_fraction,
                            layer_sizes=layer_sizes,
                            input_n=input_n,
                            arity=arity,
                            hidden_dim=hidden_dim,
                            mutation_rate=genetic_mutation_rate,
                            n_swaps_per_layer=genetic_swaps_per_layer,
                            reset_strategy=reset_strategy,
                            combined_weights=combined_weights,
                        )
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
                        hidden_dim=hidden_dim,
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

            if jp.isnan(loss):
                log.warning(
                    f"Loss is NaN at epoch {epoch}, returning last stable state"
                )
                # Save the last stable state if enabled
                _save_stable_state(
                    checkpoint_path,
                    save_stable_states,
                    last_stable_state,
                    epoch,
                    wandb_run,
                )
                return last_stable_state
            else:
                # Update last stable state
                last_stable_state = {
                    "model": model,
                    "optimizer": optimizer,
                    "pool": circuit_pool,
                    "metrics": {
                        "losses": losses.copy(),
                        "hard_losses": hard_losses.copy(),
                        "accuracies": accuracies.copy(),
                        "hard_accuracies": hard_accuracies.copy(),
                        "reset_steps": reset_steps.copy(),
                    },
                    "epoch": epoch,
                }

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

                # Add learning rate if available
                if schedule is not None:
                    schedule_value = schedule(epoch)
                else:
                    schedule_value = learning_rate
                metrics_dict["scheduler/learning_rate"] = schedule_value

                # Add early stopping metrics if enabled
                if stop_accuracy_enabled:
                    metrics_dict["early_stop/enabled"] = True
                    metrics_dict["early_stop/epochs_above_threshold"] = (
                        epochs_above_threshold
                    )
                    metrics_dict["early_stop/threshold"] = stop_accuracy_threshold
                    if first_threshold_epoch is not None:
                        metrics_dict["early_stop/first_threshold_epoch"] = (
                            first_threshold_epoch
                        )

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

                # Add early stopping info if active
                if stop_accuracy_enabled and epochs_above_threshold > 0:
                    postfix_dict["ES"] = (
                        f"{epochs_above_threshold}/{stop_accuracy_patience}"
                    )

                pbar.set_postfix(postfix_dict)

                # Step 2: Run periodic evaluation if enabled (BEFORE best model tracking)
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
                        hidden_dim=hidden_dim,
                        n_message_steps=periodic_eval_inner_steps,  # Use fixed message steps
                        loss_type=loss_type,
                        epoch=epoch,
                        wandb_run=wandb_run,
                        log_stepwise=periodic_eval_log_stepwise,
                        layer_sizes=layer_sizes,
                        log_pool_scatter=periodic_eval_log_pool_scatter,
                    )
                    # Extract final metrics for best model tracking (use IN-distribution metrics)
                    current_eval_metrics = eval_results.get("final_metrics_in", None)

                # Step 3: Get current metric value for best model tracking using modular approach
                try:
                    current_metric_value = _get_metric_value(
                        best_metric,
                        best_metric_source,
                        training_metrics,
                        current_eval_metrics,
                    )
                except (ValueError, KeyError) as e:
                    if best_metric_source == "eval" and not periodic_eval_enabled:
                        log.warning(
                            f"Best metric source is 'eval' but periodic evaluation is disabled. "
                            f"Falling back to training metrics for {best_metric}."
                        )
                        current_metric_value = _get_metric_value(
                            best_metric,
                            "training",
                            training_metrics,
                            current_eval_metrics,
                        )
                    elif best_metric_source == "eval" and current_eval_metrics is None:
                        # Evaluation is enabled but hasn't run yet this epoch, skip best model check
                        current_metric_value = None
                    else:
                        raise e

                # Check if this is the best model based on the specified metric
                is_best = False
                if current_metric_value is not None:
                    if (
                        "accuracy" in best_metric
                    ):  # For accuracy metrics, higher is better
                        if current_metric_value > best_metric_value:
                            best_metric_value = current_metric_value
                            is_best = True
                    else:  # For loss metrics, lower is better
                        if current_metric_value < best_metric_value:
                            best_metric_value = current_metric_value
                            is_best = True

                # Step 4: Save checkpoints (periodic always, best if improvement detected)
                if checkpoint_enabled:
                    _save_periodic_checkpoint(
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

                    # Save best model if enabled and is best
                    _save_best_checkpoint(
                        checkpoint_path,
                        is_best,
                        save_best,
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
                        f"{best_metric_source}_{best_metric}",  # Include source in metric name
                        current_metric_value
                        if current_metric_value is not None
                        else best_metric_value,
                        wandb_run,
                    )

                # Step 5: Check for early stopping based on accuracy
                (
                    should_break,
                    early_stop_triggered,
                    epochs_above_threshold,
                    first_threshold_epoch,
                    current_eval_metrics,
                    rng,
                ) = _check_early_stopping(
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
                    "best_metric_value": best_metric_value,
                    "best_metric": best_metric,
                    "early_stopped": early_stop_triggered,
                    "early_stop_epoch": epoch if early_stop_triggered else None,
                    "first_threshold_epoch": first_threshold_epoch,
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
