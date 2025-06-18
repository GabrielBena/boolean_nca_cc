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
import tempfile
from datetime import datetime

from boolean_nca_cc.models import CircuitGNN, CircuitSelfAttention
from boolean_nca_cc.training.utils import save_checkpoint, plot_training_curves
from boolean_nca_cc.training.schedulers import (
    should_reset_pool,
    get_current_reset_interval,
    get_learning_rate_schedule,
    get_step_beta,
)

from boolean_nca_cc.circuits.model import gen_circuit
from boolean_nca_cc.training.pool.pool import GraphPool, initialize_graph_pool
from boolean_nca_cc.training.evaluation import (
    evaluate_model_stepwise,
    evaluate_model_stepwise_batched,
    get_loss_and_update_graph,
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

        # Create and log final summary plots
        with tempfile.TemporaryDirectory() as tmpdir:
            plot_training_curves(
                {
                    "losses": results["losses"],
                    "hard_losses": results["hard_losses"],
                    "accuracies": results["accuracies"],
                    "hard_accuracies": results["hard_accuracies"],
                },
                "Training Summary",
                tmpdir,
            )
            # Log the plots to wandb
            wandb_run.log(
                {
                    "summary/loss_curve": wandb_run.Image(
                        os.path.join(tmpdir, "training_summary_loss.png")
                    ),
                    "summary/accuracy_curve": wandb_run.Image(
                        os.path.join(tmpdir, "training_summary_accuracy.png")
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
        # Map to evaluation metric keys (use fixed seed evaluation for consistency)
        eval_key_map = {
            "loss": "eval_seed/final_loss",
            "hard_loss": "eval_seed/final_hard_loss",
            "accuracy": "eval_seed/final_accuracy",
            "hard_accuracy": "eval_seed/final_hard_accuracy",
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


def _run_periodic_evaluation_dual(
    model,
    pool,
    test_seed_wires,
    test_seed_logits,
    test_random_wires,
    test_random_logits,
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
    initial_diversity: int = 1,
    eval_batch_size: int = 16,
    eval_rng_key: jax.random.PRNGKey = None,
) -> Dict:
    """
    Run dual periodic evaluation: both fixed seed and in-distribution/OOD batch evaluations.

    Args:
        model: The model to evaluate
        test_seed_wires: Test circuit wires (single circuit with fixed seed)
        test_seed_logits: Test circuit logits (single circuit with fixed seed)
        test_random_wires: Test circuit wires (batch of random circuits for OOD)
        test_random_logits: Test circuit logits (batch of random circuits for OOD)
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
        initial_diversity: Number of initial diverse circuits in the pool
        eval_batch_size: Batch size for pool sampling evaluation
        eval_rng_key: Random key for evaluation sampling

    Returns:
        Dictionary with evaluation metrics from fixed seed, in-distribution, and OOD evaluations
    """
    try:
        # 1. Run fixed seed evaluation (single circuit)
        step_metrics_seed = evaluate_model_stepwise(
            model=model,
            wires=test_seed_wires,
            logits=test_seed_logits,
            x_data=x_data,
            y_data=y_data,
            input_n=input_n,
            arity=arity,
            hidden_dim=hidden_dim,
            n_message_steps=n_message_steps,
            loss_type=loss_type,
            layer_sizes=layer_sizes,
        )

        # Get final metrics (last step) for fixed seed
        final_metrics_seed = {
            "eval_seed/final_loss": step_metrics_seed["soft_loss"][-1],
            "eval_seed/final_hard_loss": step_metrics_seed["hard_loss"][-1],
            "eval_seed/final_accuracy": step_metrics_seed["soft_accuracy"][-1],
            "eval_seed/final_hard_accuracy": step_metrics_seed["hard_accuracy"][-1],
            "eval_seed/epoch": epoch,
        }

        # 2. Run in-distribution evaluation if diversity > 1 (sample from pool)
        final_metrics_pool = {}
        step_metrics_pool = None
        if initial_diversity > 1 and eval_rng_key is not None:
            # Sample circuits from the current pool for in-distribution evaluation
            pool_sample_key, eval_rng_key = jax.random.split(eval_rng_key)
            _, pool_graphs, pool_wires, pool_logits = pool.sample(
                pool_sample_key, eval_batch_size
            )

            # Reset the pool circuits to initial state for fair evaluation
            # (remove any accumulated update steps and set loss to 0)
            pool_graphs = pool_graphs._replace(
                globals=jp.zeros_like(pool_graphs.globals)
            )

            step_metrics_pool = evaluate_model_stepwise_batched(
                model=model,
                batch_wires=pool_wires,
                batch_logits=pool_logits,
                x_data=x_data,
                y_data=y_data,
                input_n=input_n,
                arity=arity,
                hidden_dim=hidden_dim,
                n_message_steps=n_message_steps,
                loss_type=loss_type,
                layer_sizes=layer_sizes,
            )

            # Get final metrics (last step) for pool sample
            final_metrics_pool = {
                "eval_pool/final_loss": step_metrics_pool["soft_loss"][-1],
                "eval_pool/final_hard_loss": step_metrics_pool["hard_loss"][-1],
                "eval_pool/final_accuracy": step_metrics_pool["soft_accuracy"][-1],
                "eval_pool/final_hard_accuracy": step_metrics_pool["hard_accuracy"][-1],
                "eval_pool/epoch": epoch,
            }

        # 3. Run random batch evaluation (OOD testing)
        step_metrics_random = evaluate_model_stepwise_batched(
            model=model,
            batch_wires=test_random_wires,
            batch_logits=test_random_logits,
            x_data=x_data,
            y_data=y_data,
            input_n=input_n,
            arity=arity,
            hidden_dim=hidden_dim,
            n_message_steps=n_message_steps,
            loss_type=loss_type,
            layer_sizes=layer_sizes,
        )

        # Get final metrics (last step) for random batch
        final_metrics_random = {
            "eval_ood/final_loss": step_metrics_random["soft_loss"][-1],
            "eval_ood/final_hard_loss": step_metrics_random["hard_loss"][-1],
            "eval_ood/final_accuracy": step_metrics_random["soft_accuracy"][-1],
            "eval_ood/final_hard_accuracy": step_metrics_random["hard_accuracy"][-1],
            "eval_ood/epoch": epoch,
        }

        # Combine all metrics for logging
        combined_metrics = {
            **final_metrics_seed,
            **final_metrics_pool,
            **final_metrics_random,
        }

        # Log to wandb if enabled
        if wandb_run:
            wandb_run.log(combined_metrics)

            if log_pool_scatter:
                _log_pool_scatter(pool, epoch, wandb_run)

            # Optionally log step-by-step metrics for all evaluations
            if log_stepwise:
                # Fixed seed step-wise metrics
                for step_idx in range(len(step_metrics_seed["step"])):
                    step_data_seed = {
                        f"eval_seed_steps/step": step_metrics_seed["step"][step_idx],
                        f"eval_seed_steps/loss": step_metrics_seed["soft_loss"][
                            step_idx
                        ],
                        f"eval_seed_steps/hard_loss": step_metrics_seed["hard_loss"][
                            step_idx
                        ],
                        f"eval_seed_steps/accuracy": step_metrics_seed["soft_accuracy"][
                            step_idx
                        ],
                        f"eval_seed_steps/hard_accuracy": step_metrics_seed[
                            "hard_accuracy"
                        ][step_idx],
                        f"eval_seed_steps/epoch": epoch,
                    }
                    wandb_run.log(step_data_seed)

                # Pool sample step-wise metrics (if available)
                if step_metrics_pool is not None:
                    for step_idx in range(len(step_metrics_pool["step"])):
                        step_data_pool = {
                            f"eval_pool_steps/step": step_metrics_pool["step"][
                                step_idx
                            ],
                            f"eval_pool_steps/loss": step_metrics_pool["soft_loss"][
                                step_idx
                            ],
                            f"eval_pool_steps/hard_loss": step_metrics_pool[
                                "hard_loss"
                            ][step_idx],
                            f"eval_pool_steps/accuracy": step_metrics_pool[
                                "soft_accuracy"
                            ][step_idx],
                            f"eval_pool_steps/hard_accuracy": step_metrics_pool[
                                "hard_accuracy"
                            ][step_idx],
                            f"eval_pool_steps/epoch": epoch,
                        }
                        wandb_run.log(step_data_pool)

                # Random batch step-wise metrics
                for step_idx in range(len(step_metrics_random["step"])):
                    step_data_random = {
                        f"eval_ood_steps/step": step_metrics_random["step"][step_idx],
                        f"eval_ood_steps/loss": step_metrics_random["soft_loss"][
                            step_idx
                        ],
                        f"eval_ood_steps/hard_loss": step_metrics_random["hard_loss"][
                            step_idx
                        ],
                        f"eval_ood_steps/accuracy": step_metrics_random[
                            "soft_accuracy"
                        ][step_idx],
                        f"eval_ood_steps/hard_accuracy": step_metrics_random[
                            "hard_accuracy"
                        ][step_idx],
                        f"eval_ood_steps/epoch": epoch,
                    }
                    wandb_run.log(step_data_random)

        # Log summary to console
        batch_size = test_random_wires[0].shape[0]
        log_message = (
            f"Periodic Eval (epoch {epoch}):\n"
            f"  Fixed Seed: Loss={final_metrics_seed['eval_seed/final_loss']:.4f}, "
            f"Acc={final_metrics_seed['eval_seed/final_accuracy']:.4f}, "
            f"Hard Acc={final_metrics_seed['eval_seed/final_hard_accuracy']:.4f}\n"
        )

        # Add pool evaluation if available
        if final_metrics_pool:
            log_message += (
                f"  Pool Sample (batch {eval_batch_size}): Loss={final_metrics_pool['eval_pool/final_loss']:.4f}, "
                f"Acc={final_metrics_pool['eval_pool/final_accuracy']:.4f}, "
                f"Hard Acc={final_metrics_pool['eval_pool/final_hard_accuracy']:.4f}\n"
            )

        log_message += (
            f"  OOD (batch {batch_size}): Loss={final_metrics_random['eval_ood/final_loss']:.4f}, "
            f"Acc={final_metrics_random['eval_ood/final_accuracy']:.4f}, "
            f"Hard Acc={final_metrics_random['eval_ood/final_hard_accuracy']:.4f}"
        )

        log.info(log_message)

        # Return all step metrics and final metrics for best model tracking
        result = {
            "step_metrics_seed": step_metrics_seed,
            "step_metrics_random": step_metrics_random,
            "final_metrics_seed": final_metrics_seed,
            "final_metrics_random": final_metrics_random,
        }

        # Add pool metrics if available
        if step_metrics_pool is not None:
            result["step_metrics_pool"] = step_metrics_pool
            result["final_metrics_pool"] = final_metrics_pool

        return result

    except Exception as e:
        log.warning(f"Error during periodic evaluation at epoch {epoch}: {e}")
        return {}


def _run_periodic_evaluation(
    model,
    pool,
    test_wires,
    test_logits,
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
    wiring_mode: str = "random",
) -> Dict:
    """
    Run periodic evaluation on test circuits.

    Args:
        model: The model to evaluate
        test_wires: Test circuit wires (single circuit or batch)
        test_logits: Test circuit logits (single circuit or batch)
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
        wiring_mode: Wiring mode ("fixed" or "random")

    Returns:
        Dictionary with evaluation metrics including final values for best model tracking
    """
    try:
        if wiring_mode == "fixed":
            # For fixed wiring, use single circuit evaluation
            step_metrics = evaluate_model_stepwise(
                model=model,
                wires=test_wires,
                logits=test_logits,
                x_data=x_data,
                y_data=y_data,
                input_n=input_n,
                arity=arity,
                hidden_dim=hidden_dim,
                n_message_steps=n_message_steps,
                loss_type=loss_type,
                layer_sizes=layer_sizes,
            )
            eval_type = "fixed"
        else:
            # For random wiring, use batched evaluation
            # test_wires and test_logits should already be batched from initialization
            step_metrics = evaluate_model_stepwise_batched(
                model=model,
                batch_wires=test_wires,
                batch_logits=test_logits,
                x_data=x_data,
                y_data=y_data,
                input_n=input_n,
                arity=arity,
                hidden_dim=hidden_dim,
                n_message_steps=n_message_steps,
                loss_type=loss_type,
                layer_sizes=layer_sizes,
            )
            eval_type = f"random_batch_{test_wires[0].shape[0]}"

        # Get final metrics (last step)
        final_metrics = {
            "eval/final_loss": step_metrics["soft_loss"][-1],
            "eval/final_hard_loss": step_metrics["hard_loss"][-1],
            "eval/final_accuracy": step_metrics["soft_accuracy"][-1],
            "eval/final_hard_accuracy": step_metrics["hard_accuracy"][-1],
            "eval/epoch": epoch,
        }

        # Log to wandb if enabled
        if wandb_run:
            wandb_run.log(final_metrics)

            if log_pool_scatter:
                _log_pool_scatter(pool, epoch, wandb_run)

            # Optionally log step-by-step metrics
            if log_stepwise:
                for step_idx in range(len(step_metrics["step"])):
                    step_data = {
                        f"eval_steps/step": step_metrics["step"][step_idx],
                        f"eval_steps/loss": step_metrics["soft_loss"][step_idx],
                        f"eval_steps/hard_loss": step_metrics["hard_loss"][step_idx],
                        f"eval_steps/accuracy": step_metrics["soft_accuracy"][step_idx],
                        f"eval_steps/hard_accuracy": step_metrics["hard_accuracy"][
                            step_idx
                        ],
                        f"eval_steps/epoch": epoch,
                    }
                    wandb_run.log(step_data)

        # Log summary to console
        log.info(
            f"Periodic Eval ({eval_type}, epoch {epoch}): "
            f"Loss={final_metrics['eval/final_loss']:.4f}, "
            f"Acc={final_metrics['eval/final_accuracy']:.4f}, "
            f"Hard Acc={final_metrics['eval/final_hard_accuracy']:.4f}"
        )

        # Return both step metrics and final metrics for best model tracking
        return {
            "step_metrics": step_metrics,
            "final_metrics": final_metrics,
        }

    except Exception as e:
        log.warning(f"Error during periodic evaluation at epoch {epoch}: {e}")
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
    reset_pool_interval: int = 10,
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
    init_pool: GraphPool = None,
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
        init_pool: Optional pre-initialized GraphPool to continue training with
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

    # Initialize Graph Pool for training if using pool
    if init_pool is None:
        if wiring_mode not in ["fixed", "genetic"]:
            rng, pool_rng = jax.random.split(rng)
        else:
            pool_rng = wiring_fixed_key
        # Initialize a fresh pool
        circuit_pool = initialize_graph_pool(
            rng=pool_rng,
            layer_sizes=layer_sizes,
            pool_size=pool_size,
            input_n=input_n,
            arity=arity,
            hidden_dim=hidden_dim,
            loss_value=0.0,  # Initial loss will be calculated properly in first step
            wiring_mode=wiring_mode,
            initial_diversity=initial_diversity,
        )
    else:
        # Use the provided pool
        # log.info("Using provided pool for training")
        circuit_pool = init_pool

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

    # Create progress bar for training
    pbar = tqdm(range(epochs), desc="Training GNN")
    avg_steps_reset = 0

    # Track last reset epoch for scheduling
    last_reset_epoch = -1  # Initialize to -1 so first check works correctly

    # Initialize test circuits for periodic evaluation if enabled
    test_seed_wires = None
    test_seed_logits = None
    test_random_wires = None
    test_random_logits = None

    if periodic_eval_enabled:
        # Always initialize both types of test circuits

        # 1. Fixed test seed circuit (for consistent evaluation)
        test_seed_rng = jax.random.PRNGKey(periodic_eval_test_seed)
        test_seed_wires, test_seed_logits = gen_circuit(
            test_seed_rng, layer_sizes, arity=arity
        )
        log.info("Fixed test seed circuit initialized for periodic evaluation")

        # 2. Batch of random test circuits (for OOD evaluation)
        test_random_rng = jax.random.PRNGKey(
            periodic_eval_test_seed + 1000
        )  # Different seed for random batch
        test_random_rngs = jax.random.split(test_random_rng, periodic_eval_batch_size)

        # Use vmap to generate multiple circuits
        vmap_gen_circuit = jax.vmap(
            lambda rng: gen_circuit(rng, layer_sizes, arity=arity)
        )
        test_random_wires, test_random_logits = vmap_gen_circuit(test_random_rngs)

        log.info(
            f"Batch of {periodic_eval_batch_size} random test circuits initialized for periodic evaluation (OOD testing)"
        )

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
                    if wiring_mode == "fixed":
                        # Use the fixed key for generating wirings
                        fresh_key = wiring_fixed_key

                    fresh_pool = initialize_graph_pool(
                        rng=fresh_key,
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

                _log_to_wandb(wandb_run, metrics_dict, epoch, log_interval)

                # Update progress bar with current metrics
                pbar.set_postfix(
                    {
                        "Loss": f"{loss:.4f}",
                        "Accuracy": f"{accuracy:.4f}",
                        "Hard Acc": f"{hard_accuracy:.4f}",
                        "Diversity": f"{diversity:.3f}",
                        "Reset Steps": f"{avg_steps_reset:.2f}",
                        "Loss Steps": f"{loss_steps:.2f}",
                    }
                )

                # Step 2: Run periodic evaluation if enabled (BEFORE best model tracking)
                if (
                    periodic_eval_enabled
                    and test_seed_wires is not None
                    and test_seed_logits is not None
                    and test_random_wires is not None
                    and test_random_logits is not None
                    and epoch % periodic_eval_interval == 0
                ):
                    # Run enhanced evaluations: fixed seed, pool sample (if diversity > 1), and OOD
                    rng, eval_key = jax.random.split(rng)
                    eval_results = _run_periodic_evaluation_dual(
                        model=model,
                        pool=circuit_pool,
                        test_seed_wires=test_seed_wires,
                        test_seed_logits=test_seed_logits,
                        test_random_wires=test_random_wires,
                        test_random_logits=test_random_logits,
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
                        initial_diversity=initial_diversity,
                        eval_batch_size=periodic_eval_batch_size,
                        eval_rng_key=eval_key,
                    )
                    # Extract final metrics for best model tracking (use fixed seed metrics)
                    current_eval_metrics = eval_results.get("final_metrics_seed", None)

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
