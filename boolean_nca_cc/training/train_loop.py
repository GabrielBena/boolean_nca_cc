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
from boolean_nca_cc.utils import build_graph, extract_logits_from_graph
from boolean_nca_cc.training.utils import save_checkpoint, plot_training_curves
from boolean_nca_cc.circuits.train import (
    res2loss,
    binary_cross_entropy,
    compute_accuracy,
)
from boolean_nca_cc.circuits.model import gen_circuit, run_circuit
from boolean_nca_cc.training.pool import GraphPool, initialize_graph_pool

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
    log.info(f"Saving periodic checkpoint at epoch {epoch}")

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


def get_loss_from_graph(logits, wires, x, y_target, loss_type: str):
    # --- Calculate initial loss for graph globals --- START
    acts = run_circuit(logits, wires, x)
    pred = acts[-1]
    acts_hard = run_circuit(logits, wires, x, hard=True)
    pred_hard = acts_hard[-1]

    if loss_type == "bce":
        loss = binary_cross_entropy(pred, y_target)
        hard_loss = binary_cross_entropy(pred_hard, y_target)
    elif loss_type == "l4":
        res = pred - y_target
        hard_res = pred_hard - y_target
        loss = res2loss(res, power=4)
        hard_loss = res2loss(hard_res, power=4)
    elif loss_type == "l2":
        res = pred - y_target
        hard_res = pred_hard - y_target
        loss = res2loss(res, power=2)
        hard_loss = res2loss(hard_res, power=2)
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")
    # --- Calculate initial loss for graph globals --- END

    return loss, (hard_loss, pred, pred_hard)


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
    loss_type: str = "l4",  # Options: 'l4' or 'bce'
    # Wiring mode parameters
    wiring_mode: str = "random",  # Options: 'fixed' or 'random'
    meta_batch_size: int = 64,
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
    checkpoint_dir: str = None,
    checkpoint_interval: int = 10,
    save_best: bool = True,
    best_metric: str = "hard_accuracy",  # Options: 'loss', 'hard_loss', 'accuracy', 'hard_accuracy'
    save_stable_states: bool = True,
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
        wiring_mode: Mode for circuit wirings ('fixed' or 'random')
        meta_batch_size: Batch size for training
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
        save_stable_states: Whether to save stable states (before potential NaN losses)
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
        # Initialize scheduler parameters if None
        if lr_scheduler_params is None:
            lr_scheduler_params = {}

        # Create the learning rate schedule
        if lr_scheduler == "constant":
            schedule = optax.constant_schedule(learning_rate)
        elif lr_scheduler == "exponential":
            schedule = optax.exponential_decay(
                init_value=learning_rate,
                transition_steps=lr_scheduler_params.get("transition_steps", epochs),
                decay_rate=lr_scheduler_params.get("decay_rate", 0.9),
            )
        elif lr_scheduler == "cosine":
            schedule = optax.cosine_decay_schedule(
                init_value=learning_rate,
                decay_steps=lr_scheduler_params.get("decay_steps", epochs),
                alpha=lr_scheduler_params.get("alpha", 0.0),
            )
        elif lr_scheduler == "linear_warmup":
            # Combine warmup with another schedule (e.g., cosine)
            warmup_steps = lr_scheduler_params.get("warmup_steps", epochs // 10)
            target_schedule = (
                optax.cosine_decay_schedule(  # Default to cosine after warmup
                    init_value=learning_rate,
                    decay_steps=epochs - warmup_steps,
                    alpha=lr_scheduler_params.get("alpha", 0.0),
                )
            )
            schedule = optax.join_schedules(
                [
                    optax.linear_schedule(0.0, learning_rate, warmup_steps),
                    target_schedule,
                ],
                [warmup_steps],
            )
        else:
            raise ValueError(f"Unknown lr_scheduler: {lr_scheduler}")

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

    # Initialize Graph Pool for training if using pool
    if init_pool is None:
        if wiring_mode != "fixed":
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
        )
    else:
        # Use the provided pool
        # log.info("Using provided pool for training")
        circuit_pool = init_pool

    # Function to run a circuit and calculate loss
    def get_loss_from_wires_logits(logits, wires, x, y_target, loss_type: str):
        # Run circuit and calculate loss
        acts = run_circuit(logits, wires, x)
        pred = acts[-1]
        acts_hard = run_circuit(logits, wires, x, hard=True)
        pred_hard = acts_hard[-1]

        if loss_type == "bce":
            loss = binary_cross_entropy(pred, y_target)
            hard_loss = binary_cross_entropy(pred_hard, y_target)
        elif loss_type == "l4":
            res = pred - y_target
            hard_res = pred_hard - y_target
            loss = res2loss(res, power=4)
            hard_loss = res2loss(hard_res, power=4)
        elif loss_type == "l2":
            res = pred - y_target
            hard_res = pred_hard - y_target
            loss = res2loss(res, power=2)
            hard_loss = res2loss(hard_res, power=2)
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}")

        accuracy = compute_accuracy(pred, y_target)
        hard_accuracy = compute_accuracy(pred_hard, y_target)

        return loss, (hard_loss, pred, pred_hard, accuracy, hard_accuracy)

    # Define pool-based training step
    @partial(
        nnx.jit,
        static_argnames=(
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
        n_message_steps: int,
        loss_type: str,
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
            n_message_steps: Number of message passing steps
            loss_type: Type of loss function to use

        Returns:
            Tuple of (loss, auxiliary outputs, updated pool)
        """

        # Define loss function
        def loss_fn(gnn_model, graph, logits, wires):
            # Store original shapes for reconstruction
            logits_original_shapes = [logit.shape for logit in logits]

            # Run GNN for n_message_steps to optimize the circuit
            for step in range(n_message_steps):
                graph = gnn_model(graph)

            # Extract updated logits and run the circuit
            updated_logits = extract_logits_from_graph(graph, logits_original_shapes)
            loss, aux = get_loss_from_wires_logits(
                updated_logits, wires, x, y_target, loss_type
            )

            # Update the graph with the initial loss and current update_steps
            current_update_steps = graph.globals[..., 1]
            final_graph = graph._replace(
                globals=jp.array(
                    [loss, current_update_steps + n_message_steps], dtype=jp.float32
                )
            )

            return loss, (aux, final_graph, updated_logits)

        def batch_loss_fn(model, graphs, logits, wires):
            loss, (aux, updated_graphs, updated_logits) = nnx.vmap(
                loss_fn, in_axes=(None, 0, 0, 0)
            )(model, graphs, logits, wires)
            return jp.mean(loss), (
                jax.tree.map(lambda x: jp.mean(x, axis=0), aux),
                updated_graphs,
                updated_logits,
            )

        # Compute loss and gradients
        (loss, (aux, updated_graphs, updated_logits)), grads = nnx.value_and_grad(
            batch_loss_fn, has_aux=True
        )(model, graphs, logits, wires)

        # Update GNN parameters
        optimizer.update(grads)

        # Update pool with the updated graphs and logits (wires stay the same)
        updated_pool = pool.update(idxs, updated_graphs, batch_of_logits=updated_logits)

        return loss, aux, updated_pool

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

    result = {}
    # Training loop
    for epoch in pbar:
        # Each epoch uses a different random key
        rng, epoch_key = jax.random.split(rng)

        # Pool-based training
        # Sample a batch from the pool
        rng, sample_key = jax.random.split(epoch_key)
        idxs, graphs, wires, logits = circuit_pool.sample(sample_key, meta_batch_size)

        # Select a random subset of data for this epoch
        rng, data_key = jax.random.split(rng)
        # idx = jax.random.permutation(data_key, len(x_data))
        # x_batch = x_data[idx]
        # y_batch = y_data[idx]
        x_batch = x_data
        y_batch = y_data

        # Perform pool training step
        (
            loss,
            (hard_loss, y_pred, y_hard_pred, accuracy, hard_accuracy),
            circuit_pool,
        ) = pool_train_step(
            model,
            optimizer,
            circuit_pool,
            idxs,
            graphs,
            wires,
            logits,
            x_batch,
            y_batch,
            n_message_steps,
            loss_type=loss_type,
        )

        # Reset a fraction of the pool periodically
        if (
            epoch > 0
            and reset_pool_interval is not None
            and epoch % reset_pool_interval == 0
        ):
            rng, reset_key, fresh_key = jax.random.split(rng, 3)

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

        if jp.isnan(loss):
            log.warning(f"Loss is NaN at epoch {epoch}, returning last stable state")
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

            # Get current metric value for best model tracking
            current_metric_value = None
            if best_metric == "loss":
                current_metric_value = float(loss)
            elif best_metric == "hard_loss":
                current_metric_value = float(hard_loss)
            elif best_metric == "accuracy":
                current_metric_value = float(accuracy)
            elif best_metric == "hard_accuracy":
                current_metric_value = float(hard_accuracy)
            else:
                raise ValueError(f"Unknown best_metric: {best_metric}")

            # Log to wandb if enabled
            metrics_dict = {
                "training/epoch": epoch,
                "training/loss": float(loss),
                "training/hard_loss": float(hard_loss),
                "training/accuracy": float(accuracy),
                "training/hard_accuracy": float(hard_accuracy),
                "pool/reset_steps": float(avg_steps_reset),
                "pool/avg_update_steps": float(circuit_pool.get_average_update_steps()),
            }

            # Add learning rate if available
            try:
                metrics_dict["training/lr"] = optimizer.opt_fn.learning_rate.get_count()
            except (AttributeError, TypeError):
                pass  # Skip if learning rate not available

            _log_to_wandb(wandb_run, metrics_dict, epoch, log_interval)

            # Update progress bar with current metrics
            pbar.set_postfix(
                {
                    "Loss": f"{loss:.4f}",
                    "Accuracy": f"{accuracy:.4f}",
                    "Hard Acc": f"{hard_accuracy:.4f}",
                    "Message Steps": f"{n_message_steps}",
                    "Reset Steps": f"{int(avg_steps_reset)}",
                }
            )

            # Check if this is the best model based on the specified metric
            is_best = False
            if "accuracy" in best_metric:  # For accuracy metrics, higher is better
                if current_metric_value > best_metric_value:
                    best_metric_value = current_metric_value
                    is_best = True
            else:  # For loss metrics, lower is better
                if current_metric_value < best_metric_value:
                    best_metric_value = current_metric_value
                    is_best = True

            # Save periodic checkpoint if needed
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
                best_metric,
                current_metric_value,
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

    # Log final results to wandb
    _log_final_wandb_metrics(wandb_run, result, epochs)

    return result
