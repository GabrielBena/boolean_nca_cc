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
    get_current_message_steps_and_batch_size,
    get_step_beta,
)

from boolean_nca_cc.circuits.model import gen_circuit
from boolean_nca_cc.training.pool import GraphPool, initialize_graph_pool
from boolean_nca_cc.training.evaluation import (
    evaluate_model_stepwise,
    evaluate_model_stepwise_batched,
    get_loss_and_update_graph,
)

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


def _run_periodic_evaluation(
    model,
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
    wiring_mode: str = "random",
    eval_batch_size: int = 16,
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
        wiring_mode: Wiring mode ("fixed" or "random")
        eval_batch_size: Batch size for evaluation (used only in random mode)

    Returns:
        Dictionary with evaluation metrics
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

        return step_metrics

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
    # Reset interval scheduling
    reset_interval_schedule: Dict = None,  # Configuration dict for scheduling reset intervals with keys:
    # Message steps scheduling (curriculum learning)
    message_steps_schedule: Dict = None,  # Configuration dict for scheduling message passing steps with keys:
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
    save_stable_states: bool = True,
    # Periodic evaluation parameters
    periodic_eval_enabled: bool = False,
    periodic_eval_interval: int = 100,
    periodic_eval_test_seed: int = 42,
    periodic_eval_log_stepwise: bool = False,
    periodic_eval_batch_size: int = 16,  # Batch size for random wiring evaluation
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
        wiring_mode: Mode for circuit wirings ('fixed' or 'random')
        meta_batch_size: Batch size for training
        pool_size: Size of the graph pool
        reset_pool_fraction: Fraction of pool to reset periodically
        reset_pool_interval: Number of epochs between pool resets
        reset_strategy: Strategy for selecting graphs to reset ("uniform", "steps_biased", "loss_biased", or "combined")
        combined_weights: Tuple of weights (loss_weight, steps_weight) for combining factors in "combined" strategy
        reset_interval_schedule: Configuration dict for scheduling reset intervals with keys:
            - enabled: Whether to use scheduling
            - type: Schedule type ("constant", "linear", "exponential")
            - initial_interval: Starting interval (frequent resets early)
            - final_interval: Ending interval (rare resets late)
            - decay_rate: For exponential decay
            - transition_epochs: For linear decay (None = use total epochs)
        message_steps_schedule: Configuration dict for scheduling message passing steps with keys:
            - enabled: Whether to use message steps scheduling (curriculum learning)
            - type: Schedule type ("constant", "linear", "exponential", "step")
            - initial_steps: Starting number of message steps (should be small for easy gradients)
            - final_steps: Final number of message steps
            - constant_product: Product of (meta_batch_size * n_message_steps) that should be maintained
            - transition_epochs: For linear/step schedules (None = use total epochs)
            - growth_rate: For exponential growth
            - step_intervals: For step schedule - list of epochs where to increase steps
            - step_values: For step schedule - list of step values
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
        periodic_eval_enabled: Whether to enable periodic evaluation
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

    # Initialize reset interval scheduler if provided
    if reset_interval_schedule is None:
        reset_interval_schedule = {"enabled": False}

    # Track last reset epoch for scheduling
    last_reset_epoch = -1  # Initialize to -1 so first check works correctly

    # Initialize test circuit for periodic evaluation if enabled
    test_wires = None
    test_logits = None
    if periodic_eval_enabled:
        test_rng = jax.random.PRNGKey(periodic_eval_test_seed)

        if wiring_mode == "fixed":
            # For fixed wiring, generate a single test circuit
            test_wires, test_logits = gen_circuit(test_rng, layer_sizes, arity=arity)
            log.info(
                "Single test circuit initialized for periodic evaluation (fixed wiring)"
            )
        else:  # wiring_mode == "random"
            # For random wiring, generate a batch of test circuits for batched evaluation
            test_rngs = jax.random.split(test_rng, periodic_eval_batch_size)

            # Use vmap to generate multiple circuits
            vmap_gen_circuit = jax.vmap(
                lambda rng: gen_circuit(rng, layer_sizes, arity=arity)
            )
            test_wires, test_logits = vmap_gen_circuit(test_rngs)

            log.info(
                f"Batch of {periodic_eval_batch_size} test circuits initialized for periodic evaluation (random wiring)"
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

    result = {}
    # Training loop
    for epoch in pbar:
        # Get current message steps and batch size using scheduler
        if message_steps_schedule is None:
            message_steps_schedule = {"enabled": False}

        # Calculate constant product for memory constraint

        current_n_message_steps, current_meta_batch_size = (
            get_current_message_steps_and_batch_size(
                epoch=epoch,
                schedule_config=message_steps_schedule,
                total_epochs=epochs,
                base_steps=n_message_steps,
                base_batch_size=meta_batch_size,
            )
        )

        # Pool-based training
        # Sample a batch from the pool using the current (potentially dynamic) batch size
        rng, sample_key, loss_key = jax.random.split(rng, 3)
        idxs, graphs, wires, logits = circuit_pool.sample(
            sample_key, current_meta_batch_size
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
            current_n_message_steps,
            loss_type=loss_type,
            loss_key=loss_key,
            epoch=epoch,
        )

        *_, hard_loss, _, _, accuracy, hard_accuracy, _, _ = aux

        # Reset a fraction of the pool using scheduled intervals
        current_reset_interval = get_current_reset_interval(
            epoch, reset_interval_schedule, epochs, reset_pool_interval
        )

        if should_reset_pool(epoch, current_reset_interval, last_reset_epoch):
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

            # Update last reset epoch
            last_reset_epoch = epoch

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
                "scheduler/message_steps": current_n_message_steps,
                "scheduler/batch_size": current_meta_batch_size,
                "scheduler/pool_reset_interval": current_reset_interval,
                "pool/reset_steps": float(avg_steps_reset),
                "pool/avg_update_steps": float(circuit_pool.get_average_update_steps()),
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
                    "Msg Steps": f"{current_n_message_steps}",
                    "Batch Size": f"{current_meta_batch_size}",
                    "Reset Steps": f"{avg_steps_reset:.2f}",
                    "Loss Steps": f"{loss_steps:.2f}",
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
                    best_metric,
                    current_metric_value,
                    wandb_run,
                )

            # Run periodic evaluation if enabled
            if (
                periodic_eval_enabled
                and test_wires is not None
                and test_logits is not None
                and epoch > 0
                and epoch % periodic_eval_interval == 0
            ):
                _run_periodic_evaluation(
                    model=model,
                    test_wires=test_wires,
                    test_logits=test_logits,
                    x_data=x_data,
                    y_data=y_data,
                    input_n=input_n,
                    arity=arity,
                    hidden_dim=hidden_dim,
                    n_message_steps=current_n_message_steps,  # Use current message steps
                    loss_type=loss_type,
                    epoch=epoch,
                    wandb_run=wandb_run,
                    log_stepwise=periodic_eval_log_stepwise,
                    layer_sizes=layer_sizes,
                    wiring_mode=wiring_mode,
                    eval_batch_size=periodic_eval_batch_size,
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
