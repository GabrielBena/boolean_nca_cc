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


from boolean_nca_cc.circuits.model import gen_circuit
from boolean_nca_cc.models import CircuitSelfAttention
# from boolean_nca_cc.training.utils import save_checkpoint

from boolean_nca_cc.training.schedulers import (
    should_reset_pool,
    get_learning_rate_schedule,
)

from boolean_nca_cc.training.pool.pool import GraphPool, initialize_graph_pool
from boolean_nca_cc.training.evaluation import (
    evaluate_model_stepwise_batched,
    get_loss_and_update_graph, evaluate_circuits_in_chunks
)
# Removed unused knockout dataset imports since we now use vocabulary-based evaluation
import wandb

from boolean_nca_cc.utils.graph_builder import build_graph
from boolean_nca_cc.training.pool.structural_perturbation import    (
    create_reproducible_knockout_pattern, 
    create_knockout_vocabulary
)
from functools import partial

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
        metric_source: Source of the metric ('training', 'eval', or 'eval_ko')
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
    elif metric_source == "eval_ko":
        if eval_metrics is None:
            raise ValueError("Knockout evaluation metrics not available for eval_ko source")
        # Map to knockout evaluation metric keys
        eval_key_map = {
            "loss": "eval_ko_in/final_loss",
            "hard_loss": "eval_ko_in/final_hard_loss",
            "accuracy": "eval_ko_in/final_accuracy",
            "hard_accuracy": "eval_ko_in/final_hard_accuracy",
        }
        return eval_metrics[eval_key_map[metric_name]]
    else:
        raise ValueError(f"Unknown metric source: {metric_source}")

def run_knockout_periodic_evaluation(
    model,
    knockout_vocabulary: Optional[jp.ndarray],
    base_wires: PyTree,
    base_logits: PyTree,
    knockout_config: Dict,
    periodic_eval_test_seed: int,
    x_data,
    y_data,
    input_n,
    arity,
    circuit_hidden_dim,
    n_message_steps,
    loss_type,
    epoch,
    wandb_run,
    eval_batch_size: int,
    accumulated_pattern_data: List,  # Add this parameter
    log_stepwise=False,
    layer_sizes: Optional[List[Tuple[int, int]]] = None,
    use_scan: bool = False,
    knockout_diversity: Optional[int] = None,  # Add diversity parameter for color coding
) -> Tuple[Dict, List]:  # Return both results and updated accumulated data
    """
    Run periodic evaluation on circuits with persistent knockouts using vocabulary-based sampling.
    
    Args:
        model: The trained model to evaluate
        knockout_vocabulary: Optional vocabulary of knockout patterns
        base_wires: Base circuit wires
        base_logits: Base circuit logits
        knockout_config: Configuration for knockout evaluation
        periodic_eval_test_seed: Seed for evaluation
        x_data: Input data
        y_data: Target data
        input_n: Number of inputs
        arity: Circuit arity
        circuit_hidden_dim: Hidden dimension for circuit
        n_message_steps: Number of message passing steps
        loss_type: Type of loss function
        epoch: Current training epoch
        wandb_run: WandB run object for logging
        eval_batch_size: Batch size for evaluation
        accumulated_pattern_data: List to accumulate pattern performance data
        log_stepwise: Whether to log step-by-step metrics
        layer_sizes: Layer sizes for the circuit
        use_scan: Whether to use scan for message passing
        knockout_diversity: Diversity parameter for color coding scatter plots
        
    Returns:
        Tuple of (evaluation results, updated accumulated pattern data)
    """
    try:
        
        # Use the layer_sizes parameter that's already passed to the function
        # This is in the correct format (total_gates, group_size) from generate_layer_sizes
        true_layer_sizes = layer_sizes if layer_sizes is not None else []
        
        # 1. Sample IN-distribution knockout patterns from vocabulary
        if knockout_vocabulary is not None:
            log.info(f"Running IN-distribution Knockout evaluation using vocabulary ({eval_batch_size} patterns)...")
            
            # Sample patterns from vocabulary with replacement
            id_rng = jax.random.PRNGKey(periodic_eval_test_seed)
            pattern_indices = jax.random.choice(
                id_rng, len(knockout_vocabulary), shape=(eval_batch_size,), replace=True
            )
            in_knockout_patterns = knockout_vocabulary[pattern_indices]
        else:
            log.info(f"Running IN-distribution Knockout evaluation with fresh patterns ({eval_batch_size} patterns)...")
            
            # Generate patterns using same logic as vocabulary but with eval seed
            pattern_creator_fn = partial(
                create_reproducible_knockout_pattern,
                layer_sizes=true_layer_sizes,
                damage_prob=knockout_config["damage_prob"],
            )
            
            id_rng = jax.random.PRNGKey(periodic_eval_test_seed)
            in_pattern_keys = jax.random.split(id_rng, eval_batch_size)
            in_knockout_patterns = jax.vmap(pattern_creator_fn)(in_pattern_keys)
        
        # Replicate base circuit for the batch
        in_wires = jax.tree.map(
            lambda x: jp.repeat(x[None, ...], eval_batch_size, axis=0), base_wires
        )
        in_logits = jax.tree.map(
            lambda x: jp.repeat(x[None, ...], eval_batch_size, axis=0), base_logits
        )
        
        step_metrics_in = evaluate_circuits_in_chunks(
            eval_fn=evaluate_model_stepwise_batched,
            wires=in_wires,
            logits=in_logits,
            knockout_patterns=in_knockout_patterns,
            target_chunk_size=eval_batch_size,
            model=model,
            x_data=x_data,
            y_data=y_data,
            input_n=input_n,
            arity=arity,
            circuit_hidden_dim=circuit_hidden_dim,
            n_message_steps=n_message_steps,
            loss_type=loss_type,
            layer_sizes=layer_sizes,
            return_per_pattern=True,  # Enable per-pattern analysis
            # use_scan=use_scan,
        )

        final_metrics_in = {
            "eval_ko_in/final_loss": step_metrics_in["soft_loss"][-1],
            "eval_ko_in/final_hard_loss": step_metrics_in["hard_loss"][-1],
            "eval_ko_in/final_accuracy": step_metrics_in["soft_accuracy"][-1],
            "eval_ko_in/final_hard_accuracy": step_metrics_in["hard_accuracy"][-1],
            "eval_ko_in/epoch": epoch,
        }

        # 2. Generate OUT-of-distribution knockout patterns (always fresh, different seed)
        log.info(f"Running OUT-of-distribution Knockout evaluation with fresh patterns ({eval_batch_size} patterns)...")
        
        pattern_creator_fn = partial(
            create_reproducible_knockout_pattern,
            layer_sizes=true_layer_sizes,
            damage_prob=knockout_config["damage_prob"],
        )
        
        # Use different seed for OOD patterns
        ood_rng = jax.random.PRNGKey(periodic_eval_test_seed + 1)
        out_pattern_keys = jax.random.split(ood_rng, eval_batch_size)
        out_knockout_patterns = jax.vmap(pattern_creator_fn)(out_pattern_keys)
        
        # --- BEGIN DEBUG CHECKS ---

        # Check how many ID patterns are in the vocabulary (should be all of them)
        if knockout_vocabulary is not None and in_knockout_patterns.size > 0:
            id_in_vocab_count = 0
            for i in range(len(in_knockout_patterns)):
                id_pattern = in_knockout_patterns[i]
                is_in_vocab = jp.any(jp.all(id_pattern == knockout_vocabulary, axis=1))
                if is_in_vocab:
                    id_in_vocab_count += 1
            log.info(f"DEBUG: Number of ID patterns found in vocabulary: {id_in_vocab_count}/{len(in_knockout_patterns)}")
            if wandb_run:
                wandb_run.log({"debug/id_patterns_in_vocab_count": id_in_vocab_count})
                wandb_run.log({"debug/id_patterns_total": len(in_knockout_patterns)})

        # Check how many OOD patterns are in the vocabulary (should be 0)
        if knockout_vocabulary is not None and out_knockout_patterns.size > 0:
            # Check each OOD pattern against the vocabulary
            ood_in_vocab_count = 0
            for i in range(len(out_knockout_patterns)):
                ood_pattern = out_knockout_patterns[i]
                is_in_vocab = jp.any(jp.all(ood_pattern == knockout_vocabulary, axis=1))
                if is_in_vocab:
                    ood_in_vocab_count += 1
            
            log.info(f"DEBUG: Number of OOD patterns found in vocabulary: {ood_in_vocab_count}/{len(out_knockout_patterns)}")
            if wandb_run:
                wandb_run.log({"debug/ood_patterns_in_vocab_count": ood_in_vocab_count})
                wandb_run.log({"debug/ood_patterns_total": len(out_knockout_patterns)})


        # --- END DEBUG CHECKS ---
        
        # Replicate base circuit for the batch
        out_wires = jax.tree.map(
            lambda x: jp.repeat(x[None, ...], eval_batch_size, axis=0), base_wires
        )
        out_logits = jax.tree.map(
            lambda x: jp.repeat(x[None, ...], eval_batch_size, axis=0), base_logits
        )
        
        step_metrics_out = evaluate_circuits_in_chunks(
            eval_fn=evaluate_model_stepwise_batched,
            wires=out_wires,
            logits=out_logits,
            knockout_patterns=out_knockout_patterns,
            target_chunk_size=eval_batch_size,
            model=model,
            x_data=x_data,
            y_data=y_data,
            input_n=input_n,
            arity=arity,
            circuit_hidden_dim=circuit_hidden_dim,
            n_message_steps=n_message_steps,
            loss_type=loss_type,
            layer_sizes=layer_sizes,
            return_per_pattern=True,  # Enable per-pattern analysis
            # use_scan=use_scan,
        )

        final_metrics_out = {
            "eval_ko_out/final_loss": step_metrics_out["soft_loss"][-1],
            "eval_ko_out/final_hard_loss": step_metrics_out["hard_loss"][-1],
            "eval_ko_out/final_accuracy": step_metrics_out["soft_accuracy"][-1],
            "eval_ko_out/final_hard_accuracy": step_metrics_out["hard_accuracy"][-1],
            "eval_ko_out/epoch": epoch,
        }

        # Log main metrics normally (these will create panels)
        main_metrics = {**final_metrics_in, **final_metrics_out}
        
        if wandb_run:
            wandb_run.log(main_metrics)
            
            # Add new pattern data to accumulated data for persistent scatter plot
            if "per_pattern" in step_metrics_in:
                final_hard_accuracies_in = step_metrics_in["per_pattern"]["pattern_hard_accuracies"][-1]
                for pattern_idx, hard_acc in enumerate(final_hard_accuracies_in):
                    # Include knockout_diversity in data point for color coding
                    diversity_value = knockout_diversity if knockout_diversity is not None else 0
                    data_point = [int(epoch), pattern_idx, float(hard_acc), diversity_value]
                    accumulated_pattern_data.append(data_point)
            
            # Log persistent scatter plot with all accumulated data
            if accumulated_pattern_data:
                
                pattern_table = wandb.Table(
                    data=accumulated_pattern_data,
                    columns=["epoch", "pattern_id", "hard_accuracy", "knockout_diversity"]
                )
                
                # Create scatter plot with all accumulated points
                # Note: Color coding by diversity will be handled by wandb's automatic grouping
                # based on the knockout_diversity column in the table
                scatter_plot = wandb.plot.scatter(
                    pattern_table, 
                    "epoch", 
                    "hard_accuracy",
                    title="In-Distribution Pattern Performance by Epoch (Colored by Knockout Diversity)"
                )
                
                wandb_run.log({"pattern_performance_scatter": scatter_plot})

            if log_stepwise:
                for step_idx in range(len(step_metrics_in["step"])):
                    wandb_run.log({
                        "eval_ko_in_steps/step": step_metrics_in["step"][step_idx],
                        "eval_ko_in_steps/loss": step_metrics_in["soft_loss"][step_idx],
                        "eval_ko_in_steps/hard_loss": step_metrics_in["hard_loss"][step_idx],
                        "eval_ko_in_steps/accuracy": step_metrics_in["soft_accuracy"][step_idx],
                        "eval_ko_in_steps/hard_accuracy": step_metrics_in["hard_accuracy"][step_idx],
                        "eval_ko_in_steps/epoch": epoch,
                    })
                for step_idx in range(len(step_metrics_out["step"])):
                    wandb_run.log({
                        "eval_ko_out_steps/step": step_metrics_out["step"][step_idx],
                        "eval_ko_out_steps/loss": step_metrics_out["soft_loss"][step_idx],
                        "eval_ko_out_steps/hard_loss": step_metrics_out["hard_loss"][step_idx],
                        "eval_ko_out_steps/accuracy": step_metrics_out["soft_accuracy"][step_idx],
                        "eval_ko_out_steps/hard_accuracy": step_metrics_out["hard_accuracy"][step_idx],
                        "eval_ko_out_steps/epoch": epoch,
                    })

        log.info(
            f"Knockout Eval (epoch {epoch}):\n"
            f"  IN-distribution KO: Loss={final_metrics_in['eval_ko_in/final_loss']:.4f}, "
            f"Acc={final_metrics_in['eval_ko_in/final_accuracy']:.4f}, "
            f"Hard Acc={final_metrics_in['eval_ko_in/final_hard_accuracy']:.4f}\n"
            f"  OUT-of-distribution KO: Loss={final_metrics_out['eval_ko_out/final_loss']:.4f}, "
            f"Acc={final_metrics_out['eval_ko_out/final_accuracy']:.4f}, "
            f"Hard Acc={final_metrics_out['eval_ko_out/final_hard_accuracy']:.4f}"
        )

        return {
            "final_metrics_in": final_metrics_in,
            "final_metrics_out": final_metrics_out,
        }, accumulated_pattern_data  # Return updated accumulated data

    except Exception as e:
        log.warning(f"Error during knockout periodic evaluation at epoch {epoch}: {e}")
        return {}

def train_model(
    # Data parameters
    x_data: jp.ndarray,
    y_data: jp.ndarray,
    layer_sizes: List[Tuple[int, int]],
    # Model architecture parameters
    arity: int = 2,
    circuit_hidden_dim: int = 16,
    message_passing: bool = True,
    # node_mlp_features: List[int] = [64, 32],
    # edge_mlp_features: List[int] = [64, 32],
    use_attention: bool = False,
    # Training hyperparameters
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    epochs: int = 100,
    n_message_steps: int = 1,
    use_scan: bool = False,
    # Loss parameters
    loss_type: str = "l4",  # Options: 'l4' or 'bce'
    meta_batch_size: int = 64,
    # Pool parameters
    pool_size: int = 1024,
    reset_pool_fraction: float = 0.05,
    reset_pool_interval: int = 128,
    reset_strategy: str = "uniform",  # Options: "uniform", "steps_biased", "loss_biased", or "combined"
    combined_weights: Tuple[float, float] = (
        0.5,
        0.5,
    ),  # Weights for [loss, steps] in combined strategy
    # Perturbation configurations
    persistent_knockout_config: Optional[Dict] = None,
    knockout_diversity: Optional[int] = None,  # Size of knockout pattern vocabulary for shared training/evaluation patterns
    knockout_eval: Optional[Dict] = None,
    # Learning rate scheduling
    lr_scheduler: str = "constant",  # Options: "constant", "exponential", "cosine", "linear_warmup"
    lr_scheduler_params: Optional[Dict] = None,
    # Initialization parameters
    key: int = 0,
    wiring_fixed_key: jax.random.PRNGKey = jax.random.PRNGKey(
        42
    ),  # Fixed key for generating wirings when wiring_mode='fixed'
    init_model: Optional[CircuitSelfAttention] = None,
    init_optimizer: Optional[nnx.Optimizer] = None,
    initial_metrics: Optional[Dict] = None,
    # Checkpointing parameters
    # Periodic evaluation parameters
    periodic_eval_enabled: bool = False,
    periodic_eval_inner_steps: int = 100,
    periodic_eval_interval: int = 1024,
    periodic_eval_test_seed: int = 42,
    periodic_eval_log_stepwise: bool = False,
    periodic_eval_batch_size: int = 16,  # Batch size for random wiring evaluation
    # Wandb parameters
    wandb_logging: bool = False,
    log_interval: int = 1,
    wandb_run_config: Optional[Dict] = None,
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
        # node_mlp_features: Hidden layer sizes for the node MLP
        # edge_mlp_features: Hidden layer sizes for the edge MLP
        # use_attention: Whether to use attention-based message aggregation
        learning_rate: Learning rate for optimization
        epochs: Number of training epochs
        n_message_steps: Number of message passing steps per pool batch
        loss_type: Type of loss to use ('l4' for L4 norm or 'bce' for binary cross-entropy)
        meta_batch_size: Batch size for training
        pool_size: Size of the graph pool
        reset_pool_fraction: Fraction of pool to reset periodically
        reset_pool_interval: Number of epochs between pool resets
        reset_strategy: Strategy for selecting graphs to reset ("uniform", "steps_biased", "loss_biased", or "combined")
        combined_weights: Tuple of weights (loss_weight, steps_weight) for combining factors in "combined" strategy
        persistent_knockout_config: Configuration for persistent knockout perturbations.
        knockout_diversity: Size of knockout pattern vocabulary for shared training/evaluation patterns
        key: Random seed
        wiring_fixed_key: Fixed key for generating wirings when wiring_mode='fixed'
        init_model: Optional pre-trained GNN model to continue training
        init_optimizer: Optional pre-trained optimizer to continue training
        initial_metrics: Optional dictionary of metrics from previous training
        lr_scheduler: Learning rate scheduler type
        lr_scheduler_params: Dictionary of parameters for the scheduler
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
        # # Create a new GNN
        # rng, init_key = jax.random.split(rng)
        # model = CircuitSelfAttention(
        #     # node_mlp_features=node_mlp_features,
        #     # edge_mlp_features=edge_mlp_features,
        #     circuit_hidden_dim=circuit_hidden_dim,
        #     arity=arity,
        #     attention_dim=128,
        #     num_heads=4,
        #     num_layers=3,
        #     mlp_dim=256,
        #     mlp_dim_multiplier=2,
        #     dropout_rate=0.0,
        #     # message_passing=message_passing,
        #     # use_attention=use_attention,
        #     rngs=nnx.Rngs(params=init_key),
        # )
        pass
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
    training_pool_key = wiring_fixed_key

    circuit_pool = initialize_graph_pool(
        rng=training_pool_key,
        layer_sizes=layer_sizes,
        pool_size=pool_size,
        input_n=input_n,
        arity=arity,
        circuit_hidden_dim=circuit_hidden_dim,
        loss_value=0.0,  # Initial loss will be calculated properly in first step
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
        model: CircuitSelfAttention,
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
        knockout_patterns: Optional[jp.ndarray] = None,
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
            knockout_patterns: Optional knockout pattern for structural perturbations.
        Returns:
            Tuple of (loss, auxiliary outputs, updated pool)
        """

        def get_loss_step(loss_key):
            return n_message_steps - 1

        def loss_fn_no_scan(model, graph, logits, wires, loss_key, knockout_pattern):
            # Store original shapes for reconstruction
            logits_original_shapes = [logit.shape for logit in logits]
            loss_step = get_loss_step(loss_key)

            all_results = []

            for i in range(n_message_steps):
                graph = model(graph, knockout_pattern=knockout_pattern)

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

        def batch_loss_fn(model, graphs, logits, wires, loss_key, knockout_patterns):
            loss_fn = loss_fn_no_scan # TODO: add scan as an option

            loss_keys = jax.random.split(loss_key, graphs.n_node.shape[0])
            
            loss, (aux, updated_graphs, updated_logits, loss_steps) = nnx.vmap(
                loss_fn, in_axes=(None, 0, 0, 0, 0, 0)
            )(model, graphs, logits, wires, loss_keys, knockout_patterns)
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
                knockout_patterns=knockout_patterns,
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

    # Track best model
    # best_metric_value = float("-inf") if "accuracy" in best_metric else float("inf")

    # Create progress bar for training
    pbar = tqdm(range(epochs), desc="Training GNN")
    avg_steps_reset = 0

    # Track last reset epoch for scheduling
    last_reset_epoch = -1  # Initialize to -1 so first check works correctly

    # Initialize knockout evaluation if enabled
    knockout_eval_base_circuit = None
    if knockout_eval and knockout_eval.get("enabled"):
        # Store base circuit for on-demand evaluation (no pre-created datasets)
        log.info("Knockout evaluation enabled - will use vocabulary-based evaluation")
        knockout_eval_base_circuit = gen_circuit(wiring_fixed_key, layer_sizes, arity=arity)
        log.info("Base circuit created for knockout evaluation")

    # Initialize knockout vocabulary if knockout_diversity is configured
    knockout_vocabulary = None
    log.info(f"DEBUG: train_loop received knockout_diversity = {knockout_diversity}")
    log.info(f"DEBUG: train_loop received persistent_knockout_config = {persistent_knockout_config}")
    if (persistent_knockout_config and knockout_diversity is not None and 
        knockout_diversity > 0 and persistent_knockout_config.get("fraction", 0.0) > 0.0):
        log.info(f"Creating knockout pattern vocabulary with {knockout_diversity} patterns")
        
        # Use the same seed as evaluation to ensure pattern space is identical
        vocab_rng = jax.random.PRNGKey(periodic_eval_test_seed)
        
        # Use the layer_sizes parameter directly - it's already in the correct format
        # from generate_layer_sizes: (total_gates, group_size)
        true_layer_sizes = layer_sizes
        
        # Generate the shared knockout vocabulary
        knockout_vocabulary = create_knockout_vocabulary(
            rng=vocab_rng,
            vocabulary_size=knockout_diversity,
            layer_sizes=true_layer_sizes,
            damage_prob=persistent_knockout_config.get("damage_prob", 10.0),
        )
        
        log.info(f"Generated knockout vocabulary with shape: {knockout_vocabulary.shape}")

    # Initialize accumulated pattern data for persistent scatter plot
    # Each data point will be: [epoch, pattern_id, hard_accuracy, knockout_diversity]
    accumulated_pattern_data = []

    diversity = 0.0
    result = {}
    # Training loop
    try:
        for epoch in pbar:
            # Pool-based training
            # Sample a batch from the pool using the current (potentially dynamic) batch size
            rng, sample_key, loss_key = jax.random.split(rng, 3)
            idxs, graphs, wires, logits, knockout_patterns = circuit_pool.sample(
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
                knockout_patterns=knockout_patterns,
            )

            *_, hard_loss, _, _, accuracy, hard_accuracy, _, _ = aux

            # Reset a fraction of the pool using scheduled intervals
            if should_reset_pool(epoch, reset_pool_interval, last_reset_epoch):
                rng, reset_key, fresh_key = jax.random.split(rng, 3)

                # Original logic for fixed and random wiring modes
                # Generate fresh circuits for resetting, potentially with knockouts
                num_to_reset = max(1, round(pool_size * reset_pool_fraction))

                # Use consistent key generation for pool resets
                reset_pool_key = wiring_fixed_key

                # Sample knockout patterns from vocabulary if available, otherwise use config
                sampled_knockout_patterns = None
                if knockout_vocabulary is not None:
                    # Sample patterns with replacement from the vocabulary
                    pattern_sample_key = jax.random.fold_in(reset_key, 42)  # Use different key for pattern sampling
                    pattern_indices = jax.random.choice(
                            pattern_sample_key, len(knockout_vocabulary), shape=(num_to_reset,), replace=True
                    )
                    sampled_knockout_patterns = knockout_vocabulary[pattern_indices]


                # Create a pool of fresh circuits, applying knockout patterns
                damaged_pool = initialize_graph_pool(
                    rng=reset_pool_key,
                    layer_sizes=layer_sizes,
                    pool_size=num_to_reset,  # Only create circuits we need
                    input_n=input_n,
                    arity=arity,
                    circuit_hidden_dim=circuit_hidden_dim,
                    knockout_config=persistent_knockout_config if knockout_vocabulary is None else None,  # Use config only if no vocabulary
                    knockout_patterns=sampled_knockout_patterns,  # Pass sampled patterns
                )

                # Reset a fraction of the pool and get avg steps of reset graphs
                circuit_pool, avg_steps_reset = circuit_pool.reset_fraction(
                    reset_key,
                    reset_pool_fraction,
                    damaged_pool.graphs,
                    damaged_pool.wires,
                    damaged_pool.logits,
                    damaged_pool.knockout_patterns,  # Pass patterns
                    reset_strategy=reset_strategy,
                    combined_weights=combined_weights,
                )

                # Update last reset epoch
                last_reset_epoch = epoch
                # diversity = circuit_pool.get_wiring_diversity(layer_sizes)

            if jp.isnan(loss):
                log.warning(
                    f"Loss is NaN at epoch {epoch}, returning last stable state"
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
                    # "pool/wiring_diversity": float(diversity),
                    "pool/reset_steps": float(avg_steps_reset),
                    # "pool/avg_update_steps": float(avg_steps),
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
                postfix_dict = {
                    "Loss": f"{loss:.4f}",
                    "Accuracy": f"{accuracy:.4f}",
                    "Hard Acc": f"{hard_accuracy:.4f}",
                    # "Diversity": f"{diversity:.3f}",
                    "Reset Steps": f"{avg_steps_reset:.2f}",
                    "Loss Steps": f"{loss_steps:.2f}",
                }

                pbar.set_postfix(postfix_dict)

                # Step 2: Run knockout evaluation if enabled
                all_eval_metrics = {}
                if (
                    knockout_eval
                    and knockout_eval.get("enabled")
                    and knockout_eval_base_circuit is not None
                    and epoch % periodic_eval_interval == 0
                ):
                    base_wires, base_logits = knockout_eval_base_circuit

                    # Pass vocabulary and wandb_run for detailed checks
                    # Pass accumulated data and get updated version back
                    ko_eval_results, accumulated_pattern_data = run_knockout_periodic_evaluation(
                        model=model,
                        knockout_vocabulary=knockout_vocabulary,
                        base_wires=base_wires,
                        base_logits=base_logits,
                        knockout_config=knockout_eval,
                        periodic_eval_test_seed=periodic_eval_test_seed,
                        x_data=x_data,
                        y_data=y_data,
                        input_n=input_n,
                        arity=arity,
                        circuit_hidden_dim=circuit_hidden_dim,
                        n_message_steps=periodic_eval_inner_steps,
                        loss_type=loss_type,
                        epoch=epoch,
                        wandb_run=wandb_run,
                        eval_batch_size=periodic_eval_batch_size,
                        accumulated_pattern_data=accumulated_pattern_data,  # Pass accumulated data
                        log_stepwise=periodic_eval_log_stepwise,
                        layer_sizes=layer_sizes,
                        use_scan=use_scan,
                        knockout_diversity=knockout_diversity,  # Pass diversity for color coding
                    )
                    # Extract final metrics for best model tracking
                    if ko_eval_results and "final_metrics_in" in ko_eval_results:
                        all_eval_metrics.update(ko_eval_results["final_metrics_in"])
                        all_eval_metrics.update(ko_eval_results["final_metrics_out"])

                # Set current eval metrics to the combined dictionary if any evals ran
                current_eval_metrics = all_eval_metrics if all_eval_metrics else None

                # Return the trained GNN model and metrics
                result = {
                    "model": model,
                    "optimizer": optimizer,
                    "losses": losses,
                    "hard_losses": hard_losses,
                    "accuracies": accuracies,
                    "hard_accuracies": hard_accuracies,
                    "reset_steps": reset_steps,
                    # "best_metric_value": best_metric_value,
                    # "best_metric": best_metric,
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