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
from typing import List, Tuple, Dict, Any
from functools import partial
from tqdm.auto import tqdm

from boolean_nca_cc.models import CircuitGNN, run_gnn_scan
from boolean_nca_cc.utils import build_graph, extract_logits_from_graph
from boolean_nca_cc.circuits.train import (
    res2loss,
    binary_cross_entropy,
    compute_accuracy,
)
from boolean_nca_cc.circuits.model import gen_circuit, run_circuit
from boolean_nca_cc.training.pool import GraphPool, initialize_graph_pool

# Type alias for PyTree
PyTree = Any


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


def train_gnn(
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
    n_message_steps: int = 100,
    loss_type: str = "l4",  # Options: 'l4' or 'bce'
    # Meta-learning parameters
    meta_learning: bool = False,
    meta_batch_size: int = 64,
    # Pool parameters
    use_pool: bool = False,
    pool_size: int = 1024,
    reset_pool_fraction: float = 0.05,
    reset_pool_interval: int = 10,
    # Learning rate scheduling
    lr_scheduler: str = "constant",  # Options: "constant", "exponential", "cosine", "linear_warmup"
    lr_scheduler_params: Dict = None,
    # Initialization parameters
    key: int = 0,
    init_gnn: CircuitGNN = None,
    init_optimizer: nnx.Optimizer = None,
    init_pool: GraphPool = None,
    initial_metrics: Dict = None,
    # Curriculum parameters
    message_steps_schedule: Dict = None,
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
        n_message_steps: Number of message passing steps per epoch
        key: Random seed
        weight_decay: Weight decay for optimizer
        meta_learning: Whether to use meta-learning (train on new random circuits each step)
        meta_batch_size: Batch size for meta-learning
        use_pool: Whether to use graph pool for training instead of generating new circuits
        pool_size: Size of the graph pool if use_pool is True
        reset_pool_fraction: Fraction of pool to reset periodically
        reset_pool_interval: Number of epochs between pool resets
        init_gnn: Optional pre-trained GNN model to continue training
        init_optimizer: Optional pre-trained optimizer to continue training
        init_pool: Optional pre-initialized GraphPool to continue training with
        initial_metrics: Optional dictionary of metrics from previous training
        lr_scheduler: Learning rate scheduler type. Default: "constant".
        lr_scheduler_params: Dictionary of parameters for the scheduler.
        loss_type: Type of loss to use ('l4' for L4 norm or 'bce' for binary cross-entropy).
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
    else:
        # Continue from previous metrics
        losses = list(initial_metrics.get("losses", []))
        accuracies = list(initial_metrics.get("accuracies", []))
        hard_losses = list(initial_metrics.get("hard_losses", []))
        hard_accuracies = list(initial_metrics.get("hard_accuracies", []))

    # Initialize or reuse GNN
    if init_gnn is None:
        # Create a new GNN
        rng, init_key = jax.random.split(rng)
        gnn = CircuitGNN(
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
        gnn = init_gnn

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
        optimizer = nnx.Optimizer(
            gnn, optax.adamw(learning_rate=schedule, weight_decay=weight_decay)
        )
    else:
        # Use the provided optimizer
        optimizer = init_optimizer

    # Initialize Graph Pool for training if using pool
    if use_pool:
        if init_pool is None:
            rng, pool_key = jax.random.split(rng)
            # Initialize a fresh pool
            circuit_pool = initialize_graph_pool(
                rng=pool_key,
                layer_sizes=layer_sizes,
                pool_size=pool_size,
                input_n=input_n,
                arity=arity,
                hidden_dim=hidden_dim,
                loss_value=0.0,  # Initial loss will be calculated properly in first step
            )
        else:
            # Use the provided pool
            circuit_pool = init_pool

    # Function to run a circuit and calculate loss
    # @partial(nnx.jit, static_argnames=("loss_type",))
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
        gnn: CircuitGNN,
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
            gnn: CircuitGNN model
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

            # Calculate initial loss and update graphs
            initial_loss, _ = get_loss_from_wires_logits(
                logits, wires, x, y_target, loss_type
            )
            updated_graph = graph._replace(globals=initial_loss)

            # Run GNN for n_message_steps to optimize the circuit
            for _ in range(n_message_steps):
                updated_graph = gnn_model(updated_graph)

            # Extract updated logits and run the circuit
            updated_logits = extract_logits_from_graph(
                updated_graph, logits_original_shapes
            )
            loss, aux = get_loss_from_wires_logits(
                updated_logits, wires, x, y_target, loss_type
            )

            # Final update with the computed loss
            final_graph = updated_graph._replace(globals=loss)

            return loss, (aux, final_graph, updated_logits)

        def batch_loss_fn(gnn, graphs, logits, wires):
            loss, (aux, updated_graphs, updated_logits) = nnx.vmap(
                loss_fn, in_axes=(None, 0, 0, 0)
            )(gnn, graphs, logits, wires)
            return jp.mean(loss), (
                jax.tree.map(lambda x: jp.mean(x, axis=0), aux),
                updated_graphs,
                updated_logits,
            )

        # Compute loss and gradients
        (loss, (aux, updated_graphs, updated_logits)), grads = nnx.value_and_grad(
            batch_loss_fn, has_aux=True
        )(gnn, graphs, logits, wires)

        # Update GNN parameters
        optimizer.update(grads)

        # Update pool with the updated graphs and logits (wires stay the same)
        updated_pool = pool.update(idxs, updated_graphs, batch_of_logits=updated_logits)

        return loss, aux, updated_pool

    # Define meta-training step function (classic approach generating new circuits)
    @partial(
        nnx.jit,
        static_argnames=(
            "layer_sizes",
            "n_message_steps",
            "loss_type",
            "meta_learning",
        ),
    )
    def meta_train_step(
        gnn: CircuitGNN,
        optimizer: nnx.Optimizer,
        x: jp.ndarray,
        y_target: jp.ndarray,
        rng: jax.random.PRNGKey,
        layer_sizes: List[Tuple[int, int]],
        n_message_steps: int,
        loss_type: str,  # Pass loss_type
        meta_learning: bool,
    ):
        """
        Single meta-training step with randomly sampled circuit wirings.

        Args:
            gnn: CircuitGNN model
            optimizer: nnx Optimizer
            x: Input data
            y_target: Target output data
            rng: Random key
            layer_sizes: Circuit layer sizes
            n_message_steps: Number of message passing steps
            loss_type: Type of loss function to use

        Returns:
            Tuple of (loss, (hard_loss, accuracy, hard_accuracy))
        """

        # Internal loss function for a single circuit
        @partial(nnx.jit, static_argnames=("loss_type"))
        def loss_fn(
            gnn_model: CircuitGNN,
            rng: jax.random.PRNGKey,
            loss_type: str,
        ):
            # Sample new random circuit wiring
            rng_wires, _ = jax.random.split(rng)

            wires, logits = gen_circuit(rng_wires, layer_sizes, arity=arity)

            # Store original shapes for reconstruction
            logits_original_shapes = [logit.shape for logit in logits]

            # Build graph from the random circuit, passing the initial loss
            initial_loss, _ = get_loss_from_graph(logits, wires, x, y_target, loss_type)
            graph = build_graph(
                logits, wires, input_n, arity, hidden_dim, loss_value=initial_loss
            )

            # Run GNN for n_message_steps to optimize the circuit
            for _ in range(n_message_steps):
                graph = gnn_model(graph)
                # Extract updated logits and run the circuit
                updated_logits = extract_logits_from_graph(
                    graph, logits_original_shapes
                )
                loss, (hard_loss, y_pred, y_hard_pred) = get_loss_from_graph(
                    updated_logits, wires, x, y_target, loss_type
                )

                # Update the loss value for the graph
                graph = graph._replace(globals=loss)

            accuracy = compute_accuracy(y_pred, y_target)
            hard_accuracy = compute_accuracy(y_hard_pred, y_target)

            return loss, (
                hard_loss,
                accuracy,
                hard_accuracy,
            )

        # For meta-learning, average over multiple random circuits
        @partial(nnx.jit, static_argnames=("loss_type", "meta_learning"))
        def mean_batch_loss_fn(gnn, rng, loss_type: str, meta_learning: bool):
            # Create batch of random keys
            if meta_learning:
                batch_rng = jax.random.split(rng, meta_batch_size)
            else:
                batch_rng = jp.full((meta_batch_size,), rng)
            # Use vmap to vectorize loss function over the random keys
            # Pass loss_type to the vmapped function
            batch_loss_fn = nnx.vmap(loss_fn, in_axes=(None, 0, None))
            # Compute losses for each random circuit
            losses, aux = batch_loss_fn(gnn, rng=batch_rng, loss_type=loss_type)
            # Average losses and metrics
            return jp.mean(losses), jax.tree.map(lambda x: jp.stack(x, axis=0), aux)

        # Compute loss and gradients
        # Pass loss_type to mean_batch_loss_fn
        (loss, aux), grads = nnx.value_and_grad(mean_batch_loss_fn, has_aux=True)(
            gnn, rng=rng, loss_type=loss_type, meta_learning=meta_learning
        )

        # Update GNN parameters
        optimizer.update(grads)

        return loss, aux

    if message_steps_schedule is not None:
        schedule_type = message_steps_schedule.get("schedule_type", "staircase")
        schedule_params = message_steps_schedule.get("schedule_params", {})

        if schedule_type == "linear":
            message_steps_schedule = jax.numpy.linspace(
                schedule_params.get("start", 1),
                schedule_params.get("end", n_message_steps),
                epochs,
            )
        elif schedule_type == "exponential":
            start = jax.numpy.log(schedule_params.get("start", 1))
            end = jax.numpy.log(schedule_params.get("end", n_message_steps))
            message_steps_schedule = jax.numpy.exp(
                jax.numpy.linspace(start, end, epochs)
            )
        elif schedule_type == "staircase":
            # Create a schedule with minimal discrete steps to reduce recompilations
            num_steps = schedule_params.get("num_steps", 4)  # Default to 4 steps
            start = schedule_params.get("start", 1)
            end = schedule_params.get("end", n_message_steps)
            # Calculate step sizes to distribute evenly across epochs
            step_sizes = jp.linspace(start, end, num_steps)
            # Create array of indices for each step
            step_indices = jp.linspace(
                0, epochs - 1, num_steps, dtype=jp.int32, endpoint=False
            )

            # Calculate the length of each segment
            segment_lengths = jp.diff(
                jp.concatenate([step_indices, jp.array([epochs])])
            )

            # Create the schedule by repeating each step value for its segment length
            schedule = jp.repeat(jp.round(step_sizes).astype(jp.int32), segment_lengths)

            # Ensure the schedule has exactly 'epochs' elements by padding or truncating
            padding = jp.full(epochs - len(schedule), schedule[-1])
            schedule = jp.concatenate([schedule, padding])[:epochs]

            message_steps_schedule = schedule
        else:
            raise ValueError(f"Unknown schedule_type: {schedule_type}")

        print(f"Message steps schedule: {message_steps_schedule}")

    # Create progress bar for training
    pbar = tqdm(range(epochs), desc="Training GNN")

    # Training loop
    for epoch in pbar:
        # Each epoch uses a different random key
        rng, epoch_key = jax.random.split(rng)

        # Get current message steps from schedule if provided
        current_message_steps = (
            n_message_steps
            if message_steps_schedule is None
            else int(message_steps_schedule[epoch])
        )

        if use_pool:
            # Pool-based training
            # Sample a batch from the pool
            rng, sample_key = jax.random.split(rng)
            idxs, graphs, wires, logits = circuit_pool.sample(
                sample_key, meta_batch_size
            )

            # Select a random subset of data for this epoch
            rng, data_key = jax.random.split(rng)
            idx = jax.random.permutation(data_key, len(x_data))
            x_batch = x_data[idx][:meta_batch_size]
            y_batch = y_data[idx][:meta_batch_size]

            # Perform pool training step
            (
                loss,
                (hard_loss, y_pred, y_hard_pred, accuracy, hard_accuracy),
                circuit_pool,
            ) = pool_train_step(
                gnn,
                optimizer,
                circuit_pool,
                idxs,
                graphs,
                wires,
                logits,
                x_batch,
                y_batch,
                current_message_steps,
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
                fresh_pool = initialize_graph_pool(
                    rng=fresh_key,
                    layer_sizes=layer_sizes,
                    pool_size=pool_size,  # Use same size as circuit_pool
                    input_n=input_n,
                    arity=arity,
                    hidden_dim=hidden_dim,
                )

                # Reset a fraction of the pool
                circuit_pool = circuit_pool.reset_fraction(
                    reset_key,
                    reset_pool_fraction,
                    fresh_pool.graphs,
                    fresh_pool.wires,
                    fresh_pool.logits,
                )

        else:
            # Classic meta-learning approach
            # Select a random subset of data for this epoch
            idx = jax.random.permutation(epoch_key, len(x_data))
            x_batch = x_data[idx]
            y_batch = y_data[idx]

            # Perform meta-learning training step
            loss, aux_stack = meta_train_step(
                gnn,
                optimizer,
                x_batch,
                y_batch,
                epoch_key,
                tuple(layer_sizes),
                current_message_steps,
                loss_type=loss_type,
                meta_learning=meta_learning,
            )

            hard_loss, accuracy, hard_accuracy = jax.tree.map(
                lambda x: jp.mean(x, axis=0), aux_stack
            )

        # Record metrics
        losses.append(float(loss))
        hard_losses.append(float(hard_loss))
        accuracies.append(float(accuracy))
        hard_accuracies.append(float(hard_accuracy))

        # Update progress bar with current metrics
        pbar.set_postfix(
            {
                "Loss": f"{loss:.4f}",
                "Accuracy": f"{accuracy:.4f}",
                "Hard Acc": f"{hard_accuracy:.4f}",
                "Message Steps": f"{current_message_steps}",
            }
        )

    # Return the trained GNN model and metrics
    result = {
        "gnn": nnx.state(gnn),
        "optimizer": nnx.state(optimizer),
        "losses": losses,
        "hard_losses": hard_losses,
        "accuracies": accuracies,
        "hard_accuracies": hard_accuracies,
    }

    # Add pool to result if used
    if use_pool:
        result["pool"] = circuit_pool

    return result
