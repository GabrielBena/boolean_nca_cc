"""
Training loop implementation for GNN-based boolean circuit optimization.

This module provides functions for training GNN models to optimize
boolean circuits over multiple epochs.
"""

import jax
import jax.numpy as jp
import optax
from flax import nnx
from typing import List, Tuple, Dict
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
    # Learning rate scheduling
    lr_scheduler: str = "constant",  # Options: "constant", "exponential", "cosine", "linear_warmup"
    lr_scheduler_params: Dict = None,
    # Initialization parameters
    key: int = 0,
    init_gnn: CircuitGNN = None,
    init_optimizer: nnx.Optimizer = None,
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
        init_gnn: Optional pre-trained GNN model to continue training
        init_optimizer: Optional pre-trained optimizer to continue training
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

    # Define meta-training step function
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

            # all_graphs = []
            # all_losses = []

            # not using scan is much faster for now (weird) ?
            # graph = run_gnn_scan(gnn_model, graph, n_message_steps)

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
                # all_losses.append(loss)

            accuracy = compute_accuracy(y_pred, y_target)
            hard_accuracy = compute_accuracy(y_hard_pred, y_target)

            return loss, (
                hard_loss,
                accuracy,
                hard_accuracy,
                # all_losses,
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
        if meta_learning:
            rng, epoch_key = jax.random.split(rng)
        else:
            epoch_key = rng

        # Get current message steps from schedule if provided
        current_message_steps = (
            n_message_steps
            if message_steps_schedule is None
            else int(message_steps_schedule[epoch])
        )

        # print(f"Current message steps: {current_message_steps}")

        # Select a random subset of data for this epoch
        idx = jax.random.permutation(epoch_key, len(x_data))
        x_batch = x_data[idx]
        y_batch = y_data[idx]

        # Perform training step, passing loss_type
        loss, aux_stack = meta_train_step(
            gnn,
            optimizer,
            x_batch,
            y_batch,
            epoch_key,
            tuple(layer_sizes),
            current_message_steps,
            loss_type=loss_type,  # Pass loss_type here
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
        # all_losses.append(all_losses)
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
    return {
        "gnn": nnx.state(gnn),
        "optimizer": nnx.state(optimizer),
        "losses": losses,
        "hard_losses": hard_losses,
        "accuracies": accuracies,
        "hard_accuracies": hard_accuracies,
        # "all_losses": all_losses,
        # "all_graphs": all_graphs,
    }
