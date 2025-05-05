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
from boolean_nca_cc.training.train_step import train_step_gnn
from model import gen_circuit, run_circuit


def train_gnn(
    layer_sizes: List[Tuple[int, int]],
    x_data: jp.ndarray,
    y_data: jp.ndarray,
    arity: int = 2,
    hidden_dim: int = 16,
    message_passing: bool = True,
    node_mlp_features: List[int] = [64, 32],
    edge_mlp_features: List[int] = [64, 32],
    use_attention: bool = False,
    learning_rate: float = 1e-3,
    epochs: int = 100,
    n_message_steps: int = 100,
    key: int = 0,
    weight_decay: float = 1e-4,
    meta_learning: bool = False,
    meta_batch_size: int = 64,
    init_gnn: CircuitGNN = None,
    init_optimizer: nnx.Optimizer = None,
    initial_metrics: Dict = None,
    lr_scheduler: str = "constant",  # Options: "constant", "exponential", "cosine", "linear_warmup"
    lr_scheduler_params: Dict = None,
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

    Returns:
        Dictionary with trained GNN model and training metrics
    """
    # Initialize random key
    rng = jax.random.PRNGKey(key)

    # Get dimension from layer sizes
    input_n = layer_sizes[0][0]

    # Use regular batch size if not meta-learning
    if not meta_learning:
        meta_batch_size = 1

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
    @partial(nnx.jit, static_argnames=("layer_sizes", "n_message_steps"))
    def meta_train_step(
        gnn: CircuitGNN,
        optimizer: nnx.Optimizer,
        x: jp.ndarray,
        y_target: jp.ndarray,
        rng: jax.random.PRNGKey,
        layer_sizes: List[Tuple[int, int]],
        n_message_steps: int,
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

        Returns:
            Tuple of (loss, (hard_loss, accuracy, hard_accuracy))
        """

        # Internal loss function for a single circuit
        def loss_fn(gnn_model: CircuitGNN, rng: jax.random.PRNGKey):
            # Sample new random circuit wiring
            rng_wires, _ = jax.random.split(rng)
            wires, logits = gen_circuit(rng_wires, layer_sizes, arity=arity)

            # Store original shapes for reconstruction
            logits_original_shapes = [logit.shape for logit in logits]

            # Build graph from the random circuit
            graph = build_graph(logits, wires, input_n, arity, hidden_dim)

            # Run GNN for n_message_steps to optimize the circuit
            updated_graph = run_gnn_scan(gnn_model, graph, n_message_steps)

            # Extract updated logits and run the circuit
            updated_logits = extract_logits_from_graph(
                updated_graph, logits_original_shapes
            )
            all_acts = run_circuit(updated_logits, wires, x)
            y_pred = all_acts[-1]

            # Also measure hard circuit accuracy
            all_hard_acts = run_circuit(updated_logits, wires, x, hard=True)
            y_hard_pred = all_hard_acts[-1]

            # Compute loss and accuracy
            loss = jp.mean((y_pred - y_target) ** 4)
            hard_loss = jp.mean((y_hard_pred - y_target) ** 4)
            accuracy = jp.mean(jp.round(y_pred) == y_target)
            hard_accuracy = jp.mean(jp.round(y_hard_pred) == y_target)

            return loss, (hard_loss, accuracy, hard_accuracy)

        # For meta-learning, average over multiple random circuits
        def mean_batch_loss_fn(gnn, rng):
            # Create batch of random keys
            batch_rng = jax.random.split(rng, meta_batch_size)
            # Use vmap to vectorize loss function over the random keys
            batch_loss_fn = nnx.vmap(loss_fn, in_axes=(None, 0))
            # Compute losses for each random circuit
            losses, aux = batch_loss_fn(gnn, rng=batch_rng)
            # Average losses and metrics
            return jp.mean(losses), jax.tree.map(lambda x: jp.mean(x, axis=0), aux)

        # Compute loss and gradients
        (loss, aux), grads = nnx.value_and_grad(mean_batch_loss_fn, has_aux=True)(
            gnn, rng=rng
        )

        # Update GNN parameters
        optimizer.update(grads)

        return loss, aux

    # Create progress bar for training
    pbar = tqdm(range(epochs), desc="Training GNN")

    # Training loop
    for epoch in pbar:
        # Each epoch uses a different random key
        if meta_learning:
            rng, epoch_key = jax.random.split(rng)
        else:
            epoch_key = rng

        # Select a random subset of data for this epoch
        idx = jax.random.permutation(epoch_key, len(x_data))
        x_batch = x_data[idx]
        y_batch = y_data[idx]

        # Perform training step
        loss, (hard_loss, accuracy, hard_accuracy) = meta_train_step(
            gnn,
            optimizer,
            x_batch,
            y_batch,
            epoch_key,
            tuple(layer_sizes),
            n_message_steps,
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
            }
        )

    # Return the trained GNN model and metrics
    return {
        "gnn": gnn,
        "optimizer": optimizer,
        "losses": losses,
        "hard_losses": hard_losses,
        "accuracies": accuracies,
        "hard_accuracies": hard_accuracies,
    }
