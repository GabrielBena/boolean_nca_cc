"""
Training loop implementation for self-attention-based boolean circuit optimization.

This module provides functions for training self-attention models to optimize
boolean circuits over multiple epochs.
"""

import jax
import jax.numpy as jp
import optax
from flax import nnx
from typing import List, Tuple, Dict, Optional
from functools import partial
from tqdm.auto import tqdm

from boolean_nca_cc.models.self_attention import (
    CircuitSelfAttention,
    run_self_attention_scan,
)
from boolean_nca_cc.utils import build_graph, extract_logits_from_graph
from boolean_nca_cc.training.attn_train_step import train_step_attn
from model import gen_circuit, run_circuit


def train_self_attention(
    layer_sizes: List[Tuple[int, int]],
    x_data: jp.ndarray,
    y_data: jp.ndarray,
    arity: int = 2,
    hidden_dim: int = 16,
    num_heads: int = 4,
    num_layers: int = 3,
    mlp_dim: int = 64,
    learning_rate: float = 1e-3,
    epochs: int = 100,
    n_attention_steps: int = 100,
    key: int = 0,
    weight_decay: float = 1e-4,
    dropout_rate: float = 0.0,
    meta_learning: bool = False,
    meta_batch_size: int = 64,
    init_model: CircuitSelfAttention = None,
    init_optimizer: nnx.Optimizer = None,
    initial_metrics: Dict = None,
    lr_scheduler: str = "constant",
    lr_scheduler_params: Dict = None,
):
    """
    Train a self-attention model to optimize boolean circuit parameters.

    Args:
        layer_sizes: List of tuples (nodes, group_size) for each layer
        x_data: Input data for training [batch, input_bits]
        y_data: Target output data [batch, output_bits]
        arity: Number of inputs per gate
        hidden_dim: Dimension of hidden features
        num_heads: Number of attention heads
        num_layers: Number of self-attention layers
        mlp_dim: Dimension of feed-forward network
        learning_rate: Learning rate for optimization
        epochs: Number of training epochs
        n_attention_steps: Number of self-attention steps per epoch
        key: Random seed
        weight_decay: Weight decay for optimizer
        dropout_rate: Dropout rate for attention
        meta_learning: Whether to use meta-learning (train on new random circuits each step)
        meta_batch_size: Batch size for meta-learning
        init_model: Optional pre-trained self-attention model to continue training
        init_optimizer: Optional pre-trained optimizer to continue training
        initial_metrics: Optional dictionary of metrics from previous training
        lr_scheduler: Learning rate scheduler type. Default: "constant".
        lr_scheduler_params: Dictionary of parameters for the scheduler.

    Returns:
        Dictionary with trained model and training metrics
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

    # Initialize or reuse model
    if init_model is None:
        # Create a new self-attention model
        rng, init_key = jax.random.split(rng)
        model = CircuitSelfAttention(
            hidden_dim=hidden_dim,
            arity=arity,
            num_heads=num_heads,
            num_layers=num_layers,
            mlp_dim=mlp_dim,
            dropout_rate=dropout_rate,
            rngs=nnx.Rngs(params=init_key),
        )
    else:
        # Use the provided model
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
        optimizer = nnx.Optimizer(
            model, optax.adamw(learning_rate=schedule, weight_decay=weight_decay)
        )
    else:
        # Use the provided optimizer
        optimizer = init_optimizer

    # Define meta-training step function
    @partial(nnx.jit, static_argnames=("layer_sizes", "n_attention_steps"))
    def meta_train_step(
        model: CircuitSelfAttention,
        optimizer: nnx.Optimizer,
        x: jp.ndarray,
        y_target: jp.ndarray,
        rng: jax.random.PRNGKey,
        layer_sizes: List[Tuple[int, int]],
        n_attention_steps: int,
        deterministic: bool = True,
    ):
        """
        Single meta-training step with randomly sampled circuit wirings.

        Args:
            model: CircuitSelfAttention model
            optimizer: nnx Optimizer
            x: Input data
            y_target: Target output data
            rng: Random key
            layer_sizes: Circuit layer sizes
            n_attention_steps: Number of self-attention steps
            deterministic: If False, apply dropout

        Returns:
            Tuple of (loss, (hard_loss, accuracy, hard_accuracy))
        """

        # Internal loss function for a single circuit
        def loss_fn(attn_model: CircuitSelfAttention, rng: jax.random.PRNGKey):
            # Sample new random circuit wiring
            rng_wires, rng_dropout = jax.random.split(rng)
            wires, logits = gen_circuit(rng_wires, layer_sizes, arity=arity)

            # Store original shapes for reconstruction
            logits_original_shapes = [logit.shape for logit in logits]

            # Build graph from the random circuit
            graph = build_graph(logits, wires, input_n, arity, hidden_dim)

            # Setup dropout rng
            step_rngs = None
            if not deterministic:
                step_rngs = nnx.Rngs(dropout=rng_dropout)

            # Run self-attention for n_attention_steps to optimize the circuit
            updated_graph = run_self_attention_scan(
                attn_model,
                graph,
                n_attention_steps,
                deterministic=deterministic,
                rngs=step_rngs,
            )

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
        def mean_batch_loss_fn(model, rng):
            # Create batch of random keys
            batch_rng = jax.random.split(rng, meta_batch_size)
            # Use vmap to vectorize loss function over the random keys
            batch_loss_fn = nnx.vmap(loss_fn, in_axes=(None, 0))
            # Compute losses for each random circuit
            losses, aux = batch_loss_fn(model, rng=batch_rng)
            # Average losses and metrics
            return jp.mean(losses), jax.tree.map(lambda x: jp.mean(x, axis=0), aux)

        # Compute loss and gradients
        (loss, aux), grads = nnx.value_and_grad(mean_batch_loss_fn, has_aux=True)(
            model, rng=rng
        )

        # Update model parameters
        optimizer.update(grads)

        return loss, aux

    # Create progress bar for training
    pbar = tqdm(range(epochs), desc="Training Self-Attention")

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

        # Determine whether to use dropout
        deterministic = dropout_rate == 0.0

        # Perform training step
        loss, (hard_loss, accuracy, hard_accuracy) = meta_train_step(
            model,
            optimizer,
            x_batch,
            y_batch,
            epoch_key,
            tuple(layer_sizes),
            n_attention_steps,
            deterministic=deterministic,
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

    # Return the trained model and metrics
    return {
        "model": model,
        "optimizer": optimizer,
        "losses": losses,
        "hard_losses": hard_losses,
        "accuracies": accuracies,
        "hard_accuracies": hard_accuracies,
    }
