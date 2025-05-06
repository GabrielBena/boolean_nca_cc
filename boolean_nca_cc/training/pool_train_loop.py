"""
Enhanced training loop using circuit pool for boolean circuit optimization.

This module provides functions for training GNN models with a persistent pool
of circuits that are gradually updated during training.
"""

import jax
import jax.numpy as jp
import optax
from flax import nnx
from typing import List, Tuple, Dict, Any, Optional, Union
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
from boolean_nca_cc.training.pool import CircuitPool
from boolean_nca_cc.training.perturbation import (
    shuffle_wires,
    perturb_logits,
    apply_damage,
)


def initialize_circuit_pool(
    rng: jax.random.PRNGKey,
    layer_sizes: List[Tuple[int, int]],
    pool_size: int,
    arity: int = 2,
) -> CircuitPool:
    """
    Initialize a pool of random circuits.

    Args:
        rng: Random key
        layer_sizes: Circuit layer sizes
        pool_size: Number of circuits in the pool
        arity: Number of inputs per gate

    Returns:
        Initialized CircuitPool
    """
    # Split random key for each circuit
    rngs = jax.random.split(rng, pool_size)

    # Generate circuits in parallel using vmap
    vmap_gen_circuit = jax.vmap(lambda rng: gen_circuit(rng, layer_sizes, arity=arity))
    all_wires, all_logits = vmap_gen_circuit(rngs)

    # Create pool data
    pool_data = {
        "wires": all_wires,
        "logits": all_logits,
    }

    return CircuitPool.create(pool_data)


def train_gnn_with_pool(
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
    pool_size: int = 1024,
    reset_fraction: float = 0.5,
    perturb_fraction: float = 0.3,
    batch_size: int = 64,
    init_gnn: Optional[CircuitGNN] = None,
    init_optimizer: Optional[nnx.Optimizer] = None,
    init_pool: Optional[CircuitPool] = None,
    initial_metrics: Optional[Dict] = None,
    lr_scheduler: str = "constant",
    lr_scheduler_params: Optional[Dict] = None,
    loss_type: str = "l4",
):
    """
    Train a GNN to optimize boolean circuit parameters using a persistent pool.

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
        pool_size: Size of the circuit pool
        reset_fraction: Fraction of pool to reset each epoch
        perturb_fraction: Fraction of pool to perturb each epoch
        batch_size: Batch size for training
        init_gnn: Optional pre-trained GNN model to continue training
        init_optimizer: Optional pre-trained optimizer to continue training
        init_pool: Optional pre-initialized pool to continue training
        initial_metrics: Optional dictionary of metrics from previous training
        lr_scheduler: Learning rate scheduler type
        lr_scheduler_params: Dictionary of parameters for the scheduler
        loss_type: Type of loss to use ('l4' or 'bce')

    Returns:
        Dictionary with trained GNN model, pool, and training metrics
    """
    # Initialize random key
    rng = jax.random.PRNGKey(key)

    # Get dimension from layer sizes
    input_n = layer_sizes[0][0]

    # Initialize metrics storage
    if initial_metrics is None:
        losses = []
        accuracies = []
        hard_losses = []
        hard_accuracies = []
    else:
        losses = list(initial_metrics.get("losses", []))
        accuracies = list(initial_metrics.get("accuracies", []))
        hard_losses = list(initial_metrics.get("hard_losses", []))
        hard_accuracies = list(initial_metrics.get("hard_accuracies", []))

    # Initialize or reuse GNN
    if init_gnn is None:
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
            warmup_steps = lr_scheduler_params.get("warmup_steps", epochs // 10)
            target_schedule = optax.cosine_decay_schedule(
                init_value=learning_rate,
                decay_steps=epochs - warmup_steps,
                alpha=lr_scheduler_params.get("alpha", 0.0),
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

        # Create optimizer
        optimizer = nnx.Optimizer(
            gnn, optax.adamw(learning_rate=schedule, weight_decay=weight_decay)
        )
    else:
        optimizer = init_optimizer

    # Initialize or reuse pool
    if init_pool is None:
        rng, pool_key = jax.random.split(rng)
        pool = initialize_circuit_pool(pool_key, layer_sizes, pool_size, arity=arity)
    else:
        pool = init_pool

    # Define training step function
    @partial(nnx.jit, static_argnames=("n_message_steps", "loss_type"))
    def train_step(
        gnn: CircuitGNN,
        optimizer: nnx.Optimizer,
        pool: CircuitPool,
        batch_idxs: jp.ndarray,
        x: jp.ndarray,
        y_target: jp.ndarray,
        n_message_steps: int,
        loss_type: str,
    ):
        """Single training step with a batch from the pool."""

        # Extract batch data from pool
        batch_wires = jax.tree.map(lambda leaf: leaf[batch_idxs], pool.data["wires"])
        batch_logits = jax.tree.map(lambda leaf: leaf[batch_idxs], pool.data["logits"])

        # Define loss function for a batch of circuits
        def batch_loss_fn(gnn_model: CircuitGNN, batch_wires, batch_logits):
            # Process each circuit in the batch using vmap
            def process_circuit(wires, logits):
                # Store original shapes for reconstruction
                logits_original_shapes = [logit.shape for logit in logits]

                # Calculate initial loss for graph globals
                initial_acts = run_circuit(logits, wires, x)
                initial_pred = initial_acts[-1]

                if loss_type == "bce":
                    initial_loss = binary_cross_entropy(initial_pred, y_target)
                elif loss_type == "l4":
                    initial_res = initial_pred - y_target
                    initial_loss = res2loss(initial_res, power=4)
                elif loss_type == "l2":
                    initial_res = initial_pred - y_target
                    initial_loss = res2loss(initial_res, power=2)
                else:
                    raise ValueError(f"Unknown loss_type: {loss_type}")

                # Build graph from the circuit, passing the initial loss
                graph = build_graph(
                    logits, wires, input_n, arity, hidden_dim, loss_value=initial_loss
                )

                # Run GNN for n_message_steps to optimize the circuit
                for i in range(n_message_steps):
                    updated_graph = gnn_model(graph)

                    # Extract updated logits and run the circuit
                    updated_logits = extract_logits_from_graph(
                        updated_graph, logits_original_shapes
                    )
                    all_acts = run_circuit(updated_logits, wires, x)
                    y_pred = all_acts[-1]

                    # Also measure hard circuit accuracy
                    all_hard_acts = run_circuit(updated_logits, wires, x, hard=True)
                    y_hard_pred = all_hard_acts[-1]

                    # Compute loss based on loss_type
                    if loss_type == "bce":
                        loss = binary_cross_entropy(y_pred, y_target)
                        hard_loss = binary_cross_entropy(y_hard_pred, y_target)
                    elif loss_type == "l4":
                        res = y_pred - y_target
                        hard_res = y_hard_pred - y_target
                        loss = res2loss(res, power=4)
                        hard_loss = res2loss(hard_res, power=4)
                    elif loss_type == "l2":
                        res = y_pred - y_target
                        hard_res = y_hard_pred - y_target
                        loss = res2loss(res, power=2)
                        hard_loss = res2loss(hard_res, power=2)
                    else:
                        raise ValueError(f"Unknown loss_type: {loss_type}")

                    graph = graph._replace(globals=loss)

                # Calculate accuracy
                accuracy = compute_accuracy(y_pred, y_target)
                hard_accuracy = compute_accuracy(y_hard_pred, y_target)

                # Return updated logits along with metrics for updating the pool
                return updated_logits, (loss, hard_loss, accuracy, hard_accuracy)

            # Vectorize the process_circuit function across the batch
            vmap_process_circuit = jax.vmap(process_circuit)
            updated_logits_batch, (losses, hard_losses, accuracies, hard_accuracies) = (
                vmap_process_circuit(batch_wires, batch_logits)
            )

            # Average metrics for the entire batch
            avg_loss = jp.mean(losses)
            aux = (
                jp.mean(losses),
                jp.mean(hard_losses),
                jp.mean(accuracies),
                jp.mean(hard_accuracies),
                updated_logits_batch,
            )

            return avg_loss, aux

        # Compute loss, gradients, and updated logits
        (
            (
                loss,
                (
                    avg_loss,
                    avg_hard_loss,
                    avg_accuracy,
                    avg_hard_accuracy,
                    updated_logits_batch,
                ),
            ),
            grads,
        ) = nnx.value_and_grad(batch_loss_fn, has_aux=True)(
            gnn, batch_wires, batch_logits
        )

        # Update GNN parameters
        optimizer = optimizer.update(grads)

        # Update the pool with optimized logits
        updated_batch = {"logits": updated_logits_batch}
        updated_pool = pool.update(batch_idxs, updated_batch)

        return (
            optimizer,
            updated_pool,
            avg_loss,
            avg_hard_loss,
            avg_accuracy,
            avg_hard_accuracy,
        )

    # Create progress bar for training
    pbar = tqdm(range(epochs), desc="Training GNN with Pool")

    # Training loop
    for epoch in pbar:
        # Split key for this epoch
        rng, epoch_key, data_key, sample_key, reset_key, perturb_key = jax.random.split(
            rng, 6
        )

        # Sample a batch of circuits from the pool
        batch_idxs, _ = pool.sample(sample_key, batch_size)

        # Select a random subset of data for this epoch
        data_idx = jax.random.permutation(data_key, len(x_data))
        x_batch = x_data[data_idx]
        y_batch = y_data[data_idx]

        # Perform training step
        optimizer, pool, loss, hard_loss, accuracy, hard_accuracy = train_step(
            gnn,
            optimizer,
            pool,
            batch_idxs,
            x_batch,
            y_batch,
            n_message_steps,
            loss_type,
        )

        # Reset a fraction of the pool
        reset_count = int(pool_size * reset_fraction)
        if reset_count > 0:
            # Choose random indices to reset
            reset_idxs = jax.random.choice(
                reset_key, pool_size, shape=(reset_count,), replace=False
            )

            # Generate new random circuits
            reset_rngs = jax.random.split(reset_key, reset_count)
            vmap_gen_circuit = jax.vmap(
                lambda rng: gen_circuit(rng, layer_sizes, arity=arity)
            )
            reset_wires, reset_logits = vmap_gen_circuit(reset_rngs)

            # Update the pool with new circuits
            reset_batch = {
                "wires": reset_wires,
                "logits": reset_logits,
            }
            pool = pool.update(reset_idxs, reset_batch)

        # Perturb a fraction of the pool (excluding reset circuits)
        perturb_count = int(pool_size * perturb_fraction)
        if perturb_count > 0:
            # Create a mask to exclude reset indices
            valid_indices = jp.ones(pool_size, dtype=bool)
            if reset_count > 0:
                valid_indices = valid_indices.at[reset_idxs].set(False)
            valid_idxs = jp.where(valid_indices)[0]

            # Choose random indices to perturb from valid indices
            perturb_count = min(perturb_count, len(valid_idxs))
            if perturb_count > 0:
                perturb_sample_idx = jax.random.choice(
                    perturb_key, len(valid_idxs), shape=(perturb_count,), replace=False
                )
                perturb_idxs = valid_idxs[perturb_sample_idx]

                # Sample perturbation keys
                perturb_rngs = jax.random.split(perturb_key, perturb_count)

                # Get the circuits to perturb
                perturb_wires = jax.tree.map(
                    lambda leaf: leaf[perturb_idxs], pool.data["wires"]
                )
                perturb_logits = jax.tree.map(
                    lambda leaf: leaf[perturb_idxs], pool.data["logits"]
                )

                # Apply perturbations (randomly choose between different perturbation types)
                def apply_random_perturbation(rng, wires, logits):
                    rng1, rng2 = jax.random.split(rng)
                    perturb_type = jax.random.randint(
                        rng1, shape=(), minval=0, maxval=3
                    )

                    # Define a switch-like function using conditionals
                    wires_out, logits_out = jp.piecewise(
                        perturb_type,
                        [
                            perturb_type == 0,
                            perturb_type == 1,
                            perturb_type == 2,
                        ],
                        [
                            lambda _: shuffle_wires(rng2, wires, logits),
                            lambda _: perturb_logits(rng2, wires, logits),
                            lambda _: apply_damage(rng2, wires, logits),
                        ],
                        0,  # Dummy value, not used
                    )

                    return wires_out, logits_out

                # Vectorize the perturbation function
                vmap_perturb = jax.vmap(apply_random_perturbation)
                perturbed_wires, perturbed_logits = vmap_perturb(
                    perturb_rngs, perturb_wires, perturb_logits
                )

                # Update the pool with perturbed circuits
                perturb_batch = {
                    "wires": perturbed_wires,
                    "logits": perturbed_logits,
                }
                pool = pool.update(perturb_idxs, perturb_batch)

        # Record metrics
        losses.append(float(loss))
        hard_losses.append(float(hard_loss))
        accuracies.append(float(accuracy))
        hard_accuracies.append(float(hard_accuracy))

        # Update progress bar
        pbar.set_postfix(
            {
                "Loss": f"{loss:.4f}",
                "Accuracy": f"{accuracy:.4f}",
                "Hard Acc": f"{hard_accuracy:.4f}",
            }
        )

    # Return the trained GNN model, optimizer, pool, and metrics
    return {
        "gnn": gnn,
        "optimizer": optimizer,
        "pool": pool,
        "losses": losses,
        "hard_losses": hard_losses,
        "accuracies": accuracies,
        "hard_accuracies": hard_accuracies,
    }
