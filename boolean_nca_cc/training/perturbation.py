"""
Circuit perturbation utilities for boolean circuit NCA.

This module provides functions for perturbing circuits during training
to improve robustness and generalization.
"""

import jax
import jax.numpy as jp
from typing import List, Tuple, Dict, Any


def mutate_wires_swap(
    wires: List[jp.ndarray], key: jax.random.PRNGKey, mutation_rate: float = 0.1
) -> List[jp.ndarray]:
    """
    Mutate circuit wires by swapping connections within each layer.

    This is a conservative mutation strategy that preserves the overall
    connectivity structure while introducing variation.

    Args:
        wires: List of wire arrays for each layer, shape [(arity, group_n), ...]
        mutation_rate: Fraction of connections to swap (0.0 to 1.0)
        key: Random key for reproducible mutations

    Returns:
        List of mutated wire arrays with same structure as input
    """
    if mutation_rate <= 0.0:
        return wires

    mutated_wires = []
    keys = jax.random.split(key, len(wires))

    for layer_idx, layer_wires in enumerate(wires):
        if layer_wires.size == 0:
            mutated_wires.append(layer_wires)
            continue

        # Flatten to work with all connections in this layer
        original_shape = layer_wires.shape
        flat_connections = layer_wires.flatten()
        n_connections = len(flat_connections)

        if n_connections < 2:
            # Can't swap if less than 2 connections
            mutated_wires.append(layer_wires)
            continue

        # Calculate number of swaps to perform
        n_swaps = jp.maximum(
            1, jp.round(mutation_rate * n_connections / 2).astype(jp.int32)
        )

        # Generate random pairs of indices to swap
        layer_key = keys[layer_idx]
        swap_indices = jax.random.choice(
            layer_key, n_connections, shape=(n_swaps, 2), replace=False
        )

        # Perform swaps
        new_connections = flat_connections
        for i in range(n_swaps):
            idx1, idx2 = swap_indices[i]
            # Swap values at idx1 and idx2
            val1, val2 = new_connections[idx1], new_connections[idx2]
            new_connections = new_connections.at[idx1].set(val2)
            new_connections = new_connections.at[idx2].set(val1)

        # Reshape back to original shape
        mutated_layer = new_connections.reshape(original_shape)
        mutated_wires.append(mutated_layer)

    return mutated_wires


def mutate_wires_batch(
    batch_wires: List[jp.ndarray], key: jax.random.PRNGKey, mutation_rate: float = 0.1
) -> List[jp.ndarray]:
    """
    Apply wire mutation to a batch of circuits.

    Args:
        batch_wires: List of batched wire arrays, shape [(batch_size, arity, group_n), ...]
        mutation_rate: Fraction of connections to swap per circuit
        key: Random key for mutations

    Returns:
        List of mutated batched wire arrays
    """
    if not batch_wires or mutation_rate <= 0.0:
        return batch_wires

    batch_size = batch_wires[0].shape[0]
    keys = jax.random.split(key, batch_size)

    # Use vmap to apply mutation to each circuit in the batch
    def mutate_single_circuit(circuit_key, *circuit_wires):
        # Unbatch the wires for this single circuit
        single_wires = [w for w in circuit_wires]
        # Apply mutation (correct argument order: wires, key, mutation_rate)
        mutated_single = mutate_wires_swap(single_wires, circuit_key, mutation_rate)
        return mutated_single

    # Apply vmap across the batch dimension
    mutated_batch = jax.vmap(
        mutate_single_circuit, in_axes=(0,) + (0,) * len(batch_wires)
    )(keys, *batch_wires)

    # Reorganize the output back into list of batched arrays
    return [mutated_batch[i] for i in range(len(batch_wires))]


def shuffle_wires(
    rng: jax.random.PRNGKey, wires: List, logits: List
) -> Tuple[List, List]:
    """
    Randomly shuffle some wire connections in the circuit.

    Args:
        rng: Random key
        wires: List of wire connections for each layer
        logits: List of logits for each layer

    Returns:
        Tuple of (shuffled_wires, logits)
    """
    # Make a copy of wires to avoid modifying the original
    shuffled_wires = [w.copy() for w in wires]

    # For each layer (except input layer), shuffle some connections
    for i in range(1, len(wires)):
        layer_rng, rng = jax.random.split(rng)

        # Get the shape of the current layer's wires
        layer_shape = wires[i].shape

        # Decide which connections to shuffle (random 10%)
        shuffle_mask = jax.random.uniform(layer_rng, layer_shape) < 0.1

        if jp.any(shuffle_mask):
            # For selected connections, randomly reassign them to other nodes in previous layer
            prev_layer_size = wires[i - 1].shape[0]
            new_connections = jax.random.randint(
                layer_rng, shape=jp.sum(shuffle_mask), minval=0, maxval=prev_layer_size
            )

            # Update the connections
            shuffled_wires[i] = shuffled_wires[i].at[shuffle_mask].set(new_connections)

    return shuffled_wires, logits


def perturb_logits(
    rng: jax.random.PRNGKey, wires: List, logits: List, noise_scale: float = 0.05
) -> Tuple[List, List]:
    """
    Add small noise perturbations to logits.

    Args:
        rng: Random key
        wires: List of wire connections for each layer
        logits: List of logits for each layer
        noise_scale: Scale of the noise to add

    Returns:
        Tuple of (wires, perturbed_logits)
    """
    # Make a copy of logits to avoid modifying the original
    perturbed_logits = []

    for i, logit in enumerate(logits):
        layer_rng, rng = jax.random.split(rng)

        # Add small Gaussian noise to the logits
        noise = jax.random.normal(layer_rng, logit.shape) * noise_scale
        perturbed_logits.append(logit + noise)

    return wires, perturbed_logits


def apply_damage(
    rng: jax.random.PRNGKey, wires: List, logits: List, damage_prob: float = 0.05
) -> Tuple[List, List]:
    """
    Apply random damage to the circuit by zeroing out some logits.

    Args:
        rng: Random key
        wires: List of wire connections for each layer
        logits: List of logits for each layer
        damage_prob: Probability of damaging each logit

    Returns:
        Tuple of (wires, damaged_logits)
    """
    # Make a copy of logits to avoid modifying the original
    damaged_logits = []

    for i, logit in enumerate(logits):
        layer_rng, rng = jax.random.split(rng)

        # Generate damage mask
        damage_mask = jax.random.uniform(layer_rng, logit.shape) < damage_prob

        # Zero out damaged logits
        damaged = jp.where(damage_mask, jp.zeros_like(logit), logit)
        damaged_logits.append(damaged)

    return wires, damaged_logits
