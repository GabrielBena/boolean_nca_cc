"""
Circuit perturbation utilities for boolean circuit NCA.

This module provides functions for perturbing circuits during training
to improve robustness and generalization.
"""

import jax
import jax.numpy as jp
from typing import List, Tuple, Dict, Any


def mutate_wires_swap(
    wires: List[jp.ndarray],
    key: jax.random.PRNGKey,
    mutation_rate: float = 0.1,
    n_swaps_per_layer: int = None,
) -> List[jp.ndarray]:
    """
    Mutate circuit wires by swapping connections within each layer.

    This is a conservative mutation strategy that preserves the overall
    connectivity structure while introducing variation.

    Args:
        wires: List of wire arrays for each layer, shape [(arity, group_n), ...]
        mutation_rate: Probability for each connection to be involved in a swap (0.0 to 1.0)
                      Ignored if n_swaps_per_layer is provided
        key: Random key for reproducible mutations
        n_swaps_per_layer: If provided, use this exact number of swaps per layer instead of using mutation_rate

    Returns:
        List of mutated wire arrays with same structure as input
    """
    if mutation_rate <= 0.0 and n_swaps_per_layer is None:
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

        layer_key = keys[layer_idx]

        # Calculate maximum possible swaps for this layer
        max_swaps = n_connections // 2

        # Calculate number of swaps to perform
        if n_swaps_per_layer is not None:
            # Use the specified number of swaps (but cap at max_swaps)
            n_swaps = jp.minimum(n_swaps_per_layer, max_swaps)
        else:
            # Use mutation_rate as Bernoulli probability for each connection
            bernoulli_key, layer_key = jax.random.split(layer_key)
            mutation_mask = (
                jax.random.uniform(bernoulli_key, (n_connections,)) < mutation_rate
            )

            # Count how many connections are selected for mutation
            n_to_mutate = jp.sum(mutation_mask)

            # We need pairs for swapping, so take the floor of half the selected connections
            n_swaps = jp.minimum(n_to_mutate // 2, max_swaps)

        # Always generate max_swaps worth of swap indices to avoid shape issues
        # We'll use masking to only apply the first n_swaps
        swap_indices = jax.random.choice(
            layer_key, n_connections, shape=(max_swaps, 2), replace=False
        )

        # Create a mask for which swaps to actually apply
        swap_mask = jp.arange(max_swaps) < n_swaps

        # Apply swaps using vectorized operations
        def apply_swap(i, connections):
            idx1, idx2 = swap_indices[i]
            val1, val2 = connections[idx1], connections[idx2]

            # Only apply swap if mask is True for this swap
            should_apply = swap_mask[i]
            new_val1 = jp.where(should_apply, val2, val1)
            new_val2 = jp.where(should_apply, val1, val2)

            connections = connections.at[idx1].set(new_val1)
            connections = connections.at[idx2].set(new_val2)
            return connections

        # Apply all swaps (masked ones will be no-ops)
        new_connections = jax.lax.fori_loop(0, max_swaps, apply_swap, flat_connections)

        # Reshape back to original shape
        mutated_layer = new_connections.reshape(original_shape)
        mutated_wires.append(mutated_layer)

    return mutated_wires


def mutate_wires_batch(
    batch_wires: List[jp.ndarray],
    key: jax.random.PRNGKey,
    mutation_rate: float = 0.1,
    n_swaps_per_layer: int = None,
) -> List[jp.ndarray]:
    """
    Apply wire mutation to a batch of circuits.

    Args:
        batch_wires: List of batched wire arrays, shape [(batch_size, arity, group_n), ...]
        mutation_rate: Probability for each connection to be involved in a swap (0.0 to 1.0)
                      Ignored if n_swaps_per_layer is provided
        key: Random key for mutations
        n_swaps_per_layer: If provided, use this exact number of swaps per layer instead of using mutation_rate

    Returns:
        List of mutated batched wire arrays
    """
    if not batch_wires or (mutation_rate <= 0.0 and n_swaps_per_layer is None):
        return batch_wires

    batch_size = batch_wires[0].shape[0]
    keys = jax.random.split(key, batch_size)

    # Use vmap to apply mutation to each circuit in the batch
    def mutate_single_circuit(circuit_key, *circuit_wires):
        # Unbatch the wires for this single circuit
        single_wires = [w for w in circuit_wires]
        # Apply mutation (correct argument order: wires, key, mutation_rate, n_swaps_per_layer)
        mutated_single = mutate_wires_swap(
            single_wires, circuit_key, mutation_rate, n_swaps_per_layer
        )
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
