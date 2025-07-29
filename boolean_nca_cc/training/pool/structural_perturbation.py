"""
Structural perturbation utilities for boolean circuit graphs.

This module provides functions to modify graph topology for knockouts,
which work naturally with attention-based models by removing edges
to/from knocked out gates.
"""

from functools import partial

import jax
import jax.numpy as jp
import jraph


def create_reproducible_knockout_pattern(
    key: jax.random.PRNGKey,
    layer_sizes: list[tuple[int, int]],  # (group_n, group_size) for each layer
    damage_prob: float,
    input_n: int = 0,
) -> jp.ndarray:
    """
    Create a reproducible knockout pattern for gates in the circuit.

    Args:
        key: Random key for reproducible generation
        layer_sizes: List of (group_n, group_size) for each gate layer
        damage_prob: Exact number of knockouts to apply
        input_n: Number of input nodes (never knocked out)

    Returns:
        Boolean array indicating which nodes are knocked out
        Shape: (total_nodes,) where True = knocked out

    Note:
        - Input layer (layer 0) is never knocked out
        - Output layer (last layer) is never knocked out
        - Exactly damage_prob gates will be knocked out
    """
    total_nodes = input_n  # Start with input nodes
    for group_n, group_size in layer_sizes:
        total_nodes += group_n * group_size

    # Initialize knockout pattern (False = active, True = knocked out)
    knockout_pattern = jp.zeros(total_nodes, dtype=jp.bool_)

    # Never knock out input nodes
    current_idx = input_n

    # If damage_prob is 0, return all False
    if damage_prob == 0:
        return knockout_pattern

    # Identify output layer index (last layer)
    output_layer_idx = len(layer_sizes) - 1

    # Single loop: collect all eligible gate indices
    eligible_indices = []
    current_idx = input_n

    for layer_idx, (group_n, group_size) in enumerate(layer_sizes):
        layer_size = group_n * group_size
        layer_end = current_idx + layer_size

        # Skip input and output layers - never knock out input or output nodes
        if layer_idx == 0 or layer_idx == output_layer_idx:
            current_idx = layer_end
            continue

        # Add all gate indices for this layer
        layer_indices = jp.arange(current_idx, layer_end)
        eligible_indices.append(layer_indices)
        current_idx = layer_end

    # Handle edge case: no eligible gates
    if not eligible_indices:
        return knockout_pattern

    # Concatenate all eligible indices
    all_eligible_indices = jp.concatenate(eligible_indices)

    # Ensure damage_prob doesn't exceed total eligible gates
    num_knockouts = min(int(damage_prob), len(all_eligible_indices))

    # Randomly sample exactly num_knockouts indices
    knockout_indices = jax.random.choice(
        key, all_eligible_indices, shape=(num_knockouts,), replace=False
    )

    # Set the selected indices to True (knocked out)
    knockout_pattern = knockout_pattern.at[knockout_indices].set(True)

    return knockout_pattern


def create_knockout_vocabulary(
    rng: jax.random.PRNGKey,
    vocabulary_size: int,
    layer_sizes: list[tuple[int, int]],
    damage_prob: float,
    input_n: int,
) -> jp.ndarray:
    """
    Generates a fixed vocabulary of knockout patterns.

    Args:
        rng: JAX random key.
        vocabulary_size: The number of unique patterns to generate.
        layer_sizes: List of (nodes, group_size) for each layer.
        damage_prob: The probability of knocking out a connection.
        input_n: The number of input nodes.

    Returns:
        An array of knockout patterns of shape (vocabulary_size, ...).
    """
    pattern_creator_fn = partial(
        create_reproducible_knockout_pattern,
        layer_sizes=layer_sizes,
        damage_prob=damage_prob,
        input_n=input_n,
    )

    pattern_keys = jax.random.split(rng, vocabulary_size)
    knockout_vocabulary = jax.vmap(pattern_creator_fn)(pattern_keys)

    return knockout_vocabulary


def extract_layer_info_from_graph(graph: jraph.GraphsTuple, input_n: int) -> list[tuple[int, int]]:
    """
    Extract layer size information from a graph.

    Args:
        graph: Input graph with layer information in nodes
        input_n: Number of input nodes

    Returns:
        List of (group_n, group_size) tuples for each gate layer
    """
    if "layer" not in graph.nodes or "group" not in graph.nodes:
        raise ValueError("Graph must have 'layer' and 'group' node features")

    layers = graph.nodes["layer"]
    groups = graph.nodes["group"]

    # Skip input layer (layer 0)
    gate_mask = layers > 0
    gate_layers = layers[gate_mask]
    gate_groups = groups[gate_mask]

    layer_sizes = []

    # Get the number of layers, handling the case of empty gate_layers
    num_gate_layers = jp.max(gate_layers) if gate_layers.size > 0 else 0

    for layer_idx in range(1, num_gate_layers + 1):
        layer_mask = gate_layers == layer_idx
        if not jp.any(layer_mask):
            continue

        layer_groups = gate_groups[layer_mask]

        if layer_groups.size > 0:
            max_group = jp.max(layer_groups) + 1  # +1 because groups are 0-indexed
            group_size = jp.sum(layer_mask) // max_group
            layer_sizes.append((int(max_group), int(group_size)))

    return layer_sizes
