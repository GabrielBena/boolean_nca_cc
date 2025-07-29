"""
Structural perturbation utilities for boolean circuit graphs.

This module provides functions to modify graph topology for knockouts,
which work naturally with attention-based models by removing edges
to/from knocked out gates.
"""

import jax
import jax.numpy as jp
import jraph
from typing import List, Tuple, Optional
from functools import partial


def create_reproducible_knockout_pattern(
    key: jax.random.PRNGKey,
    layer_sizes: List[Tuple[int, int]],  # (total_gates, group_size) for each layer
    damage_prob: float,
) -> jp.ndarray:
    """
    Create a reproducible knockout pattern for gates in the circuit.
    
    Args:
        key: Random key for reproducible generation
        layer_sizes: List of (total_gates, group_size) for each gate layer
        damage_prob: Exact number of knockouts to apply
    """
    total_nodes = 0 
    for total_gates, group_size in layer_sizes:
        total_nodes += total_gates 
    
    # Initialize knockout pattern (False = active, True = knocked out)
    knockout_pattern = jp.zeros(total_nodes, dtype=jp.bool_)
    
    # Never knock out input nodes
    current_idx = 0 
    
    # If damage_prob is 0, return all False
    if damage_prob == 0:
        return knockout_pattern
    
    # Identify output layer index (last layer)
    output_layer_idx = len(layer_sizes) - 1
    
    # Single loop: collect all eligible gate indices
    eligible_indices = []
    current_idx = 0
    
    for layer_idx, (total_gates, group_size) in enumerate(layer_sizes):
        layer_size = total_gates  # total_gates is already the layer size
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
    layer_sizes: List[Tuple[int, int]],
    damage_prob: float,
) -> jp.ndarray:
    """
    Generates a fixed vocabulary of knockout patterns.

    Args:
        rng: JAX random key.
        vocabulary_size: The number of unique patterns to generate.
        layer_sizes: List of (total_gates, group_size) for each layer.
        damage_prob: The probability of knocking out a connection.

    Returns:
        An array of knockout patterns of shape (vocabulary_size, ...).
    """
    pattern_creator_fn = partial(
        create_reproducible_knockout_pattern,
        layer_sizes=layer_sizes,
        damage_prob=damage_prob,
    )

    pattern_keys = jax.random.split(rng, vocabulary_size)
    knockout_vocabulary = jax.vmap(pattern_creator_fn)(pattern_keys)

    return knockout_vocabulary
