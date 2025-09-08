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


# Default ordered indices for greedy knockout pattern selection
DEFAULT_GREEDY_ORDERED_INDICES = [
    48, 17, 52, 146, 154, 30, 35, 33, 68, 145,
    12, 99, 139, 46, 111, 144, 57, 153, 10, 64,
]


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

def create_strip_knockout_pattern(
    key: jax.random.PRNGKey,
    layer_sizes: List[Tuple[int, int]],
    damage_prob: float,
) -> jp.ndarray:
    """
    Create localized knockout patterns by damaging gates within a radius of centers.
    
    Args:
        key: Random key for reproducible generation
        layer_sizes: List of (total_gates, group_size) for each gate layer
        damage_prob: Number of gates to knock out
    """
    total_nodes = sum(total_gates for total_gates, _ in layer_sizes)
    knockout_pattern = jp.zeros(total_nodes, dtype=jp.bool_)
    
    # Calculate damage radius (if damage_prob is 0, radius will be 0 and no damage applied)
    damage_radius = int(damage_prob / 2)
    
    # Convert layer_sizes to JAX arrays for JAX-compatible operations
    layer_sizes_array = jp.array(layer_sizes)
    
    # Create layer boundaries
    layer_boundaries = jp.concatenate([
        jp.array([0]),
        jp.cumsum(layer_sizes_array[:, 0])
    ])
    
    # Create damage mask for each layer
    def damage_layer(layer_idx, layer_start, layer_end):
        # Skip input and output layers
        is_input_or_output = (layer_idx == 0) | (layer_idx == len(layer_sizes) - 1)
        
        def apply_damage():
            # Select random center within this layer
            center = layer_start + jax.random.randint(
                key, (), layer_start, layer_end
            )
            
            # Damage gates within radius (simple 1D indexing)
            start_idx = jp.maximum(layer_start, center - damage_radius)
            end_idx = jp.minimum(layer_end, center + damage_radius + 1)
            
            # Create damage mask for this layer using boolean indexing
            layer_mask = jp.zeros(total_nodes, dtype=jp.bool_)
            # Create a boolean mask for the damage range
            damage_indices = jp.arange(total_nodes)
            damage_mask = (damage_indices >= start_idx) & (damage_indices < end_idx)
            layer_mask = layer_mask | damage_mask
            return layer_mask
        
        def no_damage():
            return jp.zeros(total_nodes, dtype=jp.bool_)
        
        return jax.lax.cond(is_input_or_output, no_damage, apply_damage)
    
    # Apply damage to all layers
    layer_indices = jp.arange(len(layer_sizes))
    damage_masks = jax.vmap(damage_layer)(layer_indices, layer_boundaries[:-1], layer_boundaries[1:])
    
    # Combine all damage masks
    knockout_pattern = jp.any(damage_masks, axis=0)
    
    return knockout_pattern


def create_greedy_knockout_pattern(
    ordered_indices: List[int],
    layer_sizes: List[Tuple[int, int]],
    max_gates: int,
) -> jp.ndarray:
    """
    Create a knockout pattern using a predefined ordered list of gate indices.
    
    Args:
        ordered_indices: List of gate indices in order of selection (e.g., from greedy selection)
        layer_sizes: List of (total_gates, group_size) for each layer
        max_gates: Maximum number of gates to knock out
        
    Returns:
        Boolean array where True indicates knocked out gates
    """
    total_nodes = sum(total_gates for total_gates, _ in layer_sizes)
    knockout_pattern = jp.zeros(total_nodes, dtype=jp.bool_)
    
    # Take up to max_gates indices from the ordered list
    num_knockouts = min(max_gates, len(ordered_indices))
    selected_indices = ordered_indices[:num_knockouts]
    
    # Set the selected indices to True (knocked out)
    knockout_pattern = knockout_pattern.at[jp.array(selected_indices)].set(True)
    
    return knockout_pattern


def create_knockout_vocabulary(
    rng: jax.random.PRNGKey,
    vocabulary_size: int,
    layer_sizes: List[Tuple[int, int]],
    damage_prob: float,
    damage_mode: str = "shotgun",  # Options: "shotgun", "strip", or "greedy"
    ordered_indices: Optional[List[int]] = None,  # Used when damage_mode == "greedy"
) -> jp.ndarray:
    """
    Generates a fixed vocabulary of knockout patterns.

    Args:
        rng: JAX random key.
        vocabulary_size: The number of unique patterns to generate.
        layer_sizes: List of (total_gates, group_size) for each layer.
        damage_prob: The probability of knocking out a connection.
        damage_mode: Type of damage pattern ("shotgun" for random, "strip" for localized, "greedy" for ordered selection).
        ordered_indices: List of gate indices in order of selection for greedy mode.

    Returns:
        An array of knockout patterns of shape (vocabulary_size, ...).
    """
    # Greedy mode: deterministic pattern built from ordered indices; replicate across vocabulary
    if damage_mode == "greedy":
        indices = ordered_indices if ordered_indices is not None else DEFAULT_GREEDY_ORDERED_INDICES
        pattern = create_greedy_knockout_pattern(
            ordered_indices=indices,
            layer_sizes=layer_sizes,
            max_gates=int(damage_prob),
        )
        return jp.repeat(pattern[None, ...], vocabulary_size, axis=0)

    # Select pattern creator based on damage mode
    if damage_mode == "strip":
        pattern_creator_fn = partial(
            create_strip_knockout_pattern,
            layer_sizes=layer_sizes,
            damage_prob=damage_prob,
        )
    else:
        pattern_creator_fn = partial(
            create_reproducible_knockout_pattern,
            layer_sizes=layer_sizes,
            damage_prob=damage_prob,
        )

    pattern_keys = jax.random.split(rng, vocabulary_size)
    knockout_vocabulary = jax.vmap(pattern_creator_fn)(pattern_keys)

    return knockout_vocabulary
