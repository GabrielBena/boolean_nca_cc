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


def create_reproducible_knockout_pattern(
    key: jax.random.PRNGKey,
    layer_sizes: List[Tuple[int, int]],  # (group_n, group_size) for each layer
    damage_prob: float,
    target_layer: Optional[int] = None,
    input_n: int = 0,
) -> jp.ndarray:
    """
    Create a reproducible knockout pattern for gates in the circuit.
    
    Args:
        key: Random key for reproducible generation
        layer_sizes: List of (group_n, group_size) for each gate layer
        damage_prob: Expected number of knockouts per layer (automatically scaled by layer size)
                    E.g., damage_prob=1.0 means expect 1 knockout per layer on average
                    Actual per-gate probability = damage_prob / layer_size
        target_layer: If specified, only knock out gates in this layer
        input_n: Number of input nodes (never knocked out)
        
    Returns:
        Boolean array indicating which nodes are knocked out
        Shape: (total_nodes,) where True = knocked out
    """
    total_nodes = input_n  # Start with input nodes
    for group_n, group_size in layer_sizes:
        total_nodes += group_n * group_size
    
    # Initialize knockout pattern (False = active, True = knocked out)
    knockout_pattern = jp.zeros(total_nodes, dtype=jp.bool_)
    
    # Never knock out input nodes
    current_idx = input_n
    
    # Process each gate layer
    for layer_idx, (group_n, group_size) in enumerate(layer_sizes):
        layer_size = group_n * group_size
        layer_end = current_idx + layer_size
        
        if target_layer is None or layer_idx == target_layer:
            # Apply knockouts to this layer
            layer_key = jax.random.fold_in(key, layer_idx)
            
            # Scale probability by layer size so damage_prob represents expected knockouts per layer
            scaled_prob = damage_prob / layer_size if layer_size > 0 else 0.0
            scaled_prob = jp.clip(scaled_prob, 0.0, 1.0)
            
            # Generate knockout decisions
            random_vals = jax.random.uniform(layer_key, (layer_size,))
            layer_knockouts = random_vals < scaled_prob
            
            # Apply to knockout pattern
            knockout_pattern = knockout_pattern.at[current_idx:layer_end].set(layer_knockouts)
        
        current_idx = layer_end
    
    return knockout_pattern


def extract_layer_info_from_graph(graph: jraph.GraphsTuple, input_n: int) -> List[Tuple[int, int]]:
    """
    Extract layer size information from a graph.
    
    Args:
        graph: Input graph with layer information in nodes
        input_n: Number of input nodes
        
    Returns:
        List of (group_n, group_size) tuples for each gate layer
    """
    if 'layer' not in graph.nodes or 'group' not in graph.nodes:
        raise ValueError("Graph must have 'layer' and 'group' node features")
    
    layers = graph.nodes['layer']
    groups = graph.nodes['group']
    
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