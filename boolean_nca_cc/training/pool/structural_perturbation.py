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
        damage_prob: Expected number of knockouts. Behavior depends on target_layer:
                    - If target_layer is specified: Expected knockouts in that layer
                    - If target_layer is None: Expected knockouts across entire circuit
        target_layer: If specified, only knock out gates in this layer
        input_n: Number of input nodes (never knocked out)
        
    Returns:
        Boolean array indicating which nodes are knocked out
        Shape: (total_nodes,) where True = knocked out
        
    Note:
        - Input layer (layer 0) is never knocked out
        - Output layer (last layer) is never knocked out
        - When target_layer=None, damage is distributed across all eligible layers
    """
    total_nodes = input_n  # Start with input nodes
    for group_n, group_size in layer_sizes:
        total_nodes += group_n * group_size
    
    # Initialize knockout pattern (False = active, True = knocked out)
    knockout_pattern = jp.zeros(total_nodes, dtype=jp.bool_)
    
    # Never knock out input nodes
    current_idx = input_n
    
    # Identify output layer index (last layer)
    output_layer_idx = len(layer_sizes) - 1
    
    # Calculate total eligible gates (excluding output layer)
    if target_layer is None:
        # When target_layer is None, calculate total gates across all layers except output
        total_eligible_gates = 0
        for layer_idx, (group_n, group_size) in enumerate(layer_sizes):
            if layer_idx != output_layer_idx:  # Exclude output layer
                total_eligible_gates += group_n * group_size
    
    # Process each gate layer
    for layer_idx, (group_n, group_size) in enumerate(layer_sizes):
        layer_size = group_n * group_size
        layer_end = current_idx + layer_size
        
        # Skip output layer - never knock out output nodes
        if layer_idx == output_layer_idx:
            current_idx = layer_end
            continue
        
        # Apply knockouts based on target_layer setting
        if target_layer is None:
            # Circuit-wide mode: distribute damage_prob across all eligible gates
            if total_eligible_gates > 0:
                scaled_prob = damage_prob / total_eligible_gates
            else:
                scaled_prob = 0.0
        elif layer_idx == target_layer:
            # Layer-specific mode: scale by this layer's size
            scaled_prob = damage_prob / layer_size if layer_size > 0 else 0.0
        else:
            # Not the target layer, skip
            current_idx = layer_end
            continue
        
        # Ensure probability is valid
        scaled_prob = jp.clip(scaled_prob, 0.0, 1.0)
        
        # Generate knockout decisions for this layer
        layer_key = jax.random.fold_in(key, layer_idx)
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