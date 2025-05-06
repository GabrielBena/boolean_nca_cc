"""
Extraction utilities for graphs.

This module provides functions for extracting features from graph representations
of boolean circuits after GNN processing.
"""

import jax
import jax.numpy as jp
import jraph
from typing import List, Tuple


def extract_logits_from_graph(
    graph: jraph.GraphsTuple, logits_original_shapes: List[Tuple[int, int, int]]
) -> List[jp.ndarray]:
    """
    Extract logit tensors from a graph's node features in a JIT-compatible way.

    Args:
        graph: GraphsTuple containing node features (including flattened 'logits')
        logits_original_shapes: List of original (group_n, group_size, 2^arity) shapes
                                for each layer

    Returns:
        List of logit tensors, reshaped to their original shapes
    """
    all_logits_flat = graph.nodes["logits"]
    extracted_logits_list = []
    current_node_idx = 0

    # Precompute sizes and start indices for dynamic slicing
    layer_sizes = []
    layer_start_indices = []
    for shape in logits_original_shapes:
        group_n, group_size, logit_dim = shape
        num_gates = group_n * group_size
        layer_sizes.append(num_gates)
        layer_start_indices.append(current_node_idx)
        current_node_idx += num_gates

    # Use dynamic_slice for JIT compatibility
    for i, shape in enumerate(logits_original_shapes):
        group_n, group_size, logit_dim = shape
        start_index = layer_start_indices[i]
        size = layer_sizes[i]

        # dynamic_slice needs start indices as a tuple for each dimension
        start_indices_tuple = (start_index, 0)
        # dynamic_slice needs slice sizes as a tuple for each dimension
        slice_sizes_tuple = (size, logit_dim)

        layer_logits_flat = jax.lax.dynamic_slice(
            all_logits_flat, start_indices_tuple, slice_sizes_tuple
        )

        # Reshape back to original
        layer_logits = layer_logits_flat.reshape(group_n, group_size, logit_dim)
        extracted_logits_list.append(layer_logits)

    return extracted_logits_list
