"""
Extraction utilities for graphs.

This module provides functions for extracting features from graph representations
of boolean circuits after GNN processing.
"""

import jax
import jax.numpy as jp
import jraph


def extract_logits_from_graph(
    graph: jraph.GraphsTuple, logits_original_shapes: list[tuple[int, int, int]]
) -> list[jp.ndarray]:
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


def update_output_node_loss(
    graph: jraph.GraphsTuple,
    layer_sizes: list[tuple[int, int]],
    loss_values: jp.ndarray,
) -> jraph.GraphsTuple:
    """
    Update the loss feature for output nodes in the graph.

    Args:
        graph: GraphsTuple containing node features
        layer_sizes: List of (nodes, group_size) tuples for each layer
        loss_values: Array of loss values for each output node.
                    Expected shape: [num_output_nodes] or scalar.
                    If shape is [case_n, output_bits], it will be averaged across cases.

    Returns:
        Updated graph with loss feature populated for output nodes
    """
    # Use the helper function to get output node indices
    output_start_idx, output_end_idx = get_output_node_indices(layer_sizes)
    num_output_nodes = output_end_idx - output_start_idx

    # Handle different input shapes for loss_values
    loss_values = jp.atleast_1d(loss_values)

    # For 2D case (shape [case_n, output_bits]), average across the first dimension
    processed_loss_values = jp.mean(loss_values, axis=0) if loss_values.ndim == 2 else loss_values

    # If we have a single value, broadcast it to all output nodes
    if processed_loss_values.shape[0] == 1:
        processed_loss_values = jp.full(num_output_nodes, processed_loss_values[0])

    # Ensure we have exactly the right number of values
    # Take the first num_output_nodes values if we have more, or pad with zeros if we have fewer
    if processed_loss_values.shape[0] != num_output_nodes:
        processed_loss_values = jp.resize(processed_loss_values, num_output_nodes)

    # Update the loss feature for output nodes
    updated_loss = (
        graph.nodes["loss"].at[output_start_idx:output_end_idx].set(processed_loss_values)
    )

    # Create updated nodes with the new loss values
    updated_nodes = {**graph.nodes, "loss": updated_loss}

    return graph._replace(nodes=updated_nodes)


def get_output_node_indices(layer_sizes: list[tuple[int, int]]) -> tuple[int, int]:
    """
    Get the start and end indices of output nodes in the flattened graph.

    Args:
        layer_sizes: List of (nodes, group_size) tuples for each layer

    Returns:
        Tuple of (start_index, end_index) for output nodes
    """
    # Calculate total nodes up to the output layer
    current_node_idx = 0

    # Add input layer nodes
    current_node_idx += layer_sizes[0][0]

    # Add all gate layer nodes except the last one
    for nodes, _ in layer_sizes[1:-1]:
        current_node_idx += nodes

    # Output layer starts here
    output_start_idx = current_node_idx
    output_end_idx = current_node_idx + layer_sizes[-1][0]

    return output_start_idx, output_end_idx
