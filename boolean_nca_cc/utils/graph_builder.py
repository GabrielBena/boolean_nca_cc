"""
Graph building utilities for boolean circuits.

This module provides functions for constructing graph representations
of boolean circuits for use with graph neural networks.
"""

import jax
import jax.numpy as jp
import jraph
from typing import List, Dict, Tuple

from boolean_nca_cc.utils.positional_encoding import get_positional_encoding


def build_graph(
    logits: List[jp.ndarray],
    wires: List[jp.ndarray],
    input_n: int,
    arity: int,
    hidden_dim: int,
    bidirectional_edges: bool = True,
) -> jraph.GraphsTuple:
    """
    Construct a jraph.GraphsTuple representation of a boolean circuit.

    Args:
        logits: List of logit tensors per layer. Shape [(group_n, group_size, 2^arity), ...]
        wires: List of wire connection patterns per layer. Shape [(arity, edges_in_layer), ...]
        input_n: Number of input nodes/bits for the first layer
        arity: Fan-in for each gate
        hidden_dim: Dimension of hidden features for nodes
        bidirectional_edges: If True, create edges in both forward and backward directions

    Returns:
        A jraph.GraphsTuple representing the circuit
    """
    all_nodes_features_list = []
    all_forward_senders = []
    all_forward_receivers = []
    current_global_node_idx = 0
    layer_start_indices = []  # Store the start index of each layer

    # Process layers
    for layer_idx, (layer_logits, layer_wires) in enumerate(zip(logits, wires)):
        # layer_logits: (group_n, group_size, 2^arity)
        # layer_wires: (arity, group_n) -> connects previous inputs to this layer
        group_n, group_size, logit_dim = layer_logits.shape
        num_gates_in_layer = group_n * group_size

        # Store the starting index for this layer
        layer_start_indices.append(current_global_node_idx)

        # Node features for this layer
        layer_global_indices = jp.arange(
            current_global_node_idx, current_global_node_idx + num_gates_in_layer
        )
        layer_nodes = {
            "layer": jp.full(num_gates_in_layer, layer_idx, dtype=jp.int32),
            "group": jp.repeat(jp.arange(group_n), group_size),
            "gate_id": layer_global_indices,
            "logits": layer_logits.reshape(num_gates_in_layer, logit_dim),
            "hidden": jp.zeros((num_gates_in_layer, hidden_dim), dtype=jp.float32),
        }

        # Add Positional Encodings
        # Layer PE (using layer index)
        max_layers = len(logits)
        layer_indices = jp.full(num_gates_in_layer, layer_idx, dtype=jp.int32)
        layer_pe = get_positional_encoding(
            layer_indices, hidden_dim, max_val=max_layers + 1
        )

        # Intra-Layer PE (using position within the layer)
        intra_layer_indices = jp.arange(num_gates_in_layer, dtype=jp.int32)
        intra_layer_pe = get_positional_encoding(
            intra_layer_indices, hidden_dim, max_val=num_gates_in_layer + 1
        )

        layer_nodes["layer_pe"] = layer_pe
        layer_nodes["intra_layer_pe"] = intra_layer_pe

        all_nodes_features_list.append(layer_nodes)

        # Create forward edges (if not the first layer)
        if layer_idx > 0:
            # Receivers are the gates in the current layer
            current_layer_receivers = jp.repeat(layer_global_indices, arity)

            # Senders are gates from the previous layer
            previous_layer_start_idx = layer_start_indices[layer_idx - 1]
            global_senders_for_layer = previous_layer_start_idx + layer_wires
            tiled_senders = jp.tile(global_senders_for_layer.T, (1, group_size))
            current_layer_senders = tiled_senders.reshape(-1)  # Flatten

            all_forward_senders.append(current_layer_senders)
            all_forward_receivers.append(current_layer_receivers)

        # Update global index for the next layer
        current_global_node_idx += num_gates_in_layer

    # Consolidate Nodes and Edges
    if not all_nodes_features_list:
        return jraph.GraphsTuple(
            nodes={},
            edges=None,
            senders=jp.array([], dtype=jp.int32),
            receivers=jp.array([], dtype=jp.int32),
            n_node=jp.array([0]),
            n_edge=jp.array([0]),
            globals=None,
        )

    # Combine node features from all layers
    all_nodes = jax.tree.map(
        lambda *xs: jp.concatenate(xs, axis=0), *all_nodes_features_list
    )

    # Process edges
    if all_forward_senders:
        forward_senders = jp.concatenate(all_forward_senders)
        forward_receivers = jp.concatenate(all_forward_receivers)

        if bidirectional_edges:
            # Create backward edges by swapping senders and receivers
            backward_senders = forward_receivers
            backward_receivers = forward_senders
            # Combine forward and backward edges
            senders = jp.concatenate([forward_senders, backward_senders])
            receivers = jp.concatenate([forward_receivers, backward_receivers])
        else:
            # Use only forward edges
            senders = forward_senders
            receivers = forward_receivers
    else:
        senders = jp.array([], dtype=jp.int32)
        receivers = jp.array([], dtype=jp.int32)

    n_node = current_global_node_idx
    n_edge = len(senders)

    # Create and return the GraphsTuple
    graph = jraph.GraphsTuple(
        nodes=all_nodes,
        edges=None,  # No edge features initially
        senders=senders.astype(jp.int32),
        receivers=receivers.astype(jp.int32),
        n_node=jp.array([n_node]),
        n_edge=jp.array([n_edge]),
        globals=None,  # No global features
    )

    return graph
