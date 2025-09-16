"""
Graph building utilities for boolean circuits.

This module provides functions for constructing graph representations
of boolean circuits for use with graph neural networks.
"""

import jax
import jax.numpy as jp
import jraph

from boolean_nca_cc.utils.positional_encoding import get_positional_encoding


def build_graph(
    logits: list[jp.ndarray],
    wires: list[jp.ndarray],
    input_n: int,
    arity: int,
    circuit_hidden_dim: int,
    bidirectional_edges: bool = True,
    loss_value: jp.ndarray | int = 0,
    update_steps: jp.ndarray | int = 0,
) -> jraph.GraphsTuple:
    """
    Construct a jraph.GraphsTuple representation of a boolean circuit, including input nodes.

    Args:
        logits: List of logit tensors per layer. Shape [(group_n, group_size, 2^arity), ...]
        wires: List of wire connection patterns per layer. Shape [(arity, edges_in_layer), ...]
                The first element wires[0] connects input nodes to the first gate layer.
        input_n: Number of input nodes/bits for the first layer
        arity: Fan-in for each gate
        circuit_hidden_dim: Dimension of hidden features for nodes
        bidirectional_edges: If True, create edges in both forward and backward directions
        loss_value: Optional scalar value representing the current loss of the circuit.
        update_steps: Number of times this graph has been updated by the GNN.

    Returns:
        A jraph.GraphsTuple representing the circuit
    """
    all_nodes_features_list = []
    all_forward_senders = []
    all_forward_receivers = []
    current_global_node_idx = 0
    layer_start_indices = []  # Store the start index of each layer
    pe_dim = circuit_hidden_dim  # Dimension for positional encodings
    max_layers = len(logits) + 1  # +1 for the input layer

    # --- Input Layer Nodes ---
    layer_start_indices.append(current_global_node_idx)
    input_layer_indices = jp.arange(input_n)
    input_layer_pe = get_positional_encoding(
        jp.zeros(input_n, dtype=jp.int32), pe_dim, max_val=max_layers
    )
    input_intra_layer_pe = get_positional_encoding(
        input_layer_indices, pe_dim, max_val=input_n + 1
    )
    input_nodes = {
        "layer": jp.zeros(input_n, dtype=jp.int32),  # Input layer is layer 0
        "group": jp.zeros(input_n, dtype=jp.int32),  # No groups for inputs
        "gate_id": input_layer_indices,
        "logits": jp.zeros(
            (input_n, 2**arity), dtype=jp.float32
        ),  # Inputs have no logits
        "hidden": jp.zeros((input_n, circuit_hidden_dim), dtype=jp.float32),
        "layer_pe": input_layer_pe,
        "intra_layer_pe": input_intra_layer_pe,
        "loss": jp.zeros(input_n, dtype=jp.float32),  # Loss feature for all nodes
    }
    all_nodes_features_list.append(input_nodes)
    current_global_node_idx += input_n
    # --- End Input Layer ---

    # Process gate layers (starting from layer 1)
    for layer_idx_gates, (layer_logits, layer_wires) in enumerate(zip(logits, wires)):
        layer_idx_graph = (
            layer_idx_gates + 1
        )  # Graph layer index starts from 1 for gates
        # layer_logits: (group_n, group_size, 2^arity)
        # layer_wires: (arity, group_n) -> connects previous layer's nodes to this layer
        group_n, group_size, logit_dim = layer_logits.shape
        num_gates_in_layer = group_n * group_size

        # Store the starting index for this gate layer
        layer_start_indices.append(current_global_node_idx)

        # Node features for this layer
        layer_global_indices = jp.arange(
            current_global_node_idx, current_global_node_idx + num_gates_in_layer
        )
        layer_nodes = {
            "layer": jp.full(num_gates_in_layer, layer_idx_graph, dtype=jp.int32),
            "group": jp.repeat(jp.arange(group_n), group_size),
            "gate_id": layer_global_indices,
            "logits": layer_logits.reshape(num_gates_in_layer, logit_dim),
            "hidden": jp.zeros((num_gates_in_layer, circuit_hidden_dim), dtype=jp.float32),
            "loss": jp.zeros(
                num_gates_in_layer, dtype=jp.float32
            ),  # Loss feature for all nodes
        }

        # Add Positional Encodings
        layer_indices_pe = jp.full(num_gates_in_layer, layer_idx_graph, dtype=jp.int32)
        layer_pe = get_positional_encoding(layer_indices_pe, pe_dim, max_val=max_layers)
        intra_layer_indices = jp.arange(num_gates_in_layer, dtype=jp.int32)
        intra_layer_pe = get_positional_encoding(
            intra_layer_indices, pe_dim, max_val=num_gates_in_layer + 1
        )
        layer_nodes["layer_pe"] = layer_pe
        layer_nodes["intra_layer_pe"] = intra_layer_pe

        all_nodes_features_list.append(layer_nodes)

        # Create forward edges
        # Receivers are the gates in the current layer
        current_layer_receivers = jp.repeat(layer_global_indices, arity)

        # Senders are nodes from the previous layer (could be inputs or gates)
        previous_layer_start_idx = layer_start_indices[layer_idx_graph - 1]
        # layer_wires connects the *output* of the previous layer to the *input* of the current layer gates.
        # Indices in layer_wires are relative to the start of the previous layer.
        global_senders_for_layer = previous_layer_start_idx + layer_wires
        tiled_senders = jp.tile(global_senders_for_layer.T, (1, group_size))
        current_layer_senders = tiled_senders.reshape(-1)  # Flatten

        all_forward_senders.append(current_layer_senders)
        all_forward_receivers.append(current_layer_receivers)

        # Update global index for the next layer
        current_global_node_idx += num_gates_in_layer

    # Consolidate Nodes and Edges
    if not all_nodes_features_list:
        # Handle empty circuit case
        return jraph.GraphsTuple(
            nodes={},
            edges=None,
            senders=jp.array([], dtype=jp.int32),
            receivers=jp.array([], dtype=jp.int32),
            n_node=jp.array([0]),
            n_edge=jp.array([0]),
            globals=jp.zeros((2,), dtype=jp.float32),
        )

    # Combine node features from all layers
    # Need to handle potentially missing 'logits' in input layer if we decide not to add zeros
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

    # Combine loss_value and update_steps into a single globals array
    globals_val = jp.array([loss_value, update_steps], dtype=jp.float32)

    # Create and return the GraphsTuple
    graph = jraph.GraphsTuple(
        nodes=all_nodes,
        edges=None,  # No edge features initially
        senders=senders.astype(jp.int32),
        receivers=receivers.astype(jp.int32),
        n_node=jp.array([n_node]),
        n_edge=jp.array([n_edge]),
        globals=globals_val,
    )

    return graph
