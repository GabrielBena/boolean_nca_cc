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
    update_steps: int = 0,
    knockout_strategy: str = "untouched",  # "untouched", "no_receive", "detached"
    faulty_logit_value: float = -10.0,
    gate_knockout_mask: jp.ndarray | None = None,
    perturbation_mask: jp.ndarray | None = None,
) -> jraph.GraphsTuple:
    """
    Construct a jraph.GraphsTuple representation of a boolean circuit, including input nodes.

    Args:
        logits: List of logit tensors per layer. Shape [(group_n, group_size, 2^arity), ...]
        wires: List of wire connection patterns per layer. Shape [(arity, group_n), ...]
                The first element wires[0] connects input nodes to the first gate layer.
        input_n: Number of input nodes/bits for the first layer
        arity: Fan-in for each gate
        circuit_hidden_dim: Dimension of hidden features for nodes
        bidirectional_edges: If True, create edges in both forward and backward directions
        loss_value: Optional scalar value representing the current loss of the circuit.
        update_steps: Number of times this graph has been updated by the GNN.
        knockout_strategy: How to handle knocked-out gates in message passing:
                          - "untouched": Faulty gates participate normally in message passing (recoverable)
                          - "no_receive": Faulty gates can send but not receive messages (persistent damage, signals fault)
                          - "detached": Faulty gates are completely removed from message passing (isolated)
        faulty_logit_value: Value to set for knocked-out gate logits (should be large negative for zero output)
        gate_knockout_mask: Optional mask for permanently knocked-out gates that cannot receive updates.
                           Flat gate mask with shape (total_gates,) where 0.0 = knocked out, 1.0 = active
                           OR layered gate mask as list/tuple of arrays, one per layer with shape (gate_n,)
        perturbation_mask: Optional mask for gates with recoverable perturbations that can still receive updates.
                          Same format as gate_knockout_mask. Perturbed gates have mask value 1.0.

    Returns:
        A jraph.GraphsTuple representing the circuit
    """
    # Process knockout mask if provided
    if gate_knockout_mask is not None:
        from boolean_nca_cc.training.pool.structural_perturbation import ensure_layered_mask

        # Calculate layer sizes from logits
        layer_sizes = [(input_n, 1)]  # Input layer
        for layer_logits in logits:
            group_n, group_size, _ = layer_logits.shape
            gate_n = group_n * group_size
            layer_sizes.append((gate_n, group_size))

        layered_knockout_masks = ensure_layered_mask(gate_knockout_mask, layer_sizes)
    else:
        layered_knockout_masks = None

    # Process perturbation mask if provided
    if perturbation_mask is not None:
        from boolean_nca_cc.training.pool.structural_perturbation import ensure_layered_mask

        # Calculate layer sizes from logits if not already calculated
        if gate_knockout_mask is None:
            layer_sizes = [(input_n, 1)]  # Input layer
            for layer_logits in logits:
                group_n, group_size, _ = layer_logits.shape
                gate_n = group_n * group_size
                layer_sizes.append((gate_n, group_size))

        layered_perturbation_masks = ensure_layered_mask(perturbation_mask, layer_sizes)
    else:
        layered_perturbation_masks = None

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
    input_intra_layer_pe = get_positional_encoding(input_layer_indices, pe_dim, max_val=input_n + 1)

    # Input layer masks (always active for inputs)
    input_knockout_mask = (
        jp.ones(input_n, dtype=jp.float32)
        if layered_knockout_masks is None
        else layered_knockout_masks[0]
    )
    input_perturbation_mask = (
        jp.zeros(input_n, dtype=jp.float32)
        if layered_perturbation_masks is None
        else layered_perturbation_masks[0]
    )

    input_nodes = {
        "layer": jp.zeros(input_n, dtype=jp.int32),  # Input layer is layer 0
        "group": jp.zeros(input_n, dtype=jp.int32),  # No groups for inputs
        "gate_id": input_layer_indices,
        "logits": jp.zeros((input_n, 2**arity), dtype=jp.float32),  # Inputs have no logits
        "hidden": jp.zeros((input_n, circuit_hidden_dim), dtype=jp.float32),
        "layer_pe": input_layer_pe,
        "intra_layer_pe": input_intra_layer_pe,
        "loss": jp.zeros(input_n, dtype=jp.float32),  # Loss feature for all nodes
        "gate_knockout_mask": input_knockout_mask,  # Permanent knockout mask
        "perturbation_mask": input_perturbation_mask,  # Recoverable perturbation mask
    }
    all_nodes_features_list.append(input_nodes)
    current_global_node_idx += input_n
    # --- End Input Layer ---

    # Process gate layers (starting from layer 1)
    for layer_idx_gates, (layer_logits, layer_wires) in enumerate(zip(logits, wires, strict=False)):
        layer_idx_graph = layer_idx_gates + 1  # Graph layer index starts from 1 for gates
        # layer_logits: (group_n, group_size, 2^arity)
        # layer_wires: (arity, group_n) -> connects previous layer's nodes to this layer
        group_n, group_size, logit_dim = layer_logits.shape
        num_gates_in_layer = group_n * group_size

        # Store the starting index for this gate layer
        layer_start_indices.append(current_global_node_idx)

        # Get masks for this layer
        layer_knockout_mask = (
            jp.ones(num_gates_in_layer, dtype=jp.float32)
            if layered_knockout_masks is None
            else layered_knockout_masks[layer_idx_graph]
        )
        layer_perturbation_mask = (
            jp.zeros(num_gates_in_layer, dtype=jp.float32)
            if layered_perturbation_masks is None
            else layered_perturbation_masks[layer_idx_graph]
        )

        # Apply faulty logits to knocked-out gates if knockout mask is provided
        if layered_knockout_masks is not None:
            from boolean_nca_cc.training.pool.structural_perturbation import (
                create_faulty_gate_logits,
            )

            # Set faulty logits for knocked-out gates
            layer_logits_processed = create_faulty_gate_logits(
                layer_logits, layer_knockout_mask, faulty_value=faulty_logit_value
            )
        else:
            # Use original logits if no knockout mask provided
            layer_logits_processed = layer_logits

        # Node features for this layer
        layer_global_indices = jp.arange(
            current_global_node_idx, current_global_node_idx + num_gates_in_layer
        )
        layer_nodes = {
            "layer": jp.full(num_gates_in_layer, layer_idx_graph, dtype=jp.int32),
            "group": jp.repeat(jp.arange(group_n), group_size),
            "gate_id": layer_global_indices,
            "logits": layer_logits_processed.reshape(num_gates_in_layer, logit_dim),
            "hidden": jp.zeros((num_gates_in_layer, circuit_hidden_dim), dtype=jp.float32),
            "loss": jp.zeros(num_gates_in_layer, dtype=jp.float32),  # Loss feature for all nodes
            "gate_knockout_mask": layer_knockout_mask,  # Permanent knockout mask
            "perturbation_mask": layer_perturbation_mask,  # Recoverable perturbation mask
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

        # Create forward edges with knockout strategy filtering
        # Receivers are the gates in the current layer
        current_layer_receivers = jp.repeat(layer_global_indices, arity)

        # Senders are nodes from the previous layer (could be inputs or gates)
        previous_layer_start_idx = layer_start_indices[layer_idx_graph - 1]
        # layer_wires connects the *output* of the previous layer to the *input* of the current layer gates.
        # Indices in layer_wires are relative to the start of the previous layer.
        global_senders_for_layer = previous_layer_start_idx + layer_wires
        tiled_senders = jp.tile(global_senders_for_layer.T, (1, group_size))
        current_layer_senders = tiled_senders.reshape(-1)  # Flatten

        # Apply knockout strategy filtering
        if layered_knockout_masks is not None and knockout_strategy != "untouched":
            # Get mask for current layer gates (receivers)
            current_layer_mask = layered_knockout_masks[layer_idx_graph]
            # Expand mask to match the number of edges (arity edges per gate)
            expanded_receiver_mask = jp.repeat(current_layer_mask, arity)

            # Get mask for previous layer gates (senders)
            previous_layer_mask = layered_knockout_masks[layer_idx_graph - 1]
            # Create sender mask by indexing with wire connections
            sender_mask = previous_layer_mask[layer_wires]  # Shape: (arity, group_n)
            expanded_sender_mask = jp.tile(sender_mask.T, (1, group_size)).reshape(-1)

            if knockout_strategy == "no_receive":
                # Remove edges where receiver is knocked out (faulty gates can't receive updates)
                edge_mask = expanded_receiver_mask == 1.0
            elif knockout_strategy == "detached":
                # Remove edges where either sender or receiver is knocked out
                edge_mask = (expanded_receiver_mask == 1.0) & (expanded_sender_mask == 1.0)
            else:
                # This shouldn't happen, but default to keeping all edges
                edge_mask = jp.ones_like(expanded_receiver_mask, dtype=bool)

            # Filter edges based on the mask
            current_layer_senders = current_layer_senders[edge_mask]
            current_layer_receivers = current_layer_receivers[edge_mask]

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
    all_nodes = jax.tree.map(lambda *xs: jp.concatenate(xs, axis=0), *all_nodes_features_list)

    # Process edges
    if all_forward_senders:
        forward_senders = jp.concatenate(all_forward_senders)
        forward_receivers = jp.concatenate(all_forward_receivers)

        if bidirectional_edges:
            # Create backward edges by swapping senders and receivers
            backward_senders = forward_receivers
            backward_receivers = forward_senders

            # Apply knockout strategy filtering to backward edges if needed
            if layered_knockout_masks is not None and knockout_strategy != "untouched":
                from boolean_nca_cc.training.pool.structural_perturbation import ensure_flat_mask

                # Calculate layer sizes for flat mask conversion
                layer_sizes_for_flat = [(input_n, 1)]  # Input layer
                for layer_logits in logits:
                    group_n, group_size, _ = layer_logits.shape
                    gate_n = group_n * group_size
                    layer_sizes_for_flat.append((gate_n, group_size))

                flat_knockout_mask = ensure_flat_mask(gate_knockout_mask, layer_sizes_for_flat)
                gate_active = flat_knockout_mask == 1.0

                if len(backward_receivers) > 0:  # Only filter if there are edges
                    if knockout_strategy == "no_receive":
                        # For backward edges, remove edges where receiver is knocked out
                        backward_edge_mask = gate_active[backward_receivers]
                    elif knockout_strategy == "detached":
                        # For backward edges, remove edges where either sender or receiver is knocked out
                        backward_edge_mask = (
                            gate_active[backward_senders] & gate_active[backward_receivers]
                        )
                    else:
                        # This shouldn't happen, but default to keeping all edges
                        backward_edge_mask = jp.ones(len(backward_senders), dtype=bool)

                    # Filter backward edges
                    backward_senders = backward_senders[backward_edge_mask]
                    backward_receivers = backward_receivers[backward_edge_mask]

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
    globals_val = jp.array([float(loss_value), float(update_steps)], dtype=jp.float32)

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
