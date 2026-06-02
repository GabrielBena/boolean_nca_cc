"""
Graph building utilities for boolean circuits.

This module provides functions for constructing graph representations
of boolean circuits for use with graph neural networks.

The main function `build_graph` creates a jraph.GraphsTuple from circuit
logits and wires, with support for:
- Bidirectional message passing edges
- Neighboring connections within layers
- Gate knockout masks for permanent damage simulation
- Positional encodings for layer and intra-layer positions
"""

from typing import NamedTuple

import jax
import jax.numpy as jp
import jraph

from boolean_nca_cc.utils.positional_encoding import (
    compute_dag_distance_pe,
    compute_rwse,
    get_positional_encoding,
)


class GraphGlobals(NamedTuple):
    """
    Global features for circuit graphs.

    This NamedTuple provides a consistent structure for graph globals
    that works well with JAX pytree operations (vmap, jit, etc.).

    Base fields (used by all models):
        loss: Current circuit loss value
        update_steps: Number of optimization steps applied

    Perceiver-specific fields (optional, for cross-attention to data):
        x_data: Input data batch [N_samples, N_input_bits] or None
        y_data: Target output batch [N_samples, N_output_bits] or None
        residuals: Current prediction errors [N_samples, N_output_bits] or None
        subsample_key: Base PRNG key for stochastic token subsampling or None.
            Per-step keys are derived via jax.random.fold_in(key, update_steps),
            so this key stays constant across the scan while producing unique
            subsampling patterns at each step.
    """

    loss: float | jp.ndarray
    update_steps: int | jp.ndarray
    # Perceiver-specific fields
    x_data: jp.ndarray | None = None
    y_data: jp.ndarray | None = None
    residuals: jp.ndarray | None = None
    subsample_key: jp.ndarray | None = None


def build_graph(
    logits: list[jp.ndarray],
    wires: list[jp.ndarray],
    input_n: int,
    arity: int,
    circuit_hidden_dim: int,
    bidirectional_edges: bool = True,
    neighboring_connections: bool = False,
    loss_value: jp.ndarray | int = 0,
    update_steps: int = 0,
    faulty_logit_value: float = -10.0,
    gate_knockout_mask: jp.ndarray | list[jp.ndarray] | None = None,
    positional_encoding_max_val: float = 10000.0,
    use_dist_pe: bool = False,
    use_rwse: bool = False,
    rwse_k: int = 8,
) -> jraph.GraphsTuple:
    """
    Construct a jraph.GraphsTuple representation of a boolean circuit.

    Creates a graph where:
    - Nodes represent gates (and input nodes)
    - Edges represent wire connections between gates
    - Node features include logits, hidden state, positional encodings, and knockout mask

    Knocked-out gates (gate_knockout_mask == 0.0):
    - Have faulty logits set to produce zero output
    - Cannot receive model updates (blocked in node_update and self_attention)
    - Can still send messages to neighbors (edges preserved)

    Args:
        logits: List of logit tensors per layer. Shape [(group_n, group_size, 2^arity), ...]
        wires: List of wire connection patterns per layer. Shape [(arity, group_n), ...]
               The first element wires[0] connects input nodes to the first gate layer.
        input_n: Number of input nodes/bits for the first layer
        arity: Fan-in for each gate
        circuit_hidden_dim: Dimension of hidden features for nodes
        bidirectional_edges: If True, create edges in both forward and backward directions
        neighboring_connections: If True, create edges between adjacent gates within same layer
        loss_value: Scalar value representing the current loss of the circuit
        update_steps: Number of times this graph has been updated by the model
        faulty_logit_value: Value to set for knocked-out gate logits (large negative for zero output)
        gate_knockout_mask: Optional mask for knocked-out gates.
                           Flat format: shape (total_gates,)
                           Layered format: list of arrays, one per layer with shape (gate_n,)
                           Values: 0.0 = knocked out, 1.0 = active
        positional_encoding_max_val: Maximum value for positional encoding frequency calculation
        use_dist_pe: If True, add ``nodes["dist_pe"]`` — sinusoidal encoding of
            ``(dist_from_input, dist_to_output)`` on the directed DAG. Total dim
            ``circuit_hidden_dim`` (== ``2 * (circuit_hidden_dim // 2)``). Wire-dependent;
            must be refreshed when wires shuffle (see ``training/evaluation.py``).
        use_rwse: If True, add ``nodes["rwse"]`` — Random Walk Structural Encoding.
            Per-node K-vector of return probabilities. Wire-dependent.
        rwse_k: Walk length / RWSE feature dim when ``use_rwse`` is enabled.

    Returns:
        A jraph.GraphsTuple representing the circuit with:
        - nodes: Dict with 'layer', 'group', 'gate_id', 'logits', 'hidden',
                'layer_pe', 'intra_layer_pe', 'loss', 'gate_knockout_mask'
        - edges: None (no edge features)
        - senders, receivers: Edge connectivity
        - n_node, n_edge: Graph shape info
        - globals: Array of [loss_value, update_steps]
    """
    # Calculate layer sizes for mask conversion
    layer_sizes = [(input_n, 1)]  # Input layer
    for layer_logits in logits:
        group_n, group_size, _ = layer_logits.shape
        gate_n = group_n * group_size
        layer_sizes.append((gate_n, group_size))

    # Process knockout mask if provided
    if gate_knockout_mask is not None:
        layered_knockout_masks = _ensure_layered_mask(gate_knockout_mask, layer_sizes)
    else:
        layered_knockout_masks = None

    all_nodes_features_list = []
    all_forward_senders = []
    all_forward_receivers = []
    current_global_node_idx = 0
    layer_start_indices = []
    pe_dim = circuit_hidden_dim

    # Total number of gate layers (input layer is layer 0, last gate layer is total_layers)
    # Used to normalize layer_pe to depth fraction [0.0, 1.0] for scale-free generalization
    total_layers = len(logits)  # number of gate layers; max layer index = total_layers

    # --- Input Layer Nodes ---
    layer_start_indices.append(current_global_node_idx)
    input_layer_indices = jp.arange(input_n)
    # Normalized layer PE: input layer is at depth 0.0
    input_layer_pe = get_positional_encoding(
        jp.zeros(input_n, dtype=jp.float32), pe_dim, max_val=positional_encoding_max_val
    )
    input_intra_layer_pe = get_positional_encoding(
        input_layer_indices, pe_dim, max_val=positional_encoding_max_val
    )

    # Input layer mask (always active for inputs)
    input_knockout_mask = (
        jp.ones(input_n, dtype=jp.float32)
        if layered_knockout_masks is None
        else layered_knockout_masks[0]
    )

    input_nodes = {
        "layer": jp.zeros(input_n, dtype=jp.int32),
        "group": jp.zeros(input_n, dtype=jp.int32),
        "gate_id": input_layer_indices,
        "logits": jp.zeros((input_n, 2**arity), dtype=jp.float32),
        "hidden": jp.zeros((input_n, circuit_hidden_dim), dtype=jp.float32),
        "layer_pe": input_layer_pe,
        "intra_layer_pe": input_intra_layer_pe,
        "loss": jp.zeros(input_n, dtype=jp.float32),
        "gate_knockout_mask": input_knockout_mask,
    }
    all_nodes_features_list.append(input_nodes)
    current_global_node_idx += input_n

    # --- Gate Layers ---
    for layer_idx_gates, (layer_logits, layer_wires) in enumerate(zip(logits, wires, strict=False)):
        layer_idx_graph = layer_idx_gates + 1  # Graph layer index starts from 1 for gates
        group_n, group_size, logit_dim = layer_logits.shape
        num_gates_in_layer = group_n * group_size

        layer_start_indices.append(current_global_node_idx)

        # Get knockout mask for this layer
        layer_knockout_mask = (
            jp.ones(num_gates_in_layer, dtype=jp.float32)
            if layered_knockout_masks is None
            else layered_knockout_masks[layer_idx_graph]
        )

        # Apply faulty logits to knocked-out gates
        if layered_knockout_masks is not None:
            from boolean_nca_cc.training.pool.structural_perturbation import (
                create_faulty_gate_logits,
            )

            layer_logits_processed = create_faulty_gate_logits(
                layer_logits, layer_knockout_mask, faulty_value=faulty_logit_value
            )
        else:
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
            "loss": jp.zeros(num_gates_in_layer, dtype=jp.float32),
            "gate_knockout_mask": layer_knockout_mask,
        }

        # Add Positional Encodings
        # Normalized layer PE: depth fraction in [0.0, 1.0] for scale-free generalization
        # Input layer = 0.0, output layer = 1.0, regardless of total circuit depth
        normalized_depth = layer_idx_graph / total_layers if total_layers > 0 else 0.0
        # Scale to [0, max_val] range for good sinusoidal frequency spread
        scaled_depth = jp.full(
            num_gates_in_layer, normalized_depth * positional_encoding_max_val, dtype=jp.float32
        )
        layer_pe = get_positional_encoding(
            scaled_depth, pe_dim, max_val=positional_encoding_max_val
        )
        intra_layer_indices = jp.arange(num_gates_in_layer, dtype=jp.int32)
        intra_layer_pe = get_positional_encoding(
            intra_layer_indices, pe_dim, max_val=positional_encoding_max_val
        )
        layer_nodes["layer_pe"] = layer_pe
        layer_nodes["intra_layer_pe"] = intra_layer_pe

        all_nodes_features_list.append(layer_nodes)

        # Create forward edges
        current_layer_receivers = jp.repeat(layer_global_indices, arity)
        previous_layer_start_idx = layer_start_indices[layer_idx_graph - 1]
        global_senders_for_layer = previous_layer_start_idx + layer_wires
        tiled_senders = jp.tile(global_senders_for_layer.T, (1, group_size))
        current_layer_senders = tiled_senders.reshape(-1)

        all_forward_senders.append(current_layer_senders)
        all_forward_receivers.append(current_layer_receivers)

        current_global_node_idx += num_gates_in_layer

    # Handle empty circuit case
    if not all_nodes_features_list:
        return jraph.GraphsTuple(
            nodes={},
            edges=None,
            senders=jp.array([], dtype=jp.int32),
            receivers=jp.array([], dtype=jp.int32),
            n_node=jp.array([0]),
            n_edge=jp.array([0]),
            globals=GraphGlobals(loss=0.0, update_steps=0),
        )

    # Combine node features from all layers
    all_nodes = jax.tree.map(lambda *xs: jp.concatenate(xs, axis=0), *all_nodes_features_list)

    # Process edges
    if all_forward_senders:
        forward_senders = jp.concatenate(all_forward_senders)
        forward_receivers = jp.concatenate(all_forward_receivers)

        if bidirectional_edges:
            backward_senders = forward_receivers
            backward_receivers = forward_senders
            senders = jp.concatenate([forward_senders, backward_senders])
            receivers = jp.concatenate([forward_receivers, backward_receivers])
        else:
            senders = forward_senders
            receivers = forward_receivers
    else:
        senders = jp.array([], dtype=jp.int32)
        receivers = jp.array([], dtype=jp.int32)

    # Add neighboring connections within layers if requested
    if neighboring_connections:
        neighboring_senders = []
        neighboring_receivers = []

        for layer_idx_graph in range(1, len(layer_start_indices)):
            layer_start_idx = layer_start_indices[layer_idx_graph]
            if layer_idx_graph < len(logits) + 1:
                layer_logits = logits[layer_idx_graph - 1]
                group_n, group_size, _ = layer_logits.shape
                num_gates_in_layer = group_n * group_size

                if num_gates_in_layer > 1:
                    for i in range(num_gates_in_layer - 1):
                        neighboring_senders.extend([layer_start_idx + i, layer_start_idx + i + 1])
                        neighboring_receivers.extend([layer_start_idx + i + 1, layer_start_idx + i])

        if neighboring_senders:
            neighboring_senders = jp.array(neighboring_senders)
            neighboring_receivers = jp.array(neighboring_receivers)

            if len(senders) > 0:
                senders = jp.concatenate([senders, neighboring_senders])
                receivers = jp.concatenate([receivers, neighboring_receivers])
            else:
                senders = neighboring_senders
                receivers = neighboring_receivers

    n_node = current_global_node_idx
    n_edge = len(senders)

    # ── Optional graph-based positional encodings ────────────────────────
    # Both are wire-dependent: they must be recomputed when wires shuffle.
    # The wire-shuffle path in training/evaluation.py refreshes these
    # alongside (senders, receivers); see ``refresh_topology`` there.
    if use_dist_pe:
        # Forward edges only — directional BFS needs the DAG, not the symmetrized graph.
        if all_forward_senders:
            forward_senders_arr = jp.concatenate(all_forward_senders).astype(jp.int32)
            forward_receivers_arr = jp.concatenate(all_forward_receivers).astype(jp.int32)
        else:
            forward_senders_arr = jp.array([], dtype=jp.int32)
            forward_receivers_arr = jp.array([], dtype=jp.int32)
        # Half the budget for each side so total matches the existing layer_pe size.
        per_side_dim = circuit_hidden_dim // 2
        # ``len(layer_sizes)`` iterations is always sufficient on a strict layered DAG
        # (one relaxation step propagates one layer).
        all_nodes["dist_pe"] = compute_dag_distance_pe(
            forward_senders_arr,
            forward_receivers_arr,
            all_nodes["layer"],
            n_node=n_node,
            n_iterations=len(layer_sizes),
            pe_dim=per_side_dim,
            max_val=positional_encoding_max_val,
        )

    if use_rwse:
        # RWSE on the bidirectional graph used by attention — captures the topology
        # the message-passing layers actually traverse.
        all_nodes["rwse"] = compute_rwse(
            senders.astype(jp.int32),
            receivers.astype(jp.int32),
            n_node=n_node,
            k=rwse_k,
        )

    # Store globals as NamedTuple for consistent access across all models
    globals_tuple = GraphGlobals(
        loss=float(loss_value),
        update_steps=int(update_steps),
    )

    return jraph.GraphsTuple(
        nodes=all_nodes,
        edges=None,
        senders=senders.astype(jp.int32),
        receivers=receivers.astype(jp.int32),
        n_node=jp.array([n_node]),
        n_edge=jp.array([n_edge]),
        globals=globals_tuple,
    )


def compute_circuit_edges(
    wires: list[jp.ndarray],
    layer_sizes: list[tuple[int, int]] | tuple[tuple[int, int], ...],
    arity: int,
    bidirectional_edges: bool = True,
    neighboring_connections: bool = False,
) -> tuple[jp.ndarray, jp.ndarray]:
    """Recompute ``(senders, receivers)`` for a circuit graph from its wires.

    Mirrors the edge-construction in :func:`build_graph` exactly so that
    node features can be reused while topology is refreshed mid-scan
    (e.g. after a wire shuffle).

    Edge layout (must match build_graph):
        [forward_senders, backward_senders?, neighboring_senders?]

    Only the wire-dependent values change between calls — output shapes
    are static given fixed ``layer_sizes`` / topology flags, so the result
    is JIT-friendly.

    Args:
        wires: List of wire arrays, one per gate layer. Shape
            ``(arity, group_n)``, values in ``[0, prev_layer_n)``.
        layer_sizes: Full topology including the input layer first
            (``[(input_n, 1), (gate_n_1, group_size_1), ...]``).
        arity: Number of inputs per gate.
        bidirectional_edges: Mirror forward edges as backward edges.
        neighboring_connections: Add intra-layer adjacent edges.

    Returns:
        Tuple ``(senders, receivers)`` as int32 arrays.
    """
    all_forward_senders = []
    all_forward_receivers = []
    layer_start_indices = [0]
    current_global_node_idx = layer_sizes[0][0]

    for layer_idx, ((gate_n, group_size), layer_wires) in enumerate(
        zip(layer_sizes[1:], wires, strict=True)
    ):
        layer_start_indices.append(current_global_node_idx)
        layer_global_indices = jp.arange(
            current_global_node_idx, current_global_node_idx + gate_n
        )
        current_layer_receivers = jp.repeat(layer_global_indices, arity)
        previous_layer_start_idx = layer_start_indices[layer_idx]
        global_senders_for_layer = previous_layer_start_idx + layer_wires
        tiled_senders = jp.tile(global_senders_for_layer.T, (1, group_size))
        current_layer_senders = tiled_senders.reshape(-1)

        all_forward_senders.append(current_layer_senders)
        all_forward_receivers.append(current_layer_receivers)
        current_global_node_idx += gate_n

    if not all_forward_senders:
        return jp.array([], dtype=jp.int32), jp.array([], dtype=jp.int32)

    forward_senders = jp.concatenate(all_forward_senders)
    forward_receivers = jp.concatenate(all_forward_receivers)

    if bidirectional_edges:
        senders = jp.concatenate([forward_senders, forward_receivers])
        receivers = jp.concatenate([forward_receivers, forward_senders])
    else:
        senders = forward_senders
        receivers = forward_receivers

    if neighboring_connections:
        neighboring_senders = []
        neighboring_receivers = []
        for layer_idx in range(1, len(layer_start_indices)):
            layer_start_idx = layer_start_indices[layer_idx]
            gate_n = layer_sizes[layer_idx][0]
            if gate_n > 1:
                for i in range(gate_n - 1):
                    neighboring_senders.extend([layer_start_idx + i, layer_start_idx + i + 1])
                    neighboring_receivers.extend([layer_start_idx + i + 1, layer_start_idx + i])
        if neighboring_senders:
            senders = jp.concatenate([senders, jp.array(neighboring_senders, dtype=jp.int32)])
            receivers = jp.concatenate([receivers, jp.array(neighboring_receivers, dtype=jp.int32)])

    return senders.astype(jp.int32), receivers.astype(jp.int32)


def refresh_wire_dependent_pe(
    nodes: dict,
    wires: list[jp.ndarray],
    senders: jp.ndarray,
    receivers: jp.ndarray,
    layer_sizes: tuple,
    arity: int,
    positional_encoding_max_val: float = 10000.0,
) -> dict:
    """Recompute wire-dependent PE entries (``dist_pe``, ``rwse``) for a new wiring.

    Mirrors the PE-computation in :func:`build_graph` exactly so that node
    features can be reused while topology is refreshed mid-scan (e.g. after
    a wire shuffle). Only refreshes keys that already exist in ``nodes`` —
    if a PE was never built into the graph, the corresponding refresh is a
    no-op. Returns a new dict; the input is not mutated.

    Args:
        nodes: Current node-feature dict (read-only).
        wires: Updated wire list (one array per gate layer).
        senders: Updated bidirectional senders (for RWSE).
        receivers: Updated bidirectional receivers (for RWSE).
        layer_sizes: Full topology tuple including input layer first.
        arity: Number of inputs per gate.
        positional_encoding_max_val: Frequency basis for ``dist_pe``. Must match
            the value used at :func:`build_graph` time.

    Returns:
        New node-feature dict with refreshed PE entries.
    """
    new_nodes = dict(nodes)
    n_node = nodes["layer"].shape[0]
    if "dist_pe" in nodes:
        fwd_s, fwd_r = compute_circuit_edges(
            wires, layer_sizes, arity,
            bidirectional_edges=False, neighboring_connections=False,
        )
        pe_dim_per_side = nodes["dist_pe"].shape[1] // 2
        new_nodes["dist_pe"] = compute_dag_distance_pe(
            fwd_s, fwd_r, nodes["layer"],
            n_node=n_node,
            n_iterations=len(layer_sizes),
            pe_dim=pe_dim_per_side,
            max_val=positional_encoding_max_val,
        )
    if "rwse" in nodes:
        k = nodes["rwse"].shape[1]
        new_nodes["rwse"] = compute_rwse(
            senders.astype(jp.int32),
            receivers.astype(jp.int32),
            n_node=n_node,
            k=k,
        )
    return new_nodes


def _ensure_layered_mask(
    gate_mask: jp.ndarray | list[jp.ndarray],
    layer_sizes: list[tuple[int, int]],
) -> list[jp.ndarray]:
    """
    Ensure gate mask is in layered format.

    Args:
        gate_mask: Flat or layered mask
        layer_sizes: Layer sizes for conversion

    Returns:
        Layered mask (list of arrays)
    """
    if isinstance(gate_mask, list | tuple):
        return list(gate_mask)
    else:
        # Flat format - convert to layered
        from boolean_nca_cc.training.pool.structural_perturbation import flat_to_layered_mask

        return flat_to_layered_mask(gate_mask, layer_sizes)
