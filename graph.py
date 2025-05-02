import jax
import jax.numpy as jp
import numpy as np
from functools import partial
from model import run_circuit, run_layer
from collections import namedtuple
import optax
from training import res2loss, binary_cross_entropy

# Define graph structure for the boolean circuit
GraphState = namedtuple(
    "GraphState", ["nodes", "edges", "node_features", "edge_features"]
)


def build_circuit_graph(logits, wires):
    """
    Build a graph representation of the boolean circuit.

    Args:
        logits: List of logits for each layer, shape [layer_n][group_n, group_size, 2^arity]
        wires: List of wire connection patterns, shape [layer_n][arity, group_n]

    Returns:
        GraphState containing node and edge information
    """
    # Calculate dimensions
    layer_n = len(logits)
    nodes_per_layer = []

    # First layer (inputs) has no logits, infer size from wires
    input_n = max(jp.max(wires[0]) + 1, 1)
    nodes_per_layer.append(input_n)

    # Count nodes in each hidden/output layer
    for layer_idx, layer_logits in enumerate(logits):
        group_n, group_size, _ = layer_logits.shape
        nodes_per_layer.append(group_n * group_size)

    # Initialize node features
    node_features = []
    cum_nodes = 0

    # Input layer nodes (no learnable parameters)
    input_features = []
    for i in range(nodes_per_layer[0]):
        # For input nodes: only store position information
        input_features.append(
            {
                "node_idx": cum_nodes + i,
                "layer_idx": 0,
                "group_idx": None,
                "gate_idx": i,
                "logits": None,  # Input nodes have no logits
            }
        )
    node_features.append(input_features)
    cum_nodes += nodes_per_layer[0]

    # Hidden and output layer nodes
    for layer_idx, layer_logits in enumerate(logits):
        group_n, group_size, lut_size = layer_logits.shape
        layer_features = []

        for group_idx in range(group_n):
            for gate_idx in range(group_size):
                node_idx = cum_nodes + group_idx * group_size + gate_idx
                # Store node information and parameters
                layer_features.append(
                    {
                        "node_idx": node_idx,
                        "layer_idx": layer_idx + 1,  # +1 because inputs are layer 0
                        "group_idx": group_idx,
                        "gate_idx": gate_idx,
                        "logits": layer_logits[
                            group_idx, gate_idx
                        ],  # LUT logits for this gate
                    }
                )

        node_features.append(layer_features)
        cum_nodes += group_n * group_size

    # Create edges based on wiring
    edges = []
    edge_features = []

    for layer_idx, layer_wires in enumerate(wires):
        arity, group_n = layer_wires.shape
        group_size = nodes_per_layer[layer_idx + 1] // group_n

        for group_idx in range(group_n):
            # Get input nodes for this group
            input_nodes = layer_wires[:, group_idx]

            # Connect each input to all gates in the group
            for gate_idx in range(group_size):
                # Target node in current layer
                target_idx = (
                    sum(nodes_per_layer[: layer_idx + 1])
                    + group_idx * group_size
                    + gate_idx
                )

                # Connect to each input node
                for input_port, source_idx in enumerate(input_nodes):
                    # Create edge from source (input) to target (gate)
                    edges.append((int(source_idx), target_idx))
                    edge_features.append(
                        {
                            "source_layer": layer_idx,
                            "target_layer": layer_idx + 1,
                            "input_port": input_port,  # Which input of the gate
                        }
                    )

    return GraphState(
        nodes=nodes_per_layer,
        edges=jp.array(edges),
        node_features=node_features,
        edge_features=edge_features,
    )


@jax.jit
def node_encoder_mlp(node_features, params):
    """
    Simple MLP to encode node features.

    Args:
        node_features: Node features to encode
        params: MLP parameters [weights1, bias1, weights2, bias2]

    Returns:
        Encoded node features
    """
    w1, b1, w2, b2 = params
    hidden = jp.tanh(jp.dot(node_features, w1) + b1)
    output = jp.dot(hidden, w2) + b2
    return output


@jax.jit
def edge_encoder_mlp(edge_features, params):
    """
    Simple MLP to encode edge features.

    Args:
        edge_features: Edge features to encode
        params: MLP parameters [weights1, bias1, weights2, bias2]

    Returns:
        Encoded edge features
    """
    w1, b1, w2, b2 = params
    hidden = jp.tanh(jp.dot(edge_features, w1) + b1)
    output = jp.dot(hidden, w2) + b2
    return output


@jax.jit
def message_mlp(source_features, edge_features, params):
    """
    MLP to compute messages from source nodes to target nodes.

    Args:
        source_features: Source node features
        edge_features: Edge features
        params: MLP parameters [weights1, bias1, weights2, bias2]

    Returns:
        Message features
    """
    combined = jp.concatenate([source_features, edge_features])
    w1, b1, w2, b2 = params
    hidden = jp.tanh(jp.dot(combined, w1) + b1)
    output = jp.dot(hidden, w2) + b2
    return output


@jax.jit
def update_mlp(node_features, aggregated_messages, params):
    """
    MLP to update node features based on aggregated messages.

    Args:
        node_features: Current node features
        aggregated_messages: Aggregated incoming messages
        params: MLP parameters [weights1, bias1, weights2, bias2]

    Returns:
        Updated node features
    """
    combined = jp.concatenate([node_features, aggregated_messages])
    w1, b1, w2, b2 = params
    hidden = jp.tanh(jp.dot(combined, w1) + b1)
    output = jp.dot(hidden, w2) + b2
    return output


def init_gnn_params(node_feat_dim, edge_feat_dim, hidden_dim, message_dim, key):
    """
    Initialize parameters for the GNN MLPs.

    Args:
        node_feat_dim: Dimension of node features
        edge_feat_dim: Dimension of edge features
        hidden_dim: Dimension of hidden layers
        message_dim: Dimension of messages
        key: JAX PRNG key

    Returns:
        Dictionary of initialized parameters
    """
    key1, key2, key3, key4 = jax.random.split(key, 4)

    # Node encoder MLP
    node_encoder_params = [
        jax.random.normal(key1, (node_feat_dim, hidden_dim)) * 0.1,
        jp.zeros(hidden_dim),
        jax.random.normal(key1, (hidden_dim, node_feat_dim)) * 0.1,
        jp.zeros(node_feat_dim),
    ]

    # Edge encoder MLP
    edge_encoder_params = [
        jax.random.normal(key2, (edge_feat_dim, hidden_dim)) * 0.1,
        jp.zeros(hidden_dim),
        jax.random.normal(key2, (hidden_dim, edge_feat_dim)) * 0.1,
        jp.zeros(edge_feat_dim),
    ]

    # Message MLP
    message_params = [
        jax.random.normal(key3, (node_feat_dim + edge_feat_dim, hidden_dim)) * 0.1,
        jp.zeros(hidden_dim),
        jax.random.normal(key3, (hidden_dim, message_dim)) * 0.1,
        jp.zeros(message_dim),
    ]

    # Update MLP
    update_params = [
        jax.random.normal(key4, (node_feat_dim + message_dim, hidden_dim)) * 0.1,
        jp.zeros(hidden_dim),
        jax.random.normal(key4, (hidden_dim, node_feat_dim)) * 0.1,
        jp.zeros(node_feat_dim),
    ]

    return {
        "node_encoder": node_encoder_params,
        "edge_encoder": edge_encoder_params,
        "message": message_params,
        "update": update_params,
    }


@partial(jax.jit, static_argnums=(4,))
def message_passing_step(
    graph_state, encoded_nodes, encoded_edges, gnn_params, steps=1
):
    """
    Perform message passing on the circuit graph.

    Args:
        graph_state: GraphState with node and edge information
        encoded_nodes: Encoded node features
        encoded_edges: Encoded edge features
        gnn_params: Parameters for the GNN MLPs
        steps: Number of message passing steps to perform

    Returns:
        Updated encoded node features
    """
    edges = graph_state.edges

    def single_step(nodes, _):
        # Compute messages for each edge
        messages = jax.vmap(
            lambda edge, edge_feat: message_mlp(
                nodes[edge[0]], edge_feat, gnn_params["message"]
            )
        )(edges, encoded_edges)

        # Aggregate messages for each node using segment_sum
        num_nodes = nodes.shape[0]
        target_indices = edges[:, 1]
        aggregated = jax.ops.segment_sum(
            messages, target_indices, num_segments=num_nodes
        )

        # Update each node based on its current features and aggregated messages
        updated_nodes = jax.vmap(
            lambda node, agg_msg: update_mlp(node, agg_msg, gnn_params["update"])
        )(nodes, aggregated)

        return updated_nodes, None

    # Run multiple message passing steps if requested
    final_nodes, _ = jax.lax.scan(single_step, encoded_nodes, None, length=steps)

    return final_nodes


def extract_logits_from_nodes(graph_state, encoded_nodes):
    """
    Extract logits parameters from node embeddings using vectorized operations.

    Args:
        graph_state: GraphState with node information (including original structure)
        encoded_nodes: Encoded node features after message passing

    Returns:
        Updated logits for each layer (list of JAX arrays)
    """
    node_info_list = graph_state.node_features  # This is a list of lists of dicts
    updated_logits_list = []
    start_node_idx = 0

    # Pre-calculate lut_size from the first non-input layer's node info
    first_layer_nodes = node_info_list[1]  # Assuming at least one non-input layer
    lut_size = 0
    if first_layer_nodes:
        lut_size = len(first_layer_nodes[0]["logits"])

    # Skip input layer (layer 0)
    start_node_idx += len(node_info_list[0])

    for layer_idx in range(1, len(node_info_list)):
        layer_nodes_info = node_info_list[layer_idx]
        num_nodes_in_layer = len(layer_nodes_info)

        if num_nodes_in_layer == 0:
            updated_logits_list.append(
                jp.zeros((0, 0, lut_size))
            )  # Handle empty layers
            continue

        # Determine layer dimensions from the first node (assuming consistency)
        group_n = max(node["group_idx"] for node in layer_nodes_info) + 1
        group_size = max(node["gate_idx"] for node in layer_nodes_info) + 1

        # Get the slice of encoded_nodes corresponding to this layer
        layer_encoded_nodes = jax.lax.dynamic_slice_in_dim(
            encoded_nodes, start_node_idx, num_nodes_in_layer, axis=0
        )

        # Extract the part corresponding to logits (assuming logits are first part)
        # Note: This assumes the projection MLP/logic places logits correctly
        layer_logits_flat = layer_encoded_nodes[:, :lut_size]

        # Reshape the flat logits into the desired (group_n, group_size, lut_size)
        # Assumes nodes are ordered correctly (group_idx then gate_idx) in build_circuit_graph
        try:
            layer_logits = layer_logits_flat.reshape((group_n, group_size, lut_size))
        except ValueError as e:
            # Provide more context on reshape error
            print(f"Error reshaping layer {layer_idx}:")
            print(f"  Expected shape: {(group_n, group_size, lut_size)}")
            print(f"  Flat shape: {layer_logits_flat.shape}")
            print(f"  Number of nodes in layer: {num_nodes_in_layer}")
            raise e

        updated_logits_list.append(layer_logits)
        start_node_idx += num_nodes_in_layer

    return updated_logits_list


def prepare_node_edge_features(graph_state):
    """
    Convert node and edge dictionaries to feature tensors using vectorized operations.

    Args:
        graph_state: GraphState with node and edge information (lists of dicts)

    Returns:
        Tuple of (node_features, edge_features) as JAX tensors
    """
    # --- Node Features ---
    flat_nodes_info = [node for layer in graph_state.node_features for node in layer]
    num_total_nodes = len(flat_nodes_info)

    if num_total_nodes == 0:
        return jp.zeros((0, 0)), jp.zeros((0, 0))  # Handle empty graph

    # Extract positional features
    node_idxs = jp.array([node["node_idx"] for node in flat_nodes_info])
    layer_idxs = jp.array([node["layer_idx"] for node in flat_nodes_info])
    # Replace None with -1 for group_idx
    group_idxs = jp.array(
        [
            node["group_idx"] if node["group_idx"] is not None else -1
            for node in flat_nodes_info
        ]
    )
    gate_idxs = jp.array([node["gate_idx"] for node in flat_nodes_info])

    pos_features = jp.stack([node_idxs, layer_idxs, group_idxs, gate_idxs], axis=1)

    # Extract and pad/truncate logits features
    max_lut_size = 0
    all_logits = []
    has_logits_mask = []
    for node in flat_nodes_info:
        logits = node.get("logits")  # Use get for safety
        if logits is not None:
            logits_array = jp.asarray(logits)  # Ensure it's a JAX array
            all_logits.append(logits_array)
            max_lut_size = max(max_lut_size, logits_array.shape[0])
            has_logits_mask.append(True)
        else:
            has_logits_mask.append(False)

    # Default lut size if no nodes had logits
    if max_lut_size == 0:
        max_lut_size = 16  # Or another sensible default

    logit_features = jp.zeros((num_total_nodes, max_lut_size))

    # Fill in logits for nodes that have them, applying padding/truncation
    current_logit_idx = 0
    padded_logits_list = []
    for i in range(num_total_nodes):
        if has_logits_mask[i]:
            original_logits = all_logits[current_logit_idx]
            logit_len = original_logits.shape[0]

            if logit_len >= max_lut_size:
                padded_logits = original_logits[:max_lut_size]
            else:
                padded_logits = jp.pad(
                    original_logits,
                    (0, max_lut_size - logit_len),
                    mode="constant",
                    constant_values=0,
                )
            padded_logits_list.append(padded_logits)
            current_logit_idx += 1
        else:
            # For nodes without logits, we just need a placeholder of the correct size
            padded_logits_list.append(jp.zeros(max_lut_size))

    # Stack the processed logits
    if padded_logits_list:
        logit_features = jp.stack(padded_logits_list)
    else:  # Handle case where no nodes exist or have logits
        logit_features = jp.zeros((num_total_nodes, max_lut_size))

    # Combine positional and logit features
    node_features = jp.concatenate([pos_features, logit_features], axis=1)

    # --- Edge Features ---
    if not graph_state.edge_features:  # Handle case with no edges
        edge_features = jp.zeros((0, 3))  # Shape (num_edges, num_features)
    else:
        source_layers = jp.array(
            [edge["source_layer"] for edge in graph_state.edge_features]
        )
        target_layers = jp.array(
            [edge["target_layer"] for edge in graph_state.edge_features]
        )
        input_ports = jp.array(
            [edge["input_port"] for edge in graph_state.edge_features]
        )
        edge_features = jp.stack([source_layers, target_layers, input_ports], axis=1)

    return node_features.astype(jp.float32), edge_features.astype(
        jp.float32
    )  # Ensure float32 for MLPs


def gnn_update_circuit(logits, wires, x, target_outputs, gnn_params, message_steps=3):
    """
    Update circuit parameters using GNN message passing.

    Args:
        logits: Current circuit logits
        wires: Circuit wiring
        x: Input tensor
        target_outputs: Target output tensor
        gnn_params: GNN parameters
        message_steps: Number of message passing steps

    Returns:
        Updated circuit logits
    """
    # Build graph from circuit
    graph_state = build_circuit_graph(logits, wires)

    # Run the circuit to get current outputs
    acts = run_circuit(logits, wires, x)
    current_outputs = acts[-1]

    # Calculate error on outputs
    errors = current_outputs - target_outputs

    # Prepare features
    node_features, edge_features = prepare_node_edge_features(graph_state)

    # Encode nodes and edges
    encoded_nodes = jax.vmap(
        lambda node: node_encoder_mlp(node, gnn_params["node_encoder"])
    )(node_features)
    encoded_edges = jax.vmap(
        lambda edge: edge_encoder_mlp(edge, gnn_params["edge_encoder"])
    )(edge_features)

    # Perform message passing
    updated_nodes = message_passing_step(
        graph_state, encoded_nodes, encoded_edges, gnn_params, message_steps
    )

    # Extract updated logits
    updated_logits = extract_logits_from_nodes(graph_state, updated_nodes)

    return updated_logits


# Training loop using GNN instead of backprop
def train_circuit_with_gnn(
    logits, wires, x, y0, gnn_params, steps=100, message_steps=3
):
    """
    Train circuit with GNN message passing instead of backprop.

    Args:
        logits: Initial circuit logits
        wires: Circuit wiring
        x: Input tensor
        y0: Target output tensor
        gnn_params: GNN parameters
        steps: Number of training steps
        message_steps: Number of message passing steps per training step

    Returns:
        Tuple of (final_logits, loss_history)
    """
    loss_history = []
    current_logits = logits

    for step in range(steps):
        # Update circuit parameters using GNN
        current_logits = gnn_update_circuit(
            current_logits, wires, x, y0, gnn_params, message_steps
        )

        # Evaluate current circuit
        acts = run_circuit(current_logits, wires, x)
        y_pred = acts[-1]

        # Calculate loss and accuracy
        loss = jp.mean(jp.square(y_pred - y0))
        accuracy = jp.mean(jp.round(y_pred) == y0)

        loss_history.append((loss, accuracy))

        # Optionally print progress
        if step % 10 == 0:
            print(f"Step {step}: Loss = {loss:.4f}, Accuracy = {accuracy:.4f}")

    return current_logits, loss_history


# Simple example of usage
def gnn_run_example(logits, wires, x, y0, n_steps=100, hidden_dim=64, message_dim=32):
    """
    Example of training a circuit with GNN-based parameter updates.
    Uses the same format as the backpropagation training loop.

    Args:
        logits: Initial circuit logits
        wires: Circuit wiring
        x: Input tensor
        y0: Target output tensor
        n_steps: Number of training steps

    Returns:
        Tuple of (final_logits, loss_log, aux_log)
    """
    from tqdm.auto import trange  # For progress bar

    # Initialize GNN parameters
    # Determine feature dimensions from data
    test_graph = build_circuit_graph(logits, wires)
    test_node_features, test_edge_features = prepare_node_edge_features(test_graph)
    node_feat_dim = test_node_features.shape[1]
    edge_feat_dim = test_edge_features.shape[1]

    # Initialize GNN parameters
    key = jax.random.PRNGKey(42)
    gnn_params = init_gnn_params(
        node_feat_dim, edge_feat_dim, hidden_dim, message_dim, key
    )

    # Training loop
    loss_log_gnn = []
    aux_log_gnn = []
    current_logits = logits

    pbar = trange(n_steps)
    for i in pbar:
        # Update circuit parameters using GNN (single step)
        graph_state = build_circuit_graph(current_logits, wires)
        node_features, edge_features = prepare_node_edge_features(graph_state)

        # Encode nodes and edges
        encoded_nodes = jax.vmap(
            lambda node: node_encoder_mlp(node, gnn_params["node_encoder"])
        )(node_features)
        encoded_edges = jax.vmap(
            lambda edge: edge_encoder_mlp(edge, gnn_params["edge_encoder"])
        )(edge_features)

        # Perform message passing (3 steps)
        updated_nodes = message_passing_step(
            graph_state, encoded_nodes, encoded_edges, gnn_params, 3
        )

        # Extract updated logits
        current_logits = extract_logits_from_nodes(graph_state, updated_nodes)

        # Evaluate current circuit
        acts = run_circuit(current_logits, wires, x)
        y_pred = acts[-1]
        hard_acts = run_circuit(current_logits, wires, x, hard=True)
        hard_y = hard_acts[-1]

        # Calculate loss and accuracy
        loss = jp.mean(jp.square(y_pred - y0))
        accuracy = jp.mean(jp.round(y_pred) == y0)
        hard_loss = jp.mean(jp.square(hard_y - y0))
        hard_accuracy = jp.mean(hard_y == y0)

        # Store metrics
        loss_log_gnn.append(loss)
        aux = {
            "accuracy": accuracy,
            "hard_loss": hard_loss,
            "hard_accuracy": hard_accuracy,
            "act": acts,
        }
        aux_log_gnn.append(aux)

        # Update progress bar
        pbar.set_postfix_str(
            f"loss: {loss.item():.3E}, accuracy: {accuracy.item():.2%}"
            + f" | hard_loss: {hard_loss.item():.3E}, hard_accuracy: {hard_accuracy.item():.2%}"
        )

    return current_logits, loss_log_gnn, aux_log_gnn


def gnn_loss_function(
    gnn_params, logits, wires, x, y0, message_steps=3, loss_type="l4"
):
    """
    Loss function for GNN parameter optimization.

    Args:
        gnn_params: GNN parameters to optimize
        logits: Initial circuit logits
        wires: Circuit wiring
        x: Input tensor
        y0: Target output tensor
        message_steps: Number of message passing steps
        loss_type: Type of loss to use ('l4' or 'bce')

    Returns:
        Loss value (scalar) and auxiliary information
    """
    # Build graph from circuit
    graph_state = build_circuit_graph(logits, wires)

    # Prepare features
    node_features, edge_features = prepare_node_edge_features(graph_state)

    # Encode nodes and edges
    encoded_nodes = jax.vmap(
        lambda node: node_encoder_mlp(node, gnn_params["node_encoder"])
    )(node_features)
    encoded_edges = jax.vmap(
        lambda edge: edge_encoder_mlp(edge, gnn_params["edge_encoder"])
    )(edge_features)

    # Perform message passing
    updated_nodes = message_passing_step(
        graph_state, encoded_nodes, encoded_edges, gnn_params, message_steps
    )

    # Extract updated logits
    updated_logits = extract_logits_from_nodes(graph_state, updated_nodes)

    # Evaluate updated circuit
    acts = run_circuit(updated_logits, wires, x)
    y_pred = acts[-1]

    # Compute loss
    if loss_type == "bce":
        loss = binary_cross_entropy(y_pred, y0)
    else:  # default to L4 norm
        res = y_pred - y0
        loss = res2loss(res)

    # Also compute hard metrics for monitoring
    hard_acts = run_circuit(updated_logits, wires, x, hard=True)
    hard_y = hard_acts[-1]
    hard_accuracy = jp.mean(hard_y == y0)

    return loss, (updated_logits, acts, hard_accuracy)


# Gradient function for GNN parameter optimization
grad_gnn_loss = jax.value_and_grad(
    lambda params, logits, wires, x, y0, steps, loss_type: gnn_loss_function(
        params, logits, wires, x, y0, steps, loss_type
    )[0],
    has_aux=False,
)


def train_gnn_params(
    gnn_params,
    logits,
    wires,
    x,
    y0,
    learning_rate=0.01,
    steps=100,
    message_steps=3,
    loss_type="l4",
):
    """
    Train GNN parameters to update circuit parameters effectively.

    Args:
        gnn_params: Initial GNN parameters
        logits: Initial circuit logits
        wires: Circuit wiring
        x: Input tensor
        y0: Target output tensor
        learning_rate: Learning rate for optimizer
        steps: Number of training steps
        message_steps: Number of message passing steps per iteration
        loss_type: Type of loss to use ('l4' or 'bce')

    Returns:
        Tuple of (updated_gnn_params, loss_history, updated_logits)
    """
    # Initialize optimizer
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(gnn_params)

    loss_history = []
    current_gnn_params = gnn_params
    current_logits = logits

    for step in range(steps):
        # First compute loss
        loss, (updated_logits, acts, hard_accuracy) = gnn_loss_function(
            current_gnn_params, current_logits, wires, x, y0, message_steps, loss_type
        )

        # Then compute gradients
        loss_value, grads = grad_gnn_loss(
            current_gnn_params, current_logits, wires, x, y0, message_steps, loss_type
        )

        # Update circuit logits for next iteration
        current_logits = updated_logits

        # Update GNN parameters
        updates, opt_state = optimizer.update(grads, opt_state, current_gnn_params)
        current_gnn_params = optax.apply_updates(current_gnn_params, updates)

        # Store metrics
        loss_history.append((loss, hard_accuracy))

        # Print progress
        if step % 10 == 0:
            print(
                f"Step {step}: Loss = {loss:.4f}, Hard Accuracy = {hard_accuracy:.4f}"
            )

    return current_gnn_params, loss_history, current_logits


def gnn_train_example(
    logits,
    wires,
    x,
    y0,
    n_steps=100,
    inner_steps=1,
    hidden_dim=64,
    message_dim=32,
    learning_rate=0.01,
    loss_type="l4",
):
    """
    Enhanced example of training a circuit with GNN-based parameter updates.
    Also updates the GNN parameters themselves based on performance.

    Args:
        logits: Initial circuit logits
        wires: Circuit wiring
        x: Input tensor
        y0: Target output tensor
        n_steps: Number of training steps
        inner_steps: Number of GNN parameter update steps per iteration
        hidden_dim: Hidden dimension of MLPs
        message_dim: Message dimension
        learning_rate: Learning rate for GNN parameter updates
        loss_type: Type of loss to use ('l4' or 'bce')

    Returns:
        Tuple of (final_logits, loss_log, aux_log, final_gnn_params)
    """
    from tqdm.auto import trange  # For progress bar

    # Initialize GNN parameters
    test_graph = build_circuit_graph(logits, wires)
    test_node_features, test_edge_features = prepare_node_edge_features(test_graph)
    node_feat_dim = test_node_features.shape[1]
    edge_feat_dim = test_edge_features.shape[1]

    # Initialize GNN parameters
    key = jax.random.PRNGKey(42)
    gnn_params = init_gnn_params(
        node_feat_dim, edge_feat_dim, hidden_dim, message_dim, key
    )

    # Initialize optimizer for GNN parameters
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(gnn_params)

    # Training loop
    loss_log = []
    aux_log = []
    current_logits = logits
    current_gnn_params = gnn_params

    pbar = trange(n_steps)
    for i in pbar:
        # First, apply current GNN to update circuit parameters
        graph_state = build_circuit_graph(current_logits, wires)
        node_features, edge_features = prepare_node_edge_features(graph_state)

        # Encode nodes and edges
        encoded_nodes = jax.vmap(
            lambda node: node_encoder_mlp(node, current_gnn_params["node_encoder"])
        )(node_features)
        encoded_edges = jax.vmap(
            lambda edge: edge_encoder_mlp(edge, current_gnn_params["edge_encoder"])
        )(edge_features)

        # Perform message passing
        updated_nodes = message_passing_step(
            graph_state, encoded_nodes, encoded_edges, current_gnn_params, 3
        )

        # Extract updated logits
        current_logits = extract_logits_from_nodes(graph_state, updated_nodes)

        # Evaluate current circuit
        acts = run_circuit(current_logits, wires, x)
        y_pred = acts[-1]
        hard_acts = run_circuit(current_logits, wires, x, hard=True)
        hard_y = hard_acts[-1]

        # Calculate loss and accuracy
        if loss_type == "bce":
            loss = binary_cross_entropy(y_pred, y0)
            hard_loss = binary_cross_entropy(hard_y, y0)
        else:  # default to L4
            res = y_pred - y0
            loss = res2loss(res)
            hard_res = hard_y - y0
            hard_loss = res2loss(hard_res)

        accuracy = jp.mean(jp.round(y_pred) == y0)
        hard_accuracy = jp.mean(hard_y == y0)

        # Store metrics
        loss_log.append(loss)
        aux = {
            "accuracy": accuracy,
            "hard_loss": hard_loss,
            "hard_accuracy": hard_accuracy,
            "act": acts,
        }
        aux_log.append(aux)

        # Now, update GNN parameters to improve the circuit updates
        for inner_step in range(inner_steps):
            # Compute gradients of loss with respect to GNN parameters
            loss_value, grads = grad_gnn_loss(
                current_gnn_params, current_logits, wires, x, y0, 3, loss_type
            )

            # Update GNN parameters
            updates, opt_state = optimizer.update(grads, opt_state, current_gnn_params)
            current_gnn_params = optax.apply_updates(current_gnn_params, updates)

        # Update progress bar
        pbar.set_postfix_str(
            f"loss: {loss.item():.3E}, accuracy: {accuracy.item():.2%}"
            + f" | hard_loss: {hard_loss.item():.3E}, hard_accuracy: {hard_accuracy.item():.2%}"
        )

    return current_logits, loss_log, aux_log, current_gnn_params
