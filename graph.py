import jax
import jax.numpy as jp
import jraph
from flax import nnx
import numpy as np
from typing import List, Tuple, Dict
from functools import partial
import optax
from model import run_circuit, gen_circuit
from tqdm.auto import tqdm


# Define the structure for node features
NodeType = Dict[str, jp.ndarray]
# Example: {'layer': jp.array([...]), 'group': jp.array([...]), 'gate_id': jp.array([...]), 'logits': jp.array([...])}
# Input nodes might have different features, e.g., only layer and gate_id, logits might be None or zero-padded.
EdgeType = jp.ndarray


# Define an MLP using nnx for updates
class MLP(nnx.Module):
    def __init__(self, features: List[int], *, rngs: nnx.Rngs):
        self.layers = [
            nnx.Linear(in_features, out_features, rngs=rngs)
            for in_features, out_features in zip(features[:-1], features[1:])
        ]

    def __call__(self, x):
        for layer in self.layers[:-1]:
            x = jax.nn.relu(layer(x))
        x = self.layers[-1](x)  # No activation on the final layer
        return x


# --- Graph Building ---


def build_graph(
    logits: List[jp.ndarray],
    wires: List[jp.ndarray],
    input_n: int,
    arity: int,
    hidden_dim: int,
) -> jraph.GraphsTuple:
    """
    Constructs a jraph.GraphsTuple representation of the boolean circuit.

    Args:
        logits: List of logit tensors per layer. Shape [(group_n, group_size, 2^arity), ...]
        wires: List of wire connection patterns per layer. Shape [(arity, edges_in_layer = out_n // group_size), ...]
               Note: wires[i] connects layer i to layer i+1.
        input_n: Number of input nodes/bits for the first layer (layer 0) - only used for wire indexing.
        arity: The fan-in for each gate.
        hidden_dim: Dimension of hidden features for nodes.

    Returns:
        A jraph.GraphsTuple representing the circuit.
    """
    all_nodes_features_list = []
    all_senders = []
    all_receivers = []
    current_global_node_idx = 0
    layer_start_indices = []  # We'll store the start index of each layer

    # Process layers (no separate input nodes)
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
            "group": jp.repeat(jp.arange(group_n), group_size, axis=0),
            "gate_id": layer_global_indices,
            "logits": layer_logits.reshape(
                num_gates_in_layer, logit_dim
            ),  # Reshape logits to match nodes
            "hidden": jp.zeros(
                (num_gates_in_layer, hidden_dim), dtype=jp.float32
            ),  # Initialize hidden features
        }
        all_nodes_features_list.append(layer_nodes)

        # Create edges (if not the first layer)
        if layer_idx > 0:
            # Receivers are the gates in the *current* layer
            current_layer_receivers = jp.repeat(layer_global_indices, arity)

            # Senders are gates from the previous layer, indexed by layer_wires
            previous_layer_start_idx = layer_start_indices[layer_idx - 1]

            # Map wire indices to global sender indices
            global_senders_for_layer = previous_layer_start_idx + layer_wires

            # Tile senders for all gates in each group
            tiled_senders = jp.tile(global_senders_for_layer.T, (1, group_size))
            current_layer_senders = tiled_senders.reshape(-1)  # Flatten

            all_senders.append(current_layer_senders)
            all_receivers.append(current_layer_receivers)
        elif layer_idx == 0:
            # For the first layer, we don't have senders from a previous layer in the graph
            # The inputs are external, so we don't create edges for them
            pass

        # Update global index for the next layer
        current_global_node_idx += num_gates_in_layer

    # Consolidate Nodes and Edges
    if not all_nodes_features_list:
        return jraph.GraphsTuple(
            nodes={},
            edges=None,
            senders=jp.array([]),
            receivers=jp.array([]),
            n_node=jp.array([0]),
            n_edge=jp.array([0]),
            globals=None,
        )

    # Combine node features from all layers
    all_nodes = jax.tree.map(
        lambda *xs: jp.concatenate(xs, axis=0), *all_nodes_features_list
    )

    if all_senders:
        senders = jp.concatenate(all_senders)
        receivers = jp.concatenate(all_receivers)
    else:
        senders = jp.array([], dtype=jp.int32)
        receivers = jp.array([], dtype=jp.int32)

    n_node = current_global_node_idx
    n_edge = len(senders)

    # Ensure correct shapes and types for GraphsTuple
    graph = jraph.GraphsTuple(
        nodes=all_nodes,  # Pytree of node features (Dict[str, Array])
        edges=None,  # No edge features
        senders=senders.astype(jp.int32),
        receivers=receivers.astype(jp.int32),
        n_node=jp.array([n_node]),
        n_edge=jp.array([n_edge]),
        globals=None,  # No global features for now
    )
    return graph


# --- GNN Update Functions ---


# Default aggregation function: sum over incoming messages
def aggregate_edges_for_nodes_fn(
    messages: jp.ndarray,  # Features of edges targetting a node
    indices: jp.ndarray,  # index of the node targetted by the edge
    num_segments: int,
):  # total number of nodes in the graph
    """Default aggregation function: sum messages per node."""
    return jraph.segment_sum(messages, indices, num_segments)


# Replace the update_node_fn with a NodeUpdateModule
# Update function of an NCA
class NodeUpdateModule(nnx.Module):
    def __init__(
        self,
        node_mlp_features: List[int],
        hidden_dim: int,
        arity: int,
        message_passing: bool = False,
        *,
        rngs: nnx.Rngs,
    ):
        self.arity = arity
        self.hidden_dim = hidden_dim
        self.message_passing = message_passing
        logit_dim = 2**arity

        # --- Calculate MLP input size ---
        current_features_size = logit_dim + hidden_dim
        if message_passing:
            # Aggregated messages will also have logit_dim + hidden_dim
            aggregated_message_size = logit_dim + hidden_dim
            mlp_input_size = current_features_size + aggregated_message_size
        else:
            mlp_input_size = current_features_size

        # Output needs to contain updated logits and updated hidden features
        mlp_output_size = logit_dim + hidden_dim

        # Define MLP architecture
        mlp_features = [mlp_input_size, *node_mlp_features, mlp_output_size]
        self.mlp = nnx.Sequential(
            *[
                layer
                for features in zip(mlp_features[:-1], mlp_features[1:])
                for layer in [nnx.Linear(*features, rngs=rngs), jax.nn.relu]
            ][:-1]  # Remove the last relu
        )

    def __call__(
        self,
        nodes: NodeType,
        sent_attributes: jp.ndarray,
        received_attributes: jp.ndarray,
        globals_,
    ):
        """Update node features using an MLP."""
        # Extract current node features
        current_logits = nodes["logits"]  # Shape: (num_nodes, 2**arity)
        current_hidden = nodes["hidden"]  # Shape: (num_nodes, hidden_dim)
        logit_dim = 2**self.arity

        # Combine current features
        current_node_combined_features = jp.concatenate(
            [current_logits, current_hidden], axis=-1
        )

        # Determine inputs based on message passing flag
        if self.message_passing and sent_attributes is not None:
            # Input = current_features + aggregated_messages
            mlp_input = jp.concatenate(
                [current_node_combined_features, sent_attributes], axis=-1
            )
        else:
            # Input = current_features only
            mlp_input = current_node_combined_features

        # Apply MLP to get combined updated features
        updated_combined_features = self.mlp(mlp_input)

        # Split the output into updated logits and hidden features
        updated_logits = updated_combined_features[..., :logit_dim]
        updated_hidden = updated_combined_features[..., logit_dim:]

        # Update only the 'logits' and 'hidden' fields, preserving others
        new_node_features = {k: v for k, v in nodes.items()}
        new_node_features["logits"] = updated_logits
        new_node_features["hidden"] = updated_hidden

        return new_node_features


# Perceive function of an NCA
class EdgeUpdateModule(nnx.Module):
    def __init__(
        self,
        edge_mlp_features: List[int],
        hidden_dim: int,
        arity: int,
        *,
        rngs: nnx.Rngs,
    ):
        self.hidden_dim = hidden_dim
        self.arity = arity
        self.edge_mlp_features = edge_mlp_features
        mlp_input_size = 2**arity + hidden_dim
        mlp_output_size = 2**arity + hidden_dim
        mlp_features = [mlp_input_size, *edge_mlp_features, mlp_output_size]

        self.edge_mlp = nnx.Sequential(
            *[
                layer
                for features in zip(mlp_features[:-1], mlp_features[1:])
                for layer in [nnx.Linear(*features, rngs=rngs), jax.nn.relu]
            ][:-1]  # Remove the last relu
        )

    def __call__(
        self,
        edge_features: EdgeType,
        sender_node_features: NodeType,
        receiver_node_features: NodeType,
        globals_,
    ):
        """Generate messages to be sent along edges."""
        # Pass sender's logits and hidden features as messages
        sender_logits = sender_node_features["logits"]
        sender_hidden = sender_node_features["hidden"]
        message = jp.concatenate([sender_logits, sender_hidden], axis=-1)
        message = self.edge_mlp(message)
        return message  # Shape: (num_edges, 2**arity + hidden_dim)


# --- GNN Network Definition ---


class CircuitGNN(nnx.Module):
    def __init__(
        self,
        # mlp update dims
        node_mlp_features: List[int] = [64, 32],
        edge_mlp_features: List[int] = [64, 32],
        # number of hidden channels in circuit "states"
        hidden_dim: int = 16,
        arity: int = 2,
        message_passing: bool = False,
        *,
        rngs: nnx.Rngs,
    ):
        self.arity = arity
        self.message_passing = message_passing
        self.hidden_dim = hidden_dim

        # Create the node and edge update modules directly
        self.node_update = NodeUpdateModule(
            node_mlp_features=node_mlp_features,
            hidden_dim=hidden_dim,
            arity=arity,
            message_passing=message_passing,
            rngs=rngs,
        )

        self.edge_update = EdgeUpdateModule(
            edge_mlp_features=edge_mlp_features,
            hidden_dim=hidden_dim,
            arity=arity,
            rngs=rngs,
        )

        # Store the aggregation function
        self.aggregate_fn = aggregate_edges_for_nodes_fn

    def __call__(self, graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
        """Applies one step of GNN message passing manually."""
        nodes, edge_features_in, receivers, senders, globals_, n_node, n_edge = graph

        # 1. Compute messages using the edge update module
        # Get sender node features (using tree.tree_map for pytree compatibility)
        sender_node_features = jax.tree.map(lambda n: n[senders], nodes)
        # Get receiver node features (potentially needed by edge_update, though unused in current EdgeUpdateModule)
        receiver_node_features = jax.tree.map(lambda n: n[receivers], nodes)

        # messages shape: (num_edges, 2**arity + hidden_dim)
        messages = self.edge_update(
            edge_features_in,  # Pass existing edge features (currently None)
            sender_node_features,
            receiver_node_features,
            globals_,
        )

        # 2. Aggregate messages for each node
        # Ensure num_nodes is calculated correctly, possibly using jraph.utils
        # For segment_sum, num_segments should be the total number of nodes.
        num_nodes = jax.tree.leaves(nodes)[0].shape[
            0
        ]  # More robust way needed if nodes can be None

        # aggregated_messages shape: (num_nodes, 2**arity + hidden_dim)
        aggregated_messages = self.aggregate_fn(messages, receivers, num_nodes)

        # 3. Update node features using the node update module
        # Pass the aggregated messages as sent_attributes
        updated_nodes = self.node_update(
            nodes,  # Current node features
            aggregated_messages,  # Aggregated messages from incoming edges
            None,  # received_attributes (unused by NodeUpdateModule)
            globals_,  # Global features
        )

        # Return the graph with updated node features
        # Note: Edge features are not updated in this specific setup
        return graph._replace(nodes=updated_nodes)


@partial(nnx.jit, static_argnames=("num_steps",))
def run_gnn_scan(
    gnn: CircuitGNN, graph: jraph.GraphsTuple, num_steps: int
) -> jraph.GraphsTuple:
    """
    Applies the GNN message passing iteratively for a fixed number of steps using jax.lax.scan.

    Args:
        gnn: The CircuitGNN model to apply.
        graph: The initial GraphsTuple.
        num_steps: The number of message passing steps to perform.

    Returns:
        The GraphsTuple after num_steps of message passing.
    """
    # First, ensure edges field is not None but a proper tensor to maintain pytree structure
    if graph.edges is None:
        # Initialize edges based on the message structure
        # The shape should match what update_edge_fn would produce
        senders_count = graph.senders.shape[0] if graph.senders.shape[0] > 0 else 0
        logit_dim = 2**gnn.arity
        hidden_dim = gnn.hidden_dim

        # Create empty edges with the right shape
        edges = jp.zeros((senders_count, logit_dim + hidden_dim), dtype=jp.float32)
        graph = graph._replace(edges=edges)

    def scan_body(carry_graph, _):
        # Apply one step of the GNN
        updated_graph = gnn(carry_graph)
        # Return the updated graph as the new carry value
        return updated_graph, None

    # Run the scan
    final_graph, _ = jax.lax.scan(scan_body, graph, None, length=num_steps)

    return final_graph


# --- Example Usage ---


def test_gnn(
    logits,
    wires,
    input_n,
    arity=2,
    hidden_dim=16,
    message_passing=True,
    steps=1,
    gnn=None,
    test_gradients=False,
):
    """
    Demonstrate the graph construction and GNN message passing.

    Args:
        logits: List of logit tensors for each layer
        wires: List of wire connection patterns
        input_n: Number of input nodes (only used for wiring)
        arity: Number of inputs per gate
        hidden_dim: Dimension of hidden features
        message_passing: Whether to use message passing (otherwise just MLP transform)
        steps: Number of message passing steps to run
        gnn: Optional pre-initialized GNN model
        test_gradients: Whether to test gradients
    Returns:
        Tuple of (updated_logits, graph) with updated parameters after message passing
    """
    # Store original shapes for reconstruction
    logits_original_shapes = [logit.shape for logit in logits]

    # 1. Build the graph representation
    print(
        f"Building graph from {len(logits)} layers, input_n={input_n}, arity={arity}, hidden_dim={hidden_dim}"
    )
    graph = build_graph(logits, wires, input_n, arity, hidden_dim)
    print(f"Graph has {graph.n_node[0]} nodes and {graph.n_edge[0]} edges")

    if gnn is None:
        # 2. Initialize the GNN
        print(
            f"Initializing GNN with message_passing={message_passing}, hidden_dim={hidden_dim}"
        )
        rng = jax.random.PRNGKey(0)
        node_mlp_features = [64, 32]  # Example hidden layers for MLP
        gnn = CircuitGNN(
            node_mlp_features=node_mlp_features,
            hidden_dim=hidden_dim,
            arity=arity,
            message_passing=message_passing,
            rngs=nnx.Rngs(params=rng),
        )

    # 3. Run message passing steps using scan
    print(f"Running {steps} message passing steps using scan")
    updated_graph = run_gnn_scan(gnn, graph, steps)
    print("Scan completed")

    # 4. Extract updated logits from the graph
    updated_logits = extract_logits_from_graph(updated_graph, logits_original_shapes)

    if test_gradients:
        # 5. Test if gradients flow through the model
        print("Testing gradient flow...")

        def loss_fn(gnn):
            # Simple loss function: sum of all logits
            test_graph = run_gnn_scan(gnn, graph, 1)
            return jp.sum(test_graph.nodes["logits"])

        loss, grads = nnx.value_and_grad(loss_fn)(gnn)
        # Check if gradients are non-zero
        grad_tree = jax.tree.map(lambda x: jp.abs(x).sum(), grads)
        grad_norm = sum(jax.tree.leaves(grad_tree))
        print(f"Gradient norm: {grad_norm}")
        if grad_norm == 0:
            print("WARNING: Zero gradients - gradient flow not working!")
        else:
            print("Gradient flow confirmed - gradients are non-zero")

    return updated_logits, updated_graph, gnn


# --- Training Functions ---


def extract_logits_from_graph(
    graph: jraph.GraphsTuple, logits_original_shapes: List[Tuple[int, int, int]]
) -> List[jp.ndarray]:
    """
    Extracts the list of logit tensors from the graph's node features in a JIT-compatible way.

    Args:
        graph: The GraphsTuple containing updated node features (including flattened 'logits').
        logits_original_shapes: List of original (group_n, group_size, 2^arity) shapes for each layer.

    Returns:
        A list of logit tensors, reshaped to their original shapes.
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

        # dynamic_slice needs start indices as a tuple/list for each dimension
        start_indices_tuple = (start_index, 0)
        # dynamic_slice needs slice sizes as a tuple/list for each dimension
        slice_sizes_tuple = (size, logit_dim)

        layer_logits_flat = jax.lax.dynamic_slice(
            all_logits_flat, start_indices_tuple, slice_sizes_tuple
        )

        # Reshape back to original
        layer_logits = layer_logits_flat.reshape(group_n, group_size, logit_dim)
        extracted_logits_list.append(layer_logits)

    return extracted_logits_list


@partial(nnx.jit, static_argnames=("logits_original_shapes", "n_message_steps"))
def train_step_gnn(
    gnn: CircuitGNN,
    optimizer: nnx.Optimizer,
    graph: jraph.GraphsTuple,
    wires: List[jp.ndarray],
    x: jp.ndarray,
    y_target: jp.ndarray,
    logits_original_shapes: List[Tuple[int, int, int]],
    n_message_steps: int,
):
    """
    Single training step for the GNN.

    Args:
        gnn: CircuitGNN model
        optimizer: nnx Optimizer instance
        graph: GraphsTuple representation of the circuit
        wires: List of wire connection patterns
        x: Input tensor
        y_target: Target output tensor
        logits_original_shapes: List of (group_n, group_size, 2^arity) shapes for each layer
        n_message_steps: Number of message passing steps to run

    Returns:
        Tuple of (loss, (accuracy, updated_logits, updated_graph), grads)
    """

    # Define loss_fn to take the model object directly
    def loss_fn(gnn_model: CircuitGNN):
        # Run message passing for n steps
        updated_graph = run_gnn_scan(gnn_model, graph, n_message_steps)

        # Extract updated logits from the final graph
        updated_logits = extract_logits_from_graph(
            updated_graph, logits_original_shapes
        )

        # Run the circuit with updated logits
        # --- This is a potential place where gradients might be cut off --- #
        # --- Ensure model.run_circuit is differentiable --- #
        all_acts = run_circuit(updated_logits, wires, x)
        y_pred = all_acts[-1]
        all_hard_acts = run_circuit(updated_logits, wires, x, hard=True)
        y_hard_pred = all_hard_acts[-1]

        # Compute loss and accuracy
        loss = jp.mean((y_pred - y_target) ** 4)
        hard_loss = jp.mean((y_hard_pred - y_target) ** 4)
        accuracy = jp.mean(jp.round(y_pred) == y_target)
        hard_accuracy = jp.mean(jp.round(y_hard_pred) == y_target)
        return loss, (hard_loss, accuracy, hard_accuracy, updated_logits, updated_graph)

    # Compute loss and gradients w.r.t the model object
    (loss, aux), grads = nnx.value_and_grad(loss_fn, has_aux=True)(gnn)

    # Update GNN parameters using the optimizer
    optimizer.update(grads)

    # Return loss, auxiliary outputs (accuracy, logits, graph), and gradients
    return loss, aux, grads


def train_gnn(
    layer_sizes: List[Tuple[int, int]],
    x_data: jp.ndarray,
    y_data: jp.ndarray,
    arity: int = 2,
    hidden_dim: int = 16,
    message_passing: bool = True,
    node_mlp_features: List[int] = [64, 32],
    edge_mlp_features: List[int] = [64, 32],
    learning_rate: float = 1e-3,
    epochs: int = 100,
    n_message_steps: int = 100,
    key: int = 0,
    weight_decay: float = 1e-4,
    meta_learning: bool = False,
    meta_batch_size: int = 64,
):
    """
    Meta-train the GNN to optimize circuit parameters for random wirings.

    Instead of updating a single circuit, this function trains the GNN to optimize
    any circuit with randomly sampled wirings at each step.

    Args:
        layer_sizes: List of tuples (nodes, group_size) for each layer
        x_data: Input data for training (batch, input_bits)
        y_data: Target output data (batch, output_bits)
        arity: Number of inputs per gate
        hidden_dim: Dimension of the hidden features for each node
        message_passing: Whether to use message passing
        node_mlp_features: Hidden layers for the node MLP
        learning_rate: Learning rate for optimization
        epochs: Number of training epochs
        n_message_steps: Number of message passing steps per training step
        key: Random seed
        weight_decay: Weight decay for optimizer

    Returns:
        Trained GNN model and training metrics
    """
    # Initialize random key
    rng = jax.random.PRNGKey(key)

    # Get dimensions from layer sizes
    input_n = layer_sizes[0][0]

    if not meta_learning:
        meta_batch_size = 1

    # 1. Initialize GNN
    rng, init_key = jax.random.split(rng)
    gnn = CircuitGNN(
        node_mlp_features=node_mlp_features,
        edge_mlp_features=edge_mlp_features,
        hidden_dim=hidden_dim,
        arity=arity,
        message_passing=message_passing,
        rngs=nnx.Rngs(params=init_key),
    )

    # 2. Create optimizer with weight decay
    optimizer = nnx.Optimizer(
        gnn, optax.adamw(learning_rate=learning_rate, weight_decay=weight_decay)
    )

    # 3. Define meta-training step function
    # @partial(nnx.jit, static_argnames=("layer_sizes", "n_message_steps"))
    def meta_train_step(
        gnn: CircuitGNN,
        optimizer: nnx.Optimizer,
        x: jp.ndarray,
        y_target: jp.ndarray,
        rng: jax.random.PRNGKey,
        layer_sizes: List[Tuple[int, int]],
        n_message_steps: int,
    ):
        """Single meta-training step with randomly sampled circuit wirings."""

        def loss_fn(gnn_model: CircuitGNN, rng: jax.random.PRNGKey):
            # Sample new random circuit wiring for this step
            rng_wires, _ = jax.random.split(rng)
            wires, logits = gen_circuit(rng_wires, layer_sizes, arity=arity)

            # Store original shapes for reconstruction
            logits_original_shapes = [logit.shape for logit in logits]

            # Build graph from the random circuit
            graph = build_graph(logits, wires, input_n, arity, hidden_dim)

            # Run GNN for n_message_steps to optimize the circuit
            updated_graph = run_gnn_scan(gnn_model, graph, n_message_steps)

            # Extract updated logits and run the circuit
            updated_logits = extract_logits_from_graph(
                updated_graph, logits_original_shapes
            )
            all_acts = run_circuit(updated_logits, wires, x)
            y_pred = all_acts[-1]

            # Also measure hard circuit accuracy
            all_hard_acts = run_circuit(updated_logits, wires, x, hard=True)
            y_hard_pred = all_hard_acts[-1]

            # Compute loss and accuracy
            loss = jp.mean((y_pred - y_target) ** 4)
            hard_loss = jp.mean((y_hard_pred - y_target) ** 4)
            accuracy = jp.mean(jp.round(y_pred) == y_target)
            hard_accuracy = jp.mean(jp.round(y_hard_pred) == y_target)

            return loss, (hard_loss, accuracy, hard_accuracy)

        def mean_batch_loss_fn(gnn, rng):
            batch_rng = jax.random.split(rng, meta_batch_size)
            batch_loss_fn = nnx.vmap(loss_fn, in_axes=(None, 0))
            losses, aux = batch_loss_fn(gnn, rng=batch_rng)
            return jp.mean(losses), jax.tree.map(lambda x: jp.mean(x, axis=0), aux)

        # Compute loss and gradients
        (loss, aux), grads = nnx.value_and_grad(mean_batch_loss_fn, has_aux=True)(
            gnn, rng=rng
        )

        # Update GNN parameters
        optimizer.update(grads)

        if jax.tree.reduce(lambda x, y: x and y, jax.tree.map(lambda g: (g == 0).all(), grads)) :
            print("WARNING: Gradients are all zero")

        return loss, aux

    # 4. Training loop
    losses = []
    accuracies = []
    hard_losses = []
    hard_accuracies = []

    # Create a tqdm progress bar
    pbar = tqdm(range(epochs), desc="Training GNN")

    for epoch in pbar:
        # Each epoch uses a different random key
        if meta_learning:
            rng, epoch_key = jax.random.split(rng)
        else:
            epoch_key = rng

        # Select a random subset of data for this epoch
        idx = jax.random.permutation(epoch_key, len(x_data))
        x_batch = x_data[idx]
        y_batch = y_data[idx]

        # Perform meta-training step
        loss, (hard_loss, accuracy, hard_accuracy) = meta_train_step(
            gnn,
            optimizer,
            x_batch,
            y_batch,
            epoch_key,
            tuple(layer_sizes),
            n_message_steps,
        )

        # Record metrics
        losses.append(loss)
        hard_losses.append(hard_loss)
        accuracies.append(accuracy)
        hard_accuracies.append(hard_accuracy)

        # Update progress bar with current metrics
        pbar.set_postfix(
            {
                "Loss": f"{loss:.4f}",
                "Accuracy": f"{accuracy:.4f}",
                "Hard Acc": f"{hard_accuracy:.4f}",
            }
        )

    # 5. Return the trained GNN model and metrics
    return {
        "gnn": gnn,
        "losses": losses,
        "hard_losses": hard_losses,
        "accuracies": accuracies,
        "hard_accuracies": hard_accuracies,
    }


def evaluate_gnn_stepwise(
    gnn: CircuitGNN,
    wires: List[jp.ndarray],
    logits: List[jp.ndarray],
    x_data: jp.ndarray,
    y_data: jp.ndarray,
    input_n: int,
    arity: int = 2,
    hidden_dim: int = 16,
    n_message_steps: int = 100,
):
    """
    Evaluate GNN performance by running message passing steps one by one,
    gathering metrics at each step to compare against backprop optimization.

    Args:
        gnn: Trained CircuitGNN model
        wires: List of wire connection patterns
        logits: List of initial logit tensors for each layer
        x_data: Input data for evaluation
        y_data: Target output data
        input_n: Number of input nodes
        arity: Number of inputs per gate
        hidden_dim: Dimension of hidden features
        n_message_steps: Maximum number of message passing steps to run

    Returns:
        Dictionary with metrics collected at each step
    """
    # Store original shapes for reconstruction
    logits_original_shapes = [logit.shape for logit in logits]

    # Build initial graph from circuit
    graph = build_graph(logits, wires, input_n, arity, hidden_dim)

    # Initialize metric storage
    step_metrics = {
        "step": [],
        "soft_loss": [],
        "hard_loss": [],
        "soft_accuracy": [],
        "hard_accuracy": [],
    }

    # Evaluate initial circuit (step 0)
    initial_acts = run_circuit(logits, wires, x_data)
    initial_pred = initial_acts[-1]
    initial_hard_acts = run_circuit(logits, wires, x_data, hard=True)
    initial_hard_pred = initial_hard_acts[-1]

    soft_loss = jp.mean((initial_pred - y_data) ** 2)
    hard_loss = jp.mean((initial_hard_pred - y_data) ** 2)
    soft_accuracy = jp.mean(jp.round(initial_pred) == y_data)
    hard_accuracy = jp.mean(jp.round(initial_hard_pred) == y_data)

    # Record initial metrics
    step_metrics["step"].append(0)
    step_metrics["soft_loss"].append(float(soft_loss))
    step_metrics["hard_loss"].append(float(hard_loss))
    step_metrics["soft_accuracy"].append(float(soft_accuracy))
    step_metrics["hard_accuracy"].append(float(hard_accuracy))

    # Create tqdm progress bar
    pbar = tqdm(range(1, n_message_steps + 1), desc="Evaluating GNN steps")

    # Evaluate after each message passing step
    for step in pbar:
        # Apply one step of GNN message passing
        graph = gnn(graph)

        # Extract current logits
        current_logits = extract_logits_from_graph(graph, logits_original_shapes)

        # Evaluate circuit with current logits
        acts = run_circuit(current_logits, wires, x_data)
        pred = acts[-1]
        hard_acts = run_circuit(current_logits, wires, x_data, hard=True)
        hard_pred = hard_acts[-1]

        # Calculate metrics
        soft_loss = jp.mean((pred - y_data) ** 2)
        hard_loss = jp.mean((hard_pred - y_data) ** 2)
        soft_accuracy = jp.mean(jp.round(pred) == y_data)
        hard_accuracy = jp.mean(jp.round(hard_pred) == y_data)

        # Record metrics
        step_metrics["step"].append(step)
        step_metrics["soft_loss"].append(float(soft_loss))
        step_metrics["hard_loss"].append(float(hard_loss))
        step_metrics["soft_accuracy"].append(float(soft_accuracy))
        step_metrics["hard_accuracy"].append(float(hard_accuracy))

        # Update progress bar
        pbar.set_postfix(
            {
                "Loss": f"{soft_loss:.4f}",
                "Accuracy": f"{soft_accuracy:.4f}",
                "Hard Acc": f"{hard_accuracy:.4f}",
            }
        )

    return step_metrics


def compare_gnn_vs_backprop(
    gnn: CircuitGNN,
    layer_sizes: List[Tuple[int, int]],
    x_data: jp.ndarray,
    y_data: jp.ndarray,
    arity: int = 2,
    hidden_dim: int = 16,
    n_message_steps: int = 100,
    backprop_steps: int = 100,
    bp_learning_rate: float = 0.01,
    key: int = 42,
):
    """
    Compare optimization via GNN vs direct backpropagation on the same circuit.

    Args:
        gnn: Trained CircuitGNN model
        layer_sizes: List of tuples (nodes, group_size) for each layer
        x_data: Input data for evaluation
        y_data: Target output data
        arity: Number of inputs per gate
        hidden_dim: Dimension of hidden features
        n_message_steps: Number of message passing steps for GNN
        backprop_steps: Number of optimization steps for backpropagation
        bp_learning_rate: Learning rate for backpropagation optimizer
        key: Random seed

    Returns:
        Dictionary with comparison metrics
    """
    # Generate a random circuit for testing
    rng = jax.random.PRNGKey(key)
    test_wires, test_logits = gen_circuit(rng, layer_sizes, arity=arity)
    input_n = layer_sizes[0][0]

    print("1. Running GNN optimization step-by-step")
    # Run step-wise GNN evaluation
    gnn_metrics = evaluate_gnn_stepwise(
        gnn=gnn,
        wires=test_wires,
        logits=test_logits.copy(),  # Use a copy to avoid modifying the original
        x_data=x_data,
        y_data=y_data,
        input_n=input_n,
        arity=arity,
        hidden_dim=hidden_dim,
        n_message_steps=n_message_steps,
    )

    print("2. Running direct backpropagation optimization")
    # Initialize metrics for backpropagation
    bp_metrics = {
        "step": [],
        "soft_loss": [],
        "hard_loss": [],
        "soft_accuracy": [],
        "hard_accuracy": [],
    }

    # Import the training logic
    from training import TrainState, train_step

    # Make a copy of the initial logits for backpropagation
    bp_logits = [logit.copy() for logit in test_logits]
    
    # Create optimizer
    optimizer = optax.adam(bp_learning_rate)
    opt_state = optimizer.init(bp_logits)
    
    # Create initial training state
    state = TrainState(bp_logits, opt_state)
    
    # Evaluate initial circuit (step 0)
    loss, aux, state = train_step(
        state=state,
        opt=optimizer,
        wires=test_wires,
        x=x_data,
        y0=y_data,
        loss_type="l4",  # Using L4 loss
        do_train=False   # Just evaluation, no parameter update
    )
    
    # Record initial metrics
    bp_metrics["step"].append(0)
    bp_metrics["soft_loss"].append(float(loss))
    bp_metrics["hard_loss"].append(float(aux["hard_loss"]))
    bp_metrics["soft_accuracy"].append(float(aux["accuracy"]))
    bp_metrics["hard_accuracy"].append(float(aux["hard_accuracy"]))

    # Create tqdm progress bar for backpropagation
    pbar = tqdm(range(1, backprop_steps + 1), desc="Running backprop")

    # Run backpropagation optimization using the train_step function
    for step in pbar:
        # Perform a training step
        loss, aux, state = train_step(
            state=state,
            opt=optimizer,
            wires=test_wires,
            x=x_data,
            y0=y_data,
            loss_type="l4",  # Using L4 loss
            do_train=True    # Actually update parameters
        )
        
        # Record metrics
        bp_metrics["step"].append(step)
        bp_metrics["soft_loss"].append(float(loss))
        bp_metrics["hard_loss"].append(float(aux["hard_loss"]))
        bp_metrics["soft_accuracy"].append(float(aux["accuracy"]))
        bp_metrics["hard_accuracy"].append(float(aux["hard_accuracy"]))

        # Update progress bar
        pbar.set_postfix(
            {
                "Loss": f"{loss:.4f}",
                "Accuracy": f"{aux['accuracy']:.4f}",
                "Hard Acc": f"{aux['hard_accuracy']:.4f}",
            }
        )

    # Plot comparison results
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 10))

    # Plot soft losses
    plt.subplot(2, 2, 1)
    plt.title("Soft Loss")
    plt.semilogy(
        gnn_metrics["step"], gnn_metrics["soft_loss"], label="GNN", color="teal"
    )
    plt.semilogy(
        bp_metrics["step"],
        bp_metrics["soft_loss"],
        label="Backprop",
        color="crimson",
        linestyle="--",
    )
    plt.xlabel("Steps")
    plt.ylabel("Loss (log scale)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot hard losses
    plt.subplot(2, 2, 2)
    plt.title("Hard Loss")
    plt.semilogy(
        gnn_metrics["step"], gnn_metrics["hard_loss"], label="GNN", color="teal"
    )
    plt.semilogy(
        bp_metrics["step"],
        bp_metrics["hard_loss"],
        label="Backprop",
        color="crimson",
        linestyle="--",
    )
    plt.xlabel("Steps")
    plt.ylabel("Loss (log scale)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot soft accuracies
    plt.subplot(2, 2, 3)
    plt.title("Soft Accuracy")
    plt.plot(
        gnn_metrics["step"], gnn_metrics["soft_accuracy"], label="GNN", color="teal"
    )
    plt.plot(
        bp_metrics["step"],
        bp_metrics["soft_accuracy"],
        label="Backprop",
        color="crimson",
        linestyle="--",
    )
    plt.xlabel("Steps")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot hard accuracies
    plt.subplot(2, 2, 4)
    plt.title("Hard Accuracy")
    plt.plot(
        gnn_metrics["step"], gnn_metrics["hard_accuracy"], label="GNN", color="teal"
    )
    plt.plot(
        bp_metrics["step"],
        bp_metrics["hard_accuracy"],
        label="Backprop",
        color="crimson",
        linestyle="--",
    )
    plt.xlabel("Steps")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("gnn_vs_backprop_comparison.png", dpi=150)

    return {"gnn": gnn_metrics, "backprop": bp_metrics}
