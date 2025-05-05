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


# --- Positional Encoding ---


def get_positional_encoding(indices: jp.ndarray, dim: int, max_val: float = 10000.0):
    """
    Generates sinusoidal positional encodings.

    Args:
        indices: 1D JAX array of integer positions (e.g., layer indices, node indices).
        dim: The dimension of the positional encoding vector (must be even).
        max_val: Maximum value for the denominator in the frequency calculation.

    Returns:
        JAX array of shape (len(indices), dim) containing the positional encodings.
    """
    if dim % 2 != 0:
        raise ValueError(f"Positional encoding dimension must be even, got {dim}")

    # Ensure indices are float for calculations
    positions = indices.astype(jp.float32)[:, None]  # Shape: (num_indices, 1)

    # Calculate the division term for frequencies
    # div_term shape: (dim // 2,)
    div_term = jp.exp(jp.arange(0, dim, 2, dtype=jp.float32) * -(jp.log(max_val) / dim))

    # Initialize PE matrix
    pe = jp.zeros((indices.shape[0], dim), dtype=jp.float32)

    # Calculate sin and cos components
    pe = pe.at[:, 0::2].set(jp.sin(positions * div_term))
    pe = pe.at[:, 1::2].set(jp.cos(positions * div_term))

    return pe


# --- Graph Building ---


def build_graph(
    logits: List[jp.ndarray],
    wires: List[jp.ndarray],
    input_n: int,
    arity: int,
    hidden_dim: int,
    bidirectional_edges: bool = True,
    verbose: bool = False,
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
        bidirectional_edges: If True, create edges in both forward and backward directions.

    Returns:
        A jraph.GraphsTuple representing the circuit.
    """
    all_nodes_features_list = []
    all_forward_senders = []
    all_forward_receivers = []
    current_global_node_idx = 0
    layer_start_indices = []  # We'll store the start index of each layer

    if verbose:
        print(f"USING BIDIRECTIONAL EDGES: {bidirectional_edges}")

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

        # --- Add Positional Encodings ---
        # Layer PE (using layer_idx)
        # Use a reasonable max_layers, e.g., 100, or calculate dynamically if needed
        max_layers = len(logits)  # Number of gate layers
        layer_indices = jp.full(num_gates_in_layer, layer_idx, dtype=jp.int32)
        layer_pe = get_positional_encoding(
            layer_indices, hidden_dim, max_val=max_layers + 1
        )  # Add 1 for safety

        # Intra-Layer PE (using index within the layer)
        intra_layer_indices = jp.arange(num_gates_in_layer, dtype=jp.int32)
        intra_layer_pe = get_positional_encoding(
            intra_layer_indices, hidden_dim, max_val=num_gates_in_layer + 1
        )  # Max val is layer size

        layer_nodes["layer_pe"] = layer_pe
        layer_nodes["intra_layer_pe"] = intra_layer_pe
        # --- End Positional Encodings ---

        # Create forward edges (if not the first layer)
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

            all_forward_senders.append(current_layer_senders)
            all_forward_receivers.append(current_layer_receivers)
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
        pe_dim = hidden_dim  # Dimension for each PE type

        # --- Calculate MLP input size ---
        # Logits, Hidden, Layer PE, Intra-Layer PE
        current_features_size = logit_dim + hidden_dim + pe_dim + pe_dim
        if message_passing:
            # Aggregated messages will also have logit_dim + hidden_dim
            # Edge MLP output doesn't include PEs, just logits+hidden derived features
            aggregated_message_size = logit_dim + hidden_dim
            mlp_input_size = current_features_size + aggregated_message_size
        else:
            mlp_input_size = current_features_size

        # Output needs to contain updated logits and updated hidden features
        mlp_output_size = logit_dim + hidden_dim

        # Add feature normalization layers
        self.logits_norm = nnx.BatchNorm(
            logit_dim, use_running_average=True, momentum=0.9, epsilon=1e-5, rngs=rngs
        )
        self.hidden_norm = nnx.BatchNorm(
            hidden_dim, use_running_average=True, momentum=0.9, epsilon=1e-5, rngs=rngs
        )
        # Add BatchNorm for PEs
        self.layer_pe_norm = nnx.BatchNorm(
            pe_dim, use_running_average=True, momentum=0.9, epsilon=1e-5, rngs=rngs
        )
        self.intra_layer_pe_norm = nnx.BatchNorm(
            pe_dim, use_running_average=True, momentum=0.9, epsilon=1e-5, rngs=rngs
        )
        if message_passing:
            self.message_norm = nnx.BatchNorm(
                aggregated_message_size,
                use_running_average=True,
                momentum=0.9,
                epsilon=1e-5,
                rngs=rngs,
            )

        # Define MLP architecture with BatchNorm
        mlp_features = [mlp_input_size, *node_mlp_features, mlp_output_size]
        mlp_layers = []
        for i, (in_f, out_f) in enumerate(zip(mlp_features[:-1], mlp_features[1:])):
            # Special initialization for the final layer
            if i == len(mlp_features) - 2:
                # Initialize final layer weights and biases to zero for "do nothing" start
                final_linear = nnx.Linear(
                    in_f,
                    out_f,
                    kernel_init=jax.nn.initializers.zeros,
                    bias_init=jax.nn.initializers.zeros,
                    rngs=rngs,
                )
                mlp_layers.append(final_linear)
            else:
                mlp_layers.append(nnx.Linear(in_f, out_f, rngs=rngs))
                # Add BatchNorm and ReLU
                mlp_layers.append(
                    nnx.BatchNorm(
                        out_f,
                        use_running_average=True,
                        momentum=0.9,
                        epsilon=1e-5,
                        rngs=rngs,
                    )
                )
                mlp_layers.append(jax.nn.relu)

        self.mlp = nnx.Sequential(*mlp_layers)

    def __call__(
        self,
        nodes: NodeType,
        sent_attributes: jp.ndarray,
        received_attributes: jp.ndarray,
        globals_,
    ):
        """Update node features using a residual MLP."""
        # Extract current node features
        current_logits = nodes["logits"]  # Shape: (num_nodes, 2**arity)
        current_hidden = nodes["hidden"]  # Shape: (num_nodes, hidden_dim)
        current_layer_pe = nodes["layer_pe"]  # Shape: (num_nodes, pe_dim)
        current_intra_layer_pe = nodes["intra_layer_pe"]  # Shape: (num_nodes, pe_dim)
        logit_dim = 2**self.arity

        # Normalize input features
        normalized_logits = self.logits_norm(current_logits)
        normalized_hidden = self.hidden_norm(current_hidden)
        normalized_layer_pe = self.layer_pe_norm(current_layer_pe)
        normalized_intra_layer_pe = self.intra_layer_pe_norm(current_intra_layer_pe)

        # Combine normalized features (Logits, Hidden, Layer PE, Intra-Layer PE)
        current_node_combined_features = jp.concatenate(
            [
                normalized_logits,
                normalized_hidden,
                normalized_layer_pe,
                normalized_intra_layer_pe,
            ],
            axis=-1,
        )

        # Determine inputs based on message passing flag
        if self.message_passing and sent_attributes is not None:
            # Normalize messages
            normalized_messages = self.message_norm(sent_attributes)
            # Input = current_features + normalized_messages
            mlp_input = jp.concatenate(
                [current_node_combined_features, normalized_messages], axis=-1
            )
        else:
            # Input = current_features only
            mlp_input = current_node_combined_features

        # Apply MLP to get the delta (change) in features
        delta_combined_features = self.mlp(mlp_input)

        # Split the delta into logit and hidden components
        delta_logits = delta_combined_features[..., :logit_dim]
        delta_hidden = delta_combined_features[..., logit_dim:]

        # Apply residual update with raw deltas
        updated_logits = current_logits + delta_logits
        updated_hidden = current_hidden + delta_hidden

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
        logit_dim = 2**arity  # Added logit_dim for clarity
        pe_dim = hidden_dim  # Dimension for each PE type

        # Input to edge MLP: Sender's Logits, Hidden, Layer PE, Intra-Layer PE
        mlp_input_size = logit_dim + hidden_dim + pe_dim + pe_dim
        # Output of edge MLP: Features used for aggregation (Logits + Hidden dimensions)
        mlp_output_size = logit_dim + hidden_dim
        mlp_features = [mlp_input_size, *edge_mlp_features, mlp_output_size]

        # Define Edge MLP architecture with BatchNorm
        edge_mlp_layers = []
        for i, (in_f, out_f) in enumerate(zip(mlp_features[:-1], mlp_features[1:])):
            edge_mlp_layers.append(nnx.Linear(in_f, out_f, rngs=rngs))
            # Add BatchNorm and ReLU except for the last layer
            if i < len(mlp_features) - 2:
                # BatchNorm needs the number of output features from the linear layer
                edge_mlp_layers.append(
                    nnx.BatchNorm(
                        out_f,
                        use_running_average=True,
                        momentum=0.9,
                        epsilon=1e-5,
                        rngs=rngs,
                    )
                )
                edge_mlp_layers.append(jax.nn.relu)

        self.edge_mlp = nnx.Sequential(*edge_mlp_layers)

    def __call__(
        self,
        edge_features: EdgeType,
        sender_node_features: NodeType,
        receiver_node_features: NodeType,
        globals_,
    ):
        """Generate messages to be sent along edges."""
        # Pass sender's logits, hidden features, and PEs as input to edge MLP
        sender_logits = sender_node_features["logits"]
        sender_hidden = sender_node_features["hidden"]
        sender_layer_pe = sender_node_features["layer_pe"]
        sender_intra_layer_pe = sender_node_features["intra_layer_pe"]

        # Concatenate sender features for the MLP
        sender_combined_features = jp.concatenate(
            [sender_logits, sender_hidden, sender_layer_pe, sender_intra_layer_pe],
            axis=-1,
        )

        # Apply edge MLP to generate the message
        # message shape: (num_edges, logit_dim + hidden_dim)
        message = self.edge_mlp(sender_combined_features)
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
    check_gradients: bool = False,
    init_gnn: CircuitGNN = None,
    init_optimizer: nnx.Optimizer = None,
    initial_metrics: Dict = None,
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
        init_gnn: Optional pre-trained GNN model to continue training from
        init_optimizer: Optional pre-trained optimizer to continue training from
        initial_metrics: Optional dictionary of metrics from previous training

    Returns:
        Trained GNN model and training metrics
    """
    # Initialize random key
    rng = jax.random.PRNGKey(key)

    # Get dimensions from layer sizes
    input_n = layer_sizes[0][0]

    if not meta_learning:
        meta_batch_size = 1

    # Initialize metrics storage
    if initial_metrics is None:
        # Start with empty lists
        losses = []
        accuracies = []
        hard_losses = []
        hard_accuracies = []
    else:
        # Continue from previous metrics
        losses = list(initial_metrics.get("losses", []))
        accuracies = list(initial_metrics.get("accuracies", []))
        hard_losses = list(initial_metrics.get("hard_losses", []))
        hard_accuracies = list(initial_metrics.get("hard_accuracies", []))

    # 1. Initialize or reuse GNN
    if init_gnn is None:
        # Create a new GNN
        rng, init_key = jax.random.split(rng)
        gnn = CircuitGNN(
            node_mlp_features=node_mlp_features,
            edge_mlp_features=edge_mlp_features,
            hidden_dim=hidden_dim,
            arity=arity,
            message_passing=message_passing,
            rngs=nnx.Rngs(params=init_key),
        )
    else:
        # Use the provided GNN
        gnn = init_gnn

    # 2. Create optimizer with weight decay or reuse existing optimizer
    if init_optimizer is None:
        optimizer = nnx.Optimizer(
            gnn, optax.adamw(learning_rate=learning_rate, weight_decay=weight_decay)
        )
    else:
        optimizer = init_optimizer

    # 3. Define meta-training step function
    @partial(nnx.jit, static_argnames=("layer_sizes", "n_message_steps"))
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

        # if check_gradients and jax.tree.reduce(
        #     lambda x, y: x and y, jax.tree.map(lambda g: (g == 0).all(), grads)
        # ):
        #     print("WARNING: Gradients are all zero")

        return loss, aux

    # 4. Training loop
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
        losses.append(float(loss))
        hard_losses.append(float(hard_loss))
        accuracies.append(float(accuracy))
        hard_accuracies.append(float(hard_accuracy))

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
        "optimizer": optimizer,
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
        do_train=False,  # Just evaluation, no parameter update
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
            do_train=True,  # Actually update parameters
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


# --- Visualization Functions ---


def visualize_circuit_graph(
    graph: jraph.GraphsTuple,
    figsize=(10, 8),
    node_size=80,
    layer_spacing=1.5,
    node_color_feature="layer",
    node_feature_type=None,
    feature_reduction="mean",
    pca_components=3,
    show_labels=False,
    cmap="viridis",
    edge_alpha=0.3,
    title=None,
):
    """
    Visualize a circuit graph in a clean, minimal style.

    Args:
        graph: jraph.GraphsTuple representation of the circuit
        figsize: Figure size as (width, height)
        node_size: Size of nodes in the visualization
        layer_spacing: Vertical spacing between layers
        node_color_feature: Node feature to use for coloring:
            - 'layer': Color by layer index
            - 'group': Color by group assignment
            - 'node_features': Color by node feature values (requires node_feature_type)
        node_feature_type: Type of node features to use for coloring (when node_color_feature='node_features'):
            - 'logits': Use the logits tensor
            - 'hidden': Use the hidden state tensor
        feature_reduction: Method to reduce high-dimensional features to RGB colors:
            - 'mean': Use mean across feature dimensions (single color)
            - 'sum': Use sum across feature dimensions (single color)
            - 'rgb': Use first 3 channels as RGB values
            - 'pca': Use PCA projection to 3D (requires sklearn)
        pca_components: Number of PCA components to use (default=3 for RGB)
        show_labels: Whether to show node IDs as labels
        cmap: Colormap to use for node colors (for single-channel coloring)
        edge_alpha: Transparency of edges
        title: Optional title for the plot

    Returns:
        fig, ax: The matplotlib figure and axis objects
    """
    try:
        import networkx as nx
        import matplotlib.pyplot as plt
        from matplotlib.cm import get_cmap
        import numpy as np
    except ImportError:
        print("Error: This function requires networkx and matplotlib. Install with:")
        print("pip install networkx matplotlib")
        return None, None

    # Check if feature reduction is 'pca' and try to import sklearn
    if feature_reduction == "pca":
        try:
            from sklearn.decomposition import PCA
        except ImportError:
            print("Error: PCA reduction requires scikit-learn. Install with:")
            print("pip install scikit-learn")
            print("Falling back to 'mean' reduction...")
            feature_reduction = "mean"

    # Convert GraphsTuple to NetworkX graph
    G = nx.DiGraph()

    # Get total number of nodes and edges
    n_nodes = graph.n_node[0]
    n_edges = graph.n_edge[0]

    # Add nodes with attributes
    for i in range(n_nodes):
        # Extract node features
        layer = int(graph.nodes["layer"][i])
        gate_id = int(graph.nodes["gate_id"][i])
        group = int(graph.nodes["group"][i])

        # Add node features for coloring if needed
        node_attrs = {"layer": layer, "gate_id": gate_id, "group": group}

        if node_color_feature == "node_features" and node_feature_type:
            # Add specific node features for advanced coloring
            if node_feature_type == "logits" and "logits" in graph.nodes:
                node_attrs["feature_values"] = np.array(graph.nodes["logits"][i])
            elif node_feature_type == "hidden" and "hidden" in graph.nodes:
                node_attrs["feature_values"] = np.array(graph.nodes["hidden"][i])

        # Add node with attributes
        G.add_node(i, **node_attrs)

    # Add edges
    for i in range(n_edges):
        sender = int(graph.senders[i])
        receiver = int(graph.receivers[i])
        G.add_edge(sender, receiver)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Determine node positions by layer (hierarchical layout)
    pos = {}
    layers = {}

    # Group nodes by layer
    for node, data in G.nodes(data=True):
        layer = data["layer"]
        if layer not in layers:
            layers[layer] = []
        layers[layer].append(node)

    # Set y-coordinate based on layer
    max_layer = max(layers.keys())
    for layer, nodes in layers.items():
        # Sort nodes within layer by group for cleaner visualization
        nodes.sort(key=lambda n: G.nodes[n]["group"])

        # Calculate y-coordinate (invert so layer 0 is at the top)
        y = (max_layer - layer) * layer_spacing

        # Distribute nodes horizontally
        num_nodes = len(nodes)
        for i, node in enumerate(nodes):
            x = (i - num_nodes / 2) / (num_nodes * 0.5) if num_nodes > 1 else 0
            pos[node] = (x, y)

    # Determine node colors
    if node_color_feature == "layer":
        color_values = [G.nodes[n]["layer"] for n in G.nodes()]
        node_colors = color_values
        vmin, vmax = 0, max_layer
        colorbar_label = "Layer"
        use_colormap = True

    elif node_color_feature == "group":
        color_values = [G.nodes[n]["group"] for n in G.nodes()]
        node_colors = color_values
        vmin, vmax = 0, max(color_values)
        colorbar_label = "Group"
        use_colormap = True

    elif node_color_feature == "node_features" and node_feature_type:
        # Check if all nodes have feature_values
        if not all("feature_values" in G.nodes[n] for n in G.nodes()):
            print(
                f"Warning: Not all nodes have {node_feature_type} features. Using default coloring."
            )
            node_colors = "skyblue"
            use_colormap = False
            vmin, vmax = None, None
        else:
            # Get feature values for all nodes
            feature_arrays = [G.nodes[n]["feature_values"] for n in G.nodes()]

            # Apply the requested feature reduction
            if feature_reduction == "mean":
                # Take mean of each feature array
                color_values = [np.mean(arr) for arr in feature_arrays]
                node_colors = color_values
                vmin, vmax = min(color_values), max(color_values)
                colorbar_label = f"Mean {node_feature_type}"
                use_colormap = True

            elif feature_reduction == "sum":
                # Take sum of each feature array
                color_values = [np.sum(arr) for arr in feature_arrays]
                node_colors = color_values
                vmin, vmax = min(color_values), max(color_values)
                colorbar_label = f"Sum {node_feature_type}"
                use_colormap = True

            elif feature_reduction == "rgb":
                # Use first 3 values as RGB
                # Make sure all arrays have at least 3 values
                if all(len(arr) >= 3 for arr in feature_arrays):
                    # Normalize each dimension to [0, 1]
                    rgb_arrays = []
                    for arr in feature_arrays:
                        # Only take first 3 dimensions
                        rgb = arr[:3]
                        # Apply sigmoid to values for better visualization
                        rgb = 1.0 / (1.0 + np.exp(-rgb))
                        rgb_arrays.append(rgb)

                    node_colors = rgb_arrays
                    use_colormap = False
                    vmin, vmax = None, None
                    colorbar_label = None
                else:
                    print(
                        "Warning: Some feature arrays don't have 3+ dimensions for RGB. Using mean reduction."
                    )
                    color_values = [np.mean(arr) for arr in feature_arrays]
                    node_colors = color_values
                    vmin, vmax = min(color_values), max(color_values)
                    colorbar_label = f"Mean {node_feature_type}"
                    use_colormap = True

            elif feature_reduction == "pca":
                # Make sure we have enough samples for PCA
                if len(feature_arrays) >= 3:
                    # Stack arrays for PCA
                    feature_matrix = np.stack(feature_arrays)
                    # Apply PCA
                    pca = PCA(n_components=min(pca_components, feature_matrix.shape[1]))
                    # Normalize feature matrix before PCA
                    feature_matrix = (
                        feature_matrix - np.mean(feature_matrix, axis=0)
                    ) / (np.std(feature_matrix, axis=0) + 1e-8)
                    pca_result = pca.fit_transform(feature_matrix)

                    if pca_components == 1:
                        # Use single component with colormap
                        node_colors = pca_result.flatten()
                        vmin, vmax = min(node_colors), max(node_colors)
                        colorbar_label = f"PCA of {node_feature_type}"
                        use_colormap = True
                    else:
                        # For 3+ components, use first 3 for RGB
                        rgb_pca = pca_result[:, :3]
                        # Normalize to [0, 1] for RGB
                        for col in range(rgb_pca.shape[1]):
                            col_min, col_max = (
                                rgb_pca[:, col].min(),
                                rgb_pca[:, col].max(),
                            )
                            if col_max > col_min:  # Avoid division by zero
                                rgb_pca[:, col] = (rgb_pca[:, col] - col_min) / (
                                    col_max - col_min
                                )

                        node_colors = rgb_pca
                        use_colormap = False
                        vmin, vmax = None, None
                        colorbar_label = None
                else:
                    print("Warning: Not enough nodes for PCA. Using mean reduction.")
                    color_values = [np.mean(arr) for arr in feature_arrays]
                    node_colors = color_values
                    vmin, vmax = min(color_values), max(color_values)
                    colorbar_label = f"Mean {node_feature_type}"
                    use_colormap = True
            else:
                # Invalid reduction method
                print(
                    f"Warning: Unknown feature reduction method '{feature_reduction}'. Using default coloring."
                )
                node_colors = "skyblue"
                use_colormap = False
                vmin, vmax = None, None
                colorbar_label = None
    else:
        # Default color if no feature specified
        node_colors = "skyblue"
        use_colormap = False
        vmin, vmax = None, None
        colorbar_label = None

    # Draw the graph
    if use_colormap:
        nodes = nx.draw_networkx_nodes(
            G,
            pos,
            node_size=node_size,
            node_color=node_colors,
            cmap=get_cmap(cmap),
            vmin=vmin,
            vmax=vmax,
            ax=ax,
        )
    else:
        nodes = nx.draw_networkx_nodes(
            G, pos, node_size=node_size, node_color=node_colors, ax=ax
        )

    edges = nx.draw_networkx_edges(
        G,
        pos,
        alpha=edge_alpha,
        edge_color="gray",
        arrows=True,
        arrowsize=10,
        width=0.5,
        ax=ax,
    )

    if show_labels:
        labels = {n: str(G.nodes[n]["gate_id"]) for n in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=8, ax=ax)

    # Add colorbar if coloring by a feature with a colormap
    if use_colormap and vmin is not None and vmax is not None:
        sm = plt.cm.ScalarMappable(
            cmap=get_cmap(cmap), norm=plt.Normalize(vmin=vmin, vmax=vmax)
        )
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.8, pad=0.05)
        if colorbar_label:
            cbar.set_label(colorbar_label)

    # Set title if provided
    if title:
        ax.set_title(title)
    else:
        # Create descriptive title if none provided
        if node_color_feature == "node_features" and node_feature_type:
            auto_title = f"Circuit Graph (colored by {node_feature_type} using {feature_reduction})"
            ax.set_title(auto_title)
        elif node_color_feature in ["layer", "group"]:
            ax.set_title(f"Circuit Graph (colored by {node_color_feature})")
        else:
            ax.set_title("Circuit Graph")

    # Remove axes
    ax.set_axis_off()

    plt.tight_layout()
    return fig, ax


def save_circuit_animation(
    gnn: CircuitGNN,
    graph: jraph.GraphsTuple,
    num_steps: int = 10,
    output_file: str = "circuit_evolution.gif",
    fps: int = 2,
    figsize=(10, 8),
    node_size=80,
    node_color_feature="layer",
    show_labels=False,
):
    """
    Create an animation of graph evolution over GNN message passing steps.

    Args:
        gnn: The CircuitGNN model
        graph: Initial GraphsTuple
        num_steps: Number of steps to animate
        output_file: Output filename (gif or mp4)
        fps: Frames per second in the animation
        figsize: Figure size as (width, height)
        node_size: Size of nodes in visualization
        node_color_feature: Node feature to use for coloring
        show_labels: Whether to show node labels

    Returns:
        Path to saved animation file
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation
        import networkx as nx
    except ImportError:
        print("Error: Animation requires matplotlib and networkx. Install with:")
        print("pip install matplotlib networkx")
        return None

    # Create figure and axis for animation
    fig, ax = plt.subplots(figsize=figsize)

    # Create a list to store graph states
    graph_states = [graph]

    # Generate all graph states in advance
    for step in range(1, num_steps + 1):
        # Apply message passing steps
        current_graph = run_gnn_scan(gnn, graph, step)
        graph_states.append(current_graph)

    # Create a common color scale for all frames
    all_layers = []
    for g in graph_states:
        all_layers.extend(g.nodes["layer"])
    vmin, vmax = 0, int(max(all_layers))

    # Function to convert GraphsTuple to NetworkX graph
    def graph_tuple_to_nx(g):
        G = nx.DiGraph()
        n_nodes = g.n_node[0]
        n_edges = g.n_edge[0]

        # Add nodes with attributes
        for i in range(n_nodes):
            layer = int(g.nodes["layer"][i])
            gate_id = int(g.nodes["gate_id"][i])
            group = int(g.nodes["group"][i])
            G.add_node(i, layer=layer, gate_id=gate_id, group=group)

        # Add edges
        for i in range(n_edges):
            sender = int(g.senders[i])
            receiver = int(g.receivers[i])
            G.add_edge(sender, receiver)

        return G

    # Create positions once based on the initial graph structure
    G_init = graph_tuple_to_nx(graph)

    # Group nodes by layer
    layers = {}
    for node, data in G_init.nodes(data=True):
        layer = data["layer"]
        if layer not in layers:
            layers[layer] = []
        layers[layer].append(node)

    # Compute node positions (hierarchical layout)
    pos = {}
    max_layer = max(layers.keys())
    layer_spacing_val = 1.5

    for layer, nodes in layers.items():
        # Sort nodes within layer by group for cleaner visualization
        nodes.sort(key=lambda n: G_init.nodes[n]["group"])

        # Calculate y-coordinate (invert so layer 0 is at the top)
        y = (max_layer - layer) * layer_spacing_val

        # Distribute nodes horizontally
        num_nodes = len(nodes)
        for i, node in enumerate(nodes):
            x = (i - num_nodes / 2) / (num_nodes * 0.5) if num_nodes > 1 else 0
            pos[node] = (x, y)

    # Function to draw a frame
    def draw_frame(i):
        ax.clear()

        current_graph = graph_states[i]
        G = graph_tuple_to_nx(current_graph)

        # Determine node colors
        if node_color_feature == "layer":
            color_values = [G.nodes[n]["layer"] for n in G.nodes()]
        elif node_color_feature == "group":
            color_values = [G.nodes[n]["group"] for n in G.nodes()]
        else:
            color_values = "skyblue"

        # Draw the graph
        nx.draw_networkx_nodes(
            G,
            pos,
            node_size=node_size,
            node_color=color_values,
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            ax=ax,
        )

        nx.draw_networkx_edges(
            G,
            pos,
            alpha=0.3,
            edge_color="gray",
            arrows=True,
            arrowsize=10,
            width=0.5,
            ax=ax,
        )

        if show_labels:
            labels = {n: str(G.nodes[n]["gate_id"]) for n in G.nodes()}
            nx.draw_networkx_labels(G, pos, labels=labels, font_size=8, ax=ax)

        # Add colorbar only once (first frame)
        if i == 0 and node_color_feature in ["layer", "group"]:
            sm = plt.cm.ScalarMappable(
                cmap="viridis", norm=plt.Normalize(vmin=vmin, vmax=vmax)
            )
            sm.set_array([])
            # cbar = plt.colorbar(sm, ax=ax, shrink=0.8, pad=0.05)
            # cbar.set_label(f"{node_color_feature.capitalize()}")

        # Set title
        if i == 0:
            ax.set_title("Initial Circuit")
        else:
            ax.set_title(f"After {i} Message Passing Steps")

        # Remove axes
        ax.set_axis_off()

        return (ax,)

    # Create animation
    ani = animation.FuncAnimation(
        fig,
        draw_frame,
        frames=len(graph_states),
        interval=1000 / fps,  # milliseconds
        blit=False,
    )

    # Save animation
    writer = "pillow" if output_file.endswith(".gif") else None
    ani.save(output_file, writer=writer, fps=fps)

    plt.close(fig)
    print(f"Animation saved to {output_file}")
    return output_file
