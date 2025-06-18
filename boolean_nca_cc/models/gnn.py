"""
Graph Neural Network implementation for boolean circuits.

This module provides the main GNN model for evolving boolean circuits
through message passing.
"""

import jax
import jax.numpy as jp
import jraph
from flax import nnx
from typing import List, Tuple
from functools import partial

from boolean_nca_cc.models.node_update import NodeUpdateModule
from boolean_nca_cc.models.edge_update import EdgeUpdateModule
from boolean_nca_cc.models.aggregation import aggregate_sum, AttentionAggregation


class CircuitGNN(nnx.Module):
    """
    Graph Neural Network for evolving boolean circuits.

    Implements message passing on a graph representation of a boolean circuit.
    """

    def __init__(
        self,
        node_mlp_features: List[int] = [64, 32],
        edge_mlp_features: List[int] = [64, 32],
        hidden_dim: int = 16,
        arity: int = 2,
        message_passing: bool = True,
        use_attention: bool = False,
        *,
        rngs: nnx.Rngs,
        type: str = "gnn",
        zero_init: bool = True,
        re_zero_update: bool = False,
    ):
        """
        Initialize the Circuit GNN.

        Args:
            node_mlp_features: Hidden layer sizes for the node MLP
            edge_mlp_features: Hidden layer sizes for the edge MLP
            hidden_dim: Dimension of hidden features
            arity: Number of inputs per gate in the boolean circuit
            message_passing: Whether to use message passing or only self-updates
            use_attention: Whether to use attention-based message aggregation
            rngs: Random number generators
            type: Type of model
            zero_init: Whether to initialize weights to zero
            re_zero_update: Whether to use learnable update residual rate
        """
        self.arity = arity
        self.message_passing = message_passing
        self.hidden_dim = hidden_dim

        # Create the node and edge update modules
        self.node_update = NodeUpdateModule(
            node_mlp_features=node_mlp_features,
            hidden_dim=hidden_dim,
            arity=arity,
            message_passing=message_passing,
            rngs=rngs,
            zero_init=zero_init,
            re_zero_update=re_zero_update,
        )

        self.edge_update = EdgeUpdateModule(
            edge_mlp_features=edge_mlp_features,
            hidden_dim=hidden_dim,
            arity=arity,
            rngs=rngs,
        )

        # Configure the aggregation function
        if use_attention:
            logit_dim = 2**arity
            self.aggregate_fn = AttentionAggregation(
                feature_dim=hidden_dim + logit_dim,
                num_heads=4,
                rngs=rngs,
            )
        else:
            self.aggregate_fn = aggregate_sum

    def __call__(self, graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
        """
        Apply one step of GNN message passing.

        Args:
            graph: Input graph structure with node and edge features

        Returns:
            Updated graph after one step of message passing
        """
        nodes, edge_features_in, receivers, senders, globals_, n_node, n_edge = graph

        # Note: globals_ is extracted but no longer used in computations

        # 1. Compute messages using the edge update module
        # Get sender node features
        sender_node_features = jax.tree.map(lambda n: n[senders], nodes)
        # Get receiver node features
        receiver_node_features = jax.tree.map(lambda n: n[receivers], nodes)

        # Generate messages
        messages = self.edge_update(
            edge_features_in,
            sender_node_features,
            receiver_node_features,
            None,
        )

        # 2. Aggregate messages for each node
        num_nodes = jax.tree.leaves(nodes)[0].shape[0]
        aggregated_messages = self.aggregate_fn(messages, receivers, num_nodes)

        # 3. Update node features using the node update module
        updated_nodes = self.node_update(
            nodes,
            aggregated_messages,
            None,
            None,
        )

        # Return updated graph
        return graph._replace(nodes=updated_nodes)


@partial(nnx.jit, static_argnames=("num_steps",))
def run_gnn_scan(
    gnn: CircuitGNN, graph: jraph.GraphsTuple, num_steps: int
) -> Tuple[jraph.GraphsTuple, List[jraph.GraphsTuple]]:
    """
    Run the GNN for multiple steps using scan for efficiency.

    Args:
        gnn: The CircuitGNN model to apply
        graph: Initial graph state
        num_steps: Number of steps to run

    Returns:
        final_graph: The graph after all steps
        all_graphs: List of graphs from each step (including initial)
    """

    def gnn_step(carry, _):
        graph = carry
        new_graph = gnn(graph)
        return new_graph, new_graph

    # Run scan
    final_graph, intermediate_graphs = jax.lax.scan(
        gnn_step, graph, xs=None, length=num_steps
    )

    # Combine initial graph with intermediate results
    all_graphs = [graph] + list(intermediate_graphs)

    return final_graph, all_graphs


def run_gnn_scan_with_loss(
    model: CircuitGNN,
    graph: jraph.GraphsTuple,
    num_steps: int,
    logits_original_shapes: List[Tuple],
    wires: List[jp.ndarray],
    x_data: jp.ndarray,
    y_data: jp.ndarray,
    loss_type: str,
    layer_sizes: Tuple[Tuple[int, int]],
) -> Tuple[jraph.GraphsTuple, List[jraph.GraphsTuple], jp.ndarray, List]:
    """
    Run the GNN for multiple steps with loss computation and graph updating at each step.

    This function combines model application with loss computation and graph updating,
    allowing for efficient computation of all steps and later indexing of a random step.

    Args:
        gnn: The CircuitGNN model to apply
        graph: Initial graph state
        num_steps: Number of steps to run
        logits_original_shapes: Original shapes of logits for reconstruction
        wires: Wire connection patterns
        x_data: Input data
        y_data: Target output data
        loss_type: Type of loss function to use
        layer_sizes: List of (nodes, group_size) tuples for each layer

    Returns:
        final_graph: The graph after all steps
        all_graphs: List of graphs from each step (including initial)
        all_losses: Array of losses from each step [num_steps+1]
        all_aux: List of auxiliary data from each step
    """
    from boolean_nca_cc.training.evaluation import get_loss_and_update_graph

    def gnn_step_with_loss(carry, _):
        current_graph = carry

        # Apply GNN
        model_updated_graph = model(current_graph)

        # Compute loss and update graph
        updated_graph, loss, current_logits, aux = get_loss_and_update_graph(
            model_updated_graph,
            logits_original_shapes,
            wires,
            x_data,
            y_data,
            loss_type,
            layer_sizes,
        )

        # Update graph globals with current update steps
        current_update_steps = (
            updated_graph.globals[..., 1] if updated_graph.globals is not None else 0
        )
        final_graph = updated_graph._replace(
            globals=jp.array([loss, current_update_steps + 1], dtype=jp.float32)
        )

        return final_graph, (final_graph, loss, current_logits, aux)

    # Run scan
    final_graph, step_outputs = jax.lax.scan(
        gnn_step_with_loss, graph, xs=None, length=num_steps
    )

    return final_graph, step_outputs
