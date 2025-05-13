"""
Graph Neural Network implementation for boolean circuits.

This module provides the main GNN model for evolving boolean circuits
through message passing.
"""

import jax
import jax.numpy as jp
import jraph
from flax import nnx
from typing import List
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
            globals_,
        )

        # 2. Aggregate messages for each node
        num_nodes = jax.tree.leaves(nodes)[0].shape[0]
        aggregated_messages = self.aggregate_fn(messages, receivers, num_nodes)

        # 3. Update node features using the node update module
        updated_nodes = self.node_update(
            nodes,
            aggregated_messages,
            None,
            globals_,
        )

        # Return updated graph
        return graph._replace(nodes=updated_nodes)


@partial(nnx.jit, static_argnames=("num_steps",))
def run_gnn_scan(
    gnn: CircuitGNN, graph: jraph.GraphsTuple, num_steps: int
) -> jraph.GraphsTuple:
    """
    Apply the GNN message passing iteratively for multiple steps using jax.lax.scan.

    Args:
        gnn: The CircuitGNN model
        graph: The initial graph
        num_steps: Number of message passing steps to perform

    Returns:
        Updated graph after num_steps of message passing
    """
    # Ensure edges field is properly initialized
    if graph.edges is None:
        # Initialize edges with the right shape
        senders_count = graph.senders.shape[0] if graph.senders.shape[0] > 0 else 0
        logit_dim = 2**gnn.arity
        hidden_dim = gnn.hidden_dim

        # Create empty edges with the right shape
        edges = jp.zeros((senders_count, logit_dim + hidden_dim), dtype=jp.float32)
        graph = graph._replace(edges=edges)

    def scan_body(carry_graph, _):
        # Apply one step of GNN message passing
        updated_graph = gnn(carry_graph)
        return updated_graph, None

    # Run the scan
    final_graph, _ = jax.lax.scan(scan_body, graph, None, length=num_steps)

    return final_graph
