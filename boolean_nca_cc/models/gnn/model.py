"""
Graph Neural Network implementation for boolean circuits.

This module provides the main GNN model for evolving boolean circuits
through message passing.
"""

from functools import partial

import jax
import jax.numpy as jp
import jraph
from flax import nnx

from boolean_nca_cc.circuits.train import LossConfig
from boolean_nca_cc.models.gnn.aggregation import AttentionAggregation, aggregate_sum
from boolean_nca_cc.models.gnn.edge_update import EdgeUpdateModule
from boolean_nca_cc.models.gnn.node_update import NodeUpdateModule
from boolean_nca_cc.utils.graph_builder import GraphGlobals


class CircuitGNN(nnx.Module):
    """
    Graph Neural Network for evolving boolean circuits.

    Implements message passing on a graph representation of a boolean circuit.
    """

    def __init__(
        self,
        circuit_hidden_dim: int = 16,
        mlp_dim: int
        | None = None,  # If None, will be set to circuit_hidden_dim * mlp_dim_multiplier (default 2)
        mlp_dim_multiplier: int = 2,
        mlp_n_layers: int = 2,
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
            circuit_hidden_dim: Dimension of hidden features
            mlp_dim: Hidden layer size for the MLP
            mlp_n_layers: Number of layers for the MLP
            arity: Number of inputs per gate in the boolean circuit
            message_passing: Whether to use message passing or only self-updates
            use_attention: Whether to use attention-based message aggregation
            rngs: Random number generators
            type: Type of model
        """
        self.arity = arity
        self.message_passing = message_passing
        self.circuit_hidden_dim = circuit_hidden_dim
        self.mlp_dim = mlp_dim if mlp_dim is not None else circuit_hidden_dim * mlp_dim_multiplier
        self.mlp_n_layers = mlp_n_layers

        # Create the node and edge update modules
        self.node_update = NodeUpdateModule(
            node_mlp_features=[self.mlp_dim] * mlp_n_layers,
            circuit_hidden_dim=circuit_hidden_dim,
            arity=arity,
            message_passing=message_passing,
            rngs=rngs,
        )

        self.edge_update = EdgeUpdateModule(
            edge_mlp_features=[self.mlp_dim] * mlp_n_layers,
            circuit_hidden_dim=circuit_hidden_dim,
            arity=arity,
            rngs=rngs,
        )

        # Configure the aggregation function
        if use_attention:
            logit_dim = 2**arity
            self.aggregate_fn = AttentionAggregation(
                feature_dim=circuit_hidden_dim + logit_dim,
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
) -> tuple[jraph.GraphsTuple, list[jraph.GraphsTuple]]:
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
    final_graph, intermediate_graphs = jax.lax.scan(gnn_step, graph, xs=None, length=num_steps)

    # Combine initial graph with intermediate results
    all_graphs = [graph, *list(intermediate_graphs)]

    return final_graph, all_graphs


# DEPRECATED: Use run_model_scan_with_loss from boolean_nca_cc.training.evaluation instead
# This function is kept for backward compatibility but will be removed in a future version.
def run_gnn_scan_with_loss(
    model: CircuitGNN,
    graph: jraph.GraphsTuple,
    num_steps: int,
    logits_original_shapes: list[tuple],
    wires: list[jp.ndarray],
    x_data: jp.ndarray,
    y_data: jp.ndarray,
    loss_cfg: LossConfig,
    layer_sizes: tuple[tuple[int, int], ...],
    data_fraction: float = 1.0,
    scan_key: jax.random.PRNGKey = None,
    gradient_checkpointing: bool = False,
) -> tuple[jraph.GraphsTuple, list[jraph.GraphsTuple], jp.ndarray, list]:
    """
    DEPRECATED: Use run_model_scan_with_loss from boolean_nca_cc.training.evaluation instead.

    This function wraps the unified run_model_scan_with_loss for backward compatibility.
    """
    import warnings

    from boolean_nca_cc.training.evaluation import run_model_scan_with_loss

    warnings.warn(
        "run_gnn_scan_with_loss is deprecated. "
        "Use run_model_scan_with_loss from boolean_nca_cc.training.evaluation instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    return run_model_scan_with_loss(
        model=model,
        graph=graph,
        num_steps=num_steps,
        logits_original_shapes=logits_original_shapes,
        wires=wires,
        x_data=x_data,
        y_data=y_data,
        loss_cfg=loss_cfg,
        layer_sizes=layer_sizes,
        data_fraction=data_fraction,
        scan_key=scan_key,
        gradient_checkpointing=gradient_checkpointing,
    )
