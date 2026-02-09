"""
Graph Neural Network implementation for boolean circuits.

This module provides the main GNN model for evolving boolean circuits
through sparse message passing. When use_attention=True, the aggregation
step uses sparse multi-head attention (O(E) compute/memory) that is
functionally equivalent to the topology-masked Transformer but without
the O(N^2) dense attention matrix.
"""

import jax
import jraph
from flax import nnx

from boolean_nca_cc.models.attention.base import extract_node_features
from boolean_nca_cc.models.gnn.aggregation import SparseMultiHeadAttention, aggregate_sum
from boolean_nca_cc.models.gnn.edge_update import EdgeUpdateModule
from boolean_nca_cc.models.gnn.node_update import NodeUpdateModule


class CircuitGNN(nnx.Module):
    """
    Graph Neural Network for evolving boolean circuits.

    Implements message passing on a graph representation of a boolean circuit.
    Supports two aggregation modes:
    - Sum aggregation (default): classical GNN with fixed aggregation
    - Sparse attention: learned, content-dependent aggregation (O(E) cost)

    The `attention_dim` parameter controls the latent dimension for messages
    and attention, analogous to `attention_dim` in CircuitSelfAttention.
    Setting the same value allows direct parameter-count comparison.
    """

    def __init__(
        self,
        circuit_hidden_dim: int = 16,
        attention_dim: int = 96,
        mlp_dim: int | None = None,
        mlp_dim_multiplier: int = 2,
        mlp_n_layers: int = 2,
        arity: int = 2,
        message_passing: bool = True,
        use_attention: bool = True,
        num_heads: int = 4,
        *,
        rngs: nnx.Rngs,
        type: str = "gnn",
        use_node_loss: bool = False,
        use_intra_layer_PE: bool = False,
        use_layer_PE: bool = True,
    ):
        """
        Initialize the Circuit GNN.

        Args:
            circuit_hidden_dim: Dimension of hidden features in the circuit graphs
            attention_dim: Dimension of the latent message/attention space.
                           Controls edge MLP output, attention Q/K/V, and aggregated
                           message size. Set to the same value as CircuitSelfAttention's
                           attention_dim for parameter-count parity.
            mlp_dim: FFN hidden layer size. If None, set to attention_dim * mlp_dim_multiplier
            mlp_dim_multiplier: Multiplier for mlp_dim (default 2)
            mlp_n_layers: Number of hidden layers for edge/node MLPs
            arity: Number of inputs per gate in the boolean circuit
            message_passing: Whether to use message passing or only self-updates
            use_attention: Whether to use sparse multi-head attention for aggregation
            num_heads: Number of attention heads (only used when use_attention=True)
            rngs: Random number generators
            type: Type of model
            use_node_loss: Whether to include per-node loss in features
            use_intra_layer_PE: Whether to include intra-layer positional encodings
            use_layer_PE: Whether to include layer depth positional encodings
        """
        self.arity = arity
        self.message_passing = message_passing
        self.use_attention = use_attention
        self.circuit_hidden_dim = circuit_hidden_dim
        self.attention_dim = attention_dim
        self.use_node_loss = use_node_loss
        self.use_intra_layer_PE = use_intra_layer_PE
        self.use_layer_PE = use_layer_PE
        self.mlp_dim = mlp_dim if mlp_dim is not None else attention_dim * mlp_dim_multiplier
        self.mlp_n_layers = mlp_n_layers

        # Shared PE/loss flags passed to all sub-modules
        pe_kwargs = {
            "use_intra_layer_PE": use_intra_layer_PE,
            "use_layer_PE": use_layer_PE,
            "use_node_loss": use_node_loss,
        }

        # Edge update: generates messages in attention_dim space
        # When using attention: sender-only (attention queries handle receiver-dependence)
        # When using sum: include receiver features (compensates for no learned weighting)
        self.edge_update = EdgeUpdateModule(
            edge_mlp_features=[self.mlp_dim] * mlp_n_layers,
            circuit_hidden_dim=circuit_hidden_dim,
            arity=arity,
            rngs=rngs,
            output_dim=attention_dim,
            use_receiver_features=not use_attention,
            **pe_kwargs,
        )

        # Node update: processes aggregated messages (attention_dim) + own features → deltas
        self.node_update = NodeUpdateModule(
            node_mlp_features=[self.mlp_dim] * mlp_n_layers,
            circuit_hidden_dim=circuit_hidden_dim,
            arity=arity,
            message_passing=message_passing,
            rngs=rngs,
            aggregated_message_dim=attention_dim,
            **pe_kwargs,
        )

        # Aggregation: either sum or sparse multi-head attention
        if use_attention:
            if attention_dim % num_heads != 0:
                raise ValueError(
                    f"attention_dim ({attention_dim}) must be divisible by num_heads ({num_heads})"
                )

            # Compute per-node feature dim for receiver queries
            logit_dim = 2**arity
            node_feature_dim = logit_dim + circuit_hidden_dim
            if use_intra_layer_PE:
                node_feature_dim += circuit_hidden_dim
            if use_layer_PE:
                node_feature_dim += circuit_hidden_dim
            if use_node_loss:
                node_feature_dim += 1

            # Pre-LN on receiver features before query projection
            self.receiver_query_norm = nnx.LayerNorm(node_feature_dim, rngs=rngs)
            # Pre-LN on messages before key/value projection
            self.message_kv_norm = nnx.LayerNorm(attention_dim, rngs=rngs)

            # Project receiver features to attention_dim for Q compatibility
            self.receiver_proj = nnx.Linear(node_feature_dim, attention_dim, rngs=rngs)

            self.attention_agg = SparseMultiHeadAttention(
                feature_dim=attention_dim,
                num_heads=num_heads,
                rngs=rngs,
                normalize_qk=True,
            )

    def __call__(self, graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
        """
        Apply one step of GNN message passing.

        Args:
            graph: Input graph structure with node and edge features

        Returns:
            Updated graph after one step of message passing
        """
        nodes = graph.nodes
        edge_features_in = graph.edges
        receivers = graph.receivers
        senders = graph.senders

        # 1. Compute messages using the edge update module → [E, attention_dim]
        sender_node_features = jax.tree.map(lambda n: n[senders], nodes)
        receiver_node_features = jax.tree.map(lambda n: n[receivers], nodes)

        messages = self.edge_update(
            edge_features_in,
            sender_node_features,
            receiver_node_features,
            None,
        )

        # 2. Aggregate messages for each node → [N, attention_dim]
        num_nodes = jax.tree.leaves(nodes)[0].shape[0]

        if self.use_attention:
            # Sparse multi-head attention aggregation
            receiver_node_all = extract_node_features(
                nodes, self.use_node_loss, self.use_intra_layer_PE, self.use_layer_PE
            )
            # Pre-LN on both sides (aligned with AttentionBlock convention)
            receiver_normed = self.receiver_query_norm(receiver_node_all)
            receiver_projected = self.receiver_proj(receiver_normed)
            messages_normed = self.message_kv_norm(messages)

            aggregated_messages = self.attention_agg(
                messages_normed,
                receivers,
                num_nodes,
                receiver_features=receiver_projected,
                sender_features=messages_normed,
            )
        else:
            # Simple sum aggregation
            aggregated_messages = aggregate_sum(messages, receivers, num_nodes)

        # 3. Update node features → delta logits + delta hidden
        updated_nodes = self.node_update(
            nodes,
            aggregated_messages,
            None,
            None,
        )

        return graph._replace(nodes=updated_nodes)
