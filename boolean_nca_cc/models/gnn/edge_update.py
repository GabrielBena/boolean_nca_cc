"""
Edge update modules for graph neural networks.

This module provides edge update functions for generating messages between nodes.

Design note: with bidirectional edges, every (i,j) pair has both an i→j and j→i
edge in the senders/receivers lists. This means sender-only messages already carry
information from both directions. When using attention aggregation, receiver-dependent
weighting is handled by the attention queries -- so sender-only messages suffice.
For sum aggregation, receiver features can optionally be included in the edge MLP
to compensate for the lack of learned weighting.
"""

import itertools

import jax.numpy as jp
from flax import nnx

from boolean_nca_cc.models.attention.base import extract_node_features

# Type aliases for clarity
NodeType = dict[str, jp.ndarray]
EdgeType = jp.ndarray


class EdgeUpdateModule(nnx.Module):
    """
    Edge update module for GNN message passing.

    Generates messages to be sent along edges. The output dimension is
    configurable via `output_dim` (defaults to attention_dim when set,
    or logit_dim + hidden_dim otherwise).
    """

    def __init__(
        self,
        edge_mlp_features: list[int],
        circuit_hidden_dim: int,
        arity: int,
        *,
        rngs: nnx.Rngs,
        output_dim: int | None = None,
        use_intra_layer_PE: bool = False,
        use_layer_PE: bool = True,
        use_node_loss: bool = False,
        use_receiver_features: bool = False,
    ):
        """
        Initialize the edge update module.

        Args:
            edge_mlp_features: Hidden layer sizes for the edge MLP
            circuit_hidden_dim: Dimension of hidden features
            arity: Number of inputs per gate in the boolean circuit
            rngs: Random number generators
            output_dim: Dimension of output messages. If None, defaults to
                        logit_dim + circuit_hidden_dim. Set to attention_dim
                        for parity with the self-attention model.
            use_intra_layer_PE: Whether to include intra-layer positional encodings
            use_layer_PE: Whether to include layer depth positional encodings
            use_node_loss: Whether to include per-node loss in features
            use_receiver_features: Whether to include receiver features in message
                                   computation. False when using attention aggregation.
                                   True is useful for sum aggregation.
        """
        self.circuit_hidden_dim = circuit_hidden_dim
        self.arity = arity
        self.logit_dim = 2**arity
        self.use_intra_layer_PE = use_intra_layer_PE
        self.use_layer_PE = use_layer_PE
        self.use_node_loss = use_node_loss
        self.use_receiver_features = use_receiver_features

        # Compute per-node feature dim dynamically based on PE flags
        node_feature_dim = self.logit_dim + circuit_hidden_dim  # logits + hidden (always)
        if use_intra_layer_PE:
            node_feature_dim += circuit_hidden_dim
        if use_layer_PE:
            node_feature_dim += circuit_hidden_dim
        if use_node_loss:
            node_feature_dim += 1

        # Input: sender features (+ optionally receiver features)
        if use_receiver_features:
            self.mlp_input_size = node_feature_dim * 2
        else:
            self.mlp_input_size = node_feature_dim

        # Output: configurable (attention_dim or default)
        self.mlp_output_size = output_dim if output_dim is not None else (self.logit_dim + circuit_hidden_dim)

        # Pre-LN on input features
        self.sender_norm = nnx.LayerNorm(node_feature_dim, rngs=rngs)
        if use_receiver_features:
            self.receiver_norm = nnx.LayerNorm(node_feature_dim, rngs=rngs)

        # Edge MLP with Pre-LN + GELU (aligned with attention model conventions)
        mlp_features = [
            self.mlp_input_size,
            *edge_mlp_features,
            self.mlp_output_size,
        ]

        edge_mlp_layers = []
        for i, (in_f, out_f) in enumerate(itertools.pairwise(mlp_features)):
            edge_mlp_layers.append(
                nnx.Linear(
                    in_f,
                    out_f,
                    rngs=rngs,
                    kernel_init=nnx.initializers.kaiming_normal(),
                )
            )
            # Add LayerNorm and GELU except for the last layer
            if i < len(mlp_features) - 2:
                edge_mlp_layers.append(nnx.LayerNorm(num_features=out_f, rngs=rngs))
                edge_mlp_layers.append(nnx.gelu)

        self.edge_mlp = nnx.Sequential(*edge_mlp_layers)

    def __call__(
        self,
        edge_features: EdgeType,
        sender_node_features: NodeType,
        receiver_node_features: NodeType,
        globals_=None,
    ):
        """
        Generate messages to be sent along edges.

        Returns:
            Message features [num_edges, output_dim]
        """
        sender_features = extract_node_features(
            sender_node_features, self.use_node_loss, self.use_intra_layer_PE, self.use_layer_PE
        )

        sender_normed = self.sender_norm(sender_features)

        if self.use_receiver_features:
            receiver_features = extract_node_features(
                receiver_node_features, self.use_node_loss, self.use_intra_layer_PE, self.use_layer_PE
            )
            receiver_normed = self.receiver_norm(receiver_features)
            mlp_input = jp.concatenate([sender_normed, receiver_normed], axis=-1)
        else:
            mlp_input = sender_normed

        return self.edge_mlp(mlp_input)
