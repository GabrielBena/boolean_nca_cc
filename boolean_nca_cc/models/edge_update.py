"""
Edge update modules for graph neural networks.

This module provides edge update functions for generating messages between nodes.
"""

import jax
import jax.numpy as jp
from flax import nnx
from typing import List, Dict


# Type aliases for clarity
NodeType = Dict[str, jp.ndarray]
EdgeType = jp.ndarray


class EdgeUpdateModule(nnx.Module):
    """
    Edge update module for GNN message passing.

    Generates messages to be sent along edges from sender nodes to receiver nodes.
    """

    def __init__(
        self,
        edge_mlp_features: List[int],
        hidden_dim: int,
        arity: int,
        *,
        rngs: nnx.Rngs,
    ):
        """
        Initialize the edge update module.

        Args:
            edge_mlp_features: Hidden layer sizes for the edge MLP
            hidden_dim: Dimension of hidden features
            arity: Number of inputs per gate in the boolean circuit
            rngs: Random number generators
        """
        self.hidden_dim = hidden_dim
        self.arity = arity
        self.logit_dim = 2**arity
        pe_dim = hidden_dim  # Dimension for positional encodings

        # Input: Sender's Logits, Hidden, Layer PE, Intra-Layer PE
        self.arity = arity
        self.mlp_input_size = self.logit_dim + hidden_dim + pe_dim + pe_dim
        # Output: Features used for aggregation (Logits + Hidden dimensions)
        self.mlp_output_size = self.logit_dim + hidden_dim
        self.mlp_features = [
            self.mlp_input_size,
            *edge_mlp_features,
            self.mlp_output_size,
        ]

        # Define Edge MLP architecture with batch normalization
        edge_mlp_layers = []
        for i, (in_f, out_f) in enumerate(
            zip(self.mlp_features[:-1], self.mlp_features[1:])
        ):
            edge_mlp_layers.append(nnx.Linear(in_f, out_f, rngs=rngs))
            # Add BatchNorm and ReLU except for the last layer
            if i < len(self.mlp_features) - 2:
                edge_mlp_layers.append(
                    nnx.LayerNorm(
                        num_features=out_f,
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
        globals_=None,  # Keep for compatibility but ignore
    ):
        """
        Generate messages to be sent along edges.

        Args:
            edge_features: Features of the edge (if any)
            sender_node_features: Features of the sender node
            receiver_node_features: Features of the receiver node (unused)
            globals_: Global features (ignored, kept for compatibility)

        Returns:
            Message features to be sent along the edge [num_edges, logit_dim + hidden_dim]
        """
        # Extract sender's features
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
        message = self.edge_mlp(sender_combined_features)
        return message  # Shape: [num_edges, logit_dim + hidden_dim]
