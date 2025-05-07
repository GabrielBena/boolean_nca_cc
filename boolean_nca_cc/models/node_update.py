"""
Node update modules for graph neural networks.

This module provides node update functions for processing incoming messages and
updating node features in a graph neural network.
"""

import jax
import jax.numpy as jp
from flax import nnx
from typing import List, Dict


# Type aliases for clarity
NodeType = Dict[str, jp.ndarray]


class NodeUpdateModule(nnx.Module):
    """
    Node update module for GNN message passing.

    Updates node features based on current state and incoming messages.
    """

    def __init__(
        self,
        node_mlp_features: List[int],
        hidden_dim: int,
        arity: int,
        message_passing: bool = True,
        *,
        rngs: nnx.Rngs,
    ):
        """
        Initialize the node update module.

        Args:
            node_mlp_features: Hidden layer sizes for the node MLP
            hidden_dim: Dimension of hidden features
            arity: Number of inputs per gate in the boolean circuit
            message_passing: Whether to use message passing or only self-updates
            rngs: Random number generators
        """
        self.arity = arity
        self.hidden_dim = hidden_dim
        self.message_passing = message_passing
        self.logit_dim = 2**arity
        pe_dim = hidden_dim  # Dimension for positional encodings

        # Calculate MLP input size
        # Current features: Logits, Hidden, Layer PE, Intra-Layer PE
        current_features_size = self.logit_dim + hidden_dim + pe_dim + pe_dim
        global_feature_size = 2  # Assuming globals_ is [loss, update_steps]

        if message_passing:
            # If using message passing, include aggregated messages
            # Edge message contains logits + hidden derived features
            aggregated_message_size = self.logit_dim + hidden_dim
            mlp_input_size = current_features_size + aggregated_message_size
            mlp_input_size += global_feature_size  # Add global feature size
        else:
            # Without message passing, only use current node features
            mlp_input_size = current_features_size
            mlp_input_size += global_feature_size  # Add global feature size

        # Output needs to contain updated logits and hidden features
        mlp_output_size = self.logit_dim + hidden_dim

        # Add feature normalization layers
        self.logits_norm = nnx.LayerNorm(
            self.logit_dim,
            epsilon=1e-5,
            rngs=rngs,
        )
        self.hidden_norm = nnx.LayerNorm(
            hidden_dim,
            epsilon=1e-5,
            rngs=rngs,
        )
        self.layer_pe_norm = nnx.LayerNorm(
            pe_dim,
            epsilon=1e-5,
            rngs=rngs,
        )
        self.intra_layer_pe_norm = nnx.LayerNorm(
            pe_dim,
            epsilon=1e-5,
            rngs=rngs,
        )

        if message_passing:
            self.message_norm = nnx.LayerNorm(
                aggregated_message_size,
                epsilon=1e-5,
                rngs=rngs,
            )

        # Define MLP architecture with batch normalization
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
                    nnx.LayerNorm(
                        out_f,
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
        """
        Update node features using incoming messages and current state.

        Args:
            nodes: Current node features
            sent_attributes: Aggregated messages from incoming edges
            received_attributes: Features of received messages (unused)
            globals_: Global features [loss, update_steps]

        Returns:
            Updated node features
        """
        # Extract current node features
        current_logits = nodes["logits"]  # Shape: [num_nodes, 2**arity]
        current_hidden = nodes["hidden"]  # Shape: [num_nodes, hidden_dim]
        current_layer_pe = nodes["layer_pe"]  # Shape: [num_nodes, pe_dim]
        current_intra_layer_pe = nodes["intra_layer_pe"]  # Shape: [num_nodes, pe_dim]

        # Normalize input features
        normalized_logits = self.logits_norm(current_logits)
        normalized_hidden = self.hidden_norm(current_hidden)
        normalized_layer_pe = self.layer_pe_norm(current_layer_pe)
        normalized_intra_layer_pe = self.intra_layer_pe_norm(current_intra_layer_pe)

        # Combine normalized features
        current_node_combined_features = jp.concatenate(
            [
                normalized_logits,
                normalized_hidden,
                normalized_layer_pe,
                normalized_intra_layer_pe,
            ],
            axis=-1,
        )

        # Broadcast global feature to match the number of nodes
        num_nodes = list(nodes.values())[0].shape[0]

        # Globals are [loss, update_steps], both are used for the optimization
        broadcasted_globals = jp.repeat(
            jp.reshape(globals_, (1, -1)), num_nodes, axis=0
        )

        # Determine inputs based on message passing flag
        if self.message_passing and sent_attributes is not None:
            # Normalize messages
            normalized_messages = self.message_norm(sent_attributes)
            # Input = current_features + normalized_messages
            mlp_input = jp.concatenate(
                [current_node_combined_features, normalized_messages], axis=-1
            )
            mlp_input = jp.concatenate(
                [mlp_input, broadcasted_globals], axis=-1
            )  # Add globals
        else:
            # Input = current_features only
            mlp_input = current_node_combined_features
            mlp_input = jp.concatenate(
                [mlp_input, broadcasted_globals], axis=-1
            )  # Add globals

        # Apply MLP to get the delta (change) in features
        delta_combined_features = self.mlp(mlp_input)

        # Split the delta into logit and hidden components
        delta_logits = delta_combined_features[..., : self.logit_dim]
        delta_hidden = delta_combined_features[..., self.logit_dim :]

        # Apply residual update only to non-input nodes (layer > 0)
        is_gate_node = nodes["layer"] > 0
        # Ensure mask matches feature dimensions for broadcasting
        is_gate_node_logits_mask = is_gate_node[:, None]
        is_gate_node_hidden_mask = is_gate_node[:, None]

        updated_logits = jp.where(
            is_gate_node_logits_mask, current_logits + delta_logits, current_logits
        )
        updated_hidden = jp.where(
            is_gate_node_hidden_mask, current_hidden + delta_hidden, current_hidden
        )

        # Update only the 'logits' and 'hidden' fields, preserving others
        new_node_features = {k: v for k, v in nodes.items()}
        new_node_features["logits"] = updated_logits
        new_node_features["hidden"] = updated_hidden

        return new_node_features
