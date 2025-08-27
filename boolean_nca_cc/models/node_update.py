"""
Node update modules for graph neural networks.

This module provides node update functions for processing incoming messages and
updating node features in a graph neural network.
"""

import itertools

import jax
import jax.numpy as jp
from flax import nnx

# Type aliases for clarity
NodeType = dict[str, jp.ndarray]


class NodeUpdateModule(nnx.Module):
    """
    Node update module for GNN message passing.

    Updates node features based on current state and incoming messages.
    """

    def __init__(
        self,
        node_mlp_features: list[int],
        circuit_hidden_dim: int,
        arity: int,
        message_passing: bool = True,
        *,
        rngs: nnx.Rngs,
        zero_init: bool = True,
        re_zero_update: bool = False,
    ):
        """
        Initialize the node update module.

        Args:
            node_mlp_features: Hidden layer sizes for the node MLP
            circuit_hidden_dim: Dimension of hidden features
            arity: Number of inputs per gate in the boolean circuit
            message_passing: Whether to use message passing or only self-updates
            rngs: Random number generators
            zero_init: Whether to initialize weights and biases to zero
            re_zero_update: Whether to use learnable update residual rate
        """
        self.arity = arity
        self.circuit_hidden_dim = circuit_hidden_dim
        self.message_passing = message_passing
        self.logit_dim = 2**arity
        pe_dim = circuit_hidden_dim  # Dimension for positional encodings

        # Calculate MLP input size
        # Current features: Logits, Hidden, Layer PE, Intra-Layer PE, Loss
        current_features_size = (
            self.logit_dim + circuit_hidden_dim + pe_dim + pe_dim + 1
        )  # +1 for loss feature

        if message_passing:
            # If using message passing, include aggregated messages
            # Edge message contains logits + hidden derived features
            aggregated_message_size = self.logit_dim + circuit_hidden_dim
            mlp_input_size = current_features_size + aggregated_message_size
        else:
            # Without message passing, only use current node features
            mlp_input_size = current_features_size

        # Output needs to contain updated logits and hidden features
        mlp_output_size = self.logit_dim + circuit_hidden_dim

        # Add feature normalization layers
        self.logits_norm = nnx.LayerNorm(
            self.logit_dim,
            epsilon=1e-5,
            rngs=rngs,
        )
        self.hidden_norm = nnx.LayerNorm(
            circuit_hidden_dim,
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
        # Add normalization for loss feature
        self.loss_norm = nnx.LayerNorm(
            1,  # Loss is a scalar per node
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

        for i, (in_f, out_f) in enumerate(itertools.pairwise(mlp_features)):
            # Special initialization for the final layer
            if i == len(mlp_features) - 2:
                # Use small random initialization for weights to ensure gradient flow
                # Keep logit outputs close to zero initially, but allow hidden features to learn
                final_linear = nnx.Linear(
                    in_f,
                    out_f,
                    # kernel_init=nnx.initializers.normal(
                    #     stddev=1e-4
                    # ),  # Small random init
                    kernel_init=nnx.initializers.zeros
                    if zero_init
                    else nnx.initializers.kaiming_normal(),
                    bias_init=jax.nn.initializers.zeros
                    if zero_init
                    else nnx.initializers.normal(stddev=1e-4),
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

        # Re-zero learnable scaling parameters
        self.logit_scale = (
            nnx.Param(
                jp.zeros(1),
                name="logit_scale",
                rngs=rngs,
            )
            if re_zero_update
            else 1.0
        )

        self.hidden_scale = (
            nnx.Param(
                jp.zeros(1),
                name="hidden_scale",
                rngs=rngs,
            )
            if re_zero_update
            else 1.0
        )

    def __call__(
        self,
        nodes: NodeType,
        sent_attributes: jp.ndarray,
        received_attributes: jp.ndarray,
        globals_=None,  # Keep for compatibility but ignore
    ):
        """
        Update node features using incoming messages and current state.

        Args:
            nodes: Current node features
            sent_attributes: Aggregated messages from incoming edges
            received_attributes: Features of received messages (unused)
            globals_: Global features (ignored, kept for compatibility)

        Returns:
            Updated node features
        """
        # Extract current node features
        current_logits = nodes["logits"]  # Shape: [num_nodes, 2**arity]
        current_hidden = nodes["hidden"]  # Shape: [num_nodes, circuit_hidden_dim]
        current_layer_pe = nodes["layer_pe"]  # Shape: [num_nodes, pe_dim]
        current_intra_layer_pe = nodes["intra_layer_pe"]  # Shape: [num_nodes, pe_dim]
        current_loss = nodes["loss"]  # Shape: [num_nodes]

        # Normalize input features
        normalized_logits = self.logits_norm(current_logits)
        normalized_hidden = self.hidden_norm(current_hidden)
        normalized_layer_pe = self.layer_pe_norm(current_layer_pe)
        normalized_intra_layer_pe = self.intra_layer_pe_norm(current_intra_layer_pe)
        # Add dimension for loss normalization: [num_nodes] -> [num_nodes, 1]
        normalized_loss = self.loss_norm(current_loss[:, None])  # [num_nodes, 1]

        # Combine normalized features
        current_node_combined_features = jp.concatenate(
            [
                normalized_logits,
                normalized_hidden,
                normalized_layer_pe,
                normalized_intra_layer_pe,
                normalized_loss,  # Include normalized loss feature
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
        delta_logits = delta_combined_features[..., : self.logit_dim]
        delta_hidden = delta_combined_features[..., self.logit_dim :]

        # Apply re-zero scaling to deltas
        scaled_delta_logits = self.logit_scale * delta_logits
        scaled_delta_hidden = self.hidden_scale * delta_hidden

        # Apply residual update only to non-input nodes (layer > 0)
        is_gate_node = nodes["layer"] > 0

        # Check for gate knockout mask to prevent updates to knocked-out gates
        gate_knockout_mask = nodes.get("gate_knockout_mask", None)
        if gate_knockout_mask is not None:
            # Knocked-out gates (mask == 0.0) should not receive updates
            # Only allow updates to active gates (mask == 1.0) that are also gate nodes
            update_allowed_logits = is_gate_node & (gate_knockout_mask == 1.0)
            update_allowed_hidden = is_gate_node & (gate_knockout_mask == 1.0)
        else:
            # No knockout mask, allow updates to all gate nodes
            update_allowed_logits = is_gate_node
            update_allowed_hidden = is_gate_node

        # Ensure mask matches feature dimensions for broadcasting
        update_allowed_logits_mask = update_allowed_logits[:, None]
        update_allowed_hidden_mask = update_allowed_hidden[:, None]

        updated_logits = jp.where(
            update_allowed_logits_mask,
            current_logits + scaled_delta_logits,
            current_logits,
        )
        updated_hidden = jp.where(
            update_allowed_hidden_mask,
            current_hidden + scaled_delta_hidden,
            current_hidden,
        )

        # Update only the 'logits' and 'hidden' fields, preserving others
        new_node_features = dict(nodes.items())
        new_node_features["logits"] = updated_logits
        new_node_features["hidden"] = updated_hidden

        return new_node_features
