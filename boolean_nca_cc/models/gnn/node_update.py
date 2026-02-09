"""
Node update modules for graph neural networks.

This module provides node update functions for processing incoming messages and
updating node features in a graph neural network.
"""

import itertools

import jax.numpy as jp
from flax import nnx

from boolean_nca_cc.models.attention.base import ReZero, extract_node_features

# Type aliases for clarity
NodeType = dict[str, jp.ndarray]


class NodeUpdateModule(nnx.Module):
    """
    Node update module for GNN message passing.

    Updates node features based on current state and incoming messages.
    Uses Pre-LN normalization and GELU activation, aligned with the
    attention model conventions.
    """

    def __init__(
        self,
        node_mlp_features: list[int],
        circuit_hidden_dim: int,
        arity: int,
        message_passing: bool = True,
        *,
        rngs: nnx.Rngs,
        aggregated_message_dim: int | None = None,
        use_intra_layer_PE: bool = False,
        use_layer_PE: bool = True,
        use_node_loss: bool = False,
    ):
        """
        Initialize the node update module.

        Args:
            node_mlp_features: Hidden layer sizes for the node MLP
            circuit_hidden_dim: Dimension of hidden features
            arity: Number of inputs per gate in the boolean circuit
            message_passing: Whether to use message passing or only self-updates
            rngs: Random number generators
            aggregated_message_dim: Dimension of aggregated messages from edge update +
                                    aggregation. If None, defaults to logit_dim + circuit_hidden_dim.
                                    Set to attention_dim for parity with the self-attention model.
            use_intra_layer_PE: Whether to include intra-layer positional encodings
            use_layer_PE: Whether to include layer depth positional encodings
            use_node_loss: Whether to include per-node loss in features
        """
        self.arity = arity
        self.circuit_hidden_dim = circuit_hidden_dim
        self.message_passing = message_passing
        self.logit_dim = 2**arity
        self.use_intra_layer_PE = use_intra_layer_PE
        self.use_layer_PE = use_layer_PE
        self.use_node_loss = use_node_loss

        # Compute node feature dim dynamically based on PE flags
        node_feature_dim = self.logit_dim + circuit_hidden_dim  # logits + hidden (always)
        if use_intra_layer_PE:
            node_feature_dim += circuit_hidden_dim
        if use_layer_PE:
            node_feature_dim += circuit_hidden_dim
        if use_node_loss:
            node_feature_dim += 1

        # Message dim: configurable (attention_dim) or default (logit_dim + hidden_dim)
        msg_dim = (
            aggregated_message_dim
            if aggregated_message_dim is not None
            else (self.logit_dim + circuit_hidden_dim)
        )

        mlp_input_size = node_feature_dim + msg_dim if message_passing else node_feature_dim

        # Output: delta logits + delta hidden
        mlp_output_size = self.logit_dim + circuit_hidden_dim

        # Pre-LN on input features (single LN on concatenated features,
        # aligned with attention model's self.input_norm convention)
        self.input_norm = nnx.LayerNorm(mlp_input_size, rngs=rngs)

        # Node MLP with Pre-LN + GELU (aligned with attention model FFN)
        mlp_features = [mlp_input_size, *node_mlp_features, mlp_output_size]
        mlp_layers = []

        for i, (in_f, out_f) in enumerate(itertools.pairwise(mlp_features)):
            if i == len(mlp_features) - 2:
                # Final layer: no activation
                mlp_layers.append(
                    nnx.Linear(
                        in_f,
                        out_f,
                        kernel_init=nnx.initializers.kaiming_normal(),
                        rngs=rngs,
                    )
                )
            else:
                mlp_layers.append(nnx.Linear(in_f, out_f, rngs=rngs))
                mlp_layers.append(nnx.LayerNorm(out_f, rngs=rngs))
                mlp_layers.append(nnx.gelu)

        self.mlp = nnx.Sequential(*mlp_layers)

        # ReZero learnable scaling for residual updates
        self.logit_scale = ReZero(rngs=rngs)
        self.hidden_scale = ReZero(rngs=rngs)

    def __call__(
        self,
        nodes: NodeType,
        sent_attributes: jp.ndarray,
        received_attributes: jp.ndarray,
        globals_=None,
    ):
        """
        Update node features using incoming messages and current state.

        Args:
            nodes: Current node features
            sent_attributes: Aggregated messages from incoming edges [num_nodes, msg_dim]
            received_attributes: Unused (kept for API compatibility)
            globals_: Unused (kept for API compatibility)

        Returns:
            Updated node features dict
        """
        # Extract node features using the shared utility (respects PE flags)
        node_features = extract_node_features(
            nodes, self.use_node_loss, self.use_intra_layer_PE, self.use_layer_PE
        )

        # Build MLP input
        if self.message_passing and sent_attributes is not None:
            mlp_input = jp.concatenate([node_features, sent_attributes], axis=-1)
        else:
            mlp_input = node_features

        # Pre-LN on concatenated input
        mlp_input = self.input_norm(mlp_input)

        # Apply MLP to get delta updates
        delta_combined = self.mlp(mlp_input)

        # Split into logit and hidden deltas
        delta_logits = delta_combined[..., : self.logit_dim]
        delta_hidden = delta_combined[..., self.logit_dim :]

        # Apply ReZero scaling
        scaled_delta_logits = self.logit_scale(delta_logits)
        scaled_delta_hidden = self.hidden_scale(delta_hidden)

        # Only update gate nodes (layer > 0), not input nodes
        is_gate_node = nodes["layer"] > 0

        # Handle gate knockout mask (permanent damage)
        gate_knockout_mask = nodes.get("gate_knockout_mask", None)
        if gate_knockout_mask is not None:
            update_allowed = is_gate_node & (gate_knockout_mask == 1.0)
        else:
            update_allowed = is_gate_node

        update_mask = update_allowed[:, None]

        # Residual updates with masking
        updated_logits = jp.where(
            update_mask,
            nodes["logits"] + scaled_delta_logits,
            nodes["logits"],
        )
        updated_hidden = jp.where(
            update_mask,
            nodes["hidden"] + scaled_delta_hidden,
            nodes["hidden"],
        )

        # Preserve all other node features (PEs, layer, loss, etc.)
        new_nodes = dict(nodes.items())
        new_nodes["logits"] = updated_logits
        new_nodes["hidden"] = updated_hidden

        return new_nodes
