"""
Self-attention based circuit update mechanism.

This module provides an alternative to GNN message passing by using
masked self-attention to update boolean circuit parameters.
"""

from functools import partial

import jax
import jax.numpy as jp
import jraph
from flax import nnx

from boolean_nca_cc.circuits.train import LossConfig
from boolean_nca_cc.models.attention.base import (
    AttentionBlock,
    PassThrough,
    ReZero,
    apply_knockout_mask,
    create_attention_mask,
    extract_node_features,
)


class CircuitSelfAttention(nnx.Module):
    """
    Self-attention based circuit update mechanism.

    Implements masked self-attention to update boolean circuit parameters.
    Uses ReZero for stable training throughout.
    """

    def __init__(
        self,
        n_node: int,
        circuit_hidden_dim: int = 16,
        arity: int = 2,
        attention_dim: int = 128,
        num_heads: int = 4,
        num_self_attn_layers: int = 1,
        mlp_dim: int | None = None,
        mlp_dim_multiplier: int = 2,
        dropout_rate: float = 0.0,
        use_attention_mask: bool = True,
        *,
        rngs: nnx.Rngs,
        type: str = "self_attention",
        use_node_loss: bool = False,
        re_zero_updates: bool = True,
        re_zero_attn: bool = True,
        warm_start: bool = True,
        use_intra_layer_PE: bool = False,
        use_layer_PE: bool = True,
    ):
        """
        Initialize the circuit self-attention model.

        Args:
            n_node: Fixed number of nodes in the circuit (used for attention mask precomputation;
                    learned weights are scale-free and work with any circuit size)
            circuit_hidden_dim: Dimension of hidden features in the circuit graphs
            arity: Number of inputs per gate in the boolean circuit
            attention_dim: Total dimension for attention mechanism
            num_heads: Number of attention heads
            num_self_attn_layers: Number of self-attention layers
            mlp_dim: Dimension of feed-forward network in attention blocks
            mlp_dim_multiplier: Multiplier for mlp_dim (default 2)
            dropout_rate: Dropout rate
            use_attention_mask: Whether to use self-attention masks based on circuit wiring
            rngs: Random number generators
            type: Type of model
            use_node_loss: Whether to include per-node loss in features
            re_zero_updates: Whether to use ReZero for updates
            re_zero_attn: Whether to use ReZero for attention
            use_intra_layer_PE: Whether to include intra-layer positional encodings in features
            use_layer_PE: Whether to include layer depth positional encodings in features
            warm_start: Whether to use warm start for training
        """
        self.n_node = int(n_node)
        self.arity = arity
        self.circuit_hidden_dim = circuit_hidden_dim
        self.attention_dim = attention_dim
        self.logit_dim = 2**arity
        self.dropout_rate = dropout_rate
        self.num_heads = num_heads
        self.use_attention_mask = use_attention_mask
        self.use_node_loss = use_node_loss
        self.use_intra_layer_PE = use_intra_layer_PE
        self.use_layer_PE = use_layer_PE
        self.warm_start = warm_start
        if mlp_dim is None:
            mlp_dim = attention_dim * mlp_dim_multiplier

        if attention_dim % num_heads != 0:
            raise ValueError(
                f"attention_dim ({attention_dim}) must be divisible by num_heads ({num_heads})"
            )

        # Compute input feature dimension dynamically based on PE flags
        input_feature_dim = self.logit_dim + circuit_hidden_dim  # logits + hidden (always)
        if self.use_intra_layer_PE:
            input_feature_dim += circuit_hidden_dim
        if self.use_layer_PE:
            input_feature_dim += circuit_hidden_dim
        if self.use_node_loss:
            input_feature_dim += 1

        # Input normalization + projection
        # LayerNorm on concatenated heterogeneous features (logits, hidden, PEs)
        # normalizes scale across different feature types before projection
        self.input_norm = nnx.LayerNorm(input_feature_dim, rngs=rngs)
        self.feature_proj = nnx.Linear(input_feature_dim, self.attention_dim, rngs=rngs)

        # Self-attention layers (using shared AttentionBlock with ReZero)
        self.attention_layers = nnx.List(
            [
                AttentionBlock(
                    dim=self.attention_dim,
                    mlp_dim=mlp_dim,
                    num_heads=num_heads,
                    dropout_rate=dropout_rate,
                    rngs=rngs,
                    re_zero=re_zero_attn,
                    warm_start=warm_start,
                )
                for _ in range(num_self_attn_layers)
            ]
        )

        # Final LayerNorm before output heads (standard Pre-LN practice)
        self.final_norm = nnx.LayerNorm(self.attention_dim, rngs=rngs)

        # Output projections with optional zero init
        self.logit_proj = nnx.Linear(
            self.attention_dim,
            self.logit_dim,
            use_bias=True,
            kernel_init=nnx.initializers.lecun_normal(),
            rngs=rngs,
        )
        self.hidden_proj = nnx.Linear(
            self.attention_dim,
            circuit_hidden_dim,
            use_bias=True,
            kernel_init=nnx.initializers.lecun_normal(),
            rngs=rngs,
        )

        # ReZero for final output residuals
        self.logit_rezero = ReZero(rngs=rngs) if re_zero_updates else PassThrough()
        self.hidden_rezero = ReZero(rngs=rngs) if re_zero_updates else PassThrough()

    def _create_attention_mask(
        self,
        senders: jp.ndarray,
        receivers: jp.ndarray,
        n_node: int,
    ) -> jp.ndarray:
        """Create attention mask using shared utility."""
        return create_attention_mask(senders, receivers, n_node, self.use_attention_mask)

    def __call__(
        self,
        graph: jraph.GraphsTuple,
        attention_mask: jp.ndarray | None = None,
    ) -> jraph.GraphsTuple:
        """
        Apply self-attention to update circuit parameters.

        Args:
            graph: Input graph structure with node and edge features
            attention_mask: Optional pre-computed attention mask

        Returns:
            Updated graph after self-attention
        """
        nodes, _edges, receivers, senders, _globals, _n_node, _n_edge = graph

        # Extract and concatenate node features
        features = extract_node_features(
            nodes, self.use_node_loss, self.use_intra_layer_PE, self.use_layer_PE
        )

        # Add batch dimension [1, n_node, feature_dim]
        features = features[None, ...]

        # Create attention mask if not provided
        # Derive n_node from graph (scale-free: works with any circuit size)
        # Note: this fallback path is only used outside JIT. In training/eval,
        # masks are always precomputed with a concrete n_node.
        if attention_mask is None:
            n_node = nodes["layer"].shape[0]
            attention_mask = self._create_attention_mask(senders, receivers, n_node)

        # Normalize + project features to attention dimension
        x = self.feature_proj(self.input_norm(features))

        # Apply self-attention layers
        for layer in self.attention_layers:
            x = layer(x, key_value=None, mask=attention_mask)

        # Final norm before output projections
        x = self.final_norm(x)

        # Project to updates
        logit_updates = self.logit_proj(x)[0]  # Remove batch dim
        hidden_updates = self.hidden_proj(x)[0]

        # Apply knockout mask
        logit_updates, hidden_updates = apply_knockout_mask(logit_updates, hidden_updates, nodes)

        # Apply ReZero residual updates
        updated_logits = nodes["logits"] + self.logit_rezero(logit_updates)
        updated_hidden = nodes["hidden"] + self.hidden_rezero(hidden_updates)

        # Create updated nodes
        updated_nodes = {**nodes, "logits": updated_logits, "hidden": updated_hidden}

        return graph._replace(nodes=updated_nodes)
