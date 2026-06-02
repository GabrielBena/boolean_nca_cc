"""
Gathered Neighborhood Attention for circuit optimization.

This module replaces the dense NxN attention matrix with gathered dense attention
over each node's local neighborhood. For each node, neighbor features are gathered
into a padded [max_neighbors, D] tensor and attention is computed as a standard
batched dense matmul -- fully GPU-friendly (einsum, softmax) with no scatter/gather
atomics, while only attending to actual graph neighbors.

Complexity: O(N * max_degree * D) -- between dense O(N^2 * D) and sparse-scatter O(E * D).
At small N the constant factors win over scatter-based sparse attention; at large N the
sub-quadratic scaling wins over dense attention. Best of both worlds.

The model structure mirrors CircuitSelfAttention exactly (same input projection,
FFN, output heads, ReZero, Pre-LN) so parameter counts are directly comparable.
"""

import jax.numpy as jp
import jraph
from flax import nnx

from boolean_nca_cc.models.attention.base import (
    GatheredAttentionBlock,
    PassThrough,
    ReZero,
    apply_knockout_mask,
    build_neighbor_indices,
    extract_node_features,
)


class CircuitGatheredAttention(nnx.Module):
    """
    Gathered neighborhood attention for circuit optimization.

    Computes multi-head attention restricted to each node's graph neighbors
    using dense batched operations (no scatter/gather). Structurally identical
    to CircuitSelfAttention — same projections, FFN, ReZero — but with O(N * d_max)
    attention instead of O(N^2).
    """

    def __init__(
        self,
        circuit_hidden_dim: int = 16,
        arity: int = 2,
        attention_dim: int = 96,
        num_heads: int = 8,
        mlp_dim: int | None = None,
        mlp_dim_multiplier: int = 2,
        max_neighbors: int = 16,
        use_attention_mask: bool = True,
        *,
        rngs: nnx.Rngs,
        type: str = "gathered_attention",
        use_node_loss: bool = False,
        re_zero_updates: bool = True,
        re_zero_attn: bool = True,
        warm_start: bool = True,
        use_intra_layer_PE: bool = False,
        use_layer_PE: bool = True,
        use_dist_pe: bool = False,
        use_rwse: bool = False,
        rwse_k: int = 8,
        # Accepted from self_attention.yaml defaults (ignored — not used by this model)
        num_self_attn_layers: int = 1,
        dropout_rate: float = 0.0,
        n_node: int | None = None,
    ):
        """
        Initialize the gathered neighborhood attention model.

        Args:
            circuit_hidden_dim: Dimension of hidden features in the circuit graphs
            arity: Number of inputs per gate in the boolean circuit
            attention_dim: Total dimension for attention mechanism
            num_heads: Number of attention heads
            mlp_dim: Dimension of feed-forward network hidden layer
            mlp_dim_multiplier: Multiplier for mlp_dim when mlp_dim is None
            max_neighbors: Fixed width of the gathered neighbor tensor. Must be >=
                the maximum gate degree in any circuit the model will see.
                Controls the attention compute: O(N * max_neighbors * D).
            use_attention_mask: Whether to restrict attention to graph neighbors
            rngs: Random number generators
            type: Type of model (for Hydra config)
            use_node_loss: Whether to include per-node loss in features
            re_zero_updates: Whether to use ReZero for output residuals
            re_zero_attn: Whether to use ReZero for attention/FFN residuals
            warm_start: Whether to use warm start for ReZero
            use_intra_layer_PE: Whether to include intra-layer positional encodings
            use_layer_PE: Whether to include layer depth positional encodings
            use_dist_pe: Append ``dist_pe`` (directed-DAG distance PE). Requires
                ``build_graph(..., use_dist_pe=True)``.
            use_rwse: Append ``rwse`` (Random Walk Structural Encoding). Requires
                ``build_graph(..., use_rwse=True)``.
            rwse_k: RWSE walk length / per-node feature dim. Must match graph build setting.
            num_self_attn_layers: Ignored (absorbed from self_attention config)
            dropout_rate: Ignored (absorbed from self_attention config)
            n_node: Ignored (absorbed from self_attention config)
        """
        self.arity = arity
        self.circuit_hidden_dim = circuit_hidden_dim
        self.attention_dim = attention_dim
        self.logit_dim = 2**arity
        self.num_heads = num_heads
        self.head_dim = attention_dim // num_heads
        self.max_neighbors = max_neighbors
        self.use_attention_mask = use_attention_mask
        self.use_node_loss = use_node_loss
        self.use_intra_layer_PE = use_intra_layer_PE
        self.use_layer_PE = use_layer_PE
        self.use_dist_pe = use_dist_pe
        self.use_rwse = use_rwse
        self.rwse_k = int(rwse_k)

        if mlp_dim is None:
            mlp_dim = attention_dim * mlp_dim_multiplier

        if attention_dim % num_heads != 0:
            raise ValueError(
                f"attention_dim ({attention_dim}) must be divisible by num_heads ({num_heads})"
            )

        # --- Input projection (same as CircuitSelfAttention) ---
        input_feature_dim = self.logit_dim + circuit_hidden_dim
        if use_intra_layer_PE:
            input_feature_dim += circuit_hidden_dim
        if use_layer_PE:
            input_feature_dim += circuit_hidden_dim
        if use_dist_pe:
            input_feature_dim += 2 * (circuit_hidden_dim // 2)
        if use_rwse:
            input_feature_dim += self.rwse_k
        if use_node_loss:
            input_feature_dim += 1

        self.input_norm = nnx.LayerNorm(input_feature_dim, rngs=rngs)
        self.feature_proj = nnx.Linear(input_feature_dim, attention_dim, rngs=rngs)

        # --- Gathered multi-head self-attention + FFN (shared block) ---
        self.attn_block = GatheredAttentionBlock(
            dim=attention_dim,
            mlp_dim=mlp_dim,
            num_heads=num_heads,
            rngs=rngs,
            re_zero=re_zero_attn,
            warm_start=warm_start,
        )

        # --- Output heads (same as CircuitSelfAttention) ---
        self.final_norm = nnx.LayerNorm(attention_dim, rngs=rngs)

        self.logit_proj = nnx.Linear(
            attention_dim,
            self.logit_dim,
            use_bias=True,
            kernel_init=nnx.initializers.lecun_normal(),
            rngs=rngs,
        )
        self.hidden_proj = nnx.Linear(
            attention_dim,
            circuit_hidden_dim,
            use_bias=True,
            kernel_init=nnx.initializers.lecun_normal(),
            rngs=rngs,
        )

        self.logit_rezero = ReZero(rngs=rngs) if re_zero_updates else PassThrough()
        self.hidden_rezero = ReZero(rngs=rngs) if re_zero_updates else PassThrough()

    def __call__(
        self,
        graph: jraph.GraphsTuple,
        neighbor_indices: jp.ndarray | None = None,
        neighbor_mask: jp.ndarray | None = None,
    ) -> jraph.GraphsTuple:
        """
        Apply gathered neighborhood attention to update circuit parameters.

        Args:
            graph: Input graph with node/edge features.
            neighbor_indices: Precomputed [N, max_neighbors] index array.
                If None, built on the fly from graph topology.
            neighbor_mask: Precomputed [N, max_neighbors] boolean mask.
                If None, built on the fly from graph topology.

        Returns:
            Updated graph after one attention step.
        """
        nodes = graph.nodes

        # Build neighbor indices if not precomputed
        if neighbor_indices is None or neighbor_mask is None:
            n_node = nodes["layer"].shape[0]
            neighbor_indices, neighbor_mask = build_neighbor_indices(
                graph.senders,
                graph.receivers,
                n_node,
                self.max_neighbors,
                self.use_attention_mask,
            )

        # --- Extract & project features ---
        features = extract_node_features(
            nodes,
            self.use_node_loss,
            self.use_intra_layer_PE,
            self.use_layer_PE,
            self.use_dist_pe,
            self.use_rwse,
        )
        x = self.feature_proj(self.input_norm(features))  # [N, D]

        # --- Gathered self-attention + FFN ---
        x = self.attn_block(x, neighbor_indices, neighbor_mask)

        # --- Output projections ---
        x = self.final_norm(x)
        logit_updates = self.logit_proj(x)
        hidden_updates = self.hidden_proj(x)

        # Apply knockout mask
        logit_updates, hidden_updates = apply_knockout_mask(logit_updates, hidden_updates, nodes)

        # ReZero residual updates
        updated_logits = nodes["logits"] + self.logit_rezero(logit_updates)
        updated_hidden = nodes["hidden"] + self.hidden_rezero(hidden_updates)

        updated_nodes = {**nodes, "logits": updated_logits, "hidden": updated_hidden}
        return graph._replace(nodes=updated_nodes)
