"""
Aggregation functions for graph neural networks.

This module provides different methods for aggregating messages in a graph neural network,
including a sparse multi-head attention mechanism that operates in O(E) rather than O(N^2).
"""

import jax.numpy as jp
import jraph
from flax import nnx


def aggregate_sum(messages, indices, num_segments, **kwargs):
    """
    Default aggregation function: sum messages per node.

    Args:
        messages: Features of edges targeting a node [num_edges, feature_dim]
        indices: Index of the receiver node per edge [num_edges]
        num_segments: Total number of nodes in the graph

    Returns:
        Aggregated messages per node [num_nodes, feature_dim]
    """
    return jraph.segment_sum(messages, indices, num_segments)


class SparseMultiHeadAttention(nnx.Module):
    """
    Sparse multi-head attention aggregation for graph message passing.

    Computes attention weights ONLY along existing edges (O(E) instead of O(N^2)),
    using receiver-node features as queries and sender-node features as keys/values.
    This is functionally equivalent to masked self-attention in the Transformer model,
    but with native sparse computation.

    Supports:
    - True multi-head attention with separate head projections
    - QK-normalization for stable attention logits across recurrent steps
    - Numerically stable sparse softmax via segment operations
    """

    def __init__(
        self,
        feature_dim: int,
        num_heads: int = 4,
        *,
        rngs: nnx.Rngs,
        normalize_qk: bool = True,
    ):
        """
        Initialize sparse multi-head attention.

        Args:
            feature_dim: Dimension of input feature vectors (message dim)
            num_heads: Number of attention heads
            rngs: Random number generators
            normalize_qk: Whether to apply QK-normalization via per-head LayerNorm
                          after projection, following the ViT-22B convention
                          (arxiv.org/abs/2302.05442). Consistent with Flax's
                          nnx.MultiHeadAttention normalize_qk implementation.
                          Stabilises attention logits across recurrent steps.
        """
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads
        self.normalize_qk = normalize_qk

        if feature_dim % num_heads != 0:
            raise ValueError(
                f"feature_dim ({feature_dim}) must be divisible by num_heads ({num_heads})"
            )

        # Q projection: applied to RECEIVER node features (what am I looking for?)
        self.query_proj = nnx.Linear(feature_dim, feature_dim, rngs=rngs)
        # K projection: applied to SENDER node features (what do I offer?)
        self.key_proj = nnx.Linear(feature_dim, feature_dim, rngs=rngs)
        # V projection: applied to SENDER node features (what information to send)
        self.value_proj = nnx.Linear(feature_dim, feature_dim, rngs=rngs)
        # Output projection after concatenating heads
        self.output_proj = nnx.Linear(feature_dim, feature_dim, rngs=rngs)

        # QK-normalization: per-head LayerNorm (no bias), matching Flax's
        # nnx.MultiHeadAttention implementation of normalize_qk
        if normalize_qk:
            self.query_ln = nnx.LayerNorm(self.head_dim, use_bias=False, rngs=rngs)
            self.key_ln = nnx.LayerNorm(self.head_dim, use_bias=False, rngs=rngs)

    def __call__(
        self,
        messages: jp.ndarray,
        receivers: jp.ndarray,
        num_segments: int,
        receiver_features: jp.ndarray | None = None,
        sender_features: jp.ndarray | None = None,
    ) -> jp.ndarray:
        """
        Apply sparse multi-head attention aggregation.

        Args:
            messages: Edge message features [num_edges, feature_dim]
                      (used as keys/values if sender_features is None)
            receivers: Receiver node index per edge [num_edges]
            num_segments: Total number of nodes
            receiver_features: Per-NODE features for query computation [num_nodes, feature_dim].
            sender_features: Per-EDGE sender features [num_edges, feature_dim].
                             If None, messages are used as both keys and values.

        Returns:
            Aggregated output per node [num_nodes, feature_dim]
        """
        E = messages.shape[0]
        N = num_segments
        H = self.num_heads
        D = self.head_dim

        # Handle empty graph
        if E == 0:
            return jp.zeros((N, self.feature_dim))

        # --- Keys and Values from sender/edge features ---
        kv_source = sender_features if sender_features is not None else messages
        keys = self.key_proj(kv_source).reshape(E, H, D)  # [E, H, D]
        values = self.value_proj(kv_source).reshape(E, H, D)  # [E, H, D]

        # --- Queries from receiver NODE features ---
        if receiver_features is not None:
            # Proper: queries come from each receiver node's own state
            node_queries = self.query_proj(receiver_features).reshape(N, H, D)  # [N, H, D]
            # Gather per-edge queries: each edge gets its receiver's query
            edge_queries = node_queries[receivers]  # [E, H, D]
        else:
            raise ValueError("receiver_features must be provided for SparseMultiHeadAttention")

        # --- QK-normalization via per-head LayerNorm (ViT-22B convention) ---
        # Consistent with Flax's nnx.MultiHeadAttention normalize_qk implementation.
        if self.normalize_qk:
            edge_queries = self.query_ln(edge_queries)  # [E, H, D] → LN over D per head
            keys = self.key_ln(keys)                     # [E, H, D] → LN over D per head

        # --- Attention logits (per edge, per head) ---
        # [E, H] — dot product between receiver query and sender key
        edge_logits = jp.sum(edge_queries * keys, axis=-1) / jp.sqrt(jp.float32(D))

        # --- Sparse softmax per receiver node, per head ---
        # Numerically stable: subtract max before exp
        # segment_max over edges grouped by receiver, per head
        max_logits = _segment_max_per_head(edge_logits, receivers, N, H)  # [N, H]
        edge_max = max_logits[receivers]  # [E, H]

        exp_logits = jp.exp(edge_logits - edge_max)  # [E, H]

        # Sum of exp per receiver, per head
        sum_exp = _segment_sum_per_head(exp_logits, receivers, N, H)  # [N, H]
        edge_sum_exp = sum_exp[receivers]  # [E, H]

        attention_weights = exp_logits / (edge_sum_exp + 1e-8)  # [E, H]

        # --- Weighted aggregation of values ---
        weighted_values = values * attention_weights[..., None]  # [E, H, D]

        # Flatten heads for segment_sum, then reshape back
        weighted_flat = weighted_values.reshape(E, self.feature_dim)  # [E, H*D]
        aggregated = jraph.segment_sum(weighted_flat, receivers, N)  # [N, H*D]

        # Output projection
        output = self.output_proj(aggregated)  # [N, feature_dim]

        return output


def _segment_max_per_head(
    data: jp.ndarray, segment_ids: jp.ndarray, num_segments: int, num_heads: int
) -> jp.ndarray:
    """Compute segment max independently per head. data: [E, H] -> [N, H]."""
    # Use a very negative initial value for max
    init = jp.full((num_segments, num_heads), -1e9)
    return init.at[segment_ids].max(data)


def _segment_sum_per_head(
    data: jp.ndarray, segment_ids: jp.ndarray, num_segments: int, num_heads: int
) -> jp.ndarray:
    """Compute segment sum independently per head. data: [E, H] -> [N, H]."""
    init = jp.zeros((num_segments, num_heads))
    return init.at[segment_ids].add(data)
