"""
Shared attention building blocks for circuit optimization models.

This module provides common components used by both CircuitSelfAttention
and PerceiverCircuitAttention models.
"""

import jax.numpy as jp
from flax import nnx


class ReZero(nnx.Module):
    """
    ReZero: A Simple Yet Effective Method for Training Deep Networks.

    Learns a scalar multiplier initialized to 0, providing clean gradient flow
    at initialization (identity function) and gradually learning to incorporate
    the residual branch.

    Reference: https://arxiv.org/abs/2003.04887
    """

    def __init__(self, *, rngs: nnx.Rngs):
        self.scale = nnx.Param(jp.zeros((1,)))

    def __call__(self, x: jp.ndarray) -> jp.ndarray:
        return self.scale.value * x


class PassThrough(nnx.Module):
    """
    PassThrough: A simple module that passes through the input without any changes.
    """

    def __call__(self, x: jp.ndarray) -> jp.ndarray:
        return x


class AttentionBlock(nnx.Module):
    """
    Unified attention block for both self-attention and cross-attention.

    Uses ReZero for stable training:
        x = x + ReZero(Attention(x, kv))
        x = x + ReZero(FFN(x))

    For self-attention: call with key_value=None (uses query as key/value)
    For cross-attention: call with separate key_value

    Reference: https://github.com/sooheon/perceiver-jax
    """

    def __init__(
        self,
        dim: int,
        mlp_dim: int | None = None,
        num_heads: int = 4,
        dropout_rate: float = 0.0,
        *,
        rngs: nnx.Rngs,
        re_zero: bool = True,
    ):
        """
        Initialize an attention block.

        Args:
            dim: Model dimension (for Q, K, V projections and FFN)
            mlp_dim: FFN hidden dimension (default: dim * 2)
            num_heads: Number of attention heads
            dropout_rate: Dropout rate
            rngs: Random number generators
        """
        self.dim = dim
        self.num_heads = num_heads

        if mlp_dim is None:
            mlp_dim = dim * 2

        self.attention = nnx.MultiHeadAttention(
            num_heads=num_heads,
            in_features=dim,
            dropout_rate=dropout_rate,
            deterministic=True,
            rngs=rngs,
            normalize_qk=True,
        )

        # Feed-forward network
        self.ffn = nnx.Sequential(
            nnx.Linear(
                dim,
                mlp_dim,
                rngs=rngs,
                kernel_init=nnx.initializers.kaiming_normal(),
            ),
            nnx.gelu,
            nnx.Linear(
                mlp_dim,
                dim,
                rngs=rngs,
                kernel_init=nnx.initializers.kaiming_normal(),
            ),
        )

        # ReZero scaling for attention and FFN residuals
        self.attn_rezero = ReZero(rngs=rngs) if re_zero else PassThrough()
        self.ffn_rezero = ReZero(rngs=rngs) if re_zero else PassThrough()

    def __call__(
        self,
        query: jp.ndarray,  # [batch, seq_q, dim]
        key_value: jp.ndarray | None = None,  # [batch, seq_kv, dim] or None for self-attn
        mask: jp.ndarray | None = None,  # [batch, 1, seq_q, seq_kv] or None
        output_gate: jp.ndarray | None = None,  # [batch, seq_q, 1] or None
    ) -> jp.ndarray:
        """
        Apply attention block.

        Args:
            query: Query tensor [batch, seq_q, dim]
            key_value: Key/Value tensor [batch, seq_kv, dim].
                      If None, uses query (self-attention mode).
            mask: Optional attention mask [batch, 1, seq_q, seq_kv].
                  True = attend, False = mask out.
            output_gate: Optional gate for attention output [batch, seq_q, 1].
                  Applied BEFORE residual connection. Use this for cross-attention
                  to hard-zero the contribution for nodes that shouldn't attend,
                  since softmax gives uniform attention for fully-masked rows.

        Returns:
            Updated query tensor [batch, seq_q, dim]
        """
        # Self-attention mode: Q=K=V
        if key_value is None:
            key_value = query

        # Attention
        attn_output = self.attention(
            inputs_q=query,
            inputs_k=key_value,
            inputs_v=key_value,
            mask=mask,
            deterministic=True,
            decode=False,
        )

        # Apply output gate if provided (hard zeros for non-allowed nodes)
        # This is crucial for cross-attention: softmax of fully-masked rows
        # gives uniform attention, but we want zero contribution instead.
        if output_gate is not None:
            attn_output = attn_output * output_gate

        # ReZero residual
        x = query + self.attn_rezero(attn_output)

        # FFN + ReZero residual
        x = x + self.ffn_rezero(self.ffn(x))

        return x


def create_attention_mask(
    senders: jp.ndarray,
    receivers: jp.ndarray,
    n_node: int,
    use_mask: bool = True,
) -> jp.ndarray:
    """
    Create an attention mask based on the circuit wiring.

    Args:
        senders: Array of sender node indices
        receivers: Array of receiver node indices
        n_node: Number of nodes in the circuit
        use_mask: If False, returns all-True mask (full attention)

    Returns:
        Boolean attention mask of shape [1, 1, n_node, n_node]
        where True indicates attention is allowed.
    """
    if not use_mask:
        return jp.ones((1, 1, n_node, n_node), dtype=jp.bool_)

    mask = jp.zeros((n_node, n_node), dtype=jp.bool_)

    if len(senders) > 0:
        # Set mask[receiver, sender] = True for all edges
        mask = mask.at[receivers, senders].set(True)

    # Add self-connections (diagonal)
    mask = mask | jp.eye(n_node, dtype=jp.bool_)

    # Add batch and head dimensions [1, 1, n_node, n_node]
    return mask[None, None, ...]


def extract_node_features(
    nodes: dict[str, jp.ndarray],
    use_node_loss: bool = False,
) -> jp.ndarray:
    """
    Extract and concatenate node features for attention.

    Args:
        nodes: Dictionary of node features
        use_node_loss: Whether to include per-node loss in features

    Returns:
        Concatenated features tensor of shape [n_node, feature_dim]
    """
    # Extract relevant features
    logits = nodes["logits"]  # [n_node, logit_dim]
    hidden = nodes["hidden"]  # [n_node, circuit_hidden_dim]
    layer_pe = nodes["layer_pe"]  # [n_node, circuit_hidden_dim]
    intra_layer_pe = nodes["intra_layer_pe"]  # [n_node, circuit_hidden_dim]

    features = jp.concatenate([logits, hidden, layer_pe, intra_layer_pe], axis=-1)

    if use_node_loss:
        features = jp.concatenate([features, nodes["loss"][:, None]], axis=-1)

    return features


def apply_knockout_mask(
    logit_updates: jp.ndarray,
    hidden_updates: jp.ndarray,
    nodes: dict[str, jp.ndarray],
) -> tuple[jp.ndarray, jp.ndarray]:
    """
    Apply gate knockout mask to prevent updates to knocked-out gates.

    Args:
        logit_updates: Proposed logit updates [n_node, logit_dim]
        hidden_updates: Proposed hidden updates [n_node, hidden_dim]
        nodes: Node dictionary containing optional 'gate_knockout_mask'

    Returns:
        Masked logit and hidden updates
    """
    gate_knockout_mask = nodes.get("gate_knockout_mask")
    if gate_knockout_mask is not None:
        # Knocked-out gates (mask == 0.0) should not receive updates
        update_allowed = (gate_knockout_mask == 1.0)[:, None]
        logit_updates = jp.where(update_allowed, logit_updates, 0.0)
        hidden_updates = jp.where(update_allowed, hidden_updates, 0.0)

    return logit_updates, hidden_updates
