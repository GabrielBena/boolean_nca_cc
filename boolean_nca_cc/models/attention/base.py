"""
Shared attention building blocks for circuit optimization models.

This module provides common components used by both CircuitSelfAttention,
PerceiverCircuitAttention, and CircuitGatheredAttention models.
"""

import warnings

import jax
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

    def __init__(self, *, rngs: nnx.Rngs, warm_start: bool = False):
        initial_value = jp.full((1,), 0.0 if warm_start else 1e-3)
        self.scale = nnx.Param(initial_value)

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

    Pre-LN pattern with ReZero residuals:
        x = x + ReZero(Attention(LN_q(x), LN_kv(kv)))
        x = x + ReZero(FFN(LN(x)))

    Additionally uses QK-normalization (normalize_qk) inside MultiHeadAttention
    to stabilize attention logits, complementary to the pre-LN input norms.

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
        warm_start: bool = True,
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

        # Pre-attention LayerNorms (Pre-LN pattern)
        # These normalize the input representation before Q/K/V projections.
        # Separate norms for Q and KV to support both self- and cross-attention.
        # Complementary to normalize_qk which normalizes Q/K *after* projection.
        self.layer_norm_q = nnx.LayerNorm(dim, rngs=rngs)
        self.layer_norm_kv = nnx.LayerNorm(dim, rngs=rngs)

        self.attention = nnx.MultiHeadAttention(
            num_heads=num_heads,
            in_features=dim,
            dropout_rate=dropout_rate,
            deterministic=True,
            rngs=rngs,
            normalize_qk=True,
        )

        # Feed-forward network (Pre-LN: LayerNorm before FFN)
        self.ffn = nnx.Sequential(
            nnx.LayerNorm(dim, rngs=rngs),
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
        self.attn_rezero = ReZero(rngs=rngs, warm_start=warm_start) if re_zero else PassThrough()
        self.ffn_rezero = ReZero(rngs=rngs, warm_start=warm_start) if re_zero else PassThrough()

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

        # Pre-LN: normalize inputs before attention projections
        # (original query is preserved for residual connection below)
        query_normed = self.layer_norm_q(query)
        kv_normed = self.layer_norm_kv(key_value)

        # Attention (normalize_qk additionally normalizes Q/K after projection)
        attn_output = self.attention(
            inputs_q=query_normed,
            inputs_k=kv_normed,
            inputs_v=kv_normed,
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


class GatheredAttentionBlock(nnx.Module):
    """
    Gathered neighborhood self-attention block.

    Mirrors AttentionBlock's structure (Pre-LN, QK-norm, ReZero residuals, FFN)
    but replaces the dense O(N^2) attention with gathered attention over each
    node's local graph neighborhood, computed as dense batched einsums on a
    padded [N, max_neighbors, D] tensor. Self-attention only — there is no
    graph topology between gates and external tokens, so cross-attention has
    nothing to gather over.

    Operates on un-batched [N, dim] tensors (the gather step is over the node
    axis, so a leading batch dim would force a vmap; callers using batched
    layouts should squeeze/unsqueeze around this block).
    """

    def __init__(
        self,
        dim: int,
        mlp_dim: int | None = None,
        num_heads: int = 4,
        *,
        rngs: nnx.Rngs,
        re_zero: bool = True,
        warm_start: bool = True,
    ):
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads})")

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        if mlp_dim is None:
            mlp_dim = dim * 2

        # Pre-LN: single norm since Q/K/V all come from the same source (self-attn)
        self.attn_norm = nnx.LayerNorm(dim, rngs=rngs)

        # Q/K/V/O projections (same structure as nnx.MultiHeadAttention)
        self.query_proj = nnx.Linear(dim, dim, rngs=rngs)
        self.key_proj = nnx.Linear(dim, dim, rngs=rngs)
        self.value_proj = nnx.Linear(dim, dim, rngs=rngs)
        self.output_proj = nnx.Linear(dim, dim, rngs=rngs)

        # QK-normalization: per-head LayerNorm (no bias), ViT-22B / Flax convention
        self.query_ln = nnx.LayerNorm(self.head_dim, use_bias=False, rngs=rngs)
        self.key_ln = nnx.LayerNorm(self.head_dim, use_bias=False, rngs=rngs)

        # ReZero for attention residual
        self.attn_rezero = ReZero(rngs=rngs, warm_start=warm_start) if re_zero else PassThrough()

        # FFN (Pre-LN inside)
        self.ffn = nnx.Sequential(
            nnx.LayerNorm(dim, rngs=rngs),
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
        self.ffn_rezero = ReZero(rngs=rngs, warm_start=warm_start) if re_zero else PassThrough()

    def _gathered_attention(
        self,
        x: jp.ndarray,
        neighbor_indices: jp.ndarray,
        neighbor_mask: jp.ndarray,
    ) -> jp.ndarray:
        """Multi-head self-attention restricted to each node's gathered neighbors."""
        N = x.shape[0]
        H = self.num_heads
        D = self.head_dim

        x_normed = self.attn_norm(x)

        Q = self.query_proj(x_normed).reshape(N, H, D)
        K = self.key_proj(x_normed).reshape(N, H, D)
        V = self.value_proj(x_normed).reshape(N, H, D)

        Q = self.query_ln(Q)
        K = self.key_ln(K)

        # Gather neighbor keys/values: [N, max_neighbors, H, D]
        K_gathered = K[neighbor_indices]
        V_gathered = V[neighbor_indices]

        # Q_i · K_j over the head_dim → [N, H, max_neighbors]
        logits = jp.einsum("nhd,nmhd->nhm", Q, K_gathered) / jp.sqrt(jp.float32(D))

        # Mask padding slots before softmax
        mask = neighbor_mask[:, None, :]
        logits = jp.where(mask, logits, jp.finfo(jp.float32).min)
        weights = jax.nn.softmax(logits, axis=-1)

        # Weighted aggregation over neighbors
        attn_output = jp.einsum("nhm,nmhd->nhd", weights, V_gathered)
        attn_output = attn_output.reshape(N, self.dim)

        return self.output_proj(attn_output)

    def __call__(
        self,
        x: jp.ndarray,  # [N, dim]
        neighbor_indices: jp.ndarray,  # [N, max_neighbors] int
        neighbor_mask: jp.ndarray,  # [N, max_neighbors] bool
    ) -> jp.ndarray:
        x = x + self.attn_rezero(self._gathered_attention(x, neighbor_indices, neighbor_mask))
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

    # print(f"Final mask shape: {mask.shape}")

    # Add batch and head dimensions [1, 1, n_node, n_node]
    return mask[None, None, ...]


def build_neighbor_indices(
    senders: jp.ndarray,
    receivers: jp.ndarray,
    n_node: int,
    max_neighbors: int,
    use_mask: bool = True,
) -> tuple[jp.ndarray, jp.ndarray]:
    """
    Build padded neighbor-index tensor from edge lists (pure JAX, JIT-safe).

    For each node i, collects all nodes j that i can attend to
    (i.e. j is a sender on an edge where i is the receiver), plus self-loops.
    Pads/truncates to ``max_neighbors`` so every row has the same fixed width.

    Uses only JAX operations so it can be called inside ``jax.jit`` / ``nnx.vmap``.
    ``n_node`` and ``max_neighbors`` must be Python ints (compile-time constants).

    Args:
        senders: Sender node indices from the graph edge list.
        receivers: Receiver node indices from the graph edge list.
        n_node: Total number of nodes in the circuit (Python int).
        max_neighbors: Fixed width of the neighbor tensor (Python int).
            Must be >= the maximum node degree in the graph.
        use_mask: If False, returns full (all-to-all) neighbor lists.

    Returns:
        neighbor_indices: int32 array [N, max_neighbors] — padded column indices.
            Padding slots point to node 0 (harmless; masked out by neighbor_mask).
        neighbor_mask: bool array [N, max_neighbors] — True for real neighbors.
    """
    if not use_mask:
        # Fully connected: everyone attends to everyone
        # Truncate to max_neighbors (must be >= n_node for full attention)
        indices = jp.broadcast_to(jp.arange(n_node, dtype=jp.int32), (n_node, n_node))
        indices = indices[:, :max_neighbors]
        mask = jp.arange(max_neighbors) < n_node
        mask = jp.broadcast_to(mask, (n_node, max_neighbors))
        return indices, mask

    # Build adjacency matrix [N, N] — same ops as create_attention_mask
    adj = jp.zeros((n_node, n_node), dtype=jp.bool_)
    adj = adj.at[receivers, senders].set(True)
    adj = adj | jp.eye(n_node, dtype=jp.bool_)

    # Sort each row so True (neighbor) entries come first.
    # ~adj flips True→False; argsort ascending puts False (=real neighbors) first.
    sorted_indices = jp.argsort(~adj, axis=-1, stable=True)  # [N, N]

    # Truncate to fixed width
    neighbor_indices = sorted_indices[:, :max_neighbors]  # [N, max_neighbors]

    # Build mask: position k is a real neighbor iff k < degree(i)
    degrees = adj.sum(axis=-1)  # [N]
    positions = jp.arange(max_neighbors)[None, :]  # [1, max_neighbors]
    neighbor_mask = positions < degrees[:, None]  # [N, max_neighbors]

    return neighbor_indices, neighbor_mask


def validate_gathered_topology(
    senders: jp.ndarray,
    receivers: jp.ndarray,
    n_node: int,
    max_neighbors: int,
    use_mask: bool = True,
    strict: bool = True,
    layer_indices: jp.ndarray | None = None,
    strict_input_nodes: bool = False,
) -> int:
    """
    Verify that ``max_neighbors`` is wide enough for the live graph topology.

    The gathered attention path silently truncates each node's neighbor list to
    ``max_neighbors`` (see :func:`build_neighbor_indices`). If the actual max
    in-degree exceeds this width, real neighbors get dropped and gathered
    attention produces a different result than dense self-attention.

    Two failure modes are distinguished:

    1. **Gate-node truncation (always fatal):** any gate node (layer > 0) whose
       degree exceeds ``max_neighbors`` gets its attention output corrupted, and
       that corruption flows directly into ``logit_updates`` / ``hidden_updates``
       and therefore into the circuit's predictions. There's no benign reading
       of this case.

    2. **Input-node truncation (informational by default):** layer-0 nodes
       represent the circuit's *literal input bits*. ``run_circuit`` reads
       those bits straight from the data tensor, never from
       ``nodes["logits"]``, so truncated attention on input nodes does NOT
       enter the loss directly. It only affects subsequent self-attn steps
       where gates attend to input-node K/V — a bounded, second-order error.
       Many configs intentionally use a small ``max_neighbors`` and live with
       this. Pass ``strict_input_nodes=True`` to escalate it to an error.

    Args:
        senders, receivers: Edge lists from the live graph.
        n_node: Total nodes in the circuit.
        max_neighbors: The model's configured neighbor-tensor width.
        use_mask: Mirrors :func:`build_neighbor_indices` — when False the model
            attends to every node, so the effective max degree is ``n_node``.
        strict: If True (default) raise ``ValueError`` on insufficient width
            for GATE nodes; else emit a warning for the same case.
        layer_indices: Optional [N] int array of node-layer membership. When
            provided, the validator distinguishes gate vs input-node truncation.
            Without it, the strict total-degree check applies (legacy behavior).
        strict_input_nodes: When True, input-node truncation is also fatal.
            Default False — input-node truncation is reported as a warning
            (or info-level note) but not raised, because the loss is
            insensitive to input-node attention error (see above).

    Returns:
        Actual max in-degree over all nodes (including the self-loop).
    """
    # Bail out inside JIT/vmap — this is a pre-flight check, not training logic.
    # Anything that requires concretising tracer values (like int(...)) fails under
    # tracing, and validating during the scan would be wasteful anyway: the
    # topology is identical across scan iterations, so eager call sites
    # (``_prepare_model_fn`` at training setup, ``train_model`` startup) already
    # catch any mis-config exactly once before the loop starts.
    if isinstance(senders, jax.core.Tracer):
        return -1

    if not use_mask:
        # Every node attends to every node — effective degree is n_node for all.
        max_deg = int(n_node)
        gate_max_deg = max_deg
        input_max_deg = max_deg
    else:
        adj = jp.zeros((n_node, n_node), dtype=jp.bool_)
        adj = adj.at[receivers, senders].set(True)
        adj = adj | jp.eye(n_node, dtype=jp.bool_)
        degrees = adj.sum(axis=-1)
        max_deg = int(degrees.max())
        if layer_indices is not None and not isinstance(layer_indices, jax.core.Tracer):
            is_input = jp.asarray(layer_indices) == 0
            gate_max_deg = (
                int(degrees[~is_input].max()) if bool(jp.any(~is_input)) else 0
            )
            input_max_deg = (
                int(degrees[is_input].max()) if bool(jp.any(is_input)) else 0
            )
        else:
            # No layer info: conservative — treat everything as a gate.
            gate_max_deg = max_deg
            input_max_deg = 0

    gate_truncated = max_neighbors < gate_max_deg
    input_truncated = max_neighbors < input_max_deg and input_max_deg > gate_max_deg

    if gate_truncated:
        msg = (
            "\n"
            + "=" * 72
            + "\n"
            + "  GATHERED ATTENTION TOPOLOGY MISMATCH (GATE NODES)\n"
            + f"    max_neighbors = {max_neighbors}\n"
            + f"    max gate-node in-degree = {gate_max_deg}\n"
            + f"    -> {gate_max_deg - max_neighbors} real neighbor(s) per affected gate "
            + "will be\n       SILENTLY TRUNCATED, corrupting attention outputs that\n"
            + "       flow into logit/hidden updates and the circuit loss.\n"
            + f"    Fix: set max_neighbors >= {gate_max_deg} "
            + "(or switch self_attn_kind to 'dense').\n"
            + "=" * 72
        )
        if strict:
            raise ValueError(msg)
        warnings.warn(msg, stacklevel=2)

    if input_truncated:
        msg = (
            "\n"
            + "-" * 72
            + "\n"
            + "  Gathered attention: INPUT-NODE neighborhood truncation detected\n"
            + f"    max_neighbors = {max_neighbors}\n"
            + f"    max input-node (layer=0) in-degree = {input_max_deg}\n"
            + f"    -> {input_max_deg - max_neighbors} neighbor(s) per input node truncated.\n"
            + "    This affects ONLY input nodes' own self-attention output. The\n"
            + "    circuit reads raw input bits (not input-node logits), so this\n"
            + "    does not enter the loss directly — but it does leak a small,\n"
            + "    bounded error into subsequent self-attn steps. Set\n"
            + f"    max_neighbors >= {input_max_deg} to remove it entirely.\n"
            + "-" * 72
        )
        if strict_input_nodes:
            raise ValueError(msg)
        warnings.warn(msg, stacklevel=2)

    if (
        not gate_truncated
        and not input_truncated
        and max_neighbors > 2 * max_deg
        and (max_neighbors - max_deg) > 4
    ):
        warnings.warn(
            f"max_neighbors={max_neighbors} is much larger than the actual max in-degree "
            f"({max_deg}). Each gathered self-attn layer wastes "
            f"O(N * {max_neighbors - max_deg}) compute on padding slots; "
            f"consider reducing max_neighbors to ~{max_deg}.",
            stacklevel=2,
        )

    return max_deg


def extract_node_features(
    nodes: dict[str, jp.ndarray],
    use_node_loss: bool = False,
    use_intra_layer_PE: bool = False,
    use_layer_PE: bool = False,
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

    features = jp.concatenate([logits, hidden], axis=-1)
    if use_intra_layer_PE:
        features = jp.concatenate([features, intra_layer_pe], axis=-1)
    if use_layer_PE:
        features = jp.concatenate([features, layer_pe], axis=-1)

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
