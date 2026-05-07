"""
NumPy reference implementation of the SODC inference loop.

This module is the *single source of numerical truth* for the web demo:

- It is validated against the JAX/Flax policy in
  ``boolean_nca_cc.models.attention.CircuitSelfAttention`` once we
  pin a checkpoint (M2). Until then, ``test_oracle.py`` exercises
  the math with random weights to catch shape / NaN regressions.
- It is the bit-for-bit reference the TypeScript port (``src/tmt.ts``,
  ``src/circuit.ts``) is checked against in M3.

Keeping the oracle separate from JAX lets us:

- Iterate on the TS port without a working JAX install in the loop.
- Catch quantisation drift in M2 cleanly: re-run the oracle on the
  dequantised weights and diff against the float reference.

The full inference contract — what tensors flow, in what order, with
what shapes — is documented in ``web_demo/PROGRESS.md`` under
"Inference contract". This module implements that contract verbatim.

The reference Python path we are mirroring is

    boolean_nca_cc.training.evaluation.evaluate_model_stepwise_generator

with the model being a ``CircuitSelfAttention`` (single
``AttentionBlock`` with ``normalize_qk=True`` and Pre-LN + ReZero
residuals).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace

import numpy as np
from scipy.special import erf

# ---------------------------------------------------------------------------
# Math primitives
# ---------------------------------------------------------------------------


def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """LayerNorm over the last axis. Matches ``flax.nnx.LayerNorm`` defaults."""
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mean) * (1.0 / np.sqrt(var + eps)) * gamma + beta


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax."""
    shifted = x - x.max(axis=axis, keepdims=True)
    exped = np.exp(shifted)
    return exped / exped.sum(axis=axis, keepdims=True)


def gelu_exact(x: np.ndarray) -> np.ndarray:
    """Exact erf-based GELU, matching ``jax.nn.gelu(approximate=False)``."""
    return 0.5 * x * (1.0 + erf(x / np.sqrt(2.0)))


def gelu_approx(x: np.ndarray) -> np.ndarray:
    """tanh-based GELU approximation (``jax.nn.gelu(approximate=True)``)."""
    c = np.sqrt(2.0 / np.pi)
    return 0.5 * x * (1.0 + np.tanh(c * (x + 0.044715 * x**3)))


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    out = np.empty_like(x, dtype=np.float32)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    e = np.exp(x[~pos])
    out[~pos] = e / (1.0 + e)
    return out


def linear(x: np.ndarray, w: np.ndarray, b: np.ndarray | None = None) -> np.ndarray:
    """Affine map: x @ w + b (b optional)."""
    out = x @ w
    if b is not None:
        out = out + b
    return out


def get_positional_encoding(positions: np.ndarray, dim: int, max_val: float = 10000.0) -> np.ndarray:
    """Sinusoidal positional encoding. Mirrors ``utils.positional_encoding``."""
    if dim % 2 != 0:
        raise ValueError(f"dim must be even, got {dim}")
    div_term = np.exp(np.arange(0, dim, 2, dtype=np.float32) * -(np.log(max_val) / dim))
    pe = np.zeros((positions.shape[0], dim), dtype=np.float32)
    p = positions.astype(np.float32)[:, None]
    pe[:, 0::2] = np.sin(p * div_term)
    pe[:, 1::2] = np.cos(p * div_term)
    return pe


# ---------------------------------------------------------------------------
# Multi-head self-attention with normalize_qk
# ---------------------------------------------------------------------------


@dataclass
class MHAWeights:
    """Multi-head attention weights, shaped to match ``flax.nnx.MultiHeadAttention``.

    Q/K/V kernels are ``[in_features, num_heads, head_dim]`` (DenseGeneral).
    Output kernel is ``[num_heads, head_dim, out_features]``.
    With ``normalize_qk=True``, Q and K are LayerNormed per head over
    ``head_dim`` *after* projection, before the dot product.
    """

    Wq: np.ndarray  # [D, H, d_h]
    bq: np.ndarray  # [H, d_h]
    Wk: np.ndarray
    bk: np.ndarray
    Wv: np.ndarray
    bv: np.ndarray
    Wo: np.ndarray  # [H, d_h, D]
    bo: np.ndarray  # [D]
    q_ln_gamma: np.ndarray | None = None  # [d_h]
    q_ln_beta: np.ndarray | None = None
    k_ln_gamma: np.ndarray | None = None
    k_ln_beta: np.ndarray | None = None


def multi_head_attention(
    q: np.ndarray,  # [N, D]
    kv: np.ndarray,  # [N, D]
    mask: np.ndarray | None,  # [N, N] bool, or None
    w: MHAWeights,
) -> np.ndarray:
    Q = np.einsum("nd,dhe->nhe", q, w.Wq) + w.bq
    K = np.einsum("nd,dhe->nhe", kv, w.Wk) + w.bk
    V = np.einsum("nd,dhe->nhe", kv, w.Wv) + w.bv

    if w.q_ln_gamma is not None and w.q_ln_beta is not None:
        Q = layer_norm(Q, w.q_ln_gamma, w.q_ln_beta)
    if w.k_ln_gamma is not None and w.k_ln_beta is not None:
        K = layer_norm(K, w.k_ln_gamma, w.k_ln_beta)

    head_dim = Q.shape[-1]
    scores = np.einsum("nhe,mhe->hnm", Q, K) / np.sqrt(head_dim)
    if mask is not None:
        scores = np.where(mask[None, :, :], scores, np.float32(-1e30))
    attn = softmax(scores, axis=-1)
    out = np.einsum("hnm,mhe->nhe", attn, V)
    return np.einsum("nhe,hed->nd", out, w.Wo) + w.bo


# ---------------------------------------------------------------------------
# Pre-LN + ReZero attention block (single layer of the TMT)
# ---------------------------------------------------------------------------


@dataclass
class AttentionBlockWeights:
    ln_q: tuple[np.ndarray, np.ndarray]  # (gamma, beta)
    ln_kv: tuple[np.ndarray, np.ndarray]
    mha: MHAWeights
    ffn_ln: tuple[np.ndarray, np.ndarray]
    ffn_W1: np.ndarray  # [D, mlp_dim]
    ffn_b1: np.ndarray  # [mlp_dim]
    ffn_W2: np.ndarray  # [mlp_dim, D]
    ffn_b2: np.ndarray  # [D]
    attn_rezero: float
    ffn_rezero: float
    use_gelu_approx: bool = False  # match flax default; flip via parity test


def attention_block(x: np.ndarray, mask: np.ndarray | None, w: AttentionBlockWeights) -> np.ndarray:
    q_normed = layer_norm(x, *w.ln_q)
    kv_normed = layer_norm(x, *w.ln_kv)
    attn_out = multi_head_attention(q_normed, kv_normed, mask, w.mha)
    x = x + w.attn_rezero * attn_out

    h = layer_norm(x, *w.ffn_ln)
    h = linear(h, w.ffn_W1, w.ffn_b1)
    h = gelu_approx(h) if w.use_gelu_approx else gelu_exact(h)
    h = linear(h, w.ffn_W2, w.ffn_b2)
    return x + w.ffn_rezero * h


# ---------------------------------------------------------------------------
# Gathered-neighborhood attention (mirrors CircuitGatheredAttention)
# ---------------------------------------------------------------------------


@dataclass
class GatheredAttentionWeights:
    """Weights for the ``CircuitGatheredAttention`` path.

    Differs from ``AttentionBlockWeights``:
      - Single ``attn_norm`` (was separate ``ln_q`` / ``ln_kv``).
      - Plain ``Linear[D,D]`` Q/K/V/O (was ``DenseGeneral`` with multi-head shape).
      - QK-norm has no bias (γ only, was γ + β).
      - Attention computed over a gathered ``[N, max_neighbors]``
        tensor instead of dense ``[N, N]`` with a binary mask.

    The FFN, ReZero scalars, and surrounding plumbing are unchanged.
    """

    attn_norm: tuple[np.ndarray, np.ndarray]
    Wq: np.ndarray  # [D, D]
    bq: np.ndarray  # [D]
    Wk: np.ndarray
    bk: np.ndarray
    Wv: np.ndarray
    bv: np.ndarray
    Wo: np.ndarray  # [D, D]
    bo: np.ndarray
    q_ln_gamma: np.ndarray  # [head_dim]; β is absent by construction
    k_ln_gamma: np.ndarray
    ffn_ln: tuple[np.ndarray, np.ndarray]
    ffn_W1: np.ndarray
    ffn_b1: np.ndarray
    ffn_W2: np.ndarray
    ffn_b2: np.ndarray
    attn_rezero: float
    ffn_rezero: float
    num_heads: int
    use_gelu_approx: bool = False


def build_neighbor_indices(
    attention_mask: np.ndarray, max_neighbors: int
) -> tuple[np.ndarray, np.ndarray]:
    """Convert a dense ``[N, N]`` boolean mask into padded neighbor lists.

    Mirrors ``boolean_nca_cc.models.attention.base.build_neighbor_indices``:
    each row's True entries (real neighbors + self-loop) come first, padding
    slots point at index 0 and are masked out via ``neighbor_mask``.
    """
    n_node = attention_mask.shape[0]
    if max_neighbors > n_node:
        # Match the JAX path: degenerate to fully-connected up to N.
        max_neighbors = n_node
    sorted_indices = np.argsort(~attention_mask, axis=-1, kind="stable").astype(np.int32)
    neighbor_indices = sorted_indices[:, :max_neighbors]
    degrees = attention_mask.sum(axis=-1)
    positions = np.arange(max_neighbors)[None, :]
    neighbor_mask = positions < degrees[:, None]
    return neighbor_indices, neighbor_mask


def gathered_attention_block(
    x: np.ndarray,  # [N, D]
    neighbor_indices: np.ndarray,  # [N, K]
    neighbor_mask: np.ndarray,  # [N, K] bool
    w: GatheredAttentionWeights,
) -> np.ndarray:
    """One gathered-attention block: Pre-LN attention + FFN, both ReZero residuals."""
    N, D = x.shape
    H = w.num_heads
    head_dim = D // H

    x_n = layer_norm(x, *w.attn_norm)

    Q = linear(x_n, w.Wq, w.bq).reshape(N, H, head_dim)
    K = linear(x_n, w.Wk, w.bk).reshape(N, H, head_dim)
    V = linear(x_n, w.Wv, w.bv).reshape(N, H, head_dim)

    # No-bias QK-LayerNorm: β = 0.
    zero_beta = np.zeros_like(w.q_ln_gamma)
    Q = layer_norm(Q, w.q_ln_gamma, zero_beta)
    K = layer_norm(K, w.k_ln_gamma, zero_beta)

    K_gathered = K[neighbor_indices]  # [N, K, H, head_dim]
    V_gathered = V[neighbor_indices]

    scores = np.einsum("nhd,nmhd->nhm", Q, K_gathered) / np.sqrt(head_dim)
    scores = np.where(neighbor_mask[:, None, :], scores, np.float32(-1e30))
    weights = softmax(scores, axis=-1)
    attn_out = np.einsum("nhm,nmhd->nhd", weights, V_gathered).reshape(N, D)
    attn_out = linear(attn_out, w.Wo, w.bo)

    x = x + w.attn_rezero * attn_out

    h = layer_norm(x, *w.ffn_ln)
    h = linear(h, w.ffn_W1, w.ffn_b1)
    h = gelu_approx(h) if w.use_gelu_approx else gelu_exact(h)
    h = linear(h, w.ffn_W2, w.ffn_b2)
    return x + w.ffn_rezero * h


# ---------------------------------------------------------------------------
# TMT (Topology-Masked Transformer) — the policy
# ---------------------------------------------------------------------------


@dataclass
class TMTConfig:
    arity: int
    circuit_hidden_dim: int
    attention_dim: int
    num_heads: int
    num_attn_layers: int = 1  # only meaningful for self_attention kind
    use_layer_PE: bool = True
    use_intra_layer_PE: bool = False
    use_node_loss: bool = True
    # ``self_attention`` mirrors ``CircuitSelfAttention`` (dense [N, N]
    # attention with separate LN_q / LN_kv and ``DenseGeneral`` QKV).
    # ``gathered_attention`` mirrors ``CircuitGatheredAttention`` (sparse
    # gathered ``[N, max_neighbors]`` attention with single ``attn_norm``,
    # plain ``Linear`` QKV, and unbiased QK-norm). The deployed
    # checkpoints use the gathered variant.
    model_kind: str = "gathered_attention"
    max_neighbors: int = 16  # only used by gathered_attention

    @property
    def logit_dim(self) -> int:
        return 1 << self.arity

    @property
    def feature_dim(self) -> int:
        d = self.logit_dim + self.circuit_hidden_dim
        if self.use_intra_layer_PE:
            d += self.circuit_hidden_dim
        if self.use_layer_PE:
            d += self.circuit_hidden_dim
        if self.use_node_loss:
            d += 1
        return d


@dataclass
class TMTWeights:
    config: TMTConfig
    input_norm: tuple[np.ndarray, np.ndarray]
    feature_proj_W: np.ndarray  # [feature_dim, attn_dim]
    feature_proj_b: np.ndarray  # [attn_dim]
    # ``blocks`` holds either:
    #   - ``list[AttentionBlockWeights]`` (one per layer) for ``self_attention``;
    #   - a single ``GatheredAttentionWeights`` for ``gathered_attention``.
    blocks: list[AttentionBlockWeights] | GatheredAttentionWeights
    final_norm: tuple[np.ndarray, np.ndarray]
    logit_proj_W: np.ndarray  # [attn_dim, logit_dim]
    logit_proj_b: np.ndarray
    hidden_proj_W: np.ndarray  # [attn_dim, hidden_dim]
    hidden_proj_b: np.ndarray
    logit_rezero: float
    hidden_rezero: float


# ---------------------------------------------------------------------------
# Topology + node-state containers
# ---------------------------------------------------------------------------


@dataclass
class Topology:
    """Static circuit structure: layer sizes, wires, derived edges + PEs.

    Recomputed only when the wiring changes (e.g. the user clicks
    "shuffle wires"). The TMT weights and the dynamic circuit state
    are independent of this object.
    """

    layer_sizes: list[tuple[int, int]]  # (gate_n, group_size); layer 0 = input
    wires: list[np.ndarray]  # one per *gate* layer, shape [arity, group_n]
    arity: int
    n_nodes: int
    layer_start: list[int]  # node-index offset of each layer
    output_start: int
    output_end: int
    attention_mask: np.ndarray  # [N, N] bool
    layer_pe: np.ndarray  # [N, h]
    intra_layer_pe: np.ndarray  # [N, h]
    # Optional gathered-attention caches; populated by ``build_topology``
    # when a ``max_neighbors`` value is supplied (or refreshed lazily on
    # the first ``tmt_step`` call for a gathered-kind model).
    neighbor_indices: np.ndarray | None = None
    neighbor_mask: np.ndarray | None = None


@dataclass
class CircuitState:
    """Dynamic per-node state mutated by ``tmt_step`` and ``compute_residuals``.

    ``logits`` and ``hidden`` are mutated by the policy. ``loss`` is
    refreshed by ``compute_residuals`` after each tick. ``gate_mask``
    flips when the user damages a gate.
    """

    logits: np.ndarray  # [N, 2^arity]
    hidden: np.ndarray  # [N, h]
    loss: np.ndarray  # [N]
    gate_mask: np.ndarray  # [N]


# ---------------------------------------------------------------------------
# Topology / state construction
# ---------------------------------------------------------------------------


def gen_wires(rng: np.random.Generator, in_n: int, out_n: int, arity: int, group_size: int) -> np.ndarray:
    """Random wiring matching ``circuits.model.gen_wires``.

    Returns an ``[arity, group_n]`` int array of indices into the
    previous layer's outputs. A single permutation is taken over the
    larger of (input, edge) sizes, then truncated and reshaped.
    """
    edge_n = out_n * arity // group_size
    n = max(in_n, edge_n)
    perm = rng.permutation(n)[:edge_n]
    return (perm.reshape(arity, -1) % in_n).astype(np.int32)


def build_topology(
    layer_sizes: list[tuple[int, int]],
    wires: list[np.ndarray],
    arity: int,
    hidden_dim: int,
    *,
    bidirectional_edges: bool = True,
    pe_max_val: float = 10000.0,
    max_neighbors: int | None = None,
) -> Topology:
    """Build attention mask + positional encodings from layer sizes and wires.

    Mirrors ``utils.graph_builder.build_graph`` for the parts that
    matter to the TMT (mask + PEs). Mutable per-node state lives in
    ``CircuitState`` and is built separately.
    """
    n_nodes = sum(g for g, _ in layer_sizes)
    layer_start: list[int] = [0]
    for g, _ in layer_sizes[:-1]:
        layer_start.append(layer_start[-1] + g)

    # Build the forward edge list, then mirror for bidirectional.
    forward_senders: list[np.ndarray] = []
    forward_receivers: list[np.ndarray] = []
    for li_gate, layer_w in enumerate(wires):
        layer_idx = li_gate + 1  # graph layer index (input is 0)
        gate_n, group_size = layer_sizes[layer_idx]
        group_n = gate_n // group_size

        recv_global = np.arange(layer_start[layer_idx], layer_start[layer_idx] + gate_n)
        recv = np.repeat(recv_global, arity)
        # ``layer_w`` shape is [arity, group_n]; senders per gate group
        send_global = layer_start[layer_idx - 1] + layer_w.T  # [group_n, arity]
        # Tile across group_size: every gate in a group shares the wires.
        send_tiled = np.tile(send_global, (1, group_size)).reshape(-1)
        forward_senders.append(send_tiled)
        forward_receivers.append(recv)

    if forward_senders:
        senders = np.concatenate(forward_senders)
        receivers = np.concatenate(forward_receivers)
    else:
        senders = np.zeros(0, dtype=np.int32)
        receivers = np.zeros(0, dtype=np.int32)

    # Attention mask: mask[r, s] for every edge, plus diagonal.
    mask = np.zeros((n_nodes, n_nodes), dtype=bool)
    if senders.size:
        mask[receivers, senders] = True
        if bidirectional_edges:
            mask[senders, receivers] = True
    np.fill_diagonal(mask, True)

    # Positional encodings.
    total_gate_layers = len(layer_sizes) - 1
    layer_pe_chunks: list[np.ndarray] = []
    intra_pe_chunks: list[np.ndarray] = []
    for li, (gate_n, _) in enumerate(layer_sizes):
        normalized_depth = li / total_gate_layers if total_gate_layers > 0 else 0.0
        scaled = np.full(gate_n, normalized_depth * pe_max_val, dtype=np.float32)
        layer_pe_chunks.append(get_positional_encoding(scaled, hidden_dim, pe_max_val))
        intra_pe_chunks.append(
            get_positional_encoding(np.arange(gate_n, dtype=np.int32), hidden_dim, pe_max_val)
        )
    layer_pe = np.concatenate(layer_pe_chunks, axis=0).astype(np.float32)
    intra_pe = np.concatenate(intra_pe_chunks, axis=0).astype(np.float32)

    output_start = sum(g for g, _ in layer_sizes[:-1])
    output_end = output_start + layer_sizes[-1][0]

    neighbor_indices = None
    neighbor_mask = None
    if max_neighbors is not None:
        neighbor_indices, neighbor_mask = build_neighbor_indices(mask, max_neighbors)

    return Topology(
        layer_sizes=layer_sizes,
        wires=wires,
        arity=arity,
        n_nodes=n_nodes,
        layer_start=layer_start,
        output_start=output_start,
        output_end=output_end,
        attention_mask=mask,
        layer_pe=layer_pe,
        intra_layer_pe=intra_pe,
        neighbor_indices=neighbor_indices,
        neighbor_mask=neighbor_mask,
    )


def make_nops_with_noise(
    rng: np.random.Generator, gate_n: int, arity: int, group_size: int, noise_scale: float = 0.0
) -> np.ndarray:
    """Soft-wire identity init + Gaussian noise. Mirrors ``circuits.model``."""
    one_hot = (np.arange(1 << arity) >> np.arange(arity)[:, None]) & 1  # [arity, 2^arity]
    luts = one_hot[np.arange(gate_n) % arity]  # [gate_n, 2^arity]
    base = (2.0 * luts - 1.0) * 3.0  # nop_scale=3.0
    noise = rng.normal(size=base.shape).astype(np.float32) * noise_scale
    return (base + noise).reshape(gate_n // group_size, group_size, -1).astype(np.float32)


def init_state(
    topology: Topology,
    cfg: TMTConfig,
    rng: np.random.Generator,
    *,
    noise_scale: float = 0.1,
) -> tuple[CircuitState, list[np.ndarray]]:
    """Build a fresh CircuitState plus the per-layer logits list (for ``circuit_forward``).

    Returns ``(state, layered_logits)``. ``state.logits`` is the flat
    [N, 2^arity] view consumed by the TMT; ``layered_logits`` is the
    list-of-tensors view consumed by ``circuit_forward`` and the
    JAX reference. The two are kept in lockstep by the helpers below.
    """
    # Per-layer noisy soft-wire init.
    layered: list[np.ndarray] = []
    for gate_n, group_size in topology.layer_sizes[1:]:
        layered.append(
            make_nops_with_noise(rng, gate_n, topology.arity, group_size, noise_scale=noise_scale)
        )

    flat_logits = np.zeros((topology.n_nodes, cfg.logit_dim), dtype=np.float32)
    for li, (gate_n, _) in enumerate(topology.layer_sizes[1:], start=1):
        s = topology.layer_start[li]
        flat_logits[s : s + gate_n] = layered[li - 1].reshape(gate_n, cfg.logit_dim)

    state = CircuitState(
        logits=flat_logits,
        hidden=np.zeros((topology.n_nodes, cfg.circuit_hidden_dim), dtype=np.float32),
        loss=np.zeros(topology.n_nodes, dtype=np.float32),
        gate_mask=np.ones(topology.n_nodes, dtype=np.float32),
    )
    return state, layered


def flat_to_layered_logits(state: CircuitState, topology: Topology) -> list[np.ndarray]:
    """Reshape the [N, 2^arity] flat logits into the ``circuit_forward`` per-layer format."""
    layered: list[np.ndarray] = []
    for li, (gate_n, group_size) in enumerate(topology.layer_sizes[1:], start=1):
        s = topology.layer_start[li]
        flat = state.logits[s : s + gate_n]
        group_n = gate_n // group_size
        layered.append(flat.reshape(group_n, group_size, -1).astype(np.float32))
    return layered


# ---------------------------------------------------------------------------
# Circuit forward (multilinear LUT interpolation)
# ---------------------------------------------------------------------------


def run_layer(lut: np.ndarray, inputs: list[np.ndarray]) -> np.ndarray:
    """Multilinear LUT forward, matching ``circuits.model.run_layer``."""
    for x in inputs:
        x = x[..., None, None]
        lut = (1.0 - x) * lut[..., ::2] + x * lut[..., 1::2]
    return lut.reshape(*lut.shape[:-3], -1)


def circuit_forward(
    layered_logits: list[np.ndarray],
    wires: list[np.ndarray],
    x: np.ndarray,
    *,
    hard: bool = False,
    layered_gate_mask: list[np.ndarray] | None = None,
) -> list[np.ndarray]:
    """Run the boolean circuit forward, returning per-layer activations.

    Mirrors ``circuits.model.run_circuit``. Inputs are in [0, 1];
    ``hard=True`` rounds each LUT entry to {0, 1} before the layer
    multiplies, recovering discrete Boolean logic.
    """
    if layered_gate_mask is not None:
        x = x * layered_gate_mask[0]
    acts = [x]
    masks_iter = (
        layered_gate_mask[1:]
        if layered_gate_mask is not None
        else [None] * len(layered_logits)
    )
    for ws, lgt, mask in zip(wires, layered_logits, masks_iter, strict=True):
        luts = sigmoid(lgt)
        if hard:
            luts = np.round(luts)
        x = run_layer(luts, [x[..., w] for w in ws])
        if mask is not None:
            x = x * mask
        acts.append(x)
    return acts


# ---------------------------------------------------------------------------
# Per-tick orchestration: TMT step + residual recompute
# ---------------------------------------------------------------------------


def _layered_gate_mask(topology: Topology, gate_mask_flat: np.ndarray) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    for li, (gate_n, _) in enumerate(topology.layer_sizes):
        s = topology.layer_start[li]
        out.append(gate_mask_flat[s : s + gate_n])
    return out


def extract_features(state: CircuitState, topology: Topology, cfg: TMTConfig) -> np.ndarray:
    parts: list[np.ndarray] = [state.logits, state.hidden]
    if cfg.use_intra_layer_PE:
        parts.append(topology.intra_layer_pe)
    if cfg.use_layer_PE:
        parts.append(topology.layer_pe)
    if cfg.use_node_loss:
        parts.append(state.loss[:, None])
    return np.concatenate(parts, axis=-1).astype(np.float32)


def tmt_step(state: CircuitState, topology: Topology, weights: TMTWeights) -> CircuitState:
    """Apply one TMT message-passing step. Returns a new ``CircuitState``.

    Dispatches on ``cfg.model_kind`` between the two architectures we
    support: dense self-attention with a binary mask, or gathered
    neighborhood attention. Both share the input projection, FFN,
    output heads, and ReZero plumbing.
    """
    cfg = weights.config
    f = extract_features(state, topology, cfg)
    z = layer_norm(f, *weights.input_norm)
    z = linear(z, weights.feature_proj_W, weights.feature_proj_b)

    if cfg.model_kind == "self_attention":
        if not isinstance(weights.blocks, list):
            raise TypeError("self_attention expects ``blocks: list[AttentionBlockWeights]``")
        for block in weights.blocks:
            z = attention_block(z, topology.attention_mask, block)
    elif cfg.model_kind == "gathered_attention":
        if not isinstance(weights.blocks, GatheredAttentionWeights):
            raise TypeError("gathered_attention expects ``blocks: GatheredAttentionWeights``")
        ni, nm = topology.neighbor_indices, topology.neighbor_mask
        if ni is None or nm is None:
            ni, nm = build_neighbor_indices(topology.attention_mask, cfg.max_neighbors)
        z = gathered_attention_block(z, ni, nm, weights.blocks)
    else:
        raise ValueError(f"Unknown model_kind: {cfg.model_kind!r}")

    z = layer_norm(z, *weights.final_norm)

    d_logit = linear(z, weights.logit_proj_W, weights.logit_proj_b)
    d_hidden = linear(z, weights.hidden_proj_W, weights.hidden_proj_b)

    update_allowed = (state.gate_mask == 1.0).astype(np.float32)[:, None]
    d_logit = d_logit * update_allowed
    d_hidden = d_hidden * update_allowed

    return replace(
        state,
        logits=state.logits + weights.logit_rezero * d_logit,
        hidden=state.hidden + weights.hidden_rezero * d_hidden,
    )


def compute_residuals(
    state: CircuitState,
    topology: Topology,
    x_data: np.ndarray,
    y_data: np.ndarray,
) -> tuple[CircuitState, dict[str, np.ndarray]]:
    """Run the circuit forward + fold the residuals back into the output rows of ``state.loss``.

    Returns a (new_state, aux_dict). ``aux_dict`` carries soft and
    hard predictions, residuals, and accuracies — exactly the bag the
    GUI / web demo will read off each tick.
    """
    layered = flat_to_layered_logits(state, topology)
    layered_mask = _layered_gate_mask(topology, state.gate_mask)
    soft_acts = circuit_forward(layered, topology.wires, x_data, hard=False, layered_gate_mask=layered_mask)
    hard_acts = circuit_forward(layered, topology.wires, x_data, hard=True, layered_gate_mask=layered_mask)
    pred_soft = soft_acts[-1]
    pred_hard = hard_acts[-1]
    res = pred_soft - y_data
    hard_res = pred_hard - y_data

    new_loss = np.zeros_like(state.loss)
    new_loss[topology.output_start : topology.output_end] = np.abs(res).mean(axis=0)

    aux = {
        "pred_soft": pred_soft.astype(np.float32),
        "pred_hard": pred_hard.astype(np.float32),
        "residuals": res.astype(np.float32),
        "hard_residuals": hard_res.astype(np.float32),
        "accuracy": np.float32((np.abs(res) < 0.5).mean()),
        "hard_accuracy": np.float32((pred_hard == y_data).mean()),
    }
    return replace(state, loss=new_loss), aux


# ---------------------------------------------------------------------------
# Damage primitives (matching boolean_nca_cc.training.pool.structural_perturbation)
# ---------------------------------------------------------------------------


def _eligible_gate_indices(topology: Topology) -> np.ndarray:
    """Flat indices of damage-eligible gates: hidden layers only."""
    out: list[np.ndarray] = []
    for li, (gate_n, _) in enumerate(topology.layer_sizes):
        if li == 0 or li == len(topology.layer_sizes) - 1:
            continue
        s = topology.layer_start[li]
        out.append(np.arange(s, s + gate_n, dtype=np.int32))
    return np.concatenate(out) if out else np.zeros(0, dtype=np.int32)


def damage_gate(state: CircuitState, flat_idx: int, faulty_value: float = -10.0) -> CircuitState:
    """Clamp a single gate by index. Used for click-to-damage."""
    new_logits = state.logits.copy()
    new_mask = state.gate_mask.copy()
    new_logits[flat_idx, :] = faulty_value
    new_mask[flat_idx] = 0.0
    return replace(state, logits=new_logits, gate_mask=new_mask)


def apply_shotgun(
    state: CircuitState,
    topology: Topology,
    n_gates: int,
    rng: np.random.Generator,
    faulty_value: float = -10.0,
) -> tuple[CircuitState, np.ndarray]:
    """Stamp ``n_gates`` random gates from hidden layers as damaged.

    Returns ``(new_state, chosen_indices)`` so the caller can highlight
    the new damage on the canvas.
    """
    eligible = _eligible_gate_indices(topology)
    n_gates = min(n_gates, eligible.size)
    if n_gates == 0:
        return state, np.zeros(0, dtype=np.int32)
    chosen = rng.choice(eligible, size=n_gates, replace=False)
    new_logits = state.logits.copy()
    new_mask = state.gate_mask.copy()
    new_logits[chosen, :] = faulty_value
    new_mask[chosen] = 0.0
    return replace(state, logits=new_logits, gate_mask=new_mask), chosen


def reset_circuit(
    topology: Topology, cfg: TMTConfig, rng: np.random.Generator, *, noise_scale: float = 0.1
) -> CircuitState:
    """Re-initialise to soft-wire identity + noise. Damage is cleared."""
    state, _ = init_state(topology, cfg, rng, noise_scale=noise_scale)
    return state


def shuffle_wires(
    topology: Topology, cfg: TMTConfig, rng: np.random.Generator, *, hidden_dim: int
) -> Topology:
    """Regenerate the random wiring per layer. Logits and hidden state are kept by
    the *caller*; this function only rebuilds the topology + attention mask + PEs.

    Carries ``max_neighbors`` over so the gathered cache is refreshed too.
    """
    new_wires: list[np.ndarray] = []
    for li in range(1, len(topology.layer_sizes)):
        in_n = topology.layer_sizes[li - 1][0]
        out_n, group_size = topology.layer_sizes[li]
        new_wires.append(gen_wires(rng, in_n, out_n, topology.arity, group_size))
    max_neighbors = cfg.max_neighbors if cfg.model_kind == "gathered_attention" else None
    return build_topology(
        topology.layer_sizes,
        new_wires,
        topology.arity,
        hidden_dim,
        bidirectional_edges=True,
        max_neighbors=max_neighbors,
    )


# ---------------------------------------------------------------------------
# Random-weight factory (used by the self-test; does NOT match a real checkpoint)
# ---------------------------------------------------------------------------


def make_random_weights(cfg: TMTConfig, rng: np.random.Generator) -> TMTWeights:
    """Build a TMTWeights with sensible random init. For self-tests only.

    Dispatches on ``cfg.model_kind``. The real export path lives in
    ``extract_weights.py`` (M2) and pulls each tensor out of an
    ``nnx.State`` with the canonical names.
    """

    def gauss(*shape: int, scale: float = 0.02) -> np.ndarray:
        return (rng.normal(size=shape) * scale).astype(np.float32)

    def ones(*shape: int) -> np.ndarray:
        return np.ones(shape, dtype=np.float32)

    def zeros(*shape: int) -> np.ndarray:
        return np.zeros(shape, dtype=np.float32)

    D = cfg.attention_dim
    H = cfg.num_heads
    d_h = D // H
    F = cfg.feature_dim
    L = cfg.logit_dim
    h = cfg.circuit_hidden_dim
    mlp = D * 2

    if cfg.model_kind == "self_attention":
        blocks: list[AttentionBlockWeights] = []
        for _ in range(cfg.num_attn_layers):
            mha = MHAWeights(
                Wq=gauss(D, H, d_h),
                bq=zeros(H, d_h),
                Wk=gauss(D, H, d_h),
                bk=zeros(H, d_h),
                Wv=gauss(D, H, d_h),
                bv=zeros(H, d_h),
                Wo=gauss(H, d_h, D),
                bo=zeros(D),
                q_ln_gamma=ones(d_h),
                q_ln_beta=zeros(d_h),
                k_ln_gamma=ones(d_h),
                k_ln_beta=zeros(d_h),
            )
            blocks.append(
                AttentionBlockWeights(
                    ln_q=(ones(D), zeros(D)),
                    ln_kv=(ones(D), zeros(D)),
                    mha=mha,
                    ffn_ln=(ones(D), zeros(D)),
                    ffn_W1=gauss(D, mlp),
                    ffn_b1=zeros(mlp),
                    ffn_W2=gauss(mlp, D),
                    ffn_b2=zeros(D),
                    attn_rezero=0.0,
                    ffn_rezero=0.0,
                )
            )
        attn_weights: list[AttentionBlockWeights] | GatheredAttentionWeights = blocks
    elif cfg.model_kind == "gathered_attention":
        attn_weights = GatheredAttentionWeights(
            attn_norm=(ones(D), zeros(D)),
            Wq=gauss(D, D),
            bq=zeros(D),
            Wk=gauss(D, D),
            bk=zeros(D),
            Wv=gauss(D, D),
            bv=zeros(D),
            Wo=gauss(D, D),
            bo=zeros(D),
            q_ln_gamma=ones(d_h),
            k_ln_gamma=ones(d_h),
            ffn_ln=(ones(D), zeros(D)),
            ffn_W1=gauss(D, mlp),
            ffn_b1=zeros(mlp),
            ffn_W2=gauss(mlp, D),
            ffn_b2=zeros(D),
            attn_rezero=0.0,
            ffn_rezero=0.0,
            num_heads=H,
        )
    else:
        raise ValueError(f"Unknown model_kind: {cfg.model_kind!r}")

    return TMTWeights(
        config=cfg,
        input_norm=(ones(F), zeros(F)),
        feature_proj_W=gauss(F, D),
        feature_proj_b=zeros(D),
        blocks=attn_weights,
        final_norm=(ones(D), zeros(D)),
        logit_proj_W=gauss(D, L, scale=0.01),
        logit_proj_b=zeros(L),
        hidden_proj_W=gauss(D, h, scale=0.01),
        hidden_proj_b=zeros(h),
        logit_rezero=0.0,
        hidden_rezero=0.0,
    )


# ---------------------------------------------------------------------------
# CLI: smoke test
# ---------------------------------------------------------------------------


def _cli_self_test() -> None:
    from .test_oracle import run_self_test_all_kinds

    ok = run_self_test_all_kinds(verbose=True)
    raise SystemExit(0 if ok else 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--self-test", action="store_true", help="Run the random-weight self-test."
    )
    args = parser.parse_args()
    if args.self_test:
        _cli_self_test()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
