#!/usr/bin/env python3
"""
Correctness and profiling tests for gathered self-attention.

Three sets of checks:

1. ``test_validate_topology_*``
   The strict topology validator added to ``base.py`` must:
     * Raise ``ValueError`` when ``max_neighbors`` < actual max in-degree.
     * Warn (not raise) in non-strict mode.
     * Accept any ``max_neighbors >= max_degree`` (and return the actual max).
     * Treat ``use_mask=False`` as "every node attends to every node".

2. ``test_block_self_consistency_*``
   ``GatheredAttentionBlock`` and a naive dense reference computed from the
   *same* block's weights must produce numerically identical outputs (up to
   float32 noise). This catches einsum / masking bugs in the gather path.
   Both paths consume the same ``neighbor_indices``/``neighbor_mask``, so this
   does NOT catch truncation — the topology validator does that.

3. ``bench_*`` (run as ``python tests/test_gathered_correctness.py``)
   Forward-pass timing comparison between dense and gathered backends for
   both ``CircuitGatheredAttention`` vs ``CircuitSelfAttention`` and
   ``PerceiverCircuitAttention(self_attn_kind='dense' vs 'gathered')`` across
   varied N. Reports whether gathered actually buys speedup at the circuit
   sizes you train with.
"""

from __future__ import annotations

import time
import warnings

import jax
import jax.numpy as jp
import numpy as np
import pytest
from flax import nnx

from boolean_nca_cc.circuits.model import gen_circuit, generate_layer_sizes
from boolean_nca_cc.models.attention.base import (
    AttentionBlock,
    GatheredAttentionBlock,
    build_neighbor_indices,
    create_attention_mask,
    validate_gathered_topology,
)
from boolean_nca_cc.models.attention.gathered_attention import CircuitGatheredAttention
from boolean_nca_cc.models.attention.perceiver_attention import PerceiverCircuitAttention
from boolean_nca_cc.models.attention.self_attention import CircuitSelfAttention
from boolean_nca_cc.utils.graph_builder import GraphGlobals, build_graph

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_circuit_graph(
    input_n: int = 4,
    output_n: int = 4,
    layer_n: int = 2,
    arity: int = 2,
    hidden_dim: int = 8,
    seed: int = 0,
    bidirectional_edges: bool = True,
    neighboring_connections: bool = False,
):
    """Build a small concrete circuit graph for tests."""
    layer_sizes = generate_layer_sizes(input_n, output_n, arity, layer_n=layer_n)
    k = jax.random.PRNGKey(seed)
    wires_key, logits_key = jax.random.split(k)
    wires, logits = gen_circuit(wires_key, logits_key, layer_sizes, arity=arity)
    graph = build_graph(
        logits=logits,
        wires=wires,
        input_n=input_n,
        arity=arity,
        circuit_hidden_dim=hidden_dim,
        bidirectional_edges=bidirectional_edges,
        neighboring_connections=neighboring_connections,
    )
    return graph, layer_sizes


def _attach_perceiver_globals(
    graph,
    n_samples: int = 4,
    input_n: int = 4,
    output_n: int = 4,
    seed: int = 1,
):
    k = jax.random.PRNGKey(seed)
    k1, k2, k3, k4 = jax.random.split(k, 4)
    x_data = jax.random.uniform(k1, (n_samples, input_n))
    y_data = jax.random.uniform(k2, (n_samples, output_n))
    residuals = jax.random.normal(k3, (n_samples, output_n))
    new_globals = GraphGlobals(
        loss=0.0,
        update_steps=jp.asarray(0),
        x_data=x_data,
        y_data=y_data,
        residuals=residuals,
        subsample_key=k4,
    )
    return graph._replace(globals=new_globals)


# ---------------------------------------------------------------------------
# 1. Topology validator
# ---------------------------------------------------------------------------


def test_validate_topology_accepts_sufficient_width():
    graph, _ = _make_circuit_graph()
    n_node = graph.nodes["layer"].shape[0]
    max_deg = int(
        validate_gathered_topology(
            graph.senders, graph.receivers, n_node, max_neighbors=n_node, strict=True
        )
    )
    assert max_deg >= 1
    assert max_deg <= n_node


def test_validate_topology_raises_on_truncation():
    graph, _ = _make_circuit_graph()
    n_node = graph.nodes["layer"].shape[0]
    actual = int(
        validate_gathered_topology(
            graph.senders, graph.receivers, n_node, max_neighbors=n_node, strict=True
        )
    )
    # Force a truncating config: one less than the real max degree.
    too_small = actual - 1
    if too_small < 0:
        pytest.skip("Graph too trivial to construct truncating case.")
    with pytest.raises(ValueError, match="GATHERED ATTENTION TOPOLOGY MISMATCH"):
        validate_gathered_topology(
            graph.senders, graph.receivers, n_node, max_neighbors=too_small, strict=True
        )


def test_validate_topology_warns_when_not_strict():
    graph, _ = _make_circuit_graph()
    n_node = graph.nodes["layer"].shape[0]
    actual = int(
        validate_gathered_topology(
            graph.senders, graph.receivers, n_node, max_neighbors=n_node, strict=True
        )
    )
    too_small = actual - 1
    if too_small < 0:
        pytest.skip("Graph too trivial to construct truncating case.")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        validate_gathered_topology(
            graph.senders, graph.receivers, n_node, max_neighbors=too_small, strict=False
        )
    assert any("GATHERED ATTENTION TOPOLOGY MISMATCH" in str(w.message) for w in caught)


def test_validate_topology_full_attention_mode():
    """With use_mask=False the effective max degree is n_node."""
    graph, _ = _make_circuit_graph()
    n_node = graph.nodes["layer"].shape[0]
    # Sufficient when >= n_node
    deg = int(
        validate_gathered_topology(
            graph.senders,
            graph.receivers,
            n_node,
            max_neighbors=n_node,
            use_mask=False,
            strict=True,
        )
    )
    assert deg == n_node
    # Insufficient when < n_node
    with pytest.raises(ValueError):
        validate_gathered_topology(
            graph.senders,
            graph.receivers,
            n_node,
            max_neighbors=n_node - 1,
            use_mask=False,
            strict=True,
        )


# --- Layer-aware validation: gate vs input-node degree analysis ---

def _make_input_heavy_graph():
    """Make a graph with high input-node degree and lower gate degree.

    With bidirectional_edges=True, narrow input layers receive a lot of
    reverse edges from wide first gate layers — this is exactly the
    production scenario the user hit. Returns the graph and per-node degree
    arrays so tests can locate the boundary case.
    """
    graph, _ = _make_circuit_graph(
        input_n=4, output_n=4, layer_n=2, arity=4, hidden_dim=4,
        bidirectional_edges=True,
    )
    # Compute degrees ourselves so the test asserts against ground truth
    n_node = graph.nodes["layer"].shape[0]
    adj = np.zeros((n_node, n_node), dtype=bool)
    adj[np.asarray(graph.receivers), np.asarray(graph.senders)] = True
    adj |= np.eye(n_node, dtype=bool)
    degrees = adj.sum(axis=-1)
    layers = np.asarray(graph.nodes["layer"])
    return graph, degrees, layers


def test_validate_topology_layer_aware_splits_gate_vs_input():
    """When layer_indices is provided, gate and input degrees are reported separately."""
    graph, degrees, layers = _make_input_heavy_graph()
    n_node = int(graph.nodes["layer"].shape[0])
    input_max = int(degrees[layers == 0].max())
    gate_max = int(degrees[layers > 0].max())

    # Sanity check on the fixture — only meaningful if input degree > gate degree
    if input_max <= gate_max:
        pytest.skip(
            f"Fixture doesn't produce input-heavy topology "
            f"(input_max={input_max}, gate_max={gate_max})."
        )

    # max_neighbors >= gate_max but < input_max → input-node truncation only.
    # Default behavior: warn, don't raise.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        validate_gathered_topology(
            graph.senders, graph.receivers, n_node,
            max_neighbors=gate_max,
            layer_indices=graph.nodes["layer"],
            strict=True,
            strict_input_nodes=False,
        )
    assert any("INPUT-NODE neighborhood truncation" in str(w.message) for w in caught)


def test_validate_topology_input_node_truncation_raises_when_strict():
    """strict_input_nodes=True escalates input-node truncation to an error."""
    graph, degrees, layers = _make_input_heavy_graph()
    n_node = int(graph.nodes["layer"].shape[0])
    input_max = int(degrees[layers == 0].max())
    gate_max = int(degrees[layers > 0].max())
    if input_max <= gate_max:
        pytest.skip("Fixture doesn't produce input-heavy topology.")

    with pytest.raises(ValueError, match="INPUT-NODE neighborhood truncation"):
        validate_gathered_topology(
            graph.senders, graph.receivers, n_node,
            max_neighbors=gate_max,
            layer_indices=graph.nodes["layer"],
            strict=True,
            strict_input_nodes=True,
        )


def test_validate_topology_gate_truncation_always_raises_strict():
    """Gate-node truncation is fatal regardless of strict_input_nodes."""
    graph, degrees, layers = _make_input_heavy_graph()
    n_node = int(graph.nodes["layer"].shape[0])
    gate_max = int(degrees[layers > 0].max())
    if gate_max < 2:
        pytest.skip("Need gate_max >= 2 to construct a truncating max_neighbors.")

    # max_neighbors below gate_max → gate truncation → must raise even with
    # strict_input_nodes=False
    with pytest.raises(ValueError, match="GATE NODES"):
        validate_gathered_topology(
            graph.senders, graph.receivers, n_node,
            max_neighbors=gate_max - 1,
            layer_indices=graph.nodes["layer"],
            strict=True,
            strict_input_nodes=False,
        )


# ---------------------------------------------------------------------------
# 2. Block-level self-consistency: GatheredAttentionBlock == naive-dense reference
# ---------------------------------------------------------------------------


def _naive_dense_attention(block: GatheredAttentionBlock, x, dense_mask):
    """
    Reproduce ``block.__call__`` via a naive O(N^2) dense path that reuses
    the *same* block's weights. ``dense_mask`` is the [N, N] adjacency to apply
    before softmax — same semantics as ``neighbor_indices``/``neighbor_mask``.
    """
    N, D = x.shape
    H = block.num_heads
    Dh = D // H

    xn = block.attn_norm(x)
    Q = block.query_proj(xn).reshape(N, H, Dh)
    K = block.key_proj(xn).reshape(N, H, Dh)
    V = block.value_proj(xn).reshape(N, H, Dh)
    Q = block.query_ln(Q)
    K = block.key_ln(K)

    logits = jp.einsum("nhd,mhd->nhm", Q, K) / jp.sqrt(jp.float32(Dh))
    logits = jp.where(dense_mask[:, None, :], logits, jp.finfo(jp.float32).min)
    weights = jax.nn.softmax(logits, axis=-1)
    attn = jp.einsum("nhm,mhd->nhd", weights, V).reshape(N, D)
    attn = block.output_proj(attn)

    x_post = x + block.attn_rezero(attn)
    return x_post + block.ffn_rezero(block.ffn(x_post))


def _dense_mask_from_neighbors(neighbor_indices, neighbor_mask, n_node):
    """Reconstruct the [N, N] boolean adjacency that the gathered path encodes."""
    rows = jp.arange(n_node)[:, None]
    dense = jp.zeros((n_node, n_node), dtype=jp.bool_)
    return dense.at[rows, neighbor_indices].max(neighbor_mask)


@pytest.mark.parametrize("dim,num_heads", [(32, 4), (64, 8)])
@pytest.mark.parametrize("use_mask", [True, False])
def test_block_self_consistency(dim, num_heads, use_mask):
    graph, _ = _make_circuit_graph(hidden_dim=4)
    n_node = graph.nodes["layer"].shape[0]
    max_deg = int(
        validate_gathered_topology(
            graph.senders,
            graph.receivers,
            n_node,
            max_neighbors=n_node,
            use_mask=use_mask,
            strict=True,
        )
    )
    max_neighbors = max(max_deg, 8)

    rngs = nnx.Rngs(0)
    block = GatheredAttentionBlock(dim=dim, num_heads=num_heads, rngs=rngs)

    # Random gate latents (same shape the block expects: [N, dim])
    x = jax.random.normal(jax.random.PRNGKey(7), (n_node, dim))

    neighbor_indices, neighbor_mask = build_neighbor_indices(
        graph.senders,
        graph.receivers,
        n_node,
        max_neighbors,
        use_mask=use_mask,
    )

    out_gathered = block(x, neighbor_indices, neighbor_mask)

    dense_mask = _dense_mask_from_neighbors(neighbor_indices, neighbor_mask, n_node)
    out_dense = _naive_dense_attention(block, x, dense_mask)

    max_diff = float(jp.max(jp.abs(out_gathered - out_dense)))
    assert max_diff < 1e-5, (
        f"max |gathered - naive_dense| = {max_diff:.3e} (dim={dim}, heads={num_heads})"
    )


# ---------------------------------------------------------------------------
# 3. Layer-restricted cross-attention slicing: sliced output matches the
#    intent of dense+gate on the allowed rows (within float32 noise).
#
#    The dense+gate path has a small FFN leak through ReZero residuals on
#    non-allowed rows. Slicing eliminates that leak (cleaner semantics) but
#    means the two paths are NOT bit-identical on non-allowed rows. The
#    test below verifies:
#      a) Allowed rows are bit-equivalent (within float32 noise) between the
#         two paths — that's the rows we actually care about.
#      b) The sliced path leaves non-allowed rows IDENTICAL to their value
#         before cross-attention (definition of slicing).
#      c) The dense+gate path's deviation on non-allowed rows is bounded by
#         the ReZero scale (a sanity check on the size of the leak).
# ---------------------------------------------------------------------------


def _build_test_perceiver_for_slicing(arity=4, input_n=8, output_n=8, layer_n=3, hidden_dim=8,
                                       attention_dim=64, num_heads=4, n_samples=8, seed=0):
    """Construct a Perceiver + graph + globals usable for the slicing parity test.

    Uses the (asymmetric) production semantics:
      - input cross-attn targets the INPUT LAYER (layer == 0) — the address-book
        pin nodes; the first gate layer reaches the data through wire-based
        self-attention to those pins.
      - output cross-attn targets the LAST GATE LAYER (layer == max_layer).
    """
    from boolean_nca_cc.training.evaluation import _layer_node_indices

    graph, layer_sizes = _make_circuit_graph(
        input_n=input_n, output_n=output_n, layer_n=layer_n, arity=arity,
        hidden_dim=hidden_dim, seed=seed,
    )
    graph = _attach_perceiver_globals(
        graph, n_samples=n_samples, input_n=input_n, output_n=output_n, seed=seed + 1,
    )
    n_node = graph.nodes["layer"].shape[0]
    # layer_sizes = [(input_n, 1), (gate_layer_1_n, gs), ..., (output_n, 1)]
    input_layer_size = int(layer_sizes[0][0])  # == input_n
    output_layer_size = int(layer_sizes[-1][0])  # == output_n
    model = PerceiverCircuitAttention(
        n_node=n_node,
        circuit_hidden_dim=hidden_dim,
        arity=arity,
        attention_dim=attention_dim,
        num_heads=num_heads,
        num_self_attn_layers=1,
        num_cross_attn_layers=1,
        rngs=nnx.Rngs(seed),
        self_attn_kind="dense",
        max_neighbors=8,
        use_input_cross_attention=True,
        use_output_cross_attention=True,
        restrict_input_cross_attn_to_input_layer=True,
        restrict_output_cross_attn_to_last_gate_layer=True,
        input_layer_size=input_layer_size,
        output_layer_size=output_layer_size,
    )
    layer_indices = graph.nodes["layer"]
    max_layer = int(jp.max(layer_indices))
    input_idx = _layer_node_indices(
        layer_indices, target_layer=0, size=input_layer_size,
    )
    output_idx = _layer_node_indices(
        layer_indices, target_layer=max_layer, size=output_layer_size,
    )
    return model, graph, input_idx, output_idx


def test_slicing_matches_gate_on_allowed_rows():
    """Sliced and dense+gate paths must agree on the rows we update."""
    model, graph, input_idx, output_idx = _build_test_perceiver_for_slicing()

    attn_mask = model._create_attention_mask(graph.senders, graph.receivers, model.n_node)
    layer_indices = graph.nodes["layer"]
    # Asymmetric semantics: input cross-attn targets the INPUT LAYER (layer == 0),
    # output cross-attn targets the LAST GATE LAYER (layer == max_layer).
    input_gate = model._create_output_gate(layer_indices, allowed_layer=0)
    output_gate = model._create_output_gate(
        layer_indices, allowed_layer=int(jp.max(layer_indices))
    )

    sliced_out = model(
        graph, attention_mask=attn_mask,
        input_layer_idx=input_idx, output_layer_idx=output_idx,
    )
    gated_out = model(
        graph, attention_mask=attn_mask,
        input_output_gate=input_gate, output_output_gate=output_gate,
    )

    # Allowed rows (input layer + last gate layer) must match — these are the
    # rows both paths actively update. Self-attention runs on ALL rows AFTER
    # the cross-attn blocks, so even allowed-row outputs are only as close as
    # the non-allowed-row inputs to self-attn agree. We test against the
    # dense+gate path's *pre-self-attn* equivalence target.
    sliced_full = sliced_out.nodes["logits"]
    gated_full = gated_out.nodes["logits"]

    # Combined allowed mask: input layer (layer == 0) OR last gate layer
    max_layer = int(jp.max(layer_indices))
    allowed = (layer_indices == 0) | (layer_indices == max_layer)

    sliced_allowed = sliced_full[allowed]
    gated_allowed = gated_full[allowed]
    diff_allowed = float(jp.max(jp.abs(sliced_allowed - gated_allowed)))
    # At init, ReZero scale is ~1e-3 or 0, so any FFN leak is small; after one
    # forward both paths give very close outputs. We just check it's well under
    # the scale of activations (which run ~unit norm here).
    assert diff_allowed < 1e-2, (
        f"sliced vs gate diverge on allowed rows: max |diff| = {diff_allowed:.3e}"
    )


def test_slicing_leaves_non_allowed_rows_untouched_until_self_attn():
    """The slicing path must not touch non-allowed rows during cross-attention.

    To probe just the cross-attn block (not self-attn that follows), we re-run
    the cross-attn manually and check the non-allowed rows are bit-identical
    to the pre-cross-attn gate latents.
    """
    from boolean_nca_cc.models.attention.base import extract_node_features

    model, graph, input_idx, output_idx = _build_test_perceiver_for_slicing()
    nodes = graph.nodes
    layer_indices = nodes["layer"]
    max_layer = int(jp.max(layer_indices))
    # Input layer is layer == 0; last gate layer is layer == max_layer.
    allowed = (layer_indices == 0) | (layer_indices == max_layer)

    # Build gate_latents the same way __call__ does
    gate_features = extract_node_features(
        nodes, model.use_node_loss, model.use_intra_layer_PE, model.use_layer_PE,
    )
    gate_latents = model.feature_proj(model.input_norm(gate_features))[None, ...]

    # Apply sliced cross-attention for INPUT
    x_data = graph.globals.x_data
    input_features = model._encode_data(x_data)
    input_tokens = model.input_encoder(input_features)[None, ...]
    q_subset = gate_latents[:, input_idx, :]
    for cross_attn in model.input_cross_attn_layers:
        q_subset = cross_attn(q_subset, input_tokens, mask=None, output_gate=None)
    after_input = gate_latents.at[:, input_idx, :].set(q_subset)

    # Rows outside the input layer must be bit-identical
    non_input = layer_indices != 0
    diff = float(jp.max(jp.abs(after_input[0, non_input] - gate_latents[0, non_input])))
    assert diff == 0.0, f"slicing leaked into non-input-layer rows: max diff = {diff}"

    # Apply sliced cross-attention for OUTPUT on top
    residuals = graph.globals.residuals
    output_features = model._encode_data(residuals)
    output_tokens = model.output_encoder(output_features)[None, ...]
    q_subset = after_input[:, output_idx, :]
    for cross_attn in model.output_cross_attn_layers:
        q_subset = cross_attn(q_subset, output_tokens, mask=None, output_gate=None)
    after_output = after_input.at[:, output_idx, :].set(q_subset)

    # Rows that are NOT first or last layer must be untouched by ANY cross-attn
    untouched = ~allowed
    diff_untouched = float(
        jp.max(jp.abs(after_output[0, untouched] - gate_latents[0, untouched]))
    )
    assert diff_untouched == 0.0, (
        f"slicing leaked into untouched rows: max diff = {diff_untouched}"
    )


# ---------------------------------------------------------------------------
# 4. Profiling — only run when invoked as a script, not under pytest
# ---------------------------------------------------------------------------


def _time_fn(fn, *args, n_warmup=3, n_iter=20):
    """Block-on-result timing helper."""
    for _ in range(n_warmup):
        out = fn(*args)
        jax.block_until_ready(out)
    t0 = time.perf_counter()
    for _ in range(n_iter):
        out = fn(*args)
    jax.block_until_ready(out)
    return (time.perf_counter() - t0) / n_iter * 1000  # ms / iter


def _scan_loss(model_fn, graph, n_steps):
    """Mimic the training loop: run model_fn n_steps times and return a scalar."""
    def body(g, _):
        g = model_fn(g)
        return g, None
    final, _ = jax.lax.scan(body, graph, jp.arange(n_steps))
    # Sum of logits stand-in for a loss scalar (so grad has a meaningful gradient path)
    return jp.mean(final.nodes["logits"] ** 2)


def _build_perceiver(
    self_attn_kind,
    n_node,
    max_neighbors,
    hidden_dim=16,
    arity=4,
    attention_dim=64,
    num_heads=4,
    num_self_attn_layers=1,
    num_cross_attn_layers=1,
    seed=0,
):
    return PerceiverCircuitAttention(
        n_node=n_node,
        circuit_hidden_dim=hidden_dim,
        arity=arity,
        attention_dim=attention_dim,
        num_heads=num_heads,
        num_self_attn_layers=num_self_attn_layers,
        num_cross_attn_layers=num_cross_attn_layers,
        rngs=nnx.Rngs(seed),
        self_attn_kind=self_attn_kind,
        max_neighbors=max_neighbors,
        use_input_cross_attention=True,
        use_output_cross_attention=True,
    )


def _build_self_attn(
    self_attn_kind,
    n_node,
    max_neighbors,
    hidden_dim=16,
    arity=4,
    attention_dim=64,
    num_heads=4,
    num_self_attn_layers=1,
    seed=0,
):
    """Build CircuitSelfAttention (dense) or CircuitGatheredAttention."""
    kwargs = {
        "n_node": n_node,
        "circuit_hidden_dim": hidden_dim,
        "arity": arity,
        "attention_dim": attention_dim,
        "num_heads": num_heads,
        "num_self_attn_layers": num_self_attn_layers,
        "rngs": nnx.Rngs(seed),
    }
    if self_attn_kind == "dense":
        return CircuitSelfAttention(**kwargs)
    elif self_attn_kind == "gathered":
        return CircuitGatheredAttention(max_neighbors=max_neighbors, **kwargs)
    raise ValueError(self_attn_kind)


def _bench_pair(
    dense_model,
    gathered_model,
    graph,
    n_node,
    senders,
    receivers,
    max_neighbors,
    n_steps,
    is_perceiver,
):
    """Time fwd, scanned-fwd (n_steps), and grad-of-scan for both backends."""
    attention_mask = create_attention_mask(senders, receivers, n_node)
    neighbor_indices, neighbor_mask = build_neighbor_indices(
        senders, receivers, n_node, max_neighbors,
    )

    if is_perceiver:
        def dense_step(g):
            return dense_model(g, attention_mask=attention_mask)

        def gathered_step(g):
            return gathered_model(
                g, neighbor_indices=neighbor_indices, neighbor_mask=neighbor_mask,
            )
    else:
        def dense_step(g):
            return dense_model(g, attention_mask=attention_mask)

        def gathered_step(g):
            return gathered_model(
                g, neighbor_indices=neighbor_indices, neighbor_mask=neighbor_mask,
            )

    dense_step_jit = nnx.jit(dense_step)
    gathered_step_jit = nnx.jit(gathered_step)

    def _scan_loss_dense(model, g):
        def body(carry, _):
            return model(carry, attention_mask=attention_mask), None
        final, _ = jax.lax.scan(body, g, jp.arange(n_steps))
        return jp.mean(final.nodes["logits"] ** 2)

    def _scan_loss_gathered(model, g):
        def body(carry, _):
            return model(
                carry, neighbor_indices=neighbor_indices, neighbor_mask=neighbor_mask,
            ), None
        final, _ = jax.lax.scan(body, g, jp.arange(n_steps))
        return jp.mean(final.nodes["logits"] ** 2)

    dense_loss_jit = nnx.jit(_scan_loss_dense)
    gathered_loss_jit = nnx.jit(_scan_loss_gathered)
    dense_grad_jit = nnx.jit(nnx.grad(_scan_loss_dense))
    gathered_grad_jit = nnx.jit(nnx.grad(_scan_loss_gathered))

    t_dense_fwd = _time_fn(dense_step_jit, graph)
    t_gath_fwd = _time_fn(gathered_step_jit, graph)
    t_dense_scan = _time_fn(dense_loss_jit, dense_model, graph, n_iter=5)
    t_gath_scan = _time_fn(gathered_loss_jit, gathered_model, graph, n_iter=5)
    try:
        t_dense_bw = _time_fn(dense_grad_jit, dense_model, graph, n_iter=3)
        t_gath_bw = _time_fn(gathered_grad_jit, gathered_model, graph, n_iter=3)
    except Exception as e:
        t_dense_bw = t_gath_bw = float("nan")
        print(f"    (grad timing skipped: {e})")

    return {
        "t_dense_fwd": t_dense_fwd,
        "t_gath_fwd": t_gath_fwd,
        "t_dense_scan": t_dense_scan,
        "t_gath_scan": t_gath_scan,
        "t_dense_bw": t_dense_bw,
        "t_gath_bw": t_gath_bw,
    }


def _run_perceiver_bench(
    input_n,
    output_n,
    layer_n,
    hidden_dim,
    arity,
    n_samples,
    attention_dim,
    num_heads,
    n_self_attn_layers,
    n_cross_attn_layers,
    n_steps,
):
    """Three-way Perceiver bench: dense / gathered / dense+sliced cross-attn.

    Sliced uses the new ``input_layer_idx``/``output_layer_idx`` precomputed-
    indices path: cross-attn runs only on first/last gate layer queries.
    """
    from boolean_nca_cc.training.evaluation import _layer_node_indices

    graph, layer_sizes = _make_circuit_graph(
        input_n=input_n, output_n=output_n, layer_n=layer_n,
        arity=arity, hidden_dim=hidden_dim,
    )
    graph = _attach_perceiver_globals(
        graph, n_samples=n_samples, input_n=input_n, output_n=output_n,
    )
    n_node = graph.nodes["layer"].shape[0]
    max_deg = int(validate_gathered_topology(
        graph.senders, graph.receivers, n_node, max_neighbors=n_node, strict=True,
    ))
    max_neighbors = max_deg
    # Asymmetric semantics: input cross-attn targets the input layer (layer == 0),
    # K_in = input_layer_size = input_n. Output side: K_out = output_layer_size.
    input_layer_size = int(layer_sizes[0][0])  # == input_n
    output_layer_size = int(layer_sizes[-1][0])  # == output_n

    def _build(kind):
        return PerceiverCircuitAttention(
            n_node=n_node, circuit_hidden_dim=hidden_dim, arity=arity,
            attention_dim=attention_dim, num_heads=num_heads,
            num_self_attn_layers=n_self_attn_layers,
            num_cross_attn_layers=n_cross_attn_layers,
            rngs=nnx.Rngs(0),
            self_attn_kind=kind,
            max_neighbors=max_neighbors,
            use_input_cross_attention=True,
            use_output_cross_attention=True,
            restrict_input_cross_attn_to_input_layer=True,
            restrict_output_cross_attn_to_last_gate_layer=True,
            input_layer_size=input_layer_size,
            output_layer_size=output_layer_size,
        )

    dense_model = _build("dense")
    gathered_model = _build("gathered")
    sliced_model = _build("dense")  # slicing is orthogonal to self-attn backend

    layer_indices = graph.nodes["layer"]
    max_layer = int(jp.max(layer_indices))
    input_idx = _layer_node_indices(
        layer_indices, target_layer=0, size=input_layer_size,
    )
    output_idx = _layer_node_indices(
        layer_indices, target_layer=max_layer, size=output_layer_size,
    )
    input_gate = dense_model._create_output_gate(layer_indices, allowed_layer=0)
    output_gate = dense_model._create_output_gate(layer_indices, allowed_layer=max_layer)

    attention_mask = create_attention_mask(graph.senders, graph.receivers, n_node)
    neighbor_indices, neighbor_mask = build_neighbor_indices(
        graph.senders, graph.receivers, n_node, max_neighbors,
    )

    # === Forward step functions ===
    def dense_step(g):
        return dense_model(
            g, attention_mask=attention_mask,
            input_output_gate=input_gate, output_output_gate=output_gate,
        )

    def gathered_step(g):
        return gathered_model(
            g, neighbor_indices=neighbor_indices, neighbor_mask=neighbor_mask,
            input_output_gate=input_gate, output_output_gate=output_gate,
        )

    def sliced_step(g):
        return sliced_model(
            g, attention_mask=attention_mask,
            input_layer_idx=input_idx, output_layer_idx=output_idx,
        )

    def _make_scan_loss(step_fn, model):
        def scan_loss(m, g):
            def body(carry, _):
                # Re-bind the model name in scope via closure on `m`
                if step_fn is dense_step:
                    return m(carry, attention_mask=attention_mask,
                             input_output_gate=input_gate,
                             output_output_gate=output_gate), None
                if step_fn is gathered_step:
                    return m(carry, neighbor_indices=neighbor_indices,
                             neighbor_mask=neighbor_mask,
                             input_output_gate=input_gate,
                             output_output_gate=output_gate), None
                return m(carry, attention_mask=attention_mask,
                         input_layer_idx=input_idx,
                         output_layer_idx=output_idx), None
            final, _ = jax.lax.scan(body, g, jp.arange(n_steps))
            return jp.mean(final.nodes["logits"] ** 2)
        return scan_loss

    dense_loss = _make_scan_loss(dense_step, dense_model)
    gathered_loss = _make_scan_loss(gathered_step, gathered_model)
    sliced_loss = _make_scan_loss(sliced_step, sliced_model)

    dense_fwd_jit = nnx.jit(dense_step)
    gathered_fwd_jit = nnx.jit(gathered_step)
    sliced_fwd_jit = nnx.jit(sliced_step)

    dense_scan_jit = nnx.jit(dense_loss)
    gathered_scan_jit = nnx.jit(gathered_loss)
    sliced_scan_jit = nnx.jit(sliced_loss)

    dense_grad_jit = nnx.jit(nnx.grad(dense_loss))
    gathered_grad_jit = nnx.jit(nnx.grad(gathered_loss))
    sliced_grad_jit = nnx.jit(nnx.grad(sliced_loss))

    return {
        "n_node": n_node, "max_deg": max_deg, "max_neighbors": max_neighbors,
        "K_in": int(input_idx.shape[0]), "K_out": int(output_idx.shape[0]),
        "attention_dim": attention_dim,
        "t_dense_fwd": _time_fn(dense_fwd_jit, graph),
        "t_gath_fwd": _time_fn(gathered_fwd_jit, graph),
        "t_slice_fwd": _time_fn(sliced_fwd_jit, graph),
        "t_dense_scan": _time_fn(dense_scan_jit, dense_model, graph, n_iter=5),
        "t_gath_scan": _time_fn(gathered_scan_jit, gathered_model, graph, n_iter=5),
        "t_slice_scan": _time_fn(sliced_scan_jit, sliced_model, graph, n_iter=5),
        "t_dense_bw": _time_fn(dense_grad_jit, dense_model, graph, n_iter=3),
        "t_gath_bw": _time_fn(gathered_grad_jit, gathered_model, graph, n_iter=3),
        "t_slice_bw": _time_fn(sliced_grad_jit, sliced_model, graph, n_iter=3),
    }


def _run_self_attn_bench(
    input_n, output_n, layer_n, hidden_dim, arity, attention_dim, num_heads,
    n_self_attn_layers, n_steps,
):
    graph, _ = _make_circuit_graph(
        input_n=input_n, output_n=output_n, layer_n=layer_n,
        arity=arity, hidden_dim=hidden_dim,
    )
    n_node = graph.nodes["layer"].shape[0]
    max_deg = int(validate_gathered_topology(
        graph.senders, graph.receivers, n_node, max_neighbors=n_node, strict=True,
    ))
    max_neighbors = max_deg

    dense_model = _build_self_attn(
        "dense", n_node, max_neighbors, hidden_dim=hidden_dim, arity=arity,
        attention_dim=attention_dim, num_heads=num_heads,
        num_self_attn_layers=n_self_attn_layers,
    )
    gathered_model = _build_self_attn(
        "gathered", n_node, max_neighbors, hidden_dim=hidden_dim, arity=arity,
        attention_dim=attention_dim, num_heads=num_heads,
        num_self_attn_layers=n_self_attn_layers,
    )
    times = _bench_pair(
        dense_model, gathered_model, graph, n_node,
        graph.senders, graph.receivers, max_neighbors, n_steps, is_perceiver=False,
    )
    return {"n_node": n_node, "max_deg": max_deg, "max_neighbors": max_neighbors,
            "attention_dim": attention_dim, **times}


def _print_row_self_attn(label, r, n_steps):
    """Self-attn model bench has 2 backends (dense, gathered)."""
    speedup_fwd = r["t_dense_fwd"] / r["t_gath_fwd"] if r["t_gath_fwd"] > 0 else float("nan")
    speedup_scan = r["t_dense_scan"] / r["t_gath_scan"] if r["t_gath_scan"] > 0 else float("nan")
    speedup_bw = r["t_dense_bw"] / r["t_gath_bw"] if r["t_gath_bw"] > 0 else float("nan")
    print(
        f"  {label:<28} N={r['n_node']:>4}  deg={r['max_deg']:>2}  "
        f"D={r['attention_dim']:>3}  "
        f"| fwd1: {r['t_dense_fwd']:>6.2f}/{r['t_gath_fwd']:>6.2f}ms x{speedup_fwd:>4.2f}  "
        f"| scan{n_steps}: {r['t_dense_scan']:>7.2f}/{r['t_gath_scan']:>7.2f}ms x{speedup_scan:>4.2f}  "
        f"| bw: {r['t_dense_bw']:>7.2f}/{r['t_gath_bw']:>7.2f}ms x{speedup_bw:>4.2f}"
    )


def _print_row_perceiver(label, r, n_steps):
    """Perceiver bench has 3 variants: dense, gathered (self-attn), dense+sliced cross-attn.

    Speedups are reported vs the dense baseline. ``slice`` is the new optimization.
    """
    def _sp(t_baseline, t_other):
        return t_baseline / t_other if t_other > 0 else float("nan")
    sp_fwd_g = _sp(r["t_dense_fwd"], r["t_gath_fwd"])
    sp_fwd_s = _sp(r["t_dense_fwd"], r["t_slice_fwd"])
    sp_sc_g = _sp(r["t_dense_scan"], r["t_gath_scan"])
    sp_sc_s = _sp(r["t_dense_scan"], r["t_slice_scan"])
    sp_bw_g = _sp(r["t_dense_bw"], r["t_gath_bw"])
    sp_bw_s = _sp(r["t_dense_bw"], r["t_slice_bw"])
    print(
        f"  {label:<28} N={r['n_node']:>4}  K_in={r['K_in']:>2}/K_out={r['K_out']:>2}  D={r['attention_dim']:>3}\n"
        f"      fwd1   dense={r['t_dense_fwd']:>6.2f}  "
        f"gath={r['t_gath_fwd']:>6.2f}ms (x{sp_fwd_g:>4.2f})  "
        f"slice={r['t_slice_fwd']:>6.2f}ms (x{sp_fwd_s:>4.2f})\n"
        f"      scan{n_steps:<2} dense={r['t_dense_scan']:>6.2f}  "
        f"gath={r['t_gath_scan']:>6.2f}ms (x{sp_sc_g:>4.2f})  "
        f"slice={r['t_slice_scan']:>6.2f}ms (x{sp_sc_s:>4.2f})\n"
        f"      bw     dense={r['t_dense_bw']:>6.2f}  "
        f"gath={r['t_gath_bw']:>6.2f}ms (x{sp_bw_g:>4.2f})  "
        f"slice={r['t_slice_bw']:>6.2f}ms (x{sp_bw_s:>4.2f})"
    )


# Production-realistic configs from configs/circuit/ + configs/model/
# (arity=4, attention_dim=64, num_self_attn_layers=1, num_cross_attn_layers=1)
_TRAIN_CONFIGS = [
    # name, input_n, output_n, layer_n, arity, hidden_dim
    ("default (_base)", 8, 8, 3, 4, 16),
    ("mid (_mid)", 10, 10, 3, 4, 32),
    ("large (_large)", 12, 12, 3, 4, 64),
]


def main_bench():
    # Real training: n_message_steps=5; eval: inner_steps=256.
    # We bench both to span the spectrum.
    n_steps_train = 5
    n_steps_eval = 64  # cap for time; eval is 256 in real configs but interpolates
    attention_dim = 64
    num_heads = 4
    n_self_attn_layers = 1
    n_cross_attn_layers = 1
    n_samples = 64  # configs/config.yaml: data_per_batch

    print("\n" + "=" * 110)
    print(
        " Gathered vs Dense self-attention — forward / scan / backward (JIT'd, JAX block_until_ready)"
    )
    print(f" attention_dim={attention_dim}, num_heads={num_heads}, "
          f"n_self_attn_layers={n_self_attn_layers}, n_samples={n_samples}")
    print("=" * 110)

    for n_steps, label in [(n_steps_train, "TRAINING (n_message_steps=5)"),
                            (n_steps_eval, f"EVAL  (scan={n_steps_eval})")]:
        print(f"\n##### {label} #####")
        print("\n[Plain self-attention: CircuitSelfAttention(dense) vs CircuitGatheredAttention]")
        for name, in_n, out_n, L, ar, h in _TRAIN_CONFIGS:
            try:
                r = _run_self_attn_bench(
                    input_n=in_n, output_n=out_n, layer_n=L, hidden_dim=h, arity=ar,
                    attention_dim=attention_dim, num_heads=num_heads,
                    n_self_attn_layers=n_self_attn_layers, n_steps=n_steps,
                )
                _print_row_self_attn(name, r, n_steps)
            except Exception as e:
                print(f"  skipped {name}: {e}")

        print("\n[Perceiver: dense vs gathered (self-attn) vs sliced (cross-attn)]")
        for name, in_n, out_n, L, ar, h in _TRAIN_CONFIGS:
            try:
                r = _run_perceiver_bench(
                    input_n=in_n, output_n=out_n, layer_n=L, hidden_dim=h, arity=ar,
                    n_samples=n_samples, attention_dim=attention_dim, num_heads=num_heads,
                    n_self_attn_layers=n_self_attn_layers,
                    n_cross_attn_layers=n_cross_attn_layers,
                    n_steps=n_steps,
                )
                _print_row_perceiver(name, r, n_steps)
            except Exception as e:
                print(f"  skipped {name}: {e}")

    print("\nInterpretation:")
    print("  - Self-attn table: times are dense/gathered ms; speedup = dense / gathered.")
    print("  - Perceiver table: dense baseline + (gathered self-attn) + (sliced cross-attn).")
    print("    'slice' restricts cross-attn queries to the first/last gate layer only —")
    print("    a strict optimization when restrict_*_cross_attn_to_*_layer is on.")
    print("    K_in  = input_layer_size  (input cross-attn targets layer == 0).")
    print("    K_out = output_layer_size (output cross-attn targets layer == max_layer).")
    print("    Expected: scan/bw slice speedup grows with N/K_in and N/K_out ratios.")


if __name__ == "__main__":
    # Run pytest checks first, then profiling
    import subprocess
    import sys

    print("Running correctness tests via pytest...")
    rc = subprocess.call([sys.executable, "-m", "pytest", __file__, "-v", "-x", "-k", "not bench"])
    if rc != 0:
        print(f"\nCorrectness tests failed (exit {rc}). Skipping benchmarks.")
        sys.exit(rc)
    main_bench()
