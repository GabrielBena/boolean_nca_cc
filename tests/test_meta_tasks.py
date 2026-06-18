"""
Tests for the per-circuit meta-learning task stack.

Covers:
- Task samplers (k-junta correctness, marginals, determinism; library padding).
- Init-logits dispatch (soft_wires / zeros / random — semantic correctness).
- GraphPool y_task plumbing (init / update / sample / reset_fraction).

End-to-end ``train_model`` integration is covered separately by the smoke
checks invoked from the development workflow; here we keep tests narrow and
fast so they fit in the regular CI loop.
"""

import jax
import jax.numpy as jp
import pytest

from boolean_nca_cc.circuits.model import (
    INIT_LOGITS,
    gen_circuit,
    make_random_logits,
    make_zero_logits,
)
from boolean_nca_cc.tasks import (
    TASK_SAMPLERS,
    build_task_x,
    sample_arith_family_y,
    sample_k_junta_y,
    sample_library_batch,
    sample_task_batch,
)
from boolean_nca_cc.tasks.samplers import build_task_x as _build_task_x_alias
from boolean_nca_cc.training.pool.pool import (
    GraphPool,
    initialize_graph_pool,
)


# ---------------------------------------------------------------------------
# build_task_x
# ---------------------------------------------------------------------------


def test_build_task_x_shape_and_enumeration():
    for input_n in (1, 2, 4, 8):
        case_n = 1 << input_n
        x = build_task_x(input_n)
        assert x.shape == (case_n, input_n)
        assert x.dtype == jp.float32
        # Row i should be the binary representation of i (LSB first).
        # Sample a few representative indices, clamped to [0, case_n - 1].
        for i in {0, 1, min(2, case_n - 1), case_n - 1}:
            bits = [(i >> b) & 1 for b in range(input_n)]
            assert tuple(int(b) for b in x[i].tolist()) == tuple(bits), (
                f"input_n={input_n}, i={i}: x[{i}]={x[i].tolist()} vs {bits}"
            )


def test_build_task_x_alias():
    # Verify the module-level public symbol matches the underlying impl.
    assert _build_task_x_alias is build_task_x


# ---------------------------------------------------------------------------
# k-junta sampler
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("k", [1, 2, 3, 4])
def test_k_junta_balanced_marginals_exactly_half(k):
    """``balanced=True`` LUTs have exactly 2^(k-1) ones → marginal exactly 0.5."""
    y = sample_k_junta_y(
        jax.random.PRNGKey(0), input_n=6, output_n=3, k=k, balanced=True
    )
    marginals = y.mean(axis=0)
    assert jp.allclose(marginals, 0.5), f"marginals={marginals} (k={k})"


def test_k_junta_unbalanced_outputs_are_boolean():
    """``balanced=False`` LUTs can drift (degenerate all-ones / all-zeros LUTs
    are non-trivially likely at small k), so we only check the basic invariants
    here: every output value is in {0, 1} and the marginal averaged over many
    output bits is in the right neighborhood."""
    y = sample_k_junta_y(
        jax.random.PRNGKey(7), input_n=6, output_n=64, k=2, balanced=False
    )
    # All entries are exact 0/1.
    assert jp.all((y == 0.0) | (y == 1.0))
    # With 64 output bits, the *mean* marginal across bits should be close
    # to 0.5 even though individual bits can degenerate.
    mean_marginal = float(y.mean())
    assert 0.35 < mean_marginal < 0.65, f"mean marginal {mean_marginal}"


def test_k_junta_determinism():
    y_a = sample_k_junta_y(jax.random.PRNGKey(42), input_n=6, output_n=2, k=2)
    y_b = sample_k_junta_y(jax.random.PRNGKey(42), input_n=6, output_n=2, k=2)
    assert jp.array_equal(y_a, y_b)


def test_k_junta_independence_across_keys():
    y_a = sample_k_junta_y(jax.random.PRNGKey(1), input_n=6, output_n=4, k=2)
    y_b = sample_k_junta_y(jax.random.PRNGKey(2), input_n=6, output_n=4, k=2)
    assert not jp.array_equal(y_a, y_b)


def test_k_junta_actually_a_junta():
    """For a given output bit, flipping non-selected inputs must leave y unchanged.

    Strategy: sample a known seed, then for each output bit, search for a
    subset such that y depends only on that subset's bits. We can identify
    irrelevant bits by checking pairs of inputs that differ only in that bit
    — y must be identical for both.
    """
    input_n = 5
    output_n = 3
    k = 2
    y = sample_k_junta_y(jax.random.PRNGKey(3), input_n, output_n, k=k, balanced=True)
    # y has shape [2^input_n, output_n]. For each output bit, count the
    # number of input bits that are 'relevant': flipping bit b changes y[:, c]
    # for at least one input pattern.
    case_n = 1 << input_n
    for c in range(output_n):
        relevant_bits = []
        for b in range(input_n):
            mask = 1 << b
            other = jp.arange(case_n) ^ mask
            # If y is independent of bit b, y[i] == y[i ^ (1<<b)] for all i.
            changed = jp.any(y[:, c] != y[other, c])
            if bool(changed):
                relevant_bits.append(b)
        assert len(relevant_bits) <= k, (
            f"output bit {c}: depends on {relevant_bits} (more than k={k})"
        )


def test_k_junta_rejects_invalid_k():
    with pytest.raises(ValueError, match="k="):
        sample_k_junta_y(jax.random.PRNGKey(0), input_n=4, output_n=2, k=0)
    with pytest.raises(ValueError, match="k="):
        sample_k_junta_y(jax.random.PRNGKey(0), input_n=4, output_n=2, k=5)


# ---------------------------------------------------------------------------
# Library sampler
# ---------------------------------------------------------------------------


def test_library_sampler_shape_and_padding():
    """Library tasks with native output width < output_n get zero-padded."""
    pool_size = 6
    input_n = 6  # case_n = 64
    output_n = 8
    y, idx = sample_library_batch(
        jax.random.PRNGKey(0), pool_size, input_n, output_n,
        task_names=["parity", "add", "reverse"],
    )
    assert y.shape == (pool_size, 1 << input_n, output_n)
    assert idx.shape == (pool_size,)
    assert int(idx.min()) >= 0 and int(idx.max()) < 3
    # parity has 1 output bit; in those slots, bits 1..7 must be exactly zero.
    parity_slots = jp.where(idx == 0)[0]
    if parity_slots.size > 0:
        assert jp.all(y[parity_slots, :, 1:] == 0), "parity padding broken"


def test_library_sampler_rejects_empty_task_list():
    with pytest.raises(ValueError, match="task_names"):
        sample_library_batch(jax.random.PRNGKey(0), 4, 4, 4, task_names=[])


# ---------------------------------------------------------------------------
# Registry / dispatch
# ---------------------------------------------------------------------------


def test_registry_dispatch_k_junta():
    y = sample_task_batch(
        jax.random.PRNGKey(0), pool_size=4, input_n=5, output_n=3,
        cfg={"name": "k_junta", "k": 2, "balanced": True},
    )
    assert y.shape == (4, 1 << 5, 3)


def test_registry_dispatch_library():
    y = sample_task_batch(
        jax.random.PRNGKey(0), pool_size=4, input_n=6, output_n=6,
        cfg={"name": "library", "task_names": ["add", "parity"]},
    )
    assert y.shape == (4, 1 << 6, 6)


def test_registry_rejects_unknown_sampler():
    with pytest.raises(ValueError, match="unknown task sampler"):
        sample_task_batch(
            jax.random.PRNGKey(0), pool_size=2, input_n=4, output_n=2,
            cfg={"name": "nonexistent_sampler"},
        )


def test_registry_requires_name():
    with pytest.raises(ValueError, match="missing 'name'"):
        sample_task_batch(
            jax.random.PRNGKey(0), pool_size=2, input_n=4, output_n=2,
            cfg={"k": 2},
        )


def test_registry_contains_expected_samplers():
    assert "k_junta" in TASK_SAMPLERS
    assert "arith_family" in TASK_SAMPLERS
    assert "library" in TASK_SAMPLERS


# ---------------------------------------------------------------------------
# Arithmetic-family sampler (the "structured diversity" arm of "sweep both")
# ---------------------------------------------------------------------------


def _library_add_padded(input_n: int, output_n: int) -> jp.ndarray:
    """The canonical library ``add`` truth table, padded to output_n (the target
    the π-identity member of the arith family must reproduce bit-for-bit)."""
    from boolean_nca_cc.circuits.tasks import get_task_data

    (_x, y), _split, _total = get_task_data("add", 1 << input_n, input_bits=input_n)
    y = jp.asarray(y, dtype=jp.float32)
    if y.ndim == 1:
        y = y[:, None]
    if y.shape[1] < output_n:
        y = jp.concatenate(
            [y, jp.zeros((y.shape[0], output_n - y.shape[1]), dtype=jp.float32)], axis=1
        )
    elif y.shape[1] > output_n:
        y = y[:, :output_n]
    return y


@pytest.mark.parametrize("input_n,output_n", [(8, 8), (10, 12), (12, 12)])
def test_arith_family_pi_identity_equals_library_add(input_n, output_n):
    """π=identity + offset=0 must be BIT-IDENTICAL to the library ``add`` task —
    so OOD-``add`` is exactly the identity member of this family."""
    y_id = sample_arith_family_y(
        jax.random.PRNGKey(0), input_n, output_n, permute_inputs=False, max_offset=0
    )
    y_add = _library_add_padded(input_n, output_n)
    assert y_id.shape == (1 << input_n, output_n)
    assert jp.array_equal(y_id, y_add), "π-identity arith_family != library add"


def test_arith_family_permutation_is_diverse_and_boolean():
    input_n, output_n = 10, 12
    y_add = _library_add_padded(input_n, output_n)
    ys = [
        sample_arith_family_y(jax.random.PRNGKey(s), input_n, output_n, permute_inputs=True)
        for s in range(4)
    ]
    for y in ys:
        assert jp.all((y == 0.0) | (y == 1.0)), "arith_family output not boolean"
        # A random permutation almost surely differs from canonical add.
        assert jp.any(y != y_add), "permuted member coincidentally equals add"
    # Distinct seeds give distinct tasks.
    assert not jp.array_equal(ys[0], ys[1])


def test_arith_family_offset_shifts_function():
    input_n, output_n = 8, 8
    y0 = sample_arith_family_y(
        jax.random.PRNGKey(3), input_n, output_n, permute_inputs=False, max_offset=0
    )
    # With identity perm, a non-zero offset must change at least one entry for
    # SOME key (sum != sum + c). Scan a few keys to avoid a fluke draw of c=0.
    changed = any(
        jp.any(
            sample_arith_family_y(
                jax.random.PRNGKey(s), input_n, output_n, permute_inputs=False, max_offset=7
            )
            != y0
        )
        for s in range(5)
    )
    assert changed, "max_offset>0 never changed the function across 5 keys"


def test_registry_dispatch_arith_family():
    y = sample_task_batch(
        jax.random.PRNGKey(0), pool_size=4, input_n=8, output_n=8,
        cfg={"name": "arith_family", "permute_inputs": True, "max_offset": 0},
    )
    assert y.shape == (4, 1 << 8, 8)
    assert jp.all((y == 0.0) | (y == 1.0))


def _library_padded(subject, input_n, output_n):
    from boolean_nca_cc.circuits.tasks import get_task_data

    (_x, y), _split, _total = get_task_data(subject, 1 << input_n, input_bits=input_n)
    y = jp.asarray(y, dtype=jp.float32)
    if y.ndim == 1:
        y = y[:, None]
    if y.shape[1] < output_n:
        y = jp.concatenate([y, jp.zeros((y.shape[0], output_n - y.shape[1]), dtype=jp.float32)], 1)
    return y[:, :output_n]


@pytest.mark.parametrize("op", ["add", "sub"])
def test_arith_family_op_pi_identity_matches_library(op):
    """π=identity arith_family member is bit-identical to the corresponding library
    op (add or sub) — so OOD-add is within-family and sub is a genuine held-out op."""
    input_n, output_n = 12, 12
    y_id = sample_arith_family_y(
        jax.random.PRNGKey(0), input_n, output_n, op=op, permute_inputs=False, max_offset=0
    )
    assert jp.array_equal(y_id, _library_padded(op, input_n, output_n))


def test_arith_family_ops_differ():
    input_n, output_n = 12, 12
    y_add = sample_arith_family_y(jax.random.PRNGKey(0), input_n, output_n, op="add", permute_inputs=False)
    y_sub = sample_arith_family_y(jax.random.PRNGKey(0), input_n, output_n, op="sub", permute_inputs=False)
    assert jp.any(y_add != y_sub), "add and sub families coincide"


def test_sub_in_task_library():
    from boolean_nca_cc.circuits.tasks import TASKS

    assert "sub" in TASKS  # genuine held-out arithmetic OOD relative to add training


# ---------------------------------------------------------------------------
# Init-logits dispatch
# ---------------------------------------------------------------------------


def test_init_logits_registry_keys():
    assert set(INIT_LOGITS) == {"soft_wires", "zeros", "random"}


def test_zero_logits_produce_constant_half():
    """make_zero_logits → sigmoid(0) = 0.5 — verify shape and value."""
    lgt = make_zero_logits(gate_n=8, arity=2, group_size=1, key=jax.random.PRNGKey(0))
    assert lgt.shape == (8, 1, 4)
    assert jp.all(lgt == 0.0)


def test_random_logits_scale_respected():
    """Larger random_init_scale → larger std on the logits."""
    key = jax.random.PRNGKey(0)
    lgt_small = make_random_logits(
        gate_n=64, arity=2, group_size=1, key=key, random_init_scale=0.1
    )
    lgt_large = make_random_logits(
        gate_n=64, arity=2, group_size=1, key=key, random_init_scale=10.0
    )
    assert float(lgt_small.std()) < float(lgt_large.std())


def test_gen_circuit_dispatches_init_logits():
    layer_sizes = [(4, 1), (8, 1), (4, 1)]
    _, logits_soft = gen_circuit(
        jax.random.PRNGKey(0), jax.random.PRNGKey(1),
        layer_sizes, arity=2, init_logits="soft_wires", noise_scale=0.0,
    )
    _, logits_zero = gen_circuit(
        jax.random.PRNGKey(0), jax.random.PRNGKey(1),
        layer_sizes, arity=2, init_logits="zeros",
    )
    _, logits_rand = gen_circuit(
        jax.random.PRNGKey(0), jax.random.PRNGKey(1),
        layer_sizes, arity=2, init_logits="random", random_init_scale=1.0,
    )
    assert jp.all(logits_zero[0] == 0.0)
    # soft_wires + noise_scale=0 → exact passthrough logits ∈ {-3, 3}.
    assert jp.all(jp.abs(jp.abs(logits_soft[0]) - 3.0) < 1e-5)
    # random init: not zero, not the soft-wire structure.
    assert not jp.all(logits_rand[0] == 0.0)
    assert not jp.allclose(jp.abs(logits_rand[0]), 3.0)


def test_gen_circuit_rejects_unknown_init():
    layer_sizes = [(4, 1), (8, 1)]
    with pytest.raises(ValueError, match="init_logits="):
        gen_circuit(
            jax.random.PRNGKey(0), jax.random.PRNGKey(1),
            layer_sizes, arity=2, init_logits="bogus",
        )


# ---------------------------------------------------------------------------
# Pool plumbing
# ---------------------------------------------------------------------------


def _make_pool(pool_size, layer_sizes, input_n, *, y_task=None):
    return initialize_graph_pool(
        wires_key=jax.random.PRNGKey(0),
        logits_key=jax.random.PRNGKey(1),
        layer_sizes=layer_sizes, pool_size=pool_size,
        input_n=input_n, arity=2, circuit_hidden_dim=8,
        wiring_mode="random",
        init_logits="random", random_init_scale=1.0,
        y_task=y_task,
    )


def test_pool_legacy_path_has_no_y_task():
    layer_sizes = [(4, 1), (8, 1), (4, 1)]
    pool = _make_pool(8, layer_sizes, input_n=4)
    assert pool.y_task is None


def test_pool_per_circuit_path_carries_y_task():
    pool_size = 8
    input_n, output_n = 4, 4
    layer_sizes = [(input_n, 1), (8, 1), (output_n, 1)]
    y_task = sample_task_batch(
        jax.random.PRNGKey(99), pool_size, input_n, output_n,
        cfg={"name": "k_junta", "k": 2, "balanced": True},
    )
    pool = _make_pool(pool_size, layer_sizes, input_n, y_task=y_task)
    assert pool.y_task is not None
    assert pool.y_task.shape == (pool_size, 1 << input_n, output_n)
    assert jp.array_equal(pool.y_task, y_task)


def test_pool_rejects_wrong_y_task_leading_dim():
    layer_sizes = [(4, 1), (8, 1), (4, 1)]
    bad_y = jp.zeros((5, 16, 4))  # leading dim != pool_size
    with pytest.raises(ValueError, match="leading dim"):
        _make_pool(8, layer_sizes, input_n=4, y_task=bad_y)


def test_pool_sample_indexes_y_task():
    pool_size = 8
    input_n, output_n = 4, 4
    layer_sizes = [(input_n, 1), (8, 1), (output_n, 1)]
    y_task = sample_task_batch(
        jax.random.PRNGKey(99), pool_size, input_n, output_n,
        cfg={"name": "k_junta", "k": 2, "balanced": True},
    )
    pool = _make_pool(pool_size, layer_sizes, input_n, y_task=y_task)
    idxs, _, _, _, _ = pool.sample(jax.random.PRNGKey(0), batch_size=3)
    sliced = pool.y_task[idxs]
    assert sliced.shape == (3, 1 << input_n, output_n)
    # Sanity: sliced[i] must equal y_task[idxs[i]].
    for i in range(3):
        assert jp.array_equal(sliced[i], y_task[int(idxs[i])])


def test_pool_reset_fraction_replaces_y_task_only_for_reset_slots():
    pool_size = 16
    input_n, output_n = 4, 4
    layer_sizes = [(input_n, 1), (8, 1), (output_n, 1)]
    y_task = sample_task_batch(
        jax.random.PRNGKey(99), pool_size, input_n, output_n,
        cfg={"name": "k_junta", "k": 2, "balanced": True},
    )
    pool = _make_pool(pool_size, layer_sizes, input_n, y_task=y_task)
    original_y = pool.y_task.copy()

    fresh_y = sample_task_batch(
        jax.random.PRNGKey(123), pool_size, input_n, output_n,
        cfg={"name": "k_junta", "k": 3, "balanced": True},
    )
    fresh_pool = _make_pool(pool_size, layer_sizes, input_n, y_task=fresh_y)

    reset_pool, _ = pool.reset_fraction(
        key=jax.random.PRNGKey(0),
        fraction=0.5,
        new_graphs=fresh_pool.graphs,
        new_wires=fresh_pool.wires,
        new_logits=fresh_pool.logits,
        new_gate_masks=fresh_pool.gate_masks,
        new_y_task=fresh_pool.y_task,
    )

    # Exactly fraction*pool_size slots should have changed.
    changed = jp.any(reset_pool.y_task != original_y, axis=(1, 2))
    n_changed = int(changed.sum())
    assert n_changed == 8, f"expected 8 changed slots, got {n_changed}"


# ---------------------------------------------------------------------------
# Unified inner loop (online↔batch continuum) — run_inner_loop / eval
# ---------------------------------------------------------------------------


def _tiny_circuit_model_graph(input_n=4, output_n=4, hidden_dim=16, seed=0):
    """Build a tiny (wires, logits, graph, model, layer_sizes) for online-scan tests."""
    from flax import nnx

    from boolean_nca_cc.circuits.model import gen_circuit, generate_layer_sizes
    from boolean_nca_cc.models import CircuitGNN
    from boolean_nca_cc.utils.graph_builder import build_graph

    arity = 2
    layer_sizes = generate_layer_sizes(input_n, output_n, arity, layer_n=2)
    out_n = layer_sizes[-1][0]
    k1, k2, k3 = jax.random.split(jax.random.PRNGKey(seed), 3)
    wires, logits = gen_circuit(k1, k2, layer_sizes, arity=arity, init_logits="random")
    graph = build_graph(logits, wires, input_n, arity, hidden_dim)
    model = CircuitGNN(
        circuit_hidden_dim=hidden_dim,
        attention_dim=32,
        arity=arity,
        message_passing=True,
        use_attention=False,
        rngs=nnx.Rngs(params=k3),
        use_node_loss=False,
        use_layer_PE=False,
        use_intra_layer_PE=False,
    )
    return wires, logits, graph, model, layer_sizes, input_n, out_n


def test_inner_loop_scan_shapes_and_state_preserved():
    from boolean_nca_cc.circuits.train import LOSS_L4
    from boolean_nca_cc.training.evaluation import run_inner_loop

    wires, logits, graph, model, layer_sizes, input_n, out_n = _tiny_circuit_model_graph()
    x_task = build_task_x(input_n)
    y_task = sample_k_junta_y(jax.random.PRNGKey(1), input_n, out_n, k=2)

    N = 8
    final_graph, round_losses, final_logits, round_aux = run_inner_loop(
        model=model,
        graph=graph,
        logits_original_shapes=[lg.shape for lg in logits],
        wires=wires,
        x_task=x_task,
        y_task=y_task,
        loss_cfg=LOSS_L4,
        layer_sizes=layer_sizes,
        n_rounds=N,
        steps_per_round=2,
        window_size=1,
        data_mode="stream",
        scan_key=jax.random.PRNGKey(2),
    )
    # Round-loss curve has one entry per round.
    assert round_losses.shape == (N,)
    assert round_aux["hard_accuracy"].shape == (N,)
    assert round_aux["predictions"].shape[0] == N  # [N, W, output_n]
    # State shape is preserved across rounds (circuit identity intact).
    assert final_graph.nodes["logits"].shape == graph.nodes["logits"].shape
    assert len(final_logits) == len(logits)
    assert bool(jp.all(jp.isfinite(round_losses)))


def test_inner_loop_scan_grad_flows():
    """BPTT through the unified inner loop yields finite, non-trivial grads w.r.t. params."""
    from flax import nnx

    from boolean_nca_cc.circuits.train import LOSS_L4
    from boolean_nca_cc.training.evaluation import run_inner_loop

    wires, logits, graph, model, layer_sizes, input_n, out_n = _tiny_circuit_model_graph()
    x_task = build_task_x(input_n)
    y_task = sample_k_junta_y(jax.random.PRNGKey(1), input_n, out_n, k=2)

    def loss_fn(m):
        _g, round_losses, _l, _aux = run_inner_loop(
            model=m,
            graph=graph,
            logits_original_shapes=[lg.shape for lg in logits],
            wires=wires,
            x_task=x_task,
            y_task=y_task,
            loss_cfg=LOSS_L4,
            layer_sizes=layer_sizes,
            n_rounds=6,
            steps_per_round=1,
            window_size=1,
            data_mode="stream",
            scan_key=jax.random.PRNGKey(0),
        )
        return round_losses.mean()

    grads = nnx.grad(loss_fn)(model)
    leaves = jax.tree.leaves(grads)
    assert leaves, "no gradient leaves"
    assert all(bool(jp.all(jp.isfinite(g))) for g in leaves), "non-finite gradients"
    assert any(float(jp.sum(jp.abs(g))) > 0.0 for g in leaves), "all gradients zero"


def test_batch_preset_matches_scan():
    """REGRESSION: run_inner_loop in the BATCH preset (data_mode='fixed', full-table window,
    steps_per_round=1) reproduces run_model_scan_with_loss's no-damage path — identical final
    logits and equal per-round/per-step losses. The guarantee that folding batch into the
    unified continuum changed nothing for the batch regime (both paths share the documented
    sum→mean loss normalization)."""
    from boolean_nca_cc.circuits.train import LOSS_BCE
    from boolean_nca_cc.training.evaluation import (
        run_inner_loop,
        run_model_scan_with_loss,
    )

    wires, logits, graph, model, layer_sizes, input_n, out_n = _tiny_circuit_model_graph()
    x_task = build_task_x(input_n)
    case_n = x_task.shape[0]
    y_task = sample_k_junta_y(jax.random.PRNGKey(3), input_n, out_n, k=2)
    T = 5
    shapes = [lg.shape for lg in logits]

    # Legacy batch scan (no damage, full table).
    _fg_b, step_outputs = run_model_scan_with_loss(
        model=model,
        graph=graph,
        num_steps=T,
        logits_original_shapes=shapes,
        wires=wires,
        x_data=x_task,
        y_data=y_task,
        loss_cfg=LOSS_BCE,
        layer_sizes=layer_sizes,
        data_fraction=1.0,
        scan_key=jax.random.PRNGKey(9),
    )
    batch_losses = step_outputs[1]  # [T]
    batch_final_logits = jax.tree.map(lambda a: a[-1], step_outputs[2])

    # Unified loop, BATCH preset: fixed full-table window, 1 step/round, T rounds.
    _fg_u, round_losses, uni_final_logits, _aux = run_inner_loop(
        model=model,
        graph=graph,
        logits_original_shapes=shapes,
        wires=wires,
        x_task=x_task,
        y_task=y_task,
        loss_cfg=LOSS_BCE,
        layer_sizes=layer_sizes,
        n_rounds=T,
        steps_per_round=1,
        window_size=case_n,  # full table => deterministic arange => == data_fraction=1.0
        data_mode="fixed",
        scan_key=jax.random.PRNGKey(9),
    )
    # Per-step losses match (the scan records loss AFTER each model step; so does the loop).
    assert jp.allclose(batch_losses, round_losses, atol=1e-5), (
        f"batch vs unified losses differ: {batch_losses} vs {round_losses}"
    )
    # Final adapted logits match (same trajectory).
    for lb, lu in zip(batch_final_logits, uni_final_logits, strict=True):
        assert jp.allclose(lb, lu, atol=1e-5), "batch vs unified final logits differ"


def test_inner_loop_eval_full_table_readout():
    """evaluate_model_inner_loop_batched returns a full-table hard-acc adjudicator
    (a 1-element ``hard_accuracy`` list) plus per-round adaptation curves."""
    from boolean_nca_cc.circuits.train import LOSS_L4
    from boolean_nca_cc.training.evaluation import evaluate_model_inner_loop_batched

    wires, logits, graph, model, layer_sizes, input_n, out_n = _tiny_circuit_model_graph()
    x_task = build_task_x(input_n)
    y_task = sample_k_junta_y(jax.random.PRNGKey(5), input_n, out_n, k=2)  # 2D shared

    B = 4
    batch_wires = [jp.broadcast_to(w, (B,) + w.shape) for w in wires]
    batch_logits = [jp.broadcast_to(lg, (B,) + lg.shape) for lg in logits]

    N = 6
    # chunk_size=2 exercises the OOM-safe chunked path (vmap-within-chunk + concat).
    _graphs, metrics = evaluate_model_inner_loop_batched(
        model=model,
        batch_wires=batch_wires,
        batch_logits=batch_logits,
        x_task=x_task,
        y_task=y_task,
        input_n=input_n,
        arity=2,
        circuit_hidden_dim=16,
        loss_cfg=LOSS_L4,
        layer_sizes=layer_sizes,
        n_rounds=N,
        steps_per_round=1,
        window_size=1,
        data_mode="stream",
        scan_key=jax.random.PRNGKey(7),
        chunk_size=2,
    )
    assert len(metrics["hard_accuracy"]) == 1
    ha = metrics["hard_accuracy"][-1]
    assert 0.0 <= ha <= 1.0, f"full-table hard-acc out of range: {ha}"
    assert len(metrics["round_loss"]) == N
    assert "final_full_table_hard_accuracy" in metrics


def test_resolve_n_rounds_curriculum():
    from boolean_nca_cc.training.train_loop import _resolve_n_rounds

    # No schedule → constant base.
    assert _resolve_n_rounds(7, 100, 64, None) == 64
    # Stepped schedule: latest step whose fraction <= epoch/total.
    sched = [[0.0, 8], [0.5, 32], [0.8, 64]]
    assert _resolve_n_rounds(0, 100, 64, sched) == 8
    assert _resolve_n_rounds(49, 100, 64, sched) == 8
    assert _resolve_n_rounds(60, 100, 64, sched) == 32
    assert _resolve_n_rounds(95, 100, 64, sched) == 64
    # Unsorted schedule still resolves correctly.
    assert _resolve_n_rounds(60, 100, 64, [[0.8, 64], [0.0, 8], [0.5, 32]]) == 32


def test_native_output_width():
    from boolean_nca_cc.training.train_loop import _native_output_width

    # input_n=12: add/sub = 12//2+1 = 7, reverse = input_n = 12 (honest anchor),
    # parity = 1, binary_multiply = 8 (library default). All clamped to output_n.
    assert _native_output_width("add", 12, 12) == 7
    assert _native_output_width("sub", 12, 12) == 7
    assert _native_output_width("reverse", 12, 12) == 12
    assert _native_output_width("parity", 12, 12) == 1
    assert _native_output_width("add", 12, 5) == 5  # clamped to output_n


def test_inner_loop_eval_padding_corrected_metric():
    """score_output_bits=output_n reproduces the full metric; a smaller value scores
    only the leading (meaningful) bits — the padding correction."""
    from boolean_nca_cc.circuits.train import LOSS_L4
    from boolean_nca_cc.training.evaluation import evaluate_model_inner_loop_batched

    wires, logits, graph, model, layer_sizes, input_n, out_n = _tiny_circuit_model_graph()
    x_task = build_task_x(input_n)
    # A padded target: only the first bit is "meaningful", rest are 0 (parity-like padding).
    y_task = sample_k_junta_y(jax.random.PRNGKey(5), input_n, out_n, k=2).at[:, 1:].set(0.0)

    B = 4
    bw = [jp.broadcast_to(w, (B,) + w.shape) for w in wires]
    bl = [jp.broadcast_to(lg, (B,) + lg.shape) for lg in logits]
    common = dict(
        model=model, batch_wires=bw, batch_logits=bl, x_task=x_task, y_task=y_task,
        input_n=input_n, arity=2, circuit_hidden_dim=16, loss_cfg=LOSS_L4,
        layer_sizes=layer_sizes, n_rounds=5, data_mode="stream", scan_key=jax.random.PRNGKey(7),
    )
    _, m_full = evaluate_model_inner_loop_batched(**common, score_output_bits=None)
    _, m_outn = evaluate_model_inner_loop_batched(**common, score_output_bits=out_n)
    _, m_bit0 = evaluate_model_inner_loop_batched(**common, score_output_bits=1)
    # score_output_bits == output_n must reproduce the full (None) metric.
    assert abs(m_full["hard_accuracy"][-1] - m_outn["hard_accuracy"][-1]) < 1e-5
    # All in range; the 1-bit score is a valid (generally different) number.
    for m in (m_full, m_bit0):
        assert 0.0 <= m["hard_accuracy"][-1] <= 1.0
