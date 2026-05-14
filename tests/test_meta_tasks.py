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
    assert "library" in TASK_SAMPLERS


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
