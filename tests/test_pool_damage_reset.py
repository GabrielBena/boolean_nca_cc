import jax
import jax.numpy as jp

from boolean_nca_cc.training.pool.pool import (
    initialize_graph_pool,
)
from boolean_nca_cc.circuits.model import generate_layer_sizes


def _make_small_pool(pool_size: int = 16):
    key = jax.random.PRNGKey(0)
    layer_sizes = generate_layer_sizes(input_n=4, output_n=2, arity=2, layer_n=2)
    pool = initialize_graph_pool(
        rng=key,
        layer_sizes=layer_sizes,
        pool_size=pool_size,
        input_n=layer_sizes[0][0],
        arity=2,
        circuit_hidden_dim=8,
        loss_value=0.0,
    )
    return pool, layer_sizes


def test_reset_fraction_preserves_knockout_patterns_for_reset_indices():
    pool, layer_sizes = _make_small_pool(pool_size=32)

    # Set a known non-zero knockout pattern for the entire pool
    num_nodes = int(pool.knockout_patterns.shape[1])
    baseline_pattern = jp.ones((pool.size, num_nodes), dtype=jp.bool_)
    pool = pool.replace(knockout_patterns=baseline_pattern)

    # Prepare fresh graphs/wires/logits from a temporary pool
    fresh_pool, _ = _make_small_pool(pool_size=32)

    # Choose a reset key and reproduce the indices selected inside reset_fraction
    key = jax.random.PRNGKey(123)
    selection_key, _ = jax.random.split(key)
    reset_idxs, _ = pool.get_reset_indices(
        selection_key, fraction=0.25, reset_strategy="uniform", combined_weights=(0.5, 0.5)
    )

    # Execute reset using the same outer key
    pool_after, _ = pool.reset_fraction(
        key=key,
        fraction=0.25,
        new_graphs=fresh_pool.graphs,
        new_wires=fresh_pool.wires,
        new_logits=fresh_pool.logits,
        new_knockout_patterns=None,  # critical: do not alter KO patterns during reset
        reset_strategy="uniform",
        combined_weights=(0.5, 0.5),
    )

    # KO patterns must remain unchanged for exactly the reset indices
    assert jp.array_equal(
        pool_after.knockout_patterns[reset_idxs], baseline_pattern[reset_idxs]
    )


def test_damage_fraction_updates_only_knockout_patterns_and_not_graphs_or_logits():
    pool, layer_sizes = _make_small_pool(pool_size=32)

    # Take snapshots of immutable fields to compare later
    graphs_nodes_before = pool.graphs.nodes["logits"].copy()
    wires_before = jax.tree.map(lambda x: x.copy(), pool.wires)
    logits_before = jax.tree.map(lambda x: x.copy(), pool.logits)
    reset_counter_before = pool.reset_counter.copy()
    knockout_before = pool.knockout_patterns.copy()

    key = jax.random.PRNGKey(7)
    fraction = 0.3
    damage_prob = 3.0  # must be >0 to actually flip some bits

    pool_after, damaged_idxs = pool.damage_fraction(
        key=key,
        fraction=fraction,
        layer_sizes=tuple(layer_sizes),
        damage_prob=damage_prob,
        selection_strategy="uniform",
        combined_weights=(0.5, 0.5),
        knockout_vocabulary=None,
    )

    # Selection count semantics: at least 1, round(fraction * size)
    expected = int(jp.maximum(1, jp.round(fraction * pool.size)))
    # Ensure uniqueness
    assert damaged_idxs.shape[0] == expected
    assert jp.unique(damaged_idxs).shape[0] == damaged_idxs.shape[0]

    # Graphs/wires/logits must be unchanged everywhere (including damaged indices)
    assert jp.array_equal(pool_after.graphs.nodes["logits"], graphs_nodes_before)
    jax.tree.map(lambda a, b: jp.testing.assert_array_equal(a, b), pool_after.wires, wires_before)
    jax.tree.map(lambda a, b: jp.testing.assert_array_equal(a, b), pool_after.logits, logits_before)

    # Reset counter must be unchanged
    assert jp.array_equal(pool_after.reset_counter, reset_counter_before)

    # Knockout patterns must differ at the damaged indices or remain same elsewhere
    # At minimum, ensure that only the rows at damaged indices are allowed to change
    changed_mask = jp.any(pool_after.knockout_patterns != knockout_before, axis=1)
    # All changed rows must be a subset of damaged indices
    changed_indices = jp.where(changed_mask, size=pool.size)[0]
    # Every changed index must be in damaged_idxs
    assert jp.all(jp.isin(changed_indices, damaged_idxs))


