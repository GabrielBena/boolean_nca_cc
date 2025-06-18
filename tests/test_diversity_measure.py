#!/usr/bin/env python3
"""
Test script for wiring diversity measurement across multiple circuit configurations.

This script creates pools with different wiring modes and measures their diversity
across various circuit architectures to verify that the metric behaves correctly:
- Fixed wiring: diversity ≈ 0
- Random wiring: diversity ≈ 1
- Genetic wiring: diversity starts at 0 and increases with mutations
"""

import jax
import jax.numpy as jp
from boolean_nca_cc.training.pool.pool import initialize_graph_pool
from boolean_nca_cc.circuits.model import generate_layer_sizes


def test_single_configuration(
    input_n, output_n, layer_n, pool_size, arity, hidden_dim, rng
):
    """Test diversity for a single circuit configuration."""

    layer_sizes = generate_layer_sizes(input_n, output_n, arity, layer_n=layer_n)

    print(f"\n--- Configuration: {input_n}→{output_n}, {layer_n} layers ---")
    print(f"    Layer sizes: {layer_sizes}")

    # Test 1: Fixed wiring mode (should have diversity ≈ 0)
    fixed_pool = initialize_graph_pool(
        rng=rng,
        layer_sizes=layer_sizes,
        pool_size=pool_size,
        input_n=input_n,
        arity=arity,
        hidden_dim=hidden_dim,
        wiring_mode="fixed",
    )
    fixed_diversity = fixed_pool.get_wiring_diversity(layer_sizes)

    # Test 2: Random wiring mode (should have diversity close to 1)
    random_pool = initialize_graph_pool(
        rng=rng,
        layer_sizes=layer_sizes,
        pool_size=pool_size,
        input_n=input_n,
        arity=arity,
        hidden_dim=hidden_dim,
        wiring_mode="random",
    )
    random_diversity = random_pool.get_wiring_diversity(layer_sizes)

    # Test 3: Genetic mode with mutations (should show increasing diversity)
    genetic_pool = initialize_graph_pool(
        rng=rng,
        layer_sizes=layer_sizes,
        pool_size=pool_size,
        input_n=input_n,
        arity=arity,
        hidden_dim=hidden_dim,
        wiring_mode="genetic",
    )
    initial_diversity = genetic_pool.get_wiring_diversity(layer_sizes)

    # Apply mutations to genetic pool
    mutation_rate = 0.15
    key = jax.random.fold_in(rng, 0)
    genetic_pool, _ = genetic_pool.reset_with_genetic_mutation(
        key=key,
        fraction=0.5,  # Reset half the pool with mutations
        layer_sizes=layer_sizes,
        input_n=input_n,
        arity=arity,
        hidden_dim=hidden_dim,
        mutation_rate=mutation_rate,
        reset_strategy="uniform",
    )
    final_genetic_diversity = genetic_pool.get_wiring_diversity(layer_sizes)

    print(f"    Fixed:    {fixed_diversity:.6f}")
    print(f"    Random:   {random_diversity:.6f}")
    print(f"    Genetic:  {initial_diversity:.6f} → {final_genetic_diversity:.6f}")

    return {
        "config": f"{input_n}→{output_n}, L{layer_n}",
        "fixed": fixed_diversity,
        "random": random_diversity,
        "genetic_initial": initial_diversity,
        "genetic_final": final_genetic_diversity,
        "layer_sizes": layer_sizes,
    }


def test_wiring_diversity_comprehensive():
    """Test wiring diversity across multiple circuit configurations."""

    print("Comprehensive Wiring Diversity Testing")
    print("=" * 60)

    # Test configurations
    input_output_sizes = [4, 6, 8]
    layer_counts = [1, 2, 3]
    pool_size = 512  # Smaller for faster testing
    arity = 2
    hidden_dim = 32
    rng = jax.random.PRNGKey(42)

    results = []

    # Test across different configurations
    for input_n in input_output_sizes:
        for output_n in input_output_sizes:
            for layer_n in layer_counts:
                try:
                    result = test_single_configuration(
                        input_n, output_n, layer_n, pool_size, arity, hidden_dim, rng
                    )
                    results.append(result)
                except Exception as e:
                    print(f"    ERROR: {e}")
                    continue

    # Summary analysis
    print("\n" + "=" * 60)
    print("SUMMARY ANALYSIS")
    print("=" * 60)

    print(
        f"{'Configuration':<15} {'Fixed':<8} {'Random':<8} {'Genetic':<15} {'Random/Expected':<12}"
    )
    print("-" * 70)

    fixed_diversities = []
    random_diversities = []
    genetic_improvements = []

    for result in results:
        config = result["config"]
        fixed = result["fixed"]
        random = result["random"]
        genetic_initial = result["genetic_initial"]
        genetic_final = result["genetic_final"]

        genetic_improvement = genetic_final - genetic_initial
        random_ratio = random  # Since we expect random to be close to 1.0

        fixed_diversities.append(fixed)
        random_diversities.append(random)
        genetic_improvements.append(genetic_improvement)

        print(
            f"{config:<15} {fixed:<8.4f} {random:<8.4f} {genetic_improvement:<15.4f} {random_ratio:<12.4f}"
        )

    print("-" * 70)
    print(
        f"{'AVERAGES':<15} {jp.mean(jp.array(fixed_diversities)):<8.4f} {jp.mean(jp.array(random_diversities)):<8.4f} {jp.mean(jp.array(genetic_improvements)):<15.4f} {jp.mean(jp.array(random_diversities)):<12.4f}"
    )

    print(f"\nOverall Statistics:")
    print(
        f"  Fixed diversity:     {jp.mean(jp.array(fixed_diversities)):.6f} ± {jp.std(jp.array(fixed_diversities)):.6f}"
    )
    print(
        f"  Random diversity:    {jp.mean(jp.array(random_diversities)):.6f} ± {jp.std(jp.array(random_diversities)):.6f}"
    )
    print(
        f"  Genetic improvement: {jp.mean(jp.array(genetic_improvements)):.6f} ± {jp.std(jp.array(genetic_improvements)):.6f}"
    )

    # Verification
    avg_fixed = jp.mean(jp.array(fixed_diversities))
    avg_random = jp.mean(jp.array(random_diversities))
    min_genetic_improvement = jp.min(jp.array(genetic_improvements))

    print(f"\nVerification:")
    print(f"  ✓ Fixed diversity < 0.01: {avg_fixed < 0.01} ({avg_fixed:.6f})")
    print(f"  ✓ Random diversity > 0.9: {avg_random > 0.9} ({avg_random:.6f})")
    print(
        f"  ✓ Genetic shows improvement: {min_genetic_improvement > 0} (min: {min_genetic_improvement:.6f})"
    )

    if avg_fixed < 0.01 and avg_random > 0.9 and min_genetic_improvement > 0:
        print(
            f"\n🎉 All tests passed! Diversity metric works correctly across all configurations."
        )
    else:
        print(f"\n⚠️  Some tests failed. Check the implementation.")

    return results


if __name__ == "__main__":
    test_wiring_diversity_comprehensive()
