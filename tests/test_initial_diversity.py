#!/usr/bin/env python3
"""
Test script to demonstrate the initial_diversity feature.

This script shows how the initial_diversity parameter controls
the number of different initial wirings in genetic mode.
"""

import jax
import jax.numpy as jp

from boolean_nca_cc.training.pool.pool import initialize_graph_pool


def test_initial_diversity():
    """Test the initial_diversity parameter with different values."""

    # Set up test parameters
    layer_sizes = [(3, 1), (4, 1)]  # 3 input nodes, 4 hidden nodes
    pool_size = 8
    rng = jax.random.PRNGKey(42)

    print("=== Testing Initial Diversity Feature ===")
    print(f"Pool size: {pool_size}")
    print(f"Layer sizes: {layer_sizes}")
    print()

    # Test different diversity values
    diversity_values = [1, 2, 4, 8, 16]

    for diversity in diversity_values:
        print(f"Testing initial_diversity = {diversity}")

        pool = initialize_graph_pool(
            rng=rng,
            layer_sizes=layer_sizes,
            pool_size=pool_size,
            input_n=3,
            arity=2,
            hidden_dim=16,
            wiring_mode="genetic",
            initial_diversity=diversity,
        )

        # Get the first layer wires (shape: [pool_size, arity, group_n])
        layer1_wires = pool.wires[0]  # Shape: [8, 2, 4]

        # Check how many unique wirings we have
        unique_wirings = []
        for i in range(pool_size):
            wiring = layer1_wires[i].flatten()

            # Check if this wiring is already in our unique list
            is_unique = True
            for unique_wiring in unique_wirings:
                if jp.array_equal(wiring, unique_wiring):
                    is_unique = False
                    break

            if is_unique:
                unique_wirings.append(wiring)

        actual_diversity = len(unique_wirings)
        expected_diversity = min(diversity, pool_size)

        # Get entropy-based diversity measure
        entropy_diversity = pool.get_wiring_diversity(layer_sizes)

        print(f"  Expected unique wirings: {expected_diversity}")
        print(f"  Actual unique wirings: {actual_diversity}")
        print(f"  Entropy diversity: {entropy_diversity:.4f}")

        # Show the distribution
        wiring_counts = {}
        for i in range(pool_size):
            wiring = tuple(layer1_wires[i].flatten().tolist())  # Convert to tuple of Python ints
            wiring_counts[wiring] = wiring_counts.get(wiring, 0) + 1

        print(f"  Distribution: {list(wiring_counts.values())}")

        # Verify correctness
        if actual_diversity == expected_diversity:
            print("  ✓ PASS")
        else:
            print("  ✗ FAIL")
        print()


def test_wiring_diversity_measurement():
    """Test the wiring diversity measurement with different initial_diversity values."""

    layer_sizes = [(3, 1), (4, 1)]
    pool_size = 100
    rng = jax.random.PRNGKey(42)

    print("=== Testing Wiring Diversity Measurement ===")
    print("Comparing discrete unique count vs continuous entropy measure")
    print(f"Pool size: {pool_size}")
    print()

    for diversity in [1, 5, 10, 25, 50, 100]:
        pool = initialize_graph_pool(
            rng=rng,
            layer_sizes=layer_sizes,
            pool_size=pool_size,
            input_n=3,
            arity=2,
            hidden_dim=16,
            wiring_mode="genetic",
            initial_diversity=diversity,
        )

        # Count unique wirings manually
        layer1_wires = pool.wires[0]
        unique_wirings = []
        for i in range(pool_size):
            wiring = layer1_wires[i].flatten()
            is_unique = True
            for unique_wiring in unique_wirings:
                if jp.array_equal(wiring, unique_wiring):
                    is_unique = False
                    break
            if is_unique:
                unique_wirings.append(wiring)

        actual_unique = len(unique_wirings)
        measured_diversity = pool.get_wiring_diversity(layer_sizes)
        expected_diversity = min(diversity, pool_size)

        print(
            f"initial_diversity={diversity:3d} -> expected={expected_diversity:3d}, actual_unique={actual_unique:3d}, entropy={measured_diversity:.4f}"
        )

    print()
    print("Note: Entropy diversity is normalized (0.0 = all identical, 1.0 = maximum diversity)")
    print("For comparison, random mode should give entropy ≈ 1.0")

    # Test random mode for comparison
    random_pool = initialize_graph_pool(
        rng=rng,
        layer_sizes=layer_sizes,
        pool_size=pool_size,
        input_n=3,
        arity=2,
        hidden_dim=16,
        wiring_mode="random",
    )
    random_diversity = random_pool.get_wiring_diversity(layer_sizes)
    print(f"Random mode entropy: {random_diversity:.4f}")

    # Test fixed mode for comparison
    fixed_pool = initialize_graph_pool(
        rng=rng,
        layer_sizes=layer_sizes,
        pool_size=pool_size,
        input_n=3,
        arity=2,
        hidden_dim=16,
        wiring_mode="fixed",
    )
    fixed_diversity = fixed_pool.get_wiring_diversity(layer_sizes)
    print(f"Fixed mode entropy: {fixed_diversity:.4f}")


if __name__ == "__main__":
    test_initial_diversity()
    test_wiring_diversity_measurement()
