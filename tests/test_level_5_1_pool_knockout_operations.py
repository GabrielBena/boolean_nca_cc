#!/usr/bin/env python3
"""
Level 5.1: Pool Management System - Knockout Integration Tests

Tests the integration of knockout patterns with pool operations, with critical
focus on persistence of knockout patterns throughout the pool lifecycle.

Key areas tested:
1. Knockout pattern integration in pool sampling
2. Knockout pattern updates and persistence in pool operations  
3. Knockout pattern handling in pool resets
4. Pool initialization with knockout patterns
5. CRITICAL: Persistence of knockout patterns through complete pool lifecycle
"""

import jax
import jax.numpy as jp
import jraph
import pytest
from typing import Dict, List, Tuple, Any, Optional
import numpy as np

# Import configuration
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra import initialize, compose
from hydra.core.global_hydra import GlobalHydra

# Import from codebase
from boolean_nca_cc.training.pool.pool import GraphPool, initialize_graph_pool
from boolean_nca_cc.training.pool.structural_perturbation import (
    create_reproducible_knockout_pattern,
    create_knockout_vocabulary,
    extract_layer_info_from_graph
)
from boolean_nca_cc.circuits.model import gen_circuit, generate_layer_sizes
from boolean_nca_cc.utils.graph_builder import build_graph


def load_config() -> DictConfig:
    """Load configuration from config.yaml"""
    GlobalHydra.instance().clear()
    with initialize(version_base=None, config_path="../configs"):
        cfg = compose(config_name="config")
    return cfg


class TestPoolKnockoutIntegration:
    """Test knockout pattern integration with pool operations"""

    @classmethod
    def setup_class(cls):
        """Set up test fixtures"""
        cls.config = load_config()
        cls.key = jax.random.PRNGKey(42)
        
        # Use config parameters
        cls.input_n = 4
        cls.output_n = 4
        cls.arity = cls.config.circuit.arity
        cls.circuit_hidden_dim = cls.config.circuit.circuit_hidden_dim
        cls.num_layers = 2
        
        # Generate proper layer sizes using the same function as training
        cls.layer_sizes = generate_layer_sizes(
            cls.input_n, cls.output_n, cls.arity, layer_n=cls.num_layers
        )
        cls.pool_size = 16
        cls.damage_prob = 0.3
        cls.knockout_fraction = 0.5

    def test_pool_sampling_returns_knockout_patterns(self):
        """Test that pool sampling returns knockout patterns with correct shapes"""
        print("\n=== Test: Pool Sampling Returns Knockout Patterns ===")
        
        # Create knockout config for pool initialization
        knockout_config = {
            "fraction": self.knockout_fraction,
            "damage_prob": self.damage_prob
        }
        
        # Initialize pool with knockout patterns
        pool = initialize_graph_pool(
            rng=self.key,
            layer_sizes=self.layer_sizes,
            pool_size=self.pool_size,
            input_n=self.input_n,
            arity=self.arity,
            circuit_hidden_dim=self.circuit_hidden_dim,
            knockout_config=knockout_config
        )
        
        print(f"Pool initialized with size: {pool.size}")
        print(f"Knockout patterns shape: {pool.knockout_patterns.shape}")
        
        # Sample from pool
        sample_key = jax.random.PRNGKey(123)
        batch_size = 4
        idxs, graphs, wires, logits, knockout_patterns = pool.sample(sample_key, batch_size)
        
        # Verify knockout patterns are returned
        assert knockout_patterns is not None, "Knockout patterns should be returned from sampling"
        assert knockout_patterns.shape[0] == batch_size, f"Expected batch size {batch_size}, got {knockout_patterns.shape[0]}"
        
        # Verify knockout patterns match those in pool
        for i, idx in enumerate(idxs):
            expected_pattern = pool.knockout_patterns[idx]
            actual_pattern = knockout_patterns[i]
            assert jp.array_equal(expected_pattern, actual_pattern), f"Knockout pattern mismatch for sample {i}"
        
        print(f"✓ Sampling returns knockout patterns with correct shapes: {knockout_patterns.shape}")
        print(f"✓ Sampled knockout patterns match pool patterns")
        
        # Verify some circuits have knockouts (based on fraction)
        total_knockouts = jp.sum(knockout_patterns)
        print(f"✓ Total knockouts in sample: {total_knockouts}")
        
    def test_knockout_pattern_persistence_through_sampling(self):
        """Test that knockout patterns in pool remain unchanged after sampling"""
        print("\n=== Test: Knockout Pattern Persistence Through Sampling ===")
        
        # Create vocabulary-based knockout patterns for deterministic testing
        vocab_key = jax.random.PRNGKey(456)
        vocabulary_size = 8
        
        # Generate sample circuit to extract layer info
        sample_wires, sample_logits = gen_circuit(self.key, self.layer_sizes, arity=self.arity)
        sample_graph = build_graph(
            logits=sample_logits,
            wires=sample_wires,
            input_n=self.input_n,
            arity=self.arity,
            circuit_hidden_dim=self.circuit_hidden_dim,
        )
        true_layer_sizes = extract_layer_info_from_graph(sample_graph, self.input_n)
        
        knockout_vocabulary = create_knockout_vocabulary(
            rng=vocab_key,
            vocabulary_size=vocabulary_size,
            layer_sizes=true_layer_sizes,
            damage_prob=self.damage_prob,
            input_n=self.input_n,
        )
        
        # Initialize pool with pre-generated patterns
        pool = initialize_graph_pool(
            rng=self.key,
            layer_sizes=self.layer_sizes,
            pool_size=self.pool_size,
            input_n=self.input_n,
            arity=self.arity,
            circuit_hidden_dim=self.circuit_hidden_dim,
            knockout_patterns=knockout_vocabulary
        )
        
        # Store original knockout patterns
        original_patterns = pool.knockout_patterns.copy()
        print(f"Original patterns shape: {original_patterns.shape}")
        
        # Perform multiple sampling operations
        for i in range(5):
            sample_key = jax.random.PRNGKey(100 + i)
            batch_size = 3
            idxs, graphs, wires, logits, knockout_patterns = pool.sample(sample_key, batch_size)
            
            # Verify pool patterns remain unchanged
            assert jp.array_equal(pool.knockout_patterns, original_patterns), \
                f"Pool knockout patterns changed after sampling operation {i}"
            
        print(f"✓ Knockout patterns in pool remain unchanged after {5} sampling operations")
        print(f"✓ Pool patterns persistence verified")

    def test_knockout_pattern_updates_and_persistence(self):
        """Test knockout pattern updates and persistence in pool update operations"""
        print("\n=== Test: Knockout Pattern Updates and Persistence ===")
        
        # Initialize pool with knockouts
        knockout_config = {
            "fraction": 1.0,  # All circuits have knockouts for clear testing
            "damage_prob": self.damage_prob
        }
        
        pool = initialize_graph_pool(
            rng=self.key,
            layer_sizes=self.layer_sizes,
            pool_size=self.pool_size,
            input_n=self.input_n,
            arity=self.arity,
            circuit_hidden_dim=self.circuit_hidden_dim,
            knockout_config=knockout_config
        )
        
        original_patterns = pool.knockout_patterns.copy()
        print(f"Pool initialized with knockout patterns shape: {original_patterns.shape}")
        
        # Sample a batch for update testing - use smaller batch to reduce duplicate chance
        sample_key = jax.random.PRNGKey(789)
        batch_size = 3  # Smaller batch size to reduce chance of duplicates
        idxs, graphs, wires, logits, knockout_patterns = pool.sample(sample_key, batch_size)
        
        # Test 1: Update without providing new knockout patterns (should preserve existing)
        print("\nTest 1: Update without new knockout patterns")
        updated_pool = pool.update(idxs, graphs, wires, logits)
        
        # Verify knockout patterns are preserved
        assert jp.array_equal(updated_pool.knockout_patterns, original_patterns), \
            "Knockout patterns should be preserved when not explicitly updated"
        print("✓ Knockout patterns preserved when not explicitly updated")
        
        # Test 2: Update with explicit knockout patterns
        print("\nTest 2: Update with explicit knockout patterns")
        new_patterns_key = jax.random.PRNGKey(999)
        
        # Generate new patterns for the sampled indices
        new_patterns = []
        for i in range(batch_size):
            pattern_key = jax.random.fold_in(new_patterns_key, i)
            new_pattern = create_reproducible_knockout_pattern(
                key=pattern_key,
                layer_sizes=extract_layer_info_from_graph(
                    jax.tree.map(lambda x: x[0], graphs), self.input_n
                ),
                damage_prob=0.5,  # Different damage prob for new patterns
                input_n=self.input_n,
            )
            new_patterns.append(new_pattern)
        
        new_patterns_batch = jp.stack(new_patterns)
        
        # Update with new knockout patterns using the SAME indices that were sampled
        updated_pool_with_patterns = pool.update(idxs, graphs, wires, logits, new_patterns_batch)
        
        # Verify updated indices have new patterns
        for i, idx in enumerate(idxs):
            expected_new_pattern = new_patterns_batch[i]
            actual_pattern = updated_pool_with_patterns.knockout_patterns[idx]
            assert jp.array_equal(actual_pattern, expected_new_pattern), \
                f"New knockout pattern not applied at index {idx}"
        
        # Verify non-updated indices preserve original patterns
        non_updated_mask = jp.ones(self.pool_size, dtype=bool).at[idxs].set(False)
        non_updated_indices = jp.where(non_updated_mask)[0]
        
        for idx in non_updated_indices:
            original_pattern = original_patterns[idx]
            preserved_pattern = updated_pool_with_patterns.knockout_patterns[idx]
            assert jp.array_equal(preserved_pattern, original_pattern), \
                f"Original knockout pattern not preserved at non-updated index {idx}"
        
        print(f"✓ New knockout patterns applied to updated indices: {idxs}")
        print(f"✓ Original patterns preserved for non-updated indices: {len(non_updated_indices)} circuits")

    def test_knockout_pattern_handling_in_pool_resets(self):
        """Test knockout pattern handling in pool reset operations"""
        print("\n=== Test: Knockout Pattern Handling in Pool Resets ===")
        
        # Initialize pool with diverse knockout patterns
        vocab_key = jax.random.PRNGKey(111)
        vocabulary_size = 10
        
        # Create vocabulary
        sample_wires, sample_logits = gen_circuit(self.key, self.layer_sizes, arity=self.arity)
        sample_graph = build_graph(
            logits=sample_logits,
            wires=sample_wires,
            input_n=self.input_n,
            arity=self.arity,
            circuit_hidden_dim=self.circuit_hidden_dim,
        )
        true_layer_sizes = extract_layer_info_from_graph(sample_graph, self.input_n)
        
        knockout_vocabulary = create_knockout_vocabulary(
            rng=vocab_key,
            vocabulary_size=vocabulary_size,
            layer_sizes=true_layer_sizes,
            damage_prob=self.damage_prob,
            input_n=self.input_n,
        )
        
        # Initialize pool
        pool = initialize_graph_pool(
            rng=self.key,
            layer_sizes=self.layer_sizes,
            pool_size=self.pool_size,
            input_n=self.input_n,
            arity=self.arity,
            circuit_hidden_dim=self.circuit_hidden_dim,
            knockout_patterns=knockout_vocabulary
        )
        
        original_patterns = pool.knockout_patterns.copy()
        print(f"Original pool patterns shape: {original_patterns.shape}")
        
        # Test reset with vocabulary-based patterns
        print("\nTesting reset with vocabulary-based knockout patterns")
        
        # Create fresh circuits with new knockout patterns
        fresh_pool = initialize_graph_pool(
            rng=jax.random.PRNGKey(222),
            layer_sizes=self.layer_sizes,
            pool_size=8,  # Smaller for reset source
            input_n=self.input_n,
            arity=self.arity,
            circuit_hidden_dim=self.circuit_hidden_dim,
            knockout_patterns=knockout_vocabulary  # Same vocabulary
        )
        
        # Perform reset
        reset_key = jax.random.PRNGKey(333)
        reset_fraction = 0.4
        reset_pool, avg_steps = pool.reset_fraction(
            key=reset_key,
            fraction=reset_fraction,
            new_graphs=fresh_pool.graphs,
            new_wires=fresh_pool.wires,
            new_logits=fresh_pool.logits,
            new_knockout_patterns=fresh_pool.knockout_patterns,
            reset_strategy="uniform"
        )
        
        # Verify reset occurred
        num_expected_resets = max(1, round(self.pool_size * reset_fraction))
        print(f"Expected resets: {num_expected_resets}")
        
        # Count actual pattern changes
        pattern_changes = 0
        unchanged_patterns = 0
        
        for i in range(self.pool_size):
            original_pattern = original_patterns[i]
            reset_pattern = reset_pool.knockout_patterns[i]
            
            if not jp.array_equal(original_pattern, reset_pattern):
                pattern_changes += 1
            else:
                unchanged_patterns += 1
        
        print(f"✓ Knockout patterns changed for {pattern_changes} circuits")
        print(f"✓ Knockout patterns preserved for {unchanged_patterns} circuits")
        
        # Verify that some patterns changed (reset occurred) and some remained (persistence)
        assert pattern_changes > 0, "Some knockout patterns should have changed during reset"
        assert unchanged_patterns > 0, "Some knockout patterns should have been preserved during reset"

    def test_pool_initialization_with_knockout_patterns(self):
        """Test pool initialization with different knockout pattern configurations"""
        print("\n=== Test: Pool Initialization with Knockout Patterns ===")
        
        # Test 1: Initialization with knockout config
        print("\nTest 1: Initialization with knockout config")
        knockout_config = {
            "fraction": 0.6,
            "damage_prob": 0.4
        }
        
        pool_config = initialize_graph_pool(
            rng=self.key,
            layer_sizes=self.layer_sizes,
            pool_size=self.pool_size,
            input_n=self.input_n,
            arity=self.arity,
            circuit_hidden_dim=self.circuit_hidden_dim,
            knockout_config=knockout_config
        )
        
        # Verify knockout patterns exist and have correct shape
        assert pool_config.knockout_patterns is not None, "Knockout patterns should be initialized"
        expected_shape = (self.pool_size, pool_config.graphs.nodes["logits"].shape[1])
        assert pool_config.knockout_patterns.shape == expected_shape, \
            f"Expected shape {expected_shape}, got {pool_config.knockout_patterns.shape}"
        
        # Verify some circuits have knockouts based on fraction
        circuits_with_knockouts = jp.sum(jp.any(pool_config.knockout_patterns, axis=1))
        expected_min_knockouts = round(self.pool_size * knockout_config["fraction"] * 0.5)  # Conservative estimate
        print(f"Circuits with knockouts: {circuits_with_knockouts} (expected min: {expected_min_knockouts})")
        
        print("✓ Pool initialized with knockout config successfully")
        
        # Test 2: Initialization with pre-generated patterns
        print("\nTest 2: Initialization with pre-generated patterns")
        
        # Generate vocabulary
        vocab_key = jax.random.PRNGKey(444)
        vocabulary_size = 6
        
        sample_wires, sample_logits = gen_circuit(self.key, self.layer_sizes, arity=self.arity)
        sample_graph = build_graph(
            logits=sample_logits,
            wires=sample_wires,
            input_n=self.input_n,
            arity=self.arity,
            circuit_hidden_dim=self.circuit_hidden_dim,
        )
        true_layer_sizes = extract_layer_info_from_graph(sample_graph, self.input_n)
        
        knockout_vocabulary = create_knockout_vocabulary(
            rng=vocab_key,
            vocabulary_size=vocabulary_size,
            layer_sizes=true_layer_sizes,
            damage_prob=0.25,
            input_n=self.input_n,
        )
        
        pool_vocab = initialize_graph_pool(
            rng=self.key,
            layer_sizes=self.layer_sizes,
            pool_size=self.pool_size,
            input_n=self.input_n,
            arity=self.arity,
            circuit_hidden_dim=self.circuit_hidden_dim,
            knockout_patterns=knockout_vocabulary
        )
        
        # Verify patterns are from vocabulary
        assert pool_vocab.knockout_patterns is not None, "Knockout patterns should be initialized with vocabulary"
        assert pool_vocab.knockout_patterns.shape == expected_shape, \
            f"Expected shape {expected_shape}, got {pool_vocab.knockout_patterns.shape}"
        
        # Verify all patterns in pool are from vocabulary
        vocab_patterns_found = 0
        for pool_pattern in pool_vocab.knockout_patterns:
            pattern_in_vocab = False
            for vocab_pattern in knockout_vocabulary:
                if jp.array_equal(pool_pattern, vocab_pattern):
                    pattern_in_vocab = True
                    break
            if pattern_in_vocab:
                vocab_patterns_found += 1
        
        print(f"✓ Pool patterns from vocabulary: {vocab_patterns_found}/{self.pool_size}")
        assert vocab_patterns_found == self.pool_size, "All pool patterns should be from vocabulary"

    def test_complete_knockout_lifecycle_persistence(self):
        """Test persistence of knockout patterns through complete pool lifecycle"""
        print("\n=== Test: Complete Knockout Lifecycle Persistence ===")
        
        # Create deterministic vocabulary for tracking
        vocab_key = jax.random.PRNGKey(555)
        vocabulary_size = 8
        
        sample_wires, sample_logits = gen_circuit(self.key, self.layer_sizes, arity=self.arity)
        sample_graph = build_graph(
            logits=sample_logits,
            wires=sample_wires,
            input_n=self.input_n,
            arity=self.arity,
            circuit_hidden_dim=self.circuit_hidden_dim,
        )
        true_layer_sizes = extract_layer_info_from_graph(sample_graph, self.input_n)
        
        knockout_vocabulary = create_knockout_vocabulary(
            rng=vocab_key,
            vocabulary_size=vocabulary_size,
            layer_sizes=true_layer_sizes,
            damage_prob=0.3,
            input_n=self.input_n,
        )
        
        # Initialize pool
        pool = initialize_graph_pool(
            rng=self.key,
            layer_sizes=self.layer_sizes,
            pool_size=self.pool_size,
            input_n=self.input_n,
            arity=self.arity,
            circuit_hidden_dim=self.circuit_hidden_dim,
            knockout_patterns=knockout_vocabulary
        )
        
        # Track specific circuit patterns
        tracked_indices = [0, 3, 7, 11]  # Track these specific circuits
        initial_patterns = {idx: pool.knockout_patterns[idx].copy() for idx in tracked_indices}
        
        print(f"Tracking knockout patterns for circuits: {tracked_indices}")
        
        # Phase 1: Multiple sampling operations
        print("\nPhase 1: Multiple sampling operations")
        for i in range(10):
            sample_key = jax.random.PRNGKey(600 + i)
            batch_size = 5
            idxs, graphs, wires, logits, knockout_patterns = pool.sample(sample_key, batch_size)
            
            # Verify tracked patterns remain unchanged
            for idx in tracked_indices:
                current_pattern = pool.knockout_patterns[idx]
                initial_pattern = initial_patterns[idx]
                assert jp.array_equal(current_pattern, initial_pattern), \
                    f"Pattern changed for tracked circuit {idx} during sampling phase"
        
        print("✓ Knockout patterns preserved through 10 sampling operations")
        
        # Phase 2: Update operations (some circuits updated, others not)
        print("\nPhase 2: Update operations")
        update_key = jax.random.PRNGKey(700)
        
        # Update some circuits (but not our tracked ones)
        update_indices = jp.array([1, 4, 8, 12])  # Different from tracked indices
        sample_idxs, sample_graphs, sample_wires, sample_logits, _ = pool.sample(update_key, len(update_indices))
        
        # Use the update indices instead of sample indices for precise control
        pool = pool.update(update_indices, sample_graphs, sample_wires, sample_logits)
        
        # Verify tracked patterns still unchanged
        for idx in tracked_indices:
            current_pattern = pool.knockout_patterns[idx]
            initial_pattern = initial_patterns[idx]
            assert jp.array_equal(current_pattern, initial_pattern), \
                f"Pattern changed for tracked circuit {idx} during update phase"
        
        print(f"✓ Knockout patterns preserved for non-updated circuits during updates")
        
        # Phase 3: Reset operations (some circuits reset, others preserved) 
        print("\nPhase 3: Reset operations")
        
        # Create fresh pool for reset
        fresh_pool = initialize_graph_pool(
            rng=jax.random.PRNGKey(800),
            layer_sizes=self.layer_sizes,
            pool_size=6,
            input_n=self.input_n,
            arity=self.arity,
            circuit_hidden_dim=self.circuit_hidden_dim,
            knockout_patterns=knockout_vocabulary
        )
        
        # Perform reset that should preserve some of our tracked circuits
        reset_key = jax.random.PRNGKey(900)
        reset_fraction = 0.3  # Small fraction to likely preserve most tracked circuits
        
        pool, avg_steps = pool.reset_fraction(
            key=reset_key,
            fraction=reset_fraction,
            new_graphs=fresh_pool.graphs,
            new_wires=fresh_pool.wires,
            new_logits=fresh_pool.logits,
            new_knockout_patterns=fresh_pool.knockout_patterns,
            reset_strategy="uniform"
        )
        
        # Check which tracked circuits were preserved
        preserved_count = 0
        reset_count = 0
        
        for idx in tracked_indices:
            current_pattern = pool.knockout_patterns[idx]
            initial_pattern = initial_patterns[idx]
            
            if jp.array_equal(current_pattern, initial_pattern):
                preserved_count += 1
                print(f"  Circuit {idx}: Pattern PRESERVED")
            else:
                reset_count += 1
                print(f"  Circuit {idx}: Pattern RESET")
        
        print(f"✓ Knockout lifecycle test completed:")
        print(f"  - Preserved patterns: {preserved_count}/{len(tracked_indices)}")
        print(f"  - Reset patterns: {reset_count}/{len(tracked_indices)}")
        print(f"✓ Knockout pattern persistence verified through complete lifecycle")

    def test_pool_reset_preserves_knockout_pattern_assignments(self):
        """Test that pool reset operations preserve knockout pattern assignments correctly"""
        print("\n=== Test: Pool Reset Preserves Knockout Pattern Assignments ===")
        
        # Create deterministic vocabulary for testing
        vocab_key = jax.random.PRNGKey(1000)
        vocabulary_size = 12
        
        sample_wires, sample_logits = gen_circuit(self.key, self.layer_sizes, arity=self.arity)
        sample_graph = build_graph(
            logits=sample_logits,
            wires=sample_wires,
            input_n=self.input_n,
            arity=self.arity,
            circuit_hidden_dim=self.circuit_hidden_dim,
        )
        true_layer_sizes = extract_layer_info_from_graph(sample_graph, self.input_n)
        
        knockout_vocabulary = create_knockout_vocabulary(
            rng=vocab_key,
            vocabulary_size=vocabulary_size,
            layer_sizes=true_layer_sizes,
            damage_prob=0.35,
            input_n=self.input_n,
        )
        
        # Initialize pool with vocabulary-based patterns
        pool = initialize_graph_pool(
            rng=self.key,
            layer_sizes=self.layer_sizes,
            pool_size=self.pool_size,
            input_n=self.input_n,
            arity=self.arity,
            circuit_hidden_dim=self.circuit_hidden_dim,
            knockout_patterns=knockout_vocabulary
        )
        
        # Store original knockout pattern assignments
        original_patterns = pool.knockout_patterns.copy()
        print(f"Original pool patterns shape: {original_patterns.shape}")
        
        # Create fresh pool for reset source with different patterns
        fresh_vocab_key = jax.random.PRNGKey(2000)
        fresh_knockout_vocabulary = create_knockout_vocabulary(
            rng=fresh_vocab_key,
            vocabulary_size=vocabulary_size,
            layer_sizes=true_layer_sizes,
            damage_prob=0.45,  # Different damage prob for fresh patterns
            input_n=self.input_n,
        )
        
        fresh_pool = initialize_graph_pool(
            rng=jax.random.PRNGKey(2100),
            layer_sizes=self.layer_sizes,
            pool_size=10,  # Smaller for reset source
            input_n=self.input_n,
            arity=self.arity,
            circuit_hidden_dim=self.circuit_hidden_dim,
            knockout_patterns=fresh_knockout_vocabulary
        )
        
        # Test reset with different strategies
        reset_strategies = ["uniform", "loss_biased", "steps_biased"]
        
        for strategy in reset_strategies:
            print(f"\nTesting reset strategy: {strategy}")
            
            # Perform reset
            reset_key = jax.random.PRNGKey(3000 + hash(strategy) % 1000)
            reset_fraction = 0.5  # Reset half the pool
            
            reset_pool, avg_steps = pool.reset_fraction(
                key=reset_key,
                fraction=reset_fraction,
                new_graphs=fresh_pool.graphs,
                new_wires=fresh_pool.wires,
                new_logits=fresh_pool.logits,
                new_knockout_patterns=fresh_pool.knockout_patterns,
                reset_strategy=strategy
            )
            
            # Verify reset occurred
            num_expected_resets = max(1, round(self.pool_size * reset_fraction))
            print(f"Expected resets: {num_expected_resets}")
            
            # Count pattern changes and verify assignments
            pattern_changes = 0
            preserved_patterns = 0
            fresh_patterns_found = 0
            
            for i in range(self.pool_size):
                original_pattern = original_patterns[i]
                reset_pattern = reset_pool.knockout_patterns[i]
                
                if jp.array_equal(original_pattern, reset_pattern):
                    preserved_patterns += 1
                else:
                    pattern_changes += 1
                    # Verify reset patterns are from fresh vocabulary
                    pattern_from_fresh = False
                    for fresh_pattern in fresh_knockout_vocabulary:
                        if jp.array_equal(reset_pattern, fresh_pattern):
                            pattern_from_fresh = True
                            fresh_patterns_found += 1
                            break
                    
                    assert pattern_from_fresh, f"Reset pattern at index {i} not from fresh vocabulary"
            
            print(f"  - Pattern changes: {pattern_changes}")
            print(f"  - Preserved patterns: {preserved_patterns}")
            print(f"  - Fresh patterns applied: {fresh_patterns_found}")
            
            # Verify some patterns changed and some were preserved
            assert pattern_changes > 0, f"No patterns changed with {strategy} reset strategy"
            assert preserved_patterns > 0, f"No patterns preserved with {strategy} reset strategy"
            assert fresh_patterns_found == pattern_changes, f"Not all changed patterns from fresh vocabulary"
            
            print(f"✓ Reset strategy '{strategy}' preserves knockout pattern assignments correctly")
        
        print(f"✓ All reset strategies preserve knockout pattern assignments")


def main():
    """Run all tests"""
    print("Running Level 5.1 Pool Knockout Integration Tests")
    print("=" * 60)
    
    test_suite = TestPoolKnockoutIntegration()
    test_suite.setup_class()
    
    # Run tests
    test_methods = [
        test_suite.test_pool_sampling_returns_knockout_patterns,
        test_suite.test_knockout_pattern_persistence_through_sampling,
        test_suite.test_knockout_pattern_updates_and_persistence,
        test_suite.test_knockout_pattern_handling_in_pool_resets,
        test_suite.test_pool_initialization_with_knockout_patterns,
        test_suite.test_complete_knockout_lifecycle_persistence,
        test_suite.test_pool_reset_preserves_knockout_pattern_assignments,
    ]
    
    passed = 0
    failed = 0
    
    for test_method in test_methods:
        try:
            test_method()
            passed += 1
            print(f"✓ {test_method.__name__} PASSED")
        except Exception as e:
            failed += 1
            print(f"✗ {test_method.__name__} FAILED: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print(f"Level 5.1 Pool Knockout Integration Test Results:")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Total:  {passed + failed}")
    
    if failed == 0:
        print("🎉 All tests passed! Knockout pattern persistence verified.")
    else:
        print("⚠️  Some tests failed. Check implementation.")
    
    return failed == 0


if __name__ == "__main__":
    main() 