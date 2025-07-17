import jax
import jax.numpy as jnp
import numpy as np
from pathlib import Path
import yaml
from boolean_nca_cc.training.pool.structural_perturbation import (
    create_reproducible_knockout_pattern,
    create_knockout_vocabulary,
)

CONFIG_PATH = Path(__file__).parent.parent / "configs/config.yaml"
with open(CONFIG_PATH) as f:
    config = yaml.safe_load(f)

# Test parameters
TEST_N_NODES = config["circuit"]["input_bits"]
TEST_KNOCKOUT_RATE = config["pool"]["persistent_knockout"]["damage_prob"]
TEST_SEEDS = [config["test_seed"], 314, 2718]

class TestStructuralPerturbation:
    """Test suite for knockout pattern generation in structural_perturbation.py"""

    def test_create_reproducible_knockout_pattern_basic(self):
        """Basic pattern creation test"""
        key = jax.random.PRNGKey(TEST_SEEDS[0])
        pattern = create_reproducible_knockout_pattern(
            key, TEST_N_NODES, TEST_KNOCKOUT_RATE
        )
        
        assert isinstance(pattern, jnp.ndarray)
        assert pattern.shape == (TEST_N_NODES,)
        assert np.isclose(pattern.mean(), TEST_KNOCKOUT_RATE, atol=0.1)

    def test_pattern_reproducibility(self):
        """Test pattern reproducibility with same seed"""
        keys = [jax.random.PRNGKey(s) for s in TEST_SEEDS]
        
        # Generate patterns with same seeds
        patterns1 = [
            create_reproducible_knockout_pattern(k, TEST_N_NODES, TEST_KNOCKOUT_RATE)
            for k in keys
        ]
        patterns2 = [
            create_reproducible_knockout_pattern(k, TEST_N_NODES, TEST_KNOCKOUT_RATE)
            for k in keys
        ]
        
        # Verify reproducibility
        for p1, p2 in zip(patterns1, patterns2):
            assert jnp.array_equal(p1, p2)

    def test_knockout_vocabulary_creation(self):
        """Test vocabulary generation with ID/OOD split"""
        in_vocab_size = 5
        ood_vocab_size = 3
        total_vocab_size = in_vocab_size + ood_vocab_size
        
        key = jax.random.PRNGKey(TEST_SEEDS[0])
        knockout_vocab = create_knockout_vocabulary(
            key,
            in_vocab_size,
            ood_vocab_size,
            TEST_N_NODES,
            TEST_KNOCKOUT_RATE,
        )
        
        # Verify structure
        assert isinstance(knockout_vocab, dict)
        assert "in_dist" in knockout_vocab
        assert "out_dist" in knockout_vocab
        assert len(knockout_vocab["in_dist"]) == in_vocab_size
        assert len(knockout_vocab["out_dist"]) == ood_vocab_size
        
        # Verify pattern uniqueness
        all_patterns = knockout_vocab["in_dist"] + knockout_vocab["out_dist"]
        unique_patterns = set(tuple(p.tolist()) for p in all_patterns)
        assert len(unique_patterns) == total_vocab_size

    def test_vocabulary_reproducibility(self):
        """Test vocabulary generation reproducibility"""
        key1 = jax.random.PRNGKey(TEST_SEEDS[0])
        key2 = jax.random.PRNGKey(TEST_SEEDS[0])
        
        knockout_diversity = config["pool"]["persistent_knockout"]["knockout_diversity"]
        vocab1 = create_knockout_vocabulary(
            key1,
            int(knockout_diversity * 0.9),
            int(knockout_diversity * 0.1),
            TEST_N_NODES,
            TEST_KNOCKOUT_RATE
        )
        vocab2 = create_knockout_vocabulary(
            key2, 3, 2, TEST_N_NODES, TEST_KNOCKOUT_RATE
        )
        
        # Compare ID patterns
        for p1, p2 in zip(vocab1["in_dist"], vocab2["in_dist"]):
            assert jnp.array_equal(p1, p2)
            
        # Compare OOD patterns
        for p1, p2 in zip(vocab1["out_dist"], vocab2["out_dist"]):
            assert jnp.array_equal(p1, p2)

if __name__ == "__main__":
    import pytest
    pytest.main([__file__])