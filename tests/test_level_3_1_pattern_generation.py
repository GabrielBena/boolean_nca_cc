import jax
import jax.numpy as jnp
import numpy as np
from pathlib import Path
import yaml
import jraph
from boolean_nca_cc.training.pool.structural_perturbation import (
    create_reproducible_knockout_pattern,
    create_knockout_vocabulary,
    extract_layer_info_from_graph,
)
from boolean_nca_cc.circuits.model import generate_layer_sizes, gen_circuit
from boolean_nca_cc.utils.graph_builder import build_graph
import pytest

CONFIG_PATH = Path(__file__).parent.parent / "configs/config.yaml"
with open(CONFIG_PATH) as f:
    config = yaml.safe_load(f)

# Test parameters from config
TEST_INPUT_BITS = config["circuit"]["input_bits"]
TEST_OUTPUT_BITS = config["circuit"]["output_bits"]
TEST_ARITY = config["circuit"]["arity"]
TEST_NUM_LAYERS = config["circuit"]["num_layers"]
TEST_KNOCKOUT_RATE = config["pool"]["persistent_knockout"]["damage_prob"]
TEST_SEEDS = [config["test_seed"], 314, 2718]

# Generate actual layer sizes using the same function as the real circuit
TEST_LAYER_SIZES = generate_layer_sizes(TEST_INPUT_BITS, TEST_OUTPUT_BITS, TEST_ARITY, TEST_NUM_LAYERS)
TEST_INPUT_N = TEST_INPUT_BITS

class TestStructuralPerturbation:
    """Test suite for knockout pattern generation in structural_perturbation.py"""

    def test_create_reproducible_knockout_pattern_basic(self):
        """Basic pattern creation test"""
        key = jax.random.PRNGKey(TEST_SEEDS[0])
        pattern = create_reproducible_knockout_pattern(
            key, TEST_LAYER_SIZES, TEST_KNOCKOUT_RATE, TEST_INPUT_N
        )
        
        # Calculate expected total nodes: input_n + sum(group_n * group_size)
        expected_total_nodes = TEST_INPUT_N + sum(group_n * group_size for group_n, group_size in TEST_LAYER_SIZES)
        
        assert isinstance(pattern, jnp.ndarray)
        assert pattern.shape == (expected_total_nodes,)
        # Check that input nodes are never knocked out
        assert not jnp.any(pattern[:TEST_INPUT_N])

    def test_pattern_reproducibility(self):
        """Test pattern reproducibility with same seed"""
        keys = [jax.random.PRNGKey(s) for s in TEST_SEEDS]
        
        # Generate patterns with same seeds
        patterns1 = [
            create_reproducible_knockout_pattern(k, TEST_LAYER_SIZES, TEST_KNOCKOUT_RATE, TEST_INPUT_N)
            for k in keys
        ]
        patterns2 = [
            create_reproducible_knockout_pattern(k, TEST_LAYER_SIZES, TEST_KNOCKOUT_RATE, TEST_INPUT_N)
            for k in keys
        ]
        
        # Verify reproducibility
        for p1, p2 in zip(patterns1, patterns2):
            assert jnp.array_equal(p1, p2)

    def test_knockout_vocabulary_creation(self):
        """Test vocabulary generation"""
        vocab_size = 8
        
        key = jax.random.PRNGKey(TEST_SEEDS[0])
        knockout_vocab = create_knockout_vocabulary(
            key,
            vocab_size,
            TEST_LAYER_SIZES,
            TEST_KNOCKOUT_RATE,
            TEST_INPUT_N,
        )
        
        # Calculate expected total nodes: input_n + sum(group_n * group_size)
        expected_total_nodes = TEST_INPUT_N + sum(group_n * group_size for group_n, group_size in TEST_LAYER_SIZES)
        
        # Verify structure
        assert isinstance(knockout_vocab, jnp.ndarray)
        assert knockout_vocab.shape == (vocab_size, expected_total_nodes)
        
        # Verify pattern uniqueness
        unique_patterns = set(tuple(p.tolist()) for p in knockout_vocab)
        assert len(unique_patterns) == vocab_size

    def test_vocabulary_reproducibility(self):
        """Test vocabulary generation reproducibility"""
        key1 = jax.random.PRNGKey(TEST_SEEDS[0])
        key2 = jax.random.PRNGKey(TEST_SEEDS[0])
        
        vocab_size = 5
        vocab1 = create_knockout_vocabulary(
            key1,
            vocab_size,
            TEST_LAYER_SIZES,
            TEST_KNOCKOUT_RATE,
            TEST_INPUT_N
        )
        vocab2 = create_knockout_vocabulary(
            key2, vocab_size, TEST_LAYER_SIZES, TEST_KNOCKOUT_RATE, TEST_INPUT_N
        )
        
        # Compare patterns
        assert jnp.array_equal(vocab1, vocab2)

    def test_extract_layer_info_from_graph_basic(self):
        """Test basic layer info extraction from graph"""
        # Create a simple circuit and convert to graph
        key = jax.random.PRNGKey(TEST_SEEDS[0])
        layer_sizes = [(8, 1), (16, 4), (8, 1)]  # Simple 3-layer circuit
        wires, logits = gen_circuit(key, layer_sizes, arity=TEST_ARITY)
        
        # Build graph from circuit
        graph = build_graph(
            logits=logits,
            wires=wires,
            input_n=8,
            arity=TEST_ARITY,
            circuit_hidden_dim=16,
        )
        
        # Extract layer info
        extracted_layer_sizes = extract_layer_info_from_graph(graph, input_n=8)
        
        # Verify extraction matches expected structure
        # Should extract: [(4, 4), (8, 1)] for both hidden and output layers
        # Hidden layer: 16 gates / 4 groups = 4 gates per group
        # Output layer: 8 gates / 1 group = 8 gates per group
        expected_layer_sizes = [(4, 4), (8, 1)]
        assert extracted_layer_sizes == expected_layer_sizes

    def test_extract_layer_info_from_graph_complex(self):
        """Test layer info extraction from complex multi-layer graph"""
        # Use the actual circuit structure from config
        key = jax.random.PRNGKey(TEST_SEEDS[0])
        wires, logits = gen_circuit(key, TEST_LAYER_SIZES, arity=TEST_ARITY)
        
        # Build graph from circuit
        graph = build_graph(
            logits=logits,
            wires=wires,
            input_n=TEST_INPUT_N,
            arity=TEST_ARITY,
            circuit_hidden_dim=32,
        )
        
        # Extract layer info
        extracted_layer_sizes = extract_layer_info_from_graph(graph, input_n=TEST_INPUT_N)
        
        # Verify extraction produces valid layer sizes
        assert len(extracted_layer_sizes) > 0
        for group_n, group_size in extracted_layer_sizes:
            assert group_n > 0
            assert group_size > 0
            assert group_n * group_size > 0  # Each layer should have gates

    def test_extract_layer_info_round_trip(self):
        """Test round-trip: generate layer sizes -> build graph -> extract -> create patterns"""
        # Generate layer sizes
        original_layer_sizes = TEST_LAYER_SIZES
        
        # Create circuit and graph
        key = jax.random.PRNGKey(TEST_SEEDS[0])
        wires, logits = gen_circuit(key, original_layer_sizes, arity=TEST_ARITY)
        graph = build_graph(
            logits=logits,
            wires=wires,
            input_n=TEST_INPUT_N,
            arity=TEST_ARITY,
            circuit_hidden_dim=32,
        )
        
        # Extract layer sizes from graph
        extracted_layer_sizes = extract_layer_info_from_graph(graph, input_n=TEST_INPUT_N)
        
        # Create knockout patterns using extracted layer sizes
        pattern_key = jax.random.PRNGKey(TEST_SEEDS[1])
        knockout_pattern = create_reproducible_knockout_pattern(
            pattern_key, extracted_layer_sizes, TEST_KNOCKOUT_RATE, TEST_INPUT_N
        )
        
        # Verify pattern has correct shape
        expected_total_nodes = TEST_INPUT_N + sum(group_n * group_size for group_n, group_size in extracted_layer_sizes)
        assert knockout_pattern.shape == (expected_total_nodes,)
        
        # Verify input nodes are never knocked out
        assert not jnp.any(knockout_pattern[:TEST_INPUT_N])

    def test_extract_layer_info_error_handling(self):
        """Test error handling for graphs without required node features"""
        # Create a minimal graph without layer/group features
        nodes = {"logits": jnp.zeros((10, 4)), "hidden": jnp.zeros((10, 16))}
        edges = {}
        senders = jnp.array([])
        receivers = jnp.array([])
        globals_ = None
        n_node = jnp.array(10)
        n_edge = jnp.array(0)
        
        graph = jraph.GraphsTuple(
            nodes=nodes,
            edges=edges,
            senders=senders,
            receivers=receivers,
            globals=globals_,
            n_node=n_node,
            n_edge=n_edge,
        )
        
        # Should raise ValueError for missing layer/group features
        with pytest.raises(ValueError, match="Graph must have 'layer' and 'group' node features"):
            extract_layer_info_from_graph(graph, input_n=5)

if __name__ == "__main__":
    import pytest
    pytest.main([__file__])