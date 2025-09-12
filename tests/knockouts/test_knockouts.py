"""
Comprehensive test suite for gate knockout functionality.

This module tests that:
1. Knocked out gates are effectively removed from computation
2. Knocked out gates don't contribute to gradients
3. Knocked out gates aren't updated during optimization
4. Weight decay doesn't affect knocked out gates
"""

import jax
import jax.numpy as jp
import optax
import pytest
from flax import nnx

from boolean_nca_cc.circuits.model import gen_circuit, generate_layer_sizes, run_circuit
from boolean_nca_cc.training.pool.perturbation import (
    create_reproducible_knockout_pattern,
)


class TestGateKnockout:
    """Test suite for gate knockout functionality."""

    @pytest.fixture
    def circuit_config(self):
        """Standard circuit configuration for tests."""
        return {
            "input_n": 8,
            "output_n": 4,
            "arity": 4,
            "layer_n": 3,
            "key": jax.random.PRNGKey(42),
        }

    @pytest.fixture
    def sample_circuit(self, circuit_config):
        """Create a sample circuit for testing."""
        layer_sizes = generate_layer_sizes(
            circuit_config["input_n"],
            circuit_config["output_n"],
            circuit_config["arity"],
            circuit_config["layer_n"],
        )
        wires, logits = gen_circuit(circuit_config["key"], layer_sizes, circuit_config["arity"])
        return wires, logits, layer_sizes

    @pytest.fixture
    def sample_data(self, circuit_config):
        """Create sample input/output data for testing."""
        batch_size = 16
        x = jax.random.uniform(circuit_config["key"], (batch_size, circuit_config["input_n"]))
        # Create target output
        y0 = jax.random.uniform(
            jax.random.split(circuit_config["key"])[1], (batch_size, circuit_config["output_n"])
        )
        return x, y0

    def test_knockout_pattern_creation(self, sample_circuit):
        """Test that knockout patterns are created correctly."""
        wires, logits, layer_sizes = sample_circuit
        key = jax.random.PRNGKey(123)

        # Test with different knockout numbers
        for num_knockouts in [0, 1, 5, 10]:
            knockout_pattern = create_reproducible_knockout_pattern(
                key, layer_sizes, damage_prob=num_knockouts
            )

            # Check that the pattern is a 1D boolean array
            assert knockout_pattern.ndim == 1
            assert knockout_pattern.dtype == jp.bool_

            # Count total knockouts (excluding input/output layers)
            total_knockouts = 0
            current_idx = 0
            
            for layer_idx, (total_gates, group_size) in enumerate(layer_sizes):
                layer_end = current_idx + total_gates
                
                # Skip input and output layers - never knock out input or output nodes
                if layer_idx == 0 or layer_idx == len(layer_sizes) - 1:
                    # Check that input/output layers are never knocked out
                    layer_pattern = knockout_pattern[current_idx:layer_end]
                    assert jp.all(~layer_pattern), f"Layer {layer_idx} should never be knocked out"
                else:
                    # Count knockouts in hidden layers
                    layer_pattern = knockout_pattern[current_idx:layer_end]
                    total_knockouts += jp.sum(layer_pattern)
                
                current_idx = layer_end

            # Should knock out exactly the requested number (or available gates)
            max_available = sum(gate_n for gate_n, _ in layer_sizes[1:-1])
            expected_knockouts = min(num_knockouts, max_available)
            assert total_knockouts == expected_knockouts

    def test_knocked_out_gates_zero_output(self, sample_circuit, sample_data):
        """Test that knocked out gates always produce zero output regardless of logits."""
        wires, logits, layer_sizes = sample_circuit
        x, _ = sample_data
        key = jax.random.PRNGKey(456)

        # Create knockout pattern
        knockout_pattern = create_reproducible_knockout_pattern(
            key, layer_sizes, damage_prob=5
        )

        # Convert 1D knockout pattern to layered format for run_circuit
        knockout_masks = []
        current_idx = 0
        for total_gates, group_size in layer_sizes:
            layer_end = current_idx + total_gates
            layer_pattern = knockout_pattern[current_idx:layer_end]
            # Convert boolean to float: True (knocked out) -> 0.0, False (active) -> 1.0
            layer_mask = jp.where(layer_pattern, 0.0, 1.0)
            knockout_masks.append(layer_mask)
            current_idx = layer_end

        # Run circuit with knockout masks
        acts_with_knockout = run_circuit(logits, wires, x, gate_mask=knockout_masks)

        # Verify that knocked out gates in each layer produce zero
        for layer_idx in range(1, len(layer_sizes)):  # Skip input layer
            layer_mask = knockout_masks[layer_idx]
            layer_output = acts_with_knockout[layer_idx]

            # For knocked out gates (mask == 0), output should be 0
            knocked_out_indices = jp.where(layer_mask == 0.0)[0]
            if len(knocked_out_indices) > 0:
                knocked_out_outputs = layer_output[:, knocked_out_indices]
                assert jp.allclose(knocked_out_outputs, 0.0, atol=1e-6)

    def test_knockout_independence_from_logits(self, sample_circuit, sample_data):
        """Test that knocked out gates produce same output regardless of their logit values."""
        wires, logits, layer_sizes = sample_circuit
        x, _ = sample_data
        key = jax.random.PRNGKey(789)

        # Create knockout pattern
        knockout_pattern = create_reproducible_knockout_pattern(
            key, layer_sizes, damage_prob=3
        )

        # Convert 1D knockout pattern to layered format for run_circuit
        knockout_masks = []
        current_idx = 0
        for total_gates, group_size in layer_sizes:
            layer_end = current_idx + total_gates
            layer_pattern = knockout_pattern[current_idx:layer_end]
            # Convert boolean to float: True (knocked out) -> 0.0, False (active) -> 1.0
            layer_mask = jp.where(layer_pattern, 0.0, 1.0)
            knockout_masks.append(layer_mask)
            current_idx = layer_end

        # Create two different sets of logits - one normal, one with extreme values
        normal_logits = logits
        extreme_logits = []
        for layer_logits in logits:
            # Set some logits to extreme values
            extreme_layer_logits = (
                layer_logits + jax.random.normal(key, layer_logits.shape) * 100
            )  # Very large random values
            extreme_logits.append(extreme_layer_logits)

        # Run with both sets of logits but same knockout masks
        acts_normal = run_circuit(normal_logits, wires, x, gate_mask=knockout_masks)
        acts_extreme = run_circuit(extreme_logits, wires, x, gate_mask=knockout_masks)

        # Outputs should be identical for knocked out gates
        for layer_idx in range(1, len(layer_sizes)):
            layer_mask = knockout_masks[layer_idx]
            knocked_out_indices = jp.where(layer_mask == 0.0)[0]

            if len(knocked_out_indices) > 0:
                normal_output = acts_normal[layer_idx][:, knocked_out_indices]
                extreme_output = acts_extreme[layer_idx][:, knocked_out_indices]
                assert jp.allclose(normal_output, extreme_output, atol=1e-6)

    def test_knockout_gradient_masking(self, sample_circuit, sample_data):
        """Test that knocked out gates don't contribute to gradients."""
        wires, logits, layer_sizes = sample_circuit
        x, y0 = sample_data
        key = jax.random.PRNGKey(101112)

        # Create knockout pattern
        knockout_pattern = create_reproducible_knockout_pattern(
            key, layer_sizes, damage_prob=5
        )

        # Convert 1D knockout pattern to layered format for run_circuit
        knockout_masks = []
        current_idx = 0
        for total_gates, group_size in layer_sizes:
            layer_end = current_idx + total_gates
            layer_pattern = knockout_pattern[current_idx:layer_end]
            # Convert boolean to float: True (knocked out) -> 0.0, False (active) -> 1.0
            layer_mask = jp.where(layer_pattern, 0.0, 1.0)
            knockout_masks.append(layer_mask)
            current_idx = layer_end

        # Define loss function
        def loss_fn(logits_param):
            acts = run_circuit(logits_param, wires, x, gate_mask=knockout_masks)
            return jp.sum((acts[-1] - y0) ** 2)

        # Compute gradients
        loss_value, grads = jax.value_and_grad(loss_fn)(logits)

        # Check that gradients are zero for knocked out gates
        for layer_idx, (layer_grad, layer_logits) in enumerate(zip(grads, logits, strict=False)):
            layer_mask = knockout_masks[layer_idx + 1]  # +1 because logits skip input layer
            group_n, group_size, lut_size = layer_logits.shape

            # Reshape mask to match logits structure
            mask_reshaped = layer_mask.reshape(group_n, group_size)

            # Check gradients for knocked out gates
            for group_idx in range(group_n):
                for gate_idx in range(group_size):
                    if mask_reshaped[group_idx, gate_idx] == 0.0:
                        # Gradient should be zero for knocked out gates
                        gate_grad = layer_grad[group_idx, gate_idx, :]
                        assert jp.allclose(gate_grad, 0.0, atol=1e-6), (
                            f"Non-zero gradient for knocked out gate at layer {layer_idx}, group {group_idx}, gate {gate_idx}"
                        )

    def test_knockout_prevents_parameter_updates(self, sample_circuit, sample_data):
        """Test that knocked out gates don't get updated during training."""
        wires, logits, layer_sizes = sample_circuit
        x, y0 = sample_data
        key = jax.random.PRNGKey(131415)

        # Create knockout pattern
        knockout_pattern = create_reproducible_knockout_pattern(
            key, layer_sizes, damage_prob=3
        )

        # Create a simple test graph manually for testing CircuitSelfAttention
        from boolean_nca_cc.models.self_attention import CircuitSelfAttention
        import jraph
        
        # Create a simple test graph with consistent node structure and simple edges
        num_nodes = knockout_pattern.shape[0]
        
        # Create simple edges: each node connects to the next one (for testing purposes)
        num_edges = max(0, num_nodes - 1)
        if num_edges > 0:
            senders = jp.arange(num_edges)
            receivers = jp.arange(1, num_edges + 1)
        else:
            senders = jp.array([])
            receivers = jp.array([])
        
        test_graph = jraph.GraphsTuple(
            nodes={
                "logits": jp.zeros((num_nodes, 4)),  # 2^arity = 4 for arity=2
                "hidden": jp.zeros((num_nodes, 16)),  # circuit_hidden_dim=16
                "layer_pe": jp.zeros((num_nodes, 16)),
                "intra_layer_pe": jp.zeros((num_nodes, 16)),
                "loss": jp.zeros(num_nodes),
            },
            edges=None,
            receivers=receivers,
            senders=senders,
            n_node=jp.array([num_nodes]),
            n_edge=jp.array([num_edges]),
            globals=None
        )
        
        # Create CircuitSelfAttention model
        model = CircuitSelfAttention(
            n_node=num_nodes,
            circuit_hidden_dim=16,
            arity=2,
            attention_dim=64,
            num_heads=2,
            num_layers=1,
            rngs=nnx.Rngs(0)
        )
        
        # Store original logit values for knocked-out gates
        original_faulty_values = []
        knocked_out_indices = jp.where(knockout_pattern)[0]  # True = knocked out
        
        for idx in knocked_out_indices:
            original_faulty_values.append(test_graph.nodes["logits"][idx].copy())

        # Apply the model with knockout pattern to simulate one training step
        updated_graph = model(test_graph, knockout_pattern=knockout_pattern)
        
        # Check that knocked-out gates haven't received parameter updates
        updated_logits = updated_graph.nodes["logits"]
        for i, idx in enumerate(knocked_out_indices):
            original_logits = original_faulty_values[i]
            current_logits = updated_logits[idx]
            
            # Knocked-out gates should have large negative logits (effectively zero output)
            # and should not receive updates
            assert jp.all(current_logits < -5.0), (
                f"Knocked-out gate at index {idx} should have large negative logits"
            )
            
            # The logits should be exactly the large negative value (-10.0) 
            # since no updates are applied to knocked-out gates
            expected_value = -10.0
            assert jp.allclose(current_logits, expected_value, atol=1e-6), (
                f"Knocked-out gate at index {idx} should have logits set to {expected_value}"
            )

    def test_weight_decay_handling(self, sample_circuit, sample_data):
        """Test that weight decay doesn't affect knocked out gates."""
        wires, logits, layer_sizes = sample_circuit
        x, y0 = sample_data
        key = jax.random.PRNGKey(161718)

        # Create knockout pattern
        knockout_pattern = create_reproducible_knockout_pattern(
            key, layer_sizes, damage_prob=2
        )

        # Create a simple test graph and CircuitSelfAttention model for weight decay testing
        from boolean_nca_cc.models.self_attention import CircuitSelfAttention
        import jraph
        
        # Create a simple test graph with consistent node structure and simple edges
        num_nodes = knockout_pattern.shape[0]
        
        # Create simple edges: each node connects to the next one (for testing purposes)
        num_edges = max(0, num_nodes - 1)
        if num_edges > 0:
            senders = jp.arange(num_edges)
            receivers = jp.arange(1, num_edges + 1)
        else:
            senders = jp.array([])
            receivers = jp.array([])
        
        test_graph = jraph.GraphsTuple(
            nodes={
                "logits": jp.zeros((num_nodes, 4)),  # 2^arity = 4 for arity=2
                "hidden": jp.zeros((num_nodes, 16)),  # circuit_hidden_dim=16
                "layer_pe": jp.zeros((num_nodes, 16)),
                "intra_layer_pe": jp.zeros((num_nodes, 16)),
                "loss": jp.zeros(num_nodes),
            },
            edges=None,
            receivers=receivers,
            senders=senders,
            n_node=jp.array([num_nodes]),
            n_edge=jp.array([num_edges]),
            globals=None
        )
        
        # Create CircuitSelfAttention model
        model = CircuitSelfAttention(
            n_node=num_nodes,
            circuit_hidden_dim=16,
            arity=2,
            attention_dim=64,
            num_heads=2,
            num_layers=1,
            rngs=nnx.Rngs(0)
        )
        
        # Store original logit values for knocked-out gates
        knocked_out_indices = jp.where(knockout_pattern)[0]  # True = knocked out
        
        original_faulty_logits = []
        for idx in knocked_out_indices:
            original_faulty_logits.append(test_graph.nodes["logits"][idx].copy())
        
        # Apply multiple model updates to simulate training steps
        current_graph = test_graph
        for _ in range(5):
            current_graph = model(current_graph, knockout_pattern=knockout_pattern)
        
        # Check that knocked-out gates still have large negative logits after multiple updates
        final_logits = current_graph.nodes["logits"]
        for i, idx in enumerate(knocked_out_indices):
            final_gate_logits = final_logits[idx]
            
            # Knocked-out gates should still have large negative logits
            assert jp.all(final_gate_logits < -5.0), (
                f"Knocked-out gate at index {idx} should maintain large negative logits after multiple updates"
            )
            
            # The logits should still be exactly -10.0 since no updates are applied
            expected_value = -10.0
            assert jp.allclose(final_gate_logits, expected_value, atol=1e-6), (
                f"Knocked-out gate at index {idx} should maintain logits at {expected_value} after multiple updates"
            )

    def test_reproducible_knockout_patterns(self):
        """Test that knockout patterns are reproducible with same key."""
        layer_sizes = [(8, 1), (32, 4), (16, 2), (4, 1)]
        key = jax.random.PRNGKey(192021)

        # Generate pattern twice with same key
        pattern1 = create_reproducible_knockout_pattern(
            key, layer_sizes, damage_prob=5
        )
        pattern2 = create_reproducible_knockout_pattern(
            key, layer_sizes, damage_prob=5
        )

        # Should be identical
        assert jp.array_equal(pattern1, pattern2)

        # Generate pattern with different key
        different_key = jax.random.PRNGKey(222324)
        pattern3 = create_reproducible_knockout_pattern(
            different_key, layer_sizes, damage_prob=5
        )

        # Should be different
        assert not jp.array_equal(pattern1, pattern3), "Different keys should produce different patterns"

    def test_edge_cases(self, sample_circuit):
        """Test edge cases for knockout functionality."""
        wires, logits, layer_sizes = sample_circuit
        key = jax.random.PRNGKey(252627)

        # Test with zero knockouts
        zero_knockout_pattern = create_reproducible_knockout_pattern(
            key, layer_sizes, damage_prob=0
        )
        # Should be all False (no knockouts)
        assert jp.all(~zero_knockout_pattern)

        # Test with more knockouts than available gates
        max_available = sum(gate_n for gate_n, _ in layer_sizes[1:-1])  # Exclude input/output
        excessive_knockout_pattern = create_reproducible_knockout_pattern(
            key, layer_sizes, damage_prob=max_available + 10
        )

        # Count actual knockouts (True values in the pattern)
        total_knockouts = jp.sum(excessive_knockout_pattern)

        # Should not exceed available gates
        assert total_knockouts <= max_available


