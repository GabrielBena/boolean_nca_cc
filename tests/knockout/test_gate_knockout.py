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

from boolean_nca_cc.circuits.model import gen_circuit, generate_layer_sizes, run_circuit
from boolean_nca_cc.circuits.train import TrainState, train_step
from boolean_nca_cc.training.pool.structural_perturbation import (
    apply_gate_knockout_to_pool_element,
    create_faulty_gate_logits,
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
            knockout_masks = create_reproducible_knockout_pattern(
                key, layer_sizes, num_knockouts, input_n=layer_sizes[0][0]
            )

            # Check structure
            assert len(knockout_masks) == len(layer_sizes)

            # Check that input layer is never knocked out
            assert jp.all(knockout_masks[0] == 1.0)

            # Check that output layer is never knocked out
            assert jp.all(knockout_masks[-1] == 1.0)

            # Count total knockouts (excluding input/output layers)
            total_knockouts = 0
            for i in range(1, len(knockout_masks) - 1):  # Skip input and output
                total_knockouts += jp.sum(knockout_masks[i] == 0.0)

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
        knockout_masks = create_reproducible_knockout_pattern(
            key, layer_sizes, number_knokouts=5, input_n=layer_sizes[0][0]
        )

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
        knockout_masks = create_reproducible_knockout_pattern(
            key, layer_sizes, number_knokouts=3, input_n=layer_sizes[0][0]
        )

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
        knockout_masks = create_reproducible_knockout_pattern(
            key, layer_sizes, number_knokouts=5, input_n=layer_sizes[0][0]
        )

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
        knockout_masks = create_reproducible_knockout_pattern(
            key, layer_sizes, number_knokouts=3, input_n=layer_sizes[0][0]
        )

        # Apply knockout to get faulty logits
        modified_logits, _, _ = apply_gate_knockout_to_pool_element(
            key, logits, wires, layer_sizes, number_knockouts=3
        )

        # Store original faulty logit values
        original_faulty_values = []
        for layer_idx, (layer_logits, layer_mask) in enumerate(
            zip(modified_logits, knockout_masks[1:], strict=False)  # Skip input layer
        ):
            group_n, group_size, lut_size = layer_logits.shape
            mask_reshaped = layer_mask.reshape(group_n, group_size)

            layer_faulty_values = []
            for group_idx in range(group_n):
                for gate_idx in range(group_size):
                    if mask_reshaped[group_idx, gate_idx] == 0.0:
                        gate_logits = layer_logits[group_idx, gate_idx, :]
                        layer_faulty_values.append(gate_logits.copy())
            original_faulty_values.append(layer_faulty_values)

        # Create optimizer and training state
        optimizer = optax.adam(learning_rate=0.01)
        opt_state = optimizer.init(modified_logits)
        train_state = TrainState(modified_logits, opt_state)

        # Perform training step
        loss, aux, new_train_state = train_step(
            train_state,
            optimizer,
            wires,
            x,
            y0,
            loss_type="l4",
            do_train=True,
            gate_mask=knockout_masks,
        )

        # Check that faulty logits haven't changed
        new_logits = new_train_state.params
        for layer_idx, (layer_logits, layer_mask) in enumerate(
            zip(new_logits, knockout_masks[1:], strict=False)  # Skip input layer
        ):
            group_n, group_size, lut_size = layer_logits.shape
            mask_reshaped = layer_mask.reshape(group_n, group_size)

            faulty_idx = 0
            for group_idx in range(group_n):
                for gate_idx in range(group_size):
                    if mask_reshaped[group_idx, gate_idx] == 0.0:
                        gate_logits = layer_logits[group_idx, gate_idx, :]
                        original_gate_logits = original_faulty_values[layer_idx][faulty_idx]

                        # Faulty logits should remain unchanged
                        assert jp.allclose(gate_logits, original_gate_logits, atol=1e-6), (
                            f"Faulty logits changed at layer {layer_idx}, group {group_idx}, gate {gate_idx}"
                        )
                        faulty_idx += 1

    def test_weight_decay_handling(self, sample_circuit, sample_data):
        """Test that weight decay doesn't affect knocked out gates."""
        wires, logits, layer_sizes = sample_circuit
        x, y0 = sample_data
        key = jax.random.PRNGKey(161718)

        # Create knockout pattern
        knockout_masks = create_reproducible_knockout_pattern(
            key, layer_sizes, number_knokouts=2, input_n=layer_sizes[0][0]
        )

        # Apply knockout to get faulty logits
        modified_logits, _, _ = apply_gate_knockout_to_pool_element(
            key, logits, wires, layer_sizes, number_knockouts=2
        )

        # Store original faulty values
        original_faulty_logits = []
        for layer_logits in modified_logits:
            original_faulty_logits.append(layer_logits.copy())

        # Create optimizer with weight decay
        optimizer = optax.adamw(learning_rate=0.01, weight_decay=0.1)
        opt_state = optimizer.init(modified_logits)
        train_state = TrainState(modified_logits, opt_state)

        # Perform multiple training steps
        for _ in range(5):
            loss, aux, train_state = train_step(
                train_state,
                optimizer,
                wires,
                x,
                y0,
                loss_type="l4",
                do_train=True,
                gate_mask=knockout_masks,
            )

        # Check that faulty logits are preserved despite weight decay
        final_logits = train_state.params
        for layer_idx, (final_layer_logits, original_layer_logits, layer_mask) in enumerate(
            zip(final_logits, original_faulty_logits, knockout_masks[1:], strict=False)
        ):
            group_n, group_size, lut_size = final_layer_logits.shape
            mask_reshaped = layer_mask.reshape(group_n, group_size)

            for group_idx in range(group_n):
                for gate_idx in range(group_size):
                    if mask_reshaped[group_idx, gate_idx] == 0.0:
                        final_gate_logits = final_layer_logits[group_idx, gate_idx, :]
                        original_gate_logits = original_layer_logits[group_idx, gate_idx, :]

                        # Faulty logits should be preserved
                        assert jp.allclose(final_gate_logits, original_gate_logits, atol=1e-6), (
                            f"Weight decay affected faulty logits at layer {layer_idx}, group {group_idx}, gate {gate_idx}"
                        )

    def test_reproducible_knockout_patterns(self):
        """Test that knockout patterns are reproducible with same key."""
        layer_sizes = [(8, 1), (32, 4), (16, 2), (4, 1)]
        key = jax.random.PRNGKey(192021)

        # Generate pattern twice with same key
        pattern1 = create_reproducible_knockout_pattern(
            key, layer_sizes, number_knokouts=5, input_n=8
        )
        pattern2 = create_reproducible_knockout_pattern(
            key, layer_sizes, number_knokouts=5, input_n=8
        )

        # Should be identical
        for mask1, mask2 in zip(pattern1, pattern2, strict=False):
            assert jp.array_equal(mask1, mask2)

        # Generate pattern with different key
        different_key = jax.random.PRNGKey(222324)
        pattern3 = create_reproducible_knockout_pattern(
            different_key, layer_sizes, number_knokouts=5, input_n=8
        )

        # Should be different
        patterns_differ = False
        for mask1, mask3 in zip(pattern1, pattern3, strict=False):
            if not jp.array_equal(mask1, mask3):
                patterns_differ = True
                break
        assert patterns_differ, "Different keys should produce different patterns"

    def test_edge_cases(self, sample_circuit):
        """Test edge cases for knockout functionality."""
        wires, logits, layer_sizes = sample_circuit
        key = jax.random.PRNGKey(252627)

        # Test with zero knockouts
        zero_knockout_masks = create_reproducible_knockout_pattern(
            key, layer_sizes, number_knokouts=0, input_n=layer_sizes[0][0]
        )
        for mask in zero_knockout_masks:
            assert jp.all(mask == 1.0)

        # Test with more knockouts than available gates
        max_available = sum(gate_n for gate_n, _ in layer_sizes[1:-1])  # Exclude input/output
        excessive_knockout_masks = create_reproducible_knockout_pattern(
            key, layer_sizes, number_knokouts=max_available + 10, input_n=layer_sizes[0][0]
        )

        # Count actual knockouts
        total_knockouts = 0
        for i in range(1, len(excessive_knockout_masks) - 1):  # Skip input and output
            total_knockouts += jp.sum(excessive_knockout_masks[i] == 0.0)

        # Should not exceed available gates
        assert total_knockouts <= max_available

    def test_faulty_gate_logits_creation(self, sample_circuit):
        """Test creation of faulty gate logits."""
        wires, logits, layer_sizes = sample_circuit
        key = jax.random.PRNGKey(282930)

        # Create knockout pattern
        knockout_masks = create_reproducible_knockout_pattern(
            key, layer_sizes, number_knokouts=3, input_n=layer_sizes[0][0]
        )

        # Test faulty logits creation for each layer
        for _layer_idx, (layer_logits, layer_mask) in enumerate(
            zip(logits, knockout_masks[1:], strict=False)  # Skip input layer
        ):
            faulty_logits = create_faulty_gate_logits(layer_logits, layer_mask, faulty_value=-10.0)

            group_n, group_size, lut_size = layer_logits.shape
            mask_reshaped = layer_mask.reshape(group_n, group_size)

            # Check that faulty values are set correctly
            for group_idx in range(group_n):
                for gate_idx in range(group_size):
                    gate_logits = faulty_logits[group_idx, gate_idx, :]
                    if mask_reshaped[group_idx, gate_idx] == 0.0:
                        # Should be all faulty values
                        assert jp.all(gate_logits == -10.0)
                    else:
                        # Should be original values
                        original_gate_logits = layer_logits[group_idx, gate_idx, :]
                        assert jp.allclose(gate_logits, original_gate_logits)
