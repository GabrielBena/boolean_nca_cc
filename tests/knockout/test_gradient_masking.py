"""
Test gradient masking and parameter update prevention for knocked out gates.

This module specifically tests the integration between gate knockout functionality
and the training process, ensuring that gradients are properly masked and
parameters are not updated for knocked out gates.
"""

import jax
import jax.numpy as jp
import optax
import pytest

from boolean_nca_cc.circuits.model import gen_circuit, generate_layer_sizes
from boolean_nca_cc.circuits.train import TrainState, train_step, update_params
from boolean_nca_cc.training.pool.structural_perturbation import (
    apply_gate_knockout_to_pool_element,
    create_reproducible_knockout_pattern,
)


class TestGradientMasking:
    """Test gradient masking and parameter preservation for knocked out gates."""

    @pytest.fixture
    def setup_circuit_and_data(self):
        """Set up a circuit and training data for testing."""
        key = jax.random.PRNGKey(42)
        input_n, output_n, arity, layer_n = 6, 3, 3, 2

        # Generate circuit
        layer_sizes = generate_layer_sizes(input_n, output_n, arity, layer_n)
        wires, logits = gen_circuit(key, layer_sizes, arity)

        # Generate training data
        batch_size = 8
        x = jax.random.uniform(key, (batch_size, input_n))
        y0 = jax.random.uniform(jax.random.split(key)[1], (batch_size, output_n))

        return {
            "wires": wires,
            "logits": logits,
            "layer_sizes": layer_sizes,
            "x": x,
            "y0": y0,
            "key": key,
        }

    def test_gradient_masking_in_update_params(self, setup_circuit_and_data):
        """Test that update_params correctly masks gradients for knocked out gates."""
        data = setup_circuit_and_data
        key = jax.random.PRNGKey(123)

        # Create knockout pattern
        knockout_masks = create_reproducible_knockout_pattern(
            key, data["layer_sizes"], number_knokouts=3, input_n=data["layer_sizes"][0][0]
        )

        # Create fake gradients (all ones for simplicity)
        fake_gradients = [jp.ones_like(layer_logits) for layer_logits in data["logits"]]

        # Create optimizer
        optimizer = optax.sgd(learning_rate=0.1)
        opt_state = optimizer.init(data["logits"])

        # Test update_params without mask (should update all parameters)
        new_logits_no_mask, _ = update_params(
            fake_gradients, opt_state, optimizer, data["logits"], gate_mask=None
        )

        # Test update_params with mask (should preserve knocked out gates)
        new_logits_with_mask, _ = update_params(
            fake_gradients, opt_state, optimizer, data["logits"], gate_mask=knockout_masks
        )

        # Check that knocked out gates are preserved
        for layer_idx, (original_layer, updated_layer, layer_mask) in enumerate(
            zip(data["logits"], new_logits_with_mask, knockout_masks[1:])  # Skip input layer
        ):
            group_n, group_size, lut_size = original_layer.shape
            mask_reshaped = layer_mask.reshape(group_n, group_size)

            for group_idx in range(group_n):
                for gate_idx in range(group_size):
                    original_gate = original_layer[group_idx, gate_idx, :]
                    updated_gate = updated_layer[group_idx, gate_idx, :]

                    if mask_reshaped[group_idx, gate_idx] == 0.0:
                        # Knocked out gate should be unchanged
                        assert jp.allclose(original_gate, updated_gate, atol=1e-6), (
                            f"Knocked out gate changed at layer {layer_idx}, group {group_idx}, gate {gate_idx}"
                        )
                    else:
                        # Active gate should be updated (different from original)
                        assert not jp.allclose(original_gate, updated_gate, atol=1e-6), (
                            f"Active gate not updated at layer {layer_idx}, group {group_idx}, gate {gate_idx}"
                        )

    def test_weight_decay_with_masked_updates(self, setup_circuit_and_data):
        """Test that weight decay doesn't affect knocked out gates."""
        data = setup_circuit_and_data
        key = jax.random.PRNGKey(456)

        # Create knockout pattern and apply it
        knockout_masks = create_reproducible_knockout_pattern(
            key, data["layer_sizes"], number_knokouts=2, input_n=data["layer_sizes"][0][0]
        )

        # Apply knockout to get faulty logits
        faulty_logits, _, _ = apply_gate_knockout_to_pool_element(
            key, data["logits"], data["wires"], data["layer_sizes"], number_knockouts=2
        )

        # Store original faulty values
        original_faulty_logits = [layer.copy() for layer in faulty_logits]

        # Create optimizer with aggressive weight decay
        optimizer = optax.adamw(learning_rate=0.01, weight_decay=0.5)
        opt_state = optimizer.init(faulty_logits)

        # Create large gradients to test weight decay effect
        large_gradients = [jp.ones_like(layer) * 10.0 for layer in faulty_logits]

        # Apply multiple updates
        current_logits = faulty_logits
        current_opt_state = opt_state

        for _ in range(10):  # Multiple steps to amplify weight decay effect
            current_logits, current_opt_state = update_params(
                large_gradients, current_opt_state, optimizer, current_logits, knockout_masks
            )

        # Verify that knocked out gates are still preserved
        for layer_idx, (final_layer, original_layer, layer_mask) in enumerate(
            zip(current_logits, original_faulty_logits, knockout_masks[1:])
        ):
            group_n, group_size, lut_size = final_layer.shape
            mask_reshaped = layer_mask.reshape(group_n, group_size)

            for group_idx in range(group_n):
                for gate_idx in range(group_size):
                    if mask_reshaped[group_idx, gate_idx] == 0.0:
                        final_gate = final_layer[group_idx, gate_idx, :]
                        original_gate = original_layer[group_idx, gate_idx, :]

                        assert jp.allclose(final_gate, original_gate, atol=1e-6), (
                            f"Weight decay affected knocked out gate at layer {layer_idx}, group {group_idx}, gate {gate_idx}"
                        )

    def test_train_step_integration(self, setup_circuit_and_data):
        """Test that train_step properly integrates with masked parameter updates."""
        data = setup_circuit_and_data
        key = jax.random.PRNGKey(789)

        # Create knockout pattern and apply it
        knockout_masks = create_reproducible_knockout_pattern(
            key, data["layer_sizes"], number_knokouts=2, input_n=data["layer_sizes"][0][0]
        )

        faulty_logits, _, _ = apply_gate_knockout_to_pool_element(
            key, data["logits"], data["wires"], data["layer_sizes"], number_knockouts=2
        )

        # Store original faulty values
        original_faulty_logits = [layer.copy() for layer in faulty_logits]

        # Create training state
        optimizer = optax.adam(learning_rate=0.01)
        opt_state = optimizer.init(faulty_logits)
        train_state = TrainState(faulty_logits, opt_state)

        # Perform training step with gate mask
        loss, aux, new_train_state = train_step(
            train_state,
            optimizer,
            data["wires"],
            data["x"],
            data["y0"],
            loss_type="l4",
            do_train=True,
            gate_mask=knockout_masks,
        )

        # Verify that knocked out gates are preserved
        new_logits = new_train_state.params
        for layer_idx, (new_layer, original_layer, layer_mask) in enumerate(
            zip(new_logits, original_faulty_logits, knockout_masks[1:])
        ):
            group_n, group_size, lut_size = new_layer.shape
            mask_reshaped = layer_mask.reshape(group_n, group_size)

            for group_idx in range(group_n):
                for gate_idx in range(group_size):
                    if mask_reshaped[group_idx, gate_idx] == 0.0:
                        new_gate = new_layer[group_idx, gate_idx, :]
                        original_gate = original_layer[group_idx, gate_idx, :]

                        assert jp.allclose(new_gate, original_gate, atol=1e-6), (
                            f"Train step modified knocked out gate at layer {layer_idx}, group {group_idx}, gate {gate_idx}"
                        )

    def test_different_optimizer_types(self, setup_circuit_and_data):
        """Test masked updates work with different optimizer types."""
        data = setup_circuit_and_data
        key = jax.random.PRNGKey(101112)

        # Create knockout pattern
        knockout_masks = create_reproducible_knockout_pattern(
            key, data["layer_sizes"], number_knokouts=1, input_n=data["layer_sizes"][0][0]
        )

        # Test with different optimizers
        optimizers = [
            optax.sgd(learning_rate=0.1),
            optax.adam(learning_rate=0.01),
            optax.adamw(learning_rate=0.01, weight_decay=0.1),
            optax.rmsprop(learning_rate=0.01),
        ]

        for optimizer in optimizers:
            # Apply knockout
            faulty_logits, _, _ = apply_gate_knockout_to_pool_element(
                key, data["logits"], data["wires"], data["layer_sizes"], number_knockouts=1
            )

            original_faulty_logits = [layer.copy() for layer in faulty_logits]

            # Initialize optimizer state
            opt_state = optimizer.init(faulty_logits)

            # Create gradients
            gradients = [jp.ones_like(layer) for layer in faulty_logits]

            # Apply update with mask
            new_logits, _ = update_params(
                gradients, opt_state, optimizer, faulty_logits, knockout_masks
            )

            # Verify knocked out gates are preserved
            for layer_idx, (new_layer, original_layer, layer_mask) in enumerate(
                zip(new_logits, original_faulty_logits, knockout_masks[1:])
            ):
                group_n, group_size, lut_size = new_layer.shape
                mask_reshaped = layer_mask.reshape(group_n, group_size)

                for group_idx in range(group_n):
                    for gate_idx in range(group_size):
                        if mask_reshaped[group_idx, gate_idx] == 0.0:
                            new_gate = new_layer[group_idx, gate_idx, :]
                            original_gate = original_layer[group_idx, gate_idx, :]

                            assert jp.allclose(new_gate, original_gate, atol=1e-6), (
                                f"Optimizer {type(optimizer).__name__} modified knocked out gate"
                            )

    def test_gradient_computation_with_knockout(self, setup_circuit_and_data):
        """Test that gradients are computed correctly when gates are knocked out."""
        data = setup_circuit_and_data
        key = jax.random.PRNGKey(131415)

        # Create knockout pattern
        knockout_masks = create_reproducible_knockout_pattern(
            key, data["layer_sizes"], number_knokouts=2, input_n=data["layer_sizes"][0][0]
        )

        from boolean_nca_cc.circuits.model import run_circuit

        def loss_fn(logits):
            acts = run_circuit(logits, data["wires"], data["x"], gate_mask=knockout_masks)
            return jp.sum((acts[-1] - data["y0"]) ** 2)

        # Compute gradients
        loss_value, gradients = jax.value_and_grad(loss_fn)(data["logits"])

        # Verify that gradients are zero for knocked out gates
        for layer_idx, (layer_grad, layer_mask) in enumerate(
            zip(gradients, knockout_masks[1:])  # Skip input layer
        ):
            group_n, group_size, lut_size = layer_grad.shape
            mask_reshaped = layer_mask.reshape(group_n, group_size)

            for group_idx in range(group_n):
                for gate_idx in range(group_size):
                    if mask_reshaped[group_idx, gate_idx] == 0.0:
                        gate_grad = layer_grad[group_idx, gate_idx, :]
                        assert jp.allclose(gate_grad, 0.0, atol=1e-6), (
                            f"Non-zero gradient for knocked out gate at layer {layer_idx}, group {group_idx}, gate {gate_idx}"
                        )

    def test_multiple_training_steps_preserve_knockouts(self, setup_circuit_and_data):
        """Test that multiple training steps preserve knocked out gates."""
        data = setup_circuit_and_data
        key = jax.random.PRNGKey(161718)

        # Create knockout pattern and apply it
        knockout_masks = create_reproducible_knockout_pattern(
            key, data["layer_sizes"], number_knokouts=3, input_n=data["layer_sizes"][0][0]
        )

        faulty_logits, _, _ = apply_gate_knockout_to_pool_element(
            key, data["logits"], data["wires"], data["layer_sizes"], number_knockouts=3
        )

        # Store original faulty values
        original_faulty_logits = [layer.copy() for layer in faulty_logits]

        # Create training state
        optimizer = optax.adam(learning_rate=0.01)
        opt_state = optimizer.init(faulty_logits)
        train_state = TrainState(faulty_logits, opt_state)

        # Perform multiple training steps
        for step in range(20):
            loss, aux, train_state = train_step(
                train_state,
                optimizer,
                data["wires"],
                data["x"],
                data["y0"],
                loss_type="l4",
                do_train=True,
                gate_mask=knockout_masks,
            )

            # Verify knocked out gates are still preserved at each step
            current_logits = train_state.params
            for layer_idx, (current_layer, original_layer, layer_mask) in enumerate(
                zip(current_logits, original_faulty_logits, knockout_masks[1:])
            ):
                group_n, group_size, lut_size = current_layer.shape
                mask_reshaped = layer_mask.reshape(group_n, group_size)

                for group_idx in range(group_n):
                    for gate_idx in range(group_size):
                        if mask_reshaped[group_idx, gate_idx] == 0.0:
                            current_gate = current_layer[group_idx, gate_idx, :]
                            original_gate = original_layer[group_idx, gate_idx, :]

                            assert jp.allclose(current_gate, original_gate, atol=1e-6), (
                                f"Knocked out gate changed at step {step}, layer {layer_idx}, group {group_idx}, gate {gate_idx}"
                            )
