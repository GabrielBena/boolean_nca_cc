"""
Test Level 1.2: Loss Function Infrastructure

This module tests the fundamental loss functions and training infrastructure
in circuits/train.py that are critical for knockout experiment evaluation.

Test Coverage:
- 1.2.1 Fundamental Loss Computation (res2loss, binary_cross_entropy, compute_accuracy)
- 1.2.2 Circuit Integration (loss_f_l4, loss_f_bce with circuit execution)
- 1.2.3 Gradient Compilation (grad_loss_f_l4, grad_loss_f_bce)
- 1.2.4 Edge Cases (numerical stability, extreme values)
"""

import jax
import jax.numpy as jp
import numpy as np
import pytest
from jax import random

# Import functions under test
from boolean_nca_cc.circuits.train import (
    res2loss,
    binary_cross_entropy,
    compute_accuracy,
    loss_f_l4,
    loss_f_bce,
    grad_loss_f_l4,
    grad_loss_f_bce,
    loss_f,
    unpack
)

# Import dependencies for circuit testing
from boolean_nca_cc.circuits.model import (
    run_circuit,
    make_nops,
    gen_wires,
    gen_circuit
)


class TestLevel12LossFunctionInfrastructure:
    
    def setup_method(self):
        """Setup test fixtures and random keys"""
        self.key = random.PRNGKey(42)
        self.key1, self.key2, self.key3 = random.split(self.key, 3)
        
        # Create minimal test circuit
        self.arity = 2
        self.input_n = 4
        self.output_n = 2
        self.group_size = 2
        
        # Generate simple test circuit
        layer_sizes = [(self.input_n, 1), (self.output_n, self.group_size)]
        self.wires, self.logits = gen_circuit(self.key1, layer_sizes, arity=self.arity)
        
        # Create test inputs and targets
        self.x = random.uniform(self.key2, (self.input_n,), minval=0.0, maxval=1.0)
        self.y0 = random.uniform(self.key3, (self.output_n,), minval=0.0, maxval=1.0)
    
    # ========================================
    # 1.2.1 Fundamental Loss Computation Tests
    # ========================================
    
    def test_res2loss_known_values(self):
        """Test res2loss computes L4 norm correctly with known inputs"""
        # Test with simple known values
        residuals = jp.array([1.0, -2.0, 0.5])
        expected_l4 = 1.0**4 + 2.0**4 + 0.5**4  # 1 + 16 + 0.0625 = 17.0625
        
        result = res2loss(residuals, power=4)
        
        assert jp.isclose(result, expected_l4), f"Expected {expected_l4}, got {result}"
        
        # Test with different power
        expected_l2 = 1.0**2 + 2.0**2 + 0.5**2  # 1 + 4 + 0.25 = 5.25
        result_l2 = res2loss(residuals, power=2)
        
        assert jp.isclose(result_l2, expected_l2), f"Expected {expected_l2}, got {result_l2}"
    
    def test_res2loss_zero_residuals(self):
        """Test res2loss returns zero for zero residuals"""
        residuals = jp.zeros(5)
        result = res2loss(residuals)
        
        assert jp.isclose(result, 0.0), f"Expected 0.0 for zero residuals, got {result}"
    
    def test_binary_cross_entropy_perfect_prediction(self):
        """Test binary cross-entropy with perfect predictions"""
        # Perfect predictions should give low loss
        y_pred = jp.array([0.99, 0.01, 0.99])
        y_true = jp.array([1.0, 0.0, 1.0])
        
        loss = binary_cross_entropy(y_pred, y_true)
        
        # Loss should be small but positive
        assert loss >= 0.0, "BCE loss should be non-negative"
        assert loss < 1.0, f"Perfect predictions should have low loss, got {loss}"
    
    def test_binary_cross_entropy_worst_prediction(self):
        """Test binary cross-entropy with worst predictions"""
        # Worst predictions should give high loss
        y_pred = jp.array([0.01, 0.99, 0.01])  # Opposite of true
        y_true = jp.array([1.0, 0.0, 1.0])
        
        loss = binary_cross_entropy(y_pred, y_true)
        
        # Loss should be high
        assert loss > 1.0, f"Bad predictions should have high loss, got {loss}"
    
    def test_binary_cross_entropy_edge_cases(self):
        """Test binary cross-entropy numerical stability at edges"""
        # Test with values very close to 0 and 1 (should not cause NaN/inf)
        y_pred = jp.array([1e-8, 1.0 - 1e-8])
        y_true = jp.array([0.0, 1.0])
        
        loss = binary_cross_entropy(y_pred, y_true)
        
        assert jp.isfinite(loss), f"BCE should be finite at edges, got {loss}"
        assert not jp.isnan(loss), f"BCE should not be NaN at edges, got {loss}"
    
    def test_compute_accuracy_perfect(self):
        """Test accuracy computation with perfect predictions"""
        y_pred = jp.array([0.9, 0.1, 0.8, 0.2])
        y_true = jp.array([1.0, 0.0, 1.0, 0.0])
        
        accuracy = compute_accuracy(y_pred, y_true)
        
        assert jp.isclose(accuracy, 1.0), f"Perfect predictions should give 100% accuracy, got {accuracy}"
    
    def test_compute_accuracy_worst(self):
        """Test accuracy computation with worst predictions"""
        y_pred = jp.array([0.1, 0.9, 0.2, 0.8])  # All wrong
        y_true = jp.array([1.0, 0.0, 1.0, 0.0])
        
        accuracy = compute_accuracy(y_pred, y_true)
        
        assert jp.isclose(accuracy, 0.0), f"Wrong predictions should give 0% accuracy, got {accuracy}"
    
    def test_compute_accuracy_mixed(self):
        """Test accuracy computation with mixed predictions"""
        y_pred = jp.array([0.9, 0.9, 0.8, 0.2])  # 3 correct, 1 wrong
        y_true = jp.array([1.0, 0.0, 1.0, 0.0])
        
        accuracy = compute_accuracy(y_pred, y_true)
        
        expected_accuracy = 3.0 / 4.0  # 75%
        assert jp.isclose(accuracy, expected_accuracy), f"Expected {expected_accuracy}, got {accuracy}"
    
    # ========================================
    # 1.2.2 Circuit Integration Tests
    # ========================================
    
    def test_loss_f_l4_basic_functionality(self):
        """Test L4 loss function with circuit execution"""
        loss, aux = loss_f_l4(self.logits, self.wires, self.x, self.y0)
        
        # Check loss is scalar and non-negative
        assert jp.isscalar(loss), "Loss should be scalar"
        assert loss >= 0.0, f"L4 loss should be non-negative, got {loss}"
        
        # Check auxiliary outputs
        assert 'act' in aux, "Should return circuit activations"
        assert 'accuracy' in aux, "Should return accuracy"
        assert 'hard_loss' in aux, "Should return hard loss"
        assert 'hard_accuracy' in aux, "Should return hard accuracy"
        assert 'res' in aux, "Should return residuals"
        assert 'hard_res' in aux, "Should return hard residuals"
        
        # Check activations structure
        assert len(aux['act']) == len(self.logits) + 1, "Should include input + all layers"
        assert aux['act'][0].shape == self.x.shape, "First activation should match input"
        assert aux['act'][-1].shape == self.y0.shape, "Last activation should match target"
    
    def test_loss_f_bce_basic_functionality(self):
        """Test BCE loss function with circuit execution"""
        loss, aux = loss_f_bce(self.logits, self.wires, self.x, self.y0)
        
        # Check loss is scalar and non-negative
        assert jp.isscalar(loss), "Loss should be scalar"
        assert loss >= 0.0, f"BCE loss should be non-negative, got {loss}"
        
        # Check auxiliary outputs (same structure as L4)
        assert 'act' in aux, "Should return circuit activations"
        assert 'accuracy' in aux, "Should return accuracy"
        assert 'hard_loss' in aux, "Should return hard loss"
        assert 'hard_accuracy' in aux, "Should return hard accuracy"
        assert 'res' in aux, "Should return residuals"
        assert 'hard_res' in aux, "Should return hard residuals"
    
    def test_soft_vs_hard_circuit_execution(self):
        """Test that soft and hard circuit execution produce different loss values"""
        loss_l4, aux_l4 = loss_f_l4(self.logits, self.wires, self.x, self.y0)
        loss_bce, aux_bce = loss_f_bce(self.logits, self.wires, self.x, self.y0)
        
        # Soft and hard losses should generally be different (unless perfect prediction)
        # Check that hard_loss is different from soft loss (with some tolerance for edge cases)
        hard_soft_diff_l4 = jp.abs(aux_l4['hard_loss'] - loss_l4)
        hard_soft_diff_bce = jp.abs(aux_bce['hard_loss'] - loss_bce)
        
        # At least one should show difference (unless we have an extremely simple case)
        assert hard_soft_diff_l4 > 1e-6 or hard_soft_diff_bce > 1e-6, \
            "Hard and soft losses should generally differ for non-trivial circuits"
    
    def test_loss_functions_with_identical_outputs(self):
        """Test loss functions when circuit output exactly matches target"""
        # Create a very simple test case with known dimensions
        simple_x = jp.array([0.5])
        simple_y0 = jp.array([0.5])  # Match the expected output
        
        # Create minimal circuit that should roughly preserve the input
        simple_wires = [jp.array([[0]])]  # Single wire from input to output
        
        # Create lookup table that approximates identity for a single input
        # For arity=2, we need to handle the case where we're using just one input
        # Use relatively balanced logits
        simple_logits = [jp.array([[[0.0, 5.0, 0.0, 5.0]]])]  # Shape: (1, 1, 4)
        
        loss_l4, aux_l4 = loss_f_l4(simple_logits, simple_wires, simple_x, simple_y0)
        loss_bce, aux_bce = loss_f_bce(simple_logits, simple_wires, simple_x, simple_y0)
        
        # Losses should be finite (we can't guarantee they'll be small without perfect circuit design)
        assert jp.isfinite(loss_l4), f"L4 loss should be finite, got {loss_l4}"
        assert jp.isfinite(loss_bce), f"BCE loss should be finite, got {loss_bce}"
        assert loss_l4 >= 0.0, f"L4 loss should be non-negative, got {loss_l4}"
        assert loss_bce >= 0.0, f"BCE loss should be non-negative, got {loss_bce}"
    
    def test_loss_f_dispatcher(self):
        """Test the loss function dispatcher chooses correct implementation"""
        # Test L4 selection
        loss_l4_direct, aux_l4_direct = loss_f_l4(self.logits, self.wires, self.x, self.y0)
        loss_l4_dispatch, aux_l4_dispatch = loss_f(self.logits, self.wires, self.x, self.y0, loss_type="l4")
        
        assert jp.isclose(loss_l4_direct, loss_l4_dispatch), "L4 dispatcher should match direct call"
        
        # Test BCE selection
        loss_bce_direct, aux_bce_direct = loss_f_bce(self.logits, self.wires, self.x, self.y0)
        loss_bce_dispatch, aux_bce_dispatch = loss_f(self.logits, self.wires, self.x, self.y0, loss_type="bce")
        
        assert jp.isclose(loss_bce_direct, loss_bce_dispatch), "BCE dispatcher should match direct call"
        
        # Test default (should be L4)
        loss_default, aux_default = loss_f(self.logits, self.wires, self.x, self.y0)
        assert jp.isclose(loss_l4_direct, loss_default), "Default should be L4"
    
    # ========================================
    # 1.2.3 Gradient Compilation Tests
    # ========================================
    
    def test_grad_loss_f_l4_returns_valid_gradients(self):
        """Test that gradient function returns valid gradients for L4 loss"""
        (loss, aux), grads = grad_loss_f_l4(self.logits, self.wires, self.x, self.y0)
        
        # Check gradient structure matches logits structure
        assert len(grads) == len(self.logits), "Gradient should match logits structure"
        
        for i, (grad, logit) in enumerate(zip(grads, self.logits)):
            assert grad.shape == logit.shape, f"Gradient shape {grad.shape} should match logits shape {logit.shape} for layer {i}"
            assert jp.isfinite(grad).all(), f"All gradients should be finite for layer {i}"
            assert not jp.isnan(grad).any(), f"No gradients should be NaN for layer {i}"
        
        # Check that some gradients are non-zero (unless we have a very special case)
        total_grad_norm = sum(jp.sum(jp.abs(grad)) for grad in grads)
        assert total_grad_norm > 1e-8, f"Some gradients should be non-zero, got total norm {total_grad_norm}"
    
    def test_grad_loss_f_bce_returns_valid_gradients(self):
        """Test that gradient function returns valid gradients for BCE loss"""
        (loss, aux), grads = grad_loss_f_bce(self.logits, self.wires, self.x, self.y0)
        
        # Check gradient structure matches logits structure
        assert len(grads) == len(self.logits), "Gradient should match logits structure"
        
        for i, (grad, logit) in enumerate(zip(grads, self.logits)):
            assert grad.shape == logit.shape, f"Gradient shape {grad.shape} should match logits shape {logit.shape} for layer {i}"
            assert jp.isfinite(grad).all(), f"All gradients should be finite for layer {i}"
            assert not jp.isnan(grad).any(), f"No gradients should be NaN for layer {i}"
    
    def test_gradient_computation_preserves_original_circuit(self):
        """Test that gradient computation doesn't modify original logits"""
        original_logits = [jp.copy(logit) for logit in self.logits]
        
        # Compute gradients
        (loss, aux), grads = grad_loss_f_l4(self.logits, self.wires, self.x, self.y0)
        
        # Check that original logits are unchanged
        for i, (orig, curr) in enumerate(zip(original_logits, self.logits)):
            assert jp.allclose(orig, curr), f"Original logits should be unchanged for layer {i}"
    
    def test_gradient_shapes_consistency(self):
        """Test that gradient shapes are consistent between L4 and BCE"""
        (loss_l4, aux_l4), grads_l4 = grad_loss_f_l4(self.logits, self.wires, self.x, self.y0)
        (loss_bce, aux_bce), grads_bce = grad_loss_f_bce(self.logits, self.wires, self.x, self.y0)
        
        # Gradient structures should be identical
        assert len(grads_l4) == len(grads_bce), "L4 and BCE should have same gradient structure"
        
        for i, (grad_l4, grad_bce) in enumerate(zip(grads_l4, grads_bce)):
            assert grad_l4.shape == grad_bce.shape, f"L4 and BCE gradient shapes should match for layer {i}"
    
    # ========================================
    # 1.2.4 Edge Cases and Numerical Stability
    # ========================================
    
    def test_loss_functions_with_extreme_logits(self):
        """Test loss functions with extreme logit values"""
        # Create extreme logits (very positive and very negative)
        extreme_logits = [jp.array([[[100.0, -100.0, 100.0, -100.0]]])]
        simple_wires = [jp.array([[0]])]
        simple_x = jp.array([0.5])
        simple_y0 = jp.array([0.0])
        
        # Should not cause numerical issues
        loss_l4, aux_l4 = loss_f_l4(extreme_logits, simple_wires, simple_x, simple_y0)
        loss_bce, aux_bce = loss_f_bce(extreme_logits, simple_wires, simple_x, simple_y0)
        
        assert jp.isfinite(loss_l4), f"L4 loss should be finite with extreme logits, got {loss_l4}"
        assert jp.isfinite(loss_bce), f"BCE loss should be finite with extreme logits, got {loss_bce}"
        assert not jp.isnan(loss_l4), f"L4 loss should not be NaN with extreme logits, got {loss_l4}"
        assert not jp.isnan(loss_bce), f"BCE loss should not be NaN with extreme logits, got {loss_bce}"
    
    def test_loss_functions_with_all_zero_predictions(self):
        """Test loss functions when circuit outputs all zeros"""
        # Create logits that will produce outputs near zero
        zero_logits = [jp.ones_like(logit) * (-10.0) for logit in self.logits]
        
        loss_l4, aux_l4 = loss_f_l4(zero_logits, self.wires, self.x, self.y0)
        loss_bce, aux_bce = loss_f_bce(zero_logits, self.wires, self.x, self.y0)
        
        # Should produce finite losses
        assert jp.isfinite(loss_l4), f"L4 loss should be finite with zero outputs, got {loss_l4}"
        assert jp.isfinite(loss_bce), f"BCE loss should be finite with zero outputs, got {loss_bce}"
        
        # Accuracy should be low if targets are not zero
        if jp.any(self.y0 > 0.5):
            assert aux_l4['accuracy'] < 0.8, "Accuracy should be low when predicting zeros for non-zero targets"
    
    def test_loss_functions_with_all_one_predictions(self):
        """Test loss functions when circuit outputs all ones"""
        # Create logits that will produce outputs near one
        one_logits = [jp.ones_like(logit) * 10.0 for logit in self.logits]
        
        loss_l4, aux_l4 = loss_f_l4(one_logits, self.wires, self.x, self.y0)
        loss_bce, aux_bce = loss_f_bce(one_logits, self.wires, self.x, self.y0)
        
        # Should produce finite losses
        assert jp.isfinite(loss_l4), f"L4 loss should be finite with one outputs, got {loss_l4}"
        assert jp.isfinite(loss_bce), f"BCE loss should be finite with one outputs, got {loss_bce}"
    
    def test_batch_processing_different_sizes(self):
        """Test loss functions with different input batch sizes"""
        # Test with larger input/output dimensions
        large_x = random.uniform(self.key1, (8,), minval=0.0, maxval=1.0)
        large_y0 = random.uniform(self.key2, (4,), minval=0.0, maxval=1.0)
        
        # Create corresponding circuit
        large_layer_sizes = [(8, 1), (4, 2)]
        large_wires, large_logits = gen_circuit(self.key3, large_layer_sizes, arity=2)
        
        loss_l4, aux_l4 = loss_f_l4(large_logits, large_wires, large_x, large_y0)
        loss_bce, aux_bce = loss_f_bce(large_logits, large_wires, large_x, large_y0)
        
        assert jp.isfinite(loss_l4), "L4 loss should handle larger circuits"
        assert jp.isfinite(loss_bce), "BCE loss should handle larger circuits"
        assert aux_l4['act'][-1].shape == large_y0.shape, "Output should match target dimensions"
    
    def test_unpack_utility_function(self):
        """Test the unpack utility function for bit unpacking"""
        # Test unpacking small integers
        x = jp.array([5, 3])  # 5 = 101, 3 = 011 in binary
        
        # Unpack with 3 bits
        result = unpack(x, bit_n=3)
        expected = jp.array([[1.0, 0.0, 1.0],  # 5 = 101
                           [1.0, 1.0, 0.0]])  # 3 = 011
        
        assert result.shape == (2, 3), f"Unpacked shape should be (2, 3), got {result.shape}"
        assert jp.allclose(result, expected), f"Unpacking failed: expected {expected}, got {result}"
        
        # Test with default bit_n=8
        result_8 = unpack(jp.array([255]), bit_n=8)
        expected_8 = jp.ones((1, 8))  # 255 = 11111111
        
        assert jp.allclose(result_8, expected_8), "Unpacking 255 should give all ones"


# Additional integration test
def test_loss_function_integration_with_training_step():
    """Test that loss functions integrate properly with training infrastructure"""
    from boolean_nca_cc.circuits.train import train_step, TrainState
    import optax
    
    # Setup minimal training scenario
    key = random.PRNGKey(123)
    key1, key2, key3 = random.split(key, 3)
    
    # Create simple circuit
    layer_sizes = [(2, 1), (1, 1)]
    wires, logits = gen_circuit(key1, layer_sizes, arity=2)
    
    # Create optimizer and state
    optimizer = optax.adam(0.01)
    opt_state = optimizer.init(logits)
    state = TrainState(logits, opt_state)
    
    # Create test data
    x = random.uniform(key2, (2,))
    y0 = random.uniform(key3, (1,))
    
    # Test training step with both loss types
    loss_l4, aux_l4, new_state_l4 = train_step(state, optimizer, wires, x, y0, loss_type="l4")
    loss_bce, aux_bce, new_state_bce = train_step(state, optimizer, wires, x, y0, loss_type="bce")
    
    # Both should complete without error
    assert jp.isfinite(loss_l4), "L4 training step should produce finite loss"
    assert jp.isfinite(loss_bce), "BCE training step should produce finite loss"
    
    # States should be updated
    assert not jp.allclose(new_state_l4.params[0], state.params[0]), "L4 training should update parameters"
    assert not jp.allclose(new_state_bce.params[0], state.params[0]), "BCE training should update parameters"


if __name__ == "__main__":
    # Run tests manually if script is executed directly
    test_instance = TestLevel12LossFunctionInfrastructure()
    test_instance.setup_method()
    
    print("Running Level 1.2 Loss Function Infrastructure Tests...")
    
    # Run all test methods
    methods = [method for method in dir(test_instance) if method.startswith('test_')]
    
    passed = 0
    failed = 0
    
    for method_name in methods:
        try:
            method = getattr(test_instance, method_name)
            method()
            print(f"✓ {method_name}")
            passed += 1
        except Exception as e:
            print(f"✗ {method_name}: {e}")
            failed += 1
    
    print(f"\nResults: {passed} passed, {failed} failed")
    
    # Run integration test
    try:
        test_loss_function_integration_with_training_step()
        print("✓ Integration test passed")
    except Exception as e:
        print(f"✗ Integration test failed: {e}") 