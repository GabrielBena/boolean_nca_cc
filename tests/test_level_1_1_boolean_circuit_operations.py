"""
Level 1.1: Boolean Circuit Operations Testing

This module tests the foundational boolean circuit operations from circuits/model.py.
These tests validate the core functions that create and execute boolean circuits.

JAX Fundamentals:
- Functional Programming: All functions are pure with explicit random keys
- JIT Compilation: run_layer is JIT compiled for performance
- Vectorization: Functions handle batched operations naturally
- Immutable Data: All operations return new arrays without side effects
"""

import pytest
import jax
import jax.numpy as jp
import numpy as np
from boolean_nca_cc.circuits.model import (
    make_nops, run_layer, gen_wires, gen_wires_with_noise,
    gen_circuit, run_circuit, generate_layer_sizes
)


class TestMakeNops:
    """Test make_nops function - creates lookup table logits for boolean operations."""
    
    def test_make_nops_basic_shape(self):
        """Test make_nops produces correct shape for basic parameters."""
        gate_n = 8
        arity = 2
        group_size = 4
        nop_scale = 3.0
        
        logits = make_nops(gate_n, arity, group_size, nop_scale)
        
        # Expected shape: (gate_n//group_size, group_size, 2^arity)
        expected_shape = (2, 4, 4)  # 8//4=2, 4, 2^2=4
        assert logits.shape == expected_shape
        assert isinstance(logits, jp.ndarray)
    
    def test_make_nops_arity_3(self):
        """Test make_nops with arity=3 (8 possible input combinations)."""
        gate_n = 6
        arity = 3
        group_size = 2
        nop_scale = 2.0
        
        logits = make_nops(gate_n, arity, group_size, nop_scale)
        
        expected_shape = (3, 2, 8)  # 6//2=3, 2, 2^3=8
        assert logits.shape == expected_shape
    
    def test_make_nops_nop_scale_effect(self):
        """Test that nop_scale affects the magnitude of logits."""
        gate_n = 4
        arity = 2
        group_size = 2
        
        logits_scale_1 = make_nops(gate_n, arity, group_size, nop_scale=1.0)
        logits_scale_5 = make_nops(gate_n, arity, group_size, nop_scale=5.0)
        
        # Higher scale should produce larger magnitude values
        assert jp.max(jp.abs(logits_scale_5)) > jp.max(jp.abs(logits_scale_1))
    
    def test_make_nops_deterministic(self):
        """Test make_nops produces identical results for same parameters."""
        gate_n = 6
        arity = 2
        group_size = 3
        
        logits1 = make_nops(gate_n, arity, group_size)
        logits2 = make_nops(gate_n, arity, group_size)
        
        # Should be identical (no randomness in this function)
        assert jp.allclose(logits1, logits2)
    
    def test_make_nops_logits_range(self):
        """Test that logits are in expected range after sigmoid scaling."""
        gate_n = 4
        arity = 2
        group_size = 2
        nop_scale = 3.0
        
        logits = make_nops(gate_n, arity, group_size, nop_scale)
        
        # Logits should be in range [-nop_scale, nop_scale]
        assert jp.all(logits >= -nop_scale)
        assert jp.all(logits <= nop_scale)


class TestRunLayer:
    """Test run_layer function - executes boolean lookup tables."""
    
    def test_run_layer_basic_2_input(self):
        """Test run_layer with 2-input gates (arity=2)."""
        # Create a simple 2-input XOR lookup table
        # LUT for XOR: [0, 1, 1, 0] (inputs: 00->0, 01->1, 10->1, 11->0)
        lut = jp.array([[[0.0, 1.0, 1.0, 0.0]]])  # Shape: (1, 1, 4)
        inputs = [jp.array([1.0]), jp.array([0.0])]  # Inputs: A=1, B=0
        
        output = run_layer(lut, inputs)
        
        # For inputs A=1, B=0, should get output 1 (XOR)
        expected = jp.array([[1.0]])
        assert jp.allclose(output, expected, atol=1e-6)
    
    def test_run_layer_and_gate(self):
        """Test run_layer with AND gate lookup table."""
        # LUT for AND: [0, 0, 0, 1] (inputs: 00->0, 01->0, 10->0, 11->1)
        lut = jp.array([[[0.0, 0.0, 0.0, 1.0]]])
        
        # Test all input combinations
        test_cases = [
            ([0.0, 0.0], 0.0),  # 00 -> 0
            ([0.0, 1.0], 0.0),  # 01 -> 0
            ([1.0, 0.0], 0.0),  # 10 -> 0
            ([1.0, 1.0], 1.0),  # 11 -> 1
        ]
        
        for inputs, expected_output in test_cases:
            input_tensors = [jp.array([inputs[0]]), jp.array([inputs[1]])]
            output = run_layer(lut, input_tensors)
            assert jp.allclose(output, jp.array([[expected_output]]), atol=1e-6)
    
    def test_run_layer_or_gate(self):
        """Test run_layer with OR gate lookup table."""
        # LUT for OR: [0, 1, 1, 1] (inputs: 00->0, 01->1, 10->1, 11->1)
        lut = jp.array([[[0.0, 1.0, 1.0, 1.0]]])
        
        # Test all input combinations
        test_cases = [
            ([0.0, 0.0], 0.0),  # 00 -> 0
            ([0.0, 1.0], 1.0),  # 01 -> 1
            ([1.0, 0.0], 1.0),  # 10 -> 1
            ([1.0, 1.0], 1.0),  # 11 -> 1
        ]
        
        for inputs, expected_output in test_cases:
            input_tensors = [jp.array([inputs[0]]), jp.array([inputs[1]])]
            output = run_layer(lut, input_tensors)
            assert jp.allclose(output, jp.array([[expected_output]]), atol=1e-6)
    
    def test_run_layer_batched_inputs(self):
        """Test run_layer with batched inputs."""
        # LUT for XOR
        lut = jp.array([[[0.0, 1.0, 1.0, 0.0]]])
        
        # Batch of inputs: [[0,1], [1,0], [1,1]]
        input_a = jp.array([0.0, 1.0, 1.0])
        input_b = jp.array([1.0, 0.0, 1.0])
        inputs = [input_a, input_b]
        
        output = run_layer(lut, inputs)
        
        # Expected outputs for XOR: [1, 1, 0]
        expected = jp.array([[1.0, 1.0, 0.0]])
        assert jp.allclose(output, expected, atol=1e-6)
    
    def test_run_layer_3_input_gate(self):
        """Test run_layer with 3-input gate (arity=3)."""
        # LUT for 3-input AND: [0,0,0,0,0,0,0,1] (only 111->1)
        lut = jp.array([[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]])
        
        # Test 3-input AND: inputs [1,1,1] should give output 1
        inputs = [jp.array([1.0]), jp.array([1.0]), jp.array([1.0])]
        output = run_layer(lut, inputs)
        assert jp.allclose(output, jp.array([[1.0]]), atol=1e-6)
        
        # Test 3-input AND: inputs [1,0,1] should give output 0
        inputs = [jp.array([1.0]), jp.array([0.0]), jp.array([1.0])]
        output = run_layer(lut, inputs)
        assert jp.allclose(output, jp.array([[0.0]]), atol=1e-6)
    
    def test_run_layer_jit_compilation(self):
        """Test that run_layer is properly JIT compiled."""
        lut = jp.array([[[0.0, 1.0, 1.0, 0.0]]])  # XOR
        inputs = [jp.array([1.0]), jp.array([0.0])]
        
        # First call should compile, second should use compiled version
        output1 = run_layer(lut, inputs)
        output2 = run_layer(lut, inputs)
        
        assert jp.allclose(output1, output2)
        # Verify the function is actually JIT compiled
        assert hasattr(run_layer, '__wrapped__')  # JIT adds this attribute


class TestGenWires:
    """Test gen_wires function - generates random wiring connections."""
    
    def test_gen_wires_basic_shape(self):
        """Test gen_wires produces correct shape."""
        key = jax.random.PRNGKey(42)
        in_n = 4
        out_n = 6
        arity = 2
        group_size = 2
        
        wires = gen_wires(key, in_n, out_n, arity, group_size)
        
        # Expected shape: (arity, out_n//group_size)
        expected_shape = (2, 3)  # arity=2, out_n//group_size=6//2=3
        assert wires.shape == expected_shape
        assert isinstance(wires, jp.ndarray)
    
    def test_gen_wires_indices_range(self):
        """Test that wire indices are within valid range."""
        key = jax.random.PRNGKey(42)
        in_n = 4
        out_n = 8
        arity = 2
        group_size = 2
        
        wires = gen_wires(key, in_n, out_n, arity, group_size)
        
        # All indices should be in range [0, in_n)
        assert jp.all(wires >= 0)
        assert jp.all(wires < in_n)
    
    def test_gen_wires_deterministic_with_key(self):
        """Test gen_wires produces identical results with same key."""
        key = jax.random.PRNGKey(42)
        in_n = 4
        out_n = 6
        arity = 2
        group_size = 2
        
        wires1 = gen_wires(key, in_n, out_n, arity, group_size)
        wires2 = gen_wires(key, in_n, out_n, arity, group_size)
        
        assert jp.allclose(wires1, wires2)
    
    def test_gen_wires_different_with_different_keys(self):
        """Test gen_wires produces different results with different keys."""
        key1 = jax.random.PRNGKey(42)
        key2 = jax.random.PRNGKey(43)
        in_n = 4
        out_n = 6
        arity = 2
        group_size = 2
        
        wires1 = gen_wires(key1, in_n, out_n, arity, group_size)
        wires2 = gen_wires(key2, in_n, out_n, arity, group_size)
        
        # Should be different (very unlikely to be identical)
        assert not jp.allclose(wires1, wires2)
    
    def test_gen_wires_edge_cases(self):
        """Test gen_wires with edge cases."""
        key = jax.random.PRNGKey(42)
        
        # Case 1: in_n < out_n
        wires = gen_wires(key, in_n=2, out_n=4, arity=2, group_size=2)
        assert wires.shape == (2, 2)
        assert jp.all(wires < 2)
        
        # Case 2: in_n == out_n
        wires = gen_wires(key, in_n=4, out_n=4, arity=2, group_size=2)
        assert wires.shape == (2, 2)
        assert jp.all(wires < 4)


class TestGenWiresWithNoise:
    """Test gen_wires_with_noise function - generates wiring with locality bias."""
    
    def test_gen_wires_with_noise_basic(self):
        """Test gen_wires_with_noise with no local noise (should behave like gen_wires)."""
        key = jax.random.PRNGKey(42)
        in_n = 4
        out_n = 6
        arity = 2
        group_size = 2
        
        wires_no_noise = gen_wires_with_noise(key, in_n, out_n, arity, group_size, local_noise=None)
        wires_regular = gen_wires(key, in_n, out_n, arity, group_size)
        
        # Should be identical when local_noise is None
        assert jp.allclose(wires_no_noise, wires_regular)
    
    def test_gen_wires_with_noise_local_bias(self):
        """Test gen_wires_with_noise with local noise bias."""
        key = jax.random.PRNGKey(42)
        in_n = 8
        out_n = 8
        arity = 2
        group_size = 2
        local_noise = 0.1
        
        wires_local = gen_wires_with_noise(key, in_n, out_n, arity, group_size, local_noise)
        wires_regular = gen_wires(key, in_n, out_n, arity, group_size)
        
        # Should have same shape
        assert wires_local.shape == wires_regular.shape
        assert jp.all(wires_local < in_n)
        assert jp.all(wires_local >= 0)
    
    def test_gen_wires_with_noise_deterministic(self):
        """Test gen_wires_with_noise is deterministic with same key."""
        key = jax.random.PRNGKey(42)
        in_n = 6
        out_n = 6
        arity = 2
        group_size = 2
        local_noise = 0.5
        
        wires1 = gen_wires_with_noise(key, in_n, out_n, arity, group_size, local_noise)
        wires2 = gen_wires_with_noise(key, in_n, out_n, arity, group_size, local_noise)
        
        assert jp.allclose(wires1, wires2)


class TestGenCircuit:
    """Test gen_circuit function - generates complete circuits."""
    
    def test_gen_circuit_basic(self):
        """Test gen_circuit generates valid circuit structure."""
        key = jax.random.PRNGKey(42)
        layer_sizes = [(4, 1), (6, 2), (2, 1)]  # Input, hidden, output layers
        arity = 2
        
        wires, logits = gen_circuit(key, layer_sizes, arity)
        
        # Should have one set of wires/logits per layer (excluding input)
        assert len(wires) == 2  # 2 layers after input
        assert len(logits) == 2
        
        # Check shapes
        assert wires[0].shape == (arity, 3)  # arity=2, out_n//group_size=6//2=3
        assert wires[1].shape == (arity, 2)  # arity=2, out_n//group_size=2//1=2
        
        assert logits[0].shape == (3, 2, 4)  # group_n=3, group_size=2, 2^arity=4
        assert logits[1].shape == (2, 1, 4)  # group_n=2, group_size=1, 2^arity=4
    
    def test_gen_circuit_deterministic(self):
        """Test gen_circuit is deterministic with same key."""
        key = jax.random.PRNGKey(42)
        layer_sizes = [(4, 1), (6, 2), (2, 1)]
        arity = 2
        
        wires1, logits1 = gen_circuit(key, layer_sizes, arity)
        wires2, logits2 = gen_circuit(key, layer_sizes, arity)
        
        # Should be identical
        for w1, w2 in zip(wires1, wires2):
            assert jp.allclose(w1, w2)
        for l1, l2 in zip(logits1, logits2):
            assert jp.allclose(l1, l2)
    
    def test_gen_circuit_custom_init_function(self):
        """Test gen_circuit with custom initialization function."""
        key = jax.random.PRNGKey(42)
        layer_sizes = [(4, 1), (6, 2)]
        arity = 2
        
        def custom_init(gate_n, arity, group_size):
            # Custom initialization that returns zeros
            return jp.zeros((gate_n // group_size, group_size, 1 << arity))
        
        wires, logits = gen_circuit(key, layer_sizes, arity, init_logits_fn=custom_init)
        
        # Check that custom initialization was used
        assert jp.allclose(logits[0], jp.zeros((3, 2, 4)))
    
    def test_gen_circuit_verbose_mode(self):
        """Test gen_circuit verbose mode (should not crash)."""
        key = jax.random.PRNGKey(42)
        layer_sizes = [(4, 1), (6, 2), (2, 1)]
        arity = 2
        
        # Should not raise any exceptions
        wires, logits = gen_circuit(key, layer_sizes, arity, verbose=True)
        assert len(wires) == 2
        assert len(logits) == 2


class TestRunCircuit:
    """Test run_circuit function - executes complete boolean circuits."""
    
    def test_run_circuit_basic(self):
        """Test run_circuit with simple 2-layer circuit."""
        key = jax.random.PRNGKey(42)
        layer_sizes = [(2, 1), (1, 1)]
        arity = 2
        
        wires, logits = gen_circuit(key, layer_sizes, arity)
        x = jp.array([1.0, 0.0])
        acts = run_circuit(logits, wires, x)
        # Should have activations for input + each layer (len(layer_sizes))
        assert len(acts) == len(layer_sizes)
        # Check shapes
        for i, (act, (size, _)) in enumerate(zip(acts, layer_sizes)):
            assert act.shape == (size,)
    
    def test_run_circuit_with_gate_mask(self):
        key = jax.random.PRNGKey(42)
        layer_sizes = [(2, 1), (1, 1)]
        arity = 2
        wires, logits = gen_circuit(key, layer_sizes, arity)
        x = jp.array([1.0, 0.0])
        gate_mask = [jp.ones_like(x)] + [jp.zeros(s) for (s, _) in layer_sizes[1:]]
        acts = run_circuit(logits, wires, x, gate_mask=gate_mask)
        # Input unchanged
        assert jp.allclose(acts[0], x)
        # All subsequent activations zeroed
        for act in acts[1:]:
            assert jp.allclose(act, jp.zeros_like(act))
    
    def test_run_circuit_hard_outputs(self):
        key = jax.random.PRNGKey(42)
        layer_sizes = [(2, 1), (1, 1)]
        arity = 2
        wires, logits = gen_circuit(key, layer_sizes, arity)
        x = jp.array([1.0, 0.0])
        acts_soft = run_circuit(logits, wires, x, hard=False)
        acts_hard = run_circuit(logits, wires, x, hard=True)
        # Hard outputs should be binary (0 or 1) for all non-input activations
        for act in acts_hard[1:]:
            assert jp.all(jp.logical_or(act == 0, act == 1))
        # Soft outputs should be in [0, 1] for all non-input activations
        for act in acts_soft[1:]:
            assert jp.all(act >= 0) and jp.all(act <= 1)
    
    def test_run_circuit_batched_inputs(self):
        key = jax.random.PRNGKey(42)
        layer_sizes = [(2, 1), (1, 1)]
        arity = 2
        wires, logits = gen_circuit(key, layer_sizes, arity)
        x = jp.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        acts = run_circuit(logits, wires, x)
        # Check shapes for batched inputs
        batch_size = x.shape[0]
        for i, (act, (size, _)) in enumerate(zip(acts, layer_sizes)):
            assert act.shape == (batch_size, size)
    
    def test_run_circuit_consistency(self):
        """Test run_circuit produces consistent results."""
        key = jax.random.PRNGKey(42)
        layer_sizes = [(2, 1), (1, 1)]
        arity = 2
        
        wires, logits = gen_circuit(key, layer_sizes, arity)
        x = jp.array([1.0, 0.0])
        
        acts1 = run_circuit(logits, wires, x)
        acts2 = run_circuit(logits, wires, x)
        
        # Should be identical
        for a1, a2 in zip(acts1, acts2):
            assert jp.allclose(a1, a2)


class TestGenerateLayerSizes:
    """Test generate_layer_sizes function - creates layer size specifications."""
    
    def test_generate_layer_sizes_basic(self):
        """Test generate_layer_sizes with basic parameters."""
        input_n = 4
        output_n = 2
        arity = 2
        layer_n = 2
        
        layer_sizes = generate_layer_sizes(input_n, output_n, arity, layer_n)
        
        # Should have input + hidden layers + output layer
        assert len(layer_sizes) == 4  # input + 2 hidden + output
        
        # Check input layer
        assert layer_sizes[0] == (input_n, 1)
        
        # Check output layer
        assert layer_sizes[-1] == (output_n, 1)
    
    def test_generate_layer_sizes_arity_4(self):
        """Test generate_layer_sizes with arity=4."""
        input_n = 3
        output_n = 1
        arity = 4
        layer_n = 1
        
        layer_sizes = generate_layer_sizes(input_n, output_n, arity, layer_n)
        
        # Should have 3 layers: input + hidden + output
        assert len(layer_sizes) == 3
        
        # Check last hidden layer connects properly to output
        last_hidden_size, last_hidden_group = layer_sizes[-2]
        assert last_hidden_size == output_n * arity  # 1 * 4 = 4
        assert last_hidden_group == arity // 2  # 4 // 2 = 2
    
    def test_generate_layer_sizes_arity_1(self):
        """Test generate_layer_sizes with arity=1 (edge case)."""
        input_n = 2
        output_n = 1
        arity = 1
        layer_n = 1
        
        layer_sizes = generate_layer_sizes(input_n, output_n, arity, layer_n)
        
        # Should handle arity=1 correctly
        assert len(layer_sizes) == 3
        
        # Last hidden layer group size should be 1 for arity=1
        last_hidden_size, last_hidden_group = layer_sizes[-2]
        assert last_hidden_group == 1
    
    def test_generate_layer_sizes_multiple_hidden_layers(self):
        """Test generate_layer_sizes with multiple hidden layers."""
        input_n = 4
        output_n = 2
        arity = 2
        layer_n = 3
        
        layer_sizes = generate_layer_sizes(input_n, output_n, arity, layer_n)
        
        # Should have input + 3 hidden + output = 5 layers
        assert len(layer_sizes) == 5
        
        # Check hidden layers have consistent structure
        for i in range(1, len(layer_sizes) - 1):
            size, group = layer_sizes[i]
            if i == len(layer_sizes) - 2:  # Last hidden layer
                assert size == output_n * arity
                assert group == arity // 2
            else:  # Other hidden layers
                assert size == input_n * arity * 2  # layer_width
                assert group == arity


class TestBooleanCircuitIntegration:
    """Integration tests for boolean circuit components."""
    
    def test_end_to_end_circuit_creation_and_execution(self):
        """Test complete flow: generate circuit -> run circuit."""
        key = jax.random.PRNGKey(42)
        
        # Generate layer sizes
        layer_sizes = generate_layer_sizes(input_n=2, output_n=1, arity=2, layer_n=1)
        
        # Generate circuit
        wires, logits = gen_circuit(key, layer_sizes, arity=2)
        
        # Create test inputs
        test_inputs = [
            jp.array([0.0, 0.0]),
            jp.array([0.0, 1.0]),
            jp.array([1.0, 0.0]),
            jp.array([1.0, 1.0]),
        ]
        
        # Run circuit on all inputs
        for x in test_inputs:
            acts = run_circuit(logits, wires, x)
            
            # Should have activations for all layers
            assert len(acts) == len(layer_sizes)
            
            # Input activation should match input
            assert jp.allclose(acts[0], x)
            
            # All activations should be finite
            for act in acts:
                assert jp.all(jp.isfinite(act))
    
    def test_circuit_with_different_arities(self):
        """Test circuit generation and execution with different arities."""
        key = jax.random.PRNGKey(42)
        
        for arity in [1, 2, 3]:
            layer_sizes = generate_layer_sizes(input_n=2, output_n=1, arity=arity, layer_n=1)
            wires, logits = gen_circuit(key, layer_sizes, arity=arity)
            
            # Test with random input
            x = jax.random.uniform(key, shape=(2,))
            acts = run_circuit(logits, wires, x)
            
            # Should complete without errors
            assert len(acts) == len(layer_sizes)
            assert jp.all(jp.isfinite(acts[-1]))  # Output should be finite
    
    def test_circuit_gradient_flow(self):
        """Test that circuits support gradient computation."""
        key = jax.random.PRNGKey(42)
        layer_sizes = generate_layer_sizes(input_n=2, output_n=1, arity=2, layer_n=1)
        wires, logits = gen_circuit(key, layer_sizes, arity=2)
        
        def circuit_loss(logits, x):
            acts = run_circuit(logits, wires, x)
            return jp.mean(acts[-1])  # Mean of output
        
        x = jp.array([1.0, 0.0])
        
        # Should be able to compute gradients
        grad_fn = jax.grad(circuit_loss, argnums=0)
        gradients = grad_fn(logits, x)
        
        # Gradients should have same shape as logits
        assert len(gradients) == len(logits)
        for g, l in zip(gradients, logits):
            assert g.shape == l.shape
            assert jp.all(jp.isfinite(g))  # Gradients should be finite


if __name__ == "__main__":
    # Run all tests
    pytest.main([__file__, "-v"]) 