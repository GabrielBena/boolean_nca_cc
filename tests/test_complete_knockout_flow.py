#!/usr/bin/env python3
"""
Comprehensive test to verify the complete knockout flow from self-attention to circuit execution.
"""

import jax
import jax.numpy as jp
from boolean_nca_cc.circuits.model import gen_circuit, run_circuit
from boolean_nca_cc.utils.graph_builder import build_graph
from boolean_nca_cc.utils.extraction import extract_logits_from_graph
from boolean_nca_cc.models.self_attention import CircuitSelfAttention
from flax import nnx

def test_complete_knockout_flow():
    """Test the complete flow: circuit → graph → self-attention → extraction → circuit execution."""
    print("=== Testing Complete Knockout Flow ===")
    
    # Create a simple circuit
    key = jax.random.PRNGKey(42)
    layer_sizes = [(2, 1), (4, 2)]  # 2 inputs, 4 gates (2 groups of 2)
    arity = 2
    input_n = layer_sizes[0][0]
    
    # Generate circuit
    wires, logits = gen_circuit(key, layer_sizes, arity=arity)
    
    print("=== Original Circuit ===")
    print(f"Logits shape: {logits[0].shape}")
    print(f"Logits values:\n{logits[0]}")
    
    # Test original circuit execution
    x_input = jp.array([[1.0, 0.0], [0.0, 1.0]])  # 2 test inputs
    original_acts = run_circuit(logits, wires, x_input)
    original_output = original_acts[-1]
    print(f"Original circuit output:\n{original_output}")
    print()
    
    # Build graph
    graph = build_graph(
        logits=logits,
        wires=wires,
        input_n=input_n,
        arity=arity,
        circuit_hidden_dim=16
    )
    
    print("=== Graph Structure ===")
    print(f"Total nodes: {graph.nodes['logits'].shape[0]}")
    print(f"Layer info: {graph.nodes['layer']}")
    print(f"Input nodes (layer 0): {jp.sum(graph.nodes['layer'] == 0)}")
    print(f"Gate nodes (layer > 0): {jp.sum(graph.nodes['layer'] > 0)}")
    print()
    
    # Create self-attention model
    n_node = int(graph.n_node[0])
    model = CircuitSelfAttention(
        n_node=n_node,
        circuit_hidden_dim=16,
        arity=arity,
        attention_dim=128,
        num_heads=4,
        num_layers=3,
        mlp_dim=256,
        mlp_dim_multiplier=2,
        dropout_rate=0.0,
        re_zero_update=False,  # Important for testing updates
        rngs=nnx.Rngs(params=jax.random.PRNGKey(42)),
    )
    
    # Create knockout pattern (knock out first 2 gates)
    knockout_pattern = jp.zeros(n_node, dtype=jp.bool_)
    knockout_pattern = knockout_pattern.at[2].set(True)  # Knock out first gate node
    knockout_pattern = knockout_pattern.at[3].set(True)  # Knock out second gate node
    
    print("=== Knockout Pattern ===")
    print(f"Knockout pattern: {knockout_pattern}")
    print(f"Knocked-out nodes: {jp.where(knockout_pattern)[0]}")
    print()
    
    # Apply self-attention with knockout
    updated_graph = model(graph, knockout_pattern=knockout_pattern)
    
    print("=== Updated Graph ===")
    print(f"Updated logits shape: {updated_graph.nodes['logits'].shape}")
    
    # Check that knocked-out nodes have large negative logits
    knocked_out_indices = jp.where(knockout_pattern)[0]
    for idx in knocked_out_indices:
        node_logits = updated_graph.nodes['logits'][idx]
        print(f"Node {idx} logits: {node_logits}")
        # Verify they are large negative values
        assert jp.allclose(node_logits, -10.0), f"Node {idx} should have -10.0 logits"
    
    print("✅ Knockout nodes have large negative logits")
    print()
    
    # Extract logits from updated graph
    logits_original_shapes = [logit.shape for logit in logits]
    extracted_logits = extract_logits_from_graph(updated_graph, logits_original_shapes)
    
    print("=== Extracted Logits ===")
    print(f"Extracted logits shape: {extracted_logits[0].shape}")
    print(f"Extracted logits values:\n{extracted_logits[0]}")
    print()
    
    # Test circuit execution with extracted logits
    knocked_out_acts = run_circuit(extracted_logits, wires, x_input)
    knocked_out_output = knocked_out_acts[-1]
    print(f"Knocked-out circuit output:\n{knocked_out_output}")
    print()
    
    # Verify that the knocked-out gates have zero LUTs
    print("=== LUT Verification ===")
    
    # Check sigmoid of extracted logits
    extracted_luts = jax.nn.sigmoid(extracted_logits[0])
    print(f"Extracted LUTs:\n{extracted_luts}")
    
    # Verify that knocked-out gates (first two) have approximately zero LUTs
    knocked_out_gates = extracted_luts[0, :2, :]  # First two gates
    normal_gates = extracted_luts[0, 2:, :]  # Last two gates
    
    print(f"Knocked-out gates LUTs:\n{knocked_out_gates}")
    print(f"Normal gates LUTs:\n{normal_gates}")
    
    # Check that knocked-out gates have approximately zero LUTs
    assert jp.allclose(knocked_out_gates, 0.0, atol=1e-4), "Knocked-out gates should have zero LUTs"
    
    # Check that normal gates have non-zero LUTs (if any exist in this layer)
    if normal_gates.size > 0:
        assert not jp.allclose(normal_gates, 0.0), "Normal gates should have non-zero LUTs"
        print("✅ Normal gates in first layer have non-zero LUTs")
    else:
        print("ℹ️  No normal gates in first layer (all gates knocked out)")
    
    # Also check the second layer
    if len(extracted_logits) > 1:
        second_layer_luts = jax.nn.sigmoid(extracted_logits[1])
        print(f"Second layer LUTs:\n{second_layer_luts}")
        assert not jp.allclose(second_layer_luts, 0.0), "Second layer should have non-zero LUTs"
    
    print("✅ Knocked-out gates have zero LUTs")
    print("✅ Normal gates have non-zero LUTs")
    print()
    
    # Compare outputs
    print("=== Output Comparison ===")
    print(f"Original output:\n{original_output}")
    print(f"Knocked-out output:\n{knocked_out_output}")
    
    # The outputs should be different due to knocked-out gates
    assert not jp.allclose(original_output, knocked_out_output), "Outputs should be different"
    
    print("✅ Outputs are different due to knocked-out gates")
    print()
    
    print("🎉 Complete knockout flow test passed!")
    
    return {
        'original_output': original_output,
        'knocked_out_output': knocked_out_output,
        'original_logits': logits,
        'extracted_logits': extracted_logits,
        'knockout_pattern': knockout_pattern
    }

if __name__ == "__main__":
    test_complete_knockout_flow() 