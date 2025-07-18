#!/usr/bin/env python3
"""
Debug script to demonstrate the extraction issue.
"""

import jax
import jax.numpy as jp
from boolean_nca_cc.circuits.model import gen_circuit
from boolean_nca_cc.utils.graph_builder import build_graph

def extract_logits_old_version(graph, logits_original_shapes):
    """Old version: starts extraction from index 0 (includes input nodes)"""
    all_logits_flat = graph.nodes["logits"]
    extracted_logits_list = []
    current_node_idx = 0  # OLD: Start from 0
    
    for shape in logits_original_shapes:
        group_n, group_size, logit_dim = shape
        size = group_n * group_size
        
        layer_logits_flat = jax.lax.dynamic_slice(
            all_logits_flat, (current_node_idx, 0), (size, logit_dim)
        )
        layer_logits = layer_logits_flat.reshape(group_n, group_size, logit_dim)
        extracted_logits_list.append(layer_logits)
        current_node_idx += size
    
    return extracted_logits_list

def extract_logits_new_version(graph, logits_original_shapes):
    """New version: skips input nodes, starts from gate nodes"""
    all_logits_flat = graph.nodes["logits"]
    extracted_logits_list = []
    
    # NEW: Skip input nodes
    layer_info = graph.nodes["layer"]
    input_n = jp.sum(layer_info == 0)
    current_node_idx = input_n  # Start from first gate layer
    
    for shape in logits_original_shapes:
        group_n, group_size, logit_dim = shape
        size = group_n * group_size
        
        layer_logits_flat = jax.lax.dynamic_slice(
            all_logits_flat, (current_node_idx, 0), (size, logit_dim)
        )
        layer_logits = layer_logits_flat.reshape(group_n, group_size, logit_dim)
        extracted_logits_list.append(layer_logits)
        current_node_idx += size
    
    return extracted_logits_list

def main():
    # Create a simple circuit
    key = jax.random.PRNGKey(42)
    layer_sizes = [(2, 1), (4, 2)]  # 2 inputs, 4 gates (2 groups of 2)
    arity = 2
    input_n = layer_sizes[0][0]
    
    # Generate circuit
    wires, logits = gen_circuit(key, layer_sizes, arity=arity)
    
    print("=== Original Circuit Logits ===")
    for i, layer_logits in enumerate(logits):
        print(f"Layer {i+1}: shape={layer_logits.shape}")
        print(f"Values:\n{layer_logits}\n")
    
    # Build graph
    graph = build_graph(
        logits=logits,
        wires=wires,
        input_n=input_n,
        arity=arity,
        circuit_hidden_dim=16
    )
    
    print("=== Graph Node Structure ===")
    print(f"Total nodes: {graph.nodes['logits'].shape[0]}")
    print(f"Layer info: {graph.nodes['layer']}")
    print(f"Input nodes (layer 0): {jp.sum(graph.nodes['layer'] == 0)}")
    print(f"Gate nodes (layer > 0): {jp.sum(graph.nodes['layer'] > 0)}")
    print()
    
    print("=== Graph Node Logits ===")
    for i, node_logits in enumerate(graph.nodes['logits']):
        layer = graph.nodes['layer'][i]
        print(f"Node {i} (layer {layer}): {node_logits}")
    print()
    
    # Test both extraction methods
    logits_shapes = [l.shape for l in logits]
    
    old_extracted = extract_logits_old_version(graph, logits_shapes)
    new_extracted = extract_logits_new_version(graph, logits_shapes)
    
    print("=== OLD Extraction (starts from index 0) ===")
    for i, extracted in enumerate(old_extracted):
        print(f"Layer {i+1}: shape={extracted.shape}")
        print(f"Values:\n{extracted}\n")
    
    print("=== NEW Extraction (skips input nodes) ===")
    for i, extracted in enumerate(new_extracted):
        print(f"Layer {i+1}: shape={extracted.shape}")
        print(f"Values:\n{extracted}\n")
    
    print("=== Comparison ===")
    for i, (original, old, new) in enumerate(zip(logits, old_extracted, new_extracted)):
        print(f"Layer {i+1}:")
        print(f"  Original matches OLD: {jp.allclose(original, old)}")
        print(f"  Original matches NEW: {jp.allclose(original, new)}")
        print(f"  OLD is all zeros: {jp.allclose(old, jp.zeros_like(old))}")
        print(f"  NEW is all zeros: {jp.allclose(new, jp.zeros_like(new))}")
        print()

if __name__ == "__main__":
    main() 