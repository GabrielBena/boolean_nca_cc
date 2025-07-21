#!/usr/bin/env python3
"""
Test for Node Feature Zeroing Enhancement (Phase 1)

Validates that the CircuitSelfAttention model correctly zeros out the
features of knocked-out nodes, as specified in Phase 1 of the paddingdoc.
"""

import jax
import jax.numpy as jp
import jraph
import yaml
import os
import sys
from typing import Dict, List, Tuple, Optional
from flax import nnx
import numpy as np

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from boolean_nca_cc.circuits.model import gen_circuit
from boolean_nca_cc.models import CircuitSelfAttention
from boolean_nca_cc.utils.graph_builder import build_graph
from boolean_nca_cc.training.pool.structural_perturbation import (
    create_reproducible_knockout_pattern,
    extract_layer_info_from_graph,
)

def load_config():
    """Load configuration from config.yaml files"""
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs", "config.yaml")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    model_config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs", "model", "self_attention.yaml")
    with open(model_config_path, 'r') as f:
        model_config = yaml.safe_load(f)
    
    return config, model_config

def create_test_artifacts(config, model_config):
    """Create a test circuit, graph, and model."""
    arity = config["circuit"]["arity"]
    input_bits = config["circuit"]["input_bits"]
    output_bits = config["circuit"]["output_bits"]
    num_layers = config["circuit"]["num_layers"]
    circuit_hidden_dim = config["circuit"]["circuit_hidden_dim"]
    
    layer_sizes = [(input_bits, 1)]
    for i in range(num_layers - 1):
        layer_size = max(4, (input_bits + output_bits) // 2)
        layer_sizes.append((layer_size, 1))
    layer_sizes.append((output_bits, 1))
    
    input_n = layer_sizes[0][0]
    
    wires, logits = gen_circuit(jax.random.PRNGKey(42), layer_sizes, arity=arity)
    graph = build_graph(logits, wires, input_n, arity, circuit_hidden_dim)
    
    n_nodes = int(graph.n_node[0])
    
    model = CircuitSelfAttention(
        n_node=n_nodes,
        circuit_hidden_dim=circuit_hidden_dim,
        arity=arity,
        attention_dim=model_config["attention_dim"],
        num_heads=model_config["num_heads"],
        num_layers=model_config["num_layers"],
        mlp_dim=model_config["mlp_dim"],
        mlp_dim_multiplier=model_config["mlp_dim_multiplier"],
        dropout_rate=model_config["dropout_rate"],
        zero_init=model_config["zero_init"],
        re_zero_update=False,  # Essential for testing updates
        rngs=nnx.Rngs(params=jax.random.PRNGKey(42)),
    )
    
    return graph, model, input_n

def test_feature_zeroing_in_self_attention():
    """
    Validates that the __call__ method of CircuitSelfAttention correctly
    zeros out the features of knocked-out nodes.
    """
    print("=" * 80)
    print("Testing Feature Zeroing in CircuitSelfAttention")
    print("=" * 80)

    config, model_config = load_config()
    graph, model, input_n = create_test_artifacts(config, model_config)

    # Create a knockout pattern
    true_layer_sizes = extract_layer_info_from_graph(graph, input_n)
    pattern_key = jax.random.PRNGKey(123)
    knockout_pattern = create_reproducible_knockout_pattern(
        pattern_key,
        layer_sizes=true_layer_sizes,
        damage_prob=0.3,
        input_n=input_n,
    )
    
    knocked_out_indices = jp.where(knockout_pattern)[0]
    active_indices = jp.where(~knockout_pattern)[0]

    print(f"Knocked out {len(knocked_out_indices)} nodes at indices: {knocked_out_indices}")
    
    # Run a single forward pass of the model
    updated_graph = model(graph, knockout_pattern=knockout_pattern)
    
    # Extract the updated node features
    updated_nodes = updated_graph.nodes
    updated_logits = updated_nodes["logits"]
    updated_hidden = updated_nodes["hidden"]
    
    # 1. Verify that features of knocked-out nodes are zero
    knocked_out_logits = updated_logits[knocked_out_indices]
    knocked_out_hidden = updated_hidden[knocked_out_indices]
    
    assert jp.all(knocked_out_logits == 0), "Logits of knocked-out nodes should be zero."
    print("✓ Logits of knocked-out nodes are zeroed.")
    
    assert jp.all(knocked_out_hidden == 0), "Hidden features of knocked-out nodes should be zero."
    print("✓ Hidden features of knocked-out nodes are zeroed.")
    
    # 2. Verify that features of active nodes are NOT all zero
    # We expect updates to happen since re_zero_update=False
    active_logits = updated_logits[active_indices]
    active_hidden = updated_hidden[active_indices]
    
    assert not jp.all(active_logits == 0), "Logits of active nodes should not be all zero."
    print("✓ Logits of active nodes are not all zero.")

    assert not jp.all(active_hidden == 0), "Hidden features of active nodes should not be all zero."
    print("✓ Hidden features of active nodes are not all zero.")

    print("\n🎉 Test Passed: Feature zeroing works as expected in CircuitSelfAttention.")

if __name__ == "__main__":
    test_feature_zeroing_in_self_attention()
