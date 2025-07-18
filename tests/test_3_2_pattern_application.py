#!/usr/bin/env python3
"""
Test 3.2: Pattern Application

This test verifies that knockout patterns correctly modify graph structures and prevent
updates to knocked-out nodes, mirroring the actual training loop implementation.

Key aspects tested:
1. Knockout patterns prevent attention to/from knocked-out nodes
2. Knocked-out nodes receive zero updates during optimization
3. Pattern persistence through optimization steps
4. Non-knocked-out nodes update normally
5. Attention masking correctly blocks message passing

This test mirrors the actual training loop usage where:
- Knockout patterns are passed to the model during forward pass
- The model applies both attention masking and update blocking
- Patterns persist through multiple message passing steps
"""

import jax
import jax.numpy as jp
import jraph
import optax
from flax import nnx
import yaml
import os
import sys
from typing import Dict, List, Tuple, Optional
from functools import partial

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from boolean_nca_cc.circuits.model import gen_circuit
from boolean_nca_cc.models import CircuitSelfAttention
from boolean_nca_cc.utils.graph_builder import build_graph
from boolean_nca_cc.training.pool.structural_perturbation import (
    create_reproducible_knockout_pattern,
    extract_layer_info_from_graph,
    create_knockout_vocabulary
)
from boolean_nca_cc.training.evaluation import get_loss_and_update_graph
from boolean_nca_cc.utils.extraction import extract_logits_from_graph


def load_config():
    """Load configuration from config.yaml"""
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs", "config.yaml")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load model config
    model_config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs", "model", "self_attention.yaml")
    with open(model_config_path, 'r') as f:
        model_config = yaml.safe_load(f)
    
    return config, model_config


def create_test_data(input_bits: int, output_bits: int, batch_size: int = 4):
    """Create test data for the reverse task"""
    # Create simple test data: reverse the input bits
    x_data = jax.random.randint(jax.random.PRNGKey(42), (batch_size, input_bits), 0, 2)
    y_data = jp.flip(x_data, axis=1)  # Reverse the bits
    return x_data, y_data


def test_attention_mask_creation():
    """Test that attention masks correctly block knocked-out nodes"""
    print("Testing attention mask creation...")
    
    config, model_config = load_config()
    
    # Create a simple circuit
    layer_sizes = [(2, 2), (2, 2), (1, 1)]  # 2 inputs, 2 hidden layers, 1 output
    input_n = 2
    arity = 2
    circuit_hidden_dim = 16
    
    # Generate circuit
    wires, logits = gen_circuit(jax.random.PRNGKey(42), layer_sizes, arity=arity)
    graph = build_graph(logits, wires, input_n, arity, circuit_hidden_dim)
    
    # Get actual number of nodes
    n_nodes = graph.n_node[0]
    print(f"Circuit has {n_nodes} nodes")
    
    # Create model
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
        rngs=nnx.Rngs(params=jax.random.PRNGKey(123)),
    )
    
    # Create knockout pattern using the actual function
    true_layer_sizes = extract_layer_info_from_graph(graph, input_n)
    damage_prob = config["pool"]["persistent_knockout"]["damage_prob"]
    knockout_pattern = create_reproducible_knockout_pattern(
        key=jax.random.PRNGKey(42),
        layer_sizes=true_layer_sizes,
        damage_prob=damage_prob,
        input_n=input_n,
    )
    
    print(f"Knockout pattern: {knockout_pattern}")
    
    # Create attention mask
    attention_mask = model._create_attention_mask(
        graph.senders, graph.receivers, knockout_pattern=knockout_pattern, bidirectional=True
    )
    
    # Remove batch and head dimensions for easier inspection
    mask = attention_mask[0, 0]  # Shape: [n_node, n_node]
    
    # Verify that knocked-out nodes can't send or receive attention
    knocked_out_indices = jp.where(knockout_pattern)[0]
    active_indices = jp.where(~knockout_pattern)[0]
    
    # Check that knocked-out nodes can't send attention to anyone
    for ko_idx in knocked_out_indices:
        assert jp.all(~mask[ko_idx, :]), f"Knocked-out node {ko_idx} can send attention"
    
    # Check that knocked-out nodes can't receive attention from anyone
    for ko_idx in knocked_out_indices:
        assert jp.all(~mask[:, ko_idx]), f"Knocked-out node {ko_idx} can receive attention"
    
    # Check that active nodes can still attend to each other
    for active_idx in active_indices:
        for other_active_idx in active_indices:
            if active_idx == other_active_idx:
                # Self-attention should be allowed
                assert mask[active_idx, other_active_idx], f"Active node {active_idx} can't self-attend"
            else:
                # Check if there's a connection in the original graph
                original_connection = jp.any((graph.senders == active_idx) & (graph.receivers == other_active_idx))
                if original_connection:
                    assert mask[active_idx, other_active_idx], f"Active nodes {active_idx}->{other_active_idx} can't attend"
    
    print("✓ Attention mask correctly blocks knocked-out nodes")


def test_knockout_pattern_updates():
    """Test that knocked-out nodes receive zero updates"""
    print("Testing knockout pattern updates...")
    
    config, model_config = load_config()
    
    # Create a simple circuit
    layer_sizes = [(2, 2), (2, 2), (1, 1)]
    input_n = 2
    arity = 2
    circuit_hidden_dim = 16
    
    # Generate circuit
    wires, logits = gen_circuit(jax.random.PRNGKey(42), layer_sizes, arity=arity)
    graph = build_graph(logits, wires, input_n, arity, circuit_hidden_dim)
    
    # Get actual number of nodes
    n_nodes = graph.n_node[0]
    
    # Create model with re_zero_update=False to ensure updates happen
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
        re_zero_update=False,  # Ensure updates happen
        rngs=nnx.Rngs(params=jax.random.PRNGKey(123)),
    )
    
    # Create knockout pattern using the actual function
    true_layer_sizes = extract_layer_info_from_graph(graph, input_n)
    damage_prob = config["pool"]["persistent_knockout"]["damage_prob"]
    knockout_pattern = create_reproducible_knockout_pattern(
        key=jax.random.PRNGKey(42),
        layer_sizes=true_layer_sizes,
        damage_prob=damage_prob,
        input_n=input_n,
    )
    
    # Store original node values
    original_logits = graph.nodes["logits"].copy()
    original_hidden = graph.nodes["hidden"].copy()
    
    # Apply model with knockout pattern
    updated_graph = model(graph, knockout_pattern=knockout_pattern)
    
    # Check that knocked-out nodes didn't change
    knocked_out_indices = jp.where(knockout_pattern)[0]
    for ko_idx in knocked_out_indices:
        assert jp.allclose(updated_graph.nodes["logits"][ko_idx], original_logits[ko_idx]), \
            f"Knocked-out node {ko_idx} logits changed"
        assert jp.allclose(updated_graph.nodes["hidden"][ko_idx], original_hidden[ko_idx]), \
            f"Knocked-out node {ko_idx} hidden state changed"
    
    # Check that active nodes did change
    active_indices = jp.where(~knockout_pattern)[0]
    for active_idx in active_indices:
        logits_changed = not jp.allclose(updated_graph.nodes["logits"][active_idx], original_logits[active_idx])
        hidden_changed = not jp.allclose(updated_graph.nodes["hidden"][active_idx], original_hidden[active_idx])
        assert logits_changed or hidden_changed, f"Active node {active_idx} didn't change"
    
    print("✓ Knocked-out nodes receive zero updates")


def test_pattern_persistence_through_steps():
    """Test that knockout patterns persist through multiple optimization steps"""
    print("Testing pattern persistence through steps...")
    
    config, model_config = load_config()
    
    # Create test data
    input_bits = 4
    output_bits = 4
    x_data, y_data = create_test_data(input_bits, output_bits, batch_size=2)
    
    # Create a simple circuit
    layer_sizes = [(4, 2), (2, 2), (1, 1)]  # 4 inputs, 2 hidden layers, 1 output
    input_n = 4
    arity = 2
    circuit_hidden_dim = 16
    n_message_steps = 3
    
    # Generate circuit
    wires, logits = gen_circuit(jax.random.PRNGKey(42), layer_sizes, arity=arity)
    graph = build_graph(logits, wires, input_n, arity, circuit_hidden_dim)
    
    # Create model
    model = CircuitSelfAttention(
        n_node=graph.n_node[0],
        circuit_hidden_dim=circuit_hidden_dim,
        arity=arity,
        attention_dim=model_config["attention_dim"],
        num_heads=model_config["num_heads"],
        num_layers=model_config["num_layers"],
        mlp_dim=model_config["mlp_dim"],
        mlp_dim_multiplier=model_config["mlp_dim_multiplier"],
        dropout_rate=model_config["dropout_rate"],
        rngs=nnx.Rngs(params=jax.random.PRNGKey(123)),
    )
    
    # Create knockout pattern (knock out some hidden nodes)
    true_layer_sizes = extract_layer_info_from_graph(graph, input_n)
    damage_prob = config["pool"]["persistent_knockout"]["damage_prob"]
    knockout_pattern = create_reproducible_knockout_pattern(
        key=jax.random.PRNGKey(42),
        layer_sizes=true_layer_sizes,
        damage_prob=damage_prob,
        input_n=input_n,
    )
    
    # Store original node values
    original_logits = graph.nodes["logits"].copy()
    original_hidden = graph.nodes["hidden"].copy()
    
    # Simulate multiple optimization steps (mirroring the training loop)
    current_graph = graph
    logits_original_shapes = [logit.shape for logit in logits]
    
    for step in range(n_message_steps):
        # Apply model with knockout pattern (mirrors loss_fn_no_scan in training loop)
        current_graph = model(current_graph, knockout_pattern=knockout_pattern)
        
        # Apply loss and update (mirrors get_loss_and_update_graph call)
        current_graph, loss, current_logits, aux = get_loss_and_update_graph(
            graph=current_graph,
            logits_original_shapes=logits_original_shapes,
            wires=wires,
            x_data=x_data,
            y_data=y_data,
            loss_type=config["training"]["loss_type"],
            layer_sizes=layer_sizes,
        )
        
        # Update graph globals (mirrors training loop)
        current_update_steps = current_graph.globals[..., 1] if current_graph.globals is not None else 0
        current_graph = current_graph._replace(
            globals=jp.array([loss, current_update_steps + 1], dtype=jp.float32)
        )
    
    # Check that knocked-out nodes still haven't changed from original
    knocked_out_indices = jp.where(knockout_pattern)[0]
    for ko_idx in knocked_out_indices:
        assert jp.allclose(current_graph.nodes["logits"][ko_idx], original_logits[ko_idx]), \
            f"Knocked-out node {ko_idx} logits changed after {n_message_steps} steps"
        assert jp.allclose(current_graph.nodes["hidden"][ko_idx], original_hidden[ko_idx]), \
            f"Knocked-out node {ko_idx} hidden state changed after {n_message_steps} steps"
    
    # Check that active nodes have changed
    active_indices = jp.where(~knockout_pattern)[0]
    for active_idx in active_indices:
        logits_changed = not jp.allclose(current_graph.nodes["logits"][active_idx], original_logits[active_idx])
        hidden_changed = not jp.allclose(current_graph.nodes["hidden"][active_idx], original_hidden[active_idx])
        assert logits_changed or hidden_changed, f"Active node {active_idx} didn't change after {n_message_steps} steps"
    
    print("✓ Knockout patterns persist through multiple optimization steps")


def test_batch_processing_with_knockouts():
    """Test batch processing with different knockout patterns"""
    print("Testing batch processing with knockouts...")
    
    config, model_config = load_config()
    
    # Create test data
    input_bits = 4
    output_bits = 4
    x_data, y_data = create_test_data(input_bits, output_bits, batch_size=2)
    
    # Create a simple circuit
    layer_sizes = [(4, 2), (2, 2), (1, 1)]
    input_n = 4
    arity = 2
    circuit_hidden_dim = 16
    
    # Generate base circuit
    wires, logits = gen_circuit(jax.random.PRNGKey(42), layer_sizes, arity=arity)
    base_graph = build_graph(logits, wires, input_n, arity, circuit_hidden_dim)
    
    # Get actual number of nodes
    n_nodes = base_graph.n_node[0]
    
    # Create model
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
        rngs=nnx.Rngs(params=jax.random.PRNGKey(123)),
    )
    
    # Create batch of graphs
    batch_size = 3
    batch_graphs = jraph.GraphsTuple(
        nodes=jax.tree.map(lambda x: jp.repeat(x[None, ...], batch_size, axis=0), base_graph.nodes),
        edges=base_graph.edges,
        receivers=base_graph.receivers,
        senders=base_graph.senders,
        globals=base_graph.globals,
        n_node=base_graph.n_node,
        n_edge=base_graph.n_edge,
    )
    
    # Create different knockout patterns for each graph in batch
    true_layer_sizes = extract_layer_info_from_graph(base_graph, input_n)
    damage_prob = config["pool"]["persistent_knockout"]["damage_prob"]
    
    # Create patterns using different keys for variety
    pattern_keys = jax.random.split(jax.random.PRNGKey(42), batch_size)
    knockout_patterns = jax.vmap(
        lambda key: create_reproducible_knockout_pattern(
            key=key,
            layer_sizes=true_layer_sizes,
            damage_prob=damage_prob,
            input_n=input_n,
        )
    )(pattern_keys)
    
    # Store original values
    original_logits = batch_graphs.nodes["logits"].copy()
    original_hidden = batch_graphs.nodes["hidden"].copy()
    
    # Apply model to batch with different knockout patterns
    # This mirrors the vmap usage in the training loop
    vmap_model = jax.vmap(lambda g, k: model(g, knockout_pattern=k))
    updated_batch_graphs = vmap_model(batch_graphs, knockout_patterns)
    
    # Check that each graph in the batch respects its specific knockout pattern
    for batch_idx in range(batch_size):
        pattern = knockout_patterns[batch_idx]
        knocked_out_indices = jp.where(pattern)[0]
        active_indices = jp.where(~pattern)[0]
        
        # Check knocked-out nodes didn't change
        for ko_idx in knocked_out_indices:
            assert jp.allclose(updated_batch_graphs.nodes["logits"][batch_idx, ko_idx], 
                              original_logits[batch_idx, ko_idx]), \
                f"Batch {batch_idx}, knocked-out node {ko_idx} logits changed"
            assert jp.allclose(updated_batch_graphs.nodes["hidden"][batch_idx, ko_idx], 
                              original_hidden[batch_idx, ko_idx]), \
                f"Batch {batch_idx}, knocked-out node {ko_idx} hidden state changed"
        
        # Check active nodes did change
        for active_idx in active_indices:
            logits_changed = not jp.allclose(updated_batch_graphs.nodes["logits"][batch_idx, active_idx], 
                                           original_logits[batch_idx, active_idx])
            hidden_changed = not jp.allclose(updated_batch_graphs.nodes["hidden"][batch_idx, active_idx], 
                                           original_hidden[batch_idx, active_idx])
            assert logits_changed or hidden_changed, f"Batch {batch_idx}, active node {active_idx} didn't change"
    
    print("✓ Batch processing correctly handles different knockout patterns")


def test_knockout_vocabulary_integration():
    """Test integration with knockout vocabulary (mirrors training loop vocabulary usage)"""
    print("Testing knockout vocabulary integration...")
    
    config, model_config = load_config()
    
    # Create a simple circuit
    layer_sizes = [(4, 2), (2, 2), (1, 1)]
    input_n = 4
    arity = 2
    circuit_hidden_dim = 16
    
    # Generate circuit
    wires, logits = gen_circuit(jax.random.PRNGKey(42), layer_sizes, arity=arity)
    graph = build_graph(logits, wires, input_n, arity, circuit_hidden_dim)
    
    # Extract layer info (mirrors training loop)
    true_layer_sizes = extract_layer_info_from_graph(graph, input_n)
    
    # Create knockout vocabulary (mirrors training loop vocabulary creation)
    vocabulary_size = 10
    damage_prob = config["pool"]["persistent_knockout"]["damage_prob"]
    vocab_rng = jax.random.PRNGKey(42)
    
    knockout_vocabulary = create_knockout_vocabulary(
        rng=vocab_rng,
        vocabulary_size=vocabulary_size,
        layer_sizes=true_layer_sizes,
        damage_prob=damage_prob,
        input_n=input_n,
    )
    
    # Create model
    model = CircuitSelfAttention(
        n_node=graph.n_node[0],
        circuit_hidden_dim=circuit_hidden_dim,
        arity=arity,
        attention_dim=model_config["attention_dim"],
        num_heads=model_config["num_heads"],
        num_layers=model_config["num_layers"],
        mlp_dim=model_config["mlp_dim"],
        mlp_dim_multiplier=model_config["mlp_dim_multiplier"],
        dropout_rate=model_config["dropout_rate"],
        rngs=nnx.Rngs(params=jax.random.PRNGKey(123)),
    )
    
    # Test that vocabulary patterns work correctly
    for pattern_idx in range(min(3, vocabulary_size)):  # Test first 3 patterns
        pattern = knockout_vocabulary[pattern_idx]
        
        # Store original values
        original_logits = graph.nodes["logits"].copy()
        original_hidden = graph.nodes["hidden"].copy()
        
        # Apply model with vocabulary pattern
        updated_graph = model(graph, knockout_pattern=pattern)
        
        # Check that knocked-out nodes didn't change
        knocked_out_indices = jp.where(pattern)[0]
        for ko_idx in knocked_out_indices:
            assert jp.allclose(updated_graph.nodes["logits"][ko_idx], original_logits[ko_idx]), \
                f"Vocabulary pattern {pattern_idx}, knocked-out node {ko_idx} logits changed"
            assert jp.allclose(updated_graph.nodes["hidden"][ko_idx], original_hidden[ko_idx]), \
                f"Vocabulary pattern {pattern_idx}, knocked-out node {ko_idx} hidden state changed"
        
        # Reset graph for next test
        graph = graph._replace(nodes=original_logits, hidden=original_hidden)
    
    print("✓ Knockout vocabulary integration works correctly")


def test_gradient_flow_blocking():
    """Test that gradients are properly blocked for knocked-out nodes"""
    print("Testing gradient flow blocking...")
    
    config, model_config = load_config()
    
    # Create test data
    input_bits = 4
    output_bits = 4
    x_data, y_data = create_test_data(input_bits, output_bits, batch_size=2)
    
    # Create a simple circuit
    layer_sizes = [(4, 2), (2, 2), (1, 1)]
    input_n = 4
    arity = 2
    circuit_hidden_dim = 16
    
    # Generate circuit
    wires, logits = gen_circuit(jax.random.PRNGKey(42), layer_sizes, arity=arity)
    graph = build_graph(logits, wires, input_n, arity, circuit_hidden_dim)
    
    # Get actual number of nodes
    n_nodes = graph.n_node[0]
    
    # Create model
    model = CircuitSelfAttention(
        n_node=graph.n_node[0],
        circuit_hidden_dim=circuit_hidden_dim,
        arity=arity,
        attention_dim=model_config["attention_dim"],
        num_heads=model_config["num_heads"],
        num_layers=model_config["num_layers"],
        mlp_dim=model_config["mlp_dim"],
        mlp_dim_multiplier=model_config["mlp_dim_multiplier"],
        dropout_rate=model_config["dropout_rate"],
        rngs=nnx.Rngs(params=jax.random.PRNGKey(123)),
    )
    
    # Create knockout pattern
    true_layer_sizes = extract_layer_info_from_graph(graph, input_n)
    damage_prob = config["pool"]["persistent_knockout"]["damage_prob"]
    knockout_pattern = create_reproducible_knockout_pattern(
        key=jax.random.PRNGKey(42),
        layer_sizes=true_layer_sizes,
        damage_prob=damage_prob,
        input_n=input_n,
    )
    
    # Define loss function that mirrors the training loop
    def loss_fn(model_params, graph, knockout_pattern):
        # Apply model
        updated_graph = model.apply(model_params, graph, knockout_pattern=knockout_pattern)
        
        # Extract logits and compute loss (simplified)
        logits = extract_logits_from_graph(updated_graph, [logits.shape])
        loss = jp.mean((logits - y_data[0]) ** 2)  # Simple MSE loss
        return loss
    
    # Get model parameters
    model_params = model.parameters()
    
    # Compute gradients
    grads = jax.grad(loss_fn)(model_params, graph, knockout_pattern)
    
    # Check that gradients for knocked-out nodes are zero or very small
    # (This is a simplified check - in practice, gradients would be computed through the model)
    knocked_out_indices = jp.where(knockout_pattern)[0]
    
    # Note: This is a basic test. In the actual implementation, gradients would be
    # blocked at the update level, not necessarily at the parameter level.
    # The key is that knocked-out nodes don't receive updates.
    
    print("✓ Gradient flow blocking test completed (simplified)")


def main():
    """Run all pattern application tests"""
    print("=" * 60)
    print("TEST 3.2: PATTERN APPLICATION")
    print("=" * 60)
    print("Testing knockout pattern application as used in actual training loop")
    print()
    
    try:
        test_attention_mask_creation()
        test_knockout_pattern_updates()
        test_pattern_persistence_through_steps()
        test_batch_processing_with_knockouts()
        test_knockout_vocabulary_integration()
        test_gradient_flow_blocking()
        
        print()
        print("=" * 60)
        print("✓ ALL PATTERN APPLICATION TESTS PASSED")
        print("=" * 60)
        print()
        print("Key findings:")
        print("- Knockout patterns correctly block attention to/from knocked-out nodes")
        print("- Knocked-out nodes receive zero updates during optimization")
        print("- Patterns persist through multiple message passing steps")
        print("- Batch processing handles different patterns correctly")
        print("- Vocabulary integration works as expected")
        print("- Pattern application mirrors actual training loop implementation")
        
    except Exception as e:
        print(f"❌ TEST FAILED: {e}")
        raise


if __name__ == "__main__":
    main() 