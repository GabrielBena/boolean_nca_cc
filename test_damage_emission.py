#!/usr/bin/env python3
"""
Test script for damage emission functionality.

This script tests the damage_emission mode in self-attention models by:
1. Creating a simple circuit
2. Applying knockout patterns
3. Running message passing with damage_emission=True/False
4. Comparing the behavior and logging logit channels
"""

import jax
import jax.numpy as jp
import jraph
from flax import nnx
import logging

from boolean_nca_cc.circuits.model import gen_circuit, run_circuit
from boolean_nca_cc.circuits.tasks import get_task_data
from boolean_nca_cc import generate_layer_sizes
from boolean_nca_cc.utils.graph_builder import build_graph
from boolean_nca_cc.models.self_attention import CircuitSelfAttention
from boolean_nca_cc.training.evaluation import get_loss_and_update_graph

# Configure logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def create_test_circuit():
    """Create a simple test circuit."""
    # Simple 2-bit input, 2-bit output circuit
    input_n, output_n = 2, 2
    arity = 2
    layer_sizes = generate_layer_sizes(input_n, output_n, arity, layer_n=2)
    
    # Generate circuit
    key = jax.random.PRNGKey(42)
    wires, logits = gen_circuit(key, layer_sizes, arity=arity)
    
    # Generate test data
    case_n = 1 << input_n  # 4 cases
    x_data, y_data = get_task_data("binary_multiply", case_n, input_bits=input_n, output_bits=output_n)
    
    return wires, logits, x_data, y_data, layer_sizes, input_n, arity


def create_test_model(n_nodes, circuit_hidden_dim=16, damage_emission=False):
    """Create a test self-attention model."""
    key = jax.random.PRNGKey(123)
    rngs = nnx.Rngs(params=key)
    
    model = CircuitSelfAttention(
        n_node=n_nodes,
        circuit_hidden_dim=circuit_hidden_dim,
        arity=2,
        attention_dim=64,
        num_heads=4,
        num_layers=2,
        dropout_rate=0.0,
        damage_emission=damage_emission,
        rngs=rngs,
    )
    
    return model


def print_logit_channels(graph, step, prefix=""):
    """Print logit channels for debugging (non-JIT version)."""
    logits = graph.nodes["logits"]
    log.info(f"{prefix}Step {step} - Logit channels shape: {logits.shape}")
    log.info(f"{prefix}Logit range: [{jp.min(logits):.3f}, {jp.max(logits):.3f}]")
    log.info(f"{prefix}Logit mean: {jp.mean(logits):.3f}")
    
    # Print first few logit values
    log.info(f"{prefix}First 10 logit values: {logits[:10]}")
    
    # Check for -10 values (damaged nodes)
    damaged_mask = jp.abs(logits + 10.0) < 1e-6
    num_damaged = jp.sum(damaged_mask)
    log.info(f"{prefix}Number of nodes with logits ≈ -10: {num_damaged}")


def test_damage_emission():
    """Test damage emission functionality."""
    log.info("=== Testing Damage Emission ===")
    
    # Create test circuit
    wires, logits, x_data, y_data, layer_sizes, input_n, arity = create_test_circuit()
    
    # Build graph
    graph = build_graph(
        wires=wires,
        logits=logits,
        input_n=input_n,
        arity=arity,
        circuit_hidden_dim=16,
    )
    n_nodes = int(graph.n_node[0])
    
    log.info(f"Circuit has {n_nodes} nodes")
    log.info(f"Layer sizes: {layer_sizes}")
    
    # Create knockout pattern (damage first 2 gate nodes)
    knockout_pattern = jp.zeros(n_nodes, dtype=jp.bool_)
    # Skip input nodes (first 2) and damage next 2 nodes
    knockout_pattern = knockout_pattern.at[2:4].set(True)
    log.info(f"Knockout pattern: {knockout_pattern}")
    
    # Test both modes
    for damage_emission in [False, True]:
        log.info(f"\n--- Testing damage_emission={damage_emission} ---")
        
        # Create model
        model = create_test_model(n_nodes, damage_emission=damage_emission)
        
        # Print initial state
        print_logit_channels(graph, 0, f"[damage_emission={damage_emission}] ")
        
        # Run a few message passing steps
        current_graph = graph
        for step in range(3):
            # Apply self-attention with knockout pattern
            updated_graph = model(
                current_graph,
                knockout_pattern=knockout_pattern,
            )
            
            # Print logit channels after update
            print_logit_channels(updated_graph, step + 1, f"[damage_emission={damage_emission}] ")
            
            # Check attention mask behavior
            attention_mask = model._create_attention_mask(
                graph.senders, graph.receivers, knockout_pattern=knockout_pattern
            )
            mask = attention_mask[0, 0]  # Remove batch and head dimensions
            
            # Check if damaged nodes can receive attention
            damaged_receivers = mask[knockout_pattern, :]  # Rows for damaged nodes
            damaged_senders = mask[:, knockout_pattern]    # Columns for damaged nodes
            
            log.info(f"Damaged nodes can receive from others: {jp.any(damaged_receivers)}")
            log.info(f"Others can receive from damaged nodes: {jp.any(damaged_senders)}")
            
            current_graph = updated_graph
        
        # Test sigmoid behavior of damaged logits
        log.info(f"\n--- Sigmoid behavior test (damage_emission={damage_emission}) ---")
        
        # Check sigmoid of damaged logits
        damaged_logits = current_graph.nodes["logits"][knockout_pattern]
        damaged_sigmoid = jax.nn.sigmoid(damaged_logits)
        
        log.info(f"Damaged logits: {damaged_logits}")
        log.info(f"Damaged sigmoid: {damaged_sigmoid}")
        log.info(f"Damaged sigmoid near zero: {jp.allclose(damaged_sigmoid, 0, atol=1e-3)}")


def test_attention_mask_behavior():
    """Test attention mask behavior in detail."""
    log.info("\n=== Testing Attention Mask Behavior ===")
    
    # Create simple test case
    n_nodes = 6
    senders = jp.array([0, 1, 2, 3])
    receivers = jp.array([2, 3, 4, 5])
    
    # Create knockout pattern (damage nodes 2 and 3)
    knockout_pattern = jp.array([False, False, True, True, False, False])
    
    # Create model
    model = create_test_model(n_nodes, damage_emission=True)
    
    # Create attention mask
    attention_mask = model._create_attention_mask(
        senders, receivers, knockout_pattern=knockout_pattern
    )
    mask = attention_mask[0, 0]  # Remove batch and head dimensions
    
    log.info(f"Knockout pattern: {knockout_pattern}")
    log.info(f"Attention mask shape: {mask.shape}")
    log.info(f"Attention mask:\n{mask}")
    
    # Check specific behaviors
    log.info(f"Damaged nodes (2,3) can receive: {mask[2:4, :]}")
    log.info(f"Others can receive from damaged nodes (2,3): {mask[:, 2:4]}")
    log.info(f"Damaged nodes can send to others: {mask[2:4, :]}")
    log.info(f"Others can send to damaged nodes: {mask[:, 2:4]}")


if __name__ == "__main__":
    test_damage_emission()
    test_attention_mask_behavior()
