#!/usr/bin/env python3
"""
Debug script to test SEU recovery evaluation step by step.
"""

import jax
import jax.numpy as jp
from omegaconf import OmegaConf

from boolean_nca_cc.circuits.model import generate_layer_sizes
from boolean_nca_cc.circuits.tasks import get_task_data
from boolean_nca_cc.training.preconfigure import preconfigure_circuit_logits
from boolean_nca_cc.training.evaluation import get_loss_from_wires_logits, evaluate_model_stepwise_batched
from boolean_nca_cc.training.pool.perturbation import (
    sample_seu_gates,
    build_flip_masks_from_indices,
)
from boolean_nca_cc.models import CircuitSelfAttention


def main():
    # Load config
    cfg = OmegaConf.load("configs/config.yaml")
    
    # Parameters
    input_bits = int(cfg.circuit.input_bits)
    output_bits = int(cfg.circuit.output_bits)
    arity = int(cfg.circuit.arity)
    num_layers = int(cfg.circuit.num_layers)
    loss_type = str(cfg.training.loss_type)
    task_name = str(cfg.circuit.task)
    circuit_hidden_dim = int(cfg.circuit.circuit_hidden_dim)
    
    # Generate data
    case_n = 1 << input_bits
    x_data, y_data = get_task_data(task_name, case_n, input_bits=input_bits, output_bits=output_bits)
    
    # Generate circuit
    layer_sizes = generate_layer_sizes(input_bits, output_bits, arity, layer_n=num_layers)
    
    # Preconfigure circuit
    wiring_key = jax.random.PRNGKey(33)
    base_wires, base_logits = preconfigure_circuit_logits(
        wiring_key=wiring_key,
        layer_sizes=layer_sizes,
        arity=arity,
        x_data=x_data,
        y_data=y_data,
        loss_type=loss_type,
        steps=200,
        lr=1.0,
        optimizer="adamw",
        weight_decay=1e-1,
        beta1=0.8,
        beta2=0.8,
    )
    
    # Test 1: Baseline performance
    _, aux = get_loss_from_wires_logits(base_logits, base_wires, x_data, y_data, loss_type)
    baseline_acc = float(aux[4])
    print(f"Baseline hard accuracy: {baseline_acc:.6f}")
    
    # Test 2: Apply SEU damage
    gate_layer_sizes = layer_sizes[1:]
    flip_key = jax.random.PRNGKey(481)
    selected_gates = sample_seu_gates(
        key=flip_key,
        layer_sizes=gate_layer_sizes,
        num_gates=20,
        strategy="greedy",
        ordered_indices=list(cfg.pool.greedy_ordered_indices),
    )
    
    masks = build_flip_masks_from_indices(
        layer_sizes=gate_layer_sizes,
        selected_gate_indices=selected_gates,
        flips_per_gate=16,
        arity=arity,
        key=flip_key,
    )
    
    # Apply flips
    mutated_logits = []
    for layer_idx, base in enumerate(base_logits):
        mask = masks[layer_idx]
        factors = jp.where(mask, -1.0, 1.0)
        mutated_logits.append(base * factors)
    
    # Test 3: Immediate damage
    _, aux = get_loss_from_wires_logits(mutated_logits, base_wires, x_data, y_data, loss_type)
    damaged_acc = float(aux[4])
    print(f"Damaged hard accuracy: {damaged_acc:.6f}")
    print(f"Damage delta: {damaged_acc - baseline_acc:.6f}")
    
    # Test 4: Create a dummy model and test recovery
    model_key = jax.random.PRNGKey(42)
    
    # Compute n_node correctly by building a dummy graph
    from boolean_nca_cc.utils.graph_builder import build_graph
    dummy_graph = build_graph(
        wires=base_wires,
        logits=base_logits,
        input_n=input_bits,
        arity=arity,
        circuit_hidden_dim=circuit_hidden_dim,
    )
    total_nodes = int(dummy_graph.n_node[0])
    print(f"Computed n_nodes for CircuitSelfAttention: {total_nodes}")
    
    # Debug: Check actual feature dimensions
    print(f"Debug - arity: {arity}")
    print(f"Debug - 2^arity: {2**arity}")
    print(f"Debug - circuit_hidden_dim: {circuit_hidden_dim}")
    
    # Extract features from the dummy graph to check actual dimensions
    features = jp.concatenate([
        dummy_graph.nodes["logits"],  # [n_node, 2^arity]
        dummy_graph.nodes["hidden"],  # [n_node, circuit_hidden_dim]
        dummy_graph.nodes["layer_pe"],  # [n_node, circuit_hidden_dim]
        dummy_graph.nodes["intra_layer_pe"],  # [n_node, circuit_hidden_dim]
        dummy_graph.nodes["loss"][:, None]  # [n_node, 1]
    ], axis=-1)
    
    actual_feature_dim = features.shape[-1]
    expected_feature_dim = 2**arity + 3*circuit_hidden_dim + 1
    print(f"Debug - Actual feature dimension: {actual_feature_dim}")
    print(f"Debug - Expected feature dimension: {expected_feature_dim}")
    print(f"Debug - Feature dimension mismatch: {actual_feature_dim - expected_feature_dim}")
    
    # Create rngs for model initialization
    from flax import nnx
    rngs = nnx.Rngs(model_key)
    
    model = CircuitSelfAttention(
        n_node=total_nodes,
        circuit_hidden_dim=circuit_hidden_dim,
        num_heads=4,
        dropout_rate=0.0,
        rngs=rngs,
    )
    
    # Initialize model parameters
    dummy_graph = jax.tree.map(
        lambda x: x[None],  # Add batch dimension
        jax.tree.map(
            lambda x: x[0],  # Take first example
            jax.tree.map(lambda x: x[None], base_wires)  # Add batch dimension
        )
    )
    # This is a simplified way to get a dummy graph - in practice you'd use build_graph
    print("Model created, testing recovery...")
    
    # Test 5: Run stepwise evaluation on damaged logits
    batch_wires = jax.tree.map(lambda x: x[None], base_wires)
    batch_logits = jax.tree.map(lambda x: x[None], mutated_logits)
    
    print("Running stepwise evaluation...")
    
    # Debug: Check the batch graph dimensions before evaluation
    from boolean_nca_cc.utils.graph_builder import build_graph
    debug_batch_graph = build_graph(
        logits=batch_logits[0],  # Take first batch item
        wires=batch_wires[0],    # Take first batch item
        input_n=input_bits,
        arity=arity,
        circuit_hidden_dim=circuit_hidden_dim,
        loss_value=0.0,
        bidirectional_edges=True,
    )
    
    debug_features = jp.concatenate([
        debug_batch_graph.nodes["logits"],
        debug_batch_graph.nodes["hidden"],
        debug_batch_graph.nodes["layer_pe"],
        debug_batch_graph.nodes["intra_layer_pe"],
        debug_batch_graph.nodes["loss"][:, None]
    ], axis=-1)
    
    print(f"Debug - Batch graph feature dimension: {debug_features.shape[-1]}")
    print(f"Debug - Batch graph n_node: {debug_batch_graph.n_node[0]}")
    
    # Create a custom model wrapper to debug the feature dimensions
    class DebugModel:
        def __init__(self, original_model):
            self.original_model = original_model
            
        def __call__(self, graph, **kwargs):
            # Debug the feature dimensions before calling the original model
            features = jp.concatenate([
                graph.nodes["logits"],
                graph.nodes["hidden"],
                graph.nodes["layer_pe"],
                graph.nodes["intra_layer_pe"],
                graph.nodes["loss"][:, None]
            ], axis=-1)
            
            print(f"Debug - Model input feature dimension: {features.shape[-1]}")
            print(f"Debug - Model input n_node: {graph.n_node[0]}")
            
            return self.original_model(graph, **kwargs)
    
    debug_model = DebugModel(model)
    
    recovery_metrics = evaluate_model_stepwise_batched(
        model=debug_model,
        batch_wires=batch_wires,
        batch_logits=batch_logits,
        x_data=x_data,
        y_data=y_data,
        input_n=input_bits,
        arity=arity,
        circuit_hidden_dim=circuit_hidden_dim,
        n_message_steps=5,
        loss_type=loss_type,
        layer_sizes=layer_sizes,
        bidirectional_edges=True,
        layer_neighbors=False,
    )
    
    print(f"Recovery trajectory (first 3 steps):")
    for i in range(min(3, len(recovery_metrics["hard_accuracy"]))):
        print(f"  Step {i}: {recovery_metrics['hard_accuracy'][i]:.6f}")
    
    print(f"Final accuracy: {recovery_metrics['hard_accuracy'][-1]:.6f}")
    print(f"Recovery delta: {recovery_metrics['hard_accuracy'][-1] - damaged_acc:.6f}")


if __name__ == "__main__":
    main()
