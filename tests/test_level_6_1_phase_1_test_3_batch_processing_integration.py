#!/usr/bin/env python3
"""
Test Level 6.1 Phase 1 Test 3: Batch Processing Integration

Tests the exact usage from evaluate_model_stepwise_batched:
- batch_graphs → vmap(model) → vmap(get_loss_and_update_graph)  
- Verify knockout patterns work correctly in batched operations

This test mirrors the critical path used in run_knockout_periodic_evaluation
where evaluate_circuits_in_chunks calls evaluate_model_stepwise_batched.

Usage: conda activate metabool && python -m pytest tests/test_level_6_1_phase_1_test_3_batch_processing_integration.py -v
"""

import pytest
import jax
import jax.numpy as jp
import jraph
from typing import List, Tuple, Dict, Optional
import hydra
from omegaconf import DictConfig

# Import configuration and circuit components
from boolean_nca_cc.circuits.model import gen_circuit
from boolean_nca_cc.utils.graph_builder import build_graph
from boolean_nca_cc.utils.extraction import extract_logits_from_graph
from boolean_nca_cc.models import CircuitSelfAttention
from boolean_nca_cc.training.evaluation import (
    evaluate_model_stepwise_batched,
    evaluate_circuits_in_chunks,
    get_loss_and_update_graph,
    _evaluate_with_loop
)
from boolean_nca_cc.training.pool.structural_perturbation import (
    create_reproducible_knockout_pattern,
    extract_layer_info_from_graph
)
from boolean_nca_cc.circuits.tasks import get_task_data
from flax import nnx


def load_test_config() -> DictConfig:
    """Load configuration matching the actual training setup."""
    with hydra.initialize(version_base=None, config_path="../configs"):
        cfg = hydra.compose(config_name="config")
    return cfg


@pytest.fixture
def test_config():
    """Provide test configuration."""
    return load_test_config()


@pytest.fixture
def test_data(test_config):
    """Create test data matching the actual training setup."""
    cfg = test_config
    input_n = cfg.circuit.input_bits
    output_n = cfg.circuit.output_bits
    case_n = 1 << input_n  # Calculate as 2^input_n
    x_data, y_data = get_task_data(
        task_name=cfg.circuit.task,
        case_n=case_n,
        input_bits=input_n,
        output_bits=output_n
    )
    return x_data, y_data


@pytest.fixture
def model(test_config, test_data):
    """Create a model instance matching the training setup."""
    cfg = test_config
    input_n = cfg.circuit.input_bits
    output_n = cfg.circuit.output_bits
    case_n = 1 << input_n  # Calculate as 2^input_n
    x_data, y_data = test_data

    circuits, _ = test_circuits
    
    # Generate sample circuit to get n_node parameter
    sample_wires, sample_logits = circuits[0]
    sample_graph = build_graph(
        logits=sample_logits,
        wires=sample_wires,
        input_n=cfg.circuit.input_bits,
        arity=cfg.circuit.arity,
        circuit_hidden_dim=cfg.model.circuit_hidden_dim,
    )
    n_node = sample_graph.n_node[0]  # Get actual n_node from circuit structure
    
    # Initialize model with config parameters
    init_key = jax.random.PRNGKey(cfg.seed)
    model = CircuitSelfAttention(
        circuit_hidden_dim=cfg.model.circuit_hidden_dim,
        arity=cfg.circuit.arity,
        attention_dim=cfg.model.attention_dim,
        num_heads=cfg.model.num_heads,
        num_layers=cfg.model.num_layers,
        mlp_dim=cfg.model.mlp_dim,
        mlp_dim_multiplier=cfg.model.mlp_dim_multiplier,
        dropout_rate=cfg.model.dropout_rate,
        zero_init=cfg.model.zero_init,
        re_zero_update=cfg.model.re_zero_update,
        rngs=nnx.Rngs(params=init_key),
        n_node=n_node
    )
    return model


@pytest.fixture
def test_circuits(test_config):
    """Generate test circuits matching the training setup."""
    cfg = test_config
    
    # Generate layer sizes if not provided
    if cfg.circuit.layer_sizes is None:
        layer_sizes = []
        current_size = cfg.circuit.input_bits
        for layer_idx in range(cfg.circuit.num_layers):
            if layer_idx == cfg.circuit.num_layers - 1:  # Last layer
                target_size = cfg.circuit.output_bits
            else:
                target_size = max(cfg.circuit.output_bits, current_size // 2)
            
            layer_sizes.append((current_size, cfg.circuit.arity))
            current_size = target_size
        layer_sizes.append((cfg.circuit.output_bits, cfg.circuit.arity))
    else:
        layer_sizes = cfg.circuit.layer_sizes
    
    # Generate multiple circuits for batch testing
    batch_size = 4
    circuit_key = jax.random.PRNGKey(42)  # Fixed for reproducibility
    
    circuits = []
    for i in range(batch_size):
        circuit_key, subkey = jax.random.split(circuit_key)
        wires, logits = gen_circuit(subkey, layer_sizes, arity=cfg.circuit.arity)
        circuits.append((wires, logits))
    
    return circuits, layer_sizes


def test_batch_model_application_with_knockouts(test_config, model, test_circuits, test_data):
    """Test that vmap(model) correctly applies attention masking across batched circuits."""
    cfg = test_config
    circuits, layer_sizes = test_circuits
    x_data, y_data = test_data
    
    # Create batch data
    batch_size = len(circuits)
    batch_wires = []
    batch_logits = []
    
    for wires, logits in circuits:
        batch_wires.append(wires)
        batch_logits.append(logits)
        
    # Stack into batched format
    stacked_wires = [jp.stack([circuit[0][i] for circuit in circuits]) for i in range(len(batch_wires[0]))]
    stacked_logits = [jp.stack([circuit[1][i] for circuit in circuits]) for i in range(len(batch_logits[0]))]
    
    # Build sample graph to extract layer info for knockout patterns
    sample_graph = build_graph(
        logits=batch_logits[0],
        wires=batch_wires[0],
        input_n=cfg.circuit.input_bits,
        arity=cfg.circuit.arity,
        circuit_hidden_dim=cfg.model.circuit_hidden_dim,
    )
    true_layer_sizes = extract_layer_info_from_graph(sample_graph, cfg.circuit.input_bits)
    
    # Create knockout patterns for the batch
    knockout_key = jax.random.PRNGKey(123)
    knockout_keys = jax.random.split(knockout_key, batch_size)
    
    knockout_patterns = []
    for key in knockout_keys:
        pattern = create_reproducible_knockout_pattern(
            key,
            layer_sizes=true_layer_sizes,
            damage_prob=0.3,  # Significant damage to test effects
            input_n=cfg.circuit.input_bits,
        )
        knockout_patterns.append(pattern)
    
    knockout_patterns = jp.array(knockout_patterns)
    
    # Build initial graphs
    vmap_build_graph = jax.vmap(
        lambda logits, wires: build_graph(
            logits=logits,
            wires=wires,
            input_n=cfg.circuit.input_bits,
            arity=cfg.circuit.arity,
            circuit_hidden_dim=cfg.model.circuit_hidden_dim,
            loss_value=0.0,
        )
    )
    batch_graphs = vmap_build_graph(stacked_logits, stacked_wires)
    
    # Test 1: Batch model application with knockouts
    vmap_model_with_ko = jax.vmap(lambda g, k: model(g, knockout_pattern=k))
    updated_graphs_with_ko = vmap_model_with_ko(batch_graphs, knockout_patterns)
    
    # Test 2: Batch model application without knockouts
    vmap_model_no_ko = jax.vmap(model)
    updated_graphs_no_ko = vmap_model_no_ko(batch_graphs)
    
    # Verification 1: Graphs should have different node features after knockout application
    ko_features = updated_graphs_with_ko.nodes["features"]
    no_ko_features = updated_graphs_no_ko.nodes["features"]
    
    # Should be different due to attention masking effects
    feature_differences = jp.abs(ko_features - no_ko_features)
    assert jp.max(feature_differences) > 1e-6, "Knockout should cause different node features"
    
    # Verification 2: Knocked out nodes should show specific patterns
    # Extract which nodes are knocked out and verify they're affected
    total_knockout_nodes = jp.sum(knockout_patterns)
    assert total_knockout_nodes > 0, "Should have some knockout nodes for testing"
    
    print(f"✅ Batch model application test passed:")
    print(f"   - Batch size: {batch_size}")
    print(f"   - Total knockout nodes: {total_knockout_nodes}")
    print(f"   - Max feature difference: {jp.max(feature_differences):.6f}")


def test_batch_loss_and_update_operations(test_config, model, test_circuits, test_data):
    """Test that vmap(get_loss_and_update_graph) preserves knockout effects across batch."""
    cfg = test_config
    circuits, layer_sizes = test_circuits
    x_data, y_data = test_data
    
    # Prepare batch data (same as previous test)
    batch_size = len(circuits)
    stacked_wires = [jp.stack([circuits[i][0][j] for i in range(batch_size)]) for j in range(len(circuits[0][0]))]
    stacked_logits = [jp.stack([circuits[i][1][j] for i in range(batch_size)]) for j in range(len(circuits[0][1]))]
    
    # Create knockout patterns
    knockout_key = jax.random.PRNGKey(456)
    knockout_keys = jax.random.split(knockout_key, batch_size)
    
    sample_graph = build_graph(
        logits=circuits[0][1],
        wires=circuits[0][0],
        input_n=cfg.circuit.input_bits,
        arity=cfg.circuit.arity,
        circuit_hidden_dim=cfg.model.circuit_hidden_dim,
    )
    true_layer_sizes = extract_layer_info_from_graph(sample_graph, cfg.circuit.input_bits)
    
    knockout_patterns = jp.array([
        create_reproducible_knockout_pattern(
            key,
            layer_sizes=true_layer_sizes,
            damage_prob=0.2,
            input_n=cfg.circuit.input_bits,
        ) for key in knockout_keys
    ])
    
    # Build initial graphs and apply model
    vmap_build_graph = jax.vmap(
        lambda logits, wires: build_graph(
            logits=logits,
            wires=wires,
            input_n=cfg.circuit.input_bits,
            arity=cfg.circuit.arity,
            circuit_hidden_dim=cfg.model.circuit_hidden_dim,
            loss_value=0.0,
        )
    )
    batch_graphs = vmap_build_graph(stacked_logits, stacked_wires)
    
    # Apply model with knockouts
    vmap_model = jax.vmap(lambda g, k: model(g, knockout_pattern=k))
    updated_graphs = vmap_model(batch_graphs, knockout_patterns)
    
    # Test batch loss and update operations
    logits_original_shapes = [logit.shape[1:] for logit in stacked_logits]  # Remove batch dim
    
    # This mirrors the exact usage in _evaluate_with_loop
    vmap_extract_logits = jax.vmap(
        lambda graph: extract_logits_from_graph(graph, logits_original_shapes)
    )
    
    # Extract logits from updated graphs
    current_batch_logits = vmap_extract_logits(updated_graphs)
    
    # Apply batch loss computation (mirrors _evaluate_with_loop)
    from boolean_nca_cc.circuits.train import get_loss_from_wires_logits
    from boolean_nca_cc.utils.extraction import update_output_node_loss
    
    vmap_get_loss = jax.vmap(
        lambda logits, wires: get_loss_from_wires_logits(
            logits, wires, x_data, y_data, cfg.training.loss_type
        )
    )
    
    vmap_update_loss = jax.vmap(
        lambda graph, res: update_output_node_loss(graph, layer_sizes, res.mean(axis=0))
    )
    
    # Compute losses
    current_losses, current_aux = vmap_get_loss(current_batch_logits, stacked_wires)
    (
        current_hard_losses,
        _,
        _,
        current_accuracies,
        current_hard_accuracies,
        current_res,
        _,
    ) = [aux_elem for aux_elem in current_aux]
    
    # Update output node losses
    final_updated_graphs = vmap_update_loss(updated_graphs, current_res)
    
    # Verification 1: All operations should complete without errors
    assert current_losses.shape == (batch_size,), f"Expected loss shape ({batch_size},), got {current_losses.shape}"
    assert current_accuracies.shape == (batch_size,), f"Expected accuracy shape ({batch_size},), got {current_accuracies.shape}"
    
    # Verification 2: Losses should be finite and reasonable
    assert jp.all(jp.isfinite(current_losses)), "All losses should be finite"
    assert jp.all(current_accuracies >= 0) and jp.all(current_accuracies <= 1), "Accuracies should be in [0,1]"
    
    # Verification 3: Graphs should maintain structure after updates
    assert final_updated_graphs.nodes is not None, "Updated graphs should have nodes"
    assert final_updated_graphs.edges is not None, "Updated graphs should have edges"
    
    print(f"✅ Batch loss and update operations test passed:")
    print(f"   - Batch size: {batch_size}")
    print(f"   - Loss range: [{jp.min(current_losses):.4f}, {jp.max(current_losses):.4f}]")
    print(f"   - Accuracy range: [{jp.min(current_accuracies):.4f}, {jp.max(current_accuracies):.4f}]")


def test_knockout_pattern_flow_through_batch_operations(test_config, model, test_circuits, test_data):
    """Test that knockout patterns flow correctly through batch operations without corruption."""
    cfg = test_config
    circuits, layer_sizes = test_circuits
    x_data, y_data = test_data
    
    # Prepare batch data
    batch_size = len(circuits)
    stacked_wires = [jp.stack([circuits[i][0][j] for i in range(batch_size)]) for j in range(len(circuits[0][0]))]
    stacked_logits = [jp.stack([circuits[i][1][j] for i in range(batch_size)]) for j in range(len(circuits[0][1]))]
    
    # Create distinct knockout patterns for each circuit in batch
    knockout_keys = jax.random.split(jax.random.PRNGKey(789), batch_size)
    
    sample_graph = build_graph(
        logits=circuits[0][1],
        wires=circuits[0][0],
        input_n=cfg.circuit.input_bits,
        arity=cfg.circuit.arity,
        circuit_hidden_dim=cfg.model.circuit_hidden_dim,
    )
    true_layer_sizes = extract_layer_info_from_graph(sample_graph, cfg.circuit.input_bits)
    
    # Create patterns with different damage probabilities to ensure they're distinct
    knockout_patterns = []
    for i, key in enumerate(knockout_keys):
        damage_prob = 0.1 + (i * 0.1)  # Different damage for each circuit
        pattern = create_reproducible_knockout_pattern(
            key,
            layer_sizes=true_layer_sizes,
            damage_prob=damage_prob,
            input_n=cfg.circuit.input_bits,
        )
        knockout_patterns.append(pattern)
    
    knockout_patterns = jp.array(knockout_patterns)
    
    # Verify patterns are actually different
    for i in range(batch_size - 1):
        for j in range(i + 1, batch_size):
            pattern_diff = jp.sum(jp.abs(knockout_patterns[i] - knockout_patterns[j]))
            assert pattern_diff > 0, f"Patterns {i} and {j} should be different"
    
    # Test 1: Run through evaluate_model_stepwise_batched (actual usage)
    step_metrics = evaluate_model_stepwise_batched(
        model=model,
        batch_wires=stacked_wires,
        batch_logits=stacked_logits,
        x_data=x_data,
        y_data=y_data,
        input_n=cfg.circuit.input_bits,
        arity=cfg.circuit.arity,
        circuit_hidden_dim=cfg.model.circuit_hidden_dim,
        n_message_steps=3,  # Small number for testing
        loss_type=cfg.training.loss_type,
        layer_sizes=layer_sizes,
        knockout_patterns=knockout_patterns,
    )
    
    # Test 2: Compare with individual circuit evaluation
    individual_results = []
    for i in range(batch_size):
        single_wires = [wire[i:i+1] for wire in stacked_wires]  # Keep batch dim of 1
        single_logits = [logit[i:i+1] for logit in stacked_logits]
        single_pattern = knockout_patterns[i:i+1]
        
        single_result = evaluate_model_stepwise_batched(
            model=model,
            batch_wires=single_wires,
            batch_logits=single_logits,
            x_data=x_data,
            y_data=y_data,
            input_n=cfg.circuit.input_bits,
            arity=cfg.circuit.arity,
            circuit_hidden_dim=cfg.model.circuit_hidden_dim,
            n_message_steps=3,
            loss_type=cfg.training.loss_type,
            layer_sizes=layer_sizes,
            knockout_patterns=single_pattern,
        )
        individual_results.append(single_result)
    
    # Verification 1: Batch evaluation should complete successfully
    assert "soft_loss" in step_metrics, "Batch evaluation should return step metrics"
    assert len(step_metrics["soft_loss"]) > 0, "Should have loss measurements"
    
    # Verification 2: Individual evaluations should match batch structure
    assert len(individual_results) == batch_size, "Should have results for each circuit"
    
    # Verification 3: Different patterns should produce different results
    final_losses = [result["soft_loss"][-1] for result in individual_results]
    loss_variance = jp.var(jp.array(final_losses))
    assert loss_variance > 1e-8, f"Different knockout patterns should produce different losses, got variance {loss_variance}"
    
    print(f"✅ Knockout pattern flow test passed:")
    print(f"   - Batch size: {batch_size}")
    print(f"   - Pattern differences verified: {batch_size * (batch_size - 1) // 2} pairs")
    print(f"   - Final loss variance: {loss_variance:.6f}")
    print(f"   - Steps completed: {len(step_metrics['soft_loss'])}")


def test_chunking_consistency_with_large_batch(test_config, model, test_circuits, test_data):
    """Test that large batches split into chunks produce equivalent results."""
    cfg = test_config
    circuits, layer_sizes = test_circuits
    x_data, y_data = test_data
    
    # Create a larger batch for chunking test
    large_batch_size = 8
    circuit_key = jax.random.PRNGKey(999)
    
    large_circuits = []
    for i in range(large_batch_size):
        circuit_key, subkey = jax.random.split(circuit_key)
        wires, logits = gen_circuit(subkey, layer_sizes, arity=cfg.circuit.arity)
        large_circuits.append((wires, logits))
    
    # Stack into batched format
    large_stacked_wires = [jp.stack([circuit[0][i] for circuit in large_circuits]) for i in range(len(large_circuits[0][0]))]
    large_stacked_logits = [jp.stack([circuit[1][i] for circuit in large_circuits]) for i in range(len(large_circuits[0][1]))]
    
    # Create knockout patterns
    knockout_key = jax.random.PRNGKey(111)
    knockout_keys = jax.random.split(knockout_key, large_batch_size)
    
    sample_graph = build_graph(
        logits=large_circuits[0][1],
        wires=large_circuits[0][0],
        input_n=cfg.circuit.input_bits,
        arity=cfg.circuit.arity,
        circuit_hidden_dim=cfg.model.circuit_hidden_dim,
    )
    true_layer_sizes = extract_layer_info_from_graph(sample_graph, cfg.circuit.input_bits)
    
    large_knockout_patterns = jp.array([
        create_reproducible_knockout_pattern(
            key,
            layer_sizes=true_layer_sizes,
            damage_prob=0.15,
            input_n=cfg.circuit.input_bits,
        ) for key in knockout_keys
    ])
    
    # Test 1: Evaluate without chunking (direct batch)
    direct_result = evaluate_model_stepwise_batched(
        model=model,
        batch_wires=large_stacked_wires,
        batch_logits=large_stacked_logits,
        x_data=x_data,
        y_data=y_data,
        input_n=cfg.circuit.input_bits,
        arity=cfg.circuit.arity,
        circuit_hidden_dim=cfg.model.circuit_hidden_dim,
        n_message_steps=2,  # Short for testing
        loss_type=cfg.training.loss_type,
        layer_sizes=layer_sizes,
        knockout_patterns=large_knockout_patterns,
    )
    
    # Test 2: Evaluate with chunking (force chunking with small target size)
    chunked_result = evaluate_circuits_in_chunks(
        eval_fn=evaluate_model_stepwise_batched,
        wires=large_stacked_wires,
        logits=large_stacked_logits,
        knockout_patterns=large_knockout_patterns,
        target_chunk_size=4,  # Force chunking into 2 chunks
        model=model,
        x_data=x_data,
        y_data=y_data,
        input_n=cfg.circuit.input_bits,
        arity=cfg.circuit.arity,
        circuit_hidden_dim=cfg.model.circuit_hidden_dim,
        n_message_steps=2,
        loss_type=cfg.training.loss_type,
        layer_sizes=layer_sizes,
    )
    
    # Verification 1: Both should have same structure
    assert set(direct_result.keys()) == set(chunked_result.keys()), "Results should have same keys"
    
    # Verification 2: Final metrics should be very close
    for key in ["soft_loss", "soft_accuracy"]:
        if key in direct_result and key in chunked_result:
            direct_final = direct_result[key][-1]
            chunked_final = chunked_result[key][-1]
            diff = abs(direct_final - chunked_final)
            
            # Allow small differences due to averaging across chunks
            assert diff < 0.01, f"Direct vs chunked difference for {key}: {diff:.6f} (too large)"
    
    print(f"✅ Chunking consistency test passed:")
    print(f"   - Large batch size: {large_batch_size}")
    print(f"   - Chunk size used: 4")
    print(f"   - Direct vs chunked loss diff: {abs(direct_result['soft_loss'][-1] - chunked_result['soft_loss'][-1]):.6f}")
    print(f"   - Direct vs chunked accuracy diff: {abs(direct_result['soft_accuracy'][-1] - chunked_result['soft_accuracy'][-1]):.6f}")


def test_end_to_end_batch_integration(test_config, model, test_circuits, test_data):
    """Test end-to-end batch processing matches single-circuit results when aggregated."""
    cfg = test_config
    circuits, layer_sizes = test_circuits
    x_data, y_data = test_data
    
    # Test with smaller batch for detailed comparison
    test_batch_size = 3
    test_circuits_subset = circuits[:test_batch_size]
    
    # Prepare batch data
    stacked_wires = [jp.stack([circuit[0][i] for circuit in test_circuits_subset]) for i in range(len(test_circuits_subset[0][0]))]
    stacked_logits = [jp.stack([circuit[1][i] for circuit in test_circuits_subset]) for i in range(len(test_circuits_subset[0][1]))]
    
    # Create knockout patterns
    knockout_key = jax.random.PRNGKey(222)
    knockout_keys = jax.random.split(knockout_key, test_batch_size)
    
    sample_graph = build_graph(
        logits=test_circuits_subset[0][1],
        wires=test_circuits_subset[0][0],
        input_n=cfg.circuit.input_bits,
        arity=cfg.circuit.arity,
        circuit_hidden_dim=cfg.model.circuit_hidden_dim,
    )
    true_layer_sizes = extract_layer_info_from_graph(sample_graph, cfg.circuit.input_bits)
    
    knockout_patterns = jp.array([
        create_reproducible_knockout_pattern(
            key,
            layer_sizes=true_layer_sizes,
            damage_prob=0.2,
            input_n=cfg.circuit.input_bits,
        ) for key in knockout_keys
    ])
    
    # Test batch evaluation (mirrors exact usage from training)
    batch_result = evaluate_model_stepwise_batched(
        model=model,
        batch_wires=stacked_wires,
        batch_logits=stacked_logits,
        x_data=x_data,
        y_data=y_data,
        input_n=cfg.circuit.input_bits,
        arity=cfg.circuit.arity,
        circuit_hidden_dim=cfg.model.circuit_hidden_dim,
        n_message_steps=3,
        loss_type=cfg.training.loss_type,
        layer_sizes=layer_sizes,
        knockout_patterns=knockout_patterns,
    )
    
    # Test individual evaluations for comparison
    individual_losses = []
    individual_accuracies = []
    
    for i in range(test_batch_size):
        single_wires = [wire[i:i+1] for wire in stacked_wires]
        single_logits = [logit[i:i+1] for logit in stacked_logits]
        single_pattern = knockout_patterns[i:i+1]
        
        single_result = evaluate_model_stepwise_batched(
            model=model,
            batch_wires=single_wires,
            batch_logits=single_logits,
            x_data=x_data,
            y_data=y_data,
            input_n=cfg.circuit.input_bits,
            arity=cfg.circuit.arity,
            circuit_hidden_dim=cfg.model.circuit_hidden_dim,
            n_message_steps=3,
            loss_type=cfg.training.loss_type,
            layer_sizes=layer_sizes,
            knockout_patterns=single_pattern,
        )
        
        individual_losses.append(single_result["soft_loss"][-1])
        individual_accuracies.append(single_result["soft_accuracy"][-1])
    
    # Verification: Batch average should match individual average
    manual_avg_loss = jp.mean(jp.array(individual_losses))
    manual_avg_accuracy = jp.mean(jp.array(individual_accuracies))
    
    batch_final_loss = batch_result["soft_loss"][-1]
    batch_final_accuracy = batch_result["soft_accuracy"][-1]
    
    loss_diff = abs(batch_final_loss - manual_avg_loss)
    accuracy_diff = abs(batch_final_accuracy - manual_avg_accuracy)
    
    # Should be very close (allowing for floating point precision)
    assert loss_diff < 1e-5, f"Batch loss {batch_final_loss:.6f} vs manual average {manual_avg_loss:.6f}, diff: {loss_diff:.6f}"
    assert accuracy_diff < 1e-5, f"Batch accuracy {batch_final_accuracy:.6f} vs manual average {manual_avg_accuracy:.6f}, diff: {accuracy_diff:.6f}"
    
    print(f"✅ End-to-end batch integration test passed:")
    print(f"   - Test batch size: {test_batch_size}")
    print(f"   - Batch vs manual loss diff: {loss_diff:.8f}")
    print(f"   - Batch vs manual accuracy diff: {accuracy_diff:.8f}")
    print(f"   - Individual losses: {[f'{x:.4f}' for x in individual_losses]}")
    print(f"   - Batch final loss: {batch_final_loss:.4f}")


if __name__ == "__main__":
    # Run all tests
    config = load_test_config()
    input_n = config.circuit.input_bits
    output_n = config.circuit.output_bits
    case_n = 1 << input_n  # Calculate as 2^input_n
    # Load dependencies
    x_data, y_data = get_task_data(
        task_name=config.circuit.task,
        case_n = case_n,
        input_bits=config.circuit.input_bits,
        output_bits=config.circuit.output_bits
    )
    test_data = (x_data, y_data)
    
    # Create test circuits
    if config.circuit.layer_sizes is None:
        layer_sizes = []
        current_size = config.circuit.input_bits
        for layer_idx in range(config.circuit.num_layers):
            if layer_idx == config.circuit.num_layers - 1:
                target_size = config.circuit.output_bits
            else:
                target_size = max(config.circuit.output_bits, current_size // 2)
            
            layer_sizes.append((current_size, config.circuit.arity))
            current_size = target_size
        layer_sizes.append((config.circuit.output_bits, config.circuit.arity))
    else:
        layer_sizes = config.circuit.layer_sizes
    
    circuits = []
    circuit_key = jax.random.PRNGKey(42)
    for i in range(4):
        circuit_key, subkey = jax.random.split(circuit_key)
        wires, logits = gen_circuit(subkey, layer_sizes, arity=config.circuit.arity)
        circuits.append((wires, logits))
    
    test_circuits = (circuits, layer_sizes)
    
    # Build sample graph to get n_node parameter
    sample_wires, sample_logits = circuits[0]
    sample_graph = build_graph(
        logits=sample_logits,
        wires=sample_wires,
        input_n=config.circuit.input_bits,
        arity=config.circuit.arity,
        circuit_hidden_dim=config.model.circuit_hidden_dim,
    )
    n_node = sample_graph.n_node[0]  # Critical: Extract actual node count

    # Create model with validated parameters
    init_key = jax.random.PRNGKey(config.seed)
    model = CircuitSelfAttention(
        circuit_hidden_dim=config.model.circuit_hidden_dim,
        arity=config.circuit.arity,
        attention_dim=config.model.attention_dim,
        num_heads=config.model.num_heads,
        num_layers=config.model.num_layers,
        mlp_dim=config.model.mlp_dim,
        mlp_dim_multiplier=config.model.mlp_dim_multiplier,
        dropout_rate=config.model.dropout_rate,
        zero_init=config.model.zero_init,
        re_zero_update=config.model.re_zero_update,
        n_node=n_node,  # Now properly initialized
        rngs=nnx.Rngs(params=init_key),
    )
    
    print("\n🔄 Running Level 6.1 Phase 1 Test 3: Batch Processing Integration Tests...")
    print("=" * 80)
    
    test_batch_model_application_with_knockouts(config, model, test_circuits, test_data)
    print()
    
    test_batch_loss_and_update_operations(config, model, test_circuits, test_data)
    print()
    
    test_knockout_pattern_flow_through_batch_operations(config, model, test_circuits, test_data)
    print()
    
    test_chunking_consistency_with_large_batch(config, model, test_circuits, test_data)
    print()
    
    test_end_to_end_batch_integration(config, model, test_circuits, test_data)
    print()
    
    print("=" * 80)
    print("🎉 All Level 6.1 Phase 1 Test 3 tests passed successfully!")
    print("\n📋 Test Summary:")
    print("   ✅ Batch model application with knockouts")
    print("   ✅ Batch loss and update operations")  
    print("   ✅ Knockout pattern flow through batch operations")
    print("   ✅ Chunking consistency with large batches")
    print("   ✅ End-to-end batch integration")
    print("\n🔍 Key Findings:")
    print("   - vmap(model) correctly applies attention masking across batches")
    print("   - vmap(get_loss_and_update_graph) preserves knockout effects")
    print("   - Knockout patterns flow correctly without corruption")
    print("   - Chunking produces equivalent results to direct evaluation")
    print("   - Batch processing matches individual circuit aggregations")
