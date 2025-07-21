"""
Test 2: Knockout Evaluation Pipeline Integration

This test mirrors the exact usage from run_knockout_periodic_evaluation
in training loop to verify:
1. Vocabulary → Pattern Sampling Flow (IN-dist vs OUT-dist)
2. Pattern Flow Through Pipeline 
3. Model Knockout Integration
4. Metrics Differentiation
5. ID and Training use same vocabulary (despite different shapes)
"""

import jax
import jax.numpy as jp
import pytest
from functools import partial
from typing import List, Tuple, Dict, Optional

# Import the exact functions used in the training loop
from boolean_nca_cc.training.train_loop import run_knockout_periodic_evaluation
from boolean_nca_cc.training.evaluation import (
    evaluate_circuits_in_chunks,
    evaluate_model_stepwise_batched,
)
from boolean_nca_cc.training.pool.structural_perturbation import (
    create_reproducible_knockout_pattern,
    create_knockout_vocabulary,
    extract_layer_info_from_graph,
)
from boolean_nca_cc.circuits.model import gen_circuit
from boolean_nca_cc.utils.graph_builder import build_graph
from boolean_nca_cc.models import CircuitSelfAttention
from boolean_nca_cc.training.pool.pool import initialize_graph_pool
from flax import nnx

def load_config():
    """Load configuration from config.yaml files"""
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs", "config.yaml")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    model_config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs", "model", "self_attention.yaml")
    with open(model_config_path, 'r') as f:
        model_config = yaml.safe_load(f)
    
    return config, model_config

# Test configuration matching actual usage
TEST_CONFIG = {
    'layer_sizes': [(4, 4), (8, 4), (4, 4)],  # Fixed: ensure group_size divides out_n * arity
    'input_n': 4,
    'arity': 2,
    'circuit_hidden_dim': 16,
    'eval_batch_size': 8,
    'reset_batch_size': 12,  # Different from eval to test shape independence
    'knockout_diversity': 20,
    'damage_prob': 0.3,
    'periodic_eval_test_seed': 42,
    'n_message_steps': 3,
    'loss_type': 'l4',
    'pool_size': 64,
    'reset_pool_fraction': 0.2,
}

def test_setup():
    """Create test circuit, model, and vocabulary matching training setup."""
    config = TEST_CONFIG
    
    # Create test circuit (mirrors knockout_eval_base_circuit creation)
    wiring_fixed_key = jax.random.PRNGKey(42)
    base_wires, base_logits = gen_circuit(
        wiring_fixed_key, config['layer_sizes'], arity=config['arity']
    )
    
    # Create sample graph to extract layer info (mirrors training logic)
    sample_graph = build_graph(
        logits=base_logits,
        wires=base_wires,
        input_n=config['input_n'],
        arity=config['arity'],
        circuit_hidden_dim=config['circuit_hidden_dim'],
    )
    true_layer_sizes = extract_layer_info_from_graph(sample_graph, config['input_n'])
    
    # Create vocabulary (mirrors training vocabulary creation)
    vocab_rng = jax.random.PRNGKey(config['periodic_eval_test_seed'])
    knockout_vocabulary = create_knockout_vocabulary(
        rng=vocab_rng,
        vocabulary_size=config['knockout_diversity'],
        layer_sizes=true_layer_sizes,
        damage_prob=config['damage_prob'],
        input_n=config['input_n'],
    )
    
    # Create test model
    model_key = jax.random.PRNGKey(123)
    # Calculate n_node from the sample graph
    n_node = sample_graph.nodes['logits'].shape[0]
    model = CircuitSelfAttention(
        n_node=n_node,
        circuit_hidden_dim=config['circuit_hidden_dim'],
        arity=config['arity'],
        attention_dim=64,
        num_heads=2,
        num_layers=2,
        mlp_dim=128,
        mlp_dim_multiplier=2,
        dropout_rate=0.0,
        rngs=nnx.Rngs(params=model_key),
        zero_init=False,
        re_zero_update=False,
    )
    
    # Create test data
    batch_size = 8
    x_data = jax.random.bernoulli(jax.random.PRNGKey(456), 0.5, (batch_size, config['input_n']))
    y_data = jax.random.bernoulli(jax.random.PRNGKey(789), 0.5, (batch_size, 1))
    
    return {
        'base_wires': base_wires,
        'base_logits': base_logits,
        'knockout_vocabulary': knockout_vocabulary,
        'true_layer_sizes': true_layer_sizes,
        'model': model,
        'x_data': x_data,
        'y_data': y_data,
        'config': config,
    }

def test_vocabulary_pattern_sampling_flow(test_setup):
    """
    Test Point 1: Vocabulary → Pattern Sampling Flow
    Verify IN-dist samples from vocabulary, OUT-dist generates fresh patterns
    """
    setup = test_setup
    config = setup['config']
    
    # Mirror exact sampling logic from run_knockout_periodic_evaluation
    
    # 1. IN-distribution sampling (from vocabulary)
    id_rng = jax.random.PRNGKey(config['periodic_eval_test_seed'])
    pattern_indices = jax.random.choice(
        id_rng, len(setup['knockout_vocabulary']), 
        shape=(config['eval_batch_size'],), replace=True
    )
    in_knockout_patterns = setup['knockout_vocabulary'][pattern_indices]
    
    # 2. OUT-of-distribution generation (fresh patterns)
    pattern_creator_fn = partial(
        create_reproducible_knockout_pattern,
        layer_sizes=setup['true_layer_sizes'],
        damage_prob=config['damage_prob'],
        input_n=config['input_n'],
    )
    ood_rng = jax.random.PRNGKey(config['periodic_eval_test_seed'] + 1)
    out_pattern_keys = jax.random.split(ood_rng, config['eval_batch_size'])
    out_knockout_patterns = jax.vmap(pattern_creator_fn)(out_pattern_keys)
    
    # Verify patterns have correct shapes and properties
    assert in_knockout_patterns.shape == out_knockout_patterns.shape
    assert in_knockout_patterns.dtype == jp.bool_
    assert out_knockout_patterns.dtype == jp.bool_
    
    # Verify IN-dist patterns are actually from vocabulary
    for i, pattern in enumerate(in_knockout_patterns):
        # Check this pattern exists in vocabulary
        vocab_match = jp.any(jp.all(pattern == setup['knockout_vocabulary'], axis=1))
        assert vocab_match, f"IN-dist pattern {i} not found in vocabulary"
    
    # Verify OUT-dist patterns are different from IN-dist
    patterns_identical = jp.all(in_knockout_patterns == out_knockout_patterns)
    assert not patterns_identical, "IN-dist and OUT-dist patterns should be different"
    
    print("✅ Vocabulary → Pattern Sampling Flow verified")

def test_training_id_vocabulary_consistency(test_setup):
    """
    Test Point 5: ID and Training use same vocabulary (despite different shapes)
    Verify both training reset and ID evaluation sample from same vocabulary
    """
    setup = test_setup
    config = setup['config']
    
    # Simulate training reset sampling (different shape)
    reset_key = jax.random.PRNGKey(100)
    pattern_sample_key = jax.random.fold_in(reset_key, 42)
    reset_pattern_indices = jax.random.choice(
        pattern_sample_key, len(setup['knockout_vocabulary']), 
        shape=(config['reset_batch_size'],), replace=True  # Different size!
    )
    training_reset_patterns = setup['knockout_vocabulary'][reset_pattern_indices]
    
    # Simulate ID evaluation sampling (different shape, same vocabulary)
    id_rng = jax.random.PRNGKey(config['periodic_eval_test_seed'])
    eval_pattern_indices = jax.random.choice(
        id_rng, len(setup['knockout_vocabulary']), 
        shape=(config['eval_batch_size'],), replace=True  # Different size!
    )
    id_eval_patterns = setup['knockout_vocabulary'][eval_pattern_indices]
    
    # Verify both use same vocabulary (but different shapes)
    assert training_reset_patterns.shape[0] != id_eval_patterns.shape[0]
    assert training_reset_patterns.shape[1:] == id_eval_patterns.shape[1:]  # Same pattern dimensions
    
    # Verify both sets contain only vocabulary patterns
    for pattern in training_reset_patterns:
        vocab_match = jp.any(jp.all(pattern == setup['knockout_vocabulary'], axis=1))
        assert vocab_match, "Training reset pattern not from vocabulary"
        
    for pattern in id_eval_patterns:
        vocab_match = jp.any(jp.all(pattern == setup['knockout_vocabulary'], axis=1))
        assert vocab_match, "ID evaluation pattern not from vocabulary"
    
    # Verify they can sample different patterns (independence despite same source)
    if config['reset_batch_size'] == config['eval_batch_size']:
        # Only compare if same size
        different_sampling = not jp.all(training_reset_patterns == id_eval_patterns)
        assert different_sampling, "Should sample independently even from same vocabulary"
    
    print("✅ Training and ID vocabulary consistency verified")

def test_pattern_flow_through_pipeline(test_setup):
    """
    Test Point 2: Pattern Flow Through Pipeline
    Verify: run_knockout_periodic_evaluation → evaluate_circuits_in_chunks → 
            evaluate_model_stepwise_batched → vmap_model
    """
    setup = test_setup
    config = setup['config']
    
    # Create knockout config for evaluation
    knockout_config = {
        'damage_prob': config['damage_prob'],
        'enabled': True,
    }
    
    # Mirror exact call from training loop
    ko_eval_results = run_knockout_periodic_evaluation(
        model=setup['model'],
        knockout_vocabulary=setup['knockout_vocabulary'],
        base_wires=setup['base_wires'],
        base_logits=setup['base_logits'],
        knockout_config=knockout_config,
        periodic_eval_test_seed=config['periodic_eval_test_seed'],
        x_data=setup['x_data'],
        y_data=setup['y_data'],
        input_n=config['input_n'],
        arity=config['arity'],
        circuit_hidden_dim=config['circuit_hidden_dim'],
        n_message_steps=config['n_message_steps'],
        loss_type=config['loss_type'],
        epoch=0,
        wandb_run=None,
        eval_batch_size=config['eval_batch_size'],
        log_stepwise=False,
        layer_sizes=config['layer_sizes'],
        use_scan=False,
    )
    
    # Verify pipeline completed successfully
    assert ko_eval_results is not None
    assert 'final_metrics_in' in ko_eval_results
    assert 'final_metrics_out' in ko_eval_results
    
    # Verify metrics structure
    final_metrics_in = ko_eval_results['final_metrics_in']
    final_metrics_out = ko_eval_results['final_metrics_out']
    
    expected_keys = [
        'eval_ko_in/final_loss', 'eval_ko_in/final_hard_loss',
        'eval_ko_in/final_accuracy', 'eval_ko_in/final_hard_accuracy'
    ]
    for key in expected_keys:
        assert key in final_metrics_in, f"Missing IN-dist metric: {key}"
    
    expected_keys_out = [
        'eval_ko_out/final_loss', 'eval_ko_out/final_hard_loss', 
        'eval_ko_out/final_accuracy', 'eval_ko_out/final_hard_accuracy'
    ]
    for key in expected_keys_out:
        assert key in final_metrics_out, f"Missing OUT-dist metric: {key}"
    
    print("✅ Pattern Flow Through Pipeline verified")

def test_model_knockout_integration(test_setup):
    """
    Test Point 3: Model Knockout Integration
    Verify model(g, knockout_pattern=k) correctly applies attention masking
    """
    setup = test_setup
    config = setup['config']
    
    # Create test graph
    test_graph = build_graph(
        logits=setup['base_logits'],
        wires=setup['base_wires'],
        input_n=config['input_n'],
        arity=config['arity'],
        circuit_hidden_dim=config['circuit_hidden_dim'],
    )
    
    # Create knockout pattern
    knockout_pattern = setup['knockout_vocabulary'][0]  # Use first pattern
    
    # Test model with and without knockout
    updated_graph_with_ko = setup['model'](test_graph, knockout_pattern=knockout_pattern)
    updated_graph_without_ko = setup['model'](test_graph, knockout_pattern=None)
    
    # Verify model runs successfully with knockout patterns
    assert updated_graph_with_ko is not None
    assert updated_graph_without_ko is not None
    
    # Verify graphs have same structure but potentially different node features
    assert updated_graph_with_ko.nodes.keys() == updated_graph_without_ko.nodes.keys()
    assert updated_graph_with_ko.nodes['logits'].shape == updated_graph_without_ko.nodes['logits'].shape
    
    # Verify knockout affects node features (they should be different)
    logits_difference = jp.mean(jp.abs(
        updated_graph_with_ko.nodes['logits'] - updated_graph_without_ko.nodes['logits']
    ))
    
    # With knockouts, some nodes should have different updates
    # (This is a statistical test - with high probability they should differ)
    if jp.any(knockout_pattern):  # Only if there are actual knockouts
        assert logits_difference > 1e-6, "Knockout should affect node features"
    
    print("✅ Model Knockout Integration verified")

def test_metrics_differentiation(test_setup):
    """
    Test Point 4: Metrics Differentiation  
    Verify IN-dist vs OUT-dist produce different final metrics
    """
    setup = test_setup
    config = setup['config']
    
    knockout_config = {
        'damage_prob': config['damage_prob'],
        'enabled': True,
    }
    
    # Run full evaluation pipeline
    ko_eval_results = run_knockout_periodic_evaluation(
        model=setup['model'],
        knockout_vocabulary=setup['knockout_vocabulary'],
        base_wires=setup['base_wires'],
        base_logits=setup['base_logits'],
        knockout_config=knockout_config,
        periodic_eval_test_seed=config['periodic_eval_test_seed'],
        x_data=setup['x_data'],
        y_data=setup['y_data'],
        input_n=config['input_n'],
        arity=config['arity'],
        circuit_hidden_dim=config['circuit_hidden_dim'],
        n_message_steps=config['n_message_steps'],
        loss_type=config['loss_type'],
        epoch=0,
        wandb_run=None,
        eval_batch_size=config['eval_batch_size'],
        log_stepwise=False,
        layer_sizes=config['layer_sizes'],
        use_scan=False,
    )
    
    final_metrics_in = ko_eval_results['final_metrics_in']
    final_metrics_out = ko_eval_results['final_metrics_out']
    
    # Compare corresponding metrics
    in_loss = final_metrics_in['eval_ko_in/final_loss']
    out_loss = final_metrics_out['eval_ko_out/final_loss']
    
    in_accuracy = final_metrics_in['eval_ko_in/final_accuracy'] 
    out_accuracy = final_metrics_out['eval_ko_out/final_accuracy']
    
    # Debug output
    print(f"DEBUG: IN-dist loss: {in_loss}, accuracy: {in_accuracy}")
    print(f"DEBUG: OUT-dist loss: {out_loss}, accuracy: {out_accuracy}")
    
    # Check knockout patterns
    vocab_rng = jax.random.PRNGKey(config['periodic_eval_test_seed'])
    pattern_indices = jax.random.choice(
        vocab_rng, len(setup['knockout_vocabulary']), 
        shape=(config['eval_batch_size'],), replace=True
    )
    in_knockout_patterns = setup['knockout_vocabulary'][pattern_indices]
    
    ood_rng = jax.random.PRNGKey(config['periodic_eval_test_seed'] + 1)
    pattern_creator_fn = partial(
        create_reproducible_knockout_pattern,
        layer_sizes=setup['true_layer_sizes'],
        damage_prob=config['damage_prob'],
        input_n=config['input_n'],
    )
    out_pattern_keys = jax.random.split(ood_rng, config['eval_batch_size'])
    out_knockout_patterns = jax.vmap(pattern_creator_fn)(out_pattern_keys)
    
    print(f"DEBUG: IN-dist patterns sum: {jp.sum(in_knockout_patterns)}")
    print(f"DEBUG: OUT-dist patterns sum: {jp.sum(out_knockout_patterns)}")
    print(f"DEBUG: Patterns identical: {jp.all(in_knockout_patterns == out_knockout_patterns)}")
    
    # Verify metrics are different (statistical test)
    # Due to different knockout patterns, metrics should differ
    loss_difference = abs(float(in_loss) - float(out_loss))
    accuracy_difference = abs(float(in_accuracy) - float(out_accuracy))
    
    # At least one metric should be different (very high probability)
    metrics_differ = loss_difference > 1e-6 or accuracy_difference > 1e-6
    assert metrics_differ, f"IN/OUT metrics should differ: loss_diff={loss_difference}, acc_diff={accuracy_difference}"
    
    # Verify proper averaging (metrics should be reasonable values)
    assert 0 <= float(in_accuracy) <= 1, f"IN accuracy should be in [0,1]: {in_accuracy}"
    assert 0 <= float(out_accuracy) <= 1, f"OUT accuracy should be in [0,1]: {out_accuracy}"
    assert float(in_loss) >= 0, f"IN loss should be non-negative: {in_loss}"
    assert float(out_loss) >= 0, f"OUT loss should be non-negative: {out_loss}"
    
    print("✅ Metrics Differentiation verified")

def test_end_to_end_pipeline_integration():
    """
    Integration test verifying the complete knockout evaluation pipeline
    matches the exact usage pattern from training loop
    """
    print("🧪 Running Test 2: Knockout Evaluation Pipeline Integration")
    print("=" * 60)
    
    # Run all verification points
    test_setup_data = test_setup()
    
    test_vocabulary_pattern_sampling_flow(test_setup_data)
    test_training_id_vocabulary_consistency(test_setup_data)
    test_pattern_flow_through_pipeline(test_setup_data)
    test_model_knockout_integration(test_setup_data)
    test_metrics_differentiation(test_setup_data)
    
    print("=" * 60)
    print("✅ All Test 2 verification points passed!")
    print("   - Vocabulary sampling flow works correctly")
    print("   - Training and ID use same vocabulary despite different shapes")
    print("   - Pattern flow through pipeline maintains integrity")
    print("   - Model knockout integration applies attention masking")
    print("   - IN-dist vs OUT-dist metrics differentiate properly")

if __name__ == "__main__":
    test_end_to_end_pipeline_integration() 