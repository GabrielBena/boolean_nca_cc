"""
Test script for the knockout pipeline: config -> structural_perturbation -> gate_mask -> run_circuit

This script demonstrates the complete flow from configuration to circuit execution with knockouts.
"""

import jax
import jax.numpy as jp
import hydra
from omegaconf import DictConfig, OmegaConf
import logging

from boolean_nca_cc.circuits.model import gen_circuit, run_circuit, generate_layer_sizes
from boolean_nca_cc.circuits.train import create_gate_mask_from_knockout_pattern
from boolean_nca_cc.training.pool.structural_perturbation import create_reproducible_knockout_pattern
from boolean_nca_cc.circuits.tasks import get_task_data

# Configure logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def test_knockout_pipeline(cfg: DictConfig):
    """
    Test the complete knockout pipeline.
    
    Args:
        cfg: Configuration object from config.yaml
    """
    log.info("=== Testing Knockout Pipeline ===")
    log.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")
    
    # Step 1: Generate layer sizes from config
    input_n, output_n = cfg.circuit.input_bits, cfg.circuit.output_bits
    arity = cfg.circuit.arity
    
    if cfg.circuit.layer_sizes is None:
        layer_sizes = generate_layer_sizes(
            input_n, output_n, arity, layer_n=cfg.circuit.num_layers
        )
    else:
        layer_sizes = cfg.circuit.layer_sizes
    
    log.info(f"Layer sizes: {layer_sizes}")
    
    # Step 2: Generate circuit
    key = jax.random.PRNGKey(cfg.test_seed)
    wires, logits = gen_circuit(key, layer_sizes, arity=cfg.circuit.arity)
    log.info(f"Generated circuit with {len(wires)} layers")
    log.info(f"Logits shapes: {[lgt.shape for lgt in logits]}")
    
    # Step 3: Generate knockout pattern using structural_perturbation
    rng, knockout_key = jax.random.split(key)
    damage_prob = cfg.pool.persistent_knockout.damage_prob
    
    knockout_pattern = create_reproducible_knockout_pattern(
        key=knockout_key,
        layer_sizes=layer_sizes,
        damage_prob=damage_prob,
    )
    
    # Create a more aggressive knockout pattern for testing
    # Knock out more gates to make the effect more visible
    aggressive_damage_prob = min(10, len(knockout_pattern) // 4)  # Knock out ~25% of gates
    aggressive_knockout_pattern = create_reproducible_knockout_pattern(
        key=knockout_key,
        layer_sizes=layer_sizes,
        damage_prob=aggressive_damage_prob,
    )
    
    log.info(f"Knockout pattern shape: {knockout_pattern.shape}")
    log.info(f"Total gates: {len(knockout_pattern)}")
    log.info(f"Knocked out gates: {jp.sum(knockout_pattern)}")
    log.info(f"Active gates: {jp.sum(~knockout_pattern)}")
    
    log.info(f"Aggressive knockout - knocked out gates: {jp.sum(aggressive_knockout_pattern)}")
    log.info(f"Aggressive knockout - active gates: {jp.sum(~aggressive_knockout_pattern)}")
    
    # Step 4: Convert knockout pattern to gate_mask using train.py function
    gate_mask = create_gate_mask_from_knockout_pattern(knockout_pattern, layer_sizes)
    aggressive_gate_mask = create_gate_mask_from_knockout_pattern(aggressive_knockout_pattern, layer_sizes)
    
    log.info(f"Gate mask length: {len(gate_mask)}")
    for i, mask in enumerate(gate_mask):
        log.info(f"Layer {i} mask shape: {mask.shape}, active gates: {jp.sum(mask)}")
    
    # Step 5: Generate test data
    case_n = 1 << input_n
    x, y0 = get_task_data(
        cfg.circuit.task, case_n, input_bits=input_n, output_bits=output_n
    )
    log.info(f"Test data shapes: x={x.shape}, y0={y0.shape}")
    
    # Step 6: Run circuit without knockouts (baseline)
    log.info("=== Running circuit without knockouts ===")
    acts_baseline = run_circuit(logits, wires, x, gate_mask=None)
    baseline_output = acts_baseline[-1]
    baseline_accuracy = jp.mean(jp.equal(jp.round(baseline_output), y0))
    log.info(f"Baseline accuracy: {baseline_accuracy:.4f}")
    
    # Step 7: Run circuit with knockouts
    log.info("=== Running circuit with knockouts ===")
    acts_knockout = run_circuit(logits, wires, x, gate_mask=gate_mask)
    knockout_output = acts_knockout[-1]
    knockout_accuracy = jp.mean(jp.equal(jp.round(knockout_output), y0))
    log.info(f"Knockout accuracy: {knockout_accuracy:.4f}")
    
    # Step 7b: Run circuit with aggressive knockouts
    log.info("=== Running circuit with aggressive knockouts ===")
    acts_aggressive = run_circuit(logits, wires, x, gate_mask=aggressive_gate_mask)
    aggressive_output = acts_aggressive[-1]
    aggressive_accuracy = jp.mean(jp.equal(jp.round(aggressive_output), y0))
    log.info(f"Aggressive knockout accuracy: {aggressive_accuracy:.4f}")
    
    # Step 8: Compare outputs
    output_diff = jp.abs(baseline_output - knockout_output)
    aggressive_output_diff = jp.abs(baseline_output - aggressive_output)
    log.info(f"Output difference (L1): {jp.mean(output_diff):.4f}")
    log.info(f"Aggressive output difference (L1): {jp.mean(aggressive_output_diff):.4f}")
    log.info(f"Accuracy difference: {baseline_accuracy - knockout_accuracy:.4f}")
    log.info(f"Aggressive accuracy difference: {baseline_accuracy - aggressive_accuracy:.4f}")
    
    # Step 8b: Verify that gate mask is actually being applied
    log.info("=== Verifying Gate Mask Application ===")
    
    # Check that aggressive knockout has more impact than mild knockout
    mild_impact = jp.mean(output_diff)
    aggressive_impact = jp.mean(aggressive_output_diff)
    log.info(f"Mild knockout impact: {mild_impact:.6f}")
    log.info(f"Aggressive knockout impact: {aggressive_impact:.6f}")
    
    # Verify that more knockouts lead to more change (with some tolerance for randomness)
    if aggressive_impact >= mild_impact:
        log.info("✓ Gate mask application verified: more knockouts = more impact")
    else:
        log.warning("⚠ Gate mask may not be working: aggressive knockout had less impact")
    
    # Check that at least some gates are actually being knocked out
    total_knocked_out = jp.sum(aggressive_knockout_pattern)
    if total_knocked_out > 0:
        log.info(f"✓ Knockout pattern contains {total_knocked_out} knocked out gates")
    else:
        log.error("✗ No gates are being knocked out in the pattern")
    
    # Verify that the gate mask correctly reflects the knockout pattern
    total_masked_gates = sum(jp.sum(mask == 0.0) for mask in aggressive_gate_mask)
    if total_masked_gates == total_knocked_out:
        log.info("✓ Gate mask correctly reflects knockout pattern")
    else:
        log.error(f"✗ Gate mask mismatch: {total_masked_gates} masked vs {total_knocked_out} knocked out")
    
    # Step 9: Verify gate mask structure
    log.info("=== Gate Mask Structure Verification ===")
    log.info(f"Expected gate_mask length: {len(logits) + 1}")
    log.info(f"Actual gate_mask length: {len(gate_mask)}")
    
    # Verify each layer's mask shape
    for i, (lgt, mask) in enumerate(zip(logits, gate_mask[1:])):  # Skip input layer
        expected_shape = (lgt.shape[0] * lgt.shape[1],)
        actual_shape = mask.shape
        log.info(f"Layer {i+1}: expected {expected_shape}, actual {actual_shape}")
        assert actual_shape == expected_shape, f"Shape mismatch in layer {i+1}"
    
    # Verify input layer mask shape (should be broadcastable to input shape)
    input_mask_shape = gate_mask[0].shape
    expected_input_shape = (x.shape[-1],)  # Should be (input_bits,) for broadcasting
    log.info(f"Input layer: expected {expected_input_shape}, actual {input_mask_shape}")
    assert input_mask_shape == expected_input_shape, "Input mask shape mismatch"
    
    # Test broadcasting works correctly
    broadcasted_mask = gate_mask[0][None, :]  # Add batch dimension
    log.info(f"Broadcasted input mask shape: {broadcasted_mask.shape}")
    log.info(f"Input tensor shape: {x.shape}")
    # The mask should broadcast from (1, 8) to (256, 8) when applied
    assert broadcasted_mask.shape[1] == x.shape[1], "Input mask feature dimension mismatch"
    
    log.info("=== Test completed successfully! ===")
    
    return {
        "layer_sizes": layer_sizes,
        "knockout_pattern": knockout_pattern,
        "gate_mask": gate_mask,
        "baseline_accuracy": float(baseline_accuracy),
        "knockout_accuracy": float(knockout_accuracy),
        "aggressive_knockout_accuracy": float(aggressive_accuracy),
        "output_difference": float(jp.mean(output_diff)),
        "aggressive_output_difference": float(jp.mean(aggressive_output_diff)),
    }


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Main test function using Hydra for configuration.
    
    Args:
        cfg: Hydra configuration object
    """
    try:
        results = test_knockout_pipeline(cfg)
        log.info("Test results:")
        for key, value in results.items():
            if key not in ["knockout_pattern", "gate_mask"]:  # Skip large arrays
                log.info(f"  {key}: {value}")
    except Exception as e:
        log.error(f"Test failed with error: {e}")
        raise


if __name__ == "__main__":
    main() 