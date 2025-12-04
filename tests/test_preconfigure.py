#!/usr/bin/env python3
"""
Test script for preconfigure.py to verify hard accuracy after backprop optimization.
Uses config.yaml settings and prints out hard accuracy results.
"""

import jax
import jax.numpy as jp
import hydra
from omegaconf import DictConfig, OmegaConf
import logging

from boolean_nca_cc.training.preconfigure import preconfigure_circuit_logits
from boolean_nca_cc.training.evaluation import get_loss_from_wires_logits
from boolean_nca_cc.circuits.tasks import get_task_data
from boolean_nca_cc import generate_layer_sizes

# Configure logging
log = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="configs", config_name="config")
def test_preconfigure(cfg: DictConfig) -> None:
    """
    Test preconfigure function with config settings and print hard accuracy.
    """
    print("=" * 60)
    print("TESTING PRECONFIGURE FUNCTION")
    print("=" * 60)
    
    # Print configuration
    print(f"Config settings:")
    print(f"  - preconfig_steps: {cfg.backprop.epochs}")
    print(f"  - preconfig_lr: {cfg.backprop.learning_rate}")
    print(f"  - test_seed: {cfg.test_seed}")
    print(f"  - loss_type: {cfg.training.loss_type}")
    print(f"  - arity: {cfg.circuit.arity}")
    print(f"  - input_bits: {cfg.circuit.input_bits}")
    print(f"  - output_bits: {cfg.circuit.output_bits}")
    print()
    
    # Set random seed
    rng = jax.random.PRNGKey(cfg.seed)
    
    # Generate circuit layer sizes
    input_n, output_n = cfg.circuit.input_bits, cfg.circuit.output_bits
    arity = cfg.circuit.arity
    if cfg.circuit.layer_sizes is None:
        layer_sizes = generate_layer_sizes(
            input_n, output_n, arity, layer_n=cfg.circuit.num_layers
        )
    else:
        layer_sizes = cfg.circuit.layer_sizes
    
    print(f"Layer sizes: {layer_sizes}")
    
    # Get task data
    case_n = 1 << input_n  # Complete truth table: 2^input_bits
    x_data, y_data = get_task_data(
        task_name=cfg.circuit.task,
        case_n=case_n,
        input_bits=cfg.circuit.input_bits,
        output_bits=cfg.circuit.output_bits
    )
    
    print(f"Data shapes: x_data={x_data.shape}, y_data={y_data.shape}")
    print()
    
    # Create wiring key for preconfiguration
    wiring_key = jax.random.PRNGKey(cfg.test_seed)
    
    print("Running preconfiguration...")
    print(f"  - Steps: {cfg.backprop.epochs}")
    print(f"  - Learning rate: {cfg.backprop.learning_rate}")
    print(f"  - Optimizer: {cfg.backprop.optimizer}")
    print(f"  - Weight decay: {cfg.backprop.weight_decay}")
    print()
    
    # Run preconfiguration
    try:
        base_wires, base_logits = preconfigure_circuit_logits(
            wiring_key=wiring_key,
            layer_sizes=layer_sizes,
            arity=arity,
            x_data=x_data,
            y_data=y_data,
            loss_type=cfg.training.loss_type,
            steps=cfg.backprop.epochs,
            lr=cfg.backprop.learning_rate,
            optimizer=cfg.backprop.optimizer,
            weight_decay=cfg.backprop.weight_decay,
            beta1=cfg.backprop.beta1,
            beta2=cfg.backprop.beta2,
        )
        print("✓ Preconfiguration completed successfully!")
        print()
        
    except Exception as e:
        print(f"✗ Preconfiguration failed: {e}")
        return
    
    # Evaluate the preconfigured circuit
    print("Evaluating preconfigured circuit...")
    try:
        loss, aux = get_loss_from_wires_logits(
            base_logits, base_wires, x_data, y_data, cfg.training.loss_type
        )
        
        # Extract metrics from aux tuple
        hard_loss, pred, pred_hard, accuracy, hard_accuracy, res, hard_res = aux
        
        print("=" * 40)
        print("RESULTS:")
        print("=" * 40)
        print(f"Loss: {float(loss):.6f}")
        print(f"Hard Loss: {float(hard_loss):.6f}")
        print(f"Accuracy (soft): {float(accuracy):.6f}")
        print(f"Hard Accuracy: {float(hard_accuracy):.6f}")
        print()
        
        # Show some predictions vs targets
        print("Sample predictions (first 10):")
        print("Target    | Soft Pred | Hard Pred")
        print("-" * 35)
        for i in range(min(10, len(y_data))):
            target = y_data[i]
            soft_pred = pred[i]
            hard_pred = pred_hard[i]
            print(f"{target} | {soft_pred} | {hard_pred}")
        print()
        
        # Calculate additional statistics
        soft_correct = jp.equal(jp.round(pred), y_data)
        hard_correct = jp.equal(pred_hard, y_data)
        soft_accuracy_manual = jp.mean(soft_correct)
        hard_accuracy_manual = jp.mean(hard_correct)
        
        print("Manual accuracy calculation:")
        print(f"  Soft accuracy: {float(soft_accuracy_manual):.6f}")
        print(f"  Hard accuracy: {float(hard_accuracy_manual):.6f}")
        print()
        
        # Show difference between soft and hard predictions
        soft_hard_diff = jp.mean(jp.abs(pred - pred_hard))
        print(f"Mean absolute difference between soft and hard predictions: {float(soft_hard_diff):.6f}")
        
        print("=" * 60)
        print("TEST COMPLETED")
        print("=" * 60)
        
    except Exception as e:
        print(f"✗ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_preconfigure()
