#!/usr/bin/env python3
"""
Test script that iterates over wiring seeds to find which ones allow backprop
to preconfigure the circuit perfectly (hard_accuracy == 1.0).

Uses config.yaml settings and tests a range of wiring seeds.
"""

import jax
import jax.numpy as jp
import hydra
from omegaconf import DictConfig, OmegaConf
import logging
import argparse
from typing import List, Tuple

from boolean_nca_cc.training.preconfigure import preconfigure_circuit_logits
from boolean_nca_cc.training.evaluation import get_loss_from_wires_logits
from boolean_nca_cc.circuits.tasks import get_task_data
from boolean_nca_cc import generate_layer_sizes

# Configure logging
log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def wiring_seed_preconfigure(cfg: DictConfig) -> None:
    """
    Iterate over wiring seeds and test which ones allow perfect preconfiguration.
    """
    # Parse command line arguments for seed range
    parser = argparse.ArgumentParser(description="Test wiring seeds for perfect preconfiguration")
    parser.add_argument(
        "--seed-start",
        type=int,
        default=0,
        help="Starting seed value (default: 0)"
    )
    parser.add_argument(
        "--seed-end",
        type=int,
        default=100,
        help="Ending seed value (exclusive, default: 100)"
    )
    parser.add_argument(
        "--seed-step",
        type=int,
        default=1,
        help="Step size for seed range (default: 1)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed results for each seed"
    )
    
    # Parse known args (Hydra will handle the rest)
    args, _ = parser.parse_known_args()
    
    print("=" * 80)
    print("WIRING SEED PRECONFIGURE TEST")
    print("=" * 80)
    
    # Print configuration
    print(f"\nConfig settings:")
    print(f"  - Task: {cfg.circuit.task}")
    print(f"  - preconfig_steps: {cfg.backprop.epochs}")
    print(f"  - preconfig_lr: {cfg.backprop.learning_rate}")
    print(f"  - loss_type: {cfg.training.loss_type}")
    print(f"  - arity: {cfg.circuit.arity}")
    print(f"  - input_bits: {cfg.circuit.input_bits}")
    print(f"  - output_bits: {cfg.circuit.output_bits}")
    print(f"\nSeed range: {args.seed_start} to {args.seed_end} (step: {args.seed_step})")
    print()
    
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
    
    # Get task data (reads from config)
    case_n = 1 << input_n  # Complete truth table: 2^input_bits
    x_data, y_data = get_task_data(
        task_name=cfg.circuit.task,
        case_n=case_n,
        max_samples=cfg.circuit.get("max_task_samples", 100000),
        sample_seed=cfg.test_seed,
        input_bits=cfg.circuit.input_bits,
        output_bits=cfg.circuit.output_bits
    )
    
    print(f"Data shapes: x_data={x_data.shape}, y_data={y_data.shape}")
    print()
    
    # Results storage
    perfect_seeds: List[int] = []
    results: List[Tuple[int, float, float, float, float]] = []  # seed, loss, hard_loss, accuracy, hard_accuracy
    
    # Iterate over wiring seeds
    seed_range = range(args.seed_start, args.seed_end, args.seed_step)
    print(f"Testing {len(seed_range)} wiring seeds...")
    print()
    
    for seed in seed_range:
        wiring_key = jax.random.PRNGKey(seed)
        
        if args.verbose:
            print(f"Testing seed {seed}...", end=" ", flush=True)
        
        try:
            # Run preconfiguration
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
            
            # Evaluate the preconfigured circuit
            loss, aux = get_loss_from_wires_logits(
                base_logits, base_wires, x_data, y_data, cfg.training.loss_type
            )
            
            # Extract metrics from aux tuple
            hard_loss, pred, pred_hard, accuracy, hard_accuracy, full_map_accuracy, res, hard_res = aux
            
            loss_val = float(loss)
            hard_loss_val = float(hard_loss)
            accuracy_val = float(accuracy)
            hard_accuracy_val = float(hard_accuracy)
            
            results.append((seed, loss_val, hard_loss_val, accuracy_val, hard_accuracy_val))
            
            # Check if perfect
            if hard_accuracy_val == 1.0:
                perfect_seeds.append(seed)
                if args.verbose:
                    print(f"✓ PERFECT (hard_accuracy={hard_accuracy_val:.6f}, loss={loss_val:.6f})")
                else:
                    print(f"Seed {seed}: ✓ PERFECT (hard_accuracy={hard_accuracy_val:.6f})")
            else:
                if args.verbose:
                    print(f"hard_accuracy={hard_accuracy_val:.6f}, loss={loss_val:.6f}")
                    
        except Exception as e:
            if args.verbose:
                print(f"✗ FAILED: {e}")
            else:
                print(f"Seed {seed}: ✗ FAILED: {e}")
            results.append((seed, float('inf'), float('inf'), 0.0, 0.0))
    
    # Print summary
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total seeds tested: {len(results)}")
    print(f"Perfect seeds (hard_accuracy == 1.0): {len(perfect_seeds)}")
    print()
    
    if perfect_seeds:
        print("Seeds with perfect preconfiguration:")
        for seed in perfect_seeds:
            # Find corresponding result
            result = next((r for r in results if r[0] == seed), None)
            if result:
                _, loss_val, hard_loss_val, accuracy_val, hard_accuracy_val = result
                print(f"  Seed {seed:4d}: hard_accuracy={hard_accuracy_val:.6f}, loss={loss_val:.6f}, hard_loss={hard_loss_val:.6f}")
        print()
    else:
        print("No seeds achieved perfect preconfiguration (hard_accuracy == 1.0)")
        print()
        print("Best seeds (top 10 by hard_accuracy):")
        sorted_results = sorted(results, key=lambda x: x[4], reverse=True)
        for seed, loss_val, hard_loss_val, accuracy_val, hard_accuracy_val in sorted_results[:10]:
            print(f"  Seed {seed:4d}: hard_accuracy={hard_accuracy_val:.6f}, loss={loss_val:.6f}, hard_loss={hard_loss_val:.6f}")
        print()
    
    # Statistics
    if results:
        accuracies = [r[4] for r in results]
        losses = [r[1] for r in results if r[1] != float('inf')]
        
        print("Statistics:")
        print(f"  Mean hard_accuracy: {sum(accuracies) / len(accuracies):.6f}")
        print(f"  Max hard_accuracy: {max(accuracies):.6f}")
        print(f"  Min hard_accuracy: {min(accuracies):.6f}")
        if losses:
            print(f"  Mean loss: {sum(losses) / len(losses):.6f}")
            print(f"  Min loss: {min(losses):.6f}")
        print()
    
    print("=" * 80)
    print("TEST COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    wiring_seed_preconfigure()

