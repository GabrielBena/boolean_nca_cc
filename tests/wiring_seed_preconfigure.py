#!/usr/bin/env python3
"""
Test script that iterates over wiring seeds to find which ones allow backprop
to preconfigure the circuit perfectly (hard_accuracy == 1.0).

Uses config.yaml settings and tests a range of wiring seeds.
Supports testing multiple tasks and reports seeds that achieve perfect accuracy on ALL tasks.

PARALLELIZATION: Uses JAX vmap to batch-process multiple seeds simultaneously.
"""

import jax
import jax.numpy as jp
import jax.lax as lax
import numpy as np
import optax
import hydra
from omegaconf import DictConfig, OmegaConf
import logging
import argparse
import sys
from typing import List, Tuple, Dict, Optional
from collections import defaultdict
from functools import partial
from tqdm.auto import tqdm

from boolean_nca_cc.circuits.model import gen_circuit, make_nops, gen_wires
from boolean_nca_cc.circuits.train import (
    res2loss, binary_cross_entropy, compute_accuracy,
    loss_f_l4, loss_f_bce, grad_loss_f_l4, grad_loss_f_bce,
)
from boolean_nca_cc.training.evaluation import get_loss_from_wires_logits
from boolean_nca_cc.circuits.tasks import get_task_data, TASKS
from boolean_nca_cc import generate_layer_sizes

# Configure logging
log = logging.getLogger(__name__)


# ============================================================================
# JIT-COMPILABLE BATCHED PRECONFIGURATION
# ============================================================================

def _gen_circuit_from_key(key, layer_sizes_tuple, arity):
    """
    Generate a circuit with JIT-compatible structure.
    
    Args:
        key: JAX random key
        layer_sizes_tuple: Tuple of (nodes, group_size) tuples (hashable for JIT)
        arity: Fan-in per gate
    
    Returns:
        Tuple of (wires_list, logits_list) as nested arrays
    """
    layer_sizes = list(layer_sizes_tuple)
    in_n = layer_sizes[0][0]
    all_wires = []
    all_logits = []
    
    for out_n, group_size in layer_sizes[1:]:
        wires = gen_wires(key, in_n, out_n, arity, group_size)
        logits = make_nops(out_n, arity, group_size)
        _, key = jax.random.split(key)
        in_n = out_n
        all_wires.append(wires)
        all_logits.append(logits)
    
    return all_wires, all_logits


def _make_preconfigure_step_fn(loss_type: str):
    """
    Create a JIT-compilable single optimization step function.
    
    Args:
        loss_type: "l4" or "bce"
    
    Returns:
        JIT-compiled step function
    """
    if loss_type == "bce":
        grad_fn = grad_loss_f_bce
    else:
        grad_fn = grad_loss_f_l4
    
    @jax.jit
    def step_fn(logits, opt_state, wires, x_data, y_data, opt_update_fn):
        """Single optimization step."""
        (loss, aux), grad = grad_fn(logits, wires, x_data, y_data, None)
        updates, new_opt_state = opt_update_fn(grad, opt_state, logits)
        new_logits = optax.apply_updates(logits, updates)
        return new_logits, new_opt_state, loss
    
    return step_fn


def preconfigure_single_seed_jit(
    wiring_key: jax.Array,
    layer_sizes_tuple: Tuple[Tuple[int, int], ...],
    arity: int,
    x_data: jp.ndarray,
    y_data: jp.ndarray,
    loss_type: str,
    steps: int,
    lr: float,
    optimizer: str,
    weight_decay: float,
    beta1: float,
    beta2: float,
) -> Tuple[List[jp.ndarray], List[jp.ndarray], float]:
    """
    JIT-friendly preconfiguration for a single seed using lax.fori_loop.
    
    Returns:
        Tuple of (wires, final_logits, final_loss)
    """
    # Generate circuit
    wires, logits = _gen_circuit_from_key(wiring_key, layer_sizes_tuple, arity)
    
    # Create optimizer
    if optimizer == "adamw":
        opt = optax.adamw(lr, b1=beta1, b2=beta2, weight_decay=weight_decay)
    else:
        opt = optax.adam(lr, b1=beta1, b2=beta2)
    
    opt_state = opt.init(logits)
    
    # Get the appropriate gradient function
    if loss_type == "bce":
        grad_fn = grad_loss_f_bce
    else:
        grad_fn = grad_loss_f_l4
    
    def body_fn(i, carry):
        """Single optimization step for lax.fori_loop."""
        logits, opt_state, _ = carry
        (loss, aux), grad = grad_fn(logits, wires, x_data, y_data, None)
        updates, new_opt_state = opt.update(grad, opt_state, logits)
        new_logits = optax.apply_updates(logits, updates)
        return (new_logits, new_opt_state, loss)
    
    # Initial state
    init_carry = (logits, opt_state, jp.float32(0.0))
    
    # Run optimization loop
    final_logits, final_opt_state, final_loss = lax.fori_loop(
        0, steps, body_fn, init_carry
    )
    
    return wires, final_logits, final_loss


def evaluate_single_circuit(
    logits: List[jp.ndarray],
    wires: List[jp.ndarray],
    x_data: jp.ndarray,
    y_data: jp.ndarray,
    loss_type: str,
) -> Tuple[float, float, float, float]:
    """
    Evaluate a single preconfigured circuit.
    
    Returns:
        Tuple of (loss, hard_loss, accuracy, hard_accuracy)
    """
    loss, aux = get_loss_from_wires_logits(logits, wires, x_data, y_data, loss_type)
    hard_loss, pred, pred_hard, accuracy, hard_accuracy, full_map_accuracy, res, hard_res = aux
    return float(loss), float(hard_loss), float(accuracy), float(hard_accuracy)


def process_seeds_batch(
    seeds: List[int],
    layer_sizes_tuple: Tuple[Tuple[int, int], ...],
    arity: int,
    x_data: jp.ndarray,
    y_data: jp.ndarray,
    loss_type: str,
    steps: int,
    lr: float,
    optimizer: str,
    weight_decay: float,
    beta1: float,
    beta2: float,
) -> List[Tuple[int, List[jp.ndarray], List[jp.ndarray], float]]:
    """
    Process a batch of seeds in parallel using vmap.
    
    Returns:
        List of (seed, wires, logits, final_loss) tuples
    """
    # Create batched keys
    keys = jp.stack([jax.random.PRNGKey(seed) for seed in seeds])
    
    # Create the vmapped preconfigure function
    # Note: We need to handle the tree structure of wires/logits carefully
    
    # Partial application of fixed arguments
    preconfigure_fn = partial(
        preconfigure_single_seed_jit,
        layer_sizes_tuple=layer_sizes_tuple,
        arity=arity,
        x_data=x_data,
        y_data=y_data,
        loss_type=loss_type,
        steps=steps,
        lr=lr,
        optimizer=optimizer,
        weight_decay=weight_decay,
        beta1=beta1,
        beta2=beta2,
    )
    
    # vmap over the wiring keys
    batched_preconfigure = jax.vmap(preconfigure_fn)
    
    # Run batched preconfiguration
    batched_wires, batched_logits, batched_losses = batched_preconfigure(keys)
    
    # Unpack results
    results = []
    batch_size = len(seeds)
    for i in range(batch_size):
        # Extract individual wires and logits from batched results
        wires_i = [w[i] for w in batched_wires]
        logits_i = [l[i] for l in batched_logits]
        loss_i = float(batched_losses[i])
        results.append((seeds[i], wires_i, logits_i, loss_i))
    
    return results


def evaluate_circuits_batch(
    circuits: List[Tuple[List[jp.ndarray], List[jp.ndarray]]],
    x_data: jp.ndarray,
    y_data: jp.ndarray,
    loss_type: str,
) -> List[Tuple[float, float, float, float]]:
    """
    Evaluate a batch of circuits in parallel using vmap.
    
    Args:
        circuits: List of (wires, logits) tuples
    
    Returns:
        List of (loss, hard_loss, accuracy, hard_accuracy) tuples
    """
    if not circuits:
        return []
    
    # Stack into batched arrays
    wires_list = [c[0] for c in circuits]
    logits_list = [c[1] for c in circuits]
    
    # Stack each layer separately
    batched_wires = [jp.stack([w[layer_idx] for w in wires_list]) 
                    for layer_idx in range(len(wires_list[0]))]
    batched_logits = [jp.stack([l[layer_idx] for l in logits_list]) 
                     for layer_idx in range(len(logits_list[0]))]
    
    # vmap the evaluation function
    vmap_get_loss = jax.vmap(
        lambda logits, wires: get_loss_from_wires_logits(
            logits, wires, x_data, y_data, loss_type
        )
    )
    
    losses, aux = vmap_get_loss(batched_logits, batched_wires)
    hard_losses, _, _, accuracies, hard_accuracies, _, _, _ = aux
    
    # Return individual results
    results = []
    for i in range(len(circuits)):
        results.append((
            float(losses[i]),
            float(hard_losses[i]),
            float(accuracies[i]),
            float(hard_accuracies[i])
        ))
    
    return results


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def wiring_seed_preconfigure(cfg: DictConfig) -> None:
    """
    Iterate over wiring seeds and test which ones allow perfect preconfiguration.
    Supports multiple tasks and reports seeds that achieve perfect accuracy on ALL tasks.
    
    PARALLELIZED VERSION: Processes seeds in batches using JAX vmap for efficiency.
    
    Usage:
        # Basic usage with Hydra overrides:
        python tests/wiring_seed_preconfigure.py backprop.epochs=200
        
        # Use custom test parameters via +test.* overrides:
        python tests/wiring_seed_preconfigure.py +test.seed_start=0 +test.seed_end=100 +test.tasks=binary_multiply,add
        
        # Control batch size for parallelization:
        python tests/wiring_seed_preconfigure.py +test.batch_size=32
        
        # All test parameters:
        #   +test.seed_start: First seed to test (default: 0)
        #   +test.seed_end: Last seed to test (exclusive, default: 100)
        #   +test.seed_step: Step between seeds (default: 1)
        #   +test.tasks: Comma-separated task names (default: circuit.task from config)
        #   +test.batch_size: Seeds to process in parallel (default: 16)
        #   +test.verbose: Enable verbose output (default: false)
    """
    # Extract test parameters from cfg (with defaults) - uses +test.* overrides
    test_cfg = cfg.get("test", {})
    
    # Default task comes from circuit.task in config.yaml
    default_task = cfg.circuit.get("task", "add")
    
    class Args:
        seed_start = test_cfg.get("seed_start", 0)
        seed_end = test_cfg.get("seed_end", 100)
        seed_step = test_cfg.get("seed_step", 1)
        verbose = test_cfg.get("verbose", False)
        tasks = test_cfg.get("tasks", default_task)  # Default to circuit.task from config
        batch_size = test_cfg.get("batch_size", 16)
    
    args = Args()
    
    # Parse tasks
    task_names = [t.strip() for t in args.tasks.split(",")]
    for task_name in task_names:
        if task_name not in TASKS:
            raise ValueError(f"Unknown task: {task_name}. Available: {list(TASKS.keys())}")
    
    # BP epochs comes from config (can be overridden via Hydra: backprop.epochs=200)
    bp_epochs = cfg.backprop.epochs
    
    print("=" * 80)
    print("WIRING SEED PRECONFIGURE TEST (PARALLELIZED)")
    print("=" * 80)
    
    # Print configuration
    print(f"\nConfig settings:")
    print(f"  - Tasks: {', '.join(task_names)}")
    print(f"  - preconfig_steps: {bp_epochs}")
    print(f"  - preconfig_lr: {cfg.backprop.learning_rate}")
    print(f"  - loss_type: {cfg.training.loss_type}")
    print(f"  - arity: {cfg.circuit.arity}")
    print(f"  - input_bits: {cfg.circuit.input_bits}")
    print(f"  - output_bits: {cfg.circuit.output_bits}")
    print(f"  - batch_size: {args.batch_size} (parallel seeds)")
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
    
    # Convert to tuple for JIT compatibility
    layer_sizes_tuple = tuple(tuple(ls) for ls in layer_sizes)
    
    print(f"Layer sizes: {layer_sizes}")
    print()
    
    # Get task data for all tasks
    case_n = 1 << input_n  # Complete truth table: 2^input_bits
    task_data: Dict[str, Tuple[jp.ndarray, jp.ndarray]] = {}
    
    for task_name in task_names:
        x_data, y_data = get_task_data(
            task_name=task_name,
            case_n=case_n,
            max_samples=cfg.circuit.get("max_task_samples", 100000),
            sample_seed=cfg.test_seed,
            input_bits=cfg.circuit.input_bits,
            output_bits=cfg.circuit.output_bits
        )
        task_data[task_name] = (x_data, y_data)
        print(f"Task '{task_name}': x_data={x_data.shape}, y_data={y_data.shape}")
    
    print()
    
    # Generate seed list
    all_seeds = list(range(args.seed_start, args.seed_end, args.seed_step))
    total_seeds = len(all_seeds)
    
    print(f"Testing {total_seeds} wiring seeds across {len(task_names)} task(s)...")
    print(f"Processing in batches of {args.batch_size}...")
    print()
    
    # Results storage: seed -> task -> (loss, hard_loss, accuracy, hard_accuracy)
    results: Dict[int, Dict[str, Tuple[float, float, float, float]]] = defaultdict(dict)
    perfect_seeds_all_tasks: List[int] = []
    
    # Backprop config
    bp_lr = cfg.backprop.learning_rate
    bp_optimizer = cfg.backprop.optimizer
    bp_weight_decay = cfg.backprop.weight_decay
    bp_beta1 = cfg.backprop.beta1
    bp_beta2 = cfg.backprop.beta2
    loss_type = cfg.training.loss_type
    
    # Process each task
    for task_name in task_names:
        x_data, y_data = task_data[task_name]
        
        print(f"\n{'='*40}")
        print(f"Processing task: {task_name}")
        print(f"{'='*40}")
        
        # Process seeds in batches
        seed_batches = [all_seeds[i:i + args.batch_size] 
                       for i in range(0, total_seeds, args.batch_size)]
        
        # Use tqdm for progress
        pbar = tqdm(seed_batches, desc=f"  {task_name}", unit="batch")
        
        for batch_seeds in pbar:
            try:
                # Batch preconfigure
                batch_results = process_seeds_batch(
                    seeds=batch_seeds,
                    layer_sizes_tuple=layer_sizes_tuple,
                    arity=arity,
                    x_data=x_data,
                    y_data=y_data,
                    loss_type=loss_type,
                    steps=bp_epochs,
                    lr=bp_lr,
                    optimizer=bp_optimizer,
                    weight_decay=bp_weight_decay,
                    beta1=bp_beta1,
                    beta2=bp_beta2,
                )
                
                # Batch evaluate
                circuits = [(wires, logits) for (seed, wires, logits, loss) in batch_results]
                eval_results = evaluate_circuits_batch(circuits, x_data, y_data, loss_type)
                
                # Store results
                for (seed, wires, logits, _), (loss, hard_loss, acc, hard_acc) in zip(batch_results, eval_results):
                    results[seed][task_name] = (loss, hard_loss, acc, hard_acc)
                    
                    if args.verbose:
                        status = "✓ PERFECT" if hard_acc == 1.0 else "✗"
                        print(f"  Seed {seed:4d}: {status} hard_accuracy={hard_acc:.6f}, loss={loss:.6f}")
                
                # Update progress bar with current stats
                perfect_count = sum(1 for s in batch_seeds if results[s].get(task_name, (0,0,0,0))[3] == 1.0)
                pbar.set_postfix({"perfect": perfect_count, "batch": len(batch_seeds)})
                
            except Exception as e:
                # Handle batch failures gracefully
                for seed in batch_seeds:
                    results[seed][task_name] = (float('inf'), float('inf'), 0.0, 0.0)
                if args.verbose:
                    print(f"  Batch {batch_seeds[0]}-{batch_seeds[-1]} FAILED: {e}")
    
    # Find seeds that are perfect on ALL tasks
    for seed in all_seeds:
        if all(results[seed].get(task, (0,0,0,0))[3] == 1.0 for task in task_names):
            perfect_seeds_all_tasks.append(seed)
    
    # Print summary
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total seeds tested: {len(results)}")
    print(f"Seeds with perfect accuracy on ALL tasks: {len(perfect_seeds_all_tasks)}")
    print()
    
    # Per-task statistics
    print("Per-task statistics:")
    for task_name in task_names:
        task_perfect_seeds = [
            seed for seed, task_results in results.items()
            if task_name in task_results and task_results[task_name][3] == 1.0
        ]
        task_accuracies = [
            task_results[task_name][3] for seed, task_results in results.items()
            if task_name in task_results and task_results[task_name][3] != 0.0
        ]
        
        print(f"  {task_name:20s}: {len(task_perfect_seeds):4d} perfect seeds", end="")
        if task_accuracies:
            acc_arr = np.array(task_accuracies)
            print(f" (mean: {acc_arr.mean():.6f}, std: {acc_arr.std():.6f}, max: {acc_arr.max():.6f})")
        else:
            print()
    print()
    
    if perfect_seeds_all_tasks:
        print("Seeds with perfect preconfiguration on ALL tasks:")
        for seed in perfect_seeds_all_tasks:
            print(f"  Seed {seed:4d}:", end="")
            for task_name in task_names:
                if task_name in results[seed]:
                    loss_val, hard_loss_val, accuracy_val, hard_accuracy_val = results[seed][task_name]
                    print(f"  {task_name}: hard_acc={hard_accuracy_val:.6f}, loss={loss_val:.6f}", end="")
            print()
        print()
    else:
        print("No seeds achieved perfect preconfiguration on ALL tasks")
        print()
        
        # Show best seeds per task
        print("Best seeds per task (top 5 by hard_accuracy):")
        for task_name in task_names:
            task_results_list = [
                (seed, task_results[task_name])
                for seed, task_results in results.items()
                if task_name in task_results and task_results[task_name][3] != 0.0
            ]
            if task_results_list:
                sorted_task_results = sorted(task_results_list, key=lambda x: x[1][3], reverse=True)
                print(f"  {task_name}:")
                for seed, (loss_val, hard_loss_val, accuracy_val, hard_accuracy_val) in sorted_task_results[:5]:
                    print(f"    Seed {seed:4d}: hard_accuracy={hard_accuracy_val:.6f}, loss={loss_val:.6f}")
        print()
    
    # Overall statistics
    all_accuracies = []
    all_losses = []
    for seed_results in results.values():
        for task_results in seed_results.values():
            if task_results[3] != 0.0:  # hard_accuracy
                all_accuracies.append(task_results[3])
            if task_results[0] != float('inf'):  # loss
                all_losses.append(task_results[0])
    
    if all_accuracies:
        acc_arr = np.array(all_accuracies)
        print("Overall statistics (across all tasks and seeds):")
        print(f"  Mean hard_accuracy: {acc_arr.mean():.6f}")
        print(f"  Std hard_accuracy:  {acc_arr.std():.6f}")
        print(f"  Max hard_accuracy:  {acc_arr.max():.6f}")
        print(f"  Min hard_accuracy:  {acc_arr.min():.6f}")
        if all_losses:
            loss_arr = np.array(all_losses)
            print(f"  Mean loss: {loss_arr.mean():.6f}")
            print(f"  Std loss:  {loss_arr.std():.6f}")
            print(f"  Min loss:  {loss_arr.min():.6f}")
        print()
    
    print("=" * 80)
    print("TEST COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    wiring_seed_preconfigure()
