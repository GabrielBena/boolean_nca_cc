"""
Single-gate knockout test: test each gate individually to see which gates affect accuracy.

- Fix wiring seed for reproducible circuit wiring
- Preconfigure logits via backprop (no structure change)
- Build eligible flat indices (skip input and output layers)
- For each eligible gate, knock it out individually and check accuracy drop

Outputs CSV with columns:
  gate_index,hard_accuracy_damaged,hard_accuracy_drop,is_used

Usage:
  python tests/single_gate_knockout_test.py \
    --config-name config \
    --output results/single_gate_knockout_results.csv

Respects Hydra config at configs/config.yaml for task/circuit/loss parameters.
"""

from __future__ import annotations

import os
import csv
from typing import Tuple, List

import jax
import jax.numpy as jp
from omegaconf import DictConfig, open_dict
import hydra
from tqdm import tqdm

from boolean_nca_cc.circuits.model import generate_layer_sizes, gen_circuit
from boolean_nca_cc.circuits.tasks import get_task_data
from boolean_nca_cc.training.evaluation import get_loss_from_wires_logits
from boolean_nca_cc.training.preconfigure import preconfigure_circuit_logits


def _ensure_layer_sizes(cfg: DictConfig) -> List[Tuple[int, int]]:
    input_n, output_n = cfg.circuit.input_bits, cfg.circuit.output_bits
    arity = cfg.circuit.arity
    if cfg.circuit.layer_sizes is None:
        layer_sizes = generate_layer_sizes(input_n, output_n, arity, layer_n=cfg.circuit.num_layers)
        with open_dict(cfg):
            cfg.circuit.layer_sizes = layer_sizes
    else:
        layer_sizes = cfg.circuit.layer_sizes
    return layer_sizes


def _compute_hard_accuracy(logits, wires, x, y, loss_type: str) -> float:
    _, aux = get_loss_from_wires_logits(logits, wires, x, y, loss_type)
    hard_accuracy = float(aux[4])
    return hard_accuracy


def _compute_all_metrics(logits, wires, x, y, loss_type: str):
    """Compute all metrics and return as a dictionary for diagnostic purposes."""
    loss, aux = get_loss_from_wires_logits(logits, wires, x, y, loss_type)
    hard_loss, pred, pred_hard, accuracy, hard_accuracy, full_map_accuracy, res, hard_res = aux
    return {
        "loss": float(loss),
        "hard_loss": float(hard_loss),
        "accuracy": float(accuracy),
        "hard_accuracy": float(hard_accuracy),
    }


def _apply_knockout_to_logits(
    logits: List[jp.ndarray],
    knockout_pattern: jp.ndarray,
    layer_sizes: List[Tuple[int, int]],
) -> List[jp.ndarray]:
    """
    Convert a node-level knockout boolean mask (True=knockout) into per-layer gate masks and zero-out corresponding gate logits.
    Uses the layer_sizes (total_gates per layer) to slice the flat pattern. Skips input layer.
    """
    offsets = []
    idx = layer_sizes[0][0]
    for total_gates, _group in layer_sizes[1:]:
        offsets.append((idx, idx + total_gates))
        idx += total_gates

    masked_logits: List[jp.ndarray] = []
    for layer_idx, layer_logits in enumerate(logits):
        start, end = offsets[layer_idx]
        layer_mask_flat = knockout_pattern[start:end]

        total_gates, group_size = layer_sizes[1 + layer_idx]
        group_n = total_gates // group_size

        layer_mask_grid = layer_mask_flat.reshape(group_n, group_size)
        gate_mask = jp.where(layer_mask_grid[..., None], 0.0, 1.0)

        masked = layer_logits * gate_mask
        masked_logits.append(masked)

    return masked_logits


def _eligible_flat_indices(layer_sizes: List[Tuple[int, int]]) -> jp.ndarray:
    total_nodes = sum(total_gates for total_gates, _ in layer_sizes)
    eligible: List[jp.ndarray] = []
    current_idx = 0
    output_layer_idx = len(layer_sizes) - 1
    for layer_idx, (total_gates, _group_size) in enumerate(layer_sizes):
        layer_end = current_idx + total_gates
        if layer_idx != 0 and layer_idx != output_layer_idx:
            eligible.append(jp.arange(current_idx, layer_end))
        current_idx = layer_end
    if not eligible:
        return jp.zeros((0,), dtype=jp.int32)
    return jp.concatenate(eligible)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    # Resolve layer sizes
    layer_sizes = _ensure_layer_sizes(cfg)

    # Data for the task
    input_n = cfg.circuit.input_bits
    output_n = cfg.circuit.output_bits
    case_n = 1 << input_n
    x_data, y_data = get_task_data(
        cfg.circuit.task, 
        case_n, 
        max_samples=cfg.circuit.get("max_task_samples", 100000),
        sample_seed=cfg.test_seed,
        input_bits=input_n, 
        output_bits=output_n
    )

    # Fix wiring seed for reproducible wiring
    wiring_seed = cfg.test_seed
    wiring_key = jax.random.PRNGKey(wiring_seed)

    # Preconfigure logits on fixed wiring
    bp_cfg = cfg.backprop if hasattr(cfg, "backprop") else {}
    training_cfg = cfg.training if hasattr(cfg, "training") else {}
    
    pre_steps = bp_cfg.get("epochs", 200)
    pre_lr = bp_cfg.get("learning_rate", 1)
    pre_optimizer = bp_cfg.get("optimizer", "adam")
    pre_weight_decay = float(bp_cfg.get("weight_decay", 1e-1))
    pre_beta1 = float(bp_cfg.get("beta1", 0.8))
    pre_beta2 = float(bp_cfg.get("beta2", 0.8))
    
    print("=" * 80)
    print("SINGLE-GATE KNOCKOUT TEST")
    print("=" * 80)
    print(f"\nTask Configuration:")
    print(f"  - Task: {cfg.circuit.task}")
    print(f"  - Input bits: {input_n}")
    print(f"  - Output bits: {output_n}")
    print(f"  - Case count: {case_n} (2^{input_n})")
    print(f"  - Loss type: {cfg.training.loss_type}")
    print(f"  - Arity: {cfg.circuit.arity}")
    print(f"  - Layer sizes: {layer_sizes}")
    
    print(f"\nRunning preconfiguration...")
    pre_wires, pre_logits = preconfigure_circuit_logits(
        wiring_key=wiring_key,
        layer_sizes=layer_sizes,
        arity=cfg.circuit.arity,
        x_data=x_data,
        y_data=y_data,
        loss_type=cfg.training.loss_type,
        steps=int(pre_steps),
        lr=float(pre_lr),
        optimizer=pre_optimizer,
        weight_decay=pre_weight_decay,
        beta1=pre_beta1,
        beta2=pre_beta2,
    )

    # Compute baseline metrics
    baseline_metrics = _compute_all_metrics(pre_logits, pre_wires, x_data, y_data, cfg.training.loss_type)
    baseline_hard_acc = baseline_metrics["hard_accuracy"]
    
    # Eligible indices (skip input and output layers)
    eligible = list(map(int, list(_eligible_flat_indices(layer_sizes))))

    # Total nodes
    total_nodes = sum(total_gates for total_gates, _ in layer_sizes)
    
    print(f"\nBaseline Metrics:")
    print(f"  - Hard accuracy: {baseline_hard_acc:.6f}")
    print(f"  - Eligible gate indices: {len(eligible)} (skipping input/output layers)")
    print(f"  - Total nodes: {total_nodes}")
    print("=" * 80)
    print()

    # Output path
    output_csv = hydra.utils.to_absolute_path(os.environ.get("SINGLE_GATE_OUTPUT", "results/single_gate_knockout_results.csv"))
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    results_rows = []

    # Test each gate individually
    for gate_idx in tqdm(eligible, desc="Testing gates", unit="gate"):
        # Create mask with only this gate knocked out
        mask = jp.zeros(total_nodes, dtype=jp.bool_)
        mask = mask.at[gate_idx].set(True)

        # Apply knockout and compute accuracy
        damaged_logits = _apply_knockout_to_logits(pre_logits, mask, layer_sizes)
        damaged_hard_acc = _compute_hard_accuracy(damaged_logits, pre_wires, x_data, y_data, cfg.training.loss_type)
        drop = baseline_hard_acc - damaged_hard_acc
        
        # Gate is "used" if knocking it out causes any accuracy drop
        is_used = drop > 0.0
        
        results_rows.append((gate_idx, damaged_hard_acc, drop, is_used))
        
        # Live print
        status = "USED" if is_used else "unused"
        print(f"Gate {gate_idx:4d}: Hard accuracy: {damaged_hard_acc:.6f} | "
              f"Drop: {drop:.6f} | {status}")

    # Write CSV
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(("gate_index", "hard_accuracy_damaged", "hard_accuracy_drop", "is_used"))
        writer.writerows(results_rows)

    # Summary statistics
    used_gates = [row[0] for row in results_rows if row[3]]
    unused_gates = [row[0] for row in results_rows if not row[3]]
    
    print()
    print("=" * 80)
    print("SINGLE-GATE KNOCKOUT TEST COMPLETE")
    print("=" * 80)
    print(f"Total gates tested: {len(eligible)}")
    print(f"Used gates (cause accuracy drop): {len(used_gates)}")
    print(f"Unused gates (no accuracy drop): {len(unused_gates)}")
    if used_gates:
        print(f"\nUsed gate indices: {used_gates[:20]}{'...' if len(used_gates) > 20 else ''}")
    if unused_gates:
        print(f"\nUnused gate indices: {unused_gates[:20]}{'...' if len(unused_gates) > 20 else ''}")
    print(f"\nResults written to: {output_csv}")
    print("=" * 80)


if __name__ == "__main__":
    main()
