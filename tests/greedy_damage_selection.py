"""
Greedy multi-gate damage selection: iteratively select gate indices whose removal
maximizes immediate hard-accuracy drop from a preconfigured circuit.

- Fix wiring seed for reproducible circuit wiring
- Preconfigure logits via backprop (no structure change)
- Build eligible flat indices (skip input and output layers)
- Greedily select indices: at each step, add the index that yields the largest
  hard-accuracy drop when combined with already selected indices

Outputs CSV with columns:
  step,num_gates,selected_index,hard_accuracy_damaged,hard_accuracy_drop,cumulative_indices

Usage:
  python -m experiments.greedy_damage_selection \
    --config-name config \
    greedy.max_gates=5 \
    --output results/greedy_damage_indices.csv

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
    hard_loss, pred, pred_hard, accuracy, hard_accuracy, res, hard_res = aux
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
    x_data, y_data = get_task_data(cfg.circuit.task, case_n, input_bits=input_n, output_bits=output_n)

    # Fix wiring seed for reproducible wiring
    wiring_seed = cfg.test_seed
    wiring_key = jax.random.PRNGKey(wiring_seed)

    # Preconfigure logits on fixed wiring
    # Match wiring_seed_preconfigure.py: use backprop.* parameters directly
    bp_cfg = cfg.backprop if hasattr(cfg, "backprop") else {}
    training_cfg = cfg.training if hasattr(cfg, "training") else {}
    
    # Use backprop.* parameters (matching wiring_seed_preconfigure.py)
    pre_steps = bp_cfg.get("epochs", 200)
    pre_lr = bp_cfg.get("learning_rate", 1)
    pre_optimizer = bp_cfg.get("optimizer", "adam")
    pre_weight_decay = float(bp_cfg.get("weight_decay", 1e-1))
    pre_beta1 = float(bp_cfg.get("beta1", 0.8))
    pre_beta2 = float(bp_cfg.get("beta2", 0.8))
    
    # Check if training.preconfig_* exists (for comparison)
    has_preconfig_steps = "preconfig_steps" in training_cfg
    has_preconfig_lr = "preconfig_lr" in training_cfg
    alt_steps = training_cfg.get("preconfig_steps", pre_steps) if has_preconfig_steps else pre_steps
    alt_lr = training_cfg.get("preconfig_lr", pre_lr) if has_preconfig_lr else pre_lr
    params_differ = (has_preconfig_steps and alt_steps != pre_steps) or (has_preconfig_lr and alt_lr != pre_lr)

    # Print diagnostic information header
    print("=" * 80)
    print("GREEDY DAMAGE SELECTION - PRECONFIGURATION DIAGNOSTICS")
    print("=" * 80)
    print(f"\nTask Configuration:")
    print(f"  - Task: {cfg.circuit.task}")
    print(f"  - Input bits: {input_n}")
    print(f"  - Output bits: {output_n}")
    print(f"  - Case count: {case_n} (2^{input_n})")
    print(f"  - Loss type: {cfg.training.loss_type}")
    print(f"  - Arity: {cfg.circuit.arity}")
    print(f"  - Layer sizes: {layer_sizes}")
    print(f"  - Data shapes: x_data={x_data.shape}, y_data={y_data.shape}")
    
    print(f"\nWiring Configuration:")
    print(f"  - Wiring seed: {wiring_seed}")
    print(f"  - Wiring key: {wiring_key}")
    
    print(f"\nPreconfiguration Parameters:")
    print(f"  - Steps: {pre_steps} (from backprop.epochs)")
    print(f"  - Learning rate: {pre_lr} (from backprop.learning_rate)")
    print(f"  - Optimizer: {pre_optimizer} (from backprop.optimizer)")
    print(f"  - Weight decay: {pre_weight_decay} (from backprop.weight_decay)")
    print(f"  - Beta1: {pre_beta1} (from backprop.beta1)")
    print(f"  - Beta2: {pre_beta2} (from backprop.beta2)")
    if params_differ:
        print(f"\n  ⚠ WARNING: training.preconfig_* parameters differ from backprop.*:")
        if has_preconfig_steps:
            print(f"    - training.preconfig_steps = {alt_steps} (not used, using backprop.epochs = {pre_steps})")
        if has_preconfig_lr:
            print(f"    - training.preconfig_lr = {alt_lr} (not used, using backprop.learning_rate = {pre_lr})")
        print(f"    This script matches wiring_seed_preconfigure.py (uses backprop.* directly)")
    else:
        print(f"\n  ✓ Parameters match wiring_seed_preconfigure.py (uses backprop.* directly)")
    
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

    # Compute all metrics for diagnostic output
    baseline_metrics = _compute_all_metrics(pre_logits, pre_wires, x_data, y_data, cfg.training.loss_type)
    baseline_hard_acc = baseline_metrics["hard_accuracy"]
    
    # Eligible indices (skip input and output layers) - compute before printing
    eligible = list(map(int, list(_eligible_flat_indices(layer_sizes))))

    # Total nodes
    total_nodes = sum(total_gates for total_gates, _ in layer_sizes)

    # Greedy settings
    max_gates_default = 160
    # Allow override via cfg.greedy.max_gates if present
    max_gates = int(getattr(getattr(cfg, "greedy", {}), "max_gates", max_gates_default))
    max_gates = max(0, min(max_gates, len(eligible)))
    
    print(f"\nPreconfigured Circuit Metrics:")
    print(f"  - Loss: {baseline_metrics['loss']:.6f}")
    print(f"  - Hard loss: {baseline_metrics['hard_loss']:.6f}")
    print(f"  - Accuracy: {baseline_metrics['accuracy']:.6f}")
    print(f"  - Hard accuracy: {baseline_metrics['hard_accuracy']:.6f}")
    if baseline_metrics['hard_accuracy'] == 1.0:
        print(f"  ✓ PERFECT PRECONFIGURATION (hard_accuracy == 1.0)")
    else:
        print(f"  ⚠ NOT PERFECT (hard_accuracy < 1.0)")
        print(f"    This may explain discrepancy with training loop results.")
    
    print(f"\nGreedy Selection Configuration:")
    print(f"  - Eligible gate indices: {len(eligible)} (skipping input/output layers)")
    print(f"  - Total nodes: {total_nodes}")
    print(f"  - Max gates to select: {max_gates}")
    print("=" * 80)
    print()

    # Output path
    output_csv = hydra.utils.to_absolute_path(os.environ.get("GREEDY_DAMAGE_OUTPUT", "results/greedy_damage_indices.csv"))
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    selected: List[int] = []
    results_rows = []

    # Pre-allocate a zero mask for speed in Python
    for step in tqdm(range(1, max_gates + 1), desc="Greedy selection", unit="gate"):
        best_idx = None
        best_drop = -1.0
        best_hard_acc = baseline_hard_acc

        for candidate in tqdm(eligible, desc=f"Step {step}: evaluating candidates", leave=False, disable=len(eligible) < 10):
            if candidate in selected:
                continue

            mask = jp.zeros(total_nodes, dtype=jp.bool_)
            if selected:
                mask = mask.at[jp.array(selected)].set(True)
            mask = mask.at[candidate].set(True)

            damaged_logits = _apply_knockout_to_logits(pre_logits, mask, layer_sizes)
            damaged_hard_acc = _compute_hard_accuracy(damaged_logits, pre_wires, x_data, y_data, cfg.training.loss_type)
            drop = baseline_hard_acc - damaged_hard_acc

            if drop > best_drop:
                best_drop = drop
                best_hard_acc = damaged_hard_acc
                best_idx = candidate

        if best_idx is None:
            break

        selected.append(best_idx)
        cumulative_indices = ",".join(map(str, selected))
        results_rows.append((step, len(selected), best_idx, best_hard_acc, best_drop, cumulative_indices))

    # Write CSV
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(("step", "num_gates", "selected_index", "hard_accuracy_damaged", "hard_accuracy_drop", "cumulative_indices"))
        writer.writerows(results_rows)

    print()
    print("=" * 80)
    print("GREEDY SELECTION COMPLETE")
    print("=" * 80)
    print(f"Selected {len(selected)} gate indices")
    print(f"Baseline hard accuracy: {baseline_hard_acc:.6f}")
    if results_rows:
        final_row = results_rows[-1]
        final_hard_acc = final_row[3]
        total_drop = baseline_hard_acc - final_hard_acc
        print(f"Final hard accuracy (after {len(selected)} gates): {final_hard_acc:.6f}")
        print(f"Total accuracy drop: {total_drop:.6f}")
    print(f"Results written to: {output_csv}")
    print("=" * 80)


if __name__ == "__main__":
    main()


