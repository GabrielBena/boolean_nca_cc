"""
Single-gate damage sweep: measure immediate hard-accuracy drop from a preconfigured circuit.

- Fix wiring seed for reproducible circuit wiring
- Preconfigure logits via backprop (no structure change)
- For a range of damage seeds, damage exactly one gate using existing knockout utilities
- Evaluate hard accuracy before any recovery; record drop per seed

Outputs CSV with columns: seed, hard_accuracy_baseline, hard_accuracy_damaged, hard_accuracy_drop

Usage:
  python -m experiments.single_gate_damage_sweep \
    --config-name config \
    damage_seeds 0 999 \
    --output results/single_gate_damage.csv

Respects Hydra config at configs/config.yaml for task/circuit/loss parameters.
Only requires: circuit.task, circuit.input_bits, circuit.output_bits, circuit.arity, circuit.num_layers,
training.loss_type, preconfigure steps/lr via training.backprop or defaults.
"""

from __future__ import annotations

import os
import csv
from typing import Tuple, List

import jax
import jax.numpy as jp
from omegaconf import DictConfig, OmegaConf, open_dict
import hydra

from boolean_nca_cc.circuits.model import generate_layer_sizes, gen_circuit
from boolean_nca_cc.circuits.tasks import get_task_data
from boolean_nca_cc.training.evaluation import get_loss_from_wires_logits
from boolean_nca_cc.training.preconfigure import preconfigure_circuit_logits
from boolean_nca_cc.training.pool.perturbation import (
    create_reproducible_knockout_pattern,
)


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


def _apply_knockout_to_logits(logits: List[jp.ndarray], knockout_pattern: jp.ndarray, layer_sizes: List[Tuple[int, int]]) -> List[jp.ndarray]:
    """
    Convert a node-level knockout boolean mask (True=knockout) into per-layer gate masks and zero-out corresponding gate logits.
    This uses the layer_sizes (total_gates per layer) to slice the flat pattern.
    We do not mask inputs or outputs (pattern generator already skips input/output); here we simply zero the logits of knocked gates.
    """
    # Build flat indices per layer for GATE LAYERS ONLY (skip input layer)
    offsets = []
    idx = layer_sizes[0][0]  # skip input layer span
    for total_gates, _group in layer_sizes[1:]:
        offsets.append((idx, idx + total_gates))
        idx += total_gates

    # logits list aligns with gate layers (same length as layer_sizes[1:])
    masked_logits: List[jp.ndarray] = []
    for layer_idx, layer_logits in enumerate(logits):
        start, end = offsets[layer_idx]
        layer_mask_flat = knockout_pattern[start:end]  # length = total_gates in this layer

        # Derive grouping to match logits shape: (group_n, group_size, logit_dim)
        total_gates, group_size = layer_sizes[1 + layer_idx]
        group_n = total_gates // group_size

        # Reshape flat mask to (group_n, group_size) and broadcast over logit_dim
        layer_mask_grid = layer_mask_flat.reshape(group_n, group_size)
        gate_mask = jp.where(layer_mask_grid[..., None], 0.0, 1.0)

        # Zero-out knocked gates' logits
        masked = layer_logits * gate_mask
        masked_logits.append(masked)

    return masked_logits


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    # Resolve layer sizes
    layer_sizes = _ensure_layer_sizes(cfg)

    # Data for the task
    input_n = cfg.circuit.input_bits
    output_n = cfg.circuit.output_bits
    case_n = 1 << input_n
    x_data, y_data = get_task_data(cfg.circuit.task, case_n, input_bits=input_n, output_bits=output_n)

    # Fix wiring seed; use cfg.test_seed for wiring determinism
    wiring_key = jax.random.PRNGKey(cfg.test_seed)
    base_wires, base_logits = gen_circuit(wiring_key, layer_sizes, arity=cfg.circuit.arity)

    # Preconfigure logits (repair-style) using provided or default optimizer params
    bp_cfg = cfg.backprop if hasattr(cfg, "backprop") else {}
    pre_steps = cfg.training.get("preconfig_steps", 200) if hasattr(cfg, "training") else 200
    pre_lr = cfg.training.get("preconfig_lr", 1) if hasattr(cfg, "training") else 1

    pre_wires, pre_logits = preconfigure_circuit_logits(
        wiring_key=wiring_key,
        layer_sizes=layer_sizes,
        arity=cfg.circuit.arity,
        x_data=x_data,
        y_data=y_data,
        loss_type=cfg.training.loss_type,
        steps=int(pre_steps),
        lr=float(pre_lr),
        optimizer=bp_cfg.get("optimizer", "adam"),
        weight_decay=float(bp_cfg.get("weight_decay", 1e-1)),
        beta1=float(bp_cfg.get("beta1", 0.8)),
        beta2=float(bp_cfg.get("beta2", 0.8)),
    )

    # Baseline hard accuracy
    baseline_hard_acc = _compute_hard_accuracy(pre_logits, pre_wires, x_data, y_data, cfg.training.loss_type)

    # Damage sweep parameters
    # Read optional CLI overrides from Hydra: `damage_seeds` start end, or use knockouts from cfg
    # Here we keep it simple: range from 0 to N-1, where N = cfg.pool.damage_knockout_diversity if present, else 1000
    default_N = 100
    N = int(getattr(cfg.pool, "damage_knockout_diversity", default_N))

    # Output path
    output_csv = hydra.utils.to_absolute_path(os.environ.get("SINGLE_GATE_SWEEP_OUTPUT", "results/single_gate_damage.csv"))
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    # Test multiple damage levels (1-5 gates)
    damage_levels = [1, 2, 3, 4, 5]

    # Run sweep, varying both knockout seed and damage level
    results = []

    for seed in range(N):
        for damage_prob in damage_levels:
            key = jax.random.PRNGKey(seed)
            knockout_pattern = create_reproducible_knockout_pattern(
                key=key,
                layer_sizes=layer_sizes,
                damage_prob=damage_prob,
            )

            damaged_logits = _apply_knockout_to_logits(pre_logits, knockout_pattern, layer_sizes)
            damaged_hard_acc = _compute_hard_accuracy(damaged_logits, pre_wires, x_data, y_data, cfg.training.loss_type)
            drop = baseline_hard_acc - damaged_hard_acc
            results.append((seed, damage_prob, baseline_hard_acc, damaged_hard_acc, drop))

    # Sort by damage level (ascending), then by performance drop (descending within each level)
    results.sort(key=lambda x: (x[1], -x[4]))

    # Write CSV with header
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(("seed", "damage_gates", "hard_accuracy_baseline", "hard_accuracy_damaged", "hard_accuracy_drop"))
        writer.writerows(results)

    print(f"Wrote results: {output_csv}")


if __name__ == "__main__":
    main()


