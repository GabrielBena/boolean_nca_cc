"""
Hamming distance analysis utilities for circuit truth tables.

This module provides functions for computing Hamming distances between
circuit truth tables under knockout patterns.
"""

import jax
import jax.numpy as jp
from typing import List, Dict, Any
from boolean_nca_cc.circuits.train import create_gate_mask_from_knockout_pattern


def _hard_truth_tables_from_logits(logits_per_layer: List[jp.ndarray]) -> List[jp.ndarray]:
    """
    Convert per-layer logits to hard LUT truth tables {0,1}.

    Returns list of arrays with shape (num_gates, table_size) per layer beyond input.
    """
    hard_tables_per_layer: List[jp.ndarray] = []
    for lgt in logits_per_layer:
        # lgt shape: (group_n, group_size, 2^arity)
        probs = jax.nn.sigmoid(lgt)
        hard = jp.round(probs)  # {0,1}
        hard = hard.reshape(-1, hard.shape[-1])  # (num_gates, 2^arity)
        hard_tables_per_layer.append(hard)
    return hard_tables_per_layer


def _active_gate_mask_from_knockout(layer_sizes: List[List[int]], knockout_pattern) -> List[jp.ndarray]:
    """
    Build a per-layer boolean mask for active gates aligned to logits layers.
    Returns list matching logits_per_layer (i.e., excludes input layer).
    """
    full_gate_masks = create_gate_mask_from_knockout_pattern(knockout_pattern, layer_sizes)
    # Drop input layer mask; keep masks for layers with logits
    return [jp.array(mask, dtype=jp.bool_) for mask in full_gate_masks[1:]]


def _hamming_distance_tables(
    baseline_tables: List[jp.ndarray],
    perturbed_tables: List[jp.ndarray],
    perturbed_active_masks: List[jp.ndarray],
) -> Dict[str, Any]:
    """
    Compute distances between baseline and perturbed hard truth tables.

    Only counts gates where perturbed_active_masks is True (knocked-out gates excluded).

    Returns:
      - overall_bitwise_fraction_diff: fraction of differing bits across all counted entries
      - per_layer_bitwise_fraction_diff: list per layer
      - per_gate_mean_hamming: mean over gates of per-table Hamming normalized by table size
      - counted_bits_total, counted_gates_total
    """
    assert len(baseline_tables) == len(perturbed_tables) == len(perturbed_active_masks)

    per_layer_bitwise_fraction: List[float] = []
    total_diff = 0.0
    total_count = 0.0
    per_gate_scores: List[float] = []

    for layer_idx, (base_layer, pert_layer, active_mask) in enumerate(
        zip(baseline_tables, perturbed_tables, perturbed_active_masks)
    ):
        # base_layer, pert_layer: (num_gates, table_size)
        table_size = base_layer.shape[-1]
        # select active gates only
        active_idx = jp.where(active_mask > 0.5)[0]
        if active_idx.size == 0:
            per_layer_bitwise_fraction.append(0.0)
            continue
        base_sel = base_layer[active_idx]
        pert_sel = pert_layer[active_idx]
        diffs = jp.not_equal(base_sel, pert_sel).astype(jp.float32)
        # per-gate normalized hamming (mean across table bits)
        per_gate = jp.mean(diffs, axis=1)
        per_gate_scores.extend(list(jax.device_get(per_gate)))
        # layer bitwise fraction
        layer_diff = jp.sum(diffs)
        layer_count = diffs.size
        per_layer_bitwise_fraction.append(float(layer_diff / layer_count))
        total_diff += float(layer_diff)
        total_count += float(layer_count)

    overall = float(total_diff / total_count) if total_count > 0 else 0.0
    per_gate_mean = float(jp.mean(jp.array(per_gate_scores))) if per_gate_scores else 0.0

    return dict(
        overall_bitwise_fraction_diff=overall,
        per_layer_bitwise_fraction_diff=per_layer_bitwise_fraction,
        per_gate_mean_hamming=per_gate_mean,
        counted_bits_total=int(total_count),
        counted_gates_total=len(per_gate_scores),
    )
