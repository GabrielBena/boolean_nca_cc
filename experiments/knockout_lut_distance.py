"""
Experiment: Backprop with fixed wiring under knockout patterns, and LUT truth-table distances.

This script:
- Loads the training config (without Hydra side-effects)
- Builds a fixed wiring using cfg.test_seed
- Runs baseline backprop training (no knockout)
- Runs backprop training per knockout pattern (same wiring)
- Computes Hamming distances between hard LUT truth tables of baseline vs perturbed
  excluding any gates that are knocked out in the perturbed configuration
- Writes a CSV summary and prints a concise report
"""

import os
import argparse
import json
from typing import List, Dict, Any

import jax
import jax.numpy as jp
from omegaconf import OmegaConf

from boolean_nca_cc.circuits.model import gen_circuit
from boolean_nca_cc.circuits.tasks import get_task_data
from boolean_nca_cc.circuits.train import (
    TrainState,
    loss_f_l4,
    loss_f_bce,
    train_step,
    create_gate_mask_from_knockout_pattern,
)
from boolean_nca_cc.circuits.model import generate_layer_sizes
from boolean_nca_cc.training.pool.structural_perturbation import (
    create_knockout_vocabulary,
)


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


def _run_bp_single(cfg, x_data, y_data, loss_type: str):
    key = jax.random.PRNGKey(cfg.test_seed)
    wires, logits = gen_circuit(key, cfg.circuit.layer_sizes, arity=cfg.circuit.arity)

    if cfg.backprop.optimizer == "adamw":
        import optax

        opt = optax.adamw(
            cfg.backprop.learning_rate,
            b1=cfg.backprop.beta1,
            b2=cfg.backprop.beta2,
            weight_decay=cfg.backprop.weight_decay,
        )
    else:
        import optax

        opt = optax.adam(cfg.backprop.learning_rate)

    state = TrainState(params=logits, opt_state=opt.init(logits))

    _train_step_fn = lambda st: train_step(
        state=st,
        opt=opt,
        wires=wires,
        x=x_data,
        y0=y_data,
        loss_type=loss_type,
        do_train=True,
    )

    for _ in range(cfg.backprop.epochs):
        _, _, state = _train_step_fn(state)

    loss_fn = loss_f_l4 if loss_type == "l4" else loss_f_bce
    final_loss, final_aux = loss_fn(state.params, wires, x_data, y_data)

    return dict(params=state.params, wires=wires, final_loss=float(final_loss))


def _run_bp_with_knockouts(cfg, x_data, y_data, loss_type: str, knockout_patterns, layer_sizes, baseline_params, baseline_wires):
    # Use baseline optimized circuit instead of generating fresh
    wires = baseline_wires
    logits = baseline_params  # Start from optimized baseline, not NOPs

    if cfg.backprop.optimizer == "adamw":
        import optax

        opt = optax.adamw(
            cfg.backprop.learning_rate,
            b1=cfg.backprop.beta1,
            b2=cfg.backprop.beta2,
            weight_decay=cfg.backprop.weight_decay,
        )
    else:
        import optax

        opt = optax.adam(cfg.backprop.learning_rate)

    results = []
    from tqdm.auto import tqdm
    for pattern in tqdm(knockout_patterns, desc="Training knockout patterns"):
        state = TrainState(params=logits, opt_state=opt.init(logits))

        def _step(st):
            return train_step(
                state=st,
                opt=opt,
                wires=wires,
                x=x_data,
                y0=y_data,
                loss_type=loss_type,
                do_train=True,
                knockout_pattern=pattern,
                layer_sizes=cfg.circuit.layer_sizes,
            )

        for _ in range(cfg.backprop.epochs):
            _, _, state = _step(state)

        # Final evaluation with knockout pattern applied
        loss_fn = loss_f_l4 if loss_type == "l4" else loss_f_bce
        final_loss, final_aux = loss_fn(
            state.params, wires, x_data, y_data,
            gate_mask=create_gate_mask_from_knockout_pattern(pattern, layer_sizes)
        )
        final_accuracy = float(final_aux["hard_accuracy"])
        final_hard_loss = float(final_aux["hard_loss"])

        results.append(dict(
            params=state.params, 
            pattern=pattern,
            final_accuracy=final_accuracy,
            final_hard_loss=final_hard_loss
        ))

    return dict(wires=wires, per_pattern=results)


def _pairwise_hamming_between_knockouts(
    ko_tables_list: List[List[jp.ndarray]],
    ko_active_masks_list: List[List[jp.ndarray]],
):
    """
    Compute an all-to-all matrix of bitwise fraction differences between KO runs.

    Uses intersection of active gates in both patterns for each pair and layer.
    Returns a 2D Python list of floats (NaN when no comparable gates).
    """
    n = len(ko_tables_list)
    import math

    matrix = [[math.nan for _ in range(n)] for _ in range(n)]

    for i in range(n):
        for j in range(n):
            total_diff = 0.0
            total_count = 0.0
            for layer_idx, (ti_layer, tj_layer, mi_layer, mj_layer) in enumerate(
                zip(ko_tables_list[i], ko_tables_list[j], ko_active_masks_list[i], ko_active_masks_list[j])
            ):
                # intersection of active gates
                inter_mask = jp.logical_and(mi_layer, mj_layer)
                inter_idx = jp.where(inter_mask)[0]
                if inter_idx.size == 0:
                    continue
                A = ti_layer[inter_idx]
                B = tj_layer[inter_idx]
                diffs = jp.not_equal(A, B).astype(jp.float32)
                total_diff += float(jp.sum(diffs))
                total_count += float(diffs.size)
            matrix[i][j] = (total_diff / total_count) if total_count > 0 else float('nan')

    return matrix


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)

    # Set output directory with f-string if not provided
    if args.output is None:
        vocab_size = cfg.backprop.knockout_vocabulary.size
        damage_prob = cfg.backprop.knockout_vocabulary.damage_prob
        args.output = f"results/knockout_lut_distance_vocab{vocab_size}_damage{damage_prob}"

    # Ensure layer_sizes are present
    input_n, output_n = cfg.circuit.input_bits, cfg.circuit.output_bits
    arity = cfg.circuit.arity
    if cfg.circuit.layer_sizes is None:
        layer_sizes = generate_layer_sizes(input_n, output_n, arity, layer_n=cfg.circuit.num_layers)
        cfg.circuit.layer_sizes = layer_sizes
    else:
        layer_sizes = cfg.circuit.layer_sizes

    # Data
    case_n = 1 << input_n
    x, y0 = get_task_data(cfg.circuit.task, case_n, input_bits=input_n, output_bits=output_n)

    # Baseline BP (no knockouts)
    bp_baseline = _run_bp_single(cfg, x, y0, loss_type=cfg.training.loss_type)

    # Knockout vocabulary
    rng = jax.random.PRNGKey(cfg.seed)
    rng, vocab_key = jax.random.split(rng)
    vocab = create_knockout_vocabulary(
        rng=vocab_key,
        vocabulary_size=cfg.backprop.knockout_vocabulary.size,
        layer_sizes=layer_sizes,
        damage_prob=cfg.backprop.knockout_vocabulary.damage_prob,
    )

    # Train per knockout
    bp_knockouts = _run_bp_with_knockouts(cfg, x, y0, loss_type=cfg.training.loss_type, knockout_patterns=vocab, layer_sizes=layer_sizes, baseline_params=bp_baseline["params"], baseline_wires=bp_baseline["wires"])

    # Truth tables
    baseline_tables = _hard_truth_tables_from_logits(bp_baseline["params"])  # per-layer

    # Compute distances per pattern (baseline vs each KO)
    os.makedirs(args.output, exist_ok=True)
    summary_rows = []
    ko_tables_list: List[List[jp.ndarray]] = []
    ko_active_masks_list: List[List[jp.ndarray]] = []

    for idx, item in enumerate(bp_knockouts["per_pattern"]):
        pattern = item["pattern"]
        pert_tables = _hard_truth_tables_from_logits(item["params"])  # per-layer
        active_masks = _active_gate_mask_from_knockout(layer_sizes, pattern)
        metrics = _hamming_distance_tables(baseline_tables, pert_tables, active_masks)

        row = {
            "pattern_idx": idx,
            "overall_bitwise_fraction_diff": metrics["overall_bitwise_fraction_diff"],
            "per_gate_mean_hamming": metrics["per_gate_mean_hamming"],
            "counted_bits_total": metrics["counted_bits_total"],
            "counted_gates_total": metrics["counted_gates_total"],
            "final_accuracy": item["final_accuracy"],
            "final_hard_loss": item["final_hard_loss"],
        }
        # add per-layer metrics as JSON for compactness
        row["per_layer_bitwise_fraction_diff"] = json.dumps(metrics["per_layer_bitwise_fraction_diff"])
        summary_rows.append(row)
        ko_tables_list.append(pert_tables)
        ko_active_masks_list.append(active_masks)

    # Save CSV
    import pandas as pd

    df = pd.DataFrame(summary_rows)
    csv_path = os.path.join(args.output, "summary.csv")
    df.to_csv(csv_path, index=False)

    # Pairwise all-to-all matrix among KO patterns using intersection masks
    pairwise = _pairwise_hamming_between_knockouts(ko_tables_list, ko_active_masks_list)
    import pandas as pd  # safe to re-import; ensures availability
    df_pair = pd.DataFrame(pairwise)
    df_pair.index.name = "pattern_i"
    df_pair.columns = [f"pattern_{j}" for j in range(len(pairwise))]
    pairwise_csv = os.path.join(args.output, "pairwise_matrix.csv")
    df_pair.to_csv(pairwise_csv)

    # Save a small report
    report = {
        "num_patterns": len(summary_rows),
        "mean_overall_bitwise_fraction_diff": float(df["overall_bitwise_fraction_diff"].mean()) if len(df) else 0.0,
        "mean_per_gate_mean_hamming": float(df["per_gate_mean_hamming"].mean()) if len(df) else 0.0,
        "mean_final_accuracy": float(df["final_accuracy"].mean()) if len(df) else 0.0,
        "mean_final_hard_loss": float(df["final_hard_loss"].mean()) if len(df) else 0.0,
        "config_path": os.path.abspath(args.config),
        "output_path": os.path.abspath(args.output),
        "pairwise_matrix_csv": os.path.abspath(pairwise_csv),
    }
    with open(os.path.join(args.output, "report.json"), "w") as f:
        json.dump(report, f, indent=2)

    # Generate visualizations
    try:
        from visualization.circular_plot import plot_circular_knockout_distances, plot_accuracy_vs_distance
        
        # Circular distance plot
        circular_plot_path = os.path.join(args.output, "circular_distances.png")
        plot_circular_knockout_distances(df, circular_plot_path)
        print(f"Circular plot saved to: {circular_plot_path}")
        
        # Accuracy vs distance scatter plot
        if 'final_accuracy' in df.columns:
            accuracy_plot_path = os.path.join(args.output, "accuracy_vs_distance.png")
            plot_accuracy_vs_distance(df, accuracy_plot_path)
            print(f"Accuracy plot saved to: {accuracy_plot_path}")
            
            # Add plot paths to report
            report["circular_plot_path"] = os.path.abspath(circular_plot_path)
            report["accuracy_plot_path"] = os.path.abspath(accuracy_plot_path)
            
    except ImportError as e:
        print(f"Warning: Could not import visualization module: {e}")
        print("Plots will not be generated.")
    except Exception as e:
        print(f"Warning: Error generating plots: {e}")
        print("Continuing without visualization...")

    # Print concise report
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()


