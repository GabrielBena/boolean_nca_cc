"""
Experiment: GNN vs Backprop comparison under knockout patterns, and LUT truth-table distances.

This script implements a streamlined, run ID-driven analysis:
- GNN run ID is the single source of truth for ALL configuration
- Automatically loads GNN model, training config, and parameters from WandB
- Uses GNN config for circuit structure, BP parameters, data generation, seeds, etc.
- No manual config overrides - everything derived from the training run
- Builds baseline circuit using GNN config parameters
- Runs baseline backprop training (no knockout)
- Runs backprop training per knockout pattern (same wiring)
- Runs GNN evaluation per knockout pattern (same wiring)
- Computes Hamming distances between hard LUT truth tables of baseline vs perturbed
  excluding any gates that are knocked out in the perturbed configuration
- Compares GNN vs BP recovery performance
- Writes a CSV summary and generates unified comparison plots

CLI: Only requires --run-id (defaults to "oxlper1c") and --methods selection.
All other parameters (vocab size, damage prob, loss type, etc.) come from GNN config.
"""

import os
import argparse
import json
import logging
from typing import List, Dict, Any

import jax
import jax.numpy as jp
from omegaconf import OmegaConf

# Setup logging
log = logging.getLogger(__name__)

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
from boolean_nca_cc.training.pool.perturbation import (
    create_knockout_vocabulary,
)
from boolean_nca_cc.training.checkpointing import (
    load_best_model_from_wandb,
    load_checkpoint,
    instantiate_model_from_config,
)
from boolean_nca_cc.training.evaluation import (
    get_loss_from_wires_logits,
    get_loss_and_update_graph,
)
from boolean_nca_cc.utils.graph_builder import build_graph
from boolean_nca_cc.utils.extraction import extract_logits_from_graph
from boolean_nca_cc.training.preconfigure import preconfigure_circuit_logits


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
        final_hard_accuracy = float(final_aux["hard_accuracy"])
        final_hard_loss = float(final_aux["hard_loss"])

        results.append(dict(
            params=state.params, 
            pattern=pattern,
            final_hard_accuracy=final_hard_accuracy,
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


def _evaluate_gnn_single_pattern(
    model,
    wires,
    baseline_logits,
    x_data,
    y_data,
    input_n: int,
    arity: int,
    circuit_hidden_dim: int,
    n_message_steps: int,
    loss_type: str,
    layer_sizes,
    knockout_pattern,
):
    """
    Run GNN evaluation for a single knockout pattern starting from the BP-optimized baseline logits.
    Returns final logits and metrics.
    """
    # Initial loss and residuals for graph seeding
    initial_loss, (
        initial_hard_loss,
        initial_pred,
        initial_pred_hard,
        initial_accuracy,
        initial_hard_accuracy,
        initial_res,
        initial_hard_res,
    ) = get_loss_from_wires_logits(baseline_logits, wires, x_data, y_data, loss_type)

    # Build initial graph and set loss features
    graph = build_graph(
        logits=baseline_logits,
        wires=wires,
        input_n=input_n,
        arity=arity,
        circuit_hidden_dim=circuit_hidden_dim,
        loss_value=initial_loss,
        bidirectional_edges=True,
    )

    # Update per-output loss features
    from boolean_nca_cc.utils import update_output_node_loss

    graph = update_output_node_loss(graph, layer_sizes, initial_res.mean(axis=0))

    logits_original_shapes = [l.shape for l in baseline_logits]

    current_graph = graph
    current_logits = baseline_logits
    current_aux = (
        initial_hard_loss,
        initial_pred,
        initial_pred_hard,
        initial_accuracy,
        initial_hard_accuracy,
        initial_res,
        initial_hard_res,
    )

    for _ in range(max(0, int(n_message_steps))):
        # Apply one step with knockout pattern
        updated_graph = model(current_graph, knockout_pattern=knockout_pattern)
        # Recompute loss and update loss features
        updated_graph, loss, current_logits, current_aux = get_loss_and_update_graph(
            updated_graph,
            logits_original_shapes,
            wires,
            x_data,
            y_data,
            loss_type,
            layer_sizes,
        )
        current_graph = updated_graph

    # Metrics from final aux
    (
        final_hard_loss,
        _pred,
        _pred_hard,
        _acc,
        final_hard_accuracy,
        _res,
        _hard_res,
    ) = current_aux

    return dict(
        logits=current_logits,
                    final_hard_accuracy=float(final_hard_accuracy),
        final_hard_loss=float(final_hard_loss),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default=None)
    # Core analysis control - GNN run ID determines everything else
    parser.add_argument("--run-id", type=str, default="oxlper1c", help="WandB run id for GNN load")
    parser.add_argument("--checkpoint", type=str, default=None, help="Local checkpoint path .pkl for GNN")
    # Analysis method selection only
    parser.add_argument("--methods", type=str, default="both", help="gnn,bp,both")
    # GNN evaluation parameters
    parser.add_argument("--n-message-steps", type=int, default=100)
    args = parser.parse_args()

    # Load GNN first to get the authoritative config
    if args.checkpoint is not None:
        loaded = load_checkpoint(args.checkpoint)
        gnn_cfg = OmegaConf.create(loaded.get("config", {}))
        gnn_model = instantiate_model_from_config(gnn_cfg, seed=gnn_cfg.get("seed", 0))
        from flax import nnx as _nnx
        _nnx.update(gnn_model, loaded["model"])
        gnn_hidden_dim = int(gnn_cfg.model.get("circuit_hidden_dim", 16))
        gnn_training_config = gnn_cfg
    else:
        # Use standard artifact filename
        filename_to_load = "best_model_eval_ko_hard_accuracy"

        gnn_model, _loaded_dict, gnn_cfg = load_best_model_from_wandb(
            run_id=args.run_id,
            seed=0,  # Use default seed, will be overridden by GNN config
            filename=filename_to_load,
        )
        gnn_hidden_dim = int(gnn_cfg.model.get("circuit_hidden_dim", 16))
        gnn_training_config = gnn_cfg

    # Use GNN config as the authoritative source for everything
    # This ensures BP baseline and analysis use identical parameters to GNN training
    cfg = gnn_cfg

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

    # Methods selection
    methods_str = (args.methods or "gnn").lower()
    if methods_str == "both":
        methods = {"bp", "gnn"}
    else:
        methods = set(m.strip() for m in methods_str.split(","))

    # GNN model is already loaded above, now check if methods include GNN
    if "gnn" not in methods:
        gnn_model = None
        gnn_hidden_dim = None
        gnn_training_config = None

    # Loss type from GNN config
    loss_type = cfg.training.loss_type

    # Baseline circuit creation - use preconfigured if GNN was trained in repair mode
    if "gnn" in methods and gnn_training_config is not None:
        # Check if GNN was trained in repair mode and use the same preconfigured circuit
        training_mode = gnn_training_config.get("training_mode", "growth")
        if training_mode == "repair":
            log.info("GNN was trained in repair mode - recreating exact preconfigured baseline circuit")
            
            # Extract preconfiguration parameters from training config
            preconfig_steps = gnn_training_config.get("preconfig_steps", 200)
            preconfig_lr = gnn_training_config.get("preconfig_lr", 1e-2)
            wiring_fixed_key = gnn_training_config.get("wiring_fixed_key", cfg.get("test_seed", 42))
            
            # Convert wiring_fixed_key to JAX key if it's an integer
            if isinstance(wiring_fixed_key, int):
                wiring_fixed_key = jax.random.PRNGKey(wiring_fixed_key)
            
            # Recreate the exact same preconfigured circuit used during training
            base_wires, base_logits = preconfigure_circuit_logits(
                wiring_key=wiring_fixed_key,
                layer_sizes=layer_sizes,
                arity=arity,
                x_data=x,
                y_data=y0,
                loss_type=loss_type,
                steps=preconfig_steps,
                lr=preconfig_lr,
            )
            
            # Use preconfigured circuit for BP baseline (no additional optimization needed)
            bp_baseline = {
                "wires": base_wires,
                "params": base_logits,
                "final_loss": 0.0  # Preconfigured circuits are already optimized
            }
            
            log.info(f"Recreated preconfigured circuit with {preconfig_steps} steps, lr={preconfig_lr}")
        else:
            # GNN was trained in growth mode - use standard BP baseline
            log.info("GNN was trained in growth mode - using standard BP baseline")
            bp_baseline = _run_bp_single(cfg, x, y0, loss_type=loss_type)
    else:
        # No GNN or no training config - use standard BP baseline
        bp_baseline = _run_bp_single(cfg, x, y0, loss_type=loss_type)

    # Damage modes from GNN config
    damage_modes = [cfg.backprop.knockout_vocabulary.get("mode", "shotgun") if hasattr(cfg.backprop, "knockout_vocabulary") else "shotgun"]



    # Truth tables
    baseline_tables = _hard_truth_tables_from_logits(bp_baseline["params"])  # per-layer

    # Storage
    os.makedirs(args.output, exist_ok=True)
    summary_rows = []
    last_method_ko_tables: List[List[jp.ndarray]] = []
    last_method_active_masks: List[List[jp.ndarray]] = []

    # Loop over damage modes
    rng = jax.random.PRNGKey(cfg.seed)
    for damage_mode in damage_modes:
        rng, vocab_key = jax.random.split(rng)
        vocab = create_knockout_vocabulary(
            rng=vocab_key,
            vocabulary_size=cfg.backprop.knockout_vocabulary.size,
            layer_sizes=layer_sizes,
            damage_prob=cfg.backprop.knockout_vocabulary.damage_prob,
            damage_mode=damage_mode,
        )

        # BP method
        if "bp" in methods:
            bp_knockouts = _run_bp_with_knockouts(
                cfg,
                x,
                y0,
                loss_type=loss_type,
                knockout_patterns=vocab,
                layer_sizes=layer_sizes,
                baseline_params=bp_baseline["params"],
                baseline_wires=bp_baseline["wires"],
            )

            method_ko_tables: List[List[jp.ndarray]] = []
            method_active_masks: List[List[jp.ndarray]] = []
            for idx, item in enumerate(bp_knockouts["per_pattern"]):
                pattern = item["pattern"]
                pert_tables = _hard_truth_tables_from_logits(item["params"])  # per-layer
                active_masks = _active_gate_mask_from_knockout(layer_sizes, pattern)
                metrics = _hamming_distance_tables(baseline_tables, pert_tables, active_masks)

                row = {
                    "pattern_idx": idx,
                    "damage_mode": damage_mode,
                    "method": "bp",
                    "overall_bitwise_fraction_diff": metrics["overall_bitwise_fraction_diff"],
                    "per_gate_mean_hamming": metrics["per_gate_mean_hamming"],
                    "counted_bits_total": metrics["counted_bits_total"],
                    "counted_gates_total": metrics["counted_gates_total"],
                    "final_hard_accuracy": item["final_hard_accuracy"],
                    "final_hard_loss": item["final_hard_loss"],
                }
                row["per_layer_bitwise_fraction_diff"] = json.dumps(metrics["per_layer_bitwise_fraction_diff"])
                summary_rows.append(row)
                method_ko_tables.append(pert_tables)
                method_active_masks.append(active_masks)

            last_method_ko_tables = method_ko_tables
            last_method_active_masks = method_active_masks

        # GNN method
        if "gnn" in methods and gnn_model is not None:
            method_ko_tables: List[List[jp.ndarray]] = []
            method_active_masks: List[List[jp.ndarray]] = []
            for idx, pattern in enumerate(vocab):
                eval_res = _evaluate_gnn_single_pattern(
                    model=gnn_model,
                    wires=bp_baseline["wires"],
                    baseline_logits=bp_baseline["params"],
                    x_data=x,
                    y_data=y0,
                    input_n=input_n,
                    arity=arity,
                    circuit_hidden_dim=gnn_hidden_dim or 16,
                    n_message_steps=args.n_message_steps,
                    loss_type=loss_type,
                    layer_sizes=layer_sizes,
                    knockout_pattern=pattern,
                )

                pert_tables = _hard_truth_tables_from_logits(eval_res["logits"])  # per-layer
                active_masks = _active_gate_mask_from_knockout(layer_sizes, pattern)
                metrics = _hamming_distance_tables(baseline_tables, pert_tables, active_masks)

                row = {
                    "pattern_idx": idx,
                    "damage_mode": damage_mode,
                    "method": "gnn",
                    "overall_bitwise_fraction_diff": metrics["overall_bitwise_fraction_diff"],
                    "per_gate_mean_hamming": metrics["per_gate_mean_hamming"],
                    "counted_bits_total": metrics["counted_bits_total"],
                    "counted_gates_total": metrics["counted_gates_total"],
                    "final_hard_accuracy": eval_res["final_hard_accuracy"],
                    "final_hard_loss": eval_res["final_hard_loss"],
                }
                row["per_layer_bitwise_fraction_diff"] = json.dumps(metrics["per_layer_bitwise_fraction_diff"])
                summary_rows.append(row)
                method_ko_tables.append(pert_tables)
                method_active_masks.append(active_masks)

            last_method_ko_tables = method_ko_tables
            last_method_active_masks = method_active_masks

    # Save CSV
    import pandas as pd

    df = pd.DataFrame(summary_rows)
    csv_path = os.path.join(args.output, "summary.csv")
    df.to_csv(csv_path, index=False)

    # Pairwise all-to-all matrix among KO patterns using intersection masks
    pairwise = []
    if last_method_ko_tables:
        pairwise = _pairwise_hamming_between_knockouts(last_method_ko_tables, last_method_active_masks)
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
        "mean_final_hard_accuracy": float(df["final_hard_accuracy"].mean()) if len(df) else 0.0,
        "mean_final_hard_loss": float(df["final_hard_loss"].mean()) if len(df) else 0.0,
        # "config_path": os.path.abspath(args.config),
        "output_path": os.path.abspath(args.output),
        "pairwise_matrix_csv": os.path.abspath(pairwise_csv),
    }
    with open(os.path.join(args.output, "report.json"), "w") as f:
        json.dump(report, f, indent=2)

    # Generate visualizations
    try:
        from visualization.circular_plot import plot_accuracy_vs_distance
        
        # Accuracy vs distance scatter plot
        if 'final_hard_accuracy' in df.columns:
            accuracy_plot_path = os.path.join(args.output, "accuracy_vs_distance.png")
            plot_accuracy_vs_distance(df, accuracy_plot_path, color_by_method=True)
            print(f"Accuracy plot saved to: {accuracy_plot_path}")
            
            # Add plot path to report
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


