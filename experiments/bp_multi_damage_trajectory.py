#!/usr/bin/env python3
"""
Generate and plot multi-damage BP recovery trajectories.

Runs backprop with repeated reversible damage-recovery cycles and records
per-step accuracy.  Produces trajectory data and a plot comparable to the
GNN multi-damage evaluation stored in stepwise_metrics.jsonl.

Optionally also runs the GNN with the same multi-damage schedule for a
side-by-side comparison on the same figure.

Usage (BP only):
    python experiments/bp_multi_damage_trajectory.py \
        --run-id 2uoa076p \
        --damage-prob 40 \
        --max-damage-per-circuit 9 \
        --greedy-injection-recover-steps 25 \
        --damage-start-offset 5 \
        --n-circuits 100

Usage (BP + GNN comparison):
    python experiments/bp_multi_damage_trajectory.py \
        --run-id 2uoa076p \
        --methods both \
        --n-message-steps 250
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime

import jax
import jax.numpy as jp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from boolean_nca_cc.circuits.model import gen_circuit, generate_layer_sizes
from boolean_nca_cc.circuits.tasks import get_task_data
from boolean_nca_cc.circuits.train import (
    TrainState,
    loss_f_bce,
    loss_f_l4,
    train_step,
)
from boolean_nca_cc.training.checkpointing import (
    load_best_model_from_wandb,
    load_config_from_wandb,
)
from boolean_nca_cc.training.evaluation import evaluate_model_stepwise_batched
from boolean_nca_cc.training.pool.structural_perturbation import (
    create_reproducible_knockout_pattern,
)
from boolean_nca_cc.training.preconfigure import preconfigure_circuit_logits

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Injection schedule
# ---------------------------------------------------------------------------

def build_injection_schedule(
    damage_start_offset: int,
    max_damage_per_circuit: int,
    greedy_injection_recover_steps: int,
    total_steps: int,
) -> list[int]:
    """Return 0-indexed steps where damage is injected.

    Mirrors the GNN evaluation schedule in ``evaluation.py``:
      offset, offset + (R+1), offset + 2*(R+1), ...
    where R = ``greedy_injection_recover_steps``.
    """
    steps = []
    for i in range(max_damage_per_circuit):
        s = damage_start_offset + i * (greedy_injection_recover_steps + 1)
        if s < total_steps:
            steps.append(s)
    return steps


# ---------------------------------------------------------------------------
# BP multi-damage
# ---------------------------------------------------------------------------

def run_bp_multi_damage(
    backprop_cfg: dict,
    layer_sizes: list,
    x_data,
    y_data,
    loss_type: str,
    baseline_params,
    baseline_wires,
    *,
    n_circuits: int = 100,
    damage_prob: int = 40,
    damage_start_offset: int = 5,
    max_damage_per_circuit: int = 9,
    greedy_injection_recover_steps: int = 25,
    reversible_bias: float = -10.0,
    seed: int = 9999,
    bp_epochs: int = 300,
    damage_injection_mode: str = "multi",
    reinject_threshold: float = 1.0,
    reinject_cooldown_steps: int = 5,
) -> pd.DataFrame:
    """Run BP with multi-damage and return per-step per-circuit accuracy.

    Returns a DataFrame with columns ``circuit_idx``, ``step``,
    ``hard_accuracy``.
    """
    import optax

    if backprop_cfg.get("optimizer", "adam") == "adamw":
        opt = optax.adamw(
            backprop_cfg.get("learning_rate", 1.0),
            b1=backprop_cfg.get("beta1", 0.9),
            b2=backprop_cfg.get("beta2", 0.999),
            weight_decay=backprop_cfg.get("weight_decay", 0.0),
        )
    else:
        opt = optax.adam(backprop_cfg.get("learning_rate", 1.0))

    loss_fn = loss_f_l4 if loss_type == "l4" else loss_f_bce

    use_threshold = damage_injection_mode == "threshold_reinject"

    if not use_threshold:
        injection_step_list = build_injection_schedule(
            damage_start_offset,
            max_damage_per_circuit,
            greedy_injection_recover_steps,
            bp_epochs,
        )
        injection_set = set(injection_step_list)
        log.info(
            "Multi-damage BP (fixed): %d events at steps %s (recover=%d, total_epochs=%d)",
            len(injection_step_list),
            injection_step_list,
            greedy_injection_recover_steps,
            bp_epochs,
        )
    else:
        log.info(
            "Multi-damage BP (threshold_reinject): threshold=%.2f, cooldown=%d, "
            "max_events=%d, total_epochs=%d",
            reinject_threshold,
            reinject_cooldown_steps,
            max_damage_per_circuit,
            bp_epochs,
        )

    rows: list[dict] = []
    from tqdm.auto import tqdm

    for circuit_idx in tqdm(range(n_circuits), desc="BP multi-damage circuits"):
        state = TrainState(params=baseline_params, opt_state=opt.init(baseline_params))
        rng = jax.random.PRNGKey(seed + circuit_idx * 1000)

        if use_threshold:
            events = 0
            is_recovered = False
            cooldown_counter = 0

            for step_count in range(bp_epochs):
                # Decide whether to inject
                if step_count == damage_start_offset and events == 0:
                    inject = True
                elif is_recovered and cooldown_counter >= reinject_cooldown_steps and events < max_damage_per_circuit:
                    inject = True
                else:
                    inject = False

                if inject:
                    rng, key = jax.random.split(rng)
                    ko_pattern = create_reproducible_knockout_pattern(
                        key, layer_sizes, damage_prob,
                    )
                    apply_damage_now = True
                    events += 1
                    is_recovered = False
                    cooldown_counter = 0
                else:
                    ko_pattern = None
                    apply_damage_now = False

                _, step_aux, state = train_step(
                    state=state,
                    opt=opt,
                    wires=baseline_wires,
                    x=x_data,
                    y0=y_data,
                    loss_type=loss_type,
                    do_train=True,
                    knockout_pattern=ko_pattern,
                    layer_sizes=layer_sizes,
                    damage_behavior="reversible",
                    reversible_bias=reversible_bias,
                    step_count=step_count,
                    apply_damage_now=apply_damage_now,
                )

                acc = float(step_aux["hard_accuracy"])
                rows.append(
                    {"circuit_idx": circuit_idx, "step": step_count, "hard_accuracy": acc}
                )

                if events > 0 and not is_recovered:
                    if acc >= reinject_threshold:
                        is_recovered = True
                        cooldown_counter = 0
                elif is_recovered:
                    cooldown_counter += 1
        else:
            injection_step_list = build_injection_schedule(
                damage_start_offset,
                max_damage_per_circuit,
                greedy_injection_recover_steps,
                bp_epochs,
            )
            injection_set = set(injection_step_list)
            event_patterns: dict[int, jax.Array] = {}
            for inj_step in injection_step_list:
                rng, key = jax.random.split(rng)
                event_patterns[inj_step] = create_reproducible_knockout_pattern(
                    key, layer_sizes, damage_prob,
                )

            for step_count in range(bp_epochs):
                if step_count in injection_set:
                    ko_pattern = event_patterns[step_count]
                    apply_damage_now = True
                else:
                    ko_pattern = None
                    apply_damage_now = False

                _, step_aux, state = train_step(
                    state=state,
                    opt=opt,
                    wires=baseline_wires,
                    x=x_data,
                    y0=y_data,
                    loss_type=loss_type,
                    do_train=True,
                    knockout_pattern=ko_pattern,
                    layer_sizes=layer_sizes,
                    damage_behavior="reversible",
                    reversible_bias=reversible_bias,
                    step_count=step_count,
                    apply_damage_now=apply_damage_now,
                )
                rows.append(
                    {
                        "circuit_idx": circuit_idx,
                        "step": step_count,
                        "hard_accuracy": float(step_aux["hard_accuracy"]),
                    }
                )

        # Final accuracy after last gradient update
        final_loss, final_aux = loss_fn(
            state.params, baseline_wires, x_data, y_data
        )
        rows.append(
            {
                "circuit_idx": circuit_idx,
                "step": bp_epochs,
                "hard_accuracy": float(final_aux["hard_accuracy"]),
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# GNN multi-damage
# ---------------------------------------------------------------------------

def run_gnn_multi_damage(
    gnn_model,
    baseline_params,
    baseline_wires,
    x_data,
    y_data,
    *,
    input_n: int,
    arity: int,
    circuit_hidden_dim: int,
    layer_sizes: list,
    loss_type: str,
    n_circuits: int = 100,
    damage_prob: int = 40,
    damage_start_offset: int = 5,
    max_damage_per_circuit: int = 9,
    greedy_injection_recover_steps: int = 25,
    n_message_steps: int = 250,
    seed: int = 9999,
    damage_injection_mode: str = "multi",
    reinject_threshold: float = 1.0,
    reinject_cooldown_steps: int = 5,
) -> pd.DataFrame:
    """Run GNN multi-damage evaluation and return per-step per-circuit accuracy."""

    # Batch baseline for all circuits
    batch_wires = jax.tree.map(
        lambda a: jp.repeat(a[None, ...], n_circuits, axis=0),
        baseline_wires,
    )
    batch_logits = jax.tree.map(
        lambda a: jp.repeat(a[None, ...], n_circuits, axis=0),
        baseline_params,
    )

    eval_results = evaluate_model_stepwise_batched(
        model=gnn_model,
        batch_wires=batch_wires,
        batch_logits=batch_logits,
        x_data=x_data,
        y_data=y_data,
        input_n=input_n,
        arity=arity,
        circuit_hidden_dim=circuit_hidden_dim,
        n_message_steps=n_message_steps,
        loss_type=loss_type,
        bidirectional_edges=True,
        layer_sizes=layer_sizes,
        knockout_patterns=None,  # ko_out: generate fresh patterns
        return_per_pattern=True,
        damage_injection_mode=damage_injection_mode,
        damage_start_offset=damage_start_offset,
        damage_mode="shotgun",
        greedy_window_size=damage_prob,
        max_damage_per_circuit=max_damage_per_circuit,
        greedy_injection_recover_steps=greedy_injection_recover_steps,
        reinject_threshold=reinject_threshold,
        reinject_cooldown_steps=reinject_cooldown_steps,
    )

    steps = eval_results.get("step", [])
    per_pattern = eval_results.get("per_pattern", {})
    pattern_hard_accuracies = per_pattern.get("pattern_hard_accuracies", None)

    rows: list[dict] = []
    if pattern_hard_accuracies is not None:
        # shape: [n_steps, batch_size]
        for step_idx, step_num in enumerate(steps):
            for circuit_idx in range(n_circuits):
                rows.append(
                    {
                        "circuit_idx": circuit_idx,
                        "step": int(step_num),
                        "hard_accuracy": float(
                            pattern_hard_accuracies[step_idx, circuit_idx]
                        ),
                    }
                )
    else:
        hard_accuracies = eval_results.get("hard_accuracy", [])
        for step_idx, step_num in enumerate(steps):
            rows.append(
                {
                    "circuit_idx": 0,
                    "step": int(step_num),
                    "hard_accuracy": float(hard_accuracies[step_idx]),
                }
            )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_multi_damage_trajectory(
    bp_df: pd.DataFrame | None,
    gnn_df: pd.DataFrame | None,
    injection_steps_bp: list[int],
    injection_steps_gnn: list[int] | None,
    output_path: str,
    *,
    dpi: int = 600,
) -> str:
    """Plot multi-damage trajectory for BP and/or GNN."""
    fig, ax = plt.subplots(figsize=(12, 4.5))

    # Damage injection markers (use BP schedule as reference)
    ref_steps = injection_steps_bp or injection_steps_gnn or []
    for i, s in enumerate(ref_steps):
        label = "Damage event" if i == 0 else None
        ax.axvline(
            s, color="#d32f2f", linewidth=0.7, linestyle="--", alpha=0.4,
            label=label,
        )

    if bp_df is not None:
        grouped = bp_df.groupby("step")["hard_accuracy"]
        mean_acc = grouped.mean()
        std_acc = grouped.std()
        steps = mean_acc.index.values
        ax.plot(
            steps, mean_acc.values, color="#1565C0", linewidth=1.5,
            label="Backprop",
        )
        ax.fill_between(
            steps, mean_acc - std_acc, mean_acc + std_acc,
            color="#1565C0", alpha=0.15, linewidth=0,
        )
        final = mean_acc.iloc[-1]
        ax.annotate(
            f"{final:.3f}", xy=(steps[-1], final),
            xytext=(-10, 6), textcoords="offset points",
            fontsize=11, color="#383838", ha="right",
        )

    if gnn_df is not None:
        grouped = gnn_df.groupby("step")["hard_accuracy"]
        mean_acc = grouped.mean()
        std_acc = grouped.std()
        steps = mean_acc.index.values
        ax.plot(
            steps, mean_acc.values, color="#34b779", linewidth=1.5,
            label="GNN (SA)",
        )
        ax.fill_between(
            steps, mean_acc - std_acc, mean_acc + std_acc,
            color="#34b779", alpha=0.15, linewidth=0,
        )
        final = mean_acc.iloc[-1]
        ax.annotate(
            f"{final:.3f}", xy=(steps[-1], final),
            xytext=(-10, -14), textcoords="offset points",
            fontsize=11, color="#383838", ha="right",
        )

    ax.set_xlabel("Step", fontsize=16)
    ax.set_ylabel("Hard Accuracy", fontsize=16)
    ax.tick_params(axis="both", which="major", labelsize=13)
    ax.set_ylim(0.7, 1.03)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(fontsize=13, loc="lower right")
        ax.get_legend().get_frame().set_alpha(1.0)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    return output_path


# ---------------------------------------------------------------------------
# Baseline creation helpers
# ---------------------------------------------------------------------------

def _resolve_layer_sizes(cfg):
    """Get layer_sizes from config, checking both top-level and circuit keys."""
    circuit_cfg = cfg.get("circuit", {})
    layer_sizes = circuit_cfg.get("layer_sizes", None)
    if layer_sizes is None:
        layer_sizes = cfg.get("layer_sizes", None)
    if layer_sizes is None:
        input_n = circuit_cfg.get("input_bits", 12)
        output_n = circuit_cfg.get("output_bits", 12)
        arity = circuit_cfg.get("arity", 4)
        num_layers = circuit_cfg.get("num_layers", 3)
        layer_sizes = generate_layer_sizes(input_n, output_n, arity, layer_n=num_layers)
    # Ensure plain Python list-of-lists (OmegaConf ListConfig → list)
    return [list(ls) for ls in layer_sizes]


def _create_baseline(cfg, layer_sizes, arity, x, y0, loss_type):
    """Recreate the preconfigured baseline circuit used during training."""
    training_mode = cfg.get("training", {}).get("training_mode", "growth")
    if training_mode == "repair":
        preconfig_steps = cfg.get("preconfig_steps", 200)
        preconfig_lr = cfg.get("preconfig_lr", 1e-2)
        wiring_fixed_key = cfg.get("wiring_fixed_key", cfg.get("test_seed", 42))
        if isinstance(wiring_fixed_key, int):
            wiring_fixed_key = jax.random.PRNGKey(wiring_fixed_key)
        wires, logits = preconfigure_circuit_logits(
            wiring_key=wiring_fixed_key,
            layer_sizes=layer_sizes,
            arity=arity,
            x_data=x,
            y_data=y0,
            loss_type=loss_type,
            steps=preconfig_steps,
            lr=preconfig_lr,
        )
        log.info(
            "Recreated preconfigured circuit (repair mode, %d steps, lr=%s)",
            preconfig_steps, preconfig_lr,
        )
    else:
        key = jax.random.PRNGKey(cfg.get("test_seed", 42))
        wires, logits = gen_circuit(key, layer_sizes, arity=arity)
        log.warning(
            "Growth mode: using raw circuit as baseline (no BP pre-training)"
        )
    return wires, logits


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multi-damage BP (and optionally GNN) recovery trajectory",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--run-id", type=str, required=True,
        help="WandB run ID for loading config (and GNN model if --methods includes gnn)",
    )
    parser.add_argument("--sweep-id", type=str, default=None)
    parser.add_argument(
        "--methods", type=str, default="bp",
        help="bp, gnn, or both",
    )
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory")
    parser.add_argument("--n-circuits", type=int, default=100,
                        help="Number of circuits (ko_out patterns) to run")
    parser.add_argument("--damage-prob", type=int, default=40,
                        help="Gates knocked out per damage event")
    parser.add_argument("--damage-start-offset", type=int, default=5,
                        help="Clean warm-up steps before first damage")
    parser.add_argument("--max-damage-per-circuit", type=int, default=9,
                        help="Number of damage events per circuit")
    parser.add_argument("--greedy-injection-recover-steps", type=int, default=25,
                        help="Recovery steps between damage events")
    parser.add_argument("--bp-epochs", type=int, default=None,
                        help="Total BP epochs (auto-computed if not set)")
    parser.add_argument("--n-message-steps", type=int, default=250,
                        help="Message steps for GNN evaluation")
    parser.add_argument("--reversible-bias", type=float, default=-10.0)
    parser.add_argument(
        "--damage-injection-mode", type=str, default="multi",
        choices=["multi", "threshold_reinject"],
        help="Injection policy: fixed-interval or threshold-triggered",
    )
    parser.add_argument(
        "--reinject-threshold", type=float, default=1.0,
        help="Hard accuracy that counts as 'recovered' (threshold_reinject mode)",
    )
    parser.add_argument(
        "--reinject-cooldown-steps", type=int, default=5,
        help="Clean steps after recovery before next damage (threshold_reinject mode)",
    )
    parser.add_argument("--seed", type=int, default=9999)
    parser.add_argument("--dpi", type=int, default=600)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    methods = set(
        m.strip()
        for m in args.methods.lower().replace("both", "bp,gnn").split(",")
    )

    # ------------------------------------------------------------------
    # Load config from WandB
    # ------------------------------------------------------------------
    filename_to_load = "best_model_eval_ko_hard_accuracy"
    cfg, _, _ = load_config_from_wandb(
        run_id=args.run_id,
        sweep_id=args.sweep_id,
        filename=filename_to_load,
        select_by_best_metric=bool(args.sweep_id),
    )

    circuit_cfg = cfg.get("circuit", {})
    input_n = circuit_cfg.get("input_bits", 12)
    output_n = circuit_cfg.get("output_bits", 12)
    arity = circuit_cfg.get("arity", 4)
    loss_type = cfg.get("training", {}).get("loss_type", "l4")
    backprop_cfg = dict(cfg.get("backprop", {}))
    layer_sizes = _resolve_layer_sizes(cfg)

    # ------------------------------------------------------------------
    # Data & baseline circuit
    # ------------------------------------------------------------------
    case_n = 1 << input_n
    x, y0 = get_task_data(
        circuit_cfg.get("task", "add"),
        case_n,
        max_samples=circuit_cfg.get("max_task_samples", 100000),
        sample_seed=cfg.get("test_seed", 42),
        input_bits=input_n,
        output_bits=output_n,
    )

    base_wires, base_logits = _create_baseline(
        cfg, layer_sizes, arity, x, y0, loss_type,
    )

    # ------------------------------------------------------------------
    # Auto-compute BP epochs
    # ------------------------------------------------------------------
    use_threshold = args.damage_injection_mode == "threshold_reinject"
    bp_epochs = args.bp_epochs
    if bp_epochs is None:
        bp_epochs = max(
            300,
            args.damage_start_offset
            + args.max_damage_per_circuit
            * (args.greedy_injection_recover_steps + 1)
            + args.greedy_injection_recover_steps,
        )

    # Fixed-interval schedules (only used for plotting markers in fixed mode)
    if use_threshold:
        injection_steps_bp = []
        injection_steps_gnn = []
    else:
        injection_steps_bp = build_injection_schedule(
            args.damage_start_offset,
            args.max_damage_per_circuit,
            args.greedy_injection_recover_steps,
            bp_epochs,
        )
        injection_steps_gnn = build_injection_schedule(
            args.damage_start_offset,
            args.max_damage_per_circuit,
            args.greedy_injection_recover_steps,
            args.n_message_steps,
        )

    # ------------------------------------------------------------------
    # Output directory
    # ------------------------------------------------------------------
    if args.output is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = (
            f"reports/figures/multi_damage_trajectory/"
            f"run{args.run_id}_dam{args.damage_prob}"
            f"_events{args.max_damage_per_circuit}_{ts}"
        )
    os.makedirs(args.output, exist_ok=True)

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------
    bp_df: pd.DataFrame | None = None
    gnn_df: pd.DataFrame | None = None

    if "bp" in methods:
        log.info(
            "Running BP multi-damage trajectory "
            "(%d circuits, %d epochs, %d events)",
            args.n_circuits, bp_epochs, len(injection_steps_bp),
        )
        bp_df = run_bp_multi_damage(
            backprop_cfg=backprop_cfg,
            layer_sizes=layer_sizes,
            x_data=x,
            y_data=y0,
            loss_type=loss_type,
            baseline_params=base_logits,
            baseline_wires=base_wires,
            n_circuits=args.n_circuits,
            damage_prob=args.damage_prob,
            damage_start_offset=args.damage_start_offset,
            max_damage_per_circuit=args.max_damage_per_circuit,
            greedy_injection_recover_steps=args.greedy_injection_recover_steps,
            reversible_bias=args.reversible_bias,
            seed=args.seed,
            bp_epochs=bp_epochs,
            damage_injection_mode=args.damage_injection_mode,
            reinject_threshold=args.reinject_threshold,
            reinject_cooldown_steps=args.reinject_cooldown_steps,
        )
        bp_csv = os.path.join(args.output, "bp_trajectory.csv")
        bp_df.to_csv(bp_csv, index=False)
        log.info("BP trajectory saved: %s (%d rows)", bp_csv, len(bp_df))

    if "gnn" in methods:
        log.info("Loading GNN model for multi-damage evaluation")
        gnn_model, _, _ = load_best_model_from_wandb(
            run_id=args.run_id,
            sweep_id=args.sweep_id,
            seed=0,
            filename=filename_to_load,
            select_by_best_metric=bool(args.sweep_id),
        )
        setattr(gnn_model, "damage_behavior", "reversible")
        gnn_hidden_dim = int(
            cfg.get("model", {}).get("circuit_hidden_dim", 16)
        )

        log.info(
            "Running GNN multi-damage evaluation "
            "(%d circuits, %d steps, %d events)",
            args.n_circuits, args.n_message_steps, len(injection_steps_gnn),
        )
        gnn_df = run_gnn_multi_damage(
            gnn_model=gnn_model,
            baseline_params=base_logits,
            baseline_wires=base_wires,
            x_data=x,
            y_data=y0,
            input_n=input_n,
            arity=arity,
            circuit_hidden_dim=gnn_hidden_dim,
            layer_sizes=layer_sizes,
            loss_type=loss_type,
            n_circuits=args.n_circuits,
            damage_prob=args.damage_prob,
            damage_start_offset=args.damage_start_offset,
            max_damage_per_circuit=args.max_damage_per_circuit,
            greedy_injection_recover_steps=args.greedy_injection_recover_steps,
            n_message_steps=args.n_message_steps,
            seed=args.seed,
            damage_injection_mode=args.damage_injection_mode,
            reinject_threshold=args.reinject_threshold,
            reinject_cooldown_steps=args.reinject_cooldown_steps,
        )
        gnn_csv = os.path.join(args.output, "gnn_trajectory.csv")
        gnn_df.to_csv(gnn_csv, index=False)
        log.info("GNN trajectory saved: %s (%d rows)", gnn_csv, len(gnn_df))

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    plot_path = os.path.join(args.output, "multi_damage_trajectory.png")
    plot_multi_damage_trajectory(
        bp_df, gnn_df,
        injection_steps_bp, injection_steps_gnn,
        plot_path, dpi=args.dpi,
    )
    log.info("Plot saved: %s", plot_path)

    # ------------------------------------------------------------------
    # Save run config for reproducibility
    # ------------------------------------------------------------------
    run_config = {
        "run_id": args.run_id,
        "methods": sorted(methods),
        "n_circuits": args.n_circuits,
        "damage_prob": args.damage_prob,
        "damage_start_offset": args.damage_start_offset,
        "max_damage_per_circuit": args.max_damage_per_circuit,
        "greedy_injection_recover_steps": args.greedy_injection_recover_steps,
        "damage_injection_mode": args.damage_injection_mode,
        "reinject_threshold": args.reinject_threshold,
        "reinject_cooldown_steps": args.reinject_cooldown_steps,
        "bp_epochs": bp_epochs,
        "n_message_steps": args.n_message_steps,
        "reversible_bias": args.reversible_bias,
        "seed": args.seed,
        "injection_steps_bp": injection_steps_bp,
        "injection_steps_gnn": injection_steps_gnn,
        "layer_sizes": layer_sizes,
        "loss_type": loss_type,
        "training_mode": cfg.get("training", {}).get("training_mode", "growth"),
    }
    with open(os.path.join(args.output, "config.json"), "w") as f:
        json.dump(run_config, f, indent=2)

    log.info("Done. Output directory: %s", args.output)


if __name__ == "__main__":
    main()
