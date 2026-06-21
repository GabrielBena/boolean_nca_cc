"""Config-matched, per-seed BP ceiling for the fixed-wiring tasks.

Fixes the notebook's leaked-global BP cell (cell 64), where BP reused a single
`loaded_config`/`pool` and produced one number per task. Here we load each
task's own sweep-run config and run `run_bp_scan` **once per seed** (matching the
sweep's seeds and per-seed wiring / test split), with and without eval damage --
so BP gets its own seed distribution and the TMT-vs-BP comparison is a fair
boxplot-vs-boxplot at matched conditions.

This is the only part of the pipeline that needs JAX + the package; it is a
handful of short `run_bp_scan` runs (3 tasks x n_seeds x 2 damage), not the
abandoned full per-run model re-evaluation.

Output: data/fixed_wiring_bp.csv with columns
    Task, task_short, seed, eval_damage{ON/OFF},
    hard_accuracy, hard_loss, accuracy, loss, run_id, test_num
"""
from __future__ import annotations

import argparse
import os

import pandas as pd

from .style import TASK_MAP
from .wandb_data import DEFAULT_ENTITY, DEFAULT_GROUP, DEFAULT_PROJECT, _args, _cfg, _flat

_HERE = os.path.dirname(__file__)
OUT = os.path.join(_HERE, "data", "fixed_wiring_bp.csv")


def _representative_run_per_task(group, entity, project) -> dict:
    """One run id per task (prefer the undamaged-training run) to read its config."""
    from wandb import Api

    api = Api()
    runs = api.runs(f"{entity}/{project}", filters={"group": group})
    chosen: dict[str, str] = {}
    for run in runs:
        flat = _flat(dict(run.config))
        task = _cfg(flat, "circuit.task", "task") or _args(run).get("circuit")
        td = _cfg(flat, "damage.enabled", default=_args(run).get("damage.enabled"))
        if task is None:
            continue
        if task not in chosen or str(td).lower() == "false":
            chosen[task] = run.id
    return chosen


def _seeds_from_sweep(group) -> list[int]:
    """The seeds actually used in the sweep (so BP matches the TMT seeds)."""
    from . import wandb_data

    df = wandb_data.load(use_cache=True, group=group)
    return sorted(int(s) for s in df["seed"].dropna().unique())


def compute(group=DEFAULT_GROUP, entity=DEFAULT_ENTITY, project=DEFAULT_PROJECT,
            batch_size=16, seeds=None):
    import jax
    import optax

    from boolean_nca_cc.circuits.model import generate_layer_sizes
    from boolean_nca_cc.circuits.tasks import get_task_data
    from boolean_nca_cc.circuits.train import LossConfig
    from boolean_nca_cc.training.checkpointing import load_config_from_wandb
    from boolean_nca_cc.training.evaluation import run_bp_scan
    from boolean_nca_cc.training.pool.pool import initialize_graph_pool
    from boolean_nca_cc.training.pool.structural_perturbation import compute_damage_params

    if seeds is None:
        seeds = _seeds_from_sweep(group)
    print(f"[bp] seeds = {seeds}")
    reps = _representative_run_per_task(group, entity, project)
    print(f"[bp] representative run per task: {reps}")

    rows = []
    for task_raw, run_id in reps.items():
        cfg, _ckpt, rid = load_config_from_wandb(
            run_id=run_id, entity=entity, filename="best_model_eval", use_cache=False
        )
        short = str(task_raw).replace("_large", "")
        case_n = 2 ** cfg.circuit.input_bits
        input_n, output_n = cfg.circuit.input_bits, cfg.circuit.output_bits
        arity, layer_n = cfg.circuit.arity, cfg.circuit.num_layers
        hidden_dim = cfg.circuit.circuit_hidden_dim
        wiring_mode = cfg.training.wiring_mode
        test_num = int(cfg.training.test_num)
        layer_sizes = generate_layer_sizes(
            input_n, output_n, arity, layer_n=layer_n, width_factor=cfg.circuit.width_factor
        )
        n_steps = cfg.eval.inner_steps
        dmg = compute_damage_params(cfg, layer_sizes, None)

        for seed in seeds:
            # per-seed keys, mirroring cell 57 (seed -> wiring + test split)
            base_key = jax.random.PRNGKey(int(seed))
            eval_key = jax.random.fold_in(base_key, 1)
            pool_key = eval_key if wiring_mode == "fixed" else jax.random.fold_in(eval_key, 1)
            wires_key, logits_key = jax.random.split(pool_key, 2)
            pool = initialize_graph_pool(
                wires_key=wires_key, logits_key=logits_key, pool_size=batch_size,
                layer_sizes=layer_sizes, arity=arity, input_n=input_n,
                circuit_hidden_dim=hidden_dim, wiring_mode=wiring_mode, noise_scale=0.1,
            )
            (x_tr, y_tr), (x_te, y_te), _ = get_task_data(
                short, case_n, input_bits=input_n, output_bits=output_n, text=cfg.circuit.text,
                train_test_split=True, test_ratio=test_num / case_n, seed=eval_key,
            )
            for eval_damage in (False, True):
                opt = optax.adamw(1, 0.8, 0.8, weight_decay=1e-1)
                _, metrics = run_bp_scan(
                    batch_wires=pool.wires, batch_params=pool.logits, opt=opt,
                    x_data=x_tr, y_data=y_tr, x_test=x_te, y_test=y_te,
                    n_steps=n_steps, layer_sizes=layer_sizes, input_n=input_n, arity=arity,
                    loss_cfg=LossConfig(cfg.loss),
                    p_fault=dmg["p_fault_eval"] if eval_damage else None,
                    p_fault_onset_step=0,
                    damage_mode="probabilistic" if eval_damage else "none",
                    damage_steps=None, knockouts_per_event=None,
                    faulty_value=dmg["faulty_logit_value"], permanent=1.0,
                )
                rows.append({
                    "Task": TASK_MAP.get(task_raw, short),
                    "task_short": short,
                    "seed": int(seed),
                    "eval_damage": "ON" if eval_damage else "OFF",
                    "hard_accuracy": float(metrics["hard_accuracy"][-1]),
                    "hard_loss": float(metrics["hard_loss"][-1]),
                    "accuracy": float(metrics["accuracy"][-1]),
                    "loss": float(metrics["loss"][-1]),
                    "run_id": rid,
                    "test_num": test_num,
                })
            print(f"[bp] {short:16s} seed={seed} "
                  f"off={rows[-2]['hard_accuracy']:.4f} on={rows[-1]['hard_accuracy']:.4f}")

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"[bp] wrote {OUT} ({len(df)} rows)")
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Compute the per-seed config-matched BP ceiling.")
    ap.add_argument("--group", default=DEFAULT_GROUP)
    a = ap.parse_args()
    compute(group=a.group)
