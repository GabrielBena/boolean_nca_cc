"""Principled re-gather of Fig 2 (fixed-wiring) evaluation results.

Re-runs the evaluation for every run in the sweep, config-driven, and aggregates
to tidy CSVs -- replacing the notebook's mix of W&B-history scalars (which lack
the damaged-eval "green" branch) and leaked-global BP. Faithfully mirrors
`get_eval_results` (trained_models cell 57) + the BP cell (cell 64).

For each sweep run we report the FINAL-step, batch-averaged test hard accuracy
(`step_metrics["hard_accuracy"][-1]`, on the held-out test split) under:
  - TMT, eval undamaged   (p_fault=None)
  - TMT, eval damaged      (probabilistic, permanent -- the "stochastic failures"
                            green branch of Fig 2)
and BP (run_bp_scan), undamaged + damaged, once per (task, seed).

Outputs (schema consumed by fig2_fixed_wiring.py):
  data/fixed_wiring_sweep.csv  - TMT, one row per (run x eval_damage)
  data/fixed_wiring_bp.csv     - BP,  one row per (task x seed x eval_damage)

Develop on CPU (`conda run -n nca`); scale out to anahita only if needed.
Run from a neutral cwd with the repo on sys.path (the repo's local wandb/ dir
shadows the package), e.g.:
  cd /tmp && conda run -n nca python -c "import sys; sys.path.append('<repo>'); \
     from paper_figures import reeval_fig2; reeval_fig2.main(limit=2)"
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from .style import TASK_MAP
from .wandb_data import DEFAULT_ENTITY, DEFAULT_GROUP, DEFAULT_PROJECT, _args, _cfg, _flat

_HERE = os.path.dirname(__file__)
TMT_OUT = os.path.join(_HERE, "data", "fixed_wiring_sweep.csv")
BP_OUT = os.path.join(_HERE, "data", "fixed_wiring_bp.csv")
BATCH_SIZE = 16


def _run_list(group, entity, project):
    """(run_id, task_raw, train_damage{ON/OFF}, seed) for every run in the group."""
    from wandb import Api

    api = Api()
    runs = api.runs(f"{entity}/{project}", filters={"group": group})
    out = []
    for run in runs:
        flat = _flat(dict(run.config))
        task = _cfg(flat, "circuit.task", "task") or _args(run).get("circuit")
        td = _cfg(flat, "damage.enabled", default=_args(run).get("damage.enabled"))
        seed = _cfg(flat, "seed", default=_args(run).get("seed"))
        out.append((run.id, str(task), "ON" if str(td).lower() == "true" else "OFF", int(seed)))
    return out


def _setup_from_config(cfg, jax):
    """Rebuild pool / test data / damage params from a run config (mirrors cell 57)."""
    from boolean_nca_cc.circuits.model import generate_layer_sizes
    from boolean_nca_cc.circuits.tasks import get_task_data
    from boolean_nca_cc.training.pool.pool import initialize_graph_pool
    from boolean_nca_cc.training.pool.structural_perturbation import compute_damage_params

    case_n = 2 ** cfg.circuit.input_bits
    base_key = jax.random.PRNGKey(cfg.seed)
    eval_key = (
        jax.random.fold_in(base_key, 1)
        if not hasattr(cfg, "test_seed")
        else jax.random.PRNGKey(cfg.test_seed)
    )
    input_n, output_n = cfg.circuit.input_bits, cfg.circuit.output_bits
    arity, layer_n = cfg.circuit.arity, cfg.circuit.num_layers
    hidden_dim = cfg.circuit.circuit_hidden_dim
    layer_sizes = generate_layer_sizes(
        input_n, output_n, arity, layer_n=layer_n, width_factor=cfg.circuit.width_factor
    )
    pool_key = eval_key if cfg.training.wiring_mode == "fixed" else jax.random.fold_in(eval_key, 1)
    wires_key, logits_key = jax.random.split(pool_key, 2)
    pool = initialize_graph_pool(
        wires_key=wires_key, logits_key=logits_key, pool_size=BATCH_SIZE,
        layer_sizes=layer_sizes, arity=arity, input_n=input_n,
        circuit_hidden_dim=hidden_dim, wiring_mode=cfg.training.wiring_mode, noise_scale=0.1,
    )
    short = str(cfg.circuit.task).replace("_large", "")
    (x_tr, y_tr), (x_te, y_te), _ = get_task_data(
        short, case_n, input_bits=input_n, output_bits=output_n, text=cfg.circuit.text,
        train_test_split=True, test_ratio=int(cfg.training.test_num) / case_n, seed=eval_key,
    )
    dmg = compute_damage_params(cfg, layer_sizes, None)
    return dict(
        pool=pool, x_tr=x_tr, y_tr=y_tr, x_te=x_te, y_te=y_te, layer_sizes=layer_sizes,
        input_n=input_n, arity=arity, hidden_dim=hidden_dim, dmg=dmg,
        n_steps=cfg.eval.inner_steps, short=short, test_num=int(cfg.training.test_num),
    )


def reeval(group=DEFAULT_GROUP, entity=DEFAULT_ENTITY, project=DEFAULT_PROJECT, limit=None):
    import jax  # noqa: F401
    import optax

    from boolean_nca_cc.circuits.train import LossConfig
    from boolean_nca_cc.training.checkpointing import (
        load_config_from_wandb,
        load_model_from_config_and_checkpoint,
    )
    from boolean_nca_cc.training.evaluation import evaluate_model_stepwise_batched, run_bp_scan

    runs = _run_list(group, entity, project)
    if limit:
        runs = runs[:limit]
    print(f"[reeval] {len(runs)} runs to evaluate")

    tmt_rows, bp_rows, bp_done = [], [], set()
    for i, (run_id, task_raw, train_damage, seed) in enumerate(runs):
        cfg, ckpt, rid = load_config_from_wandb(
            run_id=run_id, entity=entity, filename="best_model_eval", use_cache=False
        )
        s = _setup_from_config(cfg, jax)
        loss_cfg = LossConfig(dict(cfg.loss))
        Task = TASK_MAP.get(task_raw, s["short"])
        try:
            mdl = cfg.model if isinstance(cfg.model, str) else getattr(cfg.model, "type", "unknown")
        except Exception:
            mdl = "unknown"
        common = dict(
            batch_wires=s["pool"].wires, batch_logits=s["pool"].logits,
            x_data=s["x_te"], y_data=s["y_te"], input_n=s["input_n"], arity=s["arity"],
            circuit_hidden_dim=s["hidden_dim"], n_message_steps=s["n_steps"],
            loss_cfg=loss_cfg, layer_sizes=s["layer_sizes"], bidirectional_edges=True,
            keep_full_graphs=False, verbose=False,
        )
        model, _, _ = load_model_from_config_and_checkpoint(
            config=cfg, checkpoint_path=ckpt, run_id=rid, seed=cfg.seed,
            n_node=int(s["pool"].graphs.n_node[0].item()),
        )
        # TMT: undamaged + damaged (probabilistic, permanent) eval
        for eval_damage in ("OFF", "ON"):
            _, sm = evaluate_model_stepwise_batched(
                model=model,
                p_fault=s["dmg"]["p_fault_eval"] if eval_damage == "ON" else None,
                permanent_damage=True, p_fault_onset_step=0,
                **common,
            )
            tmt_rows.append(dict(
                run_id=run_id, run_name=rid, task_raw=task_raw, Task=Task, seed=seed,
                ood="in", split="test", train_damage=train_damage, eval_damage=eval_damage,
                wiring_mode=cfg.training.wiring_mode, model=mdl,
                test_num=s["test_num"], input_bits=s["input_n"],
                hard_accuracy=float(sm["hard_accuracy"][-1]),
                hard_loss=float(sm["hard_loss"][-1]), accuracy=float(sm["accuracy"][-1]),
                loss=float(sm["loss"][-1]),
            ))
        # BP: once per (task, seed)
        if (s["short"], seed) not in bp_done:
            bp_done.add((s["short"], seed))
            for eval_damage in ("OFF", "ON"):
                opt = optax.adamw(1, 0.8, 0.8, weight_decay=1e-1)
                _, bm = run_bp_scan(
                    batch_wires=s["pool"].wires, batch_params=s["pool"].logits, opt=opt,
                    x_data=s["x_tr"], y_data=s["y_tr"], x_test=s["x_te"], y_test=s["y_te"],
                    n_steps=s["n_steps"], layer_sizes=s["layer_sizes"], input_n=s["input_n"],
                    arity=s["arity"], loss_cfg=loss_cfg,
                    p_fault=s["dmg"]["p_fault_eval"] if eval_damage == "ON" else None,
                    p_fault_onset_step=0,
                    damage_mode="probabilistic" if eval_damage == "ON" else "none",
                    damage_steps=None, knockouts_per_event=None,
                    faulty_value=s["dmg"]["faulty_logit_value"], permanent=1.0,
                )
                bp_rows.append(dict(
                    Task=Task, task_short=s["short"], seed=seed, eval_damage=eval_damage,
                    hard_accuracy=float(bm["hard_accuracy"][-1]),
                    hard_loss=float(bm["hard_loss"][-1]), accuracy=float(bm["accuracy"][-1]),
                    loss=float(bm["loss"][-1]), test_num=s["test_num"],
                ))
        # incremental save so partial runs are usable
        os.makedirs(os.path.dirname(TMT_OUT), exist_ok=True)
        pd.DataFrame(tmt_rows).to_csv(TMT_OUT, index=False)
        pd.DataFrame(bp_rows).to_csv(BP_OUT, index=False)
        last = tmt_rows[-1]
        print(f"[{i + 1}/{len(runs)}] {Task:22s} train={train_damage} seed={seed} "
              f"TMT off={tmt_rows[-2]['hard_accuracy']:.3f} on={last['hard_accuracy']:.3f}")

    print(f"[reeval] wrote {TMT_OUT} ({len(tmt_rows)} rows) + {BP_OUT} ({len(bp_rows)} rows)")
    return pd.DataFrame(tmt_rows), pd.DataFrame(bp_rows)


def main(limit=None, group=DEFAULT_GROUP):
    reeval(group=group, limit=limit)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Re-gather Fig 2 eval results, config-driven.")
    ap.add_argument("--limit", type=int, default=None, help="evaluate only the first N runs (smoke test)")
    ap.add_argument("--group", default=DEFAULT_GROUP)
    a = ap.parse_args()
    main(limit=a.limit, group=a.group)
