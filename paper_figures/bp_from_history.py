"""BP ceiling for Fig 2, read from each run's *logged* W&B summary.

The Fig-2 sweep already logs a config-matched BP baseline per run
(`BP_results/none/*` undamaged, `BP_results/probabilistic/*` damaged) -- exactly
what the original (uncommitted) `boolean_nca_exploration.ipynb` plotted. Reading
those logged scalars reproduces the paper's BP exactly, with no JAX and no
re-optimisation (cf. `bp_baseline.py`, which *recomputes* BP via `run_bp_scan`).

Output: data/fixed_wiring_bp.csv  (schema consumed by fig2_fixed_wiring.py)
    Task, task_short, seed, train_damage, eval_damage{OFF/ON}, hard_accuracy
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from .style import TASK_MAP
from .wandb_data import DEFAULT_ENTITY, DEFAULT_GROUP, DEFAULT_PROJECT, _args, _cfg, _flat

_HERE = os.path.dirname(__file__)
OUT = os.path.join(_HERE, "data", "fixed_wiring_bp.csv")
# eval-OFF -> undamaged BP; eval-ON -> probabilistic-damage BP (matches the published Fig 2)
KEY = {"OFF": "BP_results/none/hard_accuracy", "ON": "BP_results/probabilistic/hard_accuracy"}


def _f(x):
    try:
        return float(x)
    except Exception:
        return None


def _get_bp(run, key):
    """Final BP scalar: summary (flat, then nested path), else unsampled history scan.

    BP_results are logged at different steps than the eval metrics, so the default
    sampled run.history() drops them -- the value lives in run.summary.
    """
    sm = dict(run.summary)
    if key in sm and _f(sm[key]) is not None:
        return _f(sm[key])
    node = sm
    for p in key.split("/"):
        node = node[p] if isinstance(node, dict) and p in node else None
        if node is None:
            break
    if _f(node) is not None:
        return _f(node)
    best = None
    try:
        for row in run.scan_history(keys=[key]):
            v = _f(row.get(key))
            if v is not None:
                best = v if best is None else max(best, v)
    except Exception:
        pass
    return best if best is not None else np.nan


def pull(group=DEFAULT_GROUP, entity=DEFAULT_ENTITY, project=DEFAULT_PROJECT):
    from wandb import Api

    api = Api()
    runs = api.runs(f"{entity}/{project}", filters={"group": group})
    print(f"[bp] {len(runs)} runs in '{group}'")
    rows = []
    for run in runs:
        flat = _flat(dict(run.config))
        task = _cfg(flat, "circuit.task", "task") or _args(run).get("circuit")
        seed = _cfg(flat, "seed", default=_args(run).get("seed"))
        td = _cfg(flat, "damage.enabled", default=_args(run).get("damage.enabled"))
        for ev in ("OFF", "ON"):
            rows.append({
                "Task": TASK_MAP.get(str(task), str(task)),
                "task_short": str(task).replace("_large", ""),
                "seed": int(seed) if seed is not None else None,
                "train_damage": "ON" if str(td).lower() == "true" else "OFF",
                "eval_damage": ev,
                "hard_accuracy": _get_bp(run, KEY[ev]),
            })
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"[bp] wrote {OUT} ({len(df)} rows)")
    return df


def load(use_cache: bool = True, **kw) -> pd.DataFrame:
    """Load the BP ceiling, pulling + caching to CSV on first call.

    OUT is committed to the repo (see paper_figures/.gitignore) with a real pull
    already in it, so this is W&B-free by default -- pass use_cache=False to
    force a fresh pull.
    """
    if use_cache and os.path.exists(OUT):
        print(f"[cache] reading {OUT}")
        return pd.read_csv(OUT)
    return pull(**kw)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Pull the logged BP ceiling for Fig 2.")
    ap.add_argument("--no-cache", action="store_true", help="re-pull from W&B even if cached")
    ap.add_argument("--group", default=DEFAULT_GROUP)
    a = ap.parse_args()
    df = load(use_cache=not a.no_cache, group=a.group)
    s = df.groupby(["Task", "eval_damage"]).agg(
        mean=("hard_accuracy", "mean"), n=("hard_accuracy", "size"),
        n_nan=("hard_accuracy", lambda x: int(x.isna().sum())),
    ).reset_index()
    print(s.to_string(index=False))
