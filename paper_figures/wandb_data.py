"""Config-driven pull of the fixed-wiring sweep from W&B.

Replaces the notebook's ad-hoc, global-state cells. For each run in the sweep
group we read the *logged* final-eval metrics (no model re-evaluation needed)
together with the run's own config, and emit one tidy long-form DataFrame.

Crucially, the held-out test split is read from each run's config
(`training.test_num`) rather than hard-coded -- so the paper's "N held out for
testing" claim is verified against what actually ran (reviewer R1c).

Tidy schema (one row per run x eval-damage condition):
    run_id, run_name, task_raw, Task, seed, ood, split,
    train_damage{ON/OFF}, eval_damage{ON/OFF}, wiring_mode, model,
    test_num, input_bits, hard_accuracy, hard_loss, accuracy, loss
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from .style import DAMAGE_MAP, TASK_MAP

# Fig-2 sweep of record: the Feb-12 fixed-wiring sweep -- the one whose runs log
# the damaged-eval ("green") scalar (post the Feb-11 "always eval damaged" change).
# The earlier sweep_20260209_231530 predates damaged-eval logging, so its green is
# NaN; the published Fig 2 (uncommitted boolean_nca_exploration.ipynb) used this one.
DEFAULT_GROUP = "sweep_20260212_141940"
DEFAULT_ENTITY = "gbena"
DEFAULT_PROJECT = "boolean-nca-cc"

_HERE = os.path.dirname(__file__)
CACHE = os.path.join(_HERE, "data", "fixed_wiring_sweep.csv")


def _flat(d: dict, prefix: str = "") -> dict:
    """Flatten a nested config dict to dotted keys (e.g. training.test_num)."""
    out = {}
    for k, v in (d or {}).items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flat(v, key + "."))
        else:
            out[key] = v
    return out


def _cfg(flat: dict, *names, default=None):
    """First present, non-null value among candidate (dotted) keys."""
    for n in names:
        if n in flat and flat[n] is not None:
            return flat[n]
    return default


def _args(run) -> dict:
    """Fallback config from the run's command-line args metadata."""
    md = getattr(run, "metadata", None) or {}
    return dict(m.split("=", 1) for m in md.get("args", []) if "=" in m)


def pull(
    group: str = DEFAULT_GROUP,
    entity: str = DEFAULT_ENTITY,
    project: str = DEFAULT_PROJECT,
    ood=("in",),
    splits=("test",),
    metrics=("hard_accuracy", "hard_loss", "accuracy", "loss"),
) -> pd.DataFrame:
    from wandb import Api

    api = Api()
    runs = api.runs(f"{entity}/{project}", filters={"group": group})
    print(f"[wandb] {len(runs)} runs in group '{group}'")

    rows = []
    test_nums: set[int] = set()
    for run in runs:
        flat = _flat(dict(run.config))
        args = _args(run)
        task = _cfg(flat, "circuit.task", "task") or args.get("circuit") or args.get("circuit.task")
        train_damage = _cfg(flat, "damage.enabled", default=args.get("damage.enabled"))
        seed = _cfg(flat, "seed", default=args.get("seed"))
        test_num = _cfg(flat, "training.test_num", default=args.get("training.test_num"))
        wiring = _cfg(flat, "training.wiring_mode", default=args.get("training.wiring_mode"))
        model = _cfg(flat, "model", default=args.get("model"))
        input_bits = _cfg(flat, "circuit.input_bits", default=args.get("circuit.input_bits", 12))

        if test_num is not None:
            test_nums.add(int(test_num))

        hist = run.history()  # metrics logged across training
        for o in ood:
            for split in splits:
                for ev in (False, True):  # eval-time damage off / on
                    rec = {
                        "run_id": run.id,
                        "run_name": run.name,
                        "task_raw": task,
                        "Task": TASK_MAP.get(str(task), str(task)),
                        "seed": seed,
                        "ood": o,
                        "split": split,
                        "train_damage": DAMAGE_MAP.get(train_damage, str(train_damage)),
                        "eval_damage": "ON" if ev else "OFF",
                        "wiring_mode": wiring,
                        "model": model,
                        "test_num": int(test_num) if test_num is not None else None,
                        "input_bits": int(input_bits) if input_bits is not None else None,
                    }
                    for m in metrics:
                        key = f"eval_{'damaged_' if ev else ''}{o}_{split}/final_{m}"
                        if key in hist.columns:
                            vals = hist[key].values
                            # best-epoch value, matching the notebook's reporting
                            rec[m] = float(np.nanmax(vals) if "accuracy" in m else np.nanmin(vals))
                        else:
                            rec[m] = np.nan
                    rows.append(rec)

    df = pd.DataFrame(rows)
    _report_test_split(df, test_nums)
    return df


def _report_test_split(df: pd.DataFrame, test_nums: set[int]) -> None:
    """Surface the credibility item: which test split actually ran (R1c)."""
    if len(test_nums) == 1:
        n = next(iter(test_nums))
        ib = int(df["input_bits"].dropna().iloc[0]) if df["input_bits"].notna().any() else 12
        case_n = 2 ** ib
        print(
            f"[config] test_num = {n}  -> {n}/{case_n} held out "
            f"(ratio {n / case_n:.4f}); CONSISTENT across all runs. "
            f"Use this number in the dataset-split paragraph."
        )
    elif not test_nums:
        print("[config] WARNING: test_num not found in any run config -- inspect run.config manually.")
    else:
        print(f"[config] WARNING: test_num INCONSISTENT across runs: {sorted(test_nums)}")


def load(use_cache: bool = True, **kw) -> pd.DataFrame:
    """Load tidy sweep data, pulling + caching to CSV on first call."""
    if use_cache and os.path.exists(CACHE):
        print(f"[cache] reading {CACHE}")
        return pd.read_csv(CACHE)
    df = pull(**kw)
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    df.to_csv(CACHE, index=False)
    print(f"[cache] wrote {CACHE} ({len(df)} rows)")
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Pull + cache the fixed-wiring sweep.")
    ap.add_argument("--no-cache", action="store_true", help="re-pull from W&B even if cached")
    ap.add_argument("--group", default=DEFAULT_GROUP)
    a = ap.parse_args()
    df = load(use_cache=not a.no_cache, group=a.group)
    print(df.head(12).to_string())
    print(
        "\nrows:", len(df),
        "| tasks:", sorted(df["Task"].dropna().unique().tolist()),
        "| seeds:", sorted(df["seed"].dropna().unique().tolist()),
        "| train_damage:", sorted(df["train_damage"].dropna().unique().tolist()),
        "| model:", sorted(df["model"].dropna().unique().tolist()),
        "| wiring:", sorted(df["wiring_mode"].dropna().unique().tolist()),
    )
