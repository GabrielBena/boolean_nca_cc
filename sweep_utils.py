"""
Utility functions for loading sweep results into pandas DataFrames.

Usage:
    from sweep_utils import load_sweep

    df_eval, df_step = load_sweep("sweep_wo7pa2id")
    # or just eval metrics (faster):
    df_eval = load_sweep("sweep_wo7pa2id", load_stepwise=False)
"""

import json
import os
from pathlib import Path

import pandas as pd
import yaml

SWEEP_OUTPUT_DIR = Path(__file__).parent / "sweeps" / "outputs"

# Metric columns to keep from eval_metrics.jsonl (after normalization)
EVAL_METRIC_COLS = [
    "final_loss",
    "final_hard_loss",
    "final_accuracy",
    "final_hard_accuracy",
    "final_full_map_accuracy",
]

# Metric columns to keep from stepwise_metrics.jsonl
STEP_METRIC_COLS = [
    "soft_loss",
    "hard_loss",
    "soft_accuracy",
    "hard_accuracy",
    "full_map_accuracy",
    "logits_mean",
    "logits_std",
    "logits_min",
    "logits_max",
    "hidden_l2_norm",
    "hidden_mean",
    "hidden_std",
]


def _flatten_dict(d: dict, prefix: str = "") -> dict:
    """Flatten a nested dict with dot-separated keys."""
    items = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            items.update(_flatten_dict(v, key))
        elif isinstance(v, list):
            items[key] = str(v)
        else:
            items[key] = v
    return items


def _find_varying_keys(flat_configs: list[dict]) -> list[str]:
    """Find config keys whose values differ across runs.

    Automatically detects and removes redundant keys (keys whose value
    vectors are identical to another varying key, e.g. eval.input_split_seed
    mirroring seed).
    """
    all_keys = set()
    for fc in flat_configs:
        all_keys.update(fc.keys())

    # Step 1: find keys with more than one unique value
    varying = []
    for k in sorted(all_keys):
        vals = {str(fc.get(k, "__MISSING__")) for fc in flat_configs}
        if len(vals) > 1:
            varying.append(k)

    # Step 2: deduplicate keys with identical value vectors
    # (e.g. eval.input_split_seed == seed for every run)
    # Group keys by their value vector, keep the shortest key name per group.
    value_vectors = {}
    for k in varying:
        vec = tuple(str(fc.get(k, "__MISSING__")) for fc in flat_configs)
        value_vectors[k] = vec

    groups: dict[tuple, list[str]] = {}
    for k in varying:
        vec = value_vectors[k]
        groups.setdefault(vec, []).append(k)

    deduplicated = []
    for vec, keys in groups.items():
        # prefer shortest key name (e.g. "seed" over "eval.input_split_seed")
        best = min(keys, key=len)
        deduplicated.append(best)

    return sorted(deduplicated)


def _normalize_train_columns(row: dict) -> dict:
    """Normalize train-split eval columns by stripping the
    'eval_no_damage_train/' prefix so they match test-split columns."""
    out = {}
    for k, v in row.items():
        if k.startswith("eval_no_damage_train/"):
            new_key = k.replace("eval_no_damage_train/", "")
            # skip the duplicate 'epoch' that comes prefixed
            if new_key == "epoch":
                continue
            out[new_key] = v
        else:
            out[k] = v
    return out


def _load_run_configs(sweep_dir: Path) -> tuple[list[str], list[dict], list[dict]]:
    """Load all run configs. Returns (run_dirs, flat_configs, raw_configs)."""
    run_dirs = sorted(d for d in os.listdir(sweep_dir) if (sweep_dir / d).is_dir())
    flat_configs = []
    raw_configs = []
    for rd in run_dirs:
        cfg_path = sweep_dir / rd / "config_resolved.yaml"
        with open(cfg_path) as f:
            raw = yaml.safe_load(f)
        raw_configs.append(raw)
        flat_configs.append(_flatten_dict(raw))
    return run_dirs, flat_configs, raw_configs


def load_sweep(
    sweep_id: str,
    load_stepwise: bool = False,
    sweep_output_dir: Path | str | None = None,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """Load all results from a sweep into tidy pandas DataFrames.

    Parameters
    ----------
    sweep_id : str
        The sweep identifier (directory name under sweeps/outputs/, e.g. "sweep_wo7pa2id").
    load_stepwise : bool
        If True, also loads the (large) stepwise_metrics.jsonl files
        and returns a second DataFrame.
    sweep_output_dir : Path or str, optional
        Override the default sweep output directory.

    Returns
    -------
    df_eval : pd.DataFrame
        One row per (run, epoch, split). Contains only the config parameters
        that actually vary between runs, plus the core eval metrics.
    df_step : pd.DataFrame  (only if load_stepwise=True)
        One row per (run, epoch, split, step). Same varying config params
        plus per-step metrics.
    """
    if sweep_output_dir is None:
        sweep_output_dir = SWEEP_OUTPUT_DIR
    sweep_dir = Path(sweep_output_dir) / sweep_id

    if not sweep_dir.exists():
        raise FileNotFoundError(f"Sweep directory not found: {sweep_dir}")

    # --- 1. Load configs and find varying parameters ---
    run_dirs, flat_configs, _ = _load_run_configs(sweep_dir)
    varying_keys = _find_varying_keys(flat_configs)

    # Build a lookup: run_dir -> {varying param: value}
    run_params = {}
    for rd, fc in zip(run_dirs, flat_configs):
        run_params[rd] = {k: fc.get(k) for k in varying_keys}

    print(f"Found {len(run_dirs)} runs in {sweep_dir.name}")
    print(f"Varying config keys: {varying_keys}")

    # --- 2. Load eval metrics ---
    eval_rows = []
    for rd in run_dirs:
        eval_path = sweep_dir / rd / "eval_metrics.jsonl"
        if not eval_path.exists():
            continue
        params = run_params[rd]
        with open(eval_path) as f:
            for line in f:
                row = json.loads(line)
                row = _normalize_train_columns(row)
                # Add varying config params
                for k, v in params.items():
                    row[k] = v
                eval_rows.append(row)

    df_eval = pd.DataFrame(eval_rows)

    # Keep only: run_id, epoch, split, varying config params, metric columns
    keep_cols = ["run_id", "epoch", "split", "input_split"] + varying_keys
    keep_cols += [c for c in EVAL_METRIC_COLS if c in df_eval.columns]
    # drop any duplicates in keep list while preserving order
    seen = set()
    keep_cols = [c for c in keep_cols if c not in seen and not seen.add(c)]
    df_eval = df_eval[[c for c in keep_cols if c in df_eval.columns]]

    # Nicer column names: replace dots with underscores for the config params
    rename = {k: k.replace(".", "_") for k in varying_keys if "." in k}
    df_eval = df_eval.rename(columns=rename)

    # Sort for readability
    sort_by = [rename.get(k, k) for k in varying_keys] + ["epoch", "split"]
    sort_by = [c for c in sort_by if c in df_eval.columns]
    df_eval = df_eval.sort_values(sort_by).reset_index(drop=True)

    print(f"df_eval: {df_eval.shape[0]} rows × {df_eval.shape[1]} cols")

    if not load_stepwise:
        return df_eval

    # --- 3. Load stepwise metrics ---
    step_rows = []
    for i, rd in enumerate(run_dirs):
        step_path = sweep_dir / rd / "stepwise_metrics.jsonl"
        if not step_path.exists():
            continue
        params = run_params[rd]
        with open(step_path) as f:
            for line in f:
                row = json.loads(line)
                for k, v in params.items():
                    row[k] = v
                step_rows.append(row)
        if (i + 1) % 25 == 0:
            print(f"  loaded stepwise for {i + 1}/{len(run_dirs)} runs...")

    df_step = pd.DataFrame(step_rows)

    keep_step = ["run_id", "epoch", "split", "input_split", "step"] + varying_keys
    keep_step += [c for c in STEP_METRIC_COLS if c in df_step.columns]
    seen = set()
    keep_step = [c for c in keep_step if c not in seen and not seen.add(c)]
    df_step = df_step[[c for c in keep_step if c in df_step.columns]]
    df_step = df_step.rename(columns=rename)

    sort_step = [rename.get(k, k) for k in varying_keys] + ["epoch", "split", "step"]
    sort_step = [c for c in sort_step if c in df_step.columns]
    df_step = df_step.sort_values(sort_step).reset_index(drop=True)

    print(f"df_step: {df_step.shape[0]} rows × {df_step.shape[1]} cols")

    return df_eval, df_step


def get_final_metrics(df_eval: pd.DataFrame) -> pd.DataFrame:
    """Extract the last-epoch metrics per run from df_eval.

    Useful for comparing final performance across configs.
    Returns one row per (run_id, split).
    """
    idx = df_eval.groupby(["run_id", "split"])["epoch"].idxmax()
    return df_eval.loc[idx].reset_index(drop=True)
