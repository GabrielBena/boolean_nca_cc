"""
Figure 2 — Performance vs training damage exposure (seen vs unseen).

This module fetches multiple WandB runs with varying pool.damage_knockout_diversity
and aggregates final IN/OUT hard accuracies for plotting.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import wandb
except ImportError:  # pragma: no cover - runtime environment without wandb
    wandb = None  # type: ignore

log = logging.getLogger(__name__)


@dataclass
class RunSummary:
    run_id: str
    damage_knockout_diversity: int
    training_mode: str
    hard_acc_in: Optional[float]
    hard_acc_out: Optional[float]


def _get_config_value(config: Dict[str, Any], dotted_key: str, default: Any = None) -> Any:
    """
    Safely read nested config using dotted path (e.g., "pool.damage_knockout_diversity").
    """
    current: Any = config
    for part in dotted_key.split("."):
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return default
    return current


def list_runs_grouped_by_diversity(
    project: str,
    entity: Optional[str] = None,
    filters: Optional[Dict[str, Any]] = None,
) -> List[RunSummary]:
    """
    List WandB runs and extract final IN/OUT hard accuracy along with
    pool.damage_knockout_diversity and training.training_mode.
    """
    if wandb is None:
        log.error("wandb not available. Install wandb to fetch remote results.")
        return []

    api = wandb.Api()
    project_path = f"{entity + '/' if entity else ''}{project}"

    # Build filters; avoid None which API rejects
    wandb_filters: Dict[str, Any] = filters.copy() if filters else {}

    try:
        runs = api.runs(project_path, filters=wandb_filters)
    except Exception as e:  # noqa: BLE001
        log.error(f"Failed to list runs for {project_path}: {e}")
        return []

    results: List[RunSummary] = []
    for run in runs:
        try:
            cfg = run.config or {}
            training_mode = _get_config_value(cfg, "training.training_mode", "unknown")
            diversity = _get_config_value(cfg, "pool.damage_knockout_diversity", None)

            # Skip runs without the diversity parameter
            if diversity is None:
                continue

            # Prefer run.summary values; fallback to last history value if missing
            hard_in = run.summary.get("eval_ko_in/final_hard_accuracy")
            hard_out = run.summary.get("eval_ko_out/final_hard_accuracy")

            # Handle WandB SummarySubDict objects by extracting the value
            def _extract_value(val):
                if val is None:
                    return None
                if hasattr(val, 'value'):  # SummarySubDict has .value attribute
                    return val.value
                return val

            hard_in = _extract_value(hard_in)
            hard_out = _extract_value(hard_out)

            # Some runs might not have summaries; fetch the last value from history
            if hard_in is None or hard_out is None:
                try:
                    hist = run.history(keys=[
                        "eval_ko_in/final_hard_accuracy",
                        "eval_ko_out/final_hard_accuracy",
                    ])
                    if len(hist) > 0:
                        last_row = hist.iloc[-1]
                        hard_in = hard_in if hard_in is not None else last_row.get("eval_ko_in/final_hard_accuracy")
                        hard_out = hard_out if hard_out is not None else last_row.get("eval_ko_out/final_hard_accuracy")
                except Exception as e:  # noqa: BLE001
                    log.warning(f"Could not read history for run {run.id}: {e}")

            results.append(
                RunSummary(
                    run_id=run.id,
                    damage_knockout_diversity=int(diversity),
                    training_mode=str(training_mode),
                    hard_acc_in=float(hard_in) if hard_in is not None else None,
                    hard_acc_out=float(hard_out) if hard_out is not None else None,
                )
            )
        except Exception as e:  # noqa: BLE001
            log.warning(f"Skipping run due to parsing error: {e}")

    return results


def aggregate_by_diversity(
    runs: List[RunSummary],
    modes: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Aggregate runs by diversity value and (optionally) by training mode.

    Returns a DataFrame with columns:
      diversity, training_mode, n, in_mean, in_std, out_mean, out_std
    """
    if not runs:
        return pd.DataFrame(
            columns=[
                "diversity",
                "training_mode",
                "n",
                "in_mean",
                "in_std",
                "out_mean",
                "out_std",
            ]
        )

    rows: List[Dict[str, Any]] = []
    for r in runs:
        if modes is not None and r.training_mode not in modes:
            continue
        rows.append(
            {
                "diversity": r.damage_knockout_diversity,
                "training_mode": r.training_mode,
                "in": r.hard_acc_in,
                "out": r.hard_acc_out,
            }
        )

    df = pd.DataFrame(rows)
    # Drop Nones before aggregation while preserving counts
    def _agg(series: pd.Series) -> Tuple[float, float, int]:
        vals = series.dropna().astype(float)
        if len(vals) == 0:
            return float("nan"), float("nan"), 0
        return float(vals.mean()), float(vals.std(ddof=0)), int(len(vals))

    grouped = (
        df.groupby(["diversity", "training_mode"]).agg(
            in_stats=("in", _agg),
            out_stats=("out", _agg),
        )
    )

    # Expand tuples into columns
    grouped = grouped.reset_index()
    grouped[["in_mean", "in_std", "in_n"]] = pd.DataFrame(
        grouped["in_stats"].tolist(), index=grouped.index
    )
    grouped[["out_mean", "out_std", "out_n"]] = pd.DataFrame(
        grouped["out_stats"].tolist(), index=grouped.index
    )
    grouped = grouped.drop(columns=["in_stats", "out_stats"]).sort_values(
        by=["training_mode", "diversity"]
    )
    return grouped


def get_figure2_dataset(
    project: str = "boolean-nca-cc",
    entity: Optional[str] = None,
    filters: Optional[Dict[str, Any]] = None,
    modes: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    High-level helper: list runs, extract metrics, and aggregate by diversity.
    """
    runs = list_runs_grouped_by_diversity(project=project, entity=entity, filters=filters)
    return aggregate_by_diversity(runs, modes=modes)


def plot_figure2(
    df: pd.DataFrame,
    output_path: str,
    x_ticks: Optional[List[int]] = None,
    smooth_window: int = 3,
    figsize: Tuple[int, int] = (8, 6),
    dpi: int = 300,
) -> str:
    """
    Plot hard accuracy vs damage exposure for IN and OUT distributions.
    Smoothing uses a running median on the means; std bands are not smoothed.
    """
    import matplotlib.pyplot as plt

    if df.empty:
        raise ValueError("Empty dataset for Figure 2 plot")

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    for mode, mode_df in df.groupby("training_mode"):
        mode_df = mode_df.sort_values("diversity")

        def _smooth(arr: np.ndarray) -> np.ndarray:
            if smooth_window and smooth_window > 1 and len(arr) >= smooth_window:
                # Running median smoothing
                k = smooth_window
                padded = np.pad(arr, (k // 2, k - 1 - k // 2), mode="edge")
                return np.array([np.median(padded[i : i + k]) for i in range(len(arr))])
            return arr

        x = mode_df["diversity"].to_numpy()

        in_mean = mode_df["in_mean"].to_numpy(dtype=float)
        in_std = mode_df["in_std"].to_numpy(dtype=float)
        out_mean = mode_df["out_mean"].to_numpy(dtype=float)
        out_std = mode_df["out_std"].to_numpy(dtype=float)

        in_mean_s = _smooth(in_mean.copy())
        out_mean_s = _smooth(out_mean.copy())

        # Colors: IN (cyan accents), OUT (orange accents)
        ax.plot(x, in_mean_s, color="#00bcd4", marker="o", label=f"IN ({mode})")
        ax.fill_between(x, in_mean - in_std, in_mean + in_std, color="#00bcd4", alpha=0.2)

        ax.plot(x, out_mean_s, color="#ff9800", marker="s", label=f"OUT ({mode})")
        ax.fill_between(x, out_mean - out_std, out_mean + out_std, color="#ff9800", alpha=0.2)

    ax.set_xlabel("Training damage exposure (damage_knockout_diversity)")
    ax.set_ylabel("Final hard accuracy")
    ax.set_ylim(0.55, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend()

    if x_ticks is not None and len(x_ticks) > 0:
        ax.set_xticks(x_ticks)

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


