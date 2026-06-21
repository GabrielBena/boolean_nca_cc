"""Fig 2 (redesigned): TMT vs BP on fixed topologies.

Reviewer fixes baked in:
  R1f - colour-blind-safe palette; TMT boxplots with BP as side-by-side mean+/-std
        bars (separate, not boxes-over-faint-bars); training regimes share a vertical
        axis for easy branch comparison.
  R1g - Mann-Whitney U + Cliff's delta (TMT vs BP, per panel/condition) -> CSV.
  R2  - TMT (not NCA) naming; "Binary Addition" (no underscore); a complete
        legend (no missing element labels).

Encoding:  colour = method (TMT vs BP);  x-axis = eval-time damage {undamaged,
damaged};  rows = damage-during-training {OFF top, ON bottom};  cols = task.
BP is the same ceiling in both rows (it re-optimises regardless of the TMT
training regime), shown for reference; both methods carry per-seed variance.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from . import stats, style, wandb_data

_HERE = os.path.dirname(__file__)
OUT_PDF = os.path.join(_HERE, "out", "fig2_fixed_wiring.pdf")
OUT_STATS = os.path.join(_HERE, "out", "fig2_stats.csv")
BP_CACHE = os.path.join(_HERE, "data", "fixed_wiring_bp.csv")

METHOD_COLORS = {"TMT": style.OKABE_ITO["blue"], "BP": style.OKABE_ITO["orange"]}


def _box(ax, x, vals, color):
    vals = np.asarray(vals, float)
    vals = vals[~np.isnan(vals)]
    if vals.size == 0:
        return
    bx = ax.boxplot([vals], positions=[x], widths=0.32, patch_artist=True,
                    medianprops=dict(color="black", linewidth=1.1),
                    flierprops=dict(marker="", alpha=0))
    bx["boxes"][0].set_facecolor(color)
    bx["boxes"][0].set_alpha(0.65)
    ax.scatter(x + np.random.default_rng(abs(int(x * 100))).uniform(-0.07, 0.07, vals.size),
               vals, s=7, color="black", alpha=0.5, zorder=3, linewidths=0)


def _bar(ax, x, vals, color):
    """BP as a mean bar with a +/-1 std error cap. BP variance is tiny, so a box
    collapses to an invisible sliver; a solid bar reads clearly next to the TMT
    box while staying side-by-side (not overlaid)."""
    vals = np.asarray(vals, float)
    vals = vals[~np.isnan(vals)]
    if vals.size == 0:
        return
    m = float(vals.mean())
    err = float(vals.std(ddof=1)) if vals.size > 1 else 0.0
    ax.bar(x, m, width=0.32, color=color, alpha=0.8, zorder=2,
           edgecolor="black", linewidth=0.6)
    ax.errorbar(x, m, yerr=err, fmt="none", ecolor="black",
                elinewidth=1.0, capsize=3, zorder=4)


def make_figure(tmt: pd.DataFrame, bp: pd.DataFrame | None):
    style.set_rc(base=7)
    tmt = tmt[(tmt["ood"] == "in") & (tmt["split"] == "test")].copy()
    train_levels = ["OFF", "ON"]
    tasks = style.TASK_ORDER
    # x layout: eval-undamaged group near 0, eval-damaged near 1; TMT left, BP right
    dx = 0.2
    xpos = {("OFF", "TMT"): 0 - dx, ("OFF", "BP"): 0 + dx,
            ("ON", "TMT"): 1 - dx, ("ON", "BP"): 1 + dx}

    fig, axes = plt.subplots(2, 3, figsize=(style.COL_WIDTH, 1.95), sharey=True)
    stat_rows = []
    for r, tr in enumerate(train_levels):
        for c, task in enumerate(tasks):
            ax = axes[r, c]
            for ev in ("OFF", "ON"):
                tmt_vals = tmt[(tmt["Task"] == task) & (tmt["train_damage"] == tr)
                               & (tmt["eval_damage"] == ev)]["hard_accuracy"].values
                _bar(ax, xpos[(ev, "TMT")], tmt_vals, METHOD_COLORS["TMT"])
                bp_vals = np.array([])
                if bp is not None:
                    bp_vals = bp[(bp["Task"] == task)
                                 & (bp["eval_damage"] == ev)]["hard_accuracy"].values
                    _bar(ax, xpos[(ev, "BP")], bp_vals, METHOD_COLORS["BP"])
                cmp = stats.compare(tmt_vals, bp_vals)
                stat_rows.append({"Task": task, "train_damage": tr, "eval_damage": ev,
                                  "comparison": "TMT_vs_BP", **cmp})
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["undam.", "dam."] if r == 1 else ["", ""])
            ax.set_xlim(-0.6, 1.6)
            ax.set_ylim(0.6, 1.03)
            ax.axhline(1.0, color="grey", lw=0.5, ls=":", zorder=0)
            if r == 0:
                ax.set_title(style.TASK_SHORT.get(task, task), fontweight="bold")
            if c == 0:
                ax.set_ylabel(f"Train {tr}")

    fig.supylabel("Final hard accuracy", fontsize=7)
    handles = [Patch(facecolor=METHOD_COLORS["TMT"], alpha=0.65, label="TMT (local policy)"),
               Patch(facecolor=METHOD_COLORS["BP"], alpha=0.8, label="BP (global ceiling, mean ± std)")]
    fig.legend(handles=handles, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig, pd.DataFrame(stat_rows)


def main():
    ap = argparse.ArgumentParser(description="Build the redesigned Fig 2 + stats.")
    ap.add_argument("--no-cache", action="store_true", help="re-pull sweep from W&B")
    a = ap.parse_args()

    tmt = wandb_data.load(use_cache=not a.no_cache)
    bp = pd.read_csv(BP_CACHE) if os.path.exists(BP_CACHE) else None
    if bp is None:
        print("[bp] no BP cache (data/fixed_wiring_bp.csv) -- run `python -m "
              "paper_figures.bp_baseline` first; plotting TMT only for now.")

    fig, stat_df = make_figure(tmt, bp)
    os.makedirs(os.path.dirname(OUT_PDF), exist_ok=True)
    fig.savefig(OUT_PDF)
    stat_df.to_csv(OUT_STATS, index=False)
    print(f"[fig] wrote {OUT_PDF}")
    print(f"[stats] wrote {OUT_STATS}")
    with pd.option_context("display.width", 160, "display.max_columns", 20):
        print(stat_df.to_string(index=False))


if __name__ == "__main__":
    main()
