"""Fig 10 (scale-free): hard accuracy vs circuit size, fixed + random wiring -- single-column.

Reads data/fig10_scalefree.csv (regenerated on isn1, Feb-13 checkpoint-matched code).
Columns: wiring {fixed, random}, width_factor, n_node, damage_active {True, False},
batch_idx, hard_accuracy.

Collapses the two submitted full-width subfigures into one compact single-column figure
(fixed | random). Vertical dashed line marks the training size. Damage colours match Fig 9.
"""
from __future__ import annotations

import os

import pandas as pd
import matplotlib.pyplot as plt

from paper_figures import style
from paper_figures.style import EVAL_DAMAGE_COLORS

_HERE = os.path.dirname(__file__)
CSV = os.path.join(_HERE, "data", "fig10_scalefree.csv")
OUT = os.path.join(_HERE, "out", f"fig_scale_free{style.OUT_SUFFIX}.pdf")
TRAIN_NODES = 264
DCOLOR = {False: EVAL_DAMAGE_COLORS["OFF"], True: EVAL_DAMAGE_COLORS["ON"]}
DLABEL = {False: "Damage-OFF", True: "Damage-ON"}


def _as_bool(s):
    return s.astype(str).str.lower().isin(["true", "1"])


def make_figure(df):
    df = df.copy()
    df["damage_active"] = _as_bool(df["damage_active"])
    style.set_rc(base=7)
    fig, axes = plt.subplots(1, 2, figsize=(style.COL_WIDTH, 1.55), sharey=True)
    for ax, wiring in zip(axes, ["fixed", "random"]):
        sub = df[df["wiring"] == wiring]
        for dmg in (False, True):
            g = sub[sub["damage_active"] == dmg].groupby("n_node")["hard_accuracy"].agg(["mean", "std"])
            if not len(g):
                continue
            ax.plot(g.index, g["mean"], marker="o", ms=2.5, lw=1.0, color=DCOLOR[dmg], label=DLABEL[dmg])
            ax.fill_between(g.index, g["mean"] - g["std"], g["mean"] + g["std"],
                            color=DCOLOR[dmg], alpha=0.18, lw=0)
        ax.axvline(TRAIN_NODES, color="k", ls="--", lw=0.7)
        ax.set_title(f"{wiring.capitalize()} wiring", fontweight="bold")
        ax.set_xlabel("# nodes")
    axes[0].set_ylabel("Hard accuracy")
    handles, labels = axes[0].get_legend_handles_labels()
    handles.append(plt.Line2D([], [], color="k", ls="--", lw=0.7))
    labels.append(f"Train size ({TRAIN_NODES})")
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.13), fontsize=6)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    return fig


if __name__ == "__main__":
    df = pd.read_csv(CSV)
    fig = make_figure(df)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT)
    print(f"[fig] wrote {OUT}")
