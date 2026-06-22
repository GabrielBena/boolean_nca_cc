"""Fig 4 (resilience): TMT vs BP vs no-repair under permanent damage -- single-column.

Reads data/fig4_stepwise.csv, regenerated on isn1 against the Feb-13 checkpoint-matched
code (HEAD 741fa66 -- no drift). Columns:
    damage_kind {shotgun, stochastic}, series {TMT, BP, no_repair}, step, loss, hard_accuracy

Layout mirrors the submitted figure: columns = shotgun | stochastic, rows = Loss | Accuracy.
Colours match Fig 3 (TMT blue, BP orange); no-repair is a grey dashed lower bound.
"""
from __future__ import annotations

import os
import sys

import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, '/home/gbena/code/active/boolean-circuits/boolean_nca_cc')
from paper_figures import style
from paper_figures.style import OKABE_ITO

_HERE = os.path.dirname(__file__)
CSV = os.path.join(_HERE, "data", "fig4_stepwise.csv")
OUT = os.path.join(_HERE, "out", f"fig_resilience{style.OUT_SUFFIX}.pdf")

SERIES = ["no_repair", "BP", "TMT"]  # draw order: TMT last (on top)
COLOR = {"TMT": style.METHOD["TMT"], "BP": style.METHOD["BP"], "no_repair": "0.55"}
LS = {"TMT": "-", "BP": "-", "no_repair": "--"}
LABEL = {"TMT": "TMT", "BP": "BP", "no_repair": "No-repair"}
KIND_TITLE = {"shotgun": 'Permanent "shotgun"', "stochastic": "Stochastic failures"}


def make_figure(df):
    style.set_rc(base=7)
    fig, axes = plt.subplots(2, 2, figsize=(style.COL_WIDTH, 2.1), sharex=True)
    for ci, kind in enumerate(["shotgun", "stochastic"]):
        sub = df[df["damage_kind"] == kind]
        for ri, (metric, mlabel) in enumerate([("loss", "Loss"), ("hard_accuracy", "Accuracy")]):
            ax = axes[ri, ci]
            for s in SERIES:
                d = sub[sub["series"] == s].sort_values("step")
                if len(d):
                    ax.plot(d["step"], d[metric], color=COLOR[s], ls=LS[s], lw=1.0, label=LABEL[s])
            if ri == 0:
                ax.set_title(KIND_TITLE[kind], fontweight="bold")
                ax.set_ylim(-50, 2000)   # clip the initial spike so damage events are visible
            else:
                ax.set_xlabel("Step")
                ax.set_ylim(0.90, 1.01)
            if ci == 0:
                ax.set_ylabel(mlabel)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.07))
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return fig


if __name__ == "__main__":
    df = pd.read_csv(CSV)
    fig = make_figure(df)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT)
    print(f"[fig] wrote {OUT}")
