"""Random-topology companion to Fig 2 (TMT only; BP is wiring-agnostic -- see the
fixed-topology figure, per the caption note).

Data: group `random_wiring_sweep`, config-matched to the paper (the same filter the
uncommitted boolean_nca_exploration.ipynb used). This sweep is thin -- single seed,
damage-trained only, logging *out-of-distribution* eval -- so bars are point values
with no per-seed error. Undamaged = eval_out_test, damaged = eval_damaged_out_test.

Output: out/fig_random_wiring.pdf
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from paper_figures import style
from paper_figures.style import EVAL_DAMAGE_COLORS, TASK_MAP, TASK_ORDER
from paper_figures.wandb_data import DEFAULT_ENTITY, DEFAULT_PROJECT, _args, _cfg, _flat

_HERE = os.path.dirname(__file__)
OUT_PDF = os.path.join(_HERE, "out", f"fig_random_wiring{style.OUT_SUFFIX}.pdf")
GROUP = "random_wiring_sweep"
FILTERS = {
    "group": GROUP,
    "config.model.type": "gathered_attention",
    "config.circuit.width_factor": 2,
    "config.circuit.circuit_hidden_dim": 64,
    "config.damage.target_damage_fraction": 0.1,
}
UNDAM = "eval_out_test/final_hard_accuracy"
DAM = "eval_damaged_out_test/final_hard_accuracy"


def _f(x):
    try:
        return float(x)
    except Exception:
        return None


def _get(run, key):
    sm = dict(run.summary)
    if key in sm and _f(sm[key]) is not None:
        return _f(sm[key])
    node = sm
    for p in key.split("/"):
        node = node[p] if isinstance(node, dict) and p in node else None
        if node is None:
            break
    return _f(node) if _f(node) is not None else np.nan


def pull():
    from wandb import Api
    api = Api()
    runs = api.runs(f"{DEFAULT_ENTITY}/{DEFAULT_PROJECT}", filters=FILTERS)
    print(f"[random] {len(runs)} config-matched runs")
    rows = []
    for r in runs:
        f = _flat(dict(r.config))
        task = _cfg(f, "circuit.task", "task") or _args(r).get("circuit")
        rows.append({"Task": TASK_MAP.get(str(task), str(task)),
                     "OFF": _get(r, UNDAM), "ON": _get(r, DAM)})
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    return df


def make_figure(df):
    style.set_rc(base=7)
    fig, axes = plt.subplots(1, 3, figsize=(style.COL_WIDTH, 1.35), sharey=True)
    for ax, task in zip(axes, TASK_ORDER):
        sub = df[df["Task"] == task]
        for i, ev in enumerate(("OFF", "ON")):
            vals = sub[ev].dropna().values
            if len(vals):
                ax.bar(i, float(np.mean(vals)), width=0.6, color=EVAL_DAMAGE_COLORS[ev],
                       alpha=0.85, edgecolor="black", linewidth=0.6)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["undam.", "dam."])
        ax.set_xlim(-0.7, 1.7)
        ax.set_ylim(0.5, 1.05)
        ax.axhline(1.0, color="grey", lw=0.5, ls=":", zorder=0)
        ax.set_title(style.TASK_SHORT.get(task, task), fontweight="bold")
    axes[0].set_ylabel("Final hard accuracy")
    handles = [Patch(facecolor=EVAL_DAMAGE_COLORS["OFF"], alpha=0.85, label="Undamaged eval"),
               Patch(facecolor=EVAL_DAMAGE_COLORS["ON"], alpha=0.85, label="Damaged eval")]
    fig.legend(handles=handles, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.10))
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    return fig


if __name__ == "__main__":
    df = pull()
    fig = make_figure(df)
    os.makedirs(os.path.dirname(OUT_PDF), exist_ok=True)
    fig.savefig(OUT_PDF)
    print(f"[fig] wrote {OUT_PDF}")
