"""PCA trajectories of circuit optimisation -- ported from trained_models.ipynb
(Gabriel's own run 6mo8q61y; see eval_pca_trajectories.py for the data-gen honesty
note on this being a best-effort reconstruction of the published figure).

Reads data/pca_trajectories.csv (condition, batch, step, PC1, PC2). Three panels:
none / recoverable-shotgun / permanent-shotgun, each showing one line per circuit's
LUT-logit trajectory through the jointly-fit 2D PCA space, coloured by step.
"""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from paper_figures import style

_HERE = os.path.dirname(__file__)
CSV = os.path.join(_HERE, "data", "pca_trajectories.csv")
OUT = os.path.join(_HERE, "out", f"fig_pca_trajectories{style.OUT_SUFFIX}.pdf")

CONDITIONS = ["none", "recoverable", "permanent"]
TITLE = {"none": "No damage", "recoverable": "Recoverable damage", "permanent": "Permanent damage"}


def make_figure(df: pd.DataFrame):
    style.set_rc(base=7)
    fig, axes = plt.subplots(1, 3, figsize=(style.TEXT_WIDTH, 2.3), sharex=True, sharey=True)
    n_steps = df["step"].max()
    norm = Normalize(vmin=0, vmax=n_steps)
    cmap = plt.get_cmap("viridis")

    for ax, cond in zip(axes, CONDITIONS, strict=True):
        sub = df[df["condition"] == cond]
        for _, traj in sub.groupby("batch"):
            traj = traj.sort_values("step")
            points = traj[["PC1", "PC2"]].values
            for i in range(len(points) - 1):
                ax.plot(
                    points[i : i + 2, 0], points[i : i + 2, 1],
                    color=cmap(norm(traj["step"].values[i])), lw=0.6, alpha=0.7,
                )
            ax.scatter(*points[0], color="black", s=10, marker="o", zorder=5)
            ax.scatter(*points[-1], color="black", s=14, marker="X", zorder=5)
        ax.set_title(TITLE[cond], fontweight="bold")
        ax.set_xlabel("PC1")
    axes[0].set_ylabel("PC2")

    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, fraction=0.03, pad=0.02)
    cbar.set_label("Step")
    return fig


if __name__ == "__main__":
    df = pd.read_csv(CSV)
    fig = make_figure(df)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT)
    print(f"[fig] wrote {OUT}")
