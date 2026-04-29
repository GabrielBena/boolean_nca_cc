#!/usr/bin/env python3
"""
Combined three-panel figure:
  Left column  – accuracy vs damage size (top) and hamming distance vs damage size (bottom)
  Right column – stepwise hard-accuracy trajectory (spans both rows)

Reuses the on-axes helpers from plot_perturbation_utils and the trajectory
loader from plot_stepwise_metrics.
"""

import argparse
import logging
import os
import sys

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from experiments.visualization.plot_perturbation_utils import (
    _plot_damage_size_vs_accuracy_on_ax,
    _plot_damage_size_vs_hamming_on_ax,
)
from experiments.visualization.plot_stepwise_metrics import (
    COLOR_FINAL_ACC_TEXT,
    COLOR_UNSEEN,
    ALPHA_BAND,
    load_trajectory,
)

log = logging.getLogger(__name__)


def _plot_stepwise_accuracy_on_ax(
    ax,
    run_id: str,
    metrics_dir: str = "results/metrics",
    sweep_id: str | None = None,
    split: str = "test",
    epoch: int | None = None,
    show_seen: bool = False,
    show_unseen: bool = True,
) -> None:
    """Render the stepwise hard-accuracy trajectory onto *ax*."""
    annotation_idx = 0

    def _plot_eval(eval_type_val: str, color: str, label: str) -> None:
        nonlocal annotation_idx
        traj = load_trajectory(
            run_id=run_id,
            metrics_dir=metrics_dir,
            sweep_id=sweep_id,
            split=split,
            eval_type=eval_type_val,
            epoch=epoch,
        )
        steps = traj["steps"]
        if not steps:
            log.warning("No data for eval_type=%s", eval_type_val)
            return
        steps_arr = np.array(steps)
        acc_arr = np.array(traj["hard_accuracy"])

        ax.plot(steps_arr, acc_arr, color=color, linewidth=1.5, alpha=0.9, label=label)
        last_s, last_acc = float(steps_arr[-1]), float(acc_arr[-1])
        ax.scatter(
            [last_s], [last_acc], color=color, s=28, zorder=5,
            edgecolors="white", linewidths=0.6,
        )
        ax.annotate(
            f"{last_acc:.3f}",
            xy=(last_s, last_acc),
            xytext=(-10, 6 + annotation_idx * 14),
            textcoords="offset points",
            fontsize=11, color=COLOR_FINAL_ACC_TEXT, va="bottom", ha="right",
        )
        annotation_idx += 1

        hard_std = traj["hard_accuracy_std"]
        if hard_std is not None and any(s > 0 for s in hard_std):
            std_arr = np.array(hard_std)
            ax.fill_between(
                steps_arr, acc_arr - std_arr, acc_arr + std_arr,
                color=color, alpha=ALPHA_BAND, linewidth=0,
            )

    if show_seen:
        _plot_eval("ko_in", "black", "Seen")
    if show_unseen:
        _plot_eval("ko_out", COLOR_UNSEEN, "Unseen damage patterns")

    ax.set_xlabel("Message Steps", fontsize=18)
    ax.set_ylabel("Hard Accuracy", fontsize=18)
    ax.tick_params(axis="both", which="major", labelsize=14)
    ax.set_yticks(np.arange(0.8, 1.03, 0.05))
    ax.set_ylim(0.8, 1.03)
    if ax.get_legend_handles_labels()[0]:
        leg = ax.legend(loc="lower right", fontsize=13)
        leg.get_frame().set_alpha(1.0)
        leg.get_frame().set_facecolor("white")


def plot_combined_damage_stepwise(
    summary_df: pd.DataFrame,
    run_id: str,
    output_path: str,
    *,
    width: float = 16,
    dpi: int = 600,
    color_by_method: bool = True,
    hamming_ylim_max: float | None = 0.37,
    accuracy_ylim_min: float | None = 0.985,
    accuracy_ylim_max: float | None = 1.01,
    baseline_accuracy: float | None = None,
    height_ratio: tuple = (1.5, 1),
    vlines: list | None = None,
    metrics_dir: str = "results/metrics",
    sweep_id: str | None = None,
    split: str = "test",
    epoch: int | None = None,
    show_seen: bool = False,
    show_unseen: bool = True,
    max_knockout: int | None = None,
) -> str:
    """Build and save the 3-panel figure.  Returns *output_path*."""
    if max_knockout is not None and "knockout_size" in summary_df.columns:
        summary_df = summary_df[summary_df["knockout_size"] <= max_knockout]

    reversed_ratio = (height_ratio[1], height_ratio[0])

    fig = plt.figure(figsize=(width, 6.5))
    gs = GridSpec(
        2, 2,
        figure=fig,
        width_ratios=[1, 1],
        height_ratios=list(reversed_ratio),
        hspace=0.12,
        wspace=0.30,
    )

    ax_acc = fig.add_subplot(gs[0, 0])
    ax_ham = fig.add_subplot(gs[1, 0], sharex=ax_acc)
    ax_step = fig.add_subplot(gs[:, 1])

    # --- left column: damage-size panels ---
    _plot_damage_size_vs_accuracy_on_ax(
        ax_acc, summary_df, color_by_method,
        baseline_accuracy=baseline_accuracy,
        ylim_min=accuracy_ylim_min, ylim_max=accuracy_ylim_max,
        show_xlabel=False,
    )
    _plot_damage_size_vs_hamming_on_ax(
        ax_ham, summary_df, color_by_method,
        ylim_max=hamming_ylim_max, show_xlabel=True,
    )
    if ax_ham.get_legend() is not None:
        ax_ham.get_legend().remove()
    if ax_acc.get_legend() is not None:
        ax_acc.get_legend().get_frame().set_alpha(1.0)
        ax_acc.get_legend().get_frame().set_facecolor("white")

    if vlines:
        for i, vx in enumerate(vlines):
            label = "Training Size" if i == 0 else None
            for ax in (ax_acc, ax_ham):
                ax.axvline(
                    vx, color="#404040", linewidth=1.8, linestyle="--",
                    alpha=0.7, label=label if ax is ax_acc else None,
                )
        handles, labels = ax_acc.get_legend_handles_labels()
        if "Training Size" in labels:
            vi = labels.index("Training Size")
            handles = [handles[vi]] + handles[:vi] + handles[vi + 1:]
            labels = [labels[vi]] + labels[:vi] + labels[vi + 1:]
        leg = ax_acc.legend(handles, labels, loc="best", fontsize=16, ncol=len(labels))
        leg.get_frame().set_alpha(1.0)
        leg.get_frame().set_facecolor("white")

    ax_ham.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax_acc.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    plt.setp(ax_acc.get_xticklabels(), visible=False)

    # --- right column: stepwise trajectory ---
    _plot_stepwise_accuracy_on_ax(
        ax_step, run_id,
        metrics_dir=metrics_dir, sweep_id=sweep_id,
        split=split, epoch=epoch,
        show_seen=show_seen, show_unseen=show_unseen,
    )

    # Subplot labels – A/B sit higher so the left column visually
    # matches the height of the full-span C panel.
    for label_txt, ax, y_pos in [("A", ax_acc, 1.22), ("B", ax_ham, 1.14), ("C", ax_step, 1.10)]:
        ax.text(
            -0.08, y_pos, label_txt,
            transform=ax.transAxes, fontsize=20, fontweight="bold",
            va="top", ha="right",
        )

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", pad_inches=0.15)

    pdf_path = os.path.splitext(output_path)[0] + ".pdf"
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.15)

    plt.close(fig)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Combined damage-size + stepwise-trajectory three-panel figure",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--csv", type=str, required=True,
                        help="Path to summary.csv (damage-size data)")
    parser.add_argument("--run-id", type=str, required=True,
                        help="Run ID for stepwise trajectory data")
    parser.add_argument("--output", type=str, default=None,
                        help="Output image path")
    parser.add_argument("--width", type=float, default=16,
                        help="Figure width in inches")
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--hamming-ylim-max", type=float, default=0.37)
    parser.add_argument("--accuracy-ylim-min", type=float, default=0.985)
    parser.add_argument("--accuracy-ylim-max", type=float, default=1.01)
    parser.add_argument("--height-ratio", type=str, default="1.5,1",
                        help="Gridspec bottom:top weight (hamming:accuracy)")
    parser.add_argument("--baseline-accuracy", type=float, default=None)
    parser.add_argument("--vline", type=float, action="append", default=None,
                        help="Vertical dashed line at this x (repeatable)")
    parser.add_argument("--max-knockout", type=int, default=None)
    parser.add_argument("--metrics-dir", type=str, default="results/metrics")
    parser.add_argument("--sweep-id", type=str, default=None)
    parser.add_argument("--split", type=str, default="test", choices=["test", "train"])
    parser.add_argument("--epoch", type=int, default=None)
    parser.add_argument("--seen", dest="show_seen", action="store_true")
    parser.add_argument("--no-unseen", dest="show_unseen", action="store_false")
    parser.set_defaults(show_seen=False, show_unseen=True)

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    csv_path = os.path.abspath(args.csv)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    output_path = args.output
    if output_path is None:
        output_path = os.path.join(
            os.path.dirname(csv_path), "combined_damage_stepwise.png"
        )

    df = pd.read_csv(csv_path)
    log.info("Loaded %d rows from %s", len(df), csv_path)

    height_ratio = tuple(float(x.strip()) for x in args.height_ratio.split(","))

    result = plot_combined_damage_stepwise(
        summary_df=df,
        run_id=args.run_id,
        output_path=output_path,
        width=args.width,
        dpi=args.dpi,
        color_by_method=True,
        hamming_ylim_max=args.hamming_ylim_max,
        accuracy_ylim_min=args.accuracy_ylim_min,
        accuracy_ylim_max=args.accuracy_ylim_max,
        baseline_accuracy=args.baseline_accuracy,
        height_ratio=height_ratio,
        vlines=args.vline,
        metrics_dir=args.metrics_dir,
        sweep_id=args.sweep_id,
        split=args.split,
        epoch=args.epoch,
        show_seen=args.show_seen,
        show_unseen=args.show_unseen,
        max_knockout=args.max_knockout,
    )
    log.info("Figure saved to %s", result)


if __name__ == "__main__":
    main()
