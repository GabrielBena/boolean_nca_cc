#!/usr/bin/env python3
"""
Plot stepwise metrics from local storage (results/metrics).

Barebones trajectory plot:
- Default: two side-by-side panels (hard_accuracy vs steps; loss vs steps).
- With --accuracy-only: hard_accuracy only, no plot title, final-value labels in dark grey.
No background shading. Supports single- and multi-damage-per-loop trajectories.

Data structure:
- stepwise_metrics.jsonl: one JSON record per line, appended per eval
- Each eval call writes N records (one per step) for ko_in and N for ko_out
- Records commonly include:
  epoch, split, run_id, step, hard_accuracy, hard_accuracy_std, hard_loss, soft_loss, eval_type, ...
- Multiple evals during training → filter by epoch (latest or specific)
"""

import argparse
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Add project root for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from boolean_nca_cc.utils.metrics_storage import load_stepwise_metrics

log = logging.getLogger(__name__)

# Colors matching Figure 3 trajectory style
COLOR_SEEN = "black"
COLOR_UNSEEN = "#34b779"
ALPHA_BAND = 0.2
COLOR_FINAL_ACC_TEXT = "#383838"


def _aggregate_series(
    df,
    steps: list[int],
    metric_col: str,
) -> list[float] | None:
    """Aggregate a metric per step if the column exists."""
    if metric_col not in df.columns:
        return None
    return [float(df[df["step"] == s][metric_col].mean()) for s in steps]


def _aggregate_std_series(
    df,
    steps: list[int],
    std_col: str,
) -> list[float] | None:
    """Aggregate a std metric per step if present and non-empty."""
    if std_col not in df.columns:
        return None
    std_vals = [df[df["step"] == s][std_col].mean() for s in steps]
    if not any(not np.isnan(v) and v is not None for v in std_vals):
        return None
    return [float(v) if not np.isnan(v) else 0.0 for v in std_vals]


def load_trajectory(
    run_id: str,
    metrics_dir: str = "results/metrics",
    sweep_id: str | None = None,
    split: str = "test",
    eval_type: str | None = None,
    epoch: int | None = None,
) -> dict[str, list[float] | list[int] | None]:
    """
    Load stepwise trajectory for a single run.

    Returns:
        Dictionary with per-step arrays:
        - steps
        - hard_accuracy
        - hard_accuracy_std (optional)
        - hard_loss (optional)
        - soft_loss (optional)
    """
    df = load_stepwise_metrics(
        run_id=run_id,
        metrics_dir=metrics_dir,
        format="jsonl",
        split=split,
        sweep_id=sweep_id,
        eval_type=eval_type,
        epoch=epoch,
    )
    if df.empty:
        return {
            "steps": [],
            "hard_accuracy": [],
            "hard_accuracy_std": None,
            "hard_loss": None,
            "soft_loss": None,
        }

    # Use latest epoch if not specified
    if epoch is None:
        epoch = int(df["epoch"].max())
        df = df[df["epoch"] == epoch]

    if df.empty:
        return {
            "steps": [],
            "hard_accuracy": [],
            "hard_accuracy_std": None,
            "hard_loss": None,
            "soft_loss": None,
        }

    steps = sorted(df["step"].unique())
    return {
        "steps": steps,
        "hard_accuracy": _aggregate_series(df, steps, "hard_accuracy") or [],
        "hard_accuracy_std": _aggregate_std_series(df, steps, "hard_accuracy_std"),
        "hard_loss": _aggregate_series(df, steps, "hard_loss"),
        "soft_loss": _aggregate_series(df, steps, "soft_loss"),
    }


def plot_stepwise_trajectory(
    run_id: str,
    metrics_dir: str = "results/metrics",
    sweep_id: str | None = None,
    split: str = "test",
    eval_type: str | None = None,
    epoch: int | None = None,
    show_seen: bool = False,
    show_unseen: bool = True,
    output_path: str | None = None,
    dpi: int = 600,
    accuracy_only: bool = False,
) -> plt.Figure:
    """
    Plot stepwise metrics with error bands (variance across damage patterns).

    Args:
        run_id: WandB run ID (or local run ID)
        metrics_dir: Base directory for metrics storage
        sweep_id: Optional sweep ID for lookup
        split: "test" or "train"
        eval_type: "ko_in", "ko_out", or None (plot both)
        epoch: Specific epoch or None (use latest)
        show_seen: Plot ko_in (seen) trajectory
        show_unseen: Plot ko_out (unseen) trajectory
        output_path: Path to save figure
        dpi: Figure resolution
        accuracy_only: If True, plot hard accuracy in a single panel (no title, no loss).

    Returns:
        matplotlib Figure
    """
    if accuracy_only:
        fig, ax_acc = plt.subplots(1, 1, figsize=(8, 3.5))
        ax_loss = None
    else:
        fig, (ax_acc, ax_loss) = plt.subplots(1, 2, figsize=(16, 3.5))
    any_loss_plotted = False
    final_acc_annotation_idx = 0

    def _plot_eval(eval_type_val: str, color: str, label: str) -> None:
        nonlocal any_loss_plotted, final_acc_annotation_idx
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
            log.warning(f"No data for eval_type={eval_type_val}")
            return
        steps_arr = np.array(steps)
        hard_acc_arr = np.array(traj["hard_accuracy"])
        ax_acc.plot(
            steps_arr,
            hard_acc_arr,
            color=color,
            linewidth=1.5,
            alpha=0.9,
            label=label,
        )
        last_s = float(steps_arr[-1])
        last_acc = float(hard_acc_arr[-1])
        ax_acc.scatter(
            [last_s],
            [last_acc],
            color=color,
            s=28,
            zorder=5,
            edgecolors="white",
            linewidths=0.6,
        )
        if accuracy_only:
            ax_acc.annotate(
                f"{last_acc:.3f}",
                xy=(last_s, last_acc),
                xytext=(-10, 6 + final_acc_annotation_idx * 14),
                textcoords="offset points",
                fontsize=11,
                color=COLOR_FINAL_ACC_TEXT,
                va="bottom",
                ha="right",
            )
        else:
            ax_acc.annotate(
                f"{last_acc:.3f}",
                xy=(last_s, last_acc),
                xytext=(8, 8 + final_acc_annotation_idx * 14),
                textcoords="offset points",
                fontsize=11,
                color=color,
                va="bottom",
                ha="left",
            )
        final_acc_annotation_idx += 1
        hard_std = traj["hard_accuracy_std"]
        if hard_std is not None and any(s > 0 for s in hard_std):
            hard_std_arr = np.array(hard_std)
            ax_acc.fill_between(
                steps_arr,
                hard_acc_arr - hard_std_arr,
                hard_acc_arr + hard_std_arr,
                color=color,
                alpha=ALPHA_BAND,
                linewidth=0,
            )

        # Prefer hard_loss for an apples-to-apples trajectory; fall back to soft_loss
        loss_vals = traj["hard_loss"] if traj["hard_loss"] is not None else traj["soft_loss"]
        if ax_loss is not None and loss_vals is not None:
            any_loss_plotted = True
            ax_loss.plot(
                steps_arr,
                np.array(loss_vals),
                color=color,
                linewidth=1.5,
                alpha=0.9,
                label=label,
            )

    if eval_type:
        if eval_type == "ko_in":
            _plot_eval("ko_in", COLOR_SEEN, "Seen")
        elif eval_type == "ko_out":
            _plot_eval("ko_out", COLOR_UNSEEN, "Unseen damage patterns")
        else:
            raise ValueError(f"eval_type must be 'ko_in' or 'ko_out', got {eval_type}")
    else:
        if show_seen:
            _plot_eval("ko_in", COLOR_SEEN, "Seen")
        if show_unseen:
            _plot_eval("ko_out", COLOR_UNSEEN, "Unseen damage patterns")

    ax_acc.set_xlabel("Message Steps", fontsize=18)
    ax_acc.set_ylabel("Hard Accuracy", fontsize=18)
    if not accuracy_only:
        ax_acc.set_title("Accuracy Trajectory", fontsize=20)
    ax_acc.tick_params(axis="both", which="major", labelsize=14)
    ax_acc.set_yticks(np.arange(0.8, 1.03, 0.05))
    ax_acc.set_ylim(0.8, 1.03)
    if ax_acc.get_legend_handles_labels()[0]:
        ax_acc.legend(loc="lower right", fontsize=13)
        ax_acc.get_legend().get_frame().set_alpha(1.0)

    if ax_loss is not None:
        ax_loss.set_xlabel("Message Steps", fontsize=18)
        ax_loss.set_ylabel("Loss", fontsize=18)
        ax_loss.set_title("Loss Trajectory", fontsize=20)
        ax_loss.tick_params(axis="both", which="major", labelsize=14)
        if any_loss_plotted and ax_loss.get_legend_handles_labels()[0]:
            ax_loss.legend(loc="upper right", fontsize=13)
            ax_loss.get_legend().get_frame().set_alpha(1.0)
        elif not any_loss_plotted:
            ax_loss.text(
                0.5,
                0.5,
                "No loss metrics found",
                transform=ax_loss.transAxes,
                ha="center",
                va="center",
                fontsize=13,
                color="gray",
            )

    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")

    return fig


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot stepwise metrics from local storage (barebones, no background shading)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--run-id",
        type=str,
        required=True,
        help="WandB run ID (or local run ID)",
    )
    parser.add_argument(
        "--sweep-id",
        type=str,
        default=None,
        help="WandB sweep ID for lookup (optional)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["test", "train"],
        help="Data split",
    )
    parser.add_argument(
        "--eval-type",
        type=str,
        default=None,
        choices=["ko_in", "ko_out"],
        help="Eval type to plot (default: use seen/unseen flags)",
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=None,
        help="Epoch to plot (default: latest)",
    )
    parser.add_argument(
        "--seen",
        dest="show_seen",
        action="store_true",
        help="Plot seen (ko_in) trajectory (off by default)",
    )
    parser.add_argument(
        "--no-seen",
        dest="show_seen",
        action="store_false",
        help="Do not plot seen (ko_in) trajectory",
    )
    parser.add_argument(
        "--no-unseen",
        dest="show_unseen",
        action="store_false",
        help="Do not plot unseen (ko_out) trajectory",
    )
    parser.set_defaults(show_seen=False, show_unseen=True)
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path (default: reports/figures/stepwise_{run_id}.png)",
    )
    parser.add_argument(
        "--metrics-dir",
        type=str,
        default="results/metrics",
        help="Base directory for metrics storage",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=600,
        help="Figure resolution",
    )
    parser.add_argument(
        "--accuracy-only",
        action="store_true",
        help="Single-panel hard accuracy plot (no title, dark grey final labels, no loss)",
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    output_path = args.output
    if output_path is None:
        suffix = "_acc_only" if args.accuracy_only else ""
        output_path = f"reports/figures/stepwise_{args.run_id}{suffix}.png"
        if args.epoch is not None:
            output_path = output_path.replace(".png", f"_epoch{args.epoch}.png")

    fig = plot_stepwise_trajectory(
        run_id=args.run_id,
        metrics_dir=args.metrics_dir,
        sweep_id=args.sweep_id,
        split=args.split,
        eval_type=args.eval_type,
        epoch=args.epoch,
        show_seen=args.show_seen,
        show_unseen=args.show_unseen,
        output_path=output_path,
        dpi=args.dpi,
        accuracy_only=args.accuracy_only,
    )
    plt.close(fig)
    log.info(f"Figure saved to {output_path}")


if __name__ == "__main__":
    main()
