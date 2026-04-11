#!/usr/bin/env python3
"""
Replot accuracy vs damage size from existing summary CSV,
and optionally plot per-step accuracy trajectories from summary_traj.csv.
"""

import pandas as pd
import numpy as np
import sys
import os
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from experiments.visualization.plot_perturbation_utils import (
    plot_damage_size_vs_accuracy,
    plot_accuracy_vs_distance,
)


METHOD_COLORS = {"bp": "#1f77b4", "gnn": "#ff7f0e"}
METHOD_LABELS_DEFAULT = {"bp": "BP", "gnn": "GNN"}


def plot_accuracy_trajectories(
    traj_df: pd.DataFrame,
    output_path: str,
    method_label_map: dict = None,
    figsize: tuple = None,
    alpha: float = 0.15,
):
    """
    Plot per-step accuracy trajectories showing V-shaped damage and recovery.

    One subplot per method. Within each subplot, every (knockout_size, pattern_idx)
    trajectory is drawn as a thin line coloured by knockout_size.
    """
    labels = {**METHOD_LABELS_DEFAULT, **(method_label_map or {})}
    methods = sorted(traj_df["method"].unique(), key=lambda m: m != "gnn")
    ko_sizes = sorted(traj_df["knockout_size"].unique())

    cmap = plt.cm.viridis_r
    ko_norm = plt.Normalize(vmin=min(ko_sizes), vmax=max(ko_sizes))

    n_methods = len(methods)
    if figsize is None:
        figsize = (7 * n_methods, 4.5)
    fig, axes = plt.subplots(1, n_methods, figsize=figsize, squeeze=False)

    for ax, method in zip(axes[0], methods):
        mdf = traj_df[traj_df["method"] == method]
        for ko_size in ko_sizes:
            kdf = mdf[mdf["knockout_size"] == ko_size]
            color = cmap(ko_norm(ko_size))
            for _, grp in kdf.groupby("pattern_idx"):
                ax.plot(
                    grp["step"].values,
                    grp["hard_accuracy"].values,
                    color=color,
                    alpha=alpha,
                    linewidth=0.6,
                )

        ax.set_title(labels.get(method, method), fontsize=13)
        ax.set_xlabel("Step")
        ax.set_ylabel("Hard accuracy")
        ax.set_ylim(-0.02, 1.05)
        ax.axhline(1.0, color="grey", linewidth=0.5, linestyle="--")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=ko_norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes[0].tolist(), pad=0.02, shrink=0.85)
    cbar.set_label("Knockout size", fontsize=11)

    fig.suptitle("Accuracy trajectories (damage → recovery)", fontsize=14, y=1.02)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_accuracy_trajectories_zoomed(
    traj_df: pd.DataFrame,
    output_path: str,
    method_label_map: dict = None,
    figsize: tuple = None,
    alpha: float = 0.25,
    window: int = 30,
):
    """
    Zoomed view around the damage injection step, showing only the
    bottom of the V and the first part of recovery.
    """
    labels = {**METHOD_LABELS_DEFAULT, **(method_label_map or {})}
    methods = sorted(traj_df["method"].unique(), key=lambda m: m != "gnn")
    ko_sizes = sorted(traj_df["knockout_size"].unique())

    cmap = plt.cm.viridis_r
    ko_norm = plt.Normalize(vmin=min(ko_sizes), vmax=max(ko_sizes))

    n_methods = len(methods)
    if figsize is None:
        figsize = (7 * n_methods, 4.5)
    fig, axes = plt.subplots(1, n_methods, figsize=figsize, squeeze=False)

    for ax, method in zip(axes[0], methods):
        mdf = traj_df[traj_df["method"] == method]

        # Detect damage step: the step with the global minimum accuracy
        mean_by_step = mdf.groupby("step")["hard_accuracy"].mean()
        damage_step = int(mean_by_step.idxmin())
        step_lo = max(0, damage_step - 3)
        step_hi = damage_step + window

        mdf_zoom = mdf[(mdf["step"] >= step_lo) & (mdf["step"] <= step_hi)]

        for ko_size in ko_sizes:
            kdf = mdf_zoom[mdf_zoom["knockout_size"] == ko_size]
            color = cmap(ko_norm(ko_size))
            for _, grp in kdf.groupby("pattern_idx"):
                ax.plot(
                    grp["step"].values,
                    grp["hard_accuracy"].values,
                    color=color,
                    alpha=alpha,
                    linewidth=0.8,
                )

        ax.set_title(f"{labels.get(method, method)} (zoomed)", fontsize=13)
        ax.set_xlabel("Step")
        ax.set_ylabel("Hard accuracy")
        ax.axhline(1.0, color="grey", linewidth=0.5, linestyle="--")
        ax.axvline(damage_step, color="red", linewidth=0.7, linestyle=":", label="damage step")
        ax.legend(fontsize=8, loc="lower right")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=ko_norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes[0].tolist(), pad=0.02, shrink=0.85)
    cbar.set_label("Knockout size", fontsize=11)

    fig.suptitle("Accuracy trajectories — zoomed around damage", fontsize=14, y=1.02)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Replot accuracy vs damage size from summary CSV")
    parser.add_argument("--csv", type=str, required=True,
                        help="Path to summary.csv file")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path for plot (default: same directory as CSV with name 'damage_size_vs_accuracy_REplot.png')")
    parser.add_argument("--ylim-min", type=float, default=0.97,
                        help="Minimum y-axis limit (default: 0.97)")
    parser.add_argument("--ylim-max", type=float, default=1.01,
                        help="Maximum y-axis limit (default: 1.01)")
    parser.add_argument("--no-method-colors", action="store_true",
                        help="Don't color by method (single line plot)")
    parser.add_argument("--baseline-accuracy", type=float, default=None,
                        help="Baseline accuracy to plot as horizontal reference line (default: try to load from report.json)")
    parser.add_argument("--baseline-loss", type=float, default=None,
                        help="Baseline loss (unused, kept for compatibility)")
    parser.add_argument("--also-plot-distance", action="store_true",
                        help="Also generate accuracy_vs_distance.png plot")
    parser.add_argument("--gnn-label", type=str, default=None,
                        help="Legend label for GNN method (e.g. 'NCA'). Default: 'GNN'")
    parser.add_argument("--figsize", type=str, default=None,
                        help="Figure size as 'width,height' (e.g. '8,3'). Default: 8,6")
    parser.add_argument("--traj-csv", type=str, default=None,
                        help="Path to summary_traj.csv (default: auto-detect in same directory as --csv)")
    parser.add_argument("--traj-alpha", type=float, default=0.15,
                        help="Line transparency for trajectory plots (default: 0.15)")
    parser.add_argument("--traj-zoom-window", type=int, default=30,
                        help="Number of steps after damage to show in zoomed plot (default: 30)")
    parser.add_argument("--skip-traj", action="store_true",
                        help="Skip trajectory plots even if summary_traj.csv exists")

    args = parser.parse_args()
    
    # Load the summary CSV
    csv_path = os.path.abspath(args.csv)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    csv_dir = os.path.dirname(csv_path)

    # Set output path
    if args.output is None:
        output_path = os.path.join(csv_dir, "damage_size_vs_accuracy_REplot.png")
    else:
        output_path = os.path.abspath(args.output)
    
    print(f"Loading CSV from: {csv_path}")
    df = pd.read_csv(csv_path)
    
    print(f"Loaded {len(df)} rows")
    print(f"Columns: {df.columns.tolist()}")
    if 'knockout_size' in df.columns:
        print(f"Knockout sizes: {sorted(df['knockout_size'].unique())}")
    if 'method' in df.columns:
        print(f"Methods: {df['method'].unique().tolist()}")
    
    # Try to load baseline accuracy from report.json if not provided
    baseline_accuracy = args.baseline_accuracy
    baseline_loss = args.baseline_loss
    if baseline_accuracy is None:
        report_path = os.path.join(csv_dir, "report.json")
        if os.path.exists(report_path):
            try:
                with open(report_path, 'r') as f:
                    report = json.load(f)
                    print(f"Loaded report.json from: {report_path}")
            except Exception as e:
                print(f"Warning: Could not load report.json: {e}")
    
    # Optional method label override (e.g. GNN -> NCA)
    method_label_map = None
    if args.gnn_label is not None:
        method_label_map = {"gnn": args.gnn_label, "bp": "BP"}

    # Optional figsize override
    figsize = (8, 6)
    if args.figsize is not None:
        w, h = [float(x.strip()) for x in args.figsize.split(",")]
        figsize = (w, h)

    # Generate the main plot (damage size vs accuracy)
    print(f"\nGenerating plot with ylim=({args.ylim_min}, {args.ylim_max}), figsize={figsize}...")
    plot_damage_size_vs_accuracy(
        summary_df=df,
        output_path=output_path,
        color_by_method=not args.no_method_colors,
        baseline_accuracy=baseline_accuracy,
        baseline_loss=baseline_loss,
        ylim_min=args.ylim_min,
        ylim_max=args.ylim_max,
        method_label_map=method_label_map,
        figsize=figsize,
    )
    
    print(f"Plot saved to: {output_path}")
    
    # Optionally generate accuracy vs distance plot
    if args.also_plot_distance:
        if 'final_hard_accuracy' in df.columns and 'overall_bitwise_fraction_diff' in df.columns:
            distance_plot_path = os.path.join(csv_dir, "accuracy_vs_distance_REplot.png")
            print(f"\nGenerating accuracy vs distance plot...")
            plot_accuracy_vs_distance(
                summary_df=df,
                output_path=distance_plot_path,
                color_by_method=not args.no_method_colors
            )
            print(f"Accuracy vs distance plot saved to: {distance_plot_path}")
        else:
            print("Warning: Cannot generate accuracy vs distance plot - missing required columns")

    # ── Trajectory plots ─────────────────────────────────────────────
    if not args.skip_traj:
        traj_path = args.traj_csv
        if traj_path is None:
            candidate = os.path.join(csv_dir, "summary_traj.csv")
            if os.path.exists(candidate):
                traj_path = candidate

        if traj_path is not None and os.path.exists(traj_path):
            print(f"\nLoading trajectory CSV from: {traj_path}")
            traj_df = pd.read_csv(traj_path)
            print(f"  {len(traj_df)} rows, methods={traj_df['method'].unique().tolist()}, "
                  f"KO sizes={sorted(traj_df['knockout_size'].unique())}")

            # Full trajectory
            full_traj_path = os.path.join(csv_dir, "accuracy_trajectories_REplot.png")
            print(f"  Generating full trajectory plot...")
            plot_accuracy_trajectories(
                traj_df,
                full_traj_path,
                method_label_map=method_label_map,
                alpha=args.traj_alpha,
            )
            print(f"  Saved: {full_traj_path}")

            # Zoomed trajectory
            zoom_traj_path = os.path.join(csv_dir, "accuracy_trajectories_zoomed_REplot.png")
            print(f"  Generating zoomed trajectory plot...")
            plot_accuracy_trajectories_zoomed(
                traj_df,
                zoom_traj_path,
                method_label_map=method_label_map,
                alpha=args.traj_alpha,
                window=args.traj_zoom_window,
            )
            print(f"  Saved: {zoom_traj_path}")
        else:
            print("\nNo summary_traj.csv found — skipping trajectory plots.")


if __name__ == "__main__":
    main()
