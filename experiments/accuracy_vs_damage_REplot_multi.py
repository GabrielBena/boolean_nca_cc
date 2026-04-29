#!/usr/bin/env python3
"""
Multi-seed replot wrapper for accuracy_vs_damage_plot.py outputs.

Pools summary.csv (and summary_traj.csv when present) from N run output
directories — one per training seed — into a single combined DataFrame,
then renders the same plots as accuracy_vs_damage_REplot.py. With 5 seeds
x 20 patterns per size, the scatter has 100 points per (damage_size, method)
tick for tighter statistics.

Each seed's trajectory pattern_idx is offset (seed_idx * PATTERN_OFFSET)
so per-pattern lines stay distinct in the trajectory plots.

Usage:
    python experiments/accuracy_vs_damage_REplot_multi.py \\
        --dirs reports/figures/rev_damage_scale/accuracy_scaled..._runap1b5iyb_* \\
               reports/figures/rev_damage_scale/accuracy_scaled..._run0kbp45do_* \\
               ... \\
        --output reports/figures/rev_damage_scale/multi_seed_f1pktfh5
"""

import argparse
import glob
import os
import sys

import pandas as pd

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from experiments.visualization.plot_perturbation_utils import (
    plot_damage_size_vs_accuracy,
    plot_accuracy_vs_distance,
)
from experiments.accuracy_vs_damage_REplot import (
    plot_accuracy_trajectories,
    plot_accuracy_trajectories_zoomed,
)


PATTERN_OFFSET = 10_000  # large enough to keep seeds disjoint in pattern_idx space


def _expand_dirs(raw_dirs):
    expanded = []
    for d in raw_dirs:
        matches = sorted(glob.glob(d)) if any(c in d for c in "*?[") else [d]
        if not matches:
            raise FileNotFoundError(f"No directories matched: {d}")
        expanded.extend(matches)
    return expanded


def _read_manifest(manifest_path):
    """Read one path per line, ignoring blanks and `#` comments."""
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}")
    with open(manifest_path) as f:
        lines = [ln.strip() for ln in f]
    dirs = [ln for ln in lines if ln and not ln.startswith("#")]
    if not dirs:
        raise ValueError(f"Manifest {manifest_path} contained no directories")
    return dirs


def _load_run_dir(run_dir, seed_idx):
    summary_path = os.path.join(run_dir, "summary.csv")
    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"Missing summary.csv in {run_dir}")
    df = pd.read_csv(summary_path)
    df["seed_idx"] = seed_idx
    df["source_dir"] = run_dir

    traj_path = os.path.join(run_dir, "summary_traj.csv")
    traj_df = None
    if os.path.exists(traj_path):
        traj_df = pd.read_csv(traj_path)
        traj_df["seed_idx"] = seed_idx
        traj_df["pattern_idx"] = traj_df["pattern_idx"] + seed_idx * PATTERN_OFFSET
    return df, traj_df


def main():
    parser = argparse.ArgumentParser(
        description="Combine multiple accuracy_vs_damage runs (one per seed) and replot."
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--dirs", nargs="+",
                     help="Run output directories (each containing summary.csv [+ summary_traj.csv]). "
                          "Globs are expanded.")
    src.add_argument("--manifest", type=str,
                     help="Path to a manifest file (one run dir per line). Produced by "
                          "scripts/launch_accuracy_vs_damage_sweep.sh under its LOG_DIR. "
                          "Preferred over --dirs when older sweeps share OUT_ROOT.")
    parser.add_argument("--output", type=str, required=True,
                        help="Output directory for combined CSVs and plots.")
    parser.add_argument("--ylim-min", type=float, default=0.97)
    parser.add_argument("--ylim-max", type=float, default=1.01)
    parser.add_argument("--baseline-accuracy", type=float, default=None,
                        help="Optional horizontal reference. Defaults to mean of per-seed baselines if available.")
    parser.add_argument("--gnn-label", type=str, default=None,
                        help="Legend label override for GNN/TMT method.")
    parser.add_argument("--figsize", type=str, default=None,
                        help="Figure size as 'width,height' (default: 8,6).")
    parser.add_argument("--also-plot-distance", action="store_true")
    parser.add_argument("--skip-traj", action="store_true")
    parser.add_argument("--traj-alpha", type=float, default=0.15)
    parser.add_argument("--traj-zoom-window", type=int, default=30)
    parser.add_argument("--highlight-ko-size", type=int, default=None,
                        help="Highlight a particular knockout size in trajectory plots (e.g. training KO size).")
    args = parser.parse_args()

    if args.manifest is not None:
        run_dirs = _read_manifest(args.manifest)
        print(f"Manifest: {args.manifest}")
    else:
        run_dirs = _expand_dirs(args.dirs)
    print(f"Pooling {len(run_dirs)} run directories:")
    for i, d in enumerate(run_dirs):
        print(f"  [seed_idx={i}] {d}")

    summary_frames = []
    traj_frames = []
    for i, d in enumerate(run_dirs):
        df, traj_df = _load_run_dir(d, seed_idx=i)
        summary_frames.append(df)
        if traj_df is not None:
            traj_frames.append(traj_df)

    combined = pd.concat(summary_frames, ignore_index=True)
    combined_traj = pd.concat(traj_frames, ignore_index=True) if traj_frames else None

    os.makedirs(args.output, exist_ok=True)
    combined_csv = os.path.join(args.output, "summary_combined.csv")
    combined.to_csv(combined_csv, index=False)
    print(f"\nCombined summary: {len(combined)} rows → {combined_csv}")
    if "method" in combined.columns and "knockout_size" in combined.columns:
        counts = combined.groupby(["knockout_size", "method"]).size().unstack(fill_value=0)
        print(f"Points per (knockout_size, method):\n{counts}")

    if combined_traj is not None:
        combined_traj_csv = os.path.join(args.output, "summary_traj_combined.csv")
        combined_traj.to_csv(combined_traj_csv, index=False)
        print(f"Combined trajectories: {len(combined_traj)} rows → {combined_traj_csv}")

    method_label_map = None
    if args.gnn_label is not None:
        method_label_map = {"gnn": args.gnn_label, "bp": "BP"}

    figsize = (8, 6)
    if args.figsize is not None:
        w, h = [float(x.strip()) for x in args.figsize.split(",")]
        figsize = (w, h)

    main_plot = os.path.join(args.output, "damage_size_vs_accuracy_REplot_multi.png")
    plot_damage_size_vs_accuracy(
        summary_df=combined,
        output_path=main_plot,
        color_by_method=True,
        baseline_accuracy=args.baseline_accuracy,
        ylim_min=args.ylim_min,
        ylim_max=args.ylim_max,
        method_label_map=method_label_map,
        figsize=figsize,
    )
    print(f"Wrote {main_plot}")

    if args.also_plot_distance and {"final_hard_accuracy", "overall_bitwise_fraction_diff"}.issubset(combined.columns):
        dist_plot = os.path.join(args.output, "accuracy_vs_distance_REplot_multi.png")
        plot_accuracy_vs_distance(
            summary_df=combined,
            output_path=dist_plot,
            color_by_method=True,
        )
        print(f"Wrote {dist_plot}")

    if not args.skip_traj and combined_traj is not None:
        full_traj_path = os.path.join(args.output, "accuracy_trajectories_REplot_multi.png")
        plot_accuracy_trajectories(
            combined_traj,
            full_traj_path,
            method_label_map=method_label_map,
            alpha=args.traj_alpha,
            highlight_ko_size=args.highlight_ko_size,
        )
        print(f"Wrote {full_traj_path}")

        zoom_traj_path = os.path.join(args.output, "accuracy_trajectories_zoomed_REplot_multi.png")
        plot_accuracy_trajectories_zoomed(
            combined_traj,
            zoom_traj_path,
            method_label_map=method_label_map,
            alpha=args.traj_alpha,
            window=args.traj_zoom_window,
            highlight_ko_size=args.highlight_ko_size,
        )
        print(f"Wrote {zoom_traj_path}")


if __name__ == "__main__":
    main()
