#!/usr/bin/env python3
"""
Multi-seed replot wrapper for hamming_distance_plot.py outputs.

Pools summary.csv from N run output directories — one per training seed —
into a single combined DataFrame, then renders the same plots as
hamming_distance_REplot.py. With 5 seeds x 20 patterns per size, each
(damage_size, method) tick aggregates 100 points for tighter statistics.

Usage:
    python experiments/hamming_distance_REplot_multi.py \\
        --dirs reports/figures/hamming/hamming_scaled..._runap1b5iyb_* \\
               reports/figures/hamming/hamming_scaled..._run0kbp45do_* \\
               ... \\
        --output reports/figures/hamming/multi_seed_f1pktfh5
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
    plot_damage_size_vs_hamming,
    plot_accuracy_vs_distance,
)


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
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Combine multiple hamming_distance runs (one per seed) and replot."
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--dirs", nargs="+",
                     help="Run output directories (each containing summary.csv). Globs are expanded.")
    src.add_argument("--manifest", type=str,
                     help="Path to a manifest file (one run dir per line). Produced by "
                          "scripts/launch_hamming_distance_sweep.sh under its LOG_DIR. "
                          "Preferred over --dirs when older sweeps share OUT_ROOT.")
    parser.add_argument("--output", type=str, required=True,
                        help="Output directory for combined CSV and plots.")
    parser.add_argument("--ylim-max", type=float, default=0.45,
                        help="Maximum y-axis limit for hamming plot (default: 0.45)")
    parser.add_argument("--figsize", type=str, default=None,
                        help="Figure size as 'width,height' (default: 8,6).")
    parser.add_argument("--no-method-colors", action="store_true",
                        help="Don't color by method (single line plot)")
    parser.add_argument("--also-accuracy", action="store_true",
                        help="Also generate accuracy_vs_distance.png from the pooled CSV.")
    parser.add_argument("--accuracy-ylim", type=str, default=None,
                        help="Y-axis limits for accuracy plot as 'min,max'. Default: auto-scale.")
    args = parser.parse_args()

    if args.manifest is not None:
        run_dirs = _read_manifest(args.manifest)
        print(f"Manifest: {args.manifest}")
    else:
        run_dirs = _expand_dirs(args.dirs)
    print(f"Pooling {len(run_dirs)} run directories:")
    for i, d in enumerate(run_dirs):
        print(f"  [seed_idx={i}] {d}")

    summary_frames = [_load_run_dir(d, seed_idx=i) for i, d in enumerate(run_dirs)]
    combined = pd.concat(summary_frames, ignore_index=True)

    os.makedirs(args.output, exist_ok=True)
    combined_csv = os.path.join(args.output, "summary_combined.csv")
    combined.to_csv(combined_csv, index=False)
    print(f"\nCombined summary: {len(combined)} rows → {combined_csv}")
    if "method" in combined.columns and "knockout_size" in combined.columns:
        counts = combined.groupby(["knockout_size", "method"]).size().unstack(fill_value=0)
        print(f"Points per (knockout_size, method):\n{counts}")

    figsize = (8, 6)
    if args.figsize is not None:
        w, h = [float(x.strip()) for x in args.figsize.split(",")]
        figsize = (w, h)

    main_plot = os.path.join(args.output, "damage_size_vs_hamming_REplot_multi.png")
    plot_damage_size_vs_hamming(
        summary_df=combined,
        output_path=main_plot,
        color_by_method=not args.no_method_colors,
        ylim_max=args.ylim_max,
        figsize=figsize,
    )
    print(f"Wrote {main_plot}")

    if args.also_accuracy:
        if {"final_hard_accuracy", "overall_bitwise_fraction_diff"}.issubset(combined.columns):
            ylim = None
            if args.accuracy_ylim:
                parts = [float(x.strip()) for x in args.accuracy_ylim.split(",")]
                if len(parts) == 2:
                    ylim = (parts[0], parts[1])
            acc_plot = os.path.join(args.output, "accuracy_vs_distance_REplot_multi.png")
            plot_accuracy_vs_distance(
                summary_df=combined,
                output_path=acc_plot,
                color_by_method=not args.no_method_colors,
                ylim=ylim,
            )
            print(f"Wrote {acc_plot}")
        else:
            print("Warning: skipping accuracy plot — required columns missing.")


if __name__ == "__main__":
    main()
