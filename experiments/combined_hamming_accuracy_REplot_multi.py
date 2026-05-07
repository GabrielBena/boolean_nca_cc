#!/usr/bin/env python3
"""
Multi-seed replot wrapper for the combined hamming + accuracy figure.

Pools summary.csv from N run output directories — one per training seed —
into a single combined DataFrame, then renders the same two-panel figure
as combined_hamming_accuracy_REplot.py (accuracy on top, hamming on bottom,
shared x-axis, optional dashed vlines for training KO size).

Usage:
    python experiments/combined_hamming_accuracy_REplot_multi.py \\
        --manifest logs/hamming_distance_sweep_20260505_194048/manifest.txt \\
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

from experiments.visualization.plot_perturbation_utils import plot_combined_hamming_accuracy


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
        description="Combine multiple runs (one per seed) and render the combined hamming+accuracy figure."
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--dirs", nargs="+",
                     help="Run output directories (each containing summary.csv). Globs are expanded.")
    src.add_argument("--manifest", type=str,
                     help="Path to a manifest file (one run dir per line). Produced by "
                          "scripts/launch_hamming_distance_sweep.sh under its LOG_DIR.")
    parser.add_argument("--output", type=str, required=True,
                        help="Output directory for combined CSV and the combined plot.")
    parser.add_argument("--width", type=float, default=8,
                        help="Figure width (default: 8)")
    parser.add_argument("--hamming-ylim-max", type=float, default=0.37,
                        help="Max y-axis for hamming panel (default: 0.37)")
    parser.add_argument("--accuracy-ylim-min", type=float, default=0.99,
                        help="Min y-axis for accuracy panel (default: 0.99)")
    parser.add_argument("--accuracy-ylim-max", type=float, default=1.01,
                        help="Max y-axis for accuracy panel (default: 1.01)")
    parser.add_argument("--height-ratio", type=str, default="1.5,1",
                        help="Gridspec bottom:top (hamming:accuracy); top height stays fixed like legacy (2,1), "
                        "figure shortens if bottom weight is reduced (default '1.5,1')")
    parser.add_argument("--no-method-colors", action="store_true",
                        help="Don't color by method (single color)")
    parser.add_argument("--baseline-accuracy", type=float, default=None,
                        help="Baseline accuracy horizontal line on accuracy panel")
    parser.add_argument("--vline", type=float, action="append", default=None,
                        help="Add vertical dashed line at this x value (repeatable)")
    parser.add_argument("--max-knockout", type=int, default=None,
                        help="Exclude knockout sizes above this value")
    parser.add_argument("--gnn-label", type=str, default=None,
                        help="Legend label override for GNN/TMT method.")
    parser.add_argument("--filename", type=str, default="combined_hamming_accuracy_REplot_multi.png",
                        help="Output filename for the figure (placed under --output).")
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

    if args.max_knockout is not None and 'knockout_size' in combined.columns:
        before = len(combined)
        combined = combined[combined['knockout_size'] <= args.max_knockout]
        print(f"Filtered knockout_size <= {args.max_knockout}: {before} → {len(combined)} rows")
    if 'knockout_size' in combined.columns:
        print(f"Knockout sizes: {sorted(combined['knockout_size'].unique())}")
    if 'method' in combined.columns:
        print(f"Methods: {combined['method'].unique().tolist()}")
    if "method" in combined.columns and "knockout_size" in combined.columns:
        counts = combined.groupby(["knockout_size", "method"]).size().unstack(fill_value=0)
        print(f"Points per (knockout_size, method):\n{counts}")

    height_ratio = tuple(float(x.strip()) for x in args.height_ratio.split(","))

    method_label_map = None
    if args.gnn_label is not None:
        method_label_map = {"gnn": args.gnn_label, "bp": "BP"}

    output_path = os.path.join(args.output, args.filename)
    plot_combined_hamming_accuracy(
        summary_df=combined,
        output_path=output_path,
        width=args.width,
        color_by_method=not args.no_method_colors,
        hamming_ylim_max=args.hamming_ylim_max,
        accuracy_ylim_min=args.accuracy_ylim_min,
        accuracy_ylim_max=args.accuracy_ylim_max,
        baseline_accuracy=args.baseline_accuracy,
        height_ratio=height_ratio,
        method_label_map=method_label_map,
        vlines=args.vline,
    )

    print(f"Combined plot saved to: {output_path}")


if __name__ == "__main__":
    main()
