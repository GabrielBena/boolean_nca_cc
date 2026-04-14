#!/usr/bin/env python3
"""
Replot combined hamming distance + accuracy vs damage size from existing summary CSV.
Produces a two-panel figure with shared x-axis for perfect horizontal alignment.
"""

import pandas as pd
import sys
import os
import argparse
import json

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from experiments.visualization.plot_perturbation_utils import plot_combined_hamming_accuracy


def main():
    parser = argparse.ArgumentParser(
        description="Replot combined hamming + accuracy vs damage size from summary CSV"
    )
    parser.add_argument("--csv", type=str, required=True,
                        help="Path to summary.csv file")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path (default: same dir as CSV, 'combined_hamming_accuracy.png')")
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

    args = parser.parse_args()

    csv_path = os.path.abspath(args.csv)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    if args.output is None:
        output_path = os.path.join(os.path.dirname(csv_path), "combined_hamming_accuracy.png")
    else:
        output_path = os.path.abspath(args.output)

    print(f"Loading CSV from: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows")
    if args.max_knockout is not None and 'knockout_size' in df.columns:
        before = len(df)
        df = df[df['knockout_size'] <= args.max_knockout]
        print(f"Filtered knockout_size <= {args.max_knockout}: {before} → {len(df)} rows")
    if 'knockout_size' in df.columns:
        print(f"Knockout sizes: {sorted(df['knockout_size'].unique())}")
    if 'method' in df.columns:
        print(f"Methods: {df['method'].unique().tolist()}")

    height_ratio = tuple(float(x.strip()) for x in args.height_ratio.split(","))

    plot_combined_hamming_accuracy(
        summary_df=df,
        output_path=output_path,
        width=args.width,
        color_by_method=not args.no_method_colors,
        hamming_ylim_max=args.hamming_ylim_max,
        accuracy_ylim_min=args.accuracy_ylim_min,
        accuracy_ylim_max=args.accuracy_ylim_max,
        baseline_accuracy=args.baseline_accuracy,
        height_ratio=height_ratio,
        vlines=args.vline,
    )

    print(f"Combined plot saved to: {output_path}")


if __name__ == "__main__":
    main()
