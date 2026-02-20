#!/usr/bin/env python3
"""
Replot hamming distance vs damage size from existing summary CSV.
"""

import pandas as pd
import sys
import os
import argparse

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from experiments.visualization.plot_perturbation_utils import (
    plot_damage_size_vs_hamming,
    plot_accuracy_vs_distance,
)


def main():
    parser = argparse.ArgumentParser(description="Replot hamming distance vs damage size from summary CSV")
    parser.add_argument("--csv", type=str, required=True,
                        help="Path to summary.csv file")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path for plot (default: same directory as CSV with name 'damage_size_vs_hamming_REplot.png')")
    parser.add_argument("--ylim-max", type=float, default=0.45,
                        help="Maximum y-axis limit (default: 0.45)")
    parser.add_argument("--no-method-colors", action="store_true",
                        help="Don't color by method (single line plot)")
    parser.add_argument("--also-accuracy", action="store_true",
                        help="Also generate accuracy_vs_distance.png from the same CSV")
    parser.add_argument("--accuracy-ylim", type=str, default=None,
                        help="Y-axis limits for accuracy plot as 'min,max' (e.g. '0.6,1.02'). Default: auto-scale (no fixed ylim)")
    
    args = parser.parse_args()
    
    # Load the summary CSV
    csv_path = os.path.abspath(args.csv)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    # Set output path
    if args.output is None:
        csv_dir = os.path.dirname(csv_path)
        output_path = os.path.join(csv_dir, "damage_size_vs_hamming_REplot.png")
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
    
    # Generate the damage size vs hamming plot
    print(f"\nGenerating plot with ylim_max={args.ylim_max}...")
    plot_damage_size_vs_hamming(
        summary_df=df,
        output_path=output_path,
        color_by_method=not args.no_method_colors,
        ylim_max=args.ylim_max
    )
    
    print(f"Plot saved to: {output_path}")

    # Optionally generate accuracy vs distance plot (reusing same CSV)
    if args.also_accuracy:
        if "final_hard_accuracy" in df.columns and "overall_bitwise_fraction_diff" in df.columns:
            csv_dir = os.path.dirname(csv_path)
            accuracy_plot_path = os.path.join(csv_dir, "accuracy_vs_distance.png")
            ylim = None
            if args.accuracy_ylim:
                parts = [float(x.strip()) for x in args.accuracy_ylim.split(",")]
                if len(parts) == 2:
                    ylim = (parts[0], parts[1])
            print(f"\nGenerating accuracy vs distance plot (ylim={'auto' if ylim is None else ylim})...")
            plot_accuracy_vs_distance(
                summary_df=df,
                output_path=accuracy_plot_path,
                color_by_method=not args.no_method_colors,
                ylim=ylim,
            )
            print(f"Accuracy vs distance plot saved to: {accuracy_plot_path}")
        else:
            print("Warning: Cannot generate accuracy vs distance plot - missing 'final_hard_accuracy' or 'overall_bitwise_fraction_diff' columns")


if __name__ == "__main__":
    main()
