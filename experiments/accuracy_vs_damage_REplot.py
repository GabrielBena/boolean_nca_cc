#!/usr/bin/env python3
"""
Replot accuracy vs damage size from existing summary CSV.
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

from experiments.visualization.plot_perturbation_utils import (
    plot_damage_size_vs_accuracy,
    plot_accuracy_vs_distance,
)


def main():
    parser = argparse.ArgumentParser(description="Replot accuracy vs damage size from summary CSV")
    parser.add_argument("--csv", type=str, required=True,
                        help="Path to summary.csv file")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path for plot (default: same directory as CSV with name 'damage_size_vs_accuracy_REplot.png')")
    parser.add_argument("--ylim-min", type=float, default=0.5,
                        help="Minimum y-axis limit (default: 0.5)")
    parser.add_argument("--no-method-colors", action="store_true",
                        help="Don't color by method (single line plot)")
    parser.add_argument("--baseline-accuracy", type=float, default=None,
                        help="Baseline accuracy to plot as horizontal reference line (default: try to load from report.json)")
    parser.add_argument("--baseline-loss", type=float, default=None,
                        help="Baseline loss (unused, kept for compatibility)")
    parser.add_argument("--also-plot-distance", action="store_true",
                        help="Also generate accuracy_vs_distance.png plot")
    
    args = parser.parse_args()
    
    # Load the summary CSV
    csv_path = os.path.abspath(args.csv)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    # Set output path
    if args.output is None:
        csv_dir = os.path.dirname(csv_path)
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
        csv_dir = os.path.dirname(csv_path)
        report_path = os.path.join(csv_dir, "report.json")
        if os.path.exists(report_path):
            try:
                with open(report_path, 'r') as f:
                    report = json.load(f)
                    # Report doesn't typically have baseline_accuracy, but we can try
                    # In practice, baseline accuracy might need to be computed or provided
                    print(f"Loaded report.json from: {report_path}")
            except Exception as e:
                print(f"Warning: Could not load report.json: {e}")
    
    # Generate the main plot (damage size vs accuracy)
    print(f"\nGenerating plot with ylim_min={args.ylim_min}...")
    plot_damage_size_vs_accuracy(
        summary_df=df,
        output_path=output_path,
        color_by_method=not args.no_method_colors,
        baseline_accuracy=baseline_accuracy,
        baseline_loss=baseline_loss,
        ylim_min=args.ylim_min
    )
    
    print(f"Plot saved to: {output_path}")
    
    # Optionally generate accuracy vs distance plot
    if args.also_plot_distance:
        if 'final_hard_accuracy' in df.columns and 'overall_bitwise_fraction_diff' in df.columns:
            csv_dir = os.path.dirname(csv_path)
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


if __name__ == "__main__":
    main()
