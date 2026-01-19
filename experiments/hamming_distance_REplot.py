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

from experiments.visualization.plot_perturbation_utils import plot_damage_size_vs_hamming


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
    
    # Generate the plot
    print(f"\nGenerating plot with ylim_max={args.ylim_max}...")
    plot_damage_size_vs_hamming(
        summary_df=df,
        output_path=output_path,
        color_by_method=not args.no_method_colors,
        ylim_max=args.ylim_max
    )
    
    print(f"Plot saved to: {output_path}")


if __name__ == "__main__":
    main()
