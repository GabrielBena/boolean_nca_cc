#!/usr/bin/env python3
"""
CLI script to create combined growth/regeneration plots from WandB runs.

Usage:
    python create_combined_plot.py --growth-run-id abc123 --repair-run-id def456
"""

import argparse
import sys
import os

# Add the project root to the path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from experiments.visualization.figure1_training_progress import create_combined_growth_regeneration_plot


def main():
    parser = argparse.ArgumentParser(description="Create combined growth/regeneration plot from WandB runs")
    
    parser.add_argument(
        "--growth-run-id", 
        required=True,
        help="WandB run ID for growth mode training"
    )
    
    parser.add_argument(
        "--repair-run-id", 
        required=True,
        help="WandB run ID for repair mode training"
    )
    
    parser.add_argument(
        "--project",
        default="boolean-nca-cc",
        help="WandB project name (default: boolean-nca-cc)"
    )
    
    parser.add_argument(
        "--output-dir",
        default=None,  # Will be set to project root reports/figures
        help="Output directory for the plot (default: project_root/reports/figures)"
    )
    
    parser.add_argument(
        "--filename",
        default="combined_growth_regeneration.png",
        help="Output filename (default: combined_growth_regeneration.png)"
    )
    
    args = parser.parse_args()
    
    try:
        # Set default output directory to project root reports/figures
        if args.output_dir is None:
            args.output_dir = os.path.join(project_root, "reports", "figures")
        
        print(f"Creating combined plot from:")
        print(f"  Growth run: {args.growth_run_id}")
        print(f"  Repair run: {args.repair_run_id}")
        print(f"  Project: {args.project}")
        print(f"  Output: {args.output_dir}")
        
        output_path = create_combined_growth_regeneration_plot(
            growth_run_id=args.growth_run_id,
            repair_run_id=args.repair_run_id,
            project=args.project,
            output_dir=args.output_dir,
            filename=args.filename
        )
        
        print(f"✓ Combined plot saved to: {output_path}")
        
    except Exception as e:
        print(f"✗ Error creating combined plot: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
