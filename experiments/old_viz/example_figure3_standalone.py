#!/usr/bin/env python3
"""
Example usage of standalone Figure 3 generation.

This script demonstrates how to use the standalone Figure 3 plotter to recreate
damage recovery trajectory plots from completed WandB runs without rerunning training.
"""

import sys
import os
import logging

# Add the project root to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from experiments.visualization.figure3_standalone import create_figure3_from_wandb_run

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)


def main():
    """Example usage of standalone Figure 3 generation."""
    
    # Example WandB run IDs (replace with your actual run IDs)
    sa_run_id = "your_sa_run_id_here"  # Replace with actual SA training run ID
    bp_run_id = "your_bp_run_id_here"  # Replace with actual BP run ID (optional)
    
    # Output directory
    output_dir = "reports/figures"
    
    try:
        # Create Figure 3 from WandB run data
        log.info(f"Creating Figure 3 from SA run: {sa_run_id}")
        
        output_path = create_figure3_from_wandb_run(
            run_id=sa_run_id,
            bp_run_id=bp_run_id,  # Optional BP reference
            output_dir=output_dir,
            project="boolean-nca-cc"  # Adjust project name if needed
        )
        
        log.info(f"Successfully created Figure 3: {output_path}")
        
        # You can also create multiple plots for comparison
        # output_path2 = create_figure3_from_wandb_run(
        #     run_id="another_run_id",
        #     output_dir=output_dir
        # )
        
    except Exception as e:
        log.error(f"Failed to create Figure 3: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    print("Example Figure 3 Standalone Generation")
    print("=====================================")
    print()
    print("Before running this script:")
    print("1. Replace 'your_sa_run_id_here' with an actual WandB run ID")
    print("2. Optionally replace 'your_bp_run_id_here' with a BP run ID")
    print("3. Ensure you have wandb installed: pip install wandb")
    print("4. Make sure you're logged into WandB: wandb login")
    print()
    
    # Uncomment the line below to run the example
    # exit(main())
    
    print("To run this example, uncomment the 'exit(main())' line at the bottom of this file.")
