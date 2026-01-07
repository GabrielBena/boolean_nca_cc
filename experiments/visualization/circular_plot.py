"""
Circular visualization for knockout pattern distances from baseline.

Creates a polar plot where:
- Origin (0,0) = unperturbed baseline circuit
- Radius = hamming distance (overall_bitwise_fraction_diff)
- Angle = evenly distributed around circle (pattern_idx * 2π/n_patterns)
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Optional


def plot_accuracy_vs_distance(
    summary_df: pd.DataFrame,
    output_path: str,
    figsize: tuple = (8, 3),
    dpi: int = 300,
    color_by_method: bool = True
) -> str:
    """
    Create scatter plot of final accuracy vs hamming distance.
    
    Args:
        summary_df: DataFrame with knockout results
        output_path: Path to save the plot image
        figsize: Figure dimensions
        dpi: Image resolution
        color_by_method: Whether to color points by method (GNN vs BP)
    
    Returns:
        Path to saved image file
    """
    if 'final_hard_accuracy' not in summary_df.columns or 'overall_bitwise_fraction_diff' not in summary_df.columns:
        raise ValueError("DataFrame must contain 'final_hard_accuracy' and 'overall_bitwise_fraction_diff' columns")
    
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    if color_by_method and 'method' in summary_df.columns:
        # Separate scatter plots with different colors/markers for each method
        gnn_data = summary_df[summary_df['method'] == 'gnn']
        bp_data = summary_df[summary_df['method'] == 'bp']
        
        has_gnn = len(gnn_data) > 0
        has_bp = len(bp_data) > 0
        
        # Plot GNN data
        if has_gnn:
            ax.scatter(gnn_data['overall_bitwise_fraction_diff'], 
                      gnn_data['final_hard_accuracy'], 
                      c='red', marker='o', s=100, alpha=0.7, 
                      edgecolors='black', linewidth=0.5, label='GNN')
        
        # Plot BP data
        if has_bp:
            ax.scatter(bp_data['overall_bitwise_fraction_diff'], 
                      bp_data['final_hard_accuracy'], 
                      c='blue', marker='s', s=100, alpha=0.7, 
                      edgecolors='black', linewidth=0.5, label='BP')
        
        # Only add legend if at least one method has data
        if has_gnn or has_bp:
            ax.legend(loc='lower right', fontsize=16)
    else:
        # Fallback to single color if no method column or color_by_method is False
        ax.scatter(summary_df['overall_bitwise_fraction_diff'], 
                  summary_df['final_hard_accuracy'], 
                  alpha=0.7, s=100, edgecolors='black', linewidth=0.5)
    
    # Customize plot with font sizes matching Figure 3
    ax.set_xlabel('Hamming Distance from Baseline (Fraction)', fontsize=18)
    ax.set_ylabel('Final Hard Accuracy', fontsize=18)
    ax.set_title('Circuit Performance vs Perturbation Response', fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.6, 1.02)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    return output_path

