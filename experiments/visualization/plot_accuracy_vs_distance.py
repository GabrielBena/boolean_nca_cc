"""
Accuracy vs distance visualization.

This module provides plotting functions for accuracy vs hamming distance analysis.
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
        
        # Plot GNN data
        if len(gnn_data) > 0:
            ax.scatter(gnn_data['overall_bitwise_fraction_diff'], 
                      gnn_data['final_hard_accuracy'], 
                      c='red', marker='o', s=100, alpha=0.7, 
                      edgecolors='black', linewidth=0.5, label='GNN')
        
        # Plot BP data
        if len(bp_data) > 0:
            ax.scatter(bp_data['overall_bitwise_fraction_diff'], 
                      bp_data['final_hard_accuracy'], 
                      c='blue', marker='s', s=100, alpha=0.7, 
                      edgecolors='black', linewidth=0.5, label='BP')
        
        # Add legend with font size matching Figure 3
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


def plot_damage_size_vs_hamming(
    summary_df: pd.DataFrame,
    output_path: str,
    figsize: tuple = (8, 6),
    dpi: int = 300,
    color_by_method: bool = True,
    baseline_accuracy: Optional[float] = None,
    baseline_loss: Optional[float] = None
) -> str:
    """
    Create scatter plot of damage size (knockout_size) vs hamming distance.
    Colors points by accuracy using a gradient: green (perfect) to red (poor).
    
    Args:
        summary_df: DataFrame with knockout results (must have 'knockout_size', 
                   'per_gate_mean_hamming', and 'final_hard_accuracy' columns)
        output_path: Path to save the plot image
        figsize: Figure dimensions
        dpi: Image resolution
        color_by_method: Whether to separate by method (GNN vs BP) with different markers
    
    Returns:
        Path to saved image file
    """
    required_cols = ['knockout_size', 'per_gate_mean_hamming', 'final_hard_accuracy']
    missing_cols = [col for col in required_cols if col not in summary_df.columns]
    if missing_cols:
        raise ValueError(f"DataFrame must contain columns: {missing_cols}")
    
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    # Get accuracy values for color mapping
    accuracies = summary_df['final_hard_accuracy'].values
    # Normalize accuracies to [0, 1] for colormap (assuming range is roughly [0, 1])
    # Clamp to [0, 1] to handle any values outside this range
    accuracies_clamped = np.clip(accuracies, 0, 1)
    
    # Use viridis colormap: dark purple (low accuracy) to yellow (high accuracy)
    from matplotlib.colors import Normalize
    from matplotlib.lines import Line2D
    colormap = plt.cm.get_cmap('viridis')  # Dark purple (low) -> blue -> green -> yellow (high)
    
    if color_by_method and 'method' in summary_df.columns:
        # Calculate small offset for jittering points (BP left, GNN right)
        # Use a small fraction of the x-axis range to ensure points stay near their tick
        x_range = summary_df['knockout_size'].max() - summary_df['knockout_size'].min()
        if x_range > 0:
            jitter_amount = x_range * 0.01  # 2% of range - small but visible
        else:
            # Fallback if all values are the same
            jitter_amount = 0.5
        
        # Plot each method separately with different markers but same color gradient
        for method, marker, offset in [('gnn', 'o', jitter_amount), ('bp', 's', -jitter_amount)]:
            method_data = summary_df[summary_df['method'] == method]
            if len(method_data) > 0:
                method_acc = method_data['final_hard_accuracy'].values
                method_acc_clamped = np.clip(method_acc, 0, 1)
                method_colors = colormap(method_acc_clamped)
                
                # Apply offset to x-coordinates: GNN shifts right (+), BP shifts left (-)
                x_coords = method_data['knockout_size'].values + offset
                
                scatter = ax.scatter(x_coords, 
                              method_data['per_gate_mean_hamming'],
                              c=method_acc_clamped,
                              cmap=colormap,
                              vmin=0, vmax=1,
                              marker=marker,
                              s=100,
                              alpha=0.7,
                              edgecolors='black',
                              linewidth=0.5,
                              label=method.upper())
        
        # Add colorbar for accuracy
        sm = plt.cm.ScalarMappable(cmap=colormap, norm=Normalize(vmin=0, vmax=1))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label('Final Hard Accuracy', fontsize=14)
        
        # Add method legend
        method_legend = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='black', 
                   markersize=10, label='GNN', linestyle='None'),
            Line2D([0], [0], marker='s', color='w', markerfacecolor='black', 
                   markersize=10, label='BP', linestyle='None'),
        ]
        ax.legend(handles=method_legend, loc='best', fontsize=12)
    else:
        # Single scatter plot with color gradient
        scatter = ax.scatter(summary_df['knockout_size'],
                      summary_df['per_gate_mean_hamming'],
                      c=accuracies_clamped,
                      cmap=colormap,
                      vmin=0, vmax=1,
                      s=100,
                      alpha=0.7,
                      edgecolors='black',
                      linewidth=0.5)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Final Hard Accuracy', fontsize=14)
    
    # Customize plot
    ax.set_xlabel('Damage Size (Number of Knockouts)', fontsize=18)
    ax.set_ylabel('Hamming Distance (Mean per Gate)', fontsize=18)
    ax.set_title('Hamming Distance vs Damage Size', fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    return output_path

