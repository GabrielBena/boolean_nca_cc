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


def plot_circular_knockout_distances(
    summary_df: pd.DataFrame, 
    output_path: str,
    figsize: tuple = (10, 10),
    dpi: int = 300,
    max_distance_rings: int = 5,
    show_pattern_labels: bool = True,
    color_by_accuracy: bool = True
) -> str:
    """
    Create circular plot of knockout pattern distances from baseline.
    
    Args:
        summary_df: DataFrame with knockout results (must have 'overall_bitwise_fraction_diff')
        output_path: Path to save the plot image
        figsize: Figure dimensions (width, height)
        dpi: Image resolution
        max_distance_rings: Number of concentric distance rings to show
        show_pattern_labels: Whether to show pattern index labels
        color_by_accuracy: Whether to color points by final_accuracy (if available)
    
    Returns:
        Path to saved image file
    """
    # Extract data
    hamming_distances = summary_df['overall_bitwise_fraction_diff'].values
    n_patterns = len(hamming_distances)
    
    if n_patterns == 0:
        raise ValueError("No patterns found in summary DataFrame")
    
    # Sort patterns by hamming distance (closest to baseline first)
    sorted_indices = np.argsort(hamming_distances)
    sorted_distances = hamming_distances[sorted_indices]
    
    # Calculate polar coordinates (ordered by distance)
    angles = np.linspace(0, 2*np.pi, n_patterns, endpoint=False)
    radii = sorted_distances
    
    # Create figure with polar projection
    fig, ax = plt.subplots(1, 1, figsize=figsize, subplot_kw={'projection': 'polar'})
    
    # Plot distance rings (concentric circles)
    max_radius = np.max(radii) if len(radii) > 0 else 1.0
    if max_radius > 0:
        ring_radii = np.linspace(0, max_radius, max_distance_rings + 1)
        for r in ring_radii:
            ax.plot(np.linspace(0, 2*np.pi, 100), [r]*100, 
                   color='lightgray', alpha=0.5, linewidth=0.5)
    
    # Plot pattern points
    if color_by_accuracy and 'final_accuracy' in summary_df.columns:
        accuracies = summary_df['final_accuracy'].values
        scatter = ax.scatter(angles, radii, c=accuracies, 
                           cmap='viridis', s=100, alpha=0.8, edgecolors='black', linewidth=0.5)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.8, pad=0.1)
        cbar.set_label('Final Accuracy', rotation=270, labelpad=15)
    else:
        ax.scatter(angles, radii, c='red', s=100, alpha=0.8, edgecolors='black', linewidth=0.5)
    
    # Add radial spokes from center to each pattern
    for angle, radius in zip(angles, radii):
        ax.plot([0, angle], [0, radius], color='gray', alpha=0.3, linewidth=0.5)
    
    # Add pattern labels
    if show_pattern_labels:
        # Removed pattern index labels - they are meaningless
        pass
    
    # Customize plot
    ax.set_title('Knockout Pattern Distances from Baseline\n(radius = hamming distance)', 
                pad=20, fontsize=12)
    ax.set_rlabel_position(0)
    ax.grid(True, alpha=0.3)
    
    # Set radial axis limits and labels
    ax.set_ylim(0, max_radius * 1.2)
    ax.set_rticks(ring_radii[1:])  # Skip 0
    ax.set_yticklabels([f'{r:.2f}' for r in ring_radii[1:]], fontsize=12, fontweight='bold')
    ax.set_ylabel('Hamming Distance (bitwise fraction diff)')
    
    # Set angular axis - remove pattern index labels
    ax.set_xticks(angles)
    ax.set_xticklabels([])  # No labels on angular axis
    ax.set_xlabel('Pattern Position (ordered by distance)')
    
    # Add center point (baseline)
    ax.scatter([0], [0], c='blue', s=200, marker='o', edgecolors='black', linewidth=2, zorder=5)
    ax.text(0, -max_radius * 0.1, 'Baseline\n(Unperturbed)', 
           ha='center', va='center', fontsize=10, fontweight='bold', color='blue')
    
    # Tight layout and save
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    return output_path


def plot_accuracy_vs_distance(
    summary_df: pd.DataFrame,
    output_path: str,
    figsize: tuple = (8, 6),
    dpi: int = 300
) -> str:
    """
    Create scatter plot of final accuracy vs hamming distance.
    
    Args:
        summary_df: DataFrame with knockout results
        output_path: Path to save the plot image
        figsize: Figure dimensions
        dpi: Image resolution
    
    Returns:
        Path to saved image file
    """
    if 'final_accuracy' not in summary_df.columns or 'overall_bitwise_fraction_diff' not in summary_df.columns:
        raise ValueError("DataFrame must contain 'final_accuracy' and 'overall_bitwise_fraction_diff' columns")
    
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    # Scatter plot
    ax.scatter(summary_df['overall_bitwise_fraction_diff'], 
              summary_df['final_accuracy'], 
              alpha=0.7, s=100, edgecolors='black', linewidth=0.5)
    
    # Removed pattern index labels - they are meaningless
    
    # Customize plot
    ax.set_xlabel('Hamming Distance from Baseline')
    ax.set_ylabel('Final Accuracy')
    ax.set_title('Circuit Performance vs Structural Change')
    ax.grid(True, alpha=0.3)
    
    # Add baseline point at (0, 1) if available
    if 'final_accuracy' in summary_df.columns:
        baseline_acc = 1.0  # Assuming baseline achieves perfect accuracy
        ax.scatter([0], [baseline_acc], c='blue', s=200, marker='o', 
                  edgecolors='black', linewidth=2, zorder=5)
        ax.annotate('Baseline', (0, baseline_acc), xytext=(10, 10), 
                   textcoords='offset points', fontsize=10, fontweight='bold', color='blue')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    return output_path
