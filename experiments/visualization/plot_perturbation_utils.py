"""
Accuracy vs distance visualization.

This module provides plotting functions for accuracy vs hamming distance analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, Optional


def plot_accuracy_vs_distance(
    summary_df: pd.DataFrame,
    output_path: str,
    figsize: tuple = (8, 3),
    dpi: int = 300,
    color_by_method: bool = True,
    ylim: Optional[tuple[float, float]] = (0.97, 1.02),
) -> str:
    """
    Create scatter plot of final accuracy vs hamming distance.
    
    Args:
        summary_df: DataFrame with knockout results
        output_path: Path to save the plot image
        figsize: Figure dimensions
        dpi: Image resolution
        color_by_method: Whether to color points by method (GNN vs BP)
        ylim: Optional (ymin, ymax) for y-axis. Use None for auto-scale (e.g. when
              accuracy drops below 0.97). Use e.g. (0.6, 1.02) for a generous range.
    
    Returns:
        Path to saved image file
    """
    if 'final_hard_accuracy' not in summary_df.columns or 'overall_bitwise_fraction_diff' not in summary_df.columns:
        raise ValueError("DataFrame must contain 'final_hard_accuracy' and 'overall_bitwise_fraction_diff' columns")
    
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    if color_by_method and 'method' in summary_df.columns:
        # Separate scatter plots with different colors/markers for each method.
        # Plot BP first, then GNN so GNN points are drawn on top and visible.
        gnn_data = summary_df[summary_df['method'] == 'gnn']
        bp_data = summary_df[summary_df['method'] == 'bp']
        
        if len(bp_data) > 0:
            ax.scatter(bp_data['overall_bitwise_fraction_diff'], 
                      bp_data['final_hard_accuracy'], 
                      c='#019e73', marker='s', s=100, alpha=0.7, 
                      edgecolors='black', linewidth=0.5, label='BP')
        
        if len(gnn_data) > 0:
            ax.scatter(gnn_data['overall_bitwise_fraction_diff'], 
                      gnn_data['final_hard_accuracy'], 
                      c='#e9ad39', marker='o', s=100, alpha=0.7, 
                      edgecolors='black', linewidth=0.5, label='GNN')
        
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
    if ylim is not None:
        ax.set_ylim(ylim[0], ylim[1])
    
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
    baseline_loss: Optional[float] = None,
    ylim_max: Optional[float] = None
) -> str:
    """
    Create line plot with error bars of damage size (knockout_size) vs hamming distance.
    Shows trendlines with error bars, colored by method (GNN vs BP): #e9ad39 for GNN (SA), #019e73 for BP.
    
    Args:
        summary_df: DataFrame with knockout results (must have 'knockout_size', 
                   'per_gate_mean_hamming', and 'method' columns)
        output_path: Path to save the plot image
        figsize: Figure dimensions
        dpi: Image resolution
        color_by_method: Whether to color lines by method (GNN vs BP)
        baseline_accuracy: Optional baseline accuracy (unused, kept for compatibility)
        baseline_loss: Optional baseline loss (unused, kept for compatibility)
    
    Returns:
        Path to saved image file
    """
    required_cols = ['knockout_size', 'per_gate_mean_hamming']
    missing_cols = [col for col in required_cols if col not in summary_df.columns]
    if missing_cols:
        raise ValueError(f"DataFrame must contain columns: {missing_cols}")
    
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    if color_by_method and 'method' in summary_df.columns:
        # Group by knockout_size and method, calculate mean and std
        grouped = summary_df.groupby(['knockout_size', 'method'])['per_gate_mean_hamming'].agg(['mean', 'std', 'count']).reset_index()
        
        # Plot each method separately with different colors and markers
        # SA (GNN) = #e9ad39 (aligned with plot_stepwise_metrics COLOR_UNSEEN), BP = #019e73
        for method, color, marker in [
            ('gnn', '#e9ad39', 'o'), 
            ('bp', '#019e73', 's')
        ]:
            method_data = grouped[grouped['method'] == method]
            if len(method_data) > 0:
                # Sort by knockout_size for proper line plotting
                method_data = method_data.sort_values('knockout_size')
                
                x_coords = method_data['knockout_size'].values
                y_means = method_data['mean'].values
                y_stds = method_data['std'].values
                
                # Fill NaN std values with 0 (happens when only one data point)
                y_stds = np.nan_to_num(y_stds, nan=0.0)
                
                # Plot line with markers (legend: gnn -> NCA, bp -> BP)
                legend_label = 'NCA' if method == 'gnn' else method.upper()
                ax.plot(x_coords, y_means, 
                       color=color,
                       marker=marker,
                       markersize=8,
                       linewidth=2,
                       label=legend_label,
                       alpha=0.9)
                
                # Add error bars
                ax.errorbar(x_coords, y_means, yerr=y_stds,
                           color=color,
                           alpha=0.5,
                           capsize=4,
                           capthick=1.5,
                           linestyle='None',
                           elinewidth=1.5)
        
        # Add legend
        ax.legend(loc='best', fontsize=16)
    else:
        # Fallback: group by knockout_size only
        grouped = summary_df.groupby('knockout_size')['per_gate_mean_hamming'].agg(['mean', 'std']).reset_index()
        grouped = grouped.sort_values('knockout_size')
        
        x_coords = grouped['knockout_size'].values
        y_means = grouped['mean'].values
        y_stds = grouped['std'].values
        y_stds = np.nan_to_num(y_stds, nan=0.0)
        
        ax.plot(x_coords, y_means, 
               marker='o',
               markersize=8,
               linewidth=2,
               alpha=0.9)
        
        ax.errorbar(x_coords, y_means, yerr=y_stds,
                   alpha=0.5,
                   capsize=4,
                   capthick=1.5,
                   linestyle='None',
                   elinewidth=1.5)
    
    # Determine damage behavior type from dataframe
    damage_behavior = None
    if 'damage_behavior' in summary_df.columns:
        damage_behavior_values = summary_df['damage_behavior'].unique()
        if len(damage_behavior_values) > 0:
            damage_behavior = damage_behavior_values[0]
    
    # Capitalize first letter for label
    if damage_behavior:
        damage_type = damage_behavior.capitalize()
    else:
        damage_type = "Damage"  # Fallback
    
    # Customize plot
    ax.set_xlabel(f'{damage_type} Damage Size', fontsize=18)
    ax.set_ylabel('Hamming Distance (Mean per Gate)', fontsize=18)
    ylim_upper = ylim_max if ylim_max is not None else 0.325
    ax.set_ylim(0, ylim_upper)
    ax.tick_params(axis='both', which='major', labelsize=16)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    return output_path


def plot_damage_size_vs_accuracy(
    summary_df: pd.DataFrame,
    output_path: str,
    figsize: tuple = (8, 6),
    dpi: int = 300,
    color_by_method: bool = True,
    baseline_accuracy: Optional[float] = None,
    baseline_loss: Optional[float] = None,
    ylim_min: Optional[float] = None,
    ylim_max: Optional[float] = None,
    method_label_map: Optional[Dict[str, str]] = None,
) -> str:
    """
    Create scatter plot of damage size (knockout_size) vs hard accuracy.
    Shows individual data points, colored by method (GNN vs BP): #e9ad39 for GNN (SA), #019e73 for BP.

    Args:
        summary_df: DataFrame with knockout results (must have 'knockout_size', 
                   'final_hard_accuracy', and 'method' columns)
        output_path: Path to save the plot image
        figsize: Figure dimensions
        dpi: Image resolution
        color_by_method: Whether to color lines by method (GNN vs BP)
        baseline_accuracy: Optional baseline accuracy to plot as horizontal reference line
        baseline_loss: Optional baseline loss (unused, kept for compatibility)
        ylim_min: Optional minimum y-axis limit (default: 0.97)
        ylim_max: Optional maximum y-axis limit (default: 1.02)
        method_label_map: Optional dict mapping method key (e.g. 'gnn', 'bp') to legend label (e.g. 'NCA', 'BP')

    Returns:
        Path to saved image file
    """
    required_cols = ['knockout_size', 'final_hard_accuracy']
    missing_cols = [col for col in required_cols if col not in summary_df.columns]
    if missing_cols:
        raise ValueError(f"DataFrame must contain columns: {missing_cols}")
    
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    # Calculate minimal jitter amount based on typical spacing between knockout sizes
    unique_sizes = sorted(summary_df['knockout_size'].unique())
    if len(unique_sizes) > 1:
        typical_spacing = unique_sizes[1] - unique_sizes[0]  # Use first spacing as reference
        jitter_amount = typical_spacing * 0.06  # 1.5% of typical spacing - minimal jitter
    else:
        jitter_amount = 1.0  # Fallback if only one size
    
    if color_by_method and 'method' in summary_df.columns:
        # Plot each method separately with different colors and markers
        # SA (GNN) = #e9ad39 (aligned with plot_stepwise_metrics COLOR_UNSEEN), BP = #019e73
        for method, color, marker, jitter_offset in [
            ('gnn', '#e9ad39', 'o', -jitter_amount),  # GNN slightly to the left
            ('bp', '#019e73', 's', jitter_amount)    # BP slightly to the right
        ]:
            method_data = summary_df[summary_df['method'] == method]
            if len(method_data) > 0:
                legend_label = (method_label_map or {}).get(method, method.upper())
                # Add minimal jitter to x-coordinates
                x_coords = method_data['knockout_size'].values + jitter_offset
                # Plot individual points
                ax.scatter(x_coords, 
                          method_data['final_hard_accuracy'],
                          color=color,
                          marker=marker,
                          s=50,  # marker size
                          label=legend_label,
                          alpha=0.6,
                          edgecolors='black',
                          linewidths=0.5)
        
        # Add legend
        ax.legend(loc='best', fontsize=16)
    else:
        # Fallback: plot all points without method distinction
        ax.scatter(summary_df['knockout_size'], 
                  summary_df['final_hard_accuracy'],
                  marker='o',
                  s=50,
                  alpha=0.6,
                  edgecolors='black',
                  linewidths=0.5)
    
    # Add baseline accuracy as horizontal reference line if provided
    if baseline_accuracy is not None:
        ax.axhline(y=baseline_accuracy, 
                  color='gray', 
                  linestyle='--', 
                  linewidth=1.5, 
                  alpha=0.7, 
                  label=f'Baseline ({baseline_accuracy:.3f})')
        if not (color_by_method and 'method' in summary_df.columns):
            ax.legend(loc='best', fontsize=16)
    
    # Determine damage behavior type from dataframe
    damage_behavior = None
    if 'damage_behavior' in summary_df.columns:
        damage_behavior_values = summary_df['damage_behavior'].unique()
        if len(damage_behavior_values) > 0:
            damage_behavior = damage_behavior_values[0]
    
    # Capitalize first letter for label
    if damage_behavior:
        damage_type = damage_behavior.capitalize()
    else:
        damage_type = "Damage"  # Fallback
    
    # Customize plot
    ax.set_xlabel(f'{damage_type} Damage Size', fontsize=18)
    ax.set_ylabel('Final Hard Accuracy', fontsize=18)
    ylim_lower = ylim_min if ylim_min is not None else 0.97
    ylim_upper = ylim_max if ylim_max is not None else 1.02
    ax.set_ylim(ylim_lower, ylim_upper)
    ax.tick_params(axis='both', which='major', labelsize=16)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    return output_path

