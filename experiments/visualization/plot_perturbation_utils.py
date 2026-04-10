"""
Accuracy vs distance visualization.

This module provides plotting functions for accuracy vs hamming distance analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
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
                      c='#414487', marker='s', s=100, alpha=0.7,
                      edgecolors='black', linewidth=0.5, label='BP')
        
        if len(gnn_data) > 0:
            ax.scatter(gnn_data['overall_bitwise_fraction_diff'], 
                      gnn_data['final_hard_accuracy'], 
                      c='#34b779', marker='o', s=100, alpha=0.7,
                      edgecolors='black', linewidth=0.5, label='TMT')
        
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


def _plot_damage_size_vs_hamming_on_ax(
    ax,
    summary_df: pd.DataFrame,
    color_by_method: bool = True,
    ylim_max: Optional[float] = None,
    show_xlabel: bool = True,
):
    """Plot damage size vs hamming distance onto an existing axes."""
    if color_by_method and 'method' in summary_df.columns:
        grouped = summary_df.groupby(['knockout_size', 'method'])['per_gate_mean_hamming'].agg(['mean', 'std', 'count']).reset_index()

        for method, color, marker in [
            ('gnn', '#34b779', 'o'),
            ('bp', '#414487', 's')
        ]:
            method_data = grouped[grouped['method'] == method]
            if len(method_data) > 0:
                method_data = method_data.sort_values('knockout_size')
                x_coords = method_data['knockout_size'].values
                y_means = method_data['mean'].values
                y_stds = np.nan_to_num(method_data['std'].values, nan=0.0)

                legend_label = 'TMT' if method == 'gnn' else method.upper()
                ax.plot(x_coords, y_means,
                       color=color, marker=marker, markersize=8,
                       linewidth=2, label=legend_label, alpha=0.9)
                ax.errorbar(x_coords, y_means, yerr=y_stds,
                           color=color, alpha=0.5, capsize=4,
                           capthick=1.5, linestyle='None', elinewidth=1.5)

        ax.legend(loc='best', fontsize=16)
    else:
        grouped = summary_df.groupby('knockout_size')['per_gate_mean_hamming'].agg(['mean', 'std']).reset_index()
        grouped = grouped.sort_values('knockout_size')
        x_coords = grouped['knockout_size'].values
        y_means = grouped['mean'].values
        y_stds = np.nan_to_num(grouped['std'].values, nan=0.0)

        ax.plot(x_coords, y_means, marker='o', markersize=8, linewidth=2, alpha=0.9)
        ax.errorbar(x_coords, y_means, yerr=y_stds,
                   alpha=0.5, capsize=4, capthick=1.5, linestyle='None', elinewidth=1.5)

    damage_behavior = None
    if 'damage_behavior' in summary_df.columns:
        damage_behavior_values = summary_df['damage_behavior'].unique()
        if len(damage_behavior_values) > 0:
            damage_behavior = damage_behavior_values[0]
    damage_type = damage_behavior.capitalize() if damage_behavior else "Damage"

    if show_xlabel:
        ax.set_xlabel(f'{damage_type} Damage Size', fontsize=18)
    ax.set_ylabel('Hamming Distance (Mean per Gate)', fontsize=18)
    ylim_upper = ylim_max if ylim_max is not None else 0.325
    ax.set_ylim(0, ylim_upper)
    ax.tick_params(axis='both', which='major', labelsize=16)


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
    Shows trendlines with error bars, colored by method (GNN vs BP): #34b779 for GNN (SA), #414487 for BP.
    """
    required_cols = ['knockout_size', 'per_gate_mean_hamming']
    missing_cols = [col for col in required_cols if col not in summary_df.columns]
    if missing_cols:
        raise ValueError(f"DataFrame must contain columns: {missing_cols}")

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    _plot_damage_size_vs_hamming_on_ax(ax, summary_df, color_by_method, ylim_max)

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()

    return output_path


def _plot_damage_size_vs_accuracy_on_ax(
    ax,
    summary_df: pd.DataFrame,
    color_by_method: bool = True,
    baseline_accuracy: Optional[float] = None,
    ylim_min: Optional[float] = None,
    ylim_max: Optional[float] = None,
    method_label_map: Optional[Dict[str, str]] = None,
    show_xlabel: bool = True,
):
    """Plot damage size vs accuracy onto an existing axes."""
    unique_sizes = sorted(summary_df['knockout_size'].unique())
    if len(unique_sizes) > 1:
        typical_spacing = unique_sizes[1] - unique_sizes[0]
        jitter_amount = typical_spacing * 0.06
    else:
        jitter_amount = 1.0

    if color_by_method and 'method' in summary_df.columns:
        for method, color, marker, jitter_offset in [
            ('gnn', '#34b779', 'o', -jitter_amount),
            ('bp', '#414487', 's', jitter_amount)
        ]:
            method_data = summary_df[summary_df['method'] == method]
            if len(method_data) > 0:
                _default_labels = {'gnn': 'TMT', 'bp': 'BP'}
                legend_label = (method_label_map or {}).get(method, _default_labels.get(method, method.upper()))
                x_coords = method_data['knockout_size'].values + jitter_offset
                ax.scatter(x_coords,
                          method_data['final_hard_accuracy'],
                          color=color, marker=marker, s=50,
                          label=legend_label, alpha=0.6,
                          edgecolors='black', linewidths=0.5)

        ax.legend(loc='best', fontsize=16)
    else:
        ax.scatter(summary_df['knockout_size'],
                  summary_df['final_hard_accuracy'],
                  marker='o', s=50, alpha=0.6,
                  edgecolors='black', linewidths=0.5)

    if baseline_accuracy is not None:
        ax.axhline(y=baseline_accuracy,
                  color='gray', linestyle='--', linewidth=1.5,
                  alpha=0.7, label=f'Baseline ({baseline_accuracy:.3f})')
        if not (color_by_method and 'method' in summary_df.columns):
            ax.legend(loc='best', fontsize=16)

    damage_behavior = None
    if 'damage_behavior' in summary_df.columns:
        damage_behavior_values = summary_df['damage_behavior'].unique()
        if len(damage_behavior_values) > 0:
            damage_behavior = damage_behavior_values[0]
    damage_type = damage_behavior.capitalize() if damage_behavior else "Damage"

    if show_xlabel:
        ax.set_xlabel(f'{damage_type} Damage Size', fontsize=18)
    ax.set_ylabel('Final Hard Accuracy', fontsize=18)
    ylim_lower = ylim_min if ylim_min is not None else 0.97
    ylim_upper = ylim_max if ylim_max is not None else 1.02
    ax.set_ylim(ylim_lower, ylim_upper)
    ax.tick_params(axis='both', which='major', labelsize=16)


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
    Shows individual data points, colored by method (GNN vs BP): #34b779 for GNN (SA), #414487 for BP.
    """
    required_cols = ['knockout_size', 'final_hard_accuracy']
    missing_cols = [col for col in required_cols if col not in summary_df.columns]
    if missing_cols:
        raise ValueError(f"DataFrame must contain columns: {missing_cols}")

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    _plot_damage_size_vs_accuracy_on_ax(
        ax, summary_df, color_by_method, baseline_accuracy,
        ylim_min, ylim_max, method_label_map,
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', pad_inches=0.15)
    plt.close()

    return output_path


def plot_combined_hamming_accuracy(
    summary_df: pd.DataFrame,
    output_path: str,
    width: float = 8,
    dpi: int = 300,
    color_by_method: bool = True,
    hamming_ylim_max: Optional[float] = None,
    accuracy_ylim_min: Optional[float] = 0.99,
    accuracy_ylim_max: Optional[float] = 1.01,
    baseline_accuracy: Optional[float] = None,
    height_ratio: tuple = (2, 1),
    method_label_map: Optional[Dict[str, str]] = None,
) -> str:
    """
    Combined two-panel figure: hamming distance (top) and accuracy (bottom).
    Shared x-axis ensures horizontal alignment between panels.
    """
    total_height = width * (sum(height_ratio) / height_ratio[0]) * 0.75
    fig, (ax_hamming, ax_acc) = plt.subplots(
        2, 1, figsize=(width, total_height),
        gridspec_kw={'height_ratios': list(height_ratio)},
        sharex=True,
    )

    _plot_damage_size_vs_hamming_on_ax(
        ax_hamming, summary_df, color_by_method,
        ylim_max=hamming_ylim_max, show_xlabel=False,
    )
    _plot_damage_size_vs_accuracy_on_ax(
        ax_acc, summary_df, color_by_method,
        baseline_accuracy=baseline_accuracy,
        ylim_min=accuracy_ylim_min, ylim_max=accuracy_ylim_max,
        method_label_map=method_label_map, show_xlabel=True,
    )

    # Only show legend on the top panel to avoid duplication
    if ax_acc.get_legend() is not None:
        ax_acc.get_legend().remove()

    # Align y-axis tick decimal places across both panels
    ax_hamming.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
    ax_acc.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', pad_inches=0.15)
    plt.close()

    return output_path
