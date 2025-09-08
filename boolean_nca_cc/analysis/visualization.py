"""
Visualization utilities for circuit analysis.

This module provides plotting functions for circuit performance analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Optional, List, Dict, Tuple
import logging

try:
    import wandb
except ImportError:
    wandb = None

log = logging.getLogger(__name__)


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


def create_eval_plot_prog(
    pattern_data: List[List],
    training_mode: str,
    wandb_run=None,
    output_dir: str = "reports/figures",
    filename: str = "training_progress.png",
    bp_results: Optional[Dict] = None,
    show_bp_reference: bool = False
) -> str:
    """
    Create a plot from current run data, with mode-aware labeling.
    
    Args:
        pattern_data: Current run pattern data
        training_mode: Current training mode ("growth" or "repair")
        wandb_run: Optional wandb run for fallback metrics
        output_dir: Directory to save the plot
        filename: Filename for the saved plot
        bp_results: Pre-computed backpropagation results (optional)
        show_bp_reference: Whether to show backpropagation reference line
    
    Returns:
        Path to saved plot file
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)
    
    # Create single-mode plot with mode-specific title and colors
    if pattern_data and len(pattern_data) > 0:
        df = pd.DataFrame(pattern_data, columns=['epoch', 'pattern_id', 'hard_accuracy', 'knockout_diversity'])
        epoch_stats = df.groupby('epoch')['hard_accuracy'].agg(['mean', 'std']).reset_index()
        
        epochs = epoch_stats['epoch'].values
        means = epoch_stats['mean'].values
        stds = np.nan_to_num(epoch_stats['std'].values, nan=0.0)
        
        # Choose colors based on mode (GNN/SA always red)
        color = 'red'  # GNN/SA is always red
        mode_label = 'Growth Mode' if training_mode == 'growth' else 'Repair Mode'
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax.plot(epochs, means, color=color, linewidth=2, label=f'{mode_label} Hard Accuracy')
        ax.fill_between(epochs, means - stds, means + stds, color=color, alpha=0.2, label='±1 std')
        
        # Add backpropagation reference line if requested and results available
        if show_bp_reference and bp_results is not None:
            try:
                # Calculate mean final BP accuracy across all patterns
                bp_final_accuracies = [pattern_results["final_hard_accuracy"] 
                                     for pattern_results in bp_results["patterns_performance"]]
                bp_mean_accuracy = np.mean(bp_final_accuracies)
                
                ax.axhline(y=bp_mean_accuracy, 
                          color='blue',
                          linestyle='--',
                          linewidth=1.5,
                          alpha=0.8,
                          label=f'BP Final (mean: {bp_mean_accuracy:.3f})')
            except Exception as e:
                log.warning(f"Error adding BP reference line: {e}")
        
        # Set font sizes - much larger increase for Figure 1
        ax.set_xlabel('Training Epoch', fontsize=24)
        ax.set_ylabel('Hard Accuracy', fontsize=24)
        ax.set_title(f'Training Progress - {mode_label}', fontsize=28)
        ax.tick_params(axis='both', which='major', labelsize=20)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='lower right', fontsize=18)
        ax.set_ylim(0.6, 1)
        ax.axhline(y=1.0, color='green', linestyle='--', alpha=0.5, label='Perfect Accuracy')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        log.info(f"Training progress plot saved to: {output_path}")
        return output_path
    
    else:
        # Fallback to simple plot if no pattern data
        log.warning("No pattern data available for plotting")
        return output_path
