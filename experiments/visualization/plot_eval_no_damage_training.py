"""
Plot eval_no_damage metrics across training epochs from WandB sweep.

This script loads and plots both training and eval (test) performance
to visualize the generalization gap (overfitting). It creates two side-by-side
subplots showing:
- Hard Accuracy: eval_no_damage/final_hard_accuracy (test) vs eval_no_damage_train/final_hard_accuracy (train)
- Hard Loss: eval_no_damage/final_hard_loss (test) vs eval_no_damage_train/final_hard_loss (train)
"""

import argparse
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

# Import centralized figure configuration
from experiments.visualization.figure_config import (
    setup_style,
    create_figure,
    format_axis,
    format_suptitle,
    save_figure,
    COLORS,
    LINE_STYLES,
    FONT_SIZES,
)

# Metrics are now loaded directly from local storage organized by sweep subfolders


def plot_sweep_eval_no_damage_hard_accuracy(
    sweep_id: str,
    output_path=None,
    title=None,
    figsize=None,
    dpi=300,
    project: str = "boolean-nca-cc",
    entity: str = "marcello-barylli-growai",
    show_individual_runs: bool = True,
    show_mean_std: bool = True,
    alpha_individual: float = None,
):
    """
    Plot eval_no_damage metrics (hard_accuracy and hard_loss) across training epochs for all runs in a sweep.
    Shows both training and eval (test) performance to visualize generalization gap.
    Creates two side-by-side subplots: hard_accuracy (left) and hard_loss (right).
    
    Loads metrics from local storage (results/metrics/) which are saved during training
    to avoid WandB's step grouping and overwriting issues.
    
    Args:
        sweep_id: WandB sweep ID (used to find runs in the sweep)
        output_path: Path to save the plot
        title: Plot title (auto-generated if None)
        figsize: Figure size tuple (default: None, uses standardized double-column size)
        dpi: Image resolution (default: 300 from config)
        project: WandB project name
        entity: WandB entity/username
        show_individual_runs: If True, plot each run as a separate line
        show_mean_std: If True, plot mean and std across runs
        alpha_individual: Transparency for individual run lines (default: from config)
    """
    # Setup consistent style
    setup_style()
    
    # Use default alpha if not specified
    if alpha_individual is None:
        alpha_individual = LINE_STYLES['alpha_individual']
    
    # Load metrics from local storage (organized by sweep subfolders)
    print(f"Loading metrics from local storage for sweep {sweep_id}...")
    from boolean_nca_cc.utils.metrics_storage import load_sweep_metrics
    
    # Load both test and train splits
    df_test = load_sweep_metrics(
        sweep_id=sweep_id,
        metrics_dir="results/metrics",
        format="jsonl",
        split="test"
    )
    
    df_train = load_sweep_metrics(
        sweep_id=sweep_id,
        metrics_dir="results/metrics",
        format="jsonl",
        split="train"
    )
    
    if df_test.empty and df_train.empty:
        print(f"No metrics found in local storage for sweep {sweep_id}")
        print("Make sure metrics are being saved during training (check results/metrics/ directory)")
        return None
    
    print(f"Loaded {len(df_test)} test data points and {len(df_train)} train data points from {max(df_test['run_id'].nunique() if not df_test.empty else 0, df_train['run_id'].nunique() if not df_train.empty else 0)} runs")
    
    # Set column names
    x_col = "epoch"
    y_col_test_acc = "final_hard_accuracy"  # Test split uses clean name
    y_col_train_acc = "eval_no_damage_train/final_hard_accuracy"  # Train split keeps prefix
    y_col_test_loss = "final_hard_loss"  # Test split uses clean name
    y_col_train_loss = "eval_no_damage_train/final_hard_loss"  # Train split keeps prefix
    
    # Verify columns exist
    if df_test.empty:
        print("Warning: No test metrics found")
    elif x_col not in df_test.columns or y_col_test_acc not in df_test.columns or y_col_test_loss not in df_test.columns:
        print(f"Error: Expected columns '{x_col}', '{y_col_test_acc}', and '{y_col_test_loss}' not found in test data")
        print(f"Available columns: {df_test.columns.tolist()}")
        return None
    
    if df_train.empty:
        print("Warning: No train metrics found")
    elif x_col not in df_train.columns or y_col_train_acc not in df_train.columns or y_col_train_loss not in df_train.columns:
        print(f"Error: Expected columns '{x_col}', '{y_col_train_acc}', and '{y_col_train_loss}' not found in train data")
        print(f"Available columns: {df_train.columns.tolist()}")
        return None
    
    # Normalize loss values by split fraction to make train and test comparable
    # Loss is summed (not averaged) across samples, so with 80/20 split:
    # train loss is ~4x higher than test loss due to 4x more samples
    # Normalize by fraction to get "loss per unit of data" for fair comparison
    train_fraction = 0.8  # 80% train split
    test_fraction =  0.2   # 20% test split
    
    # Create normalized loss columns for plotting
    if not df_train.empty and y_col_train_loss in df_train.columns:
        df_train = df_train.copy()
        df_train[y_col_train_loss + '_normalized'] = df_train[y_col_train_loss] / train_fraction
    if not df_test.empty and y_col_test_loss in df_test.columns:
        df_test = df_test.copy()
        df_test[y_col_test_loss + '_normalized'] = df_test[y_col_test_loss] / test_fraction
    
    # Create figure with two subplots side by side (standardized sizing)
    fig, (ax_acc, ax_loss) = create_figure(
        width_type='double',
        nrows=1,
        ncols=2,
        figsize=figsize,  # Use custom if provided, otherwise use standardized
    )
    
    # Colors for test and train (from centralized config)
    color_test = COLORS['test']
    color_train = COLORS['train']
    
    # Helper function to plot a split on a given axis
    def plot_split(ax, df, y_col, color, label_prefix, alpha_ind=alpha_individual):
        if df.empty:
            return
        
        # Check if column exists
        if y_col not in df.columns:
            print(f"Warning: Column '{y_col}' not found in dataframe, skipping plot for {label_prefix}")
            return
        
        # Remove duplicates
        initial_len = len(df)
        df_deduped = df.drop_duplicates(subset=['run_id', x_col], keep='first')
        if len(df_deduped) < initial_len:
            print(f"  Removed {initial_len - len(df_deduped)} duplicate rows")
        df = df_deduped
        
        # Plot individual runs if requested
        if show_individual_runs:
            for idx, run_id in enumerate(df['run_id'].unique()):
                run_data = df[df['run_id'] == run_id].sort_values(x_col)
                
                if len(run_data) > 0:
                    ax.plot(
                        run_data[x_col],
                        run_data[y_col],
                        alpha=alpha_ind,
                        linewidth=LINE_STYLES['linewidth_thin'],
                        color=color,
                    )
        
        # Plot mean and std if requested
        if show_mean_std:
            # Round epochs to nearest 10 to handle slight misalignments across runs
            df_rounded = df.copy()
            df_rounded['epoch_rounded'] = (df_rounded[x_col] / 10).round() * 10
            
            # Group by rounded epoch, computing mean/std of y_col and counting UNIQUE runs
            grouped = df_rounded.groupby('epoch_rounded').agg({
                y_col: ['mean', 'std'],
                'run_id': 'nunique'
            }).reset_index()
            
            # Flatten column names
            grouped.columns = ['epoch_rounded', 'mean', 'std', 'unique_runs']
            grouped = grouped.sort_values('epoch_rounded')
            
            mean_vals = grouped['mean'].values
            std_vals = grouped['std'].values
            epochs = grouped['epoch_rounded'].values
            
            # Plot mean line
            ax.plot(
                epochs,
                mean_vals,
                color=color,
                linewidth=LINE_STYLES['linewidth_thick'],
                label=f'{label_prefix} Mean',
                marker='o',
                markersize=LINE_STYLES['markersize'],
                zorder=10,
            )
            
            # Plot std shaded region
            ax.fill_between(
                epochs,
                mean_vals - std_vals,
                mean_vals + std_vals,
                alpha=LINE_STYLES['alpha_std'],
                color=color,
                label=f'{label_prefix} ±1 Std',
                zorder=5,
            )
    
    # Combine both dataframes for determining x-axis limits and run count
    df_combined = pd.concat([df_test, df_train], ignore_index=True) if not df_test.empty or not df_train.empty else pd.DataFrame()
    
    # Plot accuracy subplot (left)
    print(f"\nPlotting test/eval accuracy performance...")
    plot_split(ax_acc, df_test, y_col_test_acc, color_test, 'Test', alpha_individual)
    print(f"Plotting training accuracy performance...")
    plot_split(ax_acc, df_train, y_col_train_acc, color_train, 'Train', alpha_individual)
    
    # Plot loss subplot (right) - use normalized loss columns
    print(f"\nPlotting test/eval loss performance (normalized)...")
    plot_split(ax_loss, df_test, y_col_test_loss + '_normalized', color_test, 'Test', alpha_individual)
    print(f"Plotting training loss performance (normalized)...")
    plot_split(ax_loss, df_train, y_col_train_loss + '_normalized', color_train, 'Train', alpha_individual)
    
    # Formatting for accuracy subplot (using centralized formatting)
    format_axis(
        ax_acc,
        xlabel='Training Epoch',
        ylabel='Hard Accuracy',
        xlim=[0, 8192],
        ylim=[0, 1.05],
        grid=True,
        legend=False,  # No legend on accuracy plot
        legend_loc='best',
    )
    
    # Formatting for loss subplot (using centralized formatting)
    format_axis(
        ax_loss,
        xlabel='Training Epoch',
        ylabel='Hard Loss (Normalized)',
        xlim=[0, 8192],
        ylim=[None, None],  # Only set bottom limit
        grid=True,
        legend=show_mean_std,
        legend_loc='best',
    )
    # Loss values are normalized by split fraction (train/0.8, test/0.2) to make them
    # directly comparable. This accounts for the fact that loss is summed (not averaged)
    # across samples, so train loss would otherwise be ~4x higher than test loss.
    ax_loss.set_ylim(bottom=0)
    
    # Overall title removed per user request
    # format_suptitle(fig, title, y=1.02)
    
    plt.tight_layout()
    
    # Save or show (using centralized save function)
    if output_path:
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        save_figure(output_path, fig=fig)
    else:
        plt.show()
    
    plt.close()
    return output_path


def main():
    """Main function for plotting sweep results."""
    parser = argparse.ArgumentParser(
        description="Load and plot eval_no_damage/hard_accuracy from WandB sweep"
    )
    parser.add_argument(
        "--sweep_id",
        type=str,
        default="143rzf7h",
        help="WandB sweep ID (default: 143rzf7h)",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="boolean-nca-cc",
        help="WandB project name (default: boolean-nca-cc)",
    )
    parser.add_argument(
        "--entity",
        type=str,
        default="marcello-barylli-growai",
        help="WandB entity/username (default: marcello-barylli-growai)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/figures",
        help="Output directory for plots (default: results/figures)",
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default=None,
        help="Output filename (without extension). If None, uses sweep_id",
    )
    parser.add_argument(
        "--hide_individual",
        action="store_true",
        help="Hide individual run lines (only show mean/std)",
    )
    parser.add_argument(
        "--hide_mean_std",
        action="store_true",
        help="Hide mean/std (only show individual runs)",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Custom plot title (default: auto-generated)",
    )
    
    args = parser.parse_args()
    
    # Generate output path
    output_name = args.output_name if args.output_name else f"eval_no_damage_sweep_{args.sweep_id}"
    output_path = os.path.join(args.output_dir, f"{output_name}.png")
    
    print(f"Loading metrics from WandB sweep {args.sweep_id} and generating plot...")
    plot_sweep_eval_no_damage_hard_accuracy(
        sweep_id=args.sweep_id,
        output_path=output_path,
        title=args.title,
        project=args.project,
        entity=args.entity,
        show_individual_runs=not args.hide_individual,
        show_mean_std=not args.hide_mean_std,
    )


if __name__ == "__main__":
    main()

