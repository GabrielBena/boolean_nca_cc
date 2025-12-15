"""
Plot eval_no_damage metrics across training epochs from WandB sweep.

This script uses the general wandb_loader utility to load and plot
eval_no_damage/hard_accuracy from all runs in a sweep, showing the
variation across different seeds/configurations.
"""

import argparse
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from boolean_nca_cc.utils.wandb_loader import load_sweep_metrics


def plot_sweep_eval_no_damage_hard_accuracy(
    sweep_id: str,
    output_path=None,
    title=None,
    figsize=(12, 8),
    dpi=300,
    project: str = "boolean-nca-cc",
    entity: str = "marcello-barylli-growai",
    show_individual_runs: bool = True,
    show_mean_std: bool = True,
    alpha_individual: float = 0.3,
):
    """
    Plot eval_no_damage/final_hard_accuracy across training epochs for all runs in a sweep.
    
    Uses eval_no_damage_raw/* metrics which are logged without step_metric grouping,
    providing direct access to all logged evaluation epochs without WandB's grouping/aggregation.
    
    Args:
        sweep_id: WandB sweep ID
        output_path: Path to save the plot
        title: Plot title (auto-generated if None)
        figsize: Figure size tuple
        dpi: Image resolution
        project: WandB project name
        entity: WandB entity/username
        show_individual_runs: If True, plot each run as a separate line
        show_mean_std: If True, plot mean and std across runs
        alpha_individual: Transparency for individual run lines
    """
    # Load metrics from all runs in the sweep
    # Use eval_no_damage_raw/* metrics which are logged without step_metric grouping
    # This provides direct access to all logged values without WandB's grouping/aggregation
    print(f"Loading metrics from sweep {sweep_id}...")
    print("Using eval_no_damage_raw/* metrics (no step_metric grouping)...")
    
    df = load_sweep_metrics(
        sweep_id=sweep_id,
        x_metric="eval_no_damage_raw/epoch",
        y_metric="eval_no_damage_raw/final_hard_accuracy",
        project=project,
        entity=entity,
        include_config=True,
        config_keys=["seed"],  # Include seed to distinguish runs
    )
    
    if df.empty:
        print(f"No eval_no_damage_raw metrics found in sweep {sweep_id}")
        print("Note: Raw metrics are only available for runs logged after implementing Option 3.")
        print("For older runs, you may need to use eval_no_damage_steps/* metrics.")
        return None
    
    print(f"Loaded {len(df)} data points from {df['run_id'].nunique()} runs")
    
    # Set column names
    x_col = "eval_no_damage_raw/epoch"
    y_col = "eval_no_damage_raw/final_hard_accuracy"
    
    # Check what epochs we actually have per run
    print(f"\nDetailed epoch breakdown per run:")
    for run_id in sorted(df['run_id'].unique())[:5]:  # Show first 5 runs
        run_data = df[df['run_id'] == run_id].sort_values(x_col)
        seed = run_data['seed'].iloc[0] if 'seed' in run_data.columns else None
        epochs = sorted(run_data[x_col].unique())
        print(f"  Run {run_id[:8]} (seed={seed}): {len(epochs)} epochs: {epochs}")
    
    # Check for any train/stepwise variants that might have leaked in
    train_variants = [c for c in df.columns if '_train' in c or '_steps' in c]
    if train_variants:
        print(f"\nWarning: Found train/stepwise columns in loaded data: {train_variants}")
        print("This suggests some filtering may be needed.")
    
    # Verify we only have the metrics we want
    expected_cols = {x_col, y_col, 'run_id', 'seed'}
    unexpected_cols = set(df.columns) - expected_cols
    if unexpected_cols:
        print(f"\nNote: Additional columns in data: {unexpected_cols}")
    
    # Column names are already set above
    
    print(f"\nEpoch statistics:")
    print(f"  Min epoch: {df[x_col].min()}")
    print(f"  Max epoch: {df[x_col].max()}")
    print(f"  Unique epochs: {df[x_col].nunique()}")
    print(f"  Epochs per run (mean): {len(df) / df['run_id'].nunique():.1f}")
    
    # Show unique epochs
    unique_epochs = sorted(df[x_col].unique())
    print(f"  Unique epoch values (first 20): {unique_epochs[:20]}")
    if len(unique_epochs) > 20:
        print(f"  ... and {len(unique_epochs) - 20} more")
    
    # Show data points per run
    print(f"\nData points per run:")
    for run_id in df['run_id'].unique()[:5]:  # Show first 5 runs
        run_data = df[df['run_id'] == run_id]
        seed = run_data['seed'].iloc[0] if 'seed' in run_data.columns else None
        print(f"  Run {run_id[:8]} (seed={seed}): {len(run_data)} points, epochs: {run_data[x_col].min():.0f}-{run_data[x_col].max():.0f}")
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot individual runs if requested
    if show_individual_runs:
        print(f"\nPlotting individual runs...")
        for idx, run_id in enumerate(df['run_id'].unique()):
            run_data = df[df['run_id'] == run_id].sort_values(x_col)
            seed = run_data['seed'].iloc[0] if 'seed' in run_data.columns else None
            
            if len(run_data) > 0:
                label = f"seed={seed}" if seed is not None else f"run={run_id[:8]}"
                ax.plot(
                    run_data[x_col],
                    run_data[y_col],
                    alpha=alpha_individual,
                    linewidth=1,
                    color='gray',
                    label=label if idx == 0 else "",  # Only label first to avoid legend clutter
                )
                if idx < 3:  # Print first few for debugging
                    print(f"  Run {idx+1} (seed={seed}): {len(run_data)} data points, epochs: {run_data[x_col].min():.0f}-{run_data[x_col].max():.0f}")
    
    # Plot mean and std if requested
    if show_mean_std:
        # Note: Duplicates should already be removed by load_metric_pair, but double-check
        initial_len = len(df)
        df_deduped = df.drop_duplicates(subset=['run_id', x_col], keep='first')
        if len(df_deduped) < initial_len:
            print(f"\nWarning: Found {initial_len - len(df_deduped)} additional duplicate rows (same run_id and epoch)")
            print("This suggests duplicates weren't fully removed by the loader.")
            df = df_deduped
        
        # Group by epoch and compute statistics
        # Round epochs to nearest 10 to handle slight misalignments across runs
        df_rounded = df.copy()
        df_rounded['epoch_rounded'] = (df_rounded[x_col] / 10).round() * 10
        
        # Group by rounded epoch, computing mean/std of y_col and counting UNIQUE runs
        grouped = df_rounded.groupby('epoch_rounded').agg({
            y_col: ['mean', 'std'],
            'run_id': 'nunique'  # Count unique runs, not total rows
        }).reset_index()
        
        # Flatten column names
        grouped.columns = ['epoch_rounded', 'mean', 'std', 'unique_runs']
        grouped = grouped.sort_values('epoch_rounded')
        
        mean_vals = grouped['mean'].values
        std_vals = grouped['std'].values
        epochs = grouped['epoch_rounded'].values
        
        print(f"\nGrouped statistics:")
        print(f"  Number of epoch groups: {len(grouped)}")
        print(f"  Mean unique runs per epoch: {grouped['unique_runs'].mean():.1f}")
        print(f"  Min unique runs per epoch: {grouped['unique_runs'].min()}")
        print(f"  Max unique runs per epoch: {grouped['unique_runs'].max()}")
        
        # Show breakdown per epoch
        print(f"\nEpoch breakdown:")
        for _, row in grouped.head(10).iterrows():
            print(f"  Epoch {row['epoch_rounded']:.0f}: {row['unique_runs']:.0f} runs, mean={row['mean']:.4f}, std={row['std']:.4f}")
        
        # Plot mean line
        ax.plot(
            epochs,
            mean_vals,
            color='#d62728',  # Red
            linewidth=3,
            label='Mean',
            marker='o',
            markersize=6,
            zorder=10,
        )
        
        # Plot std shaded region
        ax.fill_between(
            epochs,
            mean_vals - std_vals,
            mean_vals + std_vals,
            alpha=0.3,
            color='#d62728',
            label='±1 Std',
            zorder=5,
        )
    
    # Formatting
    ax.set_xlabel('Training Epoch', fontsize=24)
    ax.set_ylabel('Hard Accuracy', fontsize=24)
    
    if title is None:
        title = f"Eval No Damage Hard Accuracy Across Training\nSweep: {sweep_id} ({df['run_id'].nunique()} runs)"
    ax.set_title(title, fontsize=28, fontweight='bold')
    
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])
    
    # Set x-axis limits based on actual data range (not auto-scaled to just the data points)
    if len(df) > 0:
        x_min = df[x_col].min()
        x_max = df[x_col].max()
        # Add some padding
        x_range = x_max - x_min
        if x_range > 0:
            ax.set_xlim([x_min - 0.05 * x_range, x_max + 0.05 * x_range])
        else:
            # If all data is at one epoch, show a reasonable range around it
            ax.set_xlim([max(0, x_min - 100), x_max + 100])
    
    # Add horizontal line at perfect accuracy
    ax.axhline(y=1.0, color='green', linestyle=':', alpha=0.5, linewidth=1.5)
    
    # Legend
    if show_mean_std:
        ax.legend(loc='best', fontsize=16)
    
    plt.tight_layout()
    
    # Save or show
    if output_path:
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        print(f"Saved plot to: {output_path}")
        
        if output_path.endswith('.png'):
            pdf_path = output_path.replace('.png', '.pdf')
        else:
            pdf_path = output_path + '.pdf'
        plt.savefig(pdf_path, bbox_inches='tight')
        print(f"Saved plot (PDF) to: {pdf_path}")
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

