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
    Plot eval_no_damage/hard_accuracy across training epochs for all runs in a sweep.
    
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
    # NOTE: The WandB panel shows 8 data points. The diagnostics revealed:
    # - eval_no_damage/epoch only has 3 epochs (1024, 4096, 5120) in history due to step_metric grouping
    # - eval_no_damage_steps/epoch has all 8 epochs (0, 1024, 2048, 3072, 4096, 5120, 6144, 7168)
    # The panel is likely showing eval_no_damage_steps/hard_accuracy at the final step per epoch
    # OR it's showing eval_no_damage/final_hard_accuracy but WandB UI aggregates it differently
    # We'll try both approaches and see which matches the panel better
    
    print(f"Loading metrics from sweep {sweep_id}...")
    
    # Approach 1: Try to get eval_no_damage/final_hard_accuracy with all epochs
    # (even though history() only shows 3, maybe we can get more by checking stepwise epochs)
    print("Attempting to load eval_no_damage/final_hard_accuracy...")
    df_final = load_sweep_metrics(
        sweep_id=sweep_id,
        x_metric="eval_no_damage/epoch",
        y_metric="eval_no_damage/final_hard_accuracy",
        project=project,
        entity=entity,
        include_config=True,
        config_keys=["seed"],
    )
    
    # Approach 2: Load stepwise metrics and get final step
    print("Loading stepwise metrics (which have all 8 epochs)...")
    df_steps = load_sweep_metrics(
        sweep_id=sweep_id,
        x_metric="eval_no_damage_steps/epoch",
        y_metric="eval_no_damage_steps/hard_accuracy",
        project=project,
        entity=entity,
        include_config=True,
        config_keys=["seed"],  # Include seed to distinguish runs
    )
    
    # Check what we got
    print(f"\nApproach 1 (eval_no_damage/final_hard_accuracy):")
    if not df_final.empty:
        epochs_final = sorted(df_final['eval_no_damage/epoch'].unique())
        print(f"  {len(df_final)} points, {len(epochs_final)} unique epochs: {epochs_final}")
    else:
        print(f"  No data found")
    
    print(f"\nApproach 2 (eval_no_damage_steps/hard_accuracy):")
    if df_steps.empty:
        print(f"  No eval_no_damage_steps metrics found in sweep {sweep_id}")
        if df_final.empty:
            return None
        # Fall back to df_final
        df = df_final
        x_col = "eval_no_damage/epoch"
        y_col = "eval_no_damage/final_hard_accuracy"
        print(f"\nUsing final metrics (fallback): {len(df)} points from {df['run_id'].nunique()} runs")
    else:
        print(f"  {len(df_steps)} stepwise data points from {df_steps['run_id'].nunique()} runs")
    
        # Filter to only the final step per epoch per run
        # The stepwise metrics log at every message passing step (0, 1, 2, ..., n_message_steps)
        # We want the LAST step, which should be step = periodic_eval_inner_steps (typically 200)
        step_col = "eval_no_damage_steps/step"
        if step_col not in df_steps.columns:
            # If no step column, assume the last row per epoch is the final step
            print("Warning: No step column found, using last row per epoch as final step")
            df = df_steps.groupby(['run_id', 'eval_no_damage_steps/epoch']).tail(1).reset_index(drop=True)
        else:
            # Diagnostic: Check step values for a sample run/epoch
            sample_run = df_steps['run_id'].iloc[0]
            sample_epoch = df_steps['eval_no_damage_steps/epoch'].iloc[0]
            sample_data = df_steps[(df_steps['run_id'] == sample_run) & 
                                   (df_steps['eval_no_damage_steps/epoch'] == sample_epoch)]
            if len(sample_data) > 0:
                steps = sorted(sample_data[step_col].unique())
                print(f"Sample: Run {sample_run[:8]}, Epoch {sample_epoch}: {len(steps)} steps, range {steps[0]}-{steps[-1]}")
                print(f"  Step values: {steps[:10]}{'...' if len(steps) > 10 else ''}")
                print(f"  Expected final step should be around 200 (periodic_eval_inner_steps)")
            
            # Sort by step descending, then take the first (highest) step for each (run_id, epoch)
            df_steps_sorted = df_steps.sort_values(['run_id', 'eval_no_damage_steps/epoch', step_col], 
                                                    ascending=[True, True, False])
            df = df_steps_sorted.groupby(['run_id', 'eval_no_damage_steps/epoch']).head(1).reset_index(drop=True)
            print(f"Filtered to final step per epoch: {len(df)} data points")
            
            # Verify we got the right steps
            if len(df) > 0:
                final_steps = sorted(df[step_col].unique())
                print(f"Final step values in filtered data: {final_steps[:10]}{'...' if len(final_steps) > 10 else ''}")
        
        # Set column names for stepwise approach (always set, regardless of which branch we took)
        x_col = "eval_no_damage_steps/epoch"
        y_col = "eval_no_damage_steps/hard_accuracy"
        print(f"\nUsing stepwise metrics: {len(df)} points from {df['run_id'].nunique()} runs")
    
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

