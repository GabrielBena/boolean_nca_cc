"""
Analysis script for task and seed sweeps.

This script loads metrics from a sweep that varied tasks and seeds,
and provides summary statistics and visualizations.

Robustly handles:
- Different final epochs per run (early stopping)
- Missing tasks (e.g., failed runs)
- Train vs test comparison on final epoch
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from boolean_nca_cc.utils.metrics_storage import load_sweep_metrics

# Import centralized figure configuration
import sys
import os
# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))
from experiments.visualization.figure_config import (
    setup_style,
    create_figure,
    format_axis,
    save_figure,
    COLORS,
    LINE_STYLES,
    FONT_SIZES,
)


def get_latest_epoch_per_run(df: pd.DataFrame) -> pd.DataFrame:
    """
    Get the latest epoch for each run, split, and task combination.
    
    This handles cases where runs stopped early (different final epochs).
    
    Args:
        df: DataFrame with columns ['run_id', 'split', 'task', 'epoch', ...]
    
    Returns:
        DataFrame with one row per (run_id, split, task) combination at latest epoch
    """
    if df.empty:
        return df
    
    required_cols = ['run_id', 'epoch']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Group by run_id, split (if exists), and task (if exists)
    group_cols = ['run_id']
    if 'split' in df.columns:
        group_cols.append('split')
    if 'task' in df.columns:
        group_cols.append('task')
    
    # Get latest epoch per group
    latest_df = df.loc[df.groupby(group_cols)['epoch'].idxmax()].copy()
    
    return latest_df


def analyze_task_sweep(
    sweep_id: str,
    metrics_dir: str = "results/metrics",
    expected_tasks: list = None,
    output_path: str = None,
):
    """
    Analyze a sweep that varied tasks and seeds.
    
    Robustly handles:
    - Different final epochs per run (selects latest epoch per run)
    - Missing tasks (warns but continues)
    - Train vs test comparison on final epoch
    
    Args:
        sweep_id: WandB sweep ID
        metrics_dir: Base directory for metrics storage
        expected_tasks: Optional list of expected task names (for validation)
        output_path: Optional custom output path for plot
    
    Returns:
        DataFrame with latest epoch metrics per run
    """
    # Setup consistent style
    setup_style()
    
    # Load all metrics from the sweep
    print(f"Loading metrics for sweep: {sweep_id}")
    df = load_sweep_metrics(sweep_id=sweep_id, metrics_dir=metrics_dir)
    
    if df.empty:
        print(f"No metrics found for sweep {sweep_id}")
        return df
    
    print(f"Loaded {len(df)} metric records")
    print(f"Epoch range: {df['epoch'].min()} - {df['epoch'].max()}")
    
    # Check for required columns
    if 'task' not in df.columns:
        print("Warning: 'task' column not found in metrics. Task tracking may not be enabled.")
        print("Available columns:", df.columns.tolist())
        return df
    
    if 'split' not in df.columns:
        print("Warning: 'split' column not found. Train/test comparison not available.")
        return df
    
    if 'final_hard_accuracy' not in df.columns:
        print("Error: 'final_hard_accuracy' column not found.")
        print("Available columns:", df.columns.tolist())
        return df
    
    # Get unique values
    tasks = sorted(df['task'].unique())
    splits = sorted(df['split'].unique())
    runs = df['run_id'].nunique()
    
    print(f"Tasks found: {tasks}")
    print(f"Splits: {splits}")
    print(f"Runs: {runs}")
    
    # Check for missing tasks
    if expected_tasks:
        missing_tasks = set(expected_tasks) - set(tasks)
        if missing_tasks:
            print(f"\n⚠️  Warning: Expected tasks not found: {missing_tasks}")
            print("   These tasks may have failed or not completed.")
    
    # Select latest epoch per run (handles early stopping)
    print("\n=== Selecting Latest Epoch Per Run ===")
    df_latest = get_latest_epoch_per_run(df)
    print(f"Reduced to {len(df_latest)} records (latest epoch per run/split/task)")
    
    # Show epoch distribution
    epoch_stats = df_latest.groupby('task')['epoch'].agg(['min', 'max', 'mean', 'std']).round(0)
    print("\nFinal epoch statistics by task:")
    print(epoch_stats)
    
    # Summary statistics by task and split
    print("\n=== Summary Statistics by Task and Split (Latest Epoch) ===")
    summary = df_latest.groupby(['task', 'split']).agg({
        'final_hard_accuracy': ['mean', 'std', 'min', 'max', 'count'],
        'final_hard_loss': ['mean', 'std'],
    }).round(4)
    print(summary)
    
    # Create visualization
    print("\n=== Creating Visualization ===")
    
    # Create figure with single subplot
    fig, ax_acc = create_figure(
        width_type='single',
        nrows=1,
        ncols=1,
    )
    
    # Colors for splits
    color_test = COLORS.get('test', '#2E86AB')  # Blue
    color_train = COLORS.get('train', '#A23B72')  # Purple/Red
    
    # Prepare data for plotting
    # Get all tasks (sorted for consistent ordering)
    all_tasks = sorted(df_latest['task'].unique())
    n_tasks = len(all_tasks)
    x_pos = np.arange(n_tasks)
    width = 0.35  # Bar width
    
    # Determine train metric column name (may have prefix)
    train_acc_col = None
    if 'eval_no_damage_train/final_hard_accuracy' in df_latest.columns:
        train_acc_col = 'eval_no_damage_train/final_hard_accuracy'
    elif 'final_hard_accuracy' in df_latest.columns:
        # Check if train split uses same column
        train_df = df_latest[df_latest['split'] == 'train']
        if len(train_df) > 0 and 'final_hard_accuracy' in train_df.columns:
            train_acc_col = 'final_hard_accuracy'
    
    test_acc_col = 'final_hard_accuracy'
    
    # Plot 1: Accuracy by task (train vs test) - Box plots
    # Collect data for box plots
    train_data_list = []
    test_data_list = []
    train_positions = []
    test_positions = []
    
    for i, task in enumerate(all_tasks):
        task_df = df_latest[df_latest['task'] == task]
        
        # Train data
        train_data = None
        if train_acc_col and train_acc_col in task_df.columns:
            train_data = task_df[train_acc_col].dropna()
        else:
            # Try to get from split='train'
            train_split_df = task_df[task_df['split'] == 'train']
            if len(train_split_df) > 0 and test_acc_col in train_split_df.columns:
                train_data = train_split_df[test_acc_col].dropna()
        
        if train_data is not None and len(train_data) > 0:
            train_data_list.append(train_data.values)
            train_positions.append(i - width/2)
        
        # Test data
        test_split_df = task_df[task_df['split'] == 'test']
        if len(test_split_df) > 0 and test_acc_col in test_split_df.columns:
            test_data = test_split_df[test_acc_col].dropna()
            if len(test_data) > 0:
                test_data_list.append(test_data.values)
                test_positions.append(i + width/2)
    
    # Create box plots
    bp_train = None
    bp_test = None
    
    if train_data_list:
        bp_train = ax_acc.boxplot(
            train_data_list,
            positions=train_positions,
            widths=width * 0.8,
            patch_artist=True,
            tick_labels=None,
            showmeans=True,
            meanline=False,
            boxprops=dict(facecolor=color_train, alpha=0.7, linewidth=1.5),
            medianprops=dict(color='white', linewidth=2),
            whiskerprops=dict(color=color_train, linewidth=1.5),
            capprops=dict(color=color_train, linewidth=1.5),
            flierprops=dict(marker='o', markerfacecolor=color_train, 
                          markeredgecolor=color_train, markersize=4, alpha=0.5),
            meanprops=dict(marker='D', markerfacecolor='white', 
                          markeredgecolor=color_train, markersize=5, markeredgewidth=1.5)
        )
        # Add label for legend
        from matplotlib.patches import Rectangle
        train_patch = Rectangle((0, 0), 1, 1, facecolor=color_train, alpha=0.7, 
                                edgecolor=color_train, linewidth=1.5)
        ax_acc.plot([], [], color=color_train, marker='D', linestyle='None', 
                   markersize=5, markeredgecolor=color_train, 
                   markerfacecolor='white', markeredgewidth=1.5, label='Train (mean)')
    
    if test_data_list:
        bp_test = ax_acc.boxplot(
            test_data_list,
            positions=test_positions,
            widths=width * 0.8,
            patch_artist=True,
            tick_labels=None,
            showmeans=True,
            meanline=False,
            boxprops=dict(facecolor=color_test, alpha=0.7, linewidth=1.5),
            medianprops=dict(color='white', linewidth=2),
            whiskerprops=dict(color=color_test, linewidth=1.5),
            capprops=dict(color=color_test, linewidth=1.5),
            flierprops=dict(marker='o', markerfacecolor=color_test, 
                          markeredgecolor=color_test, markersize=4, alpha=0.5),
            meanprops=dict(marker='D', markerfacecolor='white', 
                          markeredgecolor=color_test, markersize=5, markeredgewidth=1.5)
        )
        # Add label for legend
        ax_acc.plot([], [], color=color_test, marker='D', linestyle='None', 
                   markersize=5, markeredgecolor=color_test, 
                   markerfacecolor='white', markeredgewidth=1.5, label='Test (mean)')
    
    # Format accuracy subplot
    format_axis(
        ax_acc,
        xlabel='Task',
        ylabel='Hard Accuracy',
        xlim=[-0.5, n_tasks - 0.5],
        ylim=[0.7, 1.05],
        grid=True,
        legend=True,
        legend_loc='lower right',
    )
    ax_acc.set_xticks(x_pos)
    ax_acc.set_xticklabels(all_tasks, rotation=45, ha='right')
    ax_acc.set_title('Final Epoch Performance by Task', fontsize=FONT_SIZES['title'])
    
    # Add count annotations below boxes
    for i, task in enumerate(all_tasks):
        task_df = df_latest[df_latest['task'] == task]
        
        # Count train samples
        train_count = 0
        if train_acc_col and train_acc_col in task_df.columns:
            train_data = task_df[train_acc_col].dropna()
            train_count = len(train_data)
        else:
            train_split_df = task_df[task_df['split'] == 'train']
            if len(train_split_df) > 0 and test_acc_col in train_split_df.columns:
                train_data = train_split_df[test_acc_col].dropna()
                train_count = len(train_data)
        
        # Count test samples
        test_count = 0
        test_split_df = task_df[task_df['split'] == 'test']
        if len(test_split_df) > 0 and test_acc_col in test_split_df.columns:
            test_data = test_split_df[test_acc_col].dropna()
            test_count = len(test_data)
        
        if train_count > 0:
            ax_acc.text(i - width/2, -0.05, f'n={train_count}', 
                       ha='center', va='top', fontsize=8, color=color_train)
        if test_count > 0:
            ax_acc.text(i + width/2, -0.05, f'n={test_count}', 
                       ha='center', va='top', fontsize=8, color=color_test)
    
    plt.tight_layout()
    
    # Save plot
    if output_path is None:
        output_dir = Path("reports/figures/multitask")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"task_sweep_analysis_{sweep_id}.png"
    
    # Convert Path to string for save_figure
    save_figure(str(output_path), fig=fig)
    print(f"\n✓ Plot saved to: {output_path}")
    plt.close()
    
    # Task-specific analysis
    print("\n=== Task-Specific Analysis (Latest Epoch) ===")
    for task in all_tasks:
        task_df = df_latest[df_latest['task'] == task]
        print(f"\n{task.upper()}:")
        print(f"  Runs: {task_df['run_id'].nunique()}")
        print(f"  Final epochs: {task_df['epoch'].min():.0f} - {task_df['epoch'].max():.0f} (mean: {task_df['epoch'].mean():.0f})")
        
        for split in splits:
            split_task_df = task_df[task_df['split'] == split]
            if len(split_task_df) > 0:
                # Determine correct column for this split
                if split == 'train' and 'eval_no_damage_train/final_hard_accuracy' in split_task_df.columns:
                    acc_col = 'eval_no_damage_train/final_hard_accuracy'
                else:
                    acc_col = 'final_hard_accuracy'
                
                if acc_col in split_task_df.columns:
                    acc_data = split_task_df[acc_col].dropna()
                    if len(acc_data) > 0:
                        acc_mean = acc_data.mean()
                        acc_std = acc_data.std()
                        acc_min = acc_data.min()
                        acc_max = acc_data.max()
                        print(f"  {split}: {acc_mean:.4f} ± {acc_std:.4f} (range: {acc_min:.4f} - {acc_max:.4f}, n={len(acc_data)})")
                    else:
                        print(f"  {split}: No valid data")
                else:
                    print(f"  {split}: Column '{acc_col}' not found")
            else:
                print(f"  {split}: No data")
    
    return df_latest


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python analyze_task_sweep.py <sweep_id> [expected_tasks...]")
        print("Example: python analyze_task_sweep.py 8z0rbg56 add xor parity")
        sys.exit(1)
    
    sweep_id = sys.argv[1]
    expected_tasks = sys.argv[2:] if len(sys.argv) > 2 else None
    
    df = analyze_task_sweep(sweep_id, expected_tasks=expected_tasks)
    
    if df is not None and not df.empty:
        print("\n=== Full DataFrame Info ===")
        print(df.info())
        print("\n=== Sample Data (Latest Epoch Per Run) ===")
        print(df.head(10))
