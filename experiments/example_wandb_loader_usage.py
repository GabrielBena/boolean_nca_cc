"""
Example usage of the general WandB data loading utility.

This script demonstrates how to use wandb_loader.py for various use cases.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from boolean_nca_cc.utils.wandb_loader import (
    load_metric_pair,
    load_sweep_metrics,
    list_available_metrics,
    get_run_history,
    find_runs,
)

# Example 1: Load epoch-based metrics from a single run
def example_single_run_epoch():
    """Load eval_no_damage metrics vs epoch from a single run."""
    run_id = "your_run_id_here"
    
    df = load_metric_pair(
        run_id=run_id,
        x_metric="eval_no_damage/epoch",
        y_metric="eval_no_damage/hard_accuracy",
        include_config=True,  # Include config values like seed, learning_rate, etc.
    )
    
    print(f"Loaded {len(df)} data points")
    print(f"Columns: {df.columns.tolist()}")
    print(f"\nFirst few rows:")
    print(df.head())
    
    return df


# Example 2: Load stepwise metrics (message passing steps)
def example_single_run_stepwise():
    """Load stepwise metrics (MP steps) from a single run."""
    run_id = "your_run_id_here"
    
    df = load_metric_pair(
        run_id=run_id,
        x_metric="eval_no_damage_steps/step",
        y_metric="eval_no_damage_steps/hard_accuracy",
        include_config=True,
    )
    
    print(f"Loaded {len(df)} data points (multiple per epoch)")
    print(f"\nSample data:")
    print(df.head(10))
    
    return df


# Example 3: Load from a sweep with config values
def example_sweep_with_seed():
    """Load metrics from all runs in a sweep, including seed config."""
    sweep_id = "your_sweep_id_here"
    
    df = load_sweep_metrics(
        sweep_id=sweep_id,
        x_metric="eval_no_damage/epoch",
        y_metric="eval_no_damage/hard_accuracy",
        include_config=True,
        config_keys=["seed", "training.learning_rate"],  # Only include specific configs
    )
    
    print(f"Loaded data from {df['run_id'].nunique()} runs")
    print(f"Total data points: {len(df)}")
    print(f"\nUnique seeds: {df['seed'].unique()}")
    print(f"\nSample data:")
    print(df.head())
    
    return df


# Example 4: Discover available metrics
def example_list_metrics():
    """List all available metrics in a run."""
    run_id = "your_run_id_here"
    
    metrics = list_available_metrics(run_id)
    
    print(f"Available metrics ({len(metrics)}):")
    for metric in metrics[:20]:  # Show first 20
        print(f"  - {metric}")
    
    # Filter for specific patterns
    eval_metrics = [m for m in metrics if "eval" in m]
    print(f"\nEval metrics ({len(eval_metrics)}):")
    for metric in eval_metrics[:10]:
        print(f"  - {metric}")
    
    return metrics


# Example 5: Get full history and explore
def example_full_history():
    """Get full history DataFrame and explore it."""
    run_id = "your_run_id_here"
    
    history = get_run_history(run_id)
    
    print(f"History shape: {history.shape}")
    print(f"Columns: {history.columns.tolist()}")
    print(f"\nFirst few rows:")
    print(history.head())
    
    # You can now do any pandas operations
    # e.g., filter by epoch, group by metric, etc.
    
    return history


# Example 6: Find runs and load from multiple
def example_find_and_load():
    """Find runs matching criteria and load their metrics."""
    # Find runs with specific config values
    runs = find_runs(
        filters={"config.seed": 34},
        state="finished",  # Only finished runs
    )
    
    run_ids = [run.id for run in runs]
    print(f"Found {len(run_ids)} runs with seed=34")
    
    # Load metrics from all found runs
    from boolean_nca_cc.utils.wandb_loader import load_runs_metrics
    
    df = load_runs_metrics(
        run_ids=run_ids,
        x_metric="eval_no_damage/epoch",
        y_metric="eval_no_damage/hard_accuracy",
        include_config=True,
    )
    
    print(f"Loaded {len(df)} data points from {len(run_ids)} runs")
    
    return df


if __name__ == "__main__":
    print("WandB Loader Usage Examples")
    print("=" * 50)
    print("\nUncomment the example you want to run:\n")
    
    # example_single_run_epoch()
    # example_single_run_stepwise()
    # example_sweep_with_seed()
    # example_list_metrics()
    # example_full_history()
    # example_find_and_load()

