"""
Figure generation script for boolean discovery visualization.

This script creates visualizations of inner loop trajectories for boolean function discovery.
Supports both single-run mode (load model and run evaluation) and sweep mode
(load stepwise metrics from local storage organized by sweep for statistical analysis).
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from experiments.visualization.plot_trajectory import plot_inner_loop_trajectory
from experiments.viz_base_checkpoint_test import load_model_and_data, run_eval_no_damage
from boolean_nca_cc.utils.wandb_loader import find_runs


def load_stepwise_metrics_from_wandb(
    run_id: str,
    split: str = "test",  # "test" or "train"
    project: str = "boolean-nca-cc",
    entity: str = "marcello-barylli-growai",
) -> dict:
    """
    Load stepwise metrics from WandB for a single run.
    
    Args:
        run_id: WandB run ID
        split: "test" or "train" (determines metric suffix)
        project: WandB project name
        entity: WandB entity/username
    
    Returns:
        Dictionary with stepwise metrics (step, full_map_accuracy, hard_accuracy, soft_accuracy)
    """
    import wandb
    
    api = wandb.Api()
    run = api.run(f"{entity}/{project}/{run_id}")
    
    # Determine metric prefix based on split
    metric_prefix = "eval_no_damage" if split == "test" else "eval_no_damage_train"
    
    # Load stepwise metrics
    step_metric = f"{metric_prefix}_steps/step"
    full_map_metric = f"{metric_prefix}_steps/full_map_accuracy"
    hard_metric = f"{metric_prefix}_steps/hard_accuracy"
    soft_metric = f"{metric_prefix}_steps/accuracy"
    
    # Get history (all logged metrics)
    history = run.history(keys=[step_metric, full_map_metric, hard_metric, soft_metric])
    
    if history.empty:
        print(f"Warning: No stepwise metrics found for run {run_id} (split={split})")
        print(f"  Looking for metrics: {step_metric}, {full_map_metric}, {hard_metric}, {soft_metric}")
        return None
    
    # Convert to dictionary format expected by plot_inner_loop_trajectory
    metrics = {
        "step": history[step_metric].dropna().values.tolist(),
        "full_map_accuracy": history[full_map_metric].dropna().values.tolist(),
        "hard_accuracy": history[hard_metric].dropna().values.tolist(),
        "soft_accuracy": history[soft_metric].dropna().values.tolist(),
    }
    
    return metrics


def get_run_ids_from_local_storage(
    sweep_id: str,
    metrics_dir: str = "results/metrics",
) -> list:
    """
    Get list of run IDs from local storage structure organized by sweep.
    
    Uses the local file structure: metrics_dir/sweep_{sweep_id}/run_{run_id}/
    
    Args:
        sweep_id: WandB sweep ID
        metrics_dir: Base directory for metrics storage
    
    Returns:
        List of run IDs found in the sweep directory
    """
    sweep_base = Path(metrics_dir) / f"sweep_{sweep_id}"
    
    if not sweep_base.exists():
        print(f"Sweep directory not found: {sweep_base}")
        return []
    
    # Find all run directories in this sweep
    run_dirs = [d for d in sweep_base.iterdir() if d.is_dir() and d.name.startswith("run_")]
    run_ids = [d.name.replace("run_", "") for d in run_dirs]
    
    return run_ids


def aggregate_stepwise_metrics_from_sweep(
    sweep_id: str,
    split: str = "test",
    project: str = "boolean-nca-cc",
    entity: str = "marcello-barylli-growai",
    metrics_dir: str = "results/metrics",
    use_local_storage: bool = True,  # If True, use local storage to find runs; if False, use WandB API
    epoch_filter: Optional[int] = None,  # If provided, only use metrics from this epoch
) -> dict:
    """
    Load and aggregate stepwise metrics from all runs in a sweep.
    
    Uses local storage structure (results/metrics/sweep_{sweep_id}/) to find runs,
    then loads stepwise metrics from WandB for statistical analysis.
    
    Args:
        sweep_id: WandB sweep ID
        split: "test" or "train"
        project: WandB project name
        entity: WandB entity/username
        metrics_dir: Base directory for local metrics storage
        use_local_storage: If True, use local storage structure to find runs; if False, use WandB API
        epoch_filter: Optional epoch to filter by (uses last epoch if None)
    
    Returns:
        Dictionary with aggregated stepwise metrics (mean/std across runs)
    """
    # Get run IDs from local storage structure or WandB
    if use_local_storage:
        run_ids = get_run_ids_from_local_storage(sweep_id=sweep_id, metrics_dir=metrics_dir)
        if not run_ids:
            print(f"No runs found in local storage for sweep {sweep_id}")
            print(f"  Checked: {Path(metrics_dir) / f'sweep_{sweep_id}'}")
            return None
        print(f"Found {len(run_ids)} runs in local storage for sweep {sweep_id}")
    else:
        # Fallback to WandB API
        runs = find_runs(sweep_id=sweep_id, project=project, entity=entity, state="finished")
        if not runs:
            print(f"No finished runs found in sweep {sweep_id}")
            return None
        run_ids = [run.id for run in runs]
        print(f"Found {len(run_ids)} runs via WandB API for sweep {sweep_id}")
    
    print(f"Loading stepwise metrics from {len(run_ids)} runs...")
    
    all_stepwise_data = []
    
    for run_id in run_ids:
        try:
            metrics = load_stepwise_metrics_from_wandb(
                run_id=run_id,
                split=split,
                project=project,
                entity=entity,
            )
            if metrics is not None:
                # Filter by epoch if requested
                if epoch_filter is not None:
                    # We'd need to load epoch info from WandB, for now skip this filter
                    pass
                all_stepwise_data.append(metrics)
        except Exception as e:
            print(f"Warning: Failed to load metrics for run {run_id}: {e}")
            continue
    
    if not all_stepwise_data:
        print("No stepwise metrics found for any runs in sweep")
        return None
    
    # Align steps across runs (they should be the same, but handle slight differences)
    # Use the most common step sequence
    all_steps = [m["step"] for m in all_stepwise_data]
    # Find the most common length
    step_lengths = [len(s) for s in all_steps]
    most_common_length = max(set(step_lengths), key=step_lengths.count)
    
    # Use steps from first run with most common length
    reference_steps = None
    for m in all_stepwise_data:
        if len(m["step"]) == most_common_length:
            reference_steps = m["step"]
            break
    
    if reference_steps is None:
        reference_steps = all_stepwise_data[0]["step"]
    
    # Aggregate metrics at each step
    n_steps = len(reference_steps)
    full_map_values = []
    hard_values = []
    soft_values = []
    
    for step_idx in range(n_steps):
        step_full_map = []
        step_hard = []
        step_soft = []
        
        for m in all_stepwise_data:
            if step_idx < len(m["step"]):
                # Find matching step (handle slight misalignments)
                step_val = reference_steps[step_idx]
                try:
                    match_idx = m["step"].index(step_val)
                    step_full_map.append(m["full_map_accuracy"][match_idx])
                    step_hard.append(m["hard_accuracy"][match_idx])
                    step_soft.append(m["soft_accuracy"][match_idx])
                except ValueError:
                    # Step not found, skip this run for this step
                    pass
        
        full_map_values.append(step_full_map)
        hard_values.append(step_hard)
        soft_values.append(step_soft)
    
    # Calculate mean and std
    full_map_mean = [np.mean(vals) if vals else 0.0 for vals in full_map_values]
    full_map_std = [np.std(vals) if vals else 0.0 for vals in full_map_values]
    hard_mean = [np.mean(vals) if vals else 0.0 for vals in hard_values]
    hard_std = [np.std(vals) if vals else 0.0 for vals in hard_values]
    soft_mean = [np.mean(vals) if vals else 0.0 for vals in soft_values]
    soft_std = [np.std(vals) if vals else 0.0 for vals in soft_values]
    
    # Return in format compatible with plot_inner_loop_trajectory
    # For now, return mean values (we'd need to extend the plotting function to handle std)
    return {
        "step": reference_steps,
        "full_map_accuracy": full_map_mean,
        "hard_accuracy": hard_mean,
        "soft_accuracy": soft_mean,
        # Store std for potential future use
        "_full_map_std": full_map_std,
        "_hard_std": hard_std,
        "_soft_std": soft_std,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Generate boolean discovery figure from checkpoint or sweep"
    )
    parser.add_argument(
        "--run_id",
        type=str,
        default=None,
        help="WandB run ID to load model from (single-run mode)",
    )
    parser.add_argument(
        "--sweep_id",
        type=str,
        default=None,
        help="WandB sweep ID to load metrics from (sweep mode, uses local storage structure)",
    )
    parser.add_argument(
        "--use_best_model",
        action="store_true",
        default=True,
        help="Load best model based on config checkpoint settings (default: True, single-run mode only)",
    )
    parser.add_argument(
        "--use_latest",
        action="store_true",
        help="Load latest checkpoint instead of best model (single-run mode only)",
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
        help="Output filename (without extension). If None, uses run_id or sweep_id",
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
    
    args = parser.parse_args()
    
    # Determine mode
    if args.sweep_id:
        # Sweep mode: Load stepwise metrics from WandB for all runs in sweep
        print(f"SWEEP MODE: Loading stepwise metrics from sweep {args.sweep_id}...")
        
        # Load test split metrics (uses local storage structure to find runs)
        step_metrics_test = aggregate_stepwise_metrics_from_sweep(
            sweep_id=args.sweep_id,
            split="test",
            project=args.project,
            entity=args.entity,
            use_local_storage=True,  # Use local storage structure
        )
        
        # Load train split metrics
        step_metrics_train = aggregate_stepwise_metrics_from_sweep(
            sweep_id=args.sweep_id,
            split="train",
            project=args.project,
            entity=args.entity,
            use_local_storage=True,  # Use local storage structure
        )
        
        if step_metrics_test is None:
            print("Error: Could not load stepwise metrics from sweep")
            return
        
        # If train metrics not available, use test metrics for both
        if step_metrics_train is None:
            print("Warning: Train split metrics not available, using test metrics for both subplots")
            step_metrics_train = step_metrics_test
        
        output_name = args.output_name if args.output_name else f"eval_no_damage_stepwise_sweep_{args.sweep_id}"
        
    elif args.run_id:
        # Single-run mode: Load model and run evaluation
        print(f"SINGLE-RUN MODE: Loading model from run {args.run_id}...")
        
        use_best_model = args.use_best_model and not args.use_latest
        
        # Load model and prepare data
        print("Loading model and preparing data...")
        model, config, x_train, y_train, x_test, y_test, base_wires, base_logits, layer_sizes = (
            load_model_and_data(args.run_id, use_best_model=use_best_model)
        )
        
        # Run evaluation on test data
        print("\nRunning evaluation on TEST data...")
        final_metrics_test, step_metrics_test = run_eval_no_damage(
            model=model,
            base_wires=base_wires,
            base_logits=base_logits,
            x_data=x_test,
            y_data=y_test,
            config=config,
            layer_sizes=layer_sizes,
        )
        
        # Run evaluation on train data (if split is enabled)
        if config.eval.input_split_enabled:
            print("\nRunning evaluation on TRAIN data...")
            final_metrics_train, step_metrics_train = run_eval_no_damage(
                model=model,
                base_wires=base_wires,
                base_logits=base_logits,
                x_data=x_train,
                y_data=y_train,
                config=config,
                layer_sizes=layer_sizes,
            )
        else:
            # If no split, use test metrics for both (all data is used for both)
            print("\nNo data split - using same evaluation for both subplots")
            step_metrics_train = step_metrics_test
            final_metrics_train = final_metrics_test
        
        output_name = args.output_name if args.output_name else f"eval_no_damage_stepwise_{args.run_id}"
        
    else:
        parser.error("Either --run_id or --sweep_id must be provided")
        return
    
    # Generate figure using unified plotting function
    output_path = os.path.join(args.output_dir, f"{output_name}.png")
    
    print(f"\nGenerating boolean discovery figure using unified trajectory plotter...")
    plot_inner_loop_trajectory(
        trajectory_type="boolean_discovery",
        train_metrics=step_metrics_train,
        test_metrics=step_metrics_test,
        output_path=output_path,
        title="Boolean Function Discovery",
        dpi=300,
    )


if __name__ == "__main__":
    main()

