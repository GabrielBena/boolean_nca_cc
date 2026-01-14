"""
Plot trajectory curves from saved metrics data.

This script reads stepwise metrics from local storage (results/metrics) and generates
trajectory plots similar to plot_trajectory.py, but without requiring model evaluation.
It can plot multiple sweeps/runs on a single plot, grouped by sweep parameters like damage_prob.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
from pathlib import Path
from typing import Optional, List, Dict, Tuple
import argparse
import logging
import json
from collections import defaultdict

try:
    from omegaconf import OmegaConf
except ImportError:
    OmegaConf = None

from boolean_nca_cc.utils.metrics_storage import load_stepwise_metrics

log = logging.getLogger(__name__)


def get_run_ids_from_sweep(sweep_id: str, metrics_dir: str = "results/metrics") -> List[str]:
    """Get all run IDs from a sweep directory."""
    sweep_base = Path(metrics_dir) / f"sweep_{sweep_id}"
    if not sweep_base.exists():
        log.warning(f"Sweep directory not found: {sweep_base}")
        return []
    
    run_dirs = [d for d in sweep_base.iterdir() if d.is_dir() and d.name.startswith("run_")]
    run_ids = [d.name.replace("run_", "") for d in run_dirs]
    return run_ids


def load_run_metadata(run_id: str, sweep_id: str, metrics_dir: str = "results/metrics") -> Dict:
    """Load metadata from config_metadata.json for a run."""
    run_dir = Path(metrics_dir) / f"sweep_{sweep_id}" / f"run_{run_id}"
    metadata_path = run_dir / "config_metadata.json"
    
    if not metadata_path.exists():
        log.warning(f"Metadata file not found: {metadata_path}")
        return {}
    
    with open(metadata_path, "r") as f:
        return json.load(f)


def load_damage_config(run_id: str, sweep_id: str, metrics_dir: str = "results/metrics") -> Dict:
    """Load damage configuration from config_resolved.yaml for a run."""
    run_dir = Path(metrics_dir) / f"sweep_{sweep_id}" / f"run_{run_id}"
    config_path = run_dir / "config_resolved.yaml"
    
    if not config_path.exists():
        log.warning(f"Config file not found: {config_path}")
        return {}
    
    if OmegaConf is None:
        log.warning("OmegaConf not available, cannot load damage config")
        return {}
    
    try:
        cfg = OmegaConf.load(config_path)
        knockout_eval = cfg.get("eval", {}).get("knockout_eval", {})
        
        return {
            "damage_injection_mode": knockout_eval.get("damage_injection_mode", "single"),
            "damage_start_offset": knockout_eval.get("damage_start_offset", 0),
            "greedy_injection_recover_steps": knockout_eval.get("greedy_injection_recover_steps", 10),
            "max_damage_per_circuit": knockout_eval.get("max_damage_per_circuit", 1),
        }
    except Exception as e:
        log.warning(f"Error loading damage config: {e}")
        return {}


def aggregate_stepwise_metrics_by_parameter(
    sweep_id: str,
    group_by: str = "damage_prob",  # Parameter to group runs by
    split: str = "test",
    eval_type: Optional[str] = None,  # Filter by eval_type if provided (None = load both ko_in and ko_out)
    epoch: Optional[int] = None,  # If provided, filter by specific epoch (None = use all epochs)
    metrics_dir: str = "results/metrics",
) -> Dict[str, Dict]:
    """
    Load and aggregate stepwise metrics from all runs in a sweep, grouped by a parameter.
    
    Args:
        sweep_id: WandB sweep ID
        group_by: Parameter name to group runs by (e.g., "damage_prob")
        split: "test" or "train"
        eval_type: Optional filter by eval_type ("ko_in", "ko_out", "no_damage")
        epoch: Optional filter by specific epoch (None = use all epochs)
        metrics_dir: Base directory for metrics storage
    
    Returns:
        Dictionary mapping parameter values to aggregated metrics dicts
    """
    # Get all run IDs from the sweep
    run_ids = get_run_ids_from_sweep(sweep_id, metrics_dir)
    if not run_ids:
        log.warning(f"No runs found for sweep {sweep_id}")
        return {}
    
    log.info(f"Found {len(run_ids)} runs in sweep {sweep_id}")
    
    # Group runs by the specified parameter (extract from stepwise metrics)
    runs_by_param = defaultdict(list)
    
    for run_id in run_ids:
        try:
            # Load stepwise metrics to get the parameter value (load any eval_type for grouping)
            df = load_stepwise_metrics(
                run_id=run_id,
                metrics_dir=metrics_dir,
                format="jsonl",
                split=split,
                sweep_id=sweep_id,
                eval_type=None,  # Don't filter by eval_type when grouping
                epoch=epoch,
            )
            
            if df.empty:
                log.warning(f"No stepwise metrics found for run {run_id}, skipping")
                continue
            
            # Extract parameter value from the DataFrame
            if group_by not in df.columns:
                log.warning(f"Run {run_id} missing {group_by} column in stepwise metrics, skipping")
                continue
            
            # Get the parameter value (should be consistent across all rows)
            param_value = df[group_by].iloc[0]
            
            # Handle NaN/None values
            if pd.isna(param_value):
                log.warning(f"Run {run_id} has NaN {group_by} value, skipping")
                continue
            
            runs_by_param[param_value].append(run_id)
            
        except Exception as e:
            log.warning(f"Error loading metrics for run {run_id} to get {group_by}: {e}")
            continue
    
    log.info(f"Grouped runs by {group_by}: {dict(runs_by_param)}")
    
    # Load and aggregate metrics for each parameter value
    aggregated_metrics = {}
    
    # Determine which eval types to load
    # If eval_type is None, we'll load both ko_in and ko_out
    # The plotting function will handle filtering based on show_seen
    eval_types_to_load = ["ko_in", "ko_out"] if eval_type is None else [eval_type]
    
    for param_value, run_ids_group in runs_by_param.items():
        log.info(f"Loading metrics for {group_by}={param_value} ({len(run_ids_group)} runs)...")
        
        # Load metrics for each eval type
        metrics_by_eval_type = {}
        
        for current_eval_type in eval_types_to_load:
            all_stepwise_data = []
            
            for run_id in run_ids_group:
                try:
                    df = load_stepwise_metrics(
                        run_id=run_id,
                        metrics_dir=metrics_dir,
                        format="jsonl",
                        split=split,
                        sweep_id=sweep_id,
                        eval_type=current_eval_type,
                        epoch=epoch,
                    )
                    
                    if df.empty:
                        log.warning(f"No stepwise metrics found for run {run_id}")
                        continue
                    
                    # If epoch is specified, filter to that epoch; otherwise use the latest epoch
                    if epoch is not None:
                        df = df[df["epoch"] == epoch]
                    else:
                        # Use the latest epoch for each run
                        latest_epoch = df["epoch"].max()
                        df = df[df["epoch"] == latest_epoch]
                        log.debug(f"Using epoch {latest_epoch} for run {run_id}")
                    
                    if df.empty:
                        log.warning(f"No stepwise metrics found for run {run_id} after epoch filtering")
                        continue
                    
                    # Convert DataFrame to dict format similar to plot_trajectory.py
                    # Group by step (should be unique now after epoch filtering)
                    stepwise_dict = {}
                    for step in sorted(df["step"].unique()):
                        step_df = df[df["step"] == step]
                        # Should only have one row per step after epoch filtering, but take mean just in case
                        stepwise_dict[step] = {
                            "hard_accuracy": step_df["hard_accuracy"].mean(),
                            "soft_accuracy": step_df["soft_accuracy"].mean() if "soft_accuracy" in step_df.columns else None,
                            "soft_loss": step_df["soft_loss"].mean() if "soft_loss" in step_df.columns else None,
                            "hard_loss": step_df["hard_loss"].mean() if "hard_loss" in step_df.columns else None,
                            "hard_accuracy_std": step_df["hard_accuracy_std"].mean() if "hard_accuracy_std" in step_df.columns else None,
                        }
                    
                    # Convert to list format matching plot_trajectory.py structure
                    steps = sorted(stepwise_dict.keys())
                    metrics_dict = {
                        "step": steps,
                        "hard_accuracy": [stepwise_dict[s]["hard_accuracy"] for s in steps],
                    }
                    
                    if stepwise_dict[steps[0]]["soft_accuracy"] is not None:
                        metrics_dict["soft_accuracy"] = [stepwise_dict[s]["soft_accuracy"] for s in steps]
                    
                    # Include stored std if available
                    if stepwise_dict[steps[0]]["hard_accuracy_std"] is not None:
                        metrics_dict["hard_accuracy_std"] = [stepwise_dict[s]["hard_accuracy_std"] for s in steps]
                    
                    all_stepwise_data.append(metrics_dict)
                    
                except Exception as e:
                    log.warning(f"Error loading metrics for run {run_id} (eval_type={current_eval_type}): {e}")
                    continue
            
            if not all_stepwise_data:
                log.warning(f"No stepwise metrics found for {group_by}={param_value} (eval_type={current_eval_type})")
                continue
            
            # Aggregate across runs with the same parameter value
            # Align steps across runs
            all_steps = [m["step"] for m in all_stepwise_data]
            step_lengths = [len(s) for s in all_steps]
            
            if not step_lengths:
                log.warning(f"No valid step data for {group_by}={param_value} (eval_type={current_eval_type})")
                continue
            
            most_common_length = max(set(step_lengths), key=step_lengths.count)
            
            # Use steps from first run with most common length
            reference_steps = None
            for m in all_stepwise_data:
                if len(m["step"]) == most_common_length:
                    reference_steps = m["step"]
                    break
            
            if reference_steps is None:
                # Fallback: use first run's steps
                reference_steps = all_stepwise_data[0]["step"]
                log.warning(f"Using fallback steps for {group_by}={param_value} (eval_type={current_eval_type})")
            
            # Aggregate metrics at each step across runs
            n_steps = len(reference_steps)
            hard_accuracies_per_step = []
            hard_stds_per_step = []  # For stored within-run stds
            
            # Check if stored std is available in any run
            has_stored_std = any("hard_accuracy_std" in m for m in all_stepwise_data)
            
            for step_idx in range(n_steps):
                step_val = reference_steps[step_idx]
                step_hard_accs = []
                step_stored_stds = []
                
                for m in all_stepwise_data:
                    if step_idx < len(m["step"]):
                        try:
                            match_idx = m["step"].index(step_val)
                            step_hard_accs.append(m["hard_accuracy"][match_idx])
                            
                            # Collect stored std if available
                            if has_stored_std and "hard_accuracy_std" in m:
                                if match_idx < len(m["hard_accuracy_std"]):
                                    step_stored_stds.append(m["hard_accuracy_std"][match_idx])
                        except (ValueError, IndexError):
                            pass
                
                hard_accuracies_per_step.append(step_hard_accs)
                hard_stds_per_step.append(step_stored_stds)
            
            # Calculate mean across runs at each step
            hard_mean = [np.mean(vals) if vals else 0.0 for vals in hard_accuracies_per_step]
            
            # Use stored std (mean of within-run stds) if available, otherwise compute across-run std
            if has_stored_std and all(len(stds) > 0 for stds in hard_stds_per_step):
                # Use mean of stored within-run stds (typical within-run variability)
                hard_std = [np.mean(stds) if stds else 0.0 for stds in hard_stds_per_step]
                log.debug(f"Using stored within-run std for {group_by}={param_value} (eval_type={current_eval_type})")
            else:
                # Fallback: compute std across runs (between-run variability)
                hard_std = [np.std(vals) if vals else 0.0 for vals in hard_accuracies_per_step]
                if has_stored_std:
                    log.debug(f"Some runs missing stored std, using across-run std for {group_by}={param_value} (eval_type={current_eval_type})")
                else:
                    log.debug(f"No stored std available, using across-run std for {group_by}={param_value} (eval_type={current_eval_type})")
            
            # Load damage config from first run (assuming all runs in group have same config)
            damage_config = load_damage_config(run_ids_group[0], sweep_id, metrics_dir)
            
            metrics_by_eval_type[current_eval_type] = {
                "step": reference_steps,
                "hard_accuracy": hard_mean,
                "hard_accuracy_std": hard_std,
                "n_runs": len(all_stepwise_data),
                "damage_config": damage_config,  # Store for plotting shaded regions
            }
        
        # Store metrics by eval type (or directly if only one eval type)
        if len(eval_types_to_load) == 1:
            aggregated_metrics[param_value] = metrics_by_eval_type[eval_types_to_load[0]]
        else:
            aggregated_metrics[param_value] = metrics_by_eval_type
    
    return aggregated_metrics


def plot_trajectory_from_metrics(
    aggregated_metrics: Dict[str, Dict],
    group_by: str = "damage_prob",
    title: Optional[str] = None,
    show_legend: bool = True,
    colors: Optional[List[str]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    xlim: Optional[Tuple[float, float]] = None,
    show_shaded_regions: bool = True,
    show_seen: bool = False,
) -> plt.Figure:
    """
    Plot trajectory curves from aggregated metrics.
    
    Args:
        aggregated_metrics: Dictionary mapping parameter values to metrics dicts
        group_by: Parameter name used for grouping (for labels)
        title: Optional plot title
        show_legend: Whether to show legend
        colors: Optional list of colors for each trajectory
        ylim: Optional y-axis limits
        xlim: Optional x-axis limits
    
    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Sort parameter values for consistent ordering
    param_values = sorted(aggregated_metrics.keys())
    
    # Default color palette (can be customized)
    if colors is None:
        # Use a color palette that works well for multiple curves
        default_colors = plt.cm.viridis(np.linspace(0, 1, len(param_values)))
        colors = [mcolors.to_hex(c) for c in default_colors]
    
    # Plot each trajectory
    for idx, param_value in enumerate(param_values):
        metrics = aggregated_metrics[param_value]
        color = colors[idx % len(colors)]
        
        # Check if metrics has nested eval types (ko_in, ko_out) or is flat
        if "ko_in" in metrics or "ko_out" in metrics:
            # Plot both seen (ko_in) and unseen (ko_out)
            eval_types = []
            if show_seen and "ko_in" in metrics:
                eval_types.append(("ko_in", "-"))  # solid line for seen
            if "ko_out" in metrics:
                eval_types.append(("ko_out", "--"))  # dashed line for unseen
            
            # Only add label once per damage_prob (for the first curve)
            label_added = False
            for eval_type_key, linestyle in eval_types:
                eval_metrics = metrics[eval_type_key]
                steps = np.array(eval_metrics["step"])
                hard_accuracies = np.array(eval_metrics["hard_accuracy"])
                hard_std = np.array(eval_metrics["hard_accuracy_std"])
                n_runs = eval_metrics.get("n_runs", 1)
                
                # Only add label for the first curve (unseen if available, otherwise seen)
                label = None
                if not label_added:
                    if n_runs > 1:
                        label = f"{group_by}={param_value} (n={n_runs})"
                    else:
                        label = f"{group_by}={param_value}"
                    label_added = True
                
                # Plot mean line
                ax.plot(steps, hard_accuracies,
                        color=color,
                        linewidth=1.5,
                        linestyle=linestyle,
                        alpha=0.9,
                        label=label)
                
                # Plot error bands (std across runs) - make them more visible
                if np.any(hard_std > 0):
                    ax.fill_between(steps,
                                   hard_accuracies - hard_std,
                                   hard_accuracies + hard_std,
                                   color=color,
                                   alpha=0.25,
                                   linewidth=0)  # No edge lines for cleaner look
        else:
            # Flat structure (single eval type, backward compatibility)
            steps = np.array(metrics["step"])
            hard_accuracies = np.array(metrics["hard_accuracy"])
            hard_std = np.array(metrics["hard_accuracy_std"])
            n_runs = metrics.get("n_runs", 1)
            
            # Only show n_runs in legend if > 1
            if n_runs > 1:
                label = f"{group_by}={param_value} (n={n_runs})"
            else:
                label = f"{group_by}={param_value}"
            
            # Plot mean line
            ax.plot(steps, hard_accuracies,
                    color=color,
                    linewidth=1.5,
                    alpha=0.9,
                    label=label)
            
            # Plot error bands (std across runs) - make them more visible
            if np.any(hard_std > 0):
                ax.fill_between(steps,
                               hard_accuracies - hard_std,
                               hard_accuracies + hard_std,
                               color=color,
                               alpha=0.25,
                               linewidth=0)  # No edge lines for cleaner look
    
    # Set labels and title
    ax.set_xlabel('Message Steps', fontsize=22)
    ax.set_ylabel('Hard Accuracy', fontsize=22)
    
    if title is None:
        # Check if we have both eval types and show_seen is enabled
        has_both = any("ko_in" in m or "ko_out" in m for m in aggregated_metrics.values())
        if has_both and show_seen:
            title = 'Reconfiguration Trajectory (Seen & Unseen)'
        else:
            title = 'Reconfiguration Trajectory (Unseen)'
    ax.set_title(title, fontsize=26)
    
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.grid(True, alpha=0.3)
    
    # Set y-axis ticks
    if ylim is None:
        y_ticks = np.arange(0.5, 1.03, 0.05)
        ax.set_yticks(y_ticks)
        ax.set_ylim(0.65, 1.03)
    else:
        ax.set_ylim(ylim)
    
    # Set x-axis limits
    if xlim is not None:
        ax.set_xlim(xlim)
    else:
        # Auto-determine from data (handle both nested and flat structures)
        all_max_steps = []
        for m in aggregated_metrics.values():
            if "ko_in" in m or "ko_out" in m:
                # Nested structure
                for eval_type in ["ko_in", "ko_out"]:
                    if eval_type in m and m[eval_type].get("step"):
                        all_max_steps.append(max(m[eval_type]["step"]))
            elif m.get("step"):
                # Flat structure
                all_max_steps.append(max(m["step"]))
        if all_max_steps:
            ax.set_xlim(0, max(all_max_steps) + 5)
    
    
    # Add legend
    if show_legend:
        legend = ax.legend(loc='lower right', fontsize=14)
        legend.get_frame().set_alpha(1.0)
    
    plt.tight_layout()
    
    return fig


def main():
    """CLI for generating trajectory plots from saved metrics."""
    parser = argparse.ArgumentParser(
        description="Generate trajectory plots from saved metrics",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--sweep-id",
        type=str,
        required=True,
        help="WandB sweep ID to load metrics from",
    )
    
    parser.add_argument(
        "--group-by",
        type=str,
        default="damage_prob",
        help="Parameter to group runs by (e.g., 'damage_prob')",
    )
    
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["test", "train"],
        help="Data split to use",
    )
    
    parser.add_argument(
        "--eval-type",
        type=str,
        default=None,
        choices=["ko_in", "ko_out", "no_damage", None],
        help="Evaluation type to filter by (default: None = load both ko_in and ko_out)",
    )
    
    parser.add_argument(
        "--show-seen",
        action="store_true",
        default=True,
        help="Show seen (ko_in) patterns in addition to unseen (ko_out). Unseen will be dashed. (default: True)",
    )
    
    parser.add_argument(
        "--no-show-seen",
        dest="show_seen",
        action="store_false",
        help="Don't show seen (ko_in) patterns, only show unseen (ko_out).",
    )
    
    parser.add_argument(
        "--epoch",
        type=int,
        default=None,
        help="Specific epoch to use (default: use all epochs)",
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for figure (default: auto-generated)",
    )
    
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Image resolution",
    )
    
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Optional plot title",
    )
    
    parser.add_argument(
        "--metrics-dir",
        type=str,
        default="results/metrics",
        help="Base directory for metrics storage",
    )
    
    parser.add_argument(
        "--xlim",
        type=float,
        nargs=2,
        default=None,
        help="X-axis limits (e.g., --xlim 0 100)",
    )
    
    parser.add_argument(
        "--ylim",
        type=float,
        nargs=2,
        default=None,
        help="Y-axis limits (e.g., --ylim 0.6 1.0)",
    )
    
    parser.add_argument(
        "--no-shaded-regions",
        action="store_true",
        help="Don't show shaded regions for damage phases",
    )
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    log.info(f"Loading metrics from sweep {args.sweep_id}")
    log.info(f"  Grouping by: {args.group_by}")
    log.info(f"  Split: {args.split}")
    log.info(f"  Eval type: {args.eval_type}")
    log.info(f"  Show seen: {args.show_seen}")
    log.info(f"  Epoch: {args.epoch if args.epoch is not None else 'all'}")
    
    # Determine which eval types to load
    # If eval_type is explicitly set, use that; otherwise load ko_out and optionally ko_in
    if args.eval_type is not None:
        eval_type_to_load = args.eval_type
    else:
        # Default: load ko_out, and ko_in if --show-seen is set
        eval_type_to_load = None  # None means load both ko_in and ko_out
    
    # Load and aggregate metrics
    aggregated_metrics = aggregate_stepwise_metrics_by_parameter(
        sweep_id=args.sweep_id,
        group_by=args.group_by,
        split=args.split,
        eval_type=eval_type_to_load,
        epoch=args.epoch,
        metrics_dir=args.metrics_dir,
    )
    
    if not aggregated_metrics:
        log.error("No metrics found. Check sweep_id and parameters.")
        return
    
    log.info(f"Found metrics for {len(aggregated_metrics)} parameter values")
    for param_value, metrics in aggregated_metrics.items():
        # Handle both nested (ko_in/ko_out) and flat structures
        if "ko_in" in metrics or "ko_out" in metrics:
            # Nested structure with both eval types
            eval_info = []
            for eval_type in ["ko_in", "ko_out"]:
                if eval_type in metrics:
                    eval_metrics = metrics[eval_type]
                    n_runs = eval_metrics.get("n_runs", 0)
                    n_steps = len(eval_metrics.get("step", []))
                    eval_info.append(f"{eval_type}: {n_runs} runs, {n_steps} steps")
            log.info(f"  {args.group_by}={param_value}: {', '.join(eval_info)}")
        else:
            # Flat structure (single eval type)
            n_runs = metrics.get("n_runs", 0)
            n_steps = len(metrics.get("step", []))
            log.info(f"  {args.group_by}={param_value}: {n_runs} runs, {n_steps} steps")
    
    # Generate plot
    ylim = tuple(args.ylim) if args.ylim else None
    xlim = tuple(args.xlim) if args.xlim else None
    
    fig = plot_trajectory_from_metrics(
        aggregated_metrics=aggregated_metrics,
        group_by=args.group_by,
        title=args.title,
        ylim=ylim,
        xlim=xlim,
        show_shaded_regions=not args.no_shaded_regions,
        show_seen=args.show_seen,
    )
    
    # Determine output path
    output_path = args.output
    if output_path is None:
        output_path = f"reports/figures/graceful_perma/trajectory_sweep_{args.sweep_id}_{args.group_by}.png"
    
    # Ensure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Save figure
    fig.savefig(output_path, dpi=args.dpi, bbox_inches='tight')
    plt.close(fig)
    
    log.info(f"✓ Figure saved to: {output_path}")


if __name__ == "__main__":
    main()
