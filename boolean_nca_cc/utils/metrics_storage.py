"""
Local metrics storage for reliable data access.

This module provides functions to save evaluation metrics to local files,
avoiding WandB's step grouping and overwriting issues.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional
import pandas as pd
import logging

log = logging.getLogger(__name__)


def save_eval_metrics_locally(
    metrics: Dict,
    epoch: int,
    split: str,  # "test" or "train"
    run_id: str,
    metrics_dir: str = "results/metrics",
    format: str = "jsonl",  # "jsonl" or "parquet"
    sweep_id: Optional[str] = None,  # WandB sweep ID (organizes runs by sweep)
    task: Optional[str] = None,  # Task name (e.g., "add", "xor", "parity")
) -> str:
    """
    Save evaluation metrics to local file for reliable data access.
    
    This function saves metrics to a local file, avoiding WandB's step grouping
    and overwriting issues. Metrics are saved in a structured format that can be
    easily loaded for plotting and analysis.
    
    Metrics are organized by sweep: runs in a sweep are stored under
    `metrics_dir/sweep_{sweep_id}/run_{run_id}/`, while standalone runs
    are stored under `metrics_dir/standalone/run_{run_id}/`.
    
    Args:
        metrics: Dictionary of metrics (can include both grouped and raw metric names)
        epoch: Training epoch
        split: "test" or "train" 
        run_id: WandB run ID (or generated ID if no WandB)
        metrics_dir: Base directory for metrics storage
        format: "jsonl" (append, recommended) or "parquet" (update file)
        sweep_id: Optional WandB sweep ID to organize runs by sweep
        task: Optional task name (e.g., "add", "xor", "parity") for task-aware analysis
    
    Returns:
        Path to the saved metrics file
    """
    # Determine base directory based on sweep_id
    if sweep_id:
        base_dir = Path(metrics_dir) / f"sweep_{sweep_id}"
    else:
        base_dir = Path(metrics_dir) / "standalone"
    
    # Create run-specific directory
    run_dir = base_dir / f"run_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract clean metric names (remove prefixes like "eval_no_damage/" or "eval_no_damage_raw/")
    clean_metrics = {}
    for key, value in metrics.items():
        # Remove common prefixes
        clean_key = key
        for prefix in ["eval_no_damage/", "eval_no_damage_raw/", "eval_no_damage_raw_train/"]:
            if clean_key.startswith(prefix):
                clean_key = clean_key.replace(prefix, "")
                break
        clean_metrics[clean_key] = float(value) if isinstance(value, (int, float)) else value
    
    # Prepare record
    record = {
        "epoch": int(epoch),
        "split": split,
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        **clean_metrics
    }
    
    # Add task if provided (for task-aware analysis)
    if task is not None:
        record["task"] = task
    
    if format == "jsonl":
        # Append to JSON Lines file
        jsonl_path = run_dir / "eval_metrics.jsonl"
        with open(jsonl_path, "a") as f:
            f.write(json.dumps(record) + "\n")
        return str(jsonl_path)
    elif format == "parquet":
        # Load existing, append, save
        parquet_path = run_dir / "eval_metrics.parquet"
        if parquet_path.exists():
            df = pd.read_parquet(parquet_path)
            df = pd.concat([df, pd.DataFrame([record])], ignore_index=True)
        else:
            df = pd.DataFrame([record])
        df.to_parquet(parquet_path, index=False)
        return str(parquet_path)
    else:
        raise ValueError(f"Unknown format: {format}. Use 'jsonl' or 'parquet'")


def load_run_metrics(
    run_id: str,
    metrics_dir: str = "results/metrics",
    format: str = "jsonl",
    split: Optional[str] = None,  # Filter by "test" or "train"
    sweep_id: Optional[str] = None,  # Optional sweep_id for direct lookup
    task: Optional[str] = None,  # Optional filter by task name
) -> pd.DataFrame:
    """
    Load all metrics for a single run.
    
    Supports both new sweep-based structure and old flat structure for backward compatibility.
    
    Args:
        run_id: WandB run ID (or local run ID)
        metrics_dir: Base directory for metrics storage
        format: "jsonl" or "parquet"
        split: Optional filter by "test" or "train"
        sweep_id: Optional sweep_id for direct lookup (more efficient)
        task: Optional filter by task name (e.g., "add", "xor", "parity")
    
    Returns:
        DataFrame with all metrics for the run
    """
    # If sweep_id provided, use direct path
    if sweep_id:
        run_dir = Path(metrics_dir) / f"sweep_{sweep_id}" / f"run_{run_id}"
    else:
        # Try to find in any sweep or standalone, or old flat structure
        # First check standalone
        run_dir = Path(metrics_dir) / "standalone" / f"run_{run_id}"
        if not run_dir.exists():
            # Search through all sweeps
            metrics_base = Path(metrics_dir)
            if metrics_base.exists():
                for sweep_dir in metrics_base.iterdir():
                    if sweep_dir.is_dir() and sweep_dir.name.startswith("sweep_"):
                        potential_run_dir = sweep_dir / f"run_{run_id}"
                        if potential_run_dir.exists():
                            run_dir = potential_run_dir
                            break
                else:
                    # Fallback to old flat structure for backward compatibility
                    run_dir = Path(metrics_dir) / f"run_{run_id}"
    
    if format == "jsonl":
        jsonl_path = run_dir / "eval_metrics.jsonl"
        if not jsonl_path.exists():
            log.warning(f"Metrics file not found: {jsonl_path}")
            return pd.DataFrame()
        df = pd.read_json(jsonl_path, lines=True)
    elif format == "parquet":
        parquet_path = run_dir / "eval_metrics.parquet"
        if not parquet_path.exists():
            log.warning(f"Metrics file not found: {parquet_path}")
            return pd.DataFrame()
        df = pd.read_parquet(parquet_path)
    else:
        raise ValueError(f"Unknown format: {format}. Use 'jsonl' or 'parquet'")
    
    if split is not None:
        df = df[df["split"] == split]
    
    if task is not None:
        # Only filter if task column exists (for backward compatibility with old records)
        if "task" in df.columns:
            df = df[df["task"] == task]
    
    return df


def load_sweep_metrics(
    sweep_id: str,
    metrics_dir: str = "results/metrics",
    format: str = "jsonl",
    split: Optional[str] = None,
    task: Optional[str] = None,  # Optional filter by task name
) -> pd.DataFrame:
    """
    Load metrics for all runs in a sweep from local storage.
    
    This function loads all runs from the sweep's subdirectory:
    `metrics_dir/sweep_{sweep_id}/run_{run_id}/`
    
    Args:
        sweep_id: WandB sweep ID
        metrics_dir: Base directory for metrics storage
        format: "jsonl" or "parquet"
        split: Optional filter by "test" or "train"
        task: Optional filter by task name (e.g., "add", "xor", "parity")
    
    Returns:
        DataFrame with all metrics for all runs in the sweep
    """
    sweep_base = Path(metrics_dir) / f"sweep_{sweep_id}"
    
    if not sweep_base.exists():
        log.warning(f"Sweep directory not found: {sweep_base}")
        return pd.DataFrame()
    
    # Find all run directories in this sweep
    run_dirs = [d for d in sweep_base.iterdir() if d.is_dir() and d.name.startswith("run_")]
    
    if not run_dirs:
        log.warning(f"No run directories found in sweep: {sweep_base}")
        return pd.DataFrame()
    
    all_metrics = []
    for run_dir in run_dirs:
        run_id = run_dir.name.replace("run_", "")
        try:
            # Use sweep_id for efficient direct lookup
            df = load_run_metrics(
                run_id=run_id,
                metrics_dir=metrics_dir,
                format=format,
                split=split,
                sweep_id=sweep_id,
                task=task
            )
            if not df.empty:
                all_metrics.append(df)
        except Exception as e:
            log.warning(f"Error loading metrics for run {run_id}: {e}")
    
    if not all_metrics:
        return pd.DataFrame()
    
    return pd.concat(all_metrics, ignore_index=True)


