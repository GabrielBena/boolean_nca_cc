"""
Local metrics storage for reliable data access.

This module provides functions to save evaluation metrics to local files,
avoiding WandB's step grouping and overwriting issues.
"""

import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Any
import pandas as pd
import logging

log = logging.getLogger(__name__)


def save_config_snapshot(
    cfg: Any,  # Hydra config object or dict
    run_id: str,
    metrics_dir: str = "results/metrics",
    sweep_id: Optional[str] = None,
    model_config_path: Optional[str] = None,  # Path to model config file (e.g., configs/model/self_attention.yaml)
) -> Dict[str, str]:
    """
    Save configuration snapshot for a run to ensure reproducibility.
    
    Saves:
    1. Full resolved config as YAML (with all sweep overrides)
    2. Model config file (if provided)
    3. Config metadata (sweep_id, run_id, timestamp)
    
    Args:
        cfg: Hydra config object (DictConfig) or dict
        run_id: WandB run ID (or local run ID)
        metrics_dir: Base directory for metrics storage
        sweep_id: Optional WandB sweep ID
        model_config_path: Optional path to model config file to copy
    
    Returns:
        Dictionary with paths to saved config files
    """
    try:
        from omegaconf import OmegaConf, DictConfig
    except ImportError:
        log.warning("OmegaConf not available, skipping config snapshot")
        return {}
    
    # Determine base directory based on sweep_id
    if sweep_id:
        base_dir = Path(metrics_dir) / f"sweep_{sweep_id}"
    else:
        base_dir = Path(metrics_dir) / "standalone"
    
    # Create run-specific directory
    run_dir = base_dir / f"run_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    saved_paths = {}
    
    # 1. Save full resolved config as YAML
    try:
        if isinstance(cfg, DictConfig):
            config_yaml = OmegaConf.to_yaml(cfg, resolve=True)
        elif isinstance(cfg, dict):
            # Convert dict to DictConfig for proper YAML formatting
            config_dict = OmegaConf.create(cfg)
            config_yaml = OmegaConf.to_yaml(config_dict, resolve=True)
        else:
            log.warning(f"Unknown config type: {type(cfg)}, skipping config snapshot")
            return {}
        
        config_path = run_dir / "config_resolved.yaml"
        with open(config_path, "w") as f:
            f.write(config_yaml)
        saved_paths["config_resolved"] = str(config_path)
        log.info(f"Saved resolved config to {config_path}")
    except Exception as e:
        log.warning(f"Failed to save resolved config: {e}")
    
    # 2. Copy model config file if provided
    if model_config_path:
        try:
            model_config_src = Path(model_config_path)
            if model_config_src.exists():
                # Determine model type from config
                model_type = None
                if isinstance(cfg, DictConfig):
                    model_type = cfg.get("model", {}).get("type", "unknown")
                elif isinstance(cfg, dict):
                    model_type = cfg.get("model", {}).get("type", "unknown")
                
                # Save with descriptive name
                if model_type:
                    model_config_dst = run_dir / f"model_config_{model_type}.yaml"
                else:
                    model_config_dst = run_dir / "model_config.yaml"
                
                shutil.copy2(model_config_src, model_config_dst)
                saved_paths["model_config"] = str(model_config_dst)
                log.info(f"Saved model config to {model_config_dst}")
            else:
                log.warning(f"Model config file not found: {model_config_path}")
        except Exception as e:
            log.warning(f"Failed to copy model config: {e}")
    
    # 3. Save config metadata
    try:
        metadata = {
            "run_id": run_id,
            "sweep_id": sweep_id,
            "timestamp": datetime.now().isoformat(),
            "config_type": type(cfg).__name__,
        }
        
        # Extract key config values for quick reference
        if isinstance(cfg, DictConfig):
            metadata["task"] = cfg.get("circuit", {}).get("task", None)
            metadata["training_mode"] = cfg.get("training", {}).get("training_mode", None)
            metadata["model_type"] = cfg.get("model", {}).get("type", None)
            metadata["damage_prob"] = cfg.get("pool", {}).get("damage_prob", None)
        elif isinstance(cfg, dict):
            metadata["task"] = cfg.get("circuit", {}).get("task", None)
            metadata["training_mode"] = cfg.get("training", {}).get("training_mode", None)
            metadata["model_type"] = cfg.get("model", {}).get("type", None)
            metadata["damage_prob"] = cfg.get("pool", {}).get("damage_prob", None)
        
        metadata_path = run_dir / "config_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        saved_paths["config_metadata"] = str(metadata_path)
    except Exception as e:
        log.warning(f"Failed to save config metadata: {e}")
    
    return saved_paths


def save_eval_metrics_locally(
    metrics: Dict,
    epoch: int,
    split: str,  # "test" or "train"
    run_id: str,
    metrics_dir: str = "results/metrics",
    format: str = "jsonl",  # "jsonl" or "parquet"
    sweep_id: Optional[str] = None,  # WandB sweep ID (organizes runs by sweep)
    task: Optional[str] = None,  # Task name (e.g., "add", "xor", "parity")
    # Additional metadata fields
    training_mode: Optional[str] = None,  # "growth" or "repair"
    damage_prob: Optional[float] = None,  # Damage probability (number of gates knocked out)
    damage_type: Optional[str] = None,  # "reversible" or "permanent"
    damage_mode: Optional[str] = None,  # "greedy", "shotgun", "strip", "greedy_vocabulary"
    damage_injection_mode: Optional[str] = None,  # "single" or "multi"
    eval_type: Optional[str] = None,  # "no_damage", "ko_in" (seen), "ko_out" (unseen)
    input_split: Optional[str] = None,  # "seen" or "unseen" for input combinations
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
        training_mode: Optional training mode ("growth" or "repair")
        damage_prob: Optional damage probability (number of gates knocked out)
        damage_type: Optional damage type ("reversible" or "permanent")
        damage_mode: Optional damage mode ("greedy", "shotgun", "strip", "greedy_vocabulary")
        damage_injection_mode: Optional damage injection mode ("single" or "multi")
        eval_type: Optional evaluation type ("no_damage", "ko_in" for seen, "ko_out" for unseen)
        input_split: Optional input split type ("seen" or "unseen" for input combinations)
    
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
    
    # Extract clean metric names (remove prefixes like "eval_no_damage/", "eval_ko_in/", etc.)
    clean_metrics = {}
    for key, value in metrics.items():
        # Remove common prefixes
        clean_key = key
        for prefix in ["eval_no_damage/", "eval_no_damage_raw/", "eval_no_damage_raw_train/",
                       "eval_ko_in/", "eval_ko_out/", "eval_ko_in_train/", "eval_ko_out_train/"]:
            if clean_key.startswith(prefix):
                clean_key = clean_key.replace(prefix, "")
                break
        clean_metrics[clean_key] = float(value) if isinstance(value, (int, float)) else value
    
    # Prepare record with all metadata
    record = {
        "epoch": int(epoch),
        "split": split,
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        **clean_metrics
    }
    
    # Add optional metadata fields
    if task is not None:
        record["task"] = task
    if training_mode is not None:
        record["training_mode"] = training_mode
    if damage_prob is not None:
        record["damage_prob"] = float(damage_prob)
    if damage_type is not None:
        record["damage_type"] = damage_type
    if damage_mode is not None:
        record["damage_mode"] = damage_mode
    if damage_injection_mode is not None:
        record["damage_injection_mode"] = damage_injection_mode
    if eval_type is not None:
        record["eval_type"] = eval_type
    if input_split is not None:
        record["input_split"] = input_split
    
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


def save_stepwise_metrics_locally(
    step_metrics: Dict,  # Dict with keys like "step", "soft_loss", "hard_loss", "accuracy", etc.
    epoch: int,
    split: str,  # "test" or "train"
    run_id: str,
    metrics_dir: str = "results/metrics",
    format: str = "jsonl",  # "jsonl" or "parquet"
    sweep_id: Optional[str] = None,
    task: Optional[str] = None,
    # Additional metadata fields
    training_mode: Optional[str] = None,
    damage_prob: Optional[float] = None,
    damage_type: Optional[str] = None,
    damage_mode: Optional[str] = None,
    damage_injection_mode: Optional[str] = None,
    eval_type: Optional[str] = None,  # "no_damage", "ko_in", "ko_out"
    input_split: Optional[str] = None,
) -> str:
    """
    Save stepwise evaluation metrics to local file.
    
    This function saves step-by-step metrics (e.g., metrics at each message passing step)
    for detailed trajectory analysis.
    
    Args:
        step_metrics: Dictionary with stepwise metrics. Expected keys:
            - "step": List of step indices
            - "soft_loss", "hard_loss", "soft_accuracy", "hard_accuracy", etc.: Lists of values per step
        epoch: Training epoch
        split: "test" or "train"
        run_id: WandB run ID (or generated ID if no WandB)
        metrics_dir: Base directory for metrics storage
        format: "jsonl" (append, recommended) or "parquet" (update file)
        sweep_id: Optional WandB sweep ID
        task: Optional task name
        training_mode: Optional training mode
        damage_prob: Optional damage probability
        damage_type: Optional damage type
        damage_mode: Optional damage mode
        damage_injection_mode: Optional damage injection mode
        eval_type: Optional evaluation type
        input_split: Optional input split type
    
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
    
    # Get number of steps
    if "step" not in step_metrics:
        raise ValueError("step_metrics must contain 'step' key")
    n_steps = len(step_metrics["step"])
    
    # Prepare records (one per step)
    records = []
    for step_idx in range(n_steps):
        record = {
            "epoch": int(epoch),
            "split": split,
            "run_id": run_id,
            "timestamp": datetime.now().isoformat(),
            "step": int(step_metrics["step"][step_idx]),
        }
        
        # Add all stepwise metrics
        for key, values in step_metrics.items():
            if key != "step" and isinstance(values, (list, tuple)):
                if step_idx < len(values):
                    record[key] = float(values[step_idx]) if isinstance(values[step_idx], (int, float)) else values[step_idx]
        
        # Add optional metadata fields
        if task is not None:
            record["task"] = task
        if training_mode is not None:
            record["training_mode"] = training_mode
        if damage_prob is not None:
            record["damage_prob"] = float(damage_prob)
        if damage_type is not None:
            record["damage_type"] = damage_type
        if damage_mode is not None:
            record["damage_mode"] = damage_mode
        if damage_injection_mode is not None:
            record["damage_injection_mode"] = damage_injection_mode
        if eval_type is not None:
            record["eval_type"] = eval_type
        if input_split is not None:
            record["input_split"] = input_split
        
        records.append(record)
    
    if format == "jsonl":
        # Append to JSON Lines file
        jsonl_path = run_dir / "stepwise_metrics.jsonl"
        with open(jsonl_path, "a") as f:
            for record in records:
                f.write(json.dumps(record) + "\n")
        return str(jsonl_path)
    elif format == "parquet":
        # Load existing, append, save
        parquet_path = run_dir / "stepwise_metrics.parquet"
        df_new = pd.DataFrame(records)
        if parquet_path.exists():
            df = pd.read_parquet(parquet_path)
            df = pd.concat([df, df_new], ignore_index=True)
        else:
            df = df_new
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
    training_mode: Optional[str] = None,  # Optional filter by training mode
    eval_type: Optional[str] = None,  # Optional filter by eval type
    input_split: Optional[str] = None,  # Optional filter by input split
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
    
    if training_mode is not None:
        if "training_mode" in df.columns:
            df = df[df["training_mode"] == training_mode]
    
    if eval_type is not None:
        if "eval_type" in df.columns:
            df = df[df["eval_type"] == eval_type]
    
    if input_split is not None:
        if "input_split" in df.columns:
            df = df[df["input_split"] == input_split]
    
    return df


def load_sweep_metrics(
    sweep_id: str,
    metrics_dir: str = "results/metrics",
    format: str = "jsonl",
    split: Optional[str] = None,
    task: Optional[str] = None,  # Optional filter by task name
    training_mode: Optional[str] = None,  # Optional filter by training mode
    eval_type: Optional[str] = None,  # Optional filter by eval type
    input_split: Optional[str] = None,  # Optional filter by input split
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
                task=task,
                training_mode=training_mode,
                eval_type=eval_type,
                input_split=input_split
            )
            if not df.empty:
                all_metrics.append(df)
        except Exception as e:
            log.warning(f"Error loading metrics for run {run_id}: {e}")
    
    if not all_metrics:
        return pd.DataFrame()
    
    return pd.concat(all_metrics, ignore_index=True)


def load_stepwise_metrics(
    run_id: str,
    metrics_dir: str = "results/metrics",
    format: str = "jsonl",
    split: Optional[str] = None,
    sweep_id: Optional[str] = None,
    task: Optional[str] = None,
    training_mode: Optional[str] = None,
    eval_type: Optional[str] = None,
    input_split: Optional[str] = None,
    epoch: Optional[int] = None,  # Optional filter by specific epoch
) -> pd.DataFrame:
    """
    Load stepwise metrics for a single run.
    
    Args:
        run_id: WandB run ID (or local run ID)
        metrics_dir: Base directory for metrics storage
        format: "jsonl" or "parquet"
        split: Optional filter by "test" or "train"
        sweep_id: Optional sweep_id for direct lookup
        task: Optional filter by task name
        training_mode: Optional filter by training mode
        eval_type: Optional filter by eval type
        input_split: Optional filter by input split
        epoch: Optional filter by specific epoch
    
    Returns:
        DataFrame with stepwise metrics
    """
    # If sweep_id provided, use direct path
    if sweep_id:
        run_dir = Path(metrics_dir) / f"sweep_{sweep_id}" / f"run_{run_id}"
    else:
        # Try to find in any sweep or standalone
        run_dir = Path(metrics_dir) / "standalone" / f"run_{run_id}"
        if not run_dir.exists():
            metrics_base = Path(metrics_dir)
            if metrics_base.exists():
                for sweep_dir in metrics_base.iterdir():
                    if sweep_dir.is_dir() and sweep_dir.name.startswith("sweep_"):
                        potential_run_dir = sweep_dir / f"run_{run_id}"
                        if potential_run_dir.exists():
                            run_dir = potential_run_dir
                            break
    
    if format == "jsonl":
        jsonl_path = run_dir / "stepwise_metrics.jsonl"
        if not jsonl_path.exists():
            log.warning(f"Stepwise metrics file not found: {jsonl_path}")
            return pd.DataFrame()
        df = pd.read_json(jsonl_path, lines=True)
    elif format == "parquet":
        parquet_path = run_dir / "stepwise_metrics.parquet"
        if not parquet_path.exists():
            log.warning(f"Stepwise metrics file not found: {parquet_path}")
            return pd.DataFrame()
        df = pd.read_parquet(parquet_path)
    else:
        raise ValueError(f"Unknown format: {format}. Use 'jsonl' or 'parquet'")
    
    # Apply filters
    if split is not None:
        df = df[df["split"] == split]
    if task is not None and "task" in df.columns:
        df = df[df["task"] == task]
    if training_mode is not None and "training_mode" in df.columns:
        df = df[df["training_mode"] == training_mode]
    if eval_type is not None and "eval_type" in df.columns:
        df = df[df["eval_type"] == eval_type]
    if input_split is not None and "input_split" in df.columns:
        df = df[df["input_split"] == input_split]
    if epoch is not None:
        df = df[df["epoch"] == epoch]
    
    return df


