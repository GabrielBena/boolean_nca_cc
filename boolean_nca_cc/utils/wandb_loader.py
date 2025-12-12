"""
General WandB data loading utilities.

This module provides flexible functions for loading metrics, configs, and history
from WandB runs and sweeps. It supports any X/Y metric combinations and handles
both epoch-based and stepwise metrics.

Example Usage:
    # Load epoch-based metrics from a single run
    df = load_metric_pair(
        run_id="abc123",
        x_metric="eval_no_damage/epoch",
        y_metric="eval_no_damage/hard_accuracy"
    )
    
    # Load stepwise metrics (message passing steps)
    df = load_metric_pair(
        run_id="abc123",
        x_metric="eval_no_damage_steps/step",
        y_metric="eval_no_damage_steps/hard_accuracy"
    )
    
    # Load from a sweep with config values (e.g., seed)
    df = load_sweep_metrics(
        sweep_id="xyz789",
        x_metric="eval_no_damage/epoch",
        y_metric="eval_no_damage/hard_accuracy",
        config_keys=["seed", "training.learning_rate"]
    )
    
    # Discover available metrics
    metrics = list_available_metrics(run_id="abc123")
    
    # Get full history DataFrame
    history = get_run_history(run_id="abc123")
"""

import logging
from typing import Any, Optional, Union, List, Dict
import pandas as pd
import numpy as np

try:
    import wandb
    from wandb import Api
except ImportError:
    wandb = None
    Api = None
    print("Warning: wandb not available. Cannot load data from WandB.")

from omegaconf import OmegaConf

# Setup logging
log = logging.getLogger(__name__)


def get_wandb_api() -> Any:
    """Get WandB API instance, raising error if wandb is not available."""
    if wandb is None:
        raise ImportError("wandb is not installed. Please install it to use this module.")
    return Api()


def get_run_history(
    run_id: str,
    project: str = "boolean-nca-cc",
    entity: str = "marcello-barylli-growai",
    columns: Optional[List[str]] = None,
    samples: Optional[int] = None,
) -> pd.DataFrame:
    """
    Get full history DataFrame from a WandB run.
    
    IMPORTANT NOTE: When step_metric is defined (via define_metric), WandB groups
    data by the step_metric value. This means if you log the same metric multiple
    times with the same step_metric value, history() may only return the last value
    for that step. The UI panel might show all logged values, but history() groups them.
    
    Args:
        run_id: WandB run ID
        project: WandB project name
        entity: WandB entity/username
        columns: Optional list of specific columns to retrieve (None = all columns)
        samples: Optional number of samples to retrieve (None = all)
        stream: If True, use streaming mode (may get more granular data)
    
    Returns:
        DataFrame with all logged metrics. Each row is a log call.
        For epoch-based metrics: one row per epoch (when step_metric groups them).
        For stepwise metrics: multiple rows per epoch (one per step).
    """
    if wandb is None:
        raise ImportError("wandb is not available. Cannot load metrics.")
    
    try:
        api = get_wandb_api()
        run = api.run(f"{entity}/{project}/{run_id}")
        
        log.info(f"Loading history from WandB run: {run.name} (ID: {run_id})")
        
        # Get history - WandB returns a pandas DataFrame
        # Use scan_history() to ensure we get ALL data (history() may have limits)
        # NOTE: scan_history() still respects step_metric grouping, so if multiple
        # log calls have the same step_metric value, they may be aggregated
        if samples is None:
            # Use scan_history to get all data points
            if columns:
                history = pd.DataFrame(list(run.scan_history(keys=columns)))
            else:
                history = pd.DataFrame(list(run.scan_history()))
        else:
            # Use history() if samples limit is specified
            if columns:
                history = run.history(keys=columns, samples=samples)
            else:
                history = run.history(samples=samples)
        
        # Diagnostic: Check if we're missing data due to step_metric grouping
        # When step_metric is defined, WandB might group data differently
        # Try to see all columns to understand the data structure
        if len(history) > 0:
            print(f"  [Loader] History shape: {history.shape}, columns: {len(history.columns)}")
            # Show a sample of what columns exist
            eval_cols = [c for c in history.columns if 'eval_no_damage' in c]
            if eval_cols:
                print(f"  [Loader] Eval columns found: {eval_cols[:10]}{'...' if len(eval_cols) > 10 else ''}")
        
        # Add run_id as a column for multi-run scenarios
        if 'run_id' not in history.columns:
            history['run_id'] = run_id
        
        log.info(f"Loaded {len(history)} rows with {len(history.columns)} columns")
        return history
        
    except Exception as e:
        log.error(f"Error loading history from wandb run {run_id}: {e}")
        raise


def list_available_metrics(
    run_id: str,
    project: str = "boolean-nca-cc",
    entity: str = "marcello-barylli-growai",
) -> List[str]:
    """
    List all metric columns available in a run's history.
    
    Args:
        run_id: WandB run ID
        project: WandB project name
        entity: WandB entity/username
    
    Returns:
        List of metric column names available in the run's history
    """
    history = get_run_history(run_id, project=project, entity=entity, samples=1)
    # Exclude non-metric columns
    exclude_cols = {'run_id', '_wandb', '_step', '_runtime', '_timestamp'}
    metrics = [col for col in history.columns if col not in exclude_cols]
    return sorted(metrics)


def get_run_config(
    run_id: str,
    project: str = "boolean-nca-cc",
    entity: str = "marcello-barylli-growai",
    as_dict: bool = True,
) -> Union[Dict[str, Any], Any]:
    """
    Extract config/hyperparameters from a WandB run.
    
    Args:
        run_id: WandB run ID
        project: WandB project name
        entity: WandB entity/username
        as_dict: If True, return as dict. If False, return as OmegaConf object.
    
    Returns:
        Config dictionary or OmegaConf object containing all hyperparameters
    """
    if wandb is None:
        raise ImportError("wandb is not available. Cannot load config.")
    
    try:
        api = get_wandb_api()
        run = api.run(f"{entity}/{project}/{run_id}")
        
        log.info(f"Loading config from WandB run: {run.name} (ID: {run_id})")
        
        config = run.config
        
        if as_dict:
            # Convert to plain dict, handling nested OmegaConf structures
            if hasattr(config, 'to_container'):
                return OmegaConf.to_container(config, resolve=True)
            else:
                return dict(config)
        else:
            return OmegaConf.create(config)
            
    except Exception as e:
        log.error(f"Error loading config from wandb run {run_id}: {e}")
        raise


def load_metric_pair(
    run_id: str,
    x_metric: str,
    y_metric: str,
    project: str = "boolean-nca-cc",
    entity: str = "marcello-barylli-growai",
    filters: Optional[Dict[str, Any]] = None,
    include_config: bool = False,
    config_keys: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Load X/Y metric pair from a single run.
    
    This function extracts a specific X/Y metric combination from a run's history.
    It handles both epoch-based and stepwise metrics.
    
    Args:
        run_id: WandB run ID
        x_metric: Name of the metric to use as X-axis (e.g., "eval_no_damage/epoch", 
                 "eval_no_damage_steps/step")
        y_metric: Name of the metric to use as Y-axis (e.g., "eval_no_damage/hard_accuracy",
                 "eval_no_damage_steps/hard_accuracy")
        project: WandB project name
        entity: WandB entity/username
        filters: Optional dict of filters to apply to the data (e.g., {"epoch": [100, 200, 300]})
        include_config: If True, merge config values into the DataFrame
        config_keys: If include_config=True, specific config keys to include (None = all)
    
    Returns:
        DataFrame with columns [x_metric, y_metric, run_id, ...config_cols if include_config]
        Rows are filtered to only include entries where both metrics are present.
    """
    history = get_run_history(run_id, project=project, entity=entity)
    
    # Check if metrics exist
    if x_metric not in history.columns:
        raise ValueError(f"X metric '{x_metric}' not found in run {run_id}. "
                        f"Available metrics: {list_available_metrics(run_id, project, entity)[:10]}...")
    if y_metric not in history.columns:
        raise ValueError(f"Y metric '{y_metric}' not found in run {run_id}. "
                        f"Available metrics: {list_available_metrics(run_id, project, entity)[:10]}...")
    
    # Extract X/Y pair, removing rows where either is NaN
    data = history[[x_metric, y_metric, 'run_id']].copy()
    initial_data_len = len(data)
    data = data.dropna(subset=[x_metric, y_metric])
    
    # Diagnostic: Show what we found in the raw history (print for visibility)
    # IMPORTANT: When step_metric is defined, WandB may store data in a way where
    # multiple log calls with the same step_metric value get grouped/aggregated
    # The UI panel might show all logged values, but history() might only show unique step_metric values
    
    if x_metric in history.columns and y_metric in history.columns:
        has_x = history[x_metric].notna().sum()
        has_y = history[y_metric].notna().sum()
        has_both = (history[x_metric].notna() & history[y_metric].notna()).sum()
        print(f"  [Loader] Raw history for run {run_id[:8]}: {len(history)} total rows, {has_x} with {x_metric}, {has_y} with {y_metric}, {has_both} with both")
        
        if has_both > 0:
            # Show unique epoch values in raw data
            raw_epochs = sorted(history[history[x_metric].notna() & history[y_metric].notna()][x_metric].unique())
            print(f"  [Loader] Unique epochs in raw data: {raw_epochs[:10]}{'...' if len(raw_epochs) > 10 else ''}")
            
            # CRITICAL: Check if there are multiple rows with the same epoch but different values
            # This would indicate WandB is storing multiple log calls per epoch
            epoch_data = history[history[x_metric].notna() & history[y_metric].notna()].copy()
            if len(epoch_data) > 0:
                epoch_counts = epoch_data[x_metric].value_counts()
                if epoch_counts.max() > 1:
                    print(f"  [Loader] WARNING: Found {epoch_counts.max()} rows with the same epoch value!")
                    print(f"  [Loader] Epoch value counts: {dict(epoch_counts.head(10))}")
                    # Show what the different values are for the same epoch
                    for epoch_val in epoch_counts[epoch_counts > 1].index[:3]:
                        same_epoch_rows = epoch_data[epoch_data[x_metric] == epoch_val]
                        print(f"  [Loader]   Epoch {epoch_val}: {len(same_epoch_rows)} rows with y_values: {sorted(same_epoch_rows[y_metric].unique())[:5]}")
        
        # Check if there are other eval_no_damage columns that might have more epochs
        eval_cols = [c for c in history.columns if 'eval_no_damage' in c and 'epoch' in c]
        if len(eval_cols) > 1:
            print(f"  [Loader] Found multiple eval_no_damage epoch columns: {eval_cols}")
            for col in eval_cols:
                if col != x_metric:
                    unique_vals = sorted(history[history[col].notna()][col].unique())
                    print(f"    {col}: {len(unique_vals)} unique values: {unique_vals[:10]}{'...' if len(unique_vals) > 10 else ''}")
        
        # Check what other columns have epoch-like data
        epoch_like_cols = [c for c in history.columns if 'epoch' in c.lower() and history[c].notna().sum() > 0]
        if len(epoch_like_cols) > 1:
            print(f"  [Loader] Other epoch-like columns found: {epoch_like_cols[:5]}")
    
    if len(data) < initial_data_len:
        print(f"  [Loader] After dropna: {len(data)} rows (removed {initial_data_len - len(data)} rows with NaN)")
    
    # Diagnostic: Check for potential metric name conflicts
    # If loading eval_no_damage/*, exclude eval_no_damage_train/* and eval_no_damage_steps/* rows
    if 'eval_no_damage/' in x_metric and 'eval_no_damage/' in y_metric:
        # Check if train or steps variants exist in the history
        train_cols = [c for c in history.columns if 'eval_no_damage_train' in c or 'eval_no_damage_steps' in c]
        if train_cols:
            log.debug(f"Found train/stepwise columns in history: {train_cols[:5]}...")
            # Filter out rows where train/stepwise metrics are present but our target metrics might be NaN
            # This ensures we only get rows where eval_no_damage/* (not _train or _steps) is the primary metric
            # We do this by checking that our metrics are not NaN (already done) and optionally
            # that train/stepwise variants are NaN (if they exist)
            pass  # The dropna above should already handle this, but we log for visibility
    
    # Apply filters if provided
    if filters:
        for key, values in filters.items():
            if key in history.columns:
                if isinstance(values, (list, tuple)):
                    data = data[data[key].isin(values)]
                else:
                    data = data[data[key] == values]
    
    # Deduplicate: Keep only one row per (run_id, x_metric) combination
    # This handles cases where the same metric was logged multiple times at the same epoch
    initial_len = len(data)
    if initial_len > 0:
        # Show epochs before deduplication
        unique_epochs_before = sorted(data[x_metric].unique())
        print(f"  [Loader] Before deduplication: {len(data)} rows, {len(unique_epochs_before)} unique epochs: {unique_epochs_before[:10]}{'...' if len(unique_epochs_before) > 10 else ''}")
    
    data = data.drop_duplicates(subset=['run_id', x_metric], keep='first')
    if len(data) < initial_len:
        print(f"  [Loader] Removed {initial_len - len(data)} duplicate rows (same run_id and {x_metric})")
        unique_epochs_after = sorted(data[x_metric].unique())
        print(f"  [Loader] After deduplication: {len(data)} rows, {len(unique_epochs_after)} unique epochs: {unique_epochs_after[:10]}{'...' if len(unique_epochs_after) > 10 else ''}")
    
    # Add config if requested
    if include_config:
        config = get_run_config(run_id, project=project, entity=entity, as_dict=True)
        
        # Flatten nested config dict
        flat_config = _flatten_config_dict(config)
        
        # Filter to requested keys if specified
        if config_keys:
            flat_config = {k: v for k, v in flat_config.items() if k in config_keys}
        
        # Add config columns to DataFrame
        for key, value in flat_config.items():
            data[key] = value
    
    log.info(f"Loaded {len(data)} data points for {x_metric} vs {y_metric}")
    return data


def _flatten_config_dict(config: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
    """
    Flatten nested config dictionary.
    
    Example: {"training": {"lr": 0.001}} -> {"training.lr": 0.001}
    """
    items = []
    for k, v in config.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten_config_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def find_runs(
    filters: Optional[Dict[str, Any]] = None,
    sweep_id: Optional[str] = None,
    project: str = "boolean-nca-cc",
    entity: str = "marcello-barylli-growai",
    state: Optional[str] = None,
) -> List[Any]:
    """
    Find runs matching criteria.
    
    Args:
        filters: Optional dict of filters (e.g., {"config.seed": 34})
        sweep_id: Optional sweep ID to get all runs from a sweep
        project: WandB project name
        entity: WandB entity/username
        state: Optional run state filter ("running", "finished", "crashed", etc.)
    
    Returns:
        List of WandB Run objects
    """
    if wandb is None:
        raise ImportError("wandb is not available. Cannot find runs.")
    
    try:
        api = get_wandb_api()
        
        if sweep_id:
            log.info(f"Loading runs from sweep: {sweep_id}")
            sweep = api.sweep(f"{entity}/{project}/{sweep_id}")
            runs = sweep.runs
        else:
            if not filters:
                filters = {}
            log.info(f"Finding runs with filters: {filters}")
            runs = api.runs(f"{entity}/{project}", filters=filters)
        
        # Filter by state if requested
        if state:
            runs = [r for r in runs if r.state == state]
        
        log.info(f"Found {len(runs)} matching runs")
        return list(runs)
        
    except Exception as e:
        log.error(f"Error finding runs: {e}")
        raise


def load_runs_metrics(
    run_ids: List[str],
    x_metric: str,
    y_metric: str,
    project: str = "boolean-nca-cc",
    entity: str = "marcello-barylli-growai",
    include_config: bool = True,
    config_keys: Optional[List[str]] = None,
    filters: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """
    Load X/Y metrics from multiple specific runs.
    
    Args:
        run_ids: List of WandB run IDs
        x_metric: Name of the metric to use as X-axis
        y_metric: Name of the metric to use as Y-axis
        project: WandB project name
        entity: WandB entity/username
        include_config: If True, merge config values into the DataFrame
        config_keys: If include_config=True, specific config keys to include (None = all)
        filters: Optional dict of filters to apply to each run's data
    
    Returns:
        Combined DataFrame with columns [x_metric, y_metric, run_id, ...config_cols]
        Data from all runs is concatenated.
    """
    all_data = []
    
    for run_id in run_ids:
        try:
            data = load_metric_pair(
                run_id=run_id,
                x_metric=x_metric,
                y_metric=y_metric,
                project=project,
                entity=entity,
                filters=filters,
                include_config=include_config,
                config_keys=config_keys,
            )
            all_data.append(data)
        except Exception as e:
            log.warning(f"Error loading metrics from run {run_id}: {e}. Skipping.")
            continue
    
    if not all_data:
        raise ValueError(f"No data could be loaded from any of the {len(run_ids)} runs")
    
    combined = pd.concat(all_data, ignore_index=True)
    log.info(f"Combined data from {len(all_data)} runs: {len(combined)} total data points")
    return combined


def load_sweep_metrics(
    sweep_id: str,
    x_metric: str,
    y_metric: str,
    project: str = "boolean-nca-cc",
    entity: str = "marcello-barylli-growai",
    include_config: bool = True,
    config_keys: Optional[List[str]] = None,
    filters: Optional[Dict[str, Any]] = None,
    state: Optional[str] = "finished",
) -> pd.DataFrame:
    """
    Load X/Y metrics from all runs in a sweep.
    
    This is the main function for analyzing sweep results. It loads metrics from
    all runs in a sweep and merges them with config values (e.g., seed, learning_rate).
    
    Args:
        sweep_id: WandB sweep ID
        x_metric: Name of the metric to use as X-axis
        y_metric: Name of the metric to use as Y-axis
        project: WandB project name
        entity: WandB entity/username
        include_config: If True, merge config values into the DataFrame (default: True)
        config_keys: If include_config=True, specific config keys to include (None = all)
        filters: Optional dict of filters to apply to each run's data
        state: Optional run state filter ("finished", "running", etc.)
    
    Returns:
        Combined DataFrame with columns [x_metric, y_metric, run_id, ...config_cols]
        Data from all runs in the sweep is concatenated.
    """
    # Find all runs in the sweep
    runs = find_runs(sweep_id=sweep_id, project=project, entity=entity, state=state)
    
    if not runs:
        raise ValueError(f"No runs found in sweep {sweep_id}")
    
    run_ids = [run.id for run in runs]
    log.info(f"Loading metrics from {len(run_ids)} runs in sweep {sweep_id}")
    
    return load_runs_metrics(
        run_ids=run_ids,
        x_metric=x_metric,
        y_metric=y_metric,
        project=project,
        entity=entity,
        include_config=include_config,
        config_keys=config_keys,
        filters=filters,
    )


def get_run_summary(
    run_id: str,
    project: str = "boolean-nca-cc",
    entity: str = "marcello-barylli-growai",
) -> Dict[str, Any]:
    """
    Get summary metrics (final values) from a run.
    
    Args:
        run_id: WandB run ID
        project: WandB project name
        entity: WandB entity/username
    
    Returns:
        Dictionary of summary metrics (e.g., best values, final values)
    """
    if wandb is None:
        raise ImportError("wandb is not available. Cannot load summary.")
    
    try:
        api = get_wandb_api()
        run = api.run(f"{entity}/{project}/{run_id}")
        return dict(run.summary)
    except Exception as e:
        log.error(f"Error loading summary from wandb run {run_id}: {e}")
        raise

