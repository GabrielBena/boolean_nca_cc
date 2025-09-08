"""
Visualization for training progress tracking.

Creates Figure 1 from the visualization strategy: tracking generalization 
under persistent knockouts on seen patterns over training epochs.
Supports both single-mode and combined growth/regeneration visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import List, Dict, Optional, Tuple
import logging

try:
    import wandb
except ImportError:
    wandb = None

log = logging.getLogger(__name__)


def plot_knockout_regeneration_over_training(
    pattern_data: List[List],  # [epoch, pattern_id, hard_accuracy, knockout_diversity]
    fallback_metrics: Optional[Dict] = None,  # Fallback to logged metrics if pattern data unavailable
    output_path: str = "knockout_regeneration.png",
    figsize: Tuple[int, int] = (10, 6),
    dpi: int = 300,
    include_soft_accuracy: bool = False,
) -> str:
    """
    Create Figure 1: Knockout regeneration over training (IN-distribution).
    
    Args:
        pattern_data: List of data points [epoch, pattern_id, hard_accuracy, knockout_diversity]
        fallback_metrics: Dict with keys like 'epochs', 'hard_accuracies', 'soft_accuracies' 
                         for fallback when pattern data is unavailable
        output_path: Path to save the plot image
        figsize: Figure dimensions
        dpi: Image resolution
        include_soft_accuracy: Whether to include soft accuracy bands (requires soft data)
    
    Returns:
        Path to saved image file
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    if pattern_data and len(pattern_data) > 0:
        # Convert to DataFrame for easier manipulation
        df = pd.DataFrame(pattern_data, columns=['epoch', 'pattern_id', 'hard_accuracy', 'knockout_diversity'])
        
        # Group by epoch and compute mean ± std
        epoch_stats = df.groupby('epoch')['hard_accuracy'].agg(['mean', 'std']).reset_index()
        
        epochs = epoch_stats['epoch'].values
        hard_acc_means = epoch_stats['mean'].values
        hard_acc_stds = epoch_stats['std'].values
        
        # Handle NaN std values (when only one pattern per epoch)
        hard_acc_stds = np.nan_to_num(hard_acc_stds, nan=0.0)
        
        # Plot hard accuracy with error bands
        ax.plot(epochs, hard_acc_means, color='red', linewidth=2, label='Hard Accuracy')
        ax.fill_between(epochs, 
                       hard_acc_means - hard_acc_stds, 
                       hard_acc_means + hard_acc_stds, 
                       color='red', alpha=0.2, label='±1 std')
        
        # Add soft accuracy if available and requested
        if include_soft_accuracy and 'soft_accuracy' in df.columns:
            soft_epoch_stats = df.groupby('epoch')['soft_accuracy'].agg(['mean', 'std']).reset_index()
            soft_acc_means = soft_epoch_stats['mean'].values
            soft_acc_stds = np.nan_to_num(soft_epoch_stats['std'].values, nan=0.0)
            
            ax.plot(epochs, soft_acc_means, color='red', linewidth=2, label='Soft Accuracy')
            ax.fill_between(epochs, 
                           soft_acc_means - soft_acc_stds, 
                           soft_acc_means + soft_acc_stds, 
                           color='red', alpha=0.2, label='Soft ±1 std')
        
        log.info(f"Plotted knockout regeneration using pattern data: {len(epochs)} epochs, {len(df)} total patterns")
        
    elif fallback_metrics:
        # Fallback to logged metrics without error bands
        epochs = fallback_metrics.get('epochs', [])
        hard_accuracies = fallback_metrics.get('hard_accuracies', [])
        soft_accuracies = fallback_metrics.get('soft_accuracies', [])
        
        if len(epochs) != len(hard_accuracies):
            log.warning(f"Epoch and accuracy length mismatch: {len(epochs)} vs {len(hard_accuracies)}")
            min_len = min(len(epochs), len(hard_accuracies))
            epochs = epochs[:min_len]
            hard_accuracies = hard_accuracies[:min_len]
        
        # Plot hard accuracy (GNN/SA in red)
        ax.plot(epochs, hard_accuracies, color='red', linewidth=2, marker='o', 
                markersize=4, label='Hard Accuracy')
        
        # Plot soft accuracy if available
        if soft_accuracies and len(soft_accuracies) == len(epochs):
            ax.plot(epochs, soft_accuracies, color='red', linewidth=2, marker='s', 
                    markersize=4, label='Soft Accuracy')
        
        log.info(f"Plotted knockout regeneration using fallback metrics: {len(epochs)} epochs")
        
    else:
        raise ValueError("Either pattern_data or fallback_metrics must be provided")
    
    # Customize plot with much larger font sizes
    ax.set_xlabel('Training Epoch', fontsize=24)
    ax.set_ylabel('Accuracy', fontsize=24)
    ax.set_title('Knockout Regeneration Over Training (IN-distribution)', fontsize=28)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right', fontsize=18)
    
    # Set y-axis limits to 0.6
    ax.set_ylim(0.6, 1)
    
    # Add horizontal line at perfect accuracy
    ax.axhline(y=1.0, color='green', linestyle='--', alpha=0.5, label='Perfect Accuracy')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    log.info(f"Knockout regeneration plot saved to: {output_path}")
    return output_path


def extract_metrics_from_wandb_logs(
    wandb_run, 
    metric_prefix: str = "eval_ko_in"
) -> Dict:
    """
    Extract metrics from wandb run logs for fallback visualization.
    
    Args:
        wandb_run: WandB run object
        metric_prefix: Prefix for the metrics to extract (e.g., "eval_ko_in")
    
    Returns:
        Dict with epochs, hard_accuracies, soft_accuracies lists
    """
    if wandb_run is None or wandb is None:
        return {}
    
    try:
        # Get history from wandb run
        history = wandb_run.history()
        
        # Filter for knockout evaluation metrics
        ko_epochs = []
        hard_accs = []
        soft_accs = []
        
        for _, row in history.iterrows():
            epoch_key = f"{metric_prefix}/epoch"
            hard_acc_key = f"{metric_prefix}/final_hard_accuracy"
            soft_acc_key = f"{metric_prefix}/final_accuracy"
            
            if epoch_key in row and not pd.isna(row[epoch_key]):
                ko_epochs.append(int(row[epoch_key]))
                
                if hard_acc_key in row and not pd.isna(row[hard_acc_key]):
                    hard_accs.append(float(row[hard_acc_key]))
                else:
                    hard_accs.append(None)
                
                if soft_acc_key in row and not pd.isna(row[soft_acc_key]):
                    soft_accs.append(float(row[soft_acc_key]))
                else:
                    soft_accs.append(None)
        
        # Remove None values and corresponding epochs
        valid_data = [(e, h, s) for e, h, s in zip(ko_epochs, hard_accs, soft_accs) 
                     if h is not None]
        
        if valid_data:
            epochs, hard_accs, soft_accs = zip(*valid_data)
            return {
                'epochs': list(epochs),
                'hard_accuracies': list(hard_accs),
                'soft_accuracies': list(soft_accs)
            }
        else:
            return {}
            
    except Exception as e:
        log.warning(f"Error extracting metrics from wandb logs: {e}")
        return {}


def create_knockout_regeneration_plot(
    pattern_data: Optional[List[List]] = None,
    wandb_run = None,
    output_dir: str = "reports/figures",
    filename: str = "knockout_regeneration.png"
) -> str:
    """
    High-level function to create Figure 1 with automatic data source selection.
    
    Args:
        pattern_data: Optional pattern data from persistent table
        wandb_run: Optional wandb run for fallback metrics
        output_dir: Directory to save the plot
        filename: Filename for the saved plot
    
    Returns:
        Path to saved plot file
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)
    
    # Try pattern data first, fallback to wandb metrics
    fallback_metrics = None
    if wandb_run is not None:
        fallback_metrics = extract_metrics_from_wandb_logs(wandb_run, "eval_ko_in")
    
    return plot_knockout_regeneration_over_training(
        pattern_data=pattern_data,
        fallback_metrics=fallback_metrics,
        output_path=output_path
    )


def fetch_wandb_run_data(run_id: str, project: str = "boolean-nca-cc") -> Dict:
    """
    Fetch data from a WandB run by ID.
    
    Args:
        run_id: WandB run ID
        project: WandB project name
    
    Returns:
        Dict with run data including pattern data and metrics
    """
    if wandb is None:
        log.error("WandB not available. Cannot fetch run data.")
        return {}
    
    try:
        # Initialize wandb API
        api = wandb.Api()
        run = api.run(f"{project}/{run_id}")
        
        # Get run config to extract training mode
        config = run.config
        training_mode = config.get("training", {}).get("training_mode", "unknown")
        
        # Get pattern data from wandb tables if available
        pattern_data = []
        try:
            # Look for pattern performance tables
            for artifact in run.logged_artifacts():
                if "pattern_performance" in artifact.name.lower():
                    table = artifact.get("pattern_performance")
                    if hasattr(table, 'data'):
                        pattern_data = table.data
                        break
        except Exception as e:
            log.warning(f"Could not fetch pattern data from artifacts: {e}")
        
        # Get fallback metrics from run history
        fallback_metrics = extract_metrics_from_wandb_run_history(run)
        
        return {
            "run_id": run_id,
            "training_mode": training_mode,
            "pattern_data": pattern_data,
            "fallback_metrics": fallback_metrics,
            "config": config
        }
        
    except Exception as e:
        log.error(f"Error fetching WandB run {run_id}: {e}")
        return {}


def extract_metrics_from_wandb_run_history(wandb_run) -> Dict:
    """
    Extract metrics from wandb run history for fallback visualization.
    
    Args:
        wandb_run: WandB run object from API
    
    Returns:
        Dict with epochs, hard_accuracies, soft_accuracies lists
    """
    if wandb_run is None or wandb is None:
        return {}
    
    try:
        # Get history from wandb run
        history = wandb_run.history()
        
        # Filter for knockout evaluation metrics
        ko_epochs = []
        hard_accs = []
        soft_accs = []
        
        for _, row in history.iterrows():
            epoch_key = "eval_ko_in/epoch"
            hard_acc_key = "eval_ko_in/final_hard_accuracy"
            soft_acc_key = "eval_ko_in/final_accuracy"
            
            if epoch_key in row and not pd.isna(row[epoch_key]):
                ko_epochs.append(int(row[epoch_key]))
                
                if hard_acc_key in row and not pd.isna(row[hard_acc_key]):
                    hard_accs.append(float(row[hard_acc_key]))
                else:
                    hard_accs.append(None)
                
                if soft_acc_key in row and not pd.isna(row[soft_acc_key]):
                    soft_accs.append(float(row[soft_acc_key]))
                else:
                    soft_accs.append(None)
        
        # Remove None values and corresponding epochs
        valid_data = [(e, h, s) for e, h, s in zip(ko_epochs, hard_accs, soft_accs) 
                     if h is not None]
        
        if valid_data:
            epochs, hard_accs, soft_accs = zip(*valid_data)
            return {
                'epochs': list(epochs),
                'hard_accuracies': list(hard_accs),
                'soft_accuracies': list(soft_accs)
            }
        else:
            return {}
            
    except Exception as e:
        log.warning(f"Error extracting metrics from wandb run history: {e}")
        return {}


def plot_combined_growth_regeneration(
    growth_run_data: Dict,
    repair_run_data: Dict,
    output_path: str = "combined_growth_regeneration.png",
    figsize: Tuple[int, int] = (12, 8),
    dpi: int = 300,
    include_soft_accuracy: bool = False,
    bp_results: Optional[Dict] = None,
    show_bp_reference: bool = False,
) -> str:
    """
    Create combined plot showing both growth and regeneration performance.
    
    Args:
        growth_run_data: Data from growth mode run
        repair_run_data: Data from repair mode run
        output_path: Path to save the plot image
        figsize: Figure dimensions
        dpi: Image resolution
        include_soft_accuracy: Whether to include soft accuracy bands
        bp_results: Pre-computed backpropagation results (optional)
        show_bp_reference: Whether to show backpropagation reference line
    
    Returns:
        Path to saved image file
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    # Plot growth mode data
    if growth_run_data.get("pattern_data"):
        growth_df = pd.DataFrame(growth_run_data["pattern_data"], 
                               columns=['epoch', 'pattern_id', 'hard_accuracy', 'knockout_diversity'])
        growth_epoch_stats = growth_df.groupby('epoch')['hard_accuracy'].agg(['mean', 'std']).reset_index()
        
        growth_epochs = growth_epoch_stats['epoch'].values
        growth_means = growth_epoch_stats['mean'].values
        growth_stds = np.nan_to_num(growth_epoch_stats['std'].values, nan=0.0)
        
        ax.plot(growth_epochs, growth_means, color='red', linewidth=2, 
                label='Growth Mode', marker='o', markersize=3)
        ax.fill_between(growth_epochs, 
                       growth_means - growth_stds, 
                       growth_means + growth_stds, 
                       color='red', alpha=0.2)
        
    elif growth_run_data.get("fallback_metrics"):
        fallback = growth_run_data["fallback_metrics"]
        ax.plot(fallback['epochs'], fallback['hard_accuracies'], 
                color='red', linewidth=2, marker='o', markersize=4, 
                label='Growth Mode')
    
    # Plot repair mode data
    if repair_run_data.get("pattern_data"):
        repair_df = pd.DataFrame(repair_run_data["pattern_data"], 
                               columns=['epoch', 'pattern_id', 'hard_accuracy', 'knockout_diversity'])
        repair_epoch_stats = repair_df.groupby('epoch')['hard_accuracy'].agg(['mean', 'std']).reset_index()
        
        repair_epochs = repair_epoch_stats['epoch'].values
        repair_means = repair_epoch_stats['mean'].values
        repair_stds = np.nan_to_num(repair_epoch_stats['std'].values, nan=0.0)
        
        ax.plot(repair_epochs, repair_means, color='red', linewidth=2, 
                label='Repair Mode', marker='s', markersize=3)
        ax.fill_between(repair_epochs, 
                       repair_means - repair_stds, 
                       repair_means + repair_stds, 
                       color='red', alpha=0.2)
        
    elif repair_run_data.get("fallback_metrics"):
        fallback = repair_run_data["fallback_metrics"]
        ax.plot(fallback['epochs'], fallback['hard_accuracies'], 
                color='red', linewidth=2, marker='s', markersize=4, 
                label='Repair Mode')
    
    # Add backpropagation reference line if requested and results available
    if show_bp_reference and bp_results is not None:
        try:
            # Calculate mean final BP accuracy across all patterns
            bp_final_accuracies = [pattern_results["final_hard_accuracy"] 
                                 for pattern_results in bp_results["patterns_performance"]]
            bp_mean_accuracy = np.mean(bp_final_accuracies)
            
            ax.axhline(y=bp_mean_accuracy, 
                      color='blue',
                      linestyle='--',
                      linewidth=1.5,
                      alpha=0.8,
                      label=f'BP Final (mean: {bp_mean_accuracy:.3f})')
        except Exception as e:
            log.warning(f"Error adding BP reference line: {e}")
    
    # Customize plot with much larger font sizes
    ax.set_xlabel('Training Epoch', fontsize=24)
    ax.set_ylabel('Hard Accuracy', fontsize=24)
    ax.set_title('Combined Growth vs Regeneration Performance', fontsize=28)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right', fontsize=18)
    ax.set_ylim(0.6, 1)
    
    # Add horizontal line at perfect accuracy
    ax.axhline(y=1.0, color='green', linestyle='--', alpha=0.5, label='Perfect Accuracy')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    log.info(f"Combined growth/regeneration plot saved to: {output_path}")
    return output_path


def create_combined_growth_regeneration_plot(
    growth_run_id: str,
    repair_run_id: str,
    project: str = "boolean-nca-cc",
    output_dir: str = "reports/figures",
    filename: str = "combined_growth_regeneration.png"
) -> str:
    """
    High-level function to create combined growth/regeneration plot from WandB runs.
    
    Args:
        growth_run_id: WandB run ID for growth mode
        repair_run_id: WandB run ID for repair mode
        project: WandB project name
        output_dir: Directory to save the plot
        filename: Filename for the saved plot
    
    Returns:
        Path to saved plot file
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)
    
    # Fetch data from both runs
    log.info(f"Fetching data from growth run: {growth_run_id}")
    growth_data = fetch_wandb_run_data(growth_run_id, project)
    
    log.info(f"Fetching data from repair run: {repair_run_id}")
    repair_data = fetch_wandb_run_data(repair_run_id, project)
    
    if not growth_data or not repair_data:
        raise ValueError("Could not fetch data from one or both WandB runs")
    
    # Create the combined plot
    return plot_combined_growth_regeneration(
        growth_run_data=growth_data,
        repair_run_data=repair_data,
        output_path=output_path
    )


def create_eval_plot_prog(
    pattern_data: List[List],
    training_mode: str,
    wandb_run=None,
    output_dir: str = "reports/figures",
    filename: str = "training_progress.png",
    bp_results: Optional[Dict] = None,
    show_bp_reference: bool = False
) -> str:
    """
    Create a plot from current run data, with mode-aware labeling.
    
    Args:
        pattern_data: Current run pattern data
        training_mode: Current training mode ("growth" or "repair")
        wandb_run: Optional wandb run for fallback metrics
        output_dir: Directory to save the plot
        filename: Filename for the saved plot
        bp_results: Pre-computed backpropagation results (optional)
        show_bp_reference: Whether to show backpropagation reference line
    
    Returns:
        Path to saved plot file
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)
    
    # Create single-mode plot with mode-specific title and colors
    if pattern_data and len(pattern_data) > 0:
        df = pd.DataFrame(pattern_data, columns=['epoch', 'pattern_id', 'hard_accuracy', 'knockout_diversity'])
        epoch_stats = df.groupby('epoch')['hard_accuracy'].agg(['mean', 'std']).reset_index()
        
        epochs = epoch_stats['epoch'].values
        means = epoch_stats['mean'].values
        stds = np.nan_to_num(epoch_stats['std'].values, nan=0.0)
        
        # Choose colors based on mode (GNN/SA always red)
        color = 'red'  # GNN/SA is always red
        mode_label = 'Growth Mode' if training_mode == 'growth' else 'Repair Mode'
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax.plot(epochs, means, color=color, linewidth=2, label=f'{mode_label} Hard Accuracy')
        ax.fill_between(epochs, means - stds, means + stds, color=color, alpha=0.2, label='±1 std')
        
        # Add backpropagation reference line if requested and results available
        if show_bp_reference and bp_results is not None:
            try:
                # Calculate mean final BP accuracy across all patterns
                bp_final_accuracies = [pattern_results["final_hard_accuracy"] 
                                     for pattern_results in bp_results["patterns_performance"]]
                bp_mean_accuracy = np.mean(bp_final_accuracies)
                
                ax.axhline(y=bp_mean_accuracy, 
                          color='blue',
                          linestyle='--',
                          linewidth=1.5,
                          alpha=0.8,
                          label=f'BP Final (mean: {bp_mean_accuracy:.3f})')
            except Exception as e:
                log.warning(f"Error adding BP reference line: {e}")
        
        # Set font sizes - much larger increase for Figure 1
        ax.set_xlabel('Training Epoch', fontsize=24)
        ax.set_ylabel('Hard Accuracy', fontsize=24)
        ax.set_title(f'Training Progress - {mode_label}', fontsize=28)
        ax.tick_params(axis='both', which='major', labelsize=20)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='lower right', fontsize=18)
        ax.set_ylim(0.6, 1)
        ax.axhline(y=1.0, color='green', linestyle='--', alpha=0.5, label='Perfect Accuracy')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        log.info(f"Training progress plot saved to: {output_path}")
        return output_path
    
    else:
        # Fallback to original function
        return create_knockout_regeneration_plot(
            pattern_data=pattern_data,
            wandb_run=wandb_run,
            output_dir=output_dir,
            filename=filename
        )
