"""
Figure generation script for boolean discovery visualization.

This script imports functions from viz_base_checkpoint_test.py and creates
visualizations of model evaluation metrics.
"""

import argparse
import os
import sys
import matplotlib.pyplot as plt
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from experiments.viz_base_checkpoint_test import load_model_and_data, run_eval_no_damage


def plot_eval_no_damage_stepwise(
    step_metrics,
    output_path=None,
    title="Stepwise Evaluation Metrics (No Damage)",
    figsize=(10, 6),
    dpi=300,
):
    """
    Plot stepwise readout of eval_no_damage_steps for:
    - full_map_accuracy
    - hard_accuracy
    - accuracy (soft_accuracy)
    
    Args:
        step_metrics: Dictionary with step-wise metrics from evaluation
        output_path: Path to save the plot (if None, show instead)
        title: Plot title
        figsize: Figure size tuple
        dpi: Image resolution
    """
    # Extract step data
    steps = np.array(step_metrics["step"])
    full_map_accuracy = np.array(step_metrics["full_map_accuracy"])
    hard_accuracy = np.array(step_metrics["hard_accuracy"])
    accuracy = np.array(step_metrics["soft_accuracy"])  # soft_accuracy is the "accuracy" metric
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot the three curves with larger fonts and better styling
    ax.plot(
        steps,
        full_map_accuracy,
        color='#1f77b4',  # Blue
        linewidth=2.5,
        label='Full Map Accuracy',
        marker='o',
        markersize=5,
        alpha=0.9,
        markevery=max(1, len(steps) // 20),  # Show markers every ~5% of points
    )
    ax.plot(
        steps,
        hard_accuracy,
        color='#d62728',  # Red
        linewidth=2.5,
        label='Hard Accuracy',
        marker='s',
        markersize=5,
        alpha=0.9,
        markevery=max(1, len(steps) // 20),
    )
    ax.plot(
        steps,
        accuracy,
        color='#2ca02c',  # Green
        linewidth=2,
        linestyle='--',
        label='Accuracy (Soft)',
        marker='^',
        markersize=4,
        alpha=0.8,
        markevery=max(1, len(steps) // 20),
    )
    
    # Formatting with larger fonts (matching figure1 style)
    ax.set_xlabel('Step', fontsize=24)
    ax.set_ylabel('Accuracy', fontsize=24)
    ax.set_title(title, fontsize=28, fontweight='bold')
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=18)
    ax.set_ylim([0, 1.05])  # Slightly above 1.0 to show perfect accuracy clearly
    
    # Add horizontal line at perfect accuracy
    ax.axhline(y=1.0, color='green', linestyle=':', alpha=0.5, linewidth=1.5)
    
    # Add statistics text box
    if len(steps) > 0:
        stats_lines = []
        stats_lines.append(f'Full Map Acc: {full_map_accuracy[-1]:.4f}')
        stats_lines.append(f'Hard Acc: {hard_accuracy[-1]:.4f}')
        stats_lines.append(f'Soft Acc: {accuracy[-1]:.4f}')
        stats_text = '\n'.join(stats_lines)
        ax.text(
            0.02, 0.98, stats_text,
            transform=ax.transAxes,
            fontsize=14,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        )
    
    plt.tight_layout()
    
    # Save or show
    if output_path:
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        print(f"Saved plot to: {output_path}")
        
        # Also save as PDF for better quality
        if output_path.endswith('.png'):
            pdf_path = output_path.replace('.png', '.pdf')
        else:
            pdf_path = output_path + '.pdf'
        plt.savefig(pdf_path, bbox_inches='tight')
        print(f"Saved plot (PDF) to: {pdf_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Generate boolean discovery figure from checkpoint"
    )
    parser.add_argument(
        "--run_id",
        type=str,
        required=True,
        help="WandB run ID to load model from",
    )
    parser.add_argument(
        "--use_best_model",
        action="store_true",
        default=True,
        help="Load best model based on config checkpoint settings (default: True)",
    )
    parser.add_argument(
        "--use_latest",
        action="store_true",
        help="Load latest checkpoint instead of best model",
    )
    parser.add_argument(
        "--eval_on_train",
        action="store_true",
        help="Also run evaluation on training data (when split is enabled)",
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
        help="Output filename (without extension). If None, uses run_id",
    )
    
    args = parser.parse_args()
    
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
    
    # Generate plot for test data
    output_name = args.output_name if args.output_name else f"eval_no_damage_stepwise_{args.run_id}"
    output_path = os.path.join(args.output_dir, f"{output_name}_test.png")
    
    print(f"\nGenerating stepwise plot...")
    plot_eval_no_damage_stepwise(
        step_metrics=step_metrics_test,
        output_path=output_path,
        title=f"Stepwise Evaluation Metrics (No Damage) - Test Data\nRun ID: {args.run_id}",
        dpi=300,
    )
    
    # Optionally run on train data
    if args.eval_on_train and config.eval.input_split_enabled:
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
        
        # Generate plot for train data
        output_path_train = os.path.join(args.output_dir, f"{output_name}_train.png")
        plot_eval_no_damage_stepwise(
            step_metrics=step_metrics_train,
            output_path=output_path_train,
            title=f"Stepwise Evaluation Metrics (No Damage) - Train Data\nRun ID: {args.run_id}",
            dpi=300,
        )


if __name__ == "__main__":
    main()

