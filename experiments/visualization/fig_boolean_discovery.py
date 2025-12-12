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


def plot_eval_no_damage_stepwise_comparison(
    train_metrics,
    test_metrics,
    output_path=None,
    title="Boolean Function Discovery",
    figsize=(16, 6),
    dpi=300,
):
    """
    Plot 2-subplot comparison of NCA model performance on train and test splits.
    
    Args:
        train_metrics: Dictionary with training stepwise metrics
        test_metrics: Dictionary with test stepwise metrics
        output_path: Path to save the plot
        title: Figure title
        figsize: Figure size tuple
        dpi: Image resolution
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Extract step data
    steps_train = np.array(train_metrics["step"])
    steps_test = np.array(test_metrics["step"])
    
    full_map_acc_train = np.array(train_metrics["full_map_accuracy"])
    hard_acc_train = np.array(train_metrics["hard_accuracy"])
    soft_acc_train = np.array(train_metrics["soft_accuracy"])
    
    full_map_acc_test = np.array(test_metrics["full_map_accuracy"])
    hard_acc_test = np.array(test_metrics["hard_accuracy"])
    soft_acc_test = np.array(test_metrics["soft_accuracy"])
    
    # Left subplot: Training performance
    ax1.plot(
        steps_train,
        full_map_acc_train,
        color='#1f77b4',  # Blue
        linewidth=2.5,
        label='Full Map Accuracy',
        marker='o',
        markersize=5,
        alpha=0.9,
        markevery=max(1, len(steps_train) // 20),
    )
    ax1.plot(
        steps_train,
        hard_acc_train,
        color='#d62728',  # Red
        linewidth=2.5,
        label='Hard Accuracy',
        marker='s',
        markersize=5,
        alpha=0.9,
        markevery=max(1, len(steps_train) // 20),
    )
    ax1.plot(
        steps_train,
        soft_acc_train,
        color='#2ca02c',  # Green
        linewidth=2,
        linestyle='--',
        label='Accuracy (Soft)',
        marker='^',
        markersize=4,
        alpha=0.8,
        markevery=max(1, len(steps_train) // 20),
    )
    
    ax1.set_xlabel('Step', fontsize=24)
    ax1.set_ylabel('Accuracy', fontsize=24)
    ax1.set_title('Training Performance', fontsize=24, fontweight='bold')
    ax1.tick_params(axis='both', which='major', labelsize=20)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best', fontsize=18)
    ax1.set_ylim([0, 1.05])
    ax1.axhline(y=1.0, color='green', linestyle=':', alpha=0.5, linewidth=1.5)
    
    # Add statistics text box for training
    # Commented out due to overlap with curves
    # if len(steps_train) > 0:
    #     stats_lines = []
    #     stats_lines.append(f'Full Map Acc: {full_map_acc_train[-1]:.4f}')
    #     stats_lines.append(f'Hard Acc: {hard_acc_train[-1]:.4f}')
    #     stats_lines.append(f'Soft Acc: {soft_acc_train[-1]:.4f}')
    #     stats_text = '\n'.join(stats_lines)
    #     ax1.text(
    #         0.98, 0.98, stats_text,
    #         transform=ax1.transAxes,
    #         fontsize=14,
    #         verticalalignment='top',
    #         horizontalalignment='right',
    #         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    #     )
    
    # Right subplot: Test performance
    ax2.plot(
        steps_test,
        full_map_acc_test,
        color='#1f77b4',  # Blue
        linewidth=2.5,
        label='Full Map Accuracy',
        marker='o',
        markersize=5,
        alpha=0.9,
        markevery=max(1, len(steps_test) // 20),
    )
    ax2.plot(
        steps_test,
        hard_acc_test,
        color='#d62728',  # Red
        linewidth=2.5,
        label='Hard Accuracy',
        marker='s',
        markersize=5,
        alpha=0.9,
        markevery=max(1, len(steps_test) // 20),
    )
    ax2.plot(
        steps_test,
        soft_acc_test,
        color='#2ca02c',  # Green
        linewidth=2,
        linestyle='--',
        label='Accuracy (Soft)',
        marker='^',
        markersize=4,
        alpha=0.8,
        markevery=max(1, len(steps_test) // 20),
    )
    
    ax2.set_xlabel('Step', fontsize=24)
    ax2.set_ylabel('Accuracy', fontsize=24)
    ax2.set_title('Unseen Performance', fontsize=24, fontweight='bold')
    ax2.tick_params(axis='both', which='major', labelsize=20)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='best', fontsize=18)
    ax2.set_ylim([0, 1.05])
    ax2.axhline(y=1.0, color='green', linestyle=':', alpha=0.5, linewidth=1.5)
    
    # Add statistics text box for test
    # Commented out due to overlap with curves
    # if len(steps_test) > 0:
    #     stats_lines = []
    #     stats_lines.append(f'Full Map Acc: {full_map_acc_test[-1]:.4f}')
    #     stats_lines.append(f'Hard Acc: {hard_acc_test[-1]:.4f}')
    #     stats_lines.append(f'Soft Acc: {soft_acc_test[-1]:.4f}')
    #     stats_text = '\n'.join(stats_lines)
    #     ax2.text(
    #         0.98, 0.98, stats_text,
    #         transform=ax2.transAxes,
    #         fontsize=14,
    #         verticalalignment='top',
    #         horizontalalignment='right',
    #         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    #     )
    
    # Overall title
    fig.suptitle(title, fontsize=28, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    # Save or show
    if output_path:
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        print(f"Saved plot to: {output_path}")
        
        # Also save as PDF
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
    
    # Generate 2-subplot comparison figure
    output_name = args.output_name if args.output_name else f"eval_no_damage_stepwise_{args.run_id}"
    output_path = os.path.join(args.output_dir, f"{output_name}.png")
    
    print(f"\nGenerating 2-subplot comparison figure...")
    plot_eval_no_damage_stepwise_comparison(
        train_metrics=step_metrics_train,
        test_metrics=step_metrics_test,
        output_path=output_path,
        title="Boolean Function Discovery",
        dpi=300,
    )


if __name__ == "__main__":
    main()

