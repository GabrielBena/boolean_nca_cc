"""
Figure generation script for backprop-optimized boolean discovery visualization.

This script trains a circuit using backpropagation on the training split and
visualizes performance on both training and test splits in a 2-subplot figure.
"""

import argparse
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import jax
import jax.numpy as jp
import optax
from functools import partial
from omegaconf import OmegaConf

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from boolean_nca_cc.circuits.tasks import get_task_data
from boolean_nca_cc.circuits.data_split import split_input_combinations
from boolean_nca_cc.circuits.model import gen_circuit, generate_layer_sizes
from boolean_nca_cc.circuits.train import (
    TrainState,
    train_step,
    loss_f_l4,
    loss_f_bce,
    compute_full_map_accuracy,
)


def train_backprop_simple(
    x_train,
    y_train,
    x_test,
    y_test,
    layer_sizes,
    arity,
    loss_type,
    epochs,
    learning_rate,
    weight_decay,
    optimizer_name,
    beta1,
    beta2,
    test_seed,
):
    """
    Simple backpropagation training matching existing code exactly.
    Evaluates on test set at each epoch to create generalization trajectory.
    
    Args:
        x_train: Training input data
        y_train: Training target data
        x_test: Test input data (for epoch-wise evaluation)
        y_test: Test target data (for epoch-wise evaluation)
        layer_sizes: List of (nodes, group_size) tuples
        arity: Number of inputs per gate
        loss_type: 'l4' or 'bce'
        epochs: Number of training epochs
        learning_rate: Learning rate
        weight_decay: Weight decay
        optimizer_name: 'adamw' or 'adam'
        beta1: Adam beta1 parameter
        beta2: Adam beta2 parameter
        test_seed: Random seed for circuit generation
        
    Returns:
        Dictionary with training metrics, test metrics, and final trained logits and wires
    """
    # Generate circuit (matching existing code)
    key = jax.random.PRNGKey(test_seed)
    wires, logits = gen_circuit(key, layer_sizes, arity=arity)
    
    # Setup optimizer (matching existing code)
    if optimizer_name == "adamw":
        opt = optax.adamw(
            learning_rate,
            b1=beta1,
            b2=beta2,
            weight_decay=weight_decay,
        )
    else:
        opt = optax.adam(learning_rate)
    
    state = TrainState(params=logits, opt_state=opt.init(logits))
    
    # Training loop - collect metrics
    losses = []
    hard_losses = []
    accuracies = []
    hard_accuracies = []
    full_map_accuracies = []
    
    # Test metrics at each epoch (for generalization trajectory)
    test_accuracies = []
    test_hard_accuracies = []
    test_full_map_accuracies = []
    
    # Loss function for evaluation
    loss_fn = loss_f_l4 if loss_type == "l4" else loss_f_bce
    
    # Partial function for train_step (matching existing code)
    _train_step_fn = partial(
        train_step,
        opt=opt,
        wires=wires,
        x=x_train,
        y0=y_train,
        loss_type=loss_type,
        do_train=True,
    )
    
    for i in range(epochs):
        loss, aux_metrics, new_state = _train_step_fn(state=state, step_count=i)
        state = new_state
        
        # Compute full map accuracy from predictions in aux_metrics
        # aux_metrics["act"] contains activations, last element is output layer
        full_map_acc = compute_full_map_accuracy(
            aux_metrics["act"][-1], y_train
        )
        
        accuracy = float(aux_metrics["accuracy"])
        hard_accuracy = float(aux_metrics["hard_accuracy"])
        hard_loss = float(aux_metrics["hard_loss"])
        
        losses.append(float(loss))
        hard_losses.append(hard_loss)
        accuracies.append(accuracy)
        hard_accuracies.append(hard_accuracy)
        full_map_accuracies.append(float(full_map_acc))
        
        # Evaluate on test set at this epoch (for generalization trajectory)
        _, test_aux_metrics = loss_fn(state.params, wires, x_test, y_test)
        test_full_map_acc = compute_full_map_accuracy(
            test_aux_metrics["act"][-1], y_test
        )
        test_accuracies.append(float(test_aux_metrics["accuracy"]))
        test_hard_accuracies.append(float(test_aux_metrics["hard_accuracy"]))
        test_full_map_accuracies.append(float(test_full_map_acc))
    
    # Final evaluation on training data
    loss_fn = loss_f_l4 if loss_type == "l4" else loss_f_bce
    final_loss, final_aux_metrics = loss_fn(state.params, wires, x_train, y_train)
    final_full_map_acc = compute_full_map_accuracy(
        final_aux_metrics["act"][-1], y_train
    )
    
    return {
        "losses": losses,
        "hard_losses": hard_losses,
        "accuracies": accuracies,
        "hard_accuracies": hard_accuracies,
        "full_map_accuracies": full_map_accuracies,
        "test_accuracies": test_accuracies,
        "test_hard_accuracies": test_hard_accuracies,
        "test_full_map_accuracies": test_full_map_accuracies,
        "final_logits": state.params,
        "wires": wires,
        "final_loss": float(final_loss),
        "final_accuracy": float(final_aux_metrics["accuracy"]),
        "final_hard_accuracy": float(final_aux_metrics["hard_accuracy"]),
        "final_full_map_accuracy": float(final_full_map_acc),
    }


def convert_test_metrics_to_stepwise(test_metrics, n_epochs):
    """
    Convert epoch-wise test metrics to stepwise format for visualization.
    
    Args:
        test_metrics: Dictionary with test metrics per epoch
        n_epochs: Number of epochs (for step indexing)
        
    Returns:
        Dictionary with stepwise metrics (epochs as steps)
    """
    steps = list(range(n_epochs))
    step_metrics = {
        "step": steps,
        "soft_loss": [0.0] * n_epochs,  # Not computed during epoch-wise eval
        "hard_loss": [0.0] * n_epochs,  # Not computed during epoch-wise eval
        "soft_accuracy": test_metrics["test_accuracies"],
        "hard_accuracy": test_metrics["test_hard_accuracies"],
        "full_map_accuracy": test_metrics["test_full_map_accuracies"],
    }
    
    return step_metrics


def plot_backprop_stepwise_comparison(
    train_metrics,
    test_metrics,
    output_path=None,
    title="Boolean Function Discovery (Backprop)",
    figsize=(16, 6),
    dpi=300,
):
    """
    Plot 2-subplot comparison of backprop training and test performance.
    
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
    
    ax1.set_xlabel('Epoch', fontsize=24)
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
    
    # Right subplot: Test performance (generalization trajectory over epochs)
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
    
    ax2.set_xlabel('Epoch', fontsize=24)
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
        description="Generate backprop-optimized boolean discovery figure"
    )
    parser.add_argument(
        "--run_id",
        type=str,
        default=None,
        help="WandB run ID to load config from (optional)",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default=None,
        help="Path to config YAML file (optional, defaults to configs/config.yaml)",
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
        default="backprop_discovery",
        help="Output filename (without extension, default: backprop_discovery)",
    )
    
    args = parser.parse_args()
    
    # Load config
    if args.run_id:
        # Load from wandb
        from boolean_nca_cc.training.checkpointing import load_config_from_wandb
        config, _, _ = load_config_from_wandb(
            run_id=args.run_id,
            filename="latest_checkpoint",
            select_by_best_metric=False,
        )
    elif args.config_path:
        config = OmegaConf.load(args.config_path)
    else:
        # Default to configs/config.yaml
        config_path = os.path.join(
            os.path.dirname(__file__), '../../configs/config.yaml'
        )
        config = OmegaConf.load(config_path)
    
    print("Loaded config")
    
    # Generate data and splits
    print("Generating task data...")
    x_data, y_data = get_task_data(
        task_name=config.circuit.task,
        case_n=2**config.circuit.input_bits,
        input_bits=config.circuit.input_bits,
        output_bits=config.circuit.output_bits,
    )
    
    # Split if enabled
    if config.eval.input_split_enabled:
        print(f"Splitting data: {config.eval.input_train_fraction*100:.0f}% train, "
              f"{(1-config.eval.input_train_fraction)*100:.0f}% test")
        x_train, y_train, x_test, y_test = split_input_combinations(
            x_data=x_data,
            y_data=y_data,
            train_fraction=config.eval.input_train_fraction,
            seed=config.eval.input_split_seed,
            shuffle=True,
        )
        print(f"Train set: {x_train.shape[0]} combinations, Test set: {x_test.shape[0]} combinations")
    else:
        x_train, y_train = x_data, y_data
        x_test, y_test = x_data, y_data
        print("Input split disabled - using all combinations for training and evaluation")
    
    # Generate layer sizes
    if config.circuit.layer_sizes is None:
        layer_sizes = generate_layer_sizes(
            input_n=config.circuit.input_bits,
            output_n=config.circuit.output_bits,
            arity=config.circuit.arity,
            layer_n=config.circuit.num_layers,
        )
    else:
        layer_sizes = config.circuit.layer_sizes
    
    print(f"Layer sizes: {layer_sizes}")
    
    # Train via backprop on train split (with test evaluation at each epoch)
    print("\nTraining circuit via backprop on training split...")
    print("(Evaluating on test set at each epoch for generalization trajectory)")
    backprop_config = config.get("backprop", {})
    training_results = train_backprop_simple(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        layer_sizes=layer_sizes,
        arity=config.circuit.arity,
        loss_type=config.training.loss_type,
        epochs=backprop_config.get("epochs", 200),
        learning_rate=backprop_config.get("learning_rate", 1.0),
        weight_decay=backprop_config.get("weight_decay", 1e-1),
        optimizer_name=backprop_config.get("optimizer", "adamw"),
        beta1=backprop_config.get("beta1", 0.8),
        beta2=backprop_config.get("beta2", 0.8),
        test_seed=config.test_seed,
    )
    
    print(f"Training complete. Final accuracy: {training_results['final_accuracy']:.4f}")
    
    # Convert training metrics to stepwise format (epochs as steps)
    train_stepwise = {
        "step": list(range(len(training_results["accuracies"]))),
        "soft_loss": training_results["losses"],
        "hard_loss": training_results["hard_losses"],
        "soft_accuracy": training_results["accuracies"],
        "hard_accuracy": training_results["hard_accuracies"],
        "full_map_accuracy": training_results["full_map_accuracies"],
    }
    
    # Convert test metrics to stepwise format (epochs as steps - generalization trajectory)
    print("\nConverting test metrics to stepwise format...")
    test_stepwise = convert_test_metrics_to_stepwise(
        test_metrics=training_results,
        n_epochs=len(training_results["accuracies"]),
    )
    
    print(f"Test evaluation complete. Final accuracy: {test_stepwise['soft_accuracy'][-1]:.4f}")
    
    # Generate plot
    output_path = os.path.join(args.output_dir, f"{args.output_name}.png")
    print(f"\nGenerating comparison plot...")
    plot_backprop_stepwise_comparison(
        train_metrics=train_stepwise,
        test_metrics=test_stepwise,
        output_path=output_path,
        title="Boolean Function Discovery (Backprop)",
        dpi=300,
    )


if __name__ == "__main__":
    main()
