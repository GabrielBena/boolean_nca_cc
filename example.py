"""
Example usage of the boolean_nca_cc package for optimizing boolean circuits.

This script demonstrates how to:
1. Generate a random boolean circuit
2. Train a GNN to optimize it
3. Evaluate the GNN's performance
"""

import jax
import jax.numpy as jp
import matplotlib.pyplot as plt
from typing import List, Tuple

# Import from boolean_nca_cc package
from boolean_nca_cc import (
    CircuitGNN,
    build_graph,
    extract_logits_from_graph,
    train_gnn,
    evaluate_gnn_stepwise,
    run_gnn_scan,
)

# Import circuit generation functions from the original model module
from model import gen_circuit, run_circuit, generate_layer_sizes


def main():
    # Set a random seed for reproducibility
    key = jax.random.PRNGKey(42)

    # --- Step 1: Generate a simple dataset (XOR) ---
    print("Generating XOR dataset...")

    # XOR dataset
    x_data = jp.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=jp.float32)

    y_data = jp.array([[0], [1], [1], [0]], dtype=jp.float32)

    # --- Step 2: Generate a random boolean circuit ---
    print("Generating random boolean circuit...")

    # Circuit hyperparameters
    arity = 2  # Number of inputs per gate
    input_bits = 2
    output_bits = 1
    hidden_layers = 2

    # Generate layer sizes for the circuit
    layer_sizes = generate_layer_sizes(
        input_n=input_bits, output_n=output_bits, arity=arity, layer_n=hidden_layers
    )
    print(f"Circuit layer sizes: {layer_sizes}")

    # Generate random wiring and initial logits
    key, subkey = jax.random.split(key)
    wires, logits = gen_circuit(subkey, layer_sizes, arity=arity)

    # --- Step 3: Train a GNN to optimize the circuit ---
    print("Training GNN to optimize the circuit...")

    # GNN hyperparameters
    hidden_dim = 16
    node_mlp_features = [64, 32]
    edge_mlp_features = [64, 32]
    message_passing = True
    use_attention = False
    learning_rate = 1e-3
    epochs = 50
    n_message_steps = 20

    # Train the GNN
    training_results = train_gnn(
        layer_sizes=layer_sizes,
        x_data=x_data,
        y_data=y_data,
        arity=arity,
        hidden_dim=hidden_dim,
        message_passing=message_passing,
        node_mlp_features=node_mlp_features,
        edge_mlp_features=edge_mlp_features,
        use_attention=use_attention,
        learning_rate=learning_rate,
        epochs=epochs,
        n_message_steps=n_message_steps,
        key=int(key[0]),
        meta_learning=False,  # Train on a single circuit
    )

    # Extract the trained GNN
    trained_gnn = training_results["gnn"]

    # --- Step 4: Evaluate and visualize the results ---
    print("Evaluating GNN performance...")

    # Evaluate the GNN performance step-by-step
    eval_results = evaluate_gnn_stepwise(
        gnn=trained_gnn,
        wires=wires,
        logits=logits,
        x_data=x_data,
        y_data=y_data,
        input_n=input_bits,
        arity=arity,
        hidden_dim=hidden_dim,
        n_message_steps=50,  # Evaluate more steps to see convergence
    )

    # Plotting
    plt.figure(figsize=(12, 8))

    # Plot losses
    plt.subplot(2, 2, 1)
    plt.title("Loss During Training")
    plt.plot(training_results["losses"], label="Training Loss")
    plt.plot(training_results["hard_losses"], label="Hard Loss", linestyle="--")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot accuracies
    plt.subplot(2, 2, 2)
    plt.title("Accuracy During Training")
    plt.plot(training_results["accuracies"], label="Soft Accuracy")
    plt.plot(training_results["hard_accuracies"], label="Hard Accuracy", linestyle="--")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot evaluation losses
    plt.subplot(2, 2, 3)
    plt.title("Loss During GNN Message Passing")
    plt.plot(eval_results["step"], eval_results["soft_loss"], label="Soft Loss")
    plt.plot(
        eval_results["step"],
        eval_results["hard_loss"],
        label="Hard Loss",
        linestyle="--",
    )
    plt.xlabel("Message Passing Step")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot evaluation accuracies
    plt.subplot(2, 2, 4)
    plt.title("Accuracy During GNN Message Passing")
    plt.plot(eval_results["step"], eval_results["soft_accuracy"], label="Soft Accuracy")
    plt.plot(
        eval_results["step"],
        eval_results["hard_accuracy"],
        label="Hard Accuracy",
        linestyle="--",
    )
    plt.xlabel("Message Passing Step")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("gnn_circuit_optimization.png", dpi=150)
    print("Results saved to 'gnn_circuit_optimization.png'")

    # --- Step 5: Test the optimized circuit ---
    print("\nTesting optimized circuit:")

    # Get the original shape of logits
    logits_original_shapes = [logit.shape for logit in logits]

    # Build a graph from the circuit
    graph = build_graph(logits, wires, input_bits, arity, hidden_dim)

    # Run GNN message passing to get optimized circuit
    updated_graph = run_gnn_scan(trained_gnn, graph, n_message_steps=20)

    # Extract the optimized logits
    optimized_logits = extract_logits_from_graph(updated_graph, logits_original_shapes)

    # Test the optimized circuit on the XOR inputs
    print("\nInput | Target | Prediction (soft) | Prediction (hard)")
    print("----------------------------------------------")

    all_soft_acts = run_circuit(optimized_logits, wires, x_data)
    soft_pred = all_soft_acts[-1]

    all_hard_acts = run_circuit(optimized_logits, wires, x_data, hard=True)
    hard_pred = all_hard_acts[-1]

    for i in range(len(x_data)):
        print(
            f"{x_data[i]} | {y_data[i][0]:.1f}    | {soft_pred[i][0]:.4f}         | {hard_pred[i][0]:.1f}"
        )


if __name__ == "__main__":
    main()
