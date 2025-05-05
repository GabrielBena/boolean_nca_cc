"""
Evaluation functions for GNN-based boolean circuit optimization.

This module provides functions for evaluating the performance of GNN
models on optimizing boolean circuits.
"""

import jax
import jax.numpy as jp
from tqdm.auto import tqdm
from typing import List, Dict, Tuple

from boolean_nca_cc.models import CircuitGNN, run_gnn_scan
from boolean_nca_cc.utils import build_graph, extract_logits_from_graph
from model import run_circuit


def evaluate_gnn_stepwise(
    gnn: CircuitGNN,
    wires: List[jp.ndarray],
    logits: List[jp.ndarray],
    x_data: jp.ndarray,
    y_data: jp.ndarray,
    input_n: int,
    arity: int = 2,
    hidden_dim: int = 16,
    n_message_steps: int = 100,
) -> Dict:
    """
    Evaluate GNN performance by running message passing steps one by one
    and collecting metrics at each step.
    
    Args:
        gnn: Trained CircuitGNN model
        wires: List of wire connection patterns
        logits: List of initial logit tensors for each layer
        x_data: Input data for evaluation
        y_data: Target output data
        input_n: Number of input nodes
        arity: Number of inputs per gate
        hidden_dim: Dimension of hidden features
        n_message_steps: Maximum number of message passing steps to run
        
    Returns:
        Dictionary with metrics collected at each step
    """
    # Store original shapes for reconstruction
    logits_original_shapes = [logit.shape for logit in logits]

    # Build initial graph from circuit
    graph = build_graph(logits, wires, input_n, arity, hidden_dim)

    # Initialize metric storage
    step_metrics = {
        "step": [],
        "soft_loss": [],
        "hard_loss": [],
        "soft_accuracy": [],
        "hard_accuracy": [],
    }

    # Evaluate initial circuit (step 0)
    initial_acts = run_circuit(logits, wires, x_data)
    initial_pred = initial_acts[-1]
    initial_hard_acts = run_circuit(logits, wires, x_data, hard=True)
    initial_hard_pred = initial_hard_acts[-1]

    soft_loss = jp.mean((initial_pred - y_data) ** 2)
    hard_loss = jp.mean((initial_hard_pred - y_data) ** 2)
    soft_accuracy = jp.mean(jp.round(initial_pred) == y_data)
    hard_accuracy = jp.mean(jp.round(initial_hard_pred) == y_data)

    # Record initial metrics
    step_metrics["step"].append(0)
    step_metrics["soft_loss"].append(float(soft_loss))
    step_metrics["hard_loss"].append(float(hard_loss))
    step_metrics["soft_accuracy"].append(float(soft_accuracy))
    step_metrics["hard_accuracy"].append(float(hard_accuracy))

    # Create progress bar for evaluation
    pbar = tqdm(range(1, n_message_steps + 1), desc="Evaluating GNN steps")

    # Evaluate after each message passing step
    for step in pbar:
        # Apply one step of GNN message passing
        graph = gnn(graph)

        # Extract current logits
        current_logits = extract_logits_from_graph(graph, logits_original_shapes)

        # Evaluate circuit with current logits
        acts = run_circuit(current_logits, wires, x_data)
        pred = acts[-1]
        hard_acts = run_circuit(current_logits, wires, x_data, hard=True)
        hard_pred = hard_acts[-1]

        # Calculate metrics
        soft_loss = jp.mean((pred - y_data) ** 2)
        hard_loss = jp.mean((hard_pred - y_data) ** 2)
        soft_accuracy = jp.mean(jp.round(pred) == y_data)
        hard_accuracy = jp.mean(jp.round(hard_pred) == y_data)

        # Record metrics
        step_metrics["step"].append(step)
        step_metrics["soft_loss"].append(float(soft_loss))
        step_metrics["hard_loss"].append(float(hard_loss))
        step_metrics["soft_accuracy"].append(float(soft_accuracy))
        step_metrics["hard_accuracy"].append(float(hard_accuracy))

        # Update progress bar
        pbar.set_postfix(
            {
                "Loss": f"{soft_loss:.4f}",
                "Accuracy": f"{soft_accuracy:.4f}",
                "Hard Acc": f"{hard_accuracy:.4f}",
            }
        )

    return step_metrics 