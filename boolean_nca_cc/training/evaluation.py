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
from boolean_nca_cc.training.train_loop import get_loss_from_graph


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
    loss_type: str = "l4",
    bidirectional_edges: bool = True,
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
        loss_type: Loss function to use
        bidirectional_edges: Whether to use bidirectional edges

    Returns:
        Dictionary with metrics collected at each step
    """
    # Store original shapes for reconstruction
    logits_original_shapes = [logit.shape for logit in logits]

    # Calculate initial loss and accuracy using the imported function
    initial_loss, (initial_hard_loss, initial_pred, initial_pred_hard) = (
        get_loss_from_graph(logits, wires, x_data, y_data, loss_type)
    )
    # Need to compute accuracy separately if not returned by the imported get_loss_from_graph
    initial_accuracy = jp.mean(jp.round(initial_pred) == y_data)
    initial_hard_accuracy = jp.mean(jp.round(initial_pred_hard) == y_data)

    # Build initial graph from circuit
    graph = build_graph(
        logits,
        wires,
        input_n,
        arity,
        hidden_dim,
        loss_value=initial_loss,
        bidirectional_edges=bidirectional_edges,
    )

    # Initialize metric storage
    step_metrics = {
        "step": [],
        "soft_loss": [],
        "hard_loss": [],
        "soft_accuracy": [],
        "hard_accuracy": [],
        "logits_mean": [],
    }

    # Record initial metrics
    step_metrics["step"].append(0)
    step_metrics["soft_loss"].append(float(initial_loss))
    step_metrics["hard_loss"].append(float(initial_hard_loss))
    step_metrics["soft_accuracy"].append(float(initial_accuracy))
    step_metrics["hard_accuracy"].append(float(initial_hard_accuracy))
    step_metrics["logits_mean"].append(float(graph.nodes["logits"].mean()))
    # Create progress bar for evaluation
    pbar = tqdm(range(1, n_message_steps + 1), desc="Evaluating GNN steps")

    # Evaluate after each message passing step
    for step in pbar:
        # Apply one step of GNN message passing
        graph = gnn(graph)
        # print(graph.nodes["logits"].mean())

        # Extract current logits
        current_logits = extract_logits_from_graph(graph, logits_original_shapes)

        # Get loss and metrics using the imported function
        loss, (hard_loss, pred, pred_hard) = get_loss_from_graph(
            current_logits, wires, x_data, y_data, loss_type
        )
        # Compute accuracy separately
        accuracy = jp.mean(jp.round(pred) == y_data)
        hard_accuracy = jp.mean(jp.round(pred_hard) == y_data)

        # Record metrics
        step_metrics["step"].append(step)
        step_metrics["soft_loss"].append(float(loss))
        step_metrics["hard_loss"].append(float(hard_loss))
        step_metrics["soft_accuracy"].append(float(accuracy))
        step_metrics["hard_accuracy"].append(float(hard_accuracy))
        step_metrics["logits_mean"].append(float(graph.nodes["logits"].mean()))
        # Update loss value for graph
        graph = graph._replace(globals=loss)

        # Update progress bar
        pbar.set_postfix(
            {
                "Loss": f"{loss:.4f}",
                "Accuracy": f"{accuracy:.4f}",
                "Hard Acc": f"{hard_accuracy:.4f}",
            }
        )

    return step_metrics
