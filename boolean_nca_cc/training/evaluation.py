"""
Evaluation functions for GNN-based boolean circuit optimization.

This module provides functions for evaluating the performance of GNN
models on optimizing boolean circuits.
"""

import jax
import jax.numpy as jp
from tqdm.auto import tqdm
from typing import List, Dict, Tuple, Generator, NamedTuple

from boolean_nca_cc.models import CircuitGNN, CircuitSelfAttention
from boolean_nca_cc.utils import build_graph, extract_logits_from_graph
from boolean_nca_cc.training.train_loop import get_loss_from_graph


class StepResult(NamedTuple):
    """Result from a single step of model evaluation."""

    step: int
    loss: float
    hard_loss: float
    accuracy: float
    hard_accuracy: float
    predictions: jp.ndarray
    hard_predictions: jp.ndarray
    logits: List[jp.ndarray]
    graph: jp.ndarray  # The updated graph state


def evaluate_model_stepwise_generator(
    model: CircuitGNN | CircuitSelfAttention,
    wires: List[jp.ndarray],
    logits: List[jp.ndarray],
    x_data: jp.ndarray,
    y_data: jp.ndarray,
    input_n: int,
    arity: int = 2,
    hidden_dim: int = 16,
    max_steps: int = None,
    loss_type: str = "l4",
    bidirectional_edges: bool = True,
) -> Generator[StepResult, None, None]:
    """
    Generator that yields step-by-step evaluation results for GNN model optimization.

    This function provides the exact same computation path as training and batch evaluation,
    but yields results one step at a time. Perfect for live demos and interactive use.

    Args:
        model: Trained CircuitGNN or CircuitSelfAttention model
        wires: List of wire connection patterns
        logits: List of initial logit tensors for each layer
        x_data: Input data for evaluation
        y_data: Target output data
        input_n: Number of input nodes
        arity: Number of inputs per gate
        hidden_dim: Dimension of hidden features
        max_steps: Maximum number of steps to run (None for infinite)
        loss_type: Loss function to use
        bidirectional_edges: Whether to use bidirectional edges

    Yields:
        StepResult: Results from each step including loss, accuracy, predictions, and updated logits
    """
    # Store original shapes for reconstruction
    logits_original_shapes = [logit.shape for logit in logits]

    # Calculate initial loss and accuracy using the same function as training
    initial_loss, (initial_hard_loss, initial_pred, initial_pred_hard) = (
        get_loss_from_graph(logits, wires, x_data, y_data, loss_type)
    )
    initial_accuracy = jp.mean(jp.round(initial_pred) == y_data)
    initial_hard_accuracy = jp.mean(jp.round(initial_pred_hard) == y_data)

    # Build initial graph using the same function as training
    graph = build_graph(
        logits,
        wires,
        input_n,
        arity,
        hidden_dim,
        loss_value=initial_loss,
        bidirectional_edges=bidirectional_edges,
    )

    # Yield initial state (step 0)
    yield StepResult(
        step=0,
        loss=float(initial_loss),
        hard_loss=float(initial_hard_loss),
        accuracy=float(initial_accuracy),
        hard_accuracy=float(initial_hard_accuracy),
        predictions=initial_pred,
        hard_predictions=initial_pred_hard,
        logits=logits,
        graph=graph,
    )

    # Run optimization steps
    step = 0
    while max_steps is None or step < max_steps:
        step += 1

        # Apply one step of model processing (same as training)
        graph = model(graph)

        # Extract current logits using the same function as training
        current_logits = extract_logits_from_graph(graph, logits_original_shapes)

        # Get loss and metrics using the same function as training
        loss, (hard_loss, pred, pred_hard) = get_loss_from_graph(
            current_logits, wires, x_data, y_data, loss_type
        )

        # Compute accuracy the same way as training
        accuracy = jp.mean(jp.round(pred) == y_data)
        hard_accuracy = jp.mean(jp.round(pred_hard) == y_data)

        # Update graph globals with current loss and step (same as training)
        graph = graph._replace(globals=jp.array([loss, step], dtype=jp.float32))

        # Yield current state
        yield StepResult(
            step=step,
            loss=float(loss),
            hard_loss=float(hard_loss),
            accuracy=float(accuracy),
            hard_accuracy=float(hard_accuracy),
            predictions=pred,
            hard_predictions=pred_hard,
            logits=current_logits,
            graph=graph,
        )


def evaluate_model_stepwise(
    model: CircuitGNN | CircuitSelfAttention,
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

    This function now uses the generator implementation to ensure consistency
    with the step-by-step evaluation used in demos.

    Args:
        model: Trained CircuitGNN or CircuitSelfAttention model
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
    # Initialize metric storage
    step_metrics = {
        "step": [],
        "soft_loss": [],
        "hard_loss": [],
        "soft_accuracy": [],
        "hard_accuracy": [],
        "logits_mean": [],
    }

    # Use the generator to collect all results
    generator = evaluate_model_stepwise_generator(
        model=model,
        wires=wires,
        logits=logits,
        x_data=x_data,
        y_data=y_data,
        input_n=input_n,
        arity=arity,
        hidden_dim=hidden_dim,
        max_steps=n_message_steps,
        loss_type=loss_type,
        bidirectional_edges=bidirectional_edges,
    )

    # Create progress bar for evaluation
    pbar = tqdm(generator, total=n_message_steps + 1, desc="Evaluating model steps")

    # Collect all results
    for result in pbar:
        step_metrics["step"].append(result.step)
        step_metrics["soft_loss"].append(result.loss)
        step_metrics["hard_loss"].append(result.hard_loss)
        step_metrics["soft_accuracy"].append(result.accuracy)
        step_metrics["hard_accuracy"].append(result.hard_accuracy)
        step_metrics["logits_mean"].append(float(result.graph.nodes["logits"].mean()))

        # Update progress bar
        pbar.set_postfix(
            {
                "Loss": f"{result.loss:.4f}",
                "Accuracy": f"{result.accuracy:.4f}",
                "Hard Acc": f"{result.hard_accuracy:.4f}",
            }
        )

    return step_metrics
