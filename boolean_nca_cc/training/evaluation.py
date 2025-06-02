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
from boolean_nca_cc.circuits.model import run_circuit
from boolean_nca_cc.circuits.train import (
    res2loss,
    binary_cross_entropy,
    compute_accuracy,
)


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


def _get_loss_from_wires_logits(logits, wires, x, y_target, loss_type: str):
    """
    Calculate loss and metrics from circuit logits - exact copy from training loop.
    This ensures perfect consistency between training and evaluation/demo.
    """
    # Run circuit and calculate loss
    acts = run_circuit(logits, wires, x)
    pred = acts[-1]
    acts_hard = run_circuit(logits, wires, x, hard=True)
    pred_hard = acts_hard[-1]

    if loss_type == "bce":
        loss = binary_cross_entropy(pred, y_target)
        hard_loss = binary_cross_entropy(pred_hard, y_target)
    elif loss_type == "l4":
        res = pred - y_target
        hard_res = pred_hard - y_target
        loss = res2loss(res, power=4)
        hard_loss = res2loss(hard_res, power=4)
    elif loss_type == "l2":
        res = pred - y_target
        hard_res = pred_hard - y_target
        loss = res2loss(res, power=2)
        hard_loss = res2loss(hard_res, power=2)
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")

    accuracy = compute_accuracy(pred, y_target)
    hard_accuracy = compute_accuracy(pred_hard, y_target)

    return loss, (hard_loss, pred, pred_hard, accuracy, hard_accuracy)


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

    This function provides EXACTLY the same computation path as the training loop,
    including using the same loss function, graph initialization, and step tracking.
    Perfect for live demos and interactive use with zero discrepancy from training.

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
    # Store original shapes for reconstruction (EXACTLY like training)
    logits_original_shapes = [logit.shape for logit in logits]

    # Calculate initial loss using the EXACT same function as training
    (
        initial_loss,
        (
            initial_hard_loss,
            initial_pred,
            initial_pred_hard,
            initial_accuracy,
            initial_hard_accuracy,
        ),
    ) = _get_loss_from_wires_logits(logits, wires, x_data, y_data, loss_type)

    # Build initial graph using the same function as training
    # Initialize with update_steps = 0 (exactly like training pool initialization)
    graph = build_graph(
        logits,
        wires,
        input_n,
        arity,
        hidden_dim,
        loss_value=initial_loss,
        bidirectional_edges=bidirectional_edges,
    )

    # Initialize graph globals with [loss, update_steps] exactly like training
    current_update_steps = 0
    graph = graph._replace(
        globals=jp.array([initial_loss, current_update_steps], dtype=jp.float32)
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

    # Run optimization steps (EXACTLY like the training loop)
    step = 0
    while max_steps is None or step < max_steps:
        step += 1

        # Extract the current update_steps count from graph globals (EXACTLY like training)
        current_update_steps = 0
        if graph.globals is not None and graph.globals.shape[-1] > 1:
            current_update_steps = graph.globals[..., 1]

        # Apply one step of model processing (EXACTLY like training inner loop)
        # Note: training does multiple steps in a batch, but we do one at a time for live demo
        updated_graph = model(graph)

        # Extract current logits using the EXACT same function as training
        current_logits = extract_logits_from_graph(
            updated_graph, logits_original_shapes
        )

        # Get loss and metrics using the EXACT same function as training
        loss, (hard_loss, pred, pred_hard, accuracy, hard_accuracy) = (
            _get_loss_from_wires_logits(
                current_logits, wires, x_data, y_data, loss_type
            )
        )

        # Update with the computed loss and incremented update_steps (EXACTLY like training)
        final_update_steps = current_update_steps + 1  # We increment by 1 per step
        graph = updated_graph._replace(
            globals=jp.array([loss, final_update_steps], dtype=jp.float32)
        )

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
