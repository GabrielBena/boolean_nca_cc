"""
Evaluation functions for GNN-based boolean circuit optimization.

This module provides functions for evaluating the performance of GNN
models on optimizing boolean circuits.
"""

from collections.abc import Generator
from typing import NamedTuple

import jax
import jax.numpy as jp
from tqdm.auto import tqdm

from boolean_nca_cc.circuits.model import run_circuit
from boolean_nca_cc.circuits.train import (
    binary_cross_entropy,
    compute_accuracy,
    res2loss,
)
from boolean_nca_cc.models import CircuitGNN, CircuitSelfAttention
from boolean_nca_cc.utils import (
    # build_graph,
    extract_logits_from_graph,
    update_output_node_loss,
)
from boolean_nca_cc.utils.configured_graph_builder import configured_build_graph as build_graph


class StepResult(NamedTuple):
    """Result from a single step of model evaluation."""

    step: int
    loss: float
    hard_loss: float
    accuracy: float
    hard_accuracy: float
    predictions: jp.ndarray
    hard_predictions: jp.ndarray
    logits: list[jp.ndarray]
    graph: jp.ndarray  # The updated graph state


# Function to run a circuit and calculate loss
def get_loss_from_wires_logits(logits, wires, x, y_target, loss_type: str):
    # Run circuit and calculate loss
    acts = run_circuit(logits, wires, x)
    pred = acts[-1]
    acts_hard = run_circuit(logits, wires, x, hard=True)
    pred_hard = acts_hard[-1]

    # Always compute residuals (loss-agnostic error signals)
    res = pred - y_target  # Raw error signal for soft predictions
    hard_res = pred_hard - y_target  # Raw error signal for hard predictions

    # Compute losses based on the specific loss type
    if loss_type == "bce":
        loss = binary_cross_entropy(pred, y_target)
        hard_loss = binary_cross_entropy(pred_hard, y_target)
    elif loss_type.startswith("l"):
        assert len(loss_type) == 2, "Loss type must be of the form 'lX'"
        # Get power from loss_type
        power = int(loss_type[-1])
        loss = res2loss(res, power=power)
        hard_loss = res2loss(hard_res, power=power)
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")

    accuracy = compute_accuracy(pred, y_target)
    hard_accuracy = compute_accuracy(pred_hard, y_target)

    return loss, (
        hard_loss,
        pred,
        pred_hard,
        accuracy,
        hard_accuracy,
        res,
        hard_res,
    )


def get_loss_and_update_graph(
    graph: jp.ndarray,
    logits_original_shapes: list[tuple],
    wires: list[jp.ndarray],
    x_data: jp.ndarray,
    y_data: jp.ndarray,
    loss_type: str,
    layer_sizes: list[tuple[int, int]],
):
    """
    Extract logits from graph, compute loss and residuals, and update graph with loss information.

    This function performs the 3-step process after a model has been applied to a graph:
    1. Extract logits from the updated graph
    2. Compute loss and residuals using the circuit
    3. Update the graph's output node loss features with residuals

    Args:
        graph: Updated graph from model application
        logits_original_shapes: Original shapes of logits for reconstruction
        wires: Wire connection patterns
        x_data: Input data
        y_data: Target output data
        loss_type: Type of loss function to use
        layer_sizes: List of (nodes, group_size) tuples for each layer

    Returns:
        Tuple of (updated_graph, loss, aux_data)
        where aux_data contains (hard_loss, pred, pred_hard, accuracy, hard_accuracy, res, hard_res)
    """
    # Extract updated logits from the graph
    current_logits = extract_logits_from_graph(graph, logits_original_shapes)

    # Compute loss and auxiliary data
    loss, aux = get_loss_from_wires_logits(current_logits, wires, x_data, y_data, loss_type)

    # Extract residuals from aux for updating loss feature
    *_, res, _ = aux

    # Update the loss feature for output nodes using residuals
    # We'll use the magnitude of residuals as the loss signal for each output node
    updated_graph = update_output_node_loss(graph, layer_sizes, jp.abs(res).mean(axis=0))

    return updated_graph, loss, current_logits, aux


def evaluate_model_stepwise_generator(
    model: CircuitGNN | CircuitSelfAttention,
    wires: list[jp.ndarray],
    logits: list[jp.ndarray],
    x_data: jp.ndarray,
    y_data: jp.ndarray,
    input_n: int,
    arity: int = 2,
    circuit_hidden_dim: int = 16,
    max_steps: int | None = None,
    loss_type: str = "l4",
    bidirectional_edges: bool = True,
    layer_sizes: list[tuple[int, int]] | None = None,
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
        circuit_hidden_dim: Dimension of hidden features
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
            initial_res,
            initial_hard_res,
        ),
    ) = get_loss_from_wires_logits(logits, wires, x_data, y_data, loss_type)

    # Build initial graph using the same function as training
    # Initialize with update_steps = 0 (exactly like training pool initialization)
    graph = build_graph(
        logits,
        wires,
        input_n,
        arity,
        circuit_hidden_dim,
        loss_value=initial_loss,
        bidirectional_edges=bidirectional_edges,
    )

    # Initialize graph globals with [loss, update_steps] exactly like training
    current_update_steps = 0
    graph = graph._replace(globals=jp.array([initial_loss, current_update_steps], dtype=jp.float32))

    graph = update_output_node_loss(graph, layer_sizes, initial_res.mean(axis=0))

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

        # Use the unified get_loss_and_update_graph function for consistency
        updated_graph, loss, current_logits, aux = get_loss_and_update_graph(
            updated_graph,
            logits_original_shapes,
            wires,
            x_data,
            y_data,
            loss_type,
            layer_sizes,
        )

        # Extract auxiliary data
        (
            hard_loss,
            pred,
            pred_hard,
            accuracy,
            hard_accuracy,
            res,
            hard_res,
        ) = aux

        # Update with the computed loss and incremented update_steps (EXACTLY like training)
        updated_graph = updated_graph._replace(
            globals=jp.array([loss, current_update_steps + 1], dtype=jp.float32)
        )

        # Update the graph variable for next iteration
        graph = updated_graph

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
    wires: list[jp.ndarray],
    logits: list[jp.ndarray],
    x_data: jp.ndarray,
    y_data: jp.ndarray,
    input_n: int,
    arity: int = 2,
    circuit_hidden_dim: int = 16,
    n_message_steps: int = 100,
    loss_type: str = "l4",
    bidirectional_edges: bool = True,
    layer_sizes: list[tuple[int, int]] | None = None,
    use_tqdm: bool = False,
) -> dict:
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
        circuit_hidden_dim: Dimension of hidden features
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
        circuit_hidden_dim=circuit_hidden_dim,
        max_steps=n_message_steps,
        loss_type=loss_type,
        bidirectional_edges=bidirectional_edges,
        layer_sizes=layer_sizes,
    )

    # Create progress bar for evaluation
    if use_tqdm:
        pbar = tqdm(generator, total=n_message_steps + 1, desc="Evaluating model steps")
    else:
        pbar = generator

    # Collect all results
    for result in pbar:
        step_metrics["step"].append(result.step)
        step_metrics["soft_loss"].append(result.loss)
        step_metrics["hard_loss"].append(result.hard_loss)
        step_metrics["soft_accuracy"].append(result.accuracy)
        step_metrics["hard_accuracy"].append(result.hard_accuracy)
        step_metrics["logits_mean"].append(float(result.graph.nodes["logits"].mean()))

        if use_tqdm:
            # Update progress bar
            pbar.set_postfix(
                {
                    "Loss": f"{result.loss:.4f}",
                    "Accuracy": f"{result.accuracy:.4f}",
                    "Hard Acc": f"{result.hard_accuracy:.4f}",
                }
            )

    return step_metrics


def evaluate_model_stepwise_batched(
    model: CircuitGNN | CircuitSelfAttention,
    batch_wires: list[jp.ndarray],  # Shape: [batch_size, ...original_wire_shape...]
    batch_logits: list[jp.ndarray],  # Shape: [batch_size, ...original_logit_shape...]
    x_data: jp.ndarray,
    y_data: jp.ndarray,
    input_n: int,
    arity: int = 2,
    circuit_hidden_dim: int = 16,
    n_message_steps: int = 100,
    loss_type: str = "l4",
    bidirectional_edges: bool = True,
    layer_sizes: list[tuple[int, int]] | None = None,
) -> dict:
    """
    Vectorized evaluation of GNN performance on a batch of circuits.

    This mirrors the exact computation path of evaluate_model_stepwise but processes
    all circuits in the batch simultaneously using vectorized operations.

    Args:
        model: Trained CircuitGNN or CircuitSelfAttention model
        batch_wires: Batched wire connection patterns [batch_size, ...wire_shape...]
        batch_logits: Batched initial logit tensors [batch_size, ...logit_shape...]
        x_data: Input data for evaluation
        y_data: Target output data
        input_n: Number of input nodes
        arity: Number of inputs per gate
        circuit_hidden_dim: Dimension of hidden features
        n_message_steps: Maximum number of message passing steps to run
        loss_type: Loss function to use
        bidirectional_edges: Whether to use bidirectional edges
        layer_sizes: List of (nodes, group_size) tuples for each layer

    Returns:
        Dictionary with averaged metrics collected at each step
    """
    # Initialize metric storage - same structure as original
    step_metrics = {
        "step": [],
        "soft_loss": [],
        "hard_loss": [],
        "soft_accuracy": [],
        "hard_accuracy": [],
        "logits_mean": [],
    }

    # Store original shapes for reconstruction (EXACTLY like generator)
    logits_original_shapes = [logit.shape[1:] for logit in batch_logits]  # Remove batch dim

    # Calculate initial losses for the batch (EXACTLY like generator)
    vmap_get_loss = jax.vmap(
        lambda logits, wires: get_loss_from_wires_logits(logits, wires, x_data, y_data, loss_type)
    )

    initial_losses, initial_aux = vmap_get_loss(batch_logits, batch_wires)
    (
        initial_hard_losses,
        initial_preds,
        initial_pred_hards,
        initial_accuracies,
        initial_hard_accuracies,
        initial_res,
        initial_hard_res,
    ) = initial_aux

    # Build initial graphs using the same function as generator (vectorized)
    # We need to handle the loss_value parameter carefully to avoid concretization issues
    vmap_build_graph = jax.vmap(
        lambda logits, wires: build_graph(
            logits,
            wires,
            input_n,
            arity,
            circuit_hidden_dim,
            loss_value=0.0,  # Use dummy value, will be set in globals later
            bidirectional_edges=bidirectional_edges,
        )
    )
    batch_graphs = vmap_build_graph(batch_logits, batch_wires)

    # Initialize graph globals with [loss, update_steps] exactly like generator
    current_update_steps = jp.zeros(initial_losses.shape[0])
    batch_graphs = batch_graphs._replace(
        globals=jp.stack([initial_losses, current_update_steps], axis=1)
    )

    # Update output node losses (vectorized)
    vmap_update_loss = jax.vmap(
        lambda graph, res: update_output_node_loss(graph, layer_sizes, res.mean(axis=0))
    )
    batch_graphs = vmap_update_loss(batch_graphs, initial_res)

    # Yield initial state (step 0) - same as generator
    step_metrics["step"].append(0)
    step_metrics["soft_loss"].append(float(jp.mean(initial_losses)))
    step_metrics["hard_loss"].append(float(jp.mean(initial_hard_losses)))
    step_metrics["soft_accuracy"].append(float(jp.mean(initial_accuracies)))
    step_metrics["hard_accuracy"].append(float(jp.mean(initial_hard_accuracies)))
    step_metrics["logits_mean"].append(float(jp.mean(batch_graphs.nodes["logits"])))

    # Run optimization steps (EXACTLY like the generator loop)
    current_graphs = batch_graphs

    for step in range(1, n_message_steps + 1):
        # Extract the current update_steps count from graph globals (EXACTLY like generator)
        current_update_steps = jp.zeros(current_graphs.globals.shape[0])
        if current_graphs.globals is not None and current_graphs.globals.shape[-1] > 1:
            current_update_steps = current_graphs.globals[..., 1]

        # Apply one step of model processing (vectorized - EXACTLY like generator)
        vmap_model = jax.vmap(model)
        updated_graphs = vmap_model(current_graphs)

        # Use the unified get_loss_and_update_graph function for consistency (vectorized)
        vmap_get_loss_and_update = jax.vmap(
            lambda graph, wires: get_loss_and_update_graph(
                graph,
                logits_original_shapes,
                wires,
                x_data,
                y_data,
                loss_type,
                layer_sizes,
            )
        )

        updated_graphs, losses, current_logits, aux_data = vmap_get_loss_and_update(
            updated_graphs, batch_wires
        )

        # Extract auxiliary data (vectorized)
        (
            hard_losses,
            preds,
            pred_hards,
            accuracies,
            hard_accuracies,
            res,
            hard_res,
        ) = aux_data

        # Update with the computed loss and incremented update_steps (EXACTLY like generator)
        updated_graphs = updated_graphs._replace(
            globals=jp.stack([losses, current_update_steps + 1], axis=1)
        )

        # Update the graphs for next iteration
        current_graphs = updated_graphs

        # Store averaged metrics (same as generator yields)
        step_metrics["step"].append(step)
        step_metrics["soft_loss"].append(float(jp.mean(losses)))
        step_metrics["hard_loss"].append(float(jp.mean(hard_losses)))
        step_metrics["soft_accuracy"].append(float(jp.mean(accuracies)))
        step_metrics["hard_accuracy"].append(float(jp.mean(hard_accuracies)))
        step_metrics["logits_mean"].append(float(jp.mean(current_graphs.nodes["logits"])))

    return step_metrics
