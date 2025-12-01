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
    LOSS_L4,
    LossConfig,
    compute_accuracy,
    compute_loss_from_predictions,
)
from boolean_nca_cc.models import CircuitGNN, CircuitSelfAttention, PerceiverCircuitAttention
from boolean_nca_cc.training.pool.structural_perturbation import apply_knockout_to_circuit
from boolean_nca_cc.utils import (
    GraphGlobals,
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
def get_loss_from_wires_logits(
    logits,
    wires,
    x,
    y_target,
    loss_cfg=None,
):
    """
    Run circuit and calculate loss.

    Args:
        logits: Circuit logits
        wires: Wire connections
        x: Input data
        y_target: Target output
        loss_cfg: LossConfig NamedTuple or dict (default: LOSS_L4)
    """
    if loss_cfg is None:
        loss_cfg = LOSS_L4
    elif isinstance(loss_cfg, dict):
        loss_cfg = LossConfig.from_dict(loss_cfg)

    # Run circuit
    acts = run_circuit(logits, wires, x)
    pred = acts[-1]
    acts_hard = run_circuit(logits, wires, x, hard=True)
    pred_hard = acts_hard[-1]

    # Use unified loss computation
    loss, hard_loss, res, hard_res, accuracy, hard_accuracy = compute_loss_from_predictions(
        pred, pred_hard, y_target, loss_cfg
    )

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
    loss_cfg,
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
        loss_cfg: Loss config dict
        layer_sizes: List of (nodes, group_size) tuples for each layer

    Returns:
        Tuple of (updated_graph, loss, aux_data)
        where aux_data contains (hard_loss, pred, pred_hard, accuracy, hard_accuracy, res, hard_res)
    """
    # Extract updated logits from the graph
    current_logits = extract_logits_from_graph(graph, logits_original_shapes)

    # Compute loss and auxiliary data
    loss, aux = get_loss_from_wires_logits(
        logits=current_logits,
        wires=wires,
        x=x_data,
        y_target=y_data,
        loss_cfg=loss_cfg,
    )

    # Extract residuals from aux for updating loss feature
    *_, res, _ = aux

    # Update the loss feature for output nodes using residuals
    # We'll use the magnitude of residuals as the loss signal for each output node
    updated_graph = update_output_node_loss(graph, layer_sizes, jp.abs(res).mean(axis=0))

    return updated_graph, loss, current_logits, aux


def create_damage_steps(
    n_damage_steps: int = 0,
    max_steps: int | None = None,
    verbose: bool = False,
    damage_key: jax.random.PRNGKey = jax.random.PRNGKey(42),
    min_damage_interval: int = 16,
) -> list[int]:
    """
    Create damage steps with at least min_damage_interval spacing.
    """

    if n_damage_steps > 0:
        # Create random damage steps with at least min_damage_interval spacing
        total_steps = max_steps if max_steps is not None else 1000

        # Calculate minimum required steps to fit all damages with proper spacing
        min_required_steps = (n_damage_steps - 1) * min_damage_interval + 1
        assert total_steps >= min_required_steps, (
            f"Not enough steps ({total_steps}) to apply {n_damage_steps} damages "
            f"with min_damage_interval={min_damage_interval} (need at least {min_required_steps})"
        )

        # Generate truly random damage steps with guaranteed minimum spacing:
        # 1. "Compress" the range by removing mandatory spacing between points
        # 2. Pick n random points in that compressed range
        # 3. Sort and expand back by adding cumulative offsets
        compressed_range = total_steps - (n_damage_steps - 1) * min_damage_interval
        random_points = jax.random.randint(damage_key, (n_damage_steps,), 0, compressed_range)
        sorted_points = jp.sort(random_points)
        offsets = jp.arange(n_damage_steps) * min_damage_interval
        damage_steps = sorted_points + offsets

        if verbose:
            print(f"Random points (compressed): {sorted_points}")
            print(f"Offsets: {offsets}")
            print(f"Damage steps: {damage_steps}")

    return damage_steps


def evaluate_model_stepwise_generator(
    model: CircuitGNN | CircuitSelfAttention | PerceiverCircuitAttention,
    wires: list[jp.ndarray],
    logits: list[jp.ndarray],
    x_data: jp.ndarray,
    y_data: jp.ndarray,
    input_n: int,
    arity: int = 2,
    circuit_hidden_dim: int = 16,
    max_steps: int | None = None,
    loss_cfg=None,
    bidirectional_edges: bool = True,
    layer_sizes: list[tuple[int, int]] | None = None,
    damage_steps: list[int] | None = None,
    knockout_per_damage_step: int = 1,
    damage_key: jax.random.PRNGKey = jax.random.PRNGKey(42),
    verbose: bool = False,
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
        loss_cfg: Loss config dict (default: LOSS_L4)
        bidirectional_edges: Whether to use bidirectional edges

    Yields:
        StepResult: Results from each step including loss, accuracy, predictions, and updated logits
    """
    if loss_cfg is None:
        loss_cfg = LOSS_L4

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
    ) = get_loss_from_wires_logits(logits, wires, x_data, y_data, loss_cfg)

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
        gate_knockout_mask=None,
    )

    # Check if we have a Perceiver model that needs data in globals
    is_perceiver = isinstance(model, PerceiverCircuitAttention)

    # Initialize graph globals (always use GraphGlobals NamedTuple)
    current_update_steps = 0
    if is_perceiver:
        # Perceiver models need data in globals
        globals_tuple = GraphGlobals(
            loss=float(initial_loss),
            update_steps=current_update_steps,
            x_data=x_data,
            y_data=y_data,
            residuals=initial_res,
        )
    else:
        globals_tuple = GraphGlobals(
            loss=float(initial_loss),
            update_steps=current_update_steps,
        )

    graph = graph._replace(globals=globals_tuple)

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

    gate_mask = graph.nodes["gate_knockout_mask"]

    # Run optimization steps (EXACTLY like the training loop)
    step = 0
    while max_steps is None or step < max_steps:
        step += 1

        # Extract the current update_steps count from graph globals (EXACTLY like training)
        current_update_steps = 0
        if graph.globals is not None:
            current_update_steps = graph.globals.update_steps

        # Apply damage if needed
        if damage_steps is not None and step in damage_steps:
            damage_key, new_damage_key = jax.random.split(damage_key)
            modified_logits, modified_gate_mask = apply_knockout_to_circuit(
                new_damage_key,
                graph.nodes["logits"],
                layer_sizes,
                num_knockouts=knockout_per_damage_step,
                flat=True,
            )
            gate_mask *= modified_gate_mask

            graph.nodes["gate_knockout_mask"] = gate_mask
            graph.nodes["logits"] = modified_logits

            if verbose:
                print(f"Damage at step {step}")
                print(f"Damage in graph : {(graph.nodes['gate_knockout_mask'] == 0).sum()}")

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
            loss_cfg,
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
        if is_perceiver:
            globals_tuple = GraphGlobals(
                loss=float(loss),
                update_steps=current_update_steps + 1,
                x_data=x_data,
                y_data=y_data,
                residuals=res,
            )
        else:
            globals_tuple = GraphGlobals(
                loss=float(loss),
                update_steps=current_update_steps + 1,
            )

        updated_graph = updated_graph._replace(globals=globals_tuple)

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
    model: CircuitGNN | CircuitSelfAttention | PerceiverCircuitAttention,
    wires: list[jp.ndarray],
    logits: list[jp.ndarray],
    x_data: jp.ndarray,
    y_data: jp.ndarray,
    input_n: int,
    arity: int = 2,
    circuit_hidden_dim: int = 16,
    n_message_steps: int = 100,
    loss_cfg=None,
    bidirectional_edges: bool = True,
    layer_sizes: list[tuple[int, int]] | None = None,
    use_tqdm: bool = False,
    verbose: bool = False,
    knockout_per_damage_step: int = 1,
    damage_key: jax.random.PRNGKey = jax.random.PRNGKey(42),
    damage_steps: list[int] | None = None,
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
        loss_cfg: Loss config dict (default: LOSS_L4)
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
        loss_cfg=loss_cfg,
        bidirectional_edges=bidirectional_edges,
        layer_sizes=layer_sizes,
        verbose=verbose,
        knockout_per_damage_step=knockout_per_damage_step,
        damage_key=damage_key,
        damage_steps=damage_steps,
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
    model: CircuitGNN | CircuitSelfAttention | PerceiverCircuitAttention,
    batch_wires: list[jp.ndarray],  # Shape: [batch_size, ...original_wire_shape...]
    batch_logits: list[jp.ndarray],  # Shape: [batch_size, ...original_logit_shape...]
    x_data: jp.ndarray,
    y_data: jp.ndarray,
    input_n: int,
    arity: int = 2,
    circuit_hidden_dim: int = 16,
    n_message_steps: int = 100,
    loss_cfg=None,
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
        loss_cfg: Loss config dict (default: LOSS_L4)
        bidirectional_edges: Whether to use bidirectional edges
        layer_sizes: List of (nodes, group_size) tuples for each layer

    Returns:
        Dictionary with averaged metrics collected at each step
    """
    if loss_cfg is None:
        loss_cfg = LOSS_L4

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
        lambda logits, wires: get_loss_from_wires_logits(logits, wires, x_data, y_data, loss_cfg)
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

    # Initialize graph globals with NamedTuple (batched values)
    current_update_steps = jp.zeros(initial_losses.shape[0])
    batch_graphs = batch_graphs._replace(
        globals=GraphGlobals(
            loss=initial_losses,
            update_steps=current_update_steps,
        )
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
        if current_graphs.globals is not None:
            current_update_steps = current_graphs.globals.update_steps
        else:
            current_update_steps = jp.zeros(len(batch_logits[0]))

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
                loss_cfg,
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
            globals=GraphGlobals(
                loss=losses,
                update_steps=current_update_steps + 1,
            )
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
