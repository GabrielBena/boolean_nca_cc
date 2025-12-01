"""
Evaluation functions for GNN-based boolean circuit optimization.

This module provides functions for evaluating the performance of GNN
models on optimizing boolean circuits.
"""

from collections.abc import Generator
from typing import NamedTuple

import jax
import jax.numpy as jp
from flax import nnx
from tqdm.auto import tqdm

from boolean_nca_cc.circuits.model import run_circuit
from boolean_nca_cc.circuits.train import (
    LOSS_L4,
    LossConfig,
    compute_loss_from_predictions,
)
from boolean_nca_cc.models import CircuitGNN, CircuitSelfAttention, PerceiverCircuitAttention
from boolean_nca_cc.training.pool.structural_perturbation import apply_knockout_to_circuit
from boolean_nca_cc.utils import (
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
    residuals: jp.ndarray
    hard_residuals: jp.ndarray
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
    loss, hard_loss, residuals, hard_residuals, accuracy, hard_accuracy = (
        compute_loss_from_predictions(pred, pred_hard, y_target, loss_cfg)
    )

    return loss, {
        "loss": loss,
        "hard_loss": hard_loss,
        "predictions": pred,
        "hard_predictions": pred_hard,
        "accuracy": accuracy,
        "hard_accuracy": hard_accuracy,
        "residuals": residuals,
        "hard_residuals": hard_residuals,
    }


def get_loss_and_update_graph(
    graph: jp.ndarray,
    logits_original_shapes: list[tuple],
    wires: list[jp.ndarray],
    x_data: jp.ndarray,
    y_data: jp.ndarray,
    loss_cfg,
    layer_sizes: list[tuple[int, int]],
    update_perceiver_globals: bool = False,
):
    """
    Extract logits from graph, compute loss and residuals, and update graph with loss information.

    This function performs the 3-step process after a model has been applied to a graph:
    1. Extract logits from the updated graph
    2. Compute loss and residuals using the circuit
    3. Update the graph's output node loss features with residuals
    4. (Optional) Update GraphGlobals with new residuals for Perceiver models

    Args:
        graph: Updated graph from model application
        logits_original_shapes: Original shapes of logits for reconstruction
        wires: Wire connection patterns
        x_data: Input data
        y_data: Target output data
        loss_cfg: Loss config dict
        layer_sizes: List of (nodes, group_size) tuples for each layer
        update_perceiver_globals: If True, update graph.globals with new loss, residuals, etc.
            This avoids a redundant circuit evaluation in Perceiver scan functions.

    Returns:
        Tuple of (updated_graph, loss, current_logits, aux)
        where aux contains (hard_loss, pred, pred_hard, accuracy, hard_accuracy, res, hard_res)
    """
    from boolean_nca_cc.utils.graph_builder import GraphGlobals

    # Extract updated logits from the graph
    current_logits = extract_logits_from_graph(graph, logits_original_shapes)

    # Compute loss and auxiliary data
    loss, aux_data = get_loss_from_wires_logits(
        logits=current_logits,
        wires=wires,
        x=x_data,
        y_target=y_data,
        loss_cfg=loss_cfg,
    )

    # Extract residuals from aux for updating loss feature
    residuals = aux_data["residuals"]

    # Update the loss feature for output nodes using residuals
    # We'll use the magnitude of residuals as the loss signal for each output node
    updated_graph = update_output_node_loss(graph, layer_sizes, jp.abs(residuals).mean(axis=0))

    # Optionally update GraphGlobals for Perceiver models
    # This saves a redundant circuit evaluation since residuals are already computed
    if update_perceiver_globals and updated_graph.globals is not None:
        current_update_steps = updated_graph.globals.update_steps
        updated_graph = updated_graph._replace(
            globals=GraphGlobals(
                loss=loss,
                update_steps=current_update_steps + 1,
                x_data=x_data,
                y_data=y_data,
                residuals=residuals,
            )
        )

    return updated_graph, loss, current_logits, aux_data


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

    # Build initial graph using the same function as training
    # Initialize with update_steps = 0 (exactly like training pool initialization)
    graph = build_graph(
        logits,
        wires,
        input_n,
        arity,
        circuit_hidden_dim,
        loss_value=0.0,
        bidirectional_edges=bidirectional_edges,
        gate_knockout_mask=None,
    )

    # Check if we have a Perceiver model that needs data in globals
    is_perceiver = isinstance(model, PerceiverCircuitAttention)

    graph, _, _, aux_data = get_loss_and_update_graph(
        graph,
        logits_original_shapes,
        wires,
        x_data,
        y_data,
        loss_cfg,
        layer_sizes,
        update_perceiver_globals=is_perceiver,
    )

    # Yield initial state (step 0)
    yield StepResult(
        step=0,
        logits=logits,
        graph=graph,
        **aux_data,
    )

    gate_mask = graph.nodes["gate_knockout_mask"]

    # Run optimization steps
    step = 0
    while max_steps is None or step < max_steps:
        step += 1

        # Extract the current update_steps count from graph globals
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

        # Apply one step of model processing
        # Note: training does multiple steps in a batch, but we do one at a time for live demo
        updated_graph = model(graph)

        # Use the unified get_loss_and_update_graph function for consistency
        graph, _, current_logits, aux_data = get_loss_and_update_graph(
            updated_graph,
            logits_original_shapes,
            wires,
            x_data,
            y_data,
            loss_cfg,
            layer_sizes,
            update_perceiver_globals=is_perceiver,
        )

        # Yield current state
        yield StepResult(
            step=step,
            logits=current_logits,
            graph=graph,
            **aux_data,
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
    step_results = {
        "step": [],
        "loss": [],
        "hard_loss": [],
        "accuracy": [],
        "hard_accuracy": [],
    }
    # step_results = []

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
        for key, value in result._asdict().items():
            if key in step_results:
                step_results[key].append(value)

        # step_results.append(result)

        if use_tqdm:
            # Update progress bar
            pbar.set_postfix(
                {
                    "Loss": f"{result.loss:.4f}",
                    "Accuracy": f"{result.accuracy:.4f}",
                    "Hard Acc": f"{result.hard_accuracy:.4f}",
                }
            )

    # return jax.tree.map(lambda x: jp.array(x), step_results)
    return step_results


# Batched evaluation function. Does not support damage steps. right now
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
        "loss": [],
        "hard_loss": [],
        "accuracy": [],
        "hard_accuracy": [],
    }
    # step_results = []

    is_perceiver = isinstance(model, PerceiverCircuitAttention)

    # Build initial graphs using the same function as generator (vectorized)
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

    vmap_get_loss_and_update = jax.vmap(
        lambda graph, wires, logits: get_loss_and_update_graph(
            graph=graph,
            logits_original_shapes=[logit.shape for logit in logits],
            wires=wires,
            x_data=x_data,
            y_data=y_data,
            loss_cfg=loss_cfg,
            layer_sizes=layer_sizes,
            update_perceiver_globals=is_perceiver,
        )
    )

    for step in range(0, n_message_steps + 1):
        # Compute loss and update graph
        batch_graphs, _, current_logits, aux_data = vmap_get_loss_and_update(
            batch_graphs, batch_wires, batch_logits
        )

        # Store averaged metrics (same as generator yields)
        # result = StepResult(
        #     step=step,
        #     logits=current_logits,
        #     graph=batch_graphs,
        #     **jax.tree.map(lambda x: jp.mean(x, axis=0), aux_data),
        # )
        # step_results.append(result)

        step_metrics["step"].append(step)
        for key, value in aux_data.items():
            if key in step_metrics:
                step_metrics[key].append(float(jp.mean(value)))

        # Apply one step of model processing (vectorized)
        batch_graphs = nnx.vmap(lambda graph: model(graph))(batch_graphs)

    # return jax.tree.map(lambda x: jp.array(x), step_results)
    return step_metrics
