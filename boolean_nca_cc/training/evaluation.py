"""
Evaluation functions for GNN-based boolean circuit optimization.

This module provides functions for evaluating the performance of GNN
models on optimizing boolean circuits.
"""

from pickle import FALSE
import jax
import jax.numpy as jp
import jraph
from tqdm.auto import tqdm
from typing import List, Dict, Tuple, Generator, NamedTuple, Optional

from boolean_nca_cc.models import CircuitSelfAttention, CircuitGNN
from boolean_nca_cc.utils import (
    build_graph,
    extract_logits_from_graph,
    update_output_node_loss,
)
from boolean_nca_cc.circuits.model import run_circuit
from boolean_nca_cc.circuits.train import (
    res2loss,
    binary_cross_entropy,
    compute_accuracy,
    compute_full_map_accuracy,
)
from boolean_nca_cc.training.pool.structural_perturbation import (
    create_reproducible_knockout_pattern,
)


class StepResult(NamedTuple):
    """Result from a single step of model evaluation."""

    step: int
    loss: float
    hard_loss: float
    accuracy: float
    hard_accuracy: float
    full_map_accuracy: float  # Hard version only
    predictions: jp.ndarray
    hard_predictions: jp.ndarray
    logits: List[jp.ndarray]
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
    full_map_accuracy = compute_full_map_accuracy(pred_hard, y_target)  # Only hard version

    return loss, (
        hard_loss,
        pred,
        pred_hard,
        accuracy,
        hard_accuracy,
        full_map_accuracy,
        res,
        hard_res,
    )


def get_loss_and_update_graph(
    graph: jp.ndarray,
    logits_original_shapes: List[Tuple],
    wires: List[jp.ndarray],
    x_data: jp.ndarray,
    y_data: jp.ndarray,
    loss_type: str,
    layer_sizes: List[Tuple[int, int]],
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
        where aux_data contains (hard_loss, pred, pred_hard, accuracy, hard_accuracy, full_map_accuracy, res, hard_res)
    """
    # Extract updated logits from the graph
    current_logits = extract_logits_from_graph(graph, logits_original_shapes)

    # Compute loss and auxiliary data
    loss, aux = get_loss_from_wires_logits(
        current_logits, wires, x_data, y_data, loss_type
    )

    # Extract residuals from aux for updating loss feature
    *_, res, _ = aux

    # Update the loss feature for output nodes using residuals
    # We'll use the magnitude of residuals as the loss signal for each output node
    updated_graph = update_output_node_loss(
        graph, layer_sizes, jp.abs(res).mean(axis=0)
    )

    return updated_graph, loss, current_logits, aux


def evaluate_model_stepwise_generator(
    model: CircuitGNN | CircuitSelfAttention,
    wires: List[jp.ndarray],
    logits: List[jp.ndarray],
    x_data: jp.ndarray,
    y_data: jp.ndarray,
    input_n: int,
    arity: int = 2,
    circuit_hidden_dim: int = 16,
    max_steps: int = None,
    loss_type: str = "l4",
    bidirectional_edges: bool = True,
    layer_sizes: List[Tuple[int, int]] = None,
    layer_neighbors: bool = False,
    knockout_pattern: Optional[jp.ndarray] = None,
    reset_step_counter_on_init: bool = False,
    blind_mode: bool = False,
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
        layer_sizes: List of (nodes, group_size) tuples for each layer
        layer_neighbors: Whether to allow attention between adjacent nodes within layers
        knockout_pattern: Optional knockout pattern to pass to model for damage injection
        reset_step_counter_on_init: If True, reset step counter to 0 at initialization
                                   (enables reversible mode activation on first step)

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
            initial_full_map_accuracy,
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
    if reset_step_counter_on_init:
        current_update_steps = 0  # Force reset to 0 for reversible mode
    graph = graph._replace(
        globals=jp.array([initial_loss, current_update_steps], dtype=jp.float32)
    )

    initial_res_for_update = initial_res
    if blind_mode:
        initial_res_for_update = jp.zeros_like(initial_res)

    graph = update_output_node_loss(graph, layer_sizes, initial_res_for_update.mean(axis=0))

    # Debug: Log initial graph globals
    initial_globals_loss = float(graph.globals[0]) if graph.globals is not None else 0.0
    initial_globals_steps = float(graph.globals[1]) if graph.globals is not None and graph.globals.shape[-1] > 1 else 0.0
    print(f"[Generator Init] Step 0: globals=[loss={initial_globals_loss:.6f}, update_steps={initial_globals_steps:.0f}]")

    # Yield initial state (step 0)
    yield StepResult(
        step=0,
        loss=float(initial_loss),
        hard_loss=float(initial_hard_loss),
        accuracy=float(initial_accuracy),
        hard_accuracy=float(initial_hard_accuracy),
        full_map_accuracy=float(initial_full_map_accuracy),
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
        # Pass layer_sizes and layer_neighbors to match training eval behavior
        if isinstance(model, CircuitSelfAttention):
            updated_graph = model(
                graph,
                knockout_pattern=knockout_pattern,  # Pass pattern to model
                layer_neighbors=layer_neighbors,
                layer_sizes=layer_sizes,
            )
        else:
            # For GNN models, call without these parameters (they don't support them)
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
            full_map_accuracy,
            res,
            hard_res,
        ) = aux

        if blind_mode:
            # Overwrite the loss update performed by get_loss_and_update_graph
            updated_graph = update_output_node_loss(
                updated_graph, layer_sizes, jp.zeros_like(res).mean(axis=0)
            )

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
            full_map_accuracy=float(full_map_accuracy),
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
    circuit_hidden_dim: int = 16,
    n_message_steps: int = 100,
    loss_type: str = "l4",
    bidirectional_edges: bool = True,
    layer_sizes: List[Tuple[int, int]] = None,
    use_tqdm: bool = False,
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
        "full_map_accuracy": [],
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
        step_metrics["full_map_accuracy"].append(result.full_map_accuracy)
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
    model: CircuitSelfAttention,
    batch_wires: List[jp.ndarray],  # Shape: [batch_size, ...original_wire_shape...]
    batch_logits: List[jp.ndarray],  # Shape: [batch_size, ...original_logit_shape...]
    x_data: jp.ndarray,
    y_data: jp.ndarray,
    input_n: int,
    arity: int = 2,
    circuit_hidden_dim: int = 16,
    n_message_steps: int = 100,
    loss_type: str = "l4",
    bidirectional_edges: bool = True,
    layer_sizes: List[Tuple[int, int]] = None,
    knockout_patterns: Optional[jp.ndarray] = None,
    return_per_pattern: bool = False,  # New parameter
    layer_neighbors: bool = False,
    # Unified damage control system
    damage_mode: str = "greedy",  # Pattern type: "greedy", "greedy_vocabulary", "shotgun", "strip"
    damage_injection_mode: str = "multi",  # "single" (one damage per circuit) or "multi" (multiple damages)
    max_damage_per_circuit: int = 10,  # Maximum damage events per circuit
    # Periodic injection timing (controlled by damage_mode)
    greedy_ordered_indices: Optional[List[int]] = None,
    greedy_window_size: int = 1,
    greedy_injection_recover_steps: int = 10,
    # Damage start offset (warm-up period before first damage)
    damage_start_offset: int = 0,  # Number of steps to run before first damage injection
    damage_start_offset_random: bool = False,  # If True, randomize offset per circuit (0 to damage_start_offset)
    damage_start_offset_seed: int = 42,  # Seed for random offset generation
    # Vocabulary-based evaluation parameters
    knockout_vocabulary: Optional[jp.ndarray] = None,  # If provided => seen (sample from vocab); else => unseen (fresh)
    blind_mode: bool = False,  # If True, force loss feedback to zero (ablation study)
    # Permanent damage validation tracking
    track_damage_validation: bool = False,  # Enable detailed tracking for permanent damage validation
    # DEBUG: Track pre-update accuracy at damage injection
    track_pre_update_accuracy: bool = False,  # DEBUG: Enable tracking of accuracy before model updates
) -> Dict:
    """
    Evaluate GNN performance on a batch of circuits by running message passing steps
    and collecting averaged metrics at each step.

    Args:
        model: CircuitSelfAttention model
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
        knockout_patterns: Optional knockout patterns for each circuit
        return_per_pattern: Whether to return per-pattern metrics in addition to averages
        damage_mode: Pattern type for damage generation
        damage_injection_mode: "single" or "multi" damage per circuit
        max_damage_per_circuit: Maximum damage events per circuit
        blind_mode: If True, zeroes out the loss signal fed back to the GNN.
                    Used to test if the GNN generalizes structurally (Architect)
                    or relies on test-time optimization (Technician).

    Returns:
        Dictionary with averaged metrics collected at each step.
        If return_per_pattern=True, also includes "per_pattern" key with individual metrics.
    """
    # Validate new parameters
    if damage_injection_mode not in ["single", "multi"]:
        raise ValueError(f"damage_injection_mode must be 'single' or 'multi', got '{damage_injection_mode}'")
    # Automatically set max_damage_per_circuit=1 when mode is 'single'
    if damage_injection_mode == "single":
        max_damage_per_circuit = 1
    if max_damage_per_circuit < 1:
        raise ValueError(f"max_damage_per_circuit must be >= 1, got {max_damage_per_circuit}")
    if damage_start_offset < 0:
        raise ValueError(f"damage_start_offset must be >= 0, got {damage_start_offset}")
    batch_size = batch_wires[0].shape[0]

    # Build initial graphs for the batch
    vmap_build_graph = jax.vmap(
        lambda logits, wires: build_graph(
            logits=logits,
            wires=wires,
            input_n=input_n,
            arity=arity,
            circuit_hidden_dim=circuit_hidden_dim,
            loss_value=0.0,  # Will be calculated properly
            bidirectional_edges=bidirectional_edges,
        )
    )

    batch_graphs = vmap_build_graph(batch_logits, batch_wires)

    # Calculate initial losses for the batch
    vmap_get_loss = jax.vmap(
        lambda logits, wires: get_loss_from_wires_logits(
            logits, wires, x_data, y_data, loss_type
        )
    )

    initial_losses, initial_aux = vmap_get_loss(batch_logits, batch_wires)
    (
        initial_hard_losses,
        _,
        _,
        initial_accuracies,
        initial_hard_accuracies,
        initial_full_map_accuracies,
        initial_res,
        _,
    ) = [aux_elem for aux_elem in initial_aux]

    # Update graph globals with initial losses
    batch_graphs = batch_graphs._replace(
        globals=jp.stack([initial_losses, jp.zeros_like(initial_losses)], axis=1)
    )

    # Update output node losses for each graph in the batch
    vmap_update_loss = jax.vmap(
        lambda graph, res: update_output_node_loss(graph, layer_sizes, res.mean(axis=0))
    )
    batch_graphs = vmap_update_loss(batch_graphs, initial_res)

    # Initialize metric storage
    step_metrics = {
        "step": [],
        "soft_loss": [],
        "hard_loss": [],
        "soft_accuracy": [],
        "hard_accuracy": [],
        "full_map_accuracy": [],
        "logits_mean": [],
    }

    # Store initial metrics (averaged across batch)
    step_metrics["step"].append(0)
    step_metrics["soft_loss"].append(float(jp.mean(initial_losses)))
    step_metrics["hard_loss"].append(float(jp.mean(initial_hard_losses)))
    step_metrics["soft_accuracy"].append(float(jp.mean(initial_accuracies)))
    step_metrics["hard_accuracy"].append(float(jp.mean(initial_hard_accuracies)))
    step_metrics["full_map_accuracy"].append(float(jp.mean(initial_full_map_accuracies)))
    step_metrics["logits_mean"].append(float(jp.mean(batch_graphs.nodes["logits"])))
    

    # Store original shapes for reconstruction
    logits_original_shapes = [
        logit.shape[1:] for logit in batch_logits
    ]  # Remove batch dim

    # Use loop mode (original behavior)
    return _evaluate_with_loop(
        model=model,
        batch_graphs=batch_graphs,
        batch_wires=batch_wires,
        x_data=x_data,
        y_data=y_data,
        n_message_steps=n_message_steps,
        loss_type=loss_type,
        layer_sizes=layer_sizes,
        knockout_patterns=knockout_patterns,
        logits_original_shapes=logits_original_shapes,
        step_metrics=step_metrics,
        return_per_pattern=return_per_pattern,
        initial_hard_accuracies=initial_hard_accuracies,
        batch_logits=batch_logits,
        layer_neighbors=layer_neighbors,
        damage_mode=damage_mode,
        damage_injection_mode=damage_injection_mode,
        max_damage_per_circuit=max_damage_per_circuit,
        greedy_ordered_indices=greedy_ordered_indices,
        greedy_window_size=greedy_window_size,
        greedy_injection_recover_steps=greedy_injection_recover_steps,
        damage_start_offset=damage_start_offset,
        damage_start_offset_random=damage_start_offset_random,
        damage_start_offset_seed=damage_start_offset_seed,
        knockout_vocabulary=knockout_vocabulary,
        blind_mode=blind_mode,
        track_damage_validation=track_damage_validation,
        track_pre_update_accuracy=track_pre_update_accuracy,  # DEBUG
    )

def evaluate_circuits_in_chunks(
    eval_fn,
    wires: List[jp.ndarray],
    logits: List[jp.ndarray],
    target_chunk_size: int,
    **eval_kwargs,
) -> Dict:
    """
    Evaluate circuits in chunks to handle cases where diversity exceeds target batch size.

    Args:
        eval_fn: Evaluation function to apply to each chunk
        wires: List of wire arrays for all circuits
        logits: List of logit arrays for all circuits
        target_chunk_size: Target size for each evaluation chunk
        **eval_kwargs: Additional keyword arguments to pass to eval_fn

    Returns:
        Dictionary with averaged metrics across all chunks
    """
    total_circuits = wires[0].shape[0]

    if total_circuits <= target_chunk_size:
        # No need to chunk, evaluate all at once
        return eval_fn(batch_wires=wires, batch_logits=logits, **eval_kwargs)

    # Split into chunks and evaluate each
    num_chunks = (total_circuits + target_chunk_size - 1) // target_chunk_size
    chunk_results = []

    # Pop knockout_patterns from kwargs to handle it manually for chunking
    local_kwargs = eval_kwargs.copy()
    knockout_patterns = local_kwargs.pop("knockout_patterns", None)

    for chunk_idx in range(num_chunks):
        start_idx = chunk_idx * target_chunk_size
        end_idx = min(start_idx + target_chunk_size, total_circuits)

        # Extract chunk
        chunk_wires = [w[start_idx:end_idx] for w in wires]
        chunk_logits = [l[start_idx:end_idx] for l in logits]

        chunk_knockouts = None
        if knockout_patterns is not None:
            chunk_knockouts = knockout_patterns[start_idx:end_idx]

        # Evaluate chunk
        chunk_result = eval_fn(
            batch_wires=chunk_wires,
            batch_logits=chunk_logits,
            knockout_patterns=chunk_knockouts,
            **local_kwargs,
        )
        chunk_results.append(chunk_result)

    # Average results across chunks
    # Assume all chunk results have the same structure
    averaged_result = {}
    for key in chunk_results[0].keys():
        if key == "per_pattern":
            # Special handling for per_pattern data - concatenate across chunks
            if "per_pattern" in chunk_results[0]:
                combined_per_pattern = {}
                for pattern_key in chunk_results[0]["per_pattern"].keys():
                    # Concatenate along the batch dimension (axis=1)
                    pattern_data_list = [chunk["per_pattern"][pattern_key] for chunk in chunk_results]
                    combined_per_pattern[pattern_key] = jp.concatenate(pattern_data_list, axis=1)
                averaged_result["per_pattern"] = combined_per_pattern
        elif isinstance(chunk_results[0][key], list):
            # For step-wise metrics, average at each step
            step_averages = []
            for step_idx in range(len(chunk_results[0][key])):
                step_values = [chunk[key][step_idx] for chunk in chunk_results]
                step_averages.append(float(jp.mean(jp.array(step_values))))
            averaged_result[key] = step_averages
        else:
            # For scalar metrics, simple average
            values = [chunk[key] for chunk in chunk_results]
            averaged_result[key] = float(jp.mean(jp.array(values)))

    return averaged_result


def _evaluate_with_loop(
    model,
    batch_graphs: jraph.GraphsTuple,
    batch_wires: List[jp.ndarray],
    x_data: jp.ndarray,
    y_data: jp.ndarray,
    n_message_steps: int,
    loss_type: str,
    layer_sizes: List[Tuple[int, int]],
    knockout_patterns: Optional[jp.ndarray],
    logits_original_shapes: List[Tuple],
    step_metrics: Dict,
    return_per_pattern: bool = False,  # New parameter
    initial_hard_accuracies: Optional[jp.ndarray] = None,
    batch_logits: Optional[List[jp.ndarray]] = None,
    layer_neighbors: bool = False,
    # Unified damage control system
    damage_mode: str = "greedy",  # Pattern type: "greedy", "greedy_vocabulary", "shotgun", "strip"
    damage_injection_mode: str = "multi",  # "single" (one damage per circuit) or "multi" (multiple damages)
    max_damage_per_circuit: int = 10,  # Maximum damage events per circuit
    # Periodic injection timing (controlled by damage_mode)
    greedy_ordered_indices: Optional[List[int]] = None,
    greedy_window_size: int = 1,
    greedy_injection_recover_steps: int = 7,
    # Damage start offset (warm-up period before first damage)
    damage_start_offset: int = 0,  # Number of steps to run before first damage injection
    damage_start_offset_random: bool = False,  # If True, randomize offset per circuit (0 to damage_start_offset)
    damage_start_offset_seed: int = 42,  # Seed for random offset generation
    # Vocabulary-based evaluation parameters
    knockout_vocabulary: Optional[jp.ndarray] = None,  # If provided => seen; else => unseen (fresh)
    blind_mode: bool = False,
    # Permanent damage validation tracking
    track_damage_validation: bool = False,  # Enable detailed tracking for permanent damage validation
    # DEBUG: Track pre-update accuracy at damage injection
    track_pre_update_accuracy: bool = False,  # DEBUG: Enable tracking of accuracy before model updates
) -> Dict:
    """
    Evaluate using loop mode (original behavior).
    """
    # Validate new parameters
    if damage_injection_mode not in ["single", "multi"]:
        raise ValueError(f"damage_injection_mode must be 'single' or 'multi', got '{damage_injection_mode}'")
    # Automatically set max_damage_per_circuit=1 when mode is 'single'
    if damage_injection_mode == "single":
        max_damage_per_circuit = 1
    if max_damage_per_circuit < 1:
        raise ValueError(f"max_damage_per_circuit must be >= 1, got {max_damage_per_circuit}")
    # Run message passing steps
    current_graphs = batch_graphs
    
    # Always initialize per-pattern storage to eliminate redundancy
    per_pattern_metrics = {
        "pattern_hard_accuracies": [],  # [n_steps, batch_size]
        "pattern_logits": [],  # [n_steps, batch_size] - store logits for each pattern
    }
    
    # Store initial step 0 data in per-pattern metrics
    if initial_hard_accuracies is not None and batch_logits is not None:
        per_pattern_metrics["pattern_hard_accuracies"].append(initial_hard_accuracies)
        per_pattern_metrics["pattern_logits"].append(batch_logits)
    
    vmap_get_loss = jax.vmap(
        lambda logits, wires: get_loss_from_wires_logits(
            logits, wires, x_data, y_data, loss_type
        )
    )
    
    vmap_update_loss = jax.vmap(
        lambda graph, res: update_output_node_loss(graph, layer_sizes, res.mean(axis=0))
    )
    
    vmap_extract_logits = jax.vmap(
        lambda graph: extract_logits_from_graph(graph, logits_original_shapes)
    )

    # Prepare injection schedule
    batch_size = batch_wires[0].shape[0]
    eval_perturb_counter = jp.zeros((batch_size,), dtype=jp.int32)
    damage_prob = max(1, int(greedy_window_size))  # Number of gates to knock out

    # Optional: use caller-provided patterns instead of generating per-step
    provided_knockout_patterns = None
    if knockout_patterns is not None:
        provided_knockout_patterns = (
            jp.stack(knockout_patterns)
            if isinstance(knockout_patterns, (list, tuple))
            else knockout_patterns
        )
        if provided_knockout_patterns.shape[0] != batch_size:
            raise ValueError(
                f"knockout_patterns batch mismatch: got {provided_knockout_patterns.shape[0]}, expected {batch_size}"
            )
    
    # Compute per-circuit damage start offsets
    if damage_start_offset_random:
        # Generate random offset per circuit (0 to damage_start_offset inclusive)
        offset_rng = jax.random.PRNGKey(damage_start_offset_seed)
        per_circuit_offsets = jax.random.randint(
            offset_rng, (batch_size,), 0, damage_start_offset + 1, dtype=jp.int32
        )
    else:
        # Fixed offset for all circuits
        per_circuit_offsets = jp.full((batch_size,), damage_start_offset, dtype=jp.int32)

    # Initialize cumulative damage pattern tracking for permanent damage mode
    # This ensures previously damaged nodes remain damaged across all subsequent steps
    damage_behavior = getattr(model, "damage_behavior", "permanent")
    total_nodes = sum(total_gates for total_gates, _ in layer_sizes)
    cumulative_knockout_patterns = jp.zeros((batch_size, total_nodes), dtype=jp.bool_)
    
    # DEBUG: Pre-update accuracy tracking initialization
    # Tracks accuracy AFTER damage is applied but BEFORE model updates (raw damage impact)
    pre_update_tracking = None
    if track_pre_update_accuracy:
        pre_update_tracking = {
            "steps": [],  # Steps where damage was injected
            "pre_update_hard_accuracy": [],  # Accuracy after damage bias, before model update
            "post_update_hard_accuracy": [],  # Accuracy after model update (for comparison)
        }

    # DEBUG: Damage validation tracking initialization for single-damage permanent mode
    # Just tracks: when damage happened, which nodes, and validates permanence
    damage_validation_tracking = None
    prev_node_logits = None  # For tracking magnitude of change
    prev_node_hidden = None
    if track_damage_validation:
        damage_validation_tracking = {
            "first_damage_step": None,  # Step when damage was first injected
            "damage_pattern": None,  # The damage pattern (set once)
            "validation_failures": [],  # List of (step, circuit_idx, details) for any failures
            "total_checks": 0,
            "passed_checks": 0,
            # Per-step magnitude of change (normalized per-gate average)
            "damaged_node_avg_change": [],  # Should be ~0 for permanently damaged nodes
            "healthy_node_avg_change": [],  # Should be non-zero for healthy nodes
        }
        # Store initial node features for change tracking
        prev_node_logits = current_graphs.nodes["logits"].copy()
        prev_node_hidden = current_graphs.nodes["hidden"].copy()

    for step in range(1, n_message_steps + 1):
        # Determine if damage should be injected at this step
        recover_steps = int(max(0, greedy_injection_recover_steps))
        steps_since_first_damage = step - per_circuit_offsets - 1
        first_damage = (steps_since_first_damage == 0)
        subsequent_damage = (steps_since_first_damage > 0) & ((steps_since_first_damage % (recover_steps + 1)) == 0)
        inject_now = first_damage | subsequent_damage
        
        # Respect maximum number of injections per circuit
        can_inject_mask = eval_perturb_counter < max_damage_per_circuit
        inject_now_mask = inject_now & can_inject_mask

        # Either reuse provided patterns or generate on the fly
        if provided_knockout_patterns is not None:
            generated_patterns = provided_knockout_patterns
        else:
            pattern_keys = jax.random.split(
                jax.random.PRNGKey(step + int(jp.sum(eval_perturb_counter)) + 1000),
                batch_size
            )
            vm_create_pattern = jax.vmap(
                lambda k: create_reproducible_knockout_pattern(
                    k, layer_sizes, damage_prob
                )
            )
            generated_patterns = vm_create_pattern(pattern_keys)
        
        # Apply patterns only to circuits that should inject this step
        step_knockout_patterns = jp.where(
            inject_now_mask[:, None], generated_patterns, jp.zeros((batch_size, total_nodes), dtype=jp.bool_)
        )

        # For reversible mode, force one-shot bias by zeroing step counters on injection steps
        if damage_behavior == "reversible":
            steps_before = current_graphs.globals[:, 1]
            steps_after = jp.where(inject_now_mask, jp.zeros_like(steps_before), steps_before)
            current_graphs = current_graphs._replace(
                globals=jp.stack([current_graphs.globals[:, 0], steps_after], axis=1)
            )

        # Increment eval_perturb_counter (only on injection steps)
        eval_perturb_counter = jp.where(inject_now_mask, eval_perturb_counter + 1, eval_perturb_counter)
        
        # For permanent damage: accumulate patterns using logical OR
        if damage_behavior == "permanent":
            cumulative_knockout_patterns = cumulative_knockout_patterns | step_knockout_patterns
            
            # DEBUG: Track first damage event for validation
            if track_damage_validation and damage_validation_tracking["first_damage_step"] is None:
                if jp.any(step_knockout_patterns):
                    damage_validation_tracking["first_damage_step"] = step
                    damage_validation_tracking["damage_pattern"] = cumulative_knockout_patterns.copy()

        # For permanent damage, use cumulative patterns; for reversible, use step patterns
        effective_knockout_patterns = (
            cumulative_knockout_patterns if damage_behavior == "permanent" 
            else step_knockout_patterns
        )

        # DEBUG: Measure pre-update accuracy (after damage bias, before model updates)
        # This captures the "raw damage" impact before the SA model can compensate
        if track_pre_update_accuracy and pre_update_tracking is not None and jp.any(inject_now_mask):
            # Extract current logits from graphs
            current_batch_logits_pre = vmap_extract_logits(current_graphs)
            
            # Apply damage bias to logits (same logic as reversible mode in model)
            # The model applies: current_logits + (apply_bias * bias_mask)
            # where bias_mask = knockout_pattern[:, None] * reversible_bias
            reversible_bias = getattr(model, "reversible_bias", -10.0)
            
            def apply_damage_bias_to_logits(logits_list, patterns, bias_value):
                """Apply damage bias to logits for circuits that are being damaged."""
                # logits_list is a list of arrays, one per layer
                # patterns is [batch_size, total_nodes] boolean mask
                # We need to split patterns by layer and apply bias
                damaged_logits = []
                node_offset = 0
                for layer_idx, layer_logits in enumerate(logits_list):
                    # layer_logits shape: [batch_size, n_nodes_in_layer, ...] (may have extra dims)
                    n_nodes_layer = layer_logits.shape[1]
                    layer_pattern = patterns[:, node_offset:node_offset + n_nodes_layer]
                    # Apply bias where pattern is True
                    # Expand bias_mask to match all trailing dimensions of layer_logits
                    bias_mask = layer_pattern * bias_value  # [batch, nodes]
                    # Add trailing dimensions to broadcast with layer_logits
                    for _ in range(layer_logits.ndim - 2):
                        bias_mask = bias_mask[..., None]
                    damaged_layer = layer_logits + bias_mask
                    damaged_logits.append(damaged_layer)
                    node_offset += n_nodes_layer
                return damaged_logits
            
            # Apply damage bias to get "raw damaged" logits
            damaged_logits_pre = apply_damage_bias_to_logits(
                current_batch_logits_pre, step_knockout_patterns, reversible_bias
            )
            
            # Compute accuracy on damaged-but-not-updated logits
            pre_update_losses, pre_update_aux = vmap_get_loss(damaged_logits_pre, batch_wires)
            pre_update_hard_accuracies = pre_update_aux[4]  # hard_accuracy is index 4
            
            # Store pre-update metrics
            pre_update_tracking["steps"].append(step)
            pre_update_tracking["pre_update_hard_accuracy"].append(float(jp.mean(pre_update_hard_accuracies)))
            
            print(f"  Step {step}: DEBUG pre-update hard_accuracy = {float(jp.mean(pre_update_hard_accuracies)):.4f}")

        # Apply model with per-step patterns if available
        if effective_knockout_patterns is not None and jp.any(effective_knockout_patterns):
            vmap_model = jax.vmap(
                lambda g, k: model(
                    g,
                    knockout_pattern=k,
                    layer_neighbors=layer_neighbors,
                    layer_sizes=layer_sizes,
                )
            )
            updated_graphs = vmap_model(current_graphs, effective_knockout_patterns)
        else:
            vmap_model = jax.vmap(
                lambda g: model(
                    g,
                    layer_neighbors=layer_neighbors,
                    layer_sizes=layer_sizes,
                )
            )
            updated_graphs = vmap_model(current_graphs)

        # DEBUG: PERMANENT DAMAGE VALIDATION (Vectorized)
        # Verify damaged node features remain at expected values: logits=-10.0, hidden=0.0
        if track_damage_validation and damage_behavior == "permanent" and jp.any(cumulative_knockout_patterns):
            node_logits = updated_graphs.nodes["logits"]  # [batch_size, n_nodes, logit_dim]
            node_hidden = updated_graphs.nodes["hidden"]  # [batch_size, n_nodes, hidden_dim]
            
            # Vectorized validation - no Python loops!
            # Expand mask for broadcasting: [batch, n_nodes] -> [batch, n_nodes, 1]
            mask_expanded = cumulative_knockout_patterns[:, :, None]
            
            # Check logits: damaged should be -10.0 (use large value for non-damaged to ignore them)
            logits_diff = jp.where(mask_expanded, jp.abs(node_logits - (-10.0)), 0.0)
            logits_ok_per_circuit = jp.all(logits_diff < 1e-6, axis=(1, 2))  # [batch]
            
            # Check hidden: damaged should be 0.0
            hidden_diff = jp.where(mask_expanded, jp.abs(node_hidden), 0.0)
            hidden_ok_per_circuit = jp.all(hidden_diff < 1e-6, axis=(1, 2))  # [batch]
            
            # Count passes (only for circuits with damage)
            both_ok = logits_ok_per_circuit & hidden_ok_per_circuit
            circuits_with_damage = jp.any(cumulative_knockout_patterns, axis=1)  # [batch]
            
            num_checks = int(jp.sum(circuits_with_damage))
            num_passed = int(jp.sum(both_ok & circuits_with_damage))
            
            damage_validation_tracking["total_checks"] += num_checks
            damage_validation_tracking["passed_checks"] += num_passed
            
            # Only record failures if any exist (rare path - keeps fast path fast)
            if num_passed < num_checks:
                failed_mask = circuits_with_damage & ~both_ok
                failed_indices = jp.where(failed_mask)[0]
                for idx in failed_indices[:3]:  # Limit to first 3 for performance
                    idx = int(idx)
                    mask = cumulative_knockout_patterns[idx]
                    damaged_logits = node_logits[idx][mask]
                    damaged_hidden = node_hidden[idx][mask]
                    damage_validation_tracking["validation_failures"].append({
                        "step": step,
                        "circuit_idx": idx,
                        "logits_range": (float(jp.min(damaged_logits)), float(jp.max(damaged_logits))),
                        "hidden_range": (float(jp.min(damaged_hidden)), float(jp.max(damaged_hidden))),
                        "n_damaged_nodes": int(jp.sum(mask)),
                    })

        # DEBUG: MAGNITUDE OF CHANGE TRACKING (Vectorized)
        # Track per-gate average change for damaged vs healthy nodes
        if track_damage_validation and prev_node_logits is not None:
            node_logits = updated_graphs.nodes["logits"]
            node_hidden = updated_graphs.nodes["hidden"]
            
            # Compute absolute change from previous step
            logits_change = jp.abs(node_logits - prev_node_logits)  # [batch, n_nodes, logit_dim]
            hidden_change = jp.abs(node_hidden - prev_node_hidden)  # [batch, n_nodes, hidden_dim]
            
            # Combined change magnitude per node (sum over feature dims, then mean over batch)
            # Shape: [batch, n_nodes]
            per_node_change = jp.mean(logits_change, axis=-1) + jp.mean(hidden_change, axis=-1)
            
            # Separate into damaged vs healthy using cumulative pattern
            if jp.any(cumulative_knockout_patterns):
                damaged_mask = cumulative_knockout_patterns  # [batch, n_nodes]
                healthy_mask = ~damaged_mask
                
                # Average change per damaged gate (should be ~0)
                n_damaged = jp.sum(damaged_mask)
                damaged_avg = float(jp.sum(per_node_change * damaged_mask) / jp.maximum(n_damaged, 1))
                
                # Average change per healthy gate (should be non-zero)
                n_healthy = jp.sum(healthy_mask)
                healthy_avg = float(jp.sum(per_node_change * healthy_mask) / jp.maximum(n_healthy, 1))
            else:
                # No damage yet - all nodes are healthy
                damaged_avg = 0.0
                healthy_avg = float(jp.mean(per_node_change))
            
            damage_validation_tracking["damaged_node_avg_change"].append(damaged_avg)
            damage_validation_tracking["healthy_node_avg_change"].append(healthy_avg)
            
            # Update previous for next step
            prev_node_logits = node_logits
            prev_node_hidden = node_hidden

        # Extract logits from updated graphs
        current_batch_logits = vmap_extract_logits(updated_graphs)

        # Calculate losses for the batch
        current_losses, current_aux = vmap_get_loss(current_batch_logits, batch_wires)
        (
            current_hard_losses,
            _,
            _,
            current_accuracies,
            current_hard_accuracies,
            current_full_map_accuracies,
            current_res,
            _,
        ) = [aux_elem for aux_elem in current_aux]

        # In blind mode, we zero out the residuals before updating the graph
        # This prevents the GNN from seeing the error signal (test-time optimization ablation)
        graph_update_res = current_res
        if blind_mode:
            graph_update_res = jp.zeros_like(current_res)

        # Update output node losses
        updated_graphs = vmap_update_loss(updated_graphs, graph_update_res)

        # Update globals with new losses and incremented steps
        current_steps = updated_graphs.globals[:, 1] + 1
        updated_graphs = updated_graphs._replace(
            globals=jp.stack([current_losses, current_steps], axis=1)
        )

        # DEBUG: Track post-update accuracy at damage injection step (for comparison with pre-update)
        if track_pre_update_accuracy and pre_update_tracking is not None and jp.any(inject_now_mask):
            pre_update_tracking["post_update_hard_accuracy"].append(float(jp.mean(current_hard_accuracies)))
            print(f"  Step {step}: DEBUG post-update hard_accuracy = {float(jp.mean(current_hard_accuracies)):.4f}")

        # Always store individual pattern metrics first (single source of truth)
        per_pattern_metrics["pattern_hard_accuracies"].append(current_hard_accuracies)
        per_pattern_metrics["pattern_logits"].append(current_batch_logits)

        # Store averaged metrics derived from individual values (eliminates redundancy)
        step_metrics["step"].append(step)
        step_metrics["soft_loss"].append(float(jp.mean(current_losses)))
        step_metrics["hard_loss"].append(float(jp.mean(current_hard_losses)))
        step_metrics["soft_accuracy"].append(float(jp.mean(current_accuracies)))
        step_metrics["hard_accuracy"].append(float(jp.mean(per_pattern_metrics["pattern_hard_accuracies"][-1])))
        step_metrics["full_map_accuracy"].append(float(jp.mean(current_full_map_accuracies)))
        step_metrics["logits_mean"].append(
            float(jp.mean(updated_graphs.nodes["logits"]))
        )

        current_graphs = updated_graphs

    # Always add per-pattern data to step_metrics, only expose if requested
    # Convert lists to arrays for easier analysis
    step_metrics_per_pattern = {
        "pattern_hard_accuracies": jp.array(per_pattern_metrics["pattern_hard_accuracies"]), # [n_steps, batch_size]
        "pattern_logits": per_pattern_metrics["pattern_logits"], # [n_steps, batch_size] - list of logit PyTrees
    }
    
    if return_per_pattern:
        step_metrics["per_pattern"] = step_metrics_per_pattern

    # DEBUG: Add pre-update accuracy tracking results
    if track_pre_update_accuracy and pre_update_tracking is not None:
        step_metrics["pre_update_tracking"] = {
            "damage_injection_steps": pre_update_tracking["steps"],
            "pre_update_hard_accuracy": pre_update_tracking["pre_update_hard_accuracy"],
            "post_update_hard_accuracy": pre_update_tracking["post_update_hard_accuracy"],
        }
        # Log summary
        if pre_update_tracking["steps"]:
            for i, step_num in enumerate(pre_update_tracking["steps"]):
                pre_acc = pre_update_tracking["pre_update_hard_accuracy"][i]
                post_acc = pre_update_tracking["post_update_hard_accuracy"][i]
                delta = post_acc - pre_acc
                print(f"DEBUG: Step {step_num} - Pre-update: {pre_acc:.4f}, Post-update: {post_acc:.4f}, "
                      f"Delta: {delta:+.4f} ({'recovery' if delta > 0 else 'degradation'})")

    # DEBUG: Add damage validation summary if enabled
    if track_damage_validation and damage_validation_tracking is not None:
        total = damage_validation_tracking["total_checks"]
        passed = damage_validation_tracking["passed_checks"]
        failures = damage_validation_tracking["validation_failures"]
        
        step_metrics["damage_validation_summary"] = {
            "first_damage_step": damage_validation_tracking["first_damage_step"],
            "total_checks": total,
            "passed_checks": passed,
            "pass_rate": passed / total if total > 0 else 1.0,
            "num_failures": len(failures),
        }
        
        # DEBUG: Add stepwise magnitude of change for wandb plotting
        # damaged_node_avg_change should be ~0, healthy_node_avg_change should be non-zero
        step_metrics["damaged_node_avg_change"] = damage_validation_tracking["damaged_node_avg_change"]
        step_metrics["healthy_node_avg_change"] = damage_validation_tracking["healthy_node_avg_change"]
        
        # Log summary
        if total > 0:
            import logging
            log = logging.getLogger(__name__)
            if len(failures) == 0:
                log.info(f"DAMAGE VALIDATION PASSED: {passed}/{total} checks passed (100%)")
            else:
                log.warning(f"DAMAGE VALIDATION FAILED: {passed}/{total} checks passed ({100*passed/total:.1f}%)")
                for f in failures[:5]:  # Show first 5 failures
                    log.warning(
                        f"  Step {f['step']}, Circuit {f['circuit_idx']}: "
                        f"logits={f['logits_range']}, hidden={f['hidden_range']}, "
                        f"n_damaged={f['n_damaged_nodes']}"
                    )
                if len(failures) > 5:
                    log.warning(f"  ... and {len(failures) - 5} more failures")
        
        # Include detailed failures for debugging if needed
        if failures:
            step_metrics["damage_validation_failures"] = failures

    return step_metrics
