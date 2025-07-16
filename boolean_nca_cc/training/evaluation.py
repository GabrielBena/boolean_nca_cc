"""
Evaluation functions for GNN-based boolean circuit optimization.

This module provides functions for evaluating the performance of GNN
models on optimizing boolean circuits.
"""

import jax
import jax.numpy as jp
import jraph
from tqdm.auto import tqdm
from typing import List, Dict, Tuple, Generator, NamedTuple, Optional

from boolean_nca_cc.models import CircuitSelfAttention
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
        where aux_data contains (hard_loss, pred, pred_hard, accuracy, hard_accuracy, res, hard_res)
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


# def evaluate_model_stepwise_generator(
#     model: CircuitGNN | CircuitSelfAttention,
#     wires: List[jp.ndarray],
#     logits: List[jp.ndarray],
#     x_data: jp.ndarray,
#     y_data: jp.ndarray,
#     input_n: int,
#     arity: int = 2,
#     circuit_hidden_dim: int = 16,
#     max_steps: int = None,
#     loss_type: str = "l4",
#     bidirectional_edges: bool = True,
#     layer_sizes: List[Tuple[int, int]] = None,
# ) -> Generator[StepResult, None, None]:
#     """
#     Generator that yields step-by-step evaluation results for GNN model optimization.

#     This function provides EXACTLY the same computation path as the training loop,
#     including using the same loss function, graph initialization, and step tracking.
#     Perfect for live demos and interactive use with zero discrepancy from training.

#     Args:
#         model: Trained CircuitGNN or CircuitSelfAttention model
#         wires: List of wire connection patterns
#         logits: List of initial logit tensors for each layer
#         x_data: Input data for evaluation
#         y_data: Target output data
#         input_n: Number of input nodes
#         arity: Number of inputs per gate
#         circuit_hidden_dim: Dimension of hidden features
#         max_steps: Maximum number of steps to run (None for infinite)
#         loss_type: Loss function to use
#         bidirectional_edges: Whether to use bidirectional edges

#     Yields:
#         StepResult: Results from each step including loss, accuracy, predictions, and updated logits
#     """
#     # Store original shapes for reconstruction (EXACTLY like training)
#     logits_original_shapes = [logit.shape for logit in logits]

#     # Calculate initial loss using the EXACT same function as training
#     (
#         initial_loss,
#         (
#             initial_hard_loss,
#             initial_pred,
#             initial_pred_hard,
#             initial_accuracy,
#             initial_hard_accuracy,
#             initial_res,
#             initial_hard_res,
#         ),
#     ) = get_loss_from_wires_logits(logits, wires, x_data, y_data, loss_type)

#     # Build initial graph using the same function as training
#     # Initialize with update_steps = 0 (exactly like training pool initialization)
#     graph = build_graph(
#         logits,
#         wires,
#         input_n,
#         arity,
#         circuit_hidden_dim,
#         loss_value=initial_loss,
#         bidirectional_edges=bidirectional_edges,
#     )

#     # Initialize graph globals with [loss, update_steps] exactly like training
#     current_update_steps = 0
#     graph = graph._replace(
#         globals=jp.array([initial_loss, current_update_steps], dtype=jp.float32)
#     )

#     graph = update_output_node_loss(graph, layer_sizes, initial_res.mean(axis=0))

#     # Yield initial state (step 0)
#     yield StepResult(
#         step=0,
#         loss=float(initial_loss),
#         hard_loss=float(initial_hard_loss),
#         accuracy=float(initial_accuracy),
#         hard_accuracy=float(initial_hard_accuracy),
#         predictions=initial_pred,
#         hard_predictions=initial_pred_hard,
#         logits=logits,
#         graph=graph,
#     )

#     # Run optimization steps (EXACTLY like the training loop)
#     step = 0
#     while max_steps is None or step < max_steps:
#         step += 1

#         # Extract the current update_steps count from graph globals (EXACTLY like training)
#         current_update_steps = 0
#         if graph.globals is not None and graph.globals.shape[-1] > 1:
#             current_update_steps = graph.globals[..., 1]

#         # Apply one step of model processing (EXACTLY like training inner loop)
#         # Note: training does multiple steps in a batch, but we do one at a time for live demo
#         updated_graph = model(graph)

#         # Use the unified get_loss_and_update_graph function for consistency
#         updated_graph, loss, current_logits, aux = get_loss_and_update_graph(
#             updated_graph,
#             logits_original_shapes,
#             wires,
#             x_data,
#             y_data,
#             loss_type,
#             layer_sizes,
#         )

#         # Extract auxiliary data
#         (
#             hard_loss,
#             pred,
#             pred_hard,
#             accuracy,
#             hard_accuracy,
#             res,
#             hard_res,
#         ) = aux

#         # Update with the computed loss and incremented update_steps (EXACTLY like training)
#         updated_graph = updated_graph._replace(
#             globals=jp.array([loss, current_update_steps + 1], dtype=jp.float32)
#         )

#         # Update the graph variable for next iteration
#         graph = updated_graph

#         # Yield current state
#         yield StepResult(
#             step=step,
#             loss=float(loss),
#             hard_loss=float(hard_loss),
#             accuracy=float(accuracy),
#             hard_accuracy=float(hard_accuracy),
#             predictions=pred,
#             hard_predictions=pred_hard,
#             logits=current_logits,
#             graph=graph,
#         )


# def evaluate_model_stepwise(
#     model: CircuitGNN | CircuitSelfAttention,
#     wires: List[jp.ndarray],
#     logits: List[jp.ndarray],
#     x_data: jp.ndarray,
#     y_data: jp.ndarray,
#     input_n: int,
#     arity: int = 2,
#     circuit_hidden_dim: int = 16,
#     n_message_steps: int = 100,
#     loss_type: str = "l4",
#     bidirectional_edges: bool = True,
#     layer_sizes: List[Tuple[int, int]] = None,
#     use_tqdm: bool = False,
# ) -> Dict:
#     """
#     Evaluate GNN performance by running message passing steps one by one
#     and collecting metrics at each step.

#     This function now uses the generator implementation to ensure consistency
#     with the step-by-step evaluation used in demos.

#     Args:
#         model: Trained CircuitGNN or CircuitSelfAttention model
#         wires: List of wire connection patterns
#         logits: List of initial logit tensors for each layer
#         x_data: Input data for evaluation
#         y_data: Target output data
#         input_n: Number of input nodes
#         arity: Number of inputs per gate
#         circuit_hidden_dim: Dimension of hidden features
#         n_message_steps: Maximum number of message passing steps to run
#         loss_type: Loss function to use
#         bidirectional_edges: Whether to use bidirectional edges

#     Returns:
#         Dictionary with metrics collected at each step
#     """
#     # Initialize metric storage
#     step_metrics = {
#         "step": [],
#         "soft_loss": [],
#         "hard_loss": [],
#         "soft_accuracy": [],
#         "hard_accuracy": [],
#         "logits_mean": [],
#     }

#     # Use the generator to collect all results
#     generator = evaluate_model_stepwise_generator(
#         model=model,
#         wires=wires,
#         logits=logits,
#         x_data=x_data,
#         y_data=y_data,
#         input_n=input_n,
#         arity=arity,
#         circuit_hidden_dim=circuit_hidden_dim,
#         max_steps=n_message_steps,
#         loss_type=loss_type,
#         bidirectional_edges=bidirectional_edges,
#         layer_sizes=layer_sizes,
#     )

#     # Create progress bar for evaluation
#     if use_tqdm:
#         pbar = tqdm(generator, total=n_message_steps + 1, desc="Evaluating model steps")
#     else:
#         pbar = generator

#     # Collect all results
#     for result in pbar:
#         step_metrics["step"].append(result.step)
#         step_metrics["soft_loss"].append(result.loss)
#         step_metrics["hard_loss"].append(result.hard_loss)
#         step_metrics["soft_accuracy"].append(result.accuracy)
#         step_metrics["hard_accuracy"].append(result.hard_accuracy)
#         step_metrics["logits_mean"].append(float(result.graph.nodes["logits"].mean()))

#         if use_tqdm:
#             # Update progress bar
#             pbar.set_postfix(
#                 {
#                     "Loss": f"{result.loss:.4f}",
#                     "Accuracy": f"{result.accuracy:.4f}",
#                     "Hard Acc": f"{result.hard_accuracy:.4f}",
#                 }
#             )

#     return step_metrics


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
    # use_scan: bool = False,
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
        use_scan: Whether to use scan mode (should match training configuration)

    Returns:
        Dictionary with averaged metrics collected at each step
    """
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
        "logits_mean": [],
    }

    # Store initial metrics (averaged across batch)
    step_metrics["step"].append(0)
    step_metrics["soft_loss"].append(float(jp.mean(initial_losses)))
    step_metrics["hard_loss"].append(float(jp.mean(initial_hard_losses)))
    step_metrics["soft_accuracy"].append(float(jp.mean(initial_accuracies)))
    step_metrics["hard_accuracy"].append(float(jp.mean(initial_hard_accuracies)))
    step_metrics["logits_mean"].append(float(jp.mean(batch_graphs.nodes["logits"])))

    # Store original shapes for reconstruction
    logits_original_shapes = [
        logit.shape[1:] for logit in batch_logits
    ]  # Remove batch dim

    # if use_scan:
    #     # Use scan mode to match training behavior
    #     return _evaluate_with_scan(
    #         model=model,
    #         batch_graphs=batch_graphs,
    #         batch_wires=batch_wires,
    #         batch_logits=batch_logits,
    #         x_data=x_data,
    #         y_data=y_data,
    #         n_message_steps=n_message_steps,
    #         loss_type=loss_type,
    #         layer_sizes=layer_sizes,
    #         knockout_patterns=knockout_patterns,
    #         logits_original_shapes=logits_original_shapes,
    #         step_metrics=step_metrics,
    #     )
    # else:
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
        if isinstance(chunk_results[0][key], list):
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




# def _evaluate_with_scan(
#     model,
#     batch_graphs: jraph.GraphsTuple,
#     batch_wires: List[jp.ndarray],
#     batch_logits: List[jp.ndarray],
#     x_data: jp.ndarray,
#     y_data: jp.ndarray,
#     n_message_steps: int,
#     loss_type: str,
#     layer_sizes: List[Tuple[int, int]],
#     knockout_patterns: Optional[jp.ndarray],
#     logits_original_shapes: List[Tuple],
#     step_metrics: Dict,
# ) -> Dict:
#     """
#     Evaluate using scan mode to match training behavior.
#     """
#     from boolean_nca_cc.models.self_attention import run_self_attention_scan_with_loss
#     from boolean_nca_cc.models.gnn import run_gnn_scan_with_loss

#     def evaluate_single_circuit_scan(graph, wires, logits, knockout_pattern):
#         # Determine which scan function to use based on model type
#         if isinstance(model, CircuitSelfAttention):
#             scan_fn = run_self_attention_scan_with_loss
#         elif isinstance(model, CircuitGNN):
#             scan_fn = run_gnn_scan_with_loss
#         else:
#             raise ValueError(f"Unknown model type: {type(model)}")

#         # Run scan for all steps, computing loss and updating graph at each step
#         final_graph, step_outputs = scan_fn(
#             model=model,
#             graph=graph,
#             num_steps=n_message_steps,
#             logits_original_shapes=logits_original_shapes,
#             wires=wires,
#             x_data=x_data,
#             y_data=y_data,
#             loss_type=loss_type,
#             layer_sizes=tuple(layer_sizes),  # Convert to tuple for JAX
#             knockout_pattern=knockout_pattern,
#         )

#         return step_outputs

#     # Split batched graphs into individual graphs for processing
#     individual_graphs = jraph.unbatch(batch_graphs)
    
#     # Handle knockout patterns
#     if knockout_patterns is not None:
#         all_step_outputs = []
#         for i, (graph, wires, logits, knockout_pattern) in enumerate(
#             zip(individual_graphs, batch_wires, batch_logits, knockout_patterns)
#         ):
#             step_outputs = evaluate_single_circuit_scan(graph, wires, logits, knockout_pattern)
#             all_step_outputs.append(step_outputs)
#     else:
#         all_step_outputs = []
#         for i, (graph, wires, logits) in enumerate(
#             zip(individual_graphs, batch_wires, batch_logits)
#         ):
#             step_outputs = evaluate_single_circuit_scan(graph, wires, logits, None)
#             all_step_outputs.append(step_outputs)

#     # Extract and average metrics across batch and steps
#     # all_step_outputs is (batch_size, num_steps, step_data_tuple)
#     # step_data_tuple = (final_graph, loss, current_logits, aux)
    
#     batch_size = len(all_step_outputs)
    
#     for step in range(n_message_steps):
#         # Extract data for this step across all circuits in batch
#         step_losses = jp.array([all_step_outputs[b][step][1] for b in range(batch_size)])
#         step_aux = [all_step_outputs[b][step][3] for b in range(batch_size)]
        
#         # Extract individual aux components and average across batch
#         step_hard_losses = jp.mean(jp.array([aux[1] for aux in step_aux]))
#         step_accuracies = jp.mean(jp.array([aux[4] for aux in step_aux]))
#         step_hard_accuracies = jp.mean(jp.array([aux[5] for aux in step_aux]))
        
#         # Extract logits for mean calculation
#         step_logits = jp.array([all_step_outputs[b][step][2] for b in range(batch_size)])
#         step_logits_mean = jp.mean(jp.concatenate([jp.concatenate(logits) for logits in step_logits]))

#         # Store averaged metrics
#         step_metrics["step"].append(step + 1)
#         step_metrics["soft_loss"].append(float(jp.mean(step_losses)))
#         step_metrics["hard_loss"].append(float(step_hard_losses))
#         step_metrics["soft_accuracy"].append(float(step_accuracies))
#         step_metrics["hard_accuracy"].append(float(step_hard_accuracies))
#         step_metrics["logits_mean"].append(float(step_logits_mean))

#     return step_metrics


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
) -> Dict:
    """
    Evaluate using loop mode (original behavior).
    """
    # Run message passing steps
    current_graphs = batch_graphs
    
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

    for step in range(1, n_message_steps + 1):
        # Apply model to all graphs in batch
        if knockout_patterns is not None:
            # Use a lambda to correctly map over the knockout_pattern keyword argument
            vmap_model = jax.vmap(lambda g, k: model(g, knockout_pattern=k))
            updated_graphs = vmap_model(current_graphs, knockout_patterns)
        else:
            vmap_model = jax.vmap(model)
            updated_graphs = vmap_model(current_graphs)

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
            current_res,
            _,
        ) = [aux_elem for aux_elem in current_aux]

        # Update output node losses
        updated_graphs = vmap_update_loss(updated_graphs, current_res)

        # Update globals with new losses and incremented steps
        current_steps = updated_graphs.globals[:, 1] + 1
        updated_graphs = updated_graphs._replace(
            globals=jp.stack([current_losses, current_steps], axis=1)
        )

        # Store averaged metrics
        step_metrics["step"].append(step)
        step_metrics["soft_loss"].append(float(jp.mean(current_losses)))
        step_metrics["hard_loss"].append(float(jp.mean(current_hard_losses)))
        step_metrics["soft_accuracy"].append(float(jp.mean(current_accuracies)))
        step_metrics["hard_accuracy"].append(float(jp.mean(current_hard_accuracies)))
        step_metrics["logits_mean"].append(
            float(jp.mean(updated_graphs.nodes["logits"]))
        )

        current_graphs = updated_graphs

    return step_metrics
