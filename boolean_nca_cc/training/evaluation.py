"""
Evaluation functions for GNN-based boolean circuit optimization.

This module provides functions for evaluating the performance of GNN
models on optimizing boolean circuits.
"""

import logging
from collections.abc import Generator
from typing import NamedTuple

import jax
import jax.numpy as jp
from flax import nnx

from boolean_nca_cc.circuits.model import run_circuit
from boolean_nca_cc.circuits.train import (
    LOSS_L4,
    LossConfig,
    compute_loss_from_predictions,
)
from boolean_nca_cc.models import (
    CircuitGatheredAttention,
    CircuitGNN,
    CircuitSelfAttention,
    PerceiverCircuitAttention,
)
from boolean_nca_cc.training.pool.structural_perturbation import (
    apply_knockout_to_circuit,
    apply_probabilistic_gate_failure,
    create_eligible_gate_mask,
)
from boolean_nca_cc.utils import (
    extract_logits_from_graph,
    update_output_node_from_residuals,
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
):
    """
    Extract logits from graph, compute loss and residuals, and update graph with loss information.

    This function performs the 3-step process after a model has been applied to a graph:
    1. Extract logits from the updated graph
    2. Compute loss and residuals using the circuit
    3. Update the graph's output node loss features with residuals
    4. Update GraphGlobals with loss, update_steps, x_data, y_data, and residuals

    All models receive the same GraphGlobals structure for consistency.
    Models that don't use all fields (e.g., GNN doesn't use x_data/y_data/residuals)
    simply ignore them.

    Args:
        graph: Updated graph from model application
        logits_original_shapes: Original shapes of logits for reconstruction
        wires: Wire connection patterns
        x_data: Input data
        y_data: Target output data
        loss_cfg: Loss config dict
        layer_sizes: List of (nodes, group_size) tuples for each layer

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
    updated_graph = update_output_node_from_residuals(
        graph, layer_sizes, jp.abs(residuals).mean(axis=0)
    )

    # Always update GraphGlobals with full information for consistency across all models
    current_update_steps = (
        updated_graph.globals.update_steps if updated_graph.globals is not None else 0
    )

    # Carry forward the subsample key (constant base key; per-step keys derived via fold_in)
    subsample_key = (
        updated_graph.globals.subsample_key if updated_graph.globals is not None else None
    )

    updated_graph = updated_graph._replace(
        globals=GraphGlobals(
            loss=loss,
            update_steps=current_update_steps + 1,
            x_data=x_data,
            y_data=y_data,
            residuals=residuals,
            subsample_key=subsample_key,
        )
    )

    return updated_graph, loss, current_logits, aux_data


# =============================================================================
# Unified Model Step and Scan Functions
# =============================================================================


def apply_model_and_compute_loss(
    model_fn,
    graph: jp.ndarray,
    logits_original_shapes: list[tuple],
    wires: list[jp.ndarray],
    x_data: jp.ndarray,
    y_data: jp.ndarray,
    loss_cfg,
    layer_sizes: list[tuple[int, int]],
):
    """
    Core step function: apply model and compute loss with graph update.

    This is the single source of truth for applying a model step and computing loss.
    Used by:
    - run_model_scan_with_loss (training scan)
    - evaluate_model_stepwise_generator (demo/continuous mode)
    - evaluate_model_stepwise (JIT evaluation)

    Args:
        model_fn: Callable that takes a graph and returns an updated graph.
                  Should be pre-wrapped with any model-specific preprocessing
                  (e.g., attention masks, output gates).
        graph: Current graph state
        logits_original_shapes: Original shapes of logits for reconstruction
        wires: Wire connection patterns
        x_data: Input data
        y_data: Target output data
        loss_cfg: Loss configuration
        layer_sizes: List of (nodes, group_size) tuples for each layer

    Returns:
        Tuple of (updated_graph, loss, current_logits, aux_data)
    """
    # Apply model
    model_updated_graph = model_fn(graph)

    # Compute loss and update graph (always updates full GraphGlobals)
    updated_graph, loss, current_logits, aux_data = get_loss_and_update_graph(
        model_updated_graph,
        logits_original_shapes,
        wires,
        x_data,
        y_data,
        loss_cfg,
        layer_sizes,
    )

    return updated_graph, loss, current_logits, aux_data


def _prepare_model_fn(
    model: CircuitGatheredAttention | CircuitGNN | CircuitSelfAttention | PerceiverCircuitAttention,
    graph,
    gradient_checkpointing: bool = False,
):
    """
    Prepare a model function with precomputed masks for attention models.

    Returns a callable that takes a graph and returns an updated graph,
    with all model-specific preprocessing (attention masks, output gates) pre-applied.

    Args:
        model: The model to wrap
        graph: Graph (used to precompute masks)
        gradient_checkpointing: Whether to wrap with nnx.remat

    Returns:
        Tuple of (model_fn, attention_mask) where:
        - model_fn: Callable[graph] -> updated_graph
        - attention_mask: Precomputed mask (or None for GNN/gathered)
    """
    attention_mask = None
    input_output_gate = None
    output_output_gate = None

    if isinstance(model, PerceiverCircuitAttention):
        # Perceiver: precompute attention mask and output gates
        attention_mask = model._create_attention_mask(graph.senders, graph.receivers, model.n_node)

        layer_indices = graph.nodes["layer"]
        max_layer = jp.max(layer_indices)

        if model.restrict_input_cross_attn_to_first_layer:
            input_output_gate = model._create_output_gate(layer_indices, allowed_layer=0)

        if model.restrict_output_cross_attn_to_last_layer:
            output_output_gate = model._create_output_gate(layer_indices, allowed_layer=max_layer)

        def base_fn(g):
            return model(
                g,
                attention_mask=attention_mask,
                input_output_gate=input_output_gate,
                output_output_gate=output_output_gate,
            )

    elif isinstance(model, CircuitSelfAttention):
        # Self-attention: precompute attention mask
        attention_mask = model._create_attention_mask(graph.senders, graph.receivers, model.n_node)

        def base_fn(g):
            return model(g, attention_mask=attention_mask)

    elif isinstance(model, CircuitGatheredAttention):
        # Gathered attention: precompute neighbor indices from graph topology
        from boolean_nca_cc.models.attention.base import build_neighbor_indices

        n_node = graph.nodes["layer"].shape[0]
        neighbor_indices, neighbor_mask = build_neighbor_indices(
            graph.senders,
            graph.receivers,
            n_node,
            model.max_neighbors,
            model.use_attention_mask,
        )

        def base_fn(g):
            return model(g, neighbor_indices=neighbor_indices, neighbor_mask=neighbor_mask)

    else:
        # GNN: no preprocessing needed
        base_fn = model

    # Optionally wrap with gradient checkpointing
    model_fn = nnx.remat(base_fn) if gradient_checkpointing else base_fn

    return model_fn, attention_mask


def run_model_scan_with_loss(
    model: CircuitGatheredAttention | CircuitGNN | CircuitSelfAttention | PerceiverCircuitAttention,
    graph,
    num_steps: int,
    logits_original_shapes: list[tuple],
    wires: list[jp.ndarray],
    x_data: jp.ndarray,
    y_data: jp.ndarray,
    loss_cfg: LossConfig,
    layer_sizes: tuple[tuple[int, int], ...],
    data_fraction: float = 1.0,
    scan_key: jax.random.PRNGKey = None,
    gradient_checkpointing: bool = False,
    # Discrete damage parameters (for visualization)
    damage_steps: jp.ndarray | None = None,
    knockout_per_damage_step: int = 1,
    # Probabilistic damage parameters (for training)
    p_fault: float | None = None,
    faulty_value: float = -10.0,
    permanent_damage: bool = True,
    # Delayed probabilistic damage onset
    p_fault_onset_step: int = 0,
    # No-repair baseline: compute loss right after damage, before NCA runs
    compute_no_repair_baseline: bool = False,
):
    """
    Unified scan function for all model types with loss computation at each step.

    This replaces the model-specific scan functions:
    - run_gnn_scan_with_loss
    - run_self_attention_scan_with_loss
    - run_perceiver_scan_with_loss

    Supports two damage modes (can be used together):

    1. **Discrete mode** (damage_steps): Apply N knockouts at specific steps.
       Good for visualization with visible damage markers.

    2. **Probabilistic mode** (p_fault): Each gate has p_fault probability of
       failing at each step. Realistic continuous failure model for training.

    Args:
        model: Any supported model (CircuitGNN, CircuitSelfAttention, PerceiverCircuitAttention)
        graph: Initial graph state
        num_steps: Number of optimization steps
        logits_original_shapes: Original shapes of logits for reconstruction
        wires: Wire connection patterns
        x_data: Input data [N_samples, N_input_bits]
        y_data: Target output [N_samples, N_output_bits]
        loss_cfg: Loss configuration
        layer_sizes: Layer sizes for graph operations
        data_fraction: Fraction of data to use (for stochastic training)
        scan_key: Random key for data sampling and damage
        gradient_checkpointing: Whether to use gradient checkpointing (remat)
        damage_steps: Array of step indices at which to apply discrete damage (None = disabled)
        knockout_per_damage_step: Number of gates to knock out at each discrete damage step
        p_fault: Per-gate-per-step failure probability (None = disabled, 0.0 = no failures)
        faulty_value: Value to set for failed gate logits (large negative for zero output)
        permanent_damage: Whether to apply permanent damage to gates (True) or temporary damage (False)
        p_fault_onset_step: Step at which probabilistic damage starts (0 = from the start).
            Use to let the circuit converge first, then introduce faults for repair testing.
        compute_no_repair_baseline: If True, compute loss/accuracy right after damage is
            applied but before the NCA model runs. This measures the raw damage impact without
            repair, enabling comparison of "damaged without NCA" vs "damaged with NCA recovery".
            Adds keys ``no_repair_loss``, ``no_repair_hard_loss``, ``no_repair_accuracy``,
            ``no_repair_hard_accuracy`` to the aux_data in step outputs.

    Returns:
        Tuple of (final_graph, step_outputs) where step_outputs contains
        (graphs, losses, logits, aux_data) for each step
    """
    # Split key for data sampling, damage, and token subsampling
    if scan_key is not None:
        data_key, damage_key, subsample_key = jax.random.split(scan_key, 3)
    else:
        data_key = None
        damage_key = jax.random.PRNGKey(42)
        subsample_key = None

    # Select data subset if needed
    if data_fraction < 1.0 and data_key is not None:
        random_indices = jax.random.randint(
            key=data_key,
            shape=(int(x_data.shape[0] * data_fraction),),
            minval=0,
            maxval=x_data.shape[0],
        )
        x_batch = x_data[random_indices]
        y_batch = y_data[random_indices]
    else:
        x_batch = x_data
        y_batch = y_data

    # Prepare model function with precomputed masks
    model_fn, _ = _prepare_model_fn(model, graph, gradient_checkpointing)

    # Initialize graph globals with initial loss computation and batch data
    graph, _, _, _ = get_loss_and_update_graph(
        graph=graph,
        logits_original_shapes=logits_original_shapes,
        wires=wires,
        x_data=x_batch,
        y_data=y_batch,
        loss_cfg=loss_cfg,
        layer_sizes=layer_sizes,
    )

    # Inject subsample key for stochastic token subsampling (Perceiver only)
    # The key is constant; per-step variation comes from fold_in(key, update_steps)
    if subsample_key is not None:
        graph = graph._replace(globals=graph.globals._replace(subsample_key=subsample_key))

    # === Discrete damage setup ===
    # Precompute damage keys for all potential damage steps
    # This avoids dynamic key splitting inside the scan
    if damage_steps is not None and len(damage_steps) > 0:
        discrete_key, prob_key = jax.random.split(damage_key)
        n_damage = len(damage_steps)
        discrete_damage_keys = jax.random.split(discrete_key, n_damage)
        # Convert damage_steps to array for JIT compatibility
        damage_steps_array = jp.asarray(damage_steps)
        discrete_damage_enabled = True
    else:
        prob_key = damage_key
        discrete_damage_keys = None
        damage_steps_array = jp.array([], dtype=jp.int32)
        discrete_damage_enabled = False

    # === Probabilistic damage setup ===
    prob_damage_enabled = p_fault is not None and p_fault > 0.0
    if prob_damage_enabled:
        eligible_mask = create_eligible_gate_mask(layer_sizes)
        # Pre-split keys for all steps (for reproducibility and JIT compatibility)
        prob_damage_keys = jax.random.split(prob_key, num_steps)
    else:
        eligible_mask = None
        prob_damage_keys = None

    def apply_discrete_damage_if_needed(graph, step_idx, gate_mask):
        """Apply damage at specific steps using vectorized conditional."""
        if not discrete_damage_enabled:
            return graph, gate_mask

        # Check if current step is a damage step
        is_damage_step = jp.any(damage_steps_array == step_idx)

        # Get the damage key index (which damage event this is)
        damage_idx = jp.searchsorted(damage_steps_array, step_idx)
        damage_idx = jp.clip(damage_idx, 0, len(discrete_damage_keys) - 1)

        def do_damage(_):
            # Apply knockout
            modified_logits, modified_gate_mask = apply_knockout_to_circuit(
                discrete_damage_keys[damage_idx],
                graph.nodes["logits"],
                layer_sizes,
                num_knockouts=knockout_per_damage_step,
                flat=True,
            )
            new_gate_mask = gate_mask * modified_gate_mask
            new_gate_mask = jax.lax.cond(permanent_damage, lambda: new_gate_mask, lambda: gate_mask)

            new_nodes = {
                **graph.nodes,
                "logits": modified_logits,
                # If permanent damage, update the gate mask to the new mask, otherwise keep the original gate mask
                # "gate_knockout_mask": new_gate_mask if permanent_damage else gate_mask,
                "gate_knockout_mask": new_gate_mask,
            }
            return graph._replace(nodes=new_nodes), new_gate_mask

        def no_damage(_):
            return graph, gate_mask

        return jax.lax.cond(is_damage_step, do_damage, no_damage, None)

    def apply_probabilistic_damage(graph, step_idx, gate_mask):
        """Apply probabilistic damage: each gate has p_fault chance of failure.

        Respects ``p_fault_onset_step``: damage is only applied when
        ``step_idx >= p_fault_onset_step``, allowing the circuit to converge
        to a functional state before faults are introduced.
        """
        if not prob_damage_enabled:
            return graph, gate_mask

        # Get the pre-split key for this step
        step_key = prob_damage_keys[step_idx]

        # Apply probabilistic failure (always compute for JIT consistency)
        new_logits, new_mask = apply_probabilistic_gate_failure(
            step_key,
            graph.nodes["logits"],
            gate_mask,
            eligible_mask,
            p_fault,
            faulty_value,
        )

        new_gate_mask = jax.lax.cond(permanent_damage, lambda: new_mask, lambda: gate_mask)

        # Gate on onset step: only actually apply damage if past the onset
        past_onset = step_idx >= p_fault_onset_step
        final_logits = jp.where(past_onset, new_logits, graph.nodes["logits"])
        final_gate_mask = jp.where(past_onset, new_gate_mask, gate_mask)

        new_nodes = {
            **graph.nodes,
            "logits": final_logits,
            "gate_knockout_mask": final_gate_mask,
        }

        return graph._replace(nodes=new_nodes), final_gate_mask

    # Capture initial gate mask for damage detection
    initial_gate_mask = graph.nodes["gate_knockout_mask"]

    if compute_no_repair_baseline:
        # ------------------------------------------------------------------
        # No-repair baseline: carry a *forked* graph through the scan.
        #
        # Before the first damage event the no-repair ("nr") graph mirrors
        # the NCA-updated main graph exactly (same logits, same hidden
        # state).  The moment any gate is knocked out the nr graph "forks":
        # it keeps accumulating damage identically to the main graph but
        # *never* receives another NCA update.  This gives a clean
        # comparison of "damaged + NCA repair" vs "damaged, no repair".
        # ------------------------------------------------------------------

        def scan_step(carry, step_idx):
            (
                current_graph,
                current_gate_mask,
                nr_graph,
                nr_gate_mask,
                damage_started,
            ) = carry

            # --- Sync nr graph with main BEFORE this step's damage --------
            # While no damage has occurred yet, the nr graph is an exact
            # copy of the (NCA-updated) main graph.  Once damage_started
            # flips to True the nr graph keeps its own (un-repaired) state.
            synced_nodes = jax.tree.map(
                lambda nr_v, main_v: jp.where(damage_started, nr_v, main_v),
                nr_graph.nodes,
                current_graph.nodes,
            )
            nr_graph = nr_graph._replace(nodes=synced_nodes)
            nr_gate_mask = jp.where(damage_started, nr_gate_mask, current_gate_mask)

            # --- 1. Apply damage to BOTH graphs (same keys → same pattern) -
            current_graph, current_gate_mask = apply_probabilistic_damage(
                current_graph,
                step_idx,
                current_gate_mask,
            )
            nr_graph, nr_gate_mask = apply_probabilistic_damage(
                nr_graph,
                step_idx,
                nr_gate_mask,
            )

            current_graph, current_gate_mask = apply_discrete_damage_if_needed(
                current_graph,
                step_idx + 1,
                current_gate_mask,
            )
            nr_graph, nr_gate_mask = apply_discrete_damage_if_needed(
                nr_graph,
                step_idx + 1,
                nr_gate_mask,
            )

            # --- 2. Detect whether damage has now started ------------------
            # Any gate that was active in the initial mask and is now
            # knocked out means real damage has occurred.
            damage_started = damage_started | jp.any(nr_gate_mask < initial_gate_mask)

            # --- 3. Compute no-repair baseline loss (no NCA model) ---------
            nr_logits = extract_logits_from_graph(nr_graph, logits_original_shapes)
            _, no_repair_aux = get_loss_from_wires_logits(
                logits=nr_logits,
                wires=wires,
                x=x_batch,
                y_target=y_batch,
                loss_cfg=loss_cfg,
            )

            # --- 4. Apply NCA model ONLY to the main graph -----------------
            updated_graph, loss, current_logits, aux = apply_model_and_compute_loss(
                model_fn,
                current_graph,
                logits_original_shapes,
                wires,
                x_batch,
                y_batch,
                loss_cfg,
                layer_sizes,
            )

            # Merge no-repair metrics into aux dict
            aux = {
                **aux,
                "no_repair_loss": no_repair_aux["loss"],
                "no_repair_hard_loss": no_repair_aux["hard_loss"],
                "no_repair_accuracy": no_repair_aux["accuracy"],
                "no_repair_hard_accuracy": no_repair_aux["hard_accuracy"],
            }

            new_carry = (
                updated_graph,
                current_gate_mask,
                nr_graph,
                nr_gate_mask,
                damage_started,
            )
            return new_carry, (updated_graph, loss, current_logits, aux)

        # Initial carry: nr graph starts as a copy of the main graph
        init_carry = (
            graph,
            initial_gate_mask,
            graph,
            initial_gate_mask,
            jp.bool_(False),
        )
        (final_graph, _, _, _, _), step_outputs = jax.lax.scan(
            scan_step,
            init_carry,
            xs=jp.arange(num_steps),
            length=num_steps,
        )

    else:
        # ------------------------------------------------------------------
        # Standard scan (no baseline tracking)
        # ------------------------------------------------------------------
        def scan_step(carry, step_idx):
            current_graph, current_gate_mask = carry

            current_graph, current_gate_mask = apply_probabilistic_damage(
                current_graph,
                step_idx,
                current_gate_mask,
            )
            current_graph, current_gate_mask = apply_discrete_damage_if_needed(
                current_graph,
                step_idx + 1,
                current_gate_mask,
            )

            updated_graph, loss, current_logits, aux = apply_model_and_compute_loss(
                model_fn,
                current_graph,
                logits_original_shapes,
                wires,
                x_batch,
                y_batch,
                loss_cfg,
                layer_sizes,
            )

            return (updated_graph, current_gate_mask), (updated_graph, loss, current_logits, aux)

        (final_graph, _), step_outputs = jax.lax.scan(
            scan_step,
            (graph, initial_gate_mask),
            xs=jp.arange(num_steps),
            length=num_steps,
        )

    return final_graph, step_outputs


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
    model: CircuitGatheredAttention | CircuitGNN | CircuitSelfAttention | PerceiverCircuitAttention,
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
    # Discrete damage parameters (for visualization)
    knockout_per_damage_step: int = 1,
    damage_key: jax.random.PRNGKey = jax.random.PRNGKey(42),
    # Probabilistic damage parameters (for training)
    p_fault: float | None = None,
    faulty_value: float = -10.0,
    permanent_damage: bool = True,
    # Delayed probabilistic damage onset
    p_fault_onset_step: int = 0,
    verbose: bool = False,
) -> Generator[StepResult, None, None]:
    """
    Generator that yields step-by-step evaluation results for model optimization.

    This function uses the same core step logic as training (via apply_model_and_compute_loss),
    but yields results one at a time for live demos and interactive use.

    For JIT-compiled evaluation with known step count, use evaluate_model_stepwise instead,
    which internally uses run_model_scan_with_loss.

    Args:
        model: Trained model (CircuitGNN, CircuitSelfAttention, or PerceiverCircuitAttention)
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
        layer_sizes: List of (nodes, group_size) tuples for each layer
        damage_steps: List of step indices at which to apply damage
        knockout_per_damage_step: Number of gates to knock out at each damage step
        damage_key: Random key for damage
        verbose: Print damage info
        p_fault: Per-gate-per-step failure probability (None = disabled, 0.0 = no failures)
        faulty_value: Value to set for failed gate logits (large negative for zero output)
        permanent_damage: Whether to apply permanent damage to gates (True) or temporary damage (False)
        p_fault_onset_step: Step at which probabilistic damage starts (0 = from the start)

    Yields:
        StepResult: Results from each step including loss, accuracy, predictions, and updated logits
    """
    if loss_cfg is None:
        loss_cfg = LOSS_L4

    # Store original shapes for reconstruction
    logits_original_shapes = [logit.shape for logit in logits]

    # Build initial graph
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

    # Initialize graph with loss computation (uses unified function)
    graph, _, _, aux_data = get_loss_and_update_graph(
        graph,
        logits_original_shapes,
        wires,
        x_data,
        y_data,
        loss_cfg,
        layer_sizes,
    )

    # Yield initial state (step 0)
    yield StepResult(
        step=0,
        logits=logits,
        graph=graph,
        **aux_data,
    )

    # Prepare model function with precomputed masks (same as training)
    model_fn, _ = _prepare_model_fn(model, graph, gradient_checkpointing=False)

    gate_mask = graph.nodes["gate_knockout_mask"]

    # Prepare probabilistic damage resources (for generator, not JIT)
    prob_damage_enabled = p_fault is not None and p_fault > 0.0
    if prob_damage_enabled:
        eligible_mask = create_eligible_gate_mask(layer_sizes)
        prob_damage_key = jax.random.PRNGKey(99)  # separate key stream for prob damage

    # Run optimization steps
    step = 0
    while max_steps is None or step < max_steps:
        step += 1

        # Apply probabilistic damage (respecting onset step)
        if prob_damage_enabled and step >= p_fault_onset_step:
            prob_damage_key, step_key = jax.random.split(prob_damage_key)
            new_logits, new_mask = apply_probabilistic_gate_failure(
                step_key,
                graph.nodes["logits"],
                gate_mask,
                eligible_mask,
                p_fault,
                faulty_value,
            )
            new_gate_mask = new_mask if permanent_damage else gate_mask
            new_nodes = {
                **graph.nodes,
                "logits": new_logits,
                "gate_knockout_mask": new_gate_mask,
            }
            graph = graph._replace(nodes=new_nodes)
            gate_mask = new_gate_mask

        # Apply discrete damage if needed
        if damage_steps is not None and step in damage_steps:
            damage_key, new_damage_key = jax.random.split(damage_key)
            modified_logits, modified_gate_mask = apply_knockout_to_circuit(
                new_damage_key,
                graph.nodes["logits"],
                layer_sizes,
                num_knockouts=knockout_per_damage_step,
                flat=True,
            )

            new_gate_mask = gate_mask * modified_gate_mask
            gate_mask = new_gate_mask if permanent_damage else gate_mask

            # Update graph nodes with new damage
            new_nodes = {
                **graph.nodes,
                "gate_knockout_mask": gate_mask,
                "logits": modified_logits,
            }
            graph = graph._replace(nodes=new_nodes)

            if verbose:
                print(f"Damage at step {step}")
                print(f"Damage in graph : {(graph.nodes['gate_knockout_mask'] == 0).sum()}")

        # Apply model and compute loss using unified step function
        graph, _, current_logits, aux_data = apply_model_and_compute_loss(
            model_fn,
            graph,
            logits_original_shapes,
            wires,
            x_data,
            y_data,
            loss_cfg,
            layer_sizes,
        )

        # Yield current state
        yield StepResult(
            step=step,
            logits=current_logits,
            graph=graph,
            **aux_data,
        )


def evaluate_model_stepwise(
    model: CircuitGatheredAttention | CircuitGNN | CircuitSelfAttention | PerceiverCircuitAttention,
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
    # Probabilistic damage parameters (for training)
    p_fault: float | None = None,
    faulty_value: float = -10.0,
    permanent_damage: bool = True,
    # Discrete damage parameters (for visualization)
    damage_key: jax.random.PRNGKey = jax.random.PRNGKey(42),
    damage_steps: list[int] | None = None,
    # Delayed probabilistic damage onset
    p_fault_onset_step: int = 0,
    # No-repair baseline
    compute_no_repair_baseline: bool = False,
) -> dict:
    """
    Evaluate model performance using the unified JIT-compiled scan.

    This function uses run_model_scan_with_loss internally for efficient JIT-compiled
    evaluation. For live demos with streaming results, use evaluate_model_stepwise_generator.

    Args:
        model: Trained model (CircuitGNN, CircuitSelfAttention, or PerceiverCircuitAttention)
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
        layer_sizes: List of (nodes, group_size) tuples for each layer
        use_tqdm: Show progress bar (only affects post-processing display)
        verbose: Print verbose output
        knockout_per_damage_step: Number of gates to knock out at each damage step
        damage_key: Random key for damage
        damage_steps: List of step indices at which to apply damage
        p_fault: Per-gate-per-step failure probability (None = disabled, 0.0 = no failures)
        faulty_value: Value to set for failed gate logits (large negative for zero output)
        permanent_damage: Whether to apply permanent damage to gates (True) or temporary damage (False)
        p_fault_onset_step: Step at which probabilistic damage starts (0 = from the start)
        compute_no_repair_baseline: If True, include no-repair baseline metrics in results

    Returns:
        Dictionary with metrics collected at each step
    """
    if loss_cfg is None:
        loss_cfg = LOSS_L4

    # Store original shapes for reconstruction
    logits_original_shapes = [logit.shape for logit in logits]

    # Build initial graph
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

    # Convert damage_steps to array if provided
    damage_steps_array = jp.asarray(damage_steps) if damage_steps is not None else None

    # Run unified scan
    final_graph, step_outputs = run_model_scan_with_loss(
        model=model,
        graph=graph,
        num_steps=n_message_steps,
        logits_original_shapes=logits_original_shapes,
        wires=wires,
        x_data=x_data,
        y_data=y_data,
        loss_cfg=loss_cfg,
        layer_sizes=tuple(layer_sizes) if layer_sizes else None,
        data_fraction=1.0,
        scan_key=damage_key,
        gradient_checkpointing=False,
        damage_steps=damage_steps_array,
        knockout_per_damage_step=knockout_per_damage_step,
        p_fault=p_fault,
        faulty_value=faulty_value,
        permanent_damage=permanent_damage,
        p_fault_onset_step=p_fault_onset_step,
        compute_no_repair_baseline=compute_no_repair_baseline,
    )

    # Extract metrics from scan outputs
    # step_outputs is (graphs, losses, logits, aux_data) for each step
    _graphs, losses, _logits_list, aux_data = step_outputs

    # Build results dictionary
    step_results = {
        "step": list(range(1, n_message_steps + 1)),  # Steps 1 to n_message_steps
        "loss": [float(loss) for loss in losses],
        "hard_loss": [float(hl) for hl in aux_data["hard_loss"]],
        "accuracy": [float(a) for a in aux_data["accuracy"]],
        "hard_accuracy": [float(a) for a in aux_data["hard_accuracy"]],
        "final_graph": final_graph,
    }

    # Include no-repair baseline metrics if computed
    if compute_no_repair_baseline and "no_repair_loss" in aux_data:
        step_results["no_repair_loss"] = [float(v) for v in aux_data["no_repair_loss"]]
        step_results["no_repair_hard_loss"] = [float(v) for v in aux_data["no_repair_hard_loss"]]
        step_results["no_repair_accuracy"] = [float(v) for v in aux_data["no_repair_accuracy"]]
        step_results["no_repair_hard_accuracy"] = [
            float(v) for v in aux_data["no_repair_hard_accuracy"]
        ]

    if verbose:
        print(f"Final loss: {step_results['loss'][-1]:.4f}")
        print(f"Final accuracy: {step_results['accuracy'][-1]:.4f}")
        print(f"Final hard accuracy: {step_results['hard_accuracy'][-1]:.4f}")

        # Print summary if tqdm was requested (for compatibility)
        print(f"Evaluation complete: {n_message_steps} steps")
        print(f"  Loss: {step_results['loss'][-1]:.4f}")
        print(f"  Accuracy: {step_results['accuracy'][-1]:.4f}")
        print(f"  Hard Acc: {step_results['hard_accuracy'][-1]:.4f}")

    return step_results


log = logging.getLogger(__name__)


def evaluate_model_stepwise_batched(
    model: CircuitGatheredAttention | CircuitGNN | CircuitSelfAttention | PerceiverCircuitAttention,
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
    # Discrete damage (for visualization with fixed steps)
    damage_steps: jp.ndarray | None = None,
    knockout_per_damage_step: int = 1,
    damage_key: jax.random.PRNGKey = jax.random.PRNGKey(42),
    # Probabilistic damage (for training-consistent evaluation)
    p_fault: float | None = None,
    faulty_value: float = -10.0,
    permanent_damage: bool = True,
    # Delayed probabilistic damage onset
    p_fault_onset_step: int = 0,
    # No-repair baseline
    compute_no_repair_baseline: bool = False,
    # Chunking and details
    chunk_size: int | None = None,
    return_first_circuit_details: bool = False,
    verbose: bool = False,
) -> tuple[jp.ndarray, dict]:
    """
    Vectorized evaluation of model performance on a batch of circuits.

    Uses the unified scan function (run_model_scan_with_loss) with vmap for
    efficient batch processing. Supports both discrete and probabilistic damage
    modes, plus automatic chunking for memory efficiency.

    **Damage Modes:**
    - Discrete (damage_steps): Fixed steps for clean visualization
    - Probabilistic (p_fault): Per-gate-per-step failure, matches training conditions

    Args:
        model: Trained model (CircuitGNN, CircuitSelfAttention, or PerceiverCircuitAttention)
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
        damage_steps: Array of step indices at which to apply discrete damage (for visualization)
        knockout_per_damage_step: Number of gates to knock out at each discrete damage step
        damage_key: Random key for damage
        p_fault: Per-gate-per-step failure probability (for training-consistent evaluation)
        faulty_value: Value to set for failed gate logits
        chunk_size: If provided, process circuits in chunks of this size for memory efficiency.
                   If None, process all circuits at once.
        return_first_circuit_details: If True, include detailed StepResult list for first circuit
                                     (useful for visualization without re-running evaluation)

    Returns:
        Final graphs after evaluation.
        Dictionary with averaged metrics collected at each step.
        If return_first_circuit_details=True, also includes 'first_circuit_results' key
        with list of StepResult objects for visualization.
    """
    batch_size = batch_logits[0].shape[0]

    # Handle chunking for large batches
    if chunk_size is not None and batch_size > chunk_size:
        log.info(f"Using chunked evaluation (chunks of {chunk_size})")
        chunks = []
        for i in range(0, batch_size, chunk_size):
            end = min(i + chunk_size, batch_size)
            chunk_final_graphs, chunk_step_metrics = evaluate_model_stepwise_batched(
                model=model,
                batch_wires=[w[i:end] for w in batch_wires],
                batch_logits=[lg[i:end] for lg in batch_logits],
                x_data=x_data,
                y_data=y_data,
                input_n=input_n,
                arity=arity,
                circuit_hidden_dim=circuit_hidden_dim,
                n_message_steps=n_message_steps,
                loss_cfg=loss_cfg,
                bidirectional_edges=bidirectional_edges,
                layer_sizes=layer_sizes,
                damage_steps=damage_steps,
                knockout_per_damage_step=knockout_per_damage_step,
                damage_key=damage_key,
                p_fault=p_fault,
                faulty_value=faulty_value,
                permanent_damage=permanent_damage,
                p_fault_onset_step=p_fault_onset_step,
                compute_no_repair_baseline=compute_no_repair_baseline,
                chunk_size=None,  # Don't recurse further
                return_first_circuit_details=return_first_circuit_details and (i == 0),
            )
            chunks.append((end - i, chunk_final_graphs, chunk_step_metrics))
        # Weighted average across chunks
        total = sum(w for w, *_ in chunks)
        final_graphs = jax.tree.map(lambda x: jp.concat(x, axis=0), chunks[0][0])
        step_metrics = {"step": chunks[0][2]["step"]}
        avg_keys = ["loss", "hard_loss", "accuracy", "hard_accuracy"]
        # Include no-repair metrics if present
        if compute_no_repair_baseline:
            avg_keys += [
                "no_repair_loss",
                "no_repair_hard_loss",
                "no_repair_accuracy",
                "no_repair_hard_accuracy",
            ]
        for k in avg_keys:
            if k in chunks[0][2]:
                step_metrics[k] = [
                    sum(w * r[k][s] for w, r in chunks) / total for s in range(n_message_steps)
                ]
        if "first_circuit_results" in chunks[0][1]:
            step_metrics["first_circuit_results"] = chunks[0][1]["first_circuit_results"]
        return final_graphs, step_metrics

    if loss_cfg is None:
        loss_cfg = LOSS_L4

    # Build initial graphs (vectorized)
    vmap_build_graph = jax.vmap(
        lambda logits, wires: build_graph(
            logits,
            wires,
            input_n,
            arity,
            circuit_hidden_dim,
            loss_value=0.0,
            bidirectional_edges=bidirectional_edges,
        )
    )
    batch_graphs = vmap_build_graph(batch_logits, batch_wires)

    # Split damage keys for each batch element (use dummy key if no damage)
    batch_size = batch_logits[0].shape[0]
    if permanent_damage == "random" and damage_key is not None:
        damage_key, permanent_damage_key = jax.random.split(damage_key)
        permanent_damage = jax.random.choice(
            permanent_damage_key, jp.array([True, False]), (batch_size,)
        )
    elif permanent_damage == "random" and damage_key is None:
        permanent_damage = jax.random.choice(
            jax.random.PRNGKey(0), jp.array([True, False]), (batch_size,)
        )
    else:
        permanent_damage = jax.numpy.full(batch_size, permanent_damage)

    if damage_key is None:
        damage_key = jax.random.PRNGKey(0)  # Dummy key, won't be used if no damage_steps
    damage_keys = jax.random.split(damage_key, batch_size)

    # Run unified scan for each circuit in batch
    def run_single_scan(graph, wires, logits, scan_key, permanent_damage):
        return run_model_scan_with_loss(
            model=model,
            graph=graph,
            num_steps=n_message_steps,
            logits_original_shapes=[logit.shape for logit in logits],
            wires=wires,
            x_data=x_data,
            y_data=y_data,
            loss_cfg=loss_cfg,
            layer_sizes=layer_sizes,
            data_fraction=1.0,
            scan_key=scan_key,
            gradient_checkpointing=False,
            # Discrete damage (for visualization)
            damage_steps=damage_steps,
            knockout_per_damage_step=knockout_per_damage_step,
            # Probabilistic damage (for training-consistent evaluation)
            p_fault=p_fault,
            faulty_value=faulty_value,
            permanent_damage=permanent_damage,
            # Delayed onset and no-repair baseline
            p_fault_onset_step=p_fault_onset_step,
            compute_no_repair_baseline=compute_no_repair_baseline,
        )

    # Vmap over batch
    final_graphs, batch_step_outputs = nnx.vmap(run_single_scan)(
        batch_graphs, batch_wires, batch_logits, damage_keys, permanent_damage
    )

    # Extract and average metrics
    all_batch_graphs, losses, _, aux_data = batch_step_outputs

    # Average across batch dimension
    step_metrics = {
        "step": list(range(1, n_message_steps + 1)),
        "loss": [float(jp.mean(losses[:, i])) for i in range(n_message_steps)],
        "hard_loss": [float(jp.mean(aux_data["hard_loss"][:, i])) for i in range(n_message_steps)],
        "accuracy": [float(jp.mean(aux_data["accuracy"][:, i])) for i in range(n_message_steps)],
        "hard_accuracy": [
            float(jp.mean(aux_data["hard_accuracy"][:, i])) for i in range(n_message_steps)
        ],
        # Vmapped graphs: size [batch_size, n_steps, ...graph_shape...]
        "graphs": all_batch_graphs,
        "all_metrics": {
            "loss": jp.stack(losses),
            "hard_loss": jp.stack(aux_data["hard_loss"]),
            "accuracy": jp.stack(aux_data["accuracy"]),
            "hard_accuracy": jp.stack(aux_data["hard_accuracy"]),
        },
    }

    # Include no-repair baseline metrics if computed
    if compute_no_repair_baseline and "no_repair_loss" in aux_data:
        step_metrics["no_repair_loss"] = [
            float(jp.mean(aux_data["no_repair_loss"][:, i])) for i in range(n_message_steps)
        ]
        step_metrics["no_repair_hard_loss"] = [
            float(jp.mean(aux_data["no_repair_hard_loss"][:, i])) for i in range(n_message_steps)
        ]
        step_metrics["no_repair_accuracy"] = [
            float(jp.mean(aux_data["no_repair_accuracy"][:, i])) for i in range(n_message_steps)
        ]
        step_metrics["no_repair_hard_accuracy"] = [
            float(jp.mean(aux_data["no_repair_hard_accuracy"][:, i]))
            for i in range(n_message_steps)
        ]

    # Optionally extract detailed results for first circuit (for visualization)
    if return_first_circuit_details:
        step_metrics["first_circuit_results"] = _extract_single_circuit_step_results(
            batch_step_outputs, circuit_idx=0, n_message_steps=n_message_steps
        )

    if verbose:
        # Print summary if tqdm was requested (for compatibility)
        print(f"Evaluation complete: {n_message_steps} steps")
        print(f"  Loss: {step_metrics['loss'][-1]:.4f}")
        print(f"  Accuracy: {step_metrics['accuracy'][-1]:.4f}")
        print(f"  Hard Acc: {step_metrics['hard_accuracy'][-1]:.4f}")

    return final_graphs, step_metrics


def get_fraction_damaged_gates(batch_graphs_per_step: jp.ndarray) -> jp.ndarray:
    """
    Get the fraction of damaged gates for a given batch of graphs per step.
    Assumes that graphs are vmapped as [batch_size, n_steps, ...graph_shape...]
    Returns:
    - [n_steps] array of fraction of damaged gates per step
    """

    damages_per_steps = (batch_graphs_per_step.nodes["gate_knockout_mask"] == 0).mean(0).sum(-1)
    single_graph_layers = batch_graphs_per_step.nodes["layer"][0, 0]
    n_eligible_gates = (
        (single_graph_layers != 0) * (single_graph_layers != single_graph_layers.max())
    ).sum()

    return damages_per_steps / n_eligible_gates


def _extract_single_circuit_step_results(
    batch_step_outputs,
    circuit_idx: int,
    n_message_steps: int,
) -> list[StepResult]:
    """
    Extract StepResult objects for a single circuit from batched evaluation outputs.

    Args:
        batch_step_outputs: Tuple of (graphs, losses, logits, aux_data) from batched scan
        circuit_idx: Index of circuit to extract
        n_message_steps: Number of steps

    Returns:
        List of StepResult objects for the specified circuit
    """
    graphs, losses, logits_all, aux_data = batch_step_outputs

    # Bundle all outputs (flatten aux_data) for single tree_map extraction
    all_outputs = {"loss": losses, **aux_data, "logits": logits_all, "graph": graphs}

    def make_step_result(s):
        e = jax.tree.map(lambda x, step=s: x[circuit_idx, step], all_outputs)
        return StepResult(
            step=s + 1,
            loss=float(e["loss"]),
            hard_loss=float(e["hard_loss"]),
            accuracy=float(e["accuracy"]),
            hard_accuracy=float(e["hard_accuracy"]),
            predictions=e["predictions"],
            hard_predictions=e["hard_predictions"],
            residuals=e["residuals"],
            hard_residuals=e["hard_residuals"],
            logits=e["logits"],
            graph=e["graph"],
        )

    return [make_step_result(s) for s in range(n_message_steps)]
