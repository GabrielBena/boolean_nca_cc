"""
Evaluation functions for GNN-based boolean circuit optimization.

This module provides functions for evaluating the performance of GNN
models on optimizing boolean circuits.
"""

import logging
from collections.abc import Generator
from types import SimpleNamespace
from typing import NamedTuple

import jax
import jax.numpy as jp
import optax
from flax import nnx

from boolean_nca_cc.circuits.model import gen_wires, run_circuit
from boolean_nca_cc.circuits.train import (
    LOSS_L4,
    LossConfig,
    compute_loss_from_predictions,
    loss_f,
)
from boolean_nca_cc.models import (
    CircuitGatheredAttention,
    CircuitGNN,
    CircuitSelfAttention,
    PerceiverCircuitAttention,
)
from boolean_nca_cc.training.pool.perturbation import shuffle_wires as _shuffle_wires_partial
from boolean_nca_cc.training.pool.structural_perturbation import (
    apply_knockout_to_circuit,
    apply_probabilistic_gate_failure,
    apply_probabilistic_wire_shuffle,
    create_eligible_gate_mask,
    create_flat_knockout_pattern,
    flat_to_layered_mask,
)
from boolean_nca_cc.utils import (
    extract_logits_from_graph,
    update_output_node_from_residuals,
)
from boolean_nca_cc.utils.configured_graph_builder import configured_build_graph as build_graph
from boolean_nca_cc.utils.graph_builder import compute_circuit_edges, refresh_wire_dependent_pe


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


def _layer_node_indices(
    layer_indices,
    target_layer: int,
    size: int,
) -> jp.ndarray:
    """Static [size] int32 array of node positions where ``layer == target_layer``.

    Must be JIT-safe: this is called from ``_prepare_model_fn`` which itself can
    be traced (when invoked from inside ``nnx.vmap`` / ``nnx.jit`` during the
    training loop). We use ``jp.where(mask, size=size, fill_value=0)`` so the
    output shape is a Python-static int — downstream gather/scatter
    (``gate_latents[:, idx, :]``) requires that.

    ``size`` must be supplied by the caller (usually pulled from
    ``model.input_layer_size`` / ``model.output_layer_size``). The layer-membership
    pattern is wire-independent — it depends only on the circuit's layer
    structure, which is fixed for the lifetime of a given training run.
    """
    mask = layer_indices == target_layer
    idx = jp.where(mask, size=size, fill_value=0)[0]
    return idx.astype(jp.int32)


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
        # Perceiver: precompute output gates plus self-attn topology
        # (mask for dense, neighbor indices for gathered).
        layer_indices = graph.nodes["layer"]
        max_layer = jp.max(layer_indices)

        # Static layer-index arrays for the cross-attn slicing optimization.
        # Asymmetric semantics that mirror the model's restriction flags:
        #   input cross-attn  → input pins (layer == 0); K_in  = input_layer_size
        #   output cross-attn → last gate layer (layer == max); K_out = output_layer_size
        # Attending only on K queries instead of all N nodes saves a factor of
        # N/K in cross-attention FLOPs. See PerceiverCircuitAttention.__call__
        # for the slicing branch. The static K comes from
        # ``model.input_layer_size`` / ``output_layer_size`` set at construction
        # time (injected by train.py from layer_sizes); if those aren't set we
        # skip slicing and fall through to the dense+gate path.
        input_layer_idx = None
        output_layer_idx = None
        if model.restrict_input_cross_attn_to_input_layer and model.input_layer_size is not None:
            input_layer_idx = _layer_node_indices(
                layer_indices,
                target_layer=0,
                size=model.input_layer_size,
            )
        if (
            model.restrict_output_cross_attn_to_last_gate_layer
            and model.output_layer_size is not None
        ):
            # max_layer here is a tracer in JIT; we use the model's stored
            # size and rely on the layer-index pattern being identical across
            # all circuits (it is — layer structure is wire-independent).
            output_layer_idx = _layer_node_indices(
                layer_indices,
                target_layer=max_layer,
                size=model.output_layer_size,
            )

        # output_gate kept as a fallback (used only if slicing indices are None).
        if model.restrict_input_cross_attn_to_input_layer:
            input_output_gate = model._create_output_gate(layer_indices, allowed_layer=0)
        if model.restrict_output_cross_attn_to_last_gate_layer:
            output_output_gate = model._create_output_gate(layer_indices, allowed_layer=max_layer)

        if model.self_attn_kind == "gathered":
            from boolean_nca_cc.models.attention.base import (
                build_neighbor_indices,
                validate_gathered_topology,
            )

            validate_gathered_topology(
                graph.senders,
                graph.receivers,
                model.n_node,
                model.max_neighbors,
                model.use_attention_mask,
                strict=True,
                layer_indices=layer_indices,
            )
            neighbor_indices, neighbor_mask = build_neighbor_indices(
                graph.senders,
                graph.receivers,
                model.n_node,
                model.max_neighbors,
                model.use_attention_mask,
            )

            def base_fn(g):
                return model(
                    g,
                    neighbor_indices=neighbor_indices,
                    neighbor_mask=neighbor_mask,
                    input_output_gate=input_output_gate,
                    output_output_gate=output_output_gate,
                    input_layer_idx=input_layer_idx,
                    output_layer_idx=output_layer_idx,
                )

        else:
            attention_mask = model._create_attention_mask(
                graph.senders, graph.receivers, model.n_node
            )

            def base_fn(g):
                return model(
                    g,
                    attention_mask=attention_mask,
                    input_output_gate=input_output_gate,
                    output_output_gate=output_output_gate,
                    input_layer_idx=input_layer_idx,
                    output_layer_idx=output_layer_idx,
                )

    elif isinstance(model, CircuitSelfAttention):
        # Self-attention: precompute attention mask
        attention_mask = model._create_attention_mask(graph.senders, graph.receivers, model.n_node)

        def base_fn(g):
            return model(g, attention_mask=attention_mask)

    elif isinstance(model, CircuitGatheredAttention):
        # Gathered attention: precompute neighbor indices from graph topology
        from boolean_nca_cc.models.attention.base import (
            build_neighbor_indices,
            validate_gathered_topology,
        )

        n_node = graph.nodes["layer"].shape[0]
        validate_gathered_topology(
            graph.senders,
            graph.receivers,
            n_node,
            model.max_neighbors,
            model.use_attention_mask,
            strict=True,
            layer_indices=graph.nodes["layer"],
        )
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


def _compute_topology_cache(
    model,
    senders: jp.ndarray,
    receivers: jp.ndarray,
    n_node: int,
) -> dict | None:
    """Compute the wire-dependent precomputed topology for ``model``.

    Returns the bits of state that ``_prepare_model_fn`` would otherwise
    capture as closure variables, so the scan can refresh them when wires
    change (via :func:`compute_circuit_edges`). Wire-independent bits like
    Perceiver's ``input_output_gate`` are NOT in the cache — they stay
    frozen for the whole scan.

    Returns:
        - ``CircuitGatheredAttention`` →
          ``{"neighbor_indices": ..., "neighbor_mask": ...}``
        - ``PerceiverCircuitAttention`` with ``self_attn_kind="gathered"`` →
          ``{"neighbor_indices": ..., "neighbor_mask": ...}``
        - ``CircuitSelfAttention`` / dense ``PerceiverCircuitAttention`` →
          ``{"attention_mask": ...}``
        - ``CircuitGNN`` → ``None`` (model reads ``graph.senders/receivers`` directly)
    """
    if isinstance(model, CircuitGatheredAttention) or (
        isinstance(model, PerceiverCircuitAttention) and model.self_attn_kind == "gathered"
    ):
        from boolean_nca_cc.models.attention.base import build_neighbor_indices

        neighbor_indices, neighbor_mask = build_neighbor_indices(
            senders,
            receivers,
            n_node,
            model.max_neighbors,
            model.use_attention_mask,
        )
        return {"neighbor_indices": neighbor_indices, "neighbor_mask": neighbor_mask}
    if isinstance(model, (CircuitSelfAttention, PerceiverCircuitAttention)):
        return {"attention_mask": model._create_attention_mask(senders, receivers, n_node)}
    return None


def _apply_model_with_cache(
    model,
    graph,
    cache: dict | None,
    perceiver_input_gate=None,
    perceiver_output_gate=None,
    perceiver_input_layer_idx=None,
    perceiver_output_layer_idx=None,
):
    """Run ``model`` using an externally-supplied topology cache.

    Counterpart of the closure ``base_fn`` produced by ``_prepare_model_fn``,
    but takes the cache as an explicit argument so the scan can refresh it
    when wires change. The Perceiver-specific frozen inputs (output gates and
    layer-index arrays) are not in the cache because they are wire-independent —
    they depend only on the layer structure, which is fixed across wire shuffles.
    """
    if isinstance(model, CircuitGatheredAttention):
        return model(
            graph,
            neighbor_indices=cache["neighbor_indices"],
            neighbor_mask=cache["neighbor_mask"],
        )
    if isinstance(model, CircuitSelfAttention):
        return model(graph, attention_mask=cache["attention_mask"])
    if isinstance(model, PerceiverCircuitAttention):
        if model.self_attn_kind == "gathered":
            return model(
                graph,
                neighbor_indices=cache["neighbor_indices"],
                neighbor_mask=cache["neighbor_mask"],
                input_output_gate=perceiver_input_gate,
                output_output_gate=perceiver_output_gate,
                input_layer_idx=perceiver_input_layer_idx,
                output_layer_idx=perceiver_output_layer_idx,
            )
        return model(
            graph,
            attention_mask=cache["attention_mask"],
            input_output_gate=perceiver_input_gate,
            output_output_gate=perceiver_output_gate,
            input_layer_idx=perceiver_input_layer_idx,
            output_layer_idx=perceiver_output_layer_idx,
        )
    return model(graph)


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
    permanent_damage: float = 1.0,
    # Delayed probabilistic damage onset
    p_fault_onset_step: int = 0,
    # Bursty background damage (doubly-stochastic p_fault — "solar events"):
    # quiet base rate ``p_fault`` punctuated by windows of ``p_fault_burst``.
    # A window opens per tick with prob ``burst_start_rate`` and lasts
    # ``burst_length`` ticks. None/0.0 = flat p_fault (legacy behavior).
    p_fault_burst: float | None = None,
    burst_start_rate: float = 0.0,
    burst_length: int = 4,
    # No-repair baseline: compute loss right after damage, before NCA runs
    compute_no_repair_baseline: bool = False,
    # Wire shuffling parameters (discrete: explicit step indices)
    wire_shuffle_steps: list[int] | None = None,
    wire_shuffle_key: jax.random.PRNGKey = jax.random.PRNGKey(42),
    wire_shuffle_fraction: float = 1.0,
    # Wire shuffling parameters (probabilistic: per-step circuit-wide coin flip)
    p_shuffle: float | None = None,
    shuffle_fraction: float = 1.0,
    p_shuffle_onset_step: int = 0,
    # Circuit topology (only consulted when a wire-shuffle path is active,
    # to recompute senders/receivers/topology-cache matching the initial graph)
    arity: int | None = None,
    bidirectional_edges: bool = True,
    neighboring_connections: bool = False,
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
        permanent_damage: Per-gate probability that damage is permanent (float in [0, 1]).
            1.0 = always permanent, 0.0 = always temporary, 0.5 = independent coin flip
            per gate per timestep.
        p_fault_onset_step: Step at which probabilistic damage starts (0 = from the start).
            Use to let the circuit converge first, then introduce faults for repair testing.
        compute_no_repair_baseline: If True, compute loss/accuracy right after damage is
            applied but before the NCA model runs. This measures the raw damage impact without
            repair, enabling comparison of "damaged without NCA" vs "damaged with NCA recovery".
            Adds keys ``no_repair_loss``, ``no_repair_hard_loss``, ``no_repair_accuracy``,
            ``no_repair_hard_accuracy`` to the aux_data in step outputs.
        wire_shuffle_steps: List of step indices at which to shuffle wires (None = disabled)
        wire_shuffle_key: Random key for wire shuffling
        wire_shuffle_fraction: Fraction of wires to shuffle (0.0-1.0).
            1.0 (default) = regenerate all wires from scratch.
            < 1.0 = randomly reassign that fraction of connections per layer.
        p_shuffle: Per-step circuit-wide shuffle probability (None = disabled).
            Probabilistic counterpart to ``wire_shuffle_steps``: each step a single
            coin flip decides whether to perturb the wires (training-only path).
        shuffle_fraction: Static flavor knob for the probabilistic path
            (same semantics as ``wire_shuffle_fraction`` but applied per-step
            when the coin flip triggers).
        p_shuffle_onset_step: Step at which probabilistic shuffles start (0 = from the start).

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
    burst_enabled = (
        p_fault_burst is not None and p_fault_burst > 0.0 and burst_start_rate > 0.0
    )
    prob_damage_enabled = (p_fault is not None and p_fault > 0.0) or burst_enabled
    if prob_damage_enabled:
        eligible_mask = create_eligible_gate_mask(layer_sizes)
        # Pre-split keys for all steps (for reproducibility and JIT compatibility)
        prob_damage_keys = jax.random.split(prob_key, num_steps)
        base_p = float(p_fault) if p_fault is not None else 0.0
        if burst_enabled:
            # Realize this rollout's burst windows: Bernoulli starts per tick,
            # each start elevates p for ``burst_length`` ticks ("solar event").
            # Drawn from prob_key's stream → same sharing semantics as the
            # per-step damage draws themselves.
            sched_key = jax.random.fold_in(prob_key, 0xB5)
            starts = jax.random.bernoulli(sched_key, burst_start_rate, (num_steps,))
            tick = jp.arange(num_steps)
            in_window = (tick[:, None] >= tick[None, :]) & (
                tick[:, None] < tick[None, :] + burst_length
            )
            burst_active = jp.any(starts[None, :] & in_window, axis=1)
            p_fault_schedule = jp.where(burst_active, p_fault_burst, base_p)
        else:
            p_fault_schedule = jp.full((num_steps,), base_p)
    else:
        eligible_mask = None
        prob_damage_keys = None
        p_fault_schedule = None

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
            # Split key: one for knockout, one for permanent_damage coin flip
            knockout_key, perm_key = jax.random.split(discrete_damage_keys[damage_idx])
            # Apply knockout
            modified_logits, modified_gate_mask = apply_knockout_to_circuit(
                knockout_key,
                graph.nodes["logits"],
                layer_sizes,
                num_knockouts=knockout_per_damage_step,
                flat=True,
            )
            new_gate_mask = gate_mask * modified_gate_mask
            # Per-gate coin flip: each damaged gate independently decides if permanent
            perm_mask = jax.random.uniform(perm_key, shape=gate_mask.shape) < permanent_damage
            new_gate_mask = jp.where(perm_mask, new_gate_mask, gate_mask)

            new_nodes = {
                **graph.nodes,
                "logits": modified_logits,
                "gate_knockout_mask": new_gate_mask,
            }
            return graph._replace(nodes=new_nodes), new_gate_mask

        def no_damage(_):
            return graph, gate_mask

        return jax.lax.cond(is_damage_step, do_damage, no_damage, None)

    # === Wire shuffling setup ===
    wire_shuffle_enabled = wire_shuffle_steps is not None and len(wire_shuffle_steps) > 0
    if wire_shuffle_enabled:
        wire_shuffle_steps_array = jp.asarray(wire_shuffle_steps, dtype=jp.int32)
        arity = wires[0].shape[0]  # Infer arity from existing wires
        wire_shuffle_keys = jax.random.split(wire_shuffle_key, len(wire_shuffle_steps))
    else:
        wire_shuffle_steps_array = None
        wire_shuffle_keys = None

    def shuffle_wires_if_needed(current_wires, step_idx):
        """Shuffle circuit wires at specific steps using vectorized conditional.

        Follows the same pattern as ``apply_discrete_damage_if_needed``:
        pre-split keys + ``jax.lax.cond`` for JIT-safe conditional execution.

        When ``wire_shuffle_fraction < 1.0``, only that fraction of
        connections are randomly reassigned (partial shuffle).  At 1.0
        all wires are regenerated from scratch.
        """
        if not wire_shuffle_enabled:
            return current_wires

        # Check if current step is a shuffle step (1-indexed, same as damage_steps)
        is_shuffle_step = jp.any(wire_shuffle_steps_array == step_idx)

        # Find the matching key index via searchsorted
        shuffle_idx = jp.searchsorted(wire_shuffle_steps_array, step_idx)
        shuffle_idx = jp.clip(shuffle_idx, 0, len(wire_shuffle_keys) - 1)

        def do_shuffle(_):
            key = wire_shuffle_keys[shuffle_idx]
            if wire_shuffle_fraction < 1.0:
                # Partial shuffle: randomly reassign a fraction of connections
                return _shuffle_wires_partial(
                    key, current_wires, layer_sizes, fraction=wire_shuffle_fraction
                )
            # Full shuffle: regenerate all wires from scratch
            new_wires = []
            in_n = layer_sizes[0][0]
            keys = jax.random.split(key, len(layer_sizes) - 1)
            for i, (out_n, group_size) in enumerate(layer_sizes[1:]):
                w = gen_wires(keys[i], in_n, out_n, arity, group_size)
                in_n = out_n
                new_wires.append(w)
            return new_wires

        def no_shuffle(_):
            return current_wires

        return jax.lax.cond(is_shuffle_step, do_shuffle, no_shuffle, None)

    # === Probabilistic wire shuffle setup ===
    # Circuit-wide per-step coin flip on p_shuffle. Mirrors the dual mode of damage
    # (discrete wire_shuffle_steps + probabilistic p_shuffle).
    prob_shuffle_enabled = p_shuffle is not None and p_shuffle > 0.0
    if prob_shuffle_enabled:
        prob_shuffle_arity = wires[0].shape[0]
        prob_shuffle_key, _ = jax.random.split(damage_key)  # separate stream
        prob_shuffle_keys = jax.random.split(prob_shuffle_key, num_steps)
    else:
        prob_shuffle_keys = None
        prob_shuffle_arity = None

    def apply_probabilistic_shuffle(current_wires, step_idx):
        """Apply probabilistic per-step wire shuffle (circuit-wide coin flip).

        Respects ``p_shuffle_onset_step``: shuffles only start once
        ``step_idx >= p_shuffle_onset_step``. Gating is done by zeroing the
        effective probability before the coin flip, so both branches stay
        JIT-safe.
        """
        if not prob_shuffle_enabled:
            return current_wires

        past_onset = step_idx >= p_shuffle_onset_step
        effective_p = jp.where(past_onset, p_shuffle, 0.0)
        return apply_probabilistic_wire_shuffle(
            prob_shuffle_keys[step_idx],
            current_wires,
            layer_sizes,
            prob_shuffle_arity,
            effective_p,
            shuffle_fraction=shuffle_fraction,
        )

    # === Topology refresh setup (only when a wire-shuffle path is active) ===
    # The model precomputes attention masks / neighbor indices from the *initial*
    # graph.senders/receivers (see `_prepare_model_fn`). When wires change mid-scan
    # the precomputed cache and the graph's edges go stale, so the model attends
    # over the OLD topology while the loss evaluates the NEW one — the bug that
    # makes scan-time shuffles collapse to chance accuracy. Below we refresh both
    # on shuffle events via jax.lax.cond, so the only cost when no shuffle fires
    # is the cond check itself.
    any_shuffle_enabled = wire_shuffle_enabled or prob_shuffle_enabled
    if any_shuffle_enabled:
        _refresh_arity = arity if arity is not None else int(wires[0].shape[0])
        _n_node_static = int(graph.nodes["layer"].shape[0])

        # Static (wire-independent) Perceiver state — computed once, frozen for scan.
        # Both the legacy output gates and the new static layer-index arrays depend
        # only on ``nodes["layer"]`` (the layer membership pattern), which never
        # changes under wire shuffles. So they live outside the topology cache.
        _perceiver_input_gate = None
        _perceiver_output_gate = None
        _perceiver_input_layer_idx = None
        _perceiver_output_layer_idx = None
        if isinstance(model, PerceiverCircuitAttention):
            _layer_indices = graph.nodes["layer"]
            _max_layer = jp.max(_layer_indices)
            if model.restrict_input_cross_attn_to_input_layer:
                _perceiver_input_gate = model._create_output_gate(_layer_indices, allowed_layer=0)
                if model.input_layer_size is not None:
                    _perceiver_input_layer_idx = _layer_node_indices(
                        _layer_indices,
                        target_layer=0,
                        size=model.input_layer_size,
                    )
            if model.restrict_output_cross_attn_to_last_gate_layer:
                _perceiver_output_gate = model._create_output_gate(
                    _layer_indices, allowed_layer=_max_layer
                )
                if model.output_layer_size is not None:
                    _perceiver_output_layer_idx = _layer_node_indices(
                        _layer_indices,
                        target_layer=_max_layer,
                        size=model.output_layer_size,
                    )

        initial_topology_cache = _compute_topology_cache(
            model, graph.senders, graph.receivers, _n_node_static
        )

        def refresh_topology(current_graph, current_wires):
            """Recompute senders/receivers, wire-dependent node PEs, and topology cache."""
            new_senders, new_receivers = compute_circuit_edges(
                current_wires,
                layer_sizes,
                _refresh_arity,
                bidirectional_edges=bidirectional_edges,
                neighboring_connections=neighboring_connections,
            )
            # Refresh dist_pe / rwse if they were built into the graph. ``refresh_wire_dependent_pe``
            # is a no-op for keys that aren't present, so this stays correct when those PEs are
            # disabled in the build config.
            new_nodes = refresh_wire_dependent_pe(
                current_graph.nodes,
                current_wires,
                new_senders,
                new_receivers,
                layer_sizes,
                _refresh_arity,
            )
            new_graph = current_graph._replace(
                nodes=new_nodes, senders=new_senders, receivers=new_receivers
            )
            new_cache = _compute_topology_cache(model, new_senders, new_receivers, _n_node_static)
            return new_graph, new_cache

        def _apply_model_dynamic(g, cache):
            return _apply_model_with_cache(
                model,
                g,
                cache,
                _perceiver_input_gate,
                _perceiver_output_gate,
                _perceiver_input_layer_idx,
                _perceiver_output_layer_idx,
            )

        apply_model_remat = (
            nnx.remat(_apply_model_dynamic) if gradient_checkpointing else _apply_model_dynamic
        )

        def detect_wire_change(old_wires, new_wires):
            """Cheap per-element comparison summed across layers. ``True`` if any wire changed."""
            return jp.any(
                jp.stack([jp.any(o != n) for o, n in zip(old_wires, new_wires, strict=True)])
            )
    else:
        initial_topology_cache = None
        refresh_topology = None
        apply_model_remat = None
        detect_wire_change = None

    def apply_probabilistic_damage(graph, step_idx, gate_mask):
        """Apply probabilistic damage: each gate has p_fault chance of failure.

        Respects ``p_fault_onset_step``: damage is only applied when
        ``step_idx >= p_fault_onset_step``, allowing the circuit to converge
        to a functional state before faults are introduced.
        """
        if not prob_damage_enabled:
            return graph, gate_mask

        # Split key: one for gate failure, one for permanent_damage coin flip
        fault_key, perm_key = jax.random.split(prob_damage_keys[step_idx])

        # Apply probabilistic failure (always compute for JIT consistency)
        new_logits, new_mask = apply_probabilistic_gate_failure(
            fault_key,
            graph.nodes["logits"],
            gate_mask,
            eligible_mask,
            # Per-tick rate from the (possibly bursty) schedule; reduces to the
            # flat p_fault when bursts are disabled.
            p_fault_schedule[step_idx],
            faulty_value,
        )

        # Per-gate coin flip: each damaged gate independently decides if permanent
        perm_mask = jax.random.uniform(perm_key, shape=gate_mask.shape) < permanent_damage
        new_gate_mask = jp.where(perm_mask, new_mask, gate_mask)

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
            if any_shuffle_enabled:
                (
                    current_graph,
                    current_gate_mask,
                    nr_graph,
                    nr_gate_mask,
                    damage_started,
                    current_wires,
                    current_cache,
                ) = carry
            else:
                (
                    current_graph,
                    current_gate_mask,
                    nr_graph,
                    nr_gate_mask,
                    damage_started,
                    current_wires,
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

            # --- 1b. Shuffle wires if needed (1-indexed, same as damage_steps)
            old_wires = current_wires
            current_wires = shuffle_wires_if_needed(current_wires, step_idx + 1)
            # --- 1c. Probabilistic per-step wire shuffle (training path) ---
            #         Uses raw step_idx (0-indexed, same as apply_probabilistic_damage)
            current_wires = apply_probabilistic_shuffle(current_wires, step_idx)

            # --- 1d. Refresh graph topology + cache if wires changed -------
            if any_shuffle_enabled:
                did_shuffle = detect_wire_change(old_wires, current_wires)
                current_graph, current_cache = jax.lax.cond(
                    did_shuffle,
                    lambda _: refresh_topology(current_graph, current_wires),
                    lambda _: (current_graph, current_cache),
                    None,
                )
                # nr graph shares wires with main → keep its edges in sync too
                nr_graph = nr_graph._replace(
                    senders=current_graph.senders, receivers=current_graph.receivers
                )

            # --- 2. Detect whether damage has now started ------------------
            # Any gate that was active in the initial mask and is now
            # knocked out means real damage has occurred.
            damage_started = damage_started | jp.any(nr_gate_mask < initial_gate_mask)

            # --- 3. Compute no-repair baseline loss (no NCA model) ---------
            nr_logits = extract_logits_from_graph(nr_graph, logits_original_shapes)
            _, no_repair_aux = get_loss_from_wires_logits(
                logits=nr_logits,
                wires=current_wires,
                x=x_batch,
                y_target=y_batch,
                loss_cfg=loss_cfg,
            )

            # --- 4. Apply NCA model ONLY to the main graph -----------------
            if any_shuffle_enabled:
                model_updated_graph = apply_model_remat(current_graph, current_cache)
                updated_graph, loss, current_logits, aux = get_loss_and_update_graph(
                    model_updated_graph,
                    logits_original_shapes,
                    current_wires,
                    x_batch,
                    y_batch,
                    loss_cfg,
                    layer_sizes,
                )
            else:
                updated_graph, loss, current_logits, aux = apply_model_and_compute_loss(
                    model_fn,
                    current_graph,
                    logits_original_shapes,
                    current_wires,
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

            if any_shuffle_enabled:
                new_carry = (
                    updated_graph,
                    current_gate_mask,
                    nr_graph,
                    nr_gate_mask,
                    damage_started,
                    current_wires,
                    current_cache,
                )
            else:
                new_carry = (
                    updated_graph,
                    current_gate_mask,
                    nr_graph,
                    nr_gate_mask,
                    damage_started,
                    current_wires,
                )
            return new_carry, (updated_graph, loss, current_logits, aux)

        # Initial carry: nr graph starts as a copy of the main graph
        if any_shuffle_enabled:
            init_carry = (
                graph,
                initial_gate_mask,
                graph,
                initial_gate_mask,
                jp.bool_(False),
                wires,
                initial_topology_cache,
            )
        else:
            init_carry = (
                graph,
                initial_gate_mask,
                graph,
                initial_gate_mask,
                jp.bool_(False),
                wires,
            )
        final_carry, step_outputs = jax.lax.scan(
            scan_step,
            init_carry,
            xs=jp.arange(num_steps),
            length=num_steps,
        )
        final_graph = final_carry[0]

    else:
        # ------------------------------------------------------------------
        # Standard scan (no baseline tracking)
        # ------------------------------------------------------------------
        def scan_step(carry, step_idx):
            if any_shuffle_enabled:
                current_graph, current_gate_mask, current_wires, current_cache = carry
            else:
                current_graph, current_gate_mask, current_wires = carry

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

            # Shuffle wires if needed (1-indexed, same convention as damage_steps)
            old_wires = current_wires
            current_wires = shuffle_wires_if_needed(current_wires, step_idx + 1)
            # Probabilistic per-step wire shuffle (training path)
            # Uses raw step_idx (0-indexed, same as apply_probabilistic_damage)
            current_wires = apply_probabilistic_shuffle(current_wires, step_idx)

            # Refresh graph topology + cache if any shuffle fired
            if any_shuffle_enabled:
                did_shuffle = detect_wire_change(old_wires, current_wires)
                current_graph, current_cache = jax.lax.cond(
                    did_shuffle,
                    lambda _: refresh_topology(current_graph, current_wires),
                    lambda _: (current_graph, current_cache),
                    None,
                )
                model_updated_graph = apply_model_remat(current_graph, current_cache)
                updated_graph, loss, current_logits, aux = get_loss_and_update_graph(
                    model_updated_graph,
                    logits_original_shapes,
                    current_wires,
                    x_batch,
                    y_batch,
                    loss_cfg,
                    layer_sizes,
                )
                new_carry = (updated_graph, current_gate_mask, current_wires, current_cache)
            else:
                updated_graph, loss, current_logits, aux = apply_model_and_compute_loss(
                    model_fn,
                    current_graph,
                    logits_original_shapes,
                    current_wires,
                    x_batch,
                    y_batch,
                    loss_cfg,
                    layer_sizes,
                )
                new_carry = (updated_graph, current_gate_mask, current_wires)

            return new_carry, (updated_graph, loss, current_logits, aux)

        if any_shuffle_enabled:
            init_carry = (graph, initial_gate_mask, wires, initial_topology_cache)
        else:
            init_carry = (graph, initial_gate_mask, wires)
        final_carry, step_outputs = jax.lax.scan(
            scan_step,
            init_carry,
            xs=jp.arange(num_steps),
            length=num_steps,
        )
        final_graph = final_carry[0]

    return final_graph, step_outputs


def run_inner_loop(
    model: CircuitGatheredAttention | CircuitGNN | CircuitSelfAttention | PerceiverCircuitAttention,
    graph,
    logits_original_shapes: list[tuple],
    wires: list[jp.ndarray],
    x_task: jp.ndarray,
    y_task: jp.ndarray,
    loss_cfg: LossConfig,
    layer_sizes: tuple[tuple[int, int], ...],
    n_rounds: int,
    steps_per_round: int = 1,
    window_size: int = 1,
    data_mode: str = "stream",
    scan_key: jax.random.PRNGKey = None,
    gradient_checkpointing: bool = False,
):
    """
    UNIFIED inner loop spanning the online↔batch continuum (one body, four dials).

    Replaces the separate batch (``run_model_scan_with_loss``) and online
    (``run_model_online_scan_with_loss``) inner loops for the meta-learning continuum.
    Each ROUND r runs the SAME body (batch order):

      1. run ``steps_per_round`` (m) NCA message steps — the model adapts on the PRIOR
         round's fed residual / accumulated state.
      2. run the circuit on this round's W-input subset, compute the round loss L_r.
      3. feed the mean-over-W residual to the output nodes + stash the W-subset in globals.

    Steps 2+3 are exactly ``get_loss_and_update_graph`` (whose
    ``jp.abs(residuals).mean(axis=0)`` IS the mean-over-W: W=1 → the single residual,
    W=full → the original batch mean). So the residual feed is unified — there is no
    per-example-vs-batch-mean branch.

    DIALS
      window_size (W): inputs evaluated per round. Unifies batch ``data_per_batch`` /
        online window / Perceiver attended-count. (static)
      data_mode:
        - "fixed":  ONE W-subset drawn per rollout, reused every round  → BATCH.
                    W ≥ case_n ⇒ the FULL truth table (deterministic; == data_fraction=1.0).
        - "stream": a FRESH W-subset every round                        → ONLINE.
      steps_per_round (m): NCA message steps per round. (static)
      n_rounds (N): number of rounds. (static)

    predict-before-adapt EMERGES from ``data_mode="stream"`` — no flag. Round r's m
    message steps ran on round r-1's residual, and L_r is evaluated on round r's FRESH
    inputs *before* round r's residual is fed, so the rule must predict unseen inputs
    from prior feedback (the VSML generalization signal). In "fixed" mode every round
    sees the same inputs → iterative batch refinement. ``feedback_mode`` (scalar residual
    vs. value-aware cross-attention) is the MODEL's concern, NOT this loop's.

    At ``data_mode="fixed", window_size>=case_n, steps_per_round=1`` this is byte-identical
    (up to the loss-normalization) to ``run_model_scan_with_loss``'s no-damage path —
    see the batch-preset regression test. Damage / wire-shuffle / no-repair live ONLY in
    ``run_model_scan_with_loss`` (the demo/recovery regime), not here.

    Args:
        x_task: Full input enumeration [2^input_n, input_n] (shared across circuits).
        y_task: This circuit's full truth table [2^input_n, output_n].
        n_rounds: N — number of rounds (static).
        steps_per_round: m — NCA message steps per round (static).
        window_size: W — inputs per round (1 = pure online for stream; static).
        data_mode: "fixed" (batch) | "stream" (online).
        scan_key: per-circuit key; draws the index rows (and Perceiver subsample key).

    Returns:
        (final_graph, round_losses, final_logits, round_aux) where ``round_losses`` is
        [n_rounds], ``final_logits`` are the fully-adapted circuit's logits, and
        ``round_aux`` stacks the per-round aux dict (loss / hard_loss / accuracy /
        hard_accuracy / predictions / residuals ...) along a leading [n_rounds] axis.
    """
    from boolean_nca_cc.utils.graph_builder import GraphGlobals

    if scan_key is None:
        scan_key = jax.random.PRNGKey(42)
    stream_key, subsample_key = jax.random.split(scan_key)

    case_n = x_task.shape[0]
    # Per-round index rows [N, W], uniform with replacement (the circuit must generalize
    # to inputs it never saw — for 12-bit add, 2^12 >> N·W, so stream rounds are almost
    # all novel: the hard online-generalization regime the diagnosis wants).
    if data_mode == "fixed":
        # One subset committed per rollout, reused every round (= batch / per-rollout
        # data commitment). W >= case_n → the full deterministic truth table.
        if window_size >= case_n:
            fixed_idx = jp.arange(case_n)
        else:
            fixed_idx = jax.random.randint(stream_key, (window_size,), 0, case_n)
        idx_stream = jp.broadcast_to(fixed_idx, (n_rounds, fixed_idx.shape[0]))
    elif data_mode == "stream":
        # Fresh subset per round (= per-step data commitment / online).
        idx_stream = jax.random.randint(stream_key, (n_rounds, window_size), 0, case_n)
    else:
        raise ValueError(f"data_mode must be 'fixed' or 'stream', got {data_mode!r}")

    # Precompute the model fn (attention masks etc.) once; topology is fixed.
    model_fn, _ = _prepare_model_fn(model, graph, gradient_checkpointing)

    # ── init globals (the "prior" the round-0 message steps operate on) ──────────────
    # fixed (batch): pre-compute round-0's residual via get_loss_and_update_graph so
    #   round-0's message steps SEE it — reproduces run_model_scan_with_loss's pre-scan
    #   initialization (batch-faithful).
    # stream (online): NO pre-fed residual (residuals=0) → round-0 is the pure 0-shot
    #   prediction (predict-before-adapt purity).
    x0 = x_task[idx_stream[0]]
    y0 = y_task[idx_stream[0]]
    if data_mode == "fixed":
        graph, _, _, _ = get_loss_and_update_graph(
            graph=graph,
            logits_original_shapes=logits_original_shapes,
            wires=wires,
            x_data=x0,
            y_data=y0,
            loss_cfg=loss_cfg,
            layer_sizes=layer_sizes,
        )
        graph = graph._replace(globals=graph.globals._replace(subsample_key=subsample_key))
    else:
        graph = graph._replace(
            globals=GraphGlobals(
                loss=jp.float32(0.0),
                update_steps=0,
                x_data=x0,
                y_data=y0,
                residuals=jp.zeros_like(y0),
                subsample_key=subsample_key,
            )
        )

    def round_step(carry_graph, idx_row):
        x_r = x_task[idx_row]  # [W, input_n]
        y_r = y_task[idx_row]  # [W, output_n]

        # 1. m NCA message steps on the prior round's fed residual / accumulated state.
        def body(gg, _):
            return model_fn(gg), None

        g, _ = jax.lax.scan(body, carry_graph, xs=None, length=steps_per_round)

        # 2+3. loss on this round's W-subset + feed mean-over-W residual + stash the
        #      subset in globals (get_loss_and_update_graph does both, identically to
        #      the batch regime's per-step update).
        g, loss_r, _logits_r, aux_r = get_loss_and_update_graph(
            graph=g,
            logits_original_shapes=logits_original_shapes,
            wires=wires,
            x_data=x_r,
            y_data=y_r,
            loss_cfg=loss_cfg,
            layer_sizes=layer_sizes,
        )
        return g, (loss_r, aux_r)

    final_graph, (round_losses, round_aux) = jax.lax.scan(
        round_step, graph, xs=idx_stream, length=n_rounds
    )
    final_logits = extract_logits_from_graph(final_graph, logits_original_shapes)
    return final_graph, round_losses, final_logits, round_aux


def evaluate_model_inner_loop_batched(
    model: CircuitGatheredAttention | CircuitGNN | CircuitSelfAttention | PerceiverCircuitAttention,
    batch_wires: list[jp.ndarray],
    batch_logits: list[jp.ndarray],
    x_task: jp.ndarray,
    y_task: jp.ndarray,
    input_n: int,
    arity: int = 2,
    circuit_hidden_dim: int = 16,
    loss_cfg=None,
    layer_sizes: list[tuple[int, int]] | None = None,
    n_rounds: int = 64,
    steps_per_round: int = 1,
    window_size: int = 1,
    data_mode: str = "stream",
    scan_key: jax.random.PRNGKey = jax.random.PRNGKey(42),
    bidirectional_edges: bool = True,
    score_output_bits: int | None = None,
    chunk_size: int | None = None,
) -> tuple[jp.ndarray, dict]:
    """
    Inner-loop OOD evaluation (the meta-learning readout), spanning the online↔batch
    continuum via ``data_mode`` / ``window_size`` — the eval sibling of ``run_inner_loop``.

    For each circuit in the batch: run the unified inner loop (``run_inner_loop``) to ADAPT
    a FRESH circuit to its task — from a fixed batch when ``data_mode="fixed"`` or a stream
    when ``"stream"`` — then read the **hard accuracy on the FULL input enumeration**: the
    generalization readout that adjudicates (e.g.) the add↔reverse gap, directly comparable
    across the continuum and to the batch regime's ``final_hard_accuracy``.

    ``score_output_bits`` (padding-corrected metric): if set, accuracy is computed over only
    the first that-many output bits — the MEANINGFUL bits of a library subject, excluding the
    right-zero padding to ``output_n`` (de-confounds the cross-task OOD readout: parity/add/
    multiply are otherwise inflated by free padded bits). None = score all ``output_n`` bits.

    ``chunk_size``: process the batch of circuits in groups of this many (vmap within a chunk,
    Python loop across chunks) to cap peak VRAM — the 256-round-from-fresh eval over the OOD
    circuits OOMs a 32 GB V100 if vmapped all at once. None = vmap all circuits together.

    ``y_task`` may be 2D ``[2^input_n, output_n]`` (one task shared across the batch — the
    per-subject readout) or 3D ``[batch, 2^input_n, output_n]`` (per-circuit).

    Returns:
        (batch_graphs, step_metrics) where ``step_metrics["hard_accuracy"][-1]`` is the
        batch-mean full-table hard accuracy AFTER adaptation (the adjudicator). Also includes
        the per-round adaptation-loss / accuracy curves for diagnostics.
    """
    if loss_cfg is None:
        loss_cfg = LOSS_L4

    batch_size = batch_logits[0].shape[0]
    y_is_per_circuit = y_task.ndim == 3

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
    scan_keys = jax.random.split(scan_key, batch_size)

    def run_single(graph, wires, logits, key, y_local):
        _final_graph, round_losses, final_logits, round_aux = run_inner_loop(
            model=model,
            graph=graph,
            logits_original_shapes=[logit.shape for logit in logits],
            wires=wires,
            x_task=x_task,
            y_task=y_local,
            loss_cfg=loss_cfg,
            layer_sizes=layer_sizes,
            n_rounds=n_rounds,
            steps_per_round=steps_per_round,
            window_size=window_size,
            data_mode=data_mode,
            scan_key=key,
            gradient_checkpointing=True,
        )
        # Full-table readout after adapting (the generalization metric).
        _full_loss, full_aux = get_loss_from_wires_logits(
            logits=final_logits, wires=wires, x=x_task, y_target=y_local, loss_cfg=loss_cfg
        )
        if score_output_bits is None:
            acc, hard_acc = full_aux["accuracy"], full_aux["hard_accuracy"]
        else:
            # Padding-corrected: score only the first ``score_output_bits`` (meaningful)
            # output bits. ``score_output_bits == output_n`` reproduces the full metric.
            yy = y_local[:, :score_output_bits]
            acc = jp.mean(jp.round(full_aux["predictions"][:, :score_output_bits]) == yy)
            hard_acc = jp.mean(jp.round(full_aux["hard_predictions"][:, :score_output_bits]) == yy)
        return round_losses, round_aux["hard_accuracy"], acc, hard_acc

    y_in_axes = 0 if y_is_per_circuit else None
    vmapped = nnx.vmap(run_single, in_axes=(0, 0, 0, 0, y_in_axes))

    if chunk_size is None or chunk_size >= batch_size:
        round_losses, round_hard_acc, full_acc, full_hard_acc = vmapped(
            batch_graphs, batch_wires, batch_logits, scan_keys, y_task
        )
    else:
        # Chunk over circuits to cap peak VRAM (256-round-from-fresh OOMs the V100 if all
        # circuits are vmapped together). vmap within a chunk; Python loop + concat across.
        parts = []
        for start in range(0, batch_size, chunk_size):
            end = min(start + chunk_size, batch_size)
            g_c = jax.tree.map(lambda a, s=start, e=end: a[s:e], batch_graphs)
            w_c = [w[start:end] for w in batch_wires]
            l_c = [lg[start:end] for lg in batch_logits]
            k_c = scan_keys[start:end]
            y_c = y_task[start:end] if y_is_per_circuit else y_task
            parts.append(vmapped(g_c, w_c, l_c, k_c, y_c))
        round_losses = jp.concatenate([p[0] for p in parts], axis=0)
        round_hard_acc = jp.concatenate([p[1] for p in parts], axis=0)
        full_acc = jp.concatenate([p[2] for p in parts], axis=0)
        full_hard_acc = jp.concatenate([p[3] for p in parts], axis=0)

    step_metrics = {
        "step": list(range(1, n_rounds + 1)),
        # Per-round adaptation loss / single-round accuracy curves (diagnostics).
        "round_loss": jp.mean(round_losses, axis=0).tolist(),
        "round_hard_accuracy": jp.mean(round_hard_acc, axis=0).tolist(),
        # The adjudicator: full-table accuracy AFTER adaptation, exposed as a 1-element
        # list so callers can read ``[-1]`` exactly like the batch eval.
        "accuracy": [float(jp.mean(full_acc))],
        "hard_accuracy": [float(jp.mean(full_hard_acc))],
        "final_full_table_hard_accuracy": float(jp.mean(full_hard_acc)),
    }
    return batch_graphs, step_metrics


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
    permanent_damage: float = 1.0,
    # Delayed probabilistic damage onset
    p_fault_onset_step: int = 0,
    # Pre-existing damage carried in from outside the generator (e.g. shotgun damage
    # applied by the live demo before continuous repair starts).
    initial_gate_knockout_mask: jp.ndarray | None = None,
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
        permanent_damage: Per-gate probability that damage is permanent (float in [0, 1]).
            1.0 = always permanent, 0.0 = always temporary, 0.5 = independent coin flip
            per gate per timestep.
        p_fault_onset_step: Step at which probabilistic damage starts (0 = from the start)
        initial_gate_knockout_mask: Pre-existing damage mask to seed the graph with.
            Flat ``[total_gates]`` (or layered list) of 0.0/1.0 values, matching
            ``build_graph``'s convention. ``None`` = no prior damage (default).

    Yields:
        StepResult: Results from each step including loss, accuracy, predictions, and updated logits
    """
    if loss_cfg is None:
        loss_cfg = LOSS_L4

    # Store original shapes for reconstruction
    logits_original_shapes = [logit.shape for logit in logits]

    # Build initial graph (carrying any prior damage in via the mask).
    graph = build_graph(
        logits,
        wires,
        input_n,
        arity,
        circuit_hidden_dim,
        loss_value=0.0,
        bidirectional_edges=bidirectional_edges,
        gate_knockout_mask=initial_gate_knockout_mask,
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

    # Separate key stream for permanent_damage coin flips
    perm_damage_key = jax.random.PRNGKey(77)

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
            # Per-gate coin flip: each damaged gate independently decides if permanent
            perm_damage_key, perm_key = jax.random.split(perm_damage_key)
            perm_mask = jax.random.uniform(perm_key, shape=gate_mask.shape) < permanent_damage
            new_gate_mask = jp.where(perm_mask, new_mask, gate_mask)
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
            # Per-gate coin flip: each damaged gate independently decides if permanent
            perm_damage_key, perm_key = jax.random.split(perm_damage_key)
            perm_mask = jax.random.uniform(perm_key, shape=gate_mask.shape) < permanent_damage
            gate_mask = jp.where(perm_mask, new_gate_mask, gate_mask)

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
    permanent_damage: float = 1.0,
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
        permanent_damage: Per-gate probability that damage is permanent (float in [0, 1]).
            1.0 = always permanent, 0.0 = always temporary, 0.5 = independent coin flip
            per gate per timestep.
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
    permanent_damage: float | str = 1.0,
    # Delayed probabilistic damage onset
    p_fault_onset_step: int = 0,
    # No-repair baseline
    compute_no_repair_baseline: bool = False,
    # Wire shuffling parameters
    wire_shuffle_steps: list[int] | None = None,
    wire_shuffle_key: jax.random.PRNGKey = jax.random.PRNGKey(42),
    wire_shuffle_fraction: float = 1.0,
    # Chunking and details
    chunk_size: int | None = None,
    return_first_circuit_details: bool = False,
    # Memory: when False, drop the heavy per-step batched ``graphs`` GraphsTuple
    # and ``all_metrics`` (per-circuit per-step arrays) from the returned
    # ``step_metrics``. The cheap ``damaged_fraction_per_step`` is *always*
    # included so stepwise plotting/damage overlays keep working. Default
    # ``True`` preserves backward compatibility for direct callers (e.g.
    # paper_figures/eval_pca_trajectories.py indexes
    # ``step_metrics["all_metrics"]`` and ``step_metrics["graphs"]``). The
    # periodic-eval path in train_loop.py flips this to False to keep the
    # JAX BFC pool from being pinned by ``[batch, n_steps, ...graph_shape]``
    # tensors across the three eval scenarios.
    keep_full_graphs: bool = True,
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

    # ``y_data.ndim`` selects the per-circuit-task path the same way the
    # train loop does (see ``batch_loss_fn`` in train_loop.py):
    #   - 2D [N_samples, output_n]            → global task, shared across batch.
    #   - 3D [batch_size, N_samples, output_n] → per-circuit task path.
    # When per-circuit, chunking must slice y_data too.
    y_is_per_circuit = y_data.ndim == 3
    if y_is_per_circuit and y_data.shape[0] != batch_size:
        raise ValueError(
            f"per-circuit y_data has leading dim {y_data.shape[0]} but batch_size is {batch_size}"
        )

    # Handle chunking for large batches
    if chunk_size is not None and batch_size > chunk_size:
        log.info(f"Using chunked evaluation (chunks of {chunk_size})")
        chunks = []
        for i in range(0, batch_size, chunk_size):
            end = min(i + chunk_size, batch_size)
            y_chunk = y_data[i:end] if y_is_per_circuit else y_data
            chunk_final_graphs, chunk_step_metrics = evaluate_model_stepwise_batched(
                model=model,
                batch_wires=[w[i:end] for w in batch_wires],
                batch_logits=[lg[i:end] for lg in batch_logits],
                x_data=x_data,
                y_data=y_chunk,
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
                wire_shuffle_steps=wire_shuffle_steps,
                wire_shuffle_key=wire_shuffle_key,
                wire_shuffle_fraction=wire_shuffle_fraction,
                chunk_size=None,  # Don't recurse further
                return_first_circuit_details=return_first_circuit_details and (i == 0),
                keep_full_graphs=keep_full_graphs,
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
        # ``damaged_fraction_per_step`` is a per-step batch-averaged scalar list
        # (cheap), so we keep it across chunks regardless of ``keep_full_graphs``.
        if "damaged_fraction_per_step" in chunks[0][2]:
            step_metrics["damaged_fraction_per_step"] = [
                sum(w * r["damaged_fraction_per_step"][s] for w, r in chunks) / total
                for s in range(n_message_steps)
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
    # Resolve permanent_damage to a float probability per batch element:
    # "random" -> 0.5, True -> 1.0, False -> 0.0, float -> as-is
    if permanent_damage == "random":
        permanent_damage_val = 0.5
    elif isinstance(permanent_damage, bool):
        permanent_damage_val = 1.0 if permanent_damage else 0.0
    else:
        permanent_damage_val = float(permanent_damage)
    permanent_damage = jax.numpy.full(batch_size, permanent_damage_val)

    if damage_key is None:
        damage_key = jax.random.PRNGKey(0)  # Dummy key, won't be used if no damage_steps
    damage_keys = jax.random.split(damage_key, batch_size)

    # Split wire shuffle keys per batch element (each circuit gets different new wires)
    wire_shuffle_keys = jax.random.split(wire_shuffle_key, batch_size)

    # Run unified scan for each circuit in batch
    def run_single_scan(graph, wires, logits, scan_key, permanent_damage, ws_key, y_local):
        # ``y_local`` is per-circuit when y_is_per_circuit (in_axes=0 below)
        # and the shared y_data otherwise (in_axes=None).
        return run_model_scan_with_loss(
            model=model,
            graph=graph,
            num_steps=n_message_steps,
            logits_original_shapes=[logit.shape for logit in logits],
            wires=wires,
            x_data=x_data,
            y_data=y_local,
            loss_cfg=loss_cfg,
            layer_sizes=layer_sizes,
            data_fraction=1.0,
            scan_key=scan_key,
            gradient_checkpointing=True,
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
            # Wire shuffling
            wire_shuffle_steps=wire_shuffle_steps,
            wire_shuffle_key=ws_key,
            wire_shuffle_fraction=wire_shuffle_fraction,
        )

    # Vmap over batch — last in_axes element switches per shape of y_data.
    y_in_axes = 0 if y_is_per_circuit else None
    final_graphs, batch_step_outputs = nnx.vmap(
        run_single_scan, in_axes=(0, 0, 0, 0, 0, 0, y_in_axes)
    )(
        batch_graphs,
        batch_wires,
        batch_logits,
        damage_keys,
        permanent_damage,
        wire_shuffle_keys,
        y_data,
    )

    # Extract and average metrics
    all_batch_graphs, losses, _, aux_data = batch_step_outputs

    # ── Host-side averaging in a single device→host transfer per metric ──
    # losses / aux_data["..."] are [batch, n_steps] (vmap over scan). The old
    # form did `n_steps` separate `float(jp.mean(x[:, i]))` calls, each blocking
    # on the device and allocating a tiny output buffer — 1024 such calls per
    # eval, which is both slow and pressures the BFC allocator. One axis-mean +
    # one `.tolist()` is functionally identical, only cheaper.
    def _mean_to_list(arr):
        return jp.mean(arr, axis=0).tolist()

    step_metrics = {
        "step": list(range(1, n_message_steps + 1)),
        "loss": _mean_to_list(losses),
        "hard_loss": _mean_to_list(aux_data["hard_loss"]),
        "accuracy": _mean_to_list(aux_data["accuracy"]),
        "hard_accuracy": _mean_to_list(aux_data["hard_accuracy"]),
    }

    # Pre-compute the cheap per-step damage fraction here (it only reads
    # ``nodes["gate_knockout_mask"]`` + ``nodes["layer"]``), so the heavy
    # ``all_batch_graphs`` GraphsTuple can be dropped when not requested while
    # the stepwise damage-overlay plot still has data to render. ``.tolist()``
    # materialises to host so the source can be GC'd.
    step_metrics["damaged_fraction_per_step"] = get_fraction_damaged_gates(
        all_batch_graphs
    ).tolist()

    # ── Heavy, per-circuit-per-step tensors: gated by ``keep_full_graphs`` ──
    # Default ``True`` preserves backward compat for direct callers (notebooks
    # rely on ``step_metrics["all_metrics"][metric][b, t]``). The periodic-eval
    # path in train_loop.py flips this to ``False`` to avoid pinning
    # ``[batch, n_steps, ...graph_shape]`` tensors across the three eval
    # scenarios (out_test undamaged → damaged → discrete_damaged); see also
    # the memory audit comment in the function docstring.
    if keep_full_graphs:
        # Vmapped graphs: size [batch_size, n_steps, ...graph_shape...]
        step_metrics["graphs"] = all_batch_graphs
        # Per-circuit per-step metric tensors. ``losses`` / ``aux_data[...]``
        # are already stacked along the scan axis by ``jax.lax.scan`` and
        # vmap'd over the batch axis, so they are already [batch, n_steps];
        # the prior ``jp.stack(...)`` was a no-op-shape-wise but allocated
        # an extra buffer.
        step_metrics["all_metrics"] = {
            "loss": losses,
            "hard_loss": aux_data["hard_loss"],
            "accuracy": aux_data["accuracy"],
            "hard_accuracy": aux_data["hard_accuracy"],
        }
    else:
        # Sentinel so callers can ``result.get("graphs") is None`` check.
        step_metrics["graphs"] = None
        step_metrics["all_metrics"] = None

    # Include no-repair baseline metrics if computed
    if compute_no_repair_baseline and "no_repair_loss" in aux_data:
        step_metrics["no_repair_loss"] = _mean_to_list(aux_data["no_repair_loss"])
        step_metrics["no_repair_hard_loss"] = _mean_to_list(aux_data["no_repair_hard_loss"])
        step_metrics["no_repair_accuracy"] = _mean_to_list(aux_data["no_repair_accuracy"])
        step_metrics["no_repair_hard_accuracy"] = _mean_to_list(aux_data["no_repair_hard_accuracy"])

    # Optionally extract detailed results for first circuit (for visualization).
    # Must run before the explicit ``del`` below so ``batch_step_outputs`` is
    # still alive.
    if return_first_circuit_details:
        step_metrics["first_circuit_results"] = _extract_single_circuit_step_results(
            batch_step_outputs, circuit_idx=0, n_message_steps=n_message_steps
        )

    # When the caller doesn't want the heavy tensors, explicitly drop our
    # references so refcounting can release the underlying device buffers
    # before this function returns (Python wouldn't otherwise free them
    # until the local scope exits, which keeps them alive across
    # ``return final_graphs, step_metrics`` in some interpreters).
    if not keep_full_graphs:
        del batch_step_outputs, all_batch_graphs, losses, aux_data

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


# =====================================================================
# BP baseline functions
# =====================================================================
def _logits_to_flat_nodes(logits_list, input_n, arity):
    logit_dim = 2**arity
    parts = [jp.zeros((input_n, logit_dim))]
    for l in logits_list:
        gn, gs, ls = l.shape
        parts.append(l.reshape(gn * gs, ls))
    return jp.concatenate(parts, axis=0)


def run_bp_scan(
    opt,
    batch_params,
    batch_wires,
    x_data,
    y_data,
    n_steps,
    layer_sizes,
    input_n,
    arity,
    loss_cfg,
    damage_mode="none",
    permanent: float | str | bool = 0.0,
    damage_steps=None,
    knockouts_per_event=0,
    p_fault=None,
    p_fault_onset_step=0,
    faulty_value=-10.0,
    key=jax.random.PRNGKey(42),
    verbose=True,
    x_test=None,
    y_test=None,
    compile=True,
    use_scan=True,
):
    """
    Fully compiled BP baseline: scan(time) × vmap(batch) × jit.

    When ``use_scan=True`` (default) the entire optimization compiles to
    ONE XLA program via ``jax.lax.scan``.  First call is slow
    (compilation), subsequent calls are instant.

    When ``use_scan=False`` a Python-level for-loop replaces the scan.
    Each step is still vmapped and (optionally) JIT-compiled, but the
    compilation unit is a *single* step instead of the whole trajectory.
    This is **much** faster on CPU because:

    * XLA compilation is orders of magnitude cheaper (one step vs. N steps).
    * Progress is reported every ~10 % of steps so you can monitor the run.
    * Memory usage is lower (no need to materialise the full scan).

    Args:
        use_scan: If True, use ``jax.lax.scan`` (fast on GPU, slow
            compilation). If False, use a Python for-loop (CPU-friendly).
        permanent: Per-gate probability that damage is permanent (float in [0, 1]).
            1.0 = always permanent, 0.0 = always temporary, 0.5 = independent coin flip
            per gate per timestep.
            Also accepts "random" (equivalent to 0.5) or bool (True=1.0, False=0.0).
    """
    # Resolve permanent to a float probability
    if permanent == "random":
        permanent = 0.5
    elif isinstance(permanent, bool):
        permanent = 1.0 if permanent else 0.0
    else:
        permanent = float(permanent)

    if isinstance(loss_cfg, dict):
        loss_cfg = LossConfig.from_dict(loss_cfg)

    batch_size = batch_wires[0].shape[0]
    total_gates = sum(n for n, _ in layer_sizes)
    input_gates = layer_sizes[0][0]
    output_gates = layer_sizes[-1][0]
    eligible_start = input_gates
    eligible_end = total_gates - output_gates
    eligible_flat = create_eligible_gate_mask(layer_sizes)

    eval_x = x_test if x_test is not None else x_data
    eval_y = y_test if y_test is not None else y_data

    # ── Pre-compute scan inputs (shared across batch) ────────────
    step_indices = jp.arange(n_steps)

    damage_flags = jp.zeros(n_steps, dtype=jp.int32)
    if damage_steps is not None:
        for s in damage_steps:
            damage_flags = damage_flags.at[int(s)].set(1)

    # Per-circuit-per-step keys: (n_steps, batch, 2)
    all_keys = jax.random.split(key, n_steps * batch_size + 1)[1:]
    all_keys = all_keys.reshape(n_steps, batch_size, -1)

    # ── Initial batched state ────────────────────────────────────
    batch_opt_state = jax.vmap(opt.init)(batch_params)
    batch_gate_mask = jp.ones((batch_size, total_gates), dtype=jp.float32)

    # ── Pure helpers (all JAX-traceable) ─────────────────────────

    def _preserve(new_params, old_params, gmask_flat):
        idx = input_gates
        out = []
        for new_l, old_l in zip(new_params, old_params, strict=False):
            gn, gs, ls = new_l.shape
            n = gn * gs
            m = gmask_flat[idx : idx + n].reshape(gn, gs, 1)
            out.append(jp.where(m == 0.0, old_l, new_l))
            idx += n
        return out

    def _det_damage(params, gmask, k):
        knockout_key, perm_key = jax.random.split(k)
        ko = create_flat_knockout_pattern(
            knockout_key, total_gates, eligible_start, eligible_end, knockouts_per_event
        )
        idx = input_gates
        new_p = []
        for ll in params:
            gn, gs, ls = ll.shape
            m = ko[idx : idx + gn * gs].reshape(gn, gs, 1)
            new_p.append(jp.where(m == 0.0, faulty_value, ll))
            idx += gn * gs
        # Per-gate coin flip: each damaged gate independently decides if permanent
        perm_mask = jax.random.uniform(perm_key, shape=gmask.shape) < permanent
        new_m = jp.where(perm_mask, gmask * ko, gmask)
        return new_p, new_m

    def _prob_damage(params, gmask, k):
        fault_key, perm_key = jax.random.split(k)
        ld = params[0].shape[-1]
        parts = [jp.zeros((input_n, ld))]
        for l in params:
            gn, gs, ls = l.shape
            parts.append(l.reshape(gn * gs, ls))
        full = jp.concatenate(parts)
        # Always pass the current (accumulated) mask — gates permanently
        # damaged in earlier steps stay dead; temporary damage was never
        # recorded in the mask so it doesn't affect eligibility.
        new_full, new_m = apply_probabilistic_gate_failure(
            fault_key, full, gmask, eligible_flat, p_fault, faulty_value
        )
        # Per-gate coin flip: each damaged gate independently decides if permanent
        perm_mask = jax.random.uniform(perm_key, shape=gmask.shape) < permanent
        final_m = jp.where(perm_mask, new_m, gmask)
        idx = input_n
        new_p = []
        for ll in params:
            gn, gs, ls = ll.shape
            n = gn * gs
            new_p.append(new_full[idx : idx + n].reshape(gn, gs, ls))
            idx += n
        return new_p, final_m

    # ── Shared single-step body (used by both scan & loop) ──────

    def _step_body(params, wires, opt_state, gate_mask, step_idx, step_key, dmg_flag):
        """One optimisation step for a single circuit."""
        # ── Damage ──
        if damage_mode == "deterministic":
            params, gate_mask = jax.lax.cond(
                dmg_flag > 0,
                lambda p, gm, k: _det_damage(p, gm, k),
                lambda p, gm, _k: (p, gm),
                params,
                gate_mask,
                step_key,
            )
        elif damage_mode == "probabilistic" and p_fault is not None:
            new_p, new_m = _prob_damage(params, gate_mask, step_key)
            past = step_idx >= p_fault_onset_step
            params = jax.tree.map(lambda n, o: jp.where(past, n, o), new_p, params)
            gate_mask = jp.where(past, new_m, gate_mask)

        # ── Gradient step (on train data) ──
        gml = flat_to_layered_mask(gate_mask, layer_sizes)
        (_, _), grad = jax.value_and_grad(
            lambda p: loss_f(
                p,
                wires,
                x_data,
                y_data,
                loss_cfg=loss_cfg,
                gate_mask=gml,
            ),
            has_aux=True,
        )(params)
        upd, new_os = opt.update(grad, opt_state, params)
        new_p = optax.apply_updates(params, upd)
        new_p = _preserve(new_p, params, gate_mask)

        # ── Eval (on test data, post-update) ──
        loss, aux = loss_f(
            new_p,
            wires,
            eval_x,
            eval_y,
            loss_cfg=loss_cfg,
            gate_mask=gml,
        )

        # ── Record ──
        flat_log = _logits_to_flat_nodes(new_p, input_n, arity)

        carry = (new_p, new_os, gate_mask)
        outputs = (
            flat_log,
            gate_mask,
            loss,
            aux["hard_loss"],
            aux["accuracy"],
            aux["hard_accuracy"],
        )
        return carry, outputs

    # ==============================================================
    #  Path A: jax.lax.scan  (fast on GPU, heavy compilation)
    # ==============================================================
    if use_scan:

        def optimize_circuit(params, wires, opt_state, gate_mask, keys):
            """Scan over n_steps for one circuit. Will be vmapped over batch."""

            def scan_fn(carry, inputs):
                params, opt_state, gate_mask = carry
                step_idx, step_key, dmg_flag = inputs
                return _step_body(params, wires, opt_state, gate_mask, step_idx, step_key, dmg_flag)

            init_carry = (params, opt_state, gate_mask)
            scan_inputs = (step_indices, keys, damage_flags)
            _, outputs = jax.lax.scan(scan_fn, init_carry, scan_inputs)
            return outputs  # each element: (n_steps, ...)

        compiled_optimize = jax.vmap(
            optimize_circuit,
            in_axes=(0, 0, 0, 0, 1),
            #         params wires opt_st mask keys
            # keys shape: (n_steps, batch, 2) → vmap over axis 1
        )

        if compile:
            compiled_optimize = jax.jit(compiled_optimize)

        if verbose:
            print(
                f"Compiling scan+vmap BP "
                f"(batch={batch_size}, steps={n_steps}, damage={damage_mode})..."
            )

        all_logits, all_gate_mask, all_loss, all_hl, all_acc, all_ha = compiled_optimize(
            batch_params,
            batch_wires,
            batch_opt_state,
            batch_gate_mask,
            all_keys,
        )
        # all_*: (batch, n_steps, ...)

        if verbose:
            print("Done.")

    # ==============================================================
    #  Path B: Python for-loop  (CPU-friendly, lightweight compile)
    # ==============================================================
    else:
        batched_step = jax.vmap(
            _step_body,
            in_axes=(0, 0, 0, 0, None, 0, None),
            #        params wires opt_st mask step_idx step_key dmg_flag
        )

        if compile:
            batched_step = jax.jit(batched_step)

        if verbose:
            print(f"Running loop BP (batch={batch_size}, steps={n_steps}, damage={damage_mode})...")

        b_params, b_opt_state, b_gate_mask = (
            batch_params,
            batch_opt_state,
            batch_gate_mask,
        )
        collect = [[] for _ in range(6)]
        report_every = max(1, n_steps // 10)

        for t in range(n_steps):
            (b_params, b_opt_state, b_gate_mask), step_out = batched_step(
                b_params,
                batch_wires,
                b_opt_state,
                b_gate_mask,
                step_indices[t],
                all_keys[t],
                damage_flags[t],
            )
            for i, o in enumerate(step_out):
                collect[i].append(o)

            if verbose and (t + 1) % report_every == 0:
                step_loss = float(collect[2][-1].mean())
                step_acc = float(collect[4][-1].mean())
                print(
                    f"  Step {t + 1:>{len(str(n_steps))}}/{n_steps}  "
                    f"loss={step_loss:.4f}  acc={step_acc:.4f}"
                )

        # Stack per-step outputs → (batch, n_steps, ...)
        all_logits = jp.stack(collect[0], axis=1)
        all_gate_mask = jp.stack(collect[1], axis=1)
        all_loss = jp.stack(collect[2], axis=1)
        all_hl = jp.stack(collect[3], axis=1)
        all_acc = jp.stack(collect[4], axis=1)
        all_ha = jp.stack(collect[5], axis=1)

        if verbose:
            print("Done.")

    # ── Pack into standard format ────────────────────────────────
    mean_loss = jp.array(all_loss.mean(axis=0))
    mean_hl = jp.array(all_hl.mean(axis=0))
    mean_acc = jp.array(all_acc.mean(axis=0))
    mean_ha = jp.array(all_ha.mean(axis=0))

    step_metrics = {
        "step": list(range(1, n_steps + 1)),
        "loss": mean_loss.tolist(),
        "hard_loss": mean_hl.tolist(),
        "accuracy": mean_acc.tolist(),
        "hard_accuracy": mean_ha.tolist(),
        "soft_accuracy": mean_acc.tolist(),
        "soft_loss": mean_loss.tolist(),
        "graphs": SimpleNamespace(
            nodes={"logits": all_logits, "gate_knockout_mask": all_gate_mask}
        ),
        # ── All metrics ── #batch, #steps
        "all_metrics": {
            "loss": all_loss,
            "hard_loss": all_hl,
            "accuracy": all_acc,
            "hard_accuracy": all_ha,
        },
    }

    if verbose:
        print(
            f"Final: Loss={mean_loss[-1]:.4f}, Acc={mean_acc[-1]:.4f}, Hard Acc={mean_ha[-1]:.4f}"
        )

    return None, step_metrics
