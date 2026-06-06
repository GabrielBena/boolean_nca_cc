"""
Gate knockout utilities for boolean circuit graphs.

This module provides functions to apply permanent structural damage (knockouts)
to gates in boolean circuits during training. This is used to test the
resilience and recovery capabilities of meta-learning models.

**Gate Knockout (Permanent Structural Damage)**:
- Gates are permanently "broken" and cannot be updated by the model
- Logits are set to faulty values that produce zero output (large negative values)
- Gates are prevented from receiving model updates (update mask blocks updates)
- Represents hardware failure or permanent structural damage
- Damaged gates can still send messages to neighbors (depending on strategy)

**Two Damage Modes**:

1. Discrete Mode (for visualization):
   - Apply N knockouts at specific steps
   - Functions: `create_knockout_pattern`, `apply_knockout_to_circuit`, `apply_knockout_to_batch`

2. Probabilistic Mode (for training):
   - Each gate has p_fault probability of failing at each step
   - Realistic continuous failure model
   - Functions: `apply_probabilistic_gate_failure`, `compute_p_fault_from_expected`

Key Functions:
- `create_knockout_pattern`: Generate random knockout masks for circuit layers
- `create_faulty_gate_logits`: Set knocked-out gate logits to faulty values
- `apply_knockout_to_batch`: Apply knockouts to a batch of circuits
- `apply_probabilistic_gate_failure`: Per-step probabilistic gate failure
- `compute_p_fault_from_expected`: Auto-compute p_fault from target failures
- `create_eligible_gate_mask`: Create mask for damage-eligible gates
"""

import jax
import jax.numpy as jp

from boolean_nca_cc.circuits.model import gen_wires
from boolean_nca_cc.training.pool.perturbation import shuffle_wires as _shuffle_wires_partial


def create_knockout_pattern(
    key: jax.random.PRNGKey,
    layer_sizes: list[tuple[int, int]],
    num_knockouts: int,
) -> list[jp.ndarray]:
    """
    Create a knockout pattern for gates in a boolean circuit.

    Returns layered gate masks where knocked-out gates have value 0.0
    and active gates have value 1.0.

    Args:
        key: Random key for reproducible generation
        layer_sizes: List of (gate_n, group_size) for each layer.
                    First element is input layer (never knocked out).
                    Last element is output layer (never knocked out).
        num_knockouts: Number of gates to knock out (from hidden layers only)

    Returns:
        List of gate masks, one per layer:
        - Shape: (gate_n,) for each layer
        - Values: 0.0 = knocked out, 1.0 = active
        - Input layer (index 0): always all 1.0
        - Output layer (last index): always all 1.0
        - Hidden layers: knockouts applied here

    Example:
        >>> key = jax.random.PRNGKey(0)
        >>> layer_sizes = [(8, 1), (16, 2), (16, 2), (4, 1)]  # input, hidden, hidden, output
        >>> masks = create_knockout_pattern(key, layer_sizes, num_knockouts=3)
        >>> masks[0].sum()  # Input layer: all active
        8.0
        >>> masks[-1].sum()  # Output layer: all active
        4.0
        >>> masks[1].sum() + masks[2].sum()  # Hidden layers: 3 knocked out
        29.0  # (16 + 16) - 3 = 29
    """
    # If no knockouts requested, return all-active masks
    if num_knockouts == 0:
        return [jp.ones(gate_n, dtype=jp.float32) for gate_n, _ in layer_sizes]

    # Identify hidden layers (everything except first and last)
    output_layer_idx = len(layer_sizes) - 1

    # Collect all eligible gate indices from hidden layers
    eligible_indices = []
    layer_start_indices = []
    current_idx = 0

    for layer_idx, (gate_n, _group_size) in enumerate(layer_sizes):
        layer_start_indices.append(current_idx)

        # Skip input (0) and output (last) layers
        if layer_idx == 0 or layer_idx == output_layer_idx:
            current_idx += gate_n
            continue

        # Add all gate indices for this hidden layer
        layer_indices = jp.arange(current_idx, current_idx + gate_n)
        eligible_indices.append(layer_indices)
        current_idx += gate_n

    # Handle edge case: no hidden layers or no eligible gates
    if not eligible_indices:
        return [jp.ones(gate_n, dtype=jp.float32) for gate_n, _ in layer_sizes]

    # Concatenate all eligible indices
    all_eligible_indices = jp.concatenate(eligible_indices)

    # Clamp num_knockouts to available gates
    actual_knockouts = min(num_knockouts, len(all_eligible_indices))

    # Randomly sample knockout indices
    knockout_indices = jax.random.choice(
        key, all_eligible_indices, shape=(actual_knockouts,), replace=False
    )

    # Create layered gate masks
    gate_masks = []
    current_idx = 0

    for _layer_idx, (gate_n, _group_size) in enumerate(layer_sizes):
        # Initialize all gates as active (1.0)
        layer_mask = jp.ones(gate_n, dtype=jp.float32)

        # Find knockouts in this layer's range
        layer_start = current_idx
        layer_end = current_idx + gate_n

        layer_knockouts = knockout_indices[
            (knockout_indices >= layer_start) & (knockout_indices < layer_end)
        ]

        # Convert to local indices and apply
        if len(layer_knockouts) > 0:
            local_knockout_indices = layer_knockouts - layer_start
            layer_mask = layer_mask.at[local_knockout_indices].set(0.0)

        gate_masks.append(layer_mask)
        current_idx += gate_n

    return gate_masks


def create_faulty_gate_logits(
    original_logits: jp.ndarray,
    gate_mask: jp.ndarray,
    faulty_value: float = -10.0,
) -> jp.ndarray:
    """
    Set logits of knocked-out gates to faulty values that produce zero output.

    For knocked-out gates, all LUT entries are set to a large negative value
    so that after sigmoid, outputs become ~0 regardless of input.

    Args:
        original_logits: Logits array with shape (group_n, group_size, 2^arity)
        gate_mask: Gate mask with shape (gate_n,) where gate_n = group_n * group_size.
                  Values: 0.0 = knocked out, 1.0 = active
        faulty_value: Value to set for knocked-out gate logits (large negative)

    Returns:
        Modified logits with faulty values for knocked-out gates

    Example:
        >>> logits = jp.zeros((4, 2, 4))  # 8 gates, arity=2
        >>> mask = jp.array([1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0])  # Gate 2 knocked out
        >>> faulty_logits = create_faulty_gate_logits(logits, mask, faulty_value=-10.0)
        >>> faulty_logits[1, 0, :]  # Gate 2 (group 1, position 0)
        Array([-10., -10., -10., -10.], dtype=float32)
    """
    group_n, group_size, _ = original_logits.shape

    # Reshape mask to match logits structure: (gate_n,) -> (group_n, group_size, 1)
    mask_reshaped = gate_mask.reshape(group_n, group_size, 1)
    mask_expanded = jp.broadcast_to(mask_reshaped, original_logits.shape)

    # Apply faulty values where mask is 0 (knocked out)
    faulty_logits = jp.where(
        mask_expanded == 0.0,
        faulty_value,
        original_logits,
    )

    return faulty_logits


def layered_to_flat_mask(layered_masks: list[jp.ndarray]) -> jp.ndarray:
    """
    Convert layered gate masks to flat format.

    Args:
        layered_masks: List of masks, one per layer with shape (gate_n,)

    Returns:
        Flat mask array with shape (total_gates,)
    """
    return jp.concatenate(layered_masks, axis=0)


def flat_to_layered_mask(
    flat_mask: jp.ndarray,
    layer_sizes: list[tuple[int, int]],
) -> list[jp.ndarray]:
    """
    Convert flat gate mask to layered format.

    Args:
        flat_mask: Flat mask array with shape (total_gates,)
        layer_sizes: List of (gate_n, group_size) for each layer

    Returns:
        List of masks, one per layer with shape (gate_n,)
    """
    layered_masks = []
    current_idx = 0

    for gate_n, _group_size in layer_sizes:
        layer_mask = flat_mask[current_idx : current_idx + gate_n]
        layered_masks.append(layer_mask)
        current_idx += gate_n

    return layered_masks


def batch_layered_to_flat_mask(batch_layered_masks: list[jp.ndarray]) -> jp.ndarray:
    """
    Convert batched layered gate masks to batched flat format.

    Args:
        batch_layered_masks: List of mask arrays, one per layer
                            with shape (batch_size, gate_n)

    Returns:
        Batched flat mask array with shape (batch_size, total_gates)
    """
    return jp.concatenate(batch_layered_masks, axis=1)


def batch_flat_to_layered_mask(
    batch_flat_mask: jp.ndarray,
    layer_sizes: list[tuple[int, int]],
) -> list[jp.ndarray]:
    """
    Convert batched flat gate mask to batched layered format.

    Args:
        batch_flat_mask: Batched flat mask array with shape (batch_size, total_gates)
        layer_sizes: List of (gate_n, group_size) for each layer

    Returns:
        List of mask arrays, one per layer with shape (batch_size, gate_n)
    """
    batch_layered_masks = []
    current_idx = 0

    for gate_n, _group_size in layer_sizes:
        layer_mask = batch_flat_mask[:, current_idx : current_idx + gate_n]
        batch_layered_masks.append(layer_mask)
        current_idx += gate_n

    return batch_layered_masks


def create_flat_knockout_pattern(
    key: jax.random.PRNGKey,
    total_gates: int,
    eligible_start: int,
    eligible_end: int,
    num_knockouts: int,
) -> jp.ndarray:
    """
    Create a flat knockout mask for a circuit (vectorizable version).

    This function is designed to be vmapped across a batch of random keys.
    It works with flat masks for efficiency.

    Args:
        key: Random key for reproducible generation
        total_gates: Total number of gates across all layers
        eligible_start: Start index of eligible gates (after input layer)
        eligible_end: End index of eligible gates (before output layer)
        num_knockouts: Number of gates to knock out (can be dynamic/traced)

    Returns:
        Flat mask with shape (total_gates,), values 0.0 = knocked out, 1.0 = active
    """
    # Start with all active
    mask = jp.ones(total_gates, dtype=jp.float32)

    # Handle edge case (static check - eligible_start/end are static)
    num_eligible = eligible_end - eligible_start
    if num_eligible <= 0:
        return mask

    # Clamp knockouts to available gates
    actual_knockouts = jp.minimum(num_knockouts, num_eligible)

    # Use Gumbel-top-k trick for differentiable/vmappable sampling without replacement
    # Add Gumbel noise
    gumbel_noise = jax.random.gumbel(key, shape=(num_eligible,))

    # Get the indices that would sort the noise (descending)
    sorted_indices = jp.argsort(-gumbel_noise)

    # Compute ranks: ranks[i] = position of element i in sorted order
    # This avoids dynamic slicing which JAX can't handle with traced values
    ranks = jp.argsort(sorted_indices)

    # Create knockout mask for eligible gates: knock out where rank < actual_knockouts
    # This works with dynamic actual_knockouts since it's a boolean comparison
    eligible_knockout_mask = ranks < actual_knockouts

    # Apply to full mask: set knocked out positions to 0
    mask = mask.at[eligible_start:eligible_end].set(jp.where(eligible_knockout_mask, 0.0, 1.0))

    return mask


def apply_knockout_to_circuit(
    key: jax.random.PRNGKey,
    logits: list[jp.ndarray],
    layer_sizes: list[tuple[int, int]],
    num_knockouts: int,
    faulty_value: float = -10.0,
    flat: bool = False,
) -> tuple[list[jp.ndarray], list[jp.ndarray]]:
    """
    Apply permanent gate knockout to a single circuit.

    Knocked-out gates:
    1. Have their logits set to faulty values (produce zero output)
    2. Are prevented from receiving model updates (via mask in node features)

    Args:
        key: Random key for knockout pattern generation
        logits: List of logit arrays for each layer (excluding input layer)
        layer_sizes: Circuit layer sizes including input layer
        num_knockouts: Number of gates to permanently knock out
        faulty_value: Value for knocked-out gate logits

    Returns:
        Tuple of (modified_logits, knockout_masks):
        - modified_logits: List of logit arrays with faulty values applied
        - knockout_masks: List of mask arrays (one per layer, including input)
    """
    # Generate knockout pattern
    if flat:
        # Find eligible range (hidden layers only - skip input and output)
        total_gates = sum(gate_n for gate_n, _ in layer_sizes)
        input_gates = layer_sizes[0][0]
        output_gates = layer_sizes[-1][0] if len(layer_sizes) > 1 else 0
        eligible_start = input_gates
        eligible_end = total_gates - output_gates
        knockout_mask_flat = create_flat_knockout_pattern(
            key, total_gates, eligible_start, eligible_end, num_knockouts
        )
        # Apply faulty values where mask is 0 (knocked out)
        modified_logits = jp.where(knockout_mask_flat[:, None] == 0.0, faulty_value, logits)
        return modified_logits, knockout_mask_flat

    else:
        knockout_masks = create_knockout_pattern(key, layer_sizes, num_knockouts)

        # Apply faulty logits to knocked-out gates
        # Note: logits list doesn't include input layer, masks list does
        modified_logits = []
        for layer_idx, layer_logits in enumerate(logits):
            # Masks are indexed from 0 (input layer), logits start at layer 1
            layer_mask = knockout_masks[layer_idx + 1]
            faulty_logits = create_faulty_gate_logits(layer_logits, layer_mask, faulty_value)
            modified_logits.append(faulty_logits)

        return modified_logits, knockout_masks


def apply_knockout_to_batch(
    key: jax.random.PRNGKey,
    batch_logits: list[jp.ndarray],
    layer_sizes: list[tuple[int, int]],
    batch_num_knockouts: jp.ndarray,
    faulty_value: float = -10.0,
) -> tuple[list[jp.ndarray], list[jp.ndarray]]:
    """
    Apply permanent gate knockouts to a batch of circuits (vectorized).

    Each circuit in the batch receives an independent knockout pattern.
    Uses vmap for efficient parallel processing.

    Args:
        key: Random key for knockout generation
        batch_logits: List of batched logit arrays, each with shape (batch_size, ...)
        layer_sizes: Circuit layer sizes
        batch_num_knockouts: Number of gates to knock out per circuit, shape (batch_size,)
        faulty_value: Value for knocked-out gate logits

    Returns:
        Tuple of (modified_batch_logits, batch_knockout_masks):
        - modified_batch_logits: List of batched logit arrays with faulty values
        - batch_knockout_masks: List of batched mask arrays (batch_size, gate_n) per layer
    """
    batch_size = batch_logits[0].shape[0]

    # Compute layout info once (static)
    total_gates = sum(gate_n for gate_n, _ in layer_sizes)

    # Find eligible range (hidden layers only - skip input and output)
    input_gates = layer_sizes[0][0]
    output_gates = layer_sizes[-1][0] if len(layer_sizes) > 1 else 0
    eligible_start = input_gates
    eligible_end = total_gates - output_gates

    # Generate keys for batch
    keys = jax.random.split(key, batch_size)

    # Vectorized knockout mask creation using vmap
    vmapped_create_mask = jax.vmap(
        lambda key, num_knockouts: create_flat_knockout_pattern(
            key, total_gates, eligible_start, eligible_end, num_knockouts
        )
    )
    batch_flat_masks = vmapped_create_mask(
        keys, batch_num_knockouts
    )  # Shape: (batch_size, total_gates)

    # Apply faulty logits to each layer using vectorized operations
    modified_batch_logits = []
    current_idx = layer_sizes[0][0]  # Skip input layer

    for layer_idx, (gate_n, group_size) in enumerate(layer_sizes[1:], start=1):
        # Extract mask for this layer: (batch_size, gate_n)
        layer_masks = batch_flat_masks[:, current_idx : current_idx + gate_n]

        # Get logits for this layer: (batch_size, group_n, group_size, lut_size)
        layer_logits = batch_logits[layer_idx - 1]  # logits list excludes input layer
        group_n = gate_n // group_size

        # Reshape mask for broadcasting: (batch_size, group_n, group_size, 1)
        mask_reshaped = layer_masks.reshape(batch_size, group_n, group_size, 1)
        mask_expanded = jp.broadcast_to(mask_reshaped, layer_logits.shape)

        # Apply faulty values where mask is 0 (knocked out)
        modified_logits = jp.where(mask_expanded == 0.0, faulty_value, layer_logits)
        modified_batch_logits.append(modified_logits)

        current_idx += gate_n

    # Convert flat masks to layered format for return
    batch_knockout_masks = batch_flat_to_layered_mask(batch_flat_masks, layer_sizes)

    return modified_batch_logits, batch_knockout_masks


# =============================================================================
# Probabilistic Gate Failure Functions
# =============================================================================


def create_eligible_gate_mask(
    layer_sizes: tuple[tuple[int, int], ...] | list[tuple[int, int]],
) -> jp.ndarray:
    """
    Create a flat mask indicating which gates are eligible for damage.

    Eligible gates are in hidden layers only (not input or output).

    Args:
        layer_sizes: Tuple/list of (gate_n, group_size) for each layer.
                    First is input layer, last is output layer.

    Returns:
        Flat mask with shape (total_gates,):
        - 1.0 for hidden layer gates (eligible for damage)
        - 0.0 for input/output layer gates (protected)

    Example:
        >>> layer_sizes = [(8, 1), (16, 2), (16, 2), (4, 1)]  # 8 in, 32 hidden, 4 out
        >>> mask = create_eligible_gate_mask(layer_sizes)
        >>> mask[:8].sum()  # Input layer: all protected
        0.0
        >>> mask[8:40].sum()  # Hidden layers: all eligible
        32.0
        >>> mask[40:].sum()  # Output layer: all protected
        0.0
    """
    masks = []
    n_layers = len(layer_sizes)
    for i, (gate_n, _) in enumerate(layer_sizes):
        if i == 0 or i == n_layers - 1:  # Input or output layer
            masks.append(jp.zeros(gate_n, dtype=jp.float32))
        else:  # Hidden layer
            masks.append(jp.ones(gate_n, dtype=jp.float32))
    return jp.concatenate(masks)


def apply_probabilistic_gate_failure(
    key: jax.random.PRNGKey,
    logits: jp.ndarray,
    gate_mask: jp.ndarray,
    eligible_mask: jp.ndarray,
    p_fault: float,
    faulty_value: float = -10.0,
) -> tuple[jp.ndarray, jp.ndarray]:
    """
    Apply probabilistic gate failure: each active eligible gate has p_fault
    probability of permanent failure at this timestep.

    This function is designed to be called at every optimization step within
    a JAX scan loop. Damage accumulates permanently.

    Args:
        key: Random key for this step's failure sampling
        logits: Gate logits with shape [n_gates, logit_dim] or [n_gates]
        gate_mask: Current gate mask [n_gates], 1.0 = active, 0.0 = knocked out
        eligible_mask: Eligibility mask [n_gates], 1.0 = can fail, 0.0 = protected
        p_fault: Probability of failure per gate per step (typically very small)
        faulty_value: Value to set for failed gate logits

    Returns:
        Tuple of (new_logits, new_gate_mask):
        - new_logits: Updated logits with faulty values for newly failed gates
        - new_gate_mask: Updated mask with newly failed gates set to 0.0

    Example:
        >>> key = jax.random.PRNGKey(0)
        >>> logits = jp.zeros((44, 4))  # 44 gates, 4-entry LUT
        >>> gate_mask = jp.ones(44)
        >>> eligible_mask = create_eligible_gate_mask([(8, 1), (16, 2), (16, 2), (4, 1)])
        >>> new_logits, new_mask = apply_probabilistic_gate_failure(
        ...     key, logits, gate_mask, eligible_mask, p_fault=0.1
        ... )
        >>> # Some gates in hidden layers may now be knocked out
        >>> (new_mask[:8] == 1.0).all()  # Input layer protected
        True
        >>> (new_mask[40:] == 1.0).all()  # Output layer protected
        True
    """
    # Generate random values for each gate
    rand = jax.random.uniform(key, gate_mask.shape)

    # Only active + eligible gates can fail
    can_fail = (gate_mask == 1.0) & (eligible_mask == 1.0)
    fails_this_step = can_fail & (rand < p_fault)

    # Update mask (accumulate damage - permanent)
    new_mask = jp.where(fails_this_step, 0.0, gate_mask)

    # Update logits for newly failed gates
    if logits.ndim == 2:
        new_logits = jp.where(fails_this_step[:, None], faulty_value, logits)
    else:
        new_logits = jp.where(fails_this_step, faulty_value, logits)

    return new_logits, new_mask


def compute_p_fault_from_expected(
    expected_faulty_gates: float,
    n_eligible_gates: int,
    expected_lifetime_steps: int,
) -> float:
    """
    Compute the per-gate-per-step failure probability to achieve a target
    expected number of faulty gates over a circuit's lifetime.

    Uses the formula for independent Bernoulli trials:
        E[failures] = n * (1 - (1-p)^L)

    For small p, this approximates to: p ≈ k / (n * L)

    Args:
        expected_faulty_gates: Target expected number of gates to fail (k)
        n_eligible_gates: Number of gates that can fail (n)
        expected_lifetime_steps: Expected number of steps before reset (L)

    Returns:
        Per-gate-per-step failure probability (p)

    Raises:
        ValueError: If inputs are invalid

    Example:
        >>> # Target 4 faulty gates, 32 eligible gates, 2000 step lifetime
        >>> p = compute_p_fault_from_expected(4, 32, 2000)
        >>> p  # Approximately 0.0000625
        6.25e-05
    """
    if expected_faulty_gates <= 0:
        return 0.0
    if n_eligible_gates <= 0:
        raise ValueError("n_eligible_gates must be positive")
    if expected_lifetime_steps <= 0:
        raise ValueError("expected_lifetime_steps must be positive")

    # Ratio of expected failures to eligible gates
    k_over_n = expected_faulty_gates / n_eligible_gates

    if k_over_n >= 1.0:
        # Can't expect more failures than gates - saturate
        # Use approximate formula with k = n
        return 1.0 / expected_lifetime_steps

    # Exact formula: p = 1 - (1 - k/n)^(1/L)
    # This inverts: k/n = 1 - (1-p)^L
    p = 1.0 - (1.0 - k_over_n) ** (1.0 / expected_lifetime_steps)

    return float(p)


def count_eligible_gates(
    layer_sizes: tuple[tuple[int, int], ...] | list[tuple[int, int]],
) -> int:
    """
    Count the number of gates eligible for damage (hidden layers only).

    Args:
        layer_sizes: Tuple/list of (gate_n, group_size) for each layer

    Returns:
        Number of gates in hidden layers (excluding input and output)
    """
    if len(layer_sizes) <= 2:
        return 0  # Only input and output layers
    return sum(gate_n for gate_n, _ in layer_sizes[1:-1])


def get_total_gates(layer_sizes: list[tuple[int, int]]) -> int:
    """
    Calculate total number of gates across all layers.

    Args:
        layer_sizes: List of (gate_n, group_size) for each layer

    Returns:
        Total number of gates
    """
    return sum(gate_n for gate_n, _ in layer_sizes)


def count_knockouts_in_mask(gate_mask: jp.ndarray) -> int:
    """
    Count the number of knocked-out gates in a mask.

    Args:
        gate_mask: Flat or layered mask (0.0 = knocked out, 1.0 = active)

    Returns:
        Number of knocked-out gates
    """
    if isinstance(gate_mask, list):
        # Layered format
        return int(sum(jp.sum(1.0 - m) for m in gate_mask))
    else:
        # Flat format
        return int(jp.sum(1.0 - gate_mask))


def create_active_mask(layer_sizes: list[tuple[int, int]]) -> list[jp.ndarray]:
    """
    Create a mask with all gates active (no knockouts).

    Args:
        layer_sizes: List of (gate_n, group_size) for each layer

    Returns:
        List of all-ones masks, one per layer
    """
    return [jp.ones(gate_n, dtype=jp.float32) for gate_n, _ in layer_sizes]


def create_flat_active_mask(layer_sizes: list[tuple[int, int]]) -> jp.ndarray:
    """
    Create a flat mask with all gates active (no knockouts).

    Args:
        layer_sizes: List of (gate_n, group_size) for each layer

    Returns:
        Flat all-ones mask with shape (total_gates,)
    """
    total_gates = get_total_gates(layer_sizes)
    return jp.ones(total_gates, dtype=jp.float32)


def _p_fault_from_fraction(target_fraction: float, n_steps: int) -> float:
    """Compute per-gate-per-step failure probability for a target damaged fraction.

    Inverts:  frac = 1 - (1 - p)^L   →   p = 1 - (1 - frac)^(1/L)

    Args:
        target_fraction: Desired fraction of eligible gates damaged after n_steps.
        n_steps: Number of optimisation steps (circuit lifetime).

    Returns:
        Per-gate-per-step failure probability.
    """
    if target_fraction <= 0.0 or n_steps <= 0:
        return 0.0
    frac = min(target_fraction, 1.0)
    return float(1.0 - (1.0 - frac) ** (1.0 / n_steps))


def compute_damage_params(cfg, layer_sizes, log=None) -> dict:
    """Derive all damage parameters from ``cfg.damage.target_damage_fraction``.

    Returns a dict with:
        enabled                   - master switch
        target_fraction           - the configured fraction (echo)
        n_eligible                - number of hidden-layer gates
        p_fault_train             - p_fault tuned for pool.expected_updates steps
        p_fault_eval              - p_fault tuned for eval.inner_steps steps
        n_damage_steps            - number of discrete damage volleys in eval
        knockouts_per_event       - gates per volley (auto or explicit)
        faulty_logit_value        - logit value for damaged gates
        permanent                 - permanence setting (bool | "random")
        p_fault_onset_fraction    - fraction of steps before probabilistic damage starts
        p_fault_onset_step_train  - absolute onset step for training
        p_fault_onset_step_eval   - absolute onset step for evaluation
        compute_no_repair_baseline - whether to compute no-repair baseline metrics
    """
    onset_frac = float(cfg.damage.get("p_fault_onset_fraction", 0.0))
    train_steps = int(cfg.pool.expected_updates)
    eval_steps = int(cfg.eval.inner_steps)

    result = {
        "enabled": bool(cfg.damage.enabled),
        "target_fraction": 0.0,
        "n_eligible": 0,
        "p_fault_train": None,
        "p_fault_eval": None,
        "n_damage_steps": int(cfg.damage.get("n_damage_steps", 0)),
        "knockouts_per_event": 0,
        "faulty_logit_value": float(cfg.damage.get("faulty_logit_value", -10.0)),
        "permanent": cfg.damage.get("permanent", True),
        "p_fault_onset_fraction": onset_frac,
        "p_fault_onset_step_train": int(onset_frac * train_steps),
        "p_fault_onset_step_eval": int(onset_frac * eval_steps),
        "compute_no_repair_baseline": bool(cfg.damage.get("compute_no_repair_baseline", False)),
    }

    # if not result["enabled"]:
    #     if log:
    #         log.info("Damage system disabled")
    #     return result

    n_eligible = count_eligible_gates(layer_sizes)
    result["n_eligible"] = n_eligible
    if n_eligible <= 0:
        if log:
            log.warning("No eligible gates for damage (no hidden layers?)")
        return result

    target_frac = float(cfg.damage.get("target_damage_fraction", 0.0))
    result["target_fraction"] = target_frac
    if target_frac <= 0.0:
        if log:
            log.info("target_damage_fraction <= 0, damage effectively disabled")
        return result

    # --- p_fault (probabilistic) -----------------------------------------
    explicit_p = cfg.damage.get("p_fault")
    if explicit_p is not None:
        result["p_fault_train"] = float(explicit_p)
        # Re-derive eval p_fault from target fraction (not from the override)
        result["p_fault_eval"] = _p_fault_from_fraction(target_frac, eval_steps)
        if log:
            log.info(f"Using explicit p_fault_train = {result['p_fault_train']:.2e}")
    else:
        result["p_fault_train"] = _p_fault_from_fraction(target_frac, train_steps)
        result["p_fault_eval"] = _p_fault_from_fraction(target_frac, eval_steps)

    # --- Bursty background damage ("solar events", training only) --------
    # Doubly-stochastic p_fault: a quiet base rate punctuated by short windows
    # of elevated rate. The single ``target_damage_fraction`` budget is split
    # by ``damage_share``: quiet accrual keeps (1-share), bursts deliver share.
    # Expected gates per burst ≈ n_eligible * p_burst * length — a SOFT volley,
    # easing the eval shotgun into training while keeping the literal
    # K-gates-at-once event OOD. Eval p_fault stays flat (comparability).
    burst_cfg = cfg.damage.get("burst", None)
    burst_on = burst_cfg is not None and bool(burst_cfg.get("enabled", False))
    result["burst"] = {"enabled": False}
    if burst_on and target_frac > 0.0:
        b_len = max(1, int(burst_cfg.get("length", 4)))
        b_per_life = float(burst_cfg.get("per_lifecycle", 2.0))
        b_share = min(max(float(burst_cfg.get("damage_share", 0.5)), 0.0), 1.0)
        quiet_frac = (1.0 - b_share) * target_frac
        burst_frac = b_share * target_frac
        expected_burst_ticks = max(b_per_life * b_len, 1e-9)
        p_quiet = _p_fault_from_fraction(quiet_frac, train_steps) if quiet_frac > 0 else 0.0
        p_burst = _p_fault_from_fraction(burst_frac, expected_burst_ticks)
        # The quiet rate REPLACES the flat train rate; the schedule swaps in
        # p_burst inside windows (see run_model_scan_with_loss).
        result["p_fault_train"] = p_quiet
        result["burst"] = {
            "enabled": True,
            "p_fault_burst_train": float(p_burst),
            "burst_start_rate_train": float(b_per_life / train_steps),
            "length": b_len,
            "per_lifecycle": b_per_life,
            "damage_share": b_share,
            "expected_gates_per_burst": float(p_burst * b_len * n_eligible),
        }
        if log:
            log.info(
                f"Burst damage params (share {b_share:.0%} of {target_frac:.1%} budget):\n"
                f"  p_quiet  = {p_quiet:.2e}/tick   p_burst = {p_burst:.2e}/tick\n"
                f"  windows: {b_per_life:g} per lifecycle x {b_len} ticks "
                f"(start rate {result['burst']['burst_start_rate_train']:.2e}/tick)\n"
                f"  expected gates per burst ~ "
                f"{result['burst']['expected_gates_per_burst']:.1f} of {n_eligible}"
            )

    # --- Discrete knockouts (shotgun) ------------------------------------
    n_dmg = result["n_damage_steps"]
    explicit_ko = cfg.damage.get("knockouts_per_event")
    if explicit_ko is not None:
        result["knockouts_per_event"] = int(explicit_ko)
    elif n_dmg > 0:
        import math

        result["knockouts_per_event"] = max(1, math.ceil(target_frac * n_eligible / n_dmg))
    else:
        result["knockouts_per_event"] = 0

    # --- Logging ---------------------------------------------------------
    if log:
        onset_msg = ""
        if onset_frac > 0:
            onset_msg = (
                f"\n  onset: {onset_frac:.0%} of steps "
                f"(train={result['p_fault_onset_step_train']}, "
                f"eval={result['p_fault_onset_step_eval']})"
            )
        baseline_msg = ""
        if result["compute_no_repair_baseline"]:
            baseline_msg = "\n  no-repair baseline: enabled"
        log.info(
            f"Damage params (target {target_frac:.1%} of {n_eligible} gates):\n"
            f"  p_fault_train  = {result['p_fault_train']:.2e}  ({train_steps} steps)\n"
            f"  p_fault_eval   = {result['p_fault_eval']:.2e}  ({eval_steps} steps)\n"
            f"  discrete: {n_dmg} volleys x {result['knockouts_per_event']} knockouts"
            f"{onset_msg}{baseline_msg}"
        )

    if not result["enabled"]:
        if log:
            log.info("Damage system disabled: setting p_fault_train to None")

        # Training will use p_fault_train
        result["p_fault_train"] = None
        # We still leave p_fault_eval as is, to plot damaged/undamaged performance

    return result


# =============================================================================
# Probabilistic Wire Shuffle Functions
# =============================================================================


def apply_probabilistic_wire_shuffle(
    key: jax.random.PRNGKey,
    wires: list[jp.ndarray],
    layer_sizes: list[tuple[int, int]] | tuple[tuple[int, int], ...],
    arity: int,
    p_shuffle,
    shuffle_fraction: float = 1.0,
) -> list[jp.ndarray]:
    """
    Probabilistic wire shuffle: with probability ``p_shuffle``, perturb the
    entire circuit's wiring this step.

    Circuit-wide: a single coin flip per call decides whether ANY shuffling
    happens. When triggered:

    - ``shuffle_fraction == 1.0`` (default): regenerate all wires from scratch
      via ``gen_wires`` (matches the live demo's Regime III OOD test).
    - ``shuffle_fraction < 1.0``: reassign that fraction of connections per
      layer via :func:`shuffle_wires` (Bernoulli per entry).

    This is the per-step training analog of the discrete ``wire_shuffle_steps``
    used in evaluation/visualization, and the wire-level analog of
    :func:`apply_probabilistic_gate_failure` (which is per-gate).

    Args:
        key: Random key for both the coin flip and the candidate wires.
        wires: List of wire arrays, one per layer (excluding input).
        layer_sizes: Circuit topology including the input layer.
        arity: Number of inputs per gate.
        p_shuffle: Per-step shuffle probability (static float OR traced
            scalar — both work).
        shuffle_fraction: Static float controlling shuffle flavor (see above).

    Returns:
        New list of wire arrays (same shapes). Under JIT, both branches are
        always computed and selected via ``jp.where(do_shuffle, ...)``.
    """
    coin_key, shuffle_key = jax.random.split(key)
    do_shuffle = jax.random.uniform(coin_key) < p_shuffle

    if shuffle_fraction < 1.0:
        candidate = _shuffle_wires_partial(
            shuffle_key, wires, layer_sizes, fraction=shuffle_fraction
        )
    else:
        layer_keys = jax.random.split(shuffle_key, len(wires))
        candidate = []
        in_n = layer_sizes[0][0]
        for i, (out_n, group_size) in enumerate(layer_sizes[1:]):
            candidate.append(gen_wires(layer_keys[i], in_n, out_n, arity, group_size))
            in_n = out_n

    return [jp.where(do_shuffle, c, w) for c, w in zip(candidate, wires, strict=True)]


def _p_shuffle_from_target(target_shuffles: float, n_steps: int) -> float:
    """Compute per-step shuffle probability for a target expected count.

    For a circuit-wide Bernoulli with probability ``p`` over ``L`` steps,
    the expected number of shuffle events is ``E[N] = L * p``. So::

        p = target_shuffles / n_steps

    Clamped to ``[0, 1]``.

    Args:
        target_shuffles: Expected number of shuffle events per circuit lifetime.
        n_steps: Lifetime in optimization steps.

    Returns:
        Per-step shuffle probability.
    """
    if target_shuffles <= 0.0 or n_steps <= 0:
        return 0.0
    return float(min(target_shuffles / n_steps, 1.0))


def compute_shuffle_params(cfg, log=None) -> dict:
    """Derive probabilistic wire-shuffle parameters from
    ``cfg.shuffle.target_shuffles_per_lifecycle``.

    Mirrors :func:`compute_damage_params`. Returns a dict with::

        enabled                    - master switch
        target_shuffles            - configured target (echo)
        p_shuffle_train            - per-step shuffle prob tuned for pool.expected_updates
        p_shuffle_eval             - per-step shuffle prob tuned for eval.inner_steps
        shuffle_fraction           - 1.0 = full reshuffle, <1.0 = partial
        p_shuffle_onset_fraction   - fraction of steps before shuffles start
        p_shuffle_onset_step_train - absolute onset step for training
        p_shuffle_onset_step_eval  - absolute onset step for evaluation
    """
    onset_frac = float(cfg.shuffle.get("p_shuffle_onset_fraction", 0.0))
    train_steps = int(cfg.pool.expected_updates)
    eval_steps = int(cfg.eval.inner_steps)

    result = {
        "enabled": bool(cfg.shuffle.get("enabled", False)),
        "target_shuffles": 0.0,
        "p_shuffle_train": None,
        "p_shuffle_eval": None,
        "shuffle_fraction": float(cfg.shuffle.get("shuffle_fraction", 1.0)),
        "p_shuffle_onset_fraction": onset_frac,
        "p_shuffle_onset_step_train": int(onset_frac * train_steps),
        "p_shuffle_onset_step_eval": int(onset_frac * eval_steps),
    }

    target = float(cfg.shuffle.get("target_shuffles_per_lifecycle", 0.0))
    result["target_shuffles"] = target
    if target <= 0.0:
        if log:
            log.info("target_shuffles_per_lifecycle <= 0, shuffle effectively disabled")
        return result

    explicit_p = cfg.shuffle.get("p_shuffle")
    if explicit_p is not None:
        result["p_shuffle_train"] = float(explicit_p)
        # Re-derive eval p_shuffle from target (not from the override)
        result["p_shuffle_eval"] = _p_shuffle_from_target(target, eval_steps)
        if log:
            log.info(f"Using explicit p_shuffle_train = {result['p_shuffle_train']:.2e}")
    else:
        result["p_shuffle_train"] = _p_shuffle_from_target(target, train_steps)
        result["p_shuffle_eval"] = _p_shuffle_from_target(target, eval_steps)

    if log:
        onset_msg = ""
        if onset_frac > 0:
            onset_msg = (
                f"\n  onset: {onset_frac:.0%} of steps "
                f"(train={result['p_shuffle_onset_step_train']}, "
                f"eval={result['p_shuffle_onset_step_eval']})"
            )
        log.info(
            f"Shuffle params (target {target} shuffles/lifecycle, "
            f"fraction={result['shuffle_fraction']}):\n"
            f"  p_shuffle_train = {result['p_shuffle_train']:.2e}  ({train_steps} steps)\n"
            f"  p_shuffle_eval  = {result['p_shuffle_eval']:.2e}  ({eval_steps} steps)"
            f"{onset_msg}"
        )

    if not result["enabled"]:
        if log:
            log.info("Shuffle system disabled: setting p_shuffle_train to None")
        result["p_shuffle_train"] = None
        # Leave p_shuffle_eval as is, for symmetry/future eval usage

    return result


# --- Legacy wrapper (kept for notebook compatibility) ---------------------
def process_probabilistic_damage_configuration(cfg, layer_sizes, log=None):
    """Compute p_fault for training.  Delegates to ``compute_damage_params``."""
    params = compute_damage_params(cfg, layer_sizes, log)
    return params["p_fault_train"]
