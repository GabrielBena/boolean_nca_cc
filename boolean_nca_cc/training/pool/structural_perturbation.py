"""
Structural perturbation utilities for boolean circuit graphs.

This module provides functions to modify graph topology and gate behavior for
different types of interference with boolean circuits:

1. **Gate Knockout (Permanent Structural Damage)**:
   - Gates are permanently "broken" and cannot be updated by the model
   - Logits are set to faulty values that produce zero output
   - Gates are prevented from receiving model updates (knockout mask prevents updates)
   - Represents hardware failure or permanent structural damage
   - Use functions: apply_gate_knockout_to_pool_element, apply_gate_knockout_to_pool_batch

2. **Logits Perturbation (Recoverable Interference)**:
   - Gates have their logits temporarily modified but can still receive updates
   - Can potentially recover through message passing and model updates
   - Represents temporary noise, interference, or transient faults
   - Use functions: apply_logits_perturbation_to_pool_element, apply_logits_perturbation_to_pool_batch

The key difference is that knockout prevents recovery (no model updates), while
perturbation allows recovery (model updates still possible).
"""

from functools import partial

import jax
import jax.numpy as jp
import jraph


def create_reproducible_knockout_pattern(
    key: jax.random.PRNGKey,
    layer_sizes: list[tuple[int, int]],  # (gate_n, group_size) for each layer
    number_knokouts: float,
    input_n: int = 0,
) -> list[jp.ndarray]:
    """
    Create a reproducible knockout pattern for gates in the circuit.

    Returns layered gate masks that match the circuit structure used in run_circuit.

    Args:
        key: Random key for reproducible generation
        layer_sizes: List of (gate_n, group_size) for each gate layer
        number_knokouts: Exact number of knockouts to apply
        input_n: Number of input nodes (never knocked out)

    Returns:
        List of gate masks, one per layer:
        - Input layer: mask of shape (input_n,) with all 1.0 (never knocked out)
        - Gate layers: mask of shape (gate_n,) where 0.0 = knocked out, 1.0 = active

        This structure matches what run_circuit expects for gate_mask parameter.

    Note:
        - Input layer (layer 0) is never knocked out
        - Output layer (last layer) is never knocked out
        - Exactly number_knokouts gates will be knocked out from hidden layers
    """
    # If number_knokouts is 0, return all active masks
    if number_knokouts == 0:
        gate_masks = []
        for gate_n, _group_size in layer_sizes:
            gate_masks.append(jp.ones(gate_n, dtype=jp.float32))
        return gate_masks

    # Identify output layer index (last layer)
    output_layer_idx = len(layer_sizes) - 1

    # Collect all eligible gate indices from hidden layers
    eligible_indices = []
    layer_start_indices = []
    current_idx = 0

    for layer_idx, (gate_n, _group_size) in enumerate(layer_sizes):
        layer_start_indices.append(current_idx)

        # Skip input and output layers - never knock out input or output nodes
        if layer_idx == 0 or layer_idx == output_layer_idx:
            current_idx += gate_n
            continue

        # Add all gate indices for this hidden layer
        layer_indices = jp.arange(current_idx, current_idx + gate_n)
        eligible_indices.append(layer_indices)
        current_idx += gate_n

    # Handle edge case: no eligible gates
    if not eligible_indices:
        gate_masks = []
        for gate_n, _group_size in layer_sizes:
            gate_masks.append(jp.ones(gate_n, dtype=jp.float32))
        return gate_masks

    # Concatenate all eligible indices
    all_eligible_indices = jp.concatenate(eligible_indices)

    # Ensure number_knokouts doesn't exceed total eligible gates
    num_knockouts = min(int(number_knokouts), len(all_eligible_indices))

    # Randomly sample exactly num_knockouts indices
    knockout_indices = jax.random.choice(
        key, all_eligible_indices, shape=(num_knockouts,), replace=False
    )

    # Create layered gate masks
    gate_masks = []
    current_idx = 0

    for _layer_idx, (gate_n, _group_size) in enumerate(layer_sizes):
        # Initialize all gates as active (1.0)
        layer_mask = jp.ones(gate_n, dtype=jp.float32)

        # Find which gates in this layer should be knocked out
        layer_start = current_idx
        layer_end = current_idx + gate_n

        # Get knockout indices that fall within this layer
        layer_knockouts = knockout_indices[
            (knockout_indices >= layer_start) & (knockout_indices < layer_end)
        ]

        # Convert global indices to local layer indices
        local_knockout_indices = layer_knockouts - layer_start

        # Set knocked out gates to 0.0
        if len(local_knockout_indices) > 0:
            layer_mask = layer_mask.at[local_knockout_indices].set(0.0)

        gate_masks.append(layer_mask)
        current_idx += gate_n

    return gate_masks


def create_knockout_vocabulary(
    rng: jax.random.PRNGKey,
    vocabulary_size: int,
    layer_sizes: list[tuple[int, int]],
    number_knokouts: float,
    input_n: int,
) -> list[jp.ndarray]:
    """
    Generates a fixed vocabulary of knockout patterns.

    Args:
        rng: JAX random key.
        vocabulary_size: The number of unique patterns to generate.
        layer_sizes: List of (gate_n, group_size) for each layer.
        number_knokouts: The number of knockouts to apply per pattern.
        input_n: The number of input nodes.

    Returns:
        List of gate mask arrays, one per layer. Each array has shape
        (vocabulary_size, gate_n) where gate_n varies per layer.

        The structure matches what run_circuit expects: a list of masks
        where each mask corresponds to one layer of the circuit.
    """
    pattern_creator_fn = partial(
        create_reproducible_knockout_pattern,
        layer_sizes=layer_sizes,
        number_knokouts=number_knokouts,
        input_n=input_n,
    )

    pattern_keys = jax.random.split(rng, vocabulary_size)
    # knockout_patterns is a list of lists: [pattern_idx][layer_idx] -> gate_mask
    knockout_patterns = jax.vmap(pattern_creator_fn)(pattern_keys)

    # Transpose to get layer-first structure: [layer_idx][pattern_idx] -> gate_mask
    # This matches the expected format for batched operations
    num_layers = len(layer_sizes)
    knockout_vocabulary = []

    for layer_idx in range(num_layers):
        # Stack all patterns for this layer
        layer_masks = jp.stack(
            [knockout_patterns[pattern_idx][layer_idx] for pattern_idx in range(vocabulary_size)]
        )
        knockout_vocabulary.append(layer_masks)

    return knockout_vocabulary


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
    flat_mask: jp.ndarray, layer_sizes: list[tuple[int, int]]
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
        batch_layered_masks: List of mask arrays, one per layer with shape (batch_size, gate_n)

    Returns:
        Batched flat mask array with shape (batch_size, total_gates)
    """
    return jp.concatenate(batch_layered_masks, axis=1)


def batch_flat_to_layered_mask(
    batch_flat_mask: jp.ndarray, layer_sizes: list[tuple[int, int]]
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


def create_faulty_gate_logits(
    original_logits: jp.ndarray, gate_mask: jp.ndarray, faulty_value: float = -10.0
) -> jp.ndarray:
    """
    Set logits of knocked-out gates to faulty values that produce zero output.

    For knocked-out gates, we set all LUT entries to a large negative value so that
    after sigmoid, they become ~0, ensuring the gate always outputs 0 regardless of input.

    Args:
        original_logits: Logits array with shape (group_n, group_size, 2^arity)
        gate_mask: Gate mask with shape (gate_n,) where gate_n = group_n * group_size
        faulty_value: Value to set for knocked-out gate logits (should be large negative)

    Returns:
        Modified logits with faulty values for knocked-out gates
    """
    group_n, group_size, lut_size = original_logits.shape

    # Reshape mask to match logits structure
    mask_reshaped = gate_mask.reshape(group_n, group_size)

    # Expand mask to cover all LUT entries
    mask_expanded = mask_reshaped[..., None]  # Shape: (group_n, group_size, 1)
    mask_expanded = jp.broadcast_to(mask_expanded, original_logits.shape)

    # Apply faulty values where mask is 0 (knocked out)
    faulty_logits = jp.where(
        mask_expanded == 0.0,
        faulty_value,  # Set to faulty value for knocked-out gates
        original_logits,  # Keep original for active gates
    )

    return faulty_logits


def create_perturbed_logits(
    original_logits: jp.ndarray,
    perturbation_mask: jp.ndarray,
    key: jax.random.PRNGKey,
    perturbation_type: str = "noise",
    noise_scale: float = 1.0,
    faulty_value: float = -10.0,
) -> jp.ndarray:
    """
    Apply recoverable perturbations to gate logits.

    Unlike knockout which is permanent structural damage, perturbations are
    temporary interference that can potentially be recovered through message passing.

    Args:
        original_logits: Logits array with shape (group_n, group_size, 2^arity)
        perturbation_mask: Mask with shape (gate_n,) where 1.0 = perturb, 0.0 = leave unchanged
        key: Random key for noise generation
        perturbation_type: Type of perturbation ("noise", "zero", "negative")
        noise_scale: Scale of Gaussian noise to add (for "noise" type)
        faulty_value: Value to set for "negative" perturbation type

    Returns:
        Perturbed logits that can potentially be recovered
    """
    group_n, group_size, lut_size = original_logits.shape

    # Reshape mask to match logits structure
    mask_reshaped = perturbation_mask.reshape(group_n, group_size)

    # Expand mask to cover all LUT entries
    mask_expanded = mask_reshaped[..., None]  # Shape: (group_n, group_size, 1)
    mask_expanded = jp.broadcast_to(mask_expanded, original_logits.shape)

    if perturbation_type == "noise":
        # Add Gaussian noise to perturbed gates
        noise = jax.random.normal(key, original_logits.shape) * noise_scale
        perturbed_logits = jp.where(
            mask_expanded == 1.0,
            original_logits + noise,  # Add noise to perturbed gates
            original_logits,  # Keep original for unperturbed gates
        )
    elif perturbation_type == "zero":
        # Set perturbed gates to zero (recoverable)
        perturbed_logits = jp.where(
            mask_expanded == 1.0,
            0.0,  # Set to zero for perturbed gates
            original_logits,  # Keep original for unperturbed gates
        )
    elif perturbation_type == "negative":
        # Set perturbed gates to negative value (but still recoverable)
        perturbed_logits = jp.where(
            mask_expanded == 1.0,
            faulty_value,  # Set to faulty value for perturbed gates
            original_logits,  # Keep original for unperturbed gates
        )
    else:
        raise ValueError(f"Unknown perturbation_type: {perturbation_type}")

    return perturbed_logits


def apply_gate_knockout_to_pool_element(
    key: jax.random.PRNGKey,
    logits: list[jp.ndarray],
    wires: list[jp.ndarray],
    layer_sizes: list[tuple[int, int]],
    number_knockouts: float,
    faulty_value: float = -10.0,
) -> tuple[list[jp.ndarray], list[jp.ndarray], list[jp.ndarray]]:
    """
    Apply permanent gate knockout to a single pool element.

    Knocked-out gates:
    1. Have their logits set to faulty values (produce zero output)
    2. Are prevented from receiving model updates (permanent structural damage)
    3. Can still send messages but cannot be recovered

    Args:
        key: Random key for knockout generation
        logits: List of logit arrays for each layer
        wires: List of wire arrays for each layer (unchanged)
        layer_sizes: Circuit layer sizes
        number_knockouts: Number of gates to permanently knock out
        faulty_value: Value for knocked-out gate logits

    Returns:
        Tuple of (modified_logits, unchanged_wires, knockout_masks)
        where knockout_masks indicate permanently damaged gates
    """
    # Generate knockout pattern
    knockout_masks = create_reproducible_knockout_pattern(
        key, layer_sizes, number_knockouts, input_n=0
    )

    # Apply faulty logits to knocked-out gates
    modified_logits = []
    for layer_idx, (layer_logits, layer_mask) in enumerate(
        zip(logits, knockout_masks[1:], strict=False)
    ):
        # Skip input layer (no logits), only process gate layers
        faulty_logits = create_faulty_gate_logits(layer_logits, layer_mask, faulty_value)
        modified_logits.append(faulty_logits)

    return modified_logits, wires, knockout_masks


def apply_logits_perturbation_to_pool_element(
    key: jax.random.PRNGKey,
    logits: list[jp.ndarray],
    wires: list[jp.ndarray],
    layer_sizes: list[tuple[int, int]],
    number_perturbations: float,
    perturbation_type: str = "noise",
    noise_scale: float = 1.0,
    faulty_value: float = -10.0,
) -> tuple[list[jp.ndarray], list[jp.ndarray], list[jp.ndarray]]:
    """
    Apply recoverable perturbations to a single pool element.

    Perturbed gates:
    1. Have their logits modified but can still receive model updates
    2. Can potentially recover through message passing
    3. Represent temporary interference rather than permanent damage

    Args:
        key: Random key for perturbation generation
        logits: List of logit arrays for each layer
        wires: List of wire arrays for each layer (unchanged)
        layer_sizes: Circuit layer sizes
        number_perturbations: Number of gates to perturb
        perturbation_type: Type of perturbation to apply
        noise_scale: Scale of noise for "noise" perturbation
        faulty_value: Value for "negative" perturbation

    Returns:
        Tuple of (modified_logits, unchanged_wires, perturbation_masks)
        where perturbation_masks indicate which gates were perturbed
    """
    # Generate perturbation pattern (same structure as knockout but different purpose)
    perturbation_masks = create_reproducible_knockout_pattern(
        key, layer_sizes, number_perturbations, input_n=0
    )

    # Apply perturbations to selected gates
    modified_logits = []
    for layer_idx, (layer_logits, layer_mask) in enumerate(
        zip(logits, perturbation_masks[1:], strict=False)
    ):
        # Skip input layer (no logits), only process gate layers
        perturbation_key = jax.random.fold_in(key, layer_idx)
        perturbed_logits = create_perturbed_logits(
            layer_logits, layer_mask, perturbation_key, perturbation_type, noise_scale, faulty_value
        )
        modified_logits.append(perturbed_logits)

    return modified_logits, wires, perturbation_masks


def apply_gate_knockout_to_pool_batch(
    key: jax.random.PRNGKey,
    batch_logits: list[jp.ndarray],
    batch_wires: list[jp.ndarray],
    layer_sizes: list[tuple[int, int]],
    number_knockouts: float,
    faulty_value: float = -10.0,
) -> tuple[list[jp.ndarray], list[jp.ndarray], list[jp.ndarray]]:
    """
    Apply permanent gate knockout to a batch of pool elements.

    Args:
        key: Random key for knockout generation
        batch_logits: List of batched logit arrays, each with shape (batch_size, ...)
        batch_wires: List of batched wire arrays, each with shape (batch_size, ...)
        layer_sizes: Circuit layer sizes
        number_knockouts: Number of gates to permanently knock out per circuit
        faulty_value: Value for knocked-out gate logits

    Returns:
        Tuple of (modified_batch_logits, unchanged_batch_wires, batch_knockout_masks)
        where batch_knockout_masks indicate permanently damaged gates
    """
    batch_size = batch_logits[0].shape[0]

    # Generate keys for each element in the batch
    keys = jax.random.split(key, batch_size)

    # Extract individual elements from batched arrays
    individual_logits = [
        [batch_logits[layer_idx][batch_idx] for layer_idx in range(len(batch_logits))]
        for batch_idx in range(batch_size)
    ]
    individual_wires = [
        [batch_wires[layer_idx][batch_idx] for layer_idx in range(len(batch_wires))]
        for batch_idx in range(batch_size)
    ]

    # Apply knockouts to each element
    results = [
        apply_gate_knockout_to_pool_element(
            keys[i],
            individual_logits[i],
            individual_wires[i],
            layer_sizes,
            number_knockouts,
            faulty_value,
        )
        for i in range(batch_size)
    ]

    # Reconstruct batched format
    modified_batch_logits = []
    batch_knockout_masks = []

    for layer_idx in range(len(batch_logits)):
        # Stack modified logits for this layer
        layer_logits = jp.stack([results[i][0][layer_idx] for i in range(batch_size)])
        modified_batch_logits.append(layer_logits)

    for layer_idx in range(len(layer_sizes)):
        # Stack knockout masks for this layer
        layer_masks = jp.stack([results[i][2][layer_idx] for i in range(batch_size)])
        batch_knockout_masks.append(layer_masks)

    return modified_batch_logits, batch_wires, batch_knockout_masks


def apply_logits_perturbation_to_pool_batch(
    key: jax.random.PRNGKey,
    batch_logits: list[jp.ndarray],
    batch_wires: list[jp.ndarray],
    layer_sizes: list[tuple[int, int]],
    number_perturbations: float,
    perturbation_type: str = "noise",
    noise_scale: float = 1.0,
    faulty_value: float = -10.0,
) -> tuple[list[jp.ndarray], list[jp.ndarray], list[jp.ndarray]]:
    """
    Apply recoverable perturbations to a batch of pool elements.

    Args:
        key: Random key for perturbation generation
        batch_logits: List of batched logit arrays, each with shape (batch_size, ...)
        batch_wires: List of batched wire arrays, each with shape (batch_size, ...)
        layer_sizes: Circuit layer sizes
        number_perturbations: Number of gates to perturb per circuit
        perturbation_type: Type of perturbation to apply
        noise_scale: Scale of noise for "noise" perturbation
        faulty_value: Value for "negative" perturbation

    Returns:
        Tuple of (modified_batch_logits, unchanged_batch_wires, batch_perturbation_masks)
        where batch_perturbation_masks indicate which gates were perturbed
    """
    batch_size = batch_logits[0].shape[0]

    # Generate keys for each element in the batch
    keys = jax.random.split(key, batch_size)

    # Extract individual elements from batched arrays
    individual_logits = [
        [batch_logits[layer_idx][batch_idx] for layer_idx in range(len(batch_logits))]
        for batch_idx in range(batch_size)
    ]
    individual_wires = [
        [batch_wires[layer_idx][batch_idx] for layer_idx in range(len(batch_wires))]
        for batch_idx in range(batch_size)
    ]

    # Apply perturbations to each element
    results = [
        apply_logits_perturbation_to_pool_element(
            keys[i],
            individual_logits[i],
            individual_wires[i],
            layer_sizes,
            number_perturbations,
            perturbation_type,
            noise_scale,
            faulty_value,
        )
        for i in range(batch_size)
    ]

    # Reconstruct batched format
    modified_batch_logits = []
    batch_perturbation_masks = []

    for layer_idx in range(len(batch_logits)):
        # Stack modified logits for this layer
        layer_logits = jp.stack([results[i][0][layer_idx] for i in range(batch_size)])
        modified_batch_logits.append(layer_logits)

    for layer_idx in range(len(layer_sizes)):
        # Stack perturbation masks for this layer
        layer_masks = jp.stack([results[i][2][layer_idx] for i in range(batch_size)])
        batch_perturbation_masks.append(layer_masks)

    return modified_batch_logits, batch_wires, batch_perturbation_masks


def is_layered_mask(gate_mask) -> bool:
    """
    Check if a gate mask is in layered format.

    Args:
        gate_mask: Gate mask to check

    Returns:
        True if the mask is in layered format (list/tuple of arrays), False otherwise
    """
    return (
        isinstance(gate_mask, list | tuple)
        and len(gate_mask) > 0
        and isinstance(gate_mask[0], jp.ndarray)
    )


def is_flat_mask(gate_mask) -> bool:
    """
    Check if a gate mask is in flat format.

    Args:
        gate_mask: Gate mask to check

    Returns:
        True if the mask is in flat format (1D JAX array), False otherwise
    """
    return isinstance(gate_mask, jp.ndarray) and gate_mask.ndim == 1


def ensure_layered_mask(gate_mask, layer_sizes: list[tuple[int, int]]) -> list[jp.ndarray]:
    """
    Ensure gate mask is in layered format, converting if necessary.

    Args:
        gate_mask: Gate mask in either flat or layered format
        layer_sizes: List of (gate_n, group_size) for each layer

    Returns:
        Gate mask in layered format (list of arrays)

    Raises:
        ValueError: If the gate mask format is invalid
    """
    if gate_mask is None:
        # Return all-active masks for each layer
        return [jp.ones(gate_n, dtype=jp.float32) for gate_n, _ in layer_sizes]

    if is_layered_mask(gate_mask):
        return gate_mask
    elif is_flat_mask(gate_mask):
        return flat_to_layered_mask(gate_mask, layer_sizes)
    else:
        raise ValueError(
            f"Invalid gate mask format. Expected flat JAX array or layered list/tuple, "
            f"got {type(gate_mask)} with shape {getattr(gate_mask, 'shape', 'N/A')}"
        )


def ensure_flat_mask(gate_mask, layer_sizes: list[tuple[int, int]] | None = None) -> jp.ndarray:
    """
    Ensure gate mask is in flat format, converting if necessary.

    Args:
        gate_mask: Gate mask in either flat or layered format
        layer_sizes: List of (gate_n, group_size) for each layer (required if gate_mask is None)

    Returns:
        Gate mask in flat format (1D JAX array)

    Raises:
        ValueError: If the gate mask format is invalid
    """
    if gate_mask is None:
        if layer_sizes is None:
            raise ValueError("layer_sizes required when gate_mask is None")
        # Return all-active flat mask
        total_gates = sum(gate_n for gate_n, _ in layer_sizes)
        return jp.ones(total_gates, dtype=jp.float32)

    if is_flat_mask(gate_mask):
        return gate_mask
    elif is_layered_mask(gate_mask):
        return layered_to_flat_mask(gate_mask)
    else:
        raise ValueError(
            f"Invalid gate mask format. Expected flat JAX array or layered list/tuple, "
            f"got {type(gate_mask)} with shape {getattr(gate_mask, 'shape', 'N/A')}"
        )
