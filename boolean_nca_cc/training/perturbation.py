"""
Circuit perturbation utilities for boolean circuit NCA.

This module provides functions for perturbing circuits during training
to improve robustness and generalization.
"""

import jax
import jax.numpy as jp
from typing import List, Tuple, Dict, Any


def shuffle_wires(
    rng: jax.random.PRNGKey, wires: List, logits: List
) -> Tuple[List, List]:
    """
    Randomly shuffle some wire connections in the circuit.

    Args:
        rng: Random key
        wires: List of wire connections for each layer
        logits: List of logits for each layer

    Returns:
        Tuple of (shuffled_wires, logits)
    """
    # Make a copy of wires to avoid modifying the original
    shuffled_wires = [w.copy() for w in wires]

    # For each layer (except input layer), shuffle some connections
    for i in range(1, len(wires)):
        layer_rng, rng = jax.random.split(rng)

        # Get the shape of the current layer's wires
        layer_shape = wires[i].shape

        # Decide which connections to shuffle (random 10%)
        shuffle_mask = jax.random.uniform(layer_rng, layer_shape) < 0.1

        if jp.any(shuffle_mask):
            # For selected connections, randomly reassign them to other nodes in previous layer
            prev_layer_size = wires[i - 1].shape[0]
            new_connections = jax.random.randint(
                layer_rng, shape=jp.sum(shuffle_mask), minval=0, maxval=prev_layer_size
            )

            # Update the connections
            shuffled_wires[i] = shuffled_wires[i].at[shuffle_mask].set(new_connections)

    return shuffled_wires, logits


def perturb_logits(
    rng: jax.random.PRNGKey, wires: List, logits: List, noise_scale: float = 0.05
) -> Tuple[List, List]:
    """
    Add small noise perturbations to logits.

    Args:
        rng: Random key
        wires: List of wire connections for each layer
        logits: List of logits for each layer
        noise_scale: Scale of the noise to add

    Returns:
        Tuple of (wires, perturbed_logits)
    """
    # Make a copy of logits to avoid modifying the original
    perturbed_logits = []

    for i, logit in enumerate(logits):
        layer_rng, rng = jax.random.split(rng)

        # Add small Gaussian noise to the logits
        noise = jax.random.normal(layer_rng, logit.shape) * noise_scale
        perturbed_logits.append(logit + noise)

    return wires, perturbed_logits


def apply_damage(
    rng: jax.random.PRNGKey, wires: List, logits: List, damage_prob: float = 0.05
) -> Tuple[List, List]:
    """
    Apply random damage to the circuit by zeroing out some logits.

    Args:
        rng: Random key
        wires: List of wire connections for each layer
        logits: List of logits for each layer
        damage_prob: Probability of damaging each logit

    Returns:
        Tuple of (wires, damaged_logits)
    """
    # Make a copy of logits to avoid modifying the original
    damaged_logits = []

    for i, logit in enumerate(logits):
        layer_rng, rng = jax.random.split(rng)

        # Generate damage mask
        damage_mask = jax.random.uniform(layer_rng, logit.shape) < damage_prob

        # Zero out damaged logits
        damaged = jp.where(damage_mask, jp.zeros_like(logit), logit)
        damaged_logits.append(damaged)

    return wires, damaged_logits


def zero_luts(
    rng: jax.random.PRNGKey, wires: List, logits: List, lut_damage_prob: float = 0.05
) -> Tuple[List, List]:
    """
    Apply random damage to the circuit by zeroing out entire LUTs.

    This means that for a selected LUT, all its associated logits will be set to zero.
    A LUT typically corresponds to a gate, and its logits are often represented
    as a row in the layer's logit tensor.

    Args:
        rng: Random key
        wires: List of wire connections for each layer (passed through unchanged)
        logits: List of logit tensors for each layer. Each tensor is expected
                to have its first dimension correspond to the number of LUTs.
        lut_damage_prob: Probability of damaging each LUT in a layer.

    Returns:
        Tuple of (wires, damaged_logits)
    """
    damaged_logits = []

    for logit_layer in logits:
        layer_rng, rng = jax.random.split(rng)

        if logit_layer.ndim == 0: # Should not happen for typical logits
            damaged_logits.append(logit_layer)
            continue
            
        num_luts = logit_layer.shape[0]
        if num_luts == 0:
            damaged_logits.append(logit_layer)
            continue

        # Generate a mask to select which LUTs to damage
        lut_damage_mask = (
            jax.random.uniform(layer_rng, (num_luts,)) < lut_damage_prob
        )

        # Apply damage: set all logits for selected LUTs to 0.0
        # JAX's .at[indices].set(value) broadcasts value to the shape of the slice.
        # Use jp.where for JIT-friendliness with boolean masks
        if logit_layer.ndim > 1 and lut_damage_mask.ndim == 1: # Ensure mask is 1D and logit_layer is at least 2D
            # Reshape lut_damage_mask to be (num_luts, 1, ..., 1) to match logit_layer.ndim
            # For example, if logit_layer is (num_luts, d1, d2), mask becomes (num_luts, 1, 1)
            broadcast_shape_for_mask = (num_luts,) + (1,) * (logit_layer.ndim - 1)
            reshaped_mask = lut_damage_mask.reshape(broadcast_shape_for_mask)
            damaged_layer_logits = jp.where(reshaped_mask, 0.0, logit_layer)
        elif logit_layer.ndim == 1 and lut_damage_mask.ndim == 1: # Both are 1D (e.g. bias term for a layer)
             damaged_layer_logits = jp.where(lut_damage_mask, 0.0, logit_layer)
        else: # Fallback or if shapes are unexpected, keep original (or raise error)
            # This case might need more specific handling if logit_layer can have more diverse shapes
            # For now, let's assume typical LUT structures or simple biases
            damaged_layer_logits = logit_layer 
        damaged_logits.append(damaged_layer_logits)

    return wires, damaged_logits
