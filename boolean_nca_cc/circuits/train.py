"""
Training Module for Differentiable Boolean Circuits

This module contains functions for training boolean circuits using gradient-based
optimization. It provides utilities for converting between representations,
computing loss, and performing optimization steps.
"""

import jax
import jax.numpy as jp
import numpy as np
from collections import namedtuple
from boolean_nca_cc.circuits.model import run_circuit
import optax



def unpack(x, bit_n=8):
    """
    Unpack an integer into its constituent bits.

    Args:
        x: Input tensor of integers
        bit_n: Number of bits to unpack

    Returns:
        Float tensor with unpacked bits (0.0 or 1.0)
    """
    return jp.float32((x[..., None] >> np.r_[:bit_n]) & 1)


@jax.jit
def res2loss(res, power=4):
    """
    Compute loss from residuals using L4 norm.

    The L4 norm (squared L2) provides stronger gradients for large errors.

    Args:
        res: Residual tensor (difference between predicted and target)

    Returns:
        Loss value (scalar)
    """
    return jp.sum(jp.power(jp.abs(res), power))


@jax.jit
def binary_cross_entropy(y_pred, y_true):
    """
    Compute binary cross-entropy loss.

    This loss function is often more appropriate for binary classification tasks
    and can provide better gradients when outputs are interpreted as probabilities.

    Args:
        y_pred: Predicted outputs (probabilities in [0,1])
        y_true: Target outputs (typically 0 or 1)

    Returns:
        Loss value (scalar)
    """
    # Using JAX's built-in logistic loss which is optimized and numerically stable
    # For binary classification, this is equivalent to binary cross-entropy
    return jp.sum(
        optax.sigmoid_binary_cross_entropy(
            logits=jp.log(
                jp.clip(y_pred, 1e-7, 1.0) / jp.clip(1.0 - y_pred, 1e-7, 1.0)
            ),
            labels=y_true,
        )
    )


@jax.jit
def compute_accuracy(y_pred, y_true):
    """
    Compute the accuracy of predicted bits compared to true bits.

    Accuracy is defined as the proportion of bits that are correctly predicted.
    Predictions are rounded to the nearest integer (0 or 1).

    Args:
        y_pred: Predicted outputs (probabilities in [0,1])
        y_true: Target outputs (typically 0 or 1)

    Returns:
        Accuracy value (scalar, between 0 and 1)
    """
    # Round predictions to nearest binary value
    y_pred_rounded = jp.round(y_pred)
    # Check element-wise equality
    correct_predictions = jp.equal(y_pred_rounded, y_true)
    # Compute mean accuracy
    accuracy = jp.mean(correct_predictions)
    return accuracy


@jax.jit
def compute_full_map_accuracy(y_pred, y_true):
    """
    Compute the full-map accuracy: proportion of samples where ALL bits are correct.
    
    This is stricter than regular accuracy which measures bit-wise correctness.
    A sample is considered correct only if every bit matches exactly.
    
    Args:
        y_pred: Predicted outputs (probabilities in [0,1]), shape [n_samples, n_bits]
        y_true: Target outputs (typically 0 or 1), shape [n_samples, n_bits]
    
    Returns:
        Full-map accuracy value (scalar, between 0 and 1)
    """
    # Round predictions to nearest binary value
    y_pred_rounded = jp.round(y_pred)
    # Check element-wise equality for each sample
    correct_bits = jp.equal(y_pred_rounded, y_true)  # Shape: [n_samples, n_bits]
    # Check if ALL bits are correct for each sample (reduce along bit dimension)
    fully_correct_samples = jp.all(correct_bits, axis=1)  # Shape: [n_samples]
    # Compute proportion of fully correct samples
    full_map_accuracy = jp.mean(fully_correct_samples)
    return full_map_accuracy


# Define loss functions for both types (L4 and BCE)
def loss_f_l4(logits, wires, x, y0, gate_mask=None):
    """L4 loss function variant (for JIT compilation)"""
    act = run_circuit(logits, wires, x, gate_mask=gate_mask)
    hard_act = run_circuit(logits, wires, x, gate_mask=gate_mask, hard=True)
    y = act[-1]
    hard_y = hard_act[-1]
    res = y - y0
    hard_res = hard_y - y0
    loss = res2loss(res)
    hard_loss = res2loss(hard_res)
    accuracy = compute_accuracy(y, y0)
    hard_accuracy = compute_accuracy(hard_y, y0)
    return loss, dict(
        act=act,
        accuracy=accuracy,
        hard_loss=hard_loss,
        hard_accuracy=hard_accuracy,
        res=res,
        hard_res=hard_res,
    )


def loss_f_bce(logits, wires, x, y0, gate_mask=None):
    """BCE loss function variant (for JIT compilation)"""
    act = run_circuit(logits, wires, x, gate_mask=gate_mask)
    hard_act = run_circuit(logits, wires, x, gate_mask=gate_mask, hard=True)
    y = act[-1]
    hard_y = hard_act[-1]
    loss = binary_cross_entropy(y, y0)
    hard_loss = binary_cross_entropy(hard_y, y0)
    accuracy = compute_accuracy(y, y0)
    hard_accuracy = compute_accuracy(hard_y, y0)
    return loss, dict(
        act=act,
        accuracy=accuracy,
        hard_loss=hard_loss,
        hard_accuracy=hard_accuracy,
        res=y - y0,
        hard_res=hard_y - y0,
    )


# Pre-compile gradient functions for both loss types (with gate_mask support)
grad_loss_f_l4 = jax.jit(jax.value_and_grad(loss_f_l4, has_aux=True))
grad_loss_f_bce = jax.jit(jax.value_and_grad(loss_f_bce, has_aux=True))


# Knockout-related functions
def create_gate_mask_from_knockout_pattern(knockout_pattern, layer_sizes):
    """
    Convert a knockout pattern to a gate_mask compatible with run_circuit.
    
    Args:
        knockout_pattern: 1D boolean array where True = knocked out, False = active
        layer_sizes: List of (total_gates, group_size) for each layer
        
    Returns:
        List of gate masks for each layer, where 1.0 = active, 0.0 = knocked out
    """
    gate_masks = []
    current_idx = 0
    
    for layer_idx, (total_gates, group_size) in enumerate(layer_sizes):
        layer_end = current_idx + total_gates
        
        # Extract the knockout pattern for this layer
        layer_knockout = knockout_pattern[current_idx:layer_end]
        
        # Convert boolean to float (False -> 1.0, True -> 0.0)
        # This inverts the knockout pattern since gate_mask uses opposite convention
        layer_mask = jp.where(layer_knockout, 0.0, 1.0)
        
        gate_masks.append(layer_mask)
        
        current_idx = layer_end
    
    return gate_masks


def apply_reversible_bias_to_logits(logits, knockout_pattern, layer_sizes, bias=-10.0):
    """
    Apply one-shot bias to damaged gate logits for reversible damage mode.
    
    This zeros LUT outputs (via sigmoid of biased logits) while allowing recovery
    through normal gradient updates, analogous to self-attention reversible mode.
    
    Args:
        logits: List of logit tensors for each GATE layer (excludes input layer),
                each with shape (groups, group_size, 2^arity)
                Note: logits include output layer, but we skip it for consistency
        knockout_pattern: 1D boolean array where True = knocked out, False = active
                         Includes input layer at the start, but output layer is never True
        layer_sizes: List of (total_gates, group_size) for each layer (includes input layer)
        bias: Bias value to apply to damaged gates (default -10.0, zeros sigmoid output)
        
    Returns:
        List of logit tensors with bias applied to damaged gates
    """
    biased_logits = []
    
    # Skip input layer (layer_sizes[0]) - knockout pattern includes it but logits don't
    current_idx = layer_sizes[0][0]  # Start after input layer
    
    # Identify output layer index (last layer in layer_sizes)
    output_layer_idx = len(layer_sizes) - 1
    
    # Iterate over gate layers only (logits correspond to layer_sizes[1:])
    # Skip output layer for consistency with knockout pattern generation
    for logit_layer_idx, (total_gates, group_size) in enumerate(layer_sizes[1:]):
        # Skip output layer - never apply bias to output nodes
        layer_idx_in_sizes = logit_layer_idx + 1  # +1 because we skipped input layer
        if layer_idx_in_sizes == output_layer_idx:
            # Output layer: pass through unchanged (knockout pattern never has True for outputs)
            biased_logits.append(logits[logit_layer_idx])
            current_idx += total_gates
            continue
        
        layer_end = current_idx + total_gates
        
        # Extract the knockout pattern for this gate layer (flat: [total_gates])
        layer_knockout = knockout_pattern[current_idx:layer_end]
        layer_logits = logits[logit_layer_idx]
        
        # Logits are grouped: shape is (groups, group_size, 2^arity)
        # where groups = total_gates // group_size
        groups = total_gates // group_size
        
        # Reshape knockout pattern to match grouped structure: (total_gates,) -> (groups, group_size)
        layer_knockout_grouped = layer_knockout.reshape(groups, group_size)
        
        # Expand to match logit dimensions: (groups, group_size) -> (groups, group_size, 1)
        # This broadcasts with (groups, group_size, 2^arity)
        bias_mask = layer_knockout_grouped[:, :, None] * bias
        
        # Apply bias where True (knocked out)
        biased_layer = layer_logits + bias_mask
        
        biased_logits.append(biased_layer)
        current_idx = layer_end
    
    return biased_logits



# Function dispatcher for loss computation (deprecated - use loss functions directly)
def loss_f(logits, wires, x, y0, loss_type="l4", gate_mask=None):
    """
    Compute loss for a circuit given input and target output.

    Args:
        logits: List of logits for each layer
        wires: List of wire connection patterns
        x: Input tensor
        y0: Target output tensor
        loss_type: Type of loss to use ('l4' for L4 norm or 'bce' for binary cross-entropy)
        gate_mask: Optional gate mask for knockout experiments

    Returns:
        Tuple of (loss_value, auxiliary_dict) where auxiliary_dict contains
        intermediate activations and accuracy
    """
    if loss_type == "bce":
        return loss_f_bce(logits, wires, x, y0, gate_mask)
    else:  # Default to L4 norm
        return loss_f_l4(logits, wires, x, y0, gate_mask)


# Define a named tuple for training state to improve code clarity
TrainState = namedtuple("TrainState", "params opt_state")


# Remove JIT from this function since it has a non-tensor argument (opt)
def update_params(grad, opt_state, opt, logits):
    """Parameter update function"""
    upd, new_opt_state = opt.update(grad, opt_state, logits)
    new_logits = optax.apply_updates(logits, upd)
    return new_logits, new_opt_state


def train_step(
    state,
    opt,
    wires,
    x,
    y0,
    loss_type="l4",
    do_train=True,
    knockout_pattern=None,
    layer_sizes=None,
    damage_behavior="permanent",
    reversible_bias=-10.0,
    step_count=0,
    apply_damage_now=None,
):
    """
    Unified training step that handles both masked and unmasked training.

    Args:
        state: TrainState containing current parameters and optimizer state
        opt: Optax optimizer
        wires: Wiring configuration for the circuit
        x: Input batch
        y0: Target output batch
        loss_type: Type of loss to use ('l4' or 'bce')
        do_train: Whether to perform training or just evaluation
        knockout_pattern: Optional knockout pattern (if None, no masking)
        layer_sizes: Required if knockout_pattern is provided
        damage_behavior: "permanent" (use gate_mask) or "reversible" (use logit bias)
        reversible_bias: Bias value for reversible mode (default -10.0)
        step_count: Current training step (for reversible mode one-shot bias)

    Returns:
        Tuple of (loss_value, aux_dict, new_state) with updated parameters
    """
    logits, opt_state = state

    # Apply reversible bias when explicitly requested (one-shot injection)
    # Fallback to legacy behavior (step_count == 0) if apply_damage_now is not provided.
    should_apply_damage = (
        damage_behavior == "reversible"
        and knockout_pattern is not None
        and (
            apply_damage_now
            if apply_damage_now is not None
            else step_count == 0
        )
    )
    if should_apply_damage:
        logits = apply_reversible_bias_to_logits(logits, knockout_pattern, layer_sizes, reversible_bias)

    # Create gate_mask only for permanent mode
    gate_mask = None
    if knockout_pattern is not None and damage_behavior == "permanent":
        gate_mask = create_gate_mask_from_knockout_pattern(knockout_pattern, layer_sizes)

    if do_train:
        # Use pre-compiled gradient functions for efficiency
        if loss_type == "bce":
            (loss, aux), grad = grad_loss_f_bce(logits, wires, x, y0, gate_mask)
        else:  # Default to L4 norm
            (loss, aux), grad = grad_loss_f_l4(logits, wires, x, y0, gate_mask)
        
        # Update parameters
        new_logits, new_opt_state = update_params(grad, opt_state, opt, logits)

    else:
        # Evaluation logic
        if loss_type == "bce":
            loss, aux = loss_f_bce(logits, wires, x, y0, gate_mask)
        else:
            loss, aux = loss_f_l4(logits, wires, x, y0, gate_mask)
        new_logits = logits
        new_opt_state = opt_state

    return (loss, aux, TrainState(new_logits, new_opt_state))
