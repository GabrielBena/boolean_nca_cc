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
import matplotlib.pyplot as plt
from functools import partial


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


# Define loss functions for both types (L4 and BCE)
def loss_f_l4(logits, wires, x, y0):
    """L4 loss function variant (for JIT compilation)"""
    act = run_circuit(logits, wires, x)
    hard_act = run_circuit(logits, wires, x, hard=True)
    y = act[-1]
    hard_y = hard_act[-1]
    res = y - y0
    hard_res = hard_y - y0
    loss = res2loss(res)
    hard_loss = res2loss(hard_res)
    accuracy = compute_accuracy(y, y0)
    hard_accuracy = compute_accuracy(hard_y, y0)
    err_mask = hard_act[-1] != y0

    return loss, dict(
        act=act,
        accuracy=accuracy,
        hard_loss=hard_loss,
        hard_accuracy=hard_accuracy,
        err_mask=err_mask
    )


def loss_f_bce(logits, wires, x, y0):
    """BCE loss function variant (for JIT compilation)"""
    act = run_circuit(logits, wires, x)
    hard_act = run_circuit(logits, wires, x, hard=True)
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
    )


# Pre-compile gradient functions for both loss types
grad_loss_f_l4 = jax.jit(jax.value_and_grad(loss_f_l4, has_aux=True))
grad_loss_f_bce = jax.jit(jax.value_and_grad(loss_f_bce, has_aux=True))


# Function dispatcher for loss computation (not used in training loop)
def loss_f(logits, wires, x, y0, loss_type="l4"):
    """
    Compute loss for a circuit given input and target output.

    Args:
        logits: List of logits for each layer
        wires: List of wire connection patterns
        x: Input tensor
        y0: Target output tensor
        loss_type: Type of loss to use ('l4' for L4 norm or 'bce' for binary cross-entropy)

    Returns:
        Tuple of (loss_value, auxiliary_dict) where auxiliary_dict contains
        intermediate activations and accuracy
    """
    if loss_type == "bce":
        return loss_f_bce(logits, wires, x, y0)
    else:  # Default to L4 norm
        return loss_f_l4(logits, wires, x, y0)


# Define a named tuple for training state to improve code clarity
TrainState = namedtuple("TrainState", "params opt_state")


# Remove JIT from this function since it has a non-tensor argument (opt)
def update_params(grad, opt_state, opt, logits):
    """Parameter update function"""
    upd, new_opt_state = opt.update(grad, opt_state, logits)
    new_logits = optax.apply_updates(logits, upd)
    return new_logits, new_opt_state


def train_step(state, opt, wires, x, y0, loss_type="l4", do_train=True):
    """
    Perform a single training step.

    Args:
        state: TrainState containing current parameters and optimizer state
        opt: Optax optimizer
        wires: Wiring configuration for the circuit
        x: Input batch
        y0: Target output batch
        loss_type: Type of loss to use ('l4' or 'bce')

    Returns:
        Tuple of (loss_value, accuracy, new_state) with updated parameters
    """
    logits, opt_state = state

    # Use pre-compiled gradient function based on loss type
    if do_train:
        if loss_type == "bce":
            (loss, aux), grad = grad_loss_f_bce(logits, wires, x, y0)
        else:  # Default to L4 norm
            (loss, aux), grad = grad_loss_f_l4(logits, wires, x, y0)

        # Update parameters (without JIT since optimizer is a function)
        new_logits, new_opt_state = update_params(grad, opt_state, opt, logits)

    else:
        loss, aux = loss_f(logits, wires, x, y0, loss_type)
        new_logits = logits
        new_opt_state = opt_state

    # Return loss, accuracy, and new state
    return (
        loss,
        aux,
        TrainState(new_logits, new_opt_state),
    )
