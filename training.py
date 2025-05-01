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
from model import run_circuit
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


def res2loss(res):
    """
    Compute loss from residuals using L4 norm.

    The L4 norm (squared L2) provides stronger gradients for large errors.

    Args:
        res: Residual tensor (difference between predicted and target)

    Returns:
        Loss value (scalar)
    """
    return jp.square(jp.square(res)).sum()


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
    # Clip values to avoid log(0) issues
    eps = 1e-7
    y_pred = jp.clip(y_pred, eps, 1.0 - eps)

    # Standard binary cross-entropy formula
    bce = -(y_true * jp.log(y_pred) + (1.0 - y_true) * jp.log(1.0 - y_pred))

    return bce.sum()


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
        intermediate activations
    """
    # Run the circuit to get all activations
    act = run_circuit(logits, wires, x)
    # Extract final layer output
    y = act[-1]

    # Compute loss based on selected loss type
    if loss_type == "bce":
        loss = binary_cross_entropy(y, y0)
    else:  # Default to L4 norm
        res = y - y0
        loss = res2loss(res)

    # Return loss and auxiliary information
    return loss, dict(act=act)


# JIT-compile the gradient computation function for efficiency
def create_grad_loss_f(loss_type="l4"):
    """
    Create a JIT-compiled gradient function for the specified loss type.

    Args:
        loss_type: Type of loss to use ('l4' or 'bce')

    Returns:
        JIT-compiled value_and_grad function
    """

    def loss_fn(logits, wires, x, y0):
        return loss_f(logits, wires, x, y0, loss_type)

    return jax.jit(jax.value_and_grad(loss_fn, has_aux=True))


# Default gradient function with L4 loss
grad_loss_f = create_grad_loss_f("l4")

# Define a named tuple for training state to improve code clarity
TrainState = namedtuple("TrainState", "params opt_state")


def train_step(state, opt, wires, x, y0, loss_type="l4"):
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
        Tuple of (loss_value, new_state) with updated parameters
    """
    logits, opt_state = state

    # Get appropriate gradient function for the loss type
    grad_fn = grad_loss_f if loss_type == "l4" else create_grad_loss_f(loss_type)

    # Compute loss and gradients
    (loss, aux), grad = grad_fn(logits, wires, x, y0)

    # Compute parameter updates using optimizer
    upd, opt_state = opt.update(grad, opt_state, logits)

    # Apply updates to parameters
    logits = optax.apply_updates(logits, upd)

    # Return loss and new state
    return loss, TrainState(logits, opt_state)
