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
import matplotlib.pyplot as plt


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
    y = act[-1]
    res = y - y0
    loss = res2loss(res)
    accuracy = compute_accuracy(y, y0)
    return loss, dict(act=act, accuracy=accuracy)


def loss_f_bce(logits, wires, x, y0):
    """BCE loss function variant (for JIT compilation)"""
    act = run_circuit(logits, wires, x)
    y = act[-1]
    loss = binary_cross_entropy(y, y0)
    accuracy = compute_accuracy(y, y0)
    return loss, dict(act=act, accuracy=accuracy)


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
        Tuple of (loss_value, accuracy, new_state) with updated parameters
    """
    logits, opt_state = state

    # Use pre-compiled gradient function based on loss type
    if loss_type == "bce":
        (loss, aux), grad = grad_loss_f_bce(logits, wires, x, y0)
    else:  # Default to L4 norm
        (loss, aux), grad = grad_loss_f_l4(logits, wires, x, y0)

    # Update parameters (without JIT since optimizer is a function)
    new_logits, new_opt_state = update_params(grad, opt_state, opt, logits)

    # Extract accuracy from auxiliary data
    accuracy = aux["accuracy"]

    # Return loss, accuracy, and new state
    return loss, accuracy, TrainState(new_logits, new_opt_state)


def evaluate_and_visualize(logits, wires, x, y0, title_prefix=""):
    """
    Evaluate the circuit, calculate accuracy, and visualize the results.

    Runs the circuit in 'hard' mode, compares outputs to targets,
    and plots the predicted output, target output, and errors in a vertical layout.

    Args:
        logits: List of logits for each layer (trained parameters).
        wires: List of wire connection patterns for the circuit.
        x: Input data tensor.
        y0: Target output data tensor.
        title_prefix: Optional string to prepend to the plot title.
    """
    # 1. Run the circuit in hard mode to get binary predictions
    acts = run_circuit(logits, wires, x, hard=True)
    y_pred = acts[-1]

    # 2. Calculate accuracy
    accuracy = compute_accuracy(y_pred, y0)

    # 3. Calculate the difference map (errors)
    # Errors will be 1 where prediction != target, 0 otherwise
    errors = jp.not_equal(y_pred, y0).astype(jp.float32)

    # Ensure data is on CPU and converted to NumPy for plotting
    y_pred_np = np.array(y_pred)
    y0_np = np.array(y0)
    errors_np = np.array(errors)

    # 4. Visualization - Vertical Layout
    fig, axes = plt.subplots(
        3, 1, figsize=(20, 6), constrained_layout=True
    )  # 3 rows, 1 column
    cmap = "viridis"  # Or 'gray' or any other binary cmap

    # Plot Predicted Output
    axes[0].imshow(
        y_pred_np.T, cmap=cmap, interpolation="nearest", vmin=0, vmax=1, aspect="auto"
    )
    axes[0].set_title("Predicted Output")
    axes[0].set_xlabel("Batch Index")
    axes[0].set_ylabel("Output Bit Index")

    # Plot Target Output
    axes[1].imshow(
        y0_np.T, cmap=cmap, interpolation="nearest", vmin=0, vmax=1, aspect="auto"
    )
    axes[1].set_title("Target Output")
    axes[1].set_xlabel("Batch Index")
    axes[1].set_ylabel("Output Bit Index")

    # Plot Errors (highlighting incorrect bits)
    cmap_errors = plt.cm.colors.ListedColormap(["lightgray", "red"])
    axes[2].imshow(
        errors_np.T,
        cmap=cmap_errors,
        interpolation="nearest",
        vmin=0,
        vmax=1,
        aspect="auto",
    )
    axes[2].set_title("Errors (Incorrect Bits)")
    axes[2].set_xlabel("Batch Index")
    axes[2].set_ylabel("Output Bit Index")

    # Add overall title with accuracy
    fig.suptitle(f"{title_prefix}Evaluation - Accuracy: {accuracy:.4f}", fontsize=16)

    # Adjust layout
    plt.show()
