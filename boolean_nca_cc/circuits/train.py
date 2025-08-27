"""
Training Module for Differentiable Boolean Circuits

This module contains functions for training boolean circuits using gradient-based
optimization. It provides utilities for converting between representations,
computing loss, and performing optimization steps.
"""

from collections import namedtuple
from functools import partial

import jax
import jax.numpy as jp
import numpy as np
import optax
from flax import nnx

from boolean_nca_cc.circuits.model import run_circuit


@partial(nnx.jit, static_argnames=["sparsity_type"])
def compute_lut_sparsity_loss(logits, sparsity_type="l1"):
    """
    Compute sparsity regularization loss for LUTs to encourage simpler boolean functions.

    Args:
        logits: List of logit tensors for each layer
        sparsity_type: Type of sparsity regularization
            - "l1": L1 penalty on sigmoid outputs (encourages 0s and 1s)
            - "binary": Penalty for non-binary values (encourages discrete decisions)
            - "entropy": Entropy-based penalty (encourages low-entropy distributions)

    Returns:
        Sparsity loss value (scalar)
    """
    total_sparsity_loss = 0.0

    for lgt in logits:
        # Convert logits to LUT probabilities
        lut_probs = jax.nn.sigmoid(lgt)

        if sparsity_type == "l1":
            # L1 penalty on sigmoid outputs - encourages values close to 0 or 1
            sparsity_loss = jp.sum(jp.minimum(lut_probs, 1.0 - lut_probs))

        elif sparsity_type == "binary":
            # Penalty for non-binary values - encourages discrete 0/1 decisions
            sparsity_loss = jp.sum(lut_probs * (1.0 - lut_probs))

        elif sparsity_type == "entropy":
            # Entropy-based penalty - encourages low-entropy (deterministic) LUTs
            eps = 1e-7  # Small epsilon to avoid log(0)
            lut_probs_safe = jp.clip(lut_probs, eps, 1.0 - eps)
            entropy = -lut_probs_safe * jp.log(lut_probs_safe) - (1.0 - lut_probs_safe) * jp.log(
                1.0 - lut_probs_safe
            )
            sparsity_loss = jp.sum(entropy)

        else:
            raise ValueError(f"Unknown sparsity_type: {sparsity_type}")

        total_sparsity_loss += sparsity_loss

    return total_sparsity_loss


def create_sparse_optimizer(base_optimizer, sparsity_weight=1e-3, sparsity_type="l1"):
    """
    Create a custom optimizer that includes LUT sparsity regularization.

    Args:
        base_optimizer: Base optax optimizer (e.g., optax.adamw(...))
        sparsity_weight: Weight for sparsity regularization term
        sparsity_type: Type of sparsity regularization ("l1", "binary", or "entropy")

    Returns:
        Custom optimizer with sparsity regularization
    """

    def sparse_gradient_transform(updates, state, params=None):
        """Transform gradients to include sparsity regularization."""
        if params is None:
            return base_optimizer.update(updates, state, params)

        # Compute sparsity gradients
        def sparsity_loss_fn(logits):
            return compute_lut_sparsity_loss(logits, sparsity_type)

        sparsity_grads = jax.grad(sparsity_loss_fn)(params)

        # Add sparsity gradients to existing gradients
        combined_updates = []
        for update, sparsity_grad in zip(updates, sparsity_grads, strict=True):
            combined_update = update + sparsity_weight * sparsity_grad
            combined_updates.append(combined_update)

        # Apply base optimizer
        return base_optimizer.update(combined_updates, state, params)

    # Create a custom optimizer that wraps the base optimizer
    class SparseOptimizer:
        def __init__(self):
            self.base_optimizer = base_optimizer

        def init(self, params):
            return self.base_optimizer.init(params)

        def update(self, updates, state, params=None):
            return sparse_gradient_transform(updates, state, params)

    return SparseOptimizer()


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
            logits=jp.log(jp.clip(y_pred, 1e-7, 1.0) / jp.clip(1.0 - y_pred, 1e-7, 1.0)),
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
    return loss, {
        "act": act,
        "accuracy": accuracy,
        "hard_loss": hard_loss,
        "hard_accuracy": hard_accuracy,
        "res": res,
        "hard_res": hard_res,
    }


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
    return loss, {
        "act": act,
        "accuracy": accuracy,
        "hard_loss": hard_loss,
        "hard_accuracy": hard_accuracy,
        "res": y - y0,
        "hard_res": hard_y - y0,
    }


# Pre-compile gradient functions for both loss types
grad_loss_f_l4 = jax.jit(jax.value_and_grad(loss_f_l4, has_aux=True))
grad_loss_f_bce = jax.jit(jax.value_and_grad(loss_f_bce, has_aux=True))


# Function dispatcher for loss computation (not used in training loop)
def loss_f(logits, wires, x, y0, loss_type="l4", gate_mask=None):
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
        return loss_f_bce(logits, wires, x, y0, gate_mask=gate_mask)
    else:  # Default to L4 norm
        return loss_f_l4(logits, wires, x, y0, gate_mask=gate_mask)


# Define a named tuple for training state to improve code clarity
TrainState = namedtuple("TrainState", "params opt_state")


# Remove JIT from this function since it has a non-tensor argument (opt)
@jax.jit
def _preserve_knocked_out_params(updated_logits, original_logits, gate_masks):
    """
    JIT-compiled function to preserve knocked out parameters efficiently.

    Since gradients are automatically zero for knocked out gates (due to
    masking in run_circuit), we only need to preserve parameter values
    after optimizer updates to handle momentum, weight decay, etc.

    Args:
        updated_logits: List of updated logit arrays
        original_logits: List of original logit arrays
        gate_masks: List of gate masks (excluding input layer)

    Returns:
        Logits with knocked out gates preserved
    """
    preserved_logits = []

    for updated_layer, original_layer, layer_mask in zip(
        updated_logits, original_logits, gate_masks, strict=True
    ):
        group_n, group_size, lut_size = updated_layer.shape

        # Reshape mask and broadcast to match logits
        mask_reshaped = layer_mask.reshape(group_n, group_size, 1)
        mask_broadcast = jp.broadcast_to(mask_reshaped, updated_layer.shape)

        # Preserve knocked out gates (mask == 0), keep updated for active gates (mask == 1)
        preserved_layer = jp.where(
            mask_broadcast == 0.0,
            original_layer,  # Preserve original for knocked out gates
            updated_layer,  # Use updated for active gates
        )
        preserved_logits.append(preserved_layer)

    return preserved_logits


def update_params(grad, opt_state, opt, logits, gate_mask=None):
    """
    Parameter update function with gate mask support.

    Since gradients are automatically zero for knocked out gates (due to masking
    in run_circuit), we only need to preserve their parameter values after
    optimizer updates to prevent momentum, weight decay, etc. from affecting them.

    Args:
        grad: Gradients for logits (already zero for knocked out gates)
        opt_state: Optimizer state
        opt: Optimizer instance
        logits: Current logit parameters
        gate_mask: Optional gate mask (layered format) where 0.0 = knocked out

    Returns:
        Tuple of (new_logits, new_opt_state)
    """
    # Apply optimizer update normally (gradients are already zero for knocked out gates)
    upd, new_opt_state = opt.update(grad, opt_state, logits)
    updated_logits = optax.apply_updates(logits, upd)

    if gate_mask is not None:
        # Only need to preserve parameter values after optimizer update
        logit_gate_masks = gate_mask[1:]  # Skip input layer
        new_logits = _preserve_knocked_out_params(updated_logits, logits, logit_gate_masks)
    else:
        new_logits = updated_logits

    return new_logits, new_opt_state


def train_step(
    state,
    opt,
    wires,
    x,
    y0,
    loss_type="l4",
    do_train=True,
    gate_mask=None,
    x_test=None,
    y_test=None,
):
    """
    Perform a single training step, optionally evaluating on test data.

    Args:
        state: TrainState containing current parameters and optimizer state
        opt: Optax optimizer
        wires: Wiring configuration for the circuit
        x: Training input batch
        y0: Training target output batch
        loss_type: Type of loss to use ('l4' or 'bce')
        do_train: Whether to perform parameter updates (default True)
        gate_mask: Optional gate mask for circuit pruning
        x_test: Optional test input batch
        y_test: Optional test target output batch

    Returns:
        If test data is provided:
            Tuple of (train_loss, train_aux, test_loss, test_aux, new_state)
        If no test data:
            Tuple of (train_loss, train_aux, new_state) - maintains backward compatibility
    """
    logits, opt_state = state

    # Training step
    if do_train:
        if loss_type == "bce":
            (train_loss, train_aux), grad = grad_loss_f_bce(
                logits, wires, x, y0, gate_mask=gate_mask
            )
        else:  # Default to L4 norm
            (train_loss, train_aux), grad = grad_loss_f_l4(
                logits, wires, x, y0, gate_mask=gate_mask
            )

        # Update parameters (without JIT since optimizer is a function)
        new_logits, new_opt_state = update_params(grad, opt_state, opt, logits, gate_mask)

    else:
        train_loss, train_aux = loss_f(logits, wires, x, y0, loss_type, gate_mask=gate_mask)
        new_logits = logits
        new_opt_state = opt_state

    new_state = TrainState(new_logits, new_opt_state)

    # If test data is provided, evaluate on test set
    if x_test is not None and y_test is not None:
        # Evaluate on test data (no training, use updated parameters)
        test_loss, test_aux = loss_f(
            new_logits, wires, x_test, y_test, loss_type, gate_mask=gate_mask
        )

        return train_loss, train_aux, test_loss, test_aux, new_state
    else:
        # Backward compatibility: return original format when no test data
        return train_loss, train_aux, new_state
