"""
Training Module for Differentiable Boolean Circuits

This module contains functions for training boolean circuits using gradient-based
optimization. It provides utilities for converting between representations,
computing loss, and performing optimization steps.
"""

from collections import namedtuple
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jp
import numpy as np
import optax
from flax import nnx

from boolean_nca_cc.circuits.model import run_circuit


class LossConfig(NamedTuple):
    """Hashable loss configuration for JAX JIT compilation."""
    type: str  # "bce" or "l"
    alpha_pos: float = 1.0
    alpha_neg: float = 1.0
    gamma_pos: float = 0.0
    gamma_neg: float = 0.0
    power: int = 4  # Only used for type="l"
    name: str = "bce"

    @classmethod
    def from_dict(cls, d: dict):
        """Create LossConfig from dict (e.g., from Hydra config)."""
        # Handle None values explicitly (Hydra may set missing keys to None)
        def get_float(key, default):
            val = d.get(key, default)
            return float(val) if val is not None else float(default)
        
        def get_int(key, default):
            val = d.get(key, default)
            return int(val) if val is not None else int(default)
        
        return cls(
            type=d.get("type") or "bce",
            alpha_pos=get_float("alpha_pos", 1.0),
            alpha_neg=get_float("alpha_neg", 1.0),
            gamma_pos=get_float("gamma_pos", 0.0),
            gamma_neg=get_float("gamma_neg", 0.0),
            power=get_int("power", 4),
            name=d.get("name") or "bce",
        )
    
    def to_dict(self):
        """Convert to dict for compatibility."""
        return {
            "type": self.type,
            "alpha_pos": self.alpha_pos,
            "alpha_neg": self.alpha_neg,
            "gamma_pos": self.gamma_pos,
            "gamma_neg": self.gamma_neg,
            "power": self.power,
            "name": self.name,
        }


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
def generalized_bce(
    y_pred,
    y_true,
    alpha_pos: float = 1.0,
    alpha_neg: float = 1.0,
    gamma_pos: float = 0.0,
    gamma_neg: float = 0.0,
):
    """
    Unified binary cross-entropy with class weighting and focal modulation.
    
    L = -α_pos * (1-p)^γ_pos * y * log(p) - α_neg * p^γ_neg * (1-y) * log(1-p)
    
    This generalizes BCE, weighted BCE, focal loss, and asymmetric loss:
    
    | Loss Type              | α_pos | α_neg | γ_pos | γ_neg |
    |------------------------|-------|-------|-------|-------|
    | BCE                    | 1     | 1     | 0     | 0     |
    | Weighted BCE (w=3)     | 3     | 1     | 0     | 0     |
    | Focal (α=.75, γ=2)     | 0.75  | 0.25  | 2     | 2     |
    | Asymmetric (γ_neg=4)   | 1     | 1     | 0     | 4     |
    
    Args:
        y_pred: Predicted outputs (probabilities in [0,1])
        y_true: Target outputs (typically 0 or 1)
        alpha_pos: Weight for positive class (default 1.0)
        alpha_neg: Weight for negative class (default 1.0)
        gamma_pos: Focusing param for positives (0 = no focusing)
        gamma_neg: Focusing param for negatives (0 = no focusing)
    
    Returns:
        Loss value (scalar)
    """
    eps = 1e-7
    p = jp.clip(y_pred, eps, 1.0 - eps)
    
    pos_loss = -alpha_pos * jp.power(1.0 - p, gamma_pos) * y_true * jp.log(p)
    neg_loss = -alpha_neg * jp.power(p, gamma_neg) * (1.0 - y_true) * jp.log(1.0 - p)
    
    return jp.sum(pos_loss + neg_loss)


# Convenience aliases for backward compatibility
def binary_cross_entropy(y_pred, y_true, pos_weight: float = 1.0):
    """Weighted BCE (legacy alias). Use generalized_bce for full control."""
    return generalized_bce(y_pred, y_true, alpha_pos=pos_weight)


def focal_loss(y_pred, y_true, alpha: float = 0.75, gamma: float = 2.0):
    """Focal loss (legacy alias). Use generalized_bce for full control."""
    return generalized_bce(y_pred, y_true, alpha_pos=alpha, alpha_neg=1.0 - alpha, gamma_pos=gamma, gamma_neg=gamma)


def asymmetric_loss(y_pred, y_true, gamma_pos: float = 0.0, gamma_neg: float = 4.0):
    """Asymmetric loss (legacy alias). Use generalized_bce for full control."""
    return generalized_bce(y_pred, y_true, gamma_pos=gamma_pos, gamma_neg=gamma_neg)


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


def get_loss_params(loss_cfg):
    """
    Extract loss parameters from LossConfig or dict.
    
    Args:
        loss_cfg: LossConfig NamedTuple or dict with keys: type, alpha_pos, alpha_neg, gamma_pos, gamma_neg, power
    
    Returns:
        Tuple of (loss_type, kwargs) where loss_type is "gbce" or "l".
    """
    # Convert dict to LossConfig if needed
    if isinstance(loss_cfg, dict):
        loss_cfg = LossConfig.from_dict(loss_cfg)
    
    if loss_cfg.type == "l":
        return "l", {"power": loss_cfg.power}
    else:
        return "gbce", {
            "alpha_pos": loss_cfg.alpha_pos,
            "alpha_neg": loss_cfg.alpha_neg,
            "gamma_pos": loss_cfg.gamma_pos,
            "gamma_neg": loss_cfg.gamma_neg,
        }


def compute_loss_from_predictions(pred, pred_hard, y_target, loss_cfg):
    """
    Unified loss computation from predictions.
    
    This is the single source of truth for loss computation, used by both
    backprop training (train.py) and GNN evaluation (evaluation.py).
    
    Args:
        pred: Soft predictions (probabilities)
        pred_hard: Hard predictions (rounded to 0/1)
        y_target: Target values
        loss_cfg: LossConfig NamedTuple or dict with keys: type, alpha_pos, alpha_neg, gamma_pos, gamma_neg, power
    
    Returns:
        Tuple of (loss, hard_loss, res, hard_res, accuracy, hard_accuracy)
    """
    # Always compute residuals
    res = pred - y_target
    hard_res = pred_hard - y_target
    
    # Get loss params from config
    loss_type, kwargs = get_loss_params(loss_cfg)
    
    # Compute losses
    if loss_type == "gbce":
        loss = generalized_bce(pred, y_target, **kwargs)
        hard_loss = generalized_bce(pred_hard, y_target, **kwargs)
    elif loss_type == "l":
        loss = res2loss(res, **kwargs)
        hard_loss = res2loss(hard_res, **kwargs)
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")
    
    # Compute accuracies
    accuracy = compute_accuracy(pred, y_target)
    hard_accuracy = compute_accuracy(pred_hard, y_target)
    
    return loss, hard_loss, res, hard_res, accuracy, hard_accuracy


# Default loss configs for common cases (hashable NamedTuples for JAX JIT)
LOSS_L4 = LossConfig(type="l", power=4)
LOSS_L2 = LossConfig(type="l", power=2)
LOSS_BCE = LossConfig(type="bce", alpha_pos=1.0, alpha_neg=1.0, gamma_pos=0.0, gamma_neg=0.0)
LOSS_BCE_W2 = LossConfig(type="bce", alpha_pos=2.0, alpha_neg=1.0, gamma_pos=0.0, gamma_neg=0.0)
LOSS_BCE_W3 = LossConfig(type="bce", alpha_pos=3.0, alpha_neg=1.0, gamma_pos=0.0, gamma_neg=0.0)
LOSS_FOCAL = LossConfig(type="bce", alpha_pos=0.75, alpha_neg=0.25, gamma_pos=2.0, gamma_neg=2.0)
LOSS_ASYM = LossConfig(type="bce", alpha_pos=1.0, alpha_neg=1.0, gamma_pos=0.0, gamma_neg=4.0)


def loss_f(logits, wires, x, y0, loss_cfg=None, gate_mask=None):
    """
    Compute loss for a circuit given input and target output.

    Args:
        logits: List of logits for each layer
        wires: List of wire connection patterns
        x: Input tensor
        y0: Target output tensor
        loss_cfg: LossConfig NamedTuple or dict (default: LOSS_L4)
        gate_mask: Optional gate mask for circuit pruning

    Returns:
        Tuple of (loss_value, auxiliary_dict) where auxiliary_dict contains
        intermediate activations and accuracy
    """
    if loss_cfg is None:
        loss_cfg = LOSS_L4
    elif isinstance(loss_cfg, dict):
        loss_cfg = LossConfig.from_dict(loss_cfg)
        
    act = run_circuit(logits, wires, x, gate_mask=gate_mask)
    hard_act = run_circuit(logits, wires, x, gate_mask=gate_mask, hard=True)
    y = act[-1]
    hard_y = hard_act[-1]
    
    loss, hard_loss, res, hard_res, accuracy, hard_accuracy = compute_loss_from_predictions(
        y, hard_y, y0, loss_cfg
    )
    
    return loss, {
        "act": act,
        "accuracy": accuracy,
        "hard_loss": hard_loss,
        "hard_accuracy": hard_accuracy,
        "res": res,
        "hard_res": hard_res,
    }


# Legacy functions for backward compatibility
def loss_f_l4(logits, wires, x, y0, gate_mask=None):
    """L4 loss function."""
    return loss_f(logits, wires, x, y0, loss_cfg=LOSS_L4, gate_mask=gate_mask)


def loss_f_bce(logits, wires, x, y0, gate_mask=None):
    """BCE loss function."""
    return loss_f(logits, wires, x, y0, loss_cfg=LOSS_BCE, gate_mask=gate_mask)


def loss_f_bce_w2(logits, wires, x, y0, gate_mask=None):
    """Weighted BCE with alpha_pos=2."""
    return loss_f(logits, wires, x, y0, loss_cfg=LOSS_BCE_W2, gate_mask=gate_mask)


def loss_f_bce_w3(logits, wires, x, y0, gate_mask=None):
    """Weighted BCE with alpha_pos=3."""
    return loss_f(logits, wires, x, y0, loss_cfg=LOSS_BCE_W3, gate_mask=gate_mask)


# Pre-compile gradient functions for common loss types
grad_loss_f_l4 = jax.jit(jax.value_and_grad(loss_f_l4, has_aux=True))
grad_loss_f_bce = jax.jit(jax.value_and_grad(loss_f_bce, has_aux=True))
grad_loss_f_bce_w2 = jax.jit(jax.value_and_grad(loss_f_bce_w2, has_aux=True))
grad_loss_f_bce_w3 = jax.jit(jax.value_and_grad(loss_f_bce_w3, has_aux=True))

# Generic gradient function
def grad_loss_f(logits, wires, x, y0, loss_cfg=None, gate_mask=None):
    """Generic gradient function that works with any loss config."""
    if loss_cfg is None:
        loss_cfg = LOSS_L4
    return jax.value_and_grad(
        lambda lg: loss_f(lg, wires, x, y0, loss_cfg=loss_cfg, gate_mask=gate_mask),
        has_aux=True
    )(logits)


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
    loss_cfg=None,
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
        loss_cfg: Loss config dict (default: LOSS_L4)
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
    if loss_cfg is None:
        loss_cfg = LOSS_L4
    elif isinstance(loss_cfg, dict):
        loss_cfg = LossConfig.from_dict(loss_cfg)
        
    logits, opt_state = state

    # Training step
    if do_train:
        # Use pre-compiled gradients for common configs, generic for others
        if loss_cfg == LOSS_L4:
            (train_loss, train_aux), grad = grad_loss_f_l4(
                logits, wires, x, y0, gate_mask=gate_mask
            )
        elif loss_cfg == LOSS_BCE:
            (train_loss, train_aux), grad = grad_loss_f_bce(
                logits, wires, x, y0, gate_mask=gate_mask
            )
        elif loss_cfg == LOSS_BCE_W2:
            (train_loss, train_aux), grad = grad_loss_f_bce_w2(
                logits, wires, x, y0, gate_mask=gate_mask
            )
        elif loss_cfg == LOSS_BCE_W3:
            (train_loss, train_aux), grad = grad_loss_f_bce_w3(
                logits, wires, x, y0, gate_mask=gate_mask
            )
        else:
            # Generic gradient for custom loss configs
            (train_loss, train_aux), grad = grad_loss_f(
                logits, wires, x, y0, loss_cfg=loss_cfg, gate_mask=gate_mask
            )

        # Update parameters (without JIT since optimizer is a function)
        new_logits, new_opt_state = update_params(grad, opt_state, opt, logits, gate_mask)

    else:
        train_loss, train_aux = loss_f(logits, wires, x, y0, loss_cfg=loss_cfg, gate_mask=gate_mask)
        new_logits = logits
        new_opt_state = opt_state

    new_state = TrainState(new_logits, new_opt_state)

    # If test data is provided, evaluate on test set
    if x_test is not None and y_test is not None:
        # Evaluate on test data (no training, use updated parameters)
        test_loss, test_aux = loss_f(
            new_logits, wires, x_test, y_test, loss_cfg=loss_cfg, gate_mask=gate_mask
        )

        return train_loss, train_aux, test_loss, test_aux, new_state
    else:
        # Backward compatibility: return original format when no test data
        return train_loss, train_aux, new_state
