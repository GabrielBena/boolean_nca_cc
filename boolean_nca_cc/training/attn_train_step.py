"""
Training step implementation for self-attention-based boolean circuit optimization.

This module provides functions for performing single training steps
for optimizing boolean circuits using masked self-attention.
"""

import jax
import jax.numpy as jp
import jraph
from flax import nnx
from typing import List, Tuple, Dict, Optional
from functools import partial

from boolean_nca_cc.models.self_attention import (
    CircuitSelfAttention,
    run_self_attention_scan,
)
from boolean_nca_cc.utils import extract_logits_from_graph
from model import run_circuit


@partial(nnx.jit, static_argnames=("logits_original_shapes", "n_attention_steps"))
def train_step_attn(
    attn_model: CircuitSelfAttention,
    optimizer: nnx.Optimizer,
    graph: jraph.GraphsTuple,
    wires: List[jp.ndarray],
    x: jp.ndarray,
    y_target: jp.ndarray,
    logits_original_shapes: List[Tuple[int, int, int]],
    n_attention_steps: int,
    deterministic: bool = True,
    rngs: Optional[nnx.Rngs] = None,
):
    """
    Perform a single training step for the self-attention-based circuit optimizer.

    Args:
        attn_model: CircuitSelfAttention model
        optimizer: nnx Optimizer instance
        graph: GraphsTuple representation of the circuit
        wires: List of wire connection patterns
        x: Input tensor
        y_target: Target output tensor
        logits_original_shapes: List of (group_n, group_size, 2^arity) shapes for each layer
        n_attention_steps: Number of self-attention steps to run
        deterministic: If False, apply dropout
        rngs: Random number generators

    Returns:
        Tuple of (loss, (hard_loss, accuracy, hard_accuracy, updated_logits, updated_graph), gradients)
    """

    # Define loss function for the model
    def loss_fn(model: CircuitSelfAttention):
        # Run self-attention for n steps
        updated_graph = run_self_attention_scan(
            model,
            graph,
            n_attention_steps,
            deterministic=deterministic,
            rngs=rngs,
        )

        # Extract updated logits from the final graph
        updated_logits = extract_logits_from_graph(
            updated_graph, logits_original_shapes
        )

        # Run the circuit with updated logits (soft evaluation)
        all_acts = run_circuit(updated_logits, wires, x)
        y_pred = all_acts[-1]

        # Run with hard (rounded) values for evaluation
        all_hard_acts = run_circuit(updated_logits, wires, x, hard=True)
        y_hard_pred = all_hard_acts[-1]

        # Compute loss and accuracy
        loss = jp.mean((y_pred - y_target) ** 4)
        hard_loss = jp.mean((y_hard_pred - y_target) ** 4)
        accuracy = jp.mean(jp.round(y_pred) == y_target)
        hard_accuracy = jp.mean(jp.round(y_hard_pred) == y_target)

        return loss, (hard_loss, accuracy, hard_accuracy, updated_logits, updated_graph)

    # Compute loss and gradients
    (loss, aux), grads = nnx.value_and_grad(loss_fn, has_aux=True)(attn_model)

    # Update model parameters using the optimizer
    optimizer.update(grads)

    # Return loss, auxiliary outputs, and gradients
    return loss, aux, grads
