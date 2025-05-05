"""
Training step implementation for GNN-based boolean circuit optimization.

This module provides functions for performing single training steps
for optimizing boolean circuits using GNNs.
"""

import jax
import jax.numpy as jp
import jraph
from flax import nnx
from typing import List, Tuple, Dict
from functools import partial

from boolean_nca_cc.models import CircuitGNN, run_gnn_scan
from boolean_nca_cc.utils import extract_logits_from_graph
from model import run_circuit


@partial(nnx.jit, static_argnames=("logits_original_shapes", "n_message_steps"))
def train_step_gnn(
    gnn: CircuitGNN,
    optimizer: nnx.Optimizer,
    graph: jraph.GraphsTuple,
    wires: List[jp.ndarray],
    x: jp.ndarray,
    y_target: jp.ndarray,
    logits_original_shapes: List[Tuple[int, int, int]],
    n_message_steps: int,
):
    """
    Perform a single training step for the GNN-based circuit optimizer.
    
    Args:
        gnn: CircuitGNN model
        optimizer: nnx Optimizer instance
        graph: GraphsTuple representation of the circuit
        wires: List of wire connection patterns
        x: Input tensor
        y_target: Target output tensor
        logits_original_shapes: List of (group_n, group_size, 2^arity) shapes for each layer
        n_message_steps: Number of message passing steps to run
        
    Returns:
        Tuple of (loss, (hard_loss, accuracy, hard_accuracy, updated_logits, updated_graph), gradients)
    """
    # Define loss function for the model
    def loss_fn(gnn_model: CircuitGNN):
        # Run message passing for n steps
        updated_graph = run_gnn_scan(gnn_model, graph, n_message_steps)

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
    (loss, aux), grads = nnx.value_and_grad(loss_fn, has_aux=True)(gnn)

    # Update GNN parameters using the optimizer
    optimizer.update(grads)

    # Return loss, auxiliary outputs, and gradients
    return loss, aux, grads 