"""
Preconfiguration utility to produce a working circuit configuration (wires, logits)
by optimizing logits only on a fixed wiring via backprop, matching training loss semantics.
"""

import jax
import jax.numpy as jp
import optax
from typing import List, Tuple
import logging

from boolean_nca_cc.circuits.model import gen_circuit
from boolean_nca_cc.circuits.train import TrainState, train_step

log = logging.getLogger(__name__)


def preconfigure_circuit_logits(
    wiring_key: jax.random.PRNGKey,
    layer_sizes: List[Tuple[int, int]],
    arity: int,
    x_data: jp.ndarray,
    y_data: jp.ndarray,
    loss_type: str,
    steps: int = 200,
    lr: float = 1,
    optimizer: str = "adam",
    weight_decay: float = 0.0,
    beta1: float = 0.9,
    beta2: float = 0.999,
) -> Tuple[List[jp.ndarray], List[jp.ndarray]]:
    """
    Optimize circuit logits only on a fixed wiring to obtain a configured base circuit.
    
    Uses the same backprop configuration as the main training for consistency.

    Returns base wires and logits suitable for initializing pools and resets in reconfig mode.
    """
    # Generate fixed wiring with NOP logits as starting point
    base_wires, base_logits = gen_circuit(wiring_key, layer_sizes, arity=arity)
    
    # DEBUG: Log initial circuit state (before optimization)
    try:
        from boolean_nca_cc.circuits.train import get_loss_from_wires_logits
        initial_loss, initial_aux = get_loss_from_wires_logits(
            base_logits, base_wires, x_data, y_data, loss_type
        )
        initial_hard_loss, _, _, initial_accuracy, initial_hard_accuracy, _, _ = initial_aux
        log.debug(
            f"Preconfig initial state: loss={float(initial_loss):.6f}, "
            f"hard_loss={float(initial_hard_loss):.4f}, "
            f"accuracy={float(initial_accuracy):.4f}, "
            f"hard_accuracy={float(initial_hard_accuracy):.4f}"
        )
    except Exception as e:
        log.debug(f"Could not compute initial loss: {e}")

    # Setup optimizer using same configuration as main training
    if optimizer == "adamw":
        opt = optax.adamw(
            lr,
            b1=beta1,
            b2=beta2,
            weight_decay=weight_decay,
        )
    else:
        opt = optax.adam(lr, b1=beta1, b2=beta2)
    
    state = TrainState(params=base_logits, opt_state=opt.init(base_logits))
    
    # DEBUG: Log optimizer state info
    log.debug(f"Preconfig optimizer: {optimizer}, lr={lr}, beta1={beta1}, beta2={beta2}, weight_decay={weight_decay}")

    # Partially apply fixed args to training step
    step_fn = lambda s: train_step(
        state=s,
        opt=opt,
        wires=base_wires,
        x=x_data,
        y0=y_data,
        loss_type=loss_type,
        do_train=True,
        knockout_pattern=None,
        layer_sizes=None,
    )

    last_loss = None
    for step_i in range(int(steps)):
        loss, _aux, new_state = step_fn(state)
        state = new_state
        last_loss = loss
        
        # DEBUG: Log loss at key steps (first, middle, last)
        if step_i == 0 or step_i == steps // 2 or step_i == steps - 1:
            log.debug(f"Preconfig step {step_i}: loss={float(loss):.6f}")

    if not jp.isfinite(last_loss):
        raise RuntimeError(f"Preconfiguration produced non-finite loss: {float(last_loss)}")
    
    # DEBUG: Log final loss
    log.debug(f"Preconfig final loss: {float(last_loss):.6f}")

    return base_wires, state.params


