"""
Preconfiguration utility to produce a working circuit configuration (wires, logits)
by optimizing logits only on a fixed wiring via backprop, matching training loss semantics.
"""

import jax
import jax.numpy as jp
import optax
from typing import List, Tuple

from boolean_nca_cc.circuits.model import gen_circuit
from boolean_nca_cc.circuits.train import TrainState, train_step


def preconfigure_circuit_logits(
    wiring_key: jax.random.PRNGKey,
    layer_sizes: List[Tuple[int, int]],
    arity: int,
    x_data: jp.ndarray,
    y_data: jp.ndarray,
    loss_type: str,
    steps: int = 200,
    lr: float = 1e-2,
) -> Tuple[List[jp.ndarray], List[jp.ndarray]]:
    """
    Optimize circuit logits only on a fixed wiring to obtain a configured base circuit.

    Returns base wires and logits suitable for initializing pools and resets in reconfig mode.
    """
    # Generate fixed wiring with NOP logits as starting point
    base_wires, base_logits = gen_circuit(wiring_key, layer_sizes, arity=arity)

    # Optimizer over logits only
    opt = optax.adam(lr)
    state = TrainState(params=base_logits, opt_state=opt.init(base_logits))

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
    for _ in range(int(steps)):
        loss, _aux, new_state = step_fn(state)
        state = new_state
        last_loss = loss

    if not jp.isfinite(last_loss):
        raise RuntimeError(f"Preconfiguration produced non-finite loss: {float(last_loss)}")

    return base_wires, state.params


