"""
Differentiable Boolean Circuits

This module implements a framework for creating and running differentiable
boolean circuits. These circuits can be optimized through gradient-based methods
while representing boolean logic operations.

The implementation uses JAX for automatic differentiation and numerical operations.
"""

import jax
import jax.numpy as jp


def gen_wires(key, in_n, out_n, arity, group_size):
    """
    Generate random wiring connections between circuit layers.

    Args:
        key: JAX random key for reproducible randomness
        in_n: Number of input nodes
        out_n: Number of output nodes
        arity: Number of inputs per gate (fan-in)
        group_size: Number of gates per group

    Returns:
        Array of shape (arity, out_n//group_size) containing input indices for each gate
    """
    edge_n = out_n * arity // group_size
    n = max(in_n, edge_n)
    return jax.random.permutation(key, n)[:edge_n].reshape(arity, -1) % in_n


def make_nops(gate_n, arity, group_size, nop_scale=3.0):
    """
    Create logits for all possible boolean operations (lookup tables).

    Args:
        gate_n: Total number of gates
        arity: Number of inputs per gate
        group_size: Number of gates per group
        nop_scale: Scaling factor for logits (affects sigmoid sharpness)

    Returns:
        Logits tensor of shape (gate_n//group_size, group_size, 2^arity)

    Example:
        For arity=2, this function creates lookup tables that can represent
        any 2-input boolean function. The process works as follows:

        1. Generate indices for all input combinations: [0,1,2,3]
        2. Convert to binary: [[0,0], [0,1], [1,0], [1,1]]
        3. Create initial LUTs and convert to logits

        During training, these logits are optimized to implement the desired
        boolean function (AND, OR, XOR, etc.) through gradient descent.
    """
    I = jp.arange(1 << arity)
    bits = (I >> I[:arity, None]) & 1
    luts = bits[jp.arange(gate_n) % arity]
    logits = (2.0 * luts - 1.0) * nop_scale
    return logits.reshape(gate_n // group_size, group_size, -1)


@jax.jit
def run_layer(lut, inputs):
    """
    Run a single layer of the boolean circuit.

    This function evaluates boolean lookup tables (LUTs) given their inputs.
    The implementation uses a binary decision diagram approach.

    Args:
        lut: Lookup table values [group_n, group_size, 2^arity]
        inputs: List of input tensors, one per input bit

    Returns:
        Output tensor after processing through the layer

    Example:
        For a 2-input gate with inputs A and B, and lookup table LUT=[0,1,1,0] (XOR):

        1. First iteration (input A):
           - If A=0: Select first half of LUT [0,1]
           - If A=1: Select second half of LUT [1,0]

        2. Second iteration (input B):
           - If B=0: Select even indices from previous selection
           - If B=1: Select odd indices from previous selection

        For inputs A=1, B=0:
        1. A=1 selects [1,0]
        2. B=0 selects index 0 from [1,0], giving output 1

        This approach maintains differentiability while computing boolean functions.
    """
    for x in inputs:
        x = x[..., None, None]
        lut = (1.0 - x) * lut[..., ::2] + x * lut[..., 1::2]
    return lut.reshape(*lut.shape[:-3] + (-1,))


def run_circuit(logits, wires, x, hard=False):
    """
    Run the entire boolean circuit with multiple layers.

    Args:
        logits: List of logits for each layer
        wires: List of wire connection patterns for each layer
        x: Input tensor
        hard: If True, round outputs to binary values (0 or 1)
              If False, use soft (differentiable) outputs

    Returns:
        List of activation tensors for all layers (including input)
    """
    acts = [x]
    for ws, lgt in zip(wires, logits):
        luts = jax.nn.sigmoid(lgt)
        if hard:
            luts = jp.round(luts)
        x = run_layer(luts, [x[..., w] for w in ws])
        acts.append(x)
    return acts


def gen_circuit(key, layer_sizes, arity=4):
    """
    Generate a complete circuit with random wiring and initial operations.

    Args:
        key: JAX random key
        layer_sizes: List of tuples (nodes, group_size) for each layer
        arity: Number of inputs per gate (fan-in)

    Returns:
        Tuple of (wires, logits) where each is a list per layer
    """
    in_n = layer_sizes[0][0]
    all_wires, all_logits = [], []
    for out_n, group_size in layer_sizes:
        wires = gen_wires(key, in_n, out_n, arity, group_size)
        logits = make_nops(out_n, arity, group_size)
        _, key = jax.random.split(key)
        in_n = out_n
        all_wires.append(wires)
        all_logits.append(logits)
    return all_wires, all_logits
