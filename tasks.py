"""
Boolean Circuit Tasks

Pure functional implementation of boolean tasks for training circuits.
"""

import jax.numpy as jp
import numpy as np


def unpack(x, bit_n=8):
    """Unpack an integer into its constituent bits"""
    return jp.float32((x[..., None] >> np.r_[:bit_n]) & 1)


# Task definitions as pure functions
def binary_multiply(case_n, input_bits=8, output_bits=8):
    """Multiply lower and upper halves of input"""
    x = jp.arange(case_n)
    half = input_bits // 2
    y = (x & ((1 << half) - 1)) * (x >> half)
    return unpack(x, input_bits), unpack(y, output_bits)


def bitwise_and(case_n, input_bits=8, output_bits=8):
    """Bitwise AND between halves of input"""
    x = jp.arange(case_n)
    half = input_bits // 2
    y = (x & ((1 << half) - 1)) & (x >> half)
    return unpack(x, input_bits), unpack(y, output_bits)


def bitwise_xor(case_n, input_bits=8, output_bits=8):
    """Bitwise XOR between halves of input"""
    x = jp.arange(case_n)
    half = input_bits // 2
    y = (x & ((1 << half) - 1)) ^ (x >> half)
    return unpack(x, input_bits), unpack(y, output_bits)


def binary_add(case_n, input_bits=8, output_bits=None):
    """Add lower and upper halves of input"""
    if output_bits is None:
        output_bits = input_bits // 2 + 1

    x = jp.arange(case_n)
    half = input_bits // 2
    y = (x & ((1 << half) - 1)) + (x >> half)
    return unpack(x, input_bits), unpack(y, output_bits)


def parity(case_n, input_bits=8):
    """Compute parity (number of 1 bits is odd/even)"""
    x = jp.arange(case_n)
    y = x
    for i in range(1, input_bits):
        y ^= y >> i
    return unpack(x, input_bits), unpack(y & 1, 1)


def reverse_bits(case_n, input_bits=8):
    """Reverse the bits in the input"""
    x = jp.arange(case_n)
    y = jp.zeros_like(x)
    for i in range(input_bits):
        y = y | (((x >> i) & 1) << (input_bits - 1 - i))
    return unpack(x, input_bits), unpack(y, input_bits)


def custom_task(func, case_n, input_bits=8, output_bits=8):
    """Create a custom task with a user-defined function"""
    x = jp.arange(case_n)
    y = func(x)
    return unpack(x, input_bits), unpack(y, output_bits)


# Task lookup dictionary
TASKS = {
    "binary_multiply": binary_multiply,
    "and": bitwise_and,
    "xor": bitwise_xor,
    "add": binary_add,
    "parity": parity,
    "reverse": reverse_bits,
}


def get_task_data(task_name, case_n, **kwargs):
    """Get a task by name and generate its data"""
    if task_name not in TASKS:
        raise ValueError(f"Unknown task: {task_name}. Available: {list(TASKS.keys())}")

    return TASKS[task_name](case_n, **kwargs)
