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


def parity(case_n, input_bits=8, output_bits=None):
    """Compute parity (number of 1 bits is odd/even)"""
    if output_bits is not None and output_bits != 1:
        raise ValueError("Parity task is defined to have only 1 output bit.")
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


def copy(case_n, input_bits=8, output_bits=8):
    """Simple identity function - output equals input"""
    x = jp.arange(case_n)
    return unpack(x, input_bits), unpack(x, output_bits)


def gray_code(case_n, input_bits=8, output_bits=8):
    """Convert to Gray code (each number differs from its neighbors by 1 bit)"""
    x = jp.arange(case_n)
    y = x ^ (x >> 1)
    return unpack(x, input_bits), unpack(y, output_bits)


def popcount(case_n, input_bits=8, output_bits=None):
    """Count the number of 1 bits in the input"""
    if output_bits is None:
        # Number of bits needed to represent the count (log2 of input_bits, rounded up)
        output_bits = (input_bits.bit_length() - 1) + 1

    x = jp.arange(case_n)
    y = jp.sum(unpack(x, input_bits), axis=-1)  # Sum the bits to get the count
    return unpack(x, input_bits), unpack(y, output_bits)


# Task lookup dictionary
TASKS = {
    "binary_multiply": binary_multiply,
    "and": bitwise_and,
    "xor": bitwise_xor,
    "add": binary_add,
    "parity": parity,
    "reverse": reverse_bits,
    "copy": copy,
    "gray": gray_code,
    "popcount": popcount,
}


def get_task_data(task_name, case_n, **kwargs):
    """Get a task by name and generate its data"""
    if task_name not in TASKS:
        raise ValueError(f"Unknown task: {task_name}. Available: {list(TASKS.keys())}")

    task_specific_kwargs = {
        k: v
        for k, v in kwargs.items()
        if k
        in TASKS[task_name].__code__.co_varnames[
            : TASKS[task_name].__code__.co_argcount
        ]
    }
    return TASKS[task_name](case_n, **task_specific_kwargs)
