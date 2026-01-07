"""
Boolean Circuit Tasks

Pure functional implementation of boolean tasks for training circuits.
"""

import jax
import jax.numpy as jp
import numpy as np
import logging

log = logging.getLogger(__name__)


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


def identity(case_n, input_bits=8, output_bits=8):
    """Identity function - output equals input"""
    x = jp.arange(case_n)
    return unpack(x, input_bits), unpack(x, output_bits)


# Task lookup dictionary
TASKS = {
    "identity": identity,
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


def get_task_data(task_name, case_n, max_samples=None, sample_seed=42, **kwargs):
    """
    Get a task by name and generate its data.
    
    Args:
        task_name: Name of the task
        case_n: Number of cases to generate (2^input_bits for full truth table)
        max_samples: Optional maximum number of samples to return (caps if case_n > max_samples).
                     If None, returns all case_n samples.
        sample_seed: Random seed for reproducible sampling when capping (default 42)
        **kwargs: Task-specific parameters (input_bits, output_bits, etc.)
    
    Returns:
        Tuple of (x_data, y_data) with at most max_samples if specified
        
    Example:
        >>> # Full dataset for small input_bits
        >>> x_data, y_data = get_task_data("add", case_n=256, input_bits=8)
        >>> x_data.shape[0]  # 256
        
        >>> # Capped dataset for large input_bits
        >>> x_data, y_data = get_task_data("add", case_n=2**20, max_samples=10000, input_bits=20)
        >>> x_data.shape[0]  # 10000 (capped from 1,048,576)
    """
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
    
    # If capping is needed, generate only the required samples (don't generate all first!)
    if max_samples is not None and case_n > max_samples:
        input_bits = kwargs.get('input_bits', 'unknown')
        log.info(
            f"Generating {max_samples:,} random samples from {case_n:,} possible combinations "
            f"(input_bits={input_bits} produces too many combinations to generate all)"
        )
        
        # Generate random indices without materializing the full range
        # For very large case_n (e.g., 2^50), we can't use jax.random.randint directly (overflow)
        # Instead, generate random floats, scale them, and convert to Python ints (which handle arbitrary size)
        key = jax.random.PRNGKey(sample_seed)
        uniform_samples = jax.random.uniform(key, shape=(max_samples,))
        
        # Scale to [0, case_n) and convert to Python ints (which can handle arbitrary size)
        case_n_float = float(case_n)
        indices_float = uniform_samples * case_n_float
        
        # Convert to Python ints to preserve precision for large values, then modulo to ensure valid range
        # We'll keep these as Python ints and do computations in Python, then convert to JAX arrays
        indices_python = [int(x) % case_n for x in indices_float.tolist()]
        
        # Compute outputs for these specific integer inputs using Python ints
        # Since tasks operate on integers, we can compute y directly in Python
        input_bits_val = task_specific_kwargs.get('input_bits', 8)
        output_bits_val = task_specific_kwargs.get('output_bits', None)
        
        # Handle tasks that compute output_bits dynamically
        if output_bits_val is None:
            if task_name == "add":
                output_bits_val = input_bits_val // 2 + 1
            elif task_name == "popcount":
                output_bits_val = (input_bits_val.bit_length() - 1) + 1
            elif task_name == "parity":
                output_bits_val = 1
            else:
                output_bits_val = input_bits_val  # Default to same as input
        
        # Call task function with a wrapper that generates only for specific indices
        # Filter out input_bits and output_bits from kwargs since they're passed positionally
        filtered_kwargs = {k: v for k, v in task_specific_kwargs.items() 
                          if k not in ('input_bits', 'output_bits')}
        x_data, y_data = _generate_task_samples_for_indices(
            task_name, indices_python, input_bits_val, output_bits_val, **filtered_kwargs
        )
    else:
        # Generate full dataset (case_n is manageable)
        x_data, y_data = TASKS[task_name](case_n, **task_specific_kwargs)
    
    return x_data, y_data


def _generate_task_samples_for_indices(task_name, x_ints_list, input_bits, output_bits, **kwargs):
    """
    Generate task samples for specific integer inputs (without generating full range).
    
    This is used when we need to sample from a huge case_n without materializing all samples.
    The task_name must match one of the keys in TASKS dictionary.
    
    Args:
        x_ints_list: List of Python ints (can be very large, e.g., up to 2^50)
    """
    # Convert Python int list to numpy array for computation, then to JAX
    # Do computations element-wise to handle large integers
    import numpy as np
    
    # Convert to numpy array first (numpy can handle int64, or we'll use object dtype)
    # For very large integers, we'll compute in Python and convert results
    x_ints_np = np.array(x_ints_list, dtype=object)  # object dtype preserves Python ints
    
    # Compute outputs using vectorized operations where possible
    # For tasks that need bitwise ops, we'll do them element-wise
    if task_name == "binary_multiply":
        half = input_bits // 2
        y_ints_list = [(x & ((1 << half) - 1)) * (x >> half) for x in x_ints_list]
    elif task_name == "and":
        half = input_bits // 2
        y_ints_list = [(x & ((1 << half) - 1)) & (x >> half) for x in x_ints_list]
    elif task_name == "xor":
        half = input_bits // 2
        y_ints_list = [(x & ((1 << half) - 1)) ^ (x >> half) for x in x_ints_list]
    elif task_name == "add":
        half = input_bits // 2
        y_ints_list = [(x & ((1 << half) - 1)) + (x >> half) for x in x_ints_list]
    elif task_name == "parity":
        y_ints_list = []
        for x in x_ints_list:
            y = x
            for i in range(1, input_bits):
                y ^= y >> i
            y_ints_list.append(y & 1)
    elif task_name == "reverse":
        y_ints_list = []
        for x in x_ints_list:
            y = 0
            for i in range(input_bits):
                y = y | (((x >> i) & 1) << (input_bits - 1 - i))
            y_ints_list.append(y)
    elif task_name == "copy":
        y_ints_list = list(x_ints_list)
    elif task_name == "gray":
        y_ints_list = [x ^ (x >> 1) for x in x_ints_list]
    elif task_name == "popcount":
        # Sum bits to get count - compute in Python
        y_ints_list = [sum((x >> i) & 1 for i in range(input_bits)) for x in x_ints_list]
    elif task_name == "identity":
        y_ints_list = list(x_ints_list)
    else:
        # Should not happen if task_name was validated in get_task_data
        raise ValueError(f"Task {task_name} not supported for index-based generation")
    
    # Unpack to bit representations in Python (to handle large integers correctly)
    # Then convert to JAX arrays
    def unpack_python(x, bit_n):
        """Unpack integer to bits in Python, return as list"""
        return [float((x >> i) & 1) for i in range(bit_n)]
    
    # Unpack x_ints (input bits)
    x_data_list = [unpack_python(x, input_bits) for x in x_ints_list]
    x_data = jp.array(x_data_list, dtype=jp.float32)
    
    # Unpack y_ints (output bits) - results are smaller so int32 should work
    y_ints = jp.array(y_ints_list, dtype=jp.int32)
    
    # Handle output_bits for tasks that might need different output sizes
    if task_name == "parity":
        y_data = unpack(y_ints, 1)
    elif task_name == "add" and output_bits is not None:
        # binary_add can have different output_bits
        y_data = unpack(y_ints, output_bits)
    elif task_name == "popcount" and output_bits is not None:
        y_data = unpack(y_ints, output_bits)
    else:
        y_data = unpack(y_ints, output_bits)
    
    return x_data, y_data
