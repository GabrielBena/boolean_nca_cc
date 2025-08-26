"""
Boolean Circuit Tasks

Pure functional implementation of boolean tasks for training circuits.
"""

import jax.numpy as jp
import numpy as np
import PIL.Image
import PIL.ImageDraw


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


def text_task(case_n, input_bits=8, output_bits=8, text=None):
    """
    Text rendering task - renders text to binary pattern.
    Automatically selects appropriate text length based on aspect ratio if text is None.

    Args:
        case_n: Number of test cases (should be 2^input_bits)
        input_bits: Number of input bits
        output_bits: Number of output bits (height of rendered text)
        text: Text string to render (if None, auto-selects based on aspect ratio)

    Returns:
        input_x: Binary input patterns (case_n, input_bits)
        y0: Binary text pattern (case_n, output_bits)
    """
    # Auto-select text based on aspect ratio if not provided
    if text is None:
        # Create a temporary image to test text fitting
        temp_im = PIL.Image.new("L", (case_n, output_bits))
        temp_draw = PIL.ImageDraw.Draw(temp_im)

        # Text options ordered from longest to shortest
        text_options = [
            "Hello Neural CA! Self-Organizing Circuits are Real!",
            "Hello Neural CA! Self-Organizing!",
            "Neural CA Circuits!",
            "Neural CA",
            "NCA",
            "N",
        ]

        # Find the longest text that fits
        text = "N"  # fallback
        for candidate_text in text_options:
            # Use anchor-based bounding box for consistent measurement
            temp_center_x = case_n // 2
            temp_center_y = output_bits // 2
            bbox = temp_draw.textbbox((temp_center_x, temp_center_y), candidate_text, anchor="mm")

            # Calculate actual text bounds
            text_left = bbox[0]
            text_top = bbox[1]
            text_right = bbox[2]
            text_bottom = bbox[3]

            # Check if text fits with some margin
            if (
                text_left >= 0
                and text_top >= 0
                and text_right <= case_n
                and text_bottom <= output_bits
            ):
                text = candidate_text
                break

    # Create input patterns (sequential integers)
    x = jp.arange(case_n)
    input_x = unpack(x, input_bits)

    # Create PIL image for text rendering
    # Use case_n as width and output_bits as height
    im = PIL.Image.new("L", (case_n, output_bits))
    draw = PIL.ImageDraw.Draw(im)

    # Use anchor-based centering for more reliable positioning
    x_center = case_n // 2
    y_center = output_bits // 2

    # Render text centered using middle-middle anchor
    draw.text((x_center, y_center), text, fill=255, anchor="mm")

    # Convert to binary array and transpose to match expected shape
    text_array = np.array(im) > 100  # Threshold for binary conversion
    y0 = jp.float32(text_array.T)  # Transpose so shape is (case_n, output_bits)

    return input_x, y0


def noise_task(case_n, input_bits=8, output_bits=8, noise_p=0.5, seed=None):
    """
    Random noise task - generates random binary patterns.

    Args:
        case_n: Number of test cases (should be 2^input_bits)
        input_bits: Number of input bits
        output_bits: Number of output bits
        noise_p: Probability threshold for binary conversion (0.0-1.0)
        seed: Random seed for reproducibility (optional)

    Returns:
        input_x: Binary input patterns (case_n, input_bits)
        y0: Random binary patterns (case_n, output_bits)
    """
    # Create input patterns (sequential integers)
    x = jp.arange(case_n)
    input_x = unpack(x, input_bits)

    # Generate random noise with optional seed
    if seed is not None:
        np.random.seed(seed)

    noise = np.random.rand(case_n, output_bits)
    y0 = jp.float32(noise < noise_p)

    return input_x, y0


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
    "text": text_task,
    "noise": noise_task,
}


def get_task_data(task_name, case_n, **kwargs):
    """Get a task by name and generate its data"""
    if task_name not in TASKS:
        raise ValueError(f"Unknown task: {task_name}. Available: {list(TASKS.keys())}")

    task_specific_kwargs = {
        k: v
        for k, v in kwargs.items()
        if k in TASKS[task_name].__code__.co_varnames[: TASKS[task_name].__code__.co_argcount]
    }
    return TASKS[task_name](case_n, **task_specific_kwargs)
