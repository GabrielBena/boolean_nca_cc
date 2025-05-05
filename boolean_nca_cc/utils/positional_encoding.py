"""
Positional encoding utilities for graph neural networks.

This module provides functions for generating positional encodings
for nodes in a graph neural network.
"""

import jax.numpy as jp


def get_positional_encoding(indices: jp.ndarray, dim: int, max_val: float = 10000.0):
    """
    Generate sinusoidal positional encodings for node positions.

    Args:
        indices: 1D array of integer positions (e.g., layer indices)
        dim: Dimension of the positional encoding vector (must be even)
        max_val: Maximum value for the denominator in frequency calculation

    Returns:
        Array of shape (len(indices), dim) containing positional encodings
    """
    if dim % 2 != 0:
        raise ValueError(f"Positional encoding dimension must be even, got {dim}")

    # Ensure indices are float for calculations
    positions = indices.astype(jp.float32)[:, None]  # Shape: [num_indices, 1]

    # Calculate frequency division term
    div_term = jp.exp(jp.arange(0, dim, 2, dtype=jp.float32) * -(jp.log(max_val) / dim))

    # Initialize positional encoding matrix
    pe = jp.zeros((indices.shape[0], dim), dtype=jp.float32)

    # Calculate sine and cosine components
    pe = pe.at[:, 0::2].set(jp.sin(positions * div_term))
    pe = pe.at[:, 1::2].set(jp.cos(positions * div_term))

    return pe
