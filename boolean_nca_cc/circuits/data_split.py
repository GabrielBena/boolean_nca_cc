"""
Data splitting utilities for input combination train/test splits.

Provides reproducible splitting of input combinations for evaluating
circuit generalization to held-out input patterns.
"""

import jax
import jax.numpy as jp
from typing import Tuple
import logging

log = logging.getLogger(__name__)

def split_input_combinations(
    x_data: jp.ndarray,
    y_data: jp.ndarray,
    train_fraction: float = 0.8,
    seed: int = 42,
    shuffle: bool = True,
) -> Tuple[jp.ndarray, jp.ndarray, jp.ndarray, jp.ndarray]:
    """
    Split input combinations into train and test sets.
    
    Args:
        x_data: Input combinations [n_combinations, input_bits]
        y_data: Target outputs [n_combinations, output_bits]
        train_fraction: Fraction of combinations for training (default 0.8)
        seed: Random seed for reproducible shuffling
        shuffle: Whether to shuffle before splitting (default True)
        
    Returns:
        Tuple of (x_train, y_train, x_test, y_test)
        
    Example:
        >>> x_data, y_data = get_task_data("binary_multiply", 256, input_bits=8)
        >>> x_train, y_train, x_test, y_test = split_input_combinations(
        ...     x_data, y_data, train_fraction=0.8, seed=42
        ... )
        >>> x_train.shape[0]  # 204 (80% of 256)
        204
        >>> x_test.shape[0]   # 52 (20% of 256)
        52
    """
    if not 0.0 < train_fraction <= 1.0:
        raise ValueError(f"train_fraction must be between 0.0 and 1.0, got {train_fraction}")

    if x_data.shape[0] != y_data.shape[0]:
        raise ValueError(f"x_data and y_data must have the same number of combinations, got {x_data.shape[0]} and {y_data.shape[0]}")

    n_samples = x_data.shape[0]
    n_train = int(n_samples * train_fraction)
    
    # Ensure at least 1 sample in training set (unless train_fraction is exactly 0.0)
    if train_fraction > 0.0 and n_train == 0:
        n_train = 1
        log.warning(
            f"train_fraction={train_fraction} with {n_samples} samples resulted in 0 training samples. "
            f"Using minimum of 1 training sample instead."
        )

    indices = jp.arange(n_samples)

    if shuffle:
        key = jax.random.PRNGKey(seed)
        indices = jax.random.permutation(key, indices)
    
    
    train_indices = indices[:n_train]
    test_indices = indices[n_train:]

    x_train = x_data[train_indices]
    y_train = y_data[train_indices]
    x_test = x_data[test_indices]
    y_test = y_data[test_indices]

    return x_train, y_train, x_test, y_test