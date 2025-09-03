"""
Utility functions for analyzing pool statistics and update patterns.
"""

from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class PoolUpdateStats:
    """Statistics about pool update patterns."""

    expected_updates: float
    selection_probability: float
    update_rate_per_epoch: float
    expected_lifetime_epochs: float
    reset_frequency: float


def calculate_expected_pool_updates(
    pool_size: int,
    batch_size: int,
    n_message_steps: int,
    reset_interval: int,
    reset_fraction: float,
) -> PoolUpdateStats:
    """
    Calculate the expected number of updates a circuit receives in the pool.

    This calculation assumes:
    1. Circuits are sampled uniformly at random for each batch
    2. Resets happen at regular intervals with fixed probability
    3. Long-term steady-state behavior

    Args:
        pool_size: Total number of circuits in pool
        batch_size: Number of circuits sampled per epoch
        n_message_steps: Number of message passing steps per batch execution
        reset_interval: Number of epochs between reset events
        reset_fraction: Fraction of pool reset at each reset event

    Returns:
        PoolUpdateStats object with detailed statistics

    Raises:
        ValueError: If parameters are invalid
    """
    # Validate inputs
    if pool_size <= 0:
        raise ValueError("pool_size must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if batch_size > pool_size:
        raise ValueError("batch_size cannot exceed pool_size")
    if n_message_steps <= 0:
        raise ValueError("n_message_steps must be positive")
    if reset_interval <= 0:
        raise ValueError("reset_interval must be positive")
    if not (0 < reset_fraction <= 1):
        raise ValueError("reset_fraction must be between 0 and 1")

    # Calculate basic update dynamics
    selection_probability = batch_size / pool_size
    update_rate_per_epoch = selection_probability * n_message_steps

    # Calculate expected lifetime before reset
    # This is based on geometric distribution: E[lifetime] = 1/p where p is reset probability per cycle
    # Since resets happen every reset_interval epochs with probability reset_fraction,
    # the expected lifetime is reset_interval / reset_fraction
    expected_lifetime_epochs = reset_interval / reset_fraction

    # Calculate expected total updates
    expected_updates = update_rate_per_epoch * expected_lifetime_epochs

    # Reset frequency (resets per epoch)
    reset_frequency = reset_fraction / reset_interval

    return PoolUpdateStats(
        expected_updates=expected_updates,
        selection_probability=selection_probability,
        update_rate_per_epoch=update_rate_per_epoch,
        expected_lifetime_epochs=expected_lifetime_epochs,
        reset_frequency=reset_frequency,
    )


def analyze_config_pool_updates(config: Dict[str, Any]) -> PoolUpdateStats:
    """
    Analyze pool update statistics from a training configuration dictionary.

    Args:
        config: Configuration dictionary with pool and training parameters

    Returns:
        PoolUpdateStats object
    """
    return calculate_expected_pool_updates(
        pool_size=config["pool"]["size"],
        batch_size=config["training"]["meta_batch_size"],
        n_message_steps=config["training"]["n_message_steps"],
        reset_interval=config["pool"]["reset_interval"],
        reset_fraction=config["pool"]["reset_fraction"],
    )


def get_update_recommendations(stats: PoolUpdateStats) -> Dict[str, str]:
    """
    Get recommendations based on pool update statistics.

    Args:
        stats: PoolUpdateStats object

    Returns:
        Dictionary with recommendations
    """
    recommendations = {}

    updates = stats.expected_updates

    if updates < 5:
        recommendations["update_count"] = (
            "Very low update count. Consider increasing message steps, "
            "decreasing reset frequency, or increasing batch size."
        )
        recommendations["concern_level"] = "high"
    elif updates < 20:
        recommendations["update_count"] = (
            "Low update count. May need more updates for complex tasks."
        )
        recommendations["concern_level"] = "medium"
    elif updates < 100:
        recommendations["update_count"] = "Good update count for most tasks."
        recommendations["concern_level"] = "low"
    elif updates < 300:
        recommendations["update_count"] = (
            "High update count. Good for complex learning but watch for overfitting."
        )
        recommendations["concern_level"] = "medium"
    else:
        recommendations["update_count"] = (
            "Very high update count. May lead to overfitting on individual circuits."
        )
        recommendations["concern_level"] = "high"

    # Selection probability recommendations
    if stats.selection_probability < 0.01:
        recommendations["selection"] = (
            "Very low selection probability. Circuits may go many epochs without updates."
        )
    elif stats.selection_probability > 0.5:
        recommendations["selection"] = (
            "High selection probability. Most circuits get frequent updates."
        )

    # Reset frequency recommendations
    if stats.reset_frequency < 0.01:
        recommendations["reset"] = (
            "Low reset frequency. Pool may become stagnant over long training."
        )
    elif stats.reset_frequency > 0.1:
        recommendations["reset"] = "High reset frequency. May disrupt learning progress."

    return recommendations
