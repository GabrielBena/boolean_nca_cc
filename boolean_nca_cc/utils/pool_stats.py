"""
Utility functions for analyzing pool statistics and update patterns.
"""

from dataclasses import dataclass
from typing import Any


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


def analyze_config_pool_updates(config: dict[str, Any]) -> PoolUpdateStats:
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


def get_update_recommendations(stats: PoolUpdateStats) -> dict[str, str]:
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


def compute_pool_parameter(
    target_expected_updates: float,
    solve_for: str,
    pool_size: int | None = None,
    batch_size: int | None = None,
    n_message_steps: int | None = None,
    reset_interval: int | None = None,
    reset_fraction: float | None = None,
) -> float:
    """
    Compute one pool parameter based on target expected updates and other parameters.

    This function solves the equation:
    expected_updates = (batch_size / pool_size) * n_message_steps * (reset_interval / reset_fraction)

    Args:
        target_expected_updates: Desired expected number of updates per circuit
        solve_for: Parameter to solve for ('pool_size', 'batch_size', 'n_message_steps',
                  'reset_interval', or 'reset_fraction')
        pool_size: Total number of circuits in pool (required unless solving for this)
        batch_size: Number of circuits sampled per epoch (required unless solving for this)
        n_message_steps: Number of message passing steps per batch (required unless solving for this)
        reset_interval: Number of epochs between resets (required unless solving for this)
        reset_fraction: Fraction of pool reset at each event (required unless solving for this)

    Returns:
        The computed value for the specified parameter

    Raises:
        ValueError: If required parameters are missing or invalid

    Examples:
        # Find reset fraction needed for 100 expected updates
        reset_frac = compute_pool_parameter(
            target_expected_updates=100,
            solve_for="reset_fraction",
            pool_size=1024,
            batch_size=128,
            n_message_steps=3,
            reset_interval=32
        )

        # Find batch size needed for 50 expected updates
        batch_sz = compute_pool_parameter(
            target_expected_updates=50,
            solve_for="batch_size",
            pool_size=1024,
            n_message_steps=3,
            reset_interval=32,
            reset_fraction=0.125
        )
    """
    # Validate target
    if target_expected_updates <= 0:
        raise ValueError("target_expected_updates must be positive")

    # Check which parameter we're solving for and validate inputs
    if solve_for == "reset_fraction":
        if any(x is None for x in [pool_size, batch_size, n_message_steps, reset_interval]):
            raise ValueError(
                "pool_size, batch_size, n_message_steps, and reset_interval are required when solving for reset_fraction"
            )

        # reset_fraction = (batch_size * n_message_steps * reset_interval) / (pool_size * expected_updates)
        result = (batch_size * n_message_steps * reset_interval) / (
            pool_size * target_expected_updates
        )

        if result <= 0 or result > 1:
            raise ValueError(f"Computed reset_fraction ({result:.4f}) is not in valid range (0, 1]")

        return result

    elif solve_for == "reset_interval":
        if any(x is None for x in [pool_size, batch_size, n_message_steps, reset_fraction]):
            raise ValueError(
                "pool_size, batch_size, n_message_steps, and reset_fraction are required when solving for reset_interval"
            )

        # reset_interval = (pool_size * expected_updates * reset_fraction) / (batch_size * n_message_steps)
        result = (pool_size * target_expected_updates * reset_fraction) / (
            batch_size * n_message_steps
        )

        if result <= 0:
            raise ValueError(f"Computed reset_interval ({result:.2f}) must be positive")

        return result

    elif solve_for == "batch_size":
        if any(x is None for x in [pool_size, n_message_steps, reset_interval, reset_fraction]):
            raise ValueError(
                "pool_size, n_message_steps, reset_interval, and reset_fraction are required when solving for batch_size"
            )

        # batch_size = (pool_size * expected_updates * reset_fraction) / (n_message_steps * reset_interval)
        result = (pool_size * target_expected_updates * reset_fraction) / (
            n_message_steps * reset_interval
        )

        if result <= 0 or result > pool_size:
            raise ValueError(
                f"Computed batch_size ({result:.2f}) is not in valid range (0, {pool_size}]"
            )

        return result

    elif solve_for == "n_message_steps":
        if any(x is None for x in [pool_size, batch_size, reset_interval, reset_fraction]):
            raise ValueError(
                "pool_size, batch_size, reset_interval, and reset_fraction are required when solving for n_message_steps"
            )

        # n_message_steps = (pool_size * expected_updates * reset_fraction) / (batch_size * reset_interval)
        result = (pool_size * target_expected_updates * reset_fraction) / (
            batch_size * reset_interval
        )

        if result <= 0:
            raise ValueError(f"Computed n_message_steps ({result:.2f}) must be positive")

        return result

    elif solve_for == "pool_size":
        if any(x is None for x in [batch_size, n_message_steps, reset_interval, reset_fraction]):
            raise ValueError(
                "batch_size, n_message_steps, reset_interval, and reset_fraction are required when solving for pool_size"
            )

        # pool_size = (batch_size * n_message_steps * reset_interval) / (expected_updates * reset_fraction)
        result = (batch_size * n_message_steps * reset_interval) / (
            target_expected_updates * reset_fraction
        )

        if result <= 0 or result < batch_size:
            raise ValueError(
                f"Computed pool_size ({result:.2f}) must be at least batch_size ({batch_size})"
            )

        return result

    else:
        raise ValueError(
            f"Unknown parameter to solve for: {solve_for}. Must be one of: 'pool_size', 'batch_size', 'n_message_steps', 'reset_interval', 'reset_fraction'"
        )


def suggest_pool_configurations(
    target_expected_updates: float,
    constraints: dict[str, Any] | None = None,
    n_suggestions: int = 3,
) -> list[dict[str, Any]]:
    """
    Suggest multiple pool configurations that achieve target expected updates.

    Args:
        target_expected_updates: Desired expected number of updates per circuit
        constraints: Optional dictionary with fixed parameter values and ranges
        n_suggestions: Number of different configurations to suggest

    Returns:
        List of configuration dictionaries

    Example:
        suggestions = suggest_pool_configurations(
            target_expected_updates=100,
            constraints={
                'pool_size': 1024,  # Fixed value
                'batch_size_range': (64, 256),  # Range
                'reset_interval_range': (16, 64)
            }
        )
    """
    if constraints is None:
        constraints = {}

    suggestions = []

    # Default ranges if not specified
    default_ranges = {
        "pool_size": [512, 1024, 2048, 4096],
        "batch_size_range": (32, 512),
        "n_message_steps_range": (1, 10),
        "reset_interval_range": (8, 128),
        "reset_fraction_range": (0.01, 0.5),
    }

    # Generate different base configurations
    base_configs = [
        # Conservative: low batch size, moderate reset
        {"batch_size": 64, "n_message_steps": 3, "reset_interval": 32},
        # Aggressive: high batch size, frequent reset
        {"batch_size": 256, "n_message_steps": 5, "reset_interval": 16},
        # Balanced: moderate values
        {"batch_size": 128, "n_message_steps": 3, "reset_interval": 24},
    ]

    pool_sizes = constraints.get("pool_size", default_ranges["pool_size"])
    if isinstance(pool_sizes, int):
        pool_sizes = [pool_sizes]
    elif not isinstance(pool_sizes, list):
        pool_sizes = default_ranges["pool_size"]

    for base_config in base_configs[:n_suggestions]:
        for pool_size in pool_sizes:
            try:
                # Use provided values or defaults
                config = {"pool_size": pool_size}

                # Apply constraints
                for key in ["batch_size", "n_message_steps", "reset_interval"]:
                    if key in constraints:
                        config[key] = constraints[key]
                    else:
                        config[key] = base_config[key]

                # Compute reset fraction to achieve target
                reset_fraction = compute_pool_parameter(
                    target_expected_updates=target_expected_updates,
                    solve_for="reset_fraction",
                    **config,
                )

                config["reset_fraction"] = reset_fraction

                # Verify the configuration
                stats = calculate_expected_pool_updates(**config)
                config["computed_expected_updates"] = stats.expected_updates
                config["selection_probability"] = stats.selection_probability
                config["expected_lifetime_epochs"] = stats.expected_lifetime_epochs

                suggestions.append(config)
                break  # Found valid config for this base

            except ValueError:
                continue  # Try next pool size or base config

    return suggestions
