"""
Schedulers for training parameters like reset intervals and learning rates.

This module provides various scheduling functions for dynamically adjusting
training parameters over the course of training.
"""

import jax
import jax.numpy as jp
import optax
from typing import Dict, Any, Optional, Tuple


def get_learning_rate_schedule(
    lr_scheduler: str,
    learning_rate: float,
    epochs: int,
    lr_scheduler_params: Optional[Dict[str, Any]] = None,
) -> optax.Schedule:
    """
    Create a learning rate schedule based on configuration.

    Args:
        lr_scheduler: Type of scheduler ("constant", "exponential", "cosine", "linear_warmup")
        learning_rate: Base learning rate
        epochs: Total number of training epochs
        lr_scheduler_params: Optional parameters for the scheduler

    Returns:
        An optax Schedule object
    """
    if lr_scheduler_params is None:
        lr_scheduler_params = {}

    if lr_scheduler == "constant":
        return optax.constant_schedule(learning_rate)

    elif lr_scheduler == "exponential":
        return optax.exponential_decay(
            init_value=learning_rate,
            transition_steps=lr_scheduler_params.get("transition_steps", epochs),
            decay_rate=lr_scheduler_params.get("decay_rate", 0.9),
        )

    elif lr_scheduler == "cosine":
        return optax.cosine_decay_schedule(
            init_value=learning_rate,
            decay_steps=lr_scheduler_params.get("decay_steps", epochs),
            alpha=lr_scheduler_params.get("alpha", 0.0),
            exponent=lr_scheduler_params.get("exponent", 1.0),
        )

    elif lr_scheduler == "linear_warmup":
        # Combine warmup with another schedule (e.g., cosine)
        warmup_steps = lr_scheduler_params.get(
            "warmup_steps", epochs // lr_scheduler_params.get("warmup_steps_factor", 10)
        )
        print(f"Warmup steps: {warmup_steps}")
        target_schedule = optax.cosine_decay_schedule(  # Default to cosine after warmup
            init_value=learning_rate,
            decay_steps=epochs - warmup_steps,
            alpha=lr_scheduler_params.get("alpha", 0.0),
            exponent=lr_scheduler_params.get("exponent", 1.0),
        )
        return optax.join_schedules(
            [
                optax.linear_schedule(
                    lr_scheduler_params.get("lr_warmup_start", 0.0),
                    learning_rate,
                    warmup_steps,
                ),
                target_schedule,
            ],
            [warmup_steps],
        )
    else:
        raise ValueError(f"Unknown lr_scheduler: {lr_scheduler}")


def get_reset_interval_scheduler(
    schedule_config: Dict[str, Any], total_epochs: int
) -> callable:
    """
    Create a reset interval scheduler function based on configuration.

    Args:
        schedule_config: Configuration dictionary containing:
            - enabled: Whether to use scheduling
            - type: Type of schedule ("constant", "linear", "exponential")
            - initial_interval: Starting interval value
            - final_interval: Ending interval value
            - decay_rate: For exponential decay
            - transition_epochs: For linear decay (if None, uses total_epochs)
        total_epochs: Total number of training epochs

    Returns:
        A function that takes epoch number and returns the current interval
    """
    if not schedule_config.get("enabled", False):
        # Return constant scheduler if disabled
        base_interval = schedule_config.get("initial_interval", 128)
        return lambda epoch: base_interval

    schedule_type = schedule_config.get("type", "constant")
    initial_interval = schedule_config.get("initial_interval", 10)
    final_interval = schedule_config.get("final_interval", 500)

    if schedule_type == "constant":
        return lambda epoch: initial_interval

    elif schedule_type == "linear":
        # Linear interpolation from initial to final over transition_epochs
        transition_epochs = schedule_config.get("transition_epochs", total_epochs)
        transition_epochs = (
            transition_epochs if transition_epochs is not None else total_epochs
        )

        def linear_scheduler(epoch: int) -> int:
            # Clamp epoch to transition period
            progress = min(epoch / max(transition_epochs, 1), 1.0)
            current_interval = initial_interval + progress * (
                final_interval - initial_interval
            )
            return int(jp.round(current_interval))

        return linear_scheduler

    elif schedule_type == "exponential":
        # Exponential decay: interval(t) = initial * decay_rate^t + final * (1 - decay_rate^t)
        # This ensures it asymptotically approaches final_interval
        decay_rate = schedule_config.get("decay_rate", 0.95)

        def exponential_scheduler(epoch: int) -> int:
            # Exponential interpolation from initial to final
            decay_factor = decay_rate**epoch
            current_interval = (
                final_interval + (initial_interval - final_interval) * decay_factor
            )
            return int(jp.round(current_interval))

        return exponential_scheduler

    else:
        raise ValueError(f"Unknown schedule type: {schedule_type}")


def get_message_steps_scheduler(
    schedule_config: Dict[str, Any], total_epochs: int
) -> callable:
    """
    Create a message steps scheduler function based on configuration.

    This implements curriculum learning by starting with fewer message steps
    (easier gradients) and gradually increasing to more steps.

    Args:
        schedule_config: Configuration dictionary containing:
            - enabled: Whether to use scheduling
            - type: Type of schedule ("constant", "linear", "exponential", "step")
            - initial_steps: Starting number of message steps (should be small)
            - final_steps: Final number of message steps
            - transition_epochs: For linear/step schedules (if None, uses total_epochs)
            - growth_rate: For exponential growth
            - step_intervals: For step schedule - list of epochs where to increase steps
            - step_values: For step schedule - list of step values (must match step_intervals)
        total_epochs: Total number of training epochs

    Returns:
        A function that takes epoch number and returns the current number of message steps
    """
    if not schedule_config.get("enabled", False):
        # Return constant scheduler if disabled
        base_steps = schedule_config.get("initial_steps", 20)
        return lambda epoch: base_steps

    schedule_type = schedule_config.get("type", "constant")
    initial_steps = schedule_config.get("initial_steps", 5)
    final_steps = schedule_config.get("final_steps", 20)

    if schedule_type == "constant":
        return lambda epoch: initial_steps

    elif schedule_type == "linear":
        # Linear interpolation from initial to final over transition_epochs
        transition_epochs = schedule_config.get("transition_epochs", total_epochs)
        transition_epochs = (
            transition_epochs if transition_epochs is not None else total_epochs
        )

        def linear_scheduler(epoch: int) -> int:
            # Clamp epoch to transition period
            progress = min(epoch / max(transition_epochs, 1), 1.0)
            current_steps = initial_steps + progress * (final_steps - initial_steps)
            return int(jp.round(current_steps))

        return linear_scheduler

    elif schedule_type == "exponential":
        # Exponential growth: steps(t) = initial * growth_rate^t, clamped to final_steps
        growth_rate = schedule_config.get("growth_rate", 1.02)  # Slow growth by default

        def exponential_scheduler(epoch: int) -> int:
            current_steps = initial_steps * (growth_rate**epoch)
            # Clamp to final_steps
            current_steps = min(current_steps, final_steps)
            return int(jp.round(current_steps))

        return exponential_scheduler

    elif schedule_type == "step":
        # Step-wise increase at specific epochs
        step_intervals = schedule_config.get(
            "step_intervals",
            [total_epochs // 4, total_epochs // 2, 3 * total_epochs // 4],
        )
        step_values = schedule_config.get("step_values", None)

        if step_values is None:
            # Auto-generate step values if not provided
            num_steps = len(step_intervals) + 1
            step_values = [
                int(initial_steps + i * (final_steps - initial_steps) / (num_steps - 1))
                for i in range(num_steps)
            ]

        if len(step_values) != len(step_intervals) + 1:
            raise ValueError(
                "step_values must have one more element than step_intervals"
            )

        def step_scheduler(epoch: int) -> int:
            current_steps = step_values[0]  # Start with first value
            for i, interval_epoch in enumerate(step_intervals):
                if epoch >= interval_epoch:
                    current_steps = step_values[i + 1]
                else:
                    break
            return current_steps

        return step_scheduler

    else:
        raise ValueError(f"Unknown schedule type: {schedule_type}")


def should_reset_pool(epoch: int, reset_interval: int, last_reset_epoch: int) -> bool:
    """
    Determine if the pool should be reset at the current epoch.

    Args:
        epoch: Current epoch number
        reset_interval: Current reset interval
        last_reset_epoch: Epoch when pool was last reset

    Returns:
        True if pool should be reset
    """
    if epoch == 0:
        return False  # Don't reset on first epoch

    epochs_since_reset = epoch - last_reset_epoch
    return epochs_since_reset >= reset_interval


def get_current_reset_interval(
    epoch: int,
    schedule_config: Dict[str, Any],
    total_epochs: int,
    base_interval: int = 128,
) -> int:
    """
    Get the current reset interval for the given epoch.

    Args:
        epoch: Current epoch number
        schedule_config: Reset interval schedule configuration
        total_epochs: Total number of training epochs
        base_interval: Fallback interval if scheduling is disabled

    Returns:
        Current reset interval
    """
    if not schedule_config.get("enabled", False):
        return base_interval

    scheduler = get_reset_interval_scheduler(schedule_config, total_epochs)
    return scheduler(epoch)


def get_current_message_steps_and_batch_size(
    epoch: int,
    schedule_config: Dict[str, Any],
    total_epochs: int,
    base_steps: int = 20,
    base_batch_size: int = 16,
) -> Tuple[int, int]:
    """
    Get the current number of message steps and corresponding batch size for the given epoch.

    To maintain constant memory usage, meta_batch_size * n_message_steps = constant.
    As message steps increase, batch size decreases proportionally.

    Args:
        epoch: Current epoch number
        schedule_config: Message steps schedule configuration
        total_epochs: Total number of training epochs
        base_steps: Fallback steps if scheduling is disabled
        constant_product: The constant product (meta_batch_size * n_message_steps)
                         If None, calculated from base values

    Returns:
        Tuple of (current_message_steps, current_batch_size)
    """
    if not schedule_config.get("enabled", False):
        return base_steps, base_batch_size

    # Get current message steps
    scheduler = get_message_steps_scheduler(schedule_config, total_epochs)
    current_steps = scheduler(epoch)

    # Calculate corresponding batch size to maintain constant product
    constant_product = schedule_config.get("constant_product", None)
    current_batch_size = (
        max(1, constant_product // current_steps)
        if constant_product is not None
        else base_batch_size
    )

    return current_steps, current_batch_size


def get_step_beta(
    loss_key: jax.random.PRNGKey,
    n_message_steps: int,
    training_progress: float = 0.0,
) -> jp.ndarray:
    """
    Sample a loss step using Beta distribution that shifts from early to late steps.

    Args:
        loss_key: JAX random key for sampling
        n_message_steps: Total number of message passing steps
        training_progress: Float in [0.0, 1.0] indicating training progress

    Returns:
        Selected step index as integer

    Distribution evolution during training:

    EARLY TRAINING (progress ≈ 0.0):          MID TRAINING (progress ≈ 0.5):
    ████                                         ████
    ███                                          ██████
    ██                                           ████████
    █                                            ████████
    █                                            ██████
    █                                            ████
    0────────────────────100                     0────────────────────100
    (favors early steps)                         (balanced)

    LATE TRAINING (progress ≈ 1.0):

    █
    █
    █
    ██
    ███
    ████
    0────────────────────100
    (favors late steps)
    """
    training_progress = jp.clip(training_progress, 0.01, 0.99)  # Avoid extreme values

    # Parameters for beta distribution
    # Early: alpha > beta (left skewed, favors early steps)
    # Late: alpha < beta (right skewed, favors later steps)

    max_concentration = 4.0  # Controls how concentrated the distribution is
    min_concentration = 0.5

    beta = (
        max_concentration * (1.0 - training_progress)
        + min_concentration * training_progress
    )
    alpha = (
        min_concentration * (1.0 - training_progress)
        + max_concentration * training_progress
    )

    # Sample from beta distribution
    beta_sample = jax.random.beta(loss_key, alpha, beta)

    # Map to step range
    step = beta_sample * (n_message_steps - 1)

    return jp.round(step).astype(jp.int32).clip(1, n_message_steps - 1)
