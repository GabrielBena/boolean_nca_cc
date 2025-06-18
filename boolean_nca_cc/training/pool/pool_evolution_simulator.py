#!/usr/bin/env python3
"""
Pool Evolution Simulator

This utility simulates the evolution of pool reset steps over training epochs
to visualize how the pool dynamics work with different reset strategies.
"""

import os
import jax
import jax.numpy as jp
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Any
import hydra
from omegaconf import DictConfig, OmegaConf
import logging
from tqdm.auto import tqdm
from boolean_nca_cc.training.schedulers import (
    get_step_beta,
    should_reset_pool,
    get_current_reset_interval,
)
from functools import partial

log = logging.getLogger(__name__)


def _generate_step_advances_beta(
    key: jax.random.PRNGKey,
    n_updates: int,
    n_message_steps: int,
    training_progress: float,
) -> jp.ndarray:
    """Generate step advances using beta distribution."""
    step_keys = jax.random.split(key, n_updates)
    step_advances = jax.vmap(
        lambda k: get_step_beta(k, n_message_steps, training_progress)
    )(step_keys)
    return step_advances


def _generate_step_advances_fixed(n_updates: int, n_message_steps: int) -> jp.ndarray:
    """Generate fixed step advances."""
    return jp.full(n_updates, n_message_steps, dtype=jp.float32)


@jax.jit
def _update_pool_steps(
    pool_steps: jp.ndarray, update_idxs: jp.ndarray, step_advances: jp.ndarray
) -> jp.ndarray:
    """JIT-compiled function to update pool steps efficiently."""
    return pool_steps.at[update_idxs].add(step_advances)


@jax.jit
def _reset_circuits(pool_steps: jp.ndarray, reset_idxs: jp.ndarray) -> jp.ndarray:
    """JIT-compiled function to reset selected circuits."""
    return pool_steps.at[reset_idxs].set(0.0)


@jax.jit
def _compute_reset_probabilities(pool_steps: jp.ndarray) -> jp.ndarray:
    """JIT-compiled function to compute step-biased reset probabilities."""
    total_steps = jp.sum(pool_steps)
    return jp.where(
        total_steps > 0,
        pool_steps / total_steps,
        jp.ones_like(pool_steps) / len(pool_steps),
    )


def simulate_pool_reset_evolution(
    pool_size: int,
    epochs: int,
    n_message_steps: int,
    reset_pool_fraction: float,
    reset_pool_interval: int,
    meta_batch_size: int,
    reset_strategy: str = "steps_biased",
    use_beta_loss_step: bool = True,
    reset_interval_schedule: Dict = None,
    jump_size: int = 1,
    seed: int = 42,
) -> Dict[str, List]:
    """
    Simulate the evolution of pool reset steps over training epochs.

    Args:
        pool_size: Number of circuits in the pool
        epochs: Number of training epochs to simulate
        n_message_steps: Number of message passing steps per epoch
        reset_pool_fraction: Fraction of pool to reset when resetting
        reset_pool_interval: Base interval between pool resets
        meta_batch_size: Number of circuits sampled per epoch (like in real training)
        reset_strategy: Strategy for selecting circuits to reset ("steps_biased")
        use_beta_loss_step: Whether to use beta distribution for step advancement
        reset_interval_schedule: Optional scheduling for reset intervals
        jump_size: Number of epochs to simulate at once (for acceleration)
        seed: Random seed for reproducibility

    Returns:
        Dictionary with simulation results including:
        - epochs: List of epoch numbers
        - avg_steps: Average steps in pool at each epoch
        - reset_epochs: Epochs when resets occurred
        - reset_avg_steps: Average steps of reset circuits
        - pool_diversity: Diversity metric (std dev of steps)
    """

    # Initialize random key
    rng = jax.random.PRNGKey(seed)

    # Initialize pool with zero steps (all circuits start fresh)
    pool_steps = jp.zeros(pool_size, dtype=jp.float32)

    # Track metrics
    epochs_list = []
    avg_steps_list = []
    reset_epochs_list = []
    reset_avg_steps_list = []
    pool_diversity_list = []

    # Track last reset epoch for scheduling
    last_reset_epoch = -1

    # Initialize reset interval schedule if not provided
    if reset_interval_schedule is None:
        reset_interval_schedule = {"enabled": False}

    log.info(f"Starting pool evolution simulation:")
    log.info(f"  Pool size: {pool_size}")
    log.info(f"  Epochs: {epochs}")
    log.info(f"  Jump size: {jump_size}")
    log.info(f"  Meta batch size: {meta_batch_size}")
    log.info(f"  Message steps per epoch: {n_message_steps}")
    log.info(f"  Reset fraction: {reset_pool_fraction}")
    log.info(f"  Reset strategy: {reset_strategy}")
    log.info(f"  Use beta distribution: {use_beta_loss_step}")

    print(
        f"Starting pool evolution simulation for {epochs} epochs (jump size: {jump_size})"
    )

    # Calculate effective iterations based on jump size
    simulation_steps = list(range(0, epochs, jump_size))
    if simulation_steps[-1] != epochs - 1:
        simulation_steps.append(epochs - 1)  # Ensure we end at the final epoch

    for step_idx, epoch in enumerate(
        tqdm(simulation_steps, desc="Simulating pool evolution")
    ):
        # Determine how many epochs we're jumping (usually jump_size, except for last step)
        if step_idx < len(simulation_steps) - 1:
            current_jump = simulation_steps[step_idx + 1] - epoch
        else:
            current_jump = 1

        # Step 1: Sample and advance circuits (only meta_batch_size * current_jump circuits)
        # This simulates the realistic behavior where only some circuits are updated per epoch
        total_updates = meta_batch_size * current_jump

        # Ensure we don't try to update more circuits than exist in the pool
        total_updates = min(total_updates, pool_size)

        # Sample circuits to update (with replacement to match real training sampling)
        rng, sample_key = jax.random.split(rng)
        update_idxs = jax.random.choice(
            sample_key, pool_size, shape=(total_updates,), replace=True
        )

        # Calculate step advances for the sampled circuits using optimized functions
        if use_beta_loss_step:
            # Use beta distribution to determine actual steps taken
            training_progress = epoch / (epochs - 1) if epochs > 1 else 0.0
            rng, step_key = jax.random.split(rng)
            step_advances = _generate_step_advances_beta(
                step_key, total_updates, n_message_steps, training_progress
            )
        else:
            # Fixed step advancement
            step_advances = _generate_step_advances_fixed(
                total_updates, n_message_steps
            )

        # Update the sampled circuits efficiently using JIT-compiled function
        # Note: If a circuit is sampled multiple times, it gets updated multiple times
        # This is realistic since in training, a circuit could be sampled in consecutive epochs
        pool_steps = _update_pool_steps(pool_steps, update_idxs, step_advances)

        # Step 2: Check if we should reset the pool at this epoch
        current_reset_interval = get_current_reset_interval(
            epoch, reset_interval_schedule, epochs, reset_pool_interval
        )

        should_reset = should_reset_pool(
            epoch, current_reset_interval, last_reset_epoch
        )

        if should_reset:
            # Step 3: Select circuits to reset using steps_biased strategy
            rng, reset_key = jax.random.split(rng)

            # Calculate number of circuits to reset
            num_reset = max(1, int(reset_pool_fraction * pool_size))

            if reset_strategy == "steps_biased":
                # Selection biased by update steps (higher steps = higher probability)
                probs = _compute_reset_probabilities(pool_steps)
                reset_idxs = jax.random.choice(
                    reset_key, pool_size, shape=(num_reset,), replace=False, p=probs
                )
            else:
                # Fallback to uniform selection
                reset_idxs = jax.random.choice(
                    reset_key, pool_size, shape=(num_reset,), replace=False
                )

            # Calculate average steps of circuits being reset
            avg_steps_reset = float(jp.mean(pool_steps[reset_idxs]))

            # Reset selected circuits to 0 steps using JIT-compiled function
            pool_steps = _reset_circuits(pool_steps, reset_idxs)

            # Track reset information
            reset_epochs_list.append(epoch)
            reset_avg_steps_list.append(avg_steps_reset)
            last_reset_epoch = epoch

            log.debug(
                f"Epoch {epoch}: Reset {num_reset} circuits (avg steps: {avg_steps_reset:.2f})"
            )

        # Step 4: Record metrics
        avg_steps = float(jp.mean(pool_steps))
        diversity = float(jp.std(pool_steps))

        epochs_list.append(epoch)
        avg_steps_list.append(avg_steps)
        pool_diversity_list.append(diversity)

        if step_idx % max(1, len(simulation_steps) // 10) == 0 or step_idx < 3:
            log.info(
                f"Epoch {epoch}: Avg steps = {avg_steps:.2f}, Diversity = {diversity:.2f}, "
                f"Updated {total_updates} circuits"
            )

    return {
        "epochs": epochs_list,
        "avg_steps": avg_steps_list,
        "reset_epochs": reset_epochs_list,
        "reset_avg_steps": reset_avg_steps_list,
        "pool_diversity": pool_diversity_list,
        "final_pool_steps": pool_steps,
    }


def plot_pool_evolution(
    simulation_results: Dict[str, List],
    title: str = "Pool Reset Step Evolution",
    save_path: str = None,
):
    """
    Plot the results of pool evolution simulation.

    Args:
        simulation_results: Results from simulate_pool_reset_evolution
        title: Title for the plot
        save_path: Optional path to save the plot
    """

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(title, fontsize=16)

    epochs = simulation_results["epochs"]
    avg_steps = simulation_results["avg_steps"]
    reset_epochs = simulation_results["reset_epochs"]
    reset_avg_steps = simulation_results["reset_avg_steps"]
    pool_diversity = simulation_results["pool_diversity"]

    # Plot 1: Average steps over time
    ax1 = axes[0, 0]
    # smooth out the curve
    k = len(avg_steps) // 100
    avg_steps = np.convolve(avg_steps, np.ones(k) / k, mode="same")
    ax1.plot(epochs, avg_steps, "b-", linewidth=2, label="Pool Average Steps")

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Average Steps")
    ax1.set_title("Pool Average Steps Evolution")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Pool diversity (standard deviation of steps)
    ax2 = axes[0, 1]
    k = len(pool_diversity) // 100
    pool_diversity = np.convolve(pool_diversity, np.ones(k) / k, mode="same")
    ax2.plot(
        epochs, pool_diversity, "g-", linewidth=2, label="Pool Diversity (Std Dev)"
    )

    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Standard Deviation of Steps")
    ax2.set_title("Pool Diversity Evolution")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Reset frequency and average steps of reset circuits
    ax3 = axes[1, 0]
    if reset_epochs and reset_avg_steps:
        k = len(reset_epochs) // 100
        reset_epochs = np.convolve(reset_epochs, np.ones(k) / k, mode="same")
        reset_avg_steps = np.convolve(reset_avg_steps, np.ones(k) / k, mode="same")
        ax3.plot(reset_epochs, reset_avg_steps, "orange", alpha=0.5, linestyle="--")

    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Average Steps of Reset Circuits")
    ax3.set_title("Reset Circuit Characteristics")
    ax3.grid(True, alpha=0.3)

    # Plot 4: Distribution of final pool steps
    ax4 = axes[1, 1]
    final_steps = simulation_results["final_pool_steps"]
    ax4.hist(final_steps, bins=30, alpha=0.7, color="purple", edgecolor="black")
    ax4.axvline(
        jp.mean(final_steps),
        color="red",
        linestyle="--",
        label=f"Mean: {jp.mean(final_steps):.1f}",
    )
    ax4.axvline(
        jp.median(final_steps),
        color="orange",
        linestyle="--",
        label=f"Median: {jp.median(final_steps):.1f}",
    )

    ax4.set_xlabel("Steps")
    ax4.set_ylabel("Number of Circuits")
    ax4.set_title("Final Pool Step Distribution")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        log.info(f"Plot saved to {save_path}")

    plt.show()


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Main function that loads config and runs pool evolution simulation.
    """

    # Print configuration
    log.info("Loaded configuration:")
    log.info(OmegaConf.to_yaml(cfg))

    # Extract relevant parameters from config
    pool_size = cfg.pool.size
    epochs = cfg.training.epochs or 2**cfg.training.epochs_power_of_2
    n_message_steps = cfg.training.n_message_steps
    reset_pool_fraction = cfg.pool.reset_fraction
    reset_pool_interval = cfg.pool.reset_interval
    meta_batch_size = cfg.training.meta_batch_size
    use_beta_loss_step = cfg.training.use_beta_loss_step
    reset_interval_schedule = cfg.pool.reset_interval_schedule

    # Calculate good jump size for acceleration (aim for ~100-200 simulation steps)
    target_steps = 150
    jump_size = max(1, epochs // target_steps)

    log.info(
        f"Using jump size: {jump_size} (will simulate ~{epochs // jump_size} steps)"
    )

    # Run simulation
    log.info("Running pool evolution simulation...")
    results = simulate_pool_reset_evolution(
        pool_size=pool_size,
        epochs=epochs,
        n_message_steps=n_message_steps,
        reset_pool_fraction=reset_pool_fraction,
        reset_pool_interval=reset_pool_interval,
        meta_batch_size=meta_batch_size,
        reset_strategy="steps_biased",  # Focus on steps_biased as requested
        use_beta_loss_step=use_beta_loss_step,
        reset_interval_schedule=reset_interval_schedule,
        jump_size=jump_size,
        seed=cfg.seed,
    )

    # Create output directory
    output_dir = os.getcwd()
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Plot results
    plot_title = f"Pool Evolution Simulation (Steps Biased Reset)"
    save_path = os.path.join(plots_dir, "pool_evolution_simulation.png")

    plot_pool_evolution(results, title=plot_title, save_path=save_path)

    # Log summary statistics
    log.info("\n=== Simulation Summary ===")
    log.info(f"Total epochs simulated: {len(results['epochs'])}")
    log.info(f"Number of resets: {len(results['reset_epochs'])}")
    log.info(
        f"Average reset interval: {epochs / len(results['reset_epochs']):.1f} epochs"
        if results["reset_epochs"]
        else "No resets"
    )
    log.info(f"Final average steps: {results['avg_steps'][-1]:.2f}")
    log.info(f"Final pool diversity: {results['pool_diversity'][-1]:.2f}")

    if results["reset_avg_steps"]:
        log.info(
            f"Average steps of reset circuits: {np.mean(results['reset_avg_steps']):.2f} ± {np.std(results['reset_avg_steps']):.2f}"
        )

    # Show step distribution statistics
    final_steps = results["final_pool_steps"]
    log.info(f"Final step distribution:")
    log.info(f"  Min: {jp.min(final_steps):.1f}")
    log.info(f"  Max: {jp.max(final_steps):.1f}")
    log.info(f"  Mean: {jp.mean(final_steps):.1f}")
    log.info(f"  Median: {jp.median(final_steps):.1f}")
    log.info(f"  Std Dev: {jp.std(final_steps):.1f}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
