#!/usr/bin/env python3
"""
Pool Update Calculator

Tool to compute the expected average number of updates a circuit receives
in the training pool based on configuration parameters.
"""

import argparse
from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
import yaml


@dataclass
class PoolConfig:
    """Configuration for pool update calculation."""

    pool_size: int
    batch_size: int
    n_message_steps: int
    reset_interval: int
    reset_fraction: float

    def __post_init__(self):
        """Validate configuration."""
        if self.pool_size <= 0:
            raise ValueError("pool_size must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.batch_size > self.pool_size:
            raise ValueError("batch_size cannot exceed pool_size")
        if self.n_message_steps <= 0:
            raise ValueError("n_message_steps must be positive")
        if self.reset_interval <= 0:
            raise ValueError("reset_interval must be positive")
        if not (0 < self.reset_fraction <= 1):
            raise ValueError("reset_fraction must be between 0 and 1")


def calculate_expected_updates_analytical(config: PoolConfig) -> Dict[str, float]:
    """
    Calculate the expected number of updates analytically.

    The calculation is based on:
    1. Update rate per epoch: (batch_size / pool_size) * n_message_steps
    2. Expected lifetime before reset: reset_interval / reset_fraction
    3. Expected updates = update_rate * expected_lifetime

    Args:
        config: Pool configuration

    Returns:
        Dictionary with calculation results and breakdown
    """
    # Basic rates
    selection_probability = config.batch_size / config.pool_size
    updates_per_epoch_if_selected = config.n_message_steps
    update_rate_per_epoch = selection_probability * updates_per_epoch_if_selected

    # Reset dynamics
    # Expected number of reset cycles a circuit survives = 1 / reset_fraction
    # Expected total lifetime = (1 / reset_fraction) * reset_interval
    expected_lifetime_epochs = config.reset_interval / config.reset_fraction

    # Expected updates
    expected_updates = update_rate_per_epoch * expected_lifetime_epochs

    # Additional statistics
    epochs_per_reset_event = config.reset_interval
    circuits_reset_per_event = config.pool_size * config.reset_fraction

    return {
        "expected_updates": expected_updates,
        "selection_probability": selection_probability,
        "update_rate_per_epoch": update_rate_per_epoch,
        "expected_lifetime_epochs": expected_lifetime_epochs,
        "epochs_per_reset_event": epochs_per_reset_event,
        "circuits_reset_per_event": circuits_reset_per_event,
        "fraction_reset_per_event": config.reset_fraction,
    }


def simulate_pool_updates(
    config: PoolConfig, n_epochs: int = 10000, n_simulations: int = 100
) -> Dict[str, float]:
    """
    Simulate pool updates to verify analytical calculation.

    Args:
        config: Pool configuration
        n_epochs: Number of epochs to simulate
        n_simulations: Number of independent simulations to run

    Returns:
        Dictionary with simulation results
    """
    all_final_updates = []

    for sim in range(n_simulations):
        # Initialize pool - track updates for each circuit
        circuit_updates = np.zeros(config.pool_size)

        for epoch in range(n_epochs):
            # Sample batch (without replacement)
            selected_indices = np.random.choice(
                config.pool_size, size=config.batch_size, replace=False
            )

            # Apply updates to selected circuits
            circuit_updates[selected_indices] += config.n_message_steps

            # Check for reset
            if (epoch + 1) % config.reset_interval == 0:
                # Determine which circuits to reset
                n_reset = int(config.pool_size * config.reset_fraction)
                reset_indices = np.random.choice(config.pool_size, size=n_reset, replace=False)
                circuit_updates[reset_indices] = 0

        all_final_updates.extend(circuit_updates)

    return {
        "simulated_mean_updates": np.mean(all_final_updates),
        "simulated_std_updates": np.std(all_final_updates),
        "simulated_median_updates": np.median(all_final_updates),
        "simulated_min_updates": np.min(all_final_updates),
        "simulated_max_updates": np.max(all_final_updates),
        "n_simulations": n_simulations,
        "n_epochs": n_epochs,
    }


def load_config_from_yaml(config_path: str) -> PoolConfig:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to config YAML file

    Returns:
        PoolConfig object
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return PoolConfig(
        pool_size=config["pool"]["size"],
        batch_size=config["training"]["meta_batch_size"],
        n_message_steps=config["training"]["n_message_steps"],
        reset_interval=config["pool"]["reset_interval"],
        reset_fraction=config["pool"]["reset_fraction"],
    )


def print_results(analytical: Dict[str, float], simulation: Dict[str, float] = None):
    """
    Print calculation results in a formatted way.

    Args:
        analytical: Results from analytical calculation
        simulation: Optional results from simulation
    """
    print("=" * 60)
    print("POOL UPDATE CALCULATOR RESULTS")
    print("=" * 60)

    print("\n📊 ANALYTICAL CALCULATION:")
    print(f"Expected Updates per Circuit: {analytical['expected_updates']:.2f}")
    print(f"Selection Probability per Epoch: {analytical['selection_probability']:.4f}")
    print(f"Update Rate per Epoch: {analytical['update_rate_per_epoch']:.4f}")
    print(f"Expected Lifetime (epochs): {analytical['expected_lifetime_epochs']:.1f}")

    print(f"\n🔄 RESET DYNAMICS:")
    print(f"Reset Event Interval: {analytical['epochs_per_reset_event']} epochs")
    print(f"Circuits Reset per Event: {analytical['circuits_reset_per_event']:.1f}")
    print(f"Fraction Reset per Event: {analytical['fraction_reset_per_event']:.3f}")

    if simulation:
        print(f"\n🎯 SIMULATION VERIFICATION:")
        print(f"Simulated Mean Updates: {simulation['simulated_mean_updates']:.2f}")
        print(
            f"Analytical vs Simulated Error: {abs(analytical['expected_updates'] - simulation['simulated_mean_updates']):.2f}"
        )
        print(f"Standard Deviation: {simulation['simulated_std_updates']:.2f}")
        print(f"Median: {simulation['simulated_median_updates']:.2f}")
        print(
            f"Range: [{simulation['simulated_min_updates']:.0f}, {simulation['simulated_max_updates']:.0f}]"
        )
        print(f"Simulations: {simulation['n_simulations']} x {simulation['n_epochs']} epochs")

    print("\n💡 INTERPRETATION:")
    updates_per_circuit = analytical["expected_updates"]
    if updates_per_circuit < 10:
        print("⚠️  Low update count - circuits may not have enough training")
    elif updates_per_circuit < 50:
        print("✅ Moderate update count - reasonable for simple tasks")
    elif updates_per_circuit < 200:
        print("✅ Good update count - suitable for complex learning")
    else:
        print("📈 High update count - may lead to overfitting on individual circuits")


def main():
    parser = argparse.ArgumentParser(description="Calculate expected pool updates")
    parser.add_argument("--config", type=str, help="Path to YAML config file")
    parser.add_argument("--pool-size", type=int, default=4096, help="Pool size")
    parser.add_argument("--batch-size", type=int, default=1024, help="Batch size")
    parser.add_argument("--n-message-steps", type=int, default=3, help="Message steps per batch")
    parser.add_argument("--reset-interval", type=int, default=32, help="Reset interval (epochs)")
    parser.add_argument("--reset-fraction", type=float, default=0.25, help="Reset fraction")
    parser.add_argument("--simulate", action="store_true", help="Run simulation verification")
    parser.add_argument("--n-epochs", type=int, default=10000, help="Simulation epochs")
    parser.add_argument("--n-simulations", type=int, default=100, help="Number of simulations")

    args = parser.parse_args()

    # Load configuration
    if args.config:
        config = load_config_from_yaml(args.config)
        print(f"Loaded configuration from: {args.config}")
    else:
        config = PoolConfig(
            pool_size=args.pool_size,
            batch_size=args.batch_size,
            n_message_steps=args.n_message_steps,
            reset_interval=args.reset_interval,
            reset_fraction=args.reset_fraction,
        )
        print("Using command-line configuration")

    print(f"\n📋 CONFIGURATION:")
    print(f"Pool Size: {config.pool_size}")
    print(f"Batch Size: {config.batch_size}")
    print(f"Message Steps per Batch: {config.n_message_steps}")
    print(f"Reset Interval: {config.reset_interval} epochs")
    print(f"Reset Fraction: {config.reset_fraction:.3f}")

    # Calculate analytical result
    analytical = calculate_expected_updates_analytical(config)

    # Run simulation if requested
    simulation = None
    if args.simulate:
        print(f"\n🔄 Running simulation ({args.n_simulations} runs x {args.n_epochs} epochs)...")
        simulation = simulate_pool_updates(config, args.n_epochs, args.n_simulations)

    # Print results
    print_results(analytical, simulation)


if __name__ == "__main__":
    main()
