#!/usr/bin/env python3
"""
Initialize WandB Sweep for Boolean Circuit Hyperparameter Optimization

This script initializes a WandB sweep using the configuration file.
"""

import sys
from pathlib import Path

import yaml

import wandb


def main():
    # Load sweep configuration
    file_path = Path(__file__).resolve()
    root_path = file_path.parent
    config_path = root_path / "wandb_sweep_config.yaml"
    if not config_path.exists():
        print("❌ wandb_sweep_config.yaml not found!")
        sys.exit(1)

    with open(config_path, "r") as f:
        sweep_config = yaml.safe_load(f)

    project = sweep_config["project"]
    entity = sweep_config["entity"]

    print("🔧 Initializing WandB Sweep...")
    print(f"📊 Method: {sweep_config['method']}")
    print(f"🎯 Metric: {sweep_config['metric']['name']} ({sweep_config['metric']['goal']})")

    # Initialize the sweep
    sweep_id = wandb.sweep(sweep_config, project=project, entity=entity)

    print(f"✅ Sweep initialized successfully!")
    print(f"🆔 Sweep ID: {sweep_id}")
    print(f"🌐 Sweep URL: https://wandb.ai/{entity}/{project}/sweeps/{sweep_id}")
    print()
    print("To start the sweep agent, run:")
    print(f"  wandb agent {entity}/{project}/{sweep_id}")
    print("  # OR")
    print(f"  python sweep_agent.py {sweep_id}")

    # Save sweep ID for later use
    with open("sweep_id.txt", "w") as f:
        f.write(sweep_id)

    return sweep_id


if __name__ == "__main__":
    main()
