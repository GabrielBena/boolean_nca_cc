#!/usr/bin/env python3
"""
Start WandB Sweep Agent for Boolean Circuit Hyperparameter Optimization

This script starts a WandB sweep agent to run the hyperparameter optimization.
"""

import argparse
import sys
from pathlib import Path

import wandb


def main():
    parser = argparse.ArgumentParser(description="Start WandB sweep agent")
    parser.add_argument("sweep_id", nargs="?", help="WandB sweep ID")
    parser.add_argument("--count", type=int, help="Maximum number of runs for this agent")
    parser.add_argument("--entity", type=str, help="WandB entity", default="m2snn")
    parser.add_argument("--project", type=str, help="WandB project", default="boolean_bp_sweep")
    args = parser.parse_args()

    # Get sweep ID
    if args.sweep_id:
        sweep_id = args.sweep_id
    else:
        # Try to load from file
        sweep_id_file = Path("sweep_id.txt")
        if sweep_id_file.exists():
            sweep_id = sweep_id_file.read_text().strip()
        else:
            print("❌ No sweep ID provided and no sweep_id.txt found!")
            print("Usage: python sweep_agent.py <sweep_id>")
            print("   or: python sweep_agent.py  # reads from sweep_id.txt")
            sys.exit(1)

    print("🚀 Starting WandB Sweep Agent...")
    print(f"🆔 Sweep ID: {sweep_id}")
    print(f"🌐 Sweep URL: https://wandb.ai/{args.entity}/{args.project}/sweeps/{sweep_id}")

    if args.count:
        print(f"🔢 Max runs: {args.count}")
        wandb.agent(sweep_id, count=args.count, entity=args.entity, project=args.project)
    else:
        print("🔄 Running indefinitely (Ctrl+C to stop)")
        wandb.agent(sweep_id, entity=args.entity, project=args.project)


if __name__ == "__main__":
    main()
