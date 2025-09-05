#!/usr/bin/env python3
"""
Alternative Agent Runner with Better Process Management

This provides better control over multiple sweep agents with proper cleanup.
"""

import argparse
import signal
import subprocess
import sys
import time
from pathlib import Path


class AgentManager:
    def __init__(self):
        self.agents = []
        self.running = True

    def start_agent(self, sweep_id, agent_id):
        """Start a single sweep agent."""
        cmd = ["python", "sweep_agent.py", sweep_id]
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1,
        )
        self.agents.append((agent_id, process))
        print(f"  ✅ Agent {agent_id} started (PID: {process.pid})")
        return process

    def stop_all_agents(self):
        """Stop all running agents."""
        print(f"\n🛑 Stopping {len(self.agents)} agents...")
        for agent_id, process in self.agents:
            if process.poll() is None:  # Still running
                print(f"  Stopping agent {agent_id} (PID: {process.pid})")
                try:
                    process.terminate()
                    # Give it a chance to terminate gracefully
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                except Exception as e:
                    print(f"  Warning: Error stopping agent {agent_id}: {e}")
        print("✅ All agents stopped.")
        self.running = False

    def signal_handler(self, signum, frame):
        """Handle interrupt signals."""
        print(f"\n📡 Received signal {signum}")
        self.stop_all_agents()
        sys.exit(0)

    def monitor_agents(self):
        """Monitor running agents and restart if needed."""
        while self.running:
            time.sleep(10)  # Check every 10 seconds

            # Check for finished agents
            for agent_id, process in self.agents[:]:
                if process.poll() is not None:  # Process finished
                    return_code = process.returncode
                    if return_code == 0:
                        print(f"  ℹ️  Agent {agent_id} finished normally")
                    else:
                        print(f"  ⚠️  Agent {agent_id} crashed (code: {return_code})")
                    self.agents.remove((agent_id, process))

            # If all agents finished, exit
            if not self.agents:
                print("🏁 All agents have finished.")
                break


def main():
    parser = argparse.ArgumentParser(description="Run multiple WandB sweep agents")
    parser.add_argument("sweep_id", help="WandB sweep ID")
    parser.add_argument("--num_agents", type=int, default=1, help="Number of agents to run")
    parser.add_argument("--auto_restart", action="store_true", help="Restart crashed agents")
    args = parser.parse_args()

    if not (1 <= args.num_agents <= 16):
        print("❌ Number of agents must be between 1 and 16")
        sys.exit(1)

    print(f"🚀 STARTING {args.num_agents} WANDB SWEEP AGENTS")
    print("=" * 50)
    print(f"🆔 Sweep ID: {args.sweep_id}")
    print(f"🌐 Monitor at: https://wandb.ai/m2snn/boolean_bp_sweep/sweeps/{args.sweep_id}")
    print()

    # Create agent manager
    manager = AgentManager()

    # Set up signal handlers
    signal.signal(signal.SIGINT, manager.signal_handler)
    signal.signal(signal.SIGTERM, manager.signal_handler)

    try:
        # Start all agents
        print(f"🏃 Starting {args.num_agents} agents...")
        for i in range(1, args.num_agents + 1):
            manager.start_agent(args.sweep_id, i)
            time.sleep(1)  # Small delay between starts

        print(f"\n✅ All {args.num_agents} agents started successfully!")
        print("🔄 Monitoring agents... (Ctrl+C to stop all)")

        # Monitor agents
        manager.monitor_agents()

    except KeyboardInterrupt:
        print("\n🔴 Interrupted by user")
        manager.stop_all_agents()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        manager.stop_all_agents()
        sys.exit(1)

    print("\n🏁 Agent manager finished.")


if __name__ == "__main__":
    main()
