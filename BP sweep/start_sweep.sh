#!/bin/bash
# WandB Sweep Launcher Script
# 
# This script initializes and starts a WandB sweep for hyperparameter optimization
# of boolean circuit training.

set -e  # Exit on any error

echo "🚀 BOOLEAN CIRCUIT HYPERPARAMETER OPTIMIZATION SWEEP"
echo "====================================================="

# Check if wandb is installed
if ! command -v wandb &> /dev/null; then
    echo "❌ WandB CLI not found! Please install with: pip install wandb"
    exit 1
fi

# Check if logged in to wandb
if ! wandb login; then
    echo "❌ WandB login failed!"
    exit 1
fi

# Initialize the sweep
echo "🔧 Initializing sweep configuration..."
SWEEP_OUTPUT=$(python sweep_init.py)
SWEEP_ID=$(echo "$SWEEP_OUTPUT" | grep "Sweep ID:" | sed 's/.*Sweep ID: //')

if [ -z "$SWEEP_ID" ]; then
    echo "❌ Failed to get sweep ID!"
    echo "Full output was:"
    echo "$SWEEP_OUTPUT"
    exit 1
fi

echo "✅ Sweep initialized: $SWEEP_ID"
echo "🌐 View progress at: https://wandb.ai/m2snn/boolean_bp_sweep/sweeps/$SWEEP_ID"
echo ""

# Ask user how many agents to run
read -p "How many parallel agents to run? (1-16, default=1): " NUM_AGENTS
NUM_AGENTS=${NUM_AGENTS:-1}

if ! [[ "$NUM_AGENTS" =~ ^[1-9]$|^1[0-6]$ ]]; then
    echo "❌ Invalid number of agents. Must be 1-16."
    exit 1
fi

echo "🏃 Starting $NUM_AGENTS sweep agent(s)..."

# Use Python-based agent manager for better process control
python run_agents.py $SWEEP_ID --num_agents $NUM_AGENTS
