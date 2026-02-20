#!/bin/bash

# This script launches W&B agents in parallel on different GPUs
# for the specified sweep.
#
# Usage: ./sweeps/sweep.sh [SWEEP_ID]
# If no SWEEP_ID is provided, uses the default hardcoded one

# Use provided sweep ID or default
INPUT_ID="${1:-marcello-barylli-growai/boolean_nca_cc/t3rb9jjg}"

# If input doesn't contain '/', assume it's just the short ID and construct full path
# Using case statement for POSIX compatibility (works with both sh and bash)
case "$INPUT_ID" in
    */*)
        SWEEP_ID="$INPUT_ID"
        ;;
    *)
        SWEEP_ID="marcello-barylli-growai/boolean_nca_cc/$INPUT_ID"
        ;;
esac

echo "Starting agents for sweep: $SWEEP_ID"

# squadron 1
# CUDA_VISIBLE_DEVICES=0 wandb agent $SWEEP_ID &
# CUDA_VISIBLE_DEVICES=1 wandb agent $SWEEP_ID &    
# CUDA_VISIBLE_DEVICES=2 wandb agent $SWEEP_ID &

# squadron 2
# CUDA_VISIBLE_DEVICES=3 wandb agent $SWEEP_ID &
CUDA_VISIBLE_DEVICES=4 wandb agent $SWEEP_ID &
CUDA_VISIBLE_DEVICES=5 wandb agent $SWEEP_ID &
CUDA_VISIBLE_DEVICES=6 wandb agent $SWEEP_ID &
CUDA_VISIBLE_DEVICES=7 wandb agent $SWEEP_ID &

echo "agents launched in the background."