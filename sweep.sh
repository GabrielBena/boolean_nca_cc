#!/bin/bash

# This script launches 5 W&B agents in parallel on different GPUs
# for the specified sweep.

SWEEP_ID="marcello-barylli-growai/boolean_nca_cc/z2hd54d7"

echo "Starting agents for sweep: $SWEEP_ID"

CUDA_VISIBLE_DEVICES=3 wandb agent $SWEEP_ID &
CUDA_VISIBLE_DEVICES=4 wandb agent $SWEEP_ID &
CUDA_VISIBLE_DEVICES=5 wandb agent $SWEEP_ID &
CUDA_VISIBLE_DEVICES=6 wandb agent $SWEEP_ID &
CUDA_VISIBLE_DEVICES=7 wandb agent $SWEEP_ID &

echo "5 agents launched in the background."