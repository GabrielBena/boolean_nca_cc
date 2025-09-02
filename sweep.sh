#!/bin/bash

# This script launches 5 W&B agents in parallel on different GPUs
# for the specified sweep.

SWEEP_ID="marcello-barylli-growai/boolean_nca_cc/9yn4cfgp"

echo "Starting agents for sweep: $SWEEP_ID"

# # diversity agents
# CUDA_VISIBLE_DEVICES=0 wandb agent $SWEEP_ID &
# CUDA_VISIBLE_DEVICES=1 wandb agent $SWEEP_ID &    
# CUDA_VISIBLE_DEVICES=2 wandb agent $SWEEP_ID &
# CUDA_VISIBLE_DEVICES=3 wandb agent $SWEEP_ID &

# damage agents
CUDA_VISIBLE_DEVICES=3 wandb agent $SWEEP_ID &
CUDA_VISIBLE_DEVICES=4 wandb agent $SWEEP_ID &
CUDA_VISIBLE_DEVICES=5 wandb agent $SWEEP_ID &
# CUDA_VISIBLE_DEVICES=7 wandb agent $SWEEP_ID &

echo " agents launched in the background."