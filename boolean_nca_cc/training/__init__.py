"""
Training utilities for boolean NCA

This package contains training utilities for boolean NCA.

Key evaluation functions:
- `evaluate_model_stepwise_batched`: Main evaluation function for batched circuits (supports chunking)
- `evaluate_model_stepwise_generator`: Generator for live demos and streaming results
- `run_model_scan_with_loss`: Low-level unified scan for training and evaluation
"""

from boolean_nca_cc.training.checkpointing import configure_notebook_logging
from boolean_nca_cc.training.evaluation import (
    StepResult,
    evaluate_model_stepwise,
    evaluate_model_stepwise_batched,
    evaluate_model_stepwise_generator,
    get_loss_and_update_graph,
    get_loss_from_wires_logits,
    run_model_scan_with_loss,
)
from boolean_nca_cc.training.sharding import (
    ShardingContext,
    create_device_mesh,
    setup_virtual_devices,
)
from boolean_nca_cc.training.train_loop import train_model

__all__ = [
    # Multi-GPU utilities
    "ShardingContext",
    "StepResult",
    "configure_notebook_logging",
    "create_device_mesh",
    "evaluate_circuits_in_chunks",  # Deprecated
    "evaluate_model_stepwise",
    "evaluate_model_stepwise_batched",
    "evaluate_model_stepwise_generator",
    "get_loss_and_update_graph",
    "get_loss_from_wires_logits",
    "run_model_scan_with_loss",
    "setup_virtual_devices",
    "train_model",
]
