"""
Training utilities for boolean NCA

This package contains training utilities for boolean NCA.
"""

from boolean_nca_cc.training.train_loop import train_model
from boolean_nca_cc.training.evaluation import (
    evaluate_model_stepwise,
    evaluate_model_stepwise_batched,
    StepResult,
    get_loss_from_wires_logits,
    get_loss_and_update_graph,
)

__all__ = [
    "train_model",
    "evaluate_model_stepwise",
    "evaluate_model_stepwise_batched",
    "StepResult",
    "get_loss_from_wires_logits",
    "get_loss_and_update_graph",
]
