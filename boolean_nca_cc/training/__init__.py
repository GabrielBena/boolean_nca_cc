"""
Training utilities for boolean NCA

This package contains training utilities for boolean NCA.
"""

from boolean_nca_cc.training.evaluation import (
    StepResult,
    evaluate_model_stepwise,
    evaluate_model_stepwise_generator,
    get_loss_and_update_graph,
    get_loss_from_wires_logits,
)
from boolean_nca_cc.training.train_loop import train_model

__all__ = [
    "StepResult",
    "evaluate_model_stepwise",
    "evaluate_model_stepwise_generator",
    "get_loss_and_update_graph",
    "get_loss_from_wires_logits",
    "train_model",
]
