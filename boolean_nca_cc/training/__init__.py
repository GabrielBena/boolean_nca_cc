"""
Training utilities for boolean NCA

This package contains training utilities for boolean NCA.
"""

from boolean_nca_cc.training.train_loop import train_model
from boolean_nca_cc.training.train_step import train_step_model
from boolean_nca_cc.training.evaluation import (
    evaluate_model_stepwise,
    evaluate_model_stepwise_generator,
    StepResult,
)

__all__ = [
    "train_model",
    "train_step_model",
    "evaluate_model_stepwise",
    "evaluate_model_stepwise_generator",
    "StepResult",
]
