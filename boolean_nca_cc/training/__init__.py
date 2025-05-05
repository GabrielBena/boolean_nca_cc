"""
Training utilities for boolean NCA

This package contains training utilities for boolean NCA.
"""

from boolean_nca_cc.training.train_loop import train_gnn
from boolean_nca_cc.training.train_step import train_step_gnn
from boolean_nca_cc.training.attn_train_loop import train_self_attention
from boolean_nca_cc.training.attn_train_step import train_step_attn

__all__ = [
    "train_gnn",
    "train_step_gnn",
    "train_self_attention",
    "train_step_attn",
]
