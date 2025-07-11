"""
Models for boolean circuit evolution using GNNs.

This package provides models for graph neural network-based evolution of boolean circuits.
"""

from boolean_nca_cc.models.self_attention import (
    CircuitSelfAttention,
    run_self_attention_scan,
    run_self_attention_scan_with_loss,
)

__all__ = [
    "CircuitSelfAttention",
    "run_self_attention_scan",
    "run_self_attention_scan_with_loss",
]
