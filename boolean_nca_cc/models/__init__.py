"""
Models for boolean circuit evolution using GNNs.

This package provides models for graph neural network-based evolution of boolean circuits.

Model scan functions:
- Use `run_model_scan_with_loss` from `boolean_nca_cc.training.evaluation` for all models
- The model-specific `run_*_scan_with_loss` functions are deprecated and will be removed

For evaluation and training:
- `run_model_scan_with_loss`: Unified scan for all model types (training)
- `apply_model_and_compute_loss`: Single step function (used internally)
- `evaluate_model_stepwise`: JIT-compiled evaluation
- `evaluate_model_stepwise_generator`: Generator for live demos
"""

from boolean_nca_cc.models.attention.base import AttentionBlock, ReZero
from boolean_nca_cc.models.attention.perceiver_attention import (
    PerceiverCircuitAttention,
)
from boolean_nca_cc.models.attention.self_attention import (
    CircuitSelfAttention,
)
from boolean_nca_cc.models.gnn.aggregation import AttentionAggregation, aggregate_sum
from boolean_nca_cc.models.gnn.edge_update import EdgeUpdateModule
from boolean_nca_cc.models.gnn.model import (
    CircuitGNN,
    run_gnn_scan,
    run_gnn_scan_with_loss,  # DEPRECATED
)
from boolean_nca_cc.models.gnn.node_update import NodeUpdateModule

__all__ = [
    "AttentionAggregation",
    "AttentionBlock",
    "CircuitGNN",
    "CircuitSelfAttention",
    "EdgeUpdateModule",
    "NodeUpdateModule",
    "PerceiverCircuitAttention",
    "ReZero",
    "aggregate_sum",
]
