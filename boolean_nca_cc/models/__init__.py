"""
Models for boolean circuit evolution.

This package provides models for graph-based evolution of boolean circuits:
- CircuitGNN: Sparse message passing (O(E) cost), with optional sparse attention aggregation
- CircuitSelfAttention: Dense masked self-attention (O(N^2) cost)
- CircuitGatheredAttention: Gathered neighborhood attention (O(N * max_degree) cost)
- PerceiverCircuitAttention: Cross-attention to task data (experimental)

Training/evaluation:
- Use `run_model_scan_with_loss` from `boolean_nca_cc.training.evaluation` for all models
"""

from boolean_nca_cc.models.attention.base import AttentionBlock, ReZero
from boolean_nca_cc.models.attention.gathered_attention import (
    CircuitGatheredAttention,
)
from boolean_nca_cc.models.attention.perceiver_attention import (
    PerceiverCircuitAttention,
)
from boolean_nca_cc.models.attention.self_attention import (
    CircuitSelfAttention,
)
from boolean_nca_cc.models.gnn.aggregation import SparseMultiHeadAttention, aggregate_sum
from boolean_nca_cc.models.gnn.edge_update import EdgeUpdateModule
from boolean_nca_cc.models.gnn.model import CircuitGNN
from boolean_nca_cc.models.gnn.node_update import NodeUpdateModule

__all__ = [
    "AttentionBlock",
    "CircuitGNN",
    "CircuitGatheredAttention",
    "CircuitSelfAttention",
    "EdgeUpdateModule",
    "NodeUpdateModule",
    "PerceiverCircuitAttention",
    "ReZero",
    "SparseMultiHeadAttention",
    "aggregate_sum",
]
