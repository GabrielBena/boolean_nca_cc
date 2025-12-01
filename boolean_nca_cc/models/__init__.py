"""
Models for boolean circuit evolution using GNNs.

This package provides models for graph neural network-based evolution of boolean circuits.
"""

from boolean_nca_cc.models.attention.base import AttentionBlock, ReZero
from boolean_nca_cc.models.attention.perceiver_attention import (
    PerceiverCircuitAttention,
    run_perceiver_scan,
    run_perceiver_scan_with_loss,
)
from boolean_nca_cc.models.attention.self_attention import (
    CircuitSelfAttention,
    run_self_attention_scan,
    run_self_attention_scan_with_loss,
)
from boolean_nca_cc.models.gnn.aggregation import AttentionAggregation, aggregate_sum
from boolean_nca_cc.models.gnn.edge_update import EdgeUpdateModule
from boolean_nca_cc.models.gnn.model import CircuitGNN, run_gnn_scan, run_gnn_scan_with_loss
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
    "run_gnn_scan",
    "run_gnn_scan_with_loss",
    "run_perceiver_scan",
    "run_perceiver_scan_with_loss",
    "run_self_attention_scan",
    "run_self_attention_scan_with_loss",
]
