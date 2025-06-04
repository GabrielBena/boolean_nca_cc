"""
Models for boolean circuit evolution using GNNs.

This package provides models for graph neural network-based evolution of boolean circuits.
"""

from boolean_nca_cc.models.gnn import CircuitGNN, run_gnn_scan, run_gnn_scan_with_loss
from boolean_nca_cc.models.node_update import NodeUpdateModule
from boolean_nca_cc.models.edge_update import EdgeUpdateModule
from boolean_nca_cc.models.aggregation import AttentionAggregation, aggregate_sum
from boolean_nca_cc.models.self_attention import (
    CircuitSelfAttention,
    run_self_attention_scan,
    run_self_attention_scan_with_loss,
)

__all__ = [
    "CircuitGNN",
    "run_gnn_scan",
    "run_gnn_scan_with_loss",
    "NodeUpdateModule",
    "EdgeUpdateModule",
    "AttentionAggregation",
    "aggregate_sum",
    "CircuitSelfAttention",
    "run_self_attention_scan",
    "run_self_attention_scan_with_loss",
]
