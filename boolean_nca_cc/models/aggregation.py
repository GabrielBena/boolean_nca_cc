"""
Aggregation functions for graph neural networks.

This module provides different methods for aggregating messages in a graph neural network.
"""

import jax.numpy as jp
import jraph
from flax import nnx


def aggregate_sum(messages, indices, num_segments):
    """
    Default aggregation function: sum messages per node.

    Args:
        messages: Features of edges targeting a node
        indices: Index of the node targeted by the edge
        num_segments: Total number of nodes in the graph

    Returns:
        Aggregated messages per node
    """
    return jraph.segment_sum(messages, indices, num_segments)


class AttentionAggregation(nnx.Module):
    """
    Attention-based message aggregation.

    Implements a multi-head attention mechanism for aggregating messages in a GNN.
    """

    def __init__(self, feature_dim: int, num_heads: int = 4, *, rngs: nnx.Rngs):
        """
        Initialize attention aggregation.

        Args:
            feature_dim: Dimension of feature vectors
            num_heads: Number of attention heads
            rngs: Random number generators
        """
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads

        # Projections for attention mechanism
        self.key_proj = nnx.Linear(feature_dim, feature_dim, rngs=rngs)
        self.value_proj = nnx.Linear(feature_dim, feature_dim, rngs=rngs)
        self.query_proj = nnx.Linear(feature_dim, feature_dim, rngs=rngs)
        self.output_proj = nnx.Linear(feature_dim, feature_dim, rngs=rngs)

    def __call__(
        self,
        messages: jp.ndarray,
        receivers: jp.ndarray,
        num_segments: int,
    ):
        """
        Apply attention-based message aggregation.

        Args:
            messages: Features of edges targeting nodes [num_edges, feature_dim]
            receivers: Index of the node targeted by each edge [num_edges]
            num_segments: Total number of nodes in the graph

        Returns:
            Aggregated messages per node [num_nodes, feature_dim]
        """
        # Handle empty case
        if messages.shape[0] == 0:
            return jp.zeros((num_segments, self.feature_dim))

        # Project messages to keys and values
        keys = self.key_proj(messages)  # [num_edges, feature_dim]
        values = self.value_proj(messages)  # [num_edges, feature_dim]

        # Get unique receiver nodes (first aggregate messages per receiver)
        summed_messages = jraph.segment_sum(
            messages, receivers, num_segments
        )  # [num_nodes, feature_dim]

        # Generate queries for each node
        queries = self.query_proj(summed_messages)  # [num_nodes, feature_dim]

        # Get corresponding query for each message
        message_queries = queries[receivers]  # [num_edges, feature_dim]

        # Compute attention scores - shape: [num_edges]
        attention_scores = jp.sum(message_queries * keys, axis=-1) / jp.sqrt(self.feature_dim)

        # For each receiver, normalize scores with softmax
        # Compute max per receiver for numerical stability
        max_scores = jraph.segment_max(attention_scores, receivers, num_segments)  # [num_nodes]
        edge_max_scores = max_scores[receivers]  # [num_edges]

        # Compute exp(score - max_score)
        exp_scores = jp.exp(attention_scores - edge_max_scores)  # [num_edges]

        # Sum exp scores per receiver
        sum_exp_scores = jraph.segment_sum(exp_scores, receivers, num_segments)  # [num_nodes]
        edge_sum_exp_scores = sum_exp_scores[receivers]  # [num_edges]

        # Compute softmax weights
        attention_weights = exp_scores / (edge_sum_exp_scores + 1e-8)  # [num_edges]

        # Weight values by attention weights
        weighted_values = values * attention_weights[:, None]  # [num_edges, feature_dim]

        # Aggregate weighted values per receiver
        aggregated_values = jraph.segment_sum(
            weighted_values, receivers, num_segments
        )  # [num_nodes, feature_dim]

        # Final projection
        output = self.output_proj(aggregated_values)  # [num_nodes, feature_dim]

        return output
