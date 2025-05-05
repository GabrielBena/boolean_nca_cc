"""
Self-attention based circuit update mechanism.

This module provides an alternative to GNN message passing by using
masked self-attention to update boolean circuit parameters.
"""

import jax
import jax.numpy as jp
import jraph
from flax import nnx
from typing import List, Optional, Tuple, Dict, Callable
from functools import partial

from boolean_nca_cc.utils.positional_encoding import get_positional_encoding


class SelfAttentionLayer(nnx.Module):
    """
    Multi-head self-attention layer with custom masking.
    """

    def __init__(
        self,
        feature_dim: int,
        num_heads: int = 4,
        dropout_rate: float = 0.0,
        *,
        rngs: nnx.Rngs,
    ):
        """
        Initialize a self-attention layer.

        Args:
            feature_dim: Dimension of input features
            num_heads: Number of attention heads
            dropout_rate: Dropout rate for attention weights
            rngs: Random number generators
        """
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads
        self.dropout_rate = dropout_rate

        # Use MultiHeadAttention from nnx directly
        self.attention = nnx.MultiHeadAttention(
            num_heads=num_heads,
            in_features=feature_dim,
            dropout_rate=dropout_rate,
            decode=False,
            rngs=rngs,
        )

        # Layer normalization
        self.layer_norm = nnx.LayerNorm(feature_dim, rngs=rngs)

    def __call__(
        self,
        x: jp.ndarray,
        attention_mask: Optional[jp.ndarray] = None,
        deterministic: bool = True,
        rngs: Optional[nnx.Rngs] = None,
    ) -> jp.ndarray:
        """
        Apply masked self-attention.

        Args:
            x: Input tensor of shape [batch_size, seq_len, feature_dim]
            attention_mask: Optional mask tensor of shape [batch_size, 1, seq_len, seq_len]
                          where True indicates attention is allowed, and False means
                          attention is masked out
            deterministic: If False, apply dropout
            rngs: Random number generators for dropout

        Returns:
            Output tensor after self-attention of shape [batch_size, seq_len, feature_dim]
        """
        # Apply layer normalization
        normed_x = self.layer_norm(x)

        # Use the built-in MultiHeadAttention module
        output = self.attention(
            inputs_q=normed_x,  # Self-attention: query=key=value
            mask=attention_mask,
            deterministic=deterministic,
            rngs=rngs,
            decode=False,
        )

        # Apply residual connection
        return x + output


class SelfAttentionBlock(nnx.Module):
    """
    Self-attention block with feed-forward network.
    """

    def __init__(
        self,
        feature_dim: int,
        mlp_dim: int = 256,
        num_heads: int = 4,
        dropout_rate: float = 0.0,
        *,
        rngs: nnx.Rngs,
    ):
        """
        Initialize a self-attention block.

        Args:
            feature_dim: Dimension of input features
            mlp_dim: Dimension of hidden layer in feed-forward network
            num_heads: Number of attention heads
            dropout_rate: Dropout rate
            rngs: Random number generators
        """
        # Self-attention layer
        self.attention = SelfAttentionLayer(
            feature_dim=feature_dim,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
            rngs=rngs,
        )

        # Feed-forward network
        self.mlp = nnx.Sequential(
            nnx.LayerNorm(feature_dim, rngs=rngs),
            nnx.Linear(feature_dim, mlp_dim, rngs=rngs),
            nnx.gelu,
            nnx.Linear(mlp_dim, feature_dim, rngs=rngs),
        )

    def __call__(
        self,
        x: jp.ndarray,
        attention_mask: Optional[jp.ndarray] = None,
        deterministic: bool = True,
        rngs: Optional[nnx.Rngs] = None,
    ) -> jp.ndarray:
        """
        Apply self-attention block.

        Args:
            x: Input tensor
            attention_mask: Optional attention mask
            deterministic: If False, apply dropout
            rngs: Random number generators

        Returns:
            Output tensor after self-attention and feed-forward network
        """
        # Apply self-attention
        attn_output = self.attention(
            x,
            attention_mask=attention_mask,
            deterministic=deterministic,
            rngs=rngs,
        )

        # Apply feed-forward network with residual connection
        output = attn_output + self.mlp(attn_output)

        return output


class CircuitSelfAttention(nnx.Module):
    """
    Self-attention based circuit update mechanism.

    Implements masked self-attention to update boolean circuit parameters.
    """

    def __init__(
        self,
        hidden_dim: int = 16,
        arity: int = 2,
        num_heads: int = 4,
        num_layers: int = 3,
        mlp_dim: int = 64,
        dropout_rate: float = 0.0,
        example_graph: jraph.GraphsTuple = None,
        *,
        rngs: nnx.Rngs,
    ):
        """
        Initialize the circuit self-attention model.

        Args:
            hidden_dim: Dimension of hidden features
            arity: Number of inputs per gate in the boolean circuit
            num_heads: Number of attention heads
            num_layers: Number of self-attention layers
            mlp_dim: Dimension of feed-forward network
            dropout_rate: Dropout rate
            rngs: Random number generators
        """
        self.arity = arity
        self.hidden_dim = hidden_dim
        self.logit_dim = 2**arity
        self.dropout_rate = dropout_rate
        self.num_heads = num_heads

        # Compute the total feature dimension (logits + hidden + positional encodings)
        feature_dim = self.logit_dim + hidden_dim * 3  # logits + hidden + 2 PE's

        # Feature projection
        self.feature_proj = nnx.Linear(feature_dim, hidden_dim * 4, rngs=rngs)

        # Self-attention layers
        self.attention_layers = [
            SelfAttentionBlock(
                feature_dim=hidden_dim * 4,
                mlp_dim=mlp_dim,
                num_heads=num_heads,
                dropout_rate=dropout_rate,
                rngs=rngs,
            )
            for _ in range(num_layers)
        ]

        # Output projections for both logits and hidden features
        self.logit_proj = nnx.Linear(hidden_dim * 4, self.logit_dim, rngs=rngs)
        self.hidden_proj = nnx.Linear(hidden_dim * 4, hidden_dim, rngs=rngs)

    def _create_attention_mask(
        self,
        senders: jp.ndarray,
        receivers: jp.ndarray,
        n_node: int,
        bidirectional: bool = True,
    ) -> jp.ndarray:
        """
        Create an attention mask based on the circuit wiring.

        Args:
            senders: Array of sender node indices
            receivers: Array of receiver node indices
            n_node: Total number of nodes
            bidirectional: If True, create a symmetric mask

        Returns:
            Boolean attention mask of shape [batch_size, 1, seq_len, seq_len]
        """
        # Create a mask where edges exist in the graph
        mask = jp.zeros((n_node, n_node), dtype=jp.bool_)

        if len(senders) > 0:
            # Set mask[receiver, sender] = True for all edges
            mask = mask.at[receivers, senders].set(True)

            if bidirectional:
                # Also set mask[sender, receiver] = True for bidirectional attention
                mask = mask.at[senders, receivers].set(True)

        # Add self-connections (diagonal of True)
        mask = mask | jp.eye(n_node, dtype=jp.bool_)

        # Add batch dimension and singleton head dimension [batch_size, 1, seq_len, seq_len]
        # This format matches the MultiHeadAttention mask format
        return mask[None, None, ...]

    def _extract_features(self, nodes: Dict[str, jp.ndarray]) -> jp.ndarray:
        """
        Extract and concatenate node features for attention.

        Args:
            nodes: Dictionary of node features

        Returns:
            Concatenated features tensor of shape [n_node, feature_dim]
        """
        # Extract relevant features
        logits = nodes["logits"]  # [n_node, logit_dim]
        hidden = nodes["hidden"]  # [n_node, hidden_dim]
        layer_pe = nodes["layer_pe"]  # [n_node, hidden_dim]
        intra_layer_pe = nodes["intra_layer_pe"]  # [n_node, hidden_dim]

        # Concatenate all features
        features = jp.concatenate([logits, hidden, layer_pe, intra_layer_pe], axis=-1)

        return features

    def __call__(
        self,
        graph: jraph.GraphsTuple,
        deterministic: bool = True,
        rngs: Optional[nnx.Rngs] = None,
    ) -> jraph.GraphsTuple:
        """
        Apply self-attention to update circuit parameters.

        Args:
            graph: Input graph structure with node and edge features
            deterministic: If False, apply dropout
            rngs: Random number generators

        Returns:
            Updated graph after self-attention
        """
        nodes, edges, receivers, senders, globals_, n_node, n_edge = graph

        # Convert n_node from array to scalar
        n_node_scalar = n_node[0]

        # Extract and concatenate node features
        features = self._extract_features(nodes)

        # Add batch dimension [1, n_node, feature_dim]
        features = features[None, ...]

        # Create attention mask based on circuit wiring
        attention_mask = self._create_attention_mask(
            senders, receivers, n_node_scalar, bidirectional=True
        )

        # Project features to the attention dimension
        x = self.feature_proj(features)

        # Apply self-attention layers
        for layer in self.attention_layers:
            x = layer(
                x,
                attention_mask=attention_mask,
                deterministic=deterministic,
                rngs=rngs,
            )

        # Project to logit and hidden state updates
        logit_updates = self.logit_proj(x)
        hidden_updates = self.hidden_proj(x)

        # Remove batch dimension
        logit_updates = logit_updates[0]
        hidden_updates = hidden_updates[0]

        # Update logits and hidden features in a residual manner
        updated_logits = nodes["logits"] + logit_updates
        updated_hidden = nodes["hidden"] + hidden_updates

        # Create updated nodes dictionary
        updated_nodes = {**nodes, "logits": updated_logits, "hidden": updated_hidden}

        # Return updated graph
        return graph._replace(nodes=updated_nodes)


@partial(nnx.jit, static_argnames=("num_steps",))
def run_self_attention_scan(
    model: CircuitSelfAttention,
    graph: jraph.GraphsTuple,
    num_steps: int,
    deterministic: bool = True,
    rngs: Optional[nnx.Rngs] = None,
) -> jraph.GraphsTuple:
    """
    Apply the self-attention iteratively for multiple steps using jax.lax.scan.

    Args:
        model: The CircuitSelfAttention model
        graph: The initial graph
        num_steps: Number of attention steps to perform
        deterministic: If False, apply dropout
        rngs: Random number generators

    Returns:
        Updated graph after num_steps of self-attention
    """

    def scan_body(carry_graph, _):
        # Apply one step of self-attention
        updated_graph = model(
            carry_graph,
            deterministic=deterministic,
            rngs=rngs,
        )
        return updated_graph, None

    # Run the scan
    final_graph, _ = jax.lax.scan(scan_body, graph, None, length=num_steps)

    return final_graph
