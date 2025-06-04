"""
Self-attention based circuit update mechanism.

This module provides an alternative to GNN message passing by using
masked self-attention to update boolean circuit parameters.
"""

import jax
import jax.numpy as jp
import jraph
from flax import nnx
from typing import Optional, Dict, Tuple, List
from functools import partial


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
        self.deterministic = True

        # Use MultiHeadAttention from nnx directly
        self.attention = nnx.MultiHeadAttention(
            num_heads=num_heads,
            in_features=feature_dim,
            dropout_rate=dropout_rate,
            decode=False,
            deterministic=True,
            rngs=rngs,
        )

        # Layer normalization
        self.layer_norm = nnx.LayerNorm(feature_dim, rngs=rngs)

    def __call__(
        self,
        x: jp.ndarray,
        attention_mask: Optional[jp.ndarray] = None,
    ) -> jp.ndarray:
        """
        Apply masked self-attention.

        Args:
            x: Input tensor of shape [batch_size, seq_len, feature_dim]
            attention_mask: Optional mask tensor of shape [batch_size, 1, seq_len, seq_len]
                          where True indicates attention is allowed, and False means
                          attention is masked out

        Returns:
            Output tensor after self-attention of shape [batch_size, seq_len, feature_dim]
        """
        # Apply layer normalization
        normed_x = self.layer_norm(x)

        # Use the built-in MultiHeadAttention module
        output = self.attention(
            inputs_q=normed_x,  # Self-attention: query=key=value
            mask=attention_mask,
            deterministic=True,
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
        self.deterministic = True

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
    ) -> jp.ndarray:
        """
        Apply self-attention block.

        Args:
            x: Input tensor
            attention_mask: Optional attention mask

        Returns:
            Output tensor after self-attention and feed-forward network
        """
        # Apply self-attention
        attn_output = self.attention(
            x,
            attention_mask=attention_mask,
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
        n_node: int,
        hidden_dim: int = 16,
        arity: int = 2,
        num_heads: int = 4,
        num_layers: int = 3,
        mlp_dim: int = 64,
        dropout_rate: float = 0.0,
        *,
        rngs: nnx.Rngs,
        type: str = "self_attention",
        zero_init: bool = True,
    ):
        """
        Initialize the circuit self-attention model.

        Args:
            n_node: Fixed number of nodes in the circuit
            hidden_dim: Dimension of hidden features
            arity: Number of inputs per gate in the boolean circuit
            num_heads: Number of attention heads
            num_layers: Number of self-attention layers
            mlp_dim: Dimension of feed-forward network
            dropout_rate: Dropout rate
            rngs: Random number generators
        """
        self.n_node = int(n_node)
        self.arity = arity
        self.hidden_dim = hidden_dim
        self.logit_dim = 2**arity
        self.dropout_rate = dropout_rate
        self.num_heads = num_heads
        self.deterministic = True

        # Compute the total feature dimension (logits + hidden + positional encodings + loss)
        feature_dim = (
            self.logit_dim + hidden_dim * 3 + 1
        )  # logits + hidden + 2 PE's + loss

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
        self.logit_proj = nnx.Linear(
            hidden_dim * 4,
            self.logit_dim,
            kernel_init=nnx.initializers.zeros
            if zero_init
            else nnx.initializers.normal(stddev=1e-6),
            bias_init=nnx.initializers.zeros
            if zero_init
            else nnx.initializers.normal(stddev=1e-6),
            rngs=rngs,
        )
        self.hidden_proj = nnx.Linear(
            hidden_dim * 4,
            hidden_dim,
            kernel_init=nnx.initializers.zeros
            if zero_init
            else nnx.initializers.normal(stddev=1e-6),
            bias_init=nnx.initializers.zeros
            if zero_init
            else nnx.initializers.normal(stddev=1e-6),
            rngs=rngs,
        )

    def _create_attention_mask(
        self,
        senders: jp.ndarray,
        receivers: jp.ndarray,
        bidirectional: bool = True,
    ) -> jp.ndarray:
        """
        Create an attention mask based on the circuit wiring.

        Args:
            senders: Array of sender node indices
            receivers: Array of receiver node indices
            bidirectional: If True, create a symmetric mask

        Returns:
            Boolean attention mask of shape [batch_size, 1, seq_len, seq_len]
        """
        # Create a mask where edges exist in the graph
        mask = jp.zeros((self.n_node, self.n_node), dtype=jp.bool_)

        if len(senders) > 0:
            # Set mask[receiver, sender] = True for all edges
            mask = mask.at[receivers, senders].set(True)

            if bidirectional:
                # Also set mask[sender, receiver] = True for bidirectional attention
                mask = mask.at[senders, receivers].set(True)

        # Add self-connections (diagonal of True)
        mask = mask | jp.eye(self.n_node, dtype=jp.bool_)

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
        # Positional encodings
        layer_pe = nodes["layer_pe"]  # [n_node, hidden_dim]
        intra_layer_pe = nodes["intra_layer_pe"]  # [n_node, hidden_dim]
        # Loss feature
        loss = nodes["loss"]  # [n_node]
        # Add dimension to match other features: [n_node] -> [n_node, 1]
        loss_expanded = loss[:, None]  # [n_node, 1]

        # Concatenate all features including loss
        features = jp.concatenate(
            [logits, hidden, layer_pe, intra_layer_pe, loss_expanded], axis=-1
        )

        return features

    def __call__(
        self,
        graph: jraph.GraphsTuple,
        attention_mask: Optional[jp.ndarray] = None,
    ) -> jraph.GraphsTuple:
        """
        Apply self-attention to update circuit parameters.

        Args:
            graph: Input graph structure with node and edge features
            attention_mask: Optional pre-computed attention mask.
                          If None, it will be computed from the graph.

        Returns:
            Updated graph after self-attention
        """
        nodes, edges, receivers, senders, globals_, n_node, n_edge = graph

        # Note: globals_ is extracted but no longer used in computations

        # Extract and concatenate node features
        features = self._extract_features(nodes)

        # Add batch dimension [1, n_node, feature_dim]
        features = features[None, ...]

        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = self._create_attention_mask(
                senders, receivers, bidirectional=True
            )

        # Project features to the attention dimension
        x = self.feature_proj(features)

        # Apply self-attention layers
        for layer in self.attention_layers:
            x = layer(
                x,
                attention_mask=attention_mask,
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


@partial(nnx.jit, static_argnames=("num_steps"))
def run_self_attention_scan(
    model: CircuitSelfAttention,
    graph: jraph.GraphsTuple,
    num_steps: int,
) -> Tuple[jraph.GraphsTuple, List[jraph.GraphsTuple]]:
    """
    Apply the self-attention model iteratively for multiple steps using jax.lax.scan.

    Args:
        model: The CircuitSelfAttention model
        graph: The initial graph
        num_steps: Number of steps to perform

    Returns:
        final_graph: The graph after all steps
        all_graphs: List of graphs from each step (including initial)
    """
    # --- Compute the mask *once* before the scan ---
    attention_mask = model._create_attention_mask(
        graph.senders, graph.receivers, bidirectional=True
    )
    # ---------------------------------------------

    def scan_body(carry_graph, _):
        # Apply one step of self-attention with precomputed mask
        updated_graph = model(carry_graph, attention_mask=attention_mask)
        return updated_graph, updated_graph

    # Run the scan
    final_graph, intermediate_graphs = jax.lax.scan(
        scan_body, graph, None, length=num_steps
    )

    # Combine initial graph with intermediate results
    all_graphs = [graph] + list(intermediate_graphs)

    return final_graph, all_graphs


def run_self_attention_scan_with_loss(
    model: CircuitSelfAttention,
    graph: jraph.GraphsTuple,
    num_steps: int,
    logits_original_shapes: List[Tuple],
    wires: List[jp.ndarray],
    x_data: jp.ndarray,
    y_data: jp.ndarray,
    loss_type: str,
    layer_sizes: Tuple[Tuple[int, int], ...],
) -> Tuple[jraph.GraphsTuple, List[jraph.GraphsTuple], jp.ndarray, List]:
    """
    Run the self-attention model for multiple steps with loss computation and graph updating at each step.

    This function combines model application with loss computation and graph updating,
    allowing for efficient computation of all steps and later indexing of a random step.

    Args:
        model: The CircuitSelfAttention model to apply
        graph: Initial graph state
        num_steps: Number of steps to run
        logits_original_shapes: Original shapes of logits for reconstruction
        wires: Wire connection patterns
        x_data: Input data
        y_data: Target output data
        loss_type: Type of loss function to use
        layer_sizes: List of (nodes, group_size) tuples for each layer

    Returns:
        final_graph: The graph after all steps
        all_graphs: List of graphs from each step (including initial)
        all_losses: Array of losses from each step [num_steps+1]
        all_aux: List of auxiliary data from each step
    """
    from boolean_nca_cc.training.evaluation import get_loss_and_update_graph

    # --- Compute the mask *once* before the scan ---
    attention_mask = model._create_attention_mask(
        graph.senders, graph.receivers, bidirectional=True
    )
    # ---------------------------------------------

    def attention_step_with_loss(carry, _):
        current_graph = carry

        # Apply self-attention with precomputed mask
        model_updated_graph = model(current_graph, attention_mask=attention_mask)

        # Compute loss and update graph
        updated_graph, loss, current_logits, aux = get_loss_and_update_graph(
            model_updated_graph,
            logits_original_shapes,
            wires,
            x_data,
            y_data,
            loss_type,
            layer_sizes,
        )

        # Update graph globals with current update steps
        current_update_steps = (
            updated_graph.globals[..., 1] if updated_graph.globals is not None else 0
        )
        final_graph = updated_graph._replace(
            globals=jp.array([loss, current_update_steps + 1], dtype=jp.float32)
        )

        return final_graph, (final_graph, loss, current_logits, aux)

    # Run scan
    final_graph, step_outputs = jax.lax.scan(
        attention_step_with_loss, graph, xs=None, length=num_steps
    )

    return final_graph, step_outputs
