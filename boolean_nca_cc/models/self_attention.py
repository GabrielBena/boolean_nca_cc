"""
Self-attention based circuit update mechanism.

This module provides an alternative to GNN message passing by using
masked self-attention to update boolean circuit parameters.
"""

from functools import partial

import jax
import jax.numpy as jp
import jraph
from flax import nnx


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
        attention_mask: jp.ndarray | None = None,
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
        attention_mask: jp.ndarray | None = None,
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
        circuit_hidden_dim: int = 16,
        arity: int = 2,
        attention_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 3,
        mlp_dim: int | None = None,
        mlp_dim_multiplier: int = 2,
        dropout_rate: float = 0.0,
        *,
        rngs: nnx.Rngs,
        type: str = "self_attention",
        zero_init: bool = True,
        re_zero_update: bool = False,
    ):
        """
        Initialize the circuit self-attention model.

        Args:
            n_node: Fixed number of nodes in the circuit
            circuit_hidden_dim: Dimension of hidden features in the circuit graphs (interface constraint)
            arity: Number of inputs per gate in the boolean circuit
            attention_dim: Total dimension for attention mechanism (model's internal working dimension)
            num_heads: Number of attention heads
            num_layers: Number of self-attention layers
            mlp_dim: Dimension of feed-forward network in attention blocks
            mlp_dim_multiplier: Multiplier for mlp_dim (default 2 for efficient transformer pattern)
            dropout_rate: Dropout rate
            rngs: Random number generators
            type: Type of model
            zero_init: Whether to initialize weights to zero
            re_zero_update: Wether to use learnable update residual rate
        """
        self.n_node = int(n_node)
        self.arity = arity
        self.circuit_hidden_dim = circuit_hidden_dim
        self.attention_dim = attention_dim
        self.hidden_dim_per_head = attention_dim // num_heads
        self.logit_dim = 2**arity
        self.dropout_rate = dropout_rate
        self.num_heads = num_heads
        self.deterministic = True

        if (
            mlp_dim is None
        ):  # default to mlp_dim_multiplier * attention_dim (efficient transformer pattern)
            mlp_dim = attention_dim * mlp_dim_multiplier

        # Validate that attention_dim is divisible by num_heads
        if attention_dim % num_heads != 0:
            raise ValueError(
                f"attention_dim ({attention_dim}) must be divisible by num_heads ({num_heads})"
            )

        # Compute the total input feature dimension (logits + circuit_hidden + positional encodings + loss)
        input_feature_dim = (
            self.logit_dim + circuit_hidden_dim * 3 + 1
        )  # logits + circuit_hidden + 2 PE's + loss

        # Input projection: from circuit features to model's internal attention dimension
        self.feature_proj = nnx.Linear(input_feature_dim, self.attention_dim, rngs=rngs)

        # Self-attention layers
        self.attention_layers = [
            SelfAttentionBlock(
                feature_dim=self.attention_dim,
                mlp_dim=mlp_dim,
                num_heads=num_heads,
                dropout_rate=dropout_rate,
                rngs=rngs,
            )
            for _ in range(num_layers)
        ]

        # Output projections for both logits and hidden features
        self.logit_proj = nnx.Linear(
            self.attention_dim,
            self.logit_dim,
            use_bias=True,
            kernel_init=nnx.initializers.zeros if zero_init else nnx.initializers.kaiming_normal(),
            bias_init=nnx.initializers.zeros if zero_init else nnx.initializers.normal(stddev=1e-4),
            rngs=rngs,
        )
        self.logit_scale = (
            nnx.Param(
                jp.zeros(1),
                name="logit_scale",
                rngs=rngs,
            )
            if re_zero_update
            else 1.0
        )

        self.hidden_proj = nnx.Linear(
            self.attention_dim,
            circuit_hidden_dim,
            use_bias=True,
            kernel_init=nnx.initializers.zeros if zero_init else nnx.initializers.kaiming_normal(),
            bias_init=nnx.initializers.zeros if zero_init else nnx.initializers.normal(stddev=1e-4),
            rngs=rngs,
        )
        self.hidden_scale = (
            nnx.Param(
                jp.zeros(1),
                name="hidden_scale",
                rngs=rngs,
            )
            if re_zero_update
            else 1.0
        )

    def _create_attention_mask(
        self,
        senders: jp.ndarray,
        receivers: jp.ndarray,
        n_node: int,
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
        mask = jp.zeros((n_node, n_node), dtype=jp.bool_)

        if len(senders) > 0:
            # Set mask[receiver, sender] = True for all edges
            mask = mask.at[receivers, senders].set(True)

        # Add self-connections (diagonal of True)
        mask = mask | jp.eye(n_node, dtype=jp.bool_)

        # Add batch dimension and singleton head dimension [batch_size, 1, seq_len, seq_len]
        # This format matches the MultiHeadAttention mask format
        return mask[None, None, ...]

    def _extract_features(self, nodes: dict[str, jp.ndarray]) -> jp.ndarray:
        """
        Extract and concatenate node features for attention.

        Args:
            nodes: Dictionary of node features

        Returns:
            Concatenated features tensor of shape [n_node, feature_dim]
        """
        # Extract relevant features
        logits = nodes["logits"]  # [n_node, logit_dim]
        hidden = nodes["hidden"]  # [n_node, circuit_hidden_dim]
        # Positional encodings
        layer_pe = nodes["layer_pe"]  # [n_node, circuit_hidden_dim]
        intra_layer_pe = nodes["intra_layer_pe"]  # [n_node, circuit_hidden_dim]
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
        attention_mask: jp.ndarray | None = None,
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
            attention_mask = self._create_attention_mask(senders, receivers, self.n_node)

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

        # Check for gate knockout mask to prevent updates to knocked-out gates
        gate_knockout_mask = nodes.get("gate_knockout_mask", None)
        if gate_knockout_mask is not None:
            # Knocked-out gates (mask == 0.0) should not receive updates
            # Only allow updates to active gates (mask == 1.0)
            update_allowed_logits = gate_knockout_mask == 1.0
            update_allowed_hidden = gate_knockout_mask == 1.0

            # Ensure mask matches feature dimensions for broadcasting
            update_allowed_logits_mask = update_allowed_logits[:, None]
            update_allowed_hidden_mask = update_allowed_hidden[:, None]

            # Apply updates only to allowed gates, zero out updates for knocked-out gates
            logit_updates = jp.where(update_allowed_logits_mask, logit_updates, 0.0)
            hidden_updates = jp.where(update_allowed_hidden_mask, hidden_updates, 0.0)

        # Update logits and hidden features in a residual manner
        updated_logits = nodes["logits"] + self.logit_scale * logit_updates
        updated_hidden = nodes["hidden"] + self.hidden_scale * hidden_updates

        # Create updated nodes dictionary
        updated_nodes = {**nodes, "logits": updated_logits, "hidden": updated_hidden}

        # Return updated graph
        return graph._replace(nodes=updated_nodes)


@partial(nnx.jit, static_argnames=("num_steps"))
def run_self_attention_scan(
    model: CircuitSelfAttention,
    graph: jraph.GraphsTuple,
    num_steps: int,
) -> tuple[jraph.GraphsTuple, list[jraph.GraphsTuple]]:
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
    attention_mask = model._create_attention_mask(graph.senders, graph.receivers, model.n_node)
    # ---------------------------------------------

    def scan_body(carry_graph, _):
        # Apply one step of self-attention with precomputed mask
        updated_graph = model(carry_graph, attention_mask=attention_mask)
        return updated_graph, updated_graph

    # Run the scan
    final_graph, intermediate_graphs = jax.lax.scan(scan_body, graph, None, length=num_steps)

    # Combine initial graph with intermediate results
    all_graphs = [graph, *list(intermediate_graphs)]

    return final_graph, all_graphs


def run_self_attention_scan_with_loss(
    model: CircuitSelfAttention,
    graph: jraph.GraphsTuple,
    num_steps: int,
    logits_original_shapes: list[tuple],
    wires: list[jp.ndarray],
    x_data: jp.ndarray,
    y_data: jp.ndarray,
    loss_type: str,
    layer_sizes: tuple[tuple[int, int], ...],
) -> tuple[jraph.GraphsTuple, list[jraph.GraphsTuple], jp.ndarray, list]:
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
    attention_mask = model._create_attention_mask(graph.senders, graph.receivers, model.n_node)
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
