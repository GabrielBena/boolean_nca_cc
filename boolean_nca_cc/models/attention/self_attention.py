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

from boolean_nca_cc.circuits.train import LossConfig
from boolean_nca_cc.models.attention.base import (
    AttentionBlock,
    ReZero,
    apply_knockout_mask,
    create_attention_mask,
    extract_node_features,
)


class CircuitSelfAttention(nnx.Module):
    """
    Self-attention based circuit update mechanism.

    Implements masked self-attention to update boolean circuit parameters.
    Uses ReZero for stable training throughout.
    """

    def __init__(
        self,
        n_node: int,
        circuit_hidden_dim: int = 16,
        arity: int = 2,
        attention_dim: int = 128,
        num_heads: int = 4,
        num_self_attn_layers: int = 1,
        mlp_dim: int | None = None,
        mlp_dim_multiplier: int = 2,
        dropout_rate: float = 0.0,
        use_attention_mask: bool = True,
        *,
        rngs: nnx.Rngs,
        type: str = "self_attention",
        use_node_loss: bool = False,
        re_zero_attn: bool = True,
    ):
        """
        Initialize the circuit self-attention model.

        Args:
            n_node: Fixed number of nodes in the circuit
            circuit_hidden_dim: Dimension of hidden features in the circuit graphs
            arity: Number of inputs per gate in the boolean circuit
            attention_dim: Total dimension for attention mechanism
            num_heads: Number of attention heads
            num_self_attn_layers: Number of self-attention layers
            mlp_dim: Dimension of feed-forward network in attention blocks
            mlp_dim_multiplier: Multiplier for mlp_dim (default 2)
            dropout_rate: Dropout rate
            use_attention_mask: Whether to use self-attention masks based on circuit wiring
            rngs: Random number generators
            type: Type of model
            use_node_loss: Whether to include per-node loss in features
        """
        self.n_node = int(n_node)
        self.arity = arity
        self.circuit_hidden_dim = circuit_hidden_dim
        self.attention_dim = attention_dim
        self.logit_dim = 2**arity
        self.dropout_rate = dropout_rate
        self.num_heads = num_heads
        self.use_attention_mask = use_attention_mask
        self.use_node_loss = use_node_loss

        if mlp_dim is None:
            mlp_dim = attention_dim * mlp_dim_multiplier

        if attention_dim % num_heads != 0:
            raise ValueError(
                f"attention_dim ({attention_dim}) must be divisible by num_heads ({num_heads})"
            )

        # Compute input feature dimension
        input_feature_dim = self.logit_dim + circuit_hidden_dim * 3  # logits + hidden + 2 PEs
        if self.use_node_loss:
            input_feature_dim += 1

        # Input projection
        self.feature_proj = nnx.Linear(input_feature_dim, self.attention_dim, rngs=rngs)

        # Self-attention layers (using shared AttentionBlock with ReZero)
        self.attention_layers = nnx.List(
            [
                AttentionBlock(
                    dim=self.attention_dim,
                    mlp_dim=mlp_dim,
                    num_heads=num_heads,
                    dropout_rate=dropout_rate,
                    rngs=rngs,
                    re_zero=re_zero_attn,
                )
                for _ in range(num_self_attn_layers)
            ]
        )

        # Output projections with optional zero init
        self.logit_proj = nnx.Linear(
            self.attention_dim,
            self.logit_dim,
            use_bias=True,
            kernel_init=nnx.initializers.lecun_normal(),
            rngs=rngs,
        )
        self.hidden_proj = nnx.Linear(
            self.attention_dim,
            circuit_hidden_dim,
            use_bias=True,
            kernel_init=nnx.initializers.lecun_normal(),
            rngs=rngs,
        )

        # ReZero for final output residuals
        self.logit_rezero = ReZero(rngs=rngs)
        self.hidden_rezero = ReZero(rngs=rngs)

    def _create_attention_mask(
        self,
        senders: jp.ndarray,
        receivers: jp.ndarray,
        n_node: int,
    ) -> jp.ndarray:
        """Create attention mask using shared utility."""
        return create_attention_mask(senders, receivers, n_node, self.use_attention_mask)

    def __call__(
        self,
        graph: jraph.GraphsTuple,
        attention_mask: jp.ndarray | None = None,
    ) -> jraph.GraphsTuple:
        """
        Apply self-attention to update circuit parameters.

        Args:
            graph: Input graph structure with node and edge features
            attention_mask: Optional pre-computed attention mask

        Returns:
            Updated graph after self-attention
        """
        nodes, _edges, receivers, senders, _globals, _n_node, _n_edge = graph

        # Extract and concatenate node features
        features = extract_node_features(nodes, self.use_node_loss)

        # Add batch dimension [1, n_node, feature_dim]
        features = features[None, ...]

        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = self._create_attention_mask(senders, receivers, self.n_node)

        # Project features to attention dimension
        x = self.feature_proj(features)

        # Apply self-attention layers
        for layer in self.attention_layers:
            x = layer(x, key_value=None, mask=attention_mask)

        # Project to updates
        logit_updates = self.logit_proj(x)[0]  # Remove batch dim
        hidden_updates = self.hidden_proj(x)[0]

        # Apply knockout mask
        logit_updates, hidden_updates = apply_knockout_mask(logit_updates, hidden_updates, nodes)

        # Apply ReZero residual updates
        updated_logits = nodes["logits"] + self.logit_rezero(logit_updates)
        updated_hidden = nodes["hidden"] + self.hidden_rezero(hidden_updates)

        # Create updated nodes
        updated_nodes = {**nodes, "logits": updated_logits, "hidden": updated_hidden}

        return graph._replace(nodes=updated_nodes)


@partial(nnx.jit, static_argnames=("num_steps",))
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
    # Compute mask once before the scan
    attention_mask = model._create_attention_mask(graph.senders, graph.receivers, model.n_node)

    def scan_body(carry_graph, _):
        updated_graph = model(carry_graph, attention_mask=attention_mask)
        return updated_graph, updated_graph

    final_graph, intermediate_graphs = jax.lax.scan(scan_body, graph, None, length=num_steps)
    all_graphs = [graph, *list(intermediate_graphs)]

    return final_graph, all_graphs


# DEPRECATED: Use run_model_scan_with_loss from boolean_nca_cc.training.evaluation instead
# This function is kept for backward compatibility but will be removed in a future version.
def run_self_attention_scan_with_loss(
    model: CircuitSelfAttention,
    graph: jraph.GraphsTuple,
    num_steps: int,
    logits_original_shapes: list[tuple],
    wires: list[jp.ndarray],
    x_data: jp.ndarray,
    y_data: jp.ndarray,
    loss_cfg: LossConfig,
    layer_sizes: tuple[tuple[int, int], ...],
    data_fraction: float = 1.0,
    scan_key: jax.random.PRNGKey = None,
    gradient_checkpointing: bool = False,
) -> tuple[jraph.GraphsTuple, list[jraph.GraphsTuple], jp.ndarray, list]:
    """
    DEPRECATED: Use run_model_scan_with_loss from boolean_nca_cc.training.evaluation instead.

    This function wraps the unified run_model_scan_with_loss for backward compatibility.
    """
    import warnings

    from boolean_nca_cc.training.evaluation import run_model_scan_with_loss

    warnings.warn(
        "run_self_attention_scan_with_loss is deprecated. "
        "Use run_model_scan_with_loss from boolean_nca_cc.training.evaluation instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    return run_model_scan_with_loss(
        model=model,
        graph=graph,
        num_steps=num_steps,
        logits_original_shapes=logits_original_shapes,
        wires=wires,
        x_data=x_data,
        y_data=y_data,
        loss_cfg=loss_cfg,
        layer_sizes=layer_sizes,
        data_fraction=data_fraction,
        scan_key=scan_key,
        gradient_checkpointing=gradient_checkpointing,
    )
