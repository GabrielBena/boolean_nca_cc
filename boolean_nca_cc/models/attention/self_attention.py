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
from boolean_nca_cc.utils.graph_builder import GraphGlobals


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
        num_layers: int = 1,
        mlp_dim: int | None = None,
        mlp_dim_multiplier: int = 2,
        dropout_rate: float = 0.0,
        use_attention_mask: bool = True,
        *,
        rngs: nnx.Rngs,
        type: str = "self_attention",
        zero_init: bool = True,
        use_node_loss: bool = False,
    ):
        """
        Initialize the circuit self-attention model.

        Args:
            n_node: Fixed number of nodes in the circuit
            circuit_hidden_dim: Dimension of hidden features in the circuit graphs
            arity: Number of inputs per gate in the boolean circuit
            attention_dim: Total dimension for attention mechanism
            num_heads: Number of attention heads
            num_layers: Number of self-attention layers
            mlp_dim: Dimension of feed-forward network in attention blocks
            mlp_dim_multiplier: Multiplier for mlp_dim (default 2)
            dropout_rate: Dropout rate
            use_attention_mask: Whether to use self-attention masks based on circuit wiring
            rngs: Random number generators
            type: Type of model
            zero_init: Whether to initialize output projection weights to zero
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
                )
                for _ in range(num_layers)
            ]
        )

        # Output projections with optional zero init
        self.logit_proj = nnx.Linear(
            self.attention_dim,
            self.logit_dim,
            use_bias=True,
            kernel_init=nnx.initializers.zeros if zero_init else nnx.initializers.lecun_normal(),
            bias_init=nnx.initializers.zeros,
            rngs=rngs,
        )
        self.hidden_proj = nnx.Linear(
            self.attention_dim,
            circuit_hidden_dim,
            use_bias=True,
            kernel_init=nnx.initializers.zeros if zero_init else nnx.initializers.lecun_normal(),
            bias_init=nnx.initializers.zeros,
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
    Run the self-attention model for multiple steps with loss computation.

    Args:
        model: The CircuitSelfAttention model to apply
        graph: Initial graph state
        num_steps: Number of steps to run
        logits_original_shapes: Original shapes of logits for reconstruction
        wires: Wire connection patterns
        x_data: Input data
        y_data: Target output data
        loss_cfg: LossConfig object
        layer_sizes: List of (nodes, group_size) tuples for each layer
        data_fraction: Fraction of data to use for loss computation
        scan_key: Random key for data sampling
        gradient_checkpointing: If True, recompute model activations during backward pass

    Returns:
        final_graph: The graph after all steps
        step_outputs: Outputs from each step
    """
    from boolean_nca_cc.training.evaluation import get_loss_and_update_graph

    # Compute mask once before the scan
    attention_mask = model._create_attention_mask(graph.senders, graph.receivers, model.n_node)

    # Select random fraction of data if needed
    if data_fraction < 1.0:
        random_indices = jax.random.randint(
            key=scan_key,
            shape=(int(x_data.shape[0] * data_fraction),),
            minval=0,
            maxval=x_data.shape[0],
        )
        x_batch = x_data[random_indices]
        y_batch = y_data[random_indices]
    else:
        x_batch = x_data
        y_batch = y_data

    # Optionally wrap with gradient checkpointing
    if gradient_checkpointing:
        model_fn = nnx.remat(lambda g: model(g, attention_mask=attention_mask))
    else:
        model_fn = lambda g: model(g, attention_mask=attention_mask)  # noqa: E731

    def attention_step_with_loss(carry, _):
        current_graph = carry

        # Apply self-attention
        model_updated_graph = model_fn(current_graph)

        # Compute loss and update graph
        updated_graph, loss, current_logits, aux = get_loss_and_update_graph(
            model_updated_graph,
            logits_original_shapes,
            wires,
            x_batch,
            y_batch,
            loss_cfg,
            layer_sizes,
        )

        # Update graph globals
        current_update_steps = (
            updated_graph.globals.update_steps if updated_graph.globals is not None else 0
        )
        final_graph = updated_graph._replace(
            globals=GraphGlobals(
                loss=loss,
                update_steps=current_update_steps + 1,
            )
        )

        return final_graph, (final_graph, loss, current_logits, aux)

    # Run scan
    final_graph, step_outputs = jax.lax.scan(
        attention_step_with_loss, graph, xs=None, length=num_steps
    )

    return final_graph, step_outputs
