"""
Perceiver-style circuit optimization with cross-attention to input/output data.

This module extends the self-attention approach by adding cross-attention
to input data patterns and output residuals, giving the model access to
the information that backpropagation uses for credit assignment.

Key differences from pure self-attention:
- Gates cross-attend to input data → understand data distribution/correlations
- Gates cross-attend to output residuals → understand per-sample, per-bit errors
- Self-attention among gates → propagate information through circuit topology

The graph structure is extended to include (via GraphGlobals NamedTuple):
- globals.x_data: Input data batch [N_samples, N_input_bits]
- globals.y_data: Target output batch [N_samples, N_output_bits]
- globals.residuals: Current prediction errors [N_samples, N_output_bits]
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
from boolean_nca_cc.utils.positional_encoding import get_positional_encoding


class PerceiverCircuitAttention(nnx.Module):
    """
    Perceiver-style circuit optimizer with cross-attention to input/output data.

    Architecture per step:
    1. Encode input data as tokens (bit values + positions)
    2. Encode output residuals as tokens (error values + positions)
    3. Gate latents cross-attend to input tokens
    4. Gate latents cross-attend to output tokens
    5. Gate latents self-attend (with circuit topology mask)
    6. Project to logit/hidden updates

    Uses ReZero throughout for stable training.
    """

    def __init__(
        self,
        n_node: int,
        circuit_hidden_dim: int = 16,
        arity: int = 2,
        attention_dim: int = 128,
        num_heads: int = 4,
        num_self_attn_layers: int = 1,
        num_cross_attn_layers: int = 1,
        mlp_dim: int | None = None,
        mlp_dim_multiplier: int = 2,
        dropout_rate: float = 0.0,
        use_attention_mask: bool = True,
        *,
        rngs: nnx.Rngs,
        type: str = "perceiver_attention",
        use_node_loss: bool = False,
        # Perceiver-specific options
        use_input_cross_attention: bool = True,
        use_output_cross_attention: bool = True,
        token_pe_dim: int = 8,
        # Structural constraints
        restrict_input_cross_attn_to_first_layer: bool = False,
        restrict_output_cross_attn_to_last_layer: bool = False,
    ):
        """
        Initialize the Perceiver-style circuit attention model.

        Args:
            n_node: Fixed number of nodes in the circuit
            circuit_hidden_dim: Dimension of hidden features in the circuit graphs
            arity: Number of inputs per gate
            attention_dim: Internal attention dimension
            num_heads: Number of attention heads
            num_self_attn_layers: Number of self-attention layers
            num_cross_attn_layers: Number of cross-attention layers
            mlp_dim: Dimension of feed-forward network
            mlp_dim_multiplier: Multiplier for mlp_dim if not specified
            dropout_rate: Dropout rate
            use_attention_mask: Whether to use topology-based attention mask
            rngs: Random number generators
            type: Model type identifier
            use_node_loss: Whether to include per-node loss in features
            use_input_cross_attention: Enable cross-attention to input data
            use_output_cross_attention: Enable cross-attention to output residuals
            token_pe_dim: Dimension for sinusoidal positional encodings
            restrict_input_cross_attn_to_first_layer: Only first gate layer attends to inputs
            restrict_output_cross_attn_to_last_layer: Only output layer attends to residuals
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
        self.use_input_cross_attention = use_input_cross_attention
        self.use_output_cross_attention = use_output_cross_attention
        self.restrict_input_cross_attn_to_first_layer = restrict_input_cross_attn_to_first_layer
        self.restrict_output_cross_attn_to_last_layer = restrict_output_cross_attn_to_last_layer
        self.token_pe_dim = token_pe_dim

        if mlp_dim is None:
            mlp_dim = attention_dim * mlp_dim_multiplier

        if attention_dim % num_heads != 0:
            raise ValueError(
                f"attention_dim ({attention_dim}) must be divisible by num_heads ({num_heads})"
            )

        # === Gate feature projection ===
        input_feature_dim = self.logit_dim + circuit_hidden_dim * 3  # logits + hidden + 2 PEs
        if self.use_node_loss:
            input_feature_dim += 1

        self.feature_proj = nnx.Linear(
            input_feature_dim,
            self.attention_dim,
            rngs=rngs,
            kernel_init=nnx.initializers.kaiming_normal(),
        )

        def create_cross_attention_layers() -> tuple[nnx.Sequential, nnx.List]:
            # Encodes (bit_value, sample_pe, bit_pe) -> attention_dim
            token_input_dim = 1 + 2 * token_pe_dim
            # === Input data encoder ===
            encoder = nnx.Sequential(
                nnx.Linear(
                    token_input_dim,
                    attention_dim,
                    rngs=rngs,
                    kernel_init=nnx.initializers.kaiming_normal(),
                ),
                nnx.gelu,
                nnx.Linear(
                    attention_dim,
                    attention_dim,
                    rngs=rngs,
                    kernel_init=nnx.initializers.kaiming_normal(),
                ),
            )
            # === Cross-attention layers (using shared AttentionBlock with ReZero) ===
            cross_attn_layers = nnx.List(
                [
                    AttentionBlock(
                        dim=attention_dim,
                        mlp_dim=mlp_dim,
                        num_heads=num_heads,
                        dropout_rate=dropout_rate,
                        rngs=rngs,
                    )
                    for _ in range(num_cross_attn_layers)
                ]
            )
            return encoder, cross_attn_layers

        # === Cross-attention to input data ===
        if use_input_cross_attention:
            self.input_encoder, self.input_cross_attn_layers = create_cross_attention_layers()

        # === Cross-attention to output residuals ===
        if use_output_cross_attention:
            self.output_encoder, self.output_cross_attn_layers = create_cross_attention_layers()

        # === Self-attention layers ===
        self.self_attn_layers = nnx.List(
            [
                AttentionBlock(
                    dim=self.attention_dim,
                    mlp_dim=mlp_dim,
                    num_heads=num_heads,
                    dropout_rate=dropout_rate,
                    rngs=rngs,
                )
                for _ in range(num_self_attn_layers)
            ]
        )

        # === Output projections ===
        self.logit_proj = nnx.Linear(
            self.attention_dim,
            self.logit_dim,
            use_bias=True,
            kernel_init=nnx.initializers.kaiming_normal(),
            rngs=rngs,
        )
        self.hidden_proj = nnx.Linear(
            self.attention_dim,
            circuit_hidden_dim,
            use_bias=True,
            kernel_init=nnx.initializers.kaiming_normal(),
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
        """Create topology-based attention mask using shared utility."""
        return create_attention_mask(senders, receivers, n_node, self.use_attention_mask)

    def _create_output_gate(
        self,
        layer_indices: jp.ndarray,
        allowed_layer: int,
    ) -> jp.ndarray:
        """
        Create a hard gate for cross-attention output.

        This is applied AFTER attention to zero out contributions for nodes
        that shouldn't attend. This is necessary because softmax of a fully-masked
        row gives uniform attention rather than zero.

        Args:
            layer_indices: Layer index for each node [N_nodes]
            allowed_layer: Which layer should receive cross-attention updates

        Returns:
            Output gate [1, N_nodes, 1] where 1.0 = receive update, 0.0 = zero out
        """
        gate = (layer_indices == allowed_layer).astype(jp.float32)  # [N_nodes]
        return gate[None, :, None]  # [1, N_nodes, 1]

    def _encode_data(self, data: jp.ndarray) -> jp.ndarray:
        """
        Encode data as tokens with sinusoidal positional encodings.

        Args:
            data: Input data [N_samples, N_input_bits]

        Returns:
            Encoded tokens [N_samples * N_input_bits, attention_dim]
        """
        N_samples, N_bits = data.shape

        # Sinusoidal positional encodings
        sample_pe = get_positional_encoding(jp.arange(N_samples), self.token_pe_dim)
        bit_pe = get_positional_encoding(jp.arange(N_bits), self.token_pe_dim)

        # Broadcast to [N_samples, N_bits, pe_dim]
        sample_pe_broadcast = jp.broadcast_to(
            sample_pe[:, None, :], (N_samples, N_bits, self.token_pe_dim)
        )
        bit_pe_broadcast = jp.broadcast_to(
            bit_pe[None, :, :], (N_samples, N_bits, self.token_pe_dim)
        )
        # Concatenate [value, sample_pe, bit_pe] and flatten
        features = jp.concatenate(
            [data[:, :, None], sample_pe_broadcast, bit_pe_broadcast], axis=-1
        )
        features = features.reshape(-1, 1 + 2 * self.token_pe_dim)

        return features

    def __call__(
        self,
        graph: jraph.GraphsTuple,
        attention_mask: jp.ndarray | None = None,
        input_cross_attn_mask: jp.ndarray | None = None,
        output_cross_attn_mask: jp.ndarray | None = None,
        input_output_gate: jp.ndarray | None = None,
        output_output_gate: jp.ndarray | None = None,
        return_intermediate_latents: bool = False,
    ) -> jraph.GraphsTuple:
        """
        Apply Perceiver-style attention to update circuit parameters.

        Args:
            graph: Input graph with data in globals (GraphGlobals NamedTuple)
            attention_mask: Optional pre-computed self-attention mask
            input_cross_attn_mask: Optional pre-computed mask for input cross-attention
            output_cross_attn_mask: Optional pre-computed mask for output cross-attention
            input_output_gate: Optional pre-computed output gate for input cross-attention
                [1, N_nodes, 1] - hard zeros non-allowed layers' contributions
            output_output_gate: Optional pre-computed output gate for output cross-attention
                [1, N_nodes, 1] - hard zeros non-allowed layers' contributions

        Returns:
            Updated graph with new logits and hidden states
        """
        nodes, _edges, receivers, senders, globals_, _n_node, _n_edge = graph

        # Extract data from globals
        x_data = globals_.x_data if globals_ is not None else None
        residuals = globals_.residuals if globals_ is not None else None

        # Get layer information for structural constraints
        layer_indices = nodes["layer"]
        max_layer = jp.max(layer_indices)

        # Encode gate features
        gate_features = extract_node_features(nodes, self.use_node_loss)
        gate_latents = self.feature_proj(gate_features)[None, ...]  # [1, N_gates, dim]

        # Store intermediate latents
        intermediate_latents = []
        if return_intermediate_latents:
            intermediate_latents.append(gate_latents.copy())

        # === Cross-attention to input data ===
        if self.use_input_cross_attention and x_data is not None:
            input_features = self._encode_data(x_data)
            input_tokens = self.input_encoder(input_features)[None, ...]

            # Create input gate if layer restriction is enabled
            if input_output_gate is None and self.restrict_input_cross_attn_to_first_layer:
                input_output_gate = self._create_output_gate(layer_indices, allowed_layer=0)

            for cross_attn in self.input_cross_attn_layers:
                gate_latents = cross_attn(
                    gate_latents,
                    input_tokens,
                    mask=input_cross_attn_mask,  # Can still be passed externally for fine control
                    output_gate=input_output_gate,
                )
                if return_intermediate_latents:
                    intermediate_latents.append(gate_latents.copy())

        # === Cross-attention to output residuals ===
        if self.use_output_cross_attention and residuals is not None:
            output_features = self._encode_data(residuals)
            output_tokens = self.output_encoder(output_features)[None, ...]

            # Create output gate if layer restriction is enabled
            if output_output_gate is None and self.restrict_output_cross_attn_to_last_layer:
                output_output_gate = self._create_output_gate(
                    layer_indices, allowed_layer=max_layer
                )

            for cross_attn in self.output_cross_attn_layers:
                gate_latents = cross_attn(
                    gate_latents,
                    output_tokens,
                    mask=output_cross_attn_mask,  # Can still be passed externally for fine control
                    output_gate=output_output_gate,
                )
                if return_intermediate_latents:
                    intermediate_latents.append(gate_latents.copy())

        # === Self-attention among gates ===
        if attention_mask is None:
            attention_mask = self._create_attention_mask(senders, receivers, self.n_node)

        for layer in self.self_attn_layers:
            gate_latents = layer(gate_latents, key_value=None, mask=attention_mask)
            if return_intermediate_latents:
                intermediate_latents.append(gate_latents.copy())

        # === Project to updates ===
        logit_updates = self.logit_proj(gate_latents)[0]
        hidden_updates = self.hidden_proj(gate_latents)[0]

        # Apply knockout mask
        logit_updates, hidden_updates = apply_knockout_mask(logit_updates, hidden_updates, nodes)

        # Apply ReZero residual updates
        updated_logits = nodes["logits"] + self.logit_rezero(logit_updates)
        updated_hidden = nodes["hidden"] + self.hidden_rezero(hidden_updates)

        updated_nodes = {**nodes, "logits": updated_logits, "hidden": updated_hidden}

        if return_intermediate_latents:
            return graph._replace(nodes=updated_nodes), intermediate_latents
        else:
            return graph._replace(nodes=updated_nodes)


# =============================================================================
# Scan functions for iterative optimization
# =============================================================================


@partial(nnx.jit, static_argnames=("num_steps",))
def run_perceiver_scan(
    model: PerceiverCircuitAttention,
    graph: jraph.GraphsTuple,
    num_steps: int,
) -> tuple[jraph.GraphsTuple, list[jraph.GraphsTuple]]:
    """
    Apply the Perceiver model iteratively for multiple steps.

    Note: Requires graph.globals to contain x_data and residuals.
    """
    # Precompute all masks and gates once
    attention_mask = model._create_attention_mask(graph.senders, graph.receivers, model.n_node)

    layer_indices = graph.nodes["layer"]
    max_layer = jp.max(layer_indices)

    # Precompute output gates for layer restrictions
    # Note: We only need output gates (not attention masks) for layer-based restrictions
    # because the output gate hard-zeros contributions for non-allowed layers.
    input_output_gate = None
    if model.restrict_input_cross_attn_to_first_layer and graph.globals.x_data is not None:
        input_output_gate = model._create_output_gate(layer_indices, allowed_layer=0)

    output_output_gate = None
    if model.restrict_output_cross_attn_to_last_layer and graph.globals.residuals is not None:
        output_output_gate = model._create_output_gate(layer_indices, allowed_layer=max_layer)

    def scan_body(carry_graph, _):
        updated_graph = model(
            carry_graph,
            attention_mask=attention_mask,
            input_output_gate=input_output_gate,
            output_output_gate=output_output_gate,
        )
        return updated_graph, updated_graph

    final_graph, intermediate_graphs = jax.lax.scan(scan_body, graph, None, length=num_steps)
    all_graphs = [graph, *list(intermediate_graphs)]

    return final_graph, all_graphs


def run_perceiver_scan_with_loss(
    model: PerceiverCircuitAttention,
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
    Run the Perceiver model for multiple steps with loss computation at each step.

    Args:
        model: The PerceiverCircuitAttention model
        graph: Initial graph state
        num_steps: Number of optimization steps
        logits_original_shapes: Original shapes of logits for reconstruction
        wires: Wire connection patterns
        x_data: Input data [N_samples, N_input_bits]
        y_data: Target output [N_samples, N_output_bits]
        loss_cfg: Loss configuration
        layer_sizes: Layer sizes for graph operations
        data_fraction: Fraction of data to use
        scan_key: Random key for data sampling
        gradient_checkpointing: Whether to use gradient checkpointing

    Returns:
        final_graph: Graph after all steps
        step_outputs: Tuple of outputs from each step
    """
    from boolean_nca_cc.training.evaluation import get_loss_and_update_graph

    # Precompute masks
    attention_mask = model._create_attention_mask(graph.senders, graph.receivers, model.n_node)

    # Select data subset if needed
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

    # Precompute output gates for layer restrictions
    layer_indices = graph.nodes["layer"]
    max_layer = jp.max(layer_indices)

    input_output_gate = None
    if model.restrict_input_cross_attn_to_first_layer:
        input_output_gate = model._create_output_gate(layer_indices, allowed_layer=0)

    output_output_gate = None
    if model.restrict_output_cross_attn_to_last_layer:
        output_output_gate = model._create_output_gate(layer_indices, allowed_layer=max_layer)

    # Update initial graph with residuals
    graph, _, _, aux_data = get_loss_and_update_graph(
        graph,
        logits_original_shapes,
        wires,
        x_batch,
        y_batch,
        loss_cfg,
        layer_sizes,
        update_perceiver_globals=True,
    )

    # Optionally wrap with gradient checkpointing
    if gradient_checkpointing:
        model_fn = nnx.remat(
            lambda g: model(
                g,
                attention_mask=attention_mask,
                input_output_gate=input_output_gate,
                output_output_gate=output_output_gate,
            )
        )
    else:
        model_fn = lambda g: model(  # noqa: E731
            g,
            attention_mask=attention_mask,
            input_output_gate=input_output_gate,
            output_output_gate=output_output_gate,
        )

    def perceiver_step_with_loss(carry, _):
        current_graph = carry

        # Apply model
        model_updated_graph = model_fn(current_graph)

        # Compute loss, update graph, and update GraphGlobals with new residuals
        # (avoids redundant circuit evaluation since residuals are already computed)
        final_graph, loss, current_logits, aux = get_loss_and_update_graph(
            model_updated_graph,
            logits_original_shapes,
            wires,
            x_batch,
            y_batch,
            loss_cfg,
            layer_sizes,
            update_perceiver_globals=True,
        )

        return final_graph, (final_graph, loss, current_logits, aux)

    final_graph, step_outputs = jax.lax.scan(
        perceiver_step_with_loss, graph, xs=None, length=num_steps
    )

    return final_graph, step_outputs
