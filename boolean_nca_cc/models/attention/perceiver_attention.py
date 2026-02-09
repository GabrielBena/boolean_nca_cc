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

import jax
import jax.numpy as jp
import jraph
from flax import nnx

from boolean_nca_cc.models.attention.base import (
    AttentionBlock,
    PassThrough,
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
        re_zero_attn: bool = True,
        warm_start: bool = True,
        re_zero_updates: bool = True,
        use_intra_layer_PE: bool = False,
        use_layer_PE: bool = True,
        # Perceiver-specific options
        use_input_cross_attention: bool = True,
        use_output_cross_attention: bool = True,
        token_pe_dim: int = 8,
        samples_per_step: int | None = None,
        # Structural constraints
        restrict_input_cross_attn_to_first_layer: bool = False,
        restrict_output_cross_attn_to_last_layer: bool = False,
    ):
        """
        Initialize the Perceiver-style circuit attention model.

        Args:
            n_node: Fixed number of nodes in the circuit (used for attention mask precomputation;
                    learned weights are scale-free and work with any circuit size)
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
            re_zero_attn: Whether to use ReZero for attention
            warm_start: Whether to use warm start for training
            re_zero_updates: Whether to use ReZero for updates
            rngs: Random number generators
            type: Model type identifier
            use_node_loss: Whether to include per-node loss in features
            use_intra_layer_PE: Whether to include intra-layer positional encodings in features
            use_layer_PE: Whether to include layer depth positional encodings in features
            use_input_cross_attention: Enable cross-attention to input data
            use_output_cross_attention: Enable cross-attention to output residuals
            token_pe_dim: Dimension for sinusoidal positional encodings
            samples_per_step: Number of data samples to subsample per NCA step for cross-attention.
                None = use all samples (no subsampling). When set, each step randomly selects
                this many samples from the data batch, creating an information bottleneck that
                promotes incremental, in-context learning over many recurrent steps.
                Reduces cross-attention compute from O(N_gates * N_samples * N_bits) to
                O(N_gates * samples_per_step * N_bits) per step.
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
        self.use_intra_layer_PE = use_intra_layer_PE
        self.use_layer_PE = use_layer_PE
        self.use_input_cross_attention = use_input_cross_attention
        self.use_output_cross_attention = use_output_cross_attention
        self.restrict_input_cross_attn_to_first_layer = restrict_input_cross_attn_to_first_layer
        self.restrict_output_cross_attn_to_last_layer = restrict_output_cross_attn_to_last_layer
        self.token_pe_dim = token_pe_dim
        self.samples_per_step = samples_per_step
        self.re_zero_attn = re_zero_attn
        self.re_zero_updates = re_zero_updates
        self.warm_start = warm_start

        if mlp_dim is None:
            mlp_dim = attention_dim * mlp_dim_multiplier

        if attention_dim % num_heads != 0:
            raise ValueError(
                f"attention_dim ({attention_dim}) must be divisible by num_heads ({num_heads})"
            )

        # === Gate feature projection ===
        # Compute input_feature_dim dynamically based on PE flags
        input_feature_dim = self.logit_dim + circuit_hidden_dim  # logits + hidden (always)
        if self.use_intra_layer_PE:
            input_feature_dim += circuit_hidden_dim
        if self.use_layer_PE:
            input_feature_dim += circuit_hidden_dim
        if self.use_node_loss:
            input_feature_dim += 1

        # Input normalization on concatenated heterogeneous features
        self.input_norm = nnx.LayerNorm(input_feature_dim, rngs=rngs)
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
                nnx.LayerNorm(token_input_dim, rngs=rngs),
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
                        re_zero=re_zero_attn,
                        warm_start=warm_start,
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
                    re_zero=re_zero_attn,
                    warm_start=warm_start,
                )
                for _ in range(num_self_attn_layers)
            ]
        )

        # Final LayerNorm before output heads (standard Pre-LN practice)
        self.final_norm = nnx.LayerNorm(self.attention_dim, rngs=rngs)

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
        self.logit_rezero = ReZero(rngs=rngs) if re_zero_updates else PassThrough()
        self.hidden_rezero = ReZero(rngs=rngs) if re_zero_updates else PassThrough()

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
        Encode data as tokens with normalized sinusoidal positional encodings.

        Uses normalized positions [0, 1] for both sample and bit indices,
        ensuring scale-free generalization across different data batch sizes
        and different numbers of input/output bits.

        The sample PE links corresponding samples across input/output cross-attention
        (same sample index → same PE). The bit PE distinguishes bit positions.

        Args:
            data: Input data [N_samples, N_bits]

        Returns:
            Encoded tokens [N_samples * N_bits, 1 + 2 * token_pe_dim]
        """
        N_samples, N_bits = data.shape

        # Normalized positional encodings: positions in [0, 1]
        # Scale by a fixed constant for good sinusoidal frequency spread
        pe_scale = 1000.0

        # Sample PE: normalized sample position [0, 1]
        normalized_sample_pos = jp.arange(N_samples, dtype=jp.float32) / jp.maximum(
            N_samples - 1, 1
        )
        sample_pe = get_positional_encoding(
            normalized_sample_pos * pe_scale, self.token_pe_dim, max_val=pe_scale
        )

        # Bit PE: normalized bit position [0, 1]
        normalized_bit_pos = jp.arange(N_bits, dtype=jp.float32) / jp.maximum(N_bits - 1, 1)
        bit_pe = get_positional_encoding(
            normalized_bit_pos * pe_scale, self.token_pe_dim, max_val=pe_scale
        )

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

        # === Stochastic sample subsampling ===
        # Subsample data at the SAMPLE level (not token level) so input and residual
        # cross-attention see matching data points with coherent bit structure.
        # Per-step keys derived via fold_in(base_key, update_steps) — stateless & unique.
        if (
            self.samples_per_step is not None
            and globals_ is not None
            and globals_.subsample_key is not None
        ):
            step_key = jax.random.fold_in(globals_.subsample_key, globals_.update_steps)
            # Determine sample count from whichever data is available
            n_samples = (
                x_data.shape[0]
                if x_data is not None
                else (residuals.shape[0] if residuals is not None else None)
            )
            if n_samples is not None:
                sample_indices = jax.random.randint(
                    step_key, shape=(self.samples_per_step,), minval=0, maxval=n_samples
                )
                # Use same indices for both: input/residual pairs stay linked
                if x_data is not None:
                    x_data = x_data[sample_indices]
                if residuals is not None:
                    residuals = residuals[sample_indices]

        # Get layer information for structural constraints
        layer_indices = nodes["layer"]
        max_layer = jp.max(layer_indices)

        # Encode gate features (normalize heterogeneous features before projection)
        gate_features = extract_node_features(
            nodes, self.use_node_loss, self.use_intra_layer_PE, self.use_layer_PE
        )
        gate_latents = self.feature_proj(self.input_norm(gate_features))[
            None, ...
        ]  # [1, N_gates, dim]

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
        # Derive n_node from graph (scale-free: works with any circuit size)
        # Note: this fallback path is only used outside JIT. In training/eval,
        # masks are always precomputed with a concrete n_node.
        if attention_mask is None:
            n_node = nodes["layer"].shape[0]
            attention_mask = self._create_attention_mask(senders, receivers, n_node)

        for layer in self.self_attn_layers:
            gate_latents = layer(gate_latents, key_value=None, mask=attention_mask)
            if return_intermediate_latents:
                intermediate_latents.append(gate_latents.copy())

        # === Final norm + project to updates ===
        gate_latents = self.final_norm(gate_latents)
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
