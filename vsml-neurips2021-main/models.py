import chex
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import itertools
import functools
import optax

from typing import Tuple, Callable, List, Optional, Iterable, Any
from config import configurable
import vsml_layers
import recurrent

# Opt-out of Haiku's vmap split_rng requirement for compatibility with older code
hk.vmap.require_split_rng = False

LayerSpec = Any


@chex.dataclass
class DenseSpec:
    in_size: int = -1
    out_size: int = -1


@chex.dataclass
class ConvSpec:
    in_width: int = -1
    in_height: int = -1
    in_channels: int = -1
    out_width: int = -1
    out_height: int = -1
    out_channels: int = -1
    kernel_size: int = 3
    stride: int = 1

    @property
    def out_size(self):
        return self.out_width * self.out_height * self.out_channels


SPEC_TYPES = {
    'dense': DenseSpec,
    'conv': ConvSpec,
}


def create_spec(cfg):
    cfg = cfg.copy()
    constr = SPEC_TYPES[cfg.pop('type')]
    return constr(**cfg)


def complete_specs(layer_specs, input_shape, output_size):
    last_idx = len(layer_specs) - 1
    for i, spec in enumerate(layer_specs):
        prev_spec = layer_specs[i - 1] if i > 0 else None
        if isinstance(spec, ConvSpec):
            if i == 0:
                spec.in_height, spec.in_width, spec.in_channels = input_shape
            else:
                spec.in_height = prev_spec.out_height
                spec.in_width = prev_spec.out_width
                spec.in_channels = prev_spec.out_channels
            spec.out_height = int(np.ceil(spec.in_height / spec.stride))
            spec.out_width = int(np.ceil(spec.in_width / spec.stride))
        elif isinstance(spec, DenseSpec):
            if i == 0:
                spec.in_size = np.prod(input_shape)
            else:
                spec.in_size = prev_spec.out_size
            if i == last_idx:
                spec.out_size = output_size
    return layer_specs


@chex.dataclass
class LayerState:
    lstm_state: hk.LSTMState
    incoming_fwd_msg: jnp.ndarray
    incoming_bwd_msg: jnp.ndarray


@configurable('model.sub_rnn')
class SubRNN(hk.Module):

    def __init__(self, slow_size: int, msg_size: int, init_rand_proportion: float, layer_norm: bool):
        """Initializes a Sub-RNN module, which is a core component of the VSMLRNN.

        Each SubRNN consists of an LSTM and two linear layers to produce forward and backward messages.
        These messages modulate the behavior of a specific part (e.g., a weight matrix or a filter)
        of a base network layer (Dense or Conv) in the VSMLRNN.
        This corresponds to f_RNN and f_m in Algorithm 2 and Figure 1 of the paper.

        Args:
            slow_size: The hidden size of the LSTM core (N in the paper's notation, Section C.2).
            msg_size: The size of the output messages (fwd_msg, bwd_msg) (N' and N'' in paper).
            init_rand_proportion: The proportion of the LSTM cell state to initialize randomly.
                                  The rest is initialized to zeros. See `CustomLSTM.initial_vsml_state`.
            layer_norm: If True, applies Layer Normalization to the output messages.
        """
        super().__init__()
        self._lstm = recurrent.CustomLSTM(slow_size)
        self._fwd_messenger = hk.Linear(msg_size)  # Corresponds to f_m for forward messages
        self._bwd_messenger = hk.Linear(msg_size)  # Corresponds to f_m for backward messages
        if layer_norm:
            self._fwd_layer_norm = hk.LayerNorm((-1,), create_scale=True, create_offset=True)
            self._bwd_layer_norm = hk.LayerNorm((-1,), create_scale=True, create_offset=True)
        self.msg_size = msg_size
        self._init_rand_proportion = init_rand_proportion
        self._use_layer_norm = layer_norm

    def __call__(self, fwd_msg: jnp.ndarray, bwd_msg: jnp.ndarray,
                 lstm_state: hk.LSTMState) -> Tuple[jnp.ndarray, jnp.ndarray, hk.LSTMState]:
        """Performs one step of the SubRNN computation.

        Concatenates incoming forward and backward messages, passes them through the LSTM,
        and then projects the LSTM output to produce new forward and backward messages.
        This implements Equation 7 from the paper: s_ab^(k) <- f_RNN(s_ab^(k), m_a^(k), m^_b^(k)),
        and the subsequent message generation m_b^(k+1) and m^_a^(k-1).

        Args:
            fwd_msg: The incoming forward message (m_a^(k) in Eq. 7).
            bwd_msg: The incoming backward message (m^_b^(k) in Eq. 7).
            lstm_state: The current LSTM state (s_ab^(k) in Eq. 7).

        Returns:
            A tuple containing:
                - new_fwd_msg: The outgoing forward message (part of input to m_b^(k+1) generation).
                - new_bwd_msg: The outgoing backward message (part of input to m^_a^(k-1) generation).
                - new_lstm_state: The updated LSTM state.
        """
        # Concatenate incoming messages to form the input to the LSTM
        inputs = jnp.concatenate([fwd_msg, bwd_msg], axis=-1)
        # LSTM core update (f_RNN in Eq. 7)
        outputs, lstm_state = self._lstm(inputs, lstm_state)
        # Generate new forward message using a linear projection (f_m)
        fwd_msg = self._fwd_messenger(outputs)
        # Generate new backward message using a linear projection (f_m)
        bwd_msg = self._bwd_messenger(outputs)
        if self._use_layer_norm:
            fwd_msg = self._fwd_layer_norm(fwd_msg)
            bwd_msg = self._bwd_layer_norm(bwd_msg)
        return fwd_msg, bwd_msg, lstm_state

    def initial_state(self, layer_spec: LayerSpec) -> hk.LSTMState:
        """Initializes the LSTM state for this SubRNN based on a layer specification.

        The shape of the LSTM state depends on the type and dimensions of the base layer
        (Dense or Conv) it is associated with. This allows the LSTM to maintain separate states
        for different parts of the base layer's parameters (e.g., each connection in a dense layer,
        or each filter element in a conv layer).

        Args:
            layer_spec: A DenseSpec or ConvSpec object describing the base layer.

        Returns:
            An initial hk.LSTMState, with cell states partially randomized according to
            `_init_rand_proportion`.
        """
        if isinstance(layer_spec, DenseSpec):
            # For a dense layer, LSTM state shape is (in_size, out_size, hidden_size)
            shape = (layer_spec.in_size, layer_spec.out_size)
        elif isinstance(layer_spec, ConvSpec):
            # For a conv layer, LSTM state shape is (kernel_h, kernel_w, in_channels, out_channels, hidden_size)
            shape = (layer_spec.kernel_size,
                     layer_spec.kernel_size,
                     layer_spec.in_channels,
                     layer_spec.out_channels)
        # Calls CustomLSTM's method to get an initial state, which handles the random proportion.
        return self._lstm.initial_vsml_state(shape, self._init_rand_proportion)


@configurable('model.vsml_rnn')
class VSMLRNN(hk.Module):

    def __init__(self, layer_specs: List[LayerSpec], num_micro_ticks: int,
                 loss_func: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
                 tanh_bound: float, output_idx: int, backward_pass: bool,
                 separate_backward_rnn: bool, feed_label: bool, layerwise_rnns: bool):
        """Initializes the Variable Shared Meta Learning RNN (VSMLRNN) model.

        This model implements the architecture described in "Meta Learning Backpropagation And Improving It".
        It uses sub-RNNs (LSTMs) to modulate the behavior of a base network defined by layer_specs.

        Args:
            layer_specs: A list of specifications (DenseSpec or ConvSpec) defining the
                architecture of the base network.
            num_micro_ticks: The number of times the sub-RNNs update and messages are
                exchanged within each layer for a single input step of the base network.
                Corresponds to the "two ticks per input" mentioned in the paper's appendix.
            loss_func: The loss function used to compute gradients for the backward pass
                and to guide the learning of the meta-learner.
            tanh_bound: A float value to bound the output activations using tanh.
                If 0 or None, no bounding is applied.
            output_idx: Index to select the output from the final layer's forward message.
            backward_pass: If True, enables the backward pass where error signals are
                propagated back through the layers using (potentially separate) sub-RNNs.
            separate_backward_rnn: If True and backward_pass is True, uses a separate
                set of sub-RNNs for the backward pass. Otherwise, the same sub-RNNs are used.
            feed_label: If True, the true label is concatenated with the error signal
                during the backward pass.
            layerwise_rnns: If True, each layer in layer_specs gets its own instance of SubRNN.
                Otherwise, SubRNN parameters are shared across all layers.
        """
        super().__init__()
        self._layer_specs = layer_specs
        self._num_micro_ticks = num_micro_ticks
        self._tanh_bound = tanh_bound
        if layerwise_rnns:
            self._sub_rnns = [SubRNN() for _ in layer_specs]
        else:
            self._sub_rnns = [SubRNN()] * len(layer_specs)
        self._loss_func = loss_func
        self._loss_func_grad = jax.grad(loss_func)
        self._backward_pass = backward_pass
        self._feed_label = feed_label
        self._batched_tick = hk.vmap(
            functools.partial(self._tick, self._sub_rnns, reverse=False),
            in_axes=(0, 0, None),
            out_axes=(0, 0),
            split_rng=False)
        if backward_pass:
            if separate_backward_rnn:
                if layerwise_rnns:
                    self._back_sub_rnns = [SubRNN() for _ in layer_specs]
                else:
                    self._back_sub_rnns = [SubRNN()] * len(layer_specs)
            else:
                self._back_sub_rnns = self._sub_rnns
            self._reverse_batched_tick = hk.vmap(
                functools.partial(self._tick, self._back_sub_rnns, reverse=True),
                in_axes=(0, 0, None),
                out_axes=(0, 0),
                split_rng=False)
        self._output_idx = output_idx

    def _tick(self, sub_rnns, layer_states: List[LayerState], error: jnp.ndarray,
              inp: jnp.ndarray, reverse=False) -> Tuple[List[LayerState], jnp.ndarray]:
        """Performs a single "tick" of computation through all layers of the VSMLRNN.

        This involves:
        1. Initializing forward messages from the input (or backward messages from error in reverse mode).
        2. Iterating through each layer (either forward or reversed order).
        3. For each layer, running its associated sub-RNN for `_num_micro_ticks` to update LSTM states
           and generate new forward and backward messages using `vsml_layers.dense` or `vsml_layers.conv2d`.
        4. Propagating these messages to adjacent layers.
        5. Producing an output from the final layer's forward message (in forward mode).

        This method corresponds to the inner workings of the loop over layers (k) within
        Algorithm 2 (VSML: Meta Testing) from the paper, for a single time step.

        Args:
            sub_rnns: A list of SubRNN instances to be used for this tick (can be forward or backward RNNs).
            layer_states: A list of LayerState objects, holding LSTM states and incoming messages for each layer.
            error: The error signal (gradient of loss w.r.t. output) from the previous time step or target.
                   Used to initialize the backward message for the last layer in a forward pass,
                   or the primary input for a reverse pass.
            inp: The input data (e.g., image) for the current time step.
                 Used to initialize the forward message for the first layer.
            reverse: If True, messages are propagated backward from the last layer to the first.
                     Typically used during the error propagation phase if `_backward_pass` is enabled.

        Returns:
            A tuple containing:
                - updated_layer_states: The list of LayerState objects with their LSTM states modified.
                - output: The output produced by the network from the final layer's forward message.
                          (Meaningful primarily when reverse=False).
        """
        if isinstance(self._layer_specs[0], DenseSpec):
            inp = inp.flatten() # Flatten input if the first layer is dense
        sub_rnn = sub_rnns[0] # Get the SubRNN instance (might be shared or layer-specific)
        # Initialize forward message for the first layer from the input `inp`.
        # This corresponds to m_a1^(1) := x_a in Algorithm 2.
        fwd_msg = jnp.pad(inp[..., None], (*[(0, 0)] * inp.ndim, (0, sub_rnn.msg_size - 1)))
        # Initialize backward message for the last layer from the `error` signal.
        # This corresponds to m^_b1^(K) := e_b in Algorithm 2 (conceptually, for the start of bwd pass).
        bwd_msg = jnp.pad(error, ((0, 0), (0, sub_rnn.msg_size - 2)))
        layer_states[0].incoming_fwd_msg = fwd_msg
        layer_states[-1].incoming_bwd_msg = bwd_msg
        output = None

        iterable = list(enumerate(zip(layer_states, self._layer_specs, sub_rnns)))
        if reverse:
            iterable = list(reversed(iterable)) # Iterate backward if in reverse mode
        # Loop over layers (indexed by k in Algorithm 2)
        for i, (ls, lspec, srnn) in iterable:
            lstm_state, fwd_msg, bwd_msg = (ls.lstm_state,
                                            ls.incoming_fwd_msg,
                                            ls.incoming_bwd_msg)
            # Inner loop for micro-ticks (see Section C.2 of the paper)
            for _ in range(self._num_micro_ticks):
                # Arguments for the vsml_layers.dense/conv2d calls, which internally use the sub-RNN (srnn).
                args = (srnn, jnp.mean, fwd_msg, bwd_msg, lstm_state)
                if isinstance(lspec, DenseSpec):
                    # Perform dense layer operation modulated by the sub-RNN.
                    out = vsml_layers.dense(*args)
                elif isinstance(lspec, ConvSpec):
                    # Perform convolutional layer operation modulated by the sub-RNN.
                    out = vsml_layers.conv2d(*args, stride=lspec.stride)
                else:
                    raise ValueError(f'Invalid layer {lspec}')
                # `out` contains new_fwd_msg, new_bwd_msg, updated_lstm_state
                new_fwd_msg, new_bwd_msg, lstm_state = out
            ls.lstm_state = lstm_state # Store the updated LSTM state for this layer
            # Propagate backward messages to the previous layer (i-1)
            # Corresponds to m^_a^(k-1) := sum_b' f_m(s_ab'^(k))
            if i > 0:
                shape = layer_states[i - 1].incoming_bwd_msg.shape
                layer_states[i - 1].incoming_bwd_msg = new_bwd_msg.reshape(shape)
            # Propagate forward messages to the next layer (i+1)
            # Corresponds to m_b^(k+1) := sum_a' f_m(s_a'b^(k))
            if i < len(layer_states) - 1:
                shape = layer_states[i + 1].incoming_fwd_msg.shape
                layer_states[i + 1].incoming_fwd_msg = new_fwd_msg.reshape(shape)
            else:
                # If this is the last layer in the forward pass, extract the output.
                # Corresponds to y^_a := m_a1^(K+1)
                output = new_fwd_msg[:, self._output_idx]
                if self._tanh_bound:
                    output = jnp.tanh(output / self._tanh_bound) * self._tanh_bound

        return layer_states, output

    def _create_layer_state(self, spec: LayerSpec) -> LayerState:
        """Initializes the state for a single layer in the VSMLRNN.

        This includes initializing the LSTM state via the sub-RNN's `initial_state` method
        and preparing zero-initialized tensors for incoming forward and backward messages.
        Corresponds to the initialization `V_L = {s_ab^(k)} <- initialize LSTM states` in Algorithm 2.

        Args:
            spec: The LayerSpec (DenseSpec or ConvSpec) for which to create the state.

        Returns:
            A LayerState object containing the initial LSTM state and message buffers.
        """
        sub_rnn = self._sub_rnns[0] # Assumes SubRNNs share params or uses the first one as representative for msg_size
        lstm_state = sub_rnn.initial_state(spec) # Initialize LSTM state based on layer spec
        msg_size = sub_rnn.msg_size
        new_msg = functools.partial(jnp.zeros, dtype=lstm_state.hidden.dtype)

        # Initialize message buffers based on layer type (Dense or Conv)
        if isinstance(spec, DenseSpec):
            incoming_fwd_msg = new_msg((spec.in_size, msg_size))
            incoming_bwd_msg = new_msg((spec.out_size, msg_size))
        elif isinstance(spec, ConvSpec):
            incoming_fwd_msg = new_msg((spec.in_height, spec.in_width,
                                        spec.in_channels, msg_size))
            incoming_bwd_msg = new_msg((spec.out_height, spec.out_width,
                                        spec.out_channels, msg_size))

        return LayerState(lstm_state=lstm_state,
                          incoming_fwd_msg=incoming_fwd_msg,
                          incoming_bwd_msg=incoming_bwd_msg)

    def _merge_layer_states(self, layer_states: List[LayerState]) -> List[LayerState]:
        def merge(state):
            s1, s2 = jnp.split(state, [state.shape[-1] // 2], axis=-1)
            merged_s1 = jnp.mean(s1, axis=0, keepdims=True)
            new_s1 = jnp.broadcast_to(merged_s1, s1.shape)
            return jnp.concatenate((new_s1, s2), axis=-1)
        for ls in layer_states:
            ls.lstm_state = jax.tree_util.tree_map(merge, ls.lstm_state)
        return layer_states

    def __call__(self, inputs: jnp.ndarray, labels: jnp.ndarray) -> jnp.ndarray:
        """Runs the forward pass of the VSMLRNN for a sequence of inputs and labels.

        This method implements the main loop of Algorithm 2 (VSML: Meta Testing) from the paper.
        It processes a sequence of inputs (e.g., a trajectory of images and labels from a few-shot task)
        by repeatedly calling `scan_tick` for each step in the sequence.

        Args:
            inputs: A JAX ndarray of shape (time, batch, ...), representing the input sequence.
                    The `...` part depends on the input type (e.g., H, W, C for images).
            labels: A JAX ndarray of shape (time, batch, num_classes), representing the target labels
                    for each input in the sequence.

        Returns:
            A JAX ndarray of shape (time, batch, num_classes), representing the predictions (logits)
            of the network for each input in the sequence.
        """
        # Initialize layer states for all layers. This is V_L in Algorithm 2.
        layer_states = [self._create_layer_state(spec) for spec in self._layer_specs]
        # Stack layer states for each item in the batch dimension of the input sequence.
        layer_states = jax.tree_util.tree_map(lambda ls: jnp.stack([ls] * inputs.shape[1]),
                                    layer_states)
        # Initialize error signal (e_b) for the first step. Bwd message for the last layer.
        init_error = layer_states[-1].incoming_bwd_msg[..., :2] # Typically (error_grad, label_or_zero)

        # `scan_tick` processes one time step (x, y) from the input sequence.
        # It corresponds to the operations inside the main "for (x,y) in D" loop of Algorithm 2.
        def scan_tick(carry, x):
            layer_states, error = carry # Unpack carry: current layer states and error from previous step
            inp, label = x # Unpack current input and label

            # Optional: Merge states if batch_size > 1 (not detailed in paper, specific to this impl.)
            if inp.shape[0] > 1:
                layer_states = self._merge_layer_states(layer_states)

            # Forward pass through all layers for current input `inp` and previous `error`.
            # This calls `self._tick` which handles the layer-by-layer processing and micro-ticks.
            # Corresponds to lines from "m_a1^(1) := x_a" up to "y^_a := m_a1^(K+1)"
            new_layer_states, out = self._batched_tick(layer_states, error, inp)

            # Compute error gradient for the current output `out` and `label`.
            # Corresponds to "e := grad_y^ L(y^, y)"
            new_error = self._loss_func_grad(out, label)
            # Prepare the error signal for the next step or for the backward pass.
            # If `_feed_label` is true, the label is included along with the gradient.
            label_input = label if self._feed_label else jnp.zeros_like(label)
            new_error = jnp.stack([new_error, label_input], axis=-1)

            # Optional backward pass to update layer states based on the new error.
            # This part is an extension/variation not explicitly in the main Algorithm 2 loop,
            # but related to how errors can be used to further modulate LSTM states.
            if self._backward_pass:
                # Propagate `new_error` backward through layers using `_reverse_batched_tick`.
                new_layer_states, _ = self._reverse_batched_tick(new_layer_states, new_error, inp)
                # Reset error if it was consumed by the backward pass (specific to this impl.)
                new_error = jnp.zeros_like(new_error)

            # Return updated states and current output for `hk.scan`.
            return (new_layer_states, new_error), out

        # Use hk.scan to efficiently loop `scan_tick` over the time dimension of inputs and labels.
        # The carry (`layer_states`, `init_error`) is updated at each step.
        _, outputs = hk.scan(scan_tick, (layer_states, init_error),
                             (inputs, labels))
        return outputs


@configurable('model.meta_rnn')
class MetaRNN(hk.Module):

    def __init__(self, loss_func: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
                 output_size: int, num_micro_ticks: int, slow_size: int,
                 tanh_bound: float, use_conv: bool):
        super().__init__()
        if use_conv:
            self._conv = hk.Sequential([
                hk.Conv2D(64, 3, 2, padding='SAME'),
                jax.nn.relu,
                hk.Conv2D(64, 3, 2, padding='SAME'),
                jax.nn.relu,
                hk.Conv2D(64, 3, 2, padding='SAME'),
                jax.nn.tanh,
            ])
        else:
            self._conv = None
        self._num_micro_ticks = num_micro_ticks
        self._tanh_bound = tanh_bound
        self._loss_func_grad = jax.grad(loss_func)
        self._lstm = hk.LSTM(slow_size)
        self._output_proj = hk.Linear(output_size)

    def __call__(self, inputs: jnp.ndarray, labels: jnp.ndarray) -> jnp.ndarray:
        batch_size = inputs.shape[1]
        output_size = self._output_proj.output_size
        lstm_state = self._lstm.initial_state(batch_size)
        init_error = jnp.zeros((batch_size, output_size))
        init_label = jnp.zeros((batch_size, output_size))

        # TODO merge states for batch_size > 1
        def scan_tick(carry, x):
            lstm_state, error, prev_label = carry
            inp, label = x
            if self._conv is not None:
                inp = self._conv(inp)
            inp = hk.Flatten(preserve_dims=1)(inp)
            inputs = jnp.concatenate([inp, error, prev_label], axis=-1)
            for _ in range(self._num_micro_ticks):
                out, lstm_state = self._lstm(inputs, lstm_state)
            out = self._output_proj(out)
            if self._tanh_bound:
                out = jnp.tanh(out / self._tanh_bound) * self._tanh_bound
            new_error = self._loss_func_grad(out, label)
            return (lstm_state, new_error, label), out
        _, outputs = hk.scan(scan_tick, (lstm_state, init_error, init_label),
                             (inputs, labels))
        return outputs


@configurable('model.sgd')
class SGD(hk.Module):

    def __init__(self, loss_func: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
                 output_size: int, num_layers: int, hidden_size: int,
                 tanh_bound: float, optimizer: str, lr: float, use_conv: bool):
        super().__init__()
        if use_conv:
            self._conv = hk.Sequential([
                hk.Conv2D(8, 3, 2, padding='SAME'),
                jax.nn.tanh,
                hk.Conv2D(8, 3, 2, padding='SAME'),
                jax.nn.tanh,
                hk.Conv2D(8, 3, 2, padding='SAME'),
                jax.nn.tanh,
            ])
        else:
            self._conv = None
        self._tanh_bound = tanh_bound
        self._loss_func = loss_func
        self._grad_func = jax.grad(self._loss, has_aux=True)
        self._network = functools.partial(self._network,
                                          output_size=output_size,
                                          num_layers=num_layers,
                                          hidden_size=hidden_size)
        self._network = hk.without_apply_rng(hk.transform(self._network))
        self._opt = getattr(optax, optimizer)(lr)

    def _network(self, x: jnp.ndarray, output_size, num_layers, hidden_size):
        # Temporarily simplify to isolate the error
        # if self._conv is not None:
        #     x = self._conv(x)
        x = hk.Flatten(preserve_dims=1)(x)
        # for _ in range(num_layers - 1):
        #     x = hk.Linear(hidden_size)(x)
        #     x = jnp.tanh(x)
        x = hk.Linear(output_size)(x) # output_size here is now correctly a static Python int
        # if self._tanh_bound:
        #     x = jnp.tanh(x / self._tanh_bound) * self._tanh_bound
        return x

    def _loss(self, params, x, labels):
        logits = self._network.apply(params, x)
        loss = self._loss_func(logits, labels)
        return loss, logits

    def __call__(self, inputs: jnp.ndarray, labels: jnp.ndarray) -> jnp.ndarray:
        # TODO seed from outside
        rng = jax.random.PRNGKey(22)

        dummy_inp = inputs[0]
        params = self._network.init(rng, dummy_inp)
        opt_state = self._opt.init(params)

        def scan_tick(carry, x):
            params, opt_state = carry
            grads, out = self._grad_func(params, *x)
            updates, opt_state = self._opt.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            return (params, opt_state), out
        _, outputs = jax.lax.scan(scan_tick, (params, opt_state), (inputs, labels))
        return outputs


class HebbianLinear(hk.Module):

    def __init__(self, output_size: int, use_oja: bool = False,
                 with_bias: bool = True,
                 w_init: Optional[hk.initializers.Initializer] = None,
                 b_init: Optional[hk.initializers.Initializer] = None,
                 name: Optional[str] = None,):
        super().__init__(name=name)
        self.input_size = None
        self.output_size = output_size
        self.with_bias = with_bias
        self.w_init = w_init
        self.b_init = b_init or jnp.zeros
        if use_oja:
            self._fw_update = self._oja
        else:
            self._fw_update = self._hebb
        # Dim 1
        self._fw_update = jax.vmap(self._fw_update, in_axes=[0, None, None, 0])
        # Dim 2
        self._fw_update = jax.vmap(self._fw_update, in_axes=[0, None, 0, None])
        # Batch axis
        self._fw_update = jax.vmap(self._fw_update, in_axes=[None, None, 0, 0])

    def __call__(self, inputs: jnp.ndarray,
                 fast_weights: Optional[jnp.ndarray]) -> jnp.ndarray:
        if not inputs.shape:
            raise ValueError("Input must not be scalar.")

        input_size = self.input_size = inputs.shape[-1]
        output_size = self.output_size
        dtype = inputs.dtype

        w_init = self.w_init
        if w_init is None:
            stddev = 1. / np.sqrt(self.input_size)
            w_init = hk.initializers.TruncatedNormal(stddev=stddev)
        w = hk.get_parameter("w", [input_size, output_size], dtype, init=w_init)
        coeff = hk.get_parameter('coeff', [input_size, output_size], dtype,
                                 init=hk.initializers.Constant(0.01))
        fw_lr = hk.get_parameter('fw_lr', [], dtype,
                                 init=hk.initializers.Constant(-4.5))
        fw_lr = jax.nn.sigmoid(fw_lr)

        if fast_weights is None:
            fast_weights = jnp.zeros_like(w)
        out = jnp.dot(inputs, w + coeff * fast_weights)

        if self.with_bias:
            b = hk.get_parameter("b", [self.output_size], dtype, init=self.b_init)
            b = jnp.broadcast_to(b, out.shape)
            out = out + b

        # Generate new fast weights
        # TODO make softmax optional
        new_fast_weights = self._fw_update(fast_weights, fw_lr, inputs,
                                           jax.nn.softmax(out))
        # Reduce batch axis
        new_fast_weights = jnp.mean(new_fast_weights, axis=0)

        return out, new_fast_weights

    def _hebb(self, fw, fw_lr, x, y) -> jnp.ndarray:
        return (1 - fw_lr) * fw + fw_lr * x * y

    def _oja(self, fw, fw_lr, x, y) -> jnp.ndarray:
        return fw + fw_lr * y * (x - y * fw)


@configurable('model.hebbian_fw')
class HebbianFW(hk.Module):

    def __init__(self, loss_func: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
                 input_shape: int, output_size: int, use_oja: bool, tanh_bound: float,
                 use_conv: bool):
        super().__init__()
        if use_conv:
            self._conv = hk.Sequential([
                hk.Conv2D(8, 3, 2, padding='SAME'),
                jax.nn.relu,
                hk.Conv2D(8, 3, 2, padding='SAME'),
                jax.nn.relu,
                hk.Conv2D(8, 3, 2, padding='SAME'),
                jax.nn.tanh,
            ])
        else:
            self._conv = None
        self._layers = [
            HebbianLinear(output_size, use_oja)
        ]
        self.output_size = output_size
        self._tanh_bound = tanh_bound
        self._loss_func = loss_func
        self._loss_func_grad = jax.grad(loss_func)
        # Create parameters
        aux = [jnp.zeros([1, output_size])] * 2
        self._eval_layers(jnp.zeros([1, *input_shape]), itertools.repeat(None), aux)

    def _eval_layers(self, inputs: jnp.ndarray, fast_weights: Iterable[jnp.ndarray], aux):
        x = inputs
        if self._conv is not None:
            x = self._conv(x)
        x = hk.Flatten(preserve_dims=1)(x)
        x = jnp.concatenate([x, *aux], axis=-1)
        fws_out = []
        for layer, fws in zip(self._layers, fast_weights):
            x, new_fws = layer(x, fws)
            fws_out.append(new_fws)
        return x, fws_out

    def __call__(self, inputs: jnp.ndarray, labels: jnp.ndarray) -> jnp.ndarray:
        batch_size = inputs.shape[1]
        hebb_state = [jnp.zeros([layer.input_size, layer.output_size])
                      for layer in self._layers]
        init_error = jnp.zeros((batch_size, self.output_size))

        # TODO merge states for batch_size > 1
        def scan_tick(carry, x):
            hebb_state, error = carry
            inp, label = x

            aux = [jnp.zeros_like(error), jnp.zeros_like(label)]
            out, _ = self._eval_layers(inp, hebb_state, aux)
            if self._tanh_bound:
                out = jnp.tanh(out / self._tanh_bound) * self._tanh_bound
            new_error = self._loss_func_grad(out, label)

            aux = [new_error, label]
            _, hebb_state = self._eval_layers(inp, hebb_state, aux)

            return (hebb_state, new_error), out
        _, outputs = hk.scan(scan_tick, (hebb_state, init_error),
                             (inputs, labels))
        return outputs


@configurable('model.fwp')
class FWP(hk.Module):

    def __init__(self, loss_func: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
                 output_size: int, fast_size: int, tanh_bound: float):
        super().__init__()
        self._tanh_bound = tanh_bound
        self._fast_size = fast_size
        self._output_size = output_size
        self._fast_shape = (fast_size, output_size)

        self._loss_func_grad = jax.grad(loss_func)
        size = 2 * fast_size + output_size + 1
        self._slow_net = hk.Linear(size)

    def __call__(self, inputs: jnp.ndarray, labels: jnp.ndarray) -> jnp.ndarray:
        batch_size = inputs.shape[1]
        fast_state = hk.initializers.VarianceScaling()(self._fast_shape, jnp.float32)
        init_carry = jnp.zeros((3, batch_size, self._output_size))

        def scan_tick(carry, x):
            fast_state, error, prev_label, prev_out = carry
            inp, label = x
            inp = hk.Flatten(preserve_dims=1)(inp)
            inputs = jnp.concatenate([inp, prev_out, error, prev_label], axis=-1)
            split_indices = np.cumsum([self._fast_size, self._output_size,
                                       self._fast_size])
            k, v, q, beta = jnp.split(self._slow_net(inputs), split_indices, axis=-1)
            beta = jax.nn.sigmoid(beta)
            prev_v = k @ fast_state
            fast_state = fast_state + k.T @ (beta * (v - prev_v))
            out = q @ fast_state
            if self._tanh_bound:
                out = jnp.tanh(out / self._tanh_bound) * self._tanh_bound
            new_error = self._loss_func_grad(out, label)
            return (fast_state, new_error, label, out), out

        _, outputs = hk.scan(scan_tick, (fast_state, *init_carry), (inputs, labels))
        return outputs


@configurable('model.fw_memory')
class FWMemory(hk.Module):

    def __init__(self, loss_func: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
                 output_size: int, slow_size: int, tanh_bound: float, memory_size: int, use_conv: bool):
        super().__init__()
        if use_conv:
            self._conv = hk.Sequential([
                hk.Conv2D(8, 3, 2, padding='SAME'),
                jax.nn.relu,
                hk.Conv2D(8, 3, 2, padding='SAME'),
                jax.nn.relu,
                hk.Conv2D(8, 3, 2, padding='SAME'),
                jax.nn.tanh,
            ])
        else:
            self._conv = None
        self._tanh_bound = tanh_bound
        self._memory_size = memory_size
        self._loss_func_grad = jax.grad(loss_func)
        self._lstm = hk.LSTM(slow_size)
        self._output_proj = hk.Linear(output_size)
        self._write_head = hk.Linear(3 * memory_size + 1)
        self._read_head = hk.Linear(2 * memory_size)
        self._read_proj = hk.Linear(slow_size)
        self._layer_norm = hk.LayerNorm(-1, False, False)

    def __call__(self, inputs: jnp.ndarray, labels: jnp.ndarray) -> jnp.ndarray:
        batch_size = inputs.shape[1]
        output_size = self._output_proj.output_size
        lstm_state = self._lstm.initial_state(batch_size)
        init_error = jnp.zeros((batch_size, output_size))
        init_label = jnp.zeros((batch_size, output_size))
        init_memory = jnp.zeros((self._memory_size, self._memory_size ** 2))

        # TODO merge states for batch_size > 1
        def scan_tick(carry, x):
            lstm_state, memory, error, prev_label = carry
            inp, label = x
            if self._conv is not None:
                inp = self._conv(inp)
            inp = hk.Flatten(preserve_dims=1)(inp)
            inputs = jnp.concatenate([inp, error, prev_label], axis=-1)

            out, lstm_state = self._lstm(inputs, lstm_state)
            write = self._write_head(out)
            beta = jax.nn.sigmoid(write[:, -1])
            k1, k2, v = jnp.split(jax.nn.tanh(write[:, :-1]), 3, axis=-1)
            # TODO this flatten doesn't work with batch dim
            key = jnp.outer(k1, k2).flatten()
            v_old = memory @ key
            memory = memory + beta * jnp.outer((v - v_old), key)
            memory = memory / jnp.maximum(1, jnp.linalg.norm(memory))
            n, e = jnp.split(jax.nn.tanh(self._read_head(out)), 2, axis=-1)
            # TODO optionally add multiple readouts
            n = self._layer_norm(memory @ jnp.outer(n, e).flatten())
            readout = self._read_proj(n)
            out = out + readout

            out = self._output_proj(out)
            if self._tanh_bound:
                out = jnp.tanh(out / self._tanh_bound) * self._tanh_bound
            new_error = self._loss_func_grad(out, label)
            return (lstm_state, memory, new_error, label), out
        _, outputs = hk.scan(scan_tick, (lstm_state, init_memory, init_error, init_label),
                             (inputs, labels))
        return outputs
