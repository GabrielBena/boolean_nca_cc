import jax
import jax.numpy as jnp
import haiku as hk
import numpy as np
from typing import Optional, Tuple, Any, Sequence
from haiku import LSTMState


class CustomLSTM(hk.RNNCore):

    def __init__(self, hidden_size: int, name: Optional[str] = None):
        """Initializes a custom LSTM core.

        This LSTM implementation is largely standard but includes a specific initialization
        scheme (`initial_vsml_state`) tailored for the VSML model, allowing partial
        random initialization of the cell state.

        Args:
            hidden_size: The number of hidden units in the LSTM.
            name: Optional name for the Haiku module.
        """
        super().__init__(name=name)
        self.hidden_size = hidden_size

    def _initializer(self, shape: Sequence[int], dtype: Any) -> jnp.ndarray:
        input_size = shape[0]
        stddev = 1. / np.sqrt(input_size)
        return hk.initializers.TruncatedNormal(stddev=stddev)(shape, dtype)

    def _lstm_initializer(self, shape: Sequence[int], dtype: Any) -> jnp.ndarray:
        return self._initializer(shape, dtype)

    def __call__(self, inputs: jnp.ndarray,
                 prev_state: LSTMState) -> Tuple[jnp.ndarray, LSTMState]:
        """Performs a standard LSTM computation step.

        Args:
            inputs: The input to the LSTM at the current time step.
            prev_state: The LSTMState (hidden and cell) from the previous time step.

        Returns:
            A tuple containing:
                - h: The new hidden state.
                - LSTMState(h, c): The new LSTMState (hidden and cell).
        """
        if len(inputs.shape) > 2 or not inputs.shape:
            raise ValueError("LSTM input must be rank-1 or rank-2.")
        # Concatenate input and previous hidden state for efficient linear projection
        x_and_h = jnp.concatenate([inputs, prev_state.hidden], axis=-1)
        # Linear projection to compute gates and cell input
        gated = hk.Linear(4 * self.hidden_size,
                          w_init=self._lstm_initializer)(x_and_h)
        # i = input gate, g = cell gate (candidate cell state), f = forget gate, o = output gate
        i, g, f, o = jnp.split(gated, indices_or_sections=4, axis=-1)

        # Apply sigmoid activations to gates, with biases mentioned in paper appendix C.2
        # "gate biases are initialized to +5 for the forget gate and −5 for the input gate"
        f = jax.nn.sigmoid(f + 5)
        i = jax.nn.sigmoid(i - 5)

        # Update cell state: c_t = f_t * c_{t-1} + i_t * tanh(g_t)
        c = f * prev_state.cell + i * jnp.tanh(g)
        # Compute new hidden state: h_t = o_t * tanh(c_t)
        h = jax.nn.sigmoid(o) * jnp.tanh(c)
        return h, LSTMState(h, c)

    def initial_state(self, batch_size: Optional[int]) -> LSTMState:
        raise NotImplementedError()

    def initial_vsml_state(self, shape: Sequence[int], rand_proportion: float) -> LSTMState:
        """Initializes the LSTM state specifically for the VSML model.

        The hidden state is initialized to zeros.
        The cell state is initialized such that a `rand_proportion` of its trailing part
        (along the last dimension, which is `hidden_size`) is drawn from a TruncatedNormal
        distribution, while the leading part is zeros. This matches the description in
        Appendix C.2 of the VSML paper: "States are initialized randomly from independent
        standard normals... We found it helpful to only initialize a fraction (e.g. 25%) of the
        LSTM cell states randomly and set the rest to 0."

        Args:
            shape: The base shape for the LSTM state, typically reflecting the dimensions
                   of the parameters it modulates (e.g., (in_size, out_size) for a dense layer,
                   or (kernel_h, kernel_w, in_c, out_c) for a conv layer).
                   The `hidden_size` will be appended to this shape.
            rand_proportion: The fraction (0.0 to 1.0) of the cell state's last dimension
                             to be initialized randomly. The rest will be zeros.

        Returns:
            An LSTMState object with initialized hidden and cell states.
        """
        # Final shape for hidden and cell states includes hidden_size as the last dimension
        shape = tuple(shape) + (self.hidden_size,)
        hidden = jnp.zeros(shape) # Hidden state is all zeros

        # Determine shapes for randomly initialized part and zero-initialized part of the cell state
        rand_elements_count = int(shape[-1] * rand_proportion)
        rand_shape = shape[:-1] + (rand_elements_count,)
        zero_shape = shape[:-1] + (shape[-1] - rand_elements_count,)

        # Concatenate the randomly initialized part and the zero part to form the cell state
        cell = jnp.concatenate([self._initializer(rand_shape, jnp.float32), # Random part
                                jnp.zeros(zero_shape)], axis=-1)          # Zero part
        return LSTMState(hidden=hidden, cell=cell)
