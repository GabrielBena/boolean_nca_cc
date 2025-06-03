import functools
import jax
import jax.numpy as jnp
import haiku as hk
import jaxutil


def dense(sub_rnn, reduce_fn, fwd_msg, bwd_msg, state):
    """Applies a VSML-modulated dense (fully connected) layer operation.

    This function takes incoming forward and backward messages and the current LSTM states
    (which implicitly define the "weights" of the dense layer) and computes updated
    messages and states.

    The `sub_rnn` is applied to each pair of input feature message (`fwd_msg`) and
    output feature message (`bwd_msg`) along with the corresponding LSTM `state` element.
    This is achieved by vmapping the `sub_rnn` across the input and output dimensions.
    The resulting messages are then reduced (e.g., averaged) to form the final
    outgoing forward and backward messages for the entire layer.

    Args:
        sub_rnn: The SubRNN instance (hk.Module) that performs the core LSTM update
                 and message generation (f_RNN and f_m from the paper).
        reduce_fn: A function (e.g., `jnp.mean`) used to aggregate messages after
                   the `sub_rnn` has been applied. For example, to combine the contributions
                   from all input features to form the output feature message.
        fwd_msg: Forward messages from the previous layer or input.
                 Shape: `[in_features, msg_size]`.
        bwd_msg: Backward messages from the subsequent layer or error signal.
                 Shape: `[out_features, msg_size]`.
        state: The LSTM states associated with this dense layer.
               Shape: `[in_features, out_features, slow_rnn_hidden_size]`.

    Returns:
        A tuple containing:
            - new_fwd_msg: Updated forward messages for the next layer.
                           Shape: `[out_features, msg_size]`.
            - new_bwd_msg: Updated backward messages for the previous layer.
                           Shape: `[in_features, msg_size]`.
            - new_state: Updated LSTM states for this layer.
                         Shape: `[in_features, out_features, slow_rnn_hidden_size]`.
    """
    # Vmap the sub_rnn over out_features (axis 0 of bwd_msg and axis 1 of state)
    batched_over_out = hk.vmap(sub_rnn, in_axes=(None, 0, 0), split_rng=False) # Pass fwd_msg as is, map over bwd_msg and corresponding state slice
    # Vmap the already batched sub_rnn over in_features (axis 0 of fwd_msg and axis 0 of state)
    batched_over_in_and_out = hk.vmap(batched_over_out, in_axes=(0, None, 0), split_rng=False) # Map over fwd_msg, pass bwd_msg as is, map over state

    # Apply the doubly-vmapped sub_rnn.
    # Effectively, sub_rnn(fwd_msg_i, bwd_msg_j, state_ij) for all i, j.
    # Resulting shapes before reduction:
    #   fwd_msg_prime: [in_features, out_features, msg_size]
    #   bwd_msg_prime: [in_features, out_features, msg_size]
    #   state_prime:   [in_features, out_features, slow_rnn_hidden_size]
    fwd_msg_prime, bwd_msg_prime, state_prime = batched_over_in_and_out(fwd_msg, bwd_msg, state)

    # Reduce/aggregate messages.
    # New forward message for output feature j is a reduction over all input features i.
    # m_b^(k+1) = reduce_fn(f_m(s_a'b^(k))) over a' (inputs)
    new_fwd_msg = reduce_fn(fwd_msg_prime, axis=0) # Reduce along in_features dimension
    # New backward message for input feature i is a reduction over all output features j.
    # m^_a^(k-1) = reduce_fn(f_m(s_ab'^(k))) over b' (outputs)
    new_bwd_msg = reduce_fn(bwd_msg_prime, axis=1) # Reduce along out_features dimension

    return new_fwd_msg, new_bwd_msg, state_prime


def conv(base_func, reduce_fn, fwd_msg, bwd_msg, state, stride):
    # TODO generalize to arbitrary state pytrees
    kwidth = state[0].shape[0]
    pad_fwd_msg = jnp.pad(fwd_msg,
                          ((kwidth // 2, kwidth // 2),)
                          + ((0, 0),) * (fwd_msg.ndim - 1))
    width = fwd_msg.shape[0]
    pad_width = pad_fwd_msg.shape[0]

    # TODO This is inefficient
    # Shape [pad_width // stride, kwidth, in_channel, msg_size]
    gathered_fwd_msg = pad_fwd_msg[(jnp.arange(kwidth)[None]
                                   + jnp.arange(pad_width - kwidth + 1,
                                                step=stride)[:, None])]

    batched_kwidth = hk.vmap(base_func, in_axes=(0, None, 0))
    batched = hk.vmap(batched_kwidth, in_axes=(0, 0, None))

    fwd_msg, bwd_msg, state = batched(gathered_fwd_msg, bwd_msg, state)

    state = jax.tree_map(lambda s: reduce_fn(s, axis=0), state)

    # Reduce over kernel
    fwd_msg = reduce_fn(fwd_msg, axis=1)

    # Construct bwd_msg
    # TODO striding currently inefficient
    idx0 = jnp.arange(width)[:, None] + jnp.arange(kwidth)[None] - 1
    idx1 = jnp.broadcast_to(jnp.flip(jnp.arange(kwidth)[None, :]), (width, kwidth))
    msg = bwd_msg[(jnp.clip(idx0, 0, width - 1) // stride, idx1)]
    mask = jnp.logical_and(jnp.logical_and(idx0 >= 0, idx0 < width // stride),
                           idx0 % stride == 0).astype(jnp.int32)
    mask = jaxutil.broadcast_minor(mask, msg.shape)
    bwd_msg = reduce_fn(msg * mask, axis=1)

    return fwd_msg, bwd_msg, state


def conv1d(sub_rnn, reduce_fn, fwd_msg, bwd_msg, state, stride=1):
    """Applies a VSML-modulated 1D convolutional layer operation.

    This function is a wrapper around the more general `conv` function, specializing it for 1D convolutions.
    It uses `vsml_layers.dense` as the `base_func` for `conv`, meaning that for each
    convolutional filter position, the interaction between input channels and output channels
    is treated like a dense layer modulated by the `sub_rnn`.

    Args:
        sub_rnn: The SubRNN instance (hk.Module).
        reduce_fn: A function (e.g., `jnp.mean`) to aggregate messages.
        fwd_msg: Forward messages from the previous layer/input.
                 Shape: `[width, in_channels, msg_size]`.
        bwd_msg: Backward messages from the subsequent layer/error.
                 Shape: `[width // stride, out_channels, msg_size]`.
        state: LSTM states for this 1D conv layer.
               Shape: `[kernel_width, in_channels, out_channels, slow_rnn_hidden_size]`.
        stride: The stride of the convolution.

    Returns:
        A tuple (new_fwd_msg, new_bwd_msg, new_state) from the underlying `conv` call.
    """
    # The core operation at each spatial location of the kernel is a dense interaction
    # between input channels and output channels, modulated by the sub_rnn.
    base_func = functools.partial(dense, sub_rnn, reduce_fn)
    return conv(base_func, reduce_fn, fwd_msg, bwd_msg, state, stride)


def conv2d(sub_rnn, reduce_fn, fwd_msg, bwd_msg, state, stride=1):
    """Applies a VSML-modulated 2D convolutional layer operation.

    This function is a wrapper around the more general `conv` function, specializing it for 2D convolutions.
    It uses `vsml_layers.conv1d` as the `base_func` for `conv`. This means that the 2D convolution
    is implemented by applying a 1D convolution (itself VSML-modulated) across one spatial dimension,
    and then this entire 1D conv operation is scanned across the other spatial dimension, also modulated
    by the VSML mechanism (via the `conv` function's handling of its `base_func`).

    Args:
        sub_rnn: The SubRNN instance (hk.Module).
        reduce_fn: A function (e.g., `jnp.mean`) to aggregate messages.
        fwd_msg: Forward messages from the previous layer/input.
                 Shape: `[height, width, in_channels, msg_size]`.
        bwd_msg: Backward messages from the subsequent layer/error.
                 Shape: `[height // stride, width // stride, out_channels, msg_size]`.
        state: LSTM states for this 2D conv layer.
               Shape: `[kernel_height, kernel_width, in_channels, out_channels, slow_rnn_hidden_size]`.
        stride: The stride of the convolution (applied to both height and width).

    Returns:
        A tuple (new_fwd_msg, new_bwd_msg, new_state) from the underlying `conv` call.
    """
    # The 2D convolution is built upon VSML-modulated 1D convolutions.
    base_func = functools.partial(conv1d, sub_rnn, reduce_fn, stride=stride)
    # The `conv` function handles the scanning of this `base_func` (the 1D conv) across
    # the other spatial dimension, along with message passing and state updates.
    return conv(base_func, reduce_fn, fwd_msg, bwd_msg, state, stride)
