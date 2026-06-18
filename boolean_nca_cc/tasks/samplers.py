"""
Single-circuit task samplers.

Each sampler is a pure function taking a PRNG key plus task-family-specific
configuration and returning ``y_task`` of shape ``[2^input_n, output_n]`` —
the full truth table of the sampled Boolean function over the deterministic
input enumeration ``x = build_task_x(input_n)``.

Storing the full truth table per circuit (rather than the task *specification*
— e.g., (subset, LUT) for a k-junta) trades GPU memory for plumbing simplicity:
the pool's pytree stays homogeneous across task families, and the train/eval
loops read ``y_task`` like any other batched array. At ``input_n=12``,
``output_n=12``, ``pool_size=1024`` this costs ~192 MB in float32, ~48 MB in
bool — small next to the pool's graph state.
"""

from collections.abc import Sequence
from functools import partial

import jax
import jax.numpy as jp

from boolean_nca_cc.circuits.tasks import get_task_data


def build_task_x(input_n: int) -> jp.ndarray:
    """
    Deterministic enumeration of all 2^input_n input patterns.

    Shared across every pool slot — the task identity lives entirely in ``y``,
    not in ``x``. This is intentional: the policy sees the same input
    distribution for every task, so any per-task variance in residuals is
    purely attributable to the target function.

    Args:
        input_n: Number of input bits.

    Returns:
        x of shape [2^input_n, input_n], float32 values in {0.0, 1.0}.
        Row i is the binary representation of integer i (LSB first), matching
        ``boolean_nca_cc.circuits.tasks.unpack``.
    """
    case_n = 1 << input_n
    bit_indices = jp.arange(input_n)
    integers = jp.arange(case_n)
    return ((integers[:, None] >> bit_indices[None, :]) & 1).astype(jp.float32)


@partial(jax.jit, static_argnames=("input_n", "output_n", "k", "balanced"))
def sample_k_junta_y(
    key: jax.Array,
    input_n: int,
    output_n: int,
    k: int,
    balanced: bool = True,
) -> jp.ndarray:
    """
    Sample a single circuit's y_task as a per-output-bit k-junta.

    For each output bit i independently:
      1. Draw a random k-subset S_i of {0, ..., input_n-1} (without replacement).
      2. Draw a Boolean LUT over 2^k entries.
      3. Compute ``y[:, i] = LUT[ pack_bits(x[:, S_i]) ]``.

    Output bits are independent k-juntas with their own subset and LUT — the
    policy must discover both *which* inputs matter for *each* output bit.

    Args:
        input_n: Number of input bits (static).
        output_n: Number of output bits (static).
        k: Junta width (static). Must satisfy 1 <= k <= input_n.
        balanced: If True, every LUT has exactly 2^(k-1) ones (marginal exactly
            0.5 on the uniform input distribution). If False, LUT entries are
            iid Bernoulli(0.5) — marginals concentrate near 0.5 but vary.

    Returns:
        y of shape [2^input_n, output_n], float32 in {0.0, 1.0}.
    """
    if not (1 <= k <= input_n):
        raise ValueError(f"k={k} must satisfy 1 <= k <= input_n={input_n}")

    case_n = 1 << input_n
    lut_size = 1 << k
    x = build_task_x(input_n)  # [case_n, input_n], float32

    # Per-output-bit keys: one for subset draw, one for LUT draw.
    subset_keys = jax.random.split(jax.random.fold_in(key, 0), output_n)
    lut_keys = jax.random.split(jax.random.fold_in(key, 1), output_n)

    powers = (1 << jp.arange(k)).astype(jp.int32)  # [k]

    if balanced:
        # Exactly half ones — marginal of every output bit is exactly 0.5.
        base_lut = jp.concatenate(
            [jp.zeros(lut_size // 2, dtype=jp.float32),
             jp.ones(lut_size - lut_size // 2, dtype=jp.float32)]
        )

        def make_lut(lut_key):
            return jax.random.permutation(lut_key, base_lut)
    else:
        def make_lut(lut_key):
            return jax.random.bernoulli(lut_key, 0.5, shape=(lut_size,)).astype(jp.float32)

    def per_output_bit(subset_key, lut_key):
        subset = jax.random.permutation(subset_key, input_n)[:k]  # [k]
        lut = make_lut(lut_key)                                   # [2^k]
        projected = x[:, subset].astype(jp.int32)                 # [case_n, k]
        compact_idx = jp.sum(projected * powers, axis=-1)         # [case_n]
        return lut[compact_idx]                                   # [case_n]

    y_columns = jax.vmap(per_output_bit)(subset_keys, lut_keys)   # [output_n, case_n]
    return y_columns.T                                            # [case_n, output_n]


@partial(
    jax.jit, static_argnames=("input_n", "output_n", "op", "permute_inputs", "max_offset")
)
def sample_arith_family_y(
    key: jax.Array,
    input_n: int,
    output_n: int,
    op: str = "add",
    permute_inputs: bool = True,
    max_offset: int = 0,
) -> jp.ndarray:
    """
    Sample a single circuit's y_task from the ADDITION family under a random
    input-bit permutation.

    The base function is binary addition of the two operands carved from the input
    bits exactly as ``boolean_nca_cc.circuits.tasks.binary_add``: operand A = the low
    ``half = input_n // 2`` bits, operand B = the remaining ``input_n - half`` high
    bits, ``y = A + B`` unpacked LSB-first to ``output_n`` (high bits zero).

    Diversity comes from a random permutation π of the INPUT-bit positions applied
    *before* the operand split: which physical input wire feeds which operand bit
    varies per circuit. This is a topology-agnostic carry family — every sample shares
    add's carry-chain structure but routes the operands differently, so a meta-learned
    rule must learn carry computation robust to input routing rather than memorise one
    wiring. This is the add-structured counterpart to the generic ``k_junta`` family;
    "sweep both" compares them, both eval'd OOD on canonical ``add``.

    With ``permute_inputs=False`` (π = identity) and ``max_offset=0`` this returns
    BIT-IDENTICAL output to the library ``add`` task (zero-padded to ``output_n``) for
    ``output_n >= input_n // 2 + 1`` — so OOD-``add`` is exactly the π-identity member
    of this family (a legitimate held-out generalization target).

    Args:
        input_n: Number of input bits (static).
        output_n: Number of output bits (static).
        permute_inputs: If True, draw a random permutation of the input-bit positions
            (the diversity). If False, identity → canonical ``add``.
        max_offset: If > 0, add a per-circuit random constant in ``[0, max_offset]`` to
            the sum before unpacking (extra diversity; shifts away from pure add).
            Default 0 → pure permuted add.

    Returns:
        y of shape [2^input_n, output_n], float32 in {0.0, 1.0}.
    """
    half = input_n // 2
    x_bits = build_task_x(input_n)  # [case_n, input_n], LSB-first

    perm_key, offset_key = jax.random.split(key)
    perm = (
        jax.random.permutation(perm_key, input_n)
        if permute_inputs
        else jp.arange(input_n)
    )
    x_perm = x_bits[:, perm]  # [case_n, input_n] — operands carved from permuted bits

    powers_a = (1 << jp.arange(half)).astype(jp.int32)             # [half]
    powers_b = (1 << jp.arange(input_n - half)).astype(jp.int32)   # [input_n - half]
    a = jp.sum(x_perm[:, :half].astype(jp.int32) * powers_a, axis=-1)   # [case_n]
    b = jp.sum(x_perm[:, half:].astype(jp.int32) * powers_b, axis=-1)   # [case_n]
    y_int = (a - b) if op == "sub" else (a + b)
    if max_offset > 0:
        y_int = y_int + jax.random.randint(offset_key, (), 0, max_offset + 1)

    # Reduce mod 2^native_width (native_width = input_n//2 + 1, the library's add/sub
    # output width) so π-identity matches the library task bit-for-bit (no-op for add at
    # even input_n; handles sub's borrow / offset wraparound via two's-complement low bits).
    native_width = input_n // 2 + 1
    y_int = y_int & ((1 << native_width) - 1)

    # Unpack LSB-first to output_n (matches circuits.tasks.unpack; high bits zero).
    bit_idx = jp.arange(output_n)
    return ((y_int[:, None] >> bit_idx[None, :]) & 1).astype(jp.float32)


def sample_library_batch(
    key: jax.Array,
    pool_size: int,
    input_n: int,
    output_n: int,
    task_names: Sequence[str],
) -> tuple[jp.ndarray, jp.ndarray]:
    """
    Per-circuit task sampler over a fixed library of named tasks.

    Each pool slot is assigned a uniformly random task name from ``task_names``.
    This is the canonical OOD-eval sampler: train on (e.g.) k_junta, eval on
    library tasks the policy has never seen.

    Library tasks have varying native output widths (e.g., binary_add → 7 bits
    while output_n may be 12 or 48). y is zero-padded on the right up to
    ``output_n``, mirroring the convention used by the rest of the stack
    (last-layer width >= y.shape[1]; surplus output bits are "don't care"
    targets and supervised against 0 by BCE/L4 loss).

    Not JIT'd — library task generation is a Python loop over named functions.
    Called once per eval-pool construction, so the cost is negligible.

    Args:
        key: PRNG key for per-slot task assignment.
        pool_size: Number of pool slots to sample.
        input_n: Circuit input width (used to set case_n = 2^input_n).
        output_n: Circuit output width (target y width after padding).
        task_names: Library task names. Each must be a key in
            ``boolean_nca_cc.circuits.tasks.TASKS``.

    Returns:
        Tuple of:
          - y_task: [pool_size, 2^input_n, output_n], float32 in {0.0, 1.0}.
          - task_idx: [pool_size], int32 — which library task each slot drew.
            Useful for per-task eval reporting.
    """
    if len(task_names) == 0:
        raise ValueError("task_names must be non-empty")

    case_n = 1 << input_n

    # Build the [n_tasks, case_n, output_n] tensor of truth tables once.
    # Each library task is called with its own native output width (passing
    # output_bits=output_n breaks tasks like ``parity``, which is hard-coded
    # to 1 output bit and raises otherwise). y is then zero-padded on the
    # right (or truncated) to match the circuit's output width.
    truth_tables = []
    for name in task_names:
        (_x, y_full), _split, _total = get_task_data(
            name, case_n, input_bits=input_n
        )
        y_full = jp.asarray(y_full, dtype=jp.float32)
        if y_full.ndim == 1:
            y_full = y_full[:, None]
        if y_full.shape[1] < output_n:
            pad = jp.zeros((y_full.shape[0], output_n - y_full.shape[1]), dtype=jp.float32)
            y_full = jp.concatenate([y_full, pad], axis=1)
        elif y_full.shape[1] > output_n:
            y_full = y_full[:, :output_n]
        truth_tables.append(y_full)
    truth_tables = jp.stack(truth_tables, axis=0)  # [n_tasks, case_n, output_n]

    task_idx = jax.random.randint(
        key, shape=(pool_size,), minval=0, maxval=len(task_names), dtype=jp.int32
    )
    y_task = truth_tables[task_idx]  # [pool_size, case_n, output_n]
    return y_task, task_idx
