"""
Config-dispatched task batch sampling.

The pool and train loop interact with task sampling exclusively through
``sample_task_batch(key, pool_size, input_n, output_n, cfg)``. The ``cfg``
dict (typically a Hydra-resolved subtree) carries a ``name`` field naming
the sampler; remaining fields are sampler-specific kwargs. To add a new
task family:

    1. Implement a single-circuit sampler in ``samplers.py`` returning
       ``[case_n, output_n]``.
    2. Add a vmapped batch wrapper here and register it under a name in
       ``TASK_SAMPLERS``.

Everything downstream stays unchanged.
"""

from collections.abc import Mapping
from functools import partial
from typing import Any

import jax
import jax.numpy as jp

from boolean_nca_cc.tasks.samplers import (
    sample_arith_family_y,
    sample_k_junta_y,
    sample_library_batch,
)


@partial(jax.jit, static_argnames=("pool_size", "input_n", "output_n", "k", "balanced"))
def _sample_k_junta_batch(
    key: jax.Array,
    pool_size: int,
    input_n: int,
    output_n: int,
    k: int,
    balanced: bool,
) -> jp.ndarray:
    keys = jax.random.split(key, pool_size)
    return jax.vmap(
        lambda k_: sample_k_junta_y(k_, input_n, output_n, k=k, balanced=balanced)
    )(keys)


def _k_junta_from_cfg(
    key: jax.Array,
    pool_size: int,
    input_n: int,
    output_n: int,
    cfg: Mapping[str, Any],
) -> jp.ndarray:
    k = int(cfg.get("k", 4))
    balanced = bool(cfg.get("balanced", True))
    return _sample_k_junta_batch(key, pool_size, input_n, output_n, k, balanced)


@partial(
    jax.jit,
    static_argnames=("pool_size", "input_n", "output_n", "op", "permute_inputs", "max_offset"),
)
def _sample_arith_family_batch(
    key: jax.Array,
    pool_size: int,
    input_n: int,
    output_n: int,
    op: str,
    permute_inputs: bool,
    max_offset: int,
) -> jp.ndarray:
    keys = jax.random.split(key, pool_size)
    return jax.vmap(
        lambda k_: sample_arith_family_y(
            k_, input_n, output_n, op=op, permute_inputs=permute_inputs, max_offset=max_offset
        )
    )(keys)


def _arith_family_from_cfg(
    key: jax.Array,
    pool_size: int,
    input_n: int,
    output_n: int,
    cfg: Mapping[str, Any],
) -> jp.ndarray:
    op = str(cfg.get("op", "add"))
    permute_inputs = bool(cfg.get("permute_inputs", True))
    max_offset = int(cfg.get("max_offset", 0))
    return _sample_arith_family_batch(
        key, pool_size, input_n, output_n, op, permute_inputs, max_offset
    )


def _library_from_cfg(
    key: jax.Array,
    pool_size: int,
    input_n: int,
    output_n: int,
    cfg: Mapping[str, Any],
) -> jp.ndarray:
    task_names = cfg.get("task_names")
    if not task_names:
        raise ValueError("library sampler requires non-empty 'task_names' in cfg")
    y_task, _idx = sample_library_batch(
        key, pool_size, input_n, output_n, tuple(task_names)
    )
    return y_task


# Registry: name → batch sampler. Adding a new task family is one line.
TASK_SAMPLERS = {
    "k_junta": _k_junta_from_cfg,
    "arith_family": _arith_family_from_cfg,
    "library": _library_from_cfg,
}


def sample_task_batch(
    key: jax.Array,
    pool_size: int,
    input_n: int,
    output_n: int,
    cfg: Mapping[str, Any],
) -> jp.ndarray:
    """
    Dispatch to a registered task batch sampler.

    Args:
        key: PRNG key.
        pool_size: Number of pool slots.
        input_n: Circuit input width.
        output_n: Circuit output width.
        cfg: Mapping with a ``name`` key (selecting the sampler) plus
            sampler-specific kwargs.

    Returns:
        y_task of shape [pool_size, 2^input_n, output_n], float32 in {0.0, 1.0}.
    """
    if "name" not in cfg:
        raise ValueError(f"task sampler cfg missing 'name': got {dict(cfg)!r}")
    name = cfg["name"]
    if name not in TASK_SAMPLERS:
        raise ValueError(
            f"unknown task sampler {name!r}; registered: {sorted(TASK_SAMPLERS)}"
        )
    return TASK_SAMPLERS[name](key, pool_size, input_n, output_n, cfg)
