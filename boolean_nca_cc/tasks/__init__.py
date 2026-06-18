"""
Meta-learning task distribution layer.

This package provides per-circuit task sampling for pool-based meta-learning,
where each pool slot is bound to a freshly sampled Boolean function (its
"task"). It is intentionally separate from ``boolean_nca_cc.circuits.tasks``,
which provides a fixed library of named tasks (binary_add, parity, etc.).
The two coexist:

- ``boolean_nca_cc.circuits.tasks`` — single fixed task, used by global training
  and as a held-out OOD evaluation library.
- ``boolean_nca_cc.tasks`` (this package) — distributions of random tasks,
  used to sample a fresh y_task per pool slot for meta-learning.

The public surface is intentionally small:

- ``build_task_x(input_n)`` — deterministic input enumeration, shared across
  all circuits in a pool (the task identity lives entirely in y).
- ``sample_task_batch(key, ..., cfg)`` — returns y_task of shape
  ``[pool_size, 2^input_n, output_n]`` from a config-dispatched sampler.

Adding a new task family means: (1) implement a single-circuit sampler in
``samplers.py``, (2) register it in ``registry.py``. The rest of the stack
(pool, train loop, eval) is sampler-agnostic.
"""

from boolean_nca_cc.tasks.registry import (
    TASK_SAMPLERS,
    sample_task_batch,
)
from boolean_nca_cc.tasks.samplers import (
    build_task_x,
    sample_arith_family_y,
    sample_k_junta_y,
    sample_library_batch,
)

__all__ = [
    "TASK_SAMPLERS",
    "build_task_x",
    "sample_arith_family_y",
    "sample_k_junta_y",
    "sample_library_batch",
    "sample_task_batch",
]
