"""
Lightweight profiling helpers for steady-state step timing and trace capture.

Usage:
- time_once(fn, *args, **kwargs): warmup + timed call; returns seconds
- profile_once(trace_dir, fn, *args, **kwargs): emits a TB trace for a single call
"""

import os
import time
import jax


def _block_on_scalar(x):
    """Block until the first scalar leaf is ready (forces device sync)."""
    if hasattr(x, "block_until_ready"):
        x.block_until_ready()
        return
    leaves = jax.tree_util.tree_leaves(x)
    if not leaves:
        return
    leaf = leaves[0]
    if hasattr(leaf, "block_until_ready"):
        leaf.block_until_ready()


def time_once(fn, *args, **kwargs) -> float:
    """Warm up once, then time one execution; return seconds."""
    out = fn(*args, **kwargs)
    _block_on_scalar(out)
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    _block_on_scalar(out)
    return time.perf_counter() - t0


def profile_once(trace_dir: str, fn, *args, **kwargs):
    """Emit a single TensorBoard trace for fn(*args, **kwargs).

    Uses create_perfetto_link=False to avoid binding to a local port that may already be in use.
    Always attempts to stop the trace in a finally block.
    """
    os.makedirs(trace_dir, exist_ok=True)
    started = False
    try:
        # Avoid port binding issues
        jax.profiler.start_trace(trace_dir, create_perfetto_link=False)
        started = True
    except Exception:
        # If starting a trace fails (e.g., already tracing), proceed without tracing
        started = False
    try:
        out = fn(*args, **kwargs)
        _block_on_scalar(out)
        return out
    finally:
        if started:
            try:
                jax.profiler.stop_trace()
            except Exception:
                pass


