"""
Tiny per-phase profiler for the live demo.

The demo loop interleaves JAX compute, numpy host work, and ImGui draw
calls. When the whole thing feels sluggish it's not always obvious
which of those three is the culprit. ``FrameProfiler`` lets you wrap
named regions of code in ``with prof.region("name"):`` and see a
moving-average breakdown in milliseconds + percent of total frame.

Tradeoffs:

* Uses ``time.perf_counter`` only -- no external deps.
* JAX is asynchronous; if you want a region's timing to capture true
  device compute, pass ``sync_with=<jax_array>`` so the context exit
  runs ``jax.block_until_ready`` before stopping the clock. Otherwise
  the dispatch cost is what gets measured (which is sometimes what you
  want).
* Designed to run *every* frame with negligible overhead (~a few µs
  per region).
"""

from __future__ import annotations

import time
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import numpy as np

try:
    import jax

    _HAS_JAX = True
except ImportError:  # pragma: no cover - jax is a hard dep of the project
    _HAS_JAX = False


@dataclass
class _RegionStats:
    samples: deque[float] = field(default_factory=lambda: deque(maxlen=120))
    # Frame-index of the most recent sample. ``-1`` means "never hit".
    # Used by the GUI to grey-out regions whose last sample is stale
    # (e.g. ``bp.*`` rows after the user switches to TMT — those would
    # otherwise linger in the mean column for 120 frames).
    last_frame: int = -1

    def add(self, dt: float, frame: int) -> None:
        self.samples.append(dt)
        self.last_frame = frame

    def stats_ms(self) -> tuple[float, float, float]:
        """Return ``(mean, p95, last)`` in milliseconds."""
        if not self.samples:
            return 0.0, 0.0, 0.0
        arr = np.fromiter(self.samples, dtype=np.float64)
        return (
            float(arr.mean() * 1000.0),
            float(np.percentile(arr, 95) * 1000.0),
            float(arr[-1] * 1000.0),
        )


class FrameProfiler:
    """
    Lightweight, named-region timing sink.

    Usage:

        prof = FrameProfiler()

        with prof.frame():
            with prof.region("compute"):
                ...
            with prof.region("render"):
                ...

        for name, mean_ms, p95_ms, last_ms in prof.summary():
            print(name, mean_ms)
    """

    def __init__(self, window: int = 120) -> None:
        self.window = window
        self._regions: dict[str, _RegionStats] = {}
        self._frame_start: float | None = None
        self._frame_total = _RegionStats(samples=deque(maxlen=window))
        self._frame_idx = 0
        # Insertion order is preserved by dict in 3.7+, which keeps the
        # GUI breakdown in the same order regions are first declared.

    # ------------------------------------------------------------------
    @contextmanager
    def region(self, name: str, sync_with: Any | None = None):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            if sync_with is not None and _HAS_JAX:
                jax.block_until_ready(sync_with)
            dt = time.perf_counter() - t0
            slot = self._regions.get(name)
            if slot is None:
                slot = _RegionStats(samples=deque(maxlen=self.window))
                self._regions[name] = slot
            slot.add(dt, self._frame_idx)

    # ------------------------------------------------------------------
    @contextmanager
    def frame(self):
        self._frame_start = time.perf_counter()
        try:
            yield
        finally:
            if self._frame_start is not None:
                self._frame_total.add(
                    time.perf_counter() - self._frame_start, self._frame_idx
                )
                self._frame_start = None
            self._frame_idx += 1

    # ------------------------------------------------------------------
    def summary(self, stale_after: int = 30) -> list[tuple[str, float, float, float, bool]]:
        """``[(name, mean_ms, p95_ms, last_ms, is_stale), ...]`` in declaration order.

        ``is_stale`` is True if the region hasn't been hit in the last
        ``stale_after`` frames. Useful for greying out e.g. ``bp.*``
        rows after a TMT model is loaded.
        """
        out: list[tuple[str, float, float, float, bool]] = []
        cur = self._frame_idx
        for name, slot in self._regions.items():
            mean, p95, last = slot.stats_ms()
            stale = (slot.last_frame < 0) or (cur - slot.last_frame > stale_after)
            out.append((name, mean, p95, last, stale))
        return out

    def frame_stats_ms(self) -> tuple[float, float, float]:
        return self._frame_total.stats_ms()

    def fps(self) -> float:
        mean, _, _ = self._frame_total.stats_ms()
        if mean <= 0:
            return 0.0
        return 1000.0 / mean

    def reset(self) -> None:
        for slot in self._regions.values():
            slot.samples.clear()
            slot.last_frame = -1
        self._frame_total.samples.clear()
        self._frame_total.last_frame = -1
        self._frame_idx = 0
