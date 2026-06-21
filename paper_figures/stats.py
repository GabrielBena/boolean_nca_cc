"""Significance + effect-size helpers (no plotting), for reviewer R1g.

We use the Mann-Whitney U test (non-parametric, appropriate for small n=5 seeds
and bounded accuracy) together with Cliff's delta as a non-parametric effect
size, since accuracies are bounded and far from normal near the ceiling.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import mannwhitneyu


def cliffs_delta(a, b) -> float:
    """Cliff's delta effect size in [-1, 1] (P(a>b) - P(a<b))."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    if a.size == 0 or b.size == 0:
        return float("nan")
    gt = sum(int(x > y) for x in a for y in b)
    lt = sum(int(x < y) for x in a for y in b)
    return (gt - lt) / (a.size * b.size)


def interpret_delta(d: float) -> str:
    """Romano et al. thresholds for |Cliff's delta|."""
    if np.isnan(d):
        return "n/a"
    ad = abs(d)
    if ad < 0.147:
        return "negligible"
    if ad < 0.33:
        return "small"
    if ad < 0.474:
        return "medium"
    return "large"


def compare(a, b) -> dict:
    """Two-sided Mann-Whitney U + Cliff's delta for samples a vs b."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    out = {
        "n_a": int(a.size),
        "n_b": int(b.size),
        "median_a": float(np.median(a)) if a.size else float("nan"),
        "median_b": float(np.median(b)) if b.size else float("nan"),
    }
    if a.size and b.size and np.unique(np.concatenate([a, b])).size > 1:
        u, p = mannwhitneyu(a, b, alternative="two-sided")
        out["U"], out["p"] = float(u), float(p)
    else:
        # identical/degenerate samples (e.g. both pinned at 1.0): no detectable difference
        out["U"], out["p"] = float("nan"), float("nan")
    d = cliffs_delta(a, b)
    out["cliffs_delta"] = d
    out["effect"] = interpret_delta(d)
    return out
