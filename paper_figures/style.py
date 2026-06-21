"""Consistent, colour-blind-safe styling for the SODC paper figures.

Uses the Okabe-Ito palette (distinguishable under deuteranopia/protanopia/
tritanopia) to address reviewer R1f, and a single set of rcParams + label maps
so every figure is stylistically consistent (R2).
"""
from __future__ import annotations

import matplotlib as mpl

# Okabe-Ito colour-blind-safe palette
OKABE_ITO = {
    "black": "#000000",
    "orange": "#E69F00",
    "skyblue": "#56B4E9",
    "green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
}

# Eval-time damage encoding: blue (undamaged) vs vermillion (damaged).
# This pair is the most reliably distinguishable across all CVD types.
EVAL_DAMAGE_COLORS = {"OFF": OKABE_ITO["blue"], "ON": OKABE_ITO["vermillion"]}
BP_COLOR = OKABE_ITO["black"]  # BP shown as a neutral reference ceiling

# Canonical task naming (fixes the "Binary_" underscore; TMT not NCA).
TASK_ORDER = ["Bit Reversal", "Binary Addition", "Binary Multiplication"]
TASK_MAP = {
    "reverse_large": "Bit Reversal",
    "add_large": "Binary Addition",
    "binary_multiply_large": "Binary Multiplication",
    "reverse": "Bit Reversal",
    "add": "Binary Addition",
    "binary_multiply": "Binary Multiplication",
}
DAMAGE_MAP = {
    "true": "ON", "false": "OFF", "True": "ON", "False": "OFF",
    True: "ON", False: "OFF", "random": "ON", "none": "OFF",
}


def set_rc() -> None:
    """Apply consistent, publication-ready matplotlib defaults (vector-safe fonts)."""
    mpl.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.size": 9,
        "axes.titlesize": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "pdf.fonttype": 42,   # embed TrueType (no Type-3) for camera-ready
        "ps.fonttype": 42,
    })
