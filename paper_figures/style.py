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
BP_COLOR = OKABE_ITO["black"]  # BP shown as a neutral reference ceiling

# --- Palette switch -------------------------------------------------------------
# Co-author's uniform 2-colour scheme, matching the un-rerunnable Fig 6 (Hamming):
# green = TMT / undamaged (favourable), indigo = BP / damaged (baseline).
# Enable with UNIFORM_PALETTE=1; figures are then written with a `_uniform` suffix,
# leaving the default Okabe-Ito versions intact.
import os as _os
UNIFORM = {"green": "#3AB97B", "indigo": "#454A89"}
USE_UNIFORM = _os.environ.get("UNIFORM_PALETTE", "").lower() not in ("", "0", "false", "no")
OUT_SUFFIX = "_uniform" if USE_UNIFORM else ""
if USE_UNIFORM:
    METHOD = {"TMT": UNIFORM["green"], "BP": UNIFORM["indigo"]}
    EVAL_DAMAGE_COLORS = {"OFF": UNIFORM["green"], "ON": UNIFORM["indigo"]}
else:
    METHOD = {"TMT": OKABE_ITO["blue"], "BP": OKABE_ITO["orange"]}
    EVAL_DAMAGE_COLORS = {"OFF": OKABE_ITO["blue"], "ON": OKABE_ITO["vermillion"]}

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
# Short titles for narrow single-column panels (full names live in the caption)
TASK_SHORT = {
    "Bit Reversal": "Reversal",
    "Binary Addition": "Addition",
    "Binary Multiplication": "Multiplication",
}

DAMAGE_MAP = {
    "true": "ON", "false": "OFF", "True": "ON", "False": "OFF",
    True: "ON", False: "OFF", "random": "ON", "none": "OFF",
}


# Page geometry (measured from the build): columnwidth = 239.4pt, textwidth = 505.9pt
COL_WIDTH = 3.31   # inches -- single-column figure width
TEXT_WIDTH = 7.0   # inches -- full (two-column) figure width


def set_rc(base: int = 9) -> None:
    """Apply consistent, publication-ready matplotlib defaults (vector-safe fonts).

    base: base font size (pt). Use ~7 for single-column (~3.3in) figures so text
    stays legible at single-column width (no LaTeX downscaling)."""
    mpl.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.size": base,
        "axes.titlesize": base,
        "axes.labelsize": base,
        "xtick.labelsize": base - 1,
        "ytick.labelsize": base - 1,
        "legend.fontsize": base - 1,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "pdf.fonttype": 42,   # embed TrueType (no Type-3) for camera-ready
        "ps.fonttype": 42,
    })
