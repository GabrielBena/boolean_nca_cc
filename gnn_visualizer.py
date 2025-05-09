import numpy as np
import jax
import jax.numpy as jp
import PIL.Image, PIL.ImageDraw
import IPython # Keep for REPL button, can be made optional

from imgui_bundle import (
    implot,
    imgui,
    immapp,
    immvision,
    hello_imgui,
)

# Helper functions (similar to random_wires_demo.py)
def zoom(a, k=2):
    if a is None or a.size == 0:
        return np.zeros((0,0,3) if len(a.shape) == 3 else (0,0) , dtype=np.uint8)
    return np.repeat(np.repeat(a, k, axis=1), k, axis=0)

def unpack(x, bit_n=8):
    if x is None:
        return np.array([])
    return (x[..., None] >> np.arange(bit_n)) & 1

def is_point_in_box(p0, p1, p):
    (x0, y0), (x1, y1), (x, y) = p0, p1, p
    return (x0 <= x <= x1) and (y0 <= y <= y1)

# Placeholder for CircuitVisualizer class and run_gui function
# They will be added in subsequent steps. 