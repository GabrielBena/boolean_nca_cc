import time
from functools import partial
import numpy as np
import jax
import jax.numpy as jp
import optax
import PIL.Image, PIL.ImageDraw
import IPython
import orbax.checkpoint as ocp
import os

from imgui_bundle import (
    implot,
    imgui,
    immapp,
    immvision,
    hello_imgui,
)

from CircuitViz import CircuitVisualizer # Assuming CircuitViz.py is in the same directory or PYTHONPATH

################## Copied/Adapted from random_wires_demo.py ##################

# --- Constants and Circuit Definition ---
input_n, output_n = 8, 8
case_n = 1 << input_n
arity, layer_width, layer_n = 4, 64, 5
layer_sizes = (
    [(input_n, 1)]  # (gate_n, group_size)
    + [(layer_width, arity)] * (layer_n - 1)
    + [(layer_width // 2, arity // 2), (output_n, 1)]
)
max_trainstep_n = 1000 # For loss log size

# --- Helper Functions ---
def zoom(a, k=2):
    return np.repeat(np.repeat(a, k, 1), k, 0)

def unpack(x, bit_n=8):
    return (x[..., None] >> np.r_[:bit_n]) & 1

def gen_wires(key, in_n, out_n, arity_val, group_size_val, local_noise_val=None): # Renamed to avoid conflict
    edge_n = out_n * arity_val // group_size_val
    if in_n != edge_n or local_noise_val is None:
        n = max(in_n, edge_n)
        return jax.random.permutation(key, n)[:edge_n].reshape(arity_val, -1) % in_n
    i = (
        jp.arange(edge_n) + jax.random.normal(key, shape=(edge_n,)) * local_noise_val
    ).argsort()
    return i.reshape(-1, arity_val).T

def make_nops(gate_n, arity_val, group_size_val, nop_scale=3.0): # Renamed
    I = jp.arange(1 << arity_val)
    bits = (I >> I[:arity_val, None]) & 1
    luts = bits[jp.arange(gate_n) % arity_val]
    logits = (2.0 * luts - 1.0) * nop_scale
    return logits.reshape(gate_n // group_size_val, group_size_val, -1)

@jax.jit
def run_layer(lut, inputs):
    for x_in in inputs: # Renamed x to x_in
        x_in = x_in[..., None, None]
        lut = (1.0 - x_in) * lut[..., ::2] + x_in * lut[..., 1::2]
    return lut.reshape(*lut.shape[:-3] + (-1,))

def run_circuit(logits_list, wires_list, gate_mask_list, x_input, hard=False): # Renamed
    x_val = x_input * gate_mask_list[0]
    acts_list = [x_val]
    for ws, lgt, mask in zip(wires_list, logits_list, gate_mask_list[1:]):
        luts = jax.nn.sigmoid(lgt)
        if hard:
            luts = jp.round(luts)
        x_val = run_layer(luts, [x_val[..., w] for w in ws]) * mask
        acts_list.append(x_val)
    return acts_list

def res2loss(res):
    return jp.square(jp.square(res)).sum()

def loss_f(logits_list, wires_list, gate_mask_list, x_input, y0_target): # Renamed
    run_f = partial(run_circuit, logits_list, wires_list, gate_mask_list, x_input)
    act_out = run_f()
    loss = res2loss(act_out[-1] - y0_target)
    hard_act_out = run_f(hard=True)
    hard_loss = res2loss(hard_act_out[-1] - y0_target)
    err_mask = hard_act_out[-1] != y0_target
    return loss, dict(
        act=act_out, err_mask=err_mask, hard_loss=hard_loss, hard_act=hard_act_out
    )

grad_loss_f = jax.jit(jax.value_and_grad(loss_f, argnums=0, has_aux=True)) # Grad w.r.t. logits_list

# --- Gate Use Analysis Functions (for mask_unused_gates callback) ---
def calc_lut_input_use(logits_val):
    luts = jp.sign(logits_val) * 0.5 + 0.5
    arity_calc = luts.shape[-1].bit_length() - 1
    luts = luts.reshape(luts.shape[:-1] + (2,) * arity_calc)
    axes_to_flatten = -1 - np.arange(arity_calc - 1)
    input_use = []
    for i_ax in range(1, arity_calc + 1):
        m = luts.take(0, -i_ax) != luts.take(1, -i_ax)
        m = m.any(axes_to_flatten)
        input_use.append(m)
    return jp.stack(input_use)

def propagate_gate_use(prev_layer_gate_n, wires_for_layer, logits_for_layer, output_use_from_next_layer):
    output_use_reshaped = output_use_from_next_layer.reshape(logits_for_layer.shape[:2])
    gate_input_use = calc_lut_input_use(logits_for_layer) * output_use_reshaped
    wire_use_mask = gate_input_use.any(-1)
    used_wires_indices = wires_for_layer[wire_use_mask]
    prev_gate_use = np.zeros(prev_layer_gate_n, np.bool_)
    np.add.at(prev_gate_use, used_wires_indices, True) # Ensure it handles duplicates if any
    return prev_gate_use, wire_use_mask

def calc_gate_use_masks_logic(circuit_input_n, wires_list, logits_list): # Renamed
    current_layer_sizes = [circuit_input_n] + [np.prod(l.shape[:2]) for l in logits_list]
    gate_use_mask = np.ones(current_layer_sizes[-1], np.bool_)
    
    calculated_gate_masks = [gate_use_mask]
    calculated_wire_masks = []
    
    for i_layer in range(len(logits_list))[::-1]:
        gate_use_mask, wire_use_mask = propagate_gate_use(
            current_layer_sizes[i_layer], wires_list[i_layer], logits_list[i_layer], gate_use_mask
        )
        calculated_wire_masks.append(wire_use_mask)
        calculated_gate_masks.append(gate_use_mask)
    return calculated_gate_masks[::-1], calculated_wire_masks[::-1]

################## AppLogic Class ##################
class AppLogic:
    def __init__(self):
        self.logits0 = []
        for gate_n_layer, group_size_layer in layer_sizes[1:]: # First entry in layer_sizes is for input layer
            self.logits0.append(make_nops(gate_n_layer, arity, group_size_layer))
        self.logits = [jp.array(l) for l in self.logits0] # Keep as JAX arrays
        
        print(f"AppLogic: param_n: {sum(l.size for l in self.logits0)}")

        self.wires_key = jax.random.PRNGKey(42)
        self.local_noise = 0.0
        self.wires = [] # Will be populated by shuffle_wires
        self.shuffle_wires() # Initial shuffle

        self.gate_mask = [] # List of np arrays (0.0 or 1.0)
        self.wire_masks = [] # List of np bool arrays
        self.reset_gate_mask() # Initialize

        x_range = jp.arange(case_n)
        self.input_x = unpack(x_range) # JAX array (case_n, input_n)
        
        inp_img_prep = self.input_x.T # (input_n, case_n)
        inp_img_prep = np.array(inp_img_prep) # to numpy for PIL
        inp_img_prep = np.dstack([inp_img_prep] * 3)
        inp_img_prep = zoom(inp_img_prep, 4)
        self.inputs_img = np.uint8(inp_img_prep.clip(0, 1) * 255)
        
        self.active_case_i = 123
        self.y0 = None # Target outputs, JAX array

        self.tasks = dict(
            copy=x_range,
            gray=x_range ^ (x_range >> 1),
            add4=(x_range & 0xF) + (x_range >> 4),
            mul4=(x_range & 0xF) * (x_range >> 4),
            popcount=np.bitwise_count(np.array(x_range)), # bitwise_count needs np array
            text=x_range, # Placeholder, will be replaced by image
            noise=x_range, # Placeholder
        )
        self.task_names = list(self.tasks)
        self.task_idx = self.task_names.index("mul4")
        self.task_text = "All you need is JAX and ImGui"
        self.noise_p = 0.5
        self.noise_data = None # For "noise" task
        self.sample_noise()
        self.update_task() # Sets self.y0

        self.wd_log10 = -1.0
        self.opt = self.get_opt()
        self.opt_state = self.opt.init(self.logits)
        
        self.loss_log = np.zeros(max_trainstep_n, np.float32)
        self.hard_log = np.zeros(max_trainstep_n, np.float32)
        self.trainstep_i = 0
        self.is_training = True

        self.act = [] # Activations from run_circuit
        self.outputs_img = np.zeros_like(self.inputs_img) # Placeholder

    def get_opt(self):
        return optax.adamw(2.0, 0.8, 0.8, weight_decay=10**self.wd_log10)

    def reset_gate_mask(self):
        self.gate_mask = [np.ones(gs[0], dtype=np.float32) for gs in layer_sizes] # gs[0] is gate_n
        self.wire_masks = [np.ones_like(w, dtype=np.bool_) for w in self.wires if w is not None and w.size >0] # Ensure wires exist

    def _apply_calculated_masks(self):
        # This helper updates self.gate_mask and self.wire_masks from calc_gate_use_masks_logic
        # It assumes self.logits are JAX arrays, calc_gate_use_masks_logic may need them as numpy
        numpy_logits = [np.array(l) for l in self.logits]
        gate_masks_calc, wire_masks_calc = calc_gate_use_masks_logic(input_n, self.wires, numpy_logits)
        
        new_gate_mask = []
        for i in range(len(gate_masks_calc)):
            # Original gate_mask is float, calculated is bool. Multiply to combine.
            new_gate_mask.append(np.array(self.gate_mask[i] * gate_masks_calc[i], dtype=np.float32))
        self.gate_mask = new_gate_mask
        self.wire_masks = wire_masks_calc # wire_masks are directly replaced

    def shuffle_wires(self):
        current_in_n = input_n
        new_wires = []
        key = self.wires_key
        for gate_n_layer, group_size_layer in layer_sizes[1:]: # Skip input layer
            key, k1 = jax.random.split(key)
            local_noise_to_use = self.local_noise if self.local_noise > 0.0 else None
            ws = gen_wires(k1, current_in_n, gate_n_layer, arity, group_size_layer, local_noise_to_use)
            new_wires.append(ws)
            current_in_n = gate_n_layer
        self.wires = new_wires
        self.wires_key = key # Update key for next shuffle
        # Wire masks need to be reset/recalculated after shuffling
        self.wire_masks = [np.ones_like(w, dtype=np.bool_) for w in self.wires if w is not None and w.size >0]


    def sample_noise(self):
        # Ensure this uses JAX random if it feeds into JAX computations later, or numpy if for PIL image
        self.noise_data = np.random.rand(case_n, output_n) # Matches output_n for noise task target

    def update_task(self):
        task_name = self.task_names[self.task_idx]
        if task_name == "text":
            im = PIL.Image.new("L", (case_n, output_n)) # output_n for target image height
            draw = PIL.ImageDraw.Draw(im)
            # Adjust text position if needed
            font = None # Use default
            try:
                # For better text rendering, specify a font if available
                from PIL import ImageFont
                font = ImageFont.truetype("arial.ttf", 10) # Example, adjust size/font
            except ImportError:
                pass
            if hasattr(draw, 'textbbox'): # More modern PIL
                bbox = draw.textbbox((0,0), self.task_text, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
                draw.text(((case_n - text_width) / 2, (output_n - text_height) / 2 -2), self.task_text, fill=255, font=font)
            else: # Older PIL
                 draw.text((2, -2), self.task_text, fill=255) # Might need adjustment
            self.y0 = jp.float32(np.array(im).T > 100) # Transpose to (output_n, case_n)
        elif task_name == "noise":
            self.y0 = jp.float32(self.noise_data.T < self.noise_p) # Transpose
        else:
            self.y0 = jp.float32(unpack(self.tasks[task_name], bit_n=output_n).T) # (output_n, case_n)
        # Ensure y0 has shape (output_n, case_n) to match circuit output
        # Circuit output is (case_n, output_n), so target should match that structure.
        # Let's adjust run_circuit output or target shape.
        # Original demo: act[-1] is (case_n, output_n), y0 is (case_n, output_n)
        # So, unpack result should be (case_n, output_n)
        if task_name == "text":
             self.y0 = jp.array(np.array(im) > 100, dtype=jp.float32) # (case_n, output_n)
        elif task_name == "noise":
            self.y0 = jp.array(self.noise_data < self.noise_p, dtype=jp.float32) # (case_n, output_n)
        else:
            self.y0 = jp.array(unpack(self.tasks[task_name], bit_n=output_n), dtype=jp.float32) # (case_n, output_n)


    def simulation_step(self):
        # Convert gate_mask (float) to bool for parts of run_circuit if needed, or ensure run_circuit handles float mask
        # The original run_circuit multiplies by mask, so float is fine.
        (loss_val, aux_data), grad_val = grad_loss_f(
            self.logits, self.wires, self.gate_mask, self.input_x, self.y0
        )

        if self.is_training:
            updates, self.opt_state = self.opt.update(grad_val, self.opt_state, self.logits)
            self.logits = optax.apply_updates(self.logits, updates)

        self.act = aux_data["act"] # list of JAX arrays
        
        # Prepare outputs_img
        oimg_prep = self.act[-1] # (case_n, output_n)
        oimg_prep = np.array(oimg_prep.T) # (output_n, case_n) for consistency with inputs_img prep
        
        m_prep = np.array(aux_data["err_mask"].T) # (output_n, case_n)
        
        oimg_display = np.dstack([oimg_prep] * 3) # (output_n, case_n, 3)
        m_display = m_prep[..., None] * 0.5 # (output_n, case_n, 1)

        oimg_display = oimg_display * (1.0 - m_display) + m_display * np.float32([1,0,0]) # Error highlighting
        oimg_display = zoom(oimg_display, 4)
        self.outputs_img = np.uint8(oimg_display.clip(0,1)*255)

        hard_loss_val = aux_data["hard_loss"].item()
        log_idx = self.trainstep_i % len(self.loss_log)
        self.loss_log[log_idx] = loss_val.item()
        self.hard_log[log_idx] = hard_loss_val
        
        if self.is_training:
            self.trainstep_i +=1

    # --- Callback Handlers ---
    def handle_set_active_case_i(self, new_case_i):
        self.active_case_i = new_case_i
        print(f"AppLogic: Active case set to {new_case_i}")

    def handle_python_repl(self):
        print("AppLogic: Launching Python REPL. Variables `app` and `viz` should be available if run in a certain way.")
        # Make self available in the REPL if possible, might need to pass it explicitly
        IPython.embed(header="IPython REPL launched from AppLogic. `self` refers to AppLogic instance.")

    def handle_toggle_training(self, is_training_val):
        self.is_training = is_training_val
        print(f"AppLogic: Training toggled to {self.is_training}")

    def handle_reset_gates(self):
        self.logits = [jp.array(l) for l in self.logits0] # Reset to initial JAX arrays
        self.trainstep_i = 0 # Reset training progress
        print("AppLogic: Gates (logits) reset")

    def handle_reset_gates_opt(self):
        self.logits = [jp.array(l) for l in self.logits0]
        self.opt_state = self.opt.init(self.logits) # Re-initialize optimizer state
        self.trainstep_i = 0
        print("AppLogic: Gates (logits) and optimizer reset")

    def handle_shuffle_wires(self):
        # wires_key is already part of self
        self.shuffle_wires() # This now updates self.wires and self.wires_key
        self.trainstep_i = 0
        print("AppLogic: Wires shuffled")
        # Gate masks might need resetting or recalculating if dependent on wire structure for calc_gate_use
        self.reset_gate_mask() # Simplest: just reset them

    def handle_set_local_noise(self, new_noise):
        if self.local_noise != new_noise:
            self.local_noise = new_noise
            self.shuffle_wires() # Re-shuffle with new noise
            print(f"AppLogic: Local noise set to {self.local_noise}, wires reshuffled")

    def handle_set_wd_log10(self, new_wd):
        if self.wd_log10 != new_wd:
            self.wd_log10 = new_wd
            self.opt = self.get_opt() # Recreate optimizer with new weight decay
            self.opt_state = self.opt.init(self.logits) # Re-init opt state
            print(f"AppLogic: Weight decay (log10) set to {self.wd_log10}")

    def handle_reset_gate_mask_action(self):
        self.reset_gate_mask()
        print("AppLogic: Gate mask reset")

    def handle_mask_unused_gates_action(self):
        self._apply_calculated_masks()
        print("AppLogic: Unused gates masked")
        
    def handle_set_task_idx(self, new_idx):
        if self.task_idx != new_idx:
            self.task_idx = new_idx
            self.update_task() # Update self.y0
            self.trainstep_i = 0
            print(f"AppLogic: Task changed to {self.task_names[self.task_idx]}")

    def handle_set_task_text(self, new_text):
        if self.task_text != new_text and self.task_names[self.task_idx] == "text":
            self.task_text = new_text
            self.update_task()
            print(f"AppLogic: Task text updated to: {self.task_text}")

    def handle_set_noise_p(self, new_p):
        if self.noise_p != new_p and self.task_names[self.task_idx] == "noise":
            self.noise_p = new_p
            self.update_task() # Re-calculate y0 based on new noise_p
            print(f"AppLogic: Noise p updated to {self.noise_p}")

# Global instances for the application
app_logic = None
visualizer = None

def main_gui_loop():
    global app_logic, visualizer

    # 1. Update simulation state
    app_logic.simulation_step()

    # 2. Pass data to visualizer
    # Ensure all data passed matches what CircuitVisualizer expects
    # Esp. logits, wires, wire_masks, gate_mask, act should be lists of arrays
    visualizer.update_data(
        logits=app_logic.logits,
        wires=app_logic.wires,
        wire_masks=app_logic.wire_masks, # Ensure this is correctly populated
        gate_mask=app_logic.gate_mask,   # Ensure this is correctly populated
        act=app_logic.act,
        inputs_img=app_logic.inputs_img,
        outputs_img=app_logic.outputs_img,
        active_case_i=app_logic.active_case_i,
        loss_log=app_logic.loss_log,
        hard_log=app_logic.hard_log,
        trainstep_i=app_logic.trainstep_i,
        is_training=app_logic.is_training,
        wd_log10=app_logic.wd_log10,
        local_noise=app_logic.local_noise,
        task_idx=app_logic.task_idx,
        task_names=app_logic.task_names,
        task_text=app_logic.task_text,
        noise_p=app_logic.noise_p,
        logits0_ref=app_logic.logits0 # For potential use by a reset callback in visualizer if it were different
    )

    # 3. Draw GUI (visualizer handles its own drawing based on updated data)
    visualizer.gui()






# GNN FUNCTIONALITY
import matplotlib.pylab as pl
import matplotlib.pyplot as plt
import PIL.Image, PIL.ImageFont, PIL.ImageDraw
from tqdm.auto import trange
import jraph
import numpy as np

pl.style.use("dark_background")


# def np2pil(a):
#     a = np.asarray(a)
#     if a.dtype.kind == "f":
#         a = np.uint8(a.clip(0, 1) * 255)
#     return PIL.Image.fromarray(a)


# def imshow(a):
#     display(np2pil(a))


# def zoom(a, k=2):
#     return np.repeat(np.repeat(a, k, 1), k, 0)


# def tile2d(a, w=None):
#     a = np.asarray(a)
#     if w is None:
#         w = int(np.ceil(np.sqrt(len(a))))
#     th, tw = a.shape[1:3]
#     pad = (w - len(a)) % w
#     a = np.pad(a, [(0, pad)] + [(0, 0)] * (a.ndim - 1), "constant")
#     h = len(a) // w
#     a = a.reshape([h, w] + list(a.shape[1:]))
#     a = np.rollaxis(a, 2, 1).reshape([th * h, tw * w] + list(a.shape[4:]))
#     return a

import jax
import jax.numpy as jp
import optax
from flax import nnx
from boolean_nca_cc import generate_layer_sizes
from boolean_nca_cc.circuits.model import gen_circuit
from boolean_nca_cc.circuits.train import TrainState, train_step
from boolean_nca_cc.circuits.viz import evaluate_and_visualize
from boolean_nca_cc.circuits.tasks import get_task_data
from boolean_nca_cc.models import CircuitGNN, run_gnn_scan
from boolean_nca_cc.training import train_model
from boolean_nca_cc.utils import build_graph, extract_logits_from_graph
from boolean_nca_cc.circuits.train import loss_f_l4


input_n, output_n = 4, 4
arity = 2
layer_sizes = generate_layer_sizes(input_n, output_n, arity, layer_n=4)
loss_type = "l4"

key = jax.random.PRNGKey(42)
wires, logitsbp = gen_circuit(key, layer_sizes, arity=arity)

opt = optax.adamw(1, 0.8, 0.8, weight_decay=1e-1)
state = TrainState(params=logitsbp, opt_state=opt.init(logitsbp))


case_n = 1 << input_n
x = jp.arange(case_n)
x, y0 = get_task_data(
    "binary_multiply", case_n, input_bits=input_n, output_bits=output_n
)

# GNN params
hidden_dim = 128
hidden_features = 128
n_message_steps = 5
loss, aux = loss_f_l4(logitsbp, wires, x, y0)
graph = build_graph(
    logitsbp, wires, input_n, arity, hidden_dim=hidden_dim, loss_value=loss
)

gnn = CircuitGNN(
    hidden_dim=hidden_dim,
    message_passing=True,
    node_mlp_features=[hidden_features, hidden_features],
    edge_mlp_features=[hidden_features, hidden_features],
    rngs=nnx.Rngs(params=jax.random.PRNGKey(42)),
    use_attention=False,
    arity=arity,
)

opt_fn = optax.adamw(1e-4, weight_decay=1e-5)
opt = nnx.Optimizer(gnn, opt_fn)

def loss_fn(gnn: CircuitGNN, graph: jraph.GraphsTuple, wires: jax.Array = wires):
    updated_graph = run_gnn_scan(gnn, graph, n_message_steps)
    updated_logits = extract_logits_from_graph(
        updated_graph, [l.shape for l in logitsbp]
    )
    loss, aux = loss_f_l4(updated_logits, wires, x, y0)
    return loss, updated_graph

def plot_losses(
    loss_log, aux_log, figax=None, title=None, plot_accuracy=False, plot_hard=True
):
    if figax is None:
        fig, ax = plt.subplots()
    else:
        fig, ax = figax
    v = len(loss_log) // 10

    if not plot_accuracy:
        smooth_losses = np.convolve(loss_log, np.ones(v) / v, mode="valid")
        hard_losses = [aux["hard_loss"] for aux in aux_log]
        smooth_hard_losses = np.convolve(hard_losses, np.ones(v) / v, mode="valid")
        ax.plot(smooth_losses, label="soft")
        if plot_hard:
            ax.plot(jp.maximum(smooth_hard_losses, 1e-5), label="hard")
    else:
        accs = [aux["accuracy"] for aux in aux_log]
        smooth_accs = np.convolve(accs, np.ones(v) / v, mode="valid")
        ax.plot(smooth_accs, label="soft")
        hard_accs = [aux["hard_accuracy"] for aux in aux_log]
        smooth_hard_accs = np.convolve(hard_accs, np.ones(v) / v, mode="valid")
        if plot_hard:
            ax.plot(smooth_hard_accs, label="hard")

    ax.legend()
    ax.set_yscale("log")

    if title is not None:
        ax.set_title(title)

(loss, aux), grads = nnx.value_and_grad(loss_fn, has_aux=True)(gnn, graph)
opt.update(grads)
(loss, aux), grads = nnx.value_and_grad(loss_fn, has_aux=True)(gnn, graph)
assert any(jax.tree.leaves(jax.tree.map(lambda x: x.any(), grads))), "No grads"

# Before calling train_model
checkpoint_dir = "./gnn_checkpoints"
# Find the latest checkpoint, e.g., by finding the highest epoch number
# For simplicity, let's assume you know the epoch you want to load
epoch_to_load = 100 # or get this dynamically
load_path = os.path.join(checkpoint_dir, f"epoch_{epoch_to_load}")

loaded_model_state = None
loaded_optimizer_state = None

if os.path.exists(load_path):
    checkpointer = ocp.PyTreeCheckpointer()
    restored_dict = checkpointer.restore(load_path)
    loaded_model_state = restored_dict['model_state']
    loaded_optimizer_state = restored_dict['optimizer_state']
    print(f"Loaded GNN model and optimizer from {load_path}")
else:
    print(f"No checkpoint found at {load_path}, training from scratch.")

# Then, when calling train_model:
gnn_results = train_model(
    key=0,
    init_model=loaded_model_state, # Pass the nnx.State directly
    init_optimizer=loaded_optimizer_state, # Pass the nnx.State directly
    # CIRCUIT PARAMS
    layer_sizes=layer_sizes,
    x_data=x,
    y_data=y0,
    arity=arity,
    # TRAINING PARAMS
    epochs=2**15,
    n_message_steps=1, 
    meta_batch_size=256,
    # WIRING MODE PARAMS
    wiring_mode="random",
    wiring_fixed_key=jax.random.PRNGKey(42),
    # LOSS PARAMS
    loss_type="l4",
    # OPTIMIZER PARAMS
    learning_rate=1e-3,
    weight_decay=1e-5,
    # LEARNING RATE SCHEDULER
    lr_scheduler="linear_warmup",
    # Model Params
    hidden_dim=hidden_dim,
    # POOL PARAMS : current mean avg of 100 steps before reset
    use_pool=True,
    pool_size=1024,
    reset_pool_fraction=0.075,
    reset_pool_interval=2**5,
    reset_strategy="combined",
)

try:
    gnn_results["model"] = nnx.state(gnn_results["model"])
    gnn_results["optimizer"] = nnx.state(gnn_results["optimizer"])
except ValueError:
    print("already converted")
    pass

# --- ADD MODEL SAVING LOGIC HERE ---
if gnn_results and "model" in gnn_results and "optimizer" in gnn_results and "losses" in gnn_results:
    import orbax.checkpoint as ocp
    import os # For creating directories

    # Define a path for checkpoints
    checkpoint_dir = "./gnn_checkpoints"
    if not os.path.exists(checkpoint_dir):
        try:
            os.makedirs(checkpoint_dir)
            print(f"Created checkpoint directory: {checkpoint_dir}")
        except OSError as e:
            print(f"Error creating checkpoint directory {checkpoint_dir}: {e}")
            # Decide how to handle this - maybe skip saving or raise error

    if os.path.exists(checkpoint_dir): # Proceed only if directory exists or was created
        # Create a Checkpointer
        checkpointer = ocp.PyTreeCheckpointer()

        # Prepare the target to save (model state and optimizer state)
        save_target = {
            'model_state': gnn_results["model"], 
            'optimizer_state': gnn_results["optimizer"] 
        }
        
        num_epochs_completed = len(gnn_results.get("losses", [])) 
        if num_epochs_completed > 0:
            save_path_for_epoch = os.path.join(checkpoint_dir, f"epoch_{num_epochs_completed}")
            try:
                checkpointer.save(save_path_for_epoch, save_target)
                print(f"Saved GNN checkpoint at epoch {num_epochs_completed} to {save_path_for_epoch}")
            except Exception as e:
                print(f"Error saving checkpoint to {save_path_for_epoch}: {e}")
                # Optionally print more details for debugging
                # print("Model state was: ", save_target['model_state'])
                # print("Optimizer state was: ", save_target['optimizer_state'])
        else:
            print("No training epochs completed according to loss log, checkpoint not saved.")
else:
    print("Training results missing or checkpoint directory could not be created. GNN model not saved.")


# recreate aux in old format
aux_log = [
    {
        "accuracy": accuracies,
        "hard_accuracy": hard_acc,
        "hard_loss": hard_loss,
    }
    for accuracies, hard_acc, hard_loss in zip(
        gnn_results["accuracies"],
        gnn_results["hard_accuracies"],
        gnn_results["hard_losses"],
    )
]

fig, axs = plt.subplots(
    1, 2, figsize=(10, 5), sharey=False, sharex=True, constrained_layout=True
)
plot_losses(
    gnn_results["losses"], aux_log, (fig, axs[0]), title="META LOSS", plot_hard=True
)
plot_losses(
    gnn_results["losses"],
    aux_log,
    (fig, axs[1]),
    plot_accuracy=True,
    title="META ACCURACY",
    plot_hard=True,
)

gnn_save = CircuitGNN(
    hidden_dim=hidden_dim,
    message_passing=True,
    node_mlp_features=[hidden_features, hidden_features],
    edge_mlp_features=[hidden_features, hidden_features],
    rngs=nnx.Rngs(params=jax.random.PRNGKey(42)),
    use_attention=False,
    arity=arity,
)

fig, axs = plt.subplots(
    1, 2, figsize=(10, 5), sharey=False, sharex=True, constrained_layout=True
)
plot_losses(
    gnn_results["losses"], aux_log, (fig, axs[0]), title="META LOSS", plot_hard=True
)
plot_losses(
    gnn_results["losses"],
    aux_log,
    (fig, axs[1]),
    plot_accuracy=True,
    title="META ACCURACY",
    plot_hard=True,
)

v = len(gnn_results["reset_steps"]) // 10
plt.plot(np.convolve(gnn_results["reset_steps"], np.ones(v) / v, mode="valid"))

# Get the inner loop losses
from boolean_nca_cc.training.evaluation import evaluate_model_stepwise

key = jax.random.PRNGKey(42)
wires_gnn, logits_gnn = gen_circuit(key, layer_sizes, arity=arity)
step_metrics = evaluate_model_stepwise(
    gnn,
    wires_gnn,
    logits_gnn,
    x,
    y0,
    input_n,
    n_message_steps=100,
    arity=arity,
    hidden_dim=hidden_dim,
    loss_type="l4",
)

aux_log_stepwise = [
    {"accuracy": acc, "hard_accuracy": hard_acc, "hard_loss": hard_loss}
    for acc, hard_acc, hard_loss in zip(
        step_metrics["soft_accuracy"],
        step_metrics["hard_accuracy"],
        step_metrics["hard_loss"],
    )
]

gnn_log_results = {
    "losses": step_metrics["soft_loss"],
    "aux_log": aux_log_stepwise,
}


# if __name__ == "__main__":
#     app_logic = AppLogic()
    
#     visualizer = CircuitVisualizer(
#         layer_sizes=layer_sizes, # Pass the global layer_sizes
#         arity=arity,             # Pass the global arity
#         case_n=case_n,
#         input_n=input_n,
#         # --- Register Callbacks ---
#         on_set_active_case_i=app_logic.handle_set_active_case_i,
#         on_python_repl=app_logic.handle_python_repl,
#         on_toggle_training=app_logic.handle_toggle_training,
#         on_reset_gates=app_logic.handle_reset_gates,
#         on_reset_gates_opt=app_logic.handle_reset_gates_opt,
#         on_shuffle_wires=app_logic.handle_shuffle_wires,
#         on_set_local_noise=app_logic.handle_set_local_noise,
#         on_set_wd_log10=app_logic.handle_set_wd_log10,
#         on_reset_gate_mask_action=app_logic.handle_reset_gate_mask_action,
#         on_mask_unused_gates=app_logic.handle_mask_unused_gates_action,
#         on_set_task_idx=app_logic.handle_set_task_idx,
#         on_set_task_text=app_logic.handle_set_task_text,
#         on_set_noise_p=app_logic.handle_set_noise_p
#         # Note: The non-input gate click that zeros LUTs is now handled directly by CircuitVisualizer
#         # so no callback for on_toggle_gate_mask in the traditional sense for that action.
#     )
    
#     # Setup ImGui Bundle runner parameters
#     runner_params = hello_imgui.RunnerParams()
#     # Set only non-window/non-addon related runner_params if needed.
#     # For now, most things will be passed to immapp.run directly.
#     # runner_params.app_window_params.window_title = "CircuitViz Integration Test" # Will be kwarg
#     # runner_params.app_window_params.window_size_auto = True # Will be kwarg
#     # runner_params.app_window_params.restore_previous_geometry = True # Will be kwarg
#     runner_params.fps_idling.enable_idling = True # This might still be useful on RunnerParams if not a direct kwarg
#     # runner_params.fps_idling.fps_idle = 10 # Will be kwarg

#     # Addons: 
#     # runner_params.imgui_window_params.with_implot = True # Will be kwarg
#     immvision.use_rgb_color_order() # If images are RGB

#     # immapp.run expects a function that takes no arguments (the GUI loop)
#     # or a GuiFunction object, or a class with a gui method.
#     # main_gui_loop is a valid GuiFunction.
#     immapp.run(
#         gui_function=main_gui_loop, 
#         # runner_params=runner_params, # Pass runner_params if it has other settings we need
#         window_title="CircuitViz Integration Test",
#         window_size_auto=True,
#         window_restore_previous_geometry=True,
#         fps_idle=10,
#         with_implot=True
#         # Note: if runner_params is passed, these keywords might conflict or be overridden
#         # depending on the specific version and behavior of imgui-bundle.
#         # For now, we prioritize direct keywords as per random_wires_demo.py.
#     )

#     print("Application closed.") 