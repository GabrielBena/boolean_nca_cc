#!/usr/bin/env python3
# Randomly wired boolean circuits demo - FIXED VERSION FOR WSL
# Modified from Alexander Mordvintsev's original
# This version avoids immvision and uses direct imgui image display

import time
import numpy as np
import jax
import jax.numpy as jp
import optax
import PIL.Image, PIL.ImageDraw
import IPython
import ctypes
from functools import partial
import pickle
import os
from flax import nnx

# Import modules for boolean circuits
from boolean_nca_cc.circuits.model import (
    make_nops,
    run_circuit,
    generate_layer_sizes,
)
from boolean_nca_cc.circuits.train import (
    unpack,
    res2loss,
)
import boolean_nca_cc.circuits.tasks as tasks  # Import the tasks module

# Import GNN components
from boolean_nca_cc.models import CircuitGNN, run_gnn_scan
from boolean_nca_cc.models.self_attention import (
    CircuitSelfAttention,
    run_self_attention_scan,
)
from boolean_nca_cc.utils import build_graph, extract_logits_from_graph

# Import model loading utility
from boolean_nca_cc.training.utils import load_best_model_from_wandb

from imgui_bundle import (
    implot,
    imgui,
    immapp,
    hello_imgui,
)


# Extend gen_wires in model.py to support local_noise
def gen_wires_with_noise(key, in_n, out_n, arity, group_size, local_noise=None):
    """Extended version of gen_wires that supports local_noise parameter"""
    edge_n = out_n * arity // group_size
    if in_n != edge_n or local_noise is None:
        n = max(in_n, edge_n)
        return jax.random.permutation(key, n)[:edge_n].reshape(arity, -1) % in_n
    i = (
        jp.arange(edge_n) + jax.random.normal(key, shape=(edge_n,)) * local_noise
    ).argsort()
    return i.reshape(-1, arity).T


def res2loss(res):
    return jp.square(jp.square(res)).sum()


# Use the modified run_circuit that accepts gate_mask
def run_circuit_gui(logits, wires, gate_mask, x, hard=False):
    """Run circuit with gate masking for GUI use"""
    return run_circuit(logits, wires, x, gate_mask=gate_mask, hard=hard)


def loss_f_gui(logits, wires, gate_mask, x, y0):
    """Modified loss function that uses gate masking"""
    run_f = partial(run_circuit_gui, logits, wires, gate_mask, x)
    act = run_f()
    loss = res2loss(act[-1] - y0)
    hard_act = run_f(hard=True)
    hard_loss = res2loss(hard_act[-1] - y0)
    err_mask = hard_act[-1] != y0
    return loss, dict(
        act=act, err_mask=err_mask, hard_loss=hard_loss, hard_act=hard_act
    )


grad_loss_f_gui = jax.jit(jax.value_and_grad(loss_f_gui, has_aux=True))


################## circuit gate and wire use analysis ##################


def calc_lut_input_use(logits):
    # (..., lut) -> (..., input_use_mask)
    luts = jp.sign(logits) * 0.5 + 0.5
    arity = luts.shape[-1].bit_length() - 1
    luts = luts.reshape(luts.shape[:-1] + (2,) * arity)
    axes_to_flatten = -1 - np.arange(arity - 1)
    input_use = []
    for i in range(1, arity + 1):
        m = luts.take(0, -i) != luts.take(1, -i)
        m = m.any(axes_to_flatten)
        input_use.append(m)
    return jp.stack(input_use)


def propatate_gate_use(input_n, wires, logits, output_use):
    output_use = output_use.reshape(logits.shape[:2])
    gate_input_use = calc_lut_input_use(logits) * output_use
    wire_use_mask = gate_input_use.any(-1)
    used_wires = wires[wire_use_mask]
    prev_gate_use = np.zeros(input_n, np.bool_)
    prev_gate_use[used_wires] = True
    return prev_gate_use, wire_use_mask


def calc_gate_use_masks(input_n, wires, logits):
    layer_sizes = [input_n] + [np.prod(l.shape[:2]) for l in logits]
    gate_use_mask = np.ones(layer_sizes[-1], np.bool_)
    gate_masks = [gate_use_mask]
    wire_masks = []
    for i in range(len(logits))[::-1]:
        gate_use_mask, wire_use_mask = propatate_gate_use(
            layer_sizes[i], wires[i], logits[i], gate_use_mask
        )
        wire_masks.append(wire_use_mask)
        gate_masks.append(gate_use_mask)
    return gate_masks[::-1], wire_masks[::-1]


######################## interactive demo ##############################


def zoom(a, k=2):
    return np.repeat(np.repeat(a, k, 1), k, 0)


def unpack(x, bit_n=8):
    return (x[..., None] >> np.r_[:bit_n]) & 1


def is_point_in_box(p0, p1, p):
    (x0, y0), (x1, y1), (x, y) = p0, p1, p
    return (x0 <= x <= x1) and (y0 <= y <= y1)


# Helper function to create an ImGui-compatible texture from numpy array
def create_texture_from_array(img_array):
    if img_array.shape[2] == 3:  # RGB needs to be converted to RGBA
        h, w, _ = img_array.shape
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        rgba[:, :, 0:3] = img_array
        rgba[:, :, 3] = 255
        img_array = rgba

    # Make a flat RGBA array
    flat_img = img_array.ravel().view(np.uint8)

    # Create texture ID using ImGui
    texture_id = imgui.get_io().fonts.tex_id

    return texture_id, img_array.shape[1], img_array.shape[0]


max_trainstep_n = 1000


class Demo:
    def __init__(self):
        # Circuit configuration
        self.input_n = 4
        self.output_n = 4
        self.arity = 2
        self.layer_n = 2

        # Update case_n based on input_n
        self.case_n = 1 << self.input_n

        # Generate initial layer sizes
        self.layer_sizes = generate_layer_sizes(
            self.input_n, self.output_n, self.arity, self.layer_n
        )

        # Circuit initialization
        self.logits0 = []
        for gate_n, group_size in self.layer_sizes[1:]:
            self.logits0.append(make_nops(gate_n, self.arity, group_size))
        self.logits = self.logits0
        print("param_n:", sum(l.size for l in self.logits0))

        # Wiring configuration
        self.wires_key = jax.random.PRNGKey(42)
        self.local_noise = 0.0
        self.shuffle_wires()
        self.reset_gate_mask()

        # Input data
        x = jp.arange(self.case_n)
        self.input_x = unpack(x, bit_n=self.input_n)

        # Create a proper image format for display - convert to 3-channel uint8
        inp_img = self.input_x.T

        # Ensure consistent visualization with different input sizes
        # Use a more moderate zoom factor to avoid overly large displays
        zoom_factor = max(4, int(8 // self.input_n * 2))  # Reduced scaling factor
        inp_img = np.dstack([inp_img] * 3)  # Convert to 3-channel
        inp_img = zoom(inp_img, zoom_factor)
        self.inputs_img = np.uint8(inp_img.clip(0, 1) * 255)  # Convert to uint8

        print(
            f"Input image shape: {self.inputs_img.shape}, dtype: {self.inputs_img.dtype}, zoom_factor: {zoom_factor}"
        )
        self.active_case_i = 123

        # Display textures
        self.input_texture = None
        self.output_texture = None
        self.ground_truth_texture = None
        self.ground_truth_img = None

        # Visualization settings
        self.use_simple_viz = False

        # Plot settings
        self.max_loss_value = 10.0  # Maximum loss value to display
        self.min_loss_value = 1e-6  # Minimum loss value (for log scale)
        self.auto_scale_plot = True

        # Task definitions using the tasks module
        self.available_tasks = list(tasks.TASKS.keys()) + ["text", "noise"]
        self.task_idx = (
            self.available_tasks.index("binary_multiply")
            if "binary_multiply" in self.available_tasks
            else 0
        )
        self.task_text = "All you need are ones and zeros and backpropagation"
        self.noise_p = 0.5
        self.sample_noise()
        self.update_task()

        # Training setup
        self.wd_log10 = -1
        self.loss_log = np.zeros(max_trainstep_n, np.float32)
        self.hard_log = np.zeros(max_trainstep_n, np.float32)
        self.trainstep_i = 0
        self.is_training = True

        # Optimization method (backprop/GNN)
        self.optimization_methods = ["Backprop", "GNN", "Self-Attention"]
        self.optimization_method_idx = 0  # Default to Backprop

        # GNN parameters (used if GNN is selected)
        self.gnn = None
        self.gnn_hidden_dim = 16
        self.gnn_message_steps = 1  # Lower default for faster UI
        self.gnn_node_mlp_features = [64, 32]
        self.gnn_edge_mlp_features = [64, 32]
        self.gnn_enable_message_passing = True

        # Self-Attention parameters
        self.sa_model = None
        self.sa_hidden_dim = 16
        self.sa_message_steps = 1  # Lower default for faster UI
        self.sa_num_heads = 4
        self.sa_num_layers = 3
        self.sa_mlp_dim = 64
        self.sa_dropout_rate = 0.0

        # WandB loading parameters
        self.run_id = None  # Specific run ID if provided
        self.loaded_run_id = None  # Last successfully loaded run ID
        self.wandb_entity = "m2snn"
        self.wandb_project = "boolean-nca-cc"
        self.wandb_download_dir = "saves"

        self.loaded_gnn_state = None  # Variable to store loaded state
        self.loaded_sa_state = None  # Variable to store loaded self-attention state

        # Initialize optimizers (Backprop optimizer first)
        self.init_optimizers()

    def create_filter_from_params(self):
        """Create filter parameters for wandb query based on current demo settings"""
        current_task = self.available_tasks[self.task_idx]
        current_opt_method = self.optimization_methods[self.optimization_method_idx]

        # Set model type based on optimization method
        if current_opt_method == "GNN":
            model_type = "gnn"
        elif current_opt_method == "Self-Attention":
            model_type = "self_attention"
        else:
            model_type = "backprop"

        # Create filter dictionary
        filters = {
            "config.circuit.input_bits": self.input_n,
            "config.circuit.output_bits": self.output_n,
            "config.circuit.arity": self.arity,
            "config.circuit.n_layers": self.layer_n,
            "config.model.type": model_type,
        }

        # Add task to filter if it's a standard task (not text or noise)
        if current_task not in ["text", "noise"]:
            filters["config.dataset.task"] = current_task

        # Add additional model-specific parameters if available
        if current_opt_method == "GNN":
            filters["config.model.hidden_dim"] = self.gnn_hidden_dim
        elif current_opt_method == "Self-Attention":
            filters["config.model.hidden_dim"] = self.sa_hidden_dim
            filters["config.model.num_heads"] = self.sa_num_heads

        return filters

    def check_existing_model(self, run_id=None):
        """Check if a model is already downloaded for this run_id"""
        if run_id:
            # Check if we already have this specific run downloaded
            model_dir = os.path.join(self.wandb_download_dir, f"run_{run_id}")
            if os.path.exists(model_dir):
                print(f"Found existing downloaded model for run {run_id}")
                # Look for model files in this directory
                for root, dirs, files in os.walk(model_dir):
                    for file in files:
                        if file.endswith(".pkl") and "best_model" in file:
                            model_path = os.path.join(root, file)
                            print(f"Using existing model file: {model_path}")
                            return model_path

        # No existing model found
        return None

    def load_best_model(self):
        """Load the best model from wandb based on current parameters"""
        try:
            filters = None
            if self.run_id:
                # Check if we already have this model downloaded
                existing_model = self.check_existing_model(self.run_id)
                if existing_model and self.run_id == self.loaded_run_id:
                    print(f"Model for run {self.run_id} is already loaded")
                    return True

                print(f"Loading model from specific run ID: {self.run_id}")
            else:
                # Create filters from current parameters
                filters = self.create_filter_from_params()
                print(f"Loading best model with filters: {filters}")

                # Check if we already have a model with these filters
                # This is more complex and would require storing filter-to-run mappings

            # Load the model
            method_name = self.optimization_methods[self.optimization_method_idx]
            print(f"Loading {method_name} model...")

            try:
                model, loaded_dict = load_best_model_from_wandb(
                    run_id=self.run_id,
                    filters=filters,
                    seed=42,  # Use a consistent seed
                    project=self.wandb_project,
                    entity=self.wandb_entity,
                    download_dir=self.wandb_download_dir,
                )

                # Store the loaded state based on optimization method
                if method_name == "GNN":
                    # Handle potential API differences by extracting state carefully
                    try:
                        self.loaded_gnn_state = {"gnn": nnx.state(model)}
                    except (AttributeError, TypeError) as e:
                        print(f"Error getting state with nnx.state: {e}")
                        # Try alternative approach
                        if hasattr(model, "_tree"):
                            self.loaded_gnn_state = {"gnn": model._tree}
                        else:
                            # Last resort - store the whole model
                            self.loaded_gnn_state = {"gnn": model}

                    self.gnn = None  # Force reinitialization with new state
                    print("GNN model loaded successfully")
                elif method_name == "Self-Attention":
                    # Handle potential API differences by extracting state carefully
                    try:
                        self.loaded_sa_state = {"sa": nnx.state(model)}
                    except (AttributeError, TypeError) as e:
                        print(f"Error getting state with nnx.state: {e}")
                        # Try alternative approach
                        if hasattr(model, "_tree"):
                            self.loaded_sa_state = {"sa": model._tree}
                        else:
                            # Last resort - store the whole model
                            self.loaded_sa_state = {"sa": model}

                    self.sa_model = None  # Force reinitialization with new state
                    print("Self-Attention model loaded successfully")

                # Store the run ID we just loaded
                if self.run_id:
                    self.loaded_run_id = self.run_id
                elif "run_id" in loaded_dict:
                    self.loaded_run_id = loaded_dict["run_id"]
                elif hasattr(model, "run_id"):
                    self.loaded_run_id = model.run_id

                return True

            except Exception as inner_e:
                print(f"Error in model loading process: {inner_e}")
                import traceback

                print(f"Traceback: {traceback.format_exc()}")
                return False

        except Exception as e:
            print(f"Error loading model from wandb: {e}")
            import traceback

            print(f"Traceback: {traceback.format_exc()}")
            return False

    def init_optimizers(self):
        """Initialize or reinitialize optimization methods"""
        # Backprop optimizer
        self.bp_optimizer = optax.adamw(2.0, 0.8, 0.8, weight_decay=10**self.wd_log10)
        self.opt_state = self.bp_optimizer.init(self.logits)

        # GNN optimizer will be initialized on demand when selected

    def get_opt(self):
        return optax.adamw(2.0, 0.8, 0.8, weight_decay=10**self.wd_log10)

    def reset_gate_mask(self):
        """Reset all gate masks to active"""
        # Ensure we have the right number of masks
        self.gate_mask = [np.ones(gate_n) for gate_n, _ in self.layer_sizes]
        self.wire_masks = [np.ones_like(w, np.bool_) for w in self.wires]
        print(
            f"Reset gate mask: {len(self.gate_mask)} gate masks, {len(self.wire_masks)} wire masks"
        )

    def mask_unused_gates(self):
        gate_masks, self.wire_masks = calc_gate_use_masks(
            self.input_n, self.wires, self.logits
        )
        for i in range(len(gate_masks)):
            self.gate_mask[i] = np.array(self.gate_mask[i] * gate_masks[i])

    def shuffle_wires(self):
        in_n = self.input_n
        self.wires = []
        key = self.wires_key
        for gate_n, group_size in self.layer_sizes[1:]:
            key, k1 = jax.random.split(key)
            local_noise = self.local_noise if self.local_noise > 0.0 else None
            ws = gen_wires_with_noise(
                k1, in_n, gate_n, self.arity, group_size, local_noise
            )
            self.wires.append(ws)
            in_n = gate_n

        # Reset GNN when wires change
        self.gnn = None

    def sample_noise(self):
        self.noise = np.random.rand(self.case_n, self.input_n)

    def update_task(self):
        """Update the current task using the tasks module where possible"""
        task_name = self.available_tasks[self.task_idx]

        if task_name == "text":
            # Text-based task (special case)
            im = PIL.Image.new("L", (self.case_n, self.output_n))
            draw = PIL.ImageDraw.Draw(im)
            draw.text((2, -2), self.task_text, fill=255)
            self.y0 = jp.float32(np.array(im) > 100).T

            # Default input is just counting
            x = jp.arange(self.case_n)
            self.input_x = unpack(x, bit_n=self.input_n)

        elif task_name == "noise":
            # Noise-based task (special case)
            if self.noise.shape != (self.case_n, self.input_n):
                self.sample_noise()
            self.y0 = jp.float32(self.noise < self.noise_p)

            # Default input is just counting
            x = jp.arange(self.case_n)
            self.input_x = unpack(x, bit_n=self.input_n)

        else:
            # Use the tasks module for standard tasks
            try:
                # Get both input and expected output from the task
                self.input_x, self.y0 = tasks.get_task_data(
                    task_name,
                    self.case_n,
                    input_bits=self.input_n,
                    output_bits=self.output_n,
                )

                # Update input visualization
                self.update_input_visualization()

            except Exception as e:
                print(f"Error loading task '{task_name}': {e}")
                # Fallback to simple copy task
                x = jp.arange(self.case_n)
                self.input_x = unpack(x, bit_n=self.input_n)
                max_output_value = (1 << self.output_n) - 1
                clipped_output = np.minimum(x, max_output_value)
                self.y0 = jp.float32(unpack(clipped_output, bit_n=self.output_n))

                # Update input visualization
                self.update_input_visualization()

        # Reset training progress when task changes
        self.trainstep_i = 0
        self.loss_log = np.zeros(max_trainstep_n, np.float32)
        self.hard_log = np.zeros(max_trainstep_n, np.float32)

        # Update ground truth visualization whenever task changes
        self.update_ground_truth_visualization()

    def update_input_visualization(self):
        """Update input visualization based on current input_x"""
        inp_img = self.input_x.T
        zoom_factor = max(4, int(8 // self.input_n * 2))  # Reduced scaling factor
        inp_img = np.dstack([inp_img] * 3)  # Convert to 3-channel
        inp_img = zoom(inp_img, zoom_factor)
        self.inputs_img = np.uint8(inp_img.clip(0, 1) * 255)  # Convert to uint8

    def initialize_gnn(self):
        """Initialize GNN using model loaded from WandB"""
        if self.gnn is None:
            print("Initializing GNN...")
            key = jax.random.PRNGKey(42)  # Fixed seed for reproducibility

            try:
                # Try to load model from WandB if not already loaded
                if not self.loaded_gnn_state:
                    print("Attempting to load GNN model from WandB...")
                    if self.load_best_model():
                        print("Successfully loaded GNN model from WandB")
                    else:
                        print(
                            "Failed to load GNN model from WandB, creating new instance"
                        )

                # Create GNN instance (whether we have loaded state or not)
                self.gnn = CircuitGNN(
                    node_mlp_features=self.gnn_node_mlp_features,
                    edge_mlp_features=self.gnn_edge_mlp_features,
                    hidden_dim=self.gnn_hidden_dim,
                    arity=self.arity,
                    message_passing=self.gnn_enable_message_passing,
                    rngs=nnx.Rngs(params=key),
                )

                # Update with loaded state if available
                if self.loaded_gnn_state:
                    print("Applying loaded GNN state...")
                    try:
                        # Try standard update first
                        nnx.update(self.gnn, self.loaded_gnn_state["gnn"])
                    except (AttributeError, TypeError) as e:
                        print(f"Standard update failed: {e}")
                        print("Trying alternative update approach...")

                        # Alternative: Try to access the state differently
                        gnn_state = self.loaded_gnn_state["gnn"]

                        # If it's a full model, get its tree/state
                        if hasattr(gnn_state, "_tree"):
                            gnn_state = gnn_state._tree

                        # Try to update with the extracted state
                        try:
                            # Try using put for each parameter
                            for collection_name, collection in gnn_state.items():
                                for var_name, value in collection.items():
                                    try:
                                        path = f"{collection_name}.{var_name}"
                                        self.gnn.put(path, value)
                                    except Exception as inner_e:
                                        print(
                                            f"Warning: Failed to update {path}: {inner_e}"
                                        )
                        except Exception as update_e:
                            print(f"Alternative update also failed: {update_e}")
                            print("Will proceed with uninitialized model")

                    print("GNN initialized with loaded state")
                else:
                    print("GNN initialized with default parameters")

                return True

            except Exception as e:
                print(f"Error initializing GNN: {e}")
                import traceback

                print(f"Traceback: {traceback.format_exc()}")
                # Clear the loaded state to avoid repeated failures
                self.loaded_gnn_state = None
                # Fallback to backprop
                self.optimization_method_idx = 0
                print("Falling back to Backprop optimization")
                return False

    def update_circuit_backprop(self):
        """Update circuit using backpropagation"""
        try:
            (loss, aux), grad = grad_loss_f_gui(
                self.logits, self.wires, self.gate_mask, self.input_x, self.y0
            )
            if self.is_training:
                upd, self.opt_state = self.bp_optimizer.update(
                    grad, self.opt_state, self.logits
                )
                self.logits = optax.apply_updates(self.logits, upd)

            self.act = aux["act"]
            self.err_mask = aux["err_mask"]  # Store error mask for visualization
            return loss, aux["hard_loss"]
        except Exception as e:
            print(f"Error in backprop update: {e}")
            # Return fallback values to avoid crashing
            self.act = [np.zeros((self.case_n, size)) for size, _ in self.layer_sizes]
            self.err_mask = np.ones((self.case_n, self.output_n), bool)
            return self.max_loss_value, self.max_loss_value

    def update_circuit_gnn(self):
        """Update circuit using GNN"""
        try:
            # Ensure GNN is initialized
            self.initialize_gnn()

            # Run GNN for specified number of steps
            try:
                # Store original logit shapes for reconstruction
                logits_original_shapes = [logit.shape for logit in self.logits]

                # Create graph from current circuit
                circuit_graph = build_graph(
                    self.logits,
                    self.wires,
                    self.input_n,
                    self.arity,
                    self.gnn_hidden_dim,
                )

                # Run GNN for specified number of steps
                updated_graph = run_gnn_scan(
                    self.gnn, circuit_graph, self.gnn_message_steps
                )

                # Extract updated logits if training is enabled
                if self.is_training:
                    self.logits = extract_logits_from_graph(
                        updated_graph, logits_original_shapes
                    )

                # Run circuit with updated logits
                self.act = run_circuit_gui(
                    self.logits, self.wires, self.gate_mask, self.input_x
                )
                hard_act = run_circuit_gui(
                    self.logits, self.wires, self.gate_mask, self.input_x, hard=True
                )

                # Generate error mask for visualization
                self.err_mask = hard_act[-1] != self.y0

                # Calculate loss
                loss = res2loss(self.act[-1] - self.y0)
                hard_loss = res2loss(hard_act[-1] - self.y0)

                return loss, hard_loss

            except Exception as e:
                print(f"Error in GNN update: {e}")
                # Fallback to backprop
                self.optimization_method_idx = 0
                return self.update_circuit_backprop()

        except Exception as e:
            print(f"Critical error in GNN update: {e}")
            import traceback

            print(f"Traceback: {traceback.format_exc()}")

            # Return fallback values to avoid crashing
            self.act = [np.zeros((self.case_n, size)) for size, _ in self.layer_sizes]
            self.err_mask = np.ones((self.case_n, self.output_n), bool)
            return self.max_loss_value, self.max_loss_value

    def update_circuit_self_attention(self):
        """Update circuit using Self-Attention"""
        try:
            # Ensure Self-Attention model is initialized
            self.initialize_self_attention()

            # Run Self-Attention for specified number of steps
            try:
                # Store original logit shapes for reconstruction
                logits_original_shapes = [logit.shape for logit in self.logits]

                # Create graph from current circuit - same as for GNN
                circuit_graph = build_graph(
                    self.logits,
                    self.wires,
                    self.input_n,
                    self.arity,
                    self.sa_hidden_dim,  # Use self-attention hidden dim
                )

                # Run Self-Attention for specified number of steps
                updated_graph = run_self_attention_scan(
                    self.sa_model, circuit_graph, self.sa_message_steps
                )

                # Extract updated logits if training is enabled
                if self.is_training:
                    self.logits = extract_logits_from_graph(
                        updated_graph, logits_original_shapes
                    )

                # Run circuit with updated logits
                self.act = run_circuit_gui(
                    self.logits, self.wires, self.gate_mask, self.input_x
                )
                hard_act = run_circuit_gui(
                    self.logits, self.wires, self.gate_mask, self.input_x, hard=True
                )

                # Generate error mask for visualization
                self.err_mask = hard_act[-1] != self.y0

                # Calculate loss
                loss = res2loss(self.act[-1] - self.y0)
                hard_loss = res2loss(hard_act[-1] - self.y0)

                return loss, hard_loss

            except Exception as e:
                print(f"Error in Self-Attention update: {e}")
                import traceback

                print(f"Traceback: {traceback.format_exc()}")
                # Fallback to backprop
                self.optimization_method_idx = 0
                return self.update_circuit_backprop()

        except Exception as e:
            print(f"Critical error in Self-Attention update: {e}")
            import traceback

            print(f"Traceback: {traceback.format_exc()}")

            # Return fallback values to avoid crashing
            self.act = [np.zeros((self.case_n, size)) for size, _ in self.layer_sizes]
            self.err_mask = np.ones((self.case_n, self.output_n), bool)
            return self.max_loss_value, self.max_loss_value

    def update_circuit(self):
        """Update circuit using selected optimization method"""
        try:
            # Choose optimization method
            if self.optimization_methods[self.optimization_method_idx] == "Backprop":
                loss, hard_loss = self.update_circuit_backprop()
            elif self.optimization_methods[self.optimization_method_idx] == "GNN":
                loss, hard_loss = self.update_circuit_gnn()
            else:  # Self-Attention
                loss, hard_loss = self.update_circuit_self_attention()

            # Extract output activations and visualize
            oimg = self.act[-1].T
            oimg = np.dstack([oimg] * 3)

            # Apply error mask for visualization (both methods set self.err_mask)
            m = self.err_mask.T[..., None] * 0.5
            oimg = oimg * (1.0 - m) + m * np.float32([1, 0, 0])

            # Use same zoom factor as for input, but more moderate
            zoom_factor = max(4, int(8 // self.output_n * 2))  # Reduced scaling factor
            oimg = zoom(oimg, zoom_factor)
            self.outputs_img = np.uint8(oimg.clip(0, 1) * 255)

            # Create ground truth visualization
            self.update_ground_truth_visualization()

            # Update textures for display
            self.update_textures()

            # Ensure loss values are valid and bounded
            loss_value = float(loss)
            hard_loss_value = float(hard_loss)

            # Check for NaN or infinity
            if np.isnan(loss_value) or np.isinf(loss_value):
                loss_value = self.max_loss_value
                print(
                    f"Warning: Invalid loss value detected, clamping to {self.max_loss_value}"
                )

            if np.isnan(hard_loss_value) or np.isinf(hard_loss_value):
                hard_loss_value = self.max_loss_value
                print(
                    f"Warning: Invalid hard_loss value detected, clamping to {self.max_loss_value}"
                )

            # Update max loss value if auto-scaling is enabled
            if self.auto_scale_plot:
                if loss_value > self.max_loss_value:
                    self.max_loss_value = min(
                        loss_value * 1.5, 1e6
                    )  # Reasonable upper bound

                if hard_loss_value > self.max_loss_value:
                    self.max_loss_value = min(hard_loss_value * 1.5, 1e6)

            # Store the bounded values
            i = self.trainstep_i % len(self.loss_log)
            self.loss_log[i] = max(
                min(loss_value, self.max_loss_value), self.min_loss_value
            )
            self.hard_log[i] = max(
                min(hard_loss_value, self.max_loss_value), self.min_loss_value
            )

            if self.is_training:
                self.trainstep_i += 1

        except Exception as e:
            print(f"Error in update_circuit: {e}")
            import traceback

            print(f"Traceback: {traceback.format_exc()}")

            # Try to recover by regenerating the circuit completely
            print("Attempting to recover by regenerating circuit...")
            try:
                self.regenerate_circuit()
            except Exception as recovery_error:
                print(f"Failed to recover: {recovery_error}")

    def update_ground_truth_visualization(self):
        """Create visualization for ground truth (expected output)"""
        # Create ground truth visualization
        gt_img = self.y0.T
        zoom_factor = max(4, int(8 // self.output_n * 2))  # Same as for output
        gt_img = np.dstack([gt_img] * 3)  # Convert to 3-channel
        gt_img = zoom(gt_img, zoom_factor)
        self.ground_truth_img = np.uint8(gt_img.clip(0, 1) * 255)

    def update_textures(self):
        # We'll create textures each frame for simplicity
        # In a real application, you might want to cache these
        dummy_texture = imgui.get_io().fonts.tex_id
        self.input_texture = (
            dummy_texture,
            self.inputs_img.shape[1],
            self.inputs_img.shape[0],
        )
        self.output_texture = (
            dummy_texture,
            self.outputs_img.shape[1],
            self.outputs_img.shape[0],
        )
        self.ground_truth_texture = (
            dummy_texture,
            self.ground_truth_img.shape[1],
            self.ground_truth_img.shape[0],
        )

    def draw_gate_lut(self, x, y, logit):
        x0, y0 = x - 20, y - 20 - 36
        dl = imgui.get_window_draw_list()
        lut = jax.nn.sigmoid(logit).reshape(-1, 4)
        col = np.uint32(lut * 255)
        col = (col << 16) | (col << 8) | col | 0xFF000000
        for (i, j), c in np.ndenumerate(col):
            x, y = x0 + j * 10, y0 + i * 10
            dl.add_rect_filled((x, y), (x + 10, y + 10), c)

    def draw_circuit(self, pad=4, d=24, H=600):
        io = imgui.get_io()
        W = imgui.get_content_region_avail().x - pad * 2
        imgui.invisible_button("circuit", (W, H))
        base_x, base_y = imgui.get_item_rect_min()
        base_x += pad

        dl = imgui.get_window_draw_list()
        h = (H - d) / (len(self.layer_sizes) - 1)
        prev_gate_x = None
        prev_y = 0
        prev_act = None
        case = self.active_case_i
        hover_gate = None

        # Ensure activations exist
        if not hasattr(self, "act") or len(self.act) != len(self.layer_sizes):
            print(
                "Warning: Activations not initialized properly, creating empty activations"
            )
            self.act = [np.zeros((self.case_n, size)) for size, _ in self.layer_sizes]

        # Ensure wire_masks has the right length
        if len(self.wire_masks) != len(self.wires):
            print(
                f"Warning: wire_masks length mismatch. Expected {len(self.wires)}, got {len(self.wire_masks)}"
            )
            self.wire_masks = [np.ones_like(w, np.bool_) for w in self.wires]

        for li, (gate_n, group_size) in enumerate(self.layer_sizes):
            group_n = gate_n // group_size
            span_x = W / group_n
            group_w = min(d * group_size, span_x - 6)
            gate_w = group_w / group_size
            group_x = base_x + (np.arange(group_n)[:, None] + 0.5) * span_x
            gate_ofs = (np.arange(group_size) - group_size / 2 + 0.5) * gate_w
            gate_x = (group_x + gate_ofs).ravel()
            y = base_y + li * h + d / 2

            # Ensure we don't go out of bounds on activations
            if li < len(self.act):
                act = np.array(self.act[li][case])
            else:
                print(f"Warning: Missing activation for layer {li}")
                act = np.zeros(gate_n)

            for i, x in enumerate(gate_x):
                a = int(act[i] * 0xA0) if i < len(act) else 0
                col = 0xFF202020 + (a << 8)
                p0, p1 = (x - gate_w / 2, y - d / 2), (x + gate_w / 2, y + d / 2)
                dl.add_rect_filled(p0, p1, col, 4)
                if is_point_in_box(p0, p1, io.mouse_pos):
                    dl.add_rect(p0, p1, 0xA00000FF, 4, thickness=2.0)
                    if li > 0:
                        hover_gate = (
                            x,
                            y,
                            self.logits[li - 1][i // group_size, i % group_size],
                        )
                    if io.mouse_clicked[0]:
                        if li > 0:
                            self.gate_mask[li][i] = 1.0 - self.gate_mask[li][i]
                        else:
                            self.active_case_i = self.active_case_i ^ (1 << i)
                if (
                    li < len(self.gate_mask)
                    and i < len(self.gate_mask[li])
                    and self.gate_mask[li][i] == 0.0
                ):
                    dl.add_rect_filled(p0, p1, 0xA00000FF, 4)

            for x in group_x[:, 0]:
                dl.add_rect(
                    (x - group_w / 2, y - d / 2),
                    (x + group_w / 2, y + d / 2),
                    0x80FFFFFF,
                    4,
                )

            if (
                li > 0
                and prev_gate_x is not None
                and li - 1 < len(self.wires)
                and li - 1 < len(self.wire_masks)
            ):
                wires = self.wires[li - 1].T
                masks = self.wire_masks[li - 1].T
                src_x = prev_gate_x[wires]
                dst_x = (
                    group_x
                    + (np.arange(self.arity) + 0.5) / self.arity * group_w
                    - group_w / 2
                )
                my = (prev_y + y) / 2
                for x0, x1, si, m in zip(
                    src_x.ravel(), dst_x.ravel(), wires.ravel(), masks.ravel()
                ):
                    if not m:
                        continue
                    a = int(prev_act[si] * 0x60) if si < len(prev_act) else 0
                    col = 0xFF404040 + (a << 8)
                    dl.add_bezier_cubic(
                        (x0, prev_y + d / 2),
                        (x0, my),
                        (x1, my),
                        (x1, y - d / 2),
                        col,
                        1.0,
                    )
            if hover_gate is not None:
                self.draw_gate_lut(*hover_gate)

            prev_gate_x = gate_x
            prev_act = act
            prev_y = y

    def draw_lut(self, name, img, tex_id):
        try:
            view_w = imgui.get_content_region_avail().x
            img_h, img_w = img.shape[:2]

            # Draw the image directly using ImGui
            texture_id, width, height = tex_id

            # Create a simple colored rectangle instead of using texture
            dl = imgui.get_window_draw_list()
            p0 = imgui.get_cursor_screen_pos()

            # Calculate proper aspect ratio based on the actual input/output data dimensions
            if name == "inputs":
                # For inputs, maintain the natural bit_n / 2^bit_n ratio
                # This accounts for the fact that width scales exponentially (2^n) while height scales linearly (n)
                natural_aspect = self.input_n / (2**self.input_n)
                reference_aspect = 8.0 / (2**8)  # Aspect ratio of 8-bit display
                # Scale to be similar to 8-bit display but with correct proportions
                aspect = natural_aspect * (reference_aspect / natural_aspect) * 3.0
            else:  # outputs
                # For outputs, do the same calculation
                natural_aspect = self.output_n / (2**self.output_n)
                reference_aspect = 8.0 / (2**8)  # Aspect ratio of 8-bit display
                aspect = natural_aspect * (reference_aspect / natural_aspect) * 3.0

            # Maintain proper aspect ratio within reasonable bounds
            min_aspect = 0.02  # Minimum height to width ratio
            max_aspect = 0.3  # Maximum height to width ratio
            aspect = max(min_aspect, min(aspect, max_aspect))

            disp_w = view_w
            disp_h = disp_w * aspect
            p1 = (p0[0] + disp_w, p0[1] + disp_h)

            # Draw a background
            dl.add_rect_filled(p0, p1, 0xFF333333, 4.0)

            if self.use_simple_viz:
                # Very simple visualization - just show active case
                case_width = disp_w / self.case_n
                for i in range(self.case_n):
                    # Draw a simple line for each case
                    x_pos = p0[0] + i * case_width
                    is_active = i == self.active_case_i

                    # Sample color from the middle row of the image
                    middle_y = img_h // 2
                    if len(img.shape) == 3 and img.shape[2] >= 3:
                        r, g, b = [int(v) for v in img[middle_y, i % img_w, 0:3]]
                        # Make sure values are in range 0-255
                        r, g, b = r & 0xFF, g & 0xFF, b & 0xFF
                        color = 0xFF000000 | (b << 16) | (g << 8) | r
                    else:
                        # Grayscale
                        v = int(img[middle_y, i % img_w]) & 0xFF
                        color = 0xFF000000 | (v << 16) | (v << 8) | v

                    # Draw a vertical line for this case
                    dl.add_line(
                        (x_pos, p0[1]), (x_pos, p1[1]), color, 2.0 if is_active else 1.0
                    )

                    # Highlight active case
                    if is_active:
                        dl.add_rect(
                            (x_pos - case_width / 2, p0[1]),
                            (x_pos + case_width / 2, p1[1]),
                            0x8000FF00,
                            0.0,
                            thickness=2.0,
                        )
            else:
                # Calculate block size
                block_w = disp_w / img_w
                block_h = disp_h / img_h

                # Only draw rectangles at a reasonable resolution to avoid performance issues
                # If the image is too large, we'll sample it
                max_display_blocks = (
                    64  # Maximum number of blocks to display for performance
                )

                if img_w > max_display_blocks or img_h > max_display_blocks:
                    # Sample the image to reduce drawing complexity
                    x_step = max(1, img_w // max_display_blocks)
                    y_step = max(1, img_h // max_display_blocks)

                    for y in range(0, img_h, y_step):
                        for x in range(0, img_w, x_step):
                            # Calculate display rectangle
                            px = p0[0] + (x / img_w) * disp_w
                            py = p0[1] + (y / img_h) * disp_h
                            px_end = p0[0] + ((x + x_step) / img_w) * disp_w
                            py_end = p0[1] + ((y + y_step) / img_h) * disp_h

                            # Get color from image and convert to ImGui color format (RGBA)
                            if len(img.shape) == 3 and img.shape[2] >= 3:
                                r, g, b = [int(v) for v in img[y, x, 0:3]]
                                # Make sure values are in range 0-255
                                r, g, b = r & 0xFF, g & 0xFF, b & 0xFF
                                color = 0xFF000000 | (b << 16) | (g << 8) | r
                            else:
                                # Grayscale
                                v = int(img[y, x]) & 0xFF  # Ensure it's in range 0-255
                                color = 0xFF000000 | (v << 16) | (v << 8) | v

                            dl.add_rect_filled((px, py), (px_end, py_end), color)
                else:
                    # Draw at full resolution
                    for y in range(img_h):
                        for x in range(img_w):
                            px = p0[0] + x * block_w
                            py = p0[1] + y * block_h

                            # Get color from image and convert to ImGui color format (RGBA)
                            if len(img.shape) == 3 and img.shape[2] >= 3:
                                r, g, b = [int(v) for v in img[y, x, 0:3]]
                                # Make sure values are in range 0-255
                                r, g, b = r & 0xFF, g & 0xFF, b & 0xFF
                                color = 0xFF000000 | (b << 16) | (g << 8) | r
                            else:
                                # Grayscale
                                v = int(img[y, x]) & 0xFF  # Ensure it's in range 0-255
                                color = 0xFF000000 | (v << 16) | (v << 8) | v

                            dl.add_rect_filled(
                                (px, py), (px + block_w, py + block_h), color
                            )

            # Add cursor showing active case
            x = p0[0] + (disp_w * (self.active_case_i + 0.5) / self.case_n)
            dl.add_line((x, p0[1]), (x, p1[1]), 0x8000FF00, 2.0)

            # Add border
            dl.add_rect(p0, p1, 0xFFFFFFFF, 4.0)

            # Make area clickable to change active case
            imgui.invisible_button(f"{name}_area", (disp_w, disp_h))

            if imgui.is_item_hovered() and imgui.is_mouse_clicked(0):
                mx = imgui.get_io().mouse_pos.x - p0[0]
                mx_ratio = mx / disp_w
                self.active_case_i = int(mx_ratio * self.case_n)
                if self.active_case_i < 0:
                    self.active_case_i = 0
                if self.active_case_i >= self.case_n:
                    self.active_case_i = self.case_n - 1

            # Skip some space
            imgui.dummy((0, disp_h))

        except Exception as e:
            imgui.text(f"Error drawing {name}: {e}")
            import traceback

            print(f"Error in draw_lut: {traceback.format_exc()}")

    def regenerate_circuit(self):
        """Completely regenerate the circuit with current parameters"""
        print(
            f"Regenerating circuit: input_n={self.input_n}, output_n={self.output_n}, arity={self.arity}, layer_n={self.layer_n}"
        )

        # Update derived values
        self.case_n = 1 << self.input_n

        # Ensure active case is valid for new case_n
        self.active_case_i = min(self.active_case_i, self.case_n - 1)

        # Generate new layer sizes
        self.layer_sizes = generate_layer_sizes(
            self.input_n, self.output_n, self.arity, self.layer_n
        )
        print(f"New layer sizes: {self.layer_sizes}")

        # Reinitialize circuit
        self.logits0 = []
        for gate_n, group_size in self.layer_sizes[1:]:
            self.logits0.append(make_nops(gate_n, self.arity, group_size))
        self.logits = self.logits0

        # Reset gate masks
        self.reset_gate_mask()

        # Regenerate wires with new key to ensure fresh connections
        self.wires_key = jax.random.PRNGKey(42)  # Fixed seed for reproducibility
        self.shuffle_wires()

        # Initialize empty activations to ensure proper dimensions
        self.act = [np.zeros((self.case_n, size)) for size, _ in self.layer_sizes]

        # Update input/output data
        x = jp.arange(self.case_n)
        self.input_x = unpack(x, bit_n=self.input_n)
        self.update_input_visualization()

        # Reset noise if needed
        self.sample_noise()

        # Reset optimizer state
        self.init_optimizers()

        # Reset GNN if it was being used
        self.gnn = None

        # Reset training progress
        self.trainstep_i = 0
        self.loss_log = np.zeros(max_trainstep_n, np.float32)
        self.hard_log = np.zeros(max_trainstep_n, np.float32)

        # Create an empty error mask of appropriate size
        self.err_mask = np.zeros((self.case_n, self.output_n), bool)

        # Update task for new input/output dimensions
        self.update_task()

        # Run circuit once to initialize activations
        try:
            self.update_circuit()
        except Exception as e:
            print(f"Warning: Initial update_circuit failed: {e}")
            print("Continuing with default activations")

        print(
            f"Circuit regenerated with {sum(l.size for l in self.logits0)} parameters"
        )

    def gui(self):
        try:
            self.update_circuit()
            runner_params = hello_imgui.get_runner_params()
            runner_params.fps_idling.enable_idling = True
            io = imgui.get_io()

            imgui.begin_child("main", (-200, 0))

            if implot.begin_plot("Train logs", (-1, 200)):
                implot.setup_legend(implot.Location_.north_east.value)
                implot.setup_axis_scale(
                    implot.ImAxis_.y1.value, implot.Scale_.log10.value
                )

                # Set axis limits
                implot.setup_axes(
                    "Step",
                    "Loss",
                    implot.AxisFlags_.auto_fit.value,
                    implot.AxisFlags_.auto_fit.value,
                )
                implot.setup_axis_limits(
                    implot.ImAxis_.y1.value, self.min_loss_value, self.max_loss_value
                )

                # Plot bounded loss values
                implot.plot_line("loss", self.loss_log)
                implot.plot_line("hard_loss", self.hard_log)

                # Show current position
                implot.drag_line_x(
                    1, self.trainstep_i % len(self.loss_log), (0.8, 0, 0, 0.5)
                )
                implot.end_plot()

            imgui.separator_text("Inputs")
            self.draw_lut("inputs", self.inputs_img, self.input_texture)

            H = (
                imgui.get_content_region_avail().y - 320
            )  # Adjust height to make room for both output and ground truth
            self.draw_circuit(H=H)

            imgui.separator_text("Outputs vs Ground Truth")

            # Create columns for side-by-side display
            imgui.columns(2, "output_columns")

            # Output column
            imgui.text("Current Output")
            self.draw_lut("outputs", self.outputs_img, self.output_texture)

            # Ground truth column
            imgui.next_column()
            imgui.text("Expected Output (Ground Truth)")
            if hasattr(self, "ground_truth_img") and self.ground_truth_img is not None:
                self.draw_lut(
                    "ground_truth", self.ground_truth_img, self.ground_truth_texture
                )

            # End columns
            imgui.columns(1)

            imgui.end_child()
            imgui.same_line()

            imgui.begin_child("controls")

            if imgui.button("python REPL"):
                IPython.embed()

            # Visualization mode toggle
            changed, self.use_simple_viz = imgui.checkbox(
                "Simple visualization", self.use_simple_viz
            )
            if changed:
                print(
                    f"Visualization mode: {'Simple' if self.use_simple_viz else 'Detailed'}"
                )

            # Plot settings
            imgui.separator_text("Plot Settings")
            _, self.auto_scale_plot = imgui.checkbox(
                "Auto-scale plot", self.auto_scale_plot
            )

            if not self.auto_scale_plot:
                _, self.max_loss_value = imgui.slider_float(
                    "Max loss",
                    self.max_loss_value,
                    0.001,
                    100.0,
                    "%.3f",
                    imgui.SliderFlags_.logarithmic.value,
                )
                _, self.min_loss_value = imgui.slider_float(
                    "Min loss",
                    self.min_loss_value,
                    1e-6,
                    0.1,
                    "%.6f",
                    imgui.SliderFlags_.logarithmic.value,
                )

                # Ensure min < max
                if self.min_loss_value >= self.max_loss_value:
                    self.min_loss_value = self.max_loss_value / 10.0

            if imgui.button("Reset Plot Bounds"):
                self.max_loss_value = 10.0
                self.min_loss_value = 1e-6
                print("Plot bounds reset to default values")

            imgui.separator_text("Optimization")

            # Optimization method selection
            opt_changed, self.optimization_method_idx = imgui.combo(
                "Method", self.optimization_method_idx, self.optimization_methods
            )
            if opt_changed:
                selected_method = self.optimization_methods[
                    self.optimization_method_idx
                ]
                print(f"Switching to {selected_method} optimization")

                # Auto-load model when switching to GNN or Self-Attention
                if selected_method == "GNN":
                    self.gnn = None  # Force reinitialization
                    self.initialize_gnn()
                elif selected_method == "Self-Attention":
                    self.sa_model = None  # Force reinitialization
                    self.initialize_self_attention()

            # GNN parameters (only shown when GNN is selected)
            if self.optimization_methods[self.optimization_method_idx] == "GNN":
                imgui.text("GNN Parameters:")
                _, self.gnn_message_steps = imgui.slider_int(
                    "Message Steps", self.gnn_message_steps, 1, 20
                )
                _, self.gnn_hidden_dim = imgui.slider_int(
                    "Hidden Dim", self.gnn_hidden_dim, 4, 64
                )
                _, self.gnn_enable_message_passing = imgui.checkbox(
                    "Enable Message Passing", self.gnn_enable_message_passing
                )
                if imgui.button("Reinitialize GNN"):
                    self.gnn = None
                    self.initialize_gnn()
                # Add button to save GNN state
                if imgui.button("Save GNN State"):
                    if self.gnn is not None:
                        try:
                            gnn_state_to_save = {
                                "gnn": nnx.state(self.gnn),
                                # Add optimizer state if needed and available
                                # 'optimizer': self.gnn_optimizer.state_dict() if hasattr(self, 'gnn_optimizer') else None
                            }
                            with open("gnn_results.pkl", "wb") as f:
                                pickle.dump(gnn_state_to_save, f)
                            print("GNN state saved to gnn_results.pkl")
                        except Exception as e:
                            print(f"Error saving GNN state: {e}")
                    else:
                        print("No active GNN model to save.")

            # Self-Attention parameters (only shown when Self-Attention is selected)
            elif (
                self.optimization_methods[self.optimization_method_idx]
                == "Self-Attention"
            ):
                imgui.text("Self-Attention Parameters:")
                _, self.sa_message_steps = imgui.slider_int(
                    "Message Steps", self.sa_message_steps, 1, 20
                )
                _, self.sa_hidden_dim = imgui.slider_int(
                    "Hidden Dim", self.sa_hidden_dim, 4, 64
                )
                _, self.sa_num_heads = imgui.slider_int(
                    "Attention Heads", self.sa_num_heads, 1, 8
                )
                _, self.sa_num_layers = imgui.slider_int(
                    "Attention Layers", self.sa_num_layers, 1, 6
                )
                _, self.sa_mlp_dim = imgui.slider_int(
                    "MLP Dimension", self.sa_mlp_dim, 16, 128
                )
                _, self.sa_dropout_rate = imgui.slider_float(
                    "Dropout Rate", self.sa_dropout_rate, 0.0, 0.5
                )
                if imgui.button("Reinitialize Self-Attention"):
                    self.sa_model = None
                    self.initialize_self_attention()
                # Add button to save Self-Attention state
                if imgui.button("Save Self-Attention State"):
                    if self.sa_model is not None:
                        try:
                            sa_state_to_save = {
                                "sa": nnx.state(self.sa_model),
                                # Add optimizer state if needed
                            }
                            with open("gnn_results.pkl", "wb") as f:
                                pickle.dump(sa_state_to_save, f)
                            print("Self-Attention state saved to gnn_results.pkl")
                        except Exception as e:
                            print(f"Error saving Self-Attention state: {e}")
                    else:
                        print("No active Self-Attention model to save.")

            # Training controls (common to both methods)
            _, self.is_training = imgui.checkbox("is_training", self.is_training)
            if imgui.button("reset gates"):
                self.logits = self.logits0
                self.trainstep_i = 0
            if imgui.button("reset gates + opt"):
                self.logits = self.logits0
                self.init_optimizers()
                self.trainstep_i = 0
                # Reset GNN as well
                self.gnn = None
            if imgui.button("shuffle wires"):
                self.wires_key, key = jax.random.split(self.wires_key)
                self.shuffle_wires()
                self.trainstep_i = 0
            local_noise_changed, self.local_noise = imgui.slider_float(
                "local noise", self.local_noise, 0.0, 20.0
            )
            if local_noise_changed:
                self.shuffle_wires()

            # Weight decay (only affects backprop)
            wd_changed, self.wd_log10 = imgui.slider_float(
                "wd_log10", self.wd_log10, -3, 0.0
            )
            if (
                wd_changed and self.optimization_method_idx == 0
            ):  # Only reinitialize for backprop
                self.init_optimizers()

            # Add WandB integration section
            if self.optimization_methods[self.optimization_method_idx] in [
                "GNN",
                "Self-Attention",
            ]:
                imgui.separator_text("WandB Integration")

                # Run ID input field
                run_id_buffer = self.run_id if self.run_id else ""
                changed, run_id_buffer = imgui.input_text(
                    "Run ID (optional)", run_id_buffer, 256
                )
                if changed:
                    self.run_id = run_id_buffer if run_id_buffer else None
                    # Trigger reinitialization if run ID changes
                    if self.optimization_methods[self.optimization_method_idx] == "GNN":
                        self.gnn = None
                        self.initialize_gnn()
                    else:  # Self-Attention
                        self.sa_model = None
                        self.initialize_self_attention()

                imgui.text_wrapped("Enter a Run ID or leave blank to use filters")

                # Show filter parameters that will be used
                if imgui.tree_node("Filter Parameters"):
                    filters = self.create_filter_from_params()
                    for key, value in filters.items():
                        imgui.text(f"{key}: {value}")
                    imgui.tree_pop()

                # Status of loaded model
                if self.loaded_run_id:
                    imgui.text_colored(
                        f"Loaded model from run: {self.loaded_run_id}",
                        0.0,
                        1.0,
                        0.0,
                        1.0,
                    )

            imgui.separator_text("Masks")
            if imgui.button("reset gate mask"):
                self.reset_gate_mask()
            if imgui.button("mask unused gates"):
                self.mask_unused_gates()
            active_gate_n = int(sum(m.sum() for m in self.gate_mask))
            imgui.text(f"active gate n: {active_gate_n}")

            imgui.separator_text("Task")
            task_changed, self.task_idx = imgui.combo(
                "task", self.task_idx, self.available_tasks
            )
            if task_changed:
                self.update_task()
                self.trainstep_i = 0

            task_name = self.available_tasks[self.task_idx]

            # Display task description
            task_descriptions = {
                "binary_multiply": "Multiply lower and upper halves of input",
                "and": "Bitwise AND between halves of input",
                "xor": "Bitwise XOR between halves of input",
                "add": "Add lower and upper halves of input",
                "parity": "Compute parity (odd/even number of bits)",
                "reverse": "Reverse the bits in the input",
                "text": "Learn a text pattern as binary image",
                "noise": "Learn a random noise pattern",
            }

            # Show task description if available
            if task_name in task_descriptions:
                imgui.text_wrapped(f"Description: {task_descriptions[task_name]}")

            # Show task-specific controls
            if task_name == "text":
                text_changed, self.task_text = imgui.input_text("text", self.task_text)
                if text_changed:
                    self.update_task()
            if task_name == "noise":
                noise_changed, self.noise_p = imgui.slider_float(
                    "p", self.noise_p, 0.0, 1.0
                )
                if noise_changed:
                    self.update_task()

            # Show circuit information
            imgui.spacing()
            actual_params = sum(l.size for l in self.logits0)
            imgui.text(f"Circuit parameters: {actual_params}")

            imgui.separator_text("Circuit Architecture")

            # Add UI controls for circuit architecture parameters
            circuit_changed = False

            # Store original values to detect changes
            orig_input_n = self.input_n
            orig_output_n = self.output_n
            orig_arity = self.arity
            orig_layer_n = self.layer_n

            # Input bits slider
            _, new_input_n = imgui.slider_int("Input Bits", self.input_n, 2, 8)
            if new_input_n != self.input_n:
                self.input_n = new_input_n
                circuit_changed = True

            # Output bits slider
            _, new_output_n = imgui.slider_int("Output Bits", self.output_n, 2, 8)
            if new_output_n != self.output_n:
                self.output_n = new_output_n
                circuit_changed = True

            # Gate arity slider
            _, new_arity = imgui.slider_int("Gate Arity", self.arity, 2, 4)
            if new_arity != self.arity:
                self.arity = new_arity
                circuit_changed = True

            # Hidden layers slider
            _, new_layer_n = imgui.slider_int("Hidden Layers", self.layer_n, 1, 4)
            if new_layer_n != self.layer_n:
                self.layer_n = new_layer_n
                circuit_changed = True

            # If any parameter changed, regenerate the entire circuit
            if circuit_changed:
                # Print what changed for debugging
                changes = []
                if orig_input_n != self.input_n:
                    changes.append(f"input_n: {orig_input_n} → {self.input_n}")
                if orig_output_n != self.output_n:
                    changes.append(f"output_n: {orig_output_n} → {self.output_n}")
                if orig_arity != self.arity:
                    changes.append(f"arity: {orig_arity} → {self.arity}")
                if orig_layer_n != self.layer_n:
                    changes.append(f"layer_n: {orig_layer_n} → {self.layer_n}")

                print(f"Circuit parameters changed: {', '.join(changes)}")

                try:
                    # Full regeneration of the circuit
                    self.regenerate_circuit()
                except Exception as e:
                    print(f"Error regenerating circuit: {e}")
                    import traceback

                    print(f"Traceback: {traceback.format_exc()}")

                    # Revert to original values on error
                    self.input_n = orig_input_n
                    self.output_n = orig_output_n
                    self.arity = orig_arity
                    self.layer_n = orig_layer_n
                    print("Reverted to original parameters due to error")

            # Manual regenerate button
            if imgui.button("Regenerate Circuit"):
                try:
                    self.regenerate_circuit()
                except Exception as e:
                    print(f"Error regenerating circuit: {e}")
                    import traceback

                    print(f"Traceback: {traceback.format_exc()}")

            # Display current architecture summary
            imgui.text(f"Total parameters: {sum(l.size for l in self.logits0)}")
            imgui.text(f"Layer structure:")
            for i, (gate_n, group_size) in enumerate(self.layer_sizes):
                layer_type = (
                    "Input"
                    if i == 0
                    else "Output"
                    if i == len(self.layer_sizes) - 1
                    else "Hidden"
                )
                imgui.text(f"  {layer_type}: {gate_n} gates, group {group_size}")

            imgui.end_child()
        except Exception as e:
            print(f"Exception in gui: {e}")
            import traceback

            print(f"Traceback: {traceback.format_exc()}")

    def initialize_self_attention(self):
        """Initialize Self-Attention using model loaded from WandB"""
        if self.sa_model is None:
            print("Initializing Self-Attention model...")
            # Total number of nodes in the circuit
            n_node = sum(gate_n for gate_n, _ in self.layer_sizes)
            key = jax.random.PRNGKey(42)  # Fixed seed for reproducibility

            try:
                # Try to load model from WandB if not already loaded
                if not self.loaded_sa_state:
                    print("Attempting to load Self-Attention model from WandB...")
                    if self.load_best_model():
                        print("Successfully loaded Self-Attention model from WandB")
                    else:
                        print(
                            "Failed to load Self-Attention model from WandB, creating new instance"
                        )

                # Create Self-Attention instance (whether we have loaded state or not)
                self.sa_model = CircuitSelfAttention(
                    n_node=n_node,
                    hidden_dim=self.sa_hidden_dim,
                    arity=self.arity,
                    num_heads=self.sa_num_heads,
                    num_layers=self.sa_num_layers,
                    mlp_dim=self.sa_mlp_dim,
                    dropout_rate=self.sa_dropout_rate,
                    rngs=nnx.Rngs(params=key),
                )

                # Update with loaded state if available
                if self.loaded_sa_state:
                    print("Applying loaded Self-Attention state...")
                    try:
                        # Try standard update first
                        nnx.update(self.sa_model, self.loaded_sa_state["sa"])
                    except (AttributeError, TypeError) as e:
                        print(f"Standard update failed: {e}")
                        print("Trying alternative update approach...")

                        # Alternative: Try to access the state differently
                        sa_state = self.loaded_sa_state["sa"]

                        # If it's a full model, get its tree/state
                        if hasattr(sa_state, "_tree"):
                            sa_state = sa_state._tree

                        # Try to update with the extracted state
                        try:
                            # Try using put for each parameter
                            for collection_name, collection in sa_state.items():
                                for var_name, value in collection.items():
                                    try:
                                        path = f"{collection_name}.{var_name}"
                                        self.sa_model.put(path, value)
                                    except Exception as inner_e:
                                        print(
                                            f"Warning: Failed to update {path}: {inner_e}"
                                        )
                        except Exception as update_e:
                            print(f"Alternative update also failed: {update_e}")
                            print("Will proceed with uninitialized model")

                    print("Self-Attention initialized with loaded state")
                else:
                    print("Self-Attention initialized with default parameters")

                return True

            except Exception as e:
                print(f"Error initializing Self-Attention: {e}")
                import traceback

                print(f"Traceback: {traceback.format_exc()}")
                # Clear the loaded state to avoid repeated failures
                self.loaded_sa_state = None
                # Fallback to backprop
                self.optimization_method_idx = 0
                print("Falling back to Backprop optimization")
                return False


if __name__ == "__main__":
    try:
        print("Starting demo with ImGui-based drawing (avoiding immvision)...")
        demo = Demo()

        try:
            immapp.run(
                demo.gui,
                window_title="Random boolean circuit (fixed for WSL)",
                window_size=(800, 600),
                fps_idle=10,
                with_implot=True,
            )  # type: ignore
        except Exception as e:
            print(f"Error running app: {e}")
    except Exception as e:
        print(f"Error initializing demo: {e}")
