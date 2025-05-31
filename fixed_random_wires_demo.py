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
from omegaconf import OmegaConf, DictConfig

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
    """
    Computes which inputs are used by each LUT (lookup table) gate based on its logits.

    Args:
        logits: ndarray of shape (..., lut), where the last dimension represents the LUT truth table.

    Returns:
        input_use_mask: ndarray of shape (..., arity), boolean mask indicating for each LUT which inputs affect its output.
    """
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
    """
    Propagates gate usage backwards through a layer, determining which previous gates and wires are used.

    Args:
        input_n: int, number of inputs to the current layer.
        wires: ndarray, wire indices for the current layer.
        logits: ndarray, LUT logits for the current layer.
        output_use: ndarray, boolean mask indicating which gates in the current layer are used.

    Returns:
        prev_gate_use: ndarray of shape (input_n,), boolean mask indicating which previous gates are used.
        wire_use_mask: ndarray, boolean mask indicating which wires in the current layer are used.
    """
    output_use = output_use.reshape(logits.shape[:2])
    gate_input_use = calc_lut_input_use(logits) * output_use
    wire_use_mask = gate_input_use.any(-1)
    used_wires = wires[wire_use_mask]
    prev_gate_use = np.zeros(input_n, np.bool_)
    prev_gate_use[used_wires] = True
    return prev_gate_use, wire_use_mask


def calc_gate_use_masks(input_n, wires, logits):
    """
    Computes masks indicating which gates and wires are used throughout a multi-layer circuit, propagating usage from outputs to inputs.

    Args:
        input_n: int, number of input gates to the first layer.
        wires: list of ndarrays, each specifying the wire indices for a layer.
        logits: list of ndarrays, each specifying the LUT logits for a layer.

    Returns:
        gate_masks: list of ndarrays, each a boolean mask for gates in each layer (from input to output).
        wire_masks: list of ndarrays, each a boolean mask for wires in each layer (from input to output).
    """
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
        self.layer_n = 3
        self.hidden_dim = 128

        # Add GNN update counter
        self.gnn_update_counter = 0
        self.max_gnn_updates = None  # Fixed number of updates before stopping

        # Update case_n based on input_n
        self.case_n = 1 << self.input_n

        # Wiring mode - MUST be initialized before self.update_wires() is called
        self.wiring_modes = ["fixed", "random"]
        self.wiring_mode_idx = 0  # Default to fixed
        self.wiring_mode = self.wiring_modes[self.wiring_mode_idx]

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
        self.update_wires()
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
        self.use_message_viz = False

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
        self.optimization_methods = ["Backprop", "GNN"]  # , "Self-Attention"]
        self.optimization_method_idx = 0  # Default to Backprop

        # GNN parameters (used if GNN is selected)
        self.gnn = None
        self.gnn_message_steps = 1  # Lower default for faster UI
        self.gnn_node_mlp_features = [64, 32]
        self.gnn_edge_mlp_features = [64, 32]
        self.gnn_enable_message_passing = True

        # Self-Attention parameters
        self.sa_model = None
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

        # WandB loading parameters
        self.run_id = None  # Specific run ID if provided
        self.loaded_run_id = None  # Last successfully loaded run ID
        self.wandb_entity = "m2snn"
        self.wandb_project = "boolean-nca-cc"
        self.wandb_download_dir = "saves"

        self.run_from_last = 2

        # Model caching
        self.model_cache = {}  # Dictionary to cache models by run_id

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
            "config.circuit.num_layers": self.layer_n,
            "config.model.type": model_type,
            "config.training.wiring_mode": self.wiring_mode,
        }

        # Add task to filter if it's a standard task (not text or noise)
        if current_task not in ["text", "noise"]:
            filters["config.circuit.task"] = current_task

        return filters

    def get_cache_key(self):
        """Generate a cache key based on current parameters"""
        # Use a combination of parameters that define the model we need
        method = self.optimization_methods[self.optimization_method_idx]
        task = self.available_tasks[self.task_idx]
        return f"{method}_{self.input_n}_{self.output_n}_{self.arity}_{self.layer_n}_{task}_{self.wiring_mode}"

    def load_best_model(self):
        """Load the best model from wandb based on current parameters or from cache.
        Cache is keyed ONLY by actual run_id."""
        try:
            # If a specific run_id is provided by the user, check cache using that run_id
            if self.run_id and self.run_id in self.model_cache:
                print(f"Using cached model for run_id: {self.run_id}")
                cached_data = self.model_cache[self.run_id]
                self.loaded_run_id = cached_data[
                    "run_id"
                ]  # This should be == self.run_id
                self.hidden_dim = cached_data.get("hidden_dim", self.hidden_dim)

                method_name = self.optimization_methods[self.optimization_method_idx]
                if method_name == "GNN":
                    self.gnn = cached_data["model"]
                    print("GNN model loaded from cache")
                elif method_name == "Self-Attention":
                    self.sa_model = cached_data["model"]
                    print("Self-Attention model loaded from cache")
                return True

            # If self.run_id is None, or if it was set but not found in cache,
            # we proceed to load from WandB. We cannot do a run_id based cache check
            # if self.run_id is None, as the cache is keyed only by run_id.

            # Load from WandB
            filters_for_load = None
            if self.run_id:
                print(f"Loading model from specific run ID: {self.run_id}")
            else:
                filters_for_load = self.create_filter_from_params()
                print(f"Loading best model with filters: {filters_for_load}")

            method_name = self.optimization_methods[self.optimization_method_idx]
            print(f"Loading {method_name} model from WandB...")

            try:
                # Try loading with best model first
                try:
                    model, loaded_dict = load_best_model_from_wandb(
                        run_id=self.run_id,
                        filters=filters,
                        seed=42,  # Use a consistent seed
                        project=self.wandb_project,
                        entity=self.wandb_entity,
                        download_dir=self.wandb_download_dir,
                    )
                except ValueError as e:
                    if "No best model artifacts found" in str(e):
                        print(
                            "No best model artifact found, trying to load final model..."
                        )
                        # Try loading final model instead
                        model, loaded_dict = load_best_model_from_wandb(
                            run_id=self.run_id,
                            filters=filters,
                            seed=42,
                            project=self.wandb_project,
                            entity=self.wandb_entity,
                            download_dir=self.wandb_download_dir,
                            filename="final_model",  # Try final model instead of best model
                        )
                    else:
                        raise  # Re-raise if it's a different error

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

                # Update GUI parameters from loaded config
                if loaded_dict and "config" in loaded_dict:
                    print("\nUPDATING GUI PARAMS FROM LOADED WANDB RUN CONFIG:")
                    live_wandb_config = loaded_dict["config"]
                    print(
                        f"  Raw live_wandb_config from run (type {type(live_wandb_config)}):\n{OmegaConf.to_yaml(live_wandb_config) if isinstance(live_wandb_config, DictConfig) else live_wandb_config}"
                    )  # Print the raw config
                    architecture_changed = False

                    # Circuit params
                    orig_input_n = self.input_n
                    new_input_n = live_wandb_config.get("circuit", {}).get(
                        "input_bits", self.input_n
                    )
                    if new_input_n != self.input_n:
                        print(
                            f"  Updating input_n from {self.input_n} to {new_input_n}"
                        )
                        self.input_n = new_input_n
                        architecture_changed = True

                    orig_output_n = self.output_n
                    new_output_n = live_wandb_config.get("circuit", {}).get(
                        "output_bits", self.output_n
                    )
                    if new_output_n != self.output_n:
                        print(
                            f"  Updating output_n from {self.output_n} to {new_output_n}"
                        )
                        self.output_n = new_output_n
                        architecture_changed = True

                    orig_arity = self.arity
                    new_arity = live_wandb_config.get("circuit", {}).get(
                        "arity", self.arity
                    )
                    if new_arity != self.arity:
                        print(f"  Updating arity from {self.arity} to {new_arity}")
                        self.arity = new_arity
                        architecture_changed = True

                    orig_layer_n = self.layer_n
                    new_layer_n = live_wandb_config.get("circuit", {}).get(
                        "num_layers", self.layer_n
                    )
                    if new_layer_n != self.layer_n:
                        print(
                            f"  Updating layer_n from {self.layer_n} to {new_layer_n}"
                        )
                        self.layer_n = new_layer_n
                        architecture_changed = True

                    # Wiring params from run.config
                    # Note: cfg.training.wiring_mode and cfg.test_seed are typical paths
                    # We need to ensure live_wandb_config has these, or adjust paths.
                    # For run uuizzeb3, wiring_mode is under 'training', test_seed might be top-level or under 'job'/'experiment'.

                    # Attempt to get wiring_mode
                    new_wiring_mode = live_wandb_config.get("training", {}).get(
                        "wiring_mode", self.wiring_mode
                    )
                    if self.wiring_mode != new_wiring_mode:
                        print(
                            f"  Updating wiring_mode from '{self.wiring_mode}' to '{new_wiring_mode}'"
                        )
                        self.wiring_mode = new_wiring_mode
                        # Find index for GUI
                        if new_wiring_mode in self.wiring_modes:
                            self.wiring_mode_idx = self.wiring_modes.index(
                                new_wiring_mode
                            )
                        architecture_changed = True  # Requires wire regeneration

                    # Attempt to get test_seed (assuming it might be top-level in the config)
                    # Common Hydra practice is cfg.seed or cfg.job.seed or cfg.experiment.seed
                    # For this specific run, let's check common Hydra config paths if not directly 'test_seed'
                    possible_seed_paths = [
                        ("test_seed", None),
                        ("seed", None),
                        ("job", "seed"),
                        ("experiment", "seed"),
                        (
                            "training",
                            "wiring_fixed_key_seed",
                        ),  # if it was named this way
                    ]
                    new_wires_key_seed_val = 42  # Default
                    seed_path_used = "default (42)"

                    for path_key, sub_key in possible_seed_paths:
                        if path_key in live_wandb_config:
                            if sub_key:
                                if (
                                    isinstance(live_wandb_config[path_key], dict)
                                    and sub_key in live_wandb_config[path_key]
                                ):
                                    new_wires_key_seed_val = live_wandb_config[
                                        path_key
                                    ][sub_key]
                                    seed_path_used = f'live_wandb_config["{path_key}"]["{sub_key}"]'  # Corrected f-string
                                    break
                            else:
                                new_wires_key_seed_val = live_wandb_config[path_key]
                                seed_path_used = f'live_wandb_config["{path_key}"]'  # Corrected f-string
                                break

                    print(
                        f"  Attempting to use seed for wires_key from: {seed_path_used} (value: {new_wires_key_seed_val})"
                    )
                    new_wires_key = jax.random.PRNGKey(new_wires_key_seed_val)

                    # Convert PRNGKeys to string for comparison if they are JAX arrays, else compare directly
                    # This is a bit tricky as PRNGKeys are opaque. Comparing seeds is more robust.
                    # For now, let's assume if the seed value changed, the key changed.
                    # We'll rely on architecture_changed flag being set if critical params like layer sizes change.
                    # A more direct check: if the seed value used for self.wires_key is different from new_wires_key_seed_val.
                    # This requires storing the seed used for self.wires_key. Let's add self.current_wires_key_seed.

                    if (
                        not hasattr(self, "current_wires_key_seed")
                        or self.current_wires_key_seed != new_wires_key_seed_val
                    ):
                        print(
                            f"  Updating wires_key using seed {new_wires_key_seed_val} (previous seed: {getattr(self, 'current_wires_key_seed', 'N/A')})"
                        )
                        self.wires_key = new_wires_key
                        self.current_wires_key_seed = (
                            new_wires_key_seed_val  # Store current seed
                        )
                        architecture_changed = True

                    # GNN model params from run.config.model
                    # hidden_dim is handled separately below.
                    # Other GNN params like node/edge_mlp_features are used directly during GNN instantiation in utils.py

                    if architecture_changed:
                        print(
                            "  Architecture or wiring parameters changed based on WandB config, will regenerate circuit."
                        )
                        self.regenerate_circuit()  # This will use the updated self.wiring_mode and self.wires_key
                    else:
                        print(
                            "  Loaded model architecture and wiring params match current GUI settings or no relevant changes found."
                        )
                    print("FINISHED UPDATING GUI PARAMS.\n")  # Corrected newline escape

                # Store the run ID we just loaded
                if self.run_id:
                    self.loaded_run_id = self.run_id
                elif "run_id" in loaded_dict:
                    self.loaded_run_id = loaded_dict["run_id"]
                elif hasattr(model, "run_id"):
                    self.loaded_run_id = model.run_id
                else:
                    self.loaded_run_id = actual_model_run_id

                # Get hidden_dim from config
                # Try to get hidden_dim from the pickled config first
                hidden_dim_found = False
                if "config" in loaded_dict:
                    pickled_config = loaded_dict["config"]
                    if isinstance(pickled_config, dict):
                        if (
                            "model" in pickled_config
                            and isinstance(pickled_config["model"], dict)
                            and "hidden_dim" in pickled_config["model"]
                        ):
                            self.hidden_dim = pickled_config["model"]["hidden_dim"]
                            hidden_dim_found = True
                            print(
                                f"Set hidden_dim from pickled_config: {self.hidden_dim}"
                            )
                        elif (
                            "hidden_dim" in pickled_config
                        ):  # Check if it's directly under config
                            self.hidden_dim = pickled_config["hidden_dim"]
                            hidden_dim_found = True
                            print(
                                f"Set hidden_dim directly from pickled_config: {self.hidden_dim}"
                            )

                if not hidden_dim_found:
                    # Fallback: if not in pickled_config, try to get it from the live WandB run config (which was used for instantiation)
                    # This assumes 'config' variable from higher up in the load_best_model_from_wandb scope is accessible or we re-fetch.
                    # For simplicity here, let's assume it might have been set during GUI param updates earlier
                    # or that the GNN's own hidden_dim is the source of truth if available.
                    if (
                        hasattr(self.gnn, "hidden_dim")
                        and self.gnn.hidden_dim is not None
                    ):
                        self.hidden_dim = self.gnn.hidden_dim
                        print(
                            f"Set hidden_dim from instantiated GNN model attribute: {self.hidden_dim}"
                        )
                    else:
                        print(
                            f"Warning: Could not determine hidden_dim from pickled config or GNN model. Retaining current value: {self.hidden_dim}"
                        )

                # Store in cache using the actual_model_run_id as the key, if determined
                if actual_model_run_id:
                    self.model_cache[actual_model_run_id] = {
                        "run_id": actual_model_run_id,
                        "model": model,
                        "hidden_dim": self.hidden_dim,
                    }
                    print(
                        f"Model for run_id '{actual_model_run_id}' stored/updated in cache."
                    )

                # Assign the loaded model to the appropriate attribute
                if method_name == "GNN":
                    self.gnn = model
                    print("GNN model loaded successfully")
                elif method_name == "Self-Attention":
                    self.sa_model = model
                    print("Self-Attention model loaded successfully")

                # Perform a one-off evaluation and print to terminal
                print("\nPerforming one-off evaluation of loaded model...")
                # Ensure current task data is loaded for evaluation
                self.update_task()  # This ensures self.input_x and self.y0 are correct for the current task

                eval_logits_for_print = self.logits  # Start with current logits (likely NOPs if circuit was just regenerated)

                # Determine the actual model instance to use for evaluation
                current_model_instance = None
                if method_name == "GNN" and self.gnn:
                    current_model_instance = self.gnn
                    print(
                        f"GNN is loaded. Internal message_passing_steps: {getattr(current_model_instance, 'message_passing_steps', 'N/A')}"
                    )
                elif method_name == "Self-Attention" and self.sa_model:
                    current_model_instance = self.sa_model
                    print(
                        f"Self-Attention model loaded. Num layers: {getattr(current_model_instance, 'num_layers', 'N/A')}"
                    )

                if current_model_instance:
                    logits_original_shapes = [
                        logit.shape for logit in eval_logits_for_print
                    ]
                    circuit_graph = build_graph(
                        eval_logits_for_print,
                        self.wires,
                        self.input_n,
                        self.arity,
                        self.hidden_dim,
                    )
                    updated_graph = current_model_instance(
                        circuit_graph
                    )  # Call the model (GNN or SA)
                    final_eval_logits = extract_logits_from_graph(
                        updated_graph, logits_original_shapes
                    )
                else:  # Backprop or other, or model not loaded
                    print(
                        "Evaluating with current logits (no GNN/SA processing for this printout)."
                    )
                    final_eval_logits = eval_logits_for_print

                # Run circuit with these logits
                eval_act = run_circuit_gui(
                    final_eval_logits,
                    self.wires,
                    self.gate_mask,
                    self.input_x,
                    hard=False,
                )
                eval_loss = res2loss(eval_act[-1] - self.y0)

                eval_hard_act = run_circuit_gui(
                    final_eval_logits,
                    self.wires,
                    self.gate_mask,
                    self.input_x,
                    hard=True,
                )
                eval_hard_loss = res2loss(eval_hard_act[-1] - self.y0)

                print(f"INITIAL POST-LOAD EVALUATION (Terminal):")
                print(f"  Soft Loss: {float(eval_loss):.6f}")
                print(f"  Hard Loss: {float(eval_hard_loss):.6f}")
                print(f"  Evaluated with: {method_name}")
                print("---------------------------------------------\n")

                return True

            except Exception as inner_e:
                print(f"Error in model loading process: {inner_e}")
                import traceback

                print(f"Traceback: {traceback.format_exc()}")
                return False

        except Exception as e:
            print(f"Error loading model: {e}")
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

    def update_wires(self):
        if self.wiring_mode == "random":
            self.generate_random_wires()
        else:  # fixed
            self.generate_fixed_wires()

    def generate_random_wires_(self):
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

    def generate_random_wires(self):
        key, self.wires_key = jax.random.split(self.wires_key)
        self.wires, self.logits = gen_circuit(key, self.layer_sizes, self.arity)

    def generate_fixed_wires(self):
        in_n = self.input_n
        self.wires = []
        # Use a fixed key for fixed wires, but allow local_noise to have an effect
        # The self.wires_key should have been set from the loaded config's test_seed
        key_for_generation = self.wires_key
        print(
            f"    generate_fixed_wires: Using initial wires_key (from seed {self.current_wires_key_seed if hasattr(self, 'current_wires_key_seed') else 'N/A'}) for layer-wise splitting."
        )

        for i, (gate_n, group_size) in enumerate(self.layer_sizes[1:]):
            # Split the key for each layer to ensure different fixed wires per layer if noise is 0
            key_for_generation, k1 = jax.random.split(key_for_generation)
            print(f"      Layer {i + 1} fixed wires: using split key {k1}")
            local_noise = self.local_noise if self.local_noise > 0.0 else None
            ws = gen_wires_with_noise(
                k1, in_n, gate_n, self.arity, group_size, local_noise
            )
            self.wires.append(ws)
            in_n = gate_n

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

    def initialize_circuit_graph(self):
        """Initialize circuit graph"""
        # Create graph from current circuit
        circuit_graph = build_graph(
            logits=self.logits,
            wires=self.wires,
            input_n=self.input_n,
            arity=self.arity,
            hidden_dim=self.hidden_dim,
            loss_value=0.0,
            update_steps=0,
        )
        self.circuit_graph = circuit_graph
        self.gnn_update_counter = 0

    def initialize_gnn(self):
        """Initialize GNN using model loaded from WandB or cache"""
        if self.gnn is None:
            print("Initializing GNN...")
            key = jax.random.PRNGKey(42)  # Fixed seed for reproducibility

            try:
                # Try to load model from cache or WandB
                if self.load_best_model():
                    print("Successfully loaded GNN model")
                else:
                    print("Failed to load GNN model, creating new instance")

                    # Create new GNN instance with current parameters
                    self.gnn = CircuitGNN(
                        node_mlp_features=self.gnn_node_mlp_features,
                        edge_mlp_features=self.gnn_edge_mlp_features,
                        hidden_dim=self.hidden_dim,
                        arity=self.arity,
                        message_passing=self.gnn_enable_message_passing,
                        rngs=nnx.Rngs(params=key),
                    )

                self.initialize_circuit_graph()
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
            # Check if we've reached the maximum number of updates
            if (
                self.max_gnn_updates is not None
                and self.gnn_update_counter >= self.max_gnn_updates
            ):
                print(
                    f"Reached maximum GNN updates ({self.max_gnn_updates}). Stopping updates."
                )
                return self.max_loss_value, self.max_loss_value

            # Ensure GNN is initialized
            if not self.gnn:
                if (
                    not self.initialize_gnn()
                ):  # This will try to load or create a new GNN
                    # Fallback to backprop if GNN initialization failed catastrophically
                    print(
                        "Critical GNN initialization failure. Falling back to Backprop."
                    )
                    self.optimization_method_idx = 0  # Index for Backprop
                    return self.update_circuit_backprop()

            # Store original logit shapes for reconstruction
            logits_original_shapes = [logit.shape for logit in self.logits]

            # Create graph from current circuit
            circuit_graph = build_graph(
                self.logits,
                self.wires,
                self.input_n,
                self.arity,
                self.hidden_dim,  # Use the demo's current hidden_dim
            )

            # Run GNN for the number of steps specified by the GUI slider
            # The GNN model's __call__ method performs one step of message passing.
            # The loop here makes it perform self.gnn_message_steps total message passing rounds.
            # print(f"Running GNN in GUI for {self.gnn_message_steps} step(s).")
            for _ in range(self.gnn_message_steps):
                circuit_graph = self.gnn(circuit_graph)  # Update graph in place

            updated_graph = circuit_graph  # Final graph after loop

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
                    self.hidden_dim,  # Use self-attention hidden dim
                )

                # Run Self-Attention for specified number of steps
                updated_graph = self.sa_model(circuit_graph)

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
                    if (
                        self.use_message_viz
                        and self.optimization_methods[self.optimization_method_idx]
                        != "Backprop"
                    ):
                        import random

                        r = random.randint(0, 255)
                        g = random.randint(0, 255)
                        b = random.randint(0, 255)
                        a = random.randint(0, 255)  # Optional alpha value

                        # Combine into a single 32-bit integer in ARGB format
                        col = (a << 24) | (r << 16) | (g << 8) | b
                    else:
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

    def reset_gnn_counter(self):
        """Reset the GNN update counter"""
        self.gnn_update_counter = 0
        print("GNN update counter reset")

    def regenerate_circuit(self):
        """Completely regenerate the circuit with current parameters"""
        print("\nREGENERATING CIRCUIT DUE TO PARAMETER CHANGE OR MANUAL REQUEST:")
        print(
            f"  Using: input_n={self.input_n}, output_n={self.output_n}, arity={self.arity}, layer_n={self.layer_n}"
        )
        print(
            f"  Using: wiring_mode='{self.wiring_mode}', wires_key_seed={self.current_wires_key_seed if hasattr(self, 'current_wires_key_seed') else 'N/A'}"
        )

        # Reset GNN counter when regenerating circuit
        self.reset_gnn_counter()

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
        # self.wires_key = jax.random.PRNGKey(42)  # Fixed seed for reproducibility
        # Use the existing self.wires_key, which might have been updated by load_best_model
        self.update_wires()  # Changed from shuffle_wires to update_wires

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

            # Add GNN update counter display
            if (
                self.optimization_methods[self.optimization_method_idx] == "GNN"
                and self.max_gnn_updates is not None
            ):
                imgui.text(
                    f"GNN Updates: {self.gnn_update_counter}/{self.max_gnn_updates}"
                )
                if imgui.button("Reset GNN Counter"):
                    self.reset_gnn_counter()

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

            activated_message_viz, self.use_message_viz = imgui.checkbox(
                "Message visualization", self.use_message_viz
            )
            if activated_message_viz:
                print(
                    f"Visualization mode: {'Messages displayed' if self.use_message_viz else 'Messages not displayed'}"
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

                # Reset logits when switching to GNN or Self-Attention
                if selected_method in ["GNN", "Self-Attention"]:
                    self.logits = self.logits0
                    self.trainstep_i = 0
                    print("Logits reset to initial state")

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
                _, self.hidden_dim = imgui.slider_int(
                    "Hidden Dim", self.hidden_dim, 4, 64
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
                if self.optimization_methods[self.optimization_method_idx] == "GNN":
                    self.initialize_circuit_graph()

            if imgui.button("reset gates + opt"):
                self.logits = self.logits0
                self.init_optimizers()
                self.trainstep_i = 0
                # Reset GNN as well
                self.gnn = None

            if imgui.button("shuffle wires"):
                self.wires_key, key = jax.random.split(self.wires_key)
                self.update_wires()
                self.trainstep_i = 0
                self.initialize_circuit_graph()

            local_noise_changed, self.local_noise = imgui.slider_float(
                "local noise", self.local_noise, 0.0, 20.0
            )
            if local_noise_changed:
                self.update_wires()

            # Weight decay (only affects backprop)
            wd_changed, self.wd_log10 = imgui.slider_float(
                "wd_log10", self.wd_log10, -3, 0.0
            )
            if (
                wd_changed and self.optimization_method_idx == 0
            ):  # Only reinitialize for backprop
                self.init_optimizers()

            # Wiring mode selection
            imgui.separator_text("Wiring Configuration")
            wiring_mode_changed, self.wiring_mode_idx = imgui.combo(
                "Wiring Mode", self.wiring_mode_idx, self.wiring_modes
            )

            if wiring_mode_changed:
                self.wiring_mode = self.wiring_modes[self.wiring_mode_idx]
                print(f"Switched to {self.wiring_mode} wiring.")
                # Regenerate wires when mode changes
                self.update_wires()
                self.trainstep_i = 0  # Reset training progress
                self.initialize_circuit_graph()
                self.gnn = None

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

                # Add slider for run_from_last
                _, self.run_from_last = imgui.slider_int(
                    "Run from Last", self.run_from_last, 1, 10
                )
                imgui.text_wrapped(
                    "Selects n-th most recent run matching filters (if no Run ID)."
                )

                # Status of loaded model
                if self.loaded_run_id:
                    imgui.text_colored(
                        imgui.ImVec4(0.0, 1.0, 0.0, 1.0),
                        f"Loaded model from run: {self.loaded_run_id}",
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
        """Initialize Self-Attention using model loaded from WandB or cache"""
        if self.sa_model is None:
            print("Initializing Self-Attention model...")
            key = jax.random.PRNGKey(42)  # Fixed seed for reproducibility

            try:
                # Try to load model from cache or WandB
                if self.load_best_model():
                    print("Successfully loaded Self-Attention model")
                    return True
                else:
                    print("Failed to load Self-Attention model, creating new instance")
                    # Create new Self-Attention instance with current parameters
                    self.sa_model = CircuitSelfAttention(
                        n_node=sum(gate_n for gate_n, _ in self.layer_sizes),
                        hidden_dim=self.hidden_dim,
                        arity=self.arity,
                        num_heads=self.sa_num_heads,
                        num_layers=self.sa_num_layers,
                        mlp_dim=self.sa_mlp_dim,
                        dropout_rate=self.sa_dropout_rate,
                        rngs=nnx.Rngs(params=key),
                    )
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
