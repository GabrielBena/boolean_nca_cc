#!/usr/bin/env python3
# Randomly wired boolean circuits demo - FIXED VERSION FOR WSL
# Modified from Alexander Mordvintsev's original
# This version avoids immvision and uses direct imgui image display
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

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
from sklearn.decomposition import PCA
import math


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
from boolean_nca_cc.utils import build_graph, extract_logits_from_graph

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
        self.input_n = 8
        self.output_n = 8
        self.arity = 4
        self.layer_n = 4
        
        self.edge_colors = {}

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
        self.optimization_methods = ["Backprop", "GNN"]
        self.optimization_method_idx = 0  # GNN

        # GNN parameters (used if GNN is selected)
        self.gnn = None
        self.gnn_hidden_dim = 64
        self.gnn_message_steps = 1 # Lower default for faster UI
        hidden_features = 64

        self.gnn_node_mlp_features = [hidden_features, hidden_features]
        self.gnn_edge_mlp_features = [hidden_features, hidden_features]
        self.gnn_enable_message_passing = True
        self.loaded_gnn_state = None # Variable to store loaded state

        # Attempt to load GNN state
        gnn_state_file = "gnn_results.pkl"
        gnn_state_file = "tofix"

        if os.path.exists(gnn_state_file):
            print(f"Found {gnn_state_file}, attempting to load...")
            try:
                with open(gnn_state_file, "rb") as f:
                    # Load the dictionary containing GNN state and potentially optimizer state
                    loaded_data = pickle.load(f)
                    # Store the relevant parts for initialize_gnn
                    if 'model' in loaded_data:
                         self.loaded_gnn_state = loaded_data # Store the whole dict for now
                         print("GNN state loaded successfully from file.")
                         # Optionally set GNN as the default method if loaded
                         # self.optimization_method_idx = self.optimization_methods.index("GNN")
                    else:
                        print(f"Warning: '{gnn_state_file}' found but does not contain 'model' key.")

            except Exception as e:
                print(f"Error loading GNN state from {gnn_state_file}: {e}")
                print("Proceeding without loaded GNN state.")
        else:
            print(f"No {gnn_state_file} found, GNN will be initialized from scratch if needed.")

        # Initialize optimizers (Backprop optimizer first)
        self.init_optimizers()

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
        self.wire_colors = [jp.zeros_like(layer_wires) for layer_wires in self.wires]


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
        """Initialize GNN optimizer if needed"""
        if self.gnn is None:
            print("Initializing GNN...")
            # Initialize GNN using current circuit configuration
            key = jax.random.PRNGKey(42)  # Fixed seed for reproducibility
            try:
                # Check if a saved GNN state exists
                if hasattr(self, 'loaded_gnn_state') and self.loaded_gnn_state:
                    print("Loading GNN state from file...")
                    # Restore GNN state
                    self.gnn = CircuitGNN(
                        node_mlp_features=self.gnn_node_mlp_features,
                        edge_mlp_features=self.gnn_edge_mlp_features,
                        hidden_dim=self.gnn_hidden_dim,
                        arity=self.arity,
                        message_passing=self.gnn_enable_message_passing,
                        # Add other necessary params like use_attention if configured
                        rngs=nnx.Rngs(params=key) # Pass RNGs correctly
                    )
                    nnx.update(self.gnn, self.loaded_gnn_state['model'])
                    print("GNN state loaded successfully.")

                else:
                    print("Creating new GNN instance...")
                    # Create a new GNN instance
                    self.gnn = CircuitGNN(
                        node_mlp_features=self.gnn_node_mlp_features,
                        edge_mlp_features=self.gnn_edge_mlp_features,
                        hidden_dim=self.gnn_hidden_dim,
                        arity=self.arity,
                        message_passing=self.gnn_enable_message_passing,
                        # Add other necessary params like use_attention if configured
                        rngs=nnx.Rngs(params=key) # Pass RNGs correctly
                    )
                    # Optionally initialize a GNN optimizer if needed for training within the demo
                    # self.gnn_optimizer = nnx.Optimizer(self.gnn, optax.adam(1e-3))
                    print("New GNN instance created.")

                print("GNN initialized successfully")
            except Exception as e:
                print(f"Error initializing GNN: {e}")
                import traceback
                print(f"Traceback: {traceback.format_exc()}")
                # Fallback to backprop
                self.optimization_method_idx = 0

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
        run_PCA = False
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
                
                total_elements = sum(wire.shape[0]*wire.shape[1] for wire in self.wires)
                print(f"Total number of elements in self.wires sub-arrays: {total_elements}")
                

                # Run GNN for specified number of steps
                updated_graph, messages = run_gnn_scan(
                    self.gnn, circuit_graph, self.gnn_message_steps, run_PCA=True
                )
                
                # late night attempt failed, do now how to get the pin ouf of arity from messages info
                if run_PCA:
                    pca = PCA(n_components=3)
                    # Fit PCA on the data and transform
                    colors = pca.fit_transform(messages)
                    
                    senders = updated_graph.senders
                    receivers = updated_graph.receivers
                    #print(messages.shape[0])
                    
                    #print(len([el for el in updated_graph.nodes["layer"] if el==1]))
                    
                    for message_idx in range(colors.shape[0]):
                        
                        #if message_idx == 256:
                            #print("check")
          
                        sender = int(senders[message_idx]) 
                        temp = updated_graph.nodes["gate_id"]
                        sender_node_idx = [idx for idx, el in enumerate(temp) if el==sender]
                        sender_node_idx = sender_node_idx[0]
                        #print(sender, sender_node_idx)
                        sender_layer = int(updated_graph.nodes["layer"][sender_node_idx])                          
                        receiver_layer = sender_layer + 1
                        
                        # figure out on which gate in the layer the sender corresponds to
                        if sender_layer >= len(self.layer_sizes):
                            break
                        
                        if sender_layer:
                            #temp = []
                            #for layer in self.layer_sizes[:sender_layer]:
                            #    temp.append(np.prod(*layer) )
                            #n_prec_gates = sum(temp)
                            

                            n_prec_gates = sum([math.prod(layer) for layer in self.layer_sizes[:sender_layer]])
                        else:
                            n_prec_gates = 0
                        src_idx = sender-n_prec_gates
                        """
                        
                        # figure out on which gate in the layer the sender corresponds to
                        receiver = int(receivers[message_idx]) 
                        if receiver_layer == 1:
                            n_prec_gates = sum(self.layer_sizes[0])
                        else:
                            n_prec_gates = sum([np.prod(*layer) for layer in self.layer_sizes[:receiver_layer]])

                        receiver_idx = receiver-n_prec_gates
                        """
                        if receiver_layer < len(self.layer_sizes):
                            num_receiving_groups = self.layer_sizes[receiver_layer][1]
                            pin_idx = np.random.randint(0, self.arity)
                            group_idx = np.random.randint(0, num_receiving_groups)
                            self.edge_colors[(receiver_layer, src_idx, pin_idx, group_idx)] = colors[message_idx,...]
                            #print(message_idx)
                            


                        #num_gates_per_group = self.layer_sizes[receiver_layer][0]
                        #receiver_node_idxs = [el for el in range(num_gates_per_group)]*num_receiving_groups
                        #group_idxs = np.repeat(np.arange(num_receiving_groups), num_gates_per_group) 
                        
                        
                        #print(receiver_idx, len(group_idxs))

                        #pin_idx = np.random.randint(0, self.arity)
                        #pin_idx = receiver_node_idxs[receiver_idx]
                        #group_idx = int(group_idxs[receiver_idx])
                        
                        #src_idx = 
                        #group_idx = np.random.randint(0, num_receiving_groups)
                        

                        
                        
                    
                else:
                    self.edge_colors = {}
                    
                    
                

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

    def update_circuit(self):
        """Update circuit using selected optimization method"""
        try:
            # Choose optimization method
            if self.optimization_methods[self.optimization_method_idx] == "Backprop":
                loss, hard_loss = self.update_circuit_backprop()
            else:  # GNN
                loss, hard_loss = self.update_circuit_gnn()

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
            
            
    def color_from_rgb_array(self, a, alpha=255):
        """
        Takes a 3-element array [r, g, b] (values can be floats or ints),
        clamps to [0, 255], and returns a packed 32-bit ARGB color.
        """
        r = max(0, min(255, int(a[0])))
        g = max(0, min(255, int(a[1])))
        b = max(0, min(255, int(a[2])))
        return (alpha << 24) | (r << 16) | (g << 8) | b

    def to_rgb_color(self, a):
        # Normalize and clamp each component to [0, 255]
        def normalize(v):
            # Map values from [-1, 1] to [0, 255]
            return max(0, min(255, int((v + 1) * 127.5)))

        r = normalize(a[0])
        g = normalize(a[1])
        b = normalize(a[2])

        # Pack into 0xRRGGBB
        return (r << 16) + (g << 8) + b

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
                #col = 0xFF404040 + (a << 16)  # Shift `a` into the red channel
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
                
                num_receiving_groups = self.layer_sizes[li][1]
                receiver_node_idxs = [el for el in range(self.arity)]*num_receiving_groups
                group_idxs = np.repeat(np.arange(num_receiving_groups), self.arity)          
                for x0, x1, si, ri, gi,  m in zip(
                    src_x.ravel(), dst_x.ravel(), wires.ravel(), receiver_node_idxs, group_idxs, masks.ravel()
                ):
                    # si is the ID of source node
                    if not m:
                        continue
                    a = int(prev_act[si] * 0x60) if si < len(prev_act) else 0
                    
                    #a = [0,0,255]
                    #col = self.color_from_rgb_array(a)
                    
                    if self.use_message_viz and self.optimization_methods[self.optimization_method_idx] != "Backprop":
                        import random
                        r = random.randint(0, 255)
                        g = random.randint(0, 255)
                        b = random.randint(0, 255)
                        a = random.randint(0, 255)  # Optional alpha value

                        # Combine into a single 32-bit integer in ARGB format
                        col = (a << 24) | (r << 16) | (g << 8) | b
                    else:
                        col = 0xFF404040 + (a << 8)
                    
                    
                    """
                    if self.edge_colors:
                        if (li, int(si), ri, int(gi)) in self.edge_colors.keys():
                            color = self.edge_colors[(li, int(si), ri, int(gi))]  # this is 3 values for the rgb
                            col = self.to_rgb_color(color)
                        else:
                            col = 0xFF404040 + (a << 8)
                    """

                    #else:
                    #    col = 0xFF404040 + (a << 8)
                    
                    #col2 = 0xFF404040 + (a << 8)
                    
                    #r, g, b = prev_act[si] * 255, 0, 0  # Example: Red intensity based on activation
                    #col = 0xFF000000 | (int(b) << 16) | (int(g) << 8) | int(r)
                    #col = 0xFF404040 + (a << 16)  # Shift `a` into the red channel
                    
                    #r = int(r) & 0xFF  # Ensure the value is in the range 0-255

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
                print(
                    f"Switching to {self.optimization_methods[self.optimization_method_idx]} optimization"
                )
                # Initialize GNN if needed when switching to it
                if self.optimization_methods[self.optimization_method_idx] == "GNN":
                    self.initialize_gnn()

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
                                'gnn': nnx.state.get_state(self.gnn),
                                # Add optimizer state if needed and available
                            }
                            with open("gnn_results.pkl", "wb") as f:
                                pickle.dump(gnn_state_to_save, f)
                            print("GNN state saved to gnn_results.pkl")
                        except Exception as e:
                            print(f"Error saving GNN state: {e}")
                    else:
                        print("No active GNN model to save.")

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
