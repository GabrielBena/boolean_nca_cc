#!/usr/bin/env python3
"""
Training-compatible boolean circuits demo using shared infrastructure.

This demo shows live circuit optimization where:
- Backprop: Direct gradient-based optimization of circuit logits
- GNN/Self-Attention: Pre-trained models suggest logit improvements (frozen models)

No model training occurs - only circuit logit optimization.
"""

import logging

import IPython
import jax
import jax.numpy as jp
import numpy as np
import optax
from flax import nnx

# Configure logging to show INFO messages
logging.basicConfig(level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s")

# Import model components
from imgui_bundle import (
    hello_imgui,
    imgui,
    immapp,
    implot,
)

from boolean_nca_cc import generate_layer_sizes

# Import shared training infrastructure
from boolean_nca_cc.circuits.model import gen_circuit, run_circuit
from boolean_nca_cc.circuits.tasks import TASKS, get_task_data

# # Import training loop functions
# from boolean_nca_cc.training.checkpointing import (
#     load_config_from_wandb,
#     load_model_from_config_and_checkpoint,
# )
from boolean_nca_cc.training.evaluation import (
    evaluate_model_stepwise_generator,
    get_loss_from_wires_logits,
)

# # Import genetic mutation functions
# from boolean_nca_cc.training.pool.perturbation import mutate_wires_swap

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


def propagate_gate_use(input_n, wires, logits, output_use):
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
    layer_sizes = [input_n] + [np.prod(log.shape[:2]) for log in logits]
    gate_use_mask = np.ones(layer_sizes[-1], np.bool_)
    gate_masks = [gate_use_mask]
    wire_masks = []
    for i in range(len(logits))[::-1]:
        gate_use_mask, wire_use_mask = propagate_gate_use(
            layer_sizes[i], wires[i], logits[i], gate_use_mask
        )
        wire_masks.append(wire_use_mask)
        gate_masks.append(gate_use_mask)
    return gate_masks[::-1], wire_masks[::-1]


######################## helper functions ##############################


def is_point_in_box(p0, p1, p):
    """Check if point p is inside box defined by p0 and p1"""
    (x0, y0), (x1, y1), (x, y) = p0, p1, p
    return (x0 <= x <= x1) and (y0 <= y <= y1)


class LogitContainer(nnx.Module):
    """Simple container to hold circuit logits for nnx.Optimizer"""

    def __init__(self, logits):
        self.logits = logits


def zoom(a, k=2):
    """Zoom function for image visualization"""
    return np.repeat(np.repeat(a, k, 1), k, 0)


def unpack(x, bit_n=8):
    """Unpack integers to binary representation"""
    return (x[..., None] >> np.r_[:bit_n]) & 1


max_trainstep_n = 1000


class CircuitOptimizationDemo:
    """
    Demo showing live circuit optimization.

    - Backprop: Direct gradient-based logit optimization
    - GNN/Self-Attention: Pre-trained models suggest logit improvements
    """

    def __init__(self):
        # Circuit configuration
        self.input_n = 8
        self.output_n = 8
        self.arity = 4
        self.layer_n = 3
        self.hidden_dim = 64

        # Update case_n based on input_n
        self.case_n = 1 << self.input_n

        # Wiring configuration
        self.wiring_modes = ["fixed", "random"]
        self.wiring_mode_idx = 1
        self.wiring_mode = self.wiring_modes[self.wiring_mode_idx]
        self.wiring_seed = 42  # Direct control over wiring seed
        self.wiring_key = jax.random.PRNGKey(self.wiring_seed)

        # Optimization configuration
        self.loss_type = "l4"
        self.learning_rate = 1.0  # Learning rate for backprop
        self.n_message_steps = 1

        # Initialize circuit using shared functions
        self.initialize_circuit()

        # Task configuration
        self.available_tasks = list(TASKS.keys())
        self.task_idx = 6
        self.task_text = "Hello Neural CA"  # Shorter text works better with performance mode
        self.noise_p = 0.5
        self.update_task()

        # Optimization state
        self.step_i = 0
        self.is_optimizing = True
        self.loss_log = np.zeros(max_trainstep_n, np.float32)
        self.hard_log = np.zeros(max_trainstep_n, np.float32)

        # Mutation settings
        self.mutation_rate = 0.05

        # Optimization method configuration
        self.optimization_methods = ["Backprop", "GNN", "Self-Attention"]
        self.optimization_method_idx = 0

        # Model instances (only pre-trained, frozen models)
        self.frozen_model = None
        self.logit_optimizer = None  # Only for backprop

        # Model configuration for consistency with training
        self.model_hidden_dim = self.hidden_dim  # Will be updated when loading models
        self.model_use_globals = True  # Will be updated when loading self-attention models

        # Step-by-step generator for GNN/Self-Attention (unified with training code)
        self.model_generator = None
        self.last_step_result = None

        # Visualization settings
        self.use_simple_viz = False
        self.use_message_viz = False  # For circuit visualization
        self.use_full_resolution = False  # Toggle for full resolution vs performance mode
        self.max_loss_value = 10.0
        self.min_loss_value = 1e-6
        self.auto_scale_plot = True

        # Loss plot display options
        self.loss_display_modes = ["Both", "Soft Only", "Hard Only"]
        self.loss_display_mode_idx = 0  # Default to showing both

        # Gate mask management for circuit visualization
        self.gate_mask = []
        self.wire_masks = []
        self.reset_gate_mask()

        # Store activations for circuit visualization
        self.act = []
        self.err_mask = None

        # Active case for visualization
        self.active_case_i = 123 % self.case_n

        # WandB integration
        self.wandb_entity = "m2snn"
        self.wandb_project = "boolean-nca-cc"
        self.wandb_download_dir = "saves"
        self.run_id = None
        self.loaded_run_id = None

        # Initialize visualization
        self.setup_visualization()

        # Debug flag for printing dimensions
        self._debug_printed = False

        # Initialize optimization method
        self.initialize_optimization_method()

        # Initialize activations now that everything is set up
        self.initialize_activations()

    def initialize_circuit(self):
        """Initialize circuit using shared infrastructure"""
        # Generate layer sizes using shared function
        self.layer_sizes = generate_layer_sizes(
            self.input_n, self.output_n, self.arity, self.layer_n
        )

        # Generate circuit using shared function
        self.wires, self.logits = gen_circuit(self.wiring_key, self.layer_sizes, arity=self.arity)

        # Store initial logits
        self.logits0 = self.logits

        print(f"Circuit initialized with {sum(logit.size for logit in self.logits0)} parameters")
        print(f"Layer structure: {self.layer_sizes}")

        # Reset gate masks for new circuit structure
        self.reset_gate_mask()

        # Initialize empty activations (will be properly set after task setup)
        self.act = [np.zeros((self.case_n, size)) for size, _ in self.layer_sizes]
        self.err_mask = np.zeros((self.case_n, self.output_n), bool)

        # Reset the model generator when circuit changes
        self.model_generator = None
        self.last_step_result = None

    def update_task(self):
        """Update current task using shared task infrastructure"""
        task_name = self.available_tasks[self.task_idx]

        # Use shared task infrastructure for all tasks
        try:
            # Prepare task-specific parameters
            task_kwargs = {
                "input_bits": self.input_n,
                "output_bits": self.output_n,
            }

            # Add task-specific parameters
            if task_name == "text":
                task_kwargs["text"] = self.task_text
            elif task_name == "noise":
                task_kwargs["noise_p"] = self.noise_p
                # Use a consistent seed for reproducibility during demo
                task_kwargs["seed"] = 42

            self.input_x, self.y0 = get_task_data(task_name, self.case_n, **task_kwargs)
        except Exception as e:
            print(f"Error loading task '{task_name}': {e}")
            # Fallback to copy task
            x = jp.arange(self.case_n)
            self.input_x = unpack(x, bit_n=self.input_n)
            max_output_value = (1 << self.output_n) - 1
            clipped_output = np.minimum(x, max_output_value)
            self.y0 = jp.float32(unpack(clipped_output, bit_n=self.output_n))

        # Reset optimization progress
        self.step_i = 0
        self.loss_log = np.zeros(max_trainstep_n, np.float32)
        self.hard_log = np.zeros(max_trainstep_n, np.float32)

        # Reset the model generator when task changes
        self.model_generator = None
        self.last_step_result = None

        # Update visualization
        self.setup_visualization()

        # Refresh activations for new task
        self.initialize_activations()

    def setup_visualization(self):
        """Setup visualization using shared functions"""
        # Use consistent zoom factor like in notebook
        zoom_factor = 8

        # Create input visualization - transpose to match notebook format
        inp_img = self.input_x.T
        inp_img = np.dstack([inp_img] * 3)
        inp_img = zoom(inp_img, zoom_factor)
        self.inputs_img = np.uint8(inp_img.clip(0, 1) * 255)

        # Create ground truth visualization - transpose to match notebook format
        gt_img = self.y0.T
        gt_img = np.dstack([gt_img] * 3)
        gt_img = zoom(gt_img, zoom_factor)
        self.ground_truth_img = np.uint8(gt_img.clip(0, 1) * 255)

        # Initialize output image placeholder
        self.outputs_img = np.zeros_like(self.ground_truth_img)

        # Initialize textures with None - will be set when ImGui context is available
        self.input_texture = None
        self.output_texture = None
        self.ground_truth_texture = None
        self.imgui_initialized = False

        # Initialize active case
        self.active_case_i = 123 % self.case_n

    def initialize_imgui_textures(self):
        """Initialize ImGui textures once context is available"""
        if not self.imgui_initialized:
            try:
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
                self.imgui_initialized = True
            except Exception as e:
                print(f"Error initializing ImGui textures: {e}")
                # ImGui context not ready yet
                pass

    def initialize_optimization_method(self):
        """Initialize the selected optimization method"""
        method_name = self.optimization_methods[self.optimization_method_idx]

        if method_name == "Backprop":
            # Use direct optax optimizer (not nnx.Optimizer) for logits
            opt_fn = optax.adamw(self.learning_rate, 0.8, 0.8, weight_decay=1e-1)
            self.logit_opt_state = opt_fn.init(self.logits)
            self.logit_optimizer = opt_fn
            self.frozen_model = None
            # Reset generator when switching to backprop
            self.model_generator = None
            self.last_step_result = None

        elif method_name in ["GNN", "Self-Attention"]:
            # Try to load pre-trained frozen model
            if self.try_load_wandb_model():
                print(f"Loaded frozen {method_name} model from WandB")
                self.logit_optimizer = None  # No optimizer needed for frozen models
                self.logit_opt_state = None  # No optimizer state needed for frozen models
                # Initialize the generator for step-by-step evaluation
                self.initialize_model_generator()
            else:
                print(f"Could not load {method_name} model. Falling back to Backprop.")
                self.optimization_method_idx = 0
                self.initialize_optimization_method()
                return

    def initialize_model_generator(self):
        """Initialize the step-by-step model generator using the unified training code"""
        if self.frozen_model is None:
            return

        try:
            # Use the exact same generator as training and evaluation
            # For self-attention models, we need to use the correct hidden_dim from the model
            hidden_dim_for_graph = getattr(self, "model_hidden_dim", self.hidden_dim)

            print("Initializing model generator with:")
            print(f"  - hidden_dim: {hidden_dim_for_graph}")
            print(f"  - use_globals: {getattr(self, 'model_use_globals', True)}")
            print(f"  - model type: {type(self.frozen_model).__name__}")

            self.model_generator = evaluate_model_stepwise_generator(
                model=self.frozen_model,
                wires=self.wires,
                logits=self.logits,
                x_data=self.input_x,
                y_data=self.y0,
                input_n=self.input_n,
                arity=self.arity,
                circuit_hidden_dim=hidden_dim_for_graph,  # Use model's hidden_dim
                max_steps=None,  # Infinite steps for live demo
                loss_type=self.loss_type,
                bidirectional_edges=True,
                layer_sizes=self.layer_sizes,
            )

            # Get the initial state (step 0)
            self.last_step_result = next(self.model_generator)
            print(
                f"Initialized model generator with initial loss: {self.last_step_result.loss:.4f}"
            )

        except Exception as e:
            print(f"Error initializing model generator: {e}")
            import traceback

            print(f"Traceback: {traceback.format_exc()}")
            self.model_generator = None
            self.last_step_result = None

    def reset_gate_mask(self):
        """Reset all gate masks to active"""
        # Ensure we have the right number of masks
        self.gate_mask = [np.ones(gate_n) for gate_n, _ in self.layer_sizes]
        self.wire_masks = [np.ones_like(w, np.bool_) for w in self.wires]
        print(
            f"Reset gate mask: {len(self.gate_mask)} gate masks, {len(self.wire_masks)} wire masks"
        )

    def mask_unused_gates(self):
        """Mask unused gates based on circuit analysis"""
        gate_masks, self.wire_masks = calc_gate_use_masks(self.input_n, self.wires, self.logits)
        for i in range(len(gate_masks)):
            self.gate_mask[i] = np.array(self.gate_mask[i] * gate_masks[i])

    def try_load_wandb_model(self):
        """Try to load frozen model from WandB"""
        try:
            method_name = self.optimization_methods[self.optimization_method_idx]
            model_type = "gnn" if method_name == "GNN" else "self_attention"

            filters = {
                "config.circuit.input_bits": self.input_n,
                "config.circuit.output_bits": self.output_n,
                "config.circuit.arity": self.arity,
                # "config.circuit.num_layers": self.layer_n,
                "config.model.type": model_type,
                "config.training.wiring_mode": self.wiring_mode,
                "config.circuit.task": self.available_tasks[self.task_idx],
            }

            # Load frozen model
            loaded_config, checkpoint_path, run_id = load_config_from_wandb(
                run_id=self.run_id,
                filters=filters if not self.run_id else None,
                project=self.wandb_project,
                entity=self.wandb_entity,
                download_dir=self.wandb_download_dir,
                filename="latest_checkpoint",
                select_by_best_metric=False,
                run_from_last=1,
                use_cache=True,
            )

            if loaded_config.circuit.num_layers != self.layer_n:
                print(
                    f"Layer number mismatch: {loaded_config.circuit.num_layers} != {self.layer_n}"
                )
                print(f"Using layer number: {self.layer_n}")
                loaded_config.circuit.num_layers = self.layer_n

            model, loaded_dict = load_model_from_config_and_checkpoint(
                config=loaded_config,
                checkpoint_path=checkpoint_path,
                run_id=run_id,
            )

            self.frozen_model = model
            self.loaded_run_id = loaded_dict.get("run_id", "unknown")

            # Extract hidden_dim from loaded config for graph compatibility
            if hasattr(loaded_config, "model") and hasattr(loaded_config.model, "hidden_dim"):
                self.model_hidden_dim = loaded_config.model.hidden_dim
                print(f"Using model hidden_dim={self.model_hidden_dim} from loaded config")
            elif hasattr(loaded_config, "circuit") and hasattr(
                loaded_config.circuit, "circuit_hidden_dim"
            ):
                self.model_hidden_dim = loaded_config.circuit.circuit_hidden_dim
                print(f"Using circuit hidden_dim={self.model_hidden_dim} from loaded config")
            else:
                self.model_hidden_dim = self.hidden_dim  # Fallback to demo default
                print(
                    f"Could not find hidden_dim in config, using default: {self.model_hidden_dim}"
                )

            # Extract use_globals from loaded config for self-attention models
            if method_name == "Self-Attention":
                if hasattr(loaded_config, "model") and hasattr(loaded_config.model, "use_globals"):
                    self.model_use_globals = loaded_config.model.use_globals
                    print(f"Using model use_globals={self.model_use_globals} from loaded config")
                else:
                    self.model_use_globals = True  # Default fallback for compatibility
                    print(
                        f"Could not find use_globals in config, using default: {self.model_use_globals}"
                    )
            else:
                self.model_use_globals = (
                    True  # Default for GNN models (not applicable but for consistency)
                )

            return True

        except Exception as e:
            print(f"Could not load model from WandB: {e}")
            return False

    def optimize_circuit(self):
        """Perform one optimization step on the circuit logits"""
        try:
            method_name = self.optimization_methods[self.optimization_method_idx]

            if method_name == "Backprop":
                loss, hard_loss = self.optimize_backprop()
            else:
                loss, hard_loss = self.optimize_with_unified_model()

            # Update loss logs
            i = self.step_i % len(self.loss_log)
            self.loss_log[i] = max(min(float(loss), self.max_loss_value), self.min_loss_value)
            self.hard_log[i] = max(min(float(hard_loss), self.max_loss_value), self.min_loss_value)

            # Debug output every 100 steps
            if self.is_optimizing and self.step_i % 100 == 0:
                print(
                    f"Step {self.step_i}: Loss = {float(loss):.4f}, Hard Loss = {float(hard_loss):.4f}"
                )

            if self.is_optimizing:
                self.step_i += 1

            # Update visualization
            self.update_output_visualization()

        except Exception as e:
            print(f"Error in optimize_circuit: {e}")
            import traceback

            print(f"Traceback: {traceback.format_exc()}")

    def optimize_backprop(self):
        """Optimize circuit logits using backpropagation"""
        # Get current logits
        current_logits = self.logits

        # Calculate loss using the unified function for consistency
        (
            loss,
            (
                hard_loss,
                pred,
                pred_hard,
                accuracy,
                hard_accuracy,
                res,
                hard_res,
            ),
        ) = get_loss_from_wires_logits(
            current_logits, self.wires, self.input_x, self.y0, self.loss_type
        )

        if self.is_optimizing and hasattr(self, "logit_optimizer") and self.logit_optimizer:
            # Compute gradients with respect to logits
            def loss_fn(logits):
                loss, _ = get_loss_from_wires_logits(
                    logits, self.wires, self.input_x, self.y0, self.loss_type
                )
                return loss

            grad_fn = jax.grad(loss_fn)
            grads = grad_fn(current_logits)

            # Update logits using optax
            updates, self.logit_opt_state = self.logit_optimizer.update(
                grads, self.logit_opt_state, current_logits
            )
            self.logits = optax.apply_updates(current_logits, updates)

        # Store predictions for visualization
        self.current_pred = pred
        self.current_pred_hard = pred_hard

        # Generate circuit activations for visualization using shared circuit runner
        try:
            # Import the circuit runner from the shared infrastructure
            from boolean_nca_cc.circuits.model import run_circuit

            # Run circuit to get layer-by-layer activations
            # This returns [input_acts, layer1_acts, layer2_acts, ..., output_acts]
            self.act = run_circuit(
                current_logits, self.wires, self.input_x, hard=False, gate_mask=self.gate_mask
            )

            # Generate error mask for visualization
            self.err_mask = pred_hard != self.y0

        except Exception as e:
            print(f"Warning: Could not generate circuit activations: {e}")
            # Fallback: create empty activations
            self.act = [np.zeros((self.case_n, size)) for size, _ in self.layer_sizes]
            self.err_mask = np.zeros((self.case_n, self.output_n), bool)

        return loss, hard_loss

    def optimize_with_unified_model(self):
        """Use the unified generator from training code to optimize with frozen GNN/Self-Attention model"""
        if self.frozen_model is None:
            print("No frozen model loaded, falling back to backprop")
            self.optimization_method_idx = 0
            self.initialize_optimization_method()
            return self.optimize_backprop()

        try:
            # Initialize generator if needed
            if self.model_generator is None:
                self.initialize_model_generator()
                if self.model_generator is None:
                    # Fallback to backprop if generator initialization failed
                    print("Generator initialization failed, falling back to backprop")
                    self.optimization_method_idx = 0
                    self.initialize_optimization_method()
                    return self.optimize_backprop()

            if self.is_optimizing:
                # Get the next step from the generator (exactly like training)
                try:
                    # Run the specified number of message steps
                    for _ in range(self.n_message_steps):
                        self.last_step_result = next(self.model_generator)

                    # Update circuit logits with the results from the generator
                    self.logits = self.last_step_result.logits

                except StopIteration:
                    # Generator exhausted, reinitialize
                    print("Model generator exhausted, reinitializing...")
                    self.initialize_model_generator()
                    if self.model_generator is None:
                        return self.optimize_backprop()
                    self.last_step_result = next(self.model_generator)

            # Use the last step result for visualization
            if self.last_step_result is not None:
                # Store predictions for visualization (exactly like training)
                self.current_pred = self.last_step_result.predictions
                self.current_pred_hard = self.last_step_result.hard_predictions

                # Generate circuit activations for visualization using the same method as backprop
                try:
                    # Run circuit to get layer-by-layer activations
                    # This returns [input_acts, layer1_acts, layer2_acts, ..., output_acts]
                    self.act = run_circuit(
                        self.logits, self.wires, self.input_x, hard=False, gate_mask=self.gate_mask
                    )

                    # Generate error mask for visualization
                    self.err_mask = self.current_pred_hard != self.y0

                except Exception as act_e:
                    print(
                        f"Warning: Could not generate circuit activations in unified model: {act_e}"
                    )
                    # Fallback: create empty activations
                    self.act = [np.zeros((self.case_n, size)) for size, _ in self.layer_sizes]
                    self.err_mask = np.zeros((self.case_n, self.output_n), bool)

                return self.last_step_result.loss, self.last_step_result.hard_loss
            else:
                # No result yet, return current state
                (
                    loss,
                    (
                        hard_loss,
                        pred,
                        pred_hard,
                        accuracy,
                        hard_accuracy,
                        res,
                        hard_res,
                    ),
                ) = get_loss_from_wires_logits(
                    self.logits, self.wires, self.input_x, self.y0, self.loss_type
                )
                self.current_pred = pred
                self.current_pred_hard = pred_hard
                return loss, hard_loss

        except Exception as e:
            import traceback

            print(f"Error with unified model: {e}")
            print(f"Traceback: {traceback.format_exc()}")
            print("Falling back to backprop")
            # Fallback to backprop
            self.optimization_method_idx = 0
            self.initialize_optimization_method()
            return self.optimize_backprop()

    def update_output_visualization(self):
        """Update output visualization based on current predictions"""
        if not hasattr(self, "current_pred_hard"):
            return

        # Create output visualization - transpose to match notebook format
        oimg = self.current_pred.T
        oimg = np.dstack([oimg] * 3)

        # Apply error mask for visualization
        err_mask = (self.current_pred_hard != self.y0).T
        m = err_mask[..., None] * 0.5
        oimg = oimg * (1.0 - m) + m * np.float32([1, 0, 0])

        # Use consistent zoom factor like in notebook
        zoom_factor = 8
        oimg = zoom(oimg, zoom_factor)
        self.outputs_img = np.uint8(oimg.clip(0, 1) * 255)

    def regenerate_circuit(self):
        """Regenerate circuit completely"""
        print(f"Regenerating circuit: input_n={self.input_n}, output_n={self.output_n}")

        # Update derived values
        self.case_n = 1 << self.input_n
        self.active_case_i = min(self.active_case_i, self.case_n - 1)

        # Reinitialize circuit
        self.initialize_circuit()

        # Clear cached predictions to avoid shape mismatches
        if hasattr(self, "current_pred"):
            delattr(self, "current_pred")
        if hasattr(self, "current_pred_hard"):
            delattr(self, "current_pred_hard")

        # Update task and visualization
        self.update_task()

        # Reinitialize optimization method
        # self.initialize_optimization_method()

        # Reset optimization progress
        self.step_i = 0
        self.loss_log = np.zeros(max_trainstep_n, np.float32)
        self.hard_log = np.zeros(max_trainstep_n, np.float32)

        print("Circuit regenerated successfully")

    def mutate_wires_random(self, mutation_rate=None):
        """Mutate current circuit wires using genetic mutation with specified rate"""
        if mutation_rate is None:
            mutation_rate = self.mutation_rate

        try:
            # Generate a random key for mutation
            import random

            mutation_seed = random.randint(0, 99999)
            mutation_key = jax.random.PRNGKey(mutation_seed)

            # Apply mutation to current wires
            mutated_wires = mutate_wires_swap(self.wires, mutation_key, mutation_rate)

            # Update the circuit with mutated wires
            self.wires = mutated_wires

            # Reset optimization state but keep the same task
            self.step_i = 0
            self.loss_log = np.zeros(max_trainstep_n, np.float32)
            self.hard_log = np.zeros(max_trainstep_n, np.float32)

            # Reset logits to initial state for fair comparison
            self.logits = self.logits0

            # Reset the model generator when wires change
            self.model_generator = None
            self.last_step_result = None

            # Reinitialize optimization method for new circuit
            self.initialize_optimization_method()

            # Update gate masks for new wiring
            self.reset_gate_mask()

            # Refresh activations
            self.initialize_activations()

            print(f"Wires mutated with rate {mutation_rate} (seed: {mutation_seed})")

        except Exception as e:
            print(f"Error mutating wires: {e}")
            import traceback

            print(f"Traceback: {traceback.format_exc()}")

    def mutate_one_wire(self):
        """Mutate exactly one wire in one randomly selected layer"""
        try:
            # Generate a random key for mutation
            import random

            mutation_seed = random.randint(0, 99999)

            # Pick a random layer to mutate (skip if no layers have enough connections)
            available_layers = []
            for i, wire_layer in enumerate(self.wires):
                if wire_layer.size >= 2:  # Need at least 2 connections to swap
                    available_layers.append(i)

            if not available_layers:
                print("No layers available for mutation (need at least 2 connections per layer)")
                return

            # Choose random layer
            layer_to_mutate = random.choice(available_layers)

            # Create a copy of wires
            mutated_wires = [w.copy() for w in self.wires]

            # Mutate only the selected layer with exactly 1 swap
            layer_key = jax.random.PRNGKey(mutation_seed + layer_to_mutate)
            mutated_layer = mutate_wires_swap(
                [self.wires[layer_to_mutate]],
                layer_key,
                mutation_rate=0.0,
                n_swaps_per_layer=1,
            )
            mutated_wires[layer_to_mutate] = mutated_layer[0]

            # Update the circuit with mutated wires
            self.wires = mutated_wires

            # Reset optimization state but keep the same task
            self.step_i = 0
            self.loss_log = np.zeros(max_trainstep_n, np.float32)
            self.hard_log = np.zeros(max_trainstep_n, np.float32)

            # Reset logits to initial state for fair comparison
            self.logits = self.logits0

            # Reset the model generator when wires change
            self.model_generator = None
            self.last_step_result = None

            # Reinitialize optimization method for new circuit
            self.initialize_optimization_method()

            # Update gate masks for new wiring
            self.reset_gate_mask()

            # Refresh activations
            self.initialize_activations()

            print(f"Mutated exactly one wire in layer {layer_to_mutate} (seed: {mutation_seed})")

        except Exception as e:
            print(f"Error mutating one wire: {e}")
            import traceback

            print(f"Traceback: {traceback.format_exc()}")

    def reset_circuit(self):
        """Reset circuit to initial state"""
        self.logits = self.logits0
        self.step_i = 0
        self.loss_log = np.zeros(max_trainstep_n, np.float32)
        self.hard_log = np.zeros(max_trainstep_n, np.float32)

        # Reset the model generator when circuit is reset
        self.model_generator = None
        self.last_step_result = None

        # Reinitialize optimizer for backprop
        if self.optimization_methods[self.optimization_method_idx] == "Backprop":
            opt_fn = optax.adamw(self.learning_rate, 0.8, 0.8, weight_decay=1e-1)
            self.logit_opt_state = opt_fn.init(self.logits)
            self.logit_optimizer = opt_fn
        else:
            # Reinitialize generator for GNN/Self-Attention
            self.initialize_model_generator()

        print("Circuit reset to initial state")

    def draw_gate_lut(self, x, y, logit):
        """Draw the lookup table for a gate when hovering"""
        x0, y0 = x - 20, y - 20 - 36
        dl = imgui.get_window_draw_list()
        lut = jax.nn.sigmoid(logit).reshape(-1, 4)
        col = np.uint32(lut * 255)
        col = (col << 16) | (col << 8) | col | 0xFF000000
        for (i, j), c in np.ndenumerate(col):
            x_pos, y_pos = x0 + j * 10, y0 + i * 10
            dl.add_rect_filled((x_pos, y_pos), (x_pos + 10, y_pos + 10), c)

    def draw_circuit(self, pad=4, d=24, H=600):  # noqa: N803
        """Draw the detailed circuit visualization"""
        io = imgui.get_io()
        W = imgui.get_content_region_avail().x - pad * 2
        imgui.invisible_button("circuit", (W, H))
        base_x, base_y = imgui.get_item_rect_min()
        base_x += pad

        dl = imgui.get_window_draw_list()
        h = (H - d) / (len(self.layer_sizes) - 1) if len(self.layer_sizes) > 1 else H
        prev_gate_x = None
        prev_y = 0
        prev_act = None
        case = self.active_case_i
        hover_gate = None

        # Ensure activations exist and have correct dimensions
        if not hasattr(self, "act") or len(self.act) != len(self.layer_sizes):
            if not hasattr(self, "_activation_warning_shown"):
                print("Warning: Activations not initialized properly, creating empty activations")
                self._activation_warning_shown = True
            self.act = [np.zeros((self.case_n, size)) for size, _ in self.layer_sizes]

        # Ensure each activation layer has the right shape
        for li, (gate_n, _group_size) in enumerate(self.layer_sizes):
            if li >= len(self.act) or self.act[li].shape != (self.case_n, gate_n):
                if li >= len(self.act):
                    # Extend act list if needed
                    while len(self.act) <= li:
                        default_size = (
                            self.layer_sizes[len(self.act)][0]
                            if len(self.act) < len(self.layer_sizes)
                            else gate_n
                        )
                        self.act.append(np.zeros((self.case_n, default_size)))
                else:
                    # Reshape if needed
                    self.act[li] = np.zeros((self.case_n, gate_n))

        # Ensure wire_masks has the right length
        if len(self.wire_masks) != len(self.wires):
            print(
                f"Warning: wire_masks length mismatch. Expected {len(self.wires)}, got {len(self.wire_masks)}"
            )
            self.wire_masks = [np.ones_like(w, np.bool_) for w in self.wires]

        for li, (gate_n, group_size) in enumerate(self.layer_sizes):
            group_n = gate_n // group_size
            span_x = W / group_n if group_n > 0 else W
            group_w = min(d * group_size, span_x - 6)
            gate_w = group_w / group_size if group_size > 0 else group_w
            group_x = base_x + (np.arange(group_n)[:, None] + 0.5) * span_x
            gate_ofs = (np.arange(group_size) - group_size / 2 + 0.5) * gate_w
            gate_x = (group_x + gate_ofs).ravel()
            y = base_y + li * h + d / 2

            # Ensure we don't go out of bounds on activations
            if li < len(self.act):
                act = np.array(self.act[li][case]) if case < len(self.act[li]) else np.zeros(gate_n)
            else:
                print(f"Warning: Missing activation for layer {li}")
                act = np.zeros(gate_n)

            for i, x in enumerate(gate_x):
                a = int(act[i] * 0xA0) if i < len(act) else 0
                col = 0xFF202020 + (a << 8)
                p0, p1 = (x - gate_w / 2, y - d / 2), (x + gate_w / 2, y + d / 2)
                dl.add_rect_filled(p0, p1, col, 4)

                # Handle hover and click interactions
                if is_point_in_box(p0, p1, io.mouse_pos):
                    dl.add_rect(p0, p1, 0xA00000FF, 4, thickness=2.0)
                    if li > 0:
                        group_idx = i // group_size
                        gate_idx = i % group_size
                        if group_idx < len(self.logits[li - 1]) and gate_idx < len(
                            self.logits[li - 1][group_idx]
                        ):
                            hover_gate = (
                                x,
                                y,
                                self.logits[li - 1][group_idx, gate_idx],
                            )
                    if io.mouse_clicked[0]:
                        if li > 0:
                            if li < len(self.gate_mask) and i < len(self.gate_mask[li]):
                                self.gate_mask[li][i] = 1.0 - self.gate_mask[li][i]
                        else:
                            self.active_case_i = self.active_case_i ^ (1 << i)

                # Show masked gates
                if (
                    li < len(self.gate_mask)
                    and i < len(self.gate_mask[li])
                    and self.gate_mask[li][i] == 0.0
                ):
                    dl.add_rect_filled(p0, p1, 0xA00000FF, 4)

            # Draw group boundaries
            for x in group_x[:, 0]:
                dl.add_rect(
                    (x - group_w / 2, y - d / 2),
                    (x + group_w / 2, y + d / 2),
                    0x80FFFFFF,
                    4,
                )

            # Draw wires between layers
            if (
                li > 0
                and prev_gate_x is not None
                and li - 1 < len(self.wires)
                and li - 1 < len(self.wire_masks)
            ):
                wires = self.wires[li - 1].T
                masks = self.wire_masks[li - 1].T
                src_x = prev_gate_x[wires]
                dst_x = group_x + (np.arange(self.arity) + 0.5) / self.arity * group_w - group_w / 2
                my = (prev_y + y) / 2

                for x0, x1, si, m in zip(
                    src_x.ravel(), dst_x.ravel(), wires.ravel(), masks.ravel(), strict=False
                ):
                    if not m:
                        continue
                    activation_intensity = int(prev_act[si] * 0x60) if si < len(prev_act) else 0

                    if (
                        self.use_message_viz
                        and self.optimization_methods[self.optimization_method_idx] != "Backprop"
                    ):
                        # Colorful visualization for GNN/Self-Attention
                        import random

                        r = random.randint(0, 255)
                        g = random.randint(0, 255)
                        b = random.randint(0, 255)
                        alpha = random.randint(128, 255)  # Semi-transparent
                        col = (alpha << 24) | (r << 16) | (g << 8) | b
                    else:
                        col = 0xFF404040 + (activation_intensity << 8)

                    dl.add_bezier_cubic(
                        (x0, prev_y + d / 2),
                        (x0, my),
                        (x1, my),
                        (x1, y - d / 2),
                        col,
                        1.0,
                    )

            # Show LUT on hover
            if hover_gate is not None:
                self.draw_gate_lut(*hover_gate)

            prev_gate_x = gate_x
            prev_act = act
            prev_y = y

    def draw_lut(self, name, img, tex_id):
        """Draw visualization using ImGui"""
        try:
            view_w = imgui.get_content_region_avail().x
            img_h, img_w = img.shape[:2]

            # Debug: print image dimensions for text task
            if (
                name in ["outputs", "ground_truth"]
                and hasattr(self, "_debug_printed")
                and not self._debug_printed
            ):
                print(f"Debug {name}: img shape = {img.shape}, aspect = {img_h / img_w:.4f}")
                self._debug_printed = True

            # Simple aspect ratio based on actual image dimensions
            # This matches how the notebook displays the data
            natural_aspect = img_h / img_w

            # For text tasks with very wide, short images, we need to respect
            # the true aspect ratio to show the full 256×8 data properly
            if natural_aspect < 0.05:  # Very wide image (like 64×2048)
                # Use the natural aspect ratio but ensure it's visible
                aspect = max(0.03, natural_aspect)  # Allow very wide aspect ratios
            elif natural_aspect < 0.2:  # Moderately wide
                aspect = max(0.1, natural_aspect)
            else:
                aspect = max(0.1, min(natural_aspect, 1.0))

            disp_w = view_w
            disp_h = disp_w * aspect

            # Draw visualization
            dl = imgui.get_window_draw_list()
            p0 = imgui.get_cursor_screen_pos()
            p1 = (p0[0] + disp_w, p0[1] + disp_h)

            # Background
            dl.add_rect_filled(p0, p1, 0xFF333333, 4.0)

            if self.use_simple_viz:
                # Simple line visualization
                case_width = disp_w / self.case_n
                for i in range(self.case_n):
                    x_pos = p0[0] + i * case_width
                    is_active = i == self.active_case_i

                    # Sample color from middle row
                    middle_y = img_h // 2
                    if len(img.shape) == 3 and img.shape[2] >= 3:
                        r, g, b = [int(v) for v in img[middle_y, i % img_w, 0:3]]
                        r, g, b = r & 0xFF, g & 0xFF, b & 0xFF
                        color = 0xFF000000 | (b << 16) | (g << 8) | r
                    else:
                        v = int(img[middle_y, i % img_w]) & 0xFF
                        color = 0xFF000000 | (v << 16) | (v << 8) | v

                    # Draw line
                    dl.add_line((x_pos, p0[1]), (x_pos, p1[1]), color, 2.0 if is_active else 1.0)

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
                if self.use_full_resolution:
                    # Full resolution mode - show every pixel (slower but more detailed)
                    x_step = 1
                    y_step = 1
                else:
                    # Performance mode - intelligent downsampling that preserves text readability
                    # For very wide images (like text), preserve more horizontal detail
                    aspect_ratio = img_h / img_w

                    if aspect_ratio < 0.1:  # Very wide image (likely text)
                        # Preserve horizontal resolution for text readability
                        max_horizontal_samples = min(256, img_w // 4)  # Sample every 4th pixel
                        max_vertical_samples = min(64, img_h)  # Full vertical resolution
                        x_step = max(1, img_w // max_horizontal_samples)
                        y_step = max(1, img_h // max_vertical_samples)
                    else:
                        # Regular images - use original 64x64 approach
                        max_blocks = 64
                        x_step = max(1, img_w // max_blocks)
                        y_step = max(1, img_h // max_blocks)

                for y in range(0, img_h, y_step):
                    for x in range(0, img_w, x_step):
                        px = p0[0] + (x / img_w) * disp_w
                        py = p0[1] + (y / img_h) * disp_h
                        px_end = p0[0] + ((x + x_step) / img_w) * disp_w
                        py_end = p0[1] + ((y + y_step) / img_h) * disp_h

                        # Get color
                        if len(img.shape) == 3 and img.shape[2] >= 3:
                            r, g, b = [int(v) for v in img[y, x, 0:3]]
                            r, g, b = r & 0xFF, g & 0xFF, b & 0xFF
                            color = 0xFF000000 | (b << 16) | (g << 8) | r
                        else:
                            v = int(img[y, x]) & 0xFF
                            color = 0xFF000000 | (v << 16) | (v << 8) | v

                        dl.add_rect_filled((px, py), (px_end, py_end), color)

            # Active case cursor
            x = p0[0] + (disp_w * (self.active_case_i + 0.5) / self.case_n)
            dl.add_line((x, p0[1]), (x, p1[1]), 0x8000FF00, 2.0)

            # Border
            dl.add_rect(p0, p1, 0xFFFFFFFF, 4.0)

            # Make clickable
            imgui.invisible_button(f"{name}_area", (disp_w, disp_h))
            if imgui.is_item_hovered() and imgui.is_mouse_clicked(0):
                mx = imgui.get_io().mouse_pos.x - p0[0]
                mx_ratio = mx / disp_w
                self.active_case_i = max(0, min(int(mx_ratio * self.case_n), self.case_n - 1))

            # Reserve space
            imgui.dummy((0, disp_h))

        except Exception as e:
            imgui.text(f"Error drawing {name}: {e}")

    def gui(self):
        """Main GUI function"""
        try:
            # Initialize ImGui textures if not already done
            self.initialize_imgui_textures()

            # Perform one optimization step
            self.optimize_circuit()

            # Configure FPS
            runner_params = hello_imgui.get_runner_params()
            runner_params.fps_idling.enable_idling = True

            # Main content area
            imgui.begin_child("main", (-300, 0))

            # Optimization progress plot
            if implot.begin_plot("Circuit Optimization Progress", (-1, 200)):
                implot.setup_legend(implot.Location_.north_east.value)
                implot.setup_axis_scale(implot.ImAxis_.y1.value, implot.Scale_.log10.value)
                implot.setup_axes(
                    "Step",
                    "Loss",
                    implot.AxisFlags_.auto_fit.value,
                    implot.AxisFlags_.auto_fit.value,
                )
                implot.setup_axis_limits(
                    implot.ImAxis_.y1.value, self.min_loss_value, self.max_loss_value
                )

                # Plot lines based on display mode
                display_mode = self.loss_display_modes[self.loss_display_mode_idx]
                if display_mode in ["Both", "Soft Only"]:
                    implot.plot_line("soft_loss", self.loss_log)
                if display_mode in ["Both", "Hard Only"]:
                    implot.plot_line("hard_loss", self.hard_log)

                implot.drag_line_x(1, self.step_i % len(self.loss_log), (0.8, 0, 0, 0.5))

                # Right-click context menu for loss display options
                if implot.is_plot_hovered() and imgui.is_mouse_clicked(1):  # Right click
                    imgui.open_popup("loss_display_menu")

                if imgui.begin_popup("loss_display_menu"):
                    imgui.text("Loss Display Options")
                    imgui.separator()

                    for i, mode in enumerate(self.loss_display_modes):
                        selected = i == self.loss_display_mode_idx
                        if imgui.selectable(mode, selected)[0]:
                            self.loss_display_mode_idx = i
                            print(f"Loss display mode changed to: {mode}")

                    imgui.end_popup()

                implot.end_plot()

            # Input visualization
            imgui.separator_text("Inputs")
            self.draw_lut("inputs", self.inputs_img, self.input_texture)

            # Circuit visualization
            imgui.separator_text("Circuit")
            H = imgui.get_content_region_avail().y - 400  # Leave room for outputs below
            self.draw_circuit(H=max(H, 300))  # Minimum height of 300

            # Output vs Ground Truth
            imgui.separator_text("Current Output")
            self.draw_lut("outputs", self.outputs_img, self.output_texture)

            imgui.separator_text("Expected Output")
            self.draw_lut("ground_truth", self.ground_truth_img, self.ground_truth_texture)
            imgui.end_child()
            imgui.same_line()

            # Control panel
            imgui.begin_child("controls")

            if imgui.button("Python REPL"):
                IPython.embed()

            # Optimization controls
            imgui.separator_text("Circuit Optimization")

            # Play/Pause button for optimization
            if self.is_optimizing:
                if imgui.button("⏸️ Pause", (120, 0)):
                    self.is_optimizing = False
            else:
                if imgui.button("▶️ Play", (120, 0)):
                    self.is_optimizing = True

            imgui.same_line()
            imgui.text("Optimization" if self.is_optimizing else "Paused")

            if imgui.button("Reset Circuit"):
                self.reset_circuit()

            # Optimization method
            opt_changed, self.optimization_method_idx = imgui.combo(
                "Method", self.optimization_method_idx, self.optimization_methods
            )
            if opt_changed:
                print(f"Switching to {self.optimization_methods[self.optimization_method_idx]}")
                self.initialize_optimization_method()

            # Method-specific controls
            method_name = self.optimization_methods[self.optimization_method_idx]

            if method_name == "Backprop":
                imgui.text("Direct gradient-based logit optimization")
                _, self.learning_rate = imgui.slider_float(
                    "Learning Rate",
                    self.learning_rate,
                    1e-5,
                    1e-1,
                    "%.5f",
                    imgui.SliderFlags_.logarithmic.value,
                )

            elif method_name in ["GNN", "Self-Attention"]:
                _, self.n_message_steps = imgui.slider_int(
                    "Message Steps", self.n_message_steps, 1, 10
                )

                # Show model status
                if self.frozen_model is not None:
                    imgui.text_colored(
                        imgui.ImVec4(0.0, 1.0, 0.0, 1.0),
                        f"✓ Frozen {method_name} model loaded",
                    )
                    imgui.text("Model suggests logit improvements")
                else:
                    imgui.text_colored(
                        imgui.ImVec4(1.0, 0.0, 0.0, 1.0), f"✗ No {method_name} model"
                    )

                # WandB integration
                imgui.separator_text("Load Frozen Model")
                run_id_buffer = self.run_id if self.run_id else ""
                changed, run_id_buffer = imgui.input_text("Run ID", run_id_buffer, 256)
                if changed:
                    self.run_id = run_id_buffer if run_id_buffer else None

                if imgui.button("Load from WandB"):
                    if self.try_load_wandb_model():
                        print(f"Successfully loaded frozen {method_name} model")
                    else:
                        print(f"Failed to load {method_name} model")

                if self.loaded_run_id:
                    imgui.text_colored(
                        imgui.ImVec4(0.0, 1.0, 0.0, 1.0),
                        f"Loaded: {self.loaded_run_id}",
                    )

            # Circuit architecture
            imgui.separator_text("Circuit Architecture")

            orig_input_n = self.input_n
            orig_output_n = self.output_n
            orig_arity = self.arity
            orig_layer_n = self.layer_n

            _, self.input_n = imgui.slider_int("Input Bits", self.input_n, 2, 8)
            _, self.output_n = imgui.slider_int("Output Bits", self.output_n, 2, 8)
            _, self.arity = imgui.slider_int("Gate Arity", self.arity, 2, 4)
            _, self.layer_n = imgui.slider_int("Hidden Layers", self.layer_n, 1, 4)

            if (
                self.input_n != orig_input_n
                or self.output_n != orig_output_n
                or self.arity != orig_arity
                or self.layer_n != orig_layer_n
            ):
                try:
                    self.regenerate_circuit()
                    self.initialize_optimization_method()
                except Exception as e:
                    print(f"Error regenerating circuit: {e}")
                    # Revert
                    self.input_n = orig_input_n
                    self.output_n = orig_output_n
                    self.arity = orig_arity
                    self.layer_n = orig_layer_n

            if imgui.button("Regenerate Circuit"):
                self.regenerate_circuit()

            # Wiring configuration
            imgui.separator_text("Wiring")
            wiring_changed, self.wiring_mode_idx = imgui.combo(
                "Wiring Mode", self.wiring_mode_idx, self.wiring_modes
            )
            if wiring_changed:
                self.wiring_mode = self.wiring_modes[self.wiring_mode_idx]
                self.regenerate_circuit()  # This will invalidate cache

            # Wiring seed control
            seed_changed, new_seed = imgui.input_int("Wiring Seed", self.wiring_seed)
            if seed_changed:
                self.wiring_seed = max(0, new_seed)  # Ensure non-negative
                self.wiring_key = jax.random.PRNGKey(self.wiring_seed)
                self.regenerate_circuit()

            if imgui.button("Reset Seed (42)"):
                self.wiring_seed = 42
                self.wiring_key = jax.random.PRNGKey(self.wiring_seed)
                self.regenerate_circuit()

            imgui.same_line()
            if imgui.button("Shuffle Wires"):
                # Generate a random seed
                import random

                self.wiring_seed = random.randint(0, 99999)
                self.wiring_key = jax.random.PRNGKey(self.wiring_seed)
                self.regenerate_circuit()  # This will invalidate cache

            # Mutation controls
            imgui.separator()

            # Mutation rate slider
            _, self.mutation_rate = imgui.slider_float(
                "Mutation Rate", self.mutation_rate, 0.01, 0.5, "%.3f"
            )

            # Mutation buttons
            if imgui.button("Mutate Random"):
                # Apply genetic mutation to current wires using the slider value
                self.mutate_wires_random()

            imgui.same_line()
            if imgui.button("Mutate One"):
                # Mutate exactly one wire in one random layer
                self.mutate_one_wire()

            # Task selection
            imgui.separator_text("Task")
            task_changed, self.task_idx = imgui.combo("Task", self.task_idx, self.available_tasks)
            if task_changed:
                self.update_task()
                self.initialize_optimization_method()

            # Task-specific controls
            task_name = self.available_tasks[self.task_idx]
            if task_name == "text":
                text_changed, self.task_text = imgui.input_text("Text", self.task_text)
                if text_changed:
                    self.update_task()
            elif task_name == "noise":
                noise_changed, self.noise_p = imgui.slider_float("Noise p", self.noise_p, 0.0, 1.0)
                if noise_changed:
                    self.update_task()

            # Loss type
            imgui.separator_text("Loss Function")
            loss_types = ["l4", "l2", "bce"]
            loss_idx = loss_types.index(self.loss_type) if self.loss_type in loss_types else 0
            loss_changed, loss_idx = imgui.combo("Loss Type", loss_idx, loss_types)
            if loss_changed:
                self.loss_type = loss_types[loss_idx]

            # Visualization controls
            imgui.separator_text("Visualization")
            _, self.use_simple_viz = imgui.checkbox("Simple visualization", self.use_simple_viz)
            _, self.use_message_viz = imgui.checkbox("Message visualization", self.use_message_viz)
            _, self.use_full_resolution = imgui.checkbox(
                "Full resolution (slower)", self.use_full_resolution
            )
            _, self.auto_scale_plot = imgui.checkbox("Auto-scale plot", self.auto_scale_plot)

            # Circuit gate mask controls
            imgui.separator_text("Circuit Masks")
            if imgui.button("Reset Gate Mask"):
                self.reset_gate_mask()
            imgui.same_line()
            if imgui.button("Mask Unused Gates"):
                self.mask_unused_gates()

            # Show active gate count
            if hasattr(self, "gate_mask") and len(self.gate_mask) > 0:
                active_gate_n = int(sum(m.sum() for m in self.gate_mask))
                imgui.text(f"Active gates: {active_gate_n}")

            # Status information
            imgui.separator_text("Status")
            imgui.text(f"Method: {method_name}")
            imgui.text(f"Circuit Parameters: {sum(logit.size for logit in self.logits0)}")
            imgui.text(f"Optimization Step: {self.step_i}")
            imgui.text(f"Active Input Case: {self.active_case_i}")
            imgui.text(f"Wiring Seed: {self.wiring_seed}")
            imgui.text(f"Wiring Mode: {self.wiring_mode}")
            imgui.text(f"Loss Display: {self.loss_display_modes[self.loss_display_mode_idx]}")

            # Model-specific status
            if method_name == "Self-Attention" and self.frozen_model is not None:
                imgui.text(f"Model hidden_dim: {self.model_hidden_dim}")
                imgui.text(f"Model use_globals: {self.model_use_globals}")
            elif method_name == "GNN" and self.frozen_model is not None:
                imgui.text(f"Model hidden_dim: {self.model_hidden_dim}")

            if hasattr(self, "current_pred_hard"):
                try:
                    # Check shape compatibility before calculating accuracy
                    if (
                        hasattr(self, "current_pred")
                        and self.current_pred.shape == self.y0.shape
                        and self.current_pred_hard.shape == self.y0.shape
                    ):
                        accuracy = float(jp.mean(jp.round(self.current_pred) == self.y0))
                        hard_accuracy = float(jp.mean(self.current_pred_hard == self.y0))
                        imgui.text(f"Soft Accuracy: {accuracy:.3f}")
                        imgui.text(f"Hard Accuracy: {hard_accuracy:.3f}")
                    else:
                        imgui.text("Accuracy: Computing...")
                except Exception as e:
                    imgui.text(f"Accuracy: Error - {str(e)[:30]}...")

            imgui.end_child()

        except Exception as e:
            print(f"Exception in GUI: {e}")
            import traceback

            print(f"Traceback: {traceback.format_exc()}")

    def initialize_activations(self):
        """Run circuit once to generate initial activations"""
        try:
            # Make sure we have input data
            if not hasattr(self, "input_x") or not hasattr(self, "y0"):
                # Create default input data
                x = jp.arange(self.case_n)
                self.input_x = unpack(x, bit_n=self.input_n)
                self.y0 = jp.zeros((self.case_n, self.output_n))

            # Run circuit to get layer-by-layer activations
            # This returns [input_acts, layer1_acts, layer2_acts, ..., output_acts]
            self.act = run_circuit(
                self.logits, self.wires, self.input_x, hard=False, gate_mask=self.gate_mask
            )

            # Generate error mask for visualization - use final output from activations
            final_output = self.act[-1] if self.act else jp.zeros_like(self.y0)
            self.err_mask = (final_output > 0.5) != self.y0

        except Exception as e:
            print(f"Warning: Could not generate initial circuit activations: {e}")
            # Fallback: create empty activations
            self.act = [np.zeros((self.case_n, size)) for size, _ in self.layer_sizes]
            self.err_mask = np.zeros((self.case_n, self.output_n), bool)


if __name__ == "__main__":
    try:
        print("Starting Circuit Optimization Demo...")
        print("- Backprop: Direct gradient-based logit optimization")
        print("- GNN/Self-Attention: Frozen models suggest logit improvements")

        demo = CircuitOptimizationDemo()

        immapp.run(
            demo.gui,
            window_title="Circuit Optimization Demo",
            window_size=(1200, 800),
            fps_idle=10,
            with_implot=True,
        )
    except Exception as e:
        print(f"Error running demo: {e}")
        import traceback

        print(f"Traceback: {traceback.format_exc()}")
