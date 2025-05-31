#!/usr/bin/env python3
"""
Training-compatible boolean circuits demo using shared infrastructure.

This demo shows live circuit optimization where:
- Backprop: Direct gradient-based optimization of circuit logits
- GNN/Self-Attention: Pre-trained models suggest logit improvements (frozen models)

No model training occurs - only circuit logit optimization.
"""

import numpy as np
import jax
import jax.numpy as jp
import optax
import PIL.Image, PIL.ImageDraw
import IPython
from flax import nnx

# Import shared training infrastructure
from boolean_nca_cc.circuits.model import gen_circuit
from boolean_nca_cc.circuits.tasks import get_task_data, TASKS
from boolean_nca_cc import generate_layer_sizes

# Import training loop functions
from boolean_nca_cc.training.train_loop import get_loss_from_graph
from boolean_nca_cc.training.utils import load_best_model_from_wandb

# Import model components
from boolean_nca_cc.utils import build_graph, extract_logits_from_graph

from imgui_bundle import (
    implot,
    imgui,
    immapp,
    hello_imgui,
)


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
        self.input_n = 4
        self.output_n = 4
        self.arity = 2
        self.layer_n = 3
        self.hidden_dim = 64

        # Update case_n based on input_n
        self.case_n = 1 << self.input_n

        # Wiring configuration
        self.wiring_modes = ["fixed", "random"]
        self.wiring_mode_idx = 0
        self.wiring_mode = self.wiring_modes[self.wiring_mode_idx]
        self.wiring_key = jax.random.PRNGKey(42)

        # Optimization configuration
        self.loss_type = "l4"
        self.learning_rate = 1e-3
        self.n_message_steps = 1

        # Initialize circuit using shared functions
        self.initialize_circuit()

        # Task configuration
        self.available_tasks = list(TASKS.keys()) + ["text", "noise"]
        self.task_idx = 0
        self.task_text = "Hello Neural CA"
        self.noise_p = 0.5
        self.sample_noise()
        self.update_task()

        # Optimization state
        self.step_i = 0
        self.is_optimizing = True
        self.loss_log = np.zeros(max_trainstep_n, np.float32)
        self.hard_log = np.zeros(max_trainstep_n, np.float32)

        # Optimization method configuration
        self.optimization_methods = ["Backprop", "GNN", "Self-Attention"]
        self.optimization_method_idx = 0

        # Model instances (only pre-trained, frozen models)
        self.frozen_model = None
        self.logit_optimizer = None  # Only for backprop

        # Graph caching for efficiency
        self.cached_graph_structure = None
        self.graph_cache_valid = False

        # Visualization settings
        self.use_simple_viz = False
        self.max_loss_value = 10.0
        self.min_loss_value = 1e-6
        self.auto_scale_plot = True

        # WandB integration
        self.wandb_entity = "m2snn"
        self.wandb_project = "boolean-nca-cc"
        self.wandb_download_dir = "saves"
        self.run_id = None
        self.loaded_run_id = None

        # Initialize visualization
        self.setup_visualization()

        # Initialize optimization method
        self.initialize_optimization_method()

    def initialize_circuit(self):
        """Initialize circuit using shared infrastructure"""
        # Generate layer sizes using shared function
        self.layer_sizes = generate_layer_sizes(
            self.input_n, self.output_n, self.arity, self.layer_n
        )

        # Generate circuit using shared function
        self.wires, self.logits = gen_circuit(
            self.wiring_key, self.layer_sizes, arity=self.arity
        )

        # Store initial logits
        self.logits0 = self.logits

        print(
            f"Circuit initialized with {sum(l.size for l in self.logits0)} parameters"
        )
        print(f"Layer structure: {self.layer_sizes}")

    def sample_noise(self):
        """Sample noise for noise task"""
        self.noise = np.random.rand(self.case_n, self.output_n)

    def update_task(self):
        """Update current task using shared task infrastructure"""
        task_name = self.available_tasks[self.task_idx]

        if task_name == "text":
            # Text-based task
            im = PIL.Image.new("L", (self.case_n, self.output_n))
            draw = PIL.ImageDraw.Draw(im)
            draw.text((2, -2), self.task_text, fill=255)
            self.y0 = jp.float32(np.array(im) > 100).T

            x = jp.arange(self.case_n)
            self.input_x = unpack(x, bit_n=self.input_n)

        elif task_name == "noise":
            # Noise-based task
            if self.noise.shape != (self.case_n, self.output_n):
                self.sample_noise()
            self.y0 = jp.float32(self.noise < self.noise_p)

            x = jp.arange(self.case_n)
            self.input_x = unpack(x, bit_n=self.input_n)

        else:
            # Use shared task infrastructure
            try:
                self.input_x, self.y0 = get_task_data(
                    task_name,
                    self.case_n,
                    input_bits=self.input_n,
                    output_bits=self.output_n,
                )
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

        # Update visualization
        self.setup_visualization()

    def setup_visualization(self):
        """Setup visualization using shared functions"""
        # Create input visualization
        inp_img = self.input_x.T
        zoom_factor = max(4, int(8 // self.input_n * 2))
        inp_img = np.dstack([inp_img] * 3)
        inp_img = zoom(inp_img, zoom_factor)
        self.inputs_img = np.uint8(inp_img.clip(0, 1) * 255)

        # Create ground truth visualization
        gt_img = self.y0.T
        gt_img = np.dstack([gt_img] * 3)
        gt_img = zoom(gt_img, zoom_factor)
        self.ground_truth_img = np.uint8(gt_img.clip(0, 1) * 255)

        # Initialize output image placeholder
        self.outputs_img = np.zeros_like(self.ground_truth_img)

        # Initialize textures
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

        # Initialize active case
        self.active_case_i = 123 % self.case_n

    def initialize_optimization_method(self):
        """Initialize the selected optimization method"""
        method_name = self.optimization_methods[self.optimization_method_idx]

        if method_name == "Backprop":
            # Create optimizer for logits only
            opt_fn = optax.adamw(self.learning_rate)
            self.logit_optimizer = nnx.Optimizer(self.logits, opt_fn)
            self.frozen_model = None

        elif method_name in ["GNN", "Self-Attention"]:
            # Try to load pre-trained frozen model
            if self.try_load_wandb_model():
                print(f"Loaded frozen {method_name} model from WandB")
                self.logit_optimizer = None  # No optimizer needed for frozen models
                # Invalidate cache when switching to GNN/SA (different hidden_dim requirements)
                self.invalidate_graph_cache()
            else:
                print(f"Could not load {method_name} model. Falling back to Backprop.")
                self.optimization_method_idx = 0
                self.initialize_optimization_method()
                return

    def try_load_wandb_model(self):
        """Try to load frozen model from WandB"""
        try:
            method_name = self.optimization_methods[self.optimization_method_idx]
            model_type = "gnn" if method_name == "GNN" else "self_attention"

            filters = {
                "config.circuit.input_bits": self.input_n,
                "config.circuit.output_bits": self.output_n,
                "config.circuit.arity": self.arity,
                "config.circuit.num_layers": self.layer_n,
                "config.model.type": model_type,
                "config.training.wiring_mode": self.wiring_mode,
            }

            # Load frozen model
            model, loaded_dict = load_best_model_from_wandb(
                run_id=self.run_id,
                filters=filters if not self.run_id else None,
                seed=42,
                project=self.wandb_project,
                entity=self.wandb_entity,
                download_dir=self.wandb_download_dir,
            )

            self.frozen_model = model
            self.loaded_run_id = loaded_dict.get("run_id", "unknown")

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
                loss, hard_loss = self.optimize_with_frozen_model()

            # Update loss logs
            i = self.step_i % len(self.loss_log)
            self.loss_log[i] = max(
                min(float(loss), self.max_loss_value), self.min_loss_value
            )
            self.hard_log[i] = max(
                min(float(hard_loss), self.max_loss_value), self.min_loss_value
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
        # Calculate loss and get predictions for visualization
        loss, (hard_loss, pred, pred_hard) = get_loss_from_graph(
            self.logits, self.wires, self.input_x, self.y0, self.loss_type
        )

        if self.is_optimizing:
            # Compute gradients with respect to logits
            def loss_fn(logits):
                loss, _ = get_loss_from_graph(
                    logits, self.wires, self.input_x, self.y0, self.loss_type
                )
                return loss

            grad_fn = jax.grad(loss_fn)
            grads = grad_fn(self.logits)

            # Update logits using optimizer
            self.logit_optimizer.update(grads)
            self.logits = self.logit_optimizer.model

        # Store predictions for visualization
        self.current_pred = pred
        self.current_pred_hard = pred_hard

        return loss, hard_loss

    def optimize_with_frozen_model(self):
        """Use frozen GNN/Self-Attention model to suggest logit improvements"""
        if self.frozen_model is None:
            print("No frozen model loaded, falling back to backprop")
            self.optimization_method_idx = 0
            self.initialize_optimization_method()
            return self.optimize_backprop()

        try:
            # Calculate initial loss
            initial_loss, _ = get_loss_from_graph(
                self.logits, self.wires, self.input_x, self.y0, self.loss_type
            )

            # Build graph from current circuit
            circuit_graph = self.get_or_build_graph_structure()

            # Store original logit shapes for reconstruction
            logits_original_shapes = [logit.shape for logit in self.logits]

            if self.is_optimizing:
                # Use frozen model to suggest improvements (like evaluate_model_stepwise)
                updated_graph, _ = self.update_graph_with_current_state(
                    circuit_graph, initial_loss
                )

                # Apply frozen model for n_message_steps
                for _ in range(self.n_message_steps):
                    updated_graph = self.frozen_model(updated_graph)

                # Extract improved logits
                improved_logits = extract_logits_from_graph(
                    updated_graph, logits_original_shapes
                )

                # Update circuit logits with improvements
                self.logits = improved_logits

            # Calculate final loss and predictions for visualization
            loss, (hard_loss, pred, pred_hard) = get_loss_from_graph(
                self.logits, self.wires, self.input_x, self.y0, self.loss_type
            )

            # Store predictions for visualization
            self.current_pred = pred
            self.current_pred_hard = pred_hard

            return loss, hard_loss

        except Exception as e:
            print(f"Error with frozen model: {e}")
            # Fallback to backprop
            self.optimization_method_idx = 0
            self.initialize_optimization_method()
            return self.optimize_backprop()

    def update_output_visualization(self):
        """Update output visualization based on current predictions"""
        if not hasattr(self, "current_pred_hard"):
            return

        # Create output visualization
        oimg = self.current_pred.T
        oimg = np.dstack([oimg] * 3)

        # Apply error mask for visualization
        err_mask = (self.current_pred_hard != self.y0).T
        m = err_mask[..., None] * 0.5
        oimg = oimg * (1.0 - m) + m * np.float32([1, 0, 0])

        # Apply zoom
        zoom_factor = max(4, int(8 // self.output_n * 2))
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

        # Invalidate graph cache since circuit structure changed
        self.invalidate_graph_cache()

        # Update task and visualization
        self.sample_noise()
        self.update_task()

        # Reinitialize optimization method
        self.initialize_optimization_method()

        # Reset optimization progress
        self.step_i = 0
        self.loss_log = np.zeros(max_trainstep_n, np.float32)
        self.hard_log = np.zeros(max_trainstep_n, np.float32)

        print("Circuit regenerated successfully")

    def reset_circuit(self):
        """Reset circuit to initial state"""
        self.logits = self.logits0
        self.step_i = 0
        self.loss_log = np.zeros(max_trainstep_n, np.float32)
        self.hard_log = np.zeros(max_trainstep_n, np.float32)

        # Invalidate graph cache since logits changed significantly
        self.invalidate_graph_cache()

        # Reinitialize optimizer for backprop
        if self.optimization_methods[self.optimization_method_idx] == "Backprop":
            opt_fn = optax.adamw(self.learning_rate)
            self.logit_optimizer = nnx.Optimizer(self.logits, opt_fn)

        print("Circuit reset to initial state")

    def draw_lut(self, name, img, tex_id):
        """Draw visualization using ImGui"""
        try:
            view_w = imgui.get_content_region_avail().x
            img_h, img_w = img.shape[:2]

            # Calculate aspect ratio
            if name == "inputs":
                natural_aspect = self.input_n / (2**self.input_n)
                reference_aspect = 8.0 / (2**8)
                aspect = natural_aspect * (reference_aspect / natural_aspect) * 3.0
            else:
                natural_aspect = self.output_n / (2**self.output_n)
                reference_aspect = 8.0 / (2**8)
                aspect = natural_aspect * (reference_aspect / natural_aspect) * 3.0

            # Clamp aspect ratio
            aspect = max(0.02, min(aspect, 0.3))

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
                # Block visualization
                block_w = disp_w / img_w
                block_h = disp_h / img_h

                # Limit resolution for performance
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
                self.active_case_i = max(
                    0, min(int(mx_ratio * self.case_n), self.case_n - 1)
                )

            # Reserve space
            imgui.dummy((0, disp_h))

        except Exception as e:
            imgui.text(f"Error drawing {name}: {e}")

    def gui(self):
        """Main GUI function"""
        try:
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
                implot.setup_axis_scale(
                    implot.ImAxis_.y1.value, implot.Scale_.log10.value
                )
                implot.setup_axes(
                    "Step",
                    "Loss",
                    implot.AxisFlags_.auto_fit.value,
                    implot.AxisFlags_.auto_fit.value,
                )
                implot.setup_axis_limits(
                    implot.ImAxis_.y1.value, self.min_loss_value, self.max_loss_value
                )

                implot.plot_line("soft_loss", self.loss_log)
                implot.plot_line("hard_loss", self.hard_log)
                implot.drag_line_x(
                    1, self.step_i % len(self.loss_log), (0.8, 0, 0, 0.5)
                )
                implot.end_plot()

            # Input visualization
            imgui.separator_text("Inputs")
            self.draw_lut("inputs", self.inputs_img, self.input_texture)

            # Output vs Ground Truth
            imgui.separator_text("Outputs vs Ground Truth")
            imgui.columns(2, "output_columns")

            imgui.text("Current Output")
            self.draw_lut("outputs", self.outputs_img, self.output_texture)

            imgui.next_column()
            imgui.text("Expected Output")
            self.draw_lut(
                "ground_truth", self.ground_truth_img, self.ground_truth_texture
            )

            imgui.columns(1)
            imgui.end_child()
            imgui.same_line()

            # Control panel
            imgui.begin_child("controls")

            if imgui.button("Python REPL"):
                IPython.embed()

            # Optimization controls
            imgui.separator_text("Circuit Optimization")
            _, self.is_optimizing = imgui.checkbox(
                "Enable Optimization", self.is_optimizing
            )

            if imgui.button("Reset Circuit"):
                self.reset_circuit()

            # Optimization method
            opt_changed, self.optimization_method_idx = imgui.combo(
                "Method", self.optimization_method_idx, self.optimization_methods
            )
            if opt_changed:
                print(
                    f"Switching to {self.optimization_methods[self.optimization_method_idx]}"
                )
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

            if imgui.button("Shuffle Wires"):
                self.wiring_key = jax.random.split(self.wiring_key)[0]
                self.regenerate_circuit()  # This will invalidate cache

            # Task selection
            imgui.separator_text("Task")
            task_changed, self.task_idx = imgui.combo(
                "Task", self.task_idx, self.available_tasks
            )
            if task_changed:
                self.update_task()

            # Task-specific controls
            task_name = self.available_tasks[self.task_idx]
            if task_name == "text":
                text_changed, self.task_text = imgui.input_text("Text", self.task_text)
                if text_changed:
                    self.update_task()
            elif task_name == "noise":
                noise_changed, self.noise_p = imgui.slider_float(
                    "Noise p", self.noise_p, 0.0, 1.0
                )
                if noise_changed:
                    self.update_task()

            # Loss type
            imgui.separator_text("Loss Function")
            loss_types = ["l4", "l2", "bce"]
            loss_idx = (
                loss_types.index(self.loss_type) if self.loss_type in loss_types else 0
            )
            loss_changed, loss_idx = imgui.combo("Loss Type", loss_idx, loss_types)
            if loss_changed:
                self.loss_type = loss_types[loss_idx]

            # Visualization controls
            imgui.separator_text("Visualization")
            _, self.use_simple_viz = imgui.checkbox(
                "Simple visualization", self.use_simple_viz
            )
            _, self.auto_scale_plot = imgui.checkbox(
                "Auto-scale plot", self.auto_scale_plot
            )

            # Status information
            imgui.separator_text("Status")
            imgui.text(f"Method: {method_name}")
            imgui.text(f"Circuit Parameters: {sum(l.size for l in self.logits0)}")
            imgui.text(f"Optimization Step: {self.step_i}")
            imgui.text(f"Active Input Case: {self.active_case_i}")
            if hasattr(self, "current_pred_hard"):
                accuracy = float(jp.mean(jp.round(self.current_pred) == self.y0))
                hard_accuracy = float(jp.mean(self.current_pred_hard == self.y0))
                imgui.text(f"Soft Accuracy: {accuracy:.3f}")
                imgui.text(f"Hard Accuracy: {hard_accuracy:.3f}")

            imgui.end_child()

        except Exception as e:
            print(f"Exception in GUI: {e}")
            import traceback

            print(f"Traceback: {traceback.format_exc()}")

    def invalidate_graph_cache(self):
        """Invalidate the cached graph structure"""
        self.graph_cache_valid = False
        self.cached_graph_structure = None
        print("Graph cache invalidated - will rebuild on next optimization step")

    def get_or_build_graph_structure(self):
        """Get cached graph structure or build new one if invalid"""
        if not self.graph_cache_valid or self.cached_graph_structure is None:
            print("Building new graph structure...")
            # Build the base graph structure (this is expensive)
            self.cached_graph_structure = build_graph(
                logits=self.logits,
                wires=self.wires,
                input_n=self.input_n,
                arity=self.arity,
                hidden_dim=self.hidden_dim,
                loss_value=0.0,  # Placeholder, will be updated
                update_steps=0,  # Placeholder, will be updated
            )
            self.graph_cache_valid = True
            print("Graph structure cached")

        return self.cached_graph_structure

    def update_graph_with_current_state(self, base_graph, loss_value):
        """Update cached graph with current logits, loss, and step count"""
        # Extract logits and flatten them to update node features
        logits_original_shapes = [logit.shape for logit in self.logits]

        # Flatten logits to create node features (same as build_graph does)
        flattened_logits = []
        for logit_layer in self.logits:
            flattened_logits.append(logit_layer.flatten())
        all_logits = jp.concatenate(flattened_logits)

        # Pad logits to hidden_dim size per node
        n_nodes = len(all_logits)
        if self.hidden_dim > 1:
            # Create features of size hidden_dim per node
            node_features = jp.zeros((n_nodes, self.hidden_dim))
            node_features = node_features.at[:, 0].set(
                all_logits
            )  # Put logits in first feature
        else:
            node_features = all_logits.reshape(-1, 1)

        # Update the graph with new node features and globals
        updated_graph = base_graph._replace(
            nodes={"logits": node_features},
            globals=jp.array([loss_value, float(self.step_i)], dtype=jp.float32),
        )

        return updated_graph, logits_original_shapes


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
