#!/usr/bin/env python3
"""
Minimal GUI for circuit optimization with Self-Attention and Backprop.

This minimal demo shows live circuit optimization where:
- Backprop: Direct gradient-based optimization of circuit logits
- Self-Attention: Pre-trained models suggest logit improvements (frozen models)
- GAMMA RAYS: Reversible damage perturbation visualization

No model training occurs - only circuit logit optimization.
"""

import logging
import yaml

import IPython
import jax
import jax.numpy as jp
import numpy as np
import optax
from flax import nnx
from omegaconf import OmegaConf

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

# Import training loop functions
from boolean_nca_cc.training.checkpointing import (
    load_config_from_wandb,
    load_model_from_config_and_checkpoint,
)
from boolean_nca_cc.training.evaluation import (
    evaluate_model_stepwise_generator,
    get_loss_from_wires_logits,
)
from boolean_nca_cc.training.preconfigure import (
    preconfigure_circuit_logits,
)

# Import structural perturbation utilities for GAMMA RAYS mode
from boolean_nca_cc.training.pool.structural_perturbation import (
    create_greedy_subset_random_pattern,
    DEFAULT_GREEDY_ORDERED_INDICES,
)
from boolean_nca_cc.circuits.train import create_gate_mask_from_knockout_pattern


# Configure logging to show INFO messages
logging.basicConfig(level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s")

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


# DEBUG BLOCK: Extract scale parameters from loaded model (for re_zero_update models)
def _extract_scale_parameter(model, param_name: str) -> float | None:
    """
    Extract the value of a scale parameter from the model.
    
    Handles both CircuitSelfAttention (direct attributes) and CircuitGNN 
    (nested in node_update module).
    
    Args:
        model: The model instance (CircuitSelfAttention or CircuitGNN)
        param_name: Name of the parameter ('logit_scale' or 'hidden_scale')
        
    Returns:
        The scalar value of the parameter, or None if not found
    """
    try:
        # Try direct attribute first (CircuitSelfAttention)
        if hasattr(model, param_name):
            param = getattr(model, param_name)
        # Try nested in node_update (CircuitGNN)
        elif hasattr(model, 'node_update') and hasattr(model.node_update, param_name):
            param = getattr(model.node_update, param_name)
        else:
            return None
        
        # If it's a nnx.Param, extract the value
        if isinstance(param, nnx.Param):
            value = param.value
            # Convert JAX array to Python float
            if hasattr(value, 'item'):
                return float(value.item())
            elif hasattr(value, '__len__') and len(value) > 0:
                return float(value[0])
            else:
                return float(value)
        # If it's a scalar (when re_zero_update=False), return it directly
        elif isinstance(param, (int, float)):
            return float(param)
        
        return None
    except Exception as e:
        print(f"Warning: Error extracting {param_name}: {e}")
        return None
# END DEBUG BLOCK


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
    - Self-Attention: Pre-trained models suggest logit improvements
    """

    def __init__(self):
        # Circuit configuration
        self.input_n = 8
        self.output_n = 8
        self.arity = 4
        self.layer_n = 3
        self.width_factor = 2
        self.hidden_dim = 64

        # Update case_n based on input_n
        self.case_n = 1 << self.input_n

        # Wiring configuration
        self.wiring_modes = ["fixed", "random"]
        self.wiring_mode_idx = 0  # fixed (matches config.yaml training pattern)
        self.wiring_mode = self.wiring_modes[self.wiring_mode_idx]
        self.wiring_seed = 42  # Will be set from training config if available
        self.wiring_key = jax.random.PRNGKey(self.wiring_seed)

        # Simplified wiring (no training-consistent wire generation in minimal version)
        self.damage_seed = 481  # Will be set from training config if available
        self.evaluation_base_seed = 42  # Will be set from training config if available
        self.greedy_ordered_indices = None  # Prefer from training config
        self.default_damage_prob = None  # Prefer from training config
        self.training_mode = "repair"  # Will be set from config
        # Preconfiguration params (used in repair mode)
        self.preconfig_steps = 200
        self.preconfig_lr = 1.0
        self.preconfig_optimizer = "adamw"
        self.preconfig_weight_decay = 0.0
        self.preconfig_beta1 = 0.9
        self.preconfig_beta2 = 0.999

        # Optimization configuration
        self.loss_type = "l4"
        self.learning_rate = 1.0  # Learning rate for backprop
        self.n_message_steps = 1

        # Load training config defaults (circuit, seeds, loss, damage, mode)
        self._load_training_config_defaults()

        # Task configuration (ensure we have task to build x/y for preconfigure)
        self.available_tasks = list(TASKS.keys())
        if not hasattr(self, "task_idx") or not (0 <= getattr(self, "task_idx", -1) < len(self.available_tasks)):
            self.task_idx = (
                self.available_tasks.index("binary_multiply")
                if "binary_multiply" in self.available_tasks
                else 0
            )
        self.task_text = "Hello Neural CA"  # Shorter text works better with performance mode
        self.noise_p = 0.5
        # Flag to skip preconfiguration when loading preconfigured state
        self._skip_preconfig = False
        # Initialize circuit using shared functions (may preconfigure in repair mode)
        self.initialize_circuit()

        # Now that circuit exists, initialize task and visuals safely
        self.update_task(reset_logs=False)

        # Optimization state
        self.step_i = 0
        self.is_optimizing = True
        self.loss_log = np.zeros(max_trainstep_n, np.float32)
        self.hard_log = np.zeros(max_trainstep_n, np.float32)
        self.accuracy_log = np.zeros(max_trainstep_n, np.float32)
        self.hard_accuracy_log = np.zeros(max_trainstep_n, np.float32)


        # Optimization method configuration
        self.optimization_methods = ["Backprop", "Self-Attention"]
        self.optimization_method_idx = 0

        # Perturbation type (only GAMMA RAYS in minimal version)
        self.perturbation_type = "GAMMA RAYS"

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

        # Plot display options
        self.plot_types = ["Loss", "Accuracy"]
        self.plot_type_idx = 1  # Default to Loss
        self.loss_display_modes = ["Both", "Soft Only", "Hard Only"]
        self.loss_display_mode_idx = 0  # Default to showing both

        # Gate mask management for circuit visualization
        self.gate_mask = []
        self.wire_masks = []
        self.reset_gate_mask()

        # Reversible damage visualization state (for GAMMA RAYS)
        self._viz_damage_mask = []  # Per-layer masks (1.0=active, 0.0=damaged) for visualization only
        self._viz_flash_ticks = 0  # Counter for single-tick red flash (0 = no flash)
        self.damage_bias = -10.0  # Negative bias value for damaged gates
        self._current_knockout_pattern = None  # Current knockout pattern for damage injection

        # DEBUG: Scale parameter values (for re_zero_update models)
        self.model_logit_scale = None
        self.model_hidden_scale = None
        
        # DEBUG: Checkpoint metadata (epoch, step)
        self.checkpoint_epoch = None
        self.checkpoint_step = None

        # Store activations for circuit visualization
        self.act = []
        self.err_mask = None

        # Active case for visualization
        self.active_case_i = 123 % self.case_n

        # WandB integration
        self.wandb_entity = "marcello-barylli-growai"  # matches config.yaml
        self.wandb_project = "boolean-nca-cc"
        self.wandb_download_dir = "saves"
        self.run_id = None
        self.loaded_run_id = None

        # Model loading preferences
        self.load_modes = ["Latest Checkpoint", "Best Model"]
        self.load_mode_idx = 1  # Default to best model
        # prefer_metric is now auto-derived from config's checkpoint settings
        self.prefer_metric = "eval_ko_in_hard_accuracy"  # Fallback default (will be overridden by config)

        # Initialize visualization
        self.setup_visualization()

        # Debug flag for printing dimensions
        self._debug_printed = False

        # Initialize optimization method
        self.initialize_optimization_method()

        # Initialize activations now that everything is set up
        self.initialize_activations()

    # DEBUG: (Actually flagged as TEMPFIX, there is probably a less bloaty fox for this)
    def _extract_backprop_config_from_loaded_config(self, config):
        """
        Extract backprop config from loaded WandB config (matches test script approach).
        
        Args:
            config: Loaded config object (OmegaConf or dict)
            
        Returns:
            Dictionary with backprop config parameters, or None if not found
        """
        backprop_cfg = None
        config_source = None
        
        # Try loaded config first (what was actually used during training)
        # Config from WandB is an OmegaConf object, so we need to handle it properly
        if hasattr(config, "backprop"):
            backprop_cfg_raw = getattr(config, "backprop", None)
            if backprop_cfg_raw is not None:
                # Convert OmegaConf to dict (matches train.py line 419)
                # This ensures we get the same structure as training
                try:
                    backprop_cfg = OmegaConf.to_container(backprop_cfg_raw, resolve=True)
                    config_source = "loaded WandB config"
                except Exception as e:
                    print(f"Warning: Could not convert backprop config from OmegaConf: {e}")
                    # Fall back to direct access
                    if isinstance(backprop_cfg_raw, dict):
                        backprop_cfg = backprop_cfg_raw
                    else:
                        # OmegaConf object - convert manually
                        backprop_cfg = {
                            "epochs": getattr(backprop_cfg_raw, "epochs", 200),
                            "learning_rate": getattr(backprop_cfg_raw, "learning_rate", 1.0),
                            "optimizer": getattr(backprop_cfg_raw, "optimizer", "adam"),
                            "weight_decay": getattr(backprop_cfg_raw, "weight_decay", 0.0),
                            "beta1": getattr(backprop_cfg_raw, "beta1", 0.9),
                            "beta2": getattr(backprop_cfg_raw, "beta2", 0.999),
                        }
                    config_source = "loaded WandB config (manual conversion)"
        
        # Fall back to local config.yaml
        if backprop_cfg is None:
            try:
                with open("configs/config.yaml", "r") as f:
                    local_cfg = yaml.safe_load(f)
                backprop_cfg = local_cfg.get("backprop", {})
                config_source = "local config.yaml"
            except Exception as e:
                print(f"Warning: Could not load preconfig params from local config.yaml: {e}")
                backprop_cfg = {}
                config_source = "defaults"
        
        if backprop_cfg:
            print(f"Extracted preconfig params from {config_source}:")
            print(f"  steps={backprop_cfg.get('epochs', 200)}, lr={backprop_cfg.get('learning_rate', 1.0)}, optimizer={backprop_cfg.get('optimizer', 'adam')}")
            print(f"  weight_decay={backprop_cfg.get('weight_decay', 0.0)}, beta1={backprop_cfg.get('beta1', 0.9)}, beta2={backprop_cfg.get('beta2', 0.999)}")
        
        return backprop_cfg, config_source

    def _load_training_config_defaults(self):
        """Load defaults from training config.yaml and apply to GUI state."""
        try:
            with open("configs/config.yaml", "r") as f:
                cfg = yaml.safe_load(f)

            # Circuit
            circuit_cfg = cfg.get("circuit", {})
            self.input_n = circuit_cfg.get("input_bits", self.input_n)
            self.output_n = circuit_cfg.get("output_bits", self.output_n)
            self.arity = circuit_cfg.get("arity", self.arity)
            self.layer_n = circuit_cfg.get("num_layers", self.layer_n)

            # Loss
            training_cfg = cfg.get("training", {})
            self.loss_type = training_cfg.get("loss_type", self.loss_type)
            self.training_mode = training_cfg.get("training_mode", self.training_mode)

            # Seeds
            self.wiring_seed = cfg.get("test_seed", self.wiring_seed)
            self.wiring_key = jax.random.PRNGKey(self.wiring_seed)
            self.evaluation_base_seed = cfg.get("test_seed", self.evaluation_base_seed)
            self.damage_seed = cfg.get("damage_seed", self.damage_seed)

            # Damage defaults
            pool_cfg = cfg.get("pool", {})
            self.default_damage_prob = pool_cfg.get("damage_prob", self.default_damage_prob)
            self.greedy_ordered_indices = pool_cfg.get(
                "greedy_ordered_indices", DEFAULT_GREEDY_ORDERED_INDICES
            )

            # Preconfiguration params (from backprop block)
            backprop_cfg = cfg.get("backprop", {})
            self.preconfig_steps = int(backprop_cfg.get("epochs", self.preconfig_steps))
            self.preconfig_lr = float(backprop_cfg.get("learning_rate", self.preconfig_lr))
            self.preconfig_optimizer = backprop_cfg.get("optimizer", self.preconfig_optimizer)
            self.preconfig_weight_decay = float(backprop_cfg.get("weight_decay", self.preconfig_weight_decay))
            self.preconfig_beta1 = float(backprop_cfg.get("beta1", self.preconfig_beta1))
            self.preconfig_beta2 = float(backprop_cfg.get("beta2", self.preconfig_beta2))

            # Task
            task_name = circuit_cfg.get("task")
            if task_name and task_name in TASKS:
                self.available_tasks = list(TASKS.keys())
                self.task_idx = self.available_tasks.index(task_name)

            # Case count
            self.case_n = 1 << self.input_n
        except Exception as e:
            print(f"Warning: Could not load training config defaults: {e}")

    def load_preconfigured_state_from_file(self, logits_file: str, wires_file: str):
        """
        Load preconfigured circuit state (logits and wires) from NPZ files.
        
        Args:
            logits_file: Path to NPZ file containing preconfigured logits (keys: layer_0, layer_1, ...)
            wires_file: Path to NPZ file containing wires (keys: layer_0, layer_1, ...)
            
        Returns:
            Tuple of (wires, logits) as lists of JAX arrays, or None if loading fails
        """
        try:
            print(f"Loading preconfigured state from files:")
            print(f"  Logits: {logits_file}")
            print(f"  Wires: {wires_file}")
            
            # Load logits
            logits_data = np.load(logits_file)
            logits = []
            i = 0
            while f"layer_{i}" in logits_data:
                logits.append(jp.array(logits_data[f"layer_{i}"]))
                print(f"  Loaded logits layer_{i}: shape={logits_data[f'layer_{i}'].shape}, dtype={logits_data[f'layer_{i}'].dtype}")
                i += 1
            
            # Load wires
            wires_data = np.load(wires_file)
            wires = []
            i = 0
            while f"layer_{i}" in wires_data:
                wires.append(jp.array(wires_data[f"layer_{i}"]))
                print(f"  Loaded wires layer_{i}: shape={wires_data[f'layer_{i}'].shape}, dtype={wires_data[f'layer_{i}'].dtype}")
                i += 1
            
            print(f"Successfully loaded {len(logits)} logit layers and {len(wires)} wire layers")
            return wires, logits
        except Exception as e:
            print(f"Error loading preconfigured state: {e}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            return None, None

    def initialize_circuit(self):
        """Initialize circuit using shared infrastructure"""
        # Generate layer sizes using shared function
        self.layer_sizes = list(generate_layer_sizes(
            self.input_n, self.output_n, self.arity, self.layer_n
        ))

        # Use preconfigure_circuit_logits in repair mode (matches training), otherwise gen_circuit
        # Skip preconfiguration if flag is set (e.g., when loading preconfigured state)
        if self.training_mode == "repair" and self.wiring_mode == "fixed" and not self._skip_preconfig:
            # Always generate fresh task data for preconfiguration to ensure it matches current task settings
            # (matches test script approach - always generates fresh task data)
            task_name = self.available_tasks[self.task_idx]
            task_kwargs = {"input_bits": self.input_n, "output_bits": self.output_n}
            if task_name == "text":
                task_kwargs["text"] = self.task_text
            elif task_name == "noise":
                task_kwargs["noise_p"] = self.noise_p
                task_kwargs["seed"] = 42
            
            x_data, y_data = get_task_data(task_name, self.case_n, **task_kwargs)
            print(f"Task data: {task_name}, Input shape: {x_data.shape}, Output shape: {y_data.shape}, case_n={self.case_n}")

            # DEBUG: (Actually flagged as TEMPFIX, there is probably a less bloaty fox for this)
            # Log preconfig params being used (matches test script)
            print(f"Preconfiguring circuit with params (matching train_loop.py):")
            print(f"  wiring_seed={self.wiring_seed} (from config test_seed)")
            print(f"  wiring_key={self.wiring_key}")  # DEBUG: Show actual key value
            print(f"  layer_sizes={self.layer_sizes}")
            print(f"  arity={self.arity}, loss_type={self.loss_type}")
            print(f"  steps={self.preconfig_steps}, lr={self.preconfig_lr}, optimizer={self.preconfig_optimizer}")
            print(f"  weight_decay={self.preconfig_weight_decay}, beta1={self.preconfig_beta1}, beta2={self.preconfig_beta2}")
            print(f"  task_name={task_name}, case_n={self.case_n}")  # DEBUG: Show task details
            print(f"  x_data shape={x_data.shape}, y_data shape={y_data.shape}")  # DEBUG: Show data shapes
            # DEBUG: Verify data matches (convert JAX arrays to numpy for hashing)
            x_data_np = np.array(x_data) if hasattr(x_data, '__array__') else x_data
            y_data_np = np.array(y_data) if hasattr(y_data, '__array__') else y_data
            print(f"  x_data hash={hash(x_data_np.tobytes())}, y_data hash={hash(y_data_np.tobytes())}")  # DEBUG: Verify data matches
            print(f"  x_data first 5 values={x_data_np[:5] if len(x_data_np.shape) == 1 else x_data_np[:5, 0]}, y_data first 5 values={y_data_np[:5] if len(y_data_np.shape) == 1 else y_data_np[:5, 0]}")  # DEBUG: Sample values
            
            self.wires, self.logits = preconfigure_circuit_logits(
                wiring_key=self.wiring_key,
                layer_sizes=self.layer_sizes,
                arity=self.arity,
                x_data=x_data,
                y_data=y_data,
                loss_type=self.loss_type,
                steps=self.preconfig_steps,
                lr=self.preconfig_lr,
                optimizer=self.preconfig_optimizer,
                weight_decay=self.preconfig_weight_decay,
                beta1=self.preconfig_beta1,
                beta2=self.preconfig_beta2,
            )
            
            # DEBUG: Log logits statistics after preconfiguration
            logits_stats = []
            for i, logit_layer in enumerate(self.logits):
                logit_np = np.array(logit_layer) if hasattr(logit_layer, '__array__') else logit_layer
                logits_stats.append(f"layer_{i}: mean={logit_np.mean():.6f}, std={logit_np.std():.6f}, min={logit_np.min():.6f}, max={logit_np.max():.6f}")
            print(f"  Post-preconfig logits stats: {'; '.join(logits_stats)}")
            
            # DEBUG: (Actually flagged as TEMPFIX, there is probably a less bloaty fox for this)
            # Log preconfigured circuit metrics (matches test script)
            try:
                initial_loss, initial_aux = get_loss_from_wires_logits(
                    self.logits, self.wires, x_data, y_data, self.loss_type
                )
                initial_hard_loss, _, _, initial_accuracy, initial_hard_accuracy, _, _ = initial_aux
                print(f"Preconfigured circuit metrics: loss={float(initial_loss):.6f}, hard_loss={float(initial_hard_loss):.4f}, accuracy={float(initial_accuracy):.4f}, hard_accuracy={float(initial_hard_accuracy):.4f}")
                
                # Warn if preconfiguration didn't achieve perfect accuracy (training shows 1.0000)
                if initial_hard_accuracy < 0.99999999:
                    print(f"⚠️  Warning: Preconfigured circuit hard_accuracy ({initial_hard_accuracy:.4f}) is below perfect (1.0000).")
                    print(f"   This may indicate preconfig parameters don't match training.")
            except Exception as e:
                print(f"Warning: Could not compute preconfigured circuit metrics: {e}")
        else:
            # Use simple circuit generation for non-repair modes
            self.wires, self.logits = gen_circuit(
                self.wiring_key, self.layer_sizes, arity=self.arity
            )

        # Store initial logits
        self.logits0 = self.logits

        print(f"Circuit initialized with {sum(logit.size for logit in self.logits0)} parameters")
        print(f"Layer structure: {self.layer_sizes}")

        # Compute and log initial loss directly from circuit (before any model is loaded)
        # Debug: print initial loss
        if hasattr(self, "input_x") and hasattr(self, "y0"):
            try:
                initial_loss, initial_aux = get_loss_from_wires_logits(
                    self.logits, self.wires, self.input_x, self.y0, self.loss_type
                )
                initial_hard_loss, _, _, initial_accuracy, initial_hard_accuracy, _, _ = initial_aux
                print(f"[Circuit Init] Direct loss computation (before model): loss={float(initial_loss):.6f}, hard_loss={float(initial_hard_loss):.4f}, accuracy={float(initial_accuracy):.4f}, hard_accuracy={float(initial_hard_accuracy):.4f}")
            except Exception as e:
                print(f"[Circuit Init] Could not compute initial loss: {e}")
        else:
            print("[Circuit Init] Task data not yet available, will compute loss after task setup")

        # Reset gate masks for new circuit structure
        self.reset_gate_mask()

        # Initialize empty activations (will be properly set after task setup)
        self.act = [np.zeros((self.case_n, size)) for size, _ in self.layer_sizes]
        self.err_mask = np.zeros((self.case_n, self.output_n), bool)

        # Reset the model generator when circuit changes
        self.model_generator = None
        self.last_step_result = None

    def update_task(self, reset_logs=True):
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
            
            # Compute and log initial loss directly from circuit after task setup (before any model)
            # Debug: print initial loss
            if hasattr(self, "logits") and self.logits is not None:
                try:
                    initial_loss, initial_aux = get_loss_from_wires_logits(
                        self.logits, self.wires, self.input_x, self.y0, self.loss_type
                    )
                    initial_hard_loss, _, _, initial_accuracy, initial_hard_accuracy, _, _ = initial_aux
                    print(f"[Task Setup] Direct loss computation (before model): loss={float(initial_loss):.6f}, hard_loss={float(initial_hard_loss):.4f}, accuracy={float(initial_accuracy):.4f}, hard_accuracy={float(initial_hard_accuracy):.4f}")
                except Exception as e:
                    print(f"[Task Setup] Could not compute initial loss: {e}")
        except Exception as e:
            print(f"Error loading task '{task_name}': {e}")
            # Fallback to copy task
            x = jp.arange(self.case_n)
            self.input_x = unpack(x, bit_n=self.input_n)
            max_output_value = (1 << self.output_n) - 1
            clipped_output = np.minimum(x, max_output_value)
            self.y0 = jp.float32(unpack(clipped_output, bit_n=self.output_n))

        # Reset optimization progress
        if reset_logs:
            self.step_i = 0
            self.loss_log = np.zeros(max_trainstep_n, np.float32)
            self.hard_log = np.zeros(max_trainstep_n, np.float32)
            self.accuracy_log = np.zeros(max_trainstep_n, np.float32)
            self.hard_accuracy_log = np.zeros(max_trainstep_n, np.float32)

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
                # Since tex_id is not used in draw_lut, we can use None as placeholder
                # The texture tuples are stored but never actually used for rendering
                self.input_texture = (
                    None,  # tex_id not used in draw_lut function
                    self.inputs_img.shape[1],
                    self.inputs_img.shape[0],
                )
                self.output_texture = (
                    None,  # tex_id not used in draw_lut function
                    self.outputs_img.shape[1],
                    self.outputs_img.shape[0],
                )
                self.ground_truth_texture = (
                    None,  # tex_id not used in draw_lut function
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
            # DEBUG: Clear scale values and checkpoint metadata when switching to backprop
            self.model_logit_scale = None
            self.model_hidden_scale = None
            self.checkpoint_epoch = None
            self.checkpoint_step = None
            # END DEBUG

        elif method_name == "Self-Attention":
            # Try to load pre-trained frozen model
            # Check if model is already loaded - if so, skip loading and just initialize generator
            if self.frozen_model is not None:
                print(f"Model already loaded, skipping WandB load")
                # Initialize the generator for step-by-step evaluation
                self.initialize_model_generator()
            elif self.try_load_wandb_model(skip_circuit_regeneration=False):
                print(f"Loaded frozen {method_name} model from WandB")
                if self.loaded_run_id:
                    print(f"  WandB Run ID: {self.loaded_run_id}")
                self.logit_optimizer = None  # No optimizer needed for frozen models
                self.logit_opt_state = None  # No optimizer state needed for frozen models
                # Initialize the generator for step-by-step evaluation
                self.initialize_model_generator()
            else:
                print(f"Could not load {method_name} model. Falling back to Backprop.")
                # DEBUG: Clear scale values and checkpoint metadata when model loading fails
                self.model_logit_scale = None
                self.model_hidden_scale = None
                self.checkpoint_epoch = None
                self.checkpoint_step = None
                # END DEBUG
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
                logits=self.logits,  # Current logits (NOT damaged, NOT logits0)
                x_data=self.input_x,
                y_data=self.y0,
                input_n=self.input_n,
                arity=self.arity,
                circuit_hidden_dim=hidden_dim_for_graph,  # Use model's hidden_dim
                max_steps=None,  # Infinite steps for live demo
                loss_type=self.loss_type,
                bidirectional_edges=True,
                layer_sizes=self.layer_sizes,
                layer_neighbors=False,  # Match training default (can be enhanced to read from config)
                knockout_pattern=getattr(self, '_current_knockout_pattern', None),  # Pass pattern to generator
                reset_step_counter_on_init=(getattr(self, '_current_knockout_pattern', None) is not None),  # Reset if pattern exists
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

    def try_load_wandb_model(self, skip_circuit_regeneration=False):
        """Try to load frozen model from WandB
        
        Args:
            skip_circuit_regeneration: If True, skip regenerating circuit after loading model.
                                       Useful when circuit state is already set (e.g., from preconfigured state).
        """
        try:
            method_name = self.optimization_methods[self.optimization_method_idx]
            model_type = "self_attention"

            filters = {
                "config.circuit.input_bits": self.input_n,
                "config.circuit.output_bits": self.output_n,
                "config.circuit.arity": self.arity,
                # "config.circuit.num_layers": self.layer_n,
                "config.model.type": model_type,
                # Note: wiring_mode doesn't exist in training config - training always uses fixed wiring
                "config.circuit.task": self.available_tasks[self.task_idx],
                "config.training.training_mode": "repair",  # Match your config's training mode
                "config.pool.damage_mode": "greedy_vocabulary",  # Match your config's damage mode
                "config.pool.damage_injection_mode": "multi",  # Match your config's damage injection mode
            }

            # Load frozen model based on selected mode
            load_mode = self.load_modes[self.load_mode_idx]

            if load_mode == "Best Model":
                # First, load config to get checkpoint settings (for metric derivation)
                # We'll do a quick load to get the config, then reload with correct metrics
                temp_config, _, _ = load_config_from_wandb(
                    run_id=self.run_id,
                    filters=filters if not self.run_id else None,
                    project=self.wandb_project,
                    entity=self.wandb_entity,
                    download_dir=self.wandb_download_dir,
                    filename="latest_checkpoint",  # Just to get config, not the actual model
                    select_by_best_metric=False,
                    run_from_last=1,
                    use_cache=True,
                )
                
                # Derive metric name from config's checkpoint settings
                from boolean_nca_cc.training.checkpointing import derive_checkpoint_metric_from_config
                metric_name, prefer_metric = derive_checkpoint_metric_from_config(temp_config)
                print(f"Using checkpoint metric from config: {metric_name} (prefer: {prefer_metric})")
                
                # Now load the actual best model with the correct metric
                loaded_config, checkpoint_path, run_id = load_config_from_wandb(
                    run_id=self.run_id,
                    filters=filters if not self.run_id else None,
                    project=self.wandb_project,
                    entity=self.wandb_entity,
                    download_dir=self.wandb_download_dir,
                    select_by_best_metric=True,
                    run_from_last=1,
                    use_cache=True,
                    prefer_metric=prefer_metric,  # Use metric derived from config
                    metric_name=metric_name,  # Use metric name derived from config
                )

                model, loaded_dict = load_model_from_config_and_checkpoint(
                    config=loaded_config,
                    checkpoint_path=checkpoint_path,
                    run_id=run_id,
                )

                # For best model loading, we already have the instantiated model
                self.frozen_model = model
                self.loaded_run_id = run_id
                self.loaded_run_id = loaded_dict.get("run_id", "unknown")
                
                # DEBUG BLOCK: Extract checkpoint metadata (epoch, step)
                # Extract step (always available)
                self.checkpoint_step = loaded_dict.get("step")
                
                # Extract epoch from config (available for periodic/best checkpoints)
                checkpoint_config = loaded_dict.get("config", {})
                if isinstance(checkpoint_config, dict):
                    self.checkpoint_epoch = checkpoint_config.get("epoch")
                else:
                    # Config might be an OmegaConf object
                    self.checkpoint_epoch = getattr(checkpoint_config, "epoch", None)
                
                if self.checkpoint_step is not None:
                    print(f"DEBUG: Checkpoint step = {self.checkpoint_step}")
                if self.checkpoint_epoch is not None:
                    print(f"DEBUG: Checkpoint epoch = {self.checkpoint_epoch}")
                # END DEBUG BLOCK

            else:  # Latest Checkpoint
                # Use the original checkpoint loading
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

                model, loaded_dict = load_model_from_config_and_checkpoint(
                    config=loaded_config,
                    checkpoint_path=checkpoint_path,
                    run_id=run_id,
                )

                self.frozen_model = model
                self.loaded_run_id = loaded_dict.get("run_id", "unknown")
                
                # DEBUG BLOCK: Extract checkpoint metadata (epoch, step)
                # Extract step (always available)
                self.checkpoint_step = loaded_dict.get("step")
                
                # Extract epoch from config (available for periodic/best checkpoints)
                checkpoint_config = loaded_dict.get("config", {})
                if isinstance(checkpoint_config, dict):
                    self.checkpoint_epoch = checkpoint_config.get("epoch")
                else:
                    # Config might be an OmegaConf object
                    self.checkpoint_epoch = getattr(checkpoint_config, "epoch", None)
                
                if self.checkpoint_step is not None:
                    print(f"DEBUG: Checkpoint step = {self.checkpoint_step}")
                if self.checkpoint_epoch is not None:
                    print(f"DEBUG: Checkpoint epoch = {self.checkpoint_epoch}")
                # END DEBUG BLOCK

            # Align GUI state from loaded config (mirror training conditions)
            try:
                # Circuit core parameters
                self.input_n = getattr(loaded_config.circuit, "input_bits", self.input_n)
                self.output_n = getattr(loaded_config.circuit, "output_bits", self.output_n)
                self.arity = getattr(loaded_config.circuit, "arity", self.arity)
                self.layer_n = getattr(loaded_config.circuit, "num_layers", self.layer_n)

                # Seeds - extract test_seed with robust handling (matches test script)
                if hasattr(loaded_config, "test_seed"):
                    self.wiring_seed = getattr(loaded_config, "test_seed", self.wiring_seed)
                elif isinstance(loaded_config, dict) and "test_seed" in loaded_config:
                    self.wiring_seed = loaded_config["test_seed"]
                else:
                    # Try OmegaConf-style access
                    try:
                        self.wiring_seed = loaded_config.get("test_seed", self.wiring_seed) if hasattr(loaded_config, "get") else self.wiring_seed
                    except:
                        pass  # Keep existing value
                
                self.wiring_key = jax.random.PRNGKey(self.wiring_seed)
                self.damage_seed = getattr(loaded_config, "damage_seed", self.damage_seed) if hasattr(loaded_config, "damage_seed") else self.damage_seed
                print(f"Updated wiring_seed={self.wiring_seed} from loaded config (matches train.py wiring_fixed_key)")

                # Loss type
                if hasattr(loaded_config, "training") and hasattr(loaded_config.training, "loss_type"):
                    self.loss_type = loaded_config.training.loss_type

                # Task
                if hasattr(loaded_config, "circuit") and hasattr(loaded_config.circuit, "task"):
                    task_name = loaded_config.circuit.task
                    if task_name in TASKS:
                        self.available_tasks = list(TASKS.keys())
                        self.task_idx = self.available_tasks.index(task_name)

                # Damage params
                if hasattr(loaded_config, "pool"):
                    self.default_damage_prob = getattr(loaded_config.pool, "damage_prob", self.default_damage_prob)
                    if hasattr(loaded_config.pool, "greedy_ordered_indices"):
                        self.greedy_ordered_indices = list(getattr(loaded_config.pool, "greedy_ordered_indices"))
                
                # DEBUG: (Actually flagged as TEMPFIX, there is probably a less bloaty fox for this)
                # Extract and update preconfig params from loaded WandB config (matches test script)
                # This ensures we use the exact same preconfig params that were used during training
                backprop_cfg, config_source = self._extract_backprop_config_from_loaded_config(loaded_config)
                if backprop_cfg:
                    self.preconfig_steps = int(backprop_cfg.get("epochs", self.preconfig_steps))
                    self.preconfig_lr = float(backprop_cfg.get("learning_rate", self.preconfig_lr))
                    self.preconfig_optimizer = backprop_cfg.get("optimizer", self.preconfig_optimizer)
                    self.preconfig_weight_decay = float(backprop_cfg.get("weight_decay", self.preconfig_weight_decay))
                    self.preconfig_beta1 = float(backprop_cfg.get("beta1", self.preconfig_beta1))
                    self.preconfig_beta2 = float(backprop_cfg.get("beta2", self.preconfig_beta2))
                    print(f"Updated preconfig params from {config_source} to match training")
            except Exception as align_e:
                print(f"Warning: Could not fully align GUI from loaded config: {align_e}")

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
                self.model_use_globals = True  # Always True for self-attention models

            # After aligning parameters, regenerate circuit to ensure parity with training config
            # Skip if skip_circuit_regeneration is True (e.g., when loading preconfigured state)
            if not skip_circuit_regeneration:
                try:
                    self.regenerate_circuit(reset_logs=True)
                except Exception as regen_e:
                    print(f"Warning: Could not regenerate circuit after loading config: {regen_e}")

            # DEBUG BLOCK: Extract and store scale parameters from loaded model
            if self.frozen_model is not None:
                self.model_logit_scale = _extract_scale_parameter(self.frozen_model, "logit_scale")
                self.model_hidden_scale = _extract_scale_parameter(self.frozen_model, "hidden_scale")
                
                if self.model_logit_scale is not None:
                    print(f"DEBUG: Model logit_scale = {self.model_logit_scale:.9f}")
                else:
                    print("DEBUG: Model logit_scale not found (may not use re_zero_update)")
                
                if self.model_hidden_scale is not None:
                    print(f"DEBUG: Model hidden_scale = {self.model_hidden_scale:.9f}")
                else:
                    print("DEBUG: Model hidden_scale not found (may not use re_zero_update)")
            # END DEBUG BLOCK

            return True

        except Exception as e:
            print(f"Could not load model from WandB: {e}")
            return False

    def optimize_circuit(self):
        """Perform one optimization step on the circuit logits"""
        try:
            method_name = self.optimization_methods[self.optimization_method_idx]

            if method_name == "Backprop":
                loss, hard_loss, accuracy, hard_accuracy = self.optimize_backprop()
            else:
                loss, hard_loss, accuracy, hard_accuracy = self.optimize_with_unified_model()

            # Update loss logs
            i = self.step_i % len(self.loss_log)
            self.loss_log[i] = max(min(float(loss), self.max_loss_value), self.min_loss_value)
            self.hard_log[i] = max(min(float(hard_loss), self.max_loss_value), self.min_loss_value)
            self.accuracy_log[i] = max(min(float(accuracy), 1.0), 0.0)
            self.hard_accuracy_log[i] = max(min(float(hard_accuracy), 1.0), 0.0)

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

            # Use visualization damage mask during flash period, otherwise no mask
            # Note: Flash ticks are decremented in draw_circuit() to ensure proper timing
            viz_mask = None
            if self._viz_flash_ticks > 0 and len(self._viz_damage_mask) > 0:
                viz_mask = self._viz_damage_mask

            # Run circuit to get layer-by-layer activations
            # This returns [input_acts, layer1_acts, layer2_acts, ..., output_acts]
            self.act = run_circuit(
                current_logits, self.wires, self.input_x, hard=False, gate_mask=viz_mask
            )

            # Generate error mask for visualization
            self.err_mask = pred_hard != self.y0

        except Exception as e:
            print(f"Warning: Could not generate circuit activations: {e}")
            # Fallback: create empty activations
            self.act = [np.zeros((self.case_n, size)) for size, _ in self.layer_sizes]
            self.err_mask = np.zeros((self.case_n, self.output_n), bool)

        return loss, hard_loss, accuracy, hard_accuracy

    def optimize_with_unified_model(self):
        """Use the unified generator from training code to optimize with frozen Self-Attention model"""
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
                    # Use visualization damage mask during flash period, otherwise no mask
                    # Note: Flash ticks are decremented in draw_circuit() to ensure proper timing
                    viz_mask = None
                    if self._viz_flash_ticks > 0 and len(self._viz_damage_mask) > 0:
                        viz_mask = self._viz_damage_mask

                    # Run circuit to get layer-by-layer activations
                    # This returns [input_acts, layer1_acts, layer2_acts, ..., output_acts]
                    self.act = run_circuit(
                        self.logits, self.wires, self.input_x, hard=False, gate_mask=viz_mask
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

                return (
                    self.last_step_result.loss,
                    self.last_step_result.hard_loss,
                    self.last_step_result.accuracy,
                    self.last_step_result.hard_accuracy,
                )
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
                return loss, hard_loss, accuracy, hard_accuracy

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

    def regenerate_circuit(self, reset_logs=True):
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
        self.update_task(reset_logs=reset_logs)

        # Reinitialize optimization method
        # self.initialize_optimization_method()

        # Reset optimization progress
        if reset_logs:
            self.step_i = 0
            self.loss_log = np.zeros(max_trainstep_n, np.float32)
            self.hard_log = np.zeros(max_trainstep_n, np.float32)
            self.accuracy_log = np.zeros(max_trainstep_n, np.float32)
            self.hard_accuracy_log = np.zeros(max_trainstep_n, np.float32)

        print("Circuit regenerated successfully")

    def _apply_gate_damage_perturbation(self, damage_prob: int | None = None, bias: float | None = None):
        """
        Apply GAMMA RAYS damage perturbation by passing knockout pattern to model.
        
        Implements reversible damage: pattern is passed to model (not baked into logits),
        enabling proper reversible mode activation and recovery behavior that matches
        the evaluation loop. Visualization mask provides single-tick red flash.
        
        Args:
            damage_prob: Number of gates to knock out (default uses training config)
            bias: Negative bias value for knocked-out gates (default uses self.damage_bias, not used in pattern-based mode)
        """
        try:
            # Use training-configured defaults when not provided
            if damage_prob is None:
                damage_prob = int(self.default_damage_prob) if self.default_damage_prob is not None else 8
            if bias is None:
                bias = self.damage_bias
            
            # 1) Sample damage pattern (skip inputs and outputs)
            # Randomize seed each time to get different patterns on each click
            key = jax.random.PRNGKey(np.random.randint(0, 1_000_000))
            
            # Use layer_sizes directly (should be a list of tuples)
            layer_sizes_list = self.layer_sizes
            print(f"Debug: layer_sizes type: {type(layer_sizes_list)}, value: {layer_sizes_list}")
            
            pattern = create_greedy_subset_random_pattern(
                key,
                layer_sizes_list,
                int(damage_prob),
                self.greedy_ordered_indices if self.greedy_ordered_indices is not None else DEFAULT_GREEDY_ORDERED_INDICES,
            )

            # Store pattern for generator (aligns with evaluation loop)
            self._current_knockout_pattern = pattern

            # 2) Build per-layer masks; set self._viz_damage_mask for the upcoming visualization flash
            layer_gate_masks = create_gate_mask_from_knockout_pattern(pattern, layer_sizes_list)
            # Store visualization mask (1.0=active, 0.0=damaged) - for visualization only
            self._viz_damage_mask = [m.astype(np.float32) for m in layer_gate_masks]

            # 3) Initialize viz flash: self._viz_flash_ticks = 1
            self._viz_flash_ticks = 3

            # 4) Re-init generator state from CURRENT logits (NOT damaged, NOT logits0)
            # Note: We preserve step_i and log history to maintain plot continuity
            # The plot will continue from the current position, showing the damage injection
            # as a continuation of the existing curve
            # (step_i and log arrays are NOT reset - history is preserved)
            
            # Re-init generator so the very next logged tick reflects damage
            # Pattern will be passed to generator, which passes it to model
            self.model_generator = None
            self.last_step_result = None
            self.initialize_model_generator()  # Rebuild state from CURRENT logits with pattern
            
            # Initialize activations
            self.initialize_activations()

            print(f"Applied GAMMA RAYS damage: {damage_prob} gates knocked out (pattern-based)")
            print(f"  - Pattern stored, will be passed to model for reversible mode activation")
            print(f"  - Visualization flash enabled for 3 ticks")

        except Exception as e:
            print(f"Error applying gate damage perturbation: {e}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")

    def reset_circuit(self):
        """Reset circuit to initial state (full reset: clears damage and restores logits0)"""
        self.logits = self.logits0
        self.step_i = 0
        self.loss_log = np.zeros(max_trainstep_n, np.float32)
        self.hard_log = np.zeros(max_trainstep_n, np.float32)
        self.accuracy_log = np.zeros(max_trainstep_n, np.float32)
        self.hard_accuracy_log = np.zeros(max_trainstep_n, np.float32)

        # Clear reversible damage visualization state
        self._viz_damage_mask = []
        self._viz_flash_ticks = 0
        self._current_knockout_pattern = None

        # Reset the model generator when circuit is reset
        self.model_generator = None
        self.last_step_result = None

        # Reinitialize optimizer for backprop
        if self.optimization_methods[self.optimization_method_idx] == "Backprop":
            opt_fn = optax.adamw(self.learning_rate, 0.8, 0.8, weight_decay=1e-1)
            self.logit_opt_state = opt_fn.init(self.logits)
            self.logit_optimizer = opt_fn
        else:
            # Reinitialize generator for Self-Attention
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

                # Show masked gates (use viz damage mask during flash, otherwise gate_mask)
                is_damaged = False
                if self._viz_flash_ticks > 0 and len(self._viz_damage_mask) > 0:
                    # During flash period, check visualization damage mask
                    if li < len(self._viz_damage_mask) and i < len(self._viz_damage_mask[li]):
                        # Convert to numpy for comparison if needed
                        mask_val = float(self._viz_damage_mask[li][i])
                        is_damaged = (mask_val == 0.0)
                else:
                    # Normal mode: check regular gate mask
                    if li < len(self.gate_mask) and i < len(self.gate_mask[li]):
                        is_damaged = (self.gate_mask[li][i] == 0.0)
                
                if is_damaged:
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
                        # Colorful visualization for Self-Attention
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
        
        # Decrement flash ticks at the end of drawing (after all gates are drawn)
        if self._viz_flash_ticks > 0:
            self._viz_flash_ticks -= 1

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
            # Original: 0xFF333333 (dark gray)
            # Test: 0xFFEBEBEB (very light gray)
            dl.add_rect_filled(p0, p1, 0xFFFFFFFF, 4.0)

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
            # Manually draw background for child window
            # Original would be ImGui default background
            dl = imgui.get_window_draw_list()
            p_min = imgui.get_window_pos()
            p_max = (p_min[0] + imgui.get_window_width(), p_min[1] + imgui.get_window_height())
            dl.add_rect_filled(p_min, p_max, 0xFFFFFFFF, 0.0)  # White background

            # Optimization progress plot
            plot_type = self.plot_types[self.plot_type_idx]
            plot_title = f"Circuit Optimization Progress - {plot_type}"

            # Set frame background to white for the plot area
            imgui.push_style_color(int(imgui.Col_.frame_bg), imgui.ImVec4(1.0, 1.0, 1.0, 1.0))  # White background
            imgui.push_style_color(int(imgui.Col_.window_bg), imgui.ImVec4(1.0, 1.0, 1.0, 1.0))  # White background
            imgui.push_style_color(int(imgui.Col_.text), imgui.ImVec4(0.0, 0.0, 0.0, 1.0))  # Black text for labels/ticks
            imgui.push_style_color(int(imgui.Col_.popup_bg), imgui.ImVec4(1.0, 1.0, 1.0, 1.0))  # White background for legend/popups
            imgui.push_style_color(int(imgui.Col_.child_bg), imgui.ImVec4(1.0, 1.0, 1.0, 1.0))  # White background for child elements (legend)
            
            # Note: ImPlot colors are controlled through ImGui style colors
            # The plot background should inherit from frame_bg/window_bg which we set to white above
            # Text color is set to black above for labels/ticks
            # Legend background uses popup_bg/child_bg which we set to white above
            
            if implot.begin_plot(plot_title, (-1, 200)):
                implot.setup_legend(implot.Location_.north_east.value)

                # Setup axes based on plot type
                if plot_type == "Loss":
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

                    # Plot loss lines based on display mode
                    display_mode = self.loss_display_modes[self.loss_display_mode_idx]
                    if display_mode in ["Both", "Soft Only"]:
                        implot.plot_line("soft_loss", self.loss_log)
                    if display_mode in ["Both", "Hard Only"]:
                        implot.plot_line("hard_loss", self.hard_log)

                else:  # Accuracy
                    implot.setup_axis_scale(implot.ImAxis_.y1.value, implot.Scale_.linear.value)
                    implot.setup_axes(
                        "Step",
                        "Accuracy",
                        implot.AxisFlags_.auto_fit.value,
                        implot.AxisFlags_.none.value,  # Remove auto_fit for y-axis to respect manual limits
                    )
                    implot.setup_axis_limits(implot.ImAxis_.y1.value, 0.0, 1.15)

                    # Plot accuracy lines based on display mode
                    display_mode = self.loss_display_modes[self.loss_display_mode_idx]
                    if display_mode in ["Both", "Soft Only"]:
                        implot.plot_line("soft_accuracy", self.accuracy_log)
                    if display_mode in ["Both", "Hard Only"]:
                        implot.plot_line("hard_accuracy", self.hard_accuracy_log)

                implot.drag_line_x(1, self.step_i % len(self.loss_log), (0.8, 0, 0, 0.5))

                # Right-click context menu for plot options
                if implot.is_plot_hovered() and imgui.is_mouse_clicked(1):  # Right click
                    imgui.open_popup("plot_options_menu")

                if imgui.begin_popup("plot_options_menu"):
                    imgui.text("Plot Options")
                    imgui.separator()

                    # Plot type selection
                    imgui.text("Plot Type:")
                    for i, ptype in enumerate(self.plot_types):
                        selected = i == self.plot_type_idx
                        if imgui.selectable(ptype, selected)[0]:
                            self.plot_type_idx = i
                            print(f"Plot type changed to: {ptype}")

                    imgui.separator()

                    # Display mode selection
                    mode_label = "Loss Display" if plot_type == "Loss" else "Accuracy Display"
                    imgui.text(f"{mode_label} Options:")
                    for i, mode in enumerate(self.loss_display_modes):
                        selected = i == self.loss_display_mode_idx
                        if imgui.selectable(mode, selected)[0]:
                            self.loss_display_mode_idx = i
                            print(f"Display mode changed to: {mode}")

                    imgui.end_popup()

                implot.end_plot()
                imgui.pop_style_color()  # Restore child_bg
                imgui.pop_style_color()  # Restore popup_bg
                imgui.pop_style_color()  # Restore text
                imgui.pop_style_color()  # Restore window_bg
                imgui.pop_style_color()  # Restore frame_bg

            # Input visualization
            # Set separator text color to dark gray
            imgui.push_style_color(int(imgui.Col_.text), imgui.ImVec4(0.3, 0.3, 0.3, 1.0))  # Dark gray
            imgui.separator_text("Inputs")
            imgui.pop_style_color()
            self.draw_lut("inputs", self.inputs_img, self.input_texture)

            # Circuit visualization
            imgui.push_style_color(int(imgui.Col_.text), imgui.ImVec4(0.3, 0.3, 0.3, 1.0))  # Dark gray
            imgui.separator_text("Circuit")
            imgui.pop_style_color()
            H = imgui.get_content_region_avail().y - 400  # Leave room for outputs below
            self.draw_circuit(H=max(H, 300))  # Minimum height of 300

            # Output vs Ground Truth
            imgui.push_style_color(int(imgui.Col_.text), imgui.ImVec4(0.3, 0.3, 0.3, 1.0))  # Dark gray
            imgui.separator_text("Current Output")
            imgui.pop_style_color()
            self.draw_lut("outputs", self.outputs_img, self.output_texture)

            imgui.push_style_color(int(imgui.Col_.text), imgui.ImVec4(0.3, 0.3, 0.3, 1.0))  # Dark gray
            imgui.separator_text("Expected Output")
            imgui.pop_style_color()
            self.draw_lut("ground_truth", self.ground_truth_img, self.ground_truth_texture)
            imgui.end_child()
            imgui.same_line()

            # Control panel
            imgui.begin_child("controls")
            # Manually draw background for child window
            dl = imgui.get_window_draw_list()
            p_min = imgui.get_window_pos()
            p_max = (p_min[0] + imgui.get_window_width(), p_min[1] + imgui.get_window_height())
            dl.add_rect_filled(p_min, p_max, 0xFFFFFFFF, 0.0)  # White background

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

            elif method_name == "Self-Attention":
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

                # Loading mode selection
                load_changed, self.load_mode_idx = imgui.combo(
                    "Load Mode", self.load_mode_idx, self.load_modes
                )
                if load_changed:
                    print(f"Changed load mode to: {self.load_modes[self.load_mode_idx]}")

                # Show description of selected mode
                if self.load_mode_idx == 0:  # Latest Checkpoint
                    imgui.text_colored(
                        imgui.ImVec4(0.7, 0.7, 0.7, 1.0),
                        "Loads most recent checkpoint (may not be best performing)",
                    )
                else:  # Best Model
                    imgui.text_colored(
                        imgui.ImVec4(0.0, 1.0, 0.0, 1.0),
                        "Loads best performing model (recommended)",
                    )

                # Optional preferred metric for best model selection
                if self.load_mode_idx == 1:  # Best Model mode
                    prefer_metrics = [
                        "Auto (Intelligent Selection)",
                        "eval_ko_in_hard_accuracy",  # Matches checkpointing config
                        "eval_ko_out_hard_accuracy",
                        "eval_in_hard_accuracy",
                        "eval_out_hard_accuracy",
                        "eval_ko_in_hard_loss",
                        "eval_ko_out_hard_loss",
                        "eval_in_hard_loss",
                        "eval_out_hard_loss",
                        "training_hard_accuracy",
                    ]
                    prefer_metric_idx = (
                        0
                        if self.prefer_metric is None
                        else (
                            prefer_metrics.index(self.prefer_metric)
                            if self.prefer_metric in prefer_metrics
                            else 0
                        )
                    )

                    changed, prefer_metric_idx = imgui.combo(
                        "Prefer Metric", prefer_metric_idx, prefer_metrics
                    )
                    if changed:
                        self.prefer_metric = (
                            None if prefer_metric_idx == 0 else prefer_metrics[prefer_metric_idx]
                        )
                        print(f"Preferred metric: {self.prefer_metric or 'Auto'}")

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
            orig_width_factor = self.width_factor

            _, self.input_n = imgui.slider_int("Input Bits", self.input_n, 2, 8)
            _, self.output_n = imgui.slider_int("Output Bits", self.output_n, 2, 8)
            _, self.arity = imgui.slider_int("Gate Arity", self.arity, 2, 4)
            _, self.layer_n = imgui.slider_int("Hidden Layers", self.layer_n, 1, 5)
            _, self.width_factor = imgui.slider_int("Width Factor", self.width_factor, 1, 4)

            if (
                self.input_n != orig_input_n
                or self.output_n != orig_output_n
                or self.arity != orig_arity
                or self.layer_n != orig_layer_n
                or self.width_factor != orig_width_factor
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
                    self.width_factor = orig_width_factor

            if imgui.button("Regenerate Circuit"):
                self.regenerate_circuit()

            # Load preconfigured circuit state
            imgui.separator_text("Preconfigured Circuit")
            if imgui.button("Load Preconfigured State"):
                # Try to load from preconfigured_circuits folder
                import os
                logits_file = "preconfigured_circuits/preconfigured_logits_20251112_linux.npz"
                wires_file = "preconfigured_circuits/wires_20251112_linux.npz"
                
                if os.path.exists(logits_file) and os.path.exists(wires_file):
                    wires, logits = self.load_preconfigured_state_from_file(logits_file, wires_file)
                    if wires is not None and logits is not None:
                        # Verify shapes match expected layer sizes
                        expected_logits_count = len(self.layer_sizes) - 1
                        if len(logits) == expected_logits_count and len(wires) == expected_logits_count:
                            # Set flag to skip preconfiguration when initializing optimization method
                            self._skip_preconfig = True
                            
                            self.wires = wires
                            self.logits = logits
                            self.logits0 = [l.copy() for l in self.logits]
                            
                            # Reset gate masks for new circuit structure
                            self.reset_gate_mask()
                            
                            # Update task first to ensure we have current task data
                            self.update_task(reset_logs=True)
                            
                            # Compute and log loss immediately after loading (before generator init)
                            try:
                                initial_loss, initial_aux = get_loss_from_wires_logits(
                                    self.logits, self.wires, self.input_x, self.y0, self.loss_type
                                )
                                initial_hard_loss, _, _, initial_accuracy, initial_hard_accuracy, _, _ = initial_aux
                                print(f"✓ Loaded preconfigured circuit metrics: loss={float(initial_loss):.6f}, hard_loss={float(initial_hard_loss):.4f}, accuracy={float(initial_accuracy):.4f}, hard_accuracy={float(initial_hard_accuracy):.4f}")
                            except Exception as e:
                                print(f"⚠️  Warning: Could not compute loaded circuit metrics: {e}")
                            
                            # Reinitialize optimization method AFTER loss computation
                            # Pass skip_circuit_regeneration=True to prevent model loading from regenerating circuit
                            # (This will initialize generator with the new state without triggering preconfiguration)
                            if self.optimization_methods[self.optimization_method_idx] == "Self-Attention":
                                # If model is not loaded, load it but skip circuit regeneration
                                if self.frozen_model is None:
                                    if self.try_load_wandb_model(skip_circuit_regeneration=True):
                                        print(f"Loaded frozen Self-Attention model from WandB")
                                        if self.loaded_run_id:
                                            print(f"  WandB Run ID: {self.loaded_run_id}")
                                        self.logit_optimizer = None
                                        self.logit_opt_state = None
                                        # Initialize the generator for step-by-step evaluation
                                        self.initialize_model_generator()
                                    else:
                                        print(f"⚠️  Warning: Could not load model, falling back to Backprop")
                                        self.optimization_method_idx = 0
                                        self.initialize_optimization_method()
                                        # Clear flag since we're not using preconfigured state anymore
                                        self._skip_preconfig = False
                                        print("✓ Successfully loaded preconfigured circuit state")
                                else:
                                    # Model already loaded, just initialize generator
                                    self.initialize_model_generator()
                            else:
                                # For backprop, just reinitialize normally
                                self.initialize_optimization_method()
                            
                            # Clear the flag after initialization
                            self._skip_preconfig = False
                            
                            print("✓ Successfully loaded preconfigured circuit state")
                        else:
                            print(f"⚠️  Warning: Loaded circuit has {len(logits)} logit layers and {len(wires)} wire layers, expected {expected_logits_count}")
                            # Clear flag on error
                            self._skip_preconfig = False
                    else:
                        print("✗ Failed to load preconfigured state")
                        # Clear flag on error
                        self._skip_preconfig = False
                else:
                    print(f"✗ Preconfigured circuit files not found:")
                    print(f"  Expected: {logits_file}")
                    print(f"  Expected: {wires_file}")

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
                self.regenerate_circuit(reset_logs=False)  # This will invalidate cache

            # ===== GAMMA RAYS PERTURBATION =====
            imgui.separator_text("GAMMA RAYS Perturbation")
            imgui.text("Apply reversible damage to circuit gates")
            
            # PERTURB button - applies GAMMA RAYS damage
            if imgui.button("PERTURB", (120, 0)):
                # Apply gate damage perturbation (preserves plot history automatically)
                self._apply_gate_damage_perturbation()
            
            imgui.same_line()
            imgui.text("(GAMMA RAYS)")

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

            # Plot type selection
            plot_changed, self.plot_type_idx = imgui.combo(
                "Plot Type", self.plot_type_idx, self.plot_types
            )
            if plot_changed:
                print(f"Plot type changed to: {self.plot_types[self.plot_type_idx]}")

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
            imgui.text(f"Load Mode: {self.load_modes[self.load_mode_idx]}")
            if self.load_mode_idx == 1 and self.prefer_metric:  # Best Model with specific metric
                imgui.text(f"Prefer Metric: {self.prefer_metric}")
            imgui.text(f"Circuit Parameters: {sum(logit.size for logit in self.logits0)}")
            imgui.text(f"Optimization Step: {self.step_i}")
            imgui.text(f"Active Input Case: {self.active_case_i}")
            imgui.text(f"Wiring Seed: {self.wiring_seed}")
            imgui.text(f"Wiring Mode: {self.wiring_mode}")
            imgui.text(f"Plot Type: {self.plot_types[self.plot_type_idx]}")
            imgui.text(f"Display Mode: {self.loss_display_modes[self.loss_display_mode_idx]}")


            # Model-specific status
            if method_name == "Self-Attention" and self.frozen_model is not None:
                imgui.text(f"Model hidden_dim: {self.model_hidden_dim}")
                imgui.text(f"Model use_globals: {self.model_use_globals}")
                
                # DEBUG BLOCK: Display checkpoint metadata (epoch, step)
                if self.checkpoint_step is not None:
                    imgui.text(f"DEBUG: Checkpoint step = {self.checkpoint_step}")
                if self.checkpoint_epoch is not None:
                    imgui.text(f"DEBUG: Checkpoint epoch = {self.checkpoint_epoch}")
                elif self.checkpoint_step is not None:
                    # Step available but epoch not - might be an older checkpoint format
                    imgui.text_colored(
                        imgui.ImVec4(0.7, 0.7, 0.7, 1.0),
                        "DEBUG: Checkpoint epoch = N/A (not in checkpoint)"
                    )
                # END DEBUG BLOCK
                
                # DEBUG BLOCK: Display scale parameters
                if self.model_logit_scale is not None:
                    # Color code based on magnitude (red if too small, green if reasonable)
                    if abs(self.model_logit_scale) < 1e-5:
                        color = imgui.ImVec4(1.0, 0.0, 0.0, 1.0)  # Red for too small
                    elif 0.001 <= abs(self.model_logit_scale) <= 0.1:
                        color = imgui.ImVec4(0.0, 1.0, 0.0, 1.0)  # Green for reasonable
                    else:
                        color = imgui.ImVec4(1.0, 1.0, 0.0, 1.0)  # Yellow for unusual
                    imgui.text_colored(color, f"DEBUG: logit_scale = {self.model_logit_scale:.9f}")
                else:
                    imgui.text_colored(
                        imgui.ImVec4(0.7, 0.7, 0.7, 1.0),
                        "DEBUG: logit_scale = N/A (not using re_zero_update)"
                    )
                
                if self.model_hidden_scale is not None:
                    # Color code based on magnitude (red if too small, green if reasonable)
                    if abs(self.model_hidden_scale) < 1e-5:
                        color = imgui.ImVec4(1.0, 0.0, 0.0, 1.0)  # Red for too small
                    elif 0.001 <= abs(self.model_hidden_scale) <= 0.1:
                        color = imgui.ImVec4(0.0, 1.0, 0.0, 1.0)  # Green for reasonable
                    else:
                        color = imgui.ImVec4(1.0, 1.0, 0.0, 1.0)  # Yellow for unusual
                    imgui.text_colored(color, f"DEBUG: hidden_scale = {self.model_hidden_scale:.9f}")
                else:
                    imgui.text_colored(
                        imgui.ImVec4(0.7, 0.7, 0.7, 1.0),
                        "DEBUG: hidden_scale = N/A (not using re_zero_update)"
                    )
                # END DEBUG BLOCK

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
        print("Starting Minimal Circuit Optimization Demo...")
        print("- Backprop: Direct gradient-based logit optimization")
        print("- Self-Attention: Frozen models suggest logit improvements")
        print("- GAMMA RAYS: Reversible damage perturbation")

        demo = CircuitOptimizationDemo()

        immapp.run(
            demo.gui,
            window_title="Circuit Optimization Demo (Minimal)",
            window_size=(1200, 800),
            fps_idle=10,
            with_implot=True,
        )
    except Exception as e:
        print(f"Error running demo: {e}")
        import traceback

        print(f"Traceback: {traceback.format_exc()}")