#!/usr/bin/env python3
"""
Test script to compare GUI evaluation conditions vs training evaluation conditions.

This script:
1. Loads a model exactly like GUI_minimal.py does (using run_id="vayt4820")
2. Sets up the circuit the same way (preconfigured if in repair mode)
3. Runs evaluation using the training evaluation system (evaluate_circuits_in_chunks with multi-injection damage)
4. Compares results to see if the issue is in GUI evaluation conditions

This helps diagnose the accuracy drift issue by isolating whether it's:
- GUI-specific evaluation conditions
- Model behavior in general
- Something else
"""

import logging
import yaml
import jax
import jax.numpy as jp
import numpy as np
from omegaconf import OmegaConf
import matplotlib.pyplot as plt
import os

from boolean_nca_cc import generate_layer_sizes
from boolean_nca_cc.circuits.model import gen_circuit
from boolean_nca_cc.circuits.tasks import TASKS, get_task_data
from boolean_nca_cc.training.checkpointing import (
    load_config_from_wandb,
    load_model_from_config_and_checkpoint,
    derive_checkpoint_metric_from_config,
)
from boolean_nca_cc.training.evaluation import (
    evaluate_circuits_in_chunks,
    evaluate_model_stepwise_batched,
    get_loss_from_wires_logits,
)
from boolean_nca_cc.training.preconfigure import preconfigure_circuit_logits

# Configure logging
logging.basicConfig(level=logging.DEBUG, format="%(name)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)
# Enable DEBUG for preconfigure module
logging.getLogger("boolean_nca_cc.training.preconfigure").setLevel(logging.DEBUG)


def load_model_like_gui(run_id: str = "vayt4820"):
    """x
    Load model exactly like GUI_minimal.py does.
    
    Args:
        run_id: WandB run ID to load
        
    Returns:
        Tuple of (model, config, loaded_dict)
    """
    log.info(f"Loading model from WandB run ID: {run_id}")
    
    # WandB configuration (matches GUI_minimal.py)
    wandb_entity = "marcello-barylli-growai"
    wandb_project = "boolean-nca-cc"
    wandb_download_dir = "saves"
    
    # First, load config to get checkpoint settings (for metric derivation)
    temp_config, _, _ = load_config_from_wandb(
        run_id=run_id,
        filters=None,
        project=wandb_project,
        entity=wandb_entity,
        download_dir=wandb_download_dir,
        filename="latest_checkpoint",
        select_by_best_metric=False,
        run_from_last=1,
        use_cache=True,
    )
    
    # Derive metric name from config's checkpoint settings
    metric_name, prefer_metric = derive_checkpoint_metric_from_config(temp_config)
    log.info(f"Using checkpoint metric from config: {metric_name} (prefer: {prefer_metric})")
    
    # Now load the actual best model with the correct metric
    loaded_config, checkpoint_path, loaded_run_id = load_config_from_wandb(
        run_id=run_id,
        filters=None,
        project=wandb_project,
        entity=wandb_entity,
        download_dir=wandb_download_dir,
        select_by_best_metric=True,
        run_from_last=1,
        use_cache=True,
        prefer_metric=prefer_metric,
        metric_name=metric_name,
    )
    
    model, loaded_dict = load_model_from_config_and_checkpoint(
        config=loaded_config,
        checkpoint_path=checkpoint_path,
        run_id=loaded_run_id,
    )
    
    # Extract checkpoint metadata
    checkpoint_step = loaded_dict.get("step")
    checkpoint_config = loaded_dict.get("config", {})
    if isinstance(checkpoint_config, dict):
        checkpoint_epoch = checkpoint_config.get("epoch")
    else:
        checkpoint_epoch = getattr(checkpoint_config, "epoch", None)
    
    log.info(f"Loaded model from run: {loaded_run_id}")
    if checkpoint_step is not None:
        log.info(f"  Checkpoint step: {checkpoint_step}")
    if checkpoint_epoch is not None:
        log.info(f"  Checkpoint epoch: {checkpoint_epoch}")
    
    return model, loaded_config, loaded_dict


def setup_circuit_like_gui(config, wiring_key):
    """
    Set up circuit exactly like train.py does (not GUI, but training).
    
    Args:
        config: Loaded config object
        wiring_key: JAX random key for wiring generation (should use config.test_seed)
        
    Returns:
        Tuple of (wires, logits, layer_sizes, x_data, y_data, task_name)
    """
    log.info("Setting up circuit like training...")
    
    # Extract circuit parameters from config (matches train.py style)
    input_n = getattr(config.circuit, "input_bits", 8) if hasattr(config, "circuit") else 8
    output_n = getattr(config.circuit, "output_bits", 8) if hasattr(config, "circuit") else 8
    arity = getattr(config.circuit, "arity", 4) if hasattr(config, "circuit") else 4
    layer_n = getattr(config.circuit, "num_layers", 3) if hasattr(config, "circuit") else 3
    task_name = getattr(config.circuit, "task", "binary_multiply") if hasattr(config, "circuit") else "binary_multiply"
    training_mode = getattr(config.training, "training_mode", "repair") if hasattr(config, "training") else "repair"
    loss_type = getattr(config.training, "loss_type", "l4") if hasattr(config, "training") else "l4"
    
    # Get test_seed from config (matches train.py: wiring_fixed_key=jax.random.PRNGKey(cfg.test_seed))
    # Handle both OmegaConf and dict access
    if hasattr(config, "test_seed"):
        test_seed = getattr(config, "test_seed", 42)
    elif isinstance(config, dict) and "test_seed" in config:
        test_seed = config["test_seed"]
    else:
        # Try OmegaConf-style access
        try:
            test_seed = config.get("test_seed", 42) if hasattr(config, "get") else 42
        except:
            test_seed = 42
    log.info(f"Using test_seed={test_seed} from config (matches train.py wiring_fixed_key)")
    
    # Generate layer sizes
    layer_sizes = list(generate_layer_sizes(input_n, output_n, arity, layer_n))
    log.info(f"Layer sizes: {layer_sizes}")
    
    # Get task data
    case_n = 1 << input_n
    task_kwargs = {"input_bits": input_n, "output_bits": output_n}
    if task_name == "text":
        task_kwargs["text"] = "Hello Neural CA"
    elif task_name == "noise":
        task_kwargs["noise_p"] = 0.5
        task_kwargs["seed"] = 42
    
    x_data, y_data = get_task_data(task_name, case_n, **task_kwargs)
    log.info(f"Task: {task_name}, Input shape: {x_data.shape}, Output shape: {y_data.shape}")
    
    # Preconfigure circuit if in repair mode (matches GUI)
    if training_mode == "repair":
        log.info("Repair mode: preconfiguring circuit...")
        
        # Get preconfig params - prioritize loaded WandB config (what was actually used in training)
        # Then fall back to local config.yaml (matches train.py behavior)
        # train.py uses: preconfig_steps=cfg.backprop.epochs, preconfig_lr=cfg.backprop.learning_rate
        # and passes backprop_config to train_model which uses it for optimizer params
        # train.py line 419: backprop_config=OmegaConf.to_container(cfg.backprop, resolve=True) if cfg.backprop.enabled else None
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
                    log.warning(f"Could not convert backprop config from OmegaConf: {e}")
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
                log.warning(f"Could not load preconfig params from local config.yaml: {e}")
                backprop_cfg = {}
                config_source = "defaults"
        
        # Extract params (matches train_loop.py line 1385-1402)
        # train_loop.py uses: backprop_config.get("optimizer", "adam") etc.
        # Expected values from config.yaml:
        #   epochs: 200
        #   learning_rate: 1
        #   weight_decay: 1e-1 (0.1)
        #   optimizer: "adamw"
        #   beta1: 0.8
        #   beta2: 0.8
        preconfig_steps = int(backprop_cfg.get("epochs", 200))
        preconfig_lr = float(backprop_cfg.get("learning_rate", 1.0))
        preconfig_optimizer = backprop_cfg.get("optimizer", "adam")  # Default is "adam" in train_loop.py
        preconfig_weight_decay = float(backprop_cfg.get("weight_decay", 0.0))  # Should be 0.1 from config.yaml
        preconfig_beta1 = float(backprop_cfg.get("beta1", 0.9))  # Should be 0.8 from config.yaml
        preconfig_beta2 = float(backprop_cfg.get("beta2", 0.999))  # Should be 0.8 from config.yaml
        
        log.info(f"Preconfig params from {config_source} (matching train_loop.py):")
        log.info(f"  steps={preconfig_steps}, lr={preconfig_lr}, optimizer={preconfig_optimizer}")
        log.info(f"  weight_decay={preconfig_weight_decay}, beta1={preconfig_beta1}, beta2={preconfig_beta2}")
        
        # DEBUG: Platform and JAX backend info
        import platform
        log.info(f"DEBUG Platform info:")
        log.info(f"  platform={platform.platform()}")
        log.info(f"  python_version={platform.python_version()}")
        log.info(f"  jax_backend={jax.devices()[0].platform if jax.devices() else 'unknown'}")
        log.info(f"  jax_version={jax.__version__ if hasattr(jax, '__version__') else 'unknown'}")
        log.info(f"  jax_default_dtype={jax.config.jax_enable_x64 if hasattr(jax.config, 'jax_enable_x64') else 'unknown'}")
        
        wires, logits = preconfigure_circuit_logits(
            wiring_key=wiring_key,
            layer_sizes=layer_sizes,
            arity=arity,
            x_data=x_data,
            y_data=y_data,
            loss_type=loss_type,
            steps=preconfig_steps,
            lr=preconfig_lr,
            optimizer=preconfig_optimizer,
            weight_decay=preconfig_weight_decay,
            beta1=preconfig_beta1,
            beta2=preconfig_beta2,
        )
        log.info(f"Preconfigured circuit with {preconfig_steps} steps")
        
        # Compute initial loss after preconfiguration
        initial_loss, initial_aux = get_loss_from_wires_logits(
            logits, wires, x_data, y_data, loss_type
        )
        initial_hard_loss, _, _, initial_accuracy, initial_hard_accuracy, _, _ = initial_aux
        log.info(
            f"Preconfigured circuit metrics: loss={float(initial_loss):.6f}, "
            f"hard_loss={float(initial_hard_loss):.4f}, "
            f"accuracy={float(initial_accuracy):.4f}, "
            f"hard_accuracy={float(initial_hard_accuracy):.4f}"
        )
        
        # DEBUG: Print full debug info AFTER preconfiguration (to avoid affecting JAX computation)
        log.info(f"DEBUG Preconfiguration details:")
        log.info(f"  wiring_seed={test_seed} (from config test_seed)")
        log.info(f"  wiring_key={wiring_key}")
        log.info(f"  layer_sizes={layer_sizes}")
        log.info(f"  arity={arity}, loss_type={loss_type}")
        log.info(f"  steps={preconfig_steps}, lr={preconfig_lr}, optimizer={preconfig_optimizer}")
        log.info(f"  weight_decay={preconfig_weight_decay}, beta1={preconfig_beta1}, beta2={preconfig_beta2}")
        log.info(f"  task_name={task_name}, case_n={case_n}")
        log.info(f"  x_data shape={x_data.shape}, y_data shape={y_data.shape}")
        log.info(f"  x_data dtype={x_data.dtype}, y_data dtype={y_data.dtype}")
        # DEBUG: Verify data matches (convert JAX arrays to numpy for hashing - AFTER preconfig)
        x_data_np = np.array(x_data) if hasattr(x_data, '__array__') else x_data
        y_data_np = np.array(y_data) if hasattr(y_data, '__array__') else y_data
        log.info(f"  x_data hash={hash(x_data_np.tobytes())}, y_data hash={hash(y_data_np.tobytes())}")
        log.info(f"  x_data first 5 values={x_data_np[:5] if len(x_data_np.shape) == 1 else x_data_np[:5, 0]}, y_data first 5 values={y_data_np[:5] if len(y_data_np.shape) == 1 else y_data_np[:5, 0]}")
        # DEBUG: Log logits statistics after preconfiguration
        logits_stats = []
        for i, logit_layer in enumerate(logits):
            logit_np = np.array(logit_layer) if hasattr(logit_layer, '__array__') else logit_layer
            logits_stats.append(f"layer_{i}: mean={logit_np.mean():.6f}, std={logit_np.std():.6f}, min={logit_np.min():.6f}, max={logit_np.max():.6f}")
        log.info(f"  Post-preconfig logits stats: {'; '.join(logits_stats)}")
        # DEBUG: Check initial circuit state before preconfiguration (if we can)
        # This would require modifying preconfigure.py to log initial state, but for now just log what we have
        
        # Warn if preconfiguration didn't achieve perfect accuracy (training shows 1.0000)
        if initial_hard_accuracy < 0.99999999:
            log.warning(
                f"⚠️  Preconfigured circuit hard_accuracy ({initial_hard_accuracy:.4f}) is below perfect (1.0000). "
                f"This may indicate:"
            )
            log.warning(
                f"   1. Preconfig parameters don't match training (steps={preconfig_steps}, lr={preconfig_lr}, "
                f"optimizer={preconfig_optimizer}, weight_decay={preconfig_weight_decay}, "
                f"beta1={preconfig_beta1}, beta2={preconfig_beta2})"
            )
            log.warning(
                f"   2. Preconfig needs more steps to converge (currently {preconfig_steps})"
            )
            log.warning(
                f"   3. Different wiring seed (test_seed={test_seed})"
            )
            log.warning(
                f"   4. Numerical precision differences"
            )
    else:
        log.info("Growth mode: generating random circuit...")
        wires, logits = gen_circuit(wiring_key, layer_sizes, arity=arity)
        
        # Compute initial loss
        initial_loss, initial_aux = get_loss_from_wires_logits(
            logits, wires, x_data, y_data, loss_type
        )
        initial_hard_loss, _, _, initial_accuracy, initial_hard_accuracy, _, _ = initial_aux
        log.info(
            f"Initial circuit: loss={float(initial_loss):.6f}, "
            f"hard_loss={float(initial_hard_loss):.4f}, "
            f"accuracy={float(initial_accuracy):.4f}, "
            f"hard_accuracy={float(initial_hard_accuracy):.4f}"
        )
    
    return wires, logits, layer_sizes, x_data, y_data, task_name


def run_training_evaluation(
    model,
    config,
    wires,
    logits,
    layer_sizes,
    x_data,
    y_data,
    eval_batch_size: int = 16,
):
    """
    Run evaluation using training evaluation system (multi-injection damage).
    
    This matches the evaluation conditions used in train_loop.py.
    
    Args:
        model: Loaded model
        config: Loaded config object
        wires: Circuit wires
        logits: Circuit logits
        layer_sizes: Layer sizes
        x_data: Input data
        y_data: Target data
        eval_batch_size: Batch size for evaluation
        
    Returns:
        Dictionary with evaluation results
    """
    log.info("Running training evaluation with multi-injection damage...")
    
    # Extract parameters from config (matches GUI_minimal.py style)
    input_n = getattr(config.circuit, "input_bits", 8) if hasattr(config, "circuit") else 8
    arity = getattr(config.circuit, "arity", 4) if hasattr(config, "circuit") else 4
    circuit_hidden_dim = getattr(config.circuit, "circuit_hidden_dim", 64) if hasattr(config, "circuit") else 64
    loss_type = getattr(config.training, "loss_type", "l4") if hasattr(config, "training") else "l4"
    n_message_steps = getattr(config.eval, "periodic_eval_inner_steps", 200) if hasattr(config, "eval") else 200
    layer_neighbors = getattr(config.training, "layer_neighbors", False) if hasattr(config, "training") else False
    
    # Get knockout evaluation config
    eval_cfg = getattr(config, "eval", None) if hasattr(config, "eval") else None
    knockout_eval = getattr(eval_cfg, "knockout_eval", None) if eval_cfg and hasattr(eval_cfg, "knockout_eval") else None
    
    # Extract damage parameters from config (matches GUI_minimal.py style)
    pool_cfg = getattr(config, "pool", None) if hasattr(config, "pool") else None
    damage_mode = getattr(pool_cfg, "damage_mode", "greedy") if pool_cfg and hasattr(pool_cfg, "damage_mode") else "greedy"
    damage_injection_mode = getattr(pool_cfg, "damage_injection_mode", "multi") if pool_cfg and hasattr(pool_cfg, "damage_injection_mode") else "multi"
    max_damage_per_circuit = int(getattr(pool_cfg, "max_damage_per_circuit", 10) if pool_cfg and hasattr(pool_cfg, "max_damage_per_circuit") else 10)
    greedy_ordered_indices = getattr(pool_cfg, "greedy_ordered_indices", None) if pool_cfg and hasattr(pool_cfg, "greedy_ordered_indices") else None
    greedy_window_size = int(getattr(pool_cfg, "greedy_window_size", 1) if pool_cfg and hasattr(pool_cfg, "greedy_window_size") else 1)
    greedy_injection_recover_steps = int(getattr(pool_cfg, "greedy_injection_recover_steps", 10) if pool_cfg and hasattr(pool_cfg, "greedy_injection_recover_steps") else 10)
    
    # Damage start offset parameters
    damage_start_offset = int(getattr(knockout_eval, "damage_start_offset", 0) if knockout_eval and hasattr(knockout_eval, "damage_start_offset") else 0)
    damage_start_offset_random = getattr(knockout_eval, "damage_start_offset_random", False) if knockout_eval and hasattr(knockout_eval, "damage_start_offset_random") else False
    damage_start_offset_seed = int(getattr(knockout_eval, "damage_start_offset_seed", 42) if knockout_eval and hasattr(knockout_eval, "damage_start_offset_seed") else 42)
    
    log.info(f"Evaluation parameters:")
    log.info(f"  - n_message_steps: {n_message_steps}")
    log.info(f"  - damage_mode: {damage_mode}")
    log.info(f"  - damage_injection_mode: {damage_injection_mode}")
    log.info(f"  - max_damage_per_circuit: {max_damage_per_circuit}")
    log.info(f"  - damage_start_offset: {damage_start_offset}")
    log.info(f"  - greedy_injection_recover_steps: {greedy_injection_recover_steps}")
    
    # Replicate base circuit for batch
    batch_wires = jax.tree.map(
        lambda x: jp.repeat(x[None, ...], eval_batch_size, axis=0), wires
    )
    batch_logits = jax.tree.map(
        lambda x: jp.repeat(x[None, ...], eval_batch_size, axis=0), logits
    )
    
    # Run evaluation with multi-injection damage (matches train_loop.py)
    step_metrics = evaluate_circuits_in_chunks(
        eval_fn=evaluate_model_stepwise_batched,
        wires=batch_wires,
        logits=batch_logits,
        knockout_patterns=None,  # Let evaluation system handle dynamic patterns
        target_chunk_size=eval_batch_size,
        model=model,
        x_data=x_data,
        y_data=y_data,
        input_n=input_n,
        arity=arity,
        circuit_hidden_dim=circuit_hidden_dim,
        n_message_steps=n_message_steps,
        loss_type=loss_type,
        layer_sizes=layer_sizes,
        return_per_pattern=True,  # Get per-pattern data for analysis
        layer_neighbors=layer_neighbors,
        # Multi-damage parameters
        damage_mode=damage_mode,
        damage_injection_mode=damage_injection_mode,
        max_damage_per_circuit=max_damage_per_circuit,
        greedy_ordered_indices=greedy_ordered_indices,
        greedy_window_size=greedy_window_size,
        greedy_injection_recover_steps=greedy_injection_recover_steps,
        damage_start_offset=damage_start_offset,
        damage_start_offset_random=damage_start_offset_random,
        damage_start_offset_seed=damage_start_offset_seed,
        knockout_vocabulary=None,  # Force unseen patterns (or use vocabulary if available)
    )
    
    return step_metrics


def print_evaluation_results(step_metrics, initial_accuracy=None, initial_hard_accuracy=None):
    """
    Print evaluation results in a readable format.
    
    Args:
        step_metrics: Dictionary with step-wise metrics
        initial_accuracy: Initial accuracy (for comparison)
        initial_hard_accuracy: Initial hard accuracy (for comparison)
    """
    log.info("\n" + "="*80)
    log.info("EVALUATION RESULTS")
    log.info("="*80)
    
    steps = step_metrics["step"]
    soft_losses = step_metrics["soft_loss"]
    hard_losses = step_metrics["hard_loss"]
    soft_accuracies = step_metrics["soft_accuracy"]
    hard_accuracies = step_metrics["hard_accuracy"]
    
    log.info(f"\nTotal steps: {len(steps)}")
    log.info(f"Step range: {steps[0]} to {steps[-1]}")
    
    # Print initial state
    if len(steps) > 0:
        log.info(f"\nStep {steps[0]} (Initial):")
        log.info(f"  Loss: {soft_losses[0]:.6f}")
        log.info(f"  Hard Loss: {hard_losses[0]:.4f}")
        log.info(f"  Accuracy: {soft_accuracies[0]:.4f}")
        log.info(f"  Hard Accuracy: {hard_accuracies[0]:.4f}")
        if initial_hard_accuracy is not None:
            drift = hard_accuracies[0] - initial_hard_accuracy
            log.info(f"  Drift from circuit init: {drift:+.4f}")
    
    # Print key milestones
    milestones = [1, 5, 10, 50, 100, 200]
    if len(steps) > 1:
        log.info(f"\nKey Milestones:")
        for milestone in milestones:
            if milestone < len(steps):
                idx = milestone
                log.info(f"Step {steps[idx]}:")
                log.info(f"  Loss: {soft_losses[idx]:.6f} (x{soft_losses[idx]/soft_losses[0]:.2f} from initial)")
                log.info(f"  Hard Loss: {hard_losses[idx]:.4f}")
                log.info(f"  Accuracy: {soft_accuracies[idx]:.4f}")
                log.info(f"  Hard Accuracy: {hard_accuracies[idx]:.4f} (drift: {hard_accuracies[idx]-hard_accuracies[0]:+.4f})")
    
    # Print final state
    if len(steps) > 1:
        log.info(f"\nStep {steps[-1]} (Final):")
        log.info(f"  Loss: {soft_losses[-1]:.6f} (x{soft_losses[-1]/soft_losses[0]:.2f} from initial)")
        log.info(f"  Hard Loss: {hard_losses[-1]:.4f}")
        log.info(f"  Accuracy: {soft_accuracies[-1]:.4f}")
        log.info(f"  Hard Accuracy: {hard_accuracies[-1]:.4f} (drift: {hard_accuracies[-1]-hard_accuracies[0]:+.4f})")
    
    # Print per-pattern statistics if available
    if "per_pattern" in step_metrics and "pattern_hard_accuracies" in step_metrics["per_pattern"]:
        per_pattern_accuracies = step_metrics["per_pattern"]["pattern_hard_accuracies"]
        final_accuracies = per_pattern_accuracies[-1] if len(per_pattern_accuracies) > 0 else None
        if final_accuracies is not None:
            log.info(f"\nPer-Pattern Statistics (Final Step):")
            log.info(f"  Mean: {float(jp.mean(final_accuracies)):.4f}")
            log.info(f"  Std: {float(jp.std(final_accuracies)):.4f}")
            log.info(f"  Min: {float(jp.min(final_accuracies)):.4f}")
            log.info(f"  Max: {float(jp.max(final_accuracies)):.4f}")
    
    log.info("\n" + "="*80)


def plot_stepwise_accuracy(
    step_metrics,
    initial_hard_accuracy=None,
    output_dir="results",
    run_id="test",
    damage_start_offset=0,
    greedy_injection_recover_steps=10,
):
    """
    Plot step-wise hard accuracy throughout multi-injection damage evaluation.
    
    Args:
        step_metrics: Dictionary with step-wise metrics from evaluation
        initial_hard_accuracy: Initial hard accuracy (for reference line)
        output_dir: Directory to save the plot
        run_id: Run ID for filename
        damage_start_offset: Offset before first damage injection
        greedy_injection_recover_steps: Steps between damage injections
    """
    steps = step_metrics["step"]
    hard_accuracies = step_metrics["hard_accuracy"]
    soft_accuracies = step_metrics.get("soft_accuracy", [])
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot hard accuracy (primary metric)
    ax.plot(steps, hard_accuracies, 'b-', linewidth=2, label='Hard Accuracy', marker='o', markersize=3)
    
    # Plot soft accuracy if available (lighter, secondary)
    if soft_accuracies and len(soft_accuracies) == len(steps):
        ax.plot(steps, soft_accuracies, 'b--', linewidth=1, alpha=0.5, label='Soft Accuracy', marker='s', markersize=2)
    
    # Add initial accuracy reference line
    if initial_hard_accuracy is not None:
        ax.axhline(
            y=initial_hard_accuracy,
            color='g',
            linestyle=':',
            linewidth=1.5,
            alpha=0.7,
            label=f'Initial Hard Accuracy ({initial_hard_accuracy:.4f})'
        )
    
    # Mark damage injection points
    # Damage starts at: damage_start_offset + 1
    # Then every: greedy_injection_recover_steps + 1 steps
    if damage_start_offset >= 0 and greedy_injection_recover_steps > 0:
        damage_steps = []
        first_damage_step = damage_start_offset + 1
        if first_damage_step <= steps[-1]:
            damage_steps.append(first_damage_step)
            next_damage = first_damage_step + greedy_injection_recover_steps + 1
            while next_damage <= steps[-1]:
                damage_steps.append(next_damage)
                next_damage += greedy_injection_recover_steps + 1
        
        if damage_steps:
            # Mark damage injection points
            # Find closest step index for each damage step
            for i, damage_step in enumerate(damage_steps):
                # Find the closest step index
                step_idx = min(range(len(steps)), key=lambda j: abs(steps[j] - damage_step))
                closest_step = steps[step_idx]
                
                # Only mark if we're reasonably close (within 2 steps)
                if abs(closest_step - damage_step) <= 2:
                    ax.axvline(
                        x=closest_step,
                        color='r',
                        linestyle='--',
                        linewidth=1,
                        alpha=0.5,
                        label='Damage Injection' if i == 0 else ''
                    )
                    # Add annotation
                    ax.annotate(
                        f'Damage #{i+1}',
                        xy=(closest_step, hard_accuracies[step_idx]),
                        xytext=(5, 10),
                        textcoords='offset points',
                        fontsize=8,
                        color='red',
                        alpha=0.7,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='red', linewidth=0.5)
                    )
    
    # Formatting
    ax.set_xlabel('Step', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Step-wise Hard Accuracy During Multi-Injection Damage Evaluation', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=10)
    ax.set_ylim([0, 1.1])  # Slightly above 1.0 to show perfect accuracy clearly
    
    # Add statistics text box
    if len(hard_accuracies) > 0:
        initial_acc = hard_accuracies[0]
        final_acc = hard_accuracies[-1]
        drift = final_acc - initial_acc
        stats_text = f'Initial: {initial_acc:.4f}\nFinal: {final_acc:.4f}\nDrift: {drift:+.4f}'
        ax.text(
            0.02, 0.98, stats_text,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        )
    
    plt.tight_layout()
    
    # Save plot
    output_path = os.path.join(output_dir, f"stepwise_accuracy_{run_id}.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    log.info(f"Saved step-wise accuracy plot to: {output_path}")
    
    # Also save as PDF for better quality
    output_path_pdf = os.path.join(output_dir, f"stepwise_accuracy_{run_id}.pdf")
    plt.savefig(output_path_pdf, bbox_inches='tight')
    log.info(f"Saved step-wise accuracy plot (PDF) to: {output_path_pdf}")
    
    plt.close()


def main():
    """Main function to run the comparison test."""
    log.info("="*80)
    log.info("GUI vs Training Evaluation Conditions Test")
    log.info("="*80)
    
    # Configuration
    run_id = "vayt4820"
    eval_batch_size = 16  # Matches training eval batch size
    # Note: wiring_seed will be taken from loaded config's test_seed (matches train.py)
    
    # Step 1: Load model exactly like GUI
    log.info("\n[Step 1] Loading model from WandB...")
    model, config, loaded_dict = load_model_like_gui(run_id)
    
    # Step 2: Set up circuit exactly like training
    log.info("\n[Step 2] Setting up circuit...")
    # Use test_seed from config (matches train.py: wiring_fixed_key=jax.random.PRNGKey(cfg.test_seed))
    test_seed = getattr(config, "test_seed", 42) if hasattr(config, "test_seed") else 42
    wiring_key = jax.random.PRNGKey(test_seed)
    log.info(f"Using test_seed={test_seed} for wiring (from loaded config)")
    wires, logits, layer_sizes, x_data, y_data, task_name = setup_circuit_like_gui(
        config, wiring_key
    )
    
    # Step 3: Run training evaluation (multi-injection damage)
    log.info("\n[Step 3] Running training evaluation...")
    step_metrics = run_training_evaluation(
        model=model,
        config=config,
        wires=wires,
        logits=logits,
        layer_sizes=layer_sizes,
        x_data=x_data,
        y_data=y_data,
        eval_batch_size=eval_batch_size,
    )
    
    # Step 4: Print results
    log.info("\n[Step 4] Analyzing results...")
    
    # Get initial accuracy for comparison
    loss_type = getattr(config.training, "loss_type", "l4") if hasattr(config, "training") else "l4"
    initial_loss, initial_aux = get_loss_from_wires_logits(
        logits, wires, x_data, y_data, loss_type
    )
    _, _, _, initial_accuracy, initial_hard_accuracy, _, _ = initial_aux
    
    print_evaluation_results(
        step_metrics,
        initial_accuracy=float(initial_accuracy),
        initial_hard_accuracy=float(initial_hard_accuracy),
    )
    
    # Step 4.5: Plot step-wise accuracy
    log.info("\n[Step 4.5] Generating step-wise accuracy plot...")
    
    # Extract damage parameters for plotting
    eval_cfg = getattr(config, "eval", None) if hasattr(config, "eval") else None
    knockout_eval = getattr(eval_cfg, "knockout_eval", None) if eval_cfg and hasattr(eval_cfg, "knockout_eval") else None
    pool_cfg = getattr(config, "pool", None) if hasattr(config, "pool") else None
    
    damage_start_offset = int(getattr(knockout_eval, "damage_start_offset", 0) if knockout_eval and hasattr(knockout_eval, "damage_start_offset") else 0)
    greedy_injection_recover_steps = int(getattr(knockout_eval, "greedy_injection_recover_steps", 10) if knockout_eval and hasattr(knockout_eval, "greedy_injection_recover_steps") else 10)
    
    plot_stepwise_accuracy(
        step_metrics=step_metrics,
        initial_hard_accuracy=float(initial_hard_accuracy),
        output_dir="results",
        run_id=run_id,
        damage_start_offset=damage_start_offset,
        greedy_injection_recover_steps=greedy_injection_recover_steps,
    )
    
    # Step 5: Summary and interpretation
    log.info("\n" + "="*80)
    log.info("INTERPRETATION")
    log.info("="*80)
    
    hard_accuracies = step_metrics["hard_accuracy"]
    if len(hard_accuracies) > 1:
        initial_acc = hard_accuracies[0]
        final_acc = hard_accuracies[-1]
        drift = final_acc - initial_acc
        
        log.info(f"\nAccuracy Drift Analysis:")
        log.info(f"  Initial (step 0): {initial_acc:.4f}")
        log.info(f"  Final (step {step_metrics['step'][-1]}): {final_acc:.4f}")
        log.info(f"  Total drift: {drift:+.4f}")
        
        if drift < -0.1:
            log.info(f"\n⚠️  SIGNIFICANT DEGRADATION DETECTED")
            log.info(f"   The model shows significant accuracy drift even under training evaluation conditions.")
            log.info(f"   This suggests the issue is NOT specific to GUI evaluation conditions.")
        elif drift < -0.01:
            log.info(f"\n⚠️  MODERATE DEGRADATION DETECTED")
            log.info(f"   The model shows moderate accuracy drift under training evaluation conditions.")
            log.info(f"   This may indicate a general model behavior issue.")
        else:
            log.info(f"\n✓ STABLE PERFORMANCE")
            log.info(f"   The model maintains stable accuracy under training evaluation conditions.")
            log.info(f"   If GUI shows drift, the issue is likely GUI-specific evaluation conditions.")
    
    log.info("\n" + "="*80)
    log.info("Test complete!")
    log.info("="*80)


if __name__ == "__main__":
    main()

