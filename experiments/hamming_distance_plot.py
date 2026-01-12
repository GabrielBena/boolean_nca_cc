#!/usr/bin/env python3
"""
Experiment: GNN vs Backprop comparison under knockout patterns, and LUT truth-table distances.

This script implements a streamlined, run ID-driven analysis:
- GNN run ID is the single source of truth for ALL configuration
- Automatically loads GNN model, training config, and parameters from WandB
- Uses GNN config for circuit structure, BP parameters, data generation, seeds, etc.
- No manual config overrides - everything derived from the training run
- Builds baseline circuit using GNN config parameters
- Runs baseline backprop training (no knockout)
- Runs backprop training per knockout pattern (same wiring)
- Runs GNN evaluation per knockout pattern (same wiring)
- Computes Hamming distances between hard LUT truth tables of baseline vs perturbed
  excluding any gates that are knocked out in the perturbed configuration
- Compares GNN vs BP recovery performance
- Writes a CSV summary and generates unified comparison plots

CLI: Only requires --run-id (defaults to "nypyrbwh") and --methods selection.
All other parameters (vocab size, damage prob, loss type, etc.) come from GNN config.
"""

import os
import argparse
import json
import logging
from typing import List, Dict, Any

import jax
import jax.numpy as jp
from omegaconf import OmegaConf

# Setup logging
log = logging.getLogger(__name__)

from boolean_nca_cc.circuits.model import gen_circuit
from boolean_nca_cc.circuits.tasks import get_task_data
from boolean_nca_cc.circuits.train import (
    TrainState,
    loss_f_l4,
    loss_f_bce,
    train_step,
)
from boolean_nca_cc.circuits.model import generate_layer_sizes
from boolean_nca_cc.training.pool.structural_perturbation import (
    create_reproducible_knockout_pattern,
)
from boolean_nca_cc.training.checkpointing import (
    load_best_model_from_wandb,
    load_checkpoint,
    instantiate_model_from_config,
    load_config_from_wandb,
)
from boolean_nca_cc.training.evaluation import (
    get_loss_from_wires_logits,
    evaluate_model_stepwise_batched,
)
from boolean_nca_cc.training.preconfigure import preconfigure_circuit_logits
from boolean_nca_cc.analysis.hamming_utils import (
    _hard_truth_tables_from_logits,
    _active_gate_mask_from_knockout,
    _hamming_distance_tables,
)


def _run_bp_single(cfg, x_data, y_data, loss_type: str):
    key = jax.random.PRNGKey(cfg.test_seed)
    wires, logits = gen_circuit(key, cfg.circuit.layer_sizes, arity=cfg.circuit.arity)

    if cfg.backprop.optimizer == "adamw":
        import optax

        opt = optax.adamw(
            cfg.backprop.learning_rate,
            b1=cfg.backprop.beta1,
            b2=cfg.backprop.beta2,
            weight_decay=cfg.backprop.weight_decay,
        )
    else:
        import optax

        opt = optax.adam(cfg.backprop.learning_rate)

    state = TrainState(params=logits, opt_state=opt.init(logits))

    _train_step_fn = lambda st: train_step(
        state=st,
        opt=opt,
        wires=wires,
        x=x_data,
        y0=y_data,
        loss_type=loss_type,
        do_train=True,
    )

    for _ in range(cfg.backprop.epochs):
        _, _, state = _train_step_fn(state)

    loss_fn = loss_f_l4 if loss_type == "l4" else loss_f_bce
    final_loss, final_aux = loss_fn(state.params, wires, x_data, y_data)

    return dict(params=state.params, wires=wires, final_loss=float(final_loss))


def _run_bp_with_knockouts(
    cfg,
    x_data,
    y_data,
    loss_type: str,
    knockout_patterns,
    layer_sizes,
    baseline_params,
    baseline_wires,
    reversible_bias=-10.0,
    damage_start_offset=0,
):
    """
    Run backprop training with knockout patterns in reversible mode.

    Damage injection strategy:
    - Apply the knockout pattern once at step (damage_start_offset + 1) via bias
    - Let the model recover afterward (no persistent masking)
    
    Args:
        damage_start_offset: Number of steps to run before first damage injection (0 = damage at step 1)
    """
    # Use baseline optimized circuit instead of generating fresh
    wires = baseline_wires
    logits = baseline_params  # Start from optimized baseline, not NOPs

    if cfg.backprop.optimizer == "adamw":
        import optax

        opt = optax.adamw(
            cfg.backprop.learning_rate,
            b1=cfg.backprop.beta1,
            b2=cfg.backprop.beta2,
            weight_decay=cfg.backprop.weight_decay,
        )
    else:
        import optax

        opt = optax.adam(cfg.backprop.learning_rate)

    results = []
    from tqdm.auto import tqdm
    for pattern in tqdm(knockout_patterns, desc="Training knockout patterns"):
        state = TrainState(params=logits, opt_state=opt.init(logits))

        # Damage injection step: damage_start_offset + 1 (e.g., offset=15 means damage at step 16)
        damage_injection_step = damage_start_offset + 1
        
        def _step(st, step_count):
            # Determine if damage should be applied at this step
            # step_count is 0-indexed, so step_count=0 is the first step
            # damage_injection_step is 1-indexed (e.g., offset=15 means step 16)
            # So we check: step_count == (damage_injection_step - 1)
            apply_damage_now = (step_count == (damage_injection_step - 1))
            
            # Apply pattern only at the injection step for reversible damage
            knockout_pattern_for_step = pattern if apply_damage_now else None
            
            return train_step(
                state=st,
                opt=opt,
                wires=wires,
                x=x_data,
                y0=y_data,
                loss_type=loss_type,
                do_train=True,
                knockout_pattern=knockout_pattern_for_step,
                layer_sizes=cfg.circuit.layer_sizes,
                damage_behavior="reversible",
                reversible_bias=reversible_bias,
                step_count=step_count,
                apply_damage_now=apply_damage_now,
            )

        for step_count in range(cfg.backprop.epochs):
            _, _, state = _step(state, step_count)

        # Final evaluation without persistent masks (reversible only)
        loss_fn = loss_f_l4 if loss_type == "l4" else loss_f_bce
        final_loss, final_aux = loss_fn(state.params, wires, x_data, y_data)
        final_hard_accuracy = float(final_aux["hard_accuracy"])
        final_hard_loss = float(final_aux["hard_loss"])

        results.append(dict(
            params=state.params, 
            pattern=pattern,
            final_hard_accuracy=final_hard_accuracy,
            final_hard_loss=final_hard_loss
        ))

    return dict(wires=wires, per_pattern=results)


def _pairwise_hamming_between_knockouts(
    ko_tables_list: List[List[jp.ndarray]],
    ko_active_masks_list: List[List[jp.ndarray]],
):
    """
    Compute an all-to-all matrix of bitwise fraction differences between KO runs.

    Uses intersection of active gates in both patterns for each pair and layer.
    Returns a 2D Python list of floats (NaN when no comparable gates).
    """
    n = len(ko_tables_list)
    import math

    matrix = [[math.nan for _ in range(n)] for _ in range(n)]

    for i in range(n):
        for j in range(n):
            total_diff = 0.0
            total_count = 0.0
            for layer_idx, (ti_layer, tj_layer, mi_layer, mj_layer) in enumerate(
                zip(ko_tables_list[i], ko_tables_list[j], ko_active_masks_list[i], ko_active_masks_list[j])
            ):
                # intersection of active gates
                inter_mask = jp.logical_and(mi_layer, mj_layer)
                inter_idx = jp.where(inter_mask)[0]
                if inter_idx.size == 0:
                    continue
                A = ti_layer[inter_idx]
                B = tj_layer[inter_idx]
                diffs = jp.not_equal(A, B).astype(jp.float32)
                total_diff += float(jp.sum(diffs))
                total_count += float(diffs.size)
            matrix[i][j] = (total_diff / total_count) if total_count > 0 else float('nan')

    return matrix




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default=None)
    # Config loading - optional, defaults to configs/config.yaml
    parser.add_argument("--config", type=str, default="configs/config.yaml", 
                        help="Path to config YAML file (default: configs/config.yaml)")
    parser.add_argument("--run-id", type=str, default="nypyrbwh", 
                        help="WandB run id for GNN model load (default: nypyrbwh)")
    parser.add_argument("--checkpoint", type=str, default=None, 
                        help="Local checkpoint path .pkl for GNN model (optional, only needed if --methods includes 'gnn')")
    # Analysis method selection
    parser.add_argument("--methods", type=str, default="bp", help="gnn,bp,both (default: bp)")
    # GNN evaluation parameters
    parser.add_argument("--n-message-steps", type=int, default=None,
                        help="Number of message passing steps for GNN evaluation (default: 26, or from config if available, CLI override takes precedence)")
    # Knockout size variation
    parser.add_argument("--knockout-sizes", type=str, default=None, 
                        help="Comma-separated list of knockout sizes (e.g., '1,10,20,40'). If not provided, uses damage_prob from config.")
    parser.add_argument("--patterns-per-size", type=int, default=None,
                        help="Number of patterns to generate per knockout size. If not provided, uses vocabulary_size from config.")
    parser.add_argument("--reversible-bias", type=float, default=None,
                        help="Bias value for reversible damage mode (default: -10.0)")
    parser.add_argument("--damage-start-offset", type=int, default=None,
                        help="Number of steps before damage injection (default: 5, damage at step 6)")
    parser.add_argument("--exclude-knocked-out-gates", action="store_true", default=False,
                        help="Exclude knocked-out gates from hamming distance calculation (default: False, i.e., include all gates).")
    args = parser.parse_args()

    # Parse methods selection early to determine what needs to be loaded
    methods_str = (args.methods or "gnn").lower()
    if methods_str == "both":
        methods = {"bp", "gnn"}
    else:
        methods = set(m.strip() for m in methods_str.split(","))

    # Load config first (always needed for circuit parameters, even for BP-only runs)
    # This ensures BP baseline and analysis use identical parameters to GNN training
    if args.checkpoint is not None:
        loaded = load_checkpoint(args.checkpoint)
        cfg = OmegaConf.create(loaded.get("config", {}))
        gnn_training_config = cfg
    else:
        # Load config from WandB (without loading the model yet)
        # Use standard artifact filename to get config
        filename_to_load = "best_model_eval_ko_hard_accuracy"
        cfg, _, _ = load_config_from_wandb(
            run_id=args.run_id,
            filename=filename_to_load,
            select_by_best_metric=False,  # Just get config, not necessarily best model
        )
        gnn_training_config = cfg

    # Parse and set all configuration variables from args or config
    # Seed for random operations
    seed = cfg.get("seed", cfg.get("test_seed", 42))
    
    # Knockout sizes: parse from args or use config default
    if args.knockout_sizes is not None:
        knockout_sizes = [int(x.strip()) for x in args.knockout_sizes.split(",")]
    else:
        # Use damage_prob from config as single value, or default
        damage_prob = cfg.get("pool", {}).get("damage_prob", cfg.get("backprop", {}).get("knockout_vocabulary", {}).get("damage_prob", 10))
        knockout_sizes = [damage_prob]
    
    # Patterns per size: parse from args or use config default
    if args.patterns_per_size is not None:
        patterns_per_size = args.patterns_per_size
    else:
        patterns_per_size = cfg.get("pool", {}).get("vocabulary_size", cfg.get("backprop", {}).get("knockout_vocabulary", {}).get("vocabulary_size", 10))
    
    # Reversible bias: Priority: CLI override > Script default
    if args.reversible_bias is not None:
        reversible_bias = args.reversible_bias
    else:
        reversible_bias = -10.0  # Default
    
    # Only reversible mode is supported for this experiment
    damage_behavior = "reversible"
    
    # N message steps: Priority: CLI override > Script default (26)
    # Script defaults take precedence over config for simplicity
    if args.n_message_steps is not None:
        n_message_steps = args.n_message_steps
    else:
        n_message_steps = 26  # Default for this experiment
    
    # Damage start offset: Priority: CLI override > Script default (5)
    # Script defaults take precedence over config for simplicity
    if args.damage_start_offset is not None:
        damage_start_offset = args.damage_start_offset
    else:
        damage_start_offset = 5  # Default: damage at step 6, 20 recovery steps (with 26 total)
    
    # Include all gates mode: Default is True (include all gates), unless --exclude-knocked-out-gates is set
    include_all_gates = not args.exclude_knocked_out_gates
    

    # Only load GNN model if GNN is in the methods
    gnn_model = None
    gnn_hidden_dim = None
    if "gnn" in methods:
        if args.checkpoint is not None:
            loaded = load_checkpoint(args.checkpoint)
            gnn_model = instantiate_model_from_config(cfg, seed=cfg.get("seed", 0))
            from flax import nnx as _nnx
            _nnx.update(gnn_model, loaded["model"])
            gnn_hidden_dim = int(cfg.model.get("circuit_hidden_dim", 16))
            
            # Extract and display epoch/step information
            step = loaded.get("step")
            checkpoint_config = loaded.get("config", {})
            if isinstance(checkpoint_config, dict):
                epoch = checkpoint_config.get("epoch")
            else:
                epoch = getattr(checkpoint_config, "epoch", None)
            
            if step is not None:
                print(f"Loaded checkpoint at step: {step}")
            if epoch is not None:
                print(f"Loaded checkpoint at epoch: {epoch}")
            if step is None and epoch is None:
                print("Warning: Could not determine checkpoint epoch/step from loaded data")
        else:
            # Load the actual model now
            gnn_model, loaded_dict, _ = load_best_model_from_wandb(
                run_id=args.run_id,
                seed=0,  # Use default seed, will be overridden by GNN config
                filename=filename_to_load,
            )
            gnn_hidden_dim = int(cfg.model.get("circuit_hidden_dim", 16))
            
            # Extract and display epoch/step information
            step = loaded_dict.get("step")
            checkpoint_config = loaded_dict.get("config", {})
            if isinstance(checkpoint_config, dict):
                epoch = checkpoint_config.get("epoch")
            else:
                epoch = getattr(checkpoint_config, "epoch", None)
            
            if step is not None:
                print(f"Loaded checkpoint at step: {step}")
            if epoch is not None:
                print(f"Loaded checkpoint at epoch: {epoch}")
            if step is None and epoch is None:
                print("Warning: Could not determine checkpoint epoch/step from loaded data")

    # Set output directory with f-string if not provided
    if args.output is None:
        ko_sizes_str = "_".join(map(str, knockout_sizes))
        gates_mode = "allgates" if include_all_gates else "activeonly"
        args.output = f"results/hamming_scaled{ko_sizes_str}_p{patterns_per_size}_{damage_behavior}_{gates_mode}"
    log.info(f"Output directory: {args.output}")
    log.info(f"Hamming distance mode: {'include all gates' if include_all_gates else 'exclude knocked-out gates'}")

    # Ensure layer_sizes are present
    circuit_cfg = cfg.get("circuit", {})
    input_n = circuit_cfg.get("input_bits", 12)
    output_n = circuit_cfg.get("output_bits", 12)
    arity = circuit_cfg.get("arity", 4)
    num_layers = circuit_cfg.get("num_layers", 3)
    
    layer_sizes = circuit_cfg.get("layer_sizes", None)
    if layer_sizes is None:
        layer_sizes = generate_layer_sizes(input_n, output_n, arity, layer_n=num_layers)
    else:
        layer_sizes = layer_sizes

    # Data
    case_n = 1 << input_n
    x, y0 = get_task_data(
        cfg.circuit.task, 
        case_n, 
        max_samples=cfg.circuit.get("max_task_samples", 100000),
        sample_seed=cfg.test_seed,
        input_bits=input_n, 
        output_bits=output_n
    )

    # Loss type from config
    loss_type = cfg.get("training", {}).get("loss_type", "l4")

    # Baseline circuit creation - use preconfigured if GNN was trained in repair mode
    if "gnn" in methods and gnn_training_config is not None:
        # Check if GNN was trained in repair mode and use the same preconfigured circuit
        training_mode = gnn_training_config.get("training_mode", "growth")
        if training_mode == "repair":
            log.info("GNN was trained in repair mode - recreating exact preconfigured baseline circuit")
            
            # Extract preconfiguration parameters from training config
            preconfig_steps = gnn_training_config.get("preconfig_steps", 200)
            preconfig_lr = gnn_training_config.get("preconfig_lr", 1e-2)
            wiring_fixed_key = gnn_training_config.get("wiring_fixed_key", cfg.get("test_seed", 42))
            
            # Convert wiring_fixed_key to JAX key if it's an integer
            if isinstance(wiring_fixed_key, int):
                wiring_fixed_key = jax.random.PRNGKey(wiring_fixed_key)
            
            # Recreate the exact same preconfigured circuit used during training
            base_wires, base_logits = preconfigure_circuit_logits(
                wiring_key=wiring_fixed_key,
                layer_sizes=layer_sizes,
                arity=arity,
                x_data=x,
                y_data=y0,
                loss_type=loss_type,
                steps=preconfig_steps,
                lr=preconfig_lr,
            )
            
            # Use preconfigured circuit for BP baseline (no additional optimization needed)
            bp_baseline = {
                "wires": base_wires,
                "params": base_logits,
                "final_loss": 0.0  # Preconfigured circuits are already optimized
            }
            
            log.info(f"Recreated preconfigured circuit with {preconfig_steps} steps, lr={preconfig_lr}")
        else:
            # GNN was trained in growth mode - use standard BP baseline
            log.info("GNN was trained in growth mode - using standard BP baseline")
            # Create a mock config object for backprop training
            from types import SimpleNamespace
            backprop_cfg = cfg.get("backprop", {})
            mock_cfg = SimpleNamespace(
                backprop=SimpleNamespace(
                    optimizer=backprop_cfg.get("optimizer", "adam"),
                    learning_rate=backprop_cfg.get("learning_rate", 1.0),
                    beta1=backprop_cfg.get("beta1", 0.9),
                    beta2=backprop_cfg.get("beta2", 0.999),
                    weight_decay=backprop_cfg.get("weight_decay", 0.0),
                    epochs=backprop_cfg.get("epochs", 200),
                ),
                circuit=SimpleNamespace(
                    layer_sizes=layer_sizes,
                    arity=arity,
                ),
                test_seed=cfg.get("test_seed", seed),
            )
            bp_baseline = _run_bp_single(mock_cfg, x, y0, loss_type=loss_type)
    else:
        # No GNN or no training config - use standard BP baseline
        from types import SimpleNamespace
        backprop_cfg = cfg.get("backprop", {})
        mock_cfg = SimpleNamespace(
            backprop=SimpleNamespace(
                optimizer=backprop_cfg.get("optimizer", "adam"),
                learning_rate=backprop_cfg.get("learning_rate", 1.0),
                beta1=backprop_cfg.get("beta1", 0.9),
                beta2=backprop_cfg.get("beta2", 0.999),
                weight_decay=backprop_cfg.get("weight_decay", 0.0),
                epochs=backprop_cfg.get("epochs", 200),
            ),
            circuit=SimpleNamespace(
                layer_sizes=layer_sizes,
                arity=arity,
            ),
            test_seed=cfg.get("test_seed", seed),
        )
        bp_baseline = _run_bp_single(mock_cfg, x, y0, loss_type=loss_type)

    # Single damage mode for this simplified experiment
    damage_mode = "shotgun"

    # Truth tables
    baseline_tables = _hard_truth_tables_from_logits(bp_baseline["params"])  # per-layer
    
    # Compute baseline circuit performance metrics
    loss_fn = loss_f_l4 if loss_type == "l4" else loss_f_bce
    baseline_loss, baseline_aux = loss_fn(bp_baseline["params"], bp_baseline["wires"], x, y0)
    baseline_hard_accuracy = float(baseline_aux["hard_accuracy"])
    baseline_hard_loss = float(baseline_aux["hard_loss"])
    log.info(f"Baseline circuit performance: accuracy={baseline_hard_accuracy:.4f}, loss={baseline_hard_loss:.4f}")

    # Storage
    os.makedirs(args.output, exist_ok=True)
    summary_rows = []
    last_method_ko_tables: List[List[jp.ndarray]] = []
    last_method_active_masks: List[List[jp.ndarray]] = []

    # Loop over damage modes and knockout sizes
    # seed is already defined earlier
    rng = jax.random.PRNGKey(seed)
    for ko_size in knockout_sizes:
        log.info(
            f"Processing knockout size {ko_size} with damage_mode {damage_mode}, damage_behavior {damage_behavior}"
        )
        rng, patterns_key = jax.random.split(rng)
        pattern_keys = jax.random.split(patterns_key, patterns_per_size)
        vocab = jax.vmap(
            lambda k: create_reproducible_knockout_pattern(
                k, layer_sizes, ko_size
            )
        )(pattern_keys)

        # BP method
        if "bp" in methods:
            # Create a mock config object for backprop training
            from types import SimpleNamespace
            backprop_cfg = cfg.get("backprop", {})
            mock_cfg = SimpleNamespace(
                backprop=SimpleNamespace(
                    optimizer=backprop_cfg.get("optimizer", "adam"),
                    learning_rate=backprop_cfg.get("learning_rate", 1.0),
                    beta1=backprop_cfg.get("beta1", 0.9),
                    beta2=backprop_cfg.get("beta2", 0.999),
                    weight_decay=backprop_cfg.get("weight_decay", 0.0),
                    epochs=backprop_cfg.get("epochs", 200),
                ),
                circuit=SimpleNamespace(
                    layer_sizes=layer_sizes,
                    arity=arity,
                ),
                test_seed=cfg.get("test_seed", seed),
            )
            
            bp_knockouts = _run_bp_with_knockouts(
                mock_cfg,
                x,
                y0,
                loss_type=loss_type,
                knockout_patterns=vocab,
                layer_sizes=layer_sizes,
                baseline_params=bp_baseline["params"],
                baseline_wires=bp_baseline["wires"],
                reversible_bias=reversible_bias,
                damage_start_offset=damage_start_offset,
            )

            method_ko_tables: List[List[jp.ndarray]] = []
            method_active_masks: List[List[jp.ndarray]] = []
            for idx, item in enumerate(bp_knockouts["per_pattern"]):
                pattern = item["pattern"]
                pert_tables = _hard_truth_tables_from_logits(item["params"])  # per-layer
                active_masks = _active_gate_mask_from_knockout(layer_sizes, pattern)
                # Use include_all_gates setting from command line
                metrics = _hamming_distance_tables(baseline_tables, pert_tables, active_masks, include_all_gates=include_all_gates)

                row = {
                    "pattern_idx": idx,
                    "knockout_size": ko_size,
                    "damage_mode": damage_mode,
                    "damage_behavior": damage_behavior,
                    "method": "bp",
                    "overall_bitwise_fraction_diff": metrics["overall_bitwise_fraction_diff"],
                    "per_gate_mean_hamming": metrics["per_gate_mean_hamming"],
                    "counted_bits_total": metrics["counted_bits_total"],
                    "counted_gates_total": metrics["counted_gates_total"],
                    "final_hard_accuracy": item["final_hard_accuracy"],
                    "final_hard_loss": item["final_hard_loss"],
                }
                row["per_layer_bitwise_fraction_diff"] = json.dumps(metrics["per_layer_bitwise_fraction_diff"])
                summary_rows.append(row)
                method_ko_tables.append(pert_tables)
                method_active_masks.append(active_masks)

            last_method_ko_tables = method_ko_tables
            last_method_active_masks = method_active_masks

        # GNN method
        if "gnn" in methods and gnn_model is not None:
            method_ko_tables: List[List[jp.ndarray]] = []
            method_active_masks: List[List[jp.ndarray]] = []
            
            log.info(f"Evaluating {len(vocab)} GNN patterns in batch for KO size {ko_size}")
            
            # Force reversible behavior for alignment
            setattr(gnn_model, "damage_behavior", "reversible")
            
            # Batch baseline wires and logits for all patterns
            batch_wires = jax.tree.map(
                lambda x: jp.repeat(x[None, ...], patterns_per_size, axis=0),
                bp_baseline["wires"]
            )
            batch_logits = jax.tree.map(
                lambda x: jp.repeat(x[None, ...], patterns_per_size, axis=0),
                bp_baseline["params"]
            )
            
            vocab_array = vocab if not isinstance(vocab, list) else jp.stack(vocab)
            
            # Run batched evaluation using provided patterns (single injection, reversible)
            eval_results = evaluate_model_stepwise_batched(
                model=gnn_model,
                batch_wires=batch_wires,
                batch_logits=batch_logits,
                x_data=x,
                y_data=y0,
                input_n=input_n,
                arity=arity,
                circuit_hidden_dim=gnn_hidden_dim or 16,
                n_message_steps=n_message_steps,
                loss_type=loss_type,
                bidirectional_edges=True,
                layer_sizes=layer_sizes,
                knockout_patterns=vocab_array,
                return_per_pattern=True,  # Get per-pattern metrics
                damage_injection_mode="single",  # One damage per circuit
                damage_start_offset=damage_start_offset,
                damage_mode="shotgun",  # Static patterns (not periodic)
                greedy_window_size=ko_size,  # Number of gates to knock out (matches knockout size)
                track_pre_update_accuracy=True,  # DEBUG: Track accuracy before/after model updates
            )
            
            # DEBUG: Print pre-update tracking summary
            pre_update_tracking = eval_results.get("pre_update_tracking", {})
            if pre_update_tracking:
                damage_steps = pre_update_tracking.get("damage_injection_steps", [])
                pre_acc = pre_update_tracking.get("pre_update_hard_accuracy", [])
                post_acc = pre_update_tracking.get("post_update_hard_accuracy", [])
                log.info(f"DEBUG: Pre-update tracking results for ko_size={ko_size}:")
                for i, step_num in enumerate(damage_steps):
                    delta = post_acc[i] - pre_acc[i] if i < len(post_acc) else 0
                    log.info(f"  Step {step_num}: Pre={pre_acc[i]:.4f}, Post={post_acc[i]:.4f}, Delta={delta:+.4f}")

            # Access step-by-step metrics for ALL message passing steps
            # These are already stored in eval_results as lists (one entry per step)
            steps = eval_results.get("step", [])  # [0, 1, 2, ..., n_message_steps]
            hard_accuracies = eval_results.get("hard_accuracy", [])  # One per step
            soft_accuracies = eval_results.get("soft_accuracy", [])
            hard_losses = eval_results.get("hard_loss", [])
            soft_losses = eval_results.get("soft_loss", [])
            
            print(f"\n{'='*60}")
            print(f"Step-by-step performance for ko_size={ko_size}:")
            print(f"{'='*60}")
            print(f"{'Step':<6} {'Hard Acc':<10} {'Soft Acc':<10} {'Hard Loss':<12} {'Soft Loss':<12}")
            print(f"{'-'*60}")
            for i, step_num in enumerate(steps):
                print(f"{step_num:<6} {hard_accuracies[i]:<10.4f} {soft_accuracies[i]:<10.4f} "
                      f"{hard_losses[i]:<12.6f} {soft_losses[i]:<12.6f}")
            print(f"{'='*60}\n")
            
            # You can also access per-pattern step-by-step metrics:
            per_pattern = eval_results.get("per_pattern", {})
            pattern_hard_accuracies = per_pattern.get("pattern_hard_accuracies", None)
            # shape: [n_steps, batch_size] - each row is one step, each column is one pattern
            if pattern_hard_accuracies is not None:
                print(f"Per-pattern step-by-step accuracies shape: {pattern_hard_accuracies.shape}")
                # Example: Get accuracy at step 6 for all patterns
                damage_step_idx = damage_start_offset + 1  # Step where damage is injected
                if damage_step_idx < len(steps):
                    acc_at_damage = pattern_hard_accuracies[damage_step_idx]  # [batch_size]
                    print(f"  Accuracies at damage step {damage_step_idx}: {acc_at_damage}")

            # Extract per-pattern results
            per_pattern = eval_results.get("per_pattern", {})
            final_logits_list = per_pattern.get("pattern_logits", [])
            final_hard_accuracies = per_pattern.get("pattern_hard_accuracies", None)
            
            # Get final step logits (last element in the list)
            if final_logits_list:
                final_step_logits = final_logits_list[-1]  # [batch_size, ...logit_shapes...]
            else:
                raise ValueError("No per-pattern logits returned from evaluation")
            
            # Get final hard accuracies (last step)
            # final_hard_accuracies shape: [n_steps, batch_size] (JAX array)
            if final_hard_accuracies is not None:
                if isinstance(final_hard_accuracies, jp.ndarray) and final_hard_accuracies.shape[0] > 0:
                    final_hard_accuracies_array = final_hard_accuracies[-1]  # [batch_size] - last step
                elif isinstance(final_hard_accuracies, list) and len(final_hard_accuracies) > 0:
                    final_hard_accuracies_array = final_hard_accuracies[-1]  # [batch_size] - last step
                else:
                    final_hard_accuracies_array = None
            else:
                final_hard_accuracies_array = None
            
            # Process each pattern result
            for idx in range(patterns_per_size):
                # Extract logits for this pattern (unbatch from tree structure)
                pattern_final_logits = jax.tree.map(lambda x: x[idx], final_step_logits)
                
                # Compute final metrics for this pattern
                pattern_loss, pattern_aux = get_loss_from_wires_logits(
                    pattern_final_logits,
                    bp_baseline["wires"],  # Use original wires (not batched)
                    x,
                    y0,
                    loss_type
                )
                pattern_final_hard_accuracy = float(pattern_aux[4])  # hard_accuracy is index 4
                pattern_final_hard_loss = float(pattern_aux[0])  # hard_loss is index 0
                
                # Use per-pattern accuracy if available, otherwise computed
                if final_hard_accuracies_array is not None:
                    pattern_final_hard_accuracy = float(final_hard_accuracies_array[idx])
                
                # Get pattern from vocab
                pattern = vocab_array[idx]
                
                # Compute hamming distance
                pert_tables = _hard_truth_tables_from_logits(pattern_final_logits)  # per-layer
                active_masks = _active_gate_mask_from_knockout(layer_sizes, pattern)
                # Use include_all_gates setting from command line
                metrics = _hamming_distance_tables(baseline_tables, pert_tables, active_masks, include_all_gates=include_all_gates)
                
                row = {
                    "pattern_idx": idx,
                    "knockout_size": ko_size,
                    "damage_mode": damage_mode,
                    "damage_behavior": damage_behavior,
                    "method": "gnn",
                    "overall_bitwise_fraction_diff": metrics["overall_bitwise_fraction_diff"],
                    "per_gate_mean_hamming": metrics["per_gate_mean_hamming"],
                    "counted_bits_total": metrics["counted_bits_total"],
                    "counted_gates_total": metrics["counted_gates_total"],
                    "final_hard_accuracy": pattern_final_hard_accuracy,
                    "final_hard_loss": pattern_final_hard_loss,
                }
                row["per_layer_bitwise_fraction_diff"] = json.dumps(metrics["per_layer_bitwise_fraction_diff"])
                summary_rows.append(row)
                method_ko_tables.append(pert_tables)
                method_active_masks.append(active_masks)

            last_method_ko_tables = method_ko_tables
            last_method_active_masks = method_active_masks

    # Save CSV
    import pandas as pd

    df = pd.DataFrame(summary_rows)
    csv_path = os.path.join(args.output, "summary.csv")
    df.to_csv(csv_path, index=False)

    # Pairwise all-to-all matrix among KO patterns using intersection masks
    pairwise = []
    if last_method_ko_tables:
        pairwise = _pairwise_hamming_between_knockouts(last_method_ko_tables, last_method_active_masks)
    import pandas as pd  # safe to re-import; ensures availability
    df_pair = pd.DataFrame(pairwise)
    df_pair.index.name = "pattern_i"
    df_pair.columns = [f"pattern_{j}" for j in range(len(pairwise))]
    pairwise_csv = os.path.join(args.output, "pairwise_matrix.csv")
    df_pair.to_csv(pairwise_csv)

    # Save a small report
    report = {
        "num_patterns": len(summary_rows),
        "mean_overall_bitwise_fraction_diff": float(df["overall_bitwise_fraction_diff"].mean()) if len(df) else 0.0,
        "mean_per_gate_mean_hamming": float(df["per_gate_mean_hamming"].mean()) if len(df) else 0.0,
        "mean_final_hard_accuracy": float(df["final_hard_accuracy"].mean()) if len(df) else 0.0,
        "mean_final_hard_loss": float(df["final_hard_loss"].mean()) if len(df) else 0.0,
        # "config_path": os.path.abspath(args.config),
        "output_path": os.path.abspath(args.output),
        "pairwise_matrix_csv": os.path.abspath(pairwise_csv),
    }
    with open(os.path.join(args.output, "report.json"), "w") as f:
        json.dump(report, f, indent=2)

    # Generate visualizations
    try:
        # Import visualization functions - add project root to path if needed
        import sys
        import os as os_module
        # Get project root (parent of experiments directory)
        script_dir = os_module.path.dirname(os_module.path.abspath(__file__))
        project_root = os_module.path.dirname(script_dir)
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        
        from experiments.visualization.plot_accuracy_vs_distance import (
            plot_accuracy_vs_distance,
            plot_damage_size_vs_hamming,
        )  # type: ignore
        
        # Accuracy vs distance scatter plot
        if 'final_hard_accuracy' in df.columns and 'overall_bitwise_fraction_diff' in df.columns:
            accuracy_plot_path = os.path.join(args.output, "accuracy_vs_distance.png")
            plot_accuracy_vs_distance(df, accuracy_plot_path, color_by_method=True)
            print(f"Accuracy plot saved to: {accuracy_plot_path}")
            report["accuracy_plot_path"] = os.path.abspath(accuracy_plot_path)
        
        # Damage size vs hamming distance plot
        if 'knockout_size' in df.columns and 'per_gate_mean_hamming' in df.columns:
            damage_plot_path = os.path.join(args.output, "damage_size_vs_hamming.png")
            plot_damage_size_vs_hamming(
                df, 
                damage_plot_path, 
                color_by_method=True,
                baseline_accuracy=baseline_hard_accuracy,
                baseline_loss=baseline_hard_loss
            )
            print(f"Damage size vs hamming plot saved to: {damage_plot_path}")
            report["damage_size_plot_path"] = os.path.abspath(damage_plot_path)
            
    except ImportError as e:
        print(f"Warning: Could not import visualization module: {e}")
        print("Plots will not be generated.")
    except Exception as e:
        print(f"Warning: Error generating plots: {e}")
        import traceback
        traceback.print_exc()
        print("Continuing without visualization...")

    # Print concise report
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()


