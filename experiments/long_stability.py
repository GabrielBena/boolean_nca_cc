#!/usr/bin/env python3
"""
Long-stability evaluation: load a trained GNN from checkpoint (or WandB), run the
exact same eval loop as in hamming_distance_plot / evaluation.py, with arbitrarily
many message-passing steps and live readouts of hard accuracy.

Uses the same checkpointing and model loading as experiments/hamming_distance_plot.py
and boolean_nca_cc/training/checkpointing.py. No knockout/damage by default;
single circuit, no batch.
"""

import argparse
import logging
import sys
import yaml

import jax
import jax.numpy as jp
from omegaconf import OmegaConf

from boolean_nca_cc.circuits.model import gen_circuit, generate_layer_sizes
from boolean_nca_cc.circuits.tasks import get_task_data
from boolean_nca_cc.training.checkpointing import (
    load_best_model_from_wandb,
    load_checkpoint,
    instantiate_model_from_config,
    load_config_from_wandb,
)
from boolean_nca_cc.training.evaluation import (
    evaluate_model_stepwise_generator,
    evaluate_model_stepwise_batched,
)
from boolean_nca_cc.training.pool.structural_perturbation import (
    create_reproducible_knockout_pattern,
)
from boolean_nca_cc.training.preconfigure import preconfigure_circuit_logits

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Run GNN eval with arbitrary MP steps and live hard-accuracy readouts."
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default="ia2zjmru",
        help="WandB run id for GNN model (used when --checkpoint is not set)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Local checkpoint path .pkl for GNN model (overrides --run-id)",
    )
    parser.add_argument(
        "--n-message-steps",
        type=int,
        default=500,
        help="Number of message-passing steps (default: 500)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to config YAML (only used when loading from checkpoint)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only print step and hard_accuracy per line (no header/extra metrics)",
    )
    parser.add_argument(
        "--readout-every",
        type=int,
        default=50,
        help="Print metrics every N steps (default: 50); step 0 always printed",
    )
    # Perturbation (knockout) mode: same as hamming_distance_plot, then average outputs
    parser.add_argument(
        "--knockout-size",
        type=int,
        default=None,
        help="If set, run batched eval with this many gates knocked out per pattern (reversible, single injection)",
    )
    parser.add_argument(
        "--patterns-per-size",
        type=int,
        default=256,
        help="Number of knockout patterns when --knockout-size is set (default: 10)",
    )
    parser.add_argument(
        "--damage-start-offset",
        type=int,
        default=49,
        help="Steps before first damage when using knockout (default: 5, damage at step 6)",
    )
    parser.add_argument(
        "--reversible-bias",
        type=float,
        default=-10.0,
        help="Bias for reversible damage when using knockout (default: -10.0)",
    )
    args = parser.parse_args()

    # --- Load config (and optionally checkpoint state) ---
    loaded_ckpt = None
    if args.checkpoint is not None:
        loaded_ckpt = load_checkpoint(args.checkpoint)
        cfg = OmegaConf.create(loaded_ckpt.get("config", {}))
    else:
        filename_to_load = "best_model_eval_ko_hard_accuracy"
        cfg, _, _ = load_config_from_wandb(
            run_id=args.run_id,
            filename=filename_to_load,
            select_by_best_metric=False,
        )

    # --- Circuit params ---
    circuit_cfg = cfg.get("circuit", {})
    input_n = circuit_cfg.get("input_bits", 12)
    output_n = circuit_cfg.get("output_bits", 12)
    arity = circuit_cfg.get("arity", 4)
    num_layers = circuit_cfg.get("num_layers", 3)
    layer_sizes = circuit_cfg.get("layer_sizes", None)
    if layer_sizes is None:
        layer_sizes = generate_layer_sizes(input_n, output_n, arity, layer_n=num_layers)

    # --- Data ---
    case_n = 1 << input_n
    x, y0 = get_task_data(
        cfg.circuit.task,
        case_n,
        max_samples=circuit_cfg.get("max_task_samples", 100000),
        sample_seed=cfg.get("test_seed", 42),
        input_bits=input_n,
        output_bits=output_n,
    )
    loss_type = cfg.get("training", {}).get("loss_type", "l4")

    # --- Load GNN model ---
    gnn_hidden_dim = int(cfg.model.get("circuit_hidden_dim", 16))
    if args.checkpoint is not None:
        assert loaded_ckpt is not None
        gnn_model = instantiate_model_from_config(cfg, seed=cfg.get("seed", 0))
        from flax import nnx as _nnx
        _nnx.update(gnn_model, loaded_ckpt["model"])
        log.info("Loaded GNN from checkpoint: %s", args.checkpoint)
    else:
        filename_to_load = "best_model_eval_ko_hard_accuracy"
        gnn_model, loaded_dict, _ = load_best_model_from_wandb(
            run_id=args.run_id,
            seed=0,
            filename=filename_to_load,
        )
        log.info("Loaded GNN from WandB run: %s", args.run_id)

    # --- Single circuit: always use repair-mode preconfigured baseline ---
    # Preconfig params must match training: use backprop block (train.py / train_loop.py).
    # long_stability previously used top-level preconfig_steps/preconfig_lr with wrong default lr=1e-2.
    training_mode = "repair"
    if training_mode == "repair":
        log.info("Recreating preconfigured baseline circuit (repair mode)")
        # Resolve backprop config like test_gui_vs_eval_conditions and GUI: loaded cfg first, then local yaml
        backprop_cfg = None
        if hasattr(cfg, "backprop") and getattr(cfg, "backprop", None) is not None:
            try:
                backprop_cfg = OmegaConf.to_container(cfg.backprop, resolve=True)
            except Exception:
                backprop_cfg = getattr(cfg, "backprop", None)
                if not isinstance(backprop_cfg, dict):
                    backprop_cfg = {
                        "epochs": getattr(backprop_cfg, "epochs", 200),
                        "learning_rate": getattr(backprop_cfg, "learning_rate", 1.0),
                        "optimizer": getattr(backprop_cfg, "optimizer", "adam"),
                        "weight_decay": getattr(backprop_cfg, "weight_decay", 0.0),
                        "beta1": getattr(backprop_cfg, "beta1", 0.9),
                        "beta2": getattr(backprop_cfg, "beta2", 0.999),
                    }
        if backprop_cfg is None:
            try:
                with open("configs/config.yaml", "r") as f:
                    local_cfg = yaml.safe_load(f)
                backprop_cfg = local_cfg.get("backprop", {})
            except Exception:
                backprop_cfg = {}
        preconfig_steps = int(backprop_cfg.get("epochs", 200))
        preconfig_lr = float(backprop_cfg.get("learning_rate", 1.0))
        preconfig_optimizer = backprop_cfg.get("optimizer", "adam")
        preconfig_weight_decay = float(backprop_cfg.get("weight_decay", 0.0))
        preconfig_beta1 = float(backprop_cfg.get("beta1", 0.9))
        preconfig_beta2 = float(backprop_cfg.get("beta2", 0.999))
        log.info(
            "Preconfig from backprop: steps=%s, lr=%s, optimizer=%s, weight_decay=%s, beta1=%s, beta2=%s",
            preconfig_steps, preconfig_lr, preconfig_optimizer,
            preconfig_weight_decay, preconfig_beta1, preconfig_beta2,
        )
        wiring_fixed_key = cfg.get("wiring_fixed_key", cfg.get("test_seed", 42))
        if isinstance(wiring_fixed_key, int):
            wiring_fixed_key = jax.random.PRNGKey(wiring_fixed_key)
        wires, logits = preconfigure_circuit_logits(
            wiring_key=wiring_fixed_key,
            layer_sizes=layer_sizes,
            arity=arity,
            x_data=x,
            y_data=y0,
            loss_type=loss_type,
            steps=preconfig_steps,
            lr=preconfig_lr,
            optimizer=preconfig_optimizer,
            weight_decay=preconfig_weight_decay,
            beta1=preconfig_beta1,
            beta2=preconfig_beta2,
        )
    else:
        key = jax.random.PRNGKey(cfg.get("test_seed", 42))
        wires, logits = gen_circuit(key, layer_sizes, arity=arity)

    # Optional: match damage_behavior if you add knockout later
    if hasattr(gnn_model, "damage_behavior"):
        pass  # leave as in checkpoint
    # layer_sizes as list of tuples for evaluation
    layer_sizes_tuples = [(int(a), int(b)) for a, b in layer_sizes]

    use_perturbations = args.knockout_size is not None and args.knockout_size > 0
    if use_perturbations:
        log.info(
            "Perturbation mode: knockout_size=%d, patterns=%d, damage_start_offset=%d",
            args.knockout_size, args.patterns_per_size, args.damage_start_offset,
        )
        setattr(gnn_model, "damage_behavior", "reversible")
        setattr(gnn_model, "reversible_bias", args.reversible_bias)
        rng = jax.random.PRNGKey(cfg.get("test_seed", 42))
        pattern_keys = jax.random.split(rng, args.patterns_per_size)
        vocab = jax.vmap(
            lambda k: create_reproducible_knockout_pattern(
                k, layer_sizes_tuples, args.knockout_size
            )
        )(pattern_keys)
        vocab_array = vocab if not isinstance(vocab, list) else jp.stack(vocab)
        batch_wires = jax.tree.map(
            lambda x: jp.repeat(x[None, ...], args.patterns_per_size, axis=0),
            wires,
        )
        batch_logits = jax.tree.map(
            lambda x: jp.repeat(x[None, ...], args.patterns_per_size, axis=0),
            logits,
        )
        log.info(
            "Running batched evaluation (%d steps x %d circuits). "
            "First run can take 10+ min due to JAX JIT compilation; subsequent runs are faster.",
            args.n_message_steps,
            args.patterns_per_size,
        )
        sys.stdout.flush()
        step_metrics = evaluate_model_stepwise_batched(
            model=gnn_model,
            batch_wires=batch_wires,
            batch_logits=batch_logits,
            x_data=x,
            y_data=y0,
            input_n=input_n,
            arity=arity,
            circuit_hidden_dim=gnn_hidden_dim,
            n_message_steps=args.n_message_steps,
            loss_type=loss_type,
            bidirectional_edges=True,
            layer_sizes=layer_sizes_tuples,
            layer_neighbors=cfg.model.get("layer_neighbors", False),
            knockout_patterns=vocab_array,
            return_per_pattern=False,
            damage_injection_mode="single",
            damage_start_offset=args.damage_start_offset,
            damage_mode="shotgun",
            greedy_window_size=args.knockout_size,
            blind_mode=False,
        )
        # Print averaged results in same format as non-perturbation path
        if not args.quiet:
            print(f"{'Step':<8} {'Hard Acc':<12} {'Soft Acc':<12} {'Hard Loss':<12} {'Soft Loss':<12}")
            print("-" * 56)
        readout_every = max(1, args.readout_every)
        steps = step_metrics["step"]
        for i, step in enumerate(steps):
            if step % readout_every != 0:
                continue
            ha = step_metrics["hard_accuracy"][i]
            sa = step_metrics["soft_accuracy"][i]
            hl = step_metrics["hard_loss"][i]
            sl = step_metrics["soft_loss"][i]
            if args.quiet:
                print(f"{step}\t{ha:.6f}")
            else:
                print(
                    f"{step:<8} {ha:<12.4f} {sa:<12.4f} {hl:<12.6f} {sl:<12.6f}"
                )
            sys.stdout.flush()
    else:
        # --- Live eval loop (same computation path as training) ---
        if not args.quiet:
            print(f"{'Step':<8} {'Hard Acc':<12} {'Soft Acc':<12} {'Hard Loss':<12} {'Soft Loss':<12}")
            print("-" * 56)

        gen = evaluate_model_stepwise_generator(
            model=gnn_model,
            wires=wires,
            logits=logits,
            x_data=x,
            y_data=y0,
            input_n=input_n,
            arity=arity,
            circuit_hidden_dim=gnn_hidden_dim,
            max_steps=args.n_message_steps,
            loss_type=loss_type,
            bidirectional_edges=True,
            layer_sizes=layer_sizes_tuples,
            layer_neighbors=cfg.model.get("layer_neighbors", False),
            knockout_pattern=None,
            reset_step_counter_on_init=False,
            blind_mode=False,
        )

        readout_every = max(1, args.readout_every)
        for result in gen:
            if result.step % readout_every != 0:
                continue
            if args.quiet:
                print(f"{result.step}\t{result.hard_accuracy:.6f}")
            else:
                print(
                    f"{result.step:<8} {result.hard_accuracy:<12.4f} {result.accuracy:<12.4f} "
                    f"{result.hard_loss:<12.6f} {result.loss:<12.6f}"
                )
            sys.stdout.flush()

    log.info("Done. Total steps: %d", args.n_message_steps)


if __name__ == "__main__":
    main()
