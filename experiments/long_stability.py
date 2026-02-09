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
    training_mode = "repair"
    if training_mode == "repair":
        log.info("Recreating preconfigured baseline circuit (repair mode)")
        preconfig_steps = cfg.get("preconfig_steps", 200)
        preconfig_lr = cfg.get("preconfig_lr", 1e-2)
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
        )
    else:
        key = jax.random.PRNGKey(cfg.get("test_seed", 42))
        wires, logits = gen_circuit(key, layer_sizes, arity=arity)

    # Optional: match damage_behavior if you add knockout later
    if hasattr(gnn_model, "damage_behavior"):
        pass  # leave as in checkpoint
    # layer_sizes as list of tuples for evaluation
    layer_sizes_tuples = [(int(a), int(b)) for a, b in layer_sizes]

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
