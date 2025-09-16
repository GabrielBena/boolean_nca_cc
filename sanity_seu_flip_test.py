import os
import sys

import jax
import jax.numpy as jp

from omegaconf import OmegaConf

from boolean_nca_cc.circuits.model import generate_layer_sizes
from boolean_nca_cc.circuits.tasks import get_task_data
from boolean_nca_cc.training.preconfigure import preconfigure_circuit_logits
from boolean_nca_cc.training.evaluation import get_loss_from_wires_logits
from boolean_nca_cc.training.pool.perturbation import (
    sample_seu_gates,
    build_flip_masks_from_indices,
    flip_logits_with_masks,
)


def main():
    # Load config
    cfg_path = os.path.join("configs", "config.yaml")
    cfg = OmegaConf.load(cfg_path)

    # Config-derived parameters
    input_bits = int(cfg.circuit.input_bits)
    output_bits = int(cfg.circuit.output_bits)
    arity = int(cfg.circuit.arity)
    num_layers = int(cfg.circuit.num_layers)
    loss_type = str(cfg.training.loss_type)
    task_name = str(cfg.circuit.task)

    # Greedy flip settings
    num_gates_to_flip = int(cfg.pool.damage_prob) if hasattr(cfg.pool, "damage_prob") else 20
    num_gates_to_flip = 20  # Force 20 per request
    flips_per_gate = int(cfg.eval.seu.flips_per_gate)
    greedy_ordered_indices = list(cfg.pool.greedy_ordered_indices)

    # Seeds
    wiring_seed = int(cfg.test_seed)
    flip_seed = int(cfg.damage_seed)

    # Build circuit layer sizes identical to training
    layer_sizes = generate_layer_sizes(input_bits, output_bits, arity, layer_n=num_layers)

    # Build full truth table for the task
    case_n = 1 << input_bits
    x_data, y_data = get_task_data(task_name, case_n, input_bits=input_bits, output_bits=output_bits)

    # Preconfigure logits on fixed wiring to obtain a functioning circuit
    wiring_key = jax.random.PRNGKey(wiring_seed)
    base_wires, base_logits = preconfigure_circuit_logits(
        wiring_key=wiring_key,
        layer_sizes=layer_sizes,
        arity=arity,
        x_data=x_data,
        y_data=y_data,
        loss_type=loss_type,
        steps=200,
        lr=1.0,
        optimizer="adamw",
        weight_decay=1e-1,
        beta1=0.8,
        beta2=0.8,
    )

    # Baseline hard accuracy
    _, aux = get_loss_from_wires_logits(base_logits, base_wires, x_data, y_data, loss_type)
    baseline_hard_acc = float(aux[4])

    # Build greedy SEU gate selection (across all gate layers, excluding inputs)
    gate_layer_sizes = layer_sizes[1:]  # exclude inputs; includes all gate layers (incl. output)
    flip_key = jax.random.PRNGKey(flip_seed)
    selected_gates = sample_seu_gates(
        key=flip_key,
        layer_sizes=gate_layer_sizes,
        num_gates=num_gates_to_flip,
        strategy="greedy",
        ordered_indices=greedy_ordered_indices,
    )

    # Build flip masks aligned with per-layer logits shapes
    masks = build_flip_masks_from_indices(
        layer_sizes=gate_layer_sizes,
        selected_gate_indices=selected_gates,
        flips_per_gate=flips_per_gate,
        arity=arity,
        key=flip_key,
    )

    # Apply flips to logits (invert selected LUT entries)
    mutated_logits = flip_logits_with_masks(base_logits, masks, mode="invert")

    # Post-flip hard accuracy
    _, aux_post = get_loss_from_wires_logits(mutated_logits, base_wires, x_data, y_data, loss_type)
    postflip_hard_acc = float(aux_post[4])

    # Report
    print("=== SEU Sanity Check ===")
    print(f"Task: {task_name}")
    print(f"Inputs={input_bits}, Outputs={output_bits}, Arity={arity}, Layers={num_layers}")
    print(f"Greedy gates flipped: {num_gates_to_flip}, Flips per gate (LUT entries)={flips_per_gate}")
    print(f"Baseline hard accuracy: {baseline_hard_acc:.6f}")
    print(f"Post-flip hard accuracy: {postflip_hard_acc:.6f}")
    print(f"Delta: {postflip_hard_acc - baseline_hard_acc:.6f}")
    print("Effect detected:" , "YES" if abs(postflip_hard_acc - baseline_hard_acc) > 1e-6 else "NO")


if __name__ == "__main__":
    sys.exit(main())


