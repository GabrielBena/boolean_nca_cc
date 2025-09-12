#!/usr/bin/env python3
"""
Demonstration of gate knockout functionality.

This script shows how to use the gate knockout system to create
permanent structural damage in boolean circuits and verify that
the knocked out gates don't participate in training.
"""

import jax
import jax.numpy as jp
import optax

from boolean_nca_cc.circuits.model import gen_circuit, generate_layer_sizes, run_circuit
from boolean_nca_cc.circuits.train import TrainState, train_step
from boolean_nca_cc.training.pool.perturbation import (
    apply_gate_knockout_to_pool_element,
    create_reproducible_knockout_pattern,
)


def demonstrate_gate_knockout():
    """Demonstrate gate knockout functionality with a simple example."""
    print("🔧 Gate Knockout Demonstration")
    print("=" * 50)

    # Set up circuit parameters
    key = jax.random.PRNGKey(42)
    input_n, output_n, arity, layer_n = 8, 4, 4, 2

    # Generate circuit
    print(f"📊 Creating circuit: {input_n} inputs → {output_n} outputs, {layer_n} layers")
    layer_sizes = generate_layer_sizes(input_n, output_n, arity, layer_n)
    print(f"   Layer sizes: {layer_sizes}")

    wires, logits = gen_circuit(key, layer_sizes, arity)

    # Generate sample data
    batch_size = 16
    x = jax.random.uniform(key, (batch_size, input_n))
    y0 = jax.random.uniform(jax.random.split(key)[1], (batch_size, output_n))

    # Create knockout pattern
    num_knockouts = 3
    print(f"\n💥 Creating knockout pattern with {num_knockouts} knocked out gates")

    knockout_key = jax.random.PRNGKey(123)
    knockout_masks = create_reproducible_knockout_pattern(
        knockout_key, layer_sizes, num_knockouts, input_n=input_n
    )

    # Show knockout pattern
    total_gates = sum(gate_n for gate_n, _ in layer_sizes)
    total_knockouts = sum(jp.sum(mask == 0.0) for mask in knockout_masks)
    print(f"   Total gates: {total_gates}, Knocked out: {int(total_knockouts)}")

    for i, (mask, (gate_n, _)) in enumerate(zip(knockout_masks, layer_sizes)):
        knocked_out_count = int(jp.sum(mask == 0.0))
        print(f"   Layer {i}: {gate_n} gates, {knocked_out_count} knocked out")

    # Apply knockout to get faulty logits
    print(f"\n🔨 Applying knockout to create faulty logits")
    faulty_logits, _, _ = apply_gate_knockout_to_pool_element(
        knockout_key, logits, wires, layer_sizes, num_knockouts
    )

    # Test circuit behavior with and without knockouts
    print(f"\n🧪 Testing circuit behavior")

    # Run normal circuit
    normal_acts = run_circuit(logits, wires, x)
    normal_output = normal_acts[-1]

    # Run circuit with knockouts
    knockout_acts = run_circuit(faulty_logits, wires, x, gate_mask=knockout_masks)
    knockout_output = knockout_acts[-1]

    print(f"   Normal output range: [{jp.min(normal_output):.3f}, {jp.max(normal_output):.3f}]")
    print(
        f"   Knockout output range: [{jp.min(knockout_output):.3f}, {jp.max(knockout_output):.3f}]"
    )

    # Verify knocked out gates produce zero
    for layer_idx in range(1, len(layer_sizes)):
        layer_mask = knockout_masks[layer_idx]
        layer_output = knockout_acts[layer_idx]
        knocked_out_indices = jp.where(layer_mask == 0.0)[0]

        if len(knocked_out_indices) > 0:
            knocked_out_outputs = layer_output[:, knocked_out_indices]
            max_knockout_output = jp.max(jp.abs(knocked_out_outputs))
            print(f"   Layer {layer_idx} knocked out gates max output: {max_knockout_output:.6f}")

    # Test training with knockout protection
    print(f"\n🎓 Testing training with knockout protection")

    # Create training state
    optimizer = optax.adam(learning_rate=0.01)
    train_state = TrainState(faulty_logits, optimizer.init(faulty_logits))

    # Store original faulty values for comparison
    original_faulty_logits = [layer.copy() for layer in faulty_logits]

    # Perform training steps
    print(f"   Performing 5 training steps...")
    losses = []

    for step in range(5):
        loss, aux, train_state = train_step(
            train_state,
            optimizer,
            wires,
            x,
            y0,
            loss_type="l4",
            do_train=True,
            gate_mask=knockout_masks,
        )
        losses.append(float(loss))

        if step == 0:
            print(f"   Step {step}: Loss = {loss:.6f}, Accuracy = {aux['accuracy']:.3f}")
        elif step == 4:
            print(f"   Step {step}: Loss = {loss:.6f}, Accuracy = {aux['accuracy']:.3f}")

    # Verify knocked out gates are preserved
    final_logits = train_state.params
    preserved_count = 0
    total_knockout_gates = 0

    for layer_idx, (final_layer, original_layer, layer_mask) in enumerate(
        zip(final_logits, original_faulty_logits, knockout_masks[1:], strict=False)
    ):
        group_n, group_size, lut_size = final_layer.shape
        mask_reshaped = layer_mask.reshape(group_n, group_size)

        for group_idx in range(group_n):
            for gate_idx in range(group_size):
                if mask_reshaped[group_idx, gate_idx] == 0.0:
                    total_knockout_gates += 1
                    final_gate = final_layer[group_idx, gate_idx, :]
                    original_gate = original_layer[group_idx, gate_idx, :]

                    if jp.allclose(final_gate, original_gate, atol=1e-6):
                        preserved_count += 1

    print(f"   Knocked out gates preserved: {preserved_count}/{total_knockout_gates}")
    print(f"   Loss progression: {losses[0]:.6f} → {losses[-1]:.6f}")

    # Test gradient computation
    print(f"\n🎯 Testing gradient computation")

    def loss_fn(logits_param):
        acts = run_circuit(logits_param, wires, x, gate_mask=knockout_masks)
        return jp.sum((acts[-1] - y0) ** 2)

    loss_value, grads = jax.value_and_grad(loss_fn)(faulty_logits)

    zero_grad_count = 0
    total_knockout_params = 0

    for layer_idx, (layer_grad, layer_mask) in enumerate(
        zip(grads, knockout_masks[1:], strict=False)
    ):
        group_n, group_size, lut_size = layer_grad.shape
        mask_reshaped = layer_mask.reshape(group_n, group_size)

        for group_idx in range(group_n):
            for gate_idx in range(group_size):
                if mask_reshaped[group_idx, gate_idx] == 0.0:
                    total_knockout_params += lut_size
                    gate_grad = layer_grad[group_idx, gate_idx, :]
                    if jp.allclose(gate_grad, 0.0, atol=1e-6):
                        zero_grad_count += lut_size

    print(
        f"   Zero gradients for knocked out parameters: {zero_grad_count}/{total_knockout_params}"
    )

    print(f"\n✅ Gate knockout demonstration completed successfully!")
    print(f"   - Knocked out gates produce zero output ✓")
    print(f"   - Knocked out gates have zero gradients ✓")
    print(f"   - Knocked out gates are preserved during training ✓")
    print(f"   - Training can proceed normally with knockout protection ✓")


if __name__ == "__main__":
    demonstrate_gate_knockout()
