#!/usr/bin/env python3
"""
Test script to verify that sequential batch processing produces the same results
as regular batch processing (within numerical precision).
"""

import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import hydra
import jax
from flax import nnx

from boolean_nca_cc import generate_layer_sizes
from boolean_nca_cc.circuits.model import gen_circuit
from boolean_nca_cc.circuits.tasks import get_task_data
from boolean_nca_cc.training.train_loop import train_model
from boolean_nca_cc.utils.graph_builder import build_graph


def test_sequential_batching():
    """Test that sequential batching produces the same results as regular batching."""

    # Set up test parameters
    seed = 42
    rng = jax.random.PRNGKey(seed)

    # Small circuit for testing
    input_bits, output_bits = 4, 4
    arity = 2
    layer_sizes = generate_layer_sizes(input_bits, output_bits, arity, layer_n=2)

    # Generate test data
    case_n = 1 << input_bits
    x_data, y_data = get_task_data(
        "reverse", case_n, input_bits=input_bits, output_bits=output_bits
    )

    # Generate dummy circuit for graph building
    test_key = jax.random.PRNGKey(42)
    wires, logits = gen_circuit(test_key, layer_sizes, arity=arity)

    # Generate dummy graph to get n_nodes
    circuit_hidden_dim = 16
    graph = build_graph(
        wires=wires,
        logits=logits,
        input_n=input_bits,
        arity=arity,
        circuit_hidden_dim=circuit_hidden_dim,
    )
    n_nodes = int(graph.n_node[0])

    # Create model configs similar to train.py
    model_config = {
        "_target_": "boolean_nca_cc.models.CircuitSelfAttention",
        "circuit_hidden_dim": circuit_hidden_dim,
        "attention_dim": 32,
        "mlp_dim_multiplier": 2,
        "n_node": n_nodes,
        "arity": arity,
    }

    # Test parameters
    meta_batch_size = 32
    chunk_size = 8  # Will create 4 chunks
    epochs = 5

    print("Testing sequential batching:")
    print(f"Meta batch size: {meta_batch_size}")
    print(f"Chunk size: {chunk_size}")
    print(f"Expected chunks: {(meta_batch_size + chunk_size - 1) // chunk_size}")
    print(f"Circuit nodes: {n_nodes}")
    print()

    # Create models using hydra instantiation
    rng, init_key1, init_key2 = jax.random.split(rng, 3)

    # Model 1 for regular batching
    model1 = hydra.utils.instantiate(model_config, rngs=nnx.Rngs(params=init_key1))

    # Model 2 for sequential batching
    model2 = hydra.utils.instantiate(model_config, rngs=nnx.Rngs(params=init_key2))

    # Set the same parameters for both models
    params1 = nnx.state(model1, nnx.Param)
    nnx.update(model2, params1)

    # Train with regular batching
    print("Training with regular batching...")
    results_regular = train_model(
        x_data=x_data,
        y_data=y_data,
        layer_sizes=layer_sizes,
        arity=arity,
        circuit_hidden_dim=circuit_hidden_dim,
        learning_rate=1e-3,
        weight_decay=1e-4,
        epochs=epochs,
        n_message_steps=3,
        use_scan=False,
        loss_type="l4",
        wiring_mode="random",
        meta_batch_size=meta_batch_size,
        batch_chunk_size=None,  # Regular batching
        pool_size=128,
        reset_pool_fraction=0.1,
        reset_pool_interval=10,
        key=seed,
        init_model=model1,
        wandb_logging=False,
    )

    # Train with sequential batching
    print("\nTraining with sequential batching...")
    results_sequential = train_model(
        x_data=x_data,
        y_data=y_data,
        layer_sizes=layer_sizes,
        arity=arity,
        circuit_hidden_dim=circuit_hidden_dim,
        learning_rate=1e-3,
        weight_decay=1e-4,
        epochs=epochs,
        n_message_steps=3,
        use_scan=False,
        loss_type="l4",
        wiring_mode="random",
        meta_batch_size=meta_batch_size,
        batch_chunk_size=chunk_size,  # Sequential batching
        pool_size=128,
        reset_pool_fraction=0.1,
        reset_pool_interval=10,
        key=seed,
        init_model=model2,
        wandb_logging=False,
    )

    # Compare results
    print("\nComparing results:")
    print("Epoch | Regular Loss | Sequential Loss | Difference")
    print("-" * 55)

    max_diff = 0.0
    for i in range(min(len(results_regular["losses"]), len(results_sequential["losses"]))):
        regular_loss = results_regular["losses"][i]
        sequential_loss = results_sequential["losses"][i]
        diff = abs(regular_loss - sequential_loss)
        max_diff = max(max_diff, diff)

        print(f"{i:5d} | {regular_loss:11.6f} | {sequential_loss:13.6f} | {diff:10.6f}")

    print(f"\nMaximum difference: {max_diff:.8f}")

    # Check if differences are within acceptable tolerance
    tolerance = 1e-4  # Allow for some numerical differences
    if max_diff < tolerance:
        print(
            f"✓ PASS: Sequential batching produces similar results (max diff: {max_diff:.8f} < {tolerance})"
        )
        return True
    else:
        print(
            f"✗ FAIL: Sequential batching produces different results (max diff: {max_diff:.8f} >= {tolerance})"
        )
        return False


if __name__ == "__main__":
    success = test_sequential_batching()
    exit(0 if success else 1)
