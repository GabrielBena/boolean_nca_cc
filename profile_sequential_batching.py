#!/usr/bin/env python3
"""
Profiling script to compare execution times for sequential batch processing
with different chunk sizes. We expect similar execution times with minimal overhead.
"""

import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import time

import hydra
import jax
import jax.numpy as jp
from flax import nnx
from omegaconf import DictConfig, OmegaConf

from boolean_nca_cc import generate_layer_sizes
from boolean_nca_cc.circuits.model import gen_circuit
from boolean_nca_cc.circuits.tasks import get_task_data
from boolean_nca_cc.training.train_loop import train_model
from boolean_nca_cc.utils.graph_builder import build_graph


def profile_batch_processing():
    """Profile different batch processing configurations."""

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

    # Create model config
    model_config = {
        "_target_": "boolean_nca_cc.models.CircuitSelfAttention",
        "circuit_hidden_dim": circuit_hidden_dim,
        "attention_dim": 32,
        "mlp_dim_multiplier": 2,
        "n_node": n_nodes,
        "arity": arity,
    }

    # Test configurations: (meta_batch_size, chunk_size, description)
    test_configs = [
        (64, None, "Full batch (64)"),
        (64, 32, "2 chunks (64 -> 32x2)"),
        (64, 16, "4 chunks (64 -> 16x4)"),
        (32, None, "Full batch (32)"),
        (32, 16, "2 chunks (32 -> 16x2)"),
        (32, 8, "4 chunks (32 -> 8x4)"),
    ]

    epochs = 3  # Small number for quick profiling

    print(f"Profiling batch processing with {epochs} epochs each")
    print(f"Circuit nodes: {n_nodes}")
    print("=" * 70)

    results = []

    for meta_batch_size, chunk_size, description in test_configs:
        print(f"\nTesting: {description}")

        # Create fresh model for each test
        rng, init_key = jax.random.split(rng)
        model = hydra.utils.instantiate(model_config, rngs=nnx.Rngs(params=init_key))

        # Measure training time
        start_time = time.time()

        train_results = train_model(
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
            batch_chunk_size=chunk_size,
            pool_size=128,
            reset_pool_fraction=0.1,
            reset_pool_interval=10,
            key=seed,
            init_model=model,
            wandb_logging=False,
        )

        end_time = time.time()
        total_time = end_time - start_time
        time_per_epoch = total_time / epochs

        # Calculate expected chunks
        if chunk_size is None:
            num_chunks = 1
            effective_chunk_size = meta_batch_size
        else:
            num_chunks = (meta_batch_size + chunk_size - 1) // chunk_size
            effective_chunk_size = chunk_size

        results.append(
            {
                "description": description,
                "meta_batch_size": meta_batch_size,
                "chunk_size": chunk_size,
                "num_chunks": num_chunks,
                "total_time": total_time,
                "time_per_epoch": time_per_epoch,
                "final_loss": train_results["losses"][-1],
            }
        )

        print(f"  Time: {total_time:.2f}s total, {time_per_epoch:.2f}s/epoch")
        print(f"  Chunks: {num_chunks} x {effective_chunk_size}")
        print(f"  Final loss: {train_results['losses'][-1]:.4f}")

    # Analysis
    print("\n" + "=" * 70)
    print("PROFILING RESULTS:")
    print("=" * 70)

    print(
        f"{'Description':<25} {'Batch':<8} {'Chunks':<8} {'Time/Epoch':<12} {'Overhead':<10} {'Loss':<8}"
    )
    print("-" * 70)

    # Find baseline (full batch) times for comparison
    baseline_32 = next(
        r for r in results if r["meta_batch_size"] == 32 and r["chunk_size"] is None
    )
    baseline_64 = next(
        r for r in results if r["meta_batch_size"] == 64 and r["chunk_size"] is None
    )

    for result in results:
        meta_batch = result["meta_batch_size"]
        chunks = result["num_chunks"]
        time_per_epoch = result["time_per_epoch"]
        description = result["description"]
        loss = result["final_loss"]

        # Calculate overhead compared to full batch of same size
        if meta_batch == 32:
            baseline_time = baseline_32["time_per_epoch"]
        else:
            baseline_time = baseline_64["time_per_epoch"]

        if result["chunk_size"] is None:
            overhead = "baseline"
        else:
            overhead_factor = time_per_epoch / baseline_time
            overhead = f"{overhead_factor:.2f}x"

        print(
            f"{description:<25} {meta_batch:<8} {chunks:<8} {time_per_epoch:<12.3f} {overhead:<10} {loss:<8.4f}"
        )

    # Expected vs actual analysis
    print("\n" + "=" * 70)
    print("OVERHEAD ANALYSIS:")
    print("=" * 70)

    print("Expected: Sequential batching should have minimal overhead (~1.0-1.2x)")
    print("Chunking should NOT scale linearly with number of chunks")
    print()

    for result in results:
        if result["chunk_size"] is not None:  # Only chunked results
            meta_batch = result["meta_batch_size"]
            baseline = baseline_32 if meta_batch == 32 else baseline_64
            overhead_factor = result["time_per_epoch"] / baseline["time_per_epoch"]
            num_chunks = result["num_chunks"]

            print(
                f"{result['description']}: {overhead_factor:.2f}x overhead ({num_chunks} chunks)"
            )

            if overhead_factor > 2.0:
                print(
                    f"  ⚠️  WARNING: High overhead detected! Expected ~1.2x, got {overhead_factor:.2f}x"
                )
            elif overhead_factor > 1.5:
                print(
                    f"  ⚠️  CAUTION: Moderate overhead. Expected ~1.2x, got {overhead_factor:.2f}x"
                )
            else:
                print(f"  ✓ OK: Reasonable overhead")

    return results


if __name__ == "__main__":
    results = profile_batch_processing()
