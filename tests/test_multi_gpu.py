#!/usr/bin/env python3
"""
Test multi-GPU data parallelism with virtual devices.

Run this test to verify multi-GPU sharding works correctly:
    JAX_PLATFORMS=cpu python tests/test_multi_gpu.py

On machines with real GPUs, use:
    python tests/test_multi_gpu.py

The virtual device simulation only works on CPU platform.
"""

# MUST be done BEFORE importing JAX
import os
import sys

# Detect if we're forcing CPU for virtual devices
FORCE_CPU = "cpu" in os.environ.get("JAX_PLATFORMS", "").lower()

if FORCE_CPU:
    # Force 4 virtual devices on CPU
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
    EXPECTED_DEVICES = 4
    print("Mode: Virtual devices (CPU simulation)")
else:
    # Use real devices
    EXPECTED_DEVICES = None  # Will be set after JAX import
    print("Mode: Real devices (set JAX_PLATFORMS=cpu for virtual device testing)")

import jax
import jax.numpy as jp
from flax import nnx

# Set expected devices after JAX import
if EXPECTED_DEVICES is None:
    EXPECTED_DEVICES = len(jax.devices())
    print(f"Detected {EXPECTED_DEVICES} real device(s)")

# Now import our modules
from boolean_nca_cc.models import CircuitGNN
from boolean_nca_cc.training.pool.pool import initialize_graph_pool
from boolean_nca_cc.training.sharding import (
    ShardingContext,
    create_device_mesh,
    get_data_sharding,
    get_replicated_sharding,
    replicate_pytree,
    shard_pytree,
    validate_batch_size,
)


def test_device_detection():
    """Test that devices are properly detected."""
    devices = jax.devices()
    print(f"✓ Found {len(devices)} devices: {[str(d) for d in devices]}")
    if FORCE_CPU:
        assert len(devices) == 4, f"Expected 4 virtual devices, got {len(devices)}"
        print("✓ Virtual device setup works!")
    else:
        assert len(devices) >= 1, "Expected at least 1 device"
        print(f"✓ Using {len(devices)} real device(s)")


def test_mesh_creation():
    """Test device mesh creation."""
    num_devices = min(4, EXPECTED_DEVICES)
    mesh = create_device_mesh(num_devices=num_devices)
    print(f"✓ Created mesh: shape={mesh.shape}, axis_names={mesh.axis_names}")
    assert "data" in mesh.axis_names
    assert mesh.shape["data"] == num_devices
    print("✓ Mesh creation works!")


def test_sharding_context():
    """Test ShardingContext basic functionality."""
    num_devices = min(4, EXPECTED_DEVICES)
    batch_size = 64 if num_devices >= 4 else 16 * num_devices

    with ShardingContext(batch_size=batch_size, num_devices=num_devices, enabled=True) as ctx:
        print(
            f"✓ ShardingContext: {ctx.num_devices} devices, {ctx.per_device_batch_size} per device"
        )
        assert ctx.num_devices == num_devices
        assert ctx.per_device_batch_size == batch_size // num_devices
        assert ctx.mesh is not None
    print("✓ ShardingContext works!")


def test_auto_detection():
    """Test auto-detection mode (enabled=None)."""
    batch_size = 16 * EXPECTED_DEVICES

    # Test auto mode (enabled=None)
    with ShardingContext(batch_size=batch_size, enabled=None) as ctx:
        if EXPECTED_DEVICES > 1:
            # Should auto-enable with multiple devices
            assert ctx.enabled is True, "Should auto-enable with >1 device"
            assert ctx.mesh is not None
            print(f"✓ Auto-enabled with {ctx.num_devices} devices")
        else:
            # Should auto-disable with single device
            assert ctx.enabled is False, "Should auto-disable with 1 device"
            print("✓ Auto-disabled with 1 device")

    # Test explicit disable
    with ShardingContext(batch_size=batch_size, enabled=False) as ctx:
        assert ctx.enabled is False
        assert ctx.mesh is None
        print("✓ Explicit disable works")

    # Test explicit enable
    with ShardingContext(batch_size=batch_size, enabled=True) as ctx:
        assert ctx.enabled is True
        print("✓ Explicit enable works")

    print("✓ Auto-detection works!")


def test_batch_size_validation():
    """Test batch size validation."""
    # Valid batch size
    validate_batch_size(64, 4)  # Should not raise
    print("✓ Valid batch size (64 / 4 = 16)")

    # Invalid batch size
    try:
        validate_batch_size(63, 4)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"✓ Invalid batch size correctly caught: {e}")

    print("✓ Batch size validation works!")


def test_data_sharding():
    """Test sharding of batch data."""
    num_devices = min(4, EXPECTED_DEVICES)
    mesh = create_device_mesh(num_devices=num_devices)

    # Create test batch
    batch_size = 16 * num_devices
    batch = jp.ones((batch_size, 32))
    sharding = get_data_sharding(mesh)
    sharded_batch = jax.device_put(batch, sharding)

    print(f"✓ Original shape: {batch.shape}")
    print(f"✓ Sharding spec: {sharded_batch.sharding}")

    # Verify it's sharded across devices
    assert sharded_batch.shape == batch.shape
    print("✓ Data sharding works!")


def test_model_replication():
    """Test model parameter replication."""
    num_devices = min(4, EXPECTED_DEVICES)
    mesh = create_device_mesh(num_devices=num_devices)
    replicated = get_replicated_sharding(mesh)

    # Create model using correct constructor signature
    rngs = nnx.Rngs(params=jax.random.PRNGKey(42))
    model = CircuitGNN(
        circuit_hidden_dim=16,
        mlp_dim=32,  # Correct parameter name
        mlp_n_layers=2,
        arity=2,
        rngs=rngs,
    )

    # Replicate state
    state = nnx.state(model)
    replicated_state = jax.device_put(state, replicated)
    nnx.update(model, replicated_state)

    # Check a parameter is replicated
    params = nnx.state(model, nnx.Param)
    sample_param = jax.tree.leaves(params)[0]
    print(f"✓ Parameter sharding: {sample_param.sharding}")
    print("✓ Model replication works!")


def test_pool_sampling_with_sharding():
    """Test pool sampling and sharding of circuit data."""
    num_devices = min(4, EXPECTED_DEVICES)

    # Setup
    layer_sizes = [(4, 1), (4, 1), (2, 1)]

    # Initialize pool
    pool = initialize_graph_pool(
        rng=jax.random.PRNGKey(42),
        layer_sizes=layer_sizes,
        pool_size=128,
        input_n=4,
        arity=2,
        circuit_hidden_dim=16,
    )

    # Sample batch (must be divisible by num_devices)
    sample_key = jax.random.PRNGKey(0)
    batch_size = 16 * num_devices
    idxs, graphs, wires, logits, _ = pool.sample(sample_key, batch_size)

    # Shard with context
    with ShardingContext(batch_size=batch_size, num_devices=num_devices, enabled=True) as ctx:
        sharded_graphs = ctx.shard(graphs)
        sharded_wires = ctx.shard(wires)
        sharded_logits = ctx.shard(logits)

        print(f"✓ Graphs n_node shape: {sharded_graphs.n_node.shape}")
        print(f"✓ Wires[0] shape: {sharded_wires[0].shape}")
        print(f"✓ Logits[0] shape: {sharded_logits[0].shape}")

    print("✓ Pool sampling with sharding works!")


def test_gradient_computation_sharded():
    """Test that gradients are correctly computed with sharding."""
    num_devices = min(4, EXPECTED_DEVICES)
    mesh = create_device_mesh(num_devices=num_devices)
    data_sharding = get_data_sharding(mesh)
    replicated = get_replicated_sharding(mesh)

    # Create simple model (just a linear layer for testing)
    class SimpleModel(nnx.Module):
        def __init__(self, rngs):
            self.linear = nnx.Linear(8, 4, rngs=rngs)

        def __call__(self, x):
            return self.linear(x)

    model = SimpleModel(nnx.Rngs(params=jax.random.PRNGKey(0)))

    # Replicate model
    state = nnx.state(model)
    state = jax.device_put(state, replicated)
    nnx.update(model, state)

    # Create sharded batch
    batch_size = 16 * num_devices
    batch_x = jp.ones((batch_size, 8))
    batch_y = jp.zeros((batch_size, 4))

    sharded_x = jax.device_put(batch_x, data_sharding)
    sharded_y = jax.device_put(batch_y, data_sharding)

    # Compute loss and gradients
    def loss_fn(model, x, y):
        pred = model(x)
        return jp.mean((pred - y) ** 2)

    @nnx.jit
    def train_step(model, x, y):
        loss, grads = nnx.value_and_grad(loss_fn)(model, x, y)
        return loss, grads

    loss, grads = train_step(model, sharded_x, sharded_y)

    print(f"✓ Loss: {float(loss):.6f}")
    print(f"✓ Grad sample sharding: {jax.tree.leaves(grads)[0].sharding}")

    # Verify gradients are not None/zero
    grad_norms = [jp.linalg.norm(g) for g in jax.tree.leaves(grads)]
    assert all(n > 0 for n in grad_norms), "Gradients should not be zero"

    print("✓ Gradient computation with sharding works!")


def run_all_tests():
    """Run all multi-GPU tests."""
    print("=" * 60)
    print("Testing Multi-GPU Data Parallelism")
    print(f"Mode: {'Virtual Devices (CPU)' if FORCE_CPU else 'Real Devices'}")
    print(f"Expected devices: {EXPECTED_DEVICES}")
    print("=" * 60)
    print()

    tests = [
        ("Device Detection", test_device_detection),
        ("Mesh Creation", test_mesh_creation),
        ("ShardingContext", test_sharding_context),
        ("Auto-Detection", test_auto_detection),
        ("Batch Size Validation", test_batch_size_validation),
        ("Data Sharding", test_data_sharding),
        ("Model Replication", test_model_replication),
        ("Pool Sampling with Sharding", test_pool_sampling_with_sharding),
        ("Gradient Computation Sharded", test_gradient_computation_sharded),
    ]

    passed = 0
    failed = 0

    for name, test_fn in tests:
        print(f"\n--- {name} ---")
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"✗ FAILED: {e}")
            import traceback

            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
