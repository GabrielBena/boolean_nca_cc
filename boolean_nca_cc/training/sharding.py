"""
Multi-GPU data parallelism utilities using JAX explicit sharding.

This module provides infrastructure for distributing batch computation across
multiple GPUs while keeping model parameters replicated. Gradients are
automatically aggregated via XLA's all-reduce when using proper sharding.

Key concepts:
- Mesh: Logical grid of devices with named axes
- NamedSharding: Specifies how tensors are partitioned across mesh axes
- Data parallelism: Batch dimension sharded, model replicated

Usage:
    # Setup (once at training start)
    mesh = create_device_mesh()

    # Shard batch data before training step
    graphs = shard_pytree(graphs, mesh, 'data')

    # Model params stay replicated (default behavior)

For testing on single machine without multiple GPUs:
    # Set BEFORE importing JAX
    setup_virtual_devices(num_devices=4)

    # Then import and use normally
    import jax
    mesh = create_device_mesh()  # Will see 4 virtual devices
"""

import logging
import os
from typing import Any

log = logging.getLogger(__name__)

# Type alias
PyTree = Any


def setup_virtual_devices(num_devices: int = 4) -> None:
    """
    Configure JAX to simulate multiple devices on a single machine.

    MUST be called BEFORE importing JAX or any JAX-dependent modules.
    Useful for testing multi-GPU code without actual hardware.

    Args:
        num_devices: Number of virtual devices to create

    Example:
        # At the very top of your script, before any imports:
        from boolean_nca_cc.training.sharding import setup_virtual_devices
        setup_virtual_devices(4)

        # Now import everything else
        import jax
        from boolean_nca_cc.training.train_loop import train_model
    """
    os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={num_devices}"
    log.info(f"Configured JAX to use {num_devices} virtual devices")


def get_num_devices() -> int:
    """Get the number of available JAX devices."""
    import jax

    return len(jax.devices())


def create_device_mesh(num_devices: int | None = None, axis_name: str = "data"):
    """
    Create a 1D device mesh for data parallelism.

    Args:
        num_devices: Number of devices to use. None means use all available.
        axis_name: Name for the mesh axis (default: 'data')

    Returns:
        jax.sharding.Mesh: Device mesh with named axis
    """
    import jax
    from jax.experimental import mesh_utils
    from jax.sharding import Mesh

    devices = jax.devices()
    if num_devices is not None:
        devices = devices[:num_devices]

    n_devices = len(devices)

    # Create 1D mesh: all devices along the data axis
    device_mesh = mesh_utils.create_device_mesh((n_devices,), devices=devices)
    mesh = Mesh(device_mesh, (axis_name,))

    log.info(f"Created device mesh with {n_devices} devices on axis '{axis_name}'")
    return mesh


def get_data_sharding(mesh, axis_name: str = "data"):
    """
    Get sharding spec that partitions first dimension across devices.

    Use this for batch data (graphs, wires, logits) - each device gets
    a slice of the batch.

    Args:
        mesh: Device mesh from create_device_mesh()
        axis_name: Name of the axis to shard along

    Returns:
        NamedSharding for data parallelism
    """
    from jax.sharding import NamedSharding
    from jax.sharding import PartitionSpec as P

    return NamedSharding(mesh, P(axis_name))


def get_replicated_sharding(mesh):
    """
    Get sharding spec that replicates data across all devices.

    Use this for model parameters and optimizer state - every device
    gets a full copy.

    Args:
        mesh: Device mesh from create_device_mesh()

    Returns:
        NamedSharding for full replication
    """
    from jax.sharding import NamedSharding
    from jax.sharding import PartitionSpec as P

    return NamedSharding(mesh, P())


def shard_pytree(pytree: PyTree, mesh, axis_name: str = "data") -> PyTree:
    """
    Shard a pytree across devices along the batch dimension.

    Each leaf array's first dimension is split across the mesh axis.

    Args:
        pytree: JAX pytree (arrays, GraphsTuple, list of arrays, etc.)
        mesh: Device mesh from create_device_mesh()
        axis_name: Name of the mesh axis to shard along

    Returns:
        Sharded pytree with same structure
    """
    import jax

    sharding = get_data_sharding(mesh, axis_name)
    return jax.device_put(pytree, sharding)


def replicate_pytree(pytree: PyTree, mesh) -> PyTree:
    """
    Replicate a pytree across all devices.

    Every device gets a full copy of each leaf array.

    Args:
        pytree: JAX pytree to replicate
        mesh: Device mesh from create_device_mesh()

    Returns:
        Replicated pytree with same structure
    """
    import jax

    sharding = get_replicated_sharding(mesh)
    return jax.device_put(pytree, sharding)


def replicate_model_and_optimizer(model, optimizer, mesh):
    """
    Replicate model parameters and optimizer state across all devices.

    This should be called once after model/optimizer creation to ensure
    all devices have identical copies before training begins.

    Args:
        model: Flax NNX model
        optimizer: Flax NNX optimizer
        mesh: Device mesh

    Returns:
        Tuple of (model, optimizer) with replicated state
    """
    import jax
    from flax import nnx

    replicated = get_replicated_sharding(mesh)

    # Get combined state of model + optimizer
    state = nnx.state((model, optimizer))

    # Replicate all state across devices
    sharded_state = jax.device_put(state, replicated)

    # Update model and optimizer with replicated state
    nnx.update((model, optimizer), sharded_state)

    log.debug("Replicated model and optimizer state across devices")
    return model, optimizer


def gather_from_devices(pytree: PyTree) -> PyTree:
    """
    Gather a sharded pytree back to a single device (device 0).

    Useful for collecting updated pool data after training step.

    Args:
        pytree: Sharded pytree to gather

    Returns:
        Pytree on single device with full data
    """
    import jax

    # Move to single device (first device)
    single_device = jax.devices()[0]
    return jax.device_put(pytree, single_device)


def validate_batch_size(batch_size: int, num_devices: int) -> None:
    """
    Validate that batch size is divisible by number of devices.

    Args:
        batch_size: Total batch size
        num_devices: Number of devices

    Raises:
        ValueError: If batch size not divisible by device count
    """
    if batch_size % num_devices != 0:
        raise ValueError(
            f"Batch size ({batch_size}) must be divisible by number of devices ({num_devices}). "
            f"Consider using batch_size={batch_size - (batch_size % num_devices)} or "
            f"batch_size={batch_size + (num_devices - batch_size % num_devices)}"
        )


def log_sharding_info(mesh, batch_size: int) -> None:
    """Log information about the sharding configuration."""
    num_devices = mesh.shape["data"] if "data" in mesh.axis_names else 1
    per_device = batch_size // num_devices
    log.info(
        f"Multi-GPU training: {num_devices} devices, batch_size={batch_size} ({per_device}/device)"
    )


class ShardingContext:
    """
    Context manager for multi-GPU training.

    Handles mesh creation, validation, and provides sharding utilities.

    Example:
        with ShardingContext(batch_size=64, num_devices=4) as ctx:
            # Shard data
            graphs = ctx.shard(graphs)
            wires = ctx.shard(wires)

            # Replicate model
            model, optimizer = ctx.replicate(model, optimizer)

            # Train...
    """

    def __init__(
        self,
        batch_size: int,
        num_devices: int | None = None,
        axis_name: str = "data",
        enabled: bool | None = None,
    ):
        """
        Initialize sharding context.

        Args:
            batch_size: Total batch size across all devices
            num_devices: Number of devices to use (None = all available)
            axis_name: Name for mesh axis
            enabled: Multi-GPU mode:
                - None = auto (enable if >1 device available)
                - True = force enable
                - False = force disable
        """
        self.batch_size = batch_size
        self.axis_name = axis_name
        self._enabled_setting = enabled  # Store original setting
        self._mesh = None
        self._num_devices = num_devices
        self._auto_enabled = False  # Track if auto-enabled

    def __enter__(self):
        import jax

        # Determine device count
        available = len(jax.devices())

        # Handle auto-detection (enabled=None)
        if self._enabled_setting is None:
            # Auto mode: enable if more than 1 device available
            self._auto_enabled = available > 1
            if self._auto_enabled:
                log.info(f"Multi-GPU auto-enabled: {available} devices detected")
            else:
                log.info("Multi-GPU auto-disabled: only 1 device available")
        else:
            self._auto_enabled = self._enabled_setting

        if not self._auto_enabled:
            return self

        self._num_devices = self._num_devices or available

        if self._num_devices > available:
            log.warning(
                f"Requested {self._num_devices} devices but only {available} available. "
                f"Using {available}."
            )
            self._num_devices = available

        # Validate batch size
        validate_batch_size(self.batch_size, self._num_devices)

        # Create mesh
        self._mesh = create_device_mesh(self._num_devices, self.axis_name)
        log_sharding_info(self._mesh, self.batch_size)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._mesh = None
        return False

    @property
    def mesh(self):
        """Get the device mesh (None if disabled)."""
        return self._mesh

    @property
    def enabled(self) -> bool:
        """Whether multi-GPU is currently enabled (after auto-detection)."""
        return self._auto_enabled

    @property
    def num_devices(self) -> int:
        """Get number of devices being used."""
        if not self._auto_enabled or self._num_devices is None:
            return 1
        return self._num_devices

    @property
    def per_device_batch_size(self) -> int:
        """Get batch size per device."""
        return self.batch_size // self.num_devices

    def shard(self, pytree: PyTree) -> PyTree:
        """Shard pytree across devices (no-op if disabled)."""
        if not self._auto_enabled or self._mesh is None:
            return pytree
        return shard_pytree(pytree, self._mesh, self.axis_name)

    def replicate(self, model, optimizer):
        """Replicate model and optimizer (no-op if disabled)."""
        if not self._auto_enabled or self._mesh is None:
            return model, optimizer
        return replicate_model_and_optimizer(model, optimizer, self._mesh)

    def gather(self, pytree: PyTree) -> PyTree:
        """Gather sharded pytree back to single device (no-op if disabled)."""
        if not self._auto_enabled or self._mesh is None:
            return pytree
        return gather_from_devices(pytree)
