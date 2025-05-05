"""
Pool implementation for boolean circuit NCA.

This module provides a Pool class that manages a collection of circuits
for training with partial updates between steps.
"""

import jax
import jax.numpy as jp
from flax import struct
from typing import Dict, List, Tuple, Any, Union
from jax import Array
from functools import partial

PyTree = Any


class CircuitPool(struct.PyTreeNode):
    """
    Pool class for boolean circuit NCA.

    Stores a collection of circuits (wires, logits, graphs) and allows
    sampling batches with partial resets between training steps.
    """

    size: int = struct.field(pytree_node=False)
    data: PyTree  # Contains wires, logits, and possibly graphs

    @classmethod
    def create(cls, data: PyTree) -> "CircuitPool":
        """
        Create a new CircuitPool instance.

        Args:
            data: Initial data to store in the pool. Should contain at minimum
                 wires and logits, and possibly other circuit-related data.

        Returns:
            A new CircuitPool instance.
        """
        # Get size from first leaf in the data
        size = jax.tree.leaves(data)[0].shape[0]
        return cls(size=size, data=data)

    @jax.jit
    def update(self, idxs: Array, batch: PyTree) -> "CircuitPool":
        """
        Update circuits in the pool at the specified indices.

        Args:
            idxs: The indices at which to update the circuits.
            batch: The batch of circuits to update at the specified indices.

        Returns:
            A new CircuitPool instance with the updated circuits.
        """
        data = jax.tree.map(
            lambda data_leaf, batch_leaf: data_leaf.at[idxs].set(batch_leaf),
            self.data,
            batch,
        )
        return self.replace(data=data)

    @partial(jax.jit, static_argnames=("batch_size",))
    def sample(self, key: Array, batch_size: int) -> Tuple[Array, PyTree]:
        """
        Sample a batch of circuits from the pool.

        Args:
            key: A random key.
            batch_size: The size of the batch to sample.

        Returns:
            A tuple containing the batch indices in the pool and the batch of circuits.
        """
        idxs = jax.random.choice(key, self.size, shape=(batch_size,))
        batch = jax.tree.map(lambda leaf: leaf[idxs], self.data)
        return idxs, batch
