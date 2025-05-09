"""
Pool implementation for boolean circuit NCA.

This module provides a Pool class that manages a collection of circuits
for training with partial updates between steps.
"""

import jax
import jax.numpy as jp
from flax import struct
from typing import Dict, List, Tuple, Any, Union, Optional
from jax import Array
from functools import partial
import jraph

from boolean_nca_cc.circuits.model import gen_circuit
from boolean_nca_cc.utils.graph_builder import build_graph
from boolean_nca_cc.utils.extraction import extract_logits_from_graph

PyTree = Any


class GraphPool(struct.PyTreeNode):
    """
    Pool class for graph-based circuit NCA.

    Stores a single batched jraph.GraphsTuple and allows
    sampling and updating batches. Also tracks wires and logits in parallel
    for computing functional circuit loss without redundant extraction.
    """

    size: int = struct.field(pytree_node=False)
    # graphs is a single jraph.GraphsTuple where each leaf has a leading batch dimension
    graphs: jraph.GraphsTuple
    # wires is a list of wire matrices with shape [pool_size, ...wire_shape...]
    wires: PyTree = None
    # logits is a list of weight matrices corresponding to each graph
    logits: PyTree = None
    # Reset counter to track which elements were reset recently
    reset_counter: Optional[Array] = None

    @classmethod
    def create(
        cls,
        batched_graphs: jraph.GraphsTuple,
        wires: PyTree = None,
        logits: PyTree = None,
        reset_counter: Optional[Array] = None,
    ) -> "GraphPool":
        """
        Create a new GraphPool instance from a batched GraphsTuple.

        Args:
            batched_graphs: A jraph.GraphsTuple where each leaf node
                            has a leading batch dimension (pool_size).
            wires: The wire matrices corresponding to each graph.
            logits: The logit matrices corresponding to each graph.
            reset_counter: Optional counter to track reset operations.

        Returns:
            A new GraphPool instance.
        """
        # Determine size from the n_node field, assuming it's always present and batched.
        # If build_graph produces n_node of shape (1,), vmap makes it (pool_size, 1).
        # So, shape[0] gives the pool_size.
        if batched_graphs.n_node is None:
            raise ValueError(
                "batched_graphs.n_node cannot be None for GraphPool.create"
            )
        size = batched_graphs.n_node.shape[0]

        # Initialize reset counter if not provided
        if reset_counter is None:
            reset_counter = jp.zeros(size, dtype=jp.int32)

        return cls(
            size=size,
            graphs=batched_graphs,
            wires=wires,
            logits=logits,
            reset_counter=reset_counter,
        )

    @jax.jit
    def update(
        self,
        idxs: Array,
        batch_of_graphs: jraph.GraphsTuple,
        batch_of_wires: PyTree = None,
        batch_of_logits: PyTree = None,
    ) -> "GraphPool":
        """
        Update graphs in the pool at the specified indices with a batch of graphs.

        Args:
            idxs: The indices (1D array) at which to update the graphs in the pool.
            batch_of_graphs: A jraph.GraphsTuple where each leaf has a leading
                             batch dimension corresponding to the size of `idxs`.
                             This contains the new graph data.
            batch_of_wires: Optional PyTree of wires corresponding to batch_of_graphs.
                           If None, wires remain unchanged.
            batch_of_logits: Optional PyTree of logits corresponding to batch_of_graphs.
                           If None, logits remain unchanged.

        Returns:
            A new GraphPool instance with the updated graphs.
        """

        # This function `core_update_logic` is effectively the `f(x, y)` from the JAX error hint.
        # It's called only when `pool_leaf` (x) is not None.
        def core_update_logic(pool_leaf_non_none, batch_leaf_original):
            # pool_leaf_non_none is guaranteed to be an array or some non-None PyTree here.
            # batch_leaf_original comes directly from batch_of_graphs.

            if batch_leaf_original is None:
                # This means a field in the pool is an Array/PyTree (non-None),
                # but the corresponding field in the update batch is None.
                # This is an invalid update, as we cannot set an array field to None.
                raise ValueError(
                    f"Field in pool is an Array/PyTree, but update batch (batch_of_graphs) provides None for this field. "
                    f"Pool leaf type: {type(pool_leaf_non_none)}, Batch leaf value: {batch_leaf_original}"
                )

            # Both pool_leaf_non_none and batch_leaf_original are expected to be actual data (not None).
            # They should be JAX arrays or compatible PyTrees for .at[].set().
            try:
                return pool_leaf_non_none.at[idxs].set(batch_leaf_original)
            except TypeError as e:
                # This can catch errors if, e.g., pool_leaf_non_none is an array
                # but batch_leaf_original is an incompatible type (like a dict) for .at[].set().
                raise TypeError(
                    f"Type error during graph update: Incompatible types for .at[].set(). "
                    f"Pool leaf type: {type(pool_leaf_non_none)}, value: {pool_leaf_non_none}, "
                    f"Batch leaf type: {type(batch_leaf_original)}, value: {batch_leaf_original}. Original error: {e}"
                )

        # The lambda function to be mapped. Structure from JAX error hint:
        # lambda x, y: None if x is None else f(x, y)
        # x corresponds to a leaf from self.graphs (pool_leaf)
        # y corresponds to a leaf from batch_of_graphs (batch_leaf_original)
        mapped_fn = (
            lambda pool_leaf, batch_leaf_original: None
            if pool_leaf is None
            else core_update_logic(pool_leaf, batch_leaf_original)
        )

        updated_graphs_data = jax.tree.map(
            mapped_fn,
            self.graphs,  # Tree 1 (pool's current graphs)
            batch_of_graphs,  # Tree 2 (graphs to update with)
            is_leaf=lambda x: x
            is None,  # Tells tree_map to treat None values as leaves
        )

        # Update wires if provided
        updated_wires = self.wires
        if batch_of_wires is not None and self.wires is not None:
            updated_wires = jax.tree.map(
                lambda pool_wires, batch_wires: pool_wires.at[idxs].set(batch_wires),
                self.wires,
                batch_of_wires,
            )

        # Update logits if provided
        updated_logits = self.logits
        if batch_of_logits is None:
            batch_of_logits = jax.vmap(extract_logits_from_graph, in_axes=(0, None))(
                batch_of_graphs, [l.shape[1:] for l in self.logits]
            )
        if self.logits is not None:
            updated_logits = jax.tree.map(
                lambda pool_logits, batch_logits: pool_logits.at[idxs].set(
                    batch_logits
                ),
                self.logits,
                batch_of_logits,
            )

        # Reset the counter for the indices that were updated
        updated_reset_counter = (
            self.reset_counter.at[idxs].set(0)
            if self.reset_counter is not None
            else None
        )

        return self.replace(
            graphs=updated_graphs_data,
            wires=updated_wires,
            logits=updated_logits,
            reset_counter=updated_reset_counter,
        )

    @partial(jax.jit, static_argnames=("batch_size",))
    def sample(
        self, key: Array, batch_size: int
    ) -> Tuple[Array, jraph.GraphsTuple, Optional[PyTree], Optional[PyTree]]:
        """
        Sample a batch of graphs from the pool.

        Args:
            key: A random key.
            batch_size: The size of the batch to sample.

        Returns:
            A tuple containing:
                - idxs: The sampled batch indices in the pool.
                - sampled_graphs: A jraph.GraphsTuple where each leaf is sliced
                                  according to idxs, representing the sampled batch.
                - sampled_wires: Corresponding wires if available, otherwise None.
                - sampled_logits: Corresponding logits if available, otherwise None.
        """
        idxs = jax.random.choice(key, self.size, shape=(batch_size,))

        def _safe_slice_leaf(pool_leaf):
            if pool_leaf is None:
                return None
            return pool_leaf[idxs]

        sampled_graphs = jax.tree.map(_safe_slice_leaf, self.graphs)

        # Sample wires if they exist
        sampled_wires = None
        if self.wires is not None:
            sampled_wires = jax.tree.map(_safe_slice_leaf, self.wires)

        # Sample logits if they exist
        sampled_logits = None
        if self.logits is not None:
            sampled_logits = jax.tree.map(_safe_slice_leaf, self.logits)

        return idxs, sampled_graphs, sampled_wires, sampled_logits

    # Method to get average update steps of graphs in the pool
    def get_average_update_steps(self) -> float:
        """Get the average number of update steps across all graphs in the pool."""
        if self.graphs.globals is None:
            return 0.0
        # Extract update_steps from the globals (second element in each graph's globals)
        update_steps = self.graphs.globals[..., 1]
        return float(jp.mean(update_steps))

    # Method to get average update steps of a subset of graphs (for reset reporting)
    def get_average_update_steps_for_indices(self, indices: Array) -> float:
        """Get the average number of update steps for specified graph indices."""
        if self.graphs.globals is None:
            return 0.0
        # Extract update_steps for the selected indices
        update_steps = self.graphs.globals[indices, 1]
        return float(jp.mean(update_steps))

    def reset_fraction(
        self,
        key: Array,
        fraction: float,
        new_graphs: jraph.GraphsTuple,
        new_wires: PyTree = None,
        new_logits: PyTree = None,
        reset_strategy: str = "uniform",  # Options: "uniform", "steps_biased", "loss_biased", or "combined"
        combined_weights: Tuple[float, float] = (
            0.5,
            0.5,
        ),  # Weights for [loss, steps] in combined strategy
    ) -> Tuple["GraphPool", float]:
        """
        Reset a random fraction of the pool with fresh graphs.

        Args:
            key: Random key for selection
            fraction: Fraction of pool to reset (between 0 and 1)
            new_graphs: Fresh graphs to use for reset
            new_wires: Fresh wires to use for reset
            new_logits: Fresh logits to use for reset
            reset_strategy: Strategy for selecting graphs to reset:
                            "uniform" - uniform random selection
                            "steps_biased" - bias by update steps (more steps = higher probability)
                            "loss_biased" - bias by loss value (higher loss = higher probability)
                            "combined" - combine both loss and update steps for selection
            combined_weights: Tuple of weights (loss_weight, steps_weight) for the combined strategy

        Returns:
            Updated pool with reset elements and the average update steps of reset graphs
        """
        # Calculate number of elements to reset
        num_reset = jp.maximum(1, jp.round(fraction * self.size).astype(jp.int32))

        # Split the key for different random operations
        key1, key2 = jax.random.split(key)

        # Select elements to reset based on the reset strategy
        if reset_strategy == "uniform":
            reset_idxs = jax.random.choice(
                key1, self.size, shape=(num_reset,), replace=False
            )
        elif reset_strategy == "steps_biased":
            # Selection biased by update steps
            if self.graphs.globals is None:
                # Fallback to uniform selection if no update steps
                reset_idxs = jax.random.choice(
                    key1, self.size, shape=(num_reset,), replace=False
                )
            else:
                # Extract update steps for each graph
                update_steps = self.graphs.globals[..., 1]

                # Create probabilities proportional to update steps
                # Add small constant to prevent zero probabilities and normalize
                probs = update_steps + 1.0  # Add 1 to avoid zeros
                probs = probs / jp.sum(probs)  # Normalize to sum to 1

                # Sample indices based on these probabilities
                reset_idxs = jax.random.choice(
                    key1, self.size, shape=(num_reset,), replace=False, p=probs
                )
        elif reset_strategy == "loss_biased":
            # Selection biased by loss value (higher loss = higher probability of reset)
            if self.graphs.globals is None:
                # Fallback to uniform selection if no loss values
                reset_idxs = jax.random.choice(
                    key1, self.size, shape=(num_reset,), replace=False
                )
            else:
                # Extract loss values for each graph
                loss_values = self.graphs.globals[..., 0]

                # Create probabilities proportional to loss values
                # Add small constant to prevent zero probabilities and normalize
                probs = loss_values + 1e-6  # Add small epsilon to avoid zeros

                # Clip extreme values for numerical stability
                probs = jp.clip(probs, 0.0, 100.0)

                # Normalize to sum to 1
                probs = probs / jp.sum(probs)

                # Sample indices based on these probabilities
                reset_idxs = jax.random.choice(
                    key1, self.size, shape=(num_reset,), replace=False, p=probs
                )
        elif reset_strategy == "combined":
            # Combine both loss values and update steps for selection
            if self.graphs.globals is None:
                # Fallback to uniform selection if no globals
                reset_idxs = jax.random.choice(
                    key1, self.size, shape=(num_reset,), replace=False
                )
            else:
                # Extract loss values and update steps
                loss_values = self.graphs.globals[..., 0]
                update_steps = self.graphs.globals[..., 1]

                # Get weights for the combined score
                loss_weight, steps_weight = combined_weights

                # Compute normalized scores for both factors
                # For loss: higher is worse, so higher probability
                loss_scores = (loss_values - jp.min(loss_values)) / (
                    jp.max(loss_values) - jp.min(loss_values) + 1e-6
                )

                # For steps: more steps means older circuit, so higher probability
                step_scores = (update_steps - jp.min(update_steps)) / (
                    jp.max(update_steps) - jp.min(update_steps) + 1e-6
                )

                # Combine the two scores with configured weights
                combined_scores = loss_weight * loss_scores + steps_weight * step_scores

                # Add small constant for numerical stability
                probs = combined_scores + 1e-6

                # Normalize to sum to 1
                probs = probs / jp.sum(probs)

                # Sample indices based on combined probabilities
                reset_idxs = jax.random.choice(
                    key1, self.size, shape=(num_reset,), replace=False, p=probs
                )
        else:
            raise ValueError(
                f"Unknown reset_strategy: {reset_strategy}. "
                f"Must be 'uniform', 'steps_biased', 'loss_biased', or 'combined'."
            )

        # Calculate average update steps of graphs being reset
        avg_steps_reset = self.get_average_update_steps_for_indices(reset_idxs)

        # Create key for sampling new graphs and wires
        key_sample = jax.random.fold_in(key2, 0)

        # Sample elements to reset from the new graphs
        sample_idxs = jax.random.choice(
            key_sample, new_graphs.n_node.shape[0], shape=(num_reset,), replace=False
        )
        reset_graphs = jax.tree.map(
            lambda leaf: None if leaf is None else leaf[sample_idxs], new_graphs
        )

        # Sample wires if provided
        reset_wires = None
        if new_wires is not None:
            reset_wires = jax.tree.map(lambda leaf: leaf[sample_idxs], new_wires)

        # Sample logits if provided
        reset_logits = None
        if new_logits is not None:
            reset_logits = jax.tree.map(lambda leaf: leaf[sample_idxs], new_logits)

        # Update the pool with reset elements
        reset_pool = self.update(reset_idxs, reset_graphs, reset_wires, reset_logits)

        # Increment the reset counter for all elements
        if reset_pool.reset_counter is not None:
            new_counter = reset_pool.reset_counter + 1
            reset_pool = reset_pool.replace(reset_counter=new_counter)

        return reset_pool, avg_steps_reset


def initialize_graph_pool(
    rng: jax.random.PRNGKey,
    layer_sizes: List[Tuple[int, int]],
    pool_size: int,
    input_n: int,
    arity: int = 2,
    hidden_dim: int = 16,
    loss_value: float = 0.0,
    wiring_mode: str = "random",
) -> GraphPool:
    """
    Initialize a pool of graphs using a provided graph creation function.

    Args:
        rng: Random key
        layer_sizes: Circuit layer sizes
        pool_size: Number of graphs in the pool
        input_n: Number of inputs to the circuit
        arity: Number of inputs per gate
        hidden_dim: Dimension of hidden features
        loss_value: Initial loss value for graph globals
        wiring_mode: Mode for generating wirings ("random" or "fixed")

    Returns:
        Initialized GraphPool
    """
    # Generate circuit wirings based on wiring mode
    if wiring_mode == "fixed":
        # In fixed mode, generate a single wiring and repeat it
        single_wires, single_logits = gen_circuit(jax.random.PRNGKey(rng), layer_sizes, arity=arity)
        # print(f"INIT WITH RNG {rng}")

        # Replicate the same wiring for all circuits in the pool
        all_wires = jax.tree.map(
            lambda leaf: jp.repeat(leaf[None, ...], pool_size, axis=0), single_wires
        )
        all_logits = jax.tree.map(
            lambda leaf: jp.repeat(leaf[None, ...], pool_size, axis=0), single_logits
        )
    else:  # wiring_mode == "random"
        # In random mode, generate different wirings for each circuit
        rngs = jax.random.split(rng, pool_size)
        vmap_gen_circuit = jax.vmap(
            lambda rng: gen_circuit(rng, layer_sizes, arity=arity)
        )
        all_wires, all_logits = vmap_gen_circuit(rngs)

    # Generate graphs in parallel using vmap
    # Create globals with both loss value and update steps counter (initialized to 0)
    # The globals structure will be [loss_value, update_steps]
    vmap_build_graph = jax.vmap(
        lambda logit, wires: build_graph(
            logits=logit,
            wires=wires,
            input_n=input_n,
            arity=arity,
            hidden_dim=hidden_dim,
            loss_value=loss_value,
            update_steps=0,  # Initialize update steps counter to 0
        )
    )
    graphs = vmap_build_graph(all_logits, all_wires)

    # Initialize reset counter
    reset_counter = jp.zeros(pool_size, dtype=jp.int32)

    return GraphPool.create(graphs, all_wires, all_logits, reset_counter)
