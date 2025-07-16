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
from boolean_nca_cc.training.pool.structural_perturbation import (
    create_reproducible_knockout_pattern,
    extract_layer_info_from_graph,
)

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
    # Knockout patterns for persistent structural perturbations
    knockout_patterns: Optional[Array] = None

    @classmethod
    def create(
        cls,
        batched_graphs: jraph.GraphsTuple,
        wires: PyTree = None,
        logits: PyTree = None,
        reset_counter: Optional[Array] = None,
        knockout_patterns: Optional[Array] = None,
    ) -> "GraphPool":
        """
        Create a new GraphPool instance from a batched GraphsTuple.

        Args:
            batched_graphs: A jraph.GraphsTuple where each leaf node
                            has a leading batch dimension (pool_size).
            wires: The wire matrices corresponding to each graph.
            logits: The logit matrices corresponding to each graph.
            reset_counter: Optional counter to track reset operations.
            knockout_patterns: Optional boolean array for persistent knockouts.

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

        # Initialize knockout patterns if not provided
        if knockout_patterns is None:
            # Infer num_nodes from the logits feature of the graph's nodes
            num_nodes = batched_graphs.nodes["logits"].shape[1]
            knockout_patterns = jp.zeros((size, num_nodes), dtype=jp.bool_)

        return cls(
            size=size,
            graphs=batched_graphs,
            reset_counter=reset_counter,
            knockout_patterns=knockout_patterns,
        )

    @jax.jit
    def update(
        self,
        idxs: Array,
        batch_of_graphs: jraph.GraphsTuple,
        batch_of_knockout_patterns: Optional[Array] = None,
    ) -> "GraphPool":
        """
        Update graphs in the pool at the specified indices with a batch of graphs.

        Args:
            idxs: The indices (1D array) at which to update the graphs in the pool.
            batch_of_graphs: A jraph.GraphsTuple where each leaf has a leading
                             batch dimension corresponding to the size of `idxs`.
                             This contains the new graph data.
            batch_of_knockout_patterns: Optional knockout patterns for updated graphs.

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

        # Update knockout patterns if provided
        updated_knockout_patterns = self.knockout_patterns
        if (
            batch_of_knockout_patterns is not None
            and self.knockout_patterns is not None
        ):
            updated_knockout_patterns = self.knockout_patterns.at[idxs].set(
                batch_of_knockout_patterns
            )

        # Reset the counter for the indices that were updated
        updated_reset_counter = (
            self.reset_counter.at[idxs].set(0)
            if self.reset_counter is not None
            else None
        )

        return self.replace(
            graphs=updated_graphs_data,
            reset_counter=updated_reset_counter,
            knockout_patterns=updated_knockout_patterns,
        )


    @partial(jax.jit, static_argnames=("batch_size",))
    def sample(
        self, key: Array, batch_size: int
    ) -> Tuple[Array, jraph.GraphsTuple, Optional[PyTree], Optional[PyTree], Optional[Array]]:
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
                - sampled_knockout_patterns: Corresponding knockout patterns.
        """
        idxs = jax.random.choice(key, self.size, shape=(batch_size,))

        def _safe_slice_leaf(pool_leaf):
            if pool_leaf is None:
                return None
            return pool_leaf[idxs]

        sampled_graphs = jax.tree.map(_safe_slice_leaf, self.graphs)
        # Sample knockout patterns
        sampled_knockout_patterns = jax.tree.map(_safe_slice_leaf, self.knockout_patterns)

        return idxs, sampled_graphs, sampled_knockout_patterns

    # # Method to get average update steps of graphs in the pool
    # def get_average_update_steps(self) -> float:
    #     """Get the average number of update steps across all graphs in the pool."""
    #     if self.graphs.globals is None:
    #         return 0.0
    #     # Extract update_steps from the globals (second element in each graph's globals)
    #     update_steps = self.graphs.globals[..., 1]
    #     return float(jp.mean(update_steps))

    # # Method to get average update steps of a subset of graphs (for reset reporting)
    # def get_average_update_steps_for_indices(self, indices: Array) -> float:
    #     """Get the average number of update steps for specified graph indices."""
    #     if self.graphs.globals is None:
    #         return 0.0
    #     # Extract update_steps for the selected indices
    #     update_steps = self.graphs.globals[indices, 1]
    #     return float(jp.mean(update_steps))


    def reset_fraction(
        self,
        key: Array,
        fraction: float,
        new_graphs: jraph.GraphsTuple,
        new_knockout_patterns: Optional[Array] = None,
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
            new_knockout_patterns: Fresh knockout patterns for reset
            reset_strategy: Strategy for selecting graphs to reset:
                            "uniform" - uniform random selection
                            "steps_biased" - bias by update steps (more steps = higher probability)
                            "loss_biased" - bias by loss value (higher loss = higher probability)
                            "combined" - combine both loss and update steps for selection
            combined_weights: Tuple of weights (loss_weight, steps_weight) for the combined strategy

        Returns:
            Updated pool with reset elements and the average update steps of reset graphs
        """
        # Split the key for different random operations
        selection_key, sampling_key = jax.random.split(key)

        # Use the refactored selection logic
        reset_idxs, avg_steps_reset = self.get_reset_indices(
            selection_key, fraction, reset_strategy, combined_weights
        )

        num_reset = len(reset_idxs)

        # Create key for sampling new graphs and wires
        key_sample = jax.random.fold_in(sampling_key, 0)

        # Sample elements to reset from the new graphs
        sample_idxs = jax.random.choice(
            key_sample, new_graphs.n_node.shape[0], shape=(num_reset,), replace=False
        )
        reset_graphs = jax.tree.map(
            lambda leaf: None if leaf is None else leaf[sample_idxs], new_graphs
        )


        # Sample knockout patterns if provided
        reset_knockout_patterns = None
        if new_knockout_patterns is not None:
            reset_knockout_patterns = new_knockout_patterns[sample_idxs]

        # Update the pool with reset elements
        reset_pool = self.update(
            reset_idxs,
            reset_graphs,
            reset_knockout_patterns,
        )

        # Increment the reset counter for all elements
        if reset_pool.reset_counter is not None:
            new_counter = reset_pool.reset_counter + 1
            reset_pool = reset_pool.replace(reset_counter=new_counter)

        return reset_pool, avg_steps_reset

    def get_reset_indices(
        self,
        key: Array,
        fraction: float,
        reset_strategy: str = "uniform",
        combined_weights: Tuple[float, float] = (0.5, 0.5),
    ) -> Tuple[Array, float]:
        """
        Get indices of circuits to reset based on the specified strategy.

        This separates the selection logic from the update logic for more flexibility.

        Args:
            key: Random key for selection
            fraction: Fraction of pool to reset (between 0 and 1)
            reset_strategy: Strategy for selecting graphs to reset
            combined_weights: Weights for loss and steps in combined strategy

        Returns:
            Tuple of (reset_indices, avg_steps_of_reset_circuits)
        """
        # Calculate number of elements to reset
        num_reset = jp.maximum(1, jp.round(fraction * self.size).astype(jp.int32))

        # Select elements to reset based on the reset strategy
        if reset_strategy == "uniform":
            reset_idxs = jax.random.choice(
                key, self.size, shape=(num_reset,), replace=False
            )
        elif reset_strategy == "steps_biased":
            # Selection biased by update steps
            if self.graphs.globals is None:
                # Fallback to uniform selection if no update steps
                reset_idxs = jax.random.choice(
                    key, self.size, shape=(num_reset,), replace=False
                )
            else:
                # Extract update steps for each graph
                update_steps = self.graphs.globals[..., 1]

                # Create probabilities proportional to update steps
                probs = update_steps
                if not jp.any(probs):
                    probs = jp.ones(self.size) / self.size
                else:
                    probs = probs / jp.sum(probs)

                # Sample indices based on these probabilities
                reset_idxs = jax.random.choice(
                    key, self.size, shape=(num_reset,), replace=False, p=probs
                )

        elif reset_strategy == "loss_biased":
            # Selection biased by loss value (higher loss = higher probability of reset)
            if self.graphs.globals is None:
                # Fallback to uniform selection if no loss values
                reset_idxs = jax.random.choice(
                    key, self.size, shape=(num_reset,), replace=False
                )
            else:
                # Extract loss values for each graph
                loss_values = self.graphs.globals[..., 0]

                # Create probabilities proportional to loss values
                probs = loss_values
                if not jp.any(probs):
                    probs = jp.ones(self.size) / self.size
                else:
                    probs = probs / jp.sum(probs)

                # Sample indices based on these probabilities
                reset_idxs = jax.random.choice(
                    key, self.size, shape=(num_reset,), replace=False, p=probs
                )

        elif reset_strategy == "combined":
            # Combine both loss values and update steps for selection
            if self.graphs.globals is None:
                # Fallback to uniform selection if no globals
                reset_idxs = jax.random.choice(
                    key, self.size, shape=(num_reset,), replace=False
                )
            else:
                # Extract loss values and update steps
                loss_values = self.graphs.globals[..., 0]
                update_steps = self.graphs.globals[..., 1]

                # Get weights for the combined score
                loss_weight, steps_weight = combined_weights

                if not jp.any(loss_values):
                    loss_scores = jp.ones(self.size) / self.size
                else:
                    loss_scores = loss_values / jp.sum(loss_values)

                if not jp.any(update_steps):
                    step_scores = jp.ones(self.size) / self.size
                else:
                    step_scores = update_steps / jp.sum(update_steps)

                # Combine the two scores with configured weights
                combined_scores = loss_weight * loss_scores + steps_weight * step_scores
                probs = combined_scores / jp.sum(combined_scores)

                # Sample indices based on combined probabilities
                reset_idxs = jax.random.choice(
                    key, self.size, shape=(num_reset,), replace=False, p=probs
                )
        else:
            raise ValueError(
                f"Unknown reset_strategy: {reset_strategy}. "
                f"Must be 'uniform', 'steps_biased', 'loss_biased', or 'combined'."
            )

        # Calculate average update steps of graphs being reset
        avg_steps_reset = self.get_average_update_steps_for_indices(reset_idxs)

        return reset_idxs, avg_steps_reset


def initialize_graph_pool(
    rng: jax.random.PRNGKey,
    layer_sizes: List[Tuple[int, int]],
    pool_size: int,
    input_n: int,
    arity: int = 2,
    circuit_hidden_dim: int = 16,
    loss_value: float = 0.0,
    knockout_config: Optional[Dict[str, Any]] = None,
    knockout_patterns: Optional[Array] = None,  # Pre-generated knockout patterns to use directly
) -> GraphPool:
    """
    Initialize a pool of graphs using a provided graph creation function.

    Args:
        rng: Random key
        layer_sizes: Circuit layer sizes
        pool_size: Number of graphs in the pool
        input_n: Number of inputs to the circuit
        arity: Number of inputs per gate
        circuit_hidden_dim: Dimension of hidden features
        loss_value: Initial loss value for graph globals
        knockout_config: Optional configuration to apply persistent knockouts
                         to a fraction of the newly created circuits.
                         Expected keys: 'fraction', 'damage_prob'.
        knockout_patterns: Optional pre-generated knockout patterns to use directly.
                          If provided, takes precedence over knockout_config.
                          Shape: (num_patterns, num_nodes)

    Returns:
        Initialized GraphPool
    """
    # Generate circuit wirings based on wiring mode

    # Original behavior: single wiring repeated for all circuits
    single_wires, single_logits = gen_circuit(rng, layer_sizes, arity=arity)

    # Replicate the same wiring for all circuits in the pool
    all_wires = jax.tree.map(
        lambda leaf: jp.repeat(leaf[None, ...], pool_size, axis=0), single_wires
    )
    all_logits = jax.tree.map(
        lambda leaf: jp.repeat(leaf[None, ...], pool_size, axis=0),
        single_logits,
    )
    # Generate graphs in parallel using vmap
    # Create globals with both loss value and update steps counter (initialized to 0)
    # The globals structure will be [loss_value, update_steps]
    vmap_build_graph = jax.vmap(
        lambda logit, wires: build_graph(
            logits=logit,
            wires=wires,
            input_n=input_n,
            arity=arity,
            circuit_hidden_dim=circuit_hidden_dim,
            loss_value=loss_value,
            update_steps=0,  # Initialize update steps counter to 0
        )
    )
    graphs = vmap_build_graph(all_logits, all_wires)

    # Initialize reset counter
    reset_counter = jp.zeros(pool_size, dtype=jp.int32)

    # Initialize knockout patterns to all False (no knockouts)
    num_nodes = graphs.nodes["logits"].shape[1]
    pool_knockout_patterns = jp.zeros((pool_size, num_nodes), dtype=jp.bool_)

    # If pre-generated knockout patterns are provided, use them directly
    if knockout_patterns is not None:
        knockout_key, rng = jax.random.split(rng)
        num_patterns_available = knockout_patterns.shape[0]
        
        # Sample patterns with replacement from the vocabulary for the entire pool
        pattern_indices = jax.random.choice(
            knockout_key, num_patterns_available, shape=(pool_size,), replace=True
        )
        pool_knockout_patterns = knockout_patterns[pattern_indices]
        
    # If knockout config is provided and no pre-generated patterns, apply persistent knockouts to a fraction of the new pool
    elif knockout_config and knockout_config.get("fraction", 0.0) > 0.0:
        knockout_key, rng = jax.random.split(rng)
        fraction = knockout_config.get("fraction")
        damage_prob = knockout_config.get("damage_prob")

        num_to_damage = jp.round(pool_size * fraction).astype(jp.int32)
        damage_indices = jax.random.choice(
            knockout_key, pool_size, shape=(num_to_damage,), replace=False
        )

        # Extract true layer sizes from one of the generated graphs to ensure consistency
        single_graph = jax.tree.map(lambda x: x[0], graphs)
        true_layer_sizes = extract_layer_info_from_graph(single_graph, input_n)

        # Generate new patterns for the selected indices
        pattern_keys = jax.random.split(knockout_key, num_to_damage)
        vmapped_pattern_creator = jax.vmap(
            lambda k: create_reproducible_knockout_pattern(
                key=k,
                layer_sizes=true_layer_sizes,
                damage_prob=damage_prob,
                input_n=input_n,
            )
        )
        new_patterns = vmapped_pattern_creator(pattern_keys)

        # Apply the new patterns at the selected indices
        pool_knockout_patterns = pool_knockout_patterns.at[damage_indices].set(new_patterns)

    return GraphPool.create(
        graphs, all_wires, all_logits, reset_counter, pool_knockout_patterns
    )
