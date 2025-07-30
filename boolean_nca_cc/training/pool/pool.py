"""
Pool implementation for boolean circuit NCA.

This module provides a Pool class that manages a collection of circuits
for training with partial updates between steps.
"""

from functools import partial
from typing import Any

import jax
import jax.numpy as jp
import jraph
from flax import struct
from jax import Array

from boolean_nca_cc.circuits.model import gen_circuit
from boolean_nca_cc.training.pool.perturbation import mutate_wires_batch
from boolean_nca_cc.utils.extraction import extract_logits_from_graph
from boolean_nca_cc.utils.graph_builder import build_graph

PyTree = Any


class GraphPool(struct.PyTreeNode):
    """
    Pool class for graph-based circuit NCA.

    Stores a single batched jraph.GraphsTuple and allows
    sampling and updating batches. Also tracks wires and logits in parallel
    for computing functional circuit loss without redundant extraction.
    Now also tracks gate masks for knockout patterns.
    """

    size: int = struct.field(pytree_node=False)
    # graphs is a single jraph.GraphsTuple where each leaf has a leading batch dimension
    graphs: jraph.GraphsTuple
    # wires is a list of wire matrices with shape [pool_size, ...wire_shape...]
    wires: PyTree = None
    # logits is a list of weight matrices corresponding to each graph
    logits: PyTree = None
    # gate_masks is a list of mask matrices for knockout patterns, in flat format
    # Each element has shape [pool_size, total_gates] where total_gates = sum(gate_n for gate_n, _ in layer_sizes)
    gate_masks: jp.ndarray | None = None
    # Reset counter to track which elements were reset recently
    reset_counter: Array | None = None

    @classmethod
    def create(
        cls,
        batched_graphs: jraph.GraphsTuple,
        wires: PyTree = None,
        logits: PyTree = None,
        gate_masks: jp.ndarray | None = None,
        reset_counter: Array | None = None,
    ) -> "GraphPool":
        """
        Create a new GraphPool instance from a batched GraphsTuple.

        Args:
            batched_graphs: A jraph.GraphsTuple where each leaf node
                            has a leading batch dimension (pool_size).
            wires: The wire matrices corresponding to each graph.
            logits: The logit matrices corresponding to each graph.
            gate_masks: Optional flat gate masks with shape (pool_size, total_gates)
            reset_counter: Optional counter to track reset operations.

        Returns:
            A new GraphPool instance.
        """
        # Determine size from the n_node field, assuming it's always present and batched.
        # If build_graph produces n_node of shape (1,), vmap makes it (pool_size, 1).
        # So, shape[0] gives the pool_size.
        if batched_graphs.n_node is None:
            raise ValueError("batched_graphs.n_node cannot be None for GraphPool.create")
        size = batched_graphs.n_node.shape[0]

        # Initialize reset counter if not provided
        if reset_counter is None:
            reset_counter = jp.zeros(size, dtype=jp.int32)

        return cls(
            size=size,
            graphs=batched_graphs,
            wires=wires,
            logits=logits,
            gate_masks=gate_masks,
            reset_counter=reset_counter,
        )

    @jax.jit
    def update(
        self,
        idxs: Array,
        batch_of_graphs: jraph.GraphsTuple,
        batch_of_wires: PyTree = None,
        batch_of_logits: PyTree = None,
        batch_of_gate_masks: jp.ndarray | None = None,
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
            batch_of_gate_masks: Optional flat gate masks with shape (batch_size, total_gates).
                               If None, gate masks remain unchanged.

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
        def mapped_fn(pool_leaf, batch_leaf_original):
            return None if pool_leaf is None else core_update_logic(pool_leaf, batch_leaf_original)

        updated_graphs_data = jax.tree.map(
            mapped_fn,
            self.graphs,  # Tree 1 (pool's current graphs)
            batch_of_graphs,  # Tree 2 (graphs to update with)
            is_leaf=lambda x: x is None,  # Tells tree_map to treat None values as leaves
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
                batch_of_graphs, [log.shape[1:] for log in self.logits]
            )
        if self.logits is not None:
            updated_logits = jax.tree.map(
                lambda pool_logits, batch_logits: pool_logits.at[idxs].set(batch_logits),
                self.logits,
                batch_of_logits,
            )

        # Update gate_masks if provided
        updated_gate_masks = self.gate_masks
        if batch_of_gate_masks is not None and self.gate_masks is not None:
            updated_gate_masks = jax.tree.map(
                lambda pool_gate_masks, batch_gate_masks: pool_gate_masks.at[idxs].set(
                    batch_gate_masks
                ),
                self.gate_masks,
                batch_of_gate_masks,
            )

        # Reset the counter for the indices that were updated
        updated_reset_counter = (
            self.reset_counter.at[idxs].set(0) if self.reset_counter is not None else None
        )

        return self.replace(
            graphs=updated_graphs_data,
            wires=updated_wires,
            logits=updated_logits,
            gate_masks=updated_gate_masks,
            reset_counter=updated_reset_counter,
        )

    @partial(jax.jit, static_argnames=("batch_size",))
    def sample(
        self, key: Array, batch_size: int
    ) -> tuple[Array, jraph.GraphsTuple, PyTree | None, PyTree | None, jp.ndarray | None]:
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
                - sampled_gate_masks: Corresponding gate masks if available, otherwise None.
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

        # Sample gate masks if they exist
        sampled_gate_masks = None
        if self.gate_masks is not None:
            sampled_gate_masks = _safe_slice_leaf(self.gate_masks)

        return idxs, sampled_graphs, sampled_wires, sampled_logits, sampled_gate_masks

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

    def get_wiring_diversity(self, layer_sizes: list[tuple[int, int]] | None = None) -> float:
        """
        Calculate the wiring diversity of the pool using entropy-based measurement.

        Returns a value between 0 and 1:
        - 0: All circuits have identical wiring (fixed mode)
        - 1: Circuits have maximally diverse wiring (random mode)
        - 0 to 1: Varying diversity (genetic mode evolving over time)

        Args:
            layer_sizes: List of (nodes, group_size) tuples for each layer.
                        If None, attempts to infer from circuit structure.

        Returns:
            Float between 0 and 1 representing wiring diversity
        """
        if self.wires is None:
            return 0.0

        total_entropy = 0.0
        total_max_entropy = 0.0

        # Iterate through each layer's wires
        for layer_idx, layer_wires in enumerate(self.wires):
            if layer_wires.size == 0:
                continue

            # Get the shape: [pool_size, arity, group_n]
            pool_size, arity, group_n = layer_wires.shape

            # Determine the number of possible values for connections in this layer
            # Connections in layer i point to nodes in layer i-1
            if layer_sizes is not None and layer_idx < len(layer_sizes):
                # Use provided layer sizes to get the previous layer size
                prev_layer_nodes = layer_sizes[layer_idx][
                    0
                ]  # layer_idx corresponds to previous layer
            else:
                # Fallback: infer from the actual connection values
                # Find the maximum connection value in this layer across all circuits
                max_connection_value = jp.max(layer_wires)
                prev_layer_nodes = int(max_connection_value) + 1

            # Maximum entropy for this layer is log(prev_layer_nodes)
            # since connections can take values 0 to prev_layer_nodes-1
            layer_max_entropy = jp.log(float(prev_layer_nodes))

            # For each connection position (arity, group_n), calculate entropy across pool
            for a in range(arity):
                for g in range(group_n):
                    # Get values for this connection position across all circuits in pool
                    connection_values = layer_wires[:, a, g]  # Shape: [pool_size]

                    # Calculate entropy of this connection position
                    unique_values, counts = jp.unique(
                        connection_values, return_counts=True, size=prev_layer_nodes
                    )

                    # Filter out padded zeros from unique (JAX pads unique to specified size)
                    valid_mask = counts > 0
                    valid_counts = counts[valid_mask]

                    if valid_counts.size == 0:
                        continue

                    # Calculate probabilities
                    probs = valid_counts / jp.sum(valid_counts)

                    # Calculate entropy: -sum(p * log(p))
                    # Use jp.where to avoid log(0)
                    log_probs = jp.where(probs > 0, jp.log(probs), 0.0)
                    entropy = -jp.sum(probs * log_probs)

                    total_entropy += entropy
                    total_max_entropy += layer_max_entropy

        if total_max_entropy == 0.0:
            return 0.0

        # Normalize by the total maximum possible entropy
        normalized_diversity = total_entropy / total_max_entropy

        return float(jp.clip(normalized_diversity, 0.0, 1.0))

    def reset_fraction(
        self,
        key: Array,
        fraction: float,
        new_graphs: jraph.GraphsTuple,
        new_wires: PyTree = None,
        new_logits: PyTree = None,
        new_gate_masks: jp.ndarray | None = None,
        reset_strategy: str = "uniform",  # Options: "uniform", "steps_biased", "loss_biased", or "combined"
        combined_weights: tuple[float, float] = (
            0.5,
            0.5,
        ),  # Weights for [loss, steps] in combined strategy
    ) -> tuple["GraphPool", float]:
        """
        Reset a random fraction of the pool with fresh graphs.

        Args:
            key: Random key for selection
            fraction: Fraction of pool to reset (between 0 and 1)
            new_graphs: Fresh graphs to use for reset
            new_wires: Fresh wires to use for reset
            new_logits: Fresh logits to use for reset
            new_gate_masks: Fresh gate masks to use for reset (flat format)
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

        # Sample wires if provided
        reset_wires = None
        if new_wires is not None:
            reset_wires = jax.tree.map(lambda leaf: leaf[sample_idxs], new_wires)

        # Sample logits if provided
        reset_logits = None
        if new_logits is not None:
            reset_logits = jax.tree.map(lambda leaf: leaf[sample_idxs], new_logits)

        # Sample gate masks if provided
        reset_gate_masks = None
        if new_gate_masks is not None:
            reset_gate_masks = new_gate_masks[sample_idxs]

        # Update the pool with reset elements
        reset_pool = self.update(
            reset_idxs, reset_graphs, reset_wires, reset_logits, reset_gate_masks
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
        combined_weights: tuple[float, float] = (0.5, 0.5),
    ) -> tuple[Array, float]:
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
            reset_idxs = jax.random.choice(key, self.size, shape=(num_reset,), replace=False)
        elif reset_strategy == "steps_biased":
            # Selection biased by update steps
            if self.graphs.globals is None:
                # Fallback to uniform selection if no update steps
                reset_idxs = jax.random.choice(key, self.size, shape=(num_reset,), replace=False)
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
                reset_idxs = jax.random.choice(key, self.size, shape=(num_reset,), replace=False)
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
                reset_idxs = jax.random.choice(key, self.size, shape=(num_reset,), replace=False)
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

    def reset_with_genetic_mutation(
        self,
        key: Array,
        fraction: float,
        layer_sizes: list[tuple[int, int]],
        input_n: int,
        arity: int,
        circuit_hidden_dim: int,
        mutation_rate: float = 0.1,
        n_swaps_per_layer: int = 1,
        reset_strategy: str = "uniform",
        combined_weights: tuple[float, float] = (0.5, 0.5),
    ) -> tuple["GraphPool", float]:
        """
        Reset a fraction of the pool using genetic mutation of existing circuits.

        This method:
        1. Selects circuits to reset using the specified strategy
        2. Takes existing circuits from the pool and mutates their wiring
        3. Generates fresh logits for the mutated circuits
        4. Updates the pool with the mutated circuits

        Args:
            key: Random key for mutations and selection
            fraction: Fraction of pool to reset
            layer_sizes: Circuit layer sizes for logit generation
            input_n: Number of input nodes
            arity: Number of inputs per gate
            circuit_hidden_dim: Hidden dimension for graphs
            mutation_rate: Rate of wire mutations (0.0 to 1.0)
            n_swaps_per_layer: Number of swaps per layer for genetic mutation
            reset_strategy: Strategy for selecting circuits to reset
            combined_weights: Weights for combined reset strategy

        Returns:
            Tuple of (updated_pool, avg_steps_of_reset_circuits)
        """
        # Split keys for different operations
        selection_key, mutation_key, logit_key = jax.random.split(key, 3)

        # Get indices of circuits to reset
        reset_idxs, avg_steps_reset = self.get_reset_indices(
            selection_key, fraction, reset_strategy, combined_weights
        )

        num_reset = len(reset_idxs)

        # Sample existing circuits from the pool for mutation
        # We'll mutate random circuits from the pool, not necessarily the ones being reset
        source_idxs = jax.random.choice(mutation_key, self.size, shape=(num_reset,), replace=True)

        # Extract wires from the source circuits
        source_wires = jax.tree.map(lambda leaf: leaf[source_idxs], self.wires)

        # Apply genetic mutations to the source wires
        mutated_wires = mutate_wires_batch(
            source_wires, mutation_key, mutation_rate, n_swaps_per_layer
        )

        # Generate fresh logits for the mutated circuits
        from boolean_nca_cc.circuits.model import make_nops

        fresh_logits = []
        for out_n, group_size in layer_sizes[1:]:  # Skip input layer
            layer_logits = make_nops(out_n, arity, group_size)
            # Repeat for batch
            batched_logits = jp.repeat(layer_logits[None, ...], num_reset, axis=0)
            fresh_logits.append(batched_logits)

        # Build new graphs with mutated wires and fresh logits
        vmap_build_graph = jax.vmap(
            lambda logit, wires, gate_knockout_mask: build_graph(
                logits=logit,
                wires=wires,
                input_n=input_n,
                arity=arity,
                circuit_hidden_dim=circuit_hidden_dim,
                loss_value=0.0,  # Reset loss
                update_steps=0,  # Reset update steps
                gate_knockout_mask=gate_knockout_mask,
                knockout_strategy="untouched",  # Fresh circuits have no knockouts
                faulty_logit_value=-10.0,  # Default faulty value for fresh circuits
            )
        )

        # For genetic mutation, use fresh gate masks (all active)
        if self.gate_masks is not None:
            total_gates = self.gate_masks.shape[1]
            fresh_gate_masks = jp.ones((num_reset, total_gates), dtype=jp.float32)
        else:
            fresh_gate_masks = jp.zeros((num_reset, 0), dtype=jp.float32)  # Empty mask

        mutated_graphs = vmap_build_graph(fresh_logits, mutated_wires, fresh_gate_masks)

        # Update the pool with mutated circuits
        updated_pool = self.update(reset_idxs, mutated_graphs, mutated_wires, fresh_logits)

        # Reset gate masks for mutated circuits (all active)
        if self.gate_masks is not None:
            total_gates = self.gate_masks.shape[1]
            fresh_gate_masks = jp.ones((num_reset, total_gates), dtype=jp.float32)
            updated_pool = updated_pool.update(
                reset_idxs, mutated_graphs, mutated_wires, fresh_logits, fresh_gate_masks
            )

        # Increment the reset counter for all elements
        if updated_pool.reset_counter is not None:
            new_counter = updated_pool.reset_counter + 1
            updated_pool = updated_pool.replace(reset_counter=new_counter)

        return updated_pool, avg_steps_reset

    def apply_gate_knockouts_to_fraction(
        self,
        key: Array,
        fraction: float,
        layer_sizes: list[tuple[int, int]],
        number_knockouts: float,
        input_n: int,
        arity: int,
        circuit_hidden_dim: int,
        faulty_value: float = -10.0,
        selection_strategy: str = "uniform",
        combined_weights: tuple[float, float] = (0.5, 0.5),
    ) -> tuple["GraphPool", float]:
        """
        Apply permanent gate knockouts to a fraction of existing pool elements.

        This method applies structural damage where knocked-out gates:
        1. Have their logits set to faulty values (produce zero output)
        2. Cannot receive model updates (permanent damage)
        3. Can still send messages but cannot recover

        Args:
            key: Random key for knockout generation and selection
            fraction: Fraction of pool to apply knockouts to
            layer_sizes: Circuit layer sizes
            number_knockouts: Number of gates to permanently knock out per circuit
            input_n: Number of input nodes
            arity: Number of inputs per gate
            circuit_hidden_dim: Hidden dimension for graphs
            faulty_value: Value for knocked-out gate logits
            selection_strategy: Strategy for selecting circuits to perturb
            combined_weights: Weights for combined selection strategy

        Returns:
            Tuple of (updated_pool, avg_steps_of_perturbed_circuits)
        """
        from boolean_nca_cc.training.pool.structural_perturbation import (
            apply_gate_knockout_to_pool_batch,
            batch_layered_to_flat_mask,
        )
        from boolean_nca_cc.utils.graph_builder import build_graph

        # Split keys for different operations
        selection_key, knockout_key = jax.random.split(key)

        # Get indices of circuits to apply knockouts to
        knockout_idxs, avg_steps_knocked_out = self.get_reset_indices(
            selection_key, fraction, selection_strategy, combined_weights
        )

        num_knockout = len(knockout_idxs)

        # Extract logits and wires for the selected circuits
        selected_logits = jax.tree.map(lambda leaf: leaf[knockout_idxs], self.logits)
        selected_wires = jax.tree.map(lambda leaf: leaf[knockout_idxs], self.wires)

        # Apply permanent knockouts to the selected circuits
        modified_logits, _, layered_knockout_masks = apply_gate_knockout_to_pool_batch(
            knockout_key,
            selected_logits,
            selected_wires,
            layer_sizes,
            number_knockouts,
            faulty_value,
        )

        # Convert layered masks to flat format for pool storage
        flat_knockout_masks = batch_layered_to_flat_mask(layered_knockout_masks)

        # Build new graphs with modified logits and knockout masks
        vmap_build_graph = jax.vmap(
            lambda logit, wires, knockout_mask: build_graph(
                logits=logit,
                wires=wires,
                input_n=input_n,
                arity=arity,
                circuit_hidden_dim=circuit_hidden_dim,
                loss_value=0.0,  # Reset loss after knockout
                update_steps=0,  # Reset update steps after knockout
                gate_knockout_mask=knockout_mask,  # Use new knockout mask parameter
                knockout_strategy="no_receive",  # Knocked-out gates can't receive messages
                faulty_logit_value=faulty_value,
            )
        )
        modified_graphs = vmap_build_graph(modified_logits, selected_wires, flat_knockout_masks)

        # Update the pool with knocked-out circuits
        updated_pool = self.update(
            knockout_idxs, modified_graphs, selected_wires, modified_logits, flat_knockout_masks
        )

        # Increment the reset counter for knocked-out elements
        if updated_pool.reset_counter is not None:
            new_counter = updated_pool.reset_counter + 1
            updated_pool = updated_pool.replace(reset_counter=new_counter)

        return updated_pool, avg_steps_knocked_out

    def apply_logits_perturbations_to_fraction(
        self,
        key: Array,
        fraction: float,
        layer_sizes: list[tuple[int, int]],
        number_perturbations: float,
        input_n: int,
        arity: int,
        circuit_hidden_dim: int,
        perturbation_type: str = "noise",
        noise_scale: float = 1.0,
        faulty_value: float = -10.0,
        selection_strategy: str = "uniform",
        combined_weights: tuple[float, float] = (0.5, 0.5),
    ) -> tuple["GraphPool", float]:
        """
        Apply recoverable logits perturbations to a fraction of existing pool elements.

        This method applies temporary interference where perturbed gates:
        1. Have their logits modified but can still receive model updates
        2. Can potentially recover through message passing
        3. Represent temporary interference rather than permanent damage

        Args:
            key: Random key for perturbation generation and selection
            fraction: Fraction of pool to apply perturbations to
            layer_sizes: Circuit layer sizes
            number_perturbations: Number of gates to perturb per circuit
            input_n: Number of input nodes
            arity: Number of inputs per gate
            circuit_hidden_dim: Hidden dimension for graphs
            perturbation_type: Type of perturbation ("noise", "zero", "negative")
            noise_scale: Scale of noise for "noise" perturbation
            faulty_value: Value for "negative" perturbation
            selection_strategy: Strategy for selecting circuits to perturb
            combined_weights: Weights for combined selection strategy

        Returns:
            Tuple of (updated_pool, avg_steps_of_perturbed_circuits)
        """
        from boolean_nca_cc.training.pool.structural_perturbation import (
            apply_logits_perturbation_to_pool_batch,
            batch_layered_to_flat_mask,
        )
        from boolean_nca_cc.utils.graph_builder import build_graph

        # Split keys for different operations
        selection_key, perturbation_key = jax.random.split(key)

        # Get indices of circuits to apply perturbations to
        perturb_idxs, avg_steps_perturbed = self.get_reset_indices(
            selection_key, fraction, selection_strategy, combined_weights
        )

        num_perturb = len(perturb_idxs)

        # Extract logits and wires for the selected circuits
        selected_logits = jax.tree.map(lambda leaf: leaf[perturb_idxs], self.logits)
        selected_wires = jax.tree.map(lambda leaf: leaf[perturb_idxs], self.wires)

        # Apply recoverable perturbations to the selected circuits
        modified_logits, _, layered_perturbation_masks = apply_logits_perturbation_to_pool_batch(
            perturbation_key,
            selected_logits,
            selected_wires,
            layer_sizes,
            number_perturbations,
            perturbation_type,
            noise_scale,
            faulty_value,
        )

        # Convert layered masks to flat format for pool storage
        flat_perturbation_masks = batch_layered_to_flat_mask(layered_perturbation_masks)

        # Build new graphs with modified logits and perturbation masks
        vmap_build_graph = jax.vmap(
            lambda logit, wires, perturbation_mask: build_graph(
                logits=logit,
                wires=wires,
                input_n=input_n,
                arity=arity,
                circuit_hidden_dim=circuit_hidden_dim,
                loss_value=0.0,  # Reset loss after perturbation
                update_steps=0,  # Reset update steps after perturbation
                perturbation_mask=perturbation_mask,  # Use new perturbation mask parameter
                knockout_strategy="untouched",  # Perturbed gates can still participate in message passing
                faulty_logit_value=faulty_value,
            )
        )
        modified_graphs = vmap_build_graph(modified_logits, selected_wires, flat_perturbation_masks)

        # For perturbations, we don't update the gate_masks field since these are recoverable
        # The perturbation information is tracked in the graph nodes themselves
        updated_pool = self.update(perturb_idxs, modified_graphs, selected_wires, modified_logits)

        # Increment the reset counter for perturbed elements
        if updated_pool.reset_counter is not None:
            new_counter = updated_pool.reset_counter + 1
            updated_pool = updated_pool.replace(reset_counter=new_counter)

        return updated_pool, avg_steps_perturbed


def initialize_graph_pool(
    rng: jax.random.PRNGKey,
    layer_sizes: list[tuple[int, int]],
    pool_size: int,
    input_n: int,
    arity: int = 2,
    circuit_hidden_dim: int = 16,
    loss_value: float = 0.0,
    wiring_mode: str = "random",
    initial_diversity: int = 1,
    initialize_gate_masks: bool = True,
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
        wiring_mode: Mode for generating wirings ("random", "fixed", or "genetic")
                    Note: "genetic" mode initializes the same as "fixed" mode -
                    genetic mutations are applied during pool resets
        initial_diversity: Number of different initial wirings to start with.
                          Only used in "fixed" and "genetic" modes.
                          - 1: All circuits start with identical wiring (original behavior)
                          - N: Pool starts with N different wirings, each repeated pool_size//N times
                          - If N >= pool_size: Each circuit gets a unique wiring (same as "random" mode)
        initialize_gate_masks: Whether to initialize gate masks (all active initially)

    Returns:
        Initialized GraphPool
    """
    # Generate circuit wirings based on wiring mode
    if wiring_mode in ["fixed", "genetic"]:
        # Clamp initial_diversity to valid range
        effective_diversity = jp.clip(initial_diversity, 1, pool_size)

        if effective_diversity == 1:
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
        elif effective_diversity >= pool_size:
            # Each circuit gets a unique wiring (same as random mode)
            # We however need to make sure we are not always generating the same wirings
            rngs = jax.random.split(rng, pool_size)
            vmap_gen_circuit = jax.vmap(lambda rng: gen_circuit(rng, layer_sizes, arity=arity))
            all_wires, all_logits = vmap_gen_circuit(rngs)
        else:
            # Generate N different wirings and repeat them across the pool
            diversity_rngs = jax.random.split(rng, effective_diversity)
            vmap_gen_circuit = jax.vmap(lambda rng: gen_circuit(rng, layer_sizes, arity=arity))
            diverse_wires, diverse_logits = vmap_gen_circuit(diversity_rngs)

            # Calculate how many times to repeat each diverse wiring
            base_repeats = pool_size // effective_diversity
            extra_repeats = pool_size % effective_diversity

            # Create repeat counts: first 'extra_repeats' get one extra copy
            repeat_counts = jp.concatenate(
                [
                    jp.full(extra_repeats, base_repeats + 1),
                    jp.full(effective_diversity - extra_repeats, base_repeats),
                ]
            )

            # Repeat each diverse wiring according to repeat_counts
            all_wires = []
            all_logits = []

            for layer_idx in range(len(diverse_wires)):
                layer_wires = []
                layer_logits = []

                for diversity_idx in range(effective_diversity):
                    n_repeats = repeat_counts[diversity_idx]

                    # Repeat this wiring n_repeats times
                    repeated_wire = jp.repeat(
                        diverse_wires[layer_idx][diversity_idx : diversity_idx + 1],
                        n_repeats,
                        axis=0,
                    )
                    repeated_logit = jp.repeat(
                        diverse_logits[layer_idx][diversity_idx : diversity_idx + 1],
                        n_repeats,
                        axis=0,
                    )

                    layer_wires.append(repeated_wire)
                    layer_logits.append(repeated_logit)

                # Concatenate all repeats for this layer
                all_wires.append(jp.concatenate(layer_wires, axis=0))
                all_logits.append(jp.concatenate(layer_logits, axis=0))
    else:  # wiring_mode == "random"
        # In random mode, generate different wirings for each circuit (ignore initial_diversity)
        rngs = jax.random.split(rng, pool_size)
        vmap_gen_circuit = jax.vmap(lambda rng: gen_circuit(rng, layer_sizes, arity=arity))
        all_wires, all_logits = vmap_gen_circuit(rngs)

    # Initialize gate masks first
    if initialize_gate_masks:
        # Calculate total number of gates across all layers
        total_gates = sum(gate_n for gate_n, _group_size in layer_sizes)
        # Initialize all gate masks to 1.0 (active)
        all_gate_masks = jp.ones((pool_size, total_gates), dtype=jp.float32)
    else:
        all_gate_masks = None

    # Generate graphs in parallel using vmap
    # Create globals with both loss value and update steps counter (initialized to 0)
    # The globals structure will be [loss_value, update_steps]
    if initialize_gate_masks:
        vmap_build_graph = jax.vmap(
            lambda logit, wires, gate_knockout_mask: build_graph(
                logits=logit,
                wires=wires,
                input_n=input_n,
                arity=arity,
                circuit_hidden_dim=circuit_hidden_dim,
                loss_value=loss_value,
                update_steps=0,  # Initialize update steps counter to 0
                gate_knockout_mask=gate_knockout_mask,
                knockout_strategy="untouched",  # Initial circuits have no knockouts
                faulty_logit_value=-10.0,  # Default faulty value
            )
        )
        graphs = vmap_build_graph(all_logits, all_wires, all_gate_masks)
    else:
        vmap_build_graph = jax.vmap(
            lambda logit, wires: build_graph(
                logits=logit,
                wires=wires,
                input_n=input_n,
                arity=arity,
                circuit_hidden_dim=circuit_hidden_dim,
                loss_value=loss_value,
                update_steps=0,  # Initialize update steps counter to 0
                faulty_logit_value=-10.0,  # Default faulty value (not used without gate masks)
            )
        )
        graphs = vmap_build_graph(all_logits, all_wires)

    # Initialize reset counter
    reset_counter = jp.zeros(pool_size, dtype=jp.int32)

    return GraphPool.create(graphs, all_wires, all_logits, all_gate_masks, reset_counter)
