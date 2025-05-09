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
from .perturbation import zero_luts
from .perturbation import zero_luts

PyTree = Any


class GraphPool(struct.PyTreeNode):
    """
    Pool class for graph-based circuit NCA.

    Stores a single batched jraph.GraphsTuple and allows
    sampling and updating batches. Also tracks wires and logits in parallel
    for computing functional circuit loss without redundant extraction.
    """

    # Fields without default values must come first
    # Fields without default values must come first
    size: int = struct.field(pytree_node=False)
    graphs: jraph.GraphsTuple
    # Added fields for graph reconstruction, non-default
    input_n: int = struct.field(pytree_node=False)
    arity: int = struct.field(pytree_node=False)
    hidden_dim: int = struct.field(pytree_node=False)
    initial_loss_value: float = struct.field(pytree_node=False)

    # Fields with default values
    # wires is a list of wire matrices with shape [pool_size, ...wire_shape...]
    wires: PyTree = None
    # logits is a list of weight matrices corresponding to each graph
    logits: PyTree = None
    # Reset counter to track which elements were reset recently
    reset_counter: Optional[Array] = None
    # New field to track if gate_knockout has been applied to a slot
    gate_knockout_applied: Optional[Array] = None # Boolean array, True if knockout applied

    @classmethod
    def create(
        cls,
        batched_graphs: jraph.GraphsTuple,
        wires: PyTree = None,
        logits: PyTree = None,
        reset_counter: Optional[Array] = None,
        # Added parameters for graph reconstruction capabilities
        input_n: int = 0, # Default for safety, should be provided
        arity: int = 0,   # Default for safety
        hidden_dim: int = 0, # Default for safety
        initial_loss_value: float = 0.0, # Default for safety
    ) -> "GraphPool":
        """
        Create a new GraphPool instance from a batched GraphsTuple.

        Args:
            batched_graphs: A jraph.GraphsTuple where each leaf node
                            has a leading batch dimension (pool_size).
            wires: The wire matrices corresponding to each graph.
            logits: The logit matrices corresponding to each graph.
            reset_counter: Optional counter to track reset operations.
            input_n: Number of inputs to the circuit.
            arity: Number of inputs per gate.
            hidden_dim: Dimension of hidden features.
            initial_loss_value: Initial loss value for graph globals when reset.
            input_n: Number of inputs to the circuit.
            arity: Number of inputs per gate.
            hidden_dim: Dimension of hidden features.
            initial_loss_value: Initial loss value for graph globals when reset.

        Returns:
            A new GraphPool instance.
        """
        # Determine size from the n_node field, assuming it's always present and batched.
        if batched_graphs.n_node is None:
            raise ValueError(
                "batched_graphs.n_node cannot be None for GraphPool.create"
            )
        size = batched_graphs.n_node.shape[0]

        # Initialize reset counter if not provided
        if reset_counter is None:
            reset_counter = jp.zeros(size, dtype=jp.int32)
        
        # Initialize gate_knockout_applied field
        gate_knockout_applied_init = jp.zeros(size, dtype=jp.bool_)

        return cls(
            size=size,
            graphs=batched_graphs,
            # Pass through new fields
            input_n=input_n,
            arity=arity,
            hidden_dim=hidden_dim,
            initial_loss_value=initial_loss_value,
            # Original fields
            wires=wires,
            logits=logits,
            reset_counter=reset_counter,
            gate_knockout_applied=gate_knockout_applied_init,
        )

    @jax.jit
    def update(
        self,
        idxs: Array,
        batch_of_graphs: jraph.GraphsTuple,
        batch_of_wires: PyTree = None,
        batch_of_logits: PyTree = None,
        gate_knockout_status_for_idxs: Optional[Array] = None,
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
            gate_knockout_status_for_idxs: Optional boolean Array. If provided, sets the
                                           gate_knockout_applied flag for the updated idxs.

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

        # Update gate_knockout_applied status if provided
        updated_gate_knockout_applied = self.gate_knockout_applied
        if gate_knockout_status_for_idxs is not None and self.gate_knockout_applied is not None:
            updated_gate_knockout_applied = self.gate_knockout_applied.at[idxs].set(
                gate_knockout_status_for_idxs
            )

        return self.replace(
            graphs=updated_graphs_data,
            wires=updated_wires,
            logits=updated_logits,
            reset_counter=updated_reset_counter,
            gate_knockout_applied=updated_gate_knockout_applied,
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
        reset_pool = self.update(
            reset_idxs, 
            reset_graphs, 
            reset_wires, 
            reset_logits,
            gate_knockout_status_for_idxs=jp.full_like(reset_idxs, False, dtype=jp.bool_) # New circuits are not knocked out
            )

        # Increment the reset counter for all elements
        if reset_pool.reset_counter is not None:
            new_counter = reset_pool.reset_counter + 1
            reset_pool = reset_pool.replace(reset_counter=new_counter)

        return reset_pool, avg_steps_reset

    def gate_knockout(
        self,
        key: Array,
        fraction: float,
        lut_damage_prob: float,
        reset_strategy: str = "uniform",
        combined_weights: Tuple[float, float] = (0.5, 0.5),
    ) -> "GraphPool":
        """
        Selects a fraction of circuits from the pool that have not undergone gate knockout yet,
        applies LUT zeroing to them, rebuilds their graph representations with reset globals,
        and updates them in the pool. Marks these circuits as having had gate knockout applied.

        Args:
            key: Random key for selection and perturbation.
            fraction: Fraction of the pool to apply LUT zeroing to (between 0 and 1).
            lut_damage_prob: Probability of zeroing out each LUT in the selected circuits.
            reset_strategy: Strategy for selecting circuits: "uniform", "steps_biased",
                            "loss_biased", or "combined".
            combined_weights: Tuple of weights (loss_weight, steps_weight) for the
                              "combined" strategy.

        Returns:
            A new GraphPool instance with the modified circuits.
        """
        if self.size == 0:
            return self
        
        if self.wires is None or self.logits is None or self.gate_knockout_applied is None:
            # Cannot apply if essential components are missing
            print("Warning: Cannot apply gate_knockout, essential components (wires, logits, or gate_knockout_applied flag) missing.")
            return self

        # Identify eligible circuits (those not yet knocked out)
        eligible_indices = jp.where(jp.logical_not(self.gate_knockout_applied))[0]
        num_eligible = eligible_indices.shape[0]

        if num_eligible == 0:
            print("Warning: No eligible circuits for gate_knockout (all may have been knocked out already).")
            return self

        # Determine how many to target based on fraction of total pool size, capped by eligible count
        num_to_target_overall = jp.round(fraction * self.size).astype(jp.int32)
        num_to_actually_select = jp.minimum(num_to_target_overall, num_eligible)
        
        if num_to_actually_select == 0:
            return self

        key_select, key_zero_luts = jax.random.split(key)

        # Select elements from ELIGIBLE indices based on the reset strategy
        if reset_strategy == "uniform":
            selected_indices_in_eligible_array = jax.random.choice(
                key_select, num_eligible, shape=(num_to_actually_select,), replace=False
            )
        elif reset_strategy == "steps_biased":
            if self.graphs.globals is None:
                selected_indices_in_eligible_array = jax.random.choice(
                    key_select, num_eligible, shape=(num_to_actually_select,), replace=False
                )
            else:
                update_steps_eligible = self.graphs.globals[eligible_indices, 1]
                probs_eligible = update_steps_eligible + 1.0
                # Handle case where all eligible update_steps are zero, leading to sum(probs_eligible) = num_eligible
                sum_probs = jp.sum(probs_eligible)
                probs_eligible = jp.where(sum_probs == 0, jp.ones_like(probs_eligible) / num_eligible, probs_eligible / sum_probs)
                selected_indices_in_eligible_array = jax.random.choice(
                    key_select, num_eligible, shape=(num_to_actually_select,), replace=False, p=probs_eligible
                )
        elif reset_strategy == "loss_biased":
            if self.graphs.globals is None:
                selected_indices_in_eligible_array = jax.random.choice(
                    key_select, num_eligible, shape=(num_to_actually_select,), replace=False
                )
            else:
                loss_values_eligible = self.graphs.globals[eligible_indices, 0]
                probs_eligible = loss_values_eligible + 1e-6
                probs_eligible = jp.clip(probs_eligible, 0.0, 100.0)
                sum_probs = jp.sum(probs_eligible)
                probs_eligible = jp.where(sum_probs == 0, jp.ones_like(probs_eligible) / num_eligible, probs_eligible / sum_probs)
                selected_indices_in_eligible_array = jax.random.choice(
                    key_select, num_eligible, shape=(num_to_actually_select,), replace=False, p=probs_eligible
                )
        elif reset_strategy == "combined":
            if self.graphs.globals is None:
                selected_indices_in_eligible_array = jax.random.choice(
                    key_select, num_eligible, shape=(num_to_actually_select,), replace=False
                )
            else:
                loss_values_eligible = self.graphs.globals[eligible_indices, 0]
                update_steps_eligible = self.graphs.globals[eligible_indices, 1]
                loss_weight, steps_weight = combined_weights
                
                min_loss_eligible = jp.min(loss_values_eligible)
                max_loss_eligible = jp.max(loss_values_eligible)
                min_steps_eligible = jp.min(update_steps_eligible)
                max_steps_eligible = jp.max(update_steps_eligible)

                loss_scores_eligible = (loss_values_eligible - min_loss_eligible) / (
                    max_loss_eligible - min_loss_eligible + 1e-6
                )
                step_scores_eligible = (update_steps_eligible - min_steps_eligible) / (
                    max_steps_eligible - min_steps_eligible + 1e-6
                )
                combined_scores_eligible = loss_weight * loss_scores_eligible + steps_weight * step_scores_eligible
                probs_eligible = combined_scores_eligible + 1e-6
                sum_probs = jp.sum(probs_eligible)
                probs_eligible = jp.where(sum_probs == 0, jp.ones_like(probs_eligible) / num_eligible, probs_eligible / sum_probs)
                selected_indices_in_eligible_array = jax.random.choice(
                    key_select, num_eligible, shape=(num_to_actually_select,), replace=False, p=probs_eligible
                )
        else:
            raise ValueError(
                f"Unknown reset_strategy: {reset_strategy}. "
                f"Must be 'uniform', 'steps_biased', 'loss_biased', or 'combined'."
            )
        
        # Map selected indices from the eligible array back to original pool indices
        target_idxs = eligible_indices[selected_indices_in_eligible_array]
        actual_num_to_zero = target_idxs.shape[0]

        if actual_num_to_zero == 0: # Should be redundant due to earlier checks, but safe
            return self

        selected_wires_batched = jax.tree.map(lambda x: x[target_idxs], self.wires)
        selected_logits_batched = jax.tree.map(lambda x: x[target_idxs], self.logits)

        keys_for_zeroing = jax.random.split(key_zero_luts, actual_num_to_zero)
        
        vmap_zero_luts = jax.vmap(zero_luts, in_axes=(0, 0, 0, None))
        _, perturbed_logits_batched = vmap_zero_luts(
            keys_for_zeroing, selected_wires_batched, selected_logits_batched, lut_damage_prob
        )

        rebuilt_graphs_batch = jax.vmap(
            lambda l_pytree, w_pytree, loss_val, upd_steps: build_graph(
                logits=l_pytree,
                wires=w_pytree,
                input_n=self.input_n,
                arity=self.arity,
                hidden_dim=self.hidden_dim,
                loss_value=loss_val,
                update_steps=upd_steps
            ),
            in_axes=(0, 0, 0, 0) 
        )(perturbed_logits_batched, selected_wires_batched, 
          jp.full((actual_num_to_zero,), self.initial_loss_value), 
          jp.zeros((actual_num_to_zero,), dtype=jp.int32))

        updated_pool = self.update(
            target_idxs,
            rebuilt_graphs_batch,
            selected_wires_batched, 
            perturbed_logits_batched,
            gate_knockout_status_for_idxs=jp.full((actual_num_to_zero,), True, dtype=jp.bool_) # Set flag to True
        )

        return updated_pool

    def zero_luts_for_fraction(
        self,
        key: Array,
        fraction: float,
        lut_damage_prob: float,
        selection_strategy: str = "uniform", # Renamed from reset_strategy for clarity
        combined_weights: Tuple[float, float] = (0.5, 0.5),
    ) -> "GraphPool":
        """
        Selects a fraction of circuits, applies LUT damage (zeros logits),
        rebuilds their graph representations preserving their original globals (loss, update_steps),
        and updates them in the pool. The reset_counter for these circuits will be set to 0.

        Args:
            key: Random key for selection and perturbation.
            fraction: Fraction of the pool to apply LUT damage to.
            lut_damage_prob: Probability of zeroing out each LUT in selected circuits.
            selection_strategy: Strategy for selecting circuits: "uniform", "steps_biased",
                                "loss_biased", or "combined".
            combined_weights: Tuple of weights for the "combined" strategy.

        Returns:
            A new GraphPool instance with the modified circuits.
        """
        if self.size == 0:
            return self
        
        if self.wires is None or self.logits is None or self.graphs.globals is None:
            # Cannot apply if essential components are missing
            print("Warning: Cannot apply LUT damage, essential components (wires, logits, or globals) missing.")
            return self

        num_to_damage = jp.round(fraction * self.size).astype(jp.int32)
        if num_to_damage == 0:
            return self

        key_select, key_zero_luts = jax.random.split(key)

        # Select elements to apply LUT damage based on the selection_strategy
        # (Logic copied and adapted from zero_out_luts_for_fraction/reset_fraction)
        if selection_strategy == "uniform":
            target_idxs = jax.random.choice(
                key_select, self.size, shape=(num_to_damage,), replace=False
            )
        elif selection_strategy == "steps_biased":
            update_steps = self.graphs.globals[..., 1]
            probs = update_steps + 1.0
            probs = probs / jp.sum(probs)
            target_idxs = jax.random.choice(
                key_select, self.size, shape=(num_to_damage,), replace=False, p=probs
            )
        elif selection_strategy == "loss_biased":
            loss_values = self.graphs.globals[..., 0]
            probs = loss_values + 1e-6
            probs = jp.clip(probs, 0.0, 100.0) # Clip extreme loss values for stability
            probs = probs / jp.sum(probs)
            target_idxs = jax.random.choice(
                key_select, self.size, shape=(num_to_damage,), replace=False, p=probs
            )
        elif selection_strategy == "combined":
            loss_values = self.graphs.globals[..., 0]
            update_steps = self.graphs.globals[..., 1]
            loss_weight, steps_weight = combined_weights
            loss_scores = (loss_values - jp.min(loss_values)) / (
                jp.max(loss_values) - jp.min(loss_values) + 1e-6
            )
            step_scores = (update_steps - jp.min(update_steps)) / (
                jp.max(update_steps) - jp.min(update_steps) + 1e-6
            )
            combined_scores = loss_weight * loss_scores + steps_weight * step_scores
            probs = combined_scores + 1e-6
            probs = probs / jp.sum(probs)
            target_idxs = jax.random.choice(
                key_select, self.size, shape=(num_to_damage,), replace=False, p=probs
            )
        else:
            raise ValueError(
                f"Unknown selection_strategy: {selection_strategy}. "
                f"Must be 'uniform', 'steps_biased', 'loss_biased', or 'combined'."
            )
        
        actual_num_to_damage = target_idxs.shape[0]
        if actual_num_to_damage == 0:
            return self

        # Get wires, logits, and current globals for the selected circuits
        selected_wires_batched = jax.tree.map(lambda x: x[target_idxs], self.wires)
        selected_logits_batched = jax.tree.map(lambda x: x[target_idxs], self.logits)
        current_globals_batched = self.graphs.globals[target_idxs] # Shape: (actual_num_to_damage, global_features)

        # Apply zero_luts perturbation
        keys_for_zeroing = jax.random.split(key_zero_luts, actual_num_to_damage)
        
        vmap_zero_luts = jax.vmap(zero_luts, in_axes=(0, 0, 0, None))
        _, perturbed_logits_batched = vmap_zero_luts(
            keys_for_zeroing, selected_wires_batched, selected_logits_batched, lut_damage_prob
        )

        # Rebuild graphs for the perturbed circuits, preserving their original globals
        preserved_loss_values = current_globals_batched[..., 0]
        preserved_update_steps = current_globals_batched[..., 1].astype(jp.int32)

        rebuilt_graphs_batch = jax.vmap(
            lambda l_pytree, w_pytree, loss_val, upd_steps: build_graph(
                logits=l_pytree,
                wires=w_pytree,
                input_n=self.input_n,
                arity=self.arity,
                hidden_dim=self.hidden_dim,
                loss_value=loss_val, # Use preserved loss
                update_steps=upd_steps # Use preserved update steps
            ),
            in_axes=(0, 0, 0, 0) 
        )(perturbed_logits_batched, selected_wires_batched, 
          preserved_loss_values, preserved_update_steps)

        # Update the pool with the modified circuits
        # The update method will set reset_counter for target_idxs to 0.
        updated_pool = self.update(
            target_idxs,
            rebuilt_graphs_batch,
            selected_wires_batched,  # Wires are unchanged by zero_luts
            perturbed_logits_batched
        )

        return updated_pool


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
        single_wires, single_logits = gen_circuit(rng, layer_sizes, arity=arity)
        # print(f"INIT WITH RNG {rng}")

        print(f"SINGLE WIRES {single_wires}")

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
            lambda rng_single: gen_circuit(rng_single, layer_sizes, arity=arity)
        )
        all_wires, all_logits = vmap_gen_circuit(rngs)

    # Generate graphs in parallel using vmap
    # Create globals with both loss value and update steps counter (initialized to 0)
    # The globals structure will be [loss_value, update_steps]
    vmap_build_graph = jax.vmap(
        lambda logit, wires_single: build_graph(
            logits=logit,
            wires=wires_single,
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

    return GraphPool.create(
        graphs,
        all_wires,
        all_logits,
        reset_counter,
        # Pass the necessary parameters for graph reconstruction
        input_n=input_n,
        arity=arity,
        hidden_dim=hidden_dim,
        initial_loss_value=loss_value
    )
