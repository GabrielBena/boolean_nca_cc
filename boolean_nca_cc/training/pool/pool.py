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
import logging

# Setup logging
log = logging.getLogger(__name__)

from boolean_nca_cc.circuits.model import gen_circuit
from boolean_nca_cc.utils.graph_builder import build_graph
from boolean_nca_cc.utils.extraction import extract_logits_from_graph
from boolean_nca_cc.training.pool.perturbation import (
    create_reproducible_knockout_pattern,
    sample_seu_gates,
    build_flip_masks_from_indices,
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
            wires=wires,
            logits=logits,
            reset_counter=reset_counter,
            knockout_patterns=knockout_patterns,
        )

    @jax.jit
    def update(
        self,
        idxs: Array,
        batch_of_graphs: jraph.GraphsTuple,
        batch_of_wires: PyTree = None,
        batch_of_logits: PyTree = None,
        batch_of_knockout_patterns: Optional[Array] = None,
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
            wires=updated_wires,
            logits=updated_logits,
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
                - sampled_wires: Corresponding wires.
                - sampled_logits: Corresponding logits.
                - sampled_knockout_patterns: Corresponding knockout patterns.
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

        # Sample knockout patterns
        sampled_knockout_patterns = jax.tree.map(_safe_slice_leaf, self.knockout_patterns)

        return idxs, sampled_graphs, sampled_wires, sampled_logits, sampled_knockout_patterns

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
        new_wires: Optional[PyTree] = None,
        new_logits: Optional[PyTree] = None,
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
            new_wires: Fresh wires to use for reset
            new_logits: Fresh logits to use for reset
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

        # Sample wires if provided
        reset_wires = None
        if new_wires is not None:
            reset_wires = jax.tree.map(lambda leaf: leaf[sample_idxs], new_wires)

        # Sample logits if provided
        reset_logits = None
        if new_logits is not None:
            reset_logits = jax.tree.map(lambda leaf: leaf[sample_idxs], new_logits)

        # Sample knockout patterns if provided
        reset_knockout_patterns = None
        if new_knockout_patterns is not None:
            reset_knockout_patterns = new_knockout_patterns[sample_idxs]

        # Update the pool with reset elements
        reset_pool = self.update(
            reset_idxs,
            reset_graphs,
            reset_wires,
            reset_logits,
            reset_knockout_patterns,
        )

        # Increment the reset counter for all elements
        if reset_pool.reset_counter is not None:
            new_counter = reset_pool.reset_counter + 1
            reset_pool = reset_pool.replace(reset_counter=new_counter)

        return reset_pool, avg_steps_reset

    def apply_seu(
        self,
        idxs: Array,
        flip_masks_per_circuit: List[jp.ndarray],  # list per layer: (k, group_n, group_size, 2^arity)
        layer_sizes: List[Tuple[int, int]],
        input_n: int,
        arity: int,
        circuit_hidden_dim: int,
    ) -> "GraphPool":
        """
        Apply SEU flips via dense masks to selected circuits, rebuild graphs preserving globals,
        and persist mutated logits and graphs into the pool.

        Args:
            idxs: 1D indices of circuits to update.
            flip_masks_per_circuit: List of per-layer boolean masks stacked over batch idxs.
            layer_sizes: Circuit layer sizes (unused here; kept for signature parity and checks).
            input_n: Number of input nodes.
            arity: Gate arity.
            circuit_hidden_dim: Node feature dim for graph builder.

        Returns:
            Updated GraphPool instance.
        """
        # Slice current logits and wires for the selected circuits
        batch_logits = jax.tree.map(lambda leaf: leaf[idxs], self.logits)
        batch_wires = jax.tree.map(lambda leaf: leaf[idxs], self.wires)

        # Validate mask structure matches logits structure
        if len(batch_logits) != len(flip_masks_per_circuit):
            raise ValueError("flip_masks_per_circuit length must equal number of layers in logits.")

        mutated_logits = []
        total_flips = 0
        for i, (layer_logits, layer_mask) in enumerate(zip(batch_logits, flip_masks_per_circuit)):
            if layer_logits.shape != layer_mask.shape:
                raise ValueError("Flip mask batch shape must match logits batch shape for each layer.")
            factors = jp.where(layer_mask, -1.0, 1.0)
            mutated_layer = layer_logits * factors
            mutated_logits.append(mutated_layer)
            
            layer_flips = jp.sum(layer_mask).astype(int)
            total_flips += layer_flips

        # Preserve globals for each selected circuit
        old_globals = self.graphs.globals[idxs]
        loss_values = old_globals[:, 0]
        update_steps = old_globals[:, 1]

        # Build graphs per circuit with preserved globals
        vmap_build_graph = jax.vmap(
            lambda lgt, wr, lv, us: build_graph(
                logits=lgt,
                wires=wr,
                input_n=input_n,
                arity=arity,
                circuit_hidden_dim=circuit_hidden_dim,
                loss_value=lv,
                update_steps=us,
            ),
            in_axes=(0, 0, 0, 0),
        )
        new_graphs = vmap_build_graph(mutated_logits, batch_wires, loss_values, update_steps)

        # Commit updates to pool (graphs and logits only)
        updated_pool = self.update(
            idxs=idxs,
            batch_of_graphs=new_graphs,
            batch_of_logits=mutated_logits,
        )

        return updated_pool

    @jax.jit
    def apply_knockouts(
        self,
        idxs: Array,
        new_knockout_patterns: Array,
    ) -> "GraphPool":
        """
        Apply new knockout patterns at the specified indices and return an updated pool.

        Only `knockout_patterns` are mutated; graphs, wires, logits, and reset_counter
        remain unchanged.

        Args:
            idxs: 1D indices in the pool to update.
            new_knockout_patterns: Boolean knockout masks aligned with `idxs`.

        Returns:
            Updated GraphPool with modified `knockout_patterns`.
        """
        if self.knockout_patterns is None:
            # Nothing to update if knockout storage is not initialized
            return self

        updated_knockout_patterns = self.knockout_patterns.at[idxs].set(
            new_knockout_patterns
        )

        return self.replace(knockout_patterns=updated_knockout_patterns)

    # @partial(
    #     jax.jit,
    #     static_argnames=(
    #         "layer_sizes",
    #         "selection_strategy",
    #         "combined_weights",
    #     ),
    # )
    def damage_fraction(
        self,
        key: Array,
        fraction: float,
        layer_sizes: List[Tuple[int, int]],
        damage_prob: float,
        selection_strategy: str = "uniform",
        combined_weights: Tuple[float, float] = (0.5, 0.5),
        knockout_vocabulary: Optional[Array] = None,
        min_pool_updates: Optional[int] = None,
        max_pool_updates: Optional[int] = None,
    ) -> Tuple["GraphPool", Array]:
        """
        Apply sample-based damage (persistent knockouts) to a fraction of pool entries.

        Only `knockout_patterns` are updated for the selected entries. The graphs, wires,
        logits, and reset_counter remain unchanged. Selection mirrors reset semantics via
        `get_reset_indices`.

        Args:
            key: Random key for selection and pattern generation.
            fraction: Fraction of pool to damage (0..1). Minimum of 1 entry affected.
            layer_sizes: Circuit layer sizes used to shape knockout patterns.
            damage_prob: Probability of knockout per node when generating new patterns.
            selection_strategy: One of {"uniform","steps_biased","loss_biased","combined"}.
            combined_weights: Weights for combined selection strategy as (loss, steps).
            knockout_vocabulary: Optional [V, num_nodes] boolean masks; if provided,
                                 sample with replacement from the vocabulary.
            min_pool_updates: Optional minimum pool updates before circuit can be damaged.
            max_pool_updates: Optional maximum pool updates before circuit becomes too fragile.

        Returns:
            Tuple of (updated_pool, damaged_indices).
        """
        # Split key for selection and downstream sampling/generation
        selection_key, op_key = jax.random.split(key)

        # Reuse reset index selection semantics with pool update filtering for damage mode
        damaged_idxs, _avg_steps_unused = self.get_reset_indices(
            selection_key, fraction, selection_strategy, combined_weights, 
            invert_loss=True, 
            min_pool_updates_for_damage=min_pool_updates,
            max_pool_updates_for_damage=max_pool_updates
        )
        
        # Log the pool update filtering application
        log.info(f"Damage selection applied pool update filtering: circuits with {min_pool_updates}-{max_pool_updates} accumulated pool updates eligible")

        num_damaged = damaged_idxs.shape[0]

        # Early out if nothing to do (should not happen due to floor=1, but guard anyway)
        def _no_op_fn(_: Array) -> Tuple["GraphPool", Array]:
            return self, damaged_idxs

        def _apply_damage(_: Array) -> Tuple["GraphPool", Array]:
            # Prepare knockout patterns, either from provided vocabulary or by generation
            if knockout_vocabulary is not None:
                vocab_size = knockout_vocabulary.shape[0]
                sample_keys = op_key  # reuse
                vocab_indices = jax.random.choice(
                    sample_keys, vocab_size, shape=(num_damaged,), replace=True
                )
                new_patterns = knockout_vocabulary[vocab_indices]
            else:
                # Generate fresh patterns
                pattern_keys = jax.random.split(op_key, num_damaged)
                vmapped_pattern_creator = jax.vmap(
                    lambda k: create_reproducible_knockout_pattern(
                        key=k,
                        layer_sizes=layer_sizes,
                        damage_prob=damage_prob,
                    )
                )
                new_patterns = vmapped_pattern_creator(pattern_keys)

            # Apply into pool
            updated_pool = self.apply_knockouts(damaged_idxs, new_patterns)
            return updated_pool, damaged_idxs

        # Use a conditional to keep shapes consistent under jit (though floor=1 normally)
        updated_pool, out_idxs = jax.lax.cond(
            num_damaged > 0, _apply_damage, _no_op_fn, damaged_idxs
        )

        return updated_pool, out_idxs

    def get_reset_indices(
        self,
        key: Array,
        fraction: float,
        reset_strategy: str = "uniform",
        combined_weights: Tuple[float, float] = (0.5, 0.5),
        invert_loss: bool = False,  # For damage mode
        max_pool_updates_for_damage: Optional[int] = None,  # Max pool updates for damage mode only
        min_pool_updates_for_damage: Optional[int] = None,  # Min pool updates for damage mode only
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

                # Invert if requested (for damage mode)
                if invert_loss:
                    probs = 1.0 - probs
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
                
                # Invert loss component if requested (for damage mode)
                if invert_loss:
                    # Simply invert the loss scores for damage mode
                    loss_scores = 1.0 - loss_scores
                    combined_scores = loss_weight * loss_scores + steps_weight * step_scores
                    combined_scores = combined_scores / jp.sum(combined_scores)
                else:
                    combined_scores = combined_scores / jp.sum(combined_scores)
                
                probs = combined_scores

                # Sample indices based on combined probabilities
                reset_idxs = jax.random.choice(
                    key, self.size, shape=(num_reset,), replace=False, p=probs
                )
        else:
            raise ValueError(
                f"Unknown reset_strategy: {reset_strategy}. "
                f"Must be 'uniform', 'steps_biased', 'loss_biased', or 'combined'."
            )

        # Apply damage mode pool update filtering if specified
        if (max_pool_updates_for_damage is not None or min_pool_updates_for_damage is not None) and self.graphs.globals is not None:
            update_steps = self.graphs.globals[..., 1]
            
            # Build eligibility mask
            eligible_mask = jp.ones(self.size, dtype=jp.bool_)
            
            if max_pool_updates_for_damage is not None:
                eligible_mask = eligible_mask & (update_steps < max_pool_updates_for_damage)
            
            if min_pool_updates_for_damage is not None:
                eligible_mask = eligible_mask & (update_steps >= min_pool_updates_for_damage)
            
            if not jp.any(eligible_mask):
                # Fallback to uniform selection if no circuits meet criteria
                log.warning(f"No circuits meet damage criteria: min_pool_updates={min_pool_updates_for_damage}, max_pool_updates={max_pool_updates_for_damage}")
                reset_idxs = jax.random.choice(
                    key, self.size, shape=(num_reset,), replace=False
                )
            else:
                # Only select from eligible circuits
                eligible_indices = jp.where(eligible_mask)[0]
                if len(eligible_indices) < num_reset:
                    # If not enough eligible circuits, take all available
                    reset_idxs = eligible_indices
                else:
                    # Sample from eligible circuits
                    reset_idxs = jax.random.choice(
                        key, eligible_indices, shape=(num_reset,), replace=False
                    )

        # Calculate average update steps of graphs being reset
        avg_steps_reset = self.get_average_update_steps_for_indices(reset_idxs)

        return reset_idxs, avg_steps_reset

    def seu_fraction(
        self,
        key: Array,
        fraction: float,
        layer_sizes: List[Tuple[int, int]],
        gates_per_circuit: int,
        flips_per_gate: int,
        selection_strategy: str,
        gate_selection: str,
        input_n: int,
        arity: int,
        circuit_hidden_dim: int,
        greedy_ordered_indices: Optional[List[int]] = None,
    ) -> Tuple["GraphPool", Array]:
        """
        Apply SEU to a fraction of circuits by sampling gates and building flip masks.

        Returns updated pool and indices of modified circuits.
        """
        
        # Select circuits to modify using existing pool selection semantics
        sel_key, op_key = jax.random.split(key)
        idxs, _avg_steps_unused = self.get_reset_indices(
            sel_key, fraction, reset_strategy=selection_strategy
        )

        num_selected = idxs.shape[0]
        if num_selected == 0:
            return self, idxs

        # For each selected circuit, sample gates and build masks
        gate_keys = jax.random.split(op_key, num_selected)

        def build_masks_for_one(k):
            # sample gates
            g_sel = sample_seu_gates(
                k, layer_sizes[1:], gates_per_circuit, strategy=gate_selection, ordered_indices=greedy_ordered_indices
            )
            # construct masks per layer matching logits shapes
            masks = build_flip_masks_from_indices(layer_sizes[1:], g_sel, flips_per_gate=flips_per_gate, arity=arity, key=k)
            return masks

        # masks_per_circuit is list-per-layer shaped (batch, ...layer_shape...)
        masks_list = [
            jax.tree.map(lambda *xs: jp.stack(xs), *[build_masks_for_one(k) for k in gate_keys])
        ][0]

        
        # Apply SEU using mask stacks
        updated_pool = self.apply_seu(
            idxs=idxs,
            flip_masks_per_circuit=masks_list,
            layer_sizes=layer_sizes[1:],
            input_n=input_n,
            arity=arity,
            circuit_hidden_dim=circuit_hidden_dim,
        )

        return updated_pool, idxs


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
    base_wires: Optional[PyTree] = None,
    base_logits: Optional[PyTree] = None,
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

    # Determine source of single circuit to replicate
    if base_wires is not None or base_logits is not None:
        if base_wires is None or base_logits is None:
            raise ValueError("Both base_wires and base_logits must be provided together.")
        single_wires, single_logits = base_wires, base_logits
    else:
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

        # Use the layer_sizes parameter directly - knockouts don't change layer structure
        # The graph structure (nodes, layers, groups) remains static
        true_layer_sizes = layer_sizes

        # Generate new patterns for the selected indices
        pattern_keys = jax.random.split(knockout_key, num_to_damage)
        vmapped_pattern_creator = jax.vmap(
            lambda k: create_reproducible_knockout_pattern(
                key=k,
                layer_sizes=true_layer_sizes,
                damage_prob=damage_prob,
            )
        )
        new_patterns = vmapped_pattern_creator(pattern_keys)

        # Apply the new patterns at the selected indices
        pool_knockout_patterns = pool_knockout_patterns.at[damage_indices].set(new_patterns)

    return GraphPool.create(
        graphs, all_wires, all_logits, reset_counter, pool_knockout_patterns
    )
