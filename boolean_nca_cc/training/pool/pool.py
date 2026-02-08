"""
Pool implementation for boolean circuit NCA.

This module provides a GraphPool class that manages a collection of circuits
for training with partial updates between steps. It supports:

1. **Circuit Management**: Store, sample, and update circuits during training
2. **Pool Resets**: Reset fractions of the pool with fresh/mutated circuits
3. **Gate Damage**: Apply permanent gate knockouts to test resilience
4. **Diversity Tracking**: Monitor wiring diversity across the pool

The pool uses a batched jraph.GraphsTuple format where each leaf has a
leading batch dimension (pool_size).
"""

from functools import partial
from typing import Any

import jax
import jax.numpy as jp
import jraph
from flax import struct
from jax import Array

from boolean_nca_cc.circuits.model import gen_circuit
from boolean_nca_cc.utils.configured_graph_builder import configured_build_graph as build_graph
from boolean_nca_cc.utils.extraction import extract_logits_from_graph

PyTree = Any


class GraphPool(struct.PyTreeNode):
    """
    Pool class for graph-based circuit NCA.

    Stores a single batched jraph.GraphsTuple and allows sampling and updating
    batches. Also tracks wires, logits, gate masks, and damage counts.

    Attributes:
        size: Number of circuits in the pool
        graphs: Batched jraph.GraphsTuple (each leaf has leading batch dimension)
        wires: List of wire matrices with shape [pool_size, ...wire_shape...]
        logits: List of weight matrices corresponding to each graph
        gate_masks: Flat gate masks with shape [pool_size, total_gates]
                   Values: 0.0 = knocked out, 1.0 = active
        damage_count: Number of knocked-out gates per circuit [pool_size]
        reset_counter: Tracks how many epochs since each element was reset
    """

    size: int = struct.field(pytree_node=False)
    graphs: jraph.GraphsTuple
    wires: PyTree = None
    logits: PyTree = None
    gate_masks: jp.ndarray | None = None
    damage_count: jp.ndarray | None = None
    reset_counter: Array | None = None

    @classmethod
    def create(
        cls,
        batched_graphs: jraph.GraphsTuple,
        wires: PyTree = None,
        logits: PyTree = None,
        gate_masks: jp.ndarray | None = None,
        damage_count: jp.ndarray | None = None,
        reset_counter: Array | None = None,
    ) -> "GraphPool":
        """
        Create a new GraphPool instance from a batched GraphsTuple.

        Args:
            batched_graphs: A jraph.GraphsTuple where each leaf has leading batch dimension
            wires: The wire matrices corresponding to each graph
            logits: The logit matrices corresponding to each graph
            gate_masks: Flat gate masks with shape (pool_size, total_gates)
            damage_count: Number of knockouts per circuit (pool_size,)
            reset_counter: Counter for reset tracking

        Returns:
            A new GraphPool instance
        """
        if batched_graphs.n_node is None:
            raise ValueError("batched_graphs.n_node cannot be None for GraphPool.create")
        size = batched_graphs.n_node.shape[0]

        if reset_counter is None:
            reset_counter = jp.zeros(size, dtype=jp.int32)

        if damage_count is None and gate_masks is not None:
            # Initialize damage count from gate masks
            damage_count = jp.sum(1.0 - gate_masks, axis=1).astype(jp.int32)
        elif damage_count is None:
            damage_count = jp.zeros(size, dtype=jp.int32)

        return cls(
            size=size,
            graphs=batched_graphs,
            wires=wires,
            logits=logits,
            gate_masks=gate_masks,
            damage_count=damage_count,
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
        Update graphs in the pool at the specified indices.

        Args:
            idxs: The indices (1D array) at which to update
            batch_of_graphs: New graph data for the indices
            batch_of_wires: New wires (optional, keeps existing if None)
            batch_of_logits: New logits (optional, extracts from graphs if None)
            batch_of_gate_masks: New gate masks (optional, keeps existing if None)

        Returns:
            A new GraphPool instance with updated values
        """

        def core_update_logic(pool_leaf_non_none, batch_leaf_original):
            if batch_leaf_original is None:
                raise ValueError(
                    f"Field in pool is non-None but update batch provides None. "
                    f"Pool leaf type: {type(pool_leaf_non_none)}"
                )
            return pool_leaf_non_none.at[idxs].set(batch_leaf_original)

        def mapped_fn(pool_leaf, batch_leaf_original):
            return None if pool_leaf is None else core_update_logic(pool_leaf, batch_leaf_original)

        updated_graphs_data = jax.tree.map(
            mapped_fn,
            self.graphs,
            batch_of_graphs,
            is_leaf=lambda x: x is None,
        )

        # Update wires if provided
        updated_wires = self.wires
        if batch_of_wires is not None and self.wires is not None:
            updated_wires = jax.tree.map(
                lambda pool_wires, batch_wires: pool_wires.at[idxs].set(batch_wires),
                self.wires,
                batch_of_wires,
            )

        # Update logits (extract from graphs if not provided)
        updated_logits = self.logits
        if batch_of_logits is None and self.logits is not None:
            batch_of_logits = jax.vmap(extract_logits_from_graph, in_axes=(0, None))(
                batch_of_graphs, [log.shape[1:] for log in self.logits]
            )
        if self.logits is not None and batch_of_logits is not None:
            updated_logits = jax.tree.map(
                lambda pool_logits, batch_logits: pool_logits.at[idxs].set(batch_logits),
                self.logits,
                batch_of_logits,
            )

        # Update gate_masks if provided
        updated_gate_masks = self.gate_masks
        if batch_of_gate_masks is not None and self.gate_masks is not None:
            updated_gate_masks = self.gate_masks.at[idxs].set(batch_of_gate_masks)

        # Update damage count based on gate masks
        updated_damage_count = self.damage_count
        if batch_of_gate_masks is not None and self.damage_count is not None:
            batch_damage = jp.sum(1.0 - batch_of_gate_masks, axis=1).astype(jp.int32)
            updated_damage_count = self.damage_count.at[idxs].set(batch_damage)

        # Reset the counter for updated indices
        updated_reset_counter = (
            self.reset_counter.at[idxs].set(0) if self.reset_counter is not None else None
        )

        return self.replace(
            graphs=updated_graphs_data,
            wires=updated_wires,
            logits=updated_logits,
            gate_masks=updated_gate_masks,
            damage_count=updated_damage_count,
            reset_counter=updated_reset_counter,
        )

    @partial(jax.jit, static_argnames=("batch_size",))
    def sample(
        self, key: Array, batch_size: int
    ) -> tuple[Array, jraph.GraphsTuple, PyTree | None, PyTree | None, jp.ndarray | None]:
        """
        Sample a batch of graphs from the pool.

        Args:
            key: A random key
            batch_size: The size of the batch to sample

        Returns:
            Tuple of (idxs, sampled_graphs, sampled_wires, sampled_logits, sampled_gate_masks)
        """
        idxs = jax.random.choice(key, self.size, shape=(batch_size,))

        def _safe_slice_leaf(pool_leaf):
            if pool_leaf is None:
                return None
            return pool_leaf[idxs]

        sampled_graphs = jax.tree.map(_safe_slice_leaf, self.graphs)
        sampled_wires = (
            jax.tree.map(_safe_slice_leaf, self.wires) if self.wires is not None else None
        )
        sampled_logits = (
            jax.tree.map(_safe_slice_leaf, self.logits) if self.logits is not None else None
        )
        sampled_gate_masks = (
            _safe_slice_leaf(self.gate_masks) if self.gate_masks is not None else None
        )

        return idxs, sampled_graphs, sampled_wires, sampled_logits, sampled_gate_masks

    def get_average_update_steps(self) -> float:
        """Get the average number of update steps across all graphs in the pool."""
        if self.graphs.globals is None:
            return 0.0
        update_steps = self.graphs.globals.update_steps
        return float(jp.mean(update_steps))

    def get_average_update_steps_for_indices(self, indices: Array) -> float:
        """Get the average number of update steps for specified graph indices."""
        if self.graphs.globals is None:
            return 0.0
        update_steps = self.graphs.globals.update_steps[indices]
        return float(jp.mean(update_steps))

    def get_average_damage_count(self) -> float:
        """Get the average number of knocked-out gates across all circuits."""
        if self.damage_count is None:
            return 0.0
        return float(jp.mean(self.damage_count))

    def get_wiring_diversity(self, layer_sizes: list[tuple[int, int]] | None = None) -> float:
        """
        Calculate the wiring diversity of the pool using entropy-based measurement.

        Returns a value between 0 and 1:
        - 0: All circuits have identical wiring (fixed mode)
        - 1: Circuits have maximally diverse wiring (random mode)

        Args:
            layer_sizes: List of (nodes, group_size) tuples for each layer

        Returns:
            Float between 0 and 1 representing wiring diversity
        """
        if self.wires is None:
            return 0.0

        total_entropy = 0.0
        total_max_entropy = 0.0

        for layer_idx, layer_wires in enumerate(self.wires):
            if layer_wires.size == 0:
                continue

            _pool_size, arity, group_n = layer_wires.shape

            if layer_sizes is not None and layer_idx < len(layer_sizes):
                prev_layer_nodes = layer_sizes[layer_idx][0]
            else:
                max_connection_value = jp.max(layer_wires)
                prev_layer_nodes = int(max_connection_value) + 1

            layer_max_entropy = jp.log(float(prev_layer_nodes))

            for a in range(arity):
                for g in range(group_n):
                    connection_values = layer_wires[:, a, g]
                    _, counts = jp.unique(
                        connection_values, return_counts=True, size=prev_layer_nodes
                    )

                    valid_mask = counts > 0
                    valid_counts = counts[valid_mask]

                    if valid_counts.size == 0:
                        continue

                    probs = valid_counts / jp.sum(valid_counts)
                    log_probs = jp.where(probs > 0, jp.log(probs), 0.0)
                    entropy = -jp.sum(probs * log_probs)

                    total_entropy += entropy
                    total_max_entropy += layer_max_entropy

        if total_max_entropy == 0.0:
            return 0.0

        normalized_diversity = total_entropy / total_max_entropy
        return float(jp.clip(normalized_diversity, 0.0, 1.0))

    def get_reset_indices(
        self,
        key: Array,
        fraction: float,
        reset_strategy: str = "uniform",
        combined_weights: tuple[float, float] = (0.5, 0.5),
    ) -> tuple[Array, float]:
        """
        Get indices of circuits to reset/modify based on the specified strategy.

        Args:
            key: Random key for selection
            fraction: Fraction of pool to select (between 0 and 1)
            reset_strategy: Selection strategy:
                           "uniform" - random selection
                           "steps_biased" - favor circuits with more updates
                           "loss_biased" - favor circuits with higher loss
                           "combined" - combine loss and steps
            combined_weights: Weights for loss and steps in combined strategy

        Returns:
            Tuple of (selected_indices, avg_steps_of_selected)
        """
        num_select = jp.maximum(1, jp.round(fraction * self.size).astype(jp.int32))

        if reset_strategy == "uniform":
            selected_idxs = jax.random.choice(key, self.size, shape=(num_select,), replace=False)

        elif reset_strategy == "steps_biased":
            if self.graphs.globals is None:
                selected_idxs = jax.random.choice(
                    key, self.size, shape=(num_select,), replace=False
                )
            else:
                update_steps = self.graphs.globals.update_steps
                probs = update_steps
                probs = jp.where(
                    jp.sum(probs) > 0, probs / jp.sum(probs), jp.ones(self.size) / self.size
                )
                selected_idxs = jax.random.choice(
                    key, self.size, shape=(num_select,), replace=False, p=probs
                )

        elif reset_strategy == "loss_biased":
            if self.graphs.globals is None:
                selected_idxs = jax.random.choice(
                    key, self.size, shape=(num_select,), replace=False
                )
            else:
                loss_values = self.graphs.globals.loss
                probs = loss_values
                probs = jp.where(
                    jp.sum(probs) > 0, probs / jp.sum(probs), jp.ones(self.size) / self.size
                )
                selected_idxs = jax.random.choice(
                    key, self.size, shape=(num_select,), replace=False, p=probs
                )

        elif reset_strategy == "combined":
            if self.graphs.globals is None:
                selected_idxs = jax.random.choice(
                    key, self.size, shape=(num_select,), replace=False
                )
            else:
                loss_values = self.graphs.globals.loss
                update_steps = self.graphs.globals.update_steps
                loss_weight, steps_weight = combined_weights

                loss_scores = jp.where(
                    jp.sum(loss_values) > 0,
                    loss_values / jp.sum(loss_values),
                    jp.ones(self.size) / self.size,
                )
                step_scores = jp.where(
                    jp.sum(update_steps) > 0,
                    update_steps / jp.sum(update_steps),
                    jp.ones(self.size) / self.size,
                )

                combined_scores = loss_weight * loss_scores + steps_weight * step_scores
                probs = combined_scores / jp.sum(combined_scores)
                selected_idxs = jax.random.choice(
                    key, self.size, shape=(num_select,), replace=False, p=probs
                )
        else:
            raise ValueError(f"Unknown reset_strategy: {reset_strategy}")

        avg_steps = self.get_average_update_steps_for_indices(selected_idxs)
        return selected_idxs, avg_steps

    def reset_fraction(
        self,
        key: Array,
        fraction: float,
        new_graphs: jraph.GraphsTuple,
        new_wires: PyTree = None,
        new_logits: PyTree = None,
        new_gate_masks: jp.ndarray | None = None,
        reset_strategy: str = "uniform",
        combined_weights: tuple[float, float] = (0.5, 0.5),
    ) -> tuple["GraphPool", float]:
        """
        Reset a random fraction of the pool with fresh graphs.

        Damage counts are reset to zero for the reset circuits.

        Args:
            key: Random key for selection
            fraction: Fraction of pool to reset (between 0 and 1)
            new_graphs: Fresh graphs to use for reset
            new_wires: Fresh wires to use for reset
            new_logits: Fresh logits to use for reset
            new_gate_masks: Fresh gate masks (default: all active)
            reset_strategy: Strategy for selecting graphs to reset
            combined_weights: Weights for combined strategy

        Returns:
            Updated pool and average update steps of reset graphs
        """
        selection_key, sampling_key = jax.random.split(key)

        reset_idxs, avg_steps_reset = self.get_reset_indices(
            selection_key, fraction, reset_strategy, combined_weights
        )
        num_reset = len(reset_idxs)

        # Sample from new graphs
        sample_idxs = jax.random.choice(
            sampling_key, new_graphs.n_node.shape[0], shape=(num_reset,), replace=False
        )
        reset_graphs = jax.tree.map(
            lambda leaf: None if leaf is None else leaf[sample_idxs], new_graphs
        )

        reset_wires = (
            jax.tree.map(lambda leaf: leaf[sample_idxs], new_wires)
            if new_wires is not None
            else None
        )
        reset_logits = (
            jax.tree.map(lambda leaf: leaf[sample_idxs], new_logits)
            if new_logits is not None
            else None
        )
        reset_gate_masks = new_gate_masks[sample_idxs] if new_gate_masks is not None else None

        # Update pool
        reset_pool = self.update(
            reset_idxs, reset_graphs, reset_wires, reset_logits, reset_gate_masks
        )

        # Increment reset counter for all elements
        if reset_pool.reset_counter is not None:
            new_counter = reset_pool.reset_counter + 1
            reset_pool = reset_pool.replace(reset_counter=new_counter)

        return reset_pool, avg_steps_reset

    def apply_damage(
        self,
        key: Array,
        fraction: float,
        layer_sizes: list[tuple[int, int]],
        num_knockouts: int,
        input_n: int,
        arity: int,
        circuit_hidden_dim: int,
        faulty_value: float = -10.0,
        max_damage_per_circuit: int | None = None,
        selection_strategy: str = "uniform",
        combined_weights: tuple[float, float] = (0.5, 0.5),
        random_knockouts_per_event: bool = False,
        random_knockouts_per_event_min: int = 2,
        random_knockouts_per_event_max: int = 4,
    ) -> tuple["GraphPool", int]:
        """
        Apply permanent gate knockouts to a fraction of existing pool elements.

        This method applies structural damage where knocked-out gates:
        1. Have their logits set to faulty values (produce zero output)
        2. Cannot receive model updates (via gate_knockout_mask in node features)
        3. Can still send messages to neighbors

        Args:
            key: Random key for knockout generation and selection
            fraction: Fraction of pool to damage
            layer_sizes: Circuit layer sizes
            num_knockouts: Number of gates to knock out per circuit
            input_n: Number of input nodes
            arity: Number of inputs per gate
            circuit_hidden_dim: Hidden dimension for graphs
            faulty_value: Value for knocked-out gate logits
            max_damage_per_circuit: Maximum total knockouts per circuit (None = no limit)
            selection_strategy: Strategy for selecting circuits to damage
            combined_weights: Weights for combined selection strategy

        Returns:
            Tuple of (updated_pool, num_circuits_damaged)
        """
        from boolean_nca_cc.training.pool.structural_perturbation import (
            apply_knockout_to_batch,
            batch_layered_to_flat_mask,
        )

        selection_key, knockout_key = jax.random.split(key)

        # Get candidate indices
        candidate_idxs, _ = self.get_reset_indices(
            selection_key, fraction, selection_strategy, combined_weights
        )

        # Filter out circuits that have reached max damage (if specified)
        if max_damage_per_circuit is not None and self.damage_count is not None:
            candidate_damage = self.damage_count[candidate_idxs]
            eligible_mask = candidate_damage < max_damage_per_circuit
            damage_idxs = candidate_idxs[eligible_mask]
        else:
            damage_idxs = candidate_idxs

        num_damaged = len(damage_idxs)
        if num_damaged == 0:
            return self, 0

        # Extract logits and wires for selected circuits
        selected_logits = jax.tree.map(lambda leaf: leaf[damage_idxs], self.logits)
        selected_wires = jax.tree.map(lambda leaf: leaf[damage_idxs], self.wires)

        # Get current masks to combine with new knockouts
        if self.gate_masks is not None:
            current_masks = self.gate_masks[damage_idxs]
        else:
            total_gates = sum(gate_n for gate_n, _ in layer_sizes)
            current_masks = jp.ones((num_damaged, total_gates), dtype=jp.float32)

        if random_knockouts_per_event:
            knockout_key, normal_key = jax.random.split(knockout_key)
            mean = (random_knockouts_per_event_min + random_knockouts_per_event_max) / 2
            std = (random_knockouts_per_event_max - random_knockouts_per_event_min) / 8
            batch_num_knockouts = (
                (
                    jax.random.normal(
                        normal_key,
                        (num_damaged,),
                    )
                    * std
                    + mean
                )
                .clip(random_knockouts_per_event_min, random_knockouts_per_event_max)
                .astype(jp.int32)
            )
        else:
            assert num_knockouts is not None, (
                "num_knockouts must be set if random_knockouts_per_event is False"
            )
            batch_num_knockouts = jp.full((num_damaged,), num_knockouts, dtype=jp.int32)

        # Apply new knockouts
        modified_logits, layered_knockout_masks = apply_knockout_to_batch(
            knockout_key,
            selected_logits,
            layer_sizes,
            batch_num_knockouts,
            faulty_value,
        )

        # Convert to flat format and combine with existing damage
        new_flat_masks = batch_layered_to_flat_mask(layered_knockout_masks)
        combined_masks = current_masks * new_flat_masks  # Both must be 1.0 for gate to be active

        # Build new graphs with updated damage
        vmap_build_graph = jax.vmap(
            lambda logit, wires, mask: build_graph(
                logits=logit,
                wires=wires,
                input_n=input_n,
                arity=arity,
                circuit_hidden_dim=circuit_hidden_dim,
                loss_value=0.0,
                update_steps=0,
                gate_knockout_mask=mask,
                faulty_logit_value=faulty_value,
            )
        )
        modified_graphs = vmap_build_graph(modified_logits, selected_wires, combined_masks)

        # update graphs so that they keep their current loss value and update steps
        modified_graphs = modified_graphs._replace(
            globals=jax.tree.map(lambda leaf: leaf[damage_idxs], self.graphs.globals)
        )

        # Update pool
        updated_pool = self.update(
            damage_idxs, modified_graphs, selected_wires, modified_logits, combined_masks
        )

        return updated_pool, int(num_damaged)


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
    noise_scale: float = 0.0,
) -> GraphPool:
    """
    Initialize a pool of graphs for training.

    Args:
        rng: Random key
        layer_sizes: Circuit layer sizes
        pool_size: Number of graphs in the pool
        input_n: Number of inputs to the circuit
        arity: Number of inputs per gate
        circuit_hidden_dim: Dimension of hidden features
        loss_value: Initial loss value for graph globals
        wiring_mode: Mode for generating wirings:
                    "random" - unique wiring for each circuit
                    "fixed" - same wiring repeated
                    "genetic" - starts like fixed, mutations applied during resets
        initial_diversity: Number of different initial wirings (for fixed/genetic modes)
        initialize_gate_masks: Whether to initialize gate masks (all active initially)

    Returns:
        Initialized GraphPool
    """
    # Generate circuit wirings based on wiring mode
    if wiring_mode in ["fixed", "genetic"]:
        effective_diversity = jp.clip(initial_diversity, 1, pool_size)

        if effective_diversity == 1:
            # Single wiring repeated
            single_wires, single_logits = gen_circuit(
                rng, layer_sizes, arity=arity, noise_scale=noise_scale
            )
            all_wires = jax.tree.map(
                lambda leaf: jp.repeat(leaf[None, ...], pool_size, axis=0), single_wires
            )
            all_logits = jax.tree.map(
                lambda leaf: jp.repeat(leaf[None, ...], pool_size, axis=0), single_logits
            )
        elif effective_diversity >= pool_size:
            # Unique wiring for each
            rngs = jax.random.split(rng, pool_size)
            vmap_gen_circuit = jax.vmap(
                lambda rng: gen_circuit(rng, layer_sizes, arity=arity, noise_scale=noise_scale)
            )
            all_wires, all_logits = vmap_gen_circuit(rngs)
        else:
            # N different wirings repeated
            diversity_rngs = jax.random.split(rng, effective_diversity)
            vmap_gen_circuit = jax.vmap(
                lambda rng: gen_circuit(rng, layer_sizes, arity=arity, noise_scale=noise_scale)
            )
            diverse_wires, diverse_logits = vmap_gen_circuit(diversity_rngs)

            base_repeats = pool_size // effective_diversity
            extra_repeats = pool_size % effective_diversity

            repeat_counts = jp.concatenate(
                [
                    jp.full(extra_repeats, base_repeats + 1),
                    jp.full(effective_diversity - extra_repeats, base_repeats),
                ]
            )

            all_wires = []
            all_logits = []

            for layer_idx in range(len(diverse_wires)):
                layer_wires = []
                layer_logits = []

                for diversity_idx in range(effective_diversity):
                    n_repeats = repeat_counts[diversity_idx]
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

                all_wires.append(jp.concatenate(layer_wires, axis=0))
                all_logits.append(jp.concatenate(layer_logits, axis=0))
    else:  # wiring_mode == "random"
        rngs = jax.random.split(rng, pool_size)
        vmap_gen_circuit = jax.vmap(
            lambda rng: gen_circuit(rng, layer_sizes, arity=arity, noise_scale=noise_scale)
        )
        all_wires, all_logits = vmap_gen_circuit(rngs)

    # Initialize gate masks
    if initialize_gate_masks:
        total_gates = sum(gate_n for gate_n, _group_size in layer_sizes)
        all_gate_masks = jp.ones((pool_size, total_gates), dtype=jp.float32)
    else:
        all_gate_masks = None

    # Build graphs
    if initialize_gate_masks:
        vmap_build_graph = jax.vmap(
            lambda logit, wires, gate_mask: build_graph(
                logits=logit,
                wires=wires,
                input_n=input_n,
                arity=arity,
                circuit_hidden_dim=circuit_hidden_dim,
                loss_value=loss_value,
                update_steps=0,
                gate_knockout_mask=gate_mask,
                faulty_logit_value=-10.0,
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
                update_steps=0,
            )
        )
        graphs = vmap_build_graph(all_logits, all_wires)

    # Initialize counters
    reset_counter = jp.zeros(pool_size, dtype=jp.int32)
    damage_count = jp.zeros(pool_size, dtype=jp.int32)

    return GraphPool.create(
        graphs, all_wires, all_logits, all_gate_masks, damage_count, reset_counter
    )
