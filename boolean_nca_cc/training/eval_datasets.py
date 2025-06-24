"""
Evaluation dataset creation utilities for standardized circuit generation.

This module provides functions to create consistent evaluation datasets
across different parts of the codebase, ensuring reproducible evaluation.
"""

import jax
import jax.numpy as jp
from typing import List, Tuple, Optional, Dict, Any
import logging

from boolean_nca_cc.circuits.model import gen_circuit
from boolean_nca_cc.training.pool.pool import GraphPool

log = logging.getLogger(__name__)

PyTree = Any


class EvaluationDatasets:
    """
    Container for standardized evaluation datasets.

    Attributes:
        test_seed_wires: Single circuit wires for consistent reference evaluation
        test_seed_logits: Single circuit logits for consistent reference evaluation
        pool_wires: Batch of wires from training pool (if diversity > 1)
        pool_logits: Batch of logits from training pool (if diversity > 1)
        ood_wires: Batch of random circuit wires for OOD evaluation
        ood_logits: Batch of random circuit logits for OOD evaluation
        ood_batch_size: Size of OOD batch
        has_pool_data: Whether pool evaluation data is available
    """

    def __init__(
        self,
        test_seed_wires: List[jp.ndarray],
        test_seed_logits: List[jp.ndarray],
        ood_wires: List[jp.ndarray],
        ood_logits: List[jp.ndarray],
        ood_batch_size: int,
        pool_wires: Optional[List[jp.ndarray]] = None,
        pool_logits: Optional[List[jp.ndarray]] = None,
    ):
        self.test_seed_wires = test_seed_wires
        self.test_seed_logits = test_seed_logits
        self.ood_wires = ood_wires
        self.ood_logits = ood_logits
        self.ood_batch_size = ood_batch_size
        self.pool_wires = pool_wires
        self.pool_logits = pool_logits
        self.has_pool_data = pool_wires is not None and pool_logits is not None

    def get_summary(self) -> str:
        """Get a summary string of the evaluation datasets."""
        summary = (
            f"Evaluation Datasets:\n"
            f"  - Test seed: 1 circuit (reference)\n"
            f"  - OOD: {self.ood_batch_size} circuits (generalization)\n"
        )
        if self.has_pool_data:
            pool_batch_size = self.pool_wires[0].shape[0] if self.pool_wires else 0
            summary += f"  - Pool: {pool_batch_size} circuits (in-distribution)\n"
        else:
            summary += f"  - Pool: not available (diversity = 1)\n"
        return summary


def create_evaluation_datasets(
    test_seed: int,
    layer_sizes: List[Tuple[int, int]],
    arity: int,
    ood_batch_size: int,
    initial_diversity: int = 1,
    pool_diversity_size: int = 16,
    wiring_mode: str = "random",
) -> EvaluationDatasets:
    """
    Create standardized evaluation datasets for consistent evaluation.

    Args:
        test_seed: Seed for generating test circuits
        layer_sizes: Circuit layer configuration
        arity: Number of inputs per gate
        ood_batch_size: Number of random circuits for OOD evaluation
        initial_diversity: Initial diversity setting (determines if pool evaluation is used)
        pool_diversity_size: Number of circuits to create for pool evaluation (when diversity > 1)
        wiring_mode: Wiring mode for pool circuits ("random", "fixed", or "genetic")

    Returns:
        EvaluationDatasets object containing all evaluation circuits
    """
    log.info(f"Creating standardized evaluation datasets (test_seed={test_seed})")

    # 1. Test seed circuit (single reference circuit)
    test_key = jax.random.PRNGKey(test_seed)
    test_seed_wires, test_seed_logits = gen_circuit(test_key, layer_sizes, arity=arity)
    log.info("Created test seed circuit (reference)")

    # 2. OOD circuits (batch of random circuits for generalization)
    ood_rng = jax.random.PRNGKey(test_seed + 1000)  # Different seed for OOD
    ood_rngs = jax.random.split(ood_rng, ood_batch_size)

    # Use vmap to generate multiple circuits efficiently
    vmap_gen_circuit = jax.vmap(lambda rng: gen_circuit(rng, layer_sizes, arity=arity))
    ood_wires, ood_logits = vmap_gen_circuit(ood_rngs)
    log.info(f"Created {ood_batch_size} OOD circuits (generalization)")

    # 3. Pool circuits (in-distribution evaluation if diversity > 1)
    # Replicate the same logic as initialize_graph_pool for consistency
    pool_wires = None
    pool_logits = None

    if initial_diversity > 1:
        # Use a different seed for pool circuits (test_seed + 2000)
        pool_rng = jax.random.PRNGKey(test_seed + 2000)

        # Generate circuit wirings based on wiring mode (same logic as initialize_graph_pool)
        if wiring_mode in ["fixed", "genetic"]:
            # Clamp initial_diversity to valid range
            effective_diversity = jp.clip(initial_diversity, 1, pool_diversity_size)

            if effective_diversity == 1:
                # Single wiring repeated for all circuits
                single_wires, single_logits = gen_circuit(
                    pool_rng, layer_sizes, arity=arity
                )

                # Replicate the same wiring for all circuits
                pool_wires = jax.tree.map(
                    lambda leaf: jp.repeat(
                        leaf[None, ...], pool_diversity_size, axis=0
                    ),
                    single_wires,
                )
                pool_logits = jax.tree.map(
                    lambda leaf: jp.repeat(
                        leaf[None, ...], pool_diversity_size, axis=0
                    ),
                    single_logits,
                )
            elif effective_diversity >= pool_diversity_size:
                # Each circuit gets a unique wiring (same as random mode)
                pool_rngs = jax.random.split(pool_rng, pool_diversity_size)
                pool_wires, pool_logits = vmap_gen_circuit(pool_rngs)
            else:
                # Generate N different wirings and repeat them across the pool
                diversity_rngs = jax.random.split(pool_rng, effective_diversity)
                diverse_wires, diverse_logits = vmap_gen_circuit(diversity_rngs)

                # Calculate how many times to repeat each diverse wiring
                base_repeats = pool_diversity_size // effective_diversity
                extra_repeats = pool_diversity_size % effective_diversity

                # Create repeat counts: first 'extra_repeats' get one extra copy
                repeat_counts = jp.concatenate(
                    [
                        jp.full(extra_repeats, base_repeats + 1),
                        jp.full(effective_diversity - extra_repeats, base_repeats),
                    ]
                )

                # Repeat each diverse wiring according to repeat_counts
                pool_wires = []
                pool_logits = []

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
                            diverse_logits[layer_idx][
                                diversity_idx : diversity_idx + 1
                            ],
                            n_repeats,
                            axis=0,
                        )

                        layer_wires.append(repeated_wire)
                        layer_logits.append(repeated_logit)

                    # Concatenate all repeats for this layer
                    pool_wires.append(jp.concatenate(layer_wires, axis=0))
                    pool_logits.append(jp.concatenate(layer_logits, axis=0))
        else:  # wiring_mode == "random"
            # In random mode, generate different wirings for each circuit (ignore initial_diversity)
            pool_rngs = jax.random.split(pool_rng, pool_diversity_size)
            pool_wires, pool_logits = vmap_gen_circuit(pool_rngs)

        effective_diversity_log = (
            effective_diversity
            if wiring_mode in ["fixed", "genetic"]
            else pool_diversity_size
        )
        log.info(
            f"Created {pool_diversity_size} pool circuits (in-distribution, diversity={effective_diversity_log})"
        )
    else:
        log.info("Pool evaluation skipped (diversity = 1)")

    datasets = EvaluationDatasets(
        test_seed_wires=test_seed_wires,
        test_seed_logits=test_seed_logits,
        ood_wires=ood_wires,
        ood_logits=ood_logits,
        ood_batch_size=ood_batch_size,
        pool_wires=pool_wires,
        pool_logits=pool_logits,
    )

    log.info(datasets.get_summary())
    return datasets


def get_evaluation_rng_keys(
    base_rng: jax.random.PRNGKey, num_keys: int = 3
) -> Tuple[jax.random.PRNGKey, ...]:
    """
    Generate evaluation random keys for consistent key splitting.

    Args:
        base_rng: Base random key
        num_keys: Number of keys to generate (default: 3 for eval, pool, future)

    Returns:
        Tuple of random keys
    """
    return tuple(jax.random.split(base_rng, num_keys))


def reset_pool_circuits_for_evaluation(
    pool_wires: List[jp.ndarray],
    pool_logits: List[jp.ndarray],
    input_n: int,
    arity: int,
    hidden_dim: int,
) -> Tuple[Any, List[jp.ndarray], List[jp.ndarray]]:
    """
    Reset pool circuits to initial state for fair evaluation.

    This removes any accumulated update steps and resets loss to 0
    for consistent evaluation comparison.

    Args:
        pool_wires: Pool circuit wires
        pool_logits: Pool circuit logits
        input_n: Number of input nodes
        arity: Gate arity
        hidden_dim: Hidden dimension

    Returns:
        Tuple of (reset_graphs, pool_wires, pool_logits)
    """
    from boolean_nca_cc.utils.graph_builder import build_graph
    import jraph

    # Build fresh graphs from the pool circuits
    batch_size = pool_wires[0].shape[0]
    reset_graphs = []

    for i in range(batch_size):
        # Extract single circuit
        single_wires = [w[i] for w in pool_wires]
        single_logits = [l[i] for l in pool_logits]

        # Build graph with reset state (loss=0, steps=0)
        graph = build_graph(
            wires=single_wires,
            logits=single_logits,
            input_n=input_n,
            arity=arity,
            hidden_dim=hidden_dim,
        )
        # Ensure globals are reset to [0.0, 0.0] (loss, steps)
        graph = graph._replace(globals=jp.array([0.0, 0.0], dtype=jp.float32))
        reset_graphs.append(graph)

    # Stack graphs into a batch
    reset_batch_graphs = jraph.batch(reset_graphs)

    return reset_batch_graphs, pool_wires, pool_logits
