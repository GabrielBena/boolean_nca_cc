"""
Unified evaluation dataset creation utilities for standardized circuit generation.

This module provides functions to create consistent evaluation datasets
that exactly match the training distribution patterns, ensuring reproducible evaluation
with proper IN-distribution and OUT-of-distribution testing.
"""

import jax
import jax.numpy as jp
from typing import List, Tuple, Optional, Dict, Any
import logging

from boolean_nca_cc.circuits.model import gen_circuit

log = logging.getLogger(__name__)

PyTree = Any


class UnifiedEvaluationDatasets:
    """
    Container for unified evaluation datasets with consistent IN/OUT-of-distribution testing.

    Attributes:
        in_distribution_wires: Wires matching the training distribution pattern
        in_distribution_logits: Logits matching the training distribution pattern
        out_of_distribution_wires: Wires for OOD evaluation (always random)
        out_of_distribution_logits: Logits for OOD evaluation (always random)
        target_batch_size: Target batch size requested
        in_actual_batch_size: Actual IN-distribution batch size (may exceed target for full diversity)
        out_actual_batch_size: Actual OUT-of-distribution batch size
        training_config: Dictionary containing the training configuration used
    """

    def __init__(
        self,
        in_distribution_wires: List[jp.ndarray],
        in_distribution_logits: List[jp.ndarray],
        out_of_distribution_wires: List[jp.ndarray],
        out_of_distribution_logits: List[jp.ndarray],
        target_batch_size: int,
        in_actual_batch_size: int,
        out_actual_batch_size: int,
        training_config: Dict[str, Any],
    ):
        self.in_distribution_wires = in_distribution_wires
        self.in_distribution_logits = in_distribution_logits
        self.out_of_distribution_wires = out_of_distribution_wires
        self.out_of_distribution_logits = out_of_distribution_logits
        self.target_batch_size = target_batch_size
        self.in_actual_batch_size = in_actual_batch_size
        self.out_actual_batch_size = out_actual_batch_size
        self.training_config = training_config

    def get_summary(self) -> str:
        """Get a summary string of the evaluation datasets."""
        summary = (
            f"Unified Evaluation Datasets:\n"
            f"  - IN-distribution: {self.in_actual_batch_size} circuits (matches training: "
            f"mode={self.training_config['wiring_mode']}, "
            f"diversity={self.training_config['initial_diversity']})\n"
            f"  - OUT-of-distribution: {self.out_actual_batch_size} circuits (random wiring)\n"
        )
        if self.in_actual_batch_size > self.target_batch_size:
            summary += (
                f"  - Note: IN-distribution size exceeds target ({self.target_batch_size}) "
                f"to ensure full diversity coverage\n"
            )
        return summary


def create_unified_evaluation_datasets(
    evaluation_base_seed: int,
    training_wiring_mode: str,
    training_initial_diversity: int,
    layer_sizes: List[Tuple[int, int]],
    arity: int,
    eval_batch_size: int,
) -> UnifiedEvaluationDatasets:
    """
    Create unified evaluation datasets that properly match training patterns.

    This function creates exactly two types of evaluation circuits:
    1. IN-distribution: Matches the training wiring pattern exactly
    2. OUT-of-distribution: Always uses random wiring regardless of training mode

    Args:
        evaluation_base_seed: Base seed for generating all evaluation circuits
        training_wiring_mode: The wiring mode used in training ("fixed", "random", "genetic")
        training_initial_diversity: The initial diversity used in training
        layer_sizes: Circuit layer configuration
        arity: Number of inputs per gate
        eval_batch_size: Number of circuits in each evaluation set

    Returns:
        UnifiedEvaluationDatasets object containing IN and OUT distribution circuits
    """
    log.info(f"Creating unified evaluation datasets (base_seed={evaluation_base_seed})")
    log.info(
        f"Training config: mode={training_wiring_mode}, diversity={training_initial_diversity}"
    )

    # Create deterministic keys for IN and OUT distribution
    in_distribution_key = jax.random.PRNGKey(evaluation_base_seed)
    out_of_distribution_key = jax.random.PRNGKey(
        evaluation_base_seed + 10000
    )  # Clearly separated

    # 1. Create IN-distribution circuits (matching training pattern)
    log.info("Creating IN-distribution evaluation circuits...")
    in_distribution_wires, in_distribution_logits, in_actual_batch_size = (
        _create_circuit_batch_with_pattern(
            rng=in_distribution_key,
            layer_sizes=layer_sizes,
            arity=arity,
            batch_size=eval_batch_size,
            wiring_mode=training_wiring_mode,
            initial_diversity=training_initial_diversity,
        )
    )

    # 2. Create OUT-of-distribution circuits (always random)
    log.info("Creating OUT-of-distribution evaluation circuits...")
    out_distribution_wires, out_distribution_logits, out_actual_batch_size = (
        _create_circuit_batch_with_pattern(
            rng=out_of_distribution_key,
            layer_sizes=layer_sizes,
            arity=arity,
            batch_size=eval_batch_size,
            wiring_mode="random",  # Always random for OOD
            initial_diversity=eval_batch_size,  # Full diversity for OOD
        )
    )

    # Store training configuration for reference
    training_config = {
        "wiring_mode": training_wiring_mode,
        "initial_diversity": training_initial_diversity,
        "layer_sizes": layer_sizes,
        "arity": arity,
        "evaluation_base_seed": evaluation_base_seed,
    }

    datasets = UnifiedEvaluationDatasets(
        in_distribution_wires=in_distribution_wires,
        in_distribution_logits=in_distribution_logits,
        out_of_distribution_wires=out_distribution_wires,
        out_of_distribution_logits=out_distribution_logits,
        target_batch_size=eval_batch_size,
        in_actual_batch_size=in_actual_batch_size,
        out_actual_batch_size=out_actual_batch_size,
        training_config=training_config,
    )

    log.info(datasets.get_summary())
    return datasets


def _create_circuit_batch_with_pattern(
    rng: jax.random.PRNGKey,
    layer_sizes: List[Tuple[int, int]],
    arity: int,
    batch_size: int,
    wiring_mode: str,
    initial_diversity: int,
    get_all_wirings: bool = False,
) -> Tuple[List[jp.ndarray], List[jp.ndarray], int]:
    """
    Create a batch of circuits using the exact same logic as initialize_graph_pool.

    This ensures perfect consistency between training pool initialization and evaluation.
    When diversity exceeds batch_size, this function will return ALL unique wirings
    (not just a subset), allowing for comprehensive evaluation across the full diversity.

    Args:
        rng: Random key for generation
        layer_sizes: Circuit layer configuration
        arity: Number of inputs per gate
        batch_size: Target batch size (may be exceeded to cover all unique wirings)
        wiring_mode: Wiring generation mode ("fixed", "random", "genetic")
        initial_diversity: Number of unique wirings to start with

    Returns:
        Tuple of (batch_wires, batch_logits, actual_batch_size) lists
        Note: actual_batch_size may exceed batch_size to ensure full diversity coverage
    """
    # Use the exact same logic as initialize_graph_pool for consistency
    if wiring_mode in ["fixed", "genetic"]:
        # Don't clamp diversity to batch_size - we want to cover ALL unique wirings
        effective_diversity = jp.clip(initial_diversity, 1, jp.inf).astype(jp.int32)
        effective_diversity = min(effective_diversity, initial_diversity)

        if effective_diversity == 1:
            # Single wiring repeated for all circuits
            single_wires, single_logits = gen_circuit(rng, layer_sizes, arity=arity)

            # Replicate the same wiring for all circuits
            batch_wires = jax.tree.map(
                lambda leaf: jp.repeat(leaf[None, ...], batch_size, axis=0),
                single_wires,
            )
            batch_logits = jax.tree.map(
                lambda leaf: jp.repeat(leaf[None, ...], batch_size, axis=0),
                single_logits,
            )
            actual_batch_size = batch_size
        elif effective_diversity >= batch_size and get_all_wirings:
            # Generate ALL unique wirings (not just a subset)
            # This ensures comprehensive evaluation across the full diversity
            rngs = jax.random.split(rng, effective_diversity)
            vmap_gen_circuit = jax.vmap(
                lambda rng: gen_circuit(rng, layer_sizes, arity=arity)
            )
            batch_wires, batch_logits = vmap_gen_circuit(rngs)
            actual_batch_size = effective_diversity
        elif effective_diversity >= batch_size and not get_all_wirings:
            # Generate N different wirings and repeat them across the batch
            diversity_rngs = jax.random.split(rng, effective_diversity)[:batch_size]
            vmap_gen_circuit = jax.vmap(
                lambda rng: gen_circuit(rng, layer_sizes, arity=arity)
            )
            batch_wires, batch_logits = vmap_gen_circuit(diversity_rngs)
            actual_batch_size = batch_size
        else:
            # Generate N different wirings and repeat them across the batch
            diversity_rngs = jax.random.split(rng, effective_diversity)
            vmap_gen_circuit = jax.vmap(
                lambda rng: gen_circuit(rng, layer_sizes, arity=arity)
            )
            diverse_wires, diverse_logits = vmap_gen_circuit(diversity_rngs)

            # Calculate how many times to repeat each diverse wiring
            base_repeats = batch_size // effective_diversity
            extra_repeats = batch_size % effective_diversity

            # Create repeat counts: first 'extra_repeats' get one extra copy
            repeat_counts = jp.concatenate(
                [
                    jp.full(extra_repeats, base_repeats + 1),
                    jp.full(effective_diversity - extra_repeats, base_repeats),
                ]
            )

            # Repeat each diverse wiring according to repeat_counts
            batch_wires = []
            batch_logits = []

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
                batch_wires.append(jp.concatenate(layer_wires, axis=0))
                batch_logits.append(jp.concatenate(layer_logits, axis=0))
            actual_batch_size = batch_size
    else:  # wiring_mode == "random"
        # In random mode, generate different wirings for each circuit
        rngs = jax.random.split(rng, batch_size)
        vmap_gen_circuit = jax.vmap(
            lambda rng: gen_circuit(rng, layer_sizes, arity=arity)
        )
        batch_wires, batch_logits = vmap_gen_circuit(rngs)
        actual_batch_size = batch_size

    return batch_wires, batch_logits, actual_batch_size


def evaluate_circuits_in_chunks(
    eval_fn,
    wires: List[jp.ndarray],
    logits: List[jp.ndarray],
    target_chunk_size: int,
    **eval_kwargs,
) -> Dict:
    """
    Evaluate circuits in chunks to handle cases where diversity exceeds target batch size.

    Args:
        eval_fn: Evaluation function to apply to each chunk
        wires: List of wire arrays for all circuits
        logits: List of logit arrays for all circuits
        target_chunk_size: Target size for each evaluation chunk
        **eval_kwargs: Additional keyword arguments to pass to eval_fn

    Returns:
        Dictionary with averaged metrics across all chunks
    """
    total_circuits = wires[0].shape[0]

    if total_circuits <= target_chunk_size:
        # No need to chunk, evaluate all at once
        return eval_fn(batch_wires=wires, batch_logits=logits, **eval_kwargs)

    # Split into chunks and evaluate each
    num_chunks = (total_circuits + target_chunk_size - 1) // target_chunk_size
    chunk_results = []

    # Pop knockout_patterns from kwargs to handle it manually for chunking
    local_kwargs = eval_kwargs.copy()
    knockout_patterns = local_kwargs.pop("knockout_patterns", None)

    for chunk_idx in range(num_chunks):
        start_idx = chunk_idx * target_chunk_size
        end_idx = min(start_idx + target_chunk_size, total_circuits)

        # Extract chunk
        chunk_wires = [w[start_idx:end_idx] for w in wires]
        chunk_logits = [l[start_idx:end_idx] for l in logits]

        chunk_knockouts = None
        if knockout_patterns is not None:
            chunk_knockouts = knockout_patterns[start_idx:end_idx]

        # Evaluate chunk
        chunk_result = eval_fn(
            batch_wires=chunk_wires,
            batch_logits=chunk_logits,
            knockout_patterns=chunk_knockouts,
            **local_kwargs,
        )
        chunk_results.append(chunk_result)

    # Average results across chunks
    # Assume all chunk results have the same structure
    averaged_result = {}
    for key in chunk_results[0].keys():
        if isinstance(chunk_results[0][key], list):
            # For step-wise metrics, average at each step
            step_averages = []
            for step_idx in range(len(chunk_results[0][key])):
                step_values = [chunk[key][step_idx] for chunk in chunk_results]
                step_averages.append(float(jp.mean(jp.array(step_values))))
            averaged_result[key] = step_averages
        else:
            # For scalar metrics, simple average
            values = [chunk[key] for chunk in chunk_results]
            averaged_result[key] = float(jp.mean(jp.array(values)))

    return averaged_result
