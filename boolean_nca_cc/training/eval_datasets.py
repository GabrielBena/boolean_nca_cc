"""
Unified evaluation dataset creation utilities for standardized circuit generation.

This module provides functions to create consistent evaluation datasets
that exactly match the training distribution patterns, ensuring reproducible evaluation
with proper IN-distribution and OUT-of-distribution testing.
"""

import logging
from typing import Any

import jax
import jax.numpy as jp

from boolean_nca_cc.training.pool.pool import get_wires_and_logits

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
        in_distribution_y_task: Per-circuit task targets for IN-distribution eval,
            shape ``[in_actual_batch_size, 2^input_n, output_n]``, or None for
            the legacy global-task path.
        out_of_distribution_y_task: Per-circuit task targets for OOD eval (from
            a held-out task family), same shape semantics, or None.
        x_task: The shared input enumeration for per-circuit-task eval
            (``build_task_x(input_n)``), or None for legacy path. When per-
            circuit y_task arrays are present, this is the corresponding x.
    """

    def __init__(
        self,
        in_distribution_wires: list[jp.ndarray],
        in_distribution_logits: list[jp.ndarray],
        out_of_distribution_wires: list[jp.ndarray],
        out_of_distribution_logits: list[jp.ndarray],
        target_batch_size: int,
        in_actual_batch_size: int,
        out_actual_batch_size: int,
        training_config: dict[str, Any],
        in_distribution_y_task: jp.ndarray | None = None,
        out_of_distribution_y_task: jp.ndarray | None = None,
        x_task: jp.ndarray | None = None,
    ):
        self.in_distribution_wires = in_distribution_wires
        self.in_distribution_logits = in_distribution_logits
        self.out_of_distribution_wires = out_of_distribution_wires
        self.out_of_distribution_logits = out_of_distribution_logits
        self.target_batch_size = target_batch_size
        self.in_actual_batch_size = in_actual_batch_size
        self.out_actual_batch_size = out_actual_batch_size
        self.training_config = training_config
        self.in_distribution_y_task = in_distribution_y_task
        self.out_of_distribution_y_task = out_of_distribution_y_task
        self.x_task = x_task

    def get_summary(self) -> str:
        """Get a summary string of the evaluation datasets."""
        summary = f"Unified Evaluation Datasets:\n"
        if self.in_distribution_wires is not None:
            summary += (
                f"  - IN-distribution: {self.in_actual_batch_size} circuits (matches training: "
                f"mode={self.training_config['wiring_mode']}, "
                f"diversity={self.training_config['initial_diversity']})\n"
            )
        if self.out_of_distribution_wires is not None:
            summary += (
                f"  - OUT-of-distribution: {self.out_actual_batch_size} circuits (random wiring)\n"
            )

        if (
            self.in_actual_batch_size is not None
            and self.in_actual_batch_size > self.target_batch_size
        ):
            summary += (
                f"  - Note: IN-distribution size exceeds target ({self.target_batch_size}) "
                f"to ensure full diversity coverage\n"
            )
        return summary


def create_unified_evaluation_datasets(
    eval_key: jax.random.PRNGKey,
    training_wiring_mode: str,
    training_initial_diversity: int,
    layer_sizes: list[tuple[int, int]],
    arity: int,
    eval_batch_size_in: int,
    eval_batch_size_out: int,
    do_ood_evaluation: bool = True,
    pool_noise_scale: float = 0.0,
    init_logits: str = "soft_wires",
    random_init_scale: float = 1.0,
    task_sampler_cfg_in: dict | None = None,
    task_sampler_cfg_ood: dict | None = None,
) -> UnifiedEvaluationDatasets:
    """
    Create unified evaluation datasets that properly match training patterns.

    This function creates exactly two types of evaluation circuits:
    1. IN-distribution: Matches the training wiring pattern exactly
    2. OUT-of-distribution: Always uses random wiring regardless of training mode

    Args:
        eval_key: Random key for generating all evaluation circuits
        training_wiring_mode: The wiring mode used in training ("fixed", "random", "genetic")
        training_initial_diversity: The initial diversity used in training
        layer_sizes: Circuit layer configuration
        arity: Number of inputs per gate
        eval_batch_size_in: Number of circuits in each IN-distribution evaluation set
        eval_batch_size_out: Number of circuits in each OUT-of-distribution evaluation set
        do_ood_evaluation: Whether to create OUT-of-distribution evaluation circuits
        get_all_wirings: Whether to get all wirings (True) or a subset (False)
    Returns:
        UnifiedEvaluationDatasets object containing IN and OUT distribution circuits
    """
    log.info(f"Creating unified evaluation datasets (eval_key={eval_key})")
    log.info(
        f"Training config: mode={training_wiring_mode}, diversity={training_initial_diversity}"
    )

    # Create deterministic keys for IN and OUT distribution
    in_distribution_key = eval_key
    out_of_distribution_key = jax.random.fold_in(eval_key, 42)

    # 1. Create IN-distribution circuits (matching training pattern) (only for fixed and genetic)
    if training_wiring_mode in ["fixed", "genetic"]:
        log.info("Creating IN-distribution evaluation circuits...")
        wires_key, logits_key = jax.random.split(in_distribution_key)
        in_distribution_wires, in_distribution_logits = get_wires_and_logits(
            wires_key=wires_key,
            logits_key=logits_key,
            layer_sizes=layer_sizes,
            arity=arity,
            pool_size=eval_batch_size_in,
            wiring_mode=training_wiring_mode,
            initial_diversity=training_initial_diversity,
            noise_scale=pool_noise_scale,
            init_logits=init_logits,
            random_init_scale=random_init_scale,
        )
    else:
        in_distribution_wires, in_distribution_logits = None, None

    # 2. Create OUT-of-distribution circuits (always random)
    if do_ood_evaluation:
        log.info("Creating OUT-of-distribution evaluation circuits...")
        wires_key, logits_key = jax.random.split(out_of_distribution_key)
        out_distribution_wires, out_distribution_logits = get_wires_and_logits(
            wires_key=wires_key,
            logits_key=logits_key,
            layer_sizes=layer_sizes,
            arity=arity,
            pool_size=eval_batch_size_out,
            wiring_mode="random",  # Always random for OOD
            initial_diversity=eval_batch_size_out,  # Full diversity for OOD
            noise_scale=pool_noise_scale,
            init_logits=init_logits,
            random_init_scale=random_init_scale,
        )
    else:
        out_distribution_wires, out_distribution_logits = None, None

    # 3. Per-circuit y_task arrays (only when a sampler config is provided).
    #    IN-side: same family as training, held-out RNG seed → fresh in-dist tasks.
    #    OOD-side: a held-out task family (e.g., library tasks like add/parity/…).
    in_distribution_y_task = None
    out_of_distribution_y_task = None
    x_task = None
    if task_sampler_cfg_in is not None or task_sampler_cfg_ood is not None:
        from boolean_nca_cc.tasks import build_task_x, sample_task_batch

        input_n = layer_sizes[0][0]
        output_n = layer_sizes[-1][0]
        x_task = build_task_x(input_n)

        if task_sampler_cfg_in is not None and in_distribution_wires is not None:
            in_task_key = jax.random.fold_in(eval_key, 101)
            in_distribution_y_task = sample_task_batch(
                in_task_key, eval_batch_size_in, input_n, output_n, task_sampler_cfg_in
            )
        if task_sampler_cfg_ood is not None and out_distribution_wires is not None:
            ood_task_key = jax.random.fold_in(eval_key, 202)
            out_of_distribution_y_task = sample_task_batch(
                ood_task_key, eval_batch_size_out, input_n, output_n, task_sampler_cfg_ood
            )
        log.info(
            f"Per-circuit-task eval: in_sampler={task_sampler_cfg_in!r}, "
            f"ood_sampler={task_sampler_cfg_ood!r}"
        )

    # Store training configuration for reference
    training_config = {
        "wiring_mode": training_wiring_mode,
        "initial_diversity": training_initial_diversity,
        "layer_sizes": layer_sizes,
        "arity": arity,
        "in_distribution_key": in_distribution_key,
        "out_of_distribution_key": out_of_distribution_key,
        "task_sampler_cfg_in": task_sampler_cfg_in,
        "task_sampler_cfg_ood": task_sampler_cfg_ood,
    }

    datasets = UnifiedEvaluationDatasets(
        in_distribution_wires=in_distribution_wires,
        in_distribution_logits=in_distribution_logits,
        out_of_distribution_wires=out_distribution_wires,
        out_of_distribution_logits=out_distribution_logits,
        target_batch_size=max(eval_batch_size_in, eval_batch_size_out),
        in_actual_batch_size=eval_batch_size_in,
        out_actual_batch_size=eval_batch_size_out,
        training_config=training_config,
        in_distribution_y_task=in_distribution_y_task,
        out_of_distribution_y_task=out_of_distribution_y_task,
        x_task=x_task,
    )

    return datasets
