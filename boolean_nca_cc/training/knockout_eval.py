"""
Evaluation dataset creation for knockout-specific testing.
"""

import jax
import jax.numpy as jp
import jraph
from typing import List, Tuple, Dict, Any, Optional
from flax import struct

from boolean_nca_cc.circuits.model import gen_circuit
from boolean_nca_cc.utils.graph_builder import build_graph
from boolean_nca_cc.training.pool.structural_perturbation import (
    create_reproducible_knockout_pattern,
    extract_layer_info_from_graph,
)

PyTree = Any


@struct.dataclass
class KnockoutEvaluationDatasets:
    """
    Holds IN-distribution and OUT-of-distribution datasets for knockout evaluation.
    """

    # IN-distribution (KO patterns seen during eval)
    in_distribution_graphs: jraph.GraphsTuple
    in_distribution_wires: PyTree
    in_distribution_logits: PyTree
    in_distribution_knockouts: jp.ndarray
    in_actual_batch_size: int = struct.field(pytree_node=False)

    # OUT-of-distribution (KO patterns not seen during eval)
    out_of_distribution_graphs: jraph.GraphsTuple
    out_of_distribution_wires: PyTree
    out_of_distribution_logits: PyTree
    out_of_distribution_knockouts: jp.ndarray
    out_actual_batch_size: int = struct.field(pytree_node=False)

    # Common parameters
    target_batch_size: int = struct.field(pytree_node=False)
    training_config: Dict[str, Any] = struct.field(pytree_node=False)

    def get_summary(self) -> str:
        """Returns a string summary of the datasets."""
        in_chunk_info = (
            f", {self.in_actual_batch_size // self.target_batch_size} chunks"
            if self.in_actual_batch_size > self.target_batch_size
            else ""
        )
        out_chunk_info = (
            f", {self.out_actual_batch_size // self.target_batch_size} chunks"
            if self.out_actual_batch_size > self.target_batch_size
            else ""
        )

        return (
            f"Knockout Evaluation Datasets created:\n"
            f"  IN-distribution: {self.in_actual_batch_size} circuits w/ knockouts (seed={self.training_config['evaluation_base_seed']}){in_chunk_info}\n"
            f"  OUT-of-distribution: {self.out_actual_batch_size} circuits w/ knockouts (seed={self.training_config['evaluation_base_seed']+1}){out_chunk_info}\n"
            f"  Damage prob={self.training_config['damage_prob']:.2f}, Target layer={self.training_config['target_layer']}"
        )


def create_knockout_evaluation_datasets(
    evaluation_base_seed: int,
    knockout_eval_config: Dict,
    layer_sizes: List[Tuple[int, int]],
    arity: int,
    circuit_hidden_dim: int,
    eval_batch_size: int,
) -> KnockoutEvaluationDatasets:
    """
    Creates standardized IN-distribution and OUT-of-distribution datasets for knockout evaluation.

    Both datasets use the *same* fixed base wiring but apply different knockout patterns.
    - IN-distribution: Knockout patterns generated with `evaluation_base_seed`.
    - OUT-of-distribution: Knockout patterns generated with `evaluation_base_seed + 1`.

    Args:
        evaluation_base_seed: The base random seed for generating datasets.
        knockout_eval_config: Configuration dictionary for knockout evaluation.
        layer_sizes: List of (nodes, group_size) for each layer.
        arity: Arity of gates.
        circuit_hidden_dim: Hidden dimension for graph nodes.
        eval_batch_size: The target batch size for evaluation.

    Returns:
        A KnockoutEvaluationDatasets object.
    """
    base_rng = jax.random.PRNGKey(evaluation_base_seed)
    in_dist_key, out_dist_key, base_circuit_key = jax.random.split(base_rng, 3)
    input_n = layer_sizes[0][0]

    # 1. Generate a single, fixed base circuit (wiring and logits)
    base_wires, base_logits = gen_circuit(base_circuit_key, layer_sizes, arity=arity)
    base_graph = build_graph(
        logits=base_logits,
        wires=base_wires,
        input_n=input_n,
        arity=arity,
        circuit_hidden_dim=circuit_hidden_dim,
    )
    true_layer_sizes = extract_layer_info_from_graph(base_graph, input_n)


    # 2. Create IN-distribution knockout patterns
    in_pattern_keys = jax.random.split(in_dist_key, eval_batch_size)
    vmapped_in_pattern_creator = jax.vmap(
        lambda k: create_reproducible_knockout_pattern(
            key=k,
            layer_sizes=true_layer_sizes,
            damage_prob=knockout_eval_config["damage_prob"],
            target_layer=knockout_eval_config["target_layer"],
            input_n=input_n,
        )
    )
    in_knockout_patterns = vmapped_in_pattern_creator(in_pattern_keys)

    # 3. Create OUT-of-distribution knockout patterns (different seed)
    out_pattern_keys = jax.random.split(out_dist_key, eval_batch_size)
    vmapped_out_pattern_creator = jax.vmap(
        lambda k: create_reproducible_knockout_pattern(
            key=k,
            layer_sizes=true_layer_sizes,
            damage_prob=knockout_eval_config["damage_prob"],
            target_layer=knockout_eval_config["target_layer"],
            input_n=input_n,
        )
    )
    out_knockout_patterns = vmapped_out_pattern_creator(out_pattern_keys)

    # 4. Replicate base circuit for the batch
    in_graphs = jax.tree.map(lambda x: jp.repeat(x[None, ...], eval_batch_size, axis=0), base_graph)
    in_wires = jax.tree.map(lambda x: jp.repeat(x[None, ...], eval_batch_size, axis=0), base_wires)
    in_logits = jax.tree.map(lambda x: jp.repeat(x[None, ...], eval_batch_size, axis=0), base_logits)
    
    out_graphs = jax.tree.map(lambda x: jp.repeat(x[None, ...], eval_batch_size, axis=0), base_graph)
    out_wires = jax.tree.map(lambda x: jp.repeat(x[None, ...], eval_batch_size, axis=0), base_wires)
    out_logits = jax.tree.map(lambda x: jp.repeat(x[None, ...], eval_batch_size, axis=0), base_logits)


    return KnockoutEvaluationDatasets(
        in_distribution_graphs=in_graphs,
        in_distribution_wires=in_wires,
        in_distribution_logits=in_logits,
        in_distribution_knockouts=in_knockout_patterns,
        in_actual_batch_size=eval_batch_size,
        out_of_distribution_graphs=out_graphs,
        out_of_distribution_wires=out_wires,
        out_of_distribution_logits=out_logits,
        out_of_distribution_knockouts=out_knockout_patterns,
        out_actual_batch_size=eval_batch_size,
        target_batch_size=eval_batch_size,
        training_config={
            "evaluation_base_seed": evaluation_base_seed,
            "damage_prob": knockout_eval_config["damage_prob"],
            "target_layer": knockout_eval_config["target_layer"],
        },
    ) 