"""
Basic implementation for exploring degenerate circuit solution spaces.

This script implements the core exploration mechanism:
1. Loads a preconfigured circuit
2. Perturbs it multiple times using greedy indices (reversibly)
3. Lets it recover with backprop
4. Tracks unique solutions discovered

This is a minimal skeleton - no visualizations or advanced analysis yet.
"""

import argparse
import hashlib
import logging
import random
from pathlib import Path
from typing import List, Tuple, Dict, Set, Optional, Callable
import numpy as np
import jax
import jax.numpy as jp
import optax
from collections import deque, defaultdict
import pickle
import json
from datetime import datetime

from boolean_nca_cc.circuits.model import run_circuit
from boolean_nca_cc.circuits.train import compute_accuracy, TrainState, train_step, loss_f_l4, loss_f_bce
from boolean_nca_cc.circuits.train import apply_reversible_bias_to_logits
from boolean_nca_cc.circuits.tasks import get_task_data
from boolean_nca_cc.training.preconfigure import preconfigure_circuit_logits
from boolean_nca_cc.training.pool.structural_perturbation import (
    create_knockout_vocabulary,
    DEFAULT_GREEDY_ORDERED_INDICES,
)
from boolean_nca_cc.training.backprop import _train_single_knockout_pattern
from boolean_nca_cc.training.evaluation import evaluate_model_stepwise_generator
from boolean_nca_cc.models.self_attention import CircuitSelfAttention

# Setup logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def hash_circuit_logits(logits: List[jp.ndarray]) -> str:
    """
    Create a hash fingerprint of circuit logits for uniqueness detection.
    
    Args:
        logits: List of logit arrays for each layer
        
    Returns:
        Hexadecimal hash string
    """
    # Flatten all logits into a single array
    flat_logits = jp.concatenate([l.flatten() for l in logits])
    # Convert to numpy for hashing (JAX arrays aren't directly hashable)
    flat_np = np.array(flat_logits)
    # Create hash
    hash_obj = hashlib.sha256(flat_np.tobytes())
    return hash_obj.hexdigest()


def is_functional(
    logits: List[jp.ndarray],
    wires: List[jp.ndarray],
    x_data: jp.ndarray,
    y_data: jp.ndarray,
    threshold: float = 0.999,
) -> bool:
    """
    Check if circuit implements target function with sufficient accuracy.
    
    Args:
        logits: Circuit logits
        wires: Circuit wiring
        x_data: Input data
        y_data: Target output data
        threshold: Minimum accuracy threshold (default 0.95)
        
    Returns:
        True if accuracy >= threshold
    """
    pred = run_circuit(logits, wires, x_data, hard=True)[-1]
    accuracy = float(compute_accuracy(pred, y_data))
    return accuracy >= threshold


def save_exploration_results(
    results: Dict,
    output_dir: Path,
    wires: List[jp.ndarray],
    layer_sizes: List[Tuple[int, int]],
    task_name: str,
    exploration_config: Dict,
) -> None:
    """
    Save exploration results to disk for later loading and visualization.
    
    Saves:
    - solutions.npz: All unique solution logits (solution_{idx}_layer_{layer_idx})
    - metadata.pkl: All metadata (hashes, exploration results, graph structure, summary)
    - wires.npz: Circuit wiring (needed for functional testing)
    - config.json: Exploration configuration (human-readable)
    
    Args:
        results: Results dictionary from explore_degenerate_solutions
        output_dir: Directory to save results
        wires: Circuit wiring (needed for functional testing)
        layer_sizes: Circuit layer sizes
        task_name: Task name
        exploration_config: Configuration used for exploration
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    log.info(f"\nSaving exploration results to: {output_dir}")
    
    # Save unique solutions logits
    solutions_file = output_dir / "solutions.npz"
    solutions_dict = {}
    hash_to_idx = {}
    
    for idx, (circuit_hash, logits) in enumerate(results["unique_solutions"].items()):
        hash_to_idx[circuit_hash] = idx
        for layer_idx, logit_layer in enumerate(logits):
            # Convert JAX array to numpy for saving
            logit_np = np.array(logit_layer)
            solutions_dict[f"solution_{idx}_layer_{layer_idx}"] = logit_np
    
    np.savez(solutions_file, **solutions_dict)
    log.info(f"  Saved {len(results['unique_solutions'])} unique solutions to {solutions_file}")
    
    # Save wires
    wires_file = output_dir / "wires.npz"
    wires_dict = {}
    for layer_idx, wire_layer in enumerate(wires):
        wire_np = np.array(wire_layer)
        wires_dict[f"layer_{layer_idx}"] = wire_np
    np.savez(wires_file, **wires_dict)
    log.info(f"  Saved wires to {wires_file}")
    
    # Prepare metadata (convert non-serializable objects)
    metadata = {
        "unique_solutions": {
            "hashes": list(results["unique_solutions"].keys()),
            "hash_to_idx": hash_to_idx,
            "num_solutions": len(results["unique_solutions"]),
        },
        "exploration_results": results["exploration_results"],
        "summary": results["summary"],
        "root_hash": results["root_hash"],
        "layer_sizes": layer_sizes,
        "task_name": task_name,
        "exploration_config": exploration_config,
        "timestamp": datetime.now().isoformat(),
    }
    
    # Save exploration graph (simplified - convert tuples to lists for JSON compatibility)
    # We'll save a simplified version that's JSON-serializable
    exploration_graph_simple = {}
    for source_hash, edges_list in results["exploration_graph"].items():
        exploration_graph_simple[source_hash] = [
            {
                "target_hash": target_hash,
                "pattern_idx": pattern_idx,
                "metadata": metadata_dict,
            }
            for target_hash, pattern_idx, metadata_dict in edges_list
        ]
    metadata["exploration_graph"] = exploration_graph_simple
    
    # Save edges (simplified)
    edges_simple = [
        {
            "source_hash": source_hash,
            "target_hash": target_hash,
            "pattern_idx": pattern_idx,
            "metadata": metadata_dict,
        }
        for source_hash, target_hash, pattern_idx, metadata_dict in results["edges"]
    ]
    metadata["edges"] = edges_simple
    
    # Save metadata as pickle (preserves all Python objects)
    metadata_file = output_dir / "metadata.pkl"
    with open(metadata_file, "wb") as f:
        pickle.dump(metadata, f)
    log.info(f"  Saved metadata to {metadata_file}")
    
    # Save config as JSON (human-readable)
    config_file = output_dir / "config.json"
    config_json = {
        "task_name": task_name,
        "layer_sizes": layer_sizes,
        "exploration_config": exploration_config,
        "summary": {
            k: v for k, v in results["summary"].items() if isinstance(v, (int, float, str, bool))
        },
        "num_unique_solutions": len(results["unique_solutions"]),
        "timestamp": metadata["timestamp"],
    }
    with open(config_file, "w") as f:
        json.dump(config_json, f, indent=2)
    log.info(f"  Saved config to {config_file}")
    
    log.info(f"\nExploration results saved successfully!")
    log.info(f"  Solutions: {solutions_file}")
    log.info(f"  Wires: {wires_file}")
    log.info(f"  Metadata: {metadata_file}")
    log.info(f"  Config: {config_file}")


def load_exploration_results(output_dir: Path) -> Dict:
    """
    Load exploration results from disk.
    
    Args:
        output_dir: Directory containing saved exploration results
        
    Returns:
        Dictionary with:
        - unique_solutions: Dict[hash -> List[jp.ndarray]] of circuit logits
        - wires: List[jp.ndarray] of circuit wiring
        - exploration_results: List of exploration results
        - exploration_graph: Graph structure
        - edges: List of edges
        - summary: Summary statistics
        - metadata: Full metadata dictionary
        - config: Configuration dictionary
        - layer_sizes: Circuit layer sizes
        - task_name: Task name
    
    Example:
        ```python
        from experiments.explore_degenerate_solutions import load_exploration_results
        from pathlib import Path
        
        # Load results
        results = load_exploration_results(Path("exploration_results/exploration_20240101_120000"))
        
        # Access unique solutions for UMAP
        unique_solutions = results["unique_solutions"]
        for circuit_hash, logits in unique_solutions.items():
            # Flatten logits for UMAP
            flat_logits = jp.concatenate([l.flatten() for l in logits])
            # ... use flat_logits for UMAP embedding
        ```
    """
    output_dir = Path(output_dir)
    
    log.info(f"Loading exploration results from: {output_dir}")
    
    # Load solutions
    solutions_file = output_dir / "solutions.npz"
    solutions_data = np.load(solutions_file)
    
    # Reconstruct unique_solutions dictionary
    unique_solutions = {}
    metadata_file = output_dir / "metadata.pkl"
    with open(metadata_file, "rb") as f:
        metadata = pickle.load(f)
    
    hash_to_idx = metadata["unique_solutions"]["hash_to_idx"]
    idx_to_hash = {idx: hash_val for hash_val, idx in hash_to_idx.items()}
    
    # Find maximum solution index
    max_idx = max(hash_to_idx.values()) if hash_to_idx else -1
    
    for idx in range(max_idx + 1):
        if idx not in idx_to_hash:
            continue
        
        circuit_hash = idx_to_hash[idx]
        logits = []
        layer_idx = 0
        while f"solution_{idx}_layer_{layer_idx}" in solutions_data:
            logit_array = jp.array(solutions_data[f"solution_{idx}_layer_{layer_idx}"])
            logits.append(logit_array)
            layer_idx += 1
        
        if logits:
            unique_solutions[circuit_hash] = logits
    
    log.info(f"  Loaded {len(unique_solutions)} unique solutions")
    
    # Load wires
    wires_file = output_dir / "wires.npz"
    wires_data = np.load(wires_file)
    wires = []
    layer_idx = 0
    while f"layer_{layer_idx}" in wires_data:
        wire_array = jp.array(wires_data[f"layer_{layer_idx}"])
        wires.append(wire_array)
        layer_idx += 1
    
    log.info(f"  Loaded {len(wires)} wire layers")
    
    # Reconstruct exploration graph
    exploration_graph = {}
    for source_hash, edges_list in metadata["exploration_graph"].items():
        exploration_graph[source_hash] = [
            (edge["target_hash"], edge["pattern_idx"], edge["metadata"])
            for edge in edges_list
        ]
    
    # Reconstruct edges
    edges = [
        (edge["source_hash"], edge["target_hash"], edge["pattern_idx"], edge["metadata"])
        for edge in metadata["edges"]
    ]
    
    return {
        "unique_solutions": unique_solutions,
        "wires": wires,
        "exploration_results": metadata["exploration_results"],
        "exploration_graph": exploration_graph,
        "edges": edges,
        "summary": metadata["summary"],
        "root_hash": metadata["root_hash"],
        "metadata": metadata,
        "config": metadata.get("exploration_config", {}),
        "layer_sizes": metadata["layer_sizes"],
        "task_name": metadata["task_name"],
    }


def infer_layer_sizes_from_logits(logits: List[jp.ndarray], input_n: int) -> List[Tuple[int, int]]:
    """
    Infer layer_sizes from loaded logits structure.
    
    Args:
        logits: List of logit arrays, each with shape (groups, group_size, 2^arity)
        input_n: Number of input nodes
        
    Returns:
        List of (total_gates, group_size) tuples for each layer
    """
    layer_sizes = [(input_n, 1)]  # Input layer
    for logit_layer in logits:
        groups, group_size = logit_layer.shape[:2]
        total_gates = groups * group_size
        layer_sizes.append((total_gates, group_size))
    return layer_sizes


def load_preconfigured_circuit(
    logits_file: str = None,
    wires_file: str = None,
    wiring_key: jax.random.PRNGKey = None,
    layer_sizes: List[Tuple[int, int]] = None,
    arity: int = 2,
    x_data: jp.ndarray = None,
    y_data: jp.ndarray = None,
    loss_type: str = "l4",
    preconfig_steps: int = 200,
    preconfig_lr: float = 1.0,
    preconfig_optimizer: str = "adamw",
    preconfig_weight_decay: float = 1e-1,
    preconfig_beta1: float = 0.8,
    preconfig_beta2: float = 0.8,
) -> Tuple[List[jp.ndarray], List[jp.ndarray], List[Tuple[int, int]]]:
    """
    Load or generate a preconfigured circuit.
    
    Args:
        logits_file: Optional path to NPZ file with preconfigured logits
        wires_file: Optional path to NPZ file with wires
        wiring_key: Random key for generating wiring (if not loading from file)
        layer_sizes: Circuit layer sizes
        arity: Gate arity
        x_data: Input data for preconfiguration
        y_data: Target data for preconfiguration
        loss_type: Loss type for preconfiguration
        preconfig_steps: Number of preconfiguration steps
        preconfig_lr: Learning rate for preconfiguration
        
    Returns:
        Tuple of (wires, logits)
    """
    if logits_file is not None and wires_file is not None:
        log.info(f"Loading preconfigured circuit from files:")
        log.info(f"  Logits: {logits_file}")
        log.info(f"  Wires: {wires_file}")
        
        # Load logits
        logits_data = np.load(logits_file)
        logits = []
        i = 0
        while f"layer_{i}" in logits_data:
            logits.append(jp.array(logits_data[f"layer_{i}"]))
            i += 1
        
        # Load wires
        wires_data = np.load(wires_file)
        wires = []
        i = 0
        while f"layer_{i}" in wires_data:
            wires.append(jp.array(wires_data[f"layer_{i}"]))
            i += 1
        
        log.info(f"Loaded {len(logits)} logit layers and {len(wires)} wire layers")
        
        # Infer actual layer_sizes from loaded logits
        # Use input_n from provided layer_sizes if available, otherwise infer from wires or use default
        if layer_sizes is not None and len(layer_sizes) > 0:
            input_n = layer_sizes[0][0]
        elif len(wires) > 0:
            # Try to infer from first wire layer - wires[0] has shape (arity, num_output_gates)
            # The number of unique input connections can be inferred, but it's complex
            # For now, use a reasonable default based on the task
            input_n = 8  # Default fallback
            log.warning(f"Could not infer input_n from layer_sizes, using default: {input_n}")
        else:
            input_n = 8
        
        inferred_layer_sizes = infer_layer_sizes_from_logits(logits, input_n)
        log.info(f"Inferred layer_sizes from loaded logits: {inferred_layer_sizes}")
        return wires, logits, inferred_layer_sizes
    else:
        log.info("Generating new preconfigured circuit")
        if wiring_key is None:
            wiring_key = jax.random.PRNGKey(42)
        if layer_sizes is None:
            raise ValueError("layer_sizes required when generating circuit")
        if x_data is None or y_data is None:
            raise ValueError("x_data and y_data required when generating circuit")
        
        wires, logits = preconfigure_circuit_logits(
            wiring_key=wiring_key,
            layer_sizes=layer_sizes,
            arity=arity,
            x_data=x_data,
            y_data=y_data,
            loss_type=loss_type,
            steps=preconfig_steps,
            lr=preconfig_lr,
            optimizer=preconfig_optimizer,
            weight_decay=preconfig_weight_decay,
            beta1=preconfig_beta1,
            beta2=preconfig_beta2,
        )
        log.info(f"Generated preconfigured circuit with {len(logits)} layers")
        return wires, logits, layer_sizes


def _perturb_and_recover_single(
    initial_logits: List[jp.ndarray],
    knockout_pattern: jp.ndarray,
    wires: List[jp.ndarray],
    x_data: jp.ndarray,
    y_data: jp.ndarray,
    layer_sizes: List[Tuple[int, int]],
    loss_type: str,
    functional_threshold: float,
    # Backprop parameters
    opt: Optional[optax.GradientTransformation] = None,
    epochs: int = 200,
    reversible_bias: float = -10.0,
    # Self-attention parameters
    recovery_mode: str = "backprop",
    model: Optional[CircuitSelfAttention] = None,
    input_n: int = None,
    circuit_hidden_dim: int = 16,
    arity: int = 2,
    max_steps: int = 15,
    damage_behavior: str = "reversible",
) -> Optional[Dict]:
    """
    Helper function to perturb and recover a single circuit.
    
    Supports two recovery modes:
    - "backprop": Uses gradient descent with optax optimizer
    - "self_attention": Uses iterative message passing with early stopping based on hard_accuracy
    
    Args:
        initial_logits: Initial circuit logits
        knockout_pattern: Boolean array indicating damaged gates
        wires: Circuit wiring
        x_data: Input data
        y_data: Target output data
        layer_sizes: Circuit layer sizes
        loss_type: Loss function type
        functional_threshold: Minimum accuracy to consider functional
        opt: Optimizer (required for backprop mode)
        epochs: Number of training epochs (backprop mode)
        reversible_bias: Bias value for reversible damage (backprop mode)
        recovery_mode: "backprop" or "self_attention"
        model: CircuitSelfAttention model (required for self_attention mode)
        input_n: Number of input nodes (required for self_attention mode)
        circuit_hidden_dim: Hidden dimension for graph (self_attention mode)
        arity: Gate arity (self_attention mode)
        max_steps: Maximum message passing steps (self_attention mode)
        damage_behavior: "reversible" or "permanent" (self_attention mode)
    
    Returns:
        Dictionary with recovery results:
        - params: Recovered logits
        - final_hard_accuracy: Final accuracy achieved
        - steps_taken: Number of steps taken (for self_attention mode)
        or None if error occurred
    """
    try:
        if recovery_mode == "backprop":
            if opt is None:
                raise ValueError("opt parameter required for backprop recovery mode")
            result = _train_single_knockout_pattern(
                initial_logits=initial_logits,
                knockout_pattern=knockout_pattern,
                opt=opt,
                wires=wires,
                x_data=x_data,
                y_data=y_data,
                loss_type=loss_type,
                layer_sizes=layer_sizes,
                epochs=epochs,
                damage_behavior="reversible",
                reversible_bias=reversible_bias,
            )
            return result
        
        elif recovery_mode == "self_attention":
            if model is None:
                raise ValueError("model parameter required for self_attention recovery mode")
            if input_n is None:
                # Infer from layer_sizes
                input_n = layer_sizes[0][0] if layer_sizes else 8
            
            # Create generator with knockout pattern
            generator = evaluate_model_stepwise_generator(
                model=model,
                wires=wires,
                logits=initial_logits,
                x_data=x_data,
                y_data=y_data,
                input_n=input_n,
                arity=arity,
                circuit_hidden_dim=circuit_hidden_dim,
                max_steps=max_steps,
                loss_type=loss_type,
                bidirectional_edges=True,
                layer_sizes=layer_sizes,
                layer_neighbors=False,
                knockout_pattern=knockout_pattern,
                reset_step_counter_on_init=True,  # Reset to enable reversible mode on first step
            )
            
            # Consume initial state (step 0)
            initial_result = next(generator)
            
            # Iterate through steps, checking hard_accuracy at each step
            final_result = initial_result
            steps_taken = 0
            
            for step_result in generator:
                steps_taken += 1
                final_result = step_result
                
                # Check if we've reached functional threshold
                if final_result.hard_accuracy >= functional_threshold:
                    log.debug(
                        f"Early stopping at step {steps_taken}: "
                        f"hard_accuracy={final_result.hard_accuracy:.4f} >= {functional_threshold}"
                    )
                    break
            
            # Extract logits from final graph state
            from boolean_nca_cc.training.evaluation import get_loss_and_update_graph
            logits_original_shapes = [logit.shape for logit in initial_logits]
            _, _, recovered_logits, _ = get_loss_and_update_graph(
                final_result.graph,
                logits_original_shapes,
                wires,
                x_data,
                y_data,
                loss_type,
                layer_sizes,
            )
            
            return {
                "params": recovered_logits,
                "final_hard_accuracy": final_result.hard_accuracy,
                "steps_taken": steps_taken,
            }
        
        else:
            raise ValueError(f"Unknown recovery_mode: {recovery_mode}")
    
    except Exception as e:
        log.debug(f"Error during perturbation-recovery: {e}")
        return None


def parse_phase_specification(phase_str: str) -> List[Tuple[str, int, int, str, Optional[int]]]:
    """
    Parse phase specification string into list of phase tuples.
    
    Format: "type:depth:perturbations:start_from[:max_nodes],..."
    - type: "bfs" or "dfs"
    - depth: maximum depth for this phase
    - perturbations: number of perturbations per node
    - start_from: "root", "frontier", "all", or "frontier:N" (N = max nodes to sample)
    - max_nodes: Optional, only used when start_from="frontier:N"
    
    Examples:
        "bfs:2:100:root,dfs:5:1:frontier"  # e1: BFS then DFS from all frontier nodes
        "dfs:50:1:root,bfs:2:100:frontier:1"  # e2: DFS then BFS from 1 random frontier node
        "bfs:2:100:root,dfs:50:1:frontier,bfs:2:100:frontier,dfs:50:1:frontier:1"  # e3: complex pattern
    
    Args:
        phase_str: Phase specification string
        
    Returns:
        List of (phase_type, depth_limit, perturbations_per_node, start_from, max_frontier_nodes) tuples
        max_frontier_nodes is None if not specified or if start_from != "frontier"
    """
    phases = []
    for phase_part in phase_str.split(','):
        parts = phase_part.strip().split(':')
        if len(parts) < 4 or len(parts) > 5:
            raise ValueError(
                f"Invalid phase format: {phase_part}. "
                f"Expected 'type:depth:perturbations:start_from[:max_nodes]'"
            )
        
        phase_type, depth_str, perturbations_str, start_from = parts[:4]
        max_frontier_nodes = None
        
        # Parse frontier:N format
        # Handle two formats:
        # 1. "bfs:2:100:frontier:1" (5 parts, parts[3]="frontier", parts[4]="1")
        # 2. "bfs:2:100:frontier:1" where start_from could be "frontier:1" if there are only 4 parts
        if start_from.startswith("frontier"):
            if len(parts) == 5 and parts[3] == "frontier":
                # Format 1: Separate field "frontier:1" -> parts[3]="frontier", parts[4]="1"
                try:
                    max_frontier_nodes = int(parts[4])
                except ValueError:
                    raise ValueError(f"Invalid frontier max_nodes: {parts[4]}. Must be an integer")
            elif ":" in start_from:
                # Format 2: Combined "frontier:1" in start_from field
                frontier_match = start_from.split(":", 1)
                if len(frontier_match) == 2 and frontier_match[0] == "frontier":
                    start_from = "frontier"
                    try:
                        max_frontier_nodes = int(frontier_match[1])
                    except ValueError:
                        raise ValueError(f"Invalid frontier max_nodes: {frontier_match[1]}. Must be an integer")
                else:
                    raise ValueError(f"Invalid frontier format: {start_from}")
            # else: start_from is just "frontier", max_frontier_nodes stays None
        
        if phase_type not in ["bfs", "dfs"]:
            raise ValueError(f"Invalid phase type: {phase_type}. Must be 'bfs' or 'dfs'")
        if start_from not in ["root", "frontier", "all"]:
            raise ValueError(
                f"Invalid start_from: {start_from}. Must be 'root', 'frontier', or 'all'"
            )
        
        phases.append((phase_type, int(depth_str), int(perturbations_str), start_from, max_frontier_nodes))
    return phases


def _execute_bfs_phase(
    start_nodes: List[Tuple[str, List[jp.ndarray]]],
    depth_limit: int,
    perturbations_per_node: int,
    base_depth: int,
    circuit_depths: Dict[str, int],
    unique_solutions: Dict[str, List[jp.ndarray]],
    discovered_circuits: List[Tuple[str, List[jp.ndarray]]],
    exploration_graph: Dict[str, List[Tuple[str, int, Dict]]],
    edges: List[Tuple[str, str, int, Dict]],
    exploration_results: List[Dict],
    knockout_vocabulary: List[jp.ndarray],
    wires: List[jp.ndarray],
    x_data: jp.ndarray,
    y_data: jp.ndarray,
    layer_sizes: List[Tuple[int, int]],
    loss_type: str,
    functional_threshold: float,
    phase_name: str,
    # Recovery parameters
    recovery_mode: str = "backprop",
    opt: Optional[optax.GradientTransformation] = None,
    epochs: int = 200,
    reversible_bias: float = -10.0,
    model: Optional[CircuitSelfAttention] = None,
    input_n: int = None,
    circuit_hidden_dim: int = 16,
    arity: int = 2,
    max_steps: int = 15,
    damage_behavior: str = "reversible",
) -> Tuple[int, int]:
    """
    Execute a BFS phase starting from given nodes.
    
    Returns:
        Tuple of (total_perturbations, functional_recoveries)
    """
    total_perturbations = 0
    functional_recoveries = 0
    
    # BFS queue: (circuit_hash, circuit_logits, depth)
    queue = deque([(hash_val, logits, base_depth) for hash_val, logits in start_nodes])
    visited_at_depth: Dict[int, Set[str]] = {}
    
    while queue:
        circuit_hash, circuit_logits, depth = queue.popleft()
        
        if depth >= base_depth + depth_limit:
            continue
        
        # Track visited circuits at this depth
        if depth not in visited_at_depth:
            visited_at_depth[depth] = set()
        if circuit_hash in visited_at_depth[depth]:
            continue
        visited_at_depth[depth].add(circuit_hash)
        
        log.info(
            f"\n{phase_name} Depth {depth}: Exploring from circuit {circuit_hash[:16]}... "
            f"(Discovered: {len(unique_solutions)})"
        )
        
        # Sample perturbations for this node
        level_rng = jax.random.PRNGKey(42 + depth)
        num_patterns = min(perturbations_per_node, len(knockout_vocabulary))
        pattern_indices = jax.random.choice(
            level_rng, len(knockout_vocabulary), shape=(num_patterns,), replace=False
        )
        
        for pattern_idx in pattern_indices:
            knockout_pattern = knockout_vocabulary[pattern_idx]
            total_perturbations += 1
            
            result = _perturb_and_recover_single(
                initial_logits=circuit_logits,
                knockout_pattern=knockout_pattern,
                wires=wires,
                x_data=x_data,
                y_data=y_data,
                layer_sizes=layer_sizes,
                loss_type=loss_type,
                functional_threshold=functional_threshold,
                recovery_mode=recovery_mode,
                opt=opt,
                epochs=epochs,
                reversible_bias=reversible_bias,
                model=model,
                input_n=input_n,
                circuit_hidden_dim=circuit_hidden_dim,
                arity=arity,
                max_steps=max_steps,
                damage_behavior=damage_behavior,
            )
            
            if result is None:
                continue
            
            recovered_logits = result["params"]
            final_accuracy = float(result["final_hard_accuracy"])
            
            # Check if functional
            is_func = is_functional(
                recovered_logits, wires, x_data, y_data, functional_threshold
            )
            
            # Check uniqueness
            recovered_hash = hash_circuit_logits(recovered_logits)
            is_unique = recovered_hash not in unique_solutions
            
            # Track edge in exploration graph
            edge_metadata = {
                "depth": depth,
                "phase": phase_name,
                "is_cycle": (recovered_hash == circuit_hash),
                "is_revisit": (recovered_hash in unique_solutions),
                "final_accuracy": final_accuracy,
                "is_functional": is_func,
            }
            edges.append((circuit_hash, recovered_hash, int(pattern_idx), edge_metadata))
            
            if circuit_hash not in exploration_graph:
                exploration_graph[circuit_hash] = []
            exploration_graph[circuit_hash].append((recovered_hash, int(pattern_idx), edge_metadata))
            
            if is_func:
                functional_recoveries += 1
                
                # Add to unique solutions
                if is_unique:
                    unique_solutions[recovered_hash] = recovered_logits
                    discovered_circuits.append((recovered_hash, recovered_logits))
                    circuit_depths[recovered_hash] = depth + 1
                
                # Add to queue for next depth
                if depth + 1 < base_depth + depth_limit:
                    next_depth_visited = visited_at_depth.get(depth + 1, set())
                    if recovered_hash not in next_depth_visited:
                        queue.append((recovered_hash, recovered_logits, depth + 1))
            
            exploration_results.append({
                "pattern_idx": int(pattern_idx),
                "source_hash": circuit_hash,
                "recovered_hash": recovered_hash,
                "final_accuracy": final_accuracy,
                "is_functional": is_func,
                "is_unique": is_func and is_unique,
                "depth": depth,
                "phase": phase_name,
            })
    
    return total_perturbations, functional_recoveries


def _execute_dfs_phase(
    start_nodes: List[Tuple[str, List[jp.ndarray]]],
    depth_limit: int,
    perturbations_per_node: int,
    base_depth: int,
    circuit_depths: Dict[str, int],
    unique_solutions: Dict[str, List[jp.ndarray]],
    discovered_circuits: List[Tuple[str, List[jp.ndarray]]],
    exploration_graph: Dict[str, List[Tuple[str, int, Dict]]],
    edges: List[Tuple[str, str, int, Dict]],
    exploration_results: List[Dict],
    knockout_vocabulary: List[jp.ndarray],
    wires: List[jp.ndarray],
    x_data: jp.ndarray,
    y_data: jp.ndarray,
    layer_sizes: List[Tuple[int, int]],
    loss_type: str,
    functional_threshold: float,
    phase_name: str,
    # Recovery parameters
    recovery_mode: str = "backprop",
    opt: Optional[optax.GradientTransformation] = None,
    epochs: int = 200,
    reversible_bias: float = -10.0,
    model: Optional[CircuitSelfAttention] = None,
    input_n: int = None,
    circuit_hidden_dim: int = 16,
    arity: int = 2,
    max_steps: int = 15,
    damage_behavior: str = "reversible",
    max_retry_attempts: int = 0,
    # Checkpoint callback
    checkpoint_callback: Optional[Callable] = None,
    root_hash: Optional[str] = None,
) -> Tuple[int, int]:
    """
    Execute a DFS phase starting from given nodes.
    
    Args:
        max_retry_attempts: When perturbations_per_node==1 and a perturbation fails,
            retry up to this many times with different patterns before giving up.
            Default 0 (no retries).
    
    Returns:
        Tuple of (total_perturbations, functional_recoveries)
    """
    total_perturbations = 0
    functional_recoveries = 0
    
    # DFS stack: (circuit_hash, circuit_logits, depth)
    stack = [(hash_val, logits, base_depth) for hash_val, logits in start_nodes]
    visited_dfs: Set[str] = set()
    
    while stack:
        circuit_hash, circuit_logits, depth = stack.pop()
        
        if depth >= base_depth + depth_limit:
            continue
        if circuit_hash in visited_dfs:
            continue
        visited_dfs.add(circuit_hash)
        
        log.info(
            f"\n{phase_name} Depth {depth}: Exploring from circuit {circuit_hash[:16]}... "
            f"(Discovered: {len(unique_solutions)})"
        )
        
        # Periodic checkpointing (every N solutions discovered)
        if checkpoint_callback is not None and len(unique_solutions) > 0:
            try:
                checkpoint_callback(
                    unique_solutions=unique_solutions,
                    exploration_graph=exploration_graph,
                    edges=edges,
                    exploration_results=exploration_results,
                    root_hash=root_hash,
                )
            except Exception as e:
                log.warning(f"Error during checkpoint: {e}")
        
        # Sample perturbations for this node
        level_rng = jax.random.PRNGKey(42 + depth)
        num_patterns = min(perturbations_per_node, len(knockout_vocabulary))
        pattern_indices = jax.random.choice(
            level_rng, len(knockout_vocabulary), shape=(num_patterns,), replace=False
        )
        
        # Track if we got a successful functional recovery (for retry logic)
        successful_recovery = None
        # Track all unique functional recoveries to add to stack
        unique_functional_recoveries = []
        
        # Process perturbations in reverse order so we explore first one first (LIFO)
        for pattern_idx in reversed(pattern_indices):
            knockout_pattern = knockout_vocabulary[pattern_idx]
            total_perturbations += 1
            
            result = _perturb_and_recover_single(
                initial_logits=circuit_logits,
                knockout_pattern=knockout_pattern,
                wires=wires,
                x_data=x_data,
                y_data=y_data,
                layer_sizes=layer_sizes,
                loss_type=loss_type,
                functional_threshold=functional_threshold,
                recovery_mode=recovery_mode,
                opt=opt,
                epochs=epochs,
                reversible_bias=reversible_bias,
                model=model,
                input_n=input_n,
                circuit_hidden_dim=circuit_hidden_dim,
                arity=arity,
                max_steps=max_steps,
                damage_behavior=damage_behavior,
            )
            
            if result is None:
                continue
            
            recovered_logits = result["params"]
            final_accuracy = float(result["final_hard_accuracy"])
            
            # Check if functional
            is_func = is_functional(
                recovered_logits, wires, x_data, y_data, functional_threshold
            )
            
            # Check uniqueness
            recovered_hash = hash_circuit_logits(recovered_logits)
            is_unique = recovered_hash not in unique_solutions
            
            # Track edge in exploration graph
            edge_metadata = {
                "depth": depth,
                "phase": phase_name,
                "is_cycle": (recovered_hash == circuit_hash),
                "is_revisit": (recovered_hash in unique_solutions),
                "final_accuracy": final_accuracy,
                "is_functional": is_func,
            }
            edges.append((circuit_hash, recovered_hash, int(pattern_idx), edge_metadata))
            
            if circuit_hash not in exploration_graph:
                exploration_graph[circuit_hash] = []
            exploration_graph[circuit_hash].append((recovered_hash, int(pattern_idx), edge_metadata))
            
            if is_func:
                functional_recoveries += 1
                successful_recovery = (recovered_hash, recovered_logits)
                
                # Add to unique solutions
                if is_unique:
                    unique_solutions[recovered_hash] = recovered_logits
                    discovered_circuits.append((recovered_hash, recovered_logits))
                    circuit_depths[recovered_hash] = depth + 1
                    # Track unique functional recoveries for DFS continuation
                    unique_functional_recoveries.append((recovered_hash, recovered_logits))
                
                # For single perturbation mode, break early after first success
                if perturbations_per_node == 1:
                    break
            
            exploration_results.append({
                "pattern_idx": int(pattern_idx),
                "source_hash": circuit_hash,
                "recovered_hash": recovered_hash,
                "final_accuracy": final_accuracy,
                "is_functional": is_func,
                "is_unique": is_func and is_unique,
                "depth": depth,
                "phase": phase_name,
            })
        
        # Retry logic: if perturbations_per_node==1 and we didn't get a functional recovery
        if perturbations_per_node == 1 and successful_recovery is None and max_retry_attempts > 0:
            # Sample additional patterns for retry (exclude already tried patterns)
            retry_rng = jax.random.PRNGKey(42 + depth + 1000)  # Different seed for retries
            tried_patterns = set(int(idx) for idx in pattern_indices)
            remaining_patterns = [i for i in range(len(knockout_vocabulary)) if i not in tried_patterns]
            
            if remaining_patterns:
                num_retries = min(max_retry_attempts, len(remaining_patterns))
                retry_indices = jax.random.choice(
                    retry_rng, len(remaining_patterns), shape=(num_retries,), replace=False
                )
                retry_pattern_indices = [remaining_patterns[int(idx)] for idx in retry_indices]
                
                for retry_idx, pattern_idx in enumerate(retry_pattern_indices):
                    knockout_pattern = knockout_vocabulary[pattern_idx]
                    total_perturbations += 1
                    
                    result = _perturb_and_recover_single(
                        initial_logits=circuit_logits,
                        knockout_pattern=knockout_pattern,
                        wires=wires,
                        x_data=x_data,
                        y_data=y_data,
                        layer_sizes=layer_sizes,
                        loss_type=loss_type,
                        functional_threshold=functional_threshold,
                        recovery_mode=recovery_mode,
                        opt=opt,
                        epochs=epochs,
                        reversible_bias=reversible_bias,
                        model=model,
                        input_n=input_n,
                        circuit_hidden_dim=circuit_hidden_dim,
                        arity=arity,
                        max_steps=max_steps,
                        damage_behavior=damage_behavior,
                    )
                    
                    if result is None:
                        continue
                    
                    recovered_logits = result["params"]
                    final_accuracy = float(result["final_hard_accuracy"])
                    
                    is_func = is_functional(
                        recovered_logits, wires, x_data, y_data, functional_threshold
                    )
                    recovered_hash = hash_circuit_logits(recovered_logits)
                    is_unique = recovered_hash not in unique_solutions
                    
                    edge_metadata = {
                        "depth": depth,
                        "phase": phase_name,
                        "is_cycle": (recovered_hash == circuit_hash),
                        "is_revisit": (recovered_hash in unique_solutions),
                        "final_accuracy": final_accuracy,
                        "is_functional": is_func,
                        "is_retry": True,
                        "retry_attempt": retry_idx + 1,
                    }
                    edges.append((circuit_hash, recovered_hash, int(pattern_idx), edge_metadata))
                    
                    if circuit_hash not in exploration_graph:
                        exploration_graph[circuit_hash] = []
                    exploration_graph[circuit_hash].append((recovered_hash, int(pattern_idx), edge_metadata))
                    
                    exploration_results.append({
                        "pattern_idx": int(pattern_idx),
                        "source_hash": circuit_hash,
                        "recovered_hash": recovered_hash,
                        "final_accuracy": final_accuracy,
                        "is_functional": is_func,
                        "is_unique": is_func and is_unique,
                        "depth": depth,
                        "phase": phase_name,
                        "is_retry": True,
                        "retry_attempt": retry_idx + 1,
                    })
                    
                    if is_func:
                        functional_recoveries += 1
                        successful_recovery = (recovered_hash, recovered_logits)
                        
                        if is_unique:
                            unique_solutions[recovered_hash] = recovered_logits
                            discovered_circuits.append((recovered_hash, recovered_logits))
                            circuit_depths[recovered_hash] = depth + 1
                            # Track unique functional recoveries for DFS continuation
                            unique_functional_recoveries.append((recovered_hash, recovered_logits))
                        
                        # Stop retrying once we get a successful recovery
                        break
        
        # Add to stack for next depth: add all unique functional recoveries
        # This allows DFS to continue even if some recoveries were revisits
        if unique_functional_recoveries:
            if depth + 1 < base_depth + depth_limit:
                for recovered_hash, recovered_logits in unique_functional_recoveries:
                    if recovered_hash not in visited_dfs:
                        stack.append((recovered_hash, recovered_logits, depth + 1))
        elif successful_recovery is not None:
            # Fallback: if no unique recoveries but we have a functional recovery, use it
            # (this handles the case where all recoveries were revisits but we still want to continue)
            recovered_hash, recovered_logits = successful_recovery
            if depth + 1 < base_depth + depth_limit:
                if recovered_hash not in visited_dfs:
                    stack.append((recovered_hash, recovered_logits, depth + 1))
    
    return total_perturbations, functional_recoveries


def compute_per_depth_statistics(exploration_results: List[Dict]) -> Dict[int, Dict]:
    """
    Compute per-depth statistics from exploration results.
    
    Args:
        exploration_results: List of exploration result dictionaries
        
    Returns:
        Dictionary mapping depth -> {
            "total_attempts": int,
            "functional_recoveries": int,
            "failed_attempts": int,
            "success_rate": float
        }
    """
    depth_stats = defaultdict(lambda: {
        "total_attempts": 0,
        "functional_recoveries": 0,
        "failed_attempts": 0,
    })
    
    for result in exploration_results:
        depth = result.get("depth")
        if depth is None:
            # Skip results without depth (e.g., random_walk)
            continue
        
        depth_stats[depth]["total_attempts"] += 1
        if result.get("is_functional", False):
            depth_stats[depth]["functional_recoveries"] += 1
        else:
            depth_stats[depth]["failed_attempts"] += 1
    
    # Compute success rates and convert to regular dict
    per_depth_stats = {}
    for depth in sorted(depth_stats.keys()):
        stats = depth_stats[depth]
        stats["success_rate"] = (
            stats["functional_recoveries"] / stats["total_attempts"]
            if stats["total_attempts"] > 0 else 0.0
        )
        per_depth_stats[depth] = stats
    
    return per_depth_stats


def explore_degenerate_solutions(
    root_wires: List[jp.ndarray],
    root_logits: List[jp.ndarray],
    x_data: jp.ndarray,
    y_data: jp.ndarray,
    layer_sizes: List[Tuple[int, int]],
    num_perturbations: int = 100,
    damage_prob: float = 20.0,
    greedy_indices: List[int] = None,
    epochs: int = 200,
    learning_rate: float = 1.0,
    optimizer: str = "adamw",
    weight_decay: float = 1e-1,
    beta1: float = 0.8,
    beta2: float = 0.8,
    loss_type: str = "l4",
    functional_threshold: float = 0.999,
    reversible_bias: float = -10.0,
    exploration_strategy: str = "phases",
    bfs_depth: int = 2,
    bfs_perturbations_per_level: int = 100,
    random_walk_iterations: int = 500,
    random_walk_seed: int = 12345,
    phases: Optional[List[Tuple[str, int, int, str, Optional[int]]]] = None,
    # Self-attention recovery parameters
    recovery_mode: str = "backprop",
    model: Optional[CircuitSelfAttention] = None,
    input_n: Optional[int] = None,
    circuit_hidden_dim: int = 64,
    arity: int = 4,
    max_steps: int = 15,
    damage_behavior: str = "reversible",
    max_retry_attempts: int = 0,
    # Checkpoint parameters
    output_dir: Optional[Path] = None,
    checkpoint_interval: int = 1000,
    task_name: str = "binary_multiply",
    exploration_config: Optional[Dict] = None,
) -> Dict:
    """
    Explore degenerate solutions using flexible phase-based or legacy strategies.
    
    Strategy options:
    - "phases": Phase-based exploration (most flexible)
        - Use `phases` parameter to specify exploration pattern
        - Format: List of (phase_type, depth_limit, perturbations_per_node, start_from, max_frontier_nodes)
        - phase_type: "bfs" or "dfs"
        - start_from: "root", "frontier", or "all"
        - max_frontier_nodes: Optional int, limits random sampling when start_from="frontier"
        - Examples:
          * e1 (BFS→DFS): [("bfs", 2, 100, "root", None), ("dfs", 5, 1, "frontier", None)]
          * e2 (DFS→BFS→DFS): [("dfs", 50, 1, "root", None), ("bfs", 2, 100, "frontier", None), ("dfs", 50, 1, "frontier", 1)]
          * e3 (alternating): [("dfs", K1, 1, "root", None), ("bfs", D1, N1, "frontier", None), ...]
    - "hybrid": BFS phase (depth 1-2) followed by Random Walk (recommended for UMAP)
    - "bfs": Breadth-first search only
    - "random_walk": Random walk only
    
    Recovery modes:
    - "backprop": Uses gradient descent with optax optimizer (default)
    - "self_attention": Uses iterative message passing with early stopping based on hard_accuracy
    
    Args:
        root_wires: Root circuit wiring
        root_logits: Root circuit logits
        x_data: Input data
        y_data: Target output data
        layer_sizes: Circuit layer sizes
        num_perturbations: Number of perturbation patterns to try (legacy, used if strategy="single_root")
        damage_prob: Number of gates to damage per perturbation
        greedy_indices: Ordered list of greedy gate indices (defaults to DEFAULT_GREEDY_ORDERED_INDICES)
        epochs: Number of recovery epochs (backprop mode)
        learning_rate: Learning rate for recovery (default: 1.0, matching config.yaml)
        optimizer: Optimizer type ("adamw" or "adam", default: "adamw")
        weight_decay: Weight decay for optimizer (default: 1e-1)
        beta1: Beta1 parameter for optimizer (default: 0.8)
        beta2: Beta2 parameter for optimizer (default: 0.8)
        loss_type: Loss type for recovery
        functional_threshold: Minimum accuracy to consider circuit functional (default: 0.999)
        reversible_bias: Bias value for reversible damage mode (backprop mode)
        exploration_strategy: Strategy to use ("hybrid", "bfs", "random_walk", or "single_root")
        bfs_depth: Maximum depth for BFS phase (default: 2)
        bfs_perturbations_per_level: Number of perturbations to try per BFS level (default: 100)
        random_walk_iterations: Number of random walk iterations (default: 500)
        random_walk_seed: Random seed for random walk phase
        recovery_mode: Recovery mode ("backprop" or "self_attention", default: "backprop")
        model: CircuitSelfAttention model (required for self_attention mode)
        input_n: Number of input nodes (inferred from layer_sizes if not provided)
        circuit_hidden_dim: Hidden dimension for graph (self_attention mode, default: 16)
        arity: Gate arity (self_attention mode, default: 2)
        max_steps: Maximum message passing steps (self_attention mode, default: 15)
        damage_behavior: Damage behavior ("reversible" or "permanent", self_attention mode, default: "reversible")
        max_retry_attempts: When DFS perturbations_per_node==1 and a perturbation fails,
            retry up to this many times with different patterns before giving up.
            Default 0 (no retries).
        
    Returns:
        Dictionary with exploration results including:
        - unique_solutions: Dict[hash -> logits] of unique circuits
        - exploration_results: List of all perturbation-recovery results
        - exploration_graph: Dict[source_hash -> List[(target_hash, pattern_idx, metadata)]]
        - edges: List of all edges (source_hash, target_hash, pattern_idx, metadata)
        - summary: Statistics about exploration
    """
    if greedy_indices is None:
        greedy_indices = DEFAULT_GREEDY_ORDERED_INDICES
    
    # Infer input_n from layer_sizes if not provided
    if input_n is None:
        input_n = layer_sizes[0][0] if layer_sizes else 8
    
    log.info("=" * 80)
    log.info("Exploring Degenerate Circuit Solutions")
    log.info("=" * 80)
    log.info(f"Exploration strategy: {exploration_strategy}")
    log.info(f"Recovery mode: {recovery_mode}")
    log.info(f"Root circuit hash: {hash_circuit_logits(root_logits)}")
    log.info(f"Damage per perturbation: {damage_prob} gates")
    log.info(f"Functional threshold: {functional_threshold}")
    
    if recovery_mode == "backprop":
        log.info(f"Recovery epochs: {epochs}")
        log.info(f"Learning rate: {learning_rate}")
        log.info(f"Optimizer: {optimizer}")
        log.info(f"Weight decay: {weight_decay}")
        log.info(f"Beta1: {beta1}, Beta2: {beta2}")
        # Setup optimizer for recovery (matching config.yaml backprop settings)
        if optimizer == "adamw":
            opt = optax.adamw(
                learning_rate,
                b1=beta1,
                b2=beta2,
                weight_decay=weight_decay,
            )
        else:
            opt = optax.adam(learning_rate, b1=beta1, b2=beta2)
    elif recovery_mode == "self_attention":
        if model is None:
            raise ValueError("model parameter required when recovery_mode='self_attention'")
        log.info(f"Self-attention recovery:")
        log.info(f"  Max steps: {max_steps}")
        log.info(f"  Circuit hidden dim: {circuit_hidden_dim}")
        log.info(f"  Arity: {arity}")
        log.info(f"  Input n: {input_n}")
        log.info(f"  Damage behavior: {damage_behavior}")
        opt = None  # Not used for self-attention
    else:
        raise ValueError(f"Unknown recovery_mode: {recovery_mode}")
    
    # Track unique solutions (for diversity analysis)
    unique_solutions: Dict[str, List[jp.ndarray]] = {}  # hash -> logits
    root_hash = hash_circuit_logits(root_logits)
    unique_solutions[root_hash] = root_logits
    
    # Track exploration graph structure (allows cycles and revisits)
    exploration_graph: Dict[str, List[Tuple[str, int, Dict]]] = {}  # source_hash -> [(target_hash, pattern_idx, metadata)]
    edges: List[Tuple[str, str, int, Dict]] = []  # (source_hash, target_hash, pattern_idx, metadata)
    
    # Track exploration results
    exploration_results = []
    successful_recoveries = 0
    functional_recoveries = 0
    total_perturbations = 0
    
    # Generate large vocabulary for random sampling
    vocab_rng = jax.random.PRNGKey(42)
    # Calculate vocabulary size needed
    vocab_size = max(num_perturbations, bfs_perturbations_per_level * bfs_depth, random_walk_iterations)
    # For phase-based exploration, estimate vocabulary size from phases
    if exploration_strategy == "phases" and phases is not None:
        max_perturbations = max(
            perturbations_per_node for _, _, perturbations_per_node, _, _ in phases
        )
        vocab_size = max(vocab_size, max_perturbations * 2)  # Add some buffer
    knockout_vocabulary = create_knockout_vocabulary(
        rng=vocab_rng,
        vocabulary_size=vocab_size,
        layer_sizes=layer_sizes,
        damage_prob=damage_prob,
        damage_mode="greedy_vocabulary",
        ordered_indices=greedy_indices,
    )
    log.info(f"Generated vocabulary of {len(knockout_vocabulary)} perturbation patterns")
    
    # Track discovered circuits for random walk (list of (hash, logits) tuples)
    discovered_circuits: List[Tuple[str, List[jp.ndarray]]] = [(root_hash, root_logits)]
    
    # Track circuit depths for phase-based exploration
    circuit_depths: Dict[str, int] = {root_hash: 0}
    
    # Create checkpoint callback if output_dir is provided
    checkpoint_callback = None
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        def create_checkpoint_callback(interval, wires, layer_sizes, task_name, exploration_config):
            last_checkpoint_count = [0]  # Use list to allow modification in closure
            def callback(**kwargs):
                current_count = len(kwargs.get("unique_solutions", {}))
                if current_count - last_checkpoint_count[0] >= interval:
                    # Build results dictionary in the same format as explore_degenerate_solutions returns
                    exploration_results_list = kwargs.get("exploration_results", [])
                    unique_solutions_dict = kwargs.get("unique_solutions", {})
                    total_perturbations = len(exploration_results_list)
                    functional_recoveries = sum(1 for r in exploration_results_list if r.get("is_functional", False))
                    perturbation_efficiency = len(unique_solutions_dict) / total_perturbations if total_perturbations > 0 else 0.0
                    
                    results_dict = {
                        "unique_solutions": unique_solutions_dict,
                        "exploration_results": exploration_results_list,
                        "exploration_graph": kwargs.get("exploration_graph", {}),
                        "edges": kwargs.get("edges", []),
                        "summary": {
                            "total_perturbations": total_perturbations,
                            "successful_recoveries": total_perturbations,  # Approximate (all perturbations that returned results)
                            "functional_recoveries": functional_recoveries,
                            "unique_solutions": len(unique_solutions_dict),
                            "perturbation_efficiency": perturbation_efficiency,
                            "strategy": exploration_config.get("exploration_strategy", "phases"),
                            "per_depth_stats": compute_per_depth_statistics(exploration_results_list),
                        },
                        "root_hash": kwargs.get("root_hash", root_hash),
                    }
                    # Use save_exploration_results to maintain same format
                    save_exploration_results(
                        results=results_dict,
                        output_dir=output_dir,
                        wires=wires,
                        layer_sizes=layer_sizes,
                        task_name=task_name,
                        exploration_config=exploration_config or {},
                    )
                    last_checkpoint_count[0] = current_count
                    log.info(f"Checkpoint saved: {current_count} solutions in {output_dir}")
            return callback
        
        checkpoint_callback = create_checkpoint_callback(
            checkpoint_interval, root_wires, layer_sizes, task_name, exploration_config or {}
        )
        log.info(f"Checkpointing enabled: {output_dir} (every {checkpoint_interval} solutions)")
    
    # ============================================================================
    # Phase-Based Exploration (if strategy is "phases")
    # ============================================================================
    if exploration_strategy == "phases":
        if phases is None:
            raise ValueError("phases parameter required when exploration_strategy='phases'")
        
        log.info("\n" + "=" * 80)
        log.info("Phase-Based Exploration")
        log.info("=" * 80)
        log.info(f"Number of phases: {len(phases)}")
        for phase_idx, phase_tuple in enumerate(phases):
            phase_type, depth_limit, perturbations_per_node, start_from, max_frontier_nodes = phase_tuple
            frontier_info = f", max_frontier_nodes={max_frontier_nodes}" if max_frontier_nodes is not None else ""
            log.info(
                f"  Phase {phase_idx + 1}: {phase_type.upper()} "
                f"(depth={depth_limit}, perturbations={perturbations_per_node}, start_from={start_from}{frontier_info})"
            )
        
        current_max_depth = 0
        
        for phase_idx, phase_tuple in enumerate(phases):
            phase_type, depth_limit, perturbations_per_node, start_from, max_frontier_nodes = phase_tuple
            phase_name = f"Phase{phase_idx + 1}_{phase_type.upper()}"
            
            log.info("\n" + "=" * 80)
            log.info(f"{phase_name}: {phase_type.upper()} (depth_limit={depth_limit}, "
                    f"perturbations_per_node={perturbations_per_node}, start_from={start_from})")
            log.info("=" * 80)
            
            # Determine starting nodes
            if start_from == "root":
                start_nodes = [(root_hash, root_logits)]
                base_depth = 0
            elif start_from == "frontier":
                if phase_idx == 0:
                    # First phase: start from root
                    start_nodes = [(root_hash, root_logits)]
                    base_depth = 0
                else:
                    # Find nodes at maximum depth from previous phases
                    max_depth = max(circuit_depths.values())
                    frontier_nodes = [
                        (h, logits) for h, logits in discovered_circuits
                        if circuit_depths.get(h, 0) == max_depth
                    ]
                    
                    # Sample if max_frontier_nodes is specified
                    if max_frontier_nodes is not None and max_frontier_nodes < len(frontier_nodes):
                        # Randomly sample frontier nodes (using phase_idx for reproducibility)
                        random.seed(42 + phase_idx)  # Reproducible sampling per phase
                        start_nodes = random.sample(
                            frontier_nodes, 
                            min(max_frontier_nodes, len(frontier_nodes))
                        )
                        log.info(
                            f"  Randomly sampled {len(start_nodes)} from {len(frontier_nodes)} "
                            f"frontier nodes at depth {max_depth}"
                        )
                    else:
                        start_nodes = frontier_nodes
                        log.info(f"  Starting from {len(start_nodes)} frontier nodes at depth {max_depth}")
                    
                    base_depth = max_depth
            elif start_from == "all":
                start_nodes = discovered_circuits
                base_depth = min(circuit_depths.values()) if circuit_depths else 0
                log.info(f"  Starting from {len(start_nodes)} discovered circuits")
            else:
                raise ValueError(f"Invalid start_from: {start_from}")
            
            if not start_nodes:
                log.warning(f"  No starting nodes for {phase_name}, skipping")
                continue
            
            # Execute phase
            if phase_type == "bfs":
                phase_perturbations, phase_functional = _execute_bfs_phase(
                    start_nodes=start_nodes,
                    depth_limit=depth_limit,
                    perturbations_per_node=perturbations_per_node,
                    base_depth=base_depth,
                    circuit_depths=circuit_depths,
                    unique_solutions=unique_solutions,
                    discovered_circuits=discovered_circuits,
                    exploration_graph=exploration_graph,
                    edges=edges,
                    exploration_results=exploration_results,
                    knockout_vocabulary=knockout_vocabulary,
                    wires=root_wires,
                    x_data=x_data,
                    y_data=y_data,
                    layer_sizes=layer_sizes,
                    loss_type=loss_type,
                    functional_threshold=functional_threshold,
                    phase_name=phase_name,
                    recovery_mode=recovery_mode,
                    opt=opt,
                    epochs=epochs,
                    reversible_bias=reversible_bias,
                    model=model,
                    input_n=input_n,
                    circuit_hidden_dim=circuit_hidden_dim,
                    arity=arity,
                    max_steps=max_steps,
                    damage_behavior=damage_behavior,
                )
            elif phase_type == "dfs":
                phase_perturbations, phase_functional = _execute_dfs_phase(
                    start_nodes=start_nodes,
                    depth_limit=depth_limit,
                    perturbations_per_node=perturbations_per_node,
                    base_depth=base_depth,
                    circuit_depths=circuit_depths,
                    unique_solutions=unique_solutions,
                    discovered_circuits=discovered_circuits,
                    exploration_graph=exploration_graph,
                    edges=edges,
                    exploration_results=exploration_results,
                    knockout_vocabulary=knockout_vocabulary,
                    wires=root_wires,
                    x_data=x_data,
                    y_data=y_data,
                    layer_sizes=layer_sizes,
                    loss_type=loss_type,
                    functional_threshold=functional_threshold,
                    phase_name=phase_name,
                    recovery_mode=recovery_mode,
                    opt=opt,
                    epochs=epochs,
                    reversible_bias=reversible_bias,
                    model=model,
                    input_n=input_n,
                    circuit_hidden_dim=circuit_hidden_dim,
                    arity=arity,
                    max_steps=max_steps,
                    damage_behavior=damage_behavior,
                    max_retry_attempts=max_retry_attempts,
                    checkpoint_callback=checkpoint_callback,
                    root_hash=root_hash,
                )
            else:
                raise ValueError(f"Invalid phase_type: {phase_type}")
            
            total_perturbations += phase_perturbations
            functional_recoveries += phase_functional
            successful_recoveries += phase_perturbations  # Approximate (some may fail)
            
            current_max_depth = max(circuit_depths.values())
            
            log.info(f"\n{phase_name} Complete:")
            log.info(f"  Perturbations: {phase_perturbations}")
            log.info(f"  Functional recoveries: {phase_functional}")
            log.info(f"  Unique solutions: {len(unique_solutions)}")
            log.info(f"  Max depth reached: {current_max_depth}")
        
        log.info("\n" + "=" * 80)
        log.info("Phase-Based Exploration Complete")
        log.info("=" * 80)
    
    # ============================================================================
    # PHASE 1: BFS Exploration (if strategy is "hybrid" or "bfs")
    # ============================================================================
    elif exploration_strategy in ["hybrid", "bfs"]:
        log.info("\n" + "=" * 80)
        log.info("PHASE 1: Breadth-First Search")
        log.info("=" * 80)
        
        # BFS queue: (circuit_hash, circuit_logits, depth)
        queue = deque([(root_hash, root_logits, 0)])
        visited_at_depth: Dict[int, Set[str]] = {}  # Track visited circuits per depth
        
        while queue:
            circuit_hash, circuit_logits, depth = queue.popleft()
            
            if depth >= bfs_depth:
                continue
            
            # Track visited circuits at this depth
            if depth not in visited_at_depth:
                visited_at_depth[depth] = set()
            if circuit_hash in visited_at_depth[depth]:
                continue
            visited_at_depth[depth].add(circuit_hash)
            
            log.info(f"\nBFS Depth {depth}: Exploring from circuit {circuit_hash[:16]}...")
            log.info(f"  Discovered circuits so far: {len(unique_solutions)}")
            
            # Sample perturbations for this level
            level_rng = jax.random.PRNGKey(42 + depth)
            num_patterns = min(bfs_perturbations_per_level, len(knockout_vocabulary))
            pattern_indices = jax.random.choice(
                level_rng, len(knockout_vocabulary), shape=(num_patterns,), replace=False
            )
            
            new_circuits_at_depth = []
            
            for pattern_idx in pattern_indices:
                knockout_pattern = knockout_vocabulary[pattern_idx]
                total_perturbations += 1
                
                result = _perturb_and_recover_single(
                    initial_logits=circuit_logits,
                    knockout_pattern=knockout_pattern,
                    wires=root_wires,
                    x_data=x_data,
                    y_data=y_data,
                    layer_sizes=layer_sizes,
                    loss_type=loss_type,
                    functional_threshold=functional_threshold,
                    recovery_mode=recovery_mode,
                    opt=opt,
                    epochs=epochs,
                    reversible_bias=reversible_bias,
                    model=model,
                    input_n=input_n,
                    circuit_hidden_dim=circuit_hidden_dim,
                    arity=arity,
                    max_steps=max_steps,
                    damage_behavior=damage_behavior,
                )
                
                if result is None:
                    continue
                
                recovered_logits = result["params"]
                final_accuracy = float(result["final_hard_accuracy"])
                successful_recoveries += 1
                
                # Check if functional
                is_func = is_functional(
                    recovered_logits, root_wires, x_data, y_data, functional_threshold
                )
                
                # Check uniqueness
                recovered_hash = hash_circuit_logits(recovered_logits)
                is_unique = recovered_hash not in unique_solutions
                
                # Track edge in exploration graph (allows cycles)
                edge_metadata = {
                    "depth": depth,
                    "phase": "bfs",
                    "is_cycle": (recovered_hash == circuit_hash),
                    "is_revisit": (recovered_hash in unique_solutions),
                    "final_accuracy": final_accuracy,
                    "is_functional": is_func,
                }
                edges.append((circuit_hash, recovered_hash, int(pattern_idx), edge_metadata))
                
                if circuit_hash not in exploration_graph:
                    exploration_graph[circuit_hash] = []
                exploration_graph[circuit_hash].append((recovered_hash, int(pattern_idx), edge_metadata))
                
                if is_func:
                    functional_recoveries += 1
                    
                    # Add to unique solutions
                    if is_unique:
                        unique_solutions[recovered_hash] = recovered_logits
                        discovered_circuits.append((recovered_hash, recovered_logits))
                        new_circuits_at_depth.append(recovered_hash)
                        log.info(
                            f"  ✓ Unique functional solution! Hash: {recovered_hash[:16]}..., "
                            f"Accuracy: {final_accuracy:.4f}"
                        )
                    else:
                        log.info(
                            f"  → Recovered to known solution (hash: {recovered_hash[:16]}...), "
                            f"Accuracy: {final_accuracy:.4f}"
                        )
                    
                    # Add to queue for next depth (if not already visited at that depth)
                    if depth + 1 < bfs_depth:
                        next_depth_visited = visited_at_depth.get(depth + 1, set())
                        if recovered_hash not in next_depth_visited:
                            queue.append((recovered_hash, recovered_logits, depth + 1))
                else:
                    log.debug(
                        f"  ✗ Recovery failed (accuracy: {final_accuracy:.4f} < {functional_threshold})"
                    )
                
                exploration_results.append({
                    "pattern_idx": int(pattern_idx),
                    "source_hash": circuit_hash,
                    "recovered_hash": recovered_hash,
                    "final_accuracy": final_accuracy,
                    "is_functional": is_func,
                    "is_unique": is_func and is_unique,
                    "depth": depth,
                    "phase": "bfs",
                })
            
            log.info(f"  New unique circuits at depth {depth}: {len(new_circuits_at_depth)}")
        
        log.info(f"\nBFS Phase Complete:")
        log.info(f"  Total perturbations: {total_perturbations}")
        log.info(f"  Unique solutions discovered: {len(unique_solutions)}")
    
    # ============================================================================
    # PHASE 2: Random Walk Exploration (if strategy is "hybrid" or "random_walk")
    # ============================================================================
    if exploration_strategy in ["hybrid", "random_walk"]:
        log.info("\n" + "=" * 80)
        log.info("PHASE 2: Random Walk")
        log.info("=" * 80)
        log.info(f"Starting random walk with {len(discovered_circuits)} discovered circuits")
        log.info(f"Random walk iterations: {random_walk_iterations}")
        
        rw_rng = jax.random.PRNGKey(random_walk_seed)
        
        for iteration in range(random_walk_iterations):
            if iteration % 50 == 0:
                log.info(f"\nRandom Walk iteration {iteration + 1}/{random_walk_iterations}")
                log.info(f"  Discovered circuits: {len(unique_solutions)}")
            
            # Randomly select a circuit to perturb
            rw_rng, choice_key = jax.random.split(rw_rng)
            circuit_idx = int(jax.random.choice(choice_key, len(discovered_circuits)))
            circuit_hash, circuit_logits = discovered_circuits[circuit_idx]
            
            # Randomly select a perturbation pattern
            rw_rng, pattern_key = jax.random.split(rw_rng)
            pattern_idx = int(jax.random.choice(pattern_key, len(knockout_vocabulary)))
            knockout_pattern = knockout_vocabulary[pattern_idx]
            
            total_perturbations += 1
            
            result = _perturb_and_recover_single(
                initial_logits=circuit_logits,
                knockout_pattern=knockout_pattern,
                wires=root_wires,
                x_data=x_data,
                y_data=y_data,
                layer_sizes=layer_sizes,
                loss_type=loss_type,
                functional_threshold=functional_threshold,
                recovery_mode=recovery_mode,
                opt=opt,
                epochs=epochs,
                reversible_bias=reversible_bias,
                model=model,
                input_n=input_n,
                circuit_hidden_dim=circuit_hidden_dim,
                arity=arity,
                max_steps=max_steps,
                damage_behavior=damage_behavior,
            )
            
            if result is None:
                continue
            
            recovered_logits = result["params"]
            final_accuracy = float(result["final_hard_accuracy"])
            successful_recoveries += 1
            
            # Check if functional
            is_func = is_functional(
                recovered_logits, root_wires, x_data, y_data, functional_threshold
            )
            
            # Check uniqueness
            recovered_hash = hash_circuit_logits(recovered_logits)
            is_unique = recovered_hash not in unique_solutions
            
            # Track edge in exploration graph (allows cycles)
            edge_metadata = {
                "phase": "random_walk",
                "iteration": iteration,
                "is_cycle": (recovered_hash == circuit_hash),
                "is_revisit": (recovered_hash in unique_solutions),
                "final_accuracy": final_accuracy,
                "is_functional": is_func,
            }
            edges.append((circuit_hash, recovered_hash, pattern_idx, edge_metadata))
            
            if circuit_hash not in exploration_graph:
                exploration_graph[circuit_hash] = []
            exploration_graph[circuit_hash].append((recovered_hash, pattern_idx, edge_metadata))
            
            if is_func:
                functional_recoveries += 1
                
                if is_unique:
                    unique_solutions[recovered_hash] = recovered_logits
                    discovered_circuits.append((recovered_hash, recovered_logits))
                    if (iteration + 1) % 10 == 0:
                        log.info(
                            f"  ✓ New unique solution! Hash: {recovered_hash[:16]}..., "
                            f"Accuracy: {final_accuracy:.4f}"
                        )
                
                exploration_results.append({
                    "pattern_idx": pattern_idx,
                    "source_hash": circuit_hash,
                    "recovered_hash": recovered_hash,
                    "final_accuracy": final_accuracy,
                    "is_functional": is_func,
                    "is_unique": is_func and is_unique,
                    "phase": "random_walk",
                    "iteration": iteration,
                })
            else:
                exploration_results.append({
                    "pattern_idx": pattern_idx,
                    "source_hash": circuit_hash,
                    "recovered_hash": None,
                    "final_accuracy": final_accuracy,
                    "is_functional": False,
                    "is_unique": False,
                    "phase": "random_walk",
                    "iteration": iteration,
                })
        
        log.info(f"\nRandom Walk Phase Complete:")
        log.info(f"  Total perturbations: {total_perturbations}")
        log.info(f"  Unique solutions discovered: {len(unique_solutions)}")
    
    # ============================================================================
    # Legacy: Single-root exploration (if strategy is "single_root")
    # ============================================================================
    if exploration_strategy == "single_root":
        log.info("\n" + "=" * 80)
        log.info("Legacy Single-Root Exploration")
        log.info("=" * 80)
        
        for pattern_idx, knockout_pattern in enumerate(knockout_vocabulary[:num_perturbations]):
            log.info(f"\nProcessing perturbation {pattern_idx + 1}/{num_perturbations}")
            total_perturbations += 1
            
            result = _perturb_and_recover_single(
                initial_logits=root_logits,
                knockout_pattern=knockout_pattern,
                wires=root_wires,
                x_data=x_data,
                y_data=y_data,
                layer_sizes=layer_sizes,
                loss_type=loss_type,
                functional_threshold=functional_threshold,
                recovery_mode=recovery_mode,
                opt=opt,
                epochs=epochs,
                reversible_bias=reversible_bias,
                model=model,
                input_n=input_n,
                circuit_hidden_dim=circuit_hidden_dim,
                arity=arity,
                max_steps=max_steps,
                damage_behavior=damage_behavior,
            )
            
            if result is None:
                exploration_results.append({
                    "pattern_idx": pattern_idx,
                    "source_hash": root_hash,
                    "recovered_hash": None,
                    "final_accuracy": 0.0,
                    "is_functional": False,
                    "is_unique": False,
                    "phase": "single_root",
                })
                continue
            
            recovered_logits = result["params"]
            final_accuracy = float(result["final_hard_accuracy"])
            successful_recoveries += 1
            
            is_func = is_functional(
                recovered_logits, root_wires, x_data, y_data, functional_threshold
            )
            recovered_hash = hash_circuit_logits(recovered_logits)
            is_unique = recovered_hash not in unique_solutions
            
            if is_func:
                functional_recoveries += 1
                if is_unique:
                    unique_solutions[recovered_hash] = recovered_logits
                    log.info(
                        f"  ✓ Unique functional solution! Hash: {recovered_hash[:16]}..., "
                        f"Accuracy: {final_accuracy:.4f}"
                    )
                else:
                    log.info(
                        f"  → Recovered to known solution (hash: {recovered_hash[:16]}...), "
                        f"Accuracy: {final_accuracy:.4f}"
                    )
            
            exploration_results.append({
                "pattern_idx": pattern_idx,
                "source_hash": root_hash,
                "recovered_hash": recovered_hash,
                "final_accuracy": final_accuracy,
                "is_functional": is_func,
                "is_unique": is_func and is_unique,
                "phase": "single_root",
            })
    
    # ============================================================================
    # Summary Statistics
    # ============================================================================
    num_unique_solutions = len(unique_solutions)
    perturbation_efficiency = num_unique_solutions / total_perturbations if total_perturbations > 0 else 0.0
    
    # Compute per-depth statistics
    per_depth_stats = compute_per_depth_statistics(exploration_results)
    
    log.info("\n" + "=" * 80)
    log.info("Exploration Summary")
    log.info("=" * 80)
    log.info(f"Strategy: {exploration_strategy}")
    log.info(f"Total perturbations: {total_perturbations}")
    log.info(f"Successful recoveries: {successful_recoveries}")
    log.info(f"Functional recoveries: {functional_recoveries}")
    log.info(f"Unique solutions discovered: {num_unique_solutions}")
    log.info(f"Perturbation efficiency: {perturbation_efficiency:.4f} (unique/total)")
    log.info(f"Exploration graph edges: {len(edges)}")
    log.info(f"Exploration graph nodes: {len(exploration_graph)}")
    
    # Log per-depth statistics if available
    if per_depth_stats:
        log.info("\nPer-Depth Statistics:")
        for depth in sorted(per_depth_stats.keys()):
            stats = per_depth_stats[depth]
            log.info(
                f"  Depth {depth}: {stats['total_attempts']} attempts, "
                f"{stats['functional_recoveries']} functional, "
                f"{stats['failed_attempts']} failed "
                f"(success rate: {stats['success_rate']:.2%})"
            )
    
    log.info("=" * 80)
    
    return {
        "unique_solutions": unique_solutions,
        "exploration_results": exploration_results,
        "exploration_graph": exploration_graph,
        "edges": edges,
        "summary": {
            "total_perturbations": total_perturbations,
            "successful_recoveries": successful_recoveries,
            "functional_recoveries": functional_recoveries,
            "unique_solutions": num_unique_solutions,
            "perturbation_efficiency": perturbation_efficiency,
            "strategy": exploration_strategy,
            "per_depth_stats": per_depth_stats,
        },
        "root_hash": root_hash,
    }


def generate_exploration_name(
    exploration_strategy: str,
    phases: Optional[List[Tuple[str, int, int, str, Optional[int]]]] = None,
    phases_string: Optional[str] = None,
    recovery_mode: str = "backprop",
) -> str:
    """
    Generate a descriptive name for the exploration run based on strategy, phases, and recovery mode.
    
    Examples:
        - "dfs:50:3:root" with backprop -> "DFS_50_3_ROOT_BACKPROP"
        - "bfs:2:100:root,dfs:5:1:frontier" with self_attention -> "BFS_2_100_ROOT_DFS_5_1_FRONTIER_SA"
        - "hybrid" with backprop -> "HYBRID_BACKPROP"
    
    Args:
        exploration_strategy: Strategy name ("phases", "hybrid", "bfs", etc.)
        phases: List of phase tuples (if available)
        phases_string: Original phases string (if phases not parsed yet)
        recovery_mode: Recovery mode ("backprop" or "self_attention", default: "backprop")
        
    Returns:
        Descriptive name string
    """
    # Map recovery mode to display name
    recovery_name_map = {
        "backprop": "BACKPROP",
        "self_attention": "SA",  # Self-Attention
    }
    recovery_suffix = recovery_name_map.get(recovery_mode, recovery_mode.upper())
    
    if exploration_strategy == "phases" and phases is not None:
        # Generate name from phases
        phase_parts = []
        for phase_type, depth_limit, perturbations_per_node, start_from, max_frontier_nodes in phases:
            phase_name = phase_type.upper()
            start_name = start_from.upper()
            if max_frontier_nodes is not None:
                start_name = f"{start_name}_{max_frontier_nodes}"
            phase_parts.append(f"{phase_name}_{depth_limit}_{perturbations_per_node}_{start_name}")
        base_name = "_".join(phase_parts)
        return f"{base_name}_{recovery_suffix}"
    elif exploration_strategy == "phases" and phases_string is not None:
        # Parse phases_string to generate name
        try:
            parsed_phases = parse_phase_specification(phases_string)
            return generate_exploration_name(exploration_strategy, parsed_phases, None, recovery_mode)
        except:
            # Fallback to strategy name if parsing fails
            return f"{exploration_strategy.upper()}_{recovery_suffix}"
    else:
        # Use strategy name for non-phase strategies
        return f"{exploration_strategy.upper()}_{recovery_suffix}"


def main():
    parser = argparse.ArgumentParser(
        description="Explore degenerate circuit solution spaces"
    )
    # Default paths to preconfigured circuits (relative to workspace root)
    workspace_root = Path(__file__).parent.parent
    default_logits_file = workspace_root / "preconfigured_circuits" / "preconfigured_logits_20251112_linux.npz"
    default_wires_file = workspace_root / "preconfigured_circuits" / "wires_20251112_linux.npz"
    
    parser.add_argument(
        "--logits-file",
        type=str,
        default=str(default_logits_file) if default_logits_file.exists() else None,
        help=f"Path to preconfigured logits NPZ file (default: {default_logits_file})",
    )
    parser.add_argument(
        "--wires-file",
        type=str,
        default=str(default_wires_file) if default_wires_file.exists() else None,
        help=f"Path to preconfigured wires NPZ file (default: {default_wires_file})",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="binary_multiply",
        help="Task name (default: binary_multiply, matching config.yaml)",
    )
    parser.add_argument(
        "--input-bits",
        type=int,
        default=8,
        help="Number of input bits (default: 8)",
    )
    parser.add_argument(
        "--output-bits",
        type=int,
        default=8,
        help="Number of output bits (default: 8, matching config.yaml)",
    )
    parser.add_argument(
        "--num-perturbations",
        type=int,
        default=20,
        help="Number of perturbation patterns to try (default: 100)",
    )
    parser.add_argument(
        "--damage-prob",
        type=float,
        default=20.0,
        help="Number of gates to damage per perturbation (default: 20.0, matching config.yaml)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
        help="Number of recovery epochs (default: 200)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1.0,
        help="Learning rate for recovery (default: 1.0, matching config.yaml)",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adamw",
        choices=["adamw", "adam"],
        help="Optimizer type (default: adamw)",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-1,
        help="Weight decay for optimizer (default: 1e-1)",
    )
    parser.add_argument(
        "--beta1",
        type=float,
        default=0.8,
        help="Beta1 parameter for optimizer (default: 0.8)",
    )
    parser.add_argument(
        "--beta2",
        type=float,
        default=0.8,
        help="Beta2 parameter for optimizer (default: 0.8)",
    )
    parser.add_argument(
        "--wiring-seed",
        type=int,
        default=33,
        help="Random seed for wiring generation (default: 33, matching config.yaml test_seed)",
    )
    parser.add_argument(
        "--functional-threshold",
        type=float,
        default=0.999,
        help="Minimum accuracy to consider circuit functional (default: 0.999)",
    )
    parser.add_argument(
        "--recovery-mode",
        type=str,
        default="backprop",
        choices=["backprop", "self_attention"],
        help="Recovery mode: 'backprop' (gradient descent) or 'self_attention' (iterative message passing) (default: backprop)",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to trained CircuitSelfAttention model checkpoint file, or WandB run ID (e.g., 'yaw4da84'). "
             "If not provided and recovery_mode='self_attention', defaults to WandB run 'yaw4da84'.",
    )
    parser.add_argument(
        "--circuit-hidden-dim",
        type=int,
        default=64,
        help="Hidden dimension for circuit graph (self_attention mode, default: 64, matching config.yaml)",
    )
    parser.add_argument(
        "--arity",
        type=int,
        default=4,
        help="Gate arity (self_attention mode, default: 4, matching config.yaml)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=15,
        help="Maximum message passing steps for self-attention recovery (default: 15)",
    )
    parser.add_argument(
        "--damage-behavior",
        type=str,
        default="reversible",
        choices=["reversible", "permanent"],
        help="Damage behavior for self-attention recovery (default: reversible, matching config.yaml)",
    )
    parser.add_argument(
        "--reversible-bias",
        type=float,
        default=-10.0,
        help="Bias value for reversible damage mode (default: -10.0, matching config.yaml)",
    )
    parser.add_argument(
        "--loss-type",
        type=str,
        default="l4",
        choices=["l4", "l2", "bce"],
        help="Loss function type (default: l4, matching config.yaml)",
    )
    parser.add_argument(
        "--exploration-strategy",
        type=str,
        default="phases",
        choices=["hybrid", "bfs", "random_walk", "single_root", "phases"],
        help="Exploration strategy: 'hybrid' (BFS + Random Walk, recommended for UMAP), "
             "'bfs' (breadth-first only), 'random_walk' (random walk only), "
             "'single_root' (legacy: all perturbations from root), "
             "or 'phases' (flexible phase-based exploration, requires --phases) (default: phases)",
    )
    parser.add_argument(
        "--phases",
        type=str,
        default="dfs:20:1:root",
        help="Phase specification for phase-based exploration. "
             "Format: 'type:depth:perturbations:start_from[:max_nodes],...' "
             "Example: 'dfs:20:1:root' (default: simple 20-step DFS from root) "
             "or 'bfs:2:100:root,dfs:5:1:frontier' "
             "(BFS depth 2 with 100 perturbations from root, then DFS depth 5 with 1 perturbation from all frontier nodes). "
             "Use 'frontier:N' to randomly sample N frontier nodes: 'dfs:50:1:frontier:1' "
             "(DFS depth 50 from 1 randomly selected frontier node)",
    )
    parser.add_argument(
        "--bfs-depth",
        type=int,
        default=2,
        help="Maximum depth for BFS phase (default: 2)",
    )
    parser.add_argument(
        "--bfs-perturbations-per-level",
        type=int,
        default=100,
        help="Number of perturbations to try per BFS level (default: 100)",
    )
    parser.add_argument(
        "--random-walk-iterations",
        type=int,
        default=500,
        help="Number of random walk iterations (default: 500)",
    )
    parser.add_argument(
        "--random-walk-seed",
        type=int,
        default=12345,
        help="Random seed for random walk phase (default: 12345)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save exploration results (default: ./exploration_results/{timestamp})",
    )
    parser.add_argument(
        "--no-save-results",
        action="store_true",
        help="Don't save exploration results to disk (default: save results)",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=1000,
        help="Save checkpoint every N unique solutions discovered (default: 1000). Checkpoints overwrite the same output directory, so visualize_umap.py can always use the latest results.",
    )
    parser.add_argument(
        "--max-retry-attempts",
        type=int,
        default=0,
        help="When DFS perturbations_per_node=1 and a perturbation fails, retry up to this many times with different patterns (default: 0, no retries)",
    )
    
    args = parser.parse_args()
    
    # Generate task data
    case_n = 1 << args.input_bits
    x_data, y_data = get_task_data(
        args.task, case_n, input_bits=args.input_bits, output_bits=args.output_bits
    )
    log.info(f"Task: {args.task}, Input shape: {x_data.shape}, Output shape: {y_data.shape}")
    
    # Generate layer sizes (simple structure for now)
    # TODO: Make this configurable
    from boolean_nca_cc.circuits.model import generate_layer_sizes
    
    layer_sizes = list(
        generate_layer_sizes(
            args.input_bits, args.output_bits, arity=args.arity, layer_n=3
        )
    )
    log.info(f"Layer sizes: {layer_sizes}")
    
    # Load or generate preconfigured circuit (using same optimizer settings as recovery)
    wiring_key = jax.random.PRNGKey(args.wiring_seed)
    wires, logits, actual_layer_sizes = load_preconfigured_circuit(
        logits_file=args.logits_file,
        wires_file=args.wires_file,
        wiring_key=wiring_key,
        layer_sizes=layer_sizes,
        arity=args.arity,
        x_data=x_data,
        y_data=y_data,
        loss_type=args.loss_type,
        preconfig_lr=args.learning_rate,
        preconfig_optimizer=args.optimizer,
        preconfig_weight_decay=args.weight_decay,
        preconfig_beta1=args.beta1,
        preconfig_beta2=args.beta2,
    )
    
    # Use actual layer_sizes from loaded circuit (may differ from generated ones)
    if args.logits_file is not None and args.wires_file is not None:
        log.info(f"Using layer_sizes inferred from loaded circuit: {actual_layer_sizes}")
        layer_sizes = actual_layer_sizes
    
    # Assess root circuit: compute loss and hard accuracy
    from boolean_nca_cc.training.evaluation import get_loss_from_wires_logits
    
    root_loss, root_aux = get_loss_from_wires_logits(
        logits, wires, x_data, y_data, loss_type=args.loss_type
    )
    root_hard_loss, _, _, root_accuracy, root_hard_accuracy, _, _, _ = root_aux
    
    log.info("=" * 80)
    log.info("Root Circuit Assessment")
    log.info("=" * 80)
    log.info(f"Loss: {float(root_loss):.6f}")
    log.info(f"Hard Loss: {float(root_hard_loss):.6f}")
    log.info(f"Accuracy: {float(root_accuracy):.4f}")
    log.info(f"Hard Accuracy: {float(root_hard_accuracy):.4f}")
    log.info(f"Functional Threshold: {args.functional_threshold}")
    
    if root_hard_accuracy < args.functional_threshold:
        log.warning(
            f"⚠️  Root circuit hard accuracy ({root_hard_accuracy:.4f}) is below functional threshold "
            f"({args.functional_threshold})"
        )
        log.warning("  This may indicate preconfiguration issues or circuit loading problems.")
    else:
        log.info("✓ Root circuit meets functional threshold")
    log.info("=" * 80)
    
    # Parse phases if provided
    phases = None
    if args.exploration_strategy == "phases":
        # Default to simple 20-step DFS if not specified
        if args.phases is None:
            args.phases = "dfs:20:1:root"
            log.info("Using default phase specification: dfs:20:1:root (20-step DFS from root)")
        phases = parse_phase_specification(args.phases)
        log.info(f"Parsed {len(phases)} phases from specification")
    
    # Load model if using self-attention recovery
    model = None
    if args.recovery_mode == "self_attention":
        # Default to WandB run ID if not provided
        model_path_or_run_id = args.model_path if args.model_path is not None else "yaw4da84"
        
        log.info(f"Loading self-attention model from: {model_path_or_run_id}")
        try:
            from flax import nnx
            
            # Check if it's a file path or WandB run ID
            is_file_path = Path(model_path_or_run_id).exists() or model_path_or_run_id.endswith(('.pkl', '.npz', '.ckpt'))
            
            if is_file_path:
                # Load from local file
                from boolean_nca_cc.training.checkpointing import load_checkpoint_with_compatibility
                
                log.info(f"Loading model from local file: {model_path_or_run_id}")
                loaded_dict = load_checkpoint_with_compatibility(model_path_or_run_id)
                
                # Create model instance with defaults (we need config for proper initialization)
                input_n = args.input_bits
                total_nodes = sum(size[0] for size in layer_sizes)
                
                model_key = jax.random.PRNGKey(42)
                model = CircuitSelfAttention(
                    n_node=total_nodes,
                    circuit_hidden_dim=args.circuit_hidden_dim,
                    arity=args.arity,
                    rngs=nnx.Rngs(params=model_key),
                    damage_behavior=args.damage_behavior,
                )
                
                # Update model with loaded state
                if "model" in loaded_dict:
                    nnx.update(model, loaded_dict["model"])
                    log.info("Model loaded successfully from file")
                else:
                    log.warning("No 'model' key in checkpoint, using initialized model")
            else:
                # Load from WandB
                from boolean_nca_cc.training.checkpointing import (
                    load_config_from_wandb,
                    load_model_from_config_and_checkpoint,
                )
                from boolean_nca_cc.circuits.tasks import TASKS
                
                log.info(f"Loading model from WandB run ID: {model_path_or_run_id}")
                
                # Set up filters based on circuit configuration (matching GUI_minimal.py)
                task_name = args.task
                filters = {
                    "config.circuit.input_bits": args.input_bits,
                    "config.circuit.output_bits": args.output_bits,
                    "config.circuit.arity": args.arity,
                    "config.model.type": "self_attention",
                    "config.circuit.task": task_name if task_name in TASKS else "binary_multiply",
                    "config.training.training_mode": "repair",
                    "config.pool.damage_mode": "greedy_vocabulary",
                    "config.pool.damage_injection_mode": "multi",
                }
                
                # Load config and checkpoint from WandB
                # If run_id is provided, use it directly (ignore filters)
                loaded_config, checkpoint_path, run_id = load_config_from_wandb(
                    run_id=model_path_or_run_id,
                    filters=None,  # Don't use filters when run_id is specified
                    project="boolean-nca-cc",
                    entity="marcello-barylli-growai",
                    download_dir="saves",
                    filename="latest_checkpoint",
                    select_by_best_metric=False,
                    run_from_last=1,
                    use_cache=True,
                )
                
                log.info(f"Loaded config from WandB run: {run_id}")
                log.info(f"Checkpoint path: {checkpoint_path}")
                
                # Load model from config and checkpoint
                model, loaded_dict = load_model_from_config_and_checkpoint(
                    config=loaded_config,
                    checkpoint_path=checkpoint_path,
                    run_id=run_id,
                )
                
                log.info("Model loaded successfully from WandB")
                
                # Extract circuit_hidden_dim from loaded config if available
                if hasattr(loaded_config, "model") and hasattr(loaded_config.model, "hidden_dim"):
                    extracted_hidden_dim = loaded_config.model.hidden_dim
                    if args.circuit_hidden_dim != extracted_hidden_dim:
                        log.info(
                            f"Updating circuit_hidden_dim from {args.circuit_hidden_dim} "
                            f"to {extracted_hidden_dim} (from loaded config)"
                        )
                        args.circuit_hidden_dim = extracted_hidden_dim
                elif hasattr(loaded_config, "circuit") and hasattr(
                    loaded_config.circuit, "circuit_hidden_dim"
                ):
                    extracted_hidden_dim = loaded_config.circuit.circuit_hidden_dim
                    if args.circuit_hidden_dim != extracted_hidden_dim:
                        log.info(
                            f"Updating circuit_hidden_dim from {args.circuit_hidden_dim} "
                            f"to {extracted_hidden_dim} (from loaded config)"
                        )
                        args.circuit_hidden_dim = extracted_hidden_dim
        
        except Exception as e:
            log.error(f"Error loading model: {e}")
            import traceback
            log.error(f"Traceback: {traceback.format_exc()}")
            raise
    
    # Prepare exploration config and output directory BEFORE running exploration
    # (so checkpoints can be saved during exploration)
    exploration_config = {
        "exploration_strategy": args.exploration_strategy,
        "bfs_depth": args.bfs_depth,
        "bfs_perturbations_per_level": args.bfs_perturbations_per_level,
        "random_walk_iterations": args.random_walk_iterations,
        "random_walk_seed": args.random_walk_seed,
        "damage_prob": args.damage_prob,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "optimizer": args.optimizer,
        "weight_decay": args.weight_decay,
        "beta1": args.beta1,
        "beta2": args.beta2,
        "functional_threshold": args.functional_threshold,
        "input_bits": args.input_bits,
        "output_bits": args.output_bits,
    }
    if args.exploration_strategy == "phases" and phases is not None:
        exploration_config["phases"] = [
            {
                "phase_type": phase_type,
                "depth_limit": depth_limit,
                "perturbations_per_node": perturbations_per_node,
                "start_from": start_from,
                "max_frontier_nodes": max_frontier_nodes,
            }
            for phase_type, depth_limit, perturbations_per_node, start_from, max_frontier_nodes in phases
        ]
        exploration_config["phases_string"] = args.phases
    
    # Create output directory early (for checkpointing during exploration)
    save_results = not args.no_save_results
    output_dir = None
    if save_results:
        if args.output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            # Generate descriptive name from exploration strategy, phases, and recovery mode
            exploration_name = generate_exploration_name(
                exploration_strategy=args.exploration_strategy,
                phases=phases,
                phases_string=args.phases if args.exploration_strategy == "phases" else None,
                recovery_mode=args.recovery_mode,
            )
            output_dir = workspace_root / "exploration_results" / f"{exploration_name}_{timestamp}"
        else:
            output_dir = Path(args.output_dir)
    
    # Run exploration
    results = explore_degenerate_solutions(
        root_wires=wires,
        root_logits=logits,
        x_data=x_data,
        y_data=y_data,
        layer_sizes=layer_sizes,
        num_perturbations=args.num_perturbations,
        damage_prob=args.damage_prob,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        optimizer=args.optimizer,
        weight_decay=args.weight_decay,
        beta1=args.beta1,
        beta2=args.beta2,
        loss_type=args.loss_type,
        functional_threshold=args.functional_threshold,
        reversible_bias=args.reversible_bias,
        exploration_strategy=args.exploration_strategy,
        bfs_depth=args.bfs_depth,
        bfs_perturbations_per_level=args.bfs_perturbations_per_level,
        random_walk_iterations=args.random_walk_iterations,
        random_walk_seed=args.random_walk_seed,
        phases=phases,
        recovery_mode=args.recovery_mode,
        model=model,
        input_n=args.input_bits,
        circuit_hidden_dim=args.circuit_hidden_dim,
        arity=args.arity,
        max_steps=args.max_steps,
        damage_behavior=args.damage_behavior,
        max_retry_attempts=args.max_retry_attempts,
        output_dir=output_dir,
        checkpoint_interval=args.checkpoint_interval,
        task_name=args.task,
        exploration_config=exploration_config,
    )
    
    # Final save (overwrites last checkpoint with final results)
    if save_results:
        save_exploration_results(
            results=results,
            output_dir=output_dir,
            wires=wires,
            layer_sizes=layer_sizes,
            task_name=args.task,
            exploration_config=exploration_config,
        )
    
    # Print final summary
    summary = results["summary"]
    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    print(f"Unique solutions discovered: {summary['unique_solutions']}")
    print(f"Perturbation efficiency: {summary['perturbation_efficiency']:.4f}")
    print(f"Functional recovery rate: {summary['functional_recoveries'] / summary['total_perturbations']:.4f}")
    if save_results:
        print(f"Results saved to: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()

