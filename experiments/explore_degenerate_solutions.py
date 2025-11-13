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
from pathlib import Path
from typing import List, Tuple, Dict, Set, Optional
import numpy as np
import jax
import jax.numpy as jp
import optax
from collections import deque
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
    threshold: float = 1,
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
    opt: optax.GradientTransformation,
    wires: List[jp.ndarray],
    x_data: jp.ndarray,
    y_data: jp.ndarray,
    layer_sizes: List[Tuple[int, int]],
    epochs: int,
    loss_type: str,
    reversible_bias: float,
) -> Optional[Dict]:
    """
    Helper function to perturb and recover a single circuit.
    
    Returns:
        Dictionary with recovery results, or None if error occurred
    """
    try:
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
    except Exception as e:
        log.debug(f"Error during perturbation-recovery: {e}")
        return None


def explore_degenerate_solutions(
    root_wires: List[jp.ndarray],
    root_logits: List[jp.ndarray],
    x_data: jp.ndarray,
    y_data: jp.ndarray,
    layer_sizes: List[Tuple[int, int]],
    num_perturbations: int = 100,
    damage_prob: float = 5.0,
    greedy_indices: List[int] = None,
    epochs: int = 200,
    learning_rate: float = 1.0,
    optimizer: str = "adamw",
    weight_decay: float = 1e-1,
    beta1: float = 0.8,
    beta2: float = 0.8,
    loss_type: str = "l4",
    functional_threshold: float = 1.0,
    reversible_bias: float = -10.0,
    exploration_strategy: str = "hybrid",
    bfs_depth: int = 2,
    bfs_perturbations_per_level: int = 100,
    random_walk_iterations: int = 500,
    random_walk_seed: int = 12345,
) -> Dict:
    """
    Explore degenerate solutions using hybrid BFS + Random Walk strategy.
    
    Strategy options:
    - "hybrid": BFS phase (depth 1-2) followed by Random Walk (recommended for UMAP)
    - "bfs": Breadth-first search only
    - "random_walk": Random walk only
    
    Args:
        root_wires: Root circuit wiring
        root_logits: Root circuit logits
        x_data: Input data
        y_data: Target output data
        layer_sizes: Circuit layer sizes
        num_perturbations: Number of perturbation patterns to try (legacy, used if strategy="single_root")
        damage_prob: Number of gates to damage per perturbation
        greedy_indices: Ordered list of greedy gate indices (defaults to DEFAULT_GREEDY_ORDERED_INDICES)
        epochs: Number of recovery epochs
        learning_rate: Learning rate for recovery (default: 1.0, matching config.yaml)
        optimizer: Optimizer type ("adamw" or "adam", default: "adamw")
        weight_decay: Weight decay for optimizer (default: 1e-1)
        beta1: Beta1 parameter for optimizer (default: 0.8)
        beta2: Beta2 parameter for optimizer (default: 0.8)
        loss_type: Loss type for recovery
        functional_threshold: Minimum accuracy to consider circuit functional (default: 1.0)
        reversible_bias: Bias value for reversible damage mode
        exploration_strategy: Strategy to use ("hybrid", "bfs", "random_walk", or "single_root")
        bfs_depth: Maximum depth for BFS phase (default: 2)
        bfs_perturbations_per_level: Number of perturbations to try per BFS level (default: 100)
        random_walk_iterations: Number of random walk iterations (default: 500)
        random_walk_seed: Random seed for random walk phase
        
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
    
    log.info("=" * 80)
    log.info("Exploring Degenerate Circuit Solutions")
    log.info("=" * 80)
    log.info(f"Exploration strategy: {exploration_strategy}")
    log.info(f"Root circuit hash: {hash_circuit_logits(root_logits)}")
    log.info(f"Damage per perturbation: {damage_prob} gates")
    log.info(f"Recovery epochs: {epochs}")
    log.info(f"Learning rate: {learning_rate}")
    log.info(f"Optimizer: {optimizer}")
    log.info(f"Weight decay: {weight_decay}")
    log.info(f"Beta1: {beta1}, Beta2: {beta2}")
    log.info(f"Functional threshold: {functional_threshold}")
    
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
    vocab_size = max(num_perturbations, bfs_perturbations_per_level * bfs_depth, random_walk_iterations)
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
    
    # ============================================================================
    # PHASE 1: BFS Exploration (if strategy is "hybrid" or "bfs")
    # ============================================================================
    if exploration_strategy in ["hybrid", "bfs"]:
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
                    opt=opt,
                    wires=root_wires,
                    x_data=x_data,
                    y_data=y_data,
                    layer_sizes=layer_sizes,
                    epochs=epochs,
                    loss_type=loss_type,
                    reversible_bias=reversible_bias,
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
                opt=opt,
                wires=root_wires,
                x_data=x_data,
                y_data=y_data,
                layer_sizes=layer_sizes,
                epochs=epochs,
                loss_type=loss_type,
                reversible_bias=reversible_bias,
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
                opt=opt,
                wires=root_wires,
                x_data=x_data,
                y_data=y_data,
                layer_sizes=layer_sizes,
                epochs=epochs,
                loss_type=loss_type,
                reversible_bias=reversible_bias,
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
        },
        "root_hash": root_hash,
    }


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
        default=10,
        help="Number of perturbation patterns to try (default: 100)",
    )
    parser.add_argument(
        "--damage-prob",
        type=float,
        default=5.0,
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
        default=42,
        help="Random seed for wiring generation (default: 42)",
    )
    parser.add_argument(
        "--functional-threshold",
        type=float,
        default=1.0,
        help="Minimum accuracy to consider circuit functional (default: 1.0)",
    )
    parser.add_argument(
        "--exploration-strategy",
        type=str,
        default="hybrid",
        choices=["hybrid", "bfs", "random_walk", "single_root"],
        help="Exploration strategy: 'hybrid' (BFS + Random Walk, recommended for UMAP), "
             "'bfs' (breadth-first only), 'random_walk' (random walk only), "
             "or 'single_root' (legacy: all perturbations from root) (default: hybrid)",
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
            args.input_bits, args.output_bits, arity=2, layer_n=3
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
        arity=2,
        x_data=x_data,
        y_data=y_data,
        loss_type="l4",
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
    
    # Verify root circuit is functional
    root_accuracy = float(
        compute_accuracy(
            run_circuit(logits, wires, x_data, hard=True)[-1], y_data
        )
    )
    log.info(f"Root circuit accuracy: {root_accuracy:.4f}")
    if root_accuracy < args.functional_threshold:
        log.warning(
            f"Root circuit accuracy ({root_accuracy:.4f}) is below functional threshold "
            f"({args.functional_threshold})"
        )
    
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
        functional_threshold=args.functional_threshold,
        exploration_strategy=args.exploration_strategy,
        bfs_depth=args.bfs_depth,
        bfs_perturbations_per_level=args.bfs_perturbations_per_level,
        random_walk_iterations=args.random_walk_iterations,
        random_walk_seed=args.random_walk_seed,
    )
    
    # Save results by default (unless --no-save-results is specified)
    save_results = not args.no_save_results
    if save_results:
        if args.output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = workspace_root / "exploration_results" / f"exploration_{timestamp}"
        else:
            output_dir = Path(args.output_dir)
        
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

