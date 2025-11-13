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
from typing import List, Tuple, Dict, Set
import numpy as np
import jax
import jax.numpy as jp
import optax

from boolean_nca_cc.circuits.model import run_circuit
from boolean_nca_cc.circuits.train import compute_accuracy, TrainState, train_step
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
) -> Tuple[List[jp.ndarray], List[jp.ndarray]]:
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
        return wires, logits
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
        return wires, logits


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
    functional_threshold: float = 1.0,
    reversible_bias: float = -10.0,
) -> Dict:
    """
    Explore degenerate solutions by perturbing and recovering circuits.
    
    Args:
        root_wires: Root circuit wiring
        root_logits: Root circuit logits
        x_data: Input data
        y_data: Target output data
        layer_sizes: Circuit layer sizes
        num_perturbations: Number of perturbation patterns to try
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
        
    Returns:
        Dictionary with exploration results
    """
    if greedy_indices is None:
        greedy_indices = DEFAULT_GREEDY_ORDERED_INDICES
    
    log.info("=" * 80)
    log.info("Exploring Degenerate Circuit Solutions")
    log.info("=" * 80)
    log.info(f"Root circuit hash: {hash_circuit_logits(root_logits)}")
    log.info(f"Number of perturbations: {num_perturbations}")
    log.info(f"Damage per perturbation: {damage_prob} gates")
    log.info(f"Recovery epochs: {epochs}")
    log.info(f"Learning rate: {learning_rate}")
    log.info(f"Optimizer: {optimizer}")
    log.info(f"Weight decay: {weight_decay}")
    log.info(f"Beta1: {beta1}, Beta2: {beta2}")
    log.info(f"Functional threshold: {functional_threshold}")
    
    # Generate vocabulary of perturbation patterns using greedy indices
    rng = jax.random.PRNGKey(42)
    knockout_vocabulary = create_knockout_vocabulary(
        rng=rng,
        vocabulary_size=num_perturbations,
        layer_sizes=layer_sizes,
        damage_prob=damage_prob,
        damage_mode="greedy_vocabulary",
        ordered_indices=greedy_indices,
    )
    log.info(f"Generated {len(knockout_vocabulary)} perturbation patterns")
    
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
    
    # Track unique solutions
    unique_solutions: Dict[str, List[jp.ndarray]] = {}  # hash -> logits
    root_hash = hash_circuit_logits(root_logits)
    unique_solutions[root_hash] = root_logits
    
    # Track exploration results
    exploration_results = []
    successful_recoveries = 0
    functional_recoveries = 0
    
    # Process each perturbation pattern
    for pattern_idx, knockout_pattern in enumerate(knockout_vocabulary):
        log.info(f"\nProcessing perturbation {pattern_idx + 1}/{num_perturbations}")
        
        # Apply perturbation and recover
        try:
            result = _train_single_knockout_pattern(
                initial_logits=root_logits,
                knockout_pattern=knockout_pattern,
                opt=opt,
                wires=root_wires,
                x_data=x_data,
                y_data=y_data,
                loss_type=loss_type,
                layer_sizes=layer_sizes,
                epochs=epochs,
                damage_behavior="reversible",
                reversible_bias=reversible_bias,
            )
            
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
            
            if is_func:
                functional_recoveries += 1
                
                if is_unique:
                    unique_solutions[recovered_hash] = recovered_logits
                    log.info(
                        f"  ✓ Unique functional solution discovered! "
                        f"Hash: {recovered_hash[:16]}..., Accuracy: {final_accuracy:.4f}"
                    )
                else:
                    log.info(
                        f"  → Recovered to known solution (hash: {recovered_hash[:16]}...), "
                        f"Accuracy: {final_accuracy:.4f}"
                    )
            else:
                log.info(
                    f"  ✗ Recovery failed to reach functional threshold "
                    f"(accuracy: {final_accuracy:.4f} < {functional_threshold})"
                )
            
            exploration_results.append({
                "pattern_idx": pattern_idx,
                "recovered_hash": recovered_hash,
                "final_accuracy": final_accuracy,
                "is_functional": is_func,
                "is_unique": is_func and is_unique,
            })
            
        except Exception as e:
            log.error(f"  ✗ Error during perturbation {pattern_idx + 1}: {e}")
            exploration_results.append({
                "pattern_idx": pattern_idx,
                "recovered_hash": None,
                "final_accuracy": 0.0,
                "is_functional": False,
                "is_unique": False,
                "error": str(e),
            })
    
    # Summary statistics
    num_unique_solutions = len(unique_solutions)
    perturbation_efficiency = num_unique_solutions / num_perturbations if num_perturbations > 0 else 0.0
    
    log.info("\n" + "=" * 80)
    log.info("Exploration Summary")
    log.info("=" * 80)
    log.info(f"Total perturbations: {num_perturbations}")
    log.info(f"Successful recoveries: {successful_recoveries}")
    log.info(f"Functional recoveries: {functional_recoveries}")
    log.info(f"Unique solutions discovered: {num_unique_solutions}")
    log.info(f"Perturbation efficiency: {perturbation_efficiency:.4f} (unique/total)")
    log.info("=" * 80)
    
    return {
        "unique_solutions": unique_solutions,
        "exploration_results": exploration_results,
        "summary": {
            "total_perturbations": num_perturbations,
            "successful_recoveries": successful_recoveries,
            "functional_recoveries": functional_recoveries,
            "unique_solutions": num_unique_solutions,
            "perturbation_efficiency": perturbation_efficiency,
        },
        "root_hash": root_hash,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Explore degenerate circuit solution spaces"
    )
    parser.add_argument(
        "--logits-file",
        type=str,
        default=None,
        help="Path to preconfigured logits NPZ file",
    )
    parser.add_argument(
        "--wires-file",
        type=str,
        default=None,
        help="Path to preconfigured wires NPZ file",
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
        default=100,
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
        default=42,
        help="Random seed for wiring generation (default: 42)",
    )
    parser.add_argument(
        "--functional-threshold",
        type=float,
        default=1.0,
        help="Minimum accuracy to consider circuit functional (default: 1.0)",
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
    wires, logits = load_preconfigured_circuit(
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
    )
    
    # Print final summary
    summary = results["summary"]
    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    print(f"Unique solutions discovered: {summary['unique_solutions']}")
    print(f"Perturbation efficiency: {summary['perturbation_efficiency']:.4f}")
    print(f"Functional recovery rate: {summary['functional_recoveries'] / summary['total_perturbations']:.4f}")
    print("=" * 80)


if __name__ == "__main__":
    main()

