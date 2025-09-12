#!/usr/bin/env python3
"""
Standalone Figure 3 plotting script using WandB checkpoints.

This script loads a trained model from WandB and recreates the exact Figure 3
plotting pipeline used in the training loop, maximizing reuse of existing components.

Usage:
    python experiments/visualization/figure3_standalone_checkpoint.py --run-id abc123 --output-dir reports/figures
    python experiments/visualization/figure3_standalone_checkpoint.py --run-id abc123 --bp-run-id def456 --output-dir reports/figures
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Optional, Dict, Any

import jax
import jax.numpy as jp
import matplotlib.pyplot as plt
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from boolean_nca_cc.training.checkpointing import load_best_model_from_wandb
from boolean_nca_cc.training.train_loop import plot_combined_bp_sa_stepwise_performance
from boolean_nca_cc.training.backprop import _run_backpropagation_training_with_knockouts
from boolean_nca_cc.circuits.model import gen_circuit
from boolean_nca_cc.utils.graph_builder import build_graph

# Setup logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def create_mock_config_from_loaded_config(loaded_config: Dict[str, Any]) -> SimpleNamespace:
    """
    Create a mock config object from loaded checkpoint config for backprop evaluation.
    
    This recreates the exact config structure used in the training loop.
    """
    # Extract key parameters from loaded config
    circuit_config = loaded_config.get("circuit", {})
    backprop_config = loaded_config.get("backprop", {})
    training_config = loaded_config.get("training", {})
    
    # Generate layer sizes using the same function as training
    from boolean_nca_cc.circuits.model import generate_layer_sizes
    input_bits = circuit_config.get("input_bits", 8)
    output_bits = circuit_config.get("output_bits", 8)
    arity = circuit_config.get("arity", 4)
    num_layers = circuit_config.get("num_layers", 3)
    layer_sizes = generate_layer_sizes(input_bits, output_bits, arity, num_layers)
    
    # Create mock config matching training loop structure
    mock_cfg = SimpleNamespace(
        test_seed=loaded_config.get("test_seed", 42),
        circuit=SimpleNamespace(
            layer_sizes=layer_sizes,
            arity=arity
        ),
        backprop=SimpleNamespace(
            epochs=backprop_config.get("epochs", 50),
            learning_rate=backprop_config.get("learning_rate", 1e-2),
            weight_decay=backprop_config.get("weight_decay", 1e-4),
            optimizer=backprop_config.get("optimizer", "adam"),
            beta1=backprop_config.get("beta1", 0.9),
            beta2=backprop_config.get("beta2", 0.999),
            parallel=backprop_config.get("parallel", True),
            batch_size=backprop_config.get("batch_size", None)
        ),
        logging=SimpleNamespace(
            log_interval=loaded_config.get("log_interval", 16)
        )
    )
    
    # Add get method to make it compatible with dict-like access
    def get(key, default=None):
        return getattr(mock_cfg, key, default)
    
    mock_cfg.get = get
    
    # Add get method to backprop sub-object as well
    def backprop_get(key, default=None):
        return getattr(mock_cfg.backprop, key, default)
    
    mock_cfg.backprop.get = backprop_get
    
    return mock_cfg


def recreate_training_data(loaded_config: Dict[str, Any]) -> tuple[jp.ndarray, jp.ndarray]:
    """
    Recreate the training data using the same parameters and functions as the original training.
    
    This ensures we use identical data for evaluation by reusing the exact task functions.
    The training uses the complete truth table (2^input_bits samples), not a random subset.
    """
    from boolean_nca_cc.circuits.tasks import get_task_data
    
    # Extract data generation parameters from actual training config
    circuit_config = loaded_config.get("circuit", {})
    input_bits = circuit_config.get("input_bits", 8)
    output_bits = circuit_config.get("output_bits", 8)
    task = circuit_config.get("task", "binary_multiply")
    
    # Use the exact same data generation as training: complete truth table
    case_n = 1 << input_bits  # 2^input_bits = 256 for 8-bit inputs
    x_data, y_data = get_task_data(
        task_name=task,
        case_n=case_n,
        input_bits=input_bits,
        output_bits=output_bits
    )
    
    return x_data, y_data


def recreate_knockout_vocabulary(loaded_config: Dict[str, Any]) -> jp.ndarray:
    """
    Recreate the knockout vocabulary using the exact same parameters and functions as training.
    """
    from boolean_nca_cc.training.pool.perturbation import create_knockout_vocabulary
    from boolean_nca_cc.circuits.model import generate_layer_sizes
    
    # Extract knockout parameters from the actual training config
    pool_config = loaded_config.get("pool", {})
    knockout_diversity = pool_config.get("damage_knockout_diversity", 16)
    damage_pool_damage_prob = pool_config.get("damage_prob", 20)
    damage_mode = pool_config.get("damage_mode", "shotgun")
    eval_config = loaded_config.get("eval", {})
    periodic_eval_test_seed = eval_config.get("periodic_eval_test_seed", 42)
    
    # Generate layer sizes using the same function as training
    circuit_config = loaded_config.get("circuit", {})
    input_bits = circuit_config.get("input_bits", 8)
    output_bits = circuit_config.get("output_bits", 8)
    arity = circuit_config.get("arity", 4)
    num_layers = circuit_config.get("num_layers", 3)
    
    # Use the exact same layer size generation as training
    layer_sizes = generate_layer_sizes(input_bits, output_bits, arity, num_layers)
    
    # Use the exact same seed as training (periodic_eval_test_seed)
    vocab_rng = jax.random.PRNGKey(periodic_eval_test_seed)
    
    # Use the exact same function as training
    knockout_vocabulary = create_knockout_vocabulary(
        rng=vocab_rng,
        vocabulary_size=knockout_diversity,
        layer_sizes=layer_sizes,
        damage_prob=damage_pool_damage_prob,
        damage_mode=damage_mode,
    )
    
    return knockout_vocabulary


def recreate_base_circuit(loaded_config: Dict[str, Any]) -> tuple:
    """
    Recreate the base circuit used for knockout evaluation using exact training parameters.
    """
    from boolean_nca_cc.circuits.model import generate_layer_sizes
    
    # Extract circuit parameters from actual training config
    circuit_config = loaded_config.get("circuit", {})
    input_bits = circuit_config.get("input_bits", 8)
    output_bits = circuit_config.get("output_bits", 8)
    arity = circuit_config.get("arity", 4)
    num_layers = circuit_config.get("num_layers", 3)
    wiring_fixed_key = loaded_config.get("wiring_fixed_key", 42)
    
    # Generate layer sizes using the same function as training
    layer_sizes = generate_layer_sizes(input_bits, output_bits, arity, num_layers)
    
    # Use the same wiring key as training
    key = jax.random.PRNGKey(wiring_fixed_key)
    
    # Generate base circuit using exact same function as training
    # gen_circuit only takes key, layer_sizes, and arity
    wires, logits = gen_circuit(
        key=key,
        layer_sizes=layer_sizes,
        arity=arity
    )
    
    return (wires, logits)


def create_figure3_from_checkpoint(
    run_id: str,
    output_dir: str = "reports/figures",
    project: str = "boolean_nca_cc",
    entity: str = "marcello-barylli-growai",
    filename: str = "best_model_hard_accuracy",
    filetype: str = "pkl"
) -> str:
    """
    Create Figure 3 plot from WandB checkpoint, reusing exact training loop implementation.
    
    Args:
        run_id: WandB run ID for the SA model
        output_dir: Directory to save the plot
        project: WandB project name
        entity: WandB entity/username
        filename: Checkpoint filename to load
        filetype: Checkpoint file type
        
    Returns:
        Path to the saved plot
    """
    log.info(f"Loading SA model from WandB run {run_id}")
    
    # Load the best model from WandB
    model, loaded_dict, config = load_best_model_from_wandb(
        run_id=run_id,
        project=project,
        entity=entity,
        filename=filename,
        filetype=filetype
    )
    
    log.info("Model loaded successfully. Recreating training environment...")
    
    # Extract training parameters
    loaded_config = loaded_dict.get("config", {})
    training_mode = loaded_config.get("training", {}).get("mode", "growth")
    
    # Recreate training data
    x_data, y_data = recreate_training_data(loaded_config)
    log.info(f"Recreated training data: {x_data.shape} inputs, {y_data.shape} targets")
    
    # Recreate knockout vocabulary
    knockout_vocabulary = recreate_knockout_vocabulary(loaded_config)
    log.info(f"Recreated knockout vocabulary: {len(knockout_vocabulary)} patterns")
    
    # Recreate base circuit
    base_circuit = recreate_base_circuit(loaded_config)
    log.info("Recreated base circuit for evaluation")
    
    # Extract model parameters
    circuit_config = loaded_config.get("circuit", {})
    input_bits = circuit_config.get("input_bits", 8)
    output_bits = circuit_config.get("output_bits", 8)
    arity = circuit_config.get("arity", 4)
    num_layers = circuit_config.get("num_layers", 3)
    circuit_hidden_dim = circuit_config.get("circuit_hidden_dim", 64)
    loss_type = loaded_config.get("training", {}).get("loss_type", "l4")
    eval_config = loaded_config.get("eval", {})
    periodic_eval_inner_steps = eval_config.get("periodic_eval_inner_steps", 100)
    
    # Generate layer sizes using the same function as training
    from boolean_nca_cc.circuits.model import generate_layer_sizes
    layer_sizes = generate_layer_sizes(input_bits, output_bits, arity, num_layers)
    input_n = input_bits
    
    # Create mock config for backprop evaluation
    mock_cfg = create_mock_config_from_loaded_config(loaded_config)
    
    # Run backpropagation training for reference line
    log.info("Computing backpropagation results for reference line...")
    bp_results = _run_backpropagation_training_with_knockouts(
        mock_cfg, x_data, y_data, loss_type, knockout_vocabulary,
        parallel=mock_cfg.backprop.parallel,
        batch_size=mock_cfg.backprop.batch_size
    )
    log.info(f"BP results computed. Mean final accuracy: {bp_results['aggregate_metrics']['mean_final_hard_accuracy']:.3f}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create Figure 3 using the exact same function as training loop
    log.info("Creating Figure 3 plot...")
    log.info(f"Data shapes: x_data={x_data.shape}, y_data={y_data.shape}")
    log.info(f"Layer sizes: {layer_sizes}")
    log.info(f"Knockout vocabulary shape: {knockout_vocabulary.shape}")
    log.info(f"Input n: {input_n}, arity: {arity}, circuit_hidden_dim: {circuit_hidden_dim}")
    
    fig = plot_combined_bp_sa_stepwise_performance(
        cfg=mock_cfg,
        x_data=x_data,
        y_data=y_data,
        loss_type=loss_type,
        knockout_patterns=knockout_vocabulary,
        model=model,
        base_circuit=base_circuit,
        n_message_steps=periodic_eval_inner_steps,
        layer_sizes=layer_sizes,
        input_n=input_n,
        arity=arity,
        circuit_hidden_dim=circuit_hidden_dim,
        bp_results=bp_results,
        show_bp_trajectory=False,  # Figure 3 mode: BP as reference line
    )
    
    # Save the plot
    output_path = os.path.join(output_dir, f"damage_recovery_trajectories_{training_mode}_standalone.png")
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    log.info(f"Figure 3 saved to: {output_path}")
    return output_path


def main():
    """Command line interface for the standalone Figure 3 plotting script."""
    parser = argparse.ArgumentParser(
        description="Create Figure 3 plot from WandB checkpoint",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python experiments/visualization/figure3_standalone_checkpoint.py --run-id abc123
  
  # Custom output directory
  python experiments/visualization/figure3_standalone_checkpoint.py --run-id abc123 --output-dir my_figures
        """
    )
    
    parser.add_argument(
        "--run-id",
        required=True,
        help="WandB run ID for the SA model"
    )
    parser.add_argument(
        "--output-dir",
        default="reports/figures",
        help="Directory to save the plot (default: reports/figures)"
    )
    parser.add_argument(
        "--project",
        default="boolean_nca_cc",
        help="WandB project name (default: boolean_nca_cc)"
    )
    parser.add_argument(
        "--entity",
        default="marcello-barylli-growai",
        help="WandB entity/username (default: marcello-barylli-growai)"
    )
    parser.add_argument(
        "--filename",
        default="best_model_hard_accuracy",
        help="Checkpoint filename to load (default: best_model_hard_accuracy)"
    )
    parser.add_argument(
        "--filetype",
        default="pkl",
        help="Checkpoint file type (default: pkl)"
    )
    
    args = parser.parse_args()
    
    try:
        output_path = create_figure3_from_checkpoint(
            run_id=args.run_id,
            output_dir=args.output_dir,
            project=args.project,
            entity=args.entity,
            filename=args.filename,
            filetype=args.filetype
        )
        print(f"✅ Figure 3 plot created successfully: {output_path}")
        
    except Exception as e:
        log.error(f"Error creating Figure 3 plot: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
