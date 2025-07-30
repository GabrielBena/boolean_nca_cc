"""
Example script demonstrating knockout backpropagation training.

This script shows how to use the new knockout-enabled backpropagation training
to compare with GNN-based knockout training.
"""

import jax
import jax.numpy as jp
import logging
from functools import partial

from boolean_nca_cc.circuits.model import gen_circuit, generate_layer_sizes
from boolean_nca_cc.circuits.tasks import get_task_data
from boolean_nca_cc.training.pool.structural_perturbation import create_knockout_vocabulary
from train import run_backpropagation_training_with_knockout

# Configure logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def main():
    """Run knockout backpropagation training example."""
    
    # Configuration (similar to config.yaml)
    config = {
        "test_seed": 42,
        "circuit": {
            "input_bits": 4,
            "output_bits": 4,
            "arity": 2,
            "num_layers": 2,
            "layer_sizes": None,  # Will be generated
        },
        "backprop": {
            "epochs": 100,
            "learning_rate": 1e-3,
            "optimizer": "adamw",
            "beta1": 0.9,
            "beta2": 0.999,
            "weight_decay": 1e-4,
        },
        "logging": {
            "log_interval": 10,
        },
        "wandb": {
            "enabled": False,  # Set to True to enable wandb logging
        }
    }
    
    # Generate layer sizes
    layer_sizes = generate_layer_sizes(
        config["circuit"]["input_bits"],
        config["circuit"]["output_bits"],
        config["circuit"]["arity"],
        layer_n=config["circuit"]["num_layers"]
    )
    config["circuit"]["layer_sizes"] = layer_sizes
    
    log.info(f"Circuit layer sizes: {layer_sizes}")
    
    # Get task data
    case_n = 1 << config["circuit"]["input_bits"]
    x_data, y_data = get_task_data(
        "parity",  # Simple parity task
        case_n,
        input_bits=config["circuit"]["input_bits"],
        output_bits=config["circuit"]["output_bits"]
    )
    
    log.info(f"Task data shape: x={x_data.shape}, y={y_data.shape}")
    
    # Create knockout vocabulary (similar to GNN training)
    knockout_diversity = 16  # Number of unique knockout patterns
    damage_prob = 0.1  # 10% of gates knocked out per pattern
    
    vocab_rng = jax.random.PRNGKey(config["test_seed"])
    knockout_vocabulary = create_knockout_vocabulary(
        rng=vocab_rng,
        vocabulary_size=knockout_diversity,
        layer_sizes=layer_sizes,
        damage_prob=damage_prob,
    )
    
    log.info(f"Created knockout vocabulary with {len(knockout_vocabulary)} patterns")
    log.info(f"Knockout vocabulary shape: {knockout_vocabulary.shape}")
    
    # Run knockout backpropagation training
    log.info("Starting knockout backpropagation training...")
    
    results = run_backpropagation_training_with_knockout(
        cfg=config,
        x_data=x_data,
        y_data=y_data,
        knockout_patterns=knockout_vocabulary,
        loss_type="l4"
    )
    
    # Print final results
    log.info("Training completed!")
    log.info(f"Final loss: {results['losses'][-1]:.4f}")
    log.info(f"Final accuracy: {results['accuracies'][-1]:.4f}")
    log.info(f"Final hard accuracy: {results['hard_accuracies'][-1]:.4f}")
    
    # Optional: Save results
    import pickle
    with open("knockout_backprop_results.pkl", "wb") as f:
        pickle.dump(results, f)
    log.info("Results saved to knockout_backprop_results.pkl")


if __name__ == "__main__":
    main() 