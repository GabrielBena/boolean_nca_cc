#!/usr/bin/env python3
"""
Training script with logit channel logging.

This script demonstrates how to add logit monitoring to the training process.
"""

import os
import logging
import jax
import jax.numpy as jp
import hydra
from omegaconf import DictConfig, OmegaConf, open_dict
import wandb
from tqdm.auto import tqdm
from functools import partial
from flax import nnx

from boolean_nca_cc.circuits.model import gen_circuit
from boolean_nca_cc.circuits.tasks import get_task_data
from boolean_nca_cc import generate_layer_sizes
from boolean_nca_cc.training.train_loop import train_model
from boolean_nca_cc.utils.graph_builder import build_graph
from boolean_nca_cc.training.utils import cleanup_redundant_wandb_artifacts

# Configure logging
log = logging.getLogger(__name__)


class LogitMonitoringWrapper:
    """
    A wrapper around the model that logs logit information.
    """
    
    def __init__(self, original_model, log_interval=10):
        self.original_model = original_model
        self.log_interval = log_interval
        self.call_count = 0
        
        # Copy all attributes from original model
        for attr in dir(original_model):
            if not attr.startswith('_') and not callable(getattr(original_model, attr)):
                setattr(self, attr, getattr(original_model, attr))
    
    def __call__(self, graph, **kwargs):
        self.call_count += 1
        
        # Call original model
        result = self.original_model(graph, **kwargs)
        
        # Log every log_interval calls
        if self.call_count % self.log_interval == 0:
            logits = result.nodes["logits"]
            log.info(f"Model call #{self.call_count} - Logits shape: {logits.shape}")
            log.info(f"Logits range: [{jp.min(logits):.3f}, {jp.max(logits):.3f}]")
            log.info(f"Logits mean: {jp.mean(logits):.3f}")
            
            # Check for damaged nodes
            if "knockout_pattern" in kwargs and kwargs["knockout_pattern"] is not None:
                knockout_pattern = kwargs["knockout_pattern"]
                damaged_logits = logits[knockout_pattern]
                log.info(f"Damaged nodes logits: {damaged_logits}")
                log.info(f"Damaged nodes sigmoid: {jax.nn.sigmoid(damaged_logits)}")
        
        return result


def create_logit_monitoring_wrapper(original_model, log_interval=10):
    """
    Create a wrapper around the model that logs logit information.
    
    Args:
        original_model: The original CircuitSelfAttention model
        log_interval: How often to log (every N calls)
    
    Returns:
        Wrapped model with logging
    """
    return LogitMonitoringWrapper(original_model, log_interval)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Main training function with logit logging.
    """
    # Print configuration
    log.info(OmegaConf.to_yaml(cfg))
    
    # Set random seed
    rng = jax.random.PRNGKey(cfg.seed)
    
    # Create output directory
    if cfg.output.dir is not None:
        output_dir = cfg.output.dir
        os.makedirs(output_dir, exist_ok=True)
    else:
        output_dir = os.getcwd()
    log.info(f"Output directory: {output_dir}")

    # Initialize wandb if enabled
    wandb_run = None
    if cfg.wandb.enabled:
        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            name=cfg.wandb.run_name,
            dir=output_dir,
            config=OmegaConf.to_container(cfg, resolve=True),
            group=cfg.wandb.group,
        )
        wandb_run = wandb.run

    # Generate circuit layer sizes
    input_n, output_n = cfg.circuit.input_bits, cfg.circuit.output_bits
    arity = cfg.circuit.arity
    if cfg.circuit.layer_sizes is None:
        layer_sizes = generate_layer_sizes(
            input_n, output_n, arity, layer_n=cfg.circuit.num_layers
        )
        with open_dict(cfg):
            cfg.circuit.layer_sizes = layer_sizes
    else:
        layer_sizes = cfg.circuit.layer_sizes
        
    # Generate dummy circuit
    test_key = jax.random.PRNGKey(cfg.test_seed)
    wires, logits = gen_circuit(
        test_key, cfg.circuit.layer_sizes, arity=cfg.circuit.arity
    )

    # Generate dummy graph
    graph = build_graph(
        wires=wires,
        logits=logits,
        input_n=input_n,
        arity=arity,
        circuit_hidden_dim=cfg.model.circuit_hidden_dim,
    )
    n_nodes = int(graph.n_node[0])

    log.info(f"Circuit layer sizes: {layer_sizes}")
    log.info(f"Number of nodes: {n_nodes}")

    # Get task data
    case_n = 1 << input_n
    x, y0 = get_task_data(
        cfg.circuit.task, case_n, input_bits=input_n, output_bits=output_n
    )

    # Initialize model
    rng, init_rng = jax.random.split(rng)
    instantiate_overrides = {"arity": arity, "rngs": nnx.Rngs(params=init_rng)}
    
    if cfg.model.type == "self_attention":
        instantiate_overrides["n_node"] = n_nodes

    # Instantiate the model using Hydra
    try:
        model = hydra.utils.instantiate(cfg.model, **instantiate_overrides)
    except Exception as e:
        log.error(f"Error instantiating model: {e}")
        raise

    # Set damage emission if specified
    if hasattr(cfg.training, 'damage_emission'):
        model.damage_emission = cfg.training.damage_emission
        log.info(f"Set damage_emission = {cfg.training.damage_emission}")

    # Count and log model parameters
    params = nnx.state(model, nnx.Param)
    total_params = jax.tree.reduce(lambda x, y: x + y.size, params, 0)
    log.info(f"Total number of params: {total_params:,}")

    # Train model with logging
    log.info(f"Starting {cfg.model.type.upper()} training with logit logging")
    model_results = train_model(
        # Initialization parameters
        key=cfg.seed,
        init_model=model,
        # Data parameters
        x_data=x,
        y_data=y0,
        layer_sizes=layer_sizes,
        circuit_hidden_dim=cfg.model.circuit_hidden_dim,
        arity=arity,
        # Training hyperparameters
        learning_rate=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
        epochs=min(cfg.training.epochs, 100),  # Limit for testing
        n_message_steps=cfg.training.n_message_steps,
        layer_neighbors=cfg.training.get("layer_neighbors", False),
        damage_emission=cfg.training.get("damage_emission", False),
        use_scan=cfg.training.use_scan,
        # Loss parameters
        loss_type=cfg.training.loss_type,
        # Wiring mode parameters
        meta_batch_size=cfg.training.meta_batch_size,
        wiring_fixed_key=jax.random.PRNGKey(cfg.test_seed),
        # Pool parameters
        pool_size=cfg.pool.size,
        reset_pool_fraction=cfg.pool.reset_fraction,
        reset_strategy=cfg.pool.reset_strategy,
        reset_pool_interval=cfg.pool.reset_interval,
        # Damage parameters
        damage_pool_enabled=cfg.pool.get("damage_pool_enabled", False),
        damage_pool_interval=cfg.pool.get("damage_pool_interval", 0),
        damage_pool_fraction=cfg.pool.get("damage_pool_fraction", 0.0),
        damage_strategy=cfg.pool.get("damage_strategy", "uniform"),
        damage_combined_weights=tuple(cfg.pool.get("damage_combined_weights", [0.5, 0.5])),
        damage_mode=cfg.pool.get("damage_mode", "shotgun"),
        damage_pool_damage_prob=cfg.pool.get("damage_prob", 0.0),
        greedy_ordered_indices=cfg.pool.get("greedy_ordered_indices", None),
        damage_eval_steps=cfg.pool.get("damage_eval_steps", 50),
        damage_min_pool_updates=cfg.pool.get("damage_min_pool_updates", 0),
        damage_max_pool_updates=cfg.pool.get("damage_max_pool_updates", 10),
        damage_seed=cfg.damage_seed,
        knockout_diversity=cfg.pool.get("damage_knockout_diversity", 0),
        # Learning rate scheduling
        lr_scheduler=cfg.training.lr_scheduler,
        lr_scheduler_params=cfg.training.lr_scheduler_params,
        # WandB parameters
        wandb_logging=cfg.wandb.enabled,
        log_interval=cfg.logging.log_interval,
        wandb_run_config=OmegaConf.to_container(cfg, resolve=True),
        # Training mode
        training_mode=cfg.training.training_mode,
        preconfig_steps=cfg.backprop.epochs,
        preconfig_lr=cfg.backprop.learning_rate,
        # Early stopping parameters
        stop_accuracy_enabled=cfg.early_stop.enabled,
        stop_accuracy_threshold=cfg.early_stop.threshold,
        stop_accuracy_metric=cfg.early_stop.metric,
        stop_accuracy_source=cfg.early_stop.source,
        stop_accuracy_patience=cfg.early_stop.patience,
        stop_accuracy_min_epochs=cfg.early_stop.min_epochs,
    )

    # Close wandb if enabled
    if cfg.wandb.enabled:
        cleanup_redundant_wandb_artifacts(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            run_id=wandb_run.id,
            dry_run=False,
            verbose=True,
        )
        wandb.finish()

    return model_results


if __name__ == "__main__":
    main()
