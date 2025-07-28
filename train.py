"""
Main training script for Boolean Circuit NCA optimization using GNNs and Self-Attention.

This script trains models that learn to simulate the inner loop of backpropagation
when training boolean circuits, using either Graph Neural Networks or Self-Attention.
"""

import os

import logging
import jax
import optax
import hydra
from omegaconf import DictConfig, OmegaConf, open_dict
import wandb
from tqdm.auto import tqdm
from functools import partial
from flax import nnx
import pandas as pd

from boolean_nca_cc.circuits.model import gen_circuit
from boolean_nca_cc.circuits.tasks import get_task_data
from boolean_nca_cc import generate_layer_sizes
from boolean_nca_cc.circuits.train import TrainState, loss_f_l4, loss_f_bce, train_step

from boolean_nca_cc.training.train_loop import (
    train_model,
)
from boolean_nca_cc.utils.graph_builder import build_graph
from boolean_nca_cc.training.utils import (
    cleanup_redundant_wandb_artifacts,
)

# Configure logging
log = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Main training function using Hydra for configuration.

    Args:
        cfg: Hydra configuration object
    """
    # Print configuration
    log.info(OmegaConf.to_yaml(cfg))
    
    # Debug: Check knockout_diversity value
    log.info(f"DEBUG: knockout_diversity value = {cfg.pool.persistent_knockout.knockout_diversity}")
    log.info(f"DEBUG: persistent_knockout_config = {cfg.pool.persistent_knockout}")

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
    if cfg.wandb.enabled:
        wandb.config.update({"layer_sizes": layer_sizes, "n_nodes": n_nodes})

    # Get task data
    case_n = 1 << input_n
    x, y0 = get_task_data(
        cfg.circuit.task, case_n, input_bits=input_n, output_bits=output_n
    )

    # Initialize model
    rng, init_rng = jax.random.split(rng)

    # Common overrides for hydra.instantiate
    # These are values computed in train.py or essential for all models
    instantiate_overrides = {"arity": arity, "rngs": nnx.Rngs(params=init_rng)}

    # Specific overrides based on model type, which should still be in the YAML.
    # Alternatively, we could inspect cfg.model._target_ if 'type' was removed.
    if cfg.model.type == "self_attention":
        instantiate_overrides["n_node"] = n_nodes
    # CircuitGNN does not require n_node in its constructor based on original setup.

    # Instantiate the model using Hydra
    try:
        model = hydra.utils.instantiate(cfg.model, **instantiate_overrides)
    except Exception as e:
        log.error(
            f"Error instantiating model {cfg.model._target_ if '_target_' in cfg.model else cfg.model.type}: {e}"
        )
        raise

    # Count and log model parameters with detailed breakdown
    log.info(f"Model instantiated: {cfg.model.type}")

    params = nnx.state(model, nnx.Param)
    module_param_counts = {}

    # Count parameters per module
    for module_name, module_params in params.items():
        num_params = jax.tree.reduce(lambda x, y: x + y.size, module_params, 0)
        module_param_counts[module_name] = num_params
        log.info(f"Number of params in {module_name}: {num_params:,}")

    # Calculate total parameters
    total_params = jax.tree.reduce(lambda x, y: x + y.size, params, 0)
    log.info(f"Total number of params: {total_params:,}")

    # Log to wandb if enabled
    if cfg.wandb.enabled:
        wandb_metrics = {
            "model/parameter_count": total_params,
        }
        # Log per-module counts as well
        for module_name, count in module_param_counts.items():
            wandb_metrics[f"model/params_{module_name}"] = count

        wandb.log(wandb_metrics)

    # Train model
    log.info(f"Starting {cfg.model.type.upper()} training")
    log.info(f"DEBUG: About to call train_model with knockout_diversity = {cfg.pool.persistent_knockout.knockout_diversity}")
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
        epochs=cfg.training.epochs or 2**cfg.training.epochs_power_of_2,
        n_message_steps=cfg.training.n_message_steps,
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
        # Perturbation configurations
        persistent_knockout_config=cfg.pool.persistent_knockout,
        knockout_diversity=cfg.pool.persistent_knockout.knockout_diversity,
        # Learning rate scheduling
        lr_scheduler=cfg.training.lr_scheduler,
        lr_scheduler_params=cfg.training.lr_scheduler_params,
        # Checkpoint parameters
        # best_metric_source=cfg.checkpoint.best_metric_source,
        # Knockout evaluation
        knockout_eval=cfg.eval.get("knockout_eval", None),
        # Periodic evaluation parameters
        periodic_eval_enabled=cfg.eval.get("periodic_eval_enabled", False),
        periodic_eval_inner_steps=cfg.eval.get("periodic_eval_inner_steps", 100),
        periodic_eval_interval=cfg.eval.get("periodic_eval_interval", 1024),
        periodic_eval_test_seed=cfg.eval.get("periodic_eval_test_seed", 42),
        periodic_eval_log_stepwise=cfg.eval.get("periodic_eval_log_stepwise", False),
        periodic_eval_batch_size=cfg.eval.get("periodic_eval_batch_size", 16),
        # WandB parameters
        wandb_logging=cfg.wandb.enabled,
        log_interval=cfg.logging.log_interval,
        wandb_run_config=OmegaConf.to_container(cfg, resolve=True),
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