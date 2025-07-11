#!/usr/bin/env python3
"""
Main training script for Boolean Circuit NCA optimization using GNNs and Self-Attention.

This script trains models that learn to simulate the inner loop of backpropagation
when training boolean circuits, using either Graph Neural Networks or Self-Attention.
"""

import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import logging
import jax
import hydra
from omegaconf import DictConfig, OmegaConf, open_dict
import wandb
from flax import nnx
import pandas as pd

from boolean_nca_cc.circuits.model import gen_circuit
from boolean_nca_cc.circuits.tasks import get_task_data
from boolean_nca_cc import generate_layer_sizes

from boolean_nca_cc.training.train_loop import train_model

from boolean_nca_cc.utils.graph_builder import build_graph
from boolean_nca_cc.training.utils import (
    save_checkpoint,
    cleanup_redundant_wandb_artifacts,
)

# Configure logging
log = logging.getLogger(__name__)


def create_and_save_final_results(
    cfg,
    model_results,
    eval_results,  # Kept for backward compatibility but not used
    layer_sizes,
    output_dir,
    wandb_run=None,
    total_params=None,
):
    """
    Create comprehensive final results dictionary, save as CSV, and log results.

    Args:
        cfg: Configuration object
        model_results: Results from model training
        eval_results: Results from evaluation (not used)
        layer_sizes: Circuit layer sizes
        output_dir: Directory to save results
        wandb_run: WandB run object (optional)

    Returns:
        dict: Final results dictionary
    """
    # Collect comprehensive final results
    final_results = {
        # Model and training configuration
        "model_type": cfg.model.type,
        "wiring_mode": cfg.training.wiring_mode,
        "loss_type": cfg.training.loss_type,
        "learning_rate": cfg.training.learning_rate,
        "epochs_completed": len(model_results["losses"]),
        "total_epochs_planned": cfg.training.epochs
        or 2**cfg.training.epochs_power_of_2,
        "early_stopped": model_results.get("early_stopped", False),
        "early_stop_epoch": model_results.get("early_stop_epoch", None),
        "first_threshold_epoch": model_results.get("first_threshold_epoch", None),
        # Meta-learning metrics (final training values)
        "meta_loss": model_results["losses"][-1],
        "meta_hard_loss": model_results["hard_losses"][-1],
        "meta_accuracy": model_results["accuracies"][-1],
        "meta_hard_accuracy": model_results["hard_accuracies"][-1],
        # Best model performance
        "best_metric": model_results.get("best_metric", None),
        "best_metric_value": model_results.get("best_metric_value", None),
        # Circuit and model architecture info
        "input_bits": cfg.circuit.input_bits,
        "output_bits": cfg.circuit.output_bits,
        "num_layers": len(layer_sizes),
        "total_nodes": sum(size[0] for size in layer_sizes),
        "circuit_hidden_dim": cfg.model.circuit_hidden_dim,
        "attention_dim": cfg.model.get("attention_dim", None),
        "mlp_dim_multiplier": cfg.model.get("mlp_dim_multiplier", None),
        "mlp_dim": cfg.model.get("mlp_dim", None),
        "total_parameters": total_params,
        "message_steps": cfg.training.n_message_steps,
        "pool_size": cfg.pool.size,
        "pool_initial_diversity": cfg.pool.initial_diversity,
    }

    # Create pandas DataFrame
    results_df = pd.DataFrame([final_results])

    # Save DataFrame to CSV
    results_csv_path = os.path.join(output_dir, "final_results.csv")
    results_df.to_csv(results_csv_path, index=False)
    log.info(f"Final results saved to: {results_csv_path}")

    # Display the DataFrame
    log.info("Final Results Summary:")
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    log.info("\n" + str(results_df.T))  # Transpose for better readability

    # Log to wandb if enabled
    if cfg.wandb.enabled and wandb_run:
        # Log the DataFrame as a table
        wandb.log({"final_results_table": wandb.Table(dataframe=results_df)})
        # Also log individual final metrics for easy access
        wandb.log({f"final/{k}": v for k, v in final_results.items() if v is not None})

    # Final log (traditional format for backward compatibility)
    log.info(f"Training complete. Final results:")
    log.info(f"  Meta Loss: {model_results['losses'][-1]:.4f}")
    log.info(f"  Meta Hard Loss: {model_results['hard_losses'][-1]:.4f}")
    log.info(f"  Meta Accuracy: {model_results['accuracies'][-1]:.4f}")
    log.info(f"  Meta Hard Accuracy: {model_results['hard_accuracies'][-1]:.4f}")

    return final_results


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Main training function using Hydra for configuration.

    Args:
        cfg: Hydra configuration object
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

    # Prepare checkpoint directory
    if cfg.checkpoint.enabled:
        checkpoint_dir = os.path.join(output_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
    else:
        checkpoint_dir = None

    # Train model
    log.info(f"Starting {cfg.model.type.upper()} training")
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
        wiring_mode=cfg.training.wiring_mode,
        meta_batch_size=cfg.training.meta_batch_size,
        wiring_fixed_key=jax.random.PRNGKey(cfg.test_seed),
        # Pool parameters
        pool_size=cfg.pool.size,
        reset_pool_fraction=cfg.pool.reset_fraction,
        reset_strategy=cfg.pool.reset_strategy,
        reset_pool_interval=cfg.pool.reset_interval,
        initial_diversity=cfg.pool.initial_diversity,
        # Perturbation configurations
        persistent_knockout_config=cfg.pool.get("persistent_knockout", None),
        knockout_diversity=cfg.pool.get("persistent_knockout", {}).get(
            "knockout_diversity"
        ),
        # Learning rate scheduling
        lr_scheduler=cfg.training.lr_scheduler,
        lr_scheduler_params=cfg.training.lr_scheduler_params,
        # Checkpoint parameters
        # checkpoint_enabled=cfg.checkpoint.enabled,
        # checkpoint_dir=checkpoint_dir,
        # checkpoint_interval=cfg.checkpoint.interval,
        save_best=cfg.checkpoint.save_best,
        best_metric=cfg.checkpoint.best_metric,
        save_stable_states=cfg.checkpoint.save_stable_states,
        # Periodic evaluation parameters
        # periodic_eval_enabled=cfg.eval.periodic.enabled,
        # periodic_eval_interval=cfg.eval.periodic.interval,
        # periodic_eval_inner_steps=cfg.eval.inner_steps,
        # periodic_eval_test_seed=cfg.test_seed,
        # periodic_eval_log_stepwise=cfg.eval.log_stepwise,
        # periodic_eval_batch_size=cfg.eval.batch_size,
        # periodic_eval_log_pool_scatter=cfg.eval.log_pool_scatter,
        # Knockout evaluation
        knockout_eval=cfg.eval.get("knockout_eval", None),
        # WandB parameters
        wandb_logging=cfg.wandb.enabled,
        log_interval=cfg.logging.log_interval,
        wandb_run_config=OmegaConf.to_container(cfg, resolve=True),
    )

    # # Save final model if checkpointing is enabled
    # if cfg.checkpoint.enabled and not cfg.wandb.enabled:
    #     # If wandb is enabled, checkpoints are already being saved during training
    #     save_checkpoint(
    #         model_results["model"],
    #         model_results["optimizer"],
    #         {
    #             "losses": model_results["losses"],
    #             "hard_losses": model_results["hard_losses"],
    #             "accuracies": model_results["accuracies"],
    #             "hard_accuracies": model_results["hard_accuracies"],
    #             "reset_steps": model_results.get("reset_steps", []),
    #         },
    #         cfg,
    #         cfg.training.epochs or 2**cfg.training.epochs_power_of_2,
    #         checkpoint_dir,
    #         filename="final_model.pkl",
    #     )

    if "metrics" in model_results:
        model_results.update(model_results["metrics"])

    # Collect comprehensive final results
    final_results = create_and_save_final_results(
        cfg,
        model_results,
        None,  # No eval results
        layer_sizes,
        output_dir,
        wandb_run,
        total_params,
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

    return final_results


if __name__ == "__main__":
    main()
