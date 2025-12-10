"""
Minimal base script for loading checkpoints and running eval_no_damage evaluation.

This script implements the base pattern from docs/viz_base_chkpt.md for loading
a model checkpoint and running evaluation matching the training loop patterns.
"""

import argparse
import jax
import jax.numpy as jp
from omegaconf import OmegaConf

from boolean_nca_cc.circuits.tasks import get_task_data
from boolean_nca_cc.circuits.data_split import split_input_combinations
from boolean_nca_cc.circuits.model import gen_circuit, generate_layer_sizes
from boolean_nca_cc.training.checkpointing import (
    load_config_from_wandb,
    load_model_from_config_and_checkpoint,
    derive_checkpoint_metric_from_config,
)
from boolean_nca_cc.training.evaluation import (
    evaluate_circuits_in_chunks,
    evaluate_model_stepwise_batched,
)
from boolean_nca_cc.training.preconfigure import preconfigure_circuit_logits


def load_model_and_data(run_id: str, use_best_model: bool = True):
    """
    Load model, config, and prepare data splits.
    
    Args:
        run_id: WandB run ID to load from
        use_best_model: If True, load best model based on config checkpoint settings
        
    Returns:
        Tuple of (model, config, x_train, y_train, x_test, y_test, base_wires, base_logits, layer_sizes)
    """
    print(f"Loading model from run_id: {run_id}")
    
    # 1. Load config and model
    metric_name = None
    prefer_metric = None
    
    if use_best_model:
        # First, load config to get checkpoint settings (for metric derivation)
        temp_config, _, _ = load_config_from_wandb(
            run_id=run_id,
            filename="latest_checkpoint",  # Just to get config, not the actual model
            select_by_best_metric=False,
        )
        
        # Derive metric name from config's checkpoint settings
        metric_name, prefer_metric = derive_checkpoint_metric_from_config(temp_config)
        print(f"Using checkpoint metric from config: {metric_name} (prefer: {prefer_metric})")
        
        # Now load the actual best model with the correct metric
        config, checkpoint_path, run_id = load_config_from_wandb(
            run_id=run_id,
            filename="best_model",
            select_by_best_metric=True,
            metric_name=metric_name,
            prefer_metric=prefer_metric,
        )
    else:
        config, checkpoint_path, run_id = load_config_from_wandb(
            run_id=run_id,
            filename="latest_checkpoint",
        )
    
    print(f"Loaded config from checkpoint: {checkpoint_path}")
    print(f"Run ID: {run_id}")
    
    # Load model from config and checkpoint
    model, loaded_dict = load_model_from_config_and_checkpoint(
        config=config,
        checkpoint_path=checkpoint_path,
        run_id=run_id,
        seed=0,
    )
    
    # Verify checkpoint epoch and compare with wandb summary
    import wandb
    api = wandb.Api()
    run_obj = api.run(f"marcello-barylli-growai/boolean-nca-cc/{run_id}")
    summary = run_obj.summary
    
    # Get epoch from loaded checkpoint
    checkpoint_epoch = None
    if "config" in loaded_dict:
        checkpoint_config = loaded_dict["config"]
        if isinstance(checkpoint_config, dict):
            checkpoint_epoch = checkpoint_config.get("epoch")
        else:
            checkpoint_epoch = getattr(checkpoint_config, "epoch", None)
    
    # Get best epoch from wandb summary
    best_epoch = summary.get("best/epoch")
    
    print(f"\n=== Checkpoint Verification ===")
    if checkpoint_epoch is not None:
        print(f"Loaded checkpoint epoch: {checkpoint_epoch}")
    if best_epoch is not None:
        print(f"Best epoch from wandb summary: {best_epoch}")
    if checkpoint_epoch is not None and best_epoch is not None:
        if checkpoint_epoch != best_epoch:
            print(f"⚠️  WARNING: Loaded checkpoint (epoch {checkpoint_epoch}) does not match best epoch ({best_epoch})!")
        else:
            print(f"✓ Checkpoint epoch matches best epoch")
    
    # Also check the metric value
    if use_best_model and metric_name:
        metric_value = summary.get(metric_name)
        if metric_value is not None:
            print(f"Best metric value ({metric_name}): {metric_value}")
    print("=" * 30)
    
    print("Model loaded successfully")
    
    # 2. Generate data and splits
    print(f"Generating task data: {config.circuit.task}")
    x_data, y_data = get_task_data(
        task_name=config.circuit.task,
        case_n=2**config.circuit.input_bits,
        input_bits=config.circuit.input_bits,
        output_bits=config.circuit.output_bits,
    )
    
    # Split if enabled (matches training)
    if config.eval.input_split_enabled:
        print(f"Splitting data: {config.eval.input_train_fraction*100:.0f}% train, "
              f"{(1-config.eval.input_train_fraction)*100:.0f}% test")
        x_train, y_train, x_test, y_test = split_input_combinations(
            x_data=x_data,
            y_data=y_data,
            train_fraction=config.eval.input_train_fraction,
            seed=config.eval.input_split_seed,
            shuffle=True,
        )
        print(f"Train set: {x_train.shape[0]} combinations, Test set: {x_test.shape[0]} combinations")
    else:
        x_train, y_train = x_data, y_data
        x_test, y_test = x_data, y_data
        print("Input split disabled - using all combinations for training and evaluation")
    
    # 3. Generate layer sizes
    if config.circuit.layer_sizes is None:
        layer_sizes = generate_layer_sizes(
            input_n=config.circuit.input_bits,
            output_n=config.circuit.output_bits,
            arity=config.circuit.arity,
            layer_n=config.circuit.num_layers,
        )
    else:
        layer_sizes = config.circuit.layer_sizes
    
    print(f"Layer sizes: {layer_sizes}")
    
    # 4. Generate base circuit
    # Get wiring key - check if wiring_fixed_key exists in config, otherwise use test_seed
    wiring_seed = OmegaConf.select(config, "wiring_fixed_key", default=None)
    if wiring_seed is None:
        wiring_seed = OmegaConf.select(config, "test_seed", default=42)
    wiring_key = jax.random.PRNGKey(wiring_seed)
    
    if config.training.training_mode == "growth":
        print("Growth mode: generating random circuit")
        base_wires, base_logits = gen_circuit(
            wiring_key,
            layer_sizes,
            arity=config.circuit.arity
        )
    elif config.training.training_mode == "repair":
        print("Repair mode: preconfiguring circuit")
        # Get backprop config for preconfiguration
        backprop_config = OmegaConf.select(config, "backprop", default={})
        base_wires, base_logits = preconfigure_circuit_logits(
            wiring_key=wiring_key,
            layer_sizes=layer_sizes,
            arity=config.circuit.arity,
            x_data=x_data,  # Use full data for preconfig
            y_data=y_data,
            loss_type=config.training.loss_type,
            steps=OmegaConf.select(config.training, "preconfig_steps", default=200),
            lr=OmegaConf.select(config.training, "preconfig_lr", default=1e-2),
            optimizer=OmegaConf.select(backprop_config, "optimizer", default="adam"),
            weight_decay=OmegaConf.select(backprop_config, "weight_decay", default=0.0),
            beta1=OmegaConf.select(backprop_config, "beta1", default=0.9),
            beta2=OmegaConf.select(backprop_config, "beta2", default=0.999),
        )
    else:
        raise ValueError(f"Unknown training_mode: {config.training.training_mode}")
    
    print("Base circuit generated")
    
    return model, config, x_train, y_train, x_test, y_test, base_wires, base_logits, layer_sizes


def run_eval_no_damage(
    model,
    base_wires,
    base_logits,
    x_data,
    y_data,
    config,
    layer_sizes,
):
    """
    Run no-damage evaluation matching train_loop.py pattern.
    
    Args:
        model: Loaded model
        base_wires: Base circuit wires
        base_logits: Base circuit logits
        x_data: Input data
        y_data: Target data
        config: Config object
        layer_sizes: Layer sizes list
        
    Returns:
        Dictionary with evaluation metrics
    """
    print("Running eval_no_damage evaluation...")
    
    # Replicate base circuit for batch
    eval_batch_size = config.eval.periodic_eval_batch_size
    eval_wires = jax.tree.map(
        lambda x: jp.repeat(x[None, ...], eval_batch_size, axis=0),
        base_wires
    )
    eval_logits = jax.tree.map(
        lambda x: jp.repeat(x[None, ...], eval_batch_size, axis=0),
        base_logits
    )
    
    # Run evaluation (no damage)
    step_metrics = evaluate_circuits_in_chunks(
        eval_fn=evaluate_model_stepwise_batched,
        wires=eval_wires,
        logits=eval_logits,
        knockout_patterns=None,  # No damage
        target_chunk_size=eval_batch_size,
        model=model,
        x_data=x_data,
        y_data=y_data,
        input_n=config.circuit.input_bits,
        arity=config.circuit.arity,
        circuit_hidden_dim=config.circuit.circuit_hidden_dim,
        n_message_steps=config.eval.periodic_eval_inner_steps,
        loss_type=config.training.loss_type,
        layer_sizes=layer_sizes,
        return_per_pattern=False,
        layer_neighbors=config.training.layer_neighbors,
        # Disable damage injection
        damage_mode="greedy",  # Won't matter
        damage_injection_mode="single",
        max_damage_per_circuit=1,
        greedy_ordered_indices=None,
        knockout_vocabulary=None,
    )
    
    # Extract metrics
    final_metrics = {
        "final_loss": step_metrics["soft_loss"][-1],
        "final_hard_loss": step_metrics["hard_loss"][-1],
        "final_accuracy": step_metrics["soft_accuracy"][-1],
        "final_hard_accuracy": step_metrics["hard_accuracy"][-1],
        "final_full_map_accuracy": step_metrics["full_map_accuracy"][-1],
    }
    
    print("\n=== Evaluation Results ===")
    print(f"Final Loss: {final_metrics['final_loss']:.6f}")
    print(f"Final Hard Loss: {final_metrics['final_hard_loss']:.6f}")
    print(f"Final Accuracy: {final_metrics['final_accuracy']:.6f}")
    print(f"Final Hard Accuracy: {final_metrics['final_hard_accuracy']:.6f}")
    print(f"Final Full Map Accuracy: {final_metrics['final_full_map_accuracy']:.6f}")
    print("=" * 30)
    
    return final_metrics, step_metrics


def main():
    parser = argparse.ArgumentParser(
        description="Load checkpoint and run eval_no_damage evaluation"
    )
    parser.add_argument(
        "--run_id",
        type=str,
        required=True,
        help="WandB run ID to load model from",
    )
    parser.add_argument(
        "--use_best_model",
        action="store_true",
        default=True,
        help="Load best model based on config checkpoint settings (default: True)",
    )
    parser.add_argument(
        "--use_latest",
        action="store_true",
        help="Load latest checkpoint instead of best model",
    )
    parser.add_argument(
        "--eval_on_train",
        action="store_true",
        help="Also run evaluation on training data (when split is enabled)",
    )
    
    args = parser.parse_args()
    
    use_best_model = args.use_best_model and not args.use_latest
    
    # Load model and prepare data
    model, config, x_train, y_train, x_test, y_test, base_wires, base_logits, layer_sizes = (
        load_model_and_data(args.run_id, use_best_model=use_best_model)
    )
    
    # Run evaluation on test data
    print("\n" + "=" * 50)
    print("Running evaluation on TEST data")
    print("=" * 50)
    final_metrics_test, step_metrics_test = run_eval_no_damage(
        model=model,
        base_wires=base_wires,
        base_logits=base_logits,
        x_data=x_test,
        y_data=y_test,
        config=config,
        layer_sizes=layer_sizes,
    )
    
    # Optionally run on train data
    if args.eval_on_train and config.eval.input_split_enabled:
        print("\n" + "=" * 50)
        print("Running evaluation on TRAIN data")
        print("=" * 50)
        final_metrics_train, step_metrics_train = run_eval_no_damage(
            model=model,
            base_wires=base_wires,
            base_logits=base_logits,
            x_data=x_train,
            y_data=y_train,
            config=config,
            layer_sizes=layer_sizes,
        )


if __name__ == "__main__":
    main()

