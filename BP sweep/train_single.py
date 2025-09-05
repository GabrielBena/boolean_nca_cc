#!/usr/bin/env python3
"""
Single Experiment Training Script with WandB Integration

This script runs a single training experiment with full Hydra configuration support.
All parameters can be overridden via command line for hyperparameter optimization.

Usage:
    python train_single.py                                    # Default config
    python train_single.py task=binary_multiply_6bit          # Different task
    python train_single.py sparsity=l1_strong                 # Different sparsity
    python train_single.py optimizer.learning_rate=0.1        # Override specific params
    python train_single.py split_seed=123 circuit_seed=456    # Different seeds
"""

import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import sys

import hydra
import jax
import numpy as np
import optax
from omegaconf import DictConfig, OmegaConf
from tqdm import trange

import wandb

# Add parent directory to path for imports
sys.path.append("..")

from boolean_nca_cc.circuits.model import gen_circuit, generate_layer_sizes
from boolean_nca_cc.circuits.tasks import get_task_data
from boolean_nca_cc.circuits.train import TrainState, create_sparse_optimizer, train_step


def create_optimizer(cfg: DictConfig):
    """Create optimizer from configuration."""
    opt_cfg = cfg.optimizer

    if "adam" in opt_cfg.name:
        if opt_cfg.get("weight_decay", None) is None or opt_cfg.weight_decay == 0:
            base_opt = optax.adam(learning_rate=opt_cfg.learning_rate, b1=opt_cfg.b1, b2=opt_cfg.b2)
            print("Using Adam optimizer")
        else:
            base_opt = optax.adamw(
                learning_rate=opt_cfg.learning_rate,
                b1=opt_cfg.b1,
                b2=opt_cfg.b2,
                weight_decay=opt_cfg.weight_decay,
            )
            print("Using AdamW optimizer")
    else:
        raise ValueError(f"Unknown optimizer: {opt_cfg.name}")

    # Add sparsity if enabled and type is not "none"
    sparsity_enabled = getattr(cfg.sparsity, "weight", 0.0) > 0.0
    if sparsity_enabled:
        return create_sparse_optimizer(
            base_optimizer=base_opt,
            sparsity_type=cfg.sparsity.type,
            sparsity_weight=cfg.sparsity.weight,
        )
    else:
        return base_opt


def init_wandb(cfg: DictConfig):
    """Initialize wandb with configuration."""
    # Convert OmegaConf to regular dict for wandb
    config_dict = OmegaConf.to_container(cfg, resolve=True)

    # Initialize wandb
    wandb.init(
        project="boolean_bp_sweep",
        entity="m2snn",
        config=config_dict,
        tags=[
            cfg.task.name,
            f"sparsity_{cfg.sparsity.type}"
            if (getattr(cfg.sparsity, "weight", 0.0) > 0.0)
            else "no_sparsity",
            cfg.optimizer.name,
            f"{cfg.cross_validation.num_splits}_splits",
        ],
    )

    return wandb.run


def run_single_split(cfg: DictConfig, split_seed: int, split_idx: int) -> dict:
    """Run training on a single train/test split."""

    # Generate circuit
    layer_sizes = generate_layer_sizes(
        cfg.task.input_n, cfg.task.output_n, cfg.task.arity, cfg.task.layer_n, cfg.task.width_factor
    )
    circuit_key = jax.random.PRNGKey(cfg.circuit_seed)
    wires, logits = gen_circuit(circuit_key, layer_sizes, arity=cfg.task.arity)

    # Get data for this split
    case_n = 1 << cfg.task.input_n
    train_data, test_data, _ = get_task_data(
        cfg.task.name,
        case_n,
        input_bits=cfg.task.input_n,
        output_bits=cfg.task.output_n,
        train_test_split=True,
        test_ratio=cfg.task.test_ratio,
        seed=split_seed,
    )
    x_train, y_train = train_data
    x_test, y_test = test_data

    # Create optimizer
    optimizer = create_optimizer(cfg)

    # Initialize training
    state = TrainState(params=logits, opt_state=optimizer.init(logits))

    # Training loop
    train_accs, test_accs = [], []
    train_losses, test_losses = [], []

    pbar = trange(
        cfg.training.num_steps, desc=f"Split {split_idx + 1}/{cfg.cross_validation.num_splits}"
    )

    for step in pbar:
        train_loss, train_aux, test_loss, test_aux, state = train_step(
            state,
            optimizer,
            wires,
            x=x_train,
            y0=y_train,
            x_test=x_test,
            y_test=y_test,
            loss_type=cfg.training.loss_type,
            do_train=True,
        )

        train_acc = float(train_aux["accuracy"])
        test_acc = float(test_aux["accuracy"])

        train_accs.append(train_acc)
        test_accs.append(test_acc)
        train_losses.append(float(train_loss))
        test_losses.append(float(test_loss))

        # Update progress bar
        pbar.set_postfix(
            {
                "train_acc": f"{train_acc:.3f}",
                "test_acc": f"{test_acc:.3f}",
                "gap": f"{train_acc - test_acc:.3f}",
            }
        )

        # Log to wandb with split prefix
        if cfg.wandb.enabled and (
            step % cfg.wandb.log_interval == 0 or step == cfg.training.num_steps - 1
        ):
            metrics = {
                f"split_{split_idx}/step": step,
                f"split_{split_idx}/train_accuracy": train_acc,
                f"split_{split_idx}/train_loss": float(train_loss),
                f"split_{split_idx}/test_accuracy": test_acc,
                f"split_{split_idx}/test_loss": float(test_loss),
                f"split_{split_idx}/generalization_gap": train_acc - test_acc,
            }

            # Add sparsity metrics
            sparsity_enabled = getattr(cfg.sparsity, "weight", 0.0) > 0.0
            if sparsity_enabled:
                from boolean_nca_cc.circuits.train import compute_lut_sparsity_loss

                for stype in ["l1", "binary", "entropy"]:
                    sparsity_val = float(compute_lut_sparsity_loss(state.params, stype))
                    metrics[f"split_{split_idx}/sparsity_{stype}"] = sparsity_val

            # wandb.log(metrics)

    pbar.close()

    # Compute final metrics for this split
    final_train_acc = train_accs[-1]
    final_test_acc = test_accs[-1]
    generalization_gap = final_train_acc - final_test_acc
    train_acc_std = float(np.std(train_accs[-20:]))
    test_acc_std = float(np.std(test_accs[-20:]))

    # Final sparsity metrics
    sparsity_metrics = {}
    sparsity_enabled = getattr(cfg.sparsity, "weight", 0.0) > 0.0
    if sparsity_enabled:
        from boolean_nca_cc.circuits.train import compute_lut_sparsity_loss

        for stype in ["l1", "binary", "entropy"]:
            sparsity_metrics[f"sparsity_{stype}"] = float(
                compute_lut_sparsity_loss(state.params, stype)
            )

    return {
        "split_idx": split_idx,
        "split_seed": split_seed,
        "final_train_acc": final_train_acc,
        "final_test_acc": final_test_acc,
        "generalization_gap": generalization_gap,
        "train_acc_std": train_acc_std,
        "test_acc_std": test_acc_std,
        "train_samples": len(x_train),
        "test_samples": len(x_test),
        "train_accs": train_accs,
        "test_accs": test_accs,
        **sparsity_metrics,
    }


@hydra.main(version_base=None, config_path="configs", config_name="config")
def train_single(cfg: DictConfig) -> None:
    """Main training function with cross-validation support."""

    print("=" * 70)
    print("🧠 BOOLEAN CIRCUIT TRAINING (K-FOLD CROSS-VALIDATION)")
    print("=" * 70)
    print(f"Task: {cfg.task.name} ({cfg.task.input_n}→{cfg.task.output_n} bits)")
    print(f"Optimizer: {cfg.optimizer.name} (lr={cfg.optimizer.learning_rate})")
    sparsity_enabled = getattr(cfg.sparsity, "weight", 0.0) > 0.0
    print(f"Sparsity: {'Enabled' if sparsity_enabled else 'Disabled'}")
    if sparsity_enabled:
        print(f"  Type: {cfg.sparsity.type}, Weight: {cfg.sparsity.weight}")
    print(f"Steps per split: {cfg.training.num_steps}")
    print(f"Cross-validation: {cfg.cross_validation.num_splits} splits")
    print(
        f"Base split seed: {cfg.cross_validation.base_split_seed}, Circuit seed: {cfg.circuit_seed}"
    )

    # Get circuit architecture info
    layer_sizes = generate_layer_sizes(
        cfg.task.input_n, cfg.task.output_n, cfg.task.arity, cfg.task.layer_n, cfg.task.width_factor
    )
    print(f"📊 Circuit architecture: {layer_sizes}")

    # Get data size info
    case_n = 1 << cfg.task.input_n
    _, test_data, _ = get_task_data(
        cfg.task.name,
        case_n,
        input_bits=cfg.task.input_n,
        output_bits=cfg.task.output_n,
        train_test_split=True,
        test_ratio=cfg.task.test_ratio,
        seed=cfg.cross_validation.base_split_seed,
    )
    train_size = case_n - len(test_data[0])
    test_size = len(test_data[0])
    print(f"📈 Data per split: ~{train_size} train, ~{test_size} test samples")
    print()

    # Initialize wandb
    if cfg.wandb.enabled:
        run = init_wandb(cfg)
        print(f"🔗 WandB run: {run.url}")
        print()

    # Generate split seeds deterministically
    np.random.seed(cfg.cross_validation.base_split_seed)
    split_seeds = np.random.randint(0, 10000, cfg.cross_validation.num_splits)

    print(f"🔀 Split seeds: {list(split_seeds)}")
    print()

    # Run training on each split
    split_results = []
    print("🚀 Starting cross-validation training...")

    for split_idx in range(cfg.cross_validation.num_splits):
        split_seed = int(split_seeds[split_idx])
        print(f"\n{'=' * 50}")
        print(f"🔄 SPLIT {split_idx + 1}/{cfg.cross_validation.num_splits} (seed={split_seed})")
        print(f"{'=' * 50}")

        split_result = run_single_split(cfg, split_seed, split_idx)
        split_results.append(split_result)

        # Print split results
        print(f"\n📊 Split {split_idx + 1} Results:")
        print(
            f"  Train Accuracy: {split_result['final_train_acc']:.4f} ± {split_result['train_acc_std']:.4f}"
        )
        print(
            f"  Test Accuracy:  {split_result['final_test_acc']:.4f} ± {split_result['test_acc_std']:.4f}"
        )
        print(f"  Gen. Gap:       {split_result['generalization_gap']:.4f}")

    # Aggregate results across splits
    print(f"\n{'=' * 70}")
    print("📈 CROSS-VALIDATION RESULTS SUMMARY")
    print(f"{'=' * 70}")

    # Compute aggregated metrics
    train_accs = [r["final_train_acc"] for r in split_results]
    test_accs = [r["final_test_acc"] for r in split_results]
    gen_gaps = [r["generalization_gap"] for r in split_results]

    agg_metrics = {
        "mean_train_acc": float(np.mean(train_accs)),
        "std_train_acc": float(np.std(train_accs)),
        "mean_test_acc": float(np.mean(test_accs)),
        "std_test_acc": float(np.std(test_accs)),
        "mean_gen_gap": float(np.mean(gen_gaps)),
        "std_gen_gap": float(np.std(gen_gaps)),
        "min_test_acc": float(np.min(test_accs)),
        "max_test_acc": float(np.max(test_accs)),
        "min_gen_gap": float(np.min(gen_gaps)),
        "max_gen_gap": float(np.max(gen_gaps)),
    }

    # Aggregate sparsity metrics if enabled
    sparsity_enabled = getattr(cfg.sparsity, "weight", 0.0) > 0.0
    if sparsity_enabled:
        for stype in ["l1", "binary", "entropy"]:
            sparsity_vals = [r[f"sparsity_{stype}"] for r in split_results]
            agg_metrics[f"mean_sparsity_{stype}"] = float(np.mean(sparsity_vals))
            agg_metrics[f"std_sparsity_{stype}"] = float(np.std(sparsity_vals))

    # Print summary
    print(f"Cross-Validation Results ({cfg.cross_validation.num_splits} splits):")
    print(
        f"  Train Accuracy: {agg_metrics['mean_train_acc']:.4f} ± {agg_metrics['std_train_acc']:.4f}"
    )
    print(
        f"  Test Accuracy:  {agg_metrics['mean_test_acc']:.4f} ± {agg_metrics['std_test_acc']:.4f}"
    )
    print(f"  Gen. Gap:       {agg_metrics['mean_gen_gap']:.4f} ± {agg_metrics['std_gen_gap']:.4f}")
    print(
        f"  Test Acc Range: [{agg_metrics['min_test_acc']:.4f}, {agg_metrics['max_test_acc']:.4f}]"
    )
    print(f"  Gen. Gap Range: [{agg_metrics['min_gen_gap']:.4f}, {agg_metrics['max_gen_gap']:.4f}]")

    if sparsity_enabled:
        print("\n  Sparsity Metrics:")
        for stype in ["l1", "binary", "entropy"]:
            mean_key = f"mean_sparsity_{stype}"
            std_key = f"std_sparsity_{stype}"
            print(f"    {stype.upper()}: {agg_metrics[mean_key]:.4f} ± {agg_metrics[std_key]:.4f}")

    # Log aggregated results to wandb
    if cfg.wandb.enabled:
        # Log individual split final results
        for i, result in enumerate(split_results):
            split_final_metrics = {
                f"split_{i}/final_train_acc": result["final_train_acc"],
                f"split_{i}/final_test_acc": result["final_test_acc"],
                f"split_{i}/final_gen_gap": result["generalization_gap"],
                f"split_{i}/train_acc_std": result["train_acc_std"],
                f"split_{i}/test_acc_std": result["test_acc_std"],
            }
            if sparsity_enabled:
                for stype in ["l1", "binary", "entropy"]:
                    split_final_metrics[f"split_{i}/final_sparsity_{stype}"] = result[
                        f"sparsity_{stype}"
                    ]
            wandb.log(split_final_metrics)

        # Log aggregated metrics
        wandb.log(
            {
                "cv_summary/num_splits": cfg.cross_validation.num_splits,
                **{f"cv_summary/{k}": v for k, v in agg_metrics.items()},
            }
        )

        # Mark run as finished
        wandb.finish()

    print("\n✅ Cross-validation training complete!")


if __name__ == "__main__":
    train_single()
