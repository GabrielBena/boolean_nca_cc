#!/usr/bin/env python3
"""
Hydra-based Sparsity Comparison Study

This script uses Hydra for clean configuration management and supports
various study types through configuration composition.

Usage:
    python run_study_hydra.py                           # Default config
    python run_study_hydra.py experiment=full_study     # Full study
    python run_study_hydra.py task=binary_multiply_6bit # 6-bit task
    python run_study_hydra.py sparsity=l1_strong        # Strong L1 sparsity
    python run_study_hydra.py --multirun experiment=quick_study,full_study  # Multiple runs
"""

import itertools
import json

# Add parent directory to path for imports
import sys
from pathlib import Path

import hydra
import jax
import numpy as np
import optax
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from tqdm import trange

sys.path.append("..")

from boolean_nca_cc.circuits.model import gen_circuit, generate_layer_sizes
from boolean_nca_cc.circuits.tasks import get_task_data
from boolean_nca_cc.circuits.train import TrainState, create_sparse_optimizer, train_step


def create_optimizer(cfg: DictConfig):
    """Create optimizer from configuration."""
    opt_cfg = cfg.optimizer

    if opt_cfg.name == "adam":
        base_opt = optax.adam(learning_rate=opt_cfg.learning_rate, b1=opt_cfg.b1, b2=opt_cfg.b2)
    elif opt_cfg.name == "adamw":
        base_opt = optax.adamw(
            learning_rate=opt_cfg.learning_rate,
            b1=opt_cfg.b1,
            b2=opt_cfg.b2,
            weight_decay=opt_cfg.weight_decay,
        )
    else:
        raise ValueError(f"Unknown optimizer: {opt_cfg.name}")

    # Add sparsity if enabled
    if cfg.sparsity.enabled:
        return create_sparse_optimizer(
            base_optimizer=base_opt,
            sparsity_type=cfg.sparsity.type,
            sparsity_weight=cfg.sparsity.weight,
        )
    else:
        return base_opt


def run_single_experiment(
    cfg: DictConfig, split_seed: int, circuit_seed: int, sparsity_config: dict = None
) -> dict:
    """Run a single experiment with given configuration."""

    # Generate circuit
    layer_sizes = generate_layer_sizes(
        cfg.task.input_n, cfg.task.output_n, cfg.task.arity, cfg.task.layer_n, cfg.task.width_factor
    )
    circuit_key = jax.random.PRNGKey(circuit_seed)
    wires, logits = gen_circuit(circuit_key, layer_sizes, arity=cfg.task.arity)

    # Get data
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

    # Create optimizer (override sparsity if provided)
    if sparsity_config is not None:
        # Temporarily override sparsity config
        orig_sparsity = OmegaConf.to_container(cfg.sparsity)
        cfg.sparsity.update(sparsity_config)
        optimizer = create_optimizer(cfg)
        cfg.sparsity.update(orig_sparsity)  # Restore
        experiment_type = f"sparse_{sparsity_config['type']}_w{sparsity_config['weight']}"
    else:
        optimizer = create_optimizer(cfg)
        if cfg.sparsity.enabled:
            experiment_type = f"sparse_{cfg.sparsity.type}_w{cfg.sparsity.weight}"
        else:
            experiment_type = "baseline"

    # Initialize training
    state = TrainState(params=logits, opt_state=optimizer.init(logits))

    # Training loop
    train_accs, test_accs = [], []

    for step in range(cfg.experiment.num_steps):
        train_loss, train_aux, test_loss, test_aux, state = train_step(
            state,
            optimizer,
            wires,
            x=x_train,
            y0=y_train,
            x_test=x_test,
            y_test=y_test,
            loss_type=cfg.experiment.loss_type,
            do_train=True,
        )

        train_accs.append(float(train_aux["accuracy"]))
        test_accs.append(float(test_aux["accuracy"]))

    # Compute metrics
    final_train_acc = train_accs[-1]
    final_test_acc = test_accs[-1]
    generalization_gap = final_train_acc - final_test_acc

    # Sparsity metrics
    sparsity_metrics = {}
    if cfg.sparsity.enabled or sparsity_config is not None:
        from boolean_nca_cc.circuits.train import compute_lut_sparsity_loss

        for stype in ["l1", "binary", "entropy"]:
            sparsity_metrics[f"sparsity_{stype}"] = float(
                compute_lut_sparsity_loss(state.params, stype)
            )

    return {
        "experiment_type": experiment_type,
        "split_seed": split_seed,
        "circuit_seed": circuit_seed,
        "final_train_acc": final_train_acc,
        "final_test_acc": final_test_acc,
        "generalization_gap": generalization_gap,
        "train_acc_std": float(np.std(train_accs[-20:])),
        "test_acc_std": float(np.std(test_accs[-20:])),
        "train_samples": len(x_train),
        "test_samples": len(x_test),
        **sparsity_metrics,
    }


@hydra.main(version_base=None, config_path="configs", config_name="config")
def run_study(cfg: DictConfig) -> None:
    """Main study runner with Hydra configuration."""

    print("=" * 60)
    print(f"🔬 SPARSITY COMPARISON STUDY")
    print("=" * 60)
    print(f"Study: {cfg.experiment.name}")
    print(f"Task: {cfg.task.name} ({cfg.task.input_n}→{cfg.task.output_n} bits)")
    print(f"Description: {cfg.experiment.description}")
    print(f"Steps: {cfg.experiment.num_steps}")
    print(f"Split seeds: {len(cfg.experiment.split_seeds)}")
    print(f"Circuit seeds: {len(cfg.experiment.circuit_seeds)}")
    print(f"Sparsity types: {cfg.experiment.sparsity_types}")
    print(f"Sparsity weights: {cfg.experiment.sparsity_weights}")
    print()

    # Generate all experiment configurations
    experiments = []

    # Baseline experiments (no sparsity)
    for split_seed, circuit_seed in itertools.product(
        cfg.experiment.split_seeds, cfg.experiment.circuit_seeds
    ):
        experiments.append(
            {"split_seed": split_seed, "circuit_seed": circuit_seed, "sparsity_config": None}
        )

    # Sparse experiments
    for sparsity_type, sparsity_weight, split_seed, circuit_seed in itertools.product(
        cfg.experiment.sparsity_types,
        [w for w in cfg.experiment.sparsity_weights if w > 0],  # Skip 0.0
        cfg.experiment.split_seeds,
        cfg.experiment.circuit_seeds,
    ):
        experiments.append(
            {
                "split_seed": split_seed,
                "circuit_seed": circuit_seed,
                "sparsity_config": {
                    "enabled": True,
                    "type": sparsity_type,
                    "weight": sparsity_weight,
                },
            }
        )

    total_experiments = len(experiments)
    print(f"Total experiments: {total_experiments}")
    print()

    # Run experiments
    results = []

    pbar = trange(total_experiments, desc="Running experiments")
    for i, exp_config in enumerate(experiments):
        try:
            result = run_single_experiment(cfg, **exp_config)
            results.append(result)

            # Update progress bar
            exp_type = result["experiment_type"]
            gap = result["generalization_gap"]
            pbar.set_postfix(
                {
                    "type": exp_type[:15],
                    "gap": f"{gap:.3f}",
                    "test_acc": f"{result['final_test_acc']:.3f}",
                }
            )

        except Exception as e:
            print(f"Error in experiment {i + 1}: {e}")
            continue

        pbar.update(1)

    pbar.close()

    if len(results) == 0:
        print("❌ No successful experiments!")
        return

    # Convert to DataFrame and analyze
    df = pd.DataFrame(results)

    # Save results
    if cfg.save_results:
        results_path = Path(cfg.results_dir)
        results_path.mkdir(parents=True, exist_ok=True)

        # Save detailed results
        csv_path = results_path / "detailed_results.csv"
        df.to_csv(csv_path, index=False)

        # Save configuration
        config_path = results_path / "config.yaml"
        with open(config_path, "w") as f:
            OmegaConf.save(cfg, f)

        # Save summary
        summary = analyze_results(df)
        summary_path = results_path / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        print(f"📁 Results saved to: {results_path}")

    # Print summary
    print_summary(df)


def analyze_results(df: pd.DataFrame) -> dict:
    """Analyze results and generate summary statistics."""

    summary = {}

    for exp_type in df["experiment_type"].unique():
        subset = df[df["experiment_type"] == exp_type]

        summary[exp_type] = {
            "count": len(subset),
            "generalization_gap": {
                "mean": float(subset["generalization_gap"].mean()),
                "std": float(subset["generalization_gap"].std()),
                "min": float(subset["generalization_gap"].min()),
                "max": float(subset["generalization_gap"].max()),
            },
            "final_test_acc": {
                "mean": float(subset["final_test_acc"].mean()),
                "std": float(subset["final_test_acc"].std()),
                "min": float(subset["final_test_acc"].min()),
                "max": float(subset["final_test_acc"].max()),
            },
        }

    return summary


def print_summary(df: pd.DataFrame) -> None:
    """Print experiment summary."""

    print("\n" + "=" * 60)
    print("📊 RESULTS SUMMARY")
    print("=" * 60)

    # Group by experiment type
    grouped = (
        df.groupby("experiment_type")
        .agg(
            {
                "generalization_gap": ["mean", "std", "count"],
                "final_test_acc": ["mean", "std"],
                "final_train_acc": ["mean", "std"],
            }
        )
        .round(4)
    )

    print("\nGeneralization Gap (Train Acc - Test Acc):")
    print("Lower = Better Generalization")
    print("-" * 50)

    for exp_type in grouped.index:
        gap_mean = grouped.loc[exp_type, ("generalization_gap", "mean")]
        gap_std = grouped.loc[exp_type, ("generalization_gap", "std")]
        test_acc = grouped.loc[exp_type, ("final_test_acc", "mean")]
        count = int(grouped.loc[exp_type, ("generalization_gap", "count")])

        print(
            f"{exp_type:25s}: {gap_mean:6.3f} ± {gap_std:5.3f} "
            f"(test_acc: {test_acc:.3f}, n={count})"
        )

    # Best configurations
    print(f"\n🏆 Best Generalization:")
    best_gap = df.loc[df["generalization_gap"].idxmin()]
    print(f"   {best_gap['experiment_type']}: gap={best_gap['generalization_gap']:.3f}")

    print(f"\n🎯 Best Test Accuracy:")
    best_acc = df.loc[df["final_test_acc"].idxmax()]
    print(f"   {best_acc['experiment_type']}: acc={best_acc['final_test_acc']:.3f}")


if __name__ == "__main__":
    run_study()
