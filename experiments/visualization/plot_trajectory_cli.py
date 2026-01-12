#!/usr/bin/env python3
"""
CLI wrapper for plot_trajectory_from_checkpoint.

Provides command-line interface for generating trajectory plots from checkpoints.
Supports:
- Reversible/permanent damage
- Single/multi damage injection
- Vocabulary is automatically generated from config to match training
"""

import argparse
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from experiments.visualization.plot_trajectory import plot_trajectory_from_checkpoint
from boolean_nca_cc.training.pool.structural_perturbation import (
    DEFAULT_GREEDY_ORDERED_INDICES,
)
from boolean_nca_cc.training.checkpointing import (
    load_config_from_wandb,
    load_checkpoint,
)
from omegaconf import OmegaConf


def main():
    parser = argparse.ArgumentParser(
        description="Generate trajectory plots from model checkpoints",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Checkpoint loading
    checkpoint_group = parser.add_mutually_exclusive_group(required=False)
    checkpoint_group.add_argument(
        "--run-id",
        type=str,
        default="nypyrbwh",
        help="WandB run ID to load model from (default: nypyrbwh)",
    )
    checkpoint_group.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Local checkpoint path (.pkl file)",
    )
    
    parser.add_argument(
        "--use-best-model",
        action="store_true",
        default=True,
        help="Load best model based on config checkpoint settings (WandB only)",
    )
    parser.add_argument(
        "--use-latest",
        action="store_true",
        help="Load latest checkpoint instead of best model (WandB only)",
    )
    
    # Trajectory type
    parser.add_argument(
        "--trajectory-type",
        type=str,
        choices=["boolean_discovery", "damage_response"],
        default="damage_response",
        help="Type of trajectory to plot",
    )
    
    # Boolean discovery parameters
    parser.add_argument(
        "--eval-on-train",
        action="store_true",
        default=True,
        help="For boolean_discovery, also evaluate on train split",
    )
    
    # Damage behavior
    parser.add_argument(
        "--force-reversible",
        action="store_true",
        help="Force reversible damage behavior (overrides model's default)",
    )
    parser.add_argument(
        "--force-permanent",
        action="store_true",
        help="Force permanent damage behavior (overrides model's default)",
    )
    
    # Damage response parameters
    parser.add_argument(
        "--show-bp",
        action="store_true",
        help="Show backpropagation comparison trajectory",
    )
    parser.add_argument(
        "--no-ood",
        action="store_true",
        help="Don't show out-of-distribution (unseen) patterns",
    )
    parser.add_argument(
        "--damage-injection-mode",
        type=str,
        choices=["single", "multi"],
        default="single",
        help="Single or multi damage injection",
    )
    parser.add_argument(
        "--damage-mode",
        type=str,
        choices=["greedy", "greedy_vocabulary", "shotgun", "strip"],
        default="shotgun",
        help="Damage pattern type (default: shotgun - random sampling from all eligible gates)",
    )
    parser.add_argument(
        "--damage-start-offset",
        type=int,
        default=None,
        help="Number of steps before first damage injection (default: from config or 0)",
    )
    parser.add_argument(
        "--max-damage-per-circuit",
        type=int,
        default=None,
        help="Maximum damage events per circuit (default: from config or 10)",
    )
    parser.add_argument(
        "--greedy-injection-recover-steps",
        type=int,
        default=None,
        help="Recovery steps between damage injections (default: from config or 10)",
    )
    parser.add_argument(
        "--greedy-window-size",
        type=int,
        default=None,
        help="Window size for greedy patterns (default: from config or 1)",
    )
    parser.add_argument(
        "--greedy-ordered-indices",
        type=str,
        default=None,
        help="Comma-separated list of gate indices for greedy mode (e.g., '48,17,52,146'). If not provided, uses config or default.",
    )
    
    # Evaluation parameters
    parser.add_argument(
        "--n-message-steps",
        type=int,
        default=None,
        help="Number of message passing steps (default: from config)",
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=None,
        help="Evaluation batch size (default: from config)",
    )
    parser.add_argument(
        "--periodic-eval-test-seed",
        type=int,
        default=None,
        help="Seed for generating OOD patterns (default: from config)",
    )
    
    # Output parameters
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for figure (default: auto-generated based on run_id/checkpoint)",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Figure title (default: auto-generated)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Image resolution",
    )
    
    # WandB parameters
    parser.add_argument(
        "--project",
        type=str,
        default="boolean-nca-cc",
        help="WandB project name",
    )
    parser.add_argument(
        "--entity",
        type=str,
        default="marcello-barylli-growai",
        help="WandB entity/username",
    )
    
    args = parser.parse_args()
    
    # Validate that we have either run_id or checkpoint
    if not args.run_id and not args.checkpoint:
        parser.error("Either --run-id or --checkpoint must be provided")
    
    # Validate damage behavior flags
    if args.force_reversible and args.force_permanent:
        parser.error("Cannot specify both --force-reversible and --force-permanent")
    
    # Parse greedy_ordered_indices
    greedy_ordered_indices = None
    if args.greedy_ordered_indices:
        greedy_ordered_indices = [int(x.strip()) for x in args.greedy_ordered_indices.split(",")]
    else:
        # Try to load config to get defaults
        config = None
        if args.checkpoint:
            loaded = load_checkpoint(args.checkpoint)
            config = OmegaConf.create(loaded.get("config", {}))
        elif args.run_id:
            config, _, _ = load_config_from_wandb(
                run_id=args.run_id,
                filename="latest_checkpoint",
                select_by_best_metric=False,
                project=args.project,
                entity=args.entity,
            )
        
        if config:
            # Try to get from config
            pool_config = config.get("pool", {})
            greedy_ordered_indices = pool_config.get("greedy_ordered_indices", None)
            if greedy_ordered_indices is None:
                greedy_ordered_indices = config.get("greedy_ordered_indices", None)
        
        # Fall back to default if still None
        if greedy_ordered_indices is None:
            print(f"Warning: greedy_ordered_indices not found in config, using default")
            greedy_ordered_indices = DEFAULT_GREEDY_ORDERED_INDICES
    
    # Determine output path if not provided
    output_path = args.output
    if output_path is None:
        if args.run_id:
            base_name = f"trajectory_{args.run_id}"
        else:
            checkpoint_name = Path(args.checkpoint).stem
            base_name = f"trajectory_{checkpoint_name}"
        
        if args.trajectory_type == "damage_response":
            damage_behavior_str = "reversible" if args.force_reversible else ("permanent" if args.force_permanent else "auto")
            base_name += f"_{args.damage_injection_mode}_{args.damage_mode}_{damage_behavior_str}"
        
        output_path = f"results/figures/{base_name}.png"
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Set use_best_model
    use_best_model = args.use_best_model and not args.use_latest
    
    print(f"\n=== Trajectory Plot Configuration ===")
    print(f"Trajectory type: {args.trajectory_type}")
    if args.run_id:
        print(f"Run ID: {args.run_id}")
    else:
        print(f"Checkpoint: {args.checkpoint}")
    if args.trajectory_type == "damage_response":
        print(f"Damage injection mode: {args.damage_injection_mode}")
        print(f"Damage mode: {args.damage_mode}")
        if args.force_reversible:
            print(f"Damage behavior: FORCED REVERSIBLE")
        elif args.force_permanent:
            print(f"Damage behavior: FORCED PERMANENT")
        else:
            print(f"Damage behavior: from model (auto)")
        print(f"Note: Vocabulary will be generated from config to match training")
    print(f"Output: {output_path}")
    print("=" * 40)
    
    # Determine force_damage_behavior
    force_damage_behavior = None
    if args.force_reversible:
        force_damage_behavior = "reversible"
    elif args.force_permanent:
        force_damage_behavior = "permanent"
    
    # Call the plotting function
    fig = plot_trajectory_from_checkpoint(
        run_id=args.run_id,
        checkpoint_path=args.checkpoint,
        use_best_model=use_best_model,
        trajectory_type=args.trajectory_type,
        eval_on_train=args.eval_on_train,
        show_bp_trajectory=args.show_bp,
        show_ood_trajectory=not args.no_ood,
        damage_injection_mode=args.damage_injection_mode,
        damage_mode=args.damage_mode,
        damage_start_offset=args.damage_start_offset if args.damage_start_offset is not None else 0,
        max_damage_per_circuit=args.max_damage_per_circuit if args.max_damage_per_circuit is not None else 10,
        greedy_injection_recover_steps=args.greedy_injection_recover_steps if args.greedy_injection_recover_steps is not None else 10,
        greedy_ordered_indices=greedy_ordered_indices,
        greedy_window_size=args.greedy_window_size if args.greedy_window_size is not None else 1,
        n_message_steps=args.n_message_steps,
        eval_batch_size=args.eval_batch_size,
        periodic_eval_test_seed=args.periodic_eval_test_seed,
        output_path=output_path,
        title=args.title,
        dpi=args.dpi,
        project=args.project,
        entity=args.entity,
        force_damage_behavior=force_damage_behavior,
    )
    
    print(f"\n✓ Figure saved to: {output_path}")


if __name__ == "__main__":
    main()
