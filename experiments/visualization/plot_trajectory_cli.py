#!/usr/bin/env python3
"""
CLI wrapper for plot_trajectory_from_checkpoint.

Provides command-line interface for generating trajectory plots from checkpoints.
Supports:
- Reversible/permanent damage
- Single/multi damage injection
- Vocabulary loading vs on-the-fly pattern generation
"""

import argparse
import os
import sys
import pickle
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import jax
import jax.numpy as jp
from omegaconf import OmegaConf

from experiments.visualization.plot_trajectory import plot_trajectory_from_checkpoint
from boolean_nca_cc.training.pool.structural_perturbation import (
    DEFAULT_GREEDY_ORDERED_INDICES,
    create_knockout_vocabulary,
)
from boolean_nca_cc.training.checkpointing import (
    load_config_from_wandb,
    load_checkpoint,
)


def load_vocabulary_from_file(vocab_path: str) -> jp.ndarray:
    """Load knockout vocabulary from a pickle file."""
    with open(vocab_path, 'rb') as f:
        data = pickle.load(f)
        if isinstance(data, dict):
            # Try common keys
            if 'vocabulary' in data:
                return jp.array(data['vocabulary'])
            elif 'knockout_vocabulary' in data:
                return jp.array(data['knockout_vocabulary'])
            elif 'patterns' in data:
                return jp.array(data['patterns'])
            else:
                raise ValueError(f"Vocabulary file must contain 'vocabulary', 'knockout_vocabulary', or 'patterns' key. Found keys: {list(data.keys())}")
        elif isinstance(data, (list, tuple)):
            return jp.array(data)
        else:
            return jp.array(data)


def generate_vocabulary_on_the_fly(
    layer_sizes,
    vocabulary_size: int,
    damage_prob: float,
    damage_mode: str,
    greedy_ordered_indices: list,
    seed: int = 42,
) -> jp.ndarray:
    """Generate knockout vocabulary on the fly."""
    rng = jax.random.PRNGKey(seed)
    return create_knockout_vocabulary(
        rng=rng,
        vocabulary_size=vocabulary_size,
        layer_sizes=layer_sizes,
        damage_prob=damage_prob,
        damage_mode=damage_mode,
        ordered_indices=greedy_ordered_indices,
    )


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
        default="greedy",
        help="Damage pattern type",
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
    
    # Vocabulary/Pattern generation
    vocab_group = parser.add_mutually_exclusive_group()
    vocab_group.add_argument(
        "--vocab-file",
        type=str,
        default=None,
        help="Path to pickle file containing knockout vocabulary (mutually exclusive with --generate-vocab)",
    )
    vocab_group.add_argument(
        "--generate-vocab",
        action="store_true",
        help="Generate vocabulary on the fly (mutually exclusive with --vocab-file)",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=None,
        help="Vocabulary size for on-the-fly generation (default: from config or 10)",
    )
    parser.add_argument(
        "--damage-prob",
        type=float,
        default=None,
        help="Damage probability (number of gates to knock out) for vocabulary generation (default: from config)",
    )
    parser.add_argument(
        "--vocab-seed",
        type=int,
        default=42,
        help="Random seed for vocabulary generation",
    )
    
    # Static patterns (for single injection mode with static damage modes)
    parser.add_argument(
        "--static-patterns-file",
        type=str,
        default=None,
        help="Path to pickle file containing static knockout patterns (for single injection with static damage modes)",
    )
    parser.add_argument(
        "--num-static-patterns",
        type=int,
        default=None,
        help="Number of static patterns to generate on the fly (for single injection with static damage modes)",
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
        default=42,
        help="Seed for generating OOD patterns",
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
    
    # Load config first to get defaults
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
    else:
        config = None
    
    # Parse greedy_ordered_indices
    greedy_ordered_indices = None
    if args.greedy_ordered_indices:
        greedy_ordered_indices = [int(x.strip()) for x in args.greedy_ordered_indices.split(",")]
    elif config:
        # Try to get from config
        pool_config = config.get("pool", {})
        greedy_ordered_indices = pool_config.get("greedy_ordered_indices", None)
        if greedy_ordered_indices is None:
            greedy_ordered_indices = config.get("greedy_ordered_indices", None)
    
    # Fall back to default if still None
    if greedy_ordered_indices is None:
        print(f"Warning: greedy_ordered_indices not found in config, using default")
        greedy_ordered_indices = DEFAULT_GREEDY_ORDERED_INDICES
    
    # Get layer sizes for vocabulary generation
    layer_sizes = None
    if config:
        if config.circuit.layer_sizes is None:
            from boolean_nca_cc.circuits.model import generate_layer_sizes
            layer_sizes = generate_layer_sizes(
                input_n=config.circuit.input_bits,
                output_n=config.circuit.output_bits,
                arity=config.circuit.arity,
                layer_n=config.circuit.num_layers,
            )
        else:
            layer_sizes = config.circuit.layer_sizes
    
    # Handle vocabulary/pattern loading
    knockout_vocabulary = None
    knockout_patterns = None
    knockout_config = None
    
    if args.trajectory_type == "damage_response":
        # For multi-damage mode with greedy/greedy_vocabulary, vocabulary is used
        if args.damage_injection_mode == "multi" and args.damage_mode in ["greedy", "greedy_vocabulary"]:
            if args.vocab_file:
                print(f"Loading vocabulary from file: {args.vocab_file}")
                knockout_vocabulary = load_vocabulary_from_file(args.vocab_file)
            elif args.generate_vocab or args.vocab_file is None:
                # Generate on the fly
                if layer_sizes is None:
                    parser.error("Cannot generate vocabulary: layer_sizes not available. Provide --vocab-file or ensure config is loaded.")
                
                vocab_size = args.vocab_size
                if vocab_size is None and config:
                    vocab_size = config.get("pool", {}).get("damage_knockout_diversity", 
                                                             config.get("pool", {}).get("vocabulary_size", 10))
                if vocab_size is None:
                    vocab_size = 10
                
                damage_prob = args.damage_prob
                if damage_prob is None and config:
                    damage_prob = config.get("pool", {}).get("damage_prob", 10)
                if damage_prob is None:
                    damage_prob = 10
                
                print(f"Generating vocabulary on the fly: size={vocab_size}, damage_prob={damage_prob}, mode={args.damage_mode}")
                knockout_vocabulary = generate_vocabulary_on_the_fly(
                    layer_sizes=layer_sizes,
                    vocabulary_size=vocab_size,
                    damage_prob=damage_prob,
                    damage_mode=args.damage_mode,
                    greedy_ordered_indices=greedy_ordered_indices,
                    seed=args.vocab_seed,
                )
        
        # For single injection mode or static damage modes, use knockout_patterns
        elif args.damage_injection_mode == "single" or args.damage_mode in ["shotgun", "strip"]:
            if args.static_patterns_file:
                print(f"Loading static patterns from file: {args.static_patterns_file}")
                knockout_patterns = load_vocabulary_from_file(args.static_patterns_file)
            elif args.num_static_patterns or args.static_patterns_file is None:
                # Generate on the fly
                if layer_sizes is None:
                    parser.error("Cannot generate patterns: layer_sizes not available. Provide --static-patterns-file or ensure config is loaded.")
                
                num_patterns = args.num_static_patterns
                if num_patterns is None and config:
                    num_patterns = config.get("pool", {}).get("vocabulary_size", 10)
                if num_patterns is None:
                    num_patterns = 10
                
                damage_prob = args.damage_prob
                if damage_prob is None and config:
                    damage_prob = config.get("pool", {}).get("damage_prob", 10)
                if damage_prob is None:
                    damage_prob = 10
                
                print(f"Generating static patterns on the fly: num={num_patterns}, damage_prob={damage_prob}, mode={args.damage_mode}")
                from boolean_nca_cc.training.pool.structural_perturbation import create_reproducible_knockout_pattern
                from functools import partial
                
                pattern_creator_fn = partial(
                    create_reproducible_knockout_pattern,
                    layer_sizes=layer_sizes,
                    damage_prob=damage_prob,
                )
                pattern_keys = jax.random.split(jax.random.PRNGKey(args.vocab_seed), num_patterns)
                knockout_patterns = jax.vmap(pattern_creator_fn)(pattern_keys)
        
        # Set knockout_config for OOD pattern generation
        if not args.no_ood:
            damage_prob = args.damage_prob
            if damage_prob is None and config:
                damage_prob = config.get("pool", {}).get("damage_prob", 10)
            if damage_prob is None:
                damage_prob = 10
            
            knockout_config = {"damage_prob": damage_prob}
    
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
        if knockout_vocabulary is not None:
            print(f"Vocabulary: {len(knockout_vocabulary)} patterns")
        if knockout_patterns is not None:
            print(f"Static patterns: {len(knockout_patterns)} patterns")
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
        knockout_vocabulary=knockout_vocabulary,
        knockout_patterns=knockout_patterns,
        knockout_config=knockout_config,
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
