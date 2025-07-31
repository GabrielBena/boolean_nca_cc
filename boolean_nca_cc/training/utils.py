import flax
import pickle
from flax import nnx
import matplotlib.pyplot as plt
import os
import wandb
from omegaconf import OmegaConf
import logging
import hydra
import jax
import glob
import numpy as np
from boolean_nca_cc.circuits.model import gen_circuit, run_circuit
from boolean_nca_cc.circuits.model import generate_layer_sizes
from boolean_nca_cc.utils.graph_builder import build_graph
from boolean_nca_cc.training.schedulers import get_learning_rate_schedule
import jax.numpy as jp
from boolean_nca_cc.training.evaluation import evaluate_circuits_in_chunks, evaluate_model_stepwise_batched

log = logging.getLogger(__name__)

def compute_n_nodes_from_config(config):
        """Compute n_nodes for CircuitSelfAttention models by building a dummy graph."""
        # Generate circuit layer sizes
        input_n, output_n = config.circuit.input_bits, config.circuit.output_bits
        arity = config.circuit.arity

        if config.circuit.layer_sizes is None:
            layer_sizes = generate_layer_sizes(
                input_n, output_n, arity, layer_n=config.circuit.num_layers
            )
        else:
            layer_sizes = config.circuit.layer_sizes

        # Generate dummy circuit
        test_key = jax.random.PRNGKey(config.get("test_seed", 42))
        wires, logits = gen_circuit(test_key, layer_sizes, arity=arity)

        # Generate dummy graph
        graph = build_graph(
            wires=wires,
            logits=logits,
            input_n=input_n,
            arity=arity,
            circuit_hidden_dim=config.model.circuit_hidden_dim,
        )

        n_nodes = int(graph.n_node[0])
        print(f"Computed n_nodes for CircuitSelfAttention: {n_nodes}")
        return n_nodes

def cleanup_redundant_wandb_artifacts(
    run_id=None,
    filters=None,
    project="boolean-nca-cc",
    entity="m2snn",
    artifact_name_pattern=None,
    keep_tags=["best", "latest"],
    keep_recent_count=3,
    dry_run=True,
    verbose=True,
):
    """Clean up redundant wandb artifacts, keeping only those with important tags.

    This function identifies and optionally deletes artifact versions that don't have
    important tags like "best" or "latest", helping to reduce storage usage in wandb.

    Args:
        run_id: Optional specific run ID to clean up artifacts from
        filters: Optional dictionary of filters to find runs
        project: WandB project name
        entity: WandB entity/username
        artifact_name_pattern: Pattern to match artifact names (e.g. "best_model", "latest_checkpoint")
                              If None, will clean all artifacts
        keep_tags: List of tags that should be preserved (artifacts with these tags won't be deleted)
        keep_recent_count: Number of most recent artifacts to keep regardless of tags
        dry_run: If True, only show what would be deleted without actually deleting
        verbose: If True, print detailed information about the cleanup process

    Returns:
        Dictionary with cleanup statistics including:
        - total_artifacts: Total number of artifacts found
        - artifacts_to_delete: Number of artifacts marked for deletion
        - artifacts_kept: Number of artifacts kept
        - deleted_artifacts: List of deleted artifact names (empty if dry_run=True)
    """
    import wandb
    from collections import defaultdict
    from datetime import datetime

    # Initialize WandB API
    api = wandb.Api()

    # Find runs to clean up
    runs_to_process = []

    if run_id:
        if verbose:
            print(f"Looking for run with ID: {run_id}")
        run = api.run(f"{entity}/{project}/{run_id}")
        runs_to_process = [run]
        if verbose:
            print(f"Found run: {run.name}")
    else:
        if not filters:
            filters = {}
        if verbose:
            print(f"Looking for runs with filters: {filters}")
        runs = api.runs(f"{entity}/{project}", filters=filters)

        if not runs:
            raise ValueError(f"No runs found matching filters: {filters}")

        runs_to_process = list(runs)
        if verbose:
            print(f"Found {len(runs_to_process)} matching runs.")

    # Statistics tracking
    total_stats = {
        "total_artifacts": 0,
        "artifacts_to_delete": 0,
        "artifacts_kept": 0,
        "deleted_artifacts": [],
        "errors": [],
    }

    # Process each run
    for run in runs_to_process:
        if verbose:
            print(f"\n=== Processing run: {run.name} (ID: {run.id}) ===")

        try:
            # Get all artifacts for this run
            artifacts = run.logged_artifacts()

            if not artifacts:
                if verbose:
                    print("No artifacts found for this run")
                continue

            # Group artifacts by name (without version)
            artifact_groups = defaultdict(list)

            for artifact in artifacts:
                # Extract base name without version
                base_name = (
                    artifact.name.split(":")[0]
                    if ":" in artifact.name
                    else artifact.name
                )

                # Apply artifact name pattern filter if specified
                if artifact_name_pattern and artifact_name_pattern not in base_name:
                    continue

                artifact_groups[base_name].append(artifact)

            if verbose:
                print(f"Found {len(artifact_groups)} artifact groups")
                for group_name, group_artifacts in artifact_groups.items():
                    print(f"  - {group_name}: {len(group_artifacts)} versions")

            # Process each artifact group
            for group_name, group_artifacts in artifact_groups.items():
                if verbose:
                    print(f"\n--- Processing artifact group: {group_name} ---")

                total_stats["total_artifacts"] += len(group_artifacts)

                # Sort artifacts by creation time (newest first)
                try:
                    group_artifacts.sort(key=lambda x: x.created_at, reverse=True)
                except Exception as e:
                    if verbose:
                        print(
                            f"Warning: Could not sort artifacts by creation time: {e}"
                        )
                    # Fallback: try to sort by version if available
                    try:
                        group_artifacts.sort(key=lambda x: x.version, reverse=True)
                    except:
                        pass  # Keep original order if sorting fails

                artifacts_to_keep = []
                artifacts_to_delete = []

                # Categorize artifacts
                for i, artifact in enumerate(group_artifacts):
                    should_keep = False
                    keep_reason = []

                    # Check if artifact has important tags
                    artifact_tags = getattr(artifact, "tags", []) or []
                    if any(tag in artifact_tags for tag in keep_tags):
                        should_keep = True
                        matching_tags = [
                            tag for tag in artifact_tags if tag in keep_tags
                        ]
                        keep_reason.append(f"has important tags: {matching_tags}")

                    # Keep recent artifacts regardless of tags
                    if i < keep_recent_count:
                        should_keep = True
                        keep_reason.append(f"among {keep_recent_count} most recent")

                    if should_keep:
                        artifacts_to_keep.append(artifact)
                        if verbose:
                            print(f"  KEEP: {artifact.name} ({', '.join(keep_reason)})")
                    else:
                        artifacts_to_delete.append(artifact)
                        if verbose:
                            print(
                                f"  DELETE: {artifact.name} (no important tags, not recent)"
                            )

                # Update statistics
                total_stats["artifacts_kept"] += len(artifacts_to_keep)
                total_stats["artifacts_to_delete"] += len(artifacts_to_delete)

                # Perform deletion if not dry run
                if not dry_run and artifacts_to_delete:
                    if verbose:
                        print(f"Deleting {len(artifacts_to_delete)} artifacts...")

                    for artifact in artifacts_to_delete:
                        try:
                            if verbose:
                                print(f"  Deleting: {artifact.name}")
                            artifact.delete()
                            total_stats["deleted_artifacts"].append(
                                f"{run.id}:{artifact.name}"
                            )
                        except Exception as e:
                            error_msg = f"Failed to delete {artifact.name}: {e}"
                            total_stats["errors"].append(error_msg)
                            if verbose:
                                print(f"  ERROR: {error_msg}")
                elif artifacts_to_delete:
                    if verbose:
                        print(
                            f"DRY RUN: Would delete {len(artifacts_to_delete)} artifacts"
                        )
                        for artifact in artifacts_to_delete:
                            print(f"  Would delete: {artifact.name}")

        except Exception as e:
            error_msg = f"Error processing run {run.id}: {e}"
            total_stats["errors"].append(error_msg)
            if verbose:
                print(f"ERROR: {error_msg}")

    # Print summary
    if verbose:
        print(f"\n=== Cleanup Summary ===")
        print(f"Total artifacts found: {total_stats['total_artifacts']}")
        print(f"Artifacts kept: {total_stats['artifacts_kept']}")
        print(f"Artifacts marked for deletion: {total_stats['artifacts_to_delete']}")
        if not dry_run:
            print(
                f"Artifacts actually deleted: {len(total_stats['deleted_artifacts'])}"
            )
        else:
            print("DRY RUN: No artifacts were actually deleted")

        if total_stats["errors"]:
            print(f"Errors encountered: {len(total_stats['errors'])}")
            for error in total_stats["errors"]:
                print(f"  - {error}")

    return total_stats


def compare_bp_sa_performance(bp_results, sa_pattern_results, vocabulary_patterns):
    """
    Compare Backpropagation vs Self-Attention performance on knockout patterns.
    
    Args:
        bp_results: Results from _run_backpropagation_training_with_knockouts()
        sa_pattern_results: Final hard accuracies from SA evaluation (array indexed by vocabulary position)
        vocabulary_patterns: The knockout vocabulary patterns used for both BP and SA
        
    Returns:
        Dictionary with comparison results and plotting data
    """
    # Extract BP results by pattern index
    bp_pattern_accuracies = {p["pattern_idx"]: p["final_hard_accuracy"] 
                           for p in bp_results["patterns_performance"]}
  
    # SA results already indexed by vocabulary position
    sa_pattern_accuracies = sa_pattern_results
  
    # Prepare data for custom visualization
    pattern_indices = []
    bp_accuracies = []
    sa_accuracies = []
  
    for pattern_idx in range(len(vocabulary_patterns)):
        bp_acc = bp_pattern_accuracies[pattern_idx]
        sa_acc = sa_pattern_accuracies[pattern_idx]
      
        pattern_indices.append(pattern_idx)
        bp_accuracies.append(bp_acc)
        sa_accuracies.append(sa_acc)
  
    # Calculate aggregate metrics
    comparisons = []
    for i, pattern_idx in enumerate(pattern_indices):
        bp_acc = bp_accuracies[i]
        sa_acc = sa_accuracies[i]
      
        comparisons.append({
            "pattern_idx": pattern_idx,
            "bp_hard_accuracy": bp_acc,
            "sa_hard_accuracy": sa_acc,
            "accuracy_difference": sa_acc - bp_acc,
            "relative_improvement": (sa_acc - bp_acc) / bp_acc if bp_acc > 0 else 0
        })
  
    return {
        # Raw data for custom plotting
        "plot_data": {
            "pattern_indices": pattern_indices,
            "bp_accuracies": bp_accuracies,
            "sa_accuracies": sa_accuracies
        },
        # Aggregate metrics
        "comparison/patterns": comparisons,
        "comparison/mean_bp_accuracy": np.mean(bp_accuracies),
        "comparison/mean_sa_accuracy": np.mean(sa_accuracies),
        "comparison/mean_improvement": np.mean([c["accuracy_difference"] for c in comparisons]),
        "comparison/patterns_better_than_bp": sum(1 for c in comparisons if c["accuracy_difference"] > 0),
        "comparison/total_patterns": len(comparisons)
    }


def plot_bp_vs_sa_comparison(comparison_results):
    """
    Create custom plot: x=pattern_index, y=hard_accuracy, red=BP, green=SA
    
    Args:
        comparison_results: Results from compare_bp_sa_performance()
        
    Returns:
        matplotlib figure object
    """
    plot_data = comparison_results["plot_data"]
  
    plt.figure(figsize=(12, 6))
    plt.scatter(plot_data["pattern_indices"], plot_data["bp_accuracies"], 
               c='red', s=50, alpha=0.7, label='Backpropagation')
    plt.scatter(plot_data["pattern_indices"], plot_data["sa_accuracies"], 
               c='green', s=50, alpha=0.7, label='Self-Attention')
  
    plt.xlabel('Pattern Index')
    plt.ylabel('Hard Accuracy')
    plt.title('BP vs SA Performance by Pattern')
    plt.legend()
    plt.grid(True, alpha=0.3)
  
    # Add mean lines
    mean_bp = comparison_results["comparison/mean_bp_accuracy"]
    mean_sa = comparison_results["comparison/mean_sa_accuracy"]
    plt.axhline(y=mean_bp, color='red', linestyle='--', alpha=0.5, label=f'BP Mean: {mean_bp:.3f}')
    plt.axhline(y=mean_sa, color='green', linestyle='--', alpha=0.5, label=f'SA Mean: {mean_sa:.3f}')
  
    return plt.gcf()


def run_final_bp_sa_comparison(
    model,
    bp_results,
    knockout_vocabulary,
    cfg,
    x_data,
    y_data,
    layer_sizes,
    input_n,
    arity,
    circuit_hidden_dim,
    n_message_steps,
    loss_type,
):
    """
    Run final BP vs SA comparison evaluation on a fresh circuit with vocabulary patterns.
    
    Args:
        model: Trained SA model
        bp_results: Results from _run_backpropagation_training_with_knockouts()
        knockout_vocabulary: Vocabulary of knockout patterns
        cfg: Configuration object
        x_data: Input data
        y_data: Target data
        layer_sizes: Circuit layer sizes
        input_n: Number of inputs
        arity: Circuit arity
        circuit_hidden_dim: Hidden dimension for circuit
        n_message_steps: Number of message passing steps for evaluation
        loss_type: Type of loss function
        
    Returns:
        Dictionary with comparison results
    """
    log.info("Running final BP vs SA comparison evaluation")
    
    # Generate fresh circuit for final evaluation (like train_beefy.py)
    final_eval_key = jax.random.PRNGKey(cfg.eval.get("final_eval_seed", 999))
    fresh_wires, fresh_logits = gen_circuit(
        final_eval_key, layer_sizes, arity=arity
    )
    
    log.info(f"Generated fresh circuit for final evaluation")
    log.info(f"Evaluating SA model on {len(knockout_vocabulary)} vocabulary patterns")
    
    # Use entire vocabulary instead of sampling
    eval_batch_size = len(knockout_vocabulary)
    
    # Replicate fresh circuit for batch
    batch_wires = jax.tree.map(
        lambda x: jp.repeat(x[None, ...], eval_batch_size, axis=0), fresh_wires
    )
    batch_logits = jax.tree.map(
        lambda x: jp.repeat(x[None, ...], eval_batch_size, axis=0), fresh_logits
    )
    
    # Evaluate SA model on all vocabulary patterns
    step_metrics = evaluate_circuits_in_chunks(
        eval_fn=evaluate_model_stepwise_batched,
        wires=batch_wires,
        logits=batch_logits,
        knockout_patterns=knockout_vocabulary,  # Use entire vocabulary
        target_chunk_size=eval_batch_size,
        model=model,
        x_data=x_data,
        y_data=y_data,
        input_n=input_n,
        arity=arity,
        circuit_hidden_dim=circuit_hidden_dim,
        n_message_steps=n_message_steps,
        loss_type=loss_type,
        layer_sizes=layer_sizes,
        return_per_pattern=True,
    )
    
    # Extract final hard accuracies for each pattern
    sa_pattern_results = step_metrics["per_pattern"]["pattern_hard_accuracies"][-1]
    
    log.info(f"SA evaluation complete. Mean hard accuracy: {np.mean(sa_pattern_results):.4f}")
    
    # Run comparison
    comparison_results = compare_bp_sa_performance(
        bp_results=bp_results,
        sa_pattern_results=sa_pattern_results,
        vocabulary_patterns=knockout_vocabulary,
    )
    
    log.info(f"Comparison complete:")
    log.info(f"  BP mean accuracy: {comparison_results['comparison/mean_bp_accuracy']:.4f}")
    log.info(f"  SA mean accuracy: {comparison_results['comparison/mean_sa_accuracy']:.4f}")
    log.info(f"  SA better than BP on {comparison_results['comparison/patterns_better_than_bp']}/{comparison_results['comparison/total_patterns']} patterns")
    
    return comparison_results