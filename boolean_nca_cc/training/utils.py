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