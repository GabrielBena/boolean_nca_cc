"""
Training utilities for Boolean NCA Circuit Compiler.

This module provides version-agnostic checkpoint saving and loading functions
that work across different JAX/Flax versions by extracting raw numpy arrays
instead of relying on framework-specific serialization.

"""

import logging
import os
import pickle
import sys
from types import ModuleType

import flax
import hydra
import jax
import matplotlib.pyplot as plt
import numpy as np
from flax import nnx
from omegaconf import OmegaConf

import wandb
from boolean_nca_cc.circuits.model import gen_circuit, generate_layer_sizes
from boolean_nca_cc.training.schedulers import get_learning_rate_schedule
from boolean_nca_cc.utils.flax_compatibility import setup_complete_flax_compatibility
from boolean_nca_cc.utils.graph_builder import build_graph

log = logging.getLogger(__name__)


def plot_training_curves(metrics, title, output_dir):
    """Generate and save training curve plots."""
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Plot loss curves
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(metrics["losses"], label="Soft Loss")
    ax.plot(metrics["hard_losses"], label="Hard Loss")
    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Loss")
    ax.set_yscale("log")
    ax.set_title(f"{title} - Loss Curves")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()

    loss_plot_path = os.path.join(output_dir, f"{title.lower().replace(' ', '_')}_loss.png")
    plt.savefig(loss_plot_path)
    if wandb.run is not None:
        wandb.log({f"{title} Loss": wandb.Image(fig)})
    plt.close(fig)

    # Plot accuracy curves
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(metrics["accuracies"], label="Soft Accuracy")
    ax.plot(metrics["hard_accuracies"], label="Hard Accuracy")
    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"{title} - Accuracy Curves")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()

    acc_plot_path = os.path.join(output_dir, f"{title.lower().replace(' ', '_')}_accuracy.png")
    plt.savefig(acc_plot_path)
    if wandb.run is not None:
        wandb.log({f"{title} Accuracy": wandb.Image(fig)})
    plt.close(fig)


def plot_inner_loop_metrics(step_metrics, title, output_dir):
    """Plot inner loop metrics over GNN message passing steps."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Plot loss curves
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(step_metrics["step"], step_metrics["soft_loss"], label="Soft Loss")
    ax.plot(step_metrics["step"], step_metrics["hard_loss"], label="Hard Loss")
    ax.set_xlabel("Message Passing Steps")
    ax.set_ylabel("Loss")
    ax.set_yscale("log")
    ax.set_title(f"{title} - Inner Loop Loss")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()

    loss_plot_path = os.path.join(output_dir, f"{title.lower().replace(' ', '_')}_inner_loss.png")
    plt.savefig(loss_plot_path)
    if wandb.run is not None:
        wandb.log({f"{title} Inner Loop Loss": wandb.Image(fig)})
    plt.close(fig)

    # Plot accuracy curves
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(step_metrics["step"], step_metrics["soft_accuracy"], label="Soft Accuracy")
    ax.plot(step_metrics["step"], step_metrics["hard_accuracy"], label="Hard Accuracy")
    ax.set_xlabel("Message Passing Steps")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"{title} - Inner Loop Accuracy")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()

    acc_plot_path = os.path.join(
        output_dir, f"{title.lower().replace(' ', '_')}_inner_accuracy.png"
    )
    plt.savefig(acc_plot_path)
    if wandb.run is not None:
        wandb.log({f"{title} Inner Loop Accuracy": wandb.Image(fig)})
    plt.close(fig)

    # Log the final metrics
    if wandb.run is not None:
        wandb.log(
            {
                "inner_loop/final_soft_loss": step_metrics["soft_loss"][-1],
                "inner_loop/final_hard_loss": step_metrics["hard_loss"][-1],
                "inner_loop/final_soft_accuracy": step_metrics["soft_accuracy"][-1],
                "inner_loop/final_hard_accuracy": step_metrics["hard_accuracy"][-1],
            }
        )


def compare_with_backprop(gnn_metrics, bp_metrics, title, output_dir):
    """Compare GNN and backpropagation performance."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Create a figure with 2x2 subplots
    fig, axs = plt.subplots(2, 2, figsize=(16, 12), constrained_layout=True)

    # Plot soft loss comparison
    axs[0, 0].plot(gnn_metrics["losses"], label="GNN")
    axs[0, 0].plot(bp_metrics["losses"], label="Backprop")
    axs[0, 0].set_title("Soft Loss Comparison")
    axs[0, 0].set_xlabel("Training Steps")
    axs[0, 0].set_ylabel("Loss")
    axs[0, 0].set_yscale("log")
    axs[0, 0].legend()
    axs[0, 0].grid(True)

    # Plot hard loss comparison
    axs[0, 1].plot(gnn_metrics["hard_losses"], label="GNN")
    axs[0, 1].plot(bp_metrics["hard_losses"], label="Backprop")
    axs[0, 1].set_title("Hard Loss Comparison")
    axs[0, 1].set_xlabel("Training Steps")
    axs[0, 1].set_ylabel("Loss")
    axs[0, 1].set_yscale("log")
    axs[0, 1].legend()
    axs[0, 1].grid(True)

    # Plot soft accuracy comparison
    axs[1, 0].plot(gnn_metrics["accuracies"], label="GNN")
    axs[1, 0].plot(bp_metrics["accuracies"], label="Backprop")
    axs[1, 0].set_title("Soft Accuracy Comparison")
    axs[1, 0].set_xlabel("Training Steps")
    axs[1, 0].set_ylabel("Accuracy")
    axs[1, 0].legend()
    axs[1, 0].grid(True)

    # Plot hard accuracy comparison
    axs[1, 1].plot(gnn_metrics["hard_accuracies"], label="GNN")
    axs[1, 1].plot(bp_metrics["hard_accuracies"], label="Backprop")
    axs[1, 1].set_title("Hard Accuracy Comparison")
    axs[1, 1].set_xlabel("Training Steps")
    axs[1, 1].set_ylabel("Accuracy")
    axs[1, 1].legend()
    axs[1, 1].grid(True)

    plt.suptitle(f"{title} - GNN vs Backprop Comparison")

    # Save the figure
    comp_plot_path = os.path.join(output_dir, f"{title.lower().replace(' ', '_')}_comparison.png")

    if wandb.run is not None:
        wandb.log({f"{title} Comparison": wandb.Image(fig)})
    plt.savefig(comp_plot_path)
    plt.close(fig)


def plot_lr_schedule(
    lr_scheduler: str,
    learning_rate: float,
    epochs: int,
    lr_scheduler_params: dict | None = None,
    title: str = "Learning Rate Schedule",
    output_dir: str | None = None,
    save_plot: bool = True,
    show_plot: bool = False,
    eval_every_n: int = 1,
):
    """
    Plot the learning rate schedule over training epochs.

    Args:
        lr_scheduler: Type of scheduler ("constant", "exponential", "cosine", "linear_warmup")
        learning_rate: Base learning rate
        epochs: Total number of training epochs
        lr_scheduler_params: Optional parameters for the scheduler
        title: Title for the plot
        output_dir: Directory to save the plot (if save_plot=True)
        save_plot: Whether to save the plot to disk
        show_plot: Whether to display the plot

    Returns:
        tuple: (fig, ax, lr_values) - matplotlib figure, axis, and learning rate values
    """
    # Get the learning rate schedule
    schedule_fn = get_learning_rate_schedule(
        lr_scheduler=lr_scheduler,
        learning_rate=learning_rate,
        epochs=epochs,
        lr_scheduler_params=lr_scheduler_params or {},
    )

    # Evaluate the schedule over all epochs
    epoch_steps = np.arange(epochs)
    lr_values = np.array([float(schedule_fn(step)) for step in range(0, epochs, eval_every_n)])
    # lr_values = np.maximum(np.array(lr_values), 1e-6)

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epoch_steps[::eval_every_n], lr_values, linewidth=2, color="blue")
    ax.set_xlabel("Training Epoch")
    ax.set_ylabel("Learning Rate")
    ax.set_title(f"{title} ({lr_scheduler})")
    ax.grid(True, alpha=0.3)

    # Add some annotations
    if lr_scheduler != "constant":
        ax.axhline(
            y=learning_rate,
            color="red",
            linestyle="--",
            alpha=0.7,
            label=f"Base LR: {learning_rate}",
        )
        ax.legend()

    # Format y-axis for better readability
    if max(lr_values) / lr_values[lr_values > 0].min() > 100:  # Use log scale for large ranges
        ax.set_yscale("log")

    plt.tight_layout()

    # Save plot if requested
    if save_plot and output_dir:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        plot_filename = f"lr_schedule_{lr_scheduler.lower()}.png"
        plot_path = os.path.join(output_dir, plot_filename)
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        print(f"Learning rate schedule plot saved to: {plot_path}")

        # Log to wandb if available
        if wandb.run is not None:
            wandb.log({f"LR Schedule ({lr_scheduler})": wandb.Image(fig)})

    # Show plot if requested
    if show_plot:
        plt.show()
    else:
        plt.close(fig) if not save_plot else None

    return fig, ax, lr_values


def cleanup_redundant_wandb_artifacts(
    run_id=None,
    filters=None,
    project="boolean-nca-cc",
    entity="m2snn",
    artifact_name_pattern=None,
    keep_tags=("best", "latest"),
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
    from collections import defaultdict

    import wandb

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
                base_name = artifact.name.split(":")[0] if ":" in artifact.name else artifact.name

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
                        print(f"Warning: Could not sort artifacts by creation time: {e}")
                    # Fallback: try to sort by version if available
                    try:
                        group_artifacts.sort(key=lambda x: x.version, reverse=True)
                    except Exception as e:
                        if verbose:
                            print(f"Warning: Could not sort artifacts by version: {e}")
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
                        matching_tags = [tag for tag in artifact_tags if tag in keep_tags]
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
                            print(f"  DELETE: {artifact.name} (no important tags, not recent)")

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
                            total_stats["deleted_artifacts"].append(f"{run.id}:{artifact.name}")
                        except Exception as e:
                            error_msg = f"Failed to delete {artifact.name}: {e}"
                            total_stats["errors"].append(error_msg)
                            if verbose:
                                print(f"  ERROR: {error_msg}")
                elif artifacts_to_delete:
                    if verbose:
                        print(f"DRY RUN: Would delete {len(artifacts_to_delete)} artifacts")
                        for artifact in artifacts_to_delete:
                            print(f"  Would delete: {artifact.name}")

        except Exception as e:
            error_msg = f"Error processing run {run.id}: {e}"
            total_stats["errors"].append(error_msg)
            if verbose:
                print(f"ERROR: {error_msg}")

    # Print summary
    if verbose:
        print("\n=== Cleanup Summary ===")
        print(f"Total artifacts found: {total_stats['total_artifacts']}")
        print(f"Artifacts kept: {total_stats['artifacts_kept']}")
        print(f"Artifacts marked for deletion: {total_stats['artifacts_to_delete']}")
        if not dry_run:
            print(f"Artifacts actually deleted: {len(total_stats['deleted_artifacts'])}")
        else:
            print("DRY RUN: No artifacts were actually deleted")

        if total_stats["errors"]:
            print(f"Errors encountered: {len(total_stats['errors'])}")
            for error in total_stats["errors"]:
                print(f"  - {error}")

    return total_stats


def keypath_to_string(keypath):
    """Convert keypath to string using JAX utilities."""
    return "_".join(str(key).strip("()").strip("[]").strip("'") for key in keypath)


def check_gradients(grads, verbose=True, return_zero_grad_paths=False):
    """
    Check gradients for zero values and optionally print detailed information.

    Args:
        grads: Gradient tree structure from nnx.value_and_grad
        verbose: If True, print information about zero gradients
        return_zero_grad_paths: If True, return list of parameter paths with zero gradients

    Returns:
        bool: True if any gradients are non-zero, False if all gradients are zero
        list (optional): List of parameter paths with zero gradients if return_zero_grad_paths=True
    """
    # Check if any gradients are non-zero
    has_grads = any(jax.tree.leaves(jax.tree.map(lambda x: x.any(), grads)))

    if not has_grads:
        if verbose:
            print("WARNING: All gradients are zero!")
        if return_zero_grad_paths:
            return False, []
        return False

    # Get detailed information about zero gradients
    leaves_with_paths, _ = jax.tree_util.tree_flatten_with_path(grads)
    zero_grad_paths = []

    for _i, (path, leaf) in enumerate(leaves_with_paths):
        path_str = keypath_to_string(path)
        if not leaf.any():
            zero_grad_paths.append(path_str)
            if verbose:
                print(f"{path_str}, shape: {leaf.shape} has no grads")

    if verbose and zero_grad_paths:
        print(
            f"Found {len(zero_grad_paths)} parameters with zero gradients out of {len(leaves_with_paths)} total parameters"
        )
    elif verbose:
        print(f"All {len(leaves_with_paths)} parameters have non-zero gradients")

    if return_zero_grad_paths:
        return has_grads, zero_grad_paths
    return has_grads


def gradient_check_step(model, optimizer, loss_fn, *loss_args, verbose=True, **loss_kwargs):
    """
    Perform a gradient check step: compute gradients, update optimizer, check for zero gradients.

    Args:
        model: The model to check gradients for
        optimizer: The optimizer to update
        loss_fn: Loss function that takes (model, *loss_args, **loss_kwargs) and returns (loss, aux)
        *loss_args: Positional arguments to pass to loss_fn
        verbose: If True, print gradient information
        **loss_kwargs: Keyword arguments to pass to loss_fn

    Returns:
        tuple: (loss, aux, has_grads, zero_grad_paths)
            - loss: The computed loss value
            - aux: Auxiliary outputs from loss function
            - has_grads: True if any gradients are non-zero
            - zero_grad_paths: List of parameter paths with zero gradients
    """
    # Compute gradients
    (loss, aux), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model, *loss_args, **loss_kwargs)

    # Update optimizer
    optimizer.update(grads)

    # Check gradients
    has_grads, zero_grad_paths = check_gradients(
        grads, verbose=verbose, return_zero_grad_paths=True
    )

    # Assert that we have some gradients
    if not has_grads:
        raise RuntimeError("No gradients found - all parameters have zero gradients!")

    return loss, aux, has_grads, zero_grad_paths


def check_checkpoint_format(checkpoint_path):
    """Check the format and contents of a checkpoint file.

    Args:
        checkpoint_path: Path to the checkpoint file

    Returns:
        Dictionary with checkpoint information
    """
    try:
        with open(checkpoint_path, "rb") as f:
            checkpoint = pickle.load(f)
    except Exception as e:
        return {"error": f"Failed to load checkpoint: {e}"}

    info = {
        "file_path": checkpoint_path,
        "file_size_mb": os.path.getsize(checkpoint_path) / (1024 * 1024),
    }

    # Check if this is version-agnostic format
    if checkpoint.get("checkpoint_version") == "v2_agnostic":
        info["format"] = "version-agnostic (v2)"
        info["jax_version"] = checkpoint.get("jax_version", "unknown")
        info["flax_version"] = checkpoint.get("flax_version", "unknown")
        info["save_timestamp"] = checkpoint.get("save_timestamp", "unknown")
        info["has_model_params"] = checkpoint.get("model_params") is not None
        info["has_optimizer_params"] = checkpoint.get("optimizer_params") is not None
    else:
        info["format"] = "legacy (JAX/Flax serialized)"
        info["has_model"] = "model" in checkpoint
        info["has_optimizer"] = "optimizer" in checkpoint

        # Try to detect potential compatibility issues
        if "model" in checkpoint:
            try:
                model_state = checkpoint["model"]
                if hasattr(model_state, "__dict__"):
                    for attr_name in dir(model_state):
                        if "var_metadata" in attr_name.lower():
                            info["potential_compatibility_issues"] = (
                                "VariableState metadata detected"
                            )
                            break
            except Exception:
                pass

    # Common fields
    info["has_config"] = "config" in checkpoint
    info["has_metrics"] = "metrics" in checkpoint
    info["step"] = checkpoint.get("step", "unknown")

    return info
