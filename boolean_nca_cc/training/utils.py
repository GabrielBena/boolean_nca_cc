import flax
import pickle
from flax import nnx
import matplotlib.pyplot as plt
import os
import wandb
from omegaconf import OmegaConf
import logging
import hydra.utils

log = logging.getLogger(__name__)


def save_checkpoint(model, optimizer, metrics, cfg, step, output_dir, filename=None):
    """Save a checkpoint of the model and optimizer.

    Args:
        model: The model to save
        optimizer: The optimizer to save
        metrics: Dictionary of training metrics
        cfg: Configuration object or dictionary
        step: Current training step
        output_dir: Directory to save the checkpoint
        filename: Optional custom filename for the checkpoint

    Returns:
        Path to the saved checkpoint
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    checkpoint = {
        "model": nnx.state(model),
        "optimizer": nnx.state(optimizer),
        "metrics": metrics,
        "config": OmegaConf.to_container(cfg, resolve=True)
        if hasattr(cfg, "to_container")
        else cfg,
        "step": step,
    }

    if filename is None:
        checkpoint_path = os.path.join(output_dir, f"checkpoint_{step}.pkl")
    else:
        checkpoint_path = os.path.join(output_dir, filename)

    with open(checkpoint_path, "wb") as f:
        pickle.dump(checkpoint, f)

    log.info(f"Saved checkpoint to {checkpoint_path}")
    if hasattr(cfg, "wandb") and cfg.wandb.enabled:
        try:
            import wandb

            if wandb.run:
                wandb.save(checkpoint_path)
        except ImportError:
            log.warning("wandb not installed, skipping checkpoint logging to W&B")
        except Exception as e:
            log.warning(f"Error logging checkpoint to W&B: {e}")

    return checkpoint_path


def load_checkpoint(checkpoint_path):
    """Load a checkpoint of the model and optimizer."""
    with open(checkpoint_path, "rb") as f:
        checkpoint_bytes = f.read()
        checkpoint = flax.serialization.from_bytes(None, checkpoint_bytes)

    return checkpoint


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

    loss_plot_path = os.path.join(
        output_dir, f"{title.lower().replace(' ', '_')}_loss.png"
    )
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

    acc_plot_path = os.path.join(
        output_dir, f"{title.lower().replace(' ', '_')}_accuracy.png"
    )
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

    loss_plot_path = os.path.join(
        output_dir, f"{title.lower().replace(' ', '_')}_inner_loss.png"
    )
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
                f"inner_loop/final_soft_loss": step_metrics["soft_loss"][-1],
                f"inner_loop/final_hard_loss": step_metrics["hard_loss"][-1],
                f"inner_loop/final_soft_accuracy": step_metrics["soft_accuracy"][-1],
                f"inner_loop/final_hard_accuracy": step_metrics["hard_accuracy"][-1],
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
    comp_plot_path = os.path.join(
        output_dir, f"{title.lower().replace(' ', '_')}_comparison.png"
    )

    if wandb.run is not None:
        wandb.log({f"{title} Comparison": wandb.Image(fig)})
    plt.savefig(comp_plot_path)
    plt.close(fig)


def load_best_model_from_wandb(
    project_name: str = "m2snn/boolean-nca-cc",
    run_id: str = None,
    filters: dict = None,
    download_root: str = "saves",
    artifact_name_keyword: str = "best_model_hard_accuracy",
    target_filename: str = "best_model.pkl",
):
    """
    Loads a model checkpoint from a W&B run and instantiates the model.

    Identifies a run by run_id or filters. If filters match multiple runs,
    it selects the one that appears last in the list returned by W&B API
    (typically the oldest).

    Within the run, it finds artifacts with `artifact_name_keyword`.
    Among these, it selects the one that appears last (typically oldest version).

    The artifact (.pkl file) is downloaded. Its content (a dictionary) is loaded.
    The model state is extracted from this dictionary.
    The model architecture is instantiated using Hydra configuration from the W&B run.
    The loaded state is applied to the instantiated model.

    Args:
        project_name (str): W&B project name.
        run_id (str, optional): Specific W&B run ID. Overrides filters.
        filters (dict, optional): Filters to select W&B runs.
        download_root (str): Root directory for artifact download.
        artifact_name_keyword (str): Keyword for relevant model artifacts.
        target_filename (str): Filename for the downloaded .pkl model file.

    Returns:
        tuple[Optional[Any], Optional[dict]]:
            - The instantiated and state-updated model object (e.g., an nnx.Module).
              Returns None if model instantiation or state update fails.
            - The full dictionary loaded from the checkpoint artifact.
              Returns None if run/artifact loading fails before model processing.
    """
    api = wandb.Api()
    run_to_process = None

    log.info(
        f"Attempting to load best model. Project: {project_name}, Run ID: {run_id}, Filters: {filters}"
    )

    if run_id:
        try:
            full_run_path = run_id if "/" in run_id else f"{project_name}/{run_id}"
            run_to_process = api.run(full_run_path)
            log.info(f"Fetched run by ID: {run_to_process.id} (Path: {full_run_path})")
        except wandb.errors.CommError as e:
            log.error(
                f"Could not fetch run by ID '{run_id}' (tried path '{full_run_path}'): {e}"
            )
            return None, None
    elif filters:
        try:
            runs = api.runs(project_name, filters=filters)
            if not runs:
                log.warning(
                    f"No runs found for project '{project_name}' with filters: {filters}"
                )
                return None, None
            log.info(
                f"Found {len(runs)} runs matching the provided filters for project '{project_name}'."
            )
            run_to_process = runs[-1]
            log.info(
                f"Selected run '{run_to_process.id}' (the last from {len(runs)} found runs) using filters: {filters}. Note: This is typically the oldest matching run."
            )
        except Exception as e:
            log.error(
                f"Error fetching/selecting runs with filters for project '{project_name}': {e}"
            )
            return None, None
    else:
        log.error(
            "Either run_id or filters must be provided to load_best_model_from_wandb."
        )
        return None, None

    if not run_to_process:  # Should be caught earlier, but as a safeguard
        log.error("Failed to identify a run to process.")
        return None, None

    log.info(f"Processing run: {run_to_process.name} (ID: {run_to_process.id})")

    all_artifacts = run_to_process.logged_artifacts()
    if not all_artifacts:
        log.warning(f"No artifacts found for run '{run_to_process.id}'.")
        return None, None

    candidate_artifacts = [
        art for art in all_artifacts if artifact_name_keyword in art.name
    ]

    if not candidate_artifacts:
        log.warning(
            f"No artifacts matching keyword '{artifact_name_keyword}' found in run '{run_to_process.id}'. Artifacts names checked: {[art.name for art in all_artifacts]}."
        )
        return None, None

    log.info(
        f"Found {len(candidate_artifacts)} artifacts containing keyword '{artifact_name_keyword}' in their names for run '{run_to_process.id}'."
    )
    selected_artifact = candidate_artifacts[-1]
    log.info(
        f"Selected artifact: '{selected_artifact.name}' (version: {selected_artifact.version}, type: {selected_artifact.type}). Note: This is typically the oldest matching artifact."
    )

    run_specific_download_dir = os.path.join(download_root, f"run_{run_to_process.id}")
    os.makedirs(run_specific_download_dir, exist_ok=True)

    final_target_path = os.path.join(run_specific_download_dir, target_filename)
    loaded_checkpoint_data = None  # Initialize here

    try:
        log.info(
            f"Downloading artifact '{selected_artifact.name}' to '{run_specific_download_dir}'..."
        )
        downloaded_artifact_location = selected_artifact.download(
            root=run_specific_download_dir
        )

        files_in_download_dir = os.listdir(downloaded_artifact_location)
        pkl_files = [
            f
            for f in files_in_download_dir
            if f.endswith(".pkl")
            and os.path.isfile(os.path.join(downloaded_artifact_location, f))
        ]

        if not pkl_files:
            log.error(
                f"No .pkl file found in downloaded artifact '{selected_artifact.name}' content at '{downloaded_artifact_location}'. Files found: {files_in_download_dir}"
            )
            return None, None

        downloaded_pkl_filename = pkl_files[0]
        if len(pkl_files) > 1:
            log.warning(
                f"Multiple .pkl files found in artifact download location '{downloaded_artifact_location}': {pkl_files}. Using '{downloaded_pkl_filename}'."
            )

        original_pkl_path = os.path.join(
            downloaded_artifact_location, downloaded_pkl_filename
        )

        if original_pkl_path != final_target_path:
            if os.path.exists(final_target_path):
                log.info(
                    f"Target file '{final_target_path}' exists. Removing before overwrite."
                )
                os.remove(final_target_path)
            os.rename(original_pkl_path, final_target_path)
            log.info(
                f"Moved/Renamed downloaded file from '{original_pkl_path}' to '{final_target_path}'."
            )
        else:
            log.info(
                f"Downloaded artifact file already at target path: '{final_target_path}'."
            )

        log.info(
            f"Attempting to load checkpoint directly using pickle from '{final_target_path}'..."
        )
        with open(final_target_path, "rb") as f:
            loaded_checkpoint_data = pickle.load(f)
        log.info(f"Data from '{final_target_path}' loaded successfully using pickle.")

        # New logic: Instantiate model using Hydra and update with loaded state
        log.info(
            "Attempting to instantiate model using Hydra and update with loaded state..."
        )
        loaded_model_state = loaded_checkpoint_data.get("model")
        if loaded_model_state is None:
            log.error("Key 'model' not found in loaded checkpoint data.")
            return None, loaded_checkpoint_data

        run_config = run_to_process.config
        model_hydra_config = run_config.get("model")

        if (
            not isinstance(model_hydra_config, dict)
            or "_target_" not in model_hydra_config
        ):
            log.error(
                f"Valid Hydra model config (dict with '_target_') not found in W&B run config at key 'model'. Found: {model_hydra_config}"
            )
            return None, loaded_checkpoint_data

        try:
            log.info(f"Instantiating model with Hydra config: {model_hydra_config}")
            # Ensure that nnx ops are performed on a JAX-compatible device if necessary,
            # though instantiation itself is usually fine on CPU.
            # NNX usually handles device placement during module calls or state updates.
            model = hydra.utils.instantiate(model_hydra_config)
            log.info(f"Model instantiated successfully: {type(model)}")

            log.info("Updating instantiated model with loaded state...")
            nnx.update(model, loaded_model_state)
            log.info("Model state updated successfully.")
            return model, loaded_checkpoint_data
        except Exception as e:
            import traceback

            log.error(
                f"Error during model instantiation or state update: {e}\n{traceback.format_exc()}"
            )
            return None, loaded_checkpoint_data

    except wandb.errors.Error as e:
        log.error(
            f"A W&B error occurred during artifact processing for run '{run_to_process.id}', artifact '{selected_artifact.name if 'selected_artifact' in locals() else 'unknown'}': {e}"
        )
        return (
            None,
            loaded_checkpoint_data,
        )  # Return loaded_checkpoint_data if it was loaded before this error
    except FileNotFoundError as e:
        log.error(f"File not found error during download/load: {e}")
        return None, loaded_checkpoint_data
    except Exception as e:
        import traceback

        log.error(
            f"An unexpected error occurred in load_best_model_from_wandb: {e}\n{traceback.format_exc()}"
        )
        return (
            None,
            loaded_checkpoint_data,
        )  # Return loaded_checkpoint_data if available
