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
    run_id=None,
    filters=None,
    seed=0,
    project="boolean-nca-cc",
    entity="m2snn",
    download_dir="saves",
    filename="best_model_hard_accuracy",
    filetype="pkl",
    run_from_last=1,
):
    """Load the best model from wandb artifacts.

    Args:
        run_id: Optional specific run ID to load from
        filters: Optional dictionary of filters to find runs
        seed: Seed for RNG initialization
        project: WandB project name
        entity: WandB entity/username
        download_dir: Directory to download artifacts to
        filename: Filename of the best model
        filetype: Filetype of the best model

    Returns:
        Tuple of (loaded_model, loaded_dict) containing the instantiated model and full loaded state
    """
    # Construct the potential local path
    expected_checkpoint_path = None
    if run_id:
        local_artifact_dir = os.path.join(download_dir, f"run_{run_id}")
        expected_checkpoint_path = os.path.join(
            local_artifact_dir, f"{filename}.{filetype}"
        )
        print(f"Checking for local checkpoint at: {expected_checkpoint_path}")
        if os.path.exists(expected_checkpoint_path):
            print(f"Found local checkpoint for run {run_id}. Loading from disk.")
            try:
                with open(expected_checkpoint_path, "rb") as f:
                    loaded_dict = pickle.load(f)

                # Get config (assuming it's stored or can be fetched if needed for instantiation)
                # For simplicity, we assume config is part of loaded_dict or can be inferred.
                # If full config is needed and not in pickle, this part might need adjustment
                # or fetching from wandb api.run(run_id).config
                api = wandb.Api()
                run_config = api.run(f"{entity}/{project}/{run_id}").config
                config = OmegaConf.create(run_config)
                print(
                    f"Instantiating model using config from run {run_id}: {config.model._target_}"
                )

                rng = nnx.Rngs(params=jax.random.PRNGKey(seed))
                model = hydra.utils.instantiate(
                    config.model, arity=config.circuit.arity, rngs=rng
                )

                if "model" in loaded_dict:
                    try:
                        nnx.update(model, loaded_dict["model"])
                    except (AttributeError, TypeError) as e:
                        print(f"Direct update failed with error: {e}")
                        print("Trying alternative update approach for local load...")
                        model_state = loaded_dict["model"]
                        if hasattr(model_state, "_state_dict"):
                            model_state = model_state._state_dict
                        for collection_name, collection in model_state.items():
                            for var_name, value in collection.items():
                                try:
                                    path = f"{collection_name}.{var_name}"
                                    model.put(path, value)
                                except Exception as inner_e:
                                    print(
                                        f"Warning: Failed to update {path} during local load: {inner_e}"
                                    )
                else:
                    print("Warning: No 'model' key found in local loaded dictionary")

                # Ensure essential keys are present in loaded_dict
                if "run_id" not in loaded_dict:
                    loaded_dict["run_id"] = run_id
                if "config" not in loaded_dict:
                    loaded_dict["config"] = config  # Add config if not present

                return model, loaded_dict
            except Exception as e:
                print(
                    f"Error loading model from local checkpoint {expected_checkpoint_path}: {e}"
                )
                print("Proceeding to download from WandB.")

    # Initialize WandB API
    api = wandb.Api()

    # Find the run
    if run_id:
        print(f"Looking for run with ID: {run_id}")
        run = api.run(f"{entity}/{project}/{run_id}")
        print(f"Found run: {run.name}")
    else:
        if not filters:
            filters = {}
        print(f"Looking for runs with filters: {filters}")
        runs = api.runs(f"{entity}/{project}", filters=filters)

        if not runs:
            raise ValueError(f"No runs found matching filters: {filters}")

        print(f"Found {len(runs)} matching runs, using the most recent one.")
        run = runs[len(runs) - run_from_last]  # Most recent run first
        print(f"Selected run: {run.name} (ID: {run.id})")

    # Get artifacts and find best model
    print("Retrieving artifacts...")
    artifacts = run.logged_artifacts()
    best_models = [a for a in artifacts if filename in a.name]

    if not best_models:
        raise ValueError(f"No best model artifacts found for run {run.id}")

    latest_best = best_models[-1]
    print(f"Found best model artifact: {latest_best.name}")

    # Create download directory if it doesn't exist
    download_path = os.path.join(download_dir, f"run_{run.id}")
    os.makedirs(download_path, exist_ok=True)

    # Download the artifact
    print(f"Downloading artifact to {download_path}")
    artifact_dir = latest_best.download(
        root=download_path
    )  # Use root to control download folder precisely
    checkpoint_path = os.path.join(
        artifact_dir, f"{filename}.{filetype}"
    )  # Adjusted path

    # Load the saved state
    print(f"Loading model from {checkpoint_path}")
    try:
        with open(checkpoint_path, "rb") as f:  # Use the corrected checkpoint_path
            loaded_dict = pickle.load(f)

        # Get config and instantiate model
        config = OmegaConf.create(run.config)
        print(f"Instantiating model using config: {config.model._target_}")

        # Create RNG key
        rng = nnx.Rngs(params=jax.random.PRNGKey(seed))

        # Instantiate model using hydra
        model = hydra.utils.instantiate(
            config.model, arity=config.circuit.arity, rngs=rng
        )

        # Update model with loaded state - handle potential API differences
        print("Updating model with loaded state")
        if "model" in loaded_dict:
            try:
                # Try the direct update approach
                nnx.update(model, loaded_dict["model"])
            except (AttributeError, TypeError) as e:
                print(f"Direct update failed with error: {e}")
                print("Trying alternative update approach...")

                # Alternative approach - recreate the state dictionary structure if needed
                model_state = loaded_dict["model"]
                # This handles cases where the state might be a VariableState object
                if hasattr(model_state, "_state_dict"):
                    model_state = model_state._state_dict

                # Update the model with the extracted state dict
                for collection_name, collection in model_state.items():
                    for var_name, value in collection.items():
                        try:
                            path = f"{collection_name}.{var_name}"
                            model.put(path, value)
                        except Exception as inner_e:
                            print(f"Warning: Failed to update {path}: {inner_e}")

        else:
            print("Warning: No 'model' key found in loaded dictionary")

        # Add the run ID to the loaded dictionary for reference
        loaded_dict["run_id"] = run.id
        # Ensure config from the run is in loaded_dict, especially if not loading locally
        loaded_dict["config"] = config

        return model, loaded_dict

    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback

        print(f"Traceback: {traceback.format_exc()}")
        raise
