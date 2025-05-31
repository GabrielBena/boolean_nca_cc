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
    use_local_latest=False,
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
        Tuple of (loaded_model, loaded_dict, config) containing the instantiated model,
        full loaded state, and the complete hydra config used during training
    """

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
            hidden_dim=config.model.hidden_dim,
        )

        n_nodes = int(graph.n_node[0])
        print(f"Computed n_nodes for CircuitSelfAttention: {n_nodes}")
        return n_nodes

    # Construct the potential local path

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
        run_id = run.id

    expected_checkpoint_path = None
    if run_id:
        local_artifact_dir = os.path.join(download_dir, f"run_{run_id}")
        expected_checkpoint_path = os.path.join(
            local_artifact_dir, f"{filename}.{filetype}"
        )
        print(f"Checking for local checkpoint at: {expected_checkpoint_path}")
        if os.path.exists(expected_checkpoint_path) and not use_local_latest:
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

                # Common overrides for hydra.instantiate
                instantiate_overrides = {"arity": config.circuit.arity, "rngs": rng}

                # Check if this is a self-attention model and add n_node if needed
                if (
                    config.model.get("type") == "self_attention"
                    or "self_attention" in config.model.get("_target_", "").lower()
                ):
                    n_nodes = compute_n_nodes_from_config(config)
                    instantiate_overrides["n_node"] = n_nodes

                model = hydra.utils.instantiate(config.model, **instantiate_overrides)

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

                return model, loaded_dict, config
            except Exception as e:
                print(
                    f"Error loading model from local checkpoint {expected_checkpoint_path}: {e}"
                )
                print("Proceeding to download from WandB.")

        elif use_local_latest:
            print("Loading latest model from local directory.")
            # Get all checkpoint files in the download directory
            checkpoint_file = glob.glob(
                os.path.join(download_dir, "**", "*.pkl"), recursive=True
            )
            checkpoint_file.sort(key=os.path.getmtime, reverse=True)
            latest_checkpoint = checkpoint_file[0]
            print(f"Loading latest checkpoint: {latest_checkpoint}")

            with open(latest_checkpoint, "rb") as f:
                loaded_dict = pickle.load(f)

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
                model = None

            # Ensure essential keys are present in loaded_dict
            if "run_id" not in loaded_dict:
                loaded_dict["run_id"] = run_id
            if "config" not in loaded_dict:
                loaded_dict["config"] = config  # Add config if not present

            return model, loaded_dict

            # Sort by modification time (newest first)

    # Get artifacts and find best model
    print("Retrieving artifacts...")
    artifacts = run.logged_artifacts()

    # Print all available artifacts for debugging
    print("Available artifacts:")
    for a in artifacts:
        print(f"  - {a.name}")

    best_models = [a for a in artifacts if filename in a.name]

    if not best_models:
        print(f"No artifacts found matching '{filename}'")
        # Try to find any model artifacts
        model_artifacts = [a for a in artifacts if "model" in a.name.lower()]
        if model_artifacts:
            print("Found other model artifacts:")
            for a in model_artifacts:
                print(f"  - {a.name}")
            raise ValueError(
                f"No {filename} artifacts found for run {run.id}, but found other model artifacts. Please check the artifact names."
            )
        else:
            raise ValueError(f"No model artifacts found for run {run.id}")

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
        with open(checkpoint_path, "rb") as f:
            loaded_dict = pickle.load(f)
    except Exception as e:
        print(f"Error loading checkpoint from {checkpoint_path}: {e}")
        raise

    # Get config from run
    config = OmegaConf.create(run.config)
    print(f"Instantiating model using original _target_: {config.model._target_}")

    # Initialize rngs for the model constructor
    rng = nnx.Rngs(params=jax.random.PRNGKey(seed))

    current_config_model_node = config.model

    # Convert the OmegaConf node for the model to a standard Python dictionary.
    # This dictionary will contain parameters like `hidden_dim`, `edge_mlp_features`, etc.,
    # and also `_target_` if present.
    model_params_from_config = OmegaConf.to_container(
        current_config_model_node, resolve=True
    )

    if not isinstance(model_params_from_config, dict):
        print(
            f"Error: Model config from WandB did not resolve to a dictionary. Got: {type(model_params_from_config)}"
        )
        raise TypeError(
            f"Expected model config to be a dict, got {type(model_params_from_config)}"
        )

    # Get the class path from _target_ and remove it from the params dict.
    class_path = model_params_from_config.pop("_target_", None)

    if class_path is None:
        print(
            f"Warning: '_target_' was missing in the resolved model parameters. Defaulting to 'boolean_nca_cc.models.CircuitGNN'"
        )
        class_path = "boolean_nca_cc.models.CircuitGNN"

    # Get the actual class type using Hydra's utility
    try:
        ModelClazz = hydra.utils.get_class(class_path)
    except Exception as e:
        print(f"Error getting class '{class_path}': {e}")
        raise

    # Prepare the full set of keyword arguments for the model's constructor
    final_constructor_kwargs = (
        model_params_from_config.copy()
    )  # Start with params from saved config
    final_constructor_kwargs["arity"] = config.circuit.arity  # Add arity
    final_constructor_kwargs["rngs"] = rng  # Add the runtime rngs object

    # Check if this is a self-attention model and add n_node if needed
    if (
        config.model.get("type") == "self_attention"
        or "self_attention" in class_path.lower()
        or "CircuitSelfAttention" in class_path
    ):
        n_nodes = compute_n_nodes_from_config(config)
        final_constructor_kwargs["n_node"] = n_nodes

    print(
        f"Attempting to instantiate {ModelClazz} with kwargs: {final_constructor_kwargs}"
    )

    # Instantiate the model directly
    try:
        model = ModelClazz(**final_constructor_kwargs)
    except Exception as e:
        print(
            f"Error during manual instantiation of {ModelClazz} with {final_constructor_kwargs}: {e}"
        )
        raise

    # Update model with loaded state (weights from the checkpoint)
    print("Updating model with loaded state")
    if "model" in loaded_dict:
        try:
            nnx.update(model, loaded_dict["model"])
        except (AttributeError, TypeError) as e:
            print(f"Direct update failed with error: {e}")
            print("Trying alternative update approach...")
            model_state = loaded_dict["model"]
            if hasattr(model_state, "_state_dict"):
                model_state = model_state._state_dict
            for collection_name, collection in model_state.items():
                for var_name, value in collection.items():
                    try:
                        path = f"{collection_name}.{var_name}"
                        model.put(path, value)
                    except Exception as inner_e:
                        print(f"Warning: Failed to update {path}: {inner_e}")
    else:
        print("Warning: No 'model' key found in loaded dictionary")

    # Ensure essential keys are present in loaded_dict
    if "run_id" not in loaded_dict:
        loaded_dict["run_id"] = run.id
    if "config" not in loaded_dict:
        loaded_dict["config"] = config

    return model, loaded_dict, config
