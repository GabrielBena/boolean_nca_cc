import logging
import os
import pickle
import warnings
from typing import Any

import hydra
import jax
from flax import nnx
from omegaconf import OmegaConf

import wandb
from boolean_nca_cc.circuits.model import gen_circuit, generate_layer_sizes
from boolean_nca_cc.utils.graph_builder import build_graph

# Setup logging
log = logging.getLogger(__name__)


def load_checkpoint_with_compatibility_working(checkpoint_path):
    """
    Working compatibility loader that handles Flax and JAX version issues.

    This loader handles:
    - flax.nnx.nnx -> flax.nnx module remapping
    - Missing _var_metadata attributes in VariableState objects
    - JAX version compatibility (MainTrace location changes)
    """
    log.info(f"Loading checkpoint with compatibility handling: {checkpoint_path}")

    try:

        class WorkingCompatibilityUnpickler(pickle.Unpickler):
            def find_class(self, module, name):
                # Handle flax.nnx.nnx -> flax.nnx remapping
                if module.startswith("flax.nnx.nnx"):
                    log.debug(f"Remapping {module}.{name}")

                    if name == "State":
                        # Return a custom State class that can handle the old format
                        class CompatibleState(dict):
                            def __init__(self, *args, **kwargs):
                                super().__init__()
                                if args and isinstance(args[0], dict):
                                    self.update(args[0])
                                if kwargs:
                                    self.update(kwargs)

                            def __setstate__(self, state):
                                if isinstance(state, dict):
                                    self.clear()
                                    self.update(state)

                        return CompatibleState

                    elif name == "VariableState":
                        # Return a custom VariableState that can handle missing _var_metadata
                        class CompatibleVariableState:
                            def __init__(self, *args, **kwargs):
                                self.type = kwargs.get("type", nnx.Param)
                                self.value = kwargs.get("value")
                                for key, value in kwargs.items():
                                    if key not in ["_var_metadata"] and not hasattr(self, key):
                                        setattr(self, key, value)

                            def __setstate__(self, state):
                                if isinstance(state, dict):
                                    for key, value in state.items():
                                        if key != "_var_metadata":
                                            setattr(self, key, value)

                            def __getattr__(self, name):
                                if name == "_var_metadata":
                                    return None
                                raise AttributeError(
                                    f"'{self.__class__.__name__}' object has no attribute '{name}'"
                                )

                        return CompatibleVariableState

                    elif name == "Param":
                        # Return a custom Param class
                        class CompatibleParam:
                            def __init__(self, *args, **kwargs):
                                if args:
                                    self.value = args[0]
                                else:
                                    self.value = kwargs.get("value", None)
                                for key, value in kwargs.items():
                                    if key not in ["_var_metadata"] and key != "value":
                                        setattr(self, key, value)

                            def __setstate__(self, state):
                                if isinstance(state, dict):
                                    for key, value in state.items():
                                        if key != "_var_metadata":
                                            setattr(self, key, value)

                        return CompatibleParam

                    else:
                        # Generic compatible class
                        class CompatibleGeneric:
                            def __init__(self, *args, **kwargs):
                                for key, value in kwargs.items():
                                    if key != "_var_metadata":
                                        setattr(self, key, value)

                            def __setstate__(self, state):
                                if isinstance(state, dict):
                                    for key, value in state.items():
                                        if key != "_var_metadata":
                                            setattr(self, key, value)

                        return CompatibleGeneric

                # Handle JAX compatibility issues
                if "jax" in module and name == "MainTrace":
                    log.debug(f"Handling JAX compatibility for {module}.{name}")
                    try:
                        import jax

                        if hasattr(jax._src.core, "MainTrace"):
                            return jax._src.core.MainTrace
                        else:

                            class CompatibleMainTrace:
                                def __init__(self, *args, **kwargs):
                                    pass

                            return CompatibleMainTrace
                    except Exception:

                        class CompatibleMainTrace:
                            def __init__(self, *args, **kwargs):
                                pass

                        return CompatibleMainTrace

                return super().find_class(module, name)

        with open(checkpoint_path, "rb") as f:
            unpickler = WorkingCompatibilityUnpickler(f)
            checkpoint = unpickler.load()

        log.info("Successfully loaded checkpoint with working compatibility loader")

        # Convert compatible objects back to real nnx objects
        if "model" in checkpoint:
            checkpoint["model"] = convert_compatible_to_nnx(checkpoint["model"])
            log.info("Converted compatible objects to nnx format")

        return checkpoint

    except Exception as e:
        log.error(f"Working compatibility loading failed: {e}")
        raise


def convert_compatible_to_nnx(obj):
    """
    Convert compatible objects created during loading back to real nnx objects.

    This ensures that the loaded state can be properly used to update the model.
    """
    if hasattr(obj, "__class__") and "Compatible" in obj.__class__.__name__:
        class_name = obj.__class__.__name__.replace("Compatible", "")

        if class_name == "VariableState":
            # Convert to real VariableState
            try:
                var_type = getattr(obj, "type", nnx.Param)
                value = getattr(obj, "value", None)
                if value is not None:
                    return nnx.VariableState(type=var_type, value=value)
                else:
                    # Return as dict if we can't create the proper object
                    return {k: v for k, v in obj.__dict__.items() if not k.startswith("_")}
            except Exception as e:
                log.warning(f"Could not convert VariableState: {e}")
                return {k: v for k, v in obj.__dict__.items() if not k.startswith("_")}

        elif class_name == "Param":
            # Convert to real Param
            try:
                value = getattr(obj, "value", None)
                if value is not None:
                    return nnx.Param(value)
                else:
                    return getattr(obj, "value", None)
            except Exception as e:
                log.warning(f"Could not convert Param: {e}")
                return getattr(obj, "value", None)

        elif class_name == "State":
            # Convert State (usually a dict-like collection)
            if isinstance(obj, dict):
                return {k: convert_compatible_to_nnx(v) for k, v in obj.items()}
            else:
                return {
                    k: convert_compatible_to_nnx(v)
                    for k, v in obj.__dict__.items()
                    if not k.startswith("_")
                }

        else:
            # Generic conversion - return as dict
            return {
                k: convert_compatible_to_nnx(v)
                for k, v in obj.__dict__.items()
                if not k.startswith("_")
            }

    elif isinstance(obj, dict):
        # Recursively convert dictionary values
        return {k: convert_compatible_to_nnx(v) for k, v in obj.items()}

    elif isinstance(obj, (list, tuple)):
        # Recursively convert list/tuple items
        converted = [convert_compatible_to_nnx(item) for item in obj]
        return type(obj)(converted)

    else:
        # Return as-is for regular objects
        return obj


def load_checkpoint_with_compatibility(checkpoint_path):
    """
    Load a checkpoint with backward compatibility for older Flax versions.

    This uses the working compatibility loader that handles:
    - flax.nnx.nnx -> flax.nnx module remapping
    - Missing _var_metadata attributes
    - JAX version compatibility issues

    Args:
        checkpoint_path: Path to the checkpoint file

    Returns:
        Dictionary containing the checkpoint data
    """
    return load_checkpoint_with_compatibility_working(checkpoint_path)


# Removed old manual reconstruction code - no longer needed


# Removed old state conversion function - no longer needed with working loader


def configure_notebook_logging(level=logging.INFO):
    """
    Configure logging for Jupyter notebook usage.

    Call this function in your notebook before using load_best_model_from_wandb
    to see detailed progress information.

    Args:
        level: Logging level (default: logging.INFO)

    Example:
        >>> from boolean_nca_cc.training.checkpointing import configure_notebook_logging
        >>> configure_notebook_logging()
        >>> # Now you'll see logs from load_best_model_from_wandb
    """
    # Configure root logger for notebook display
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        force=True,  # Override any existing configuration
    )

    # Set the level for this specific logger, but don't add additional handlers
    # since basicConfig already sets up the root logger with a console handler
    log.setLevel(level)

    # Prevent duplication by ensuring we don't add extra handlers
    # The root logger configured by basicConfig will handle the output
    log.propagate = True  # This is the default, but being explicit

    print(f"Logging configured for notebooks. Level: {logging.getLevelName(level)}")


def instantiate_model_from_config(config, seed=0):
    """
    Instantiate a model from a loaded configuration using hydra.utils.instantiate.

    This function handles the common logic for creating models from configs,
    including computing n_nodes for self-attention models.

    Args:
        config: OmegaConf configuration object containing model parameters
        seed: Random seed for model initialization

    Returns:
        Instantiated model object
    """

    def compute_n_nodes_from_config(config):
        """Compute n_nodes for CircuitSelfAttention models by building a dummy graph."""
        # Generate circuit layer sizes
        input_n, output_n = config.circuit.input_bits, config.circuit.output_bits
        arity = config.circuit.arity

        if config.circuit.layer_sizes is None:
            layer_sizes = generate_layer_sizes(
                input_n, output_n, arity, layer_n=config.circuit.num_layers,
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

    print(f"Instantiating model using config: {config.model._target_}")

    # Initialize rngs for the model constructor
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

    # Instantiate the model using hydra
    try:
        model = hydra.utils.instantiate(config.model, **instantiate_overrides)
        print(f"Successfully instantiated model: {type(model).__name__}")
        return model
    except Exception as e:
        print(f"Error instantiating model: {e}")
        raise


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

    # log.info(f"Saved checkpoint to {checkpoint_path}")
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


# Removed old load_old_checkpoint_safe - functionality integrated into main loader


def load_checkpoint(checkpoint_path):
    """Load a checkpoint from a file with backward compatibility."""
    return load_checkpoint_with_compatibility(checkpoint_path)


def load_checkpoint_legacy(checkpoint_path):
    """Load a checkpoint from a file using standard pickle (legacy method)."""
    with open(checkpoint_path, "rb") as f:
        return pickle.load(f)


# WandB integration functions
def load_config_from_wandb(
    run_id: str | None = None,
    filters: dict[str, Any] | None = None,
    project: str = "boolean-nca-cc",
    entity: str = "marcello-barylli-growai",
    download_dir: str = "saves",
    filename: str = "best_model",
    filetype: str = "pkl",
    run_from_last: int = 1,
    use_cache: bool = True,
    force_download: bool = False,
    select_by_best_metric: bool = False,
    metric_name: str = "eval_ko_in/final_hard_accuracy",
    prefer_metric: str | None = None,
) -> tuple[Any, str, str]:
    """
    Load config and checkpoint information from WandB artifacts.

    This function finds the appropriate run, downloads the checkpoint if needed,
    and returns the config along with checkpoint path information. This allows
    for config modification before model instantiation.

    Args:
        run_id: Optional specific run ID to load from
        filters: Optional dictionary of filters to find runs
        project: WandB project name
        entity: WandB entity/username
        download_dir: Directory to download artifacts to
        filename: Filename of the best model
        filetype: Filetype of the best model
        run_from_last: Index from the end of runs list to select
        use_cache: Whether to use locally cached artifacts if available
        force_download: If True, always download from wandb even if local cache exists
        select_by_best_metric: If True, select run with highest metric
        metric_name: Name of the metric to maximize when select_by_best_metric is True

    Returns:
        Tuple of (config, checkpoint_path, run_id) containing the hydra config,
        path to the checkpoint file, and the run ID
    """

    # Initialize WandB API
    api = wandb.Api()

    # Find the run
    if run_id:
        log.info(f"Looking for run with ID: {run_id}")
        run = api.run(f"{entity}/{project}/{run_id}")
        log.info(f"Found run: {run.name}")
    else:
        if not filters:
            filters = {}
        log.info(f"Looking for runs with filters: {filters}")
        runs = api.runs(f"{entity}/{project}", filters=filters)

        if not runs:
            raise ValueError(f"No runs found matching filters: {filters}")

        log.info(f"Found {len(runs)} matching runs.")

        if select_by_best_metric:
            log.info(f"Selecting run with highest {metric_name} metric...")
            best_run = None
            best_metric_value = float("-inf")

            for candidate_run in runs:
                if candidate_run.state == "running":
                    log.info(
                        f"Run {candidate_run.name} (ID: {candidate_run.id}): Skipping unfinished run"
                    )
                    continue

                # Get the summary metrics for this run
                summary = candidate_run.summary

                # Try to get the metric value
                metric_value = summary.get(metric_name)
                if metric_value is not None:
                    log.info(
                        f"Run {candidate_run.name} (ID: {candidate_run.id}): {metric_name} = {metric_value}"
                    )
                    if metric_value > best_metric_value:
                        best_metric_value = metric_value
                        best_run = candidate_run
                else:
                    log.info(
                        f"Run {candidate_run.name} (ID: {candidate_run.id}): {metric_name} not found in summary"
                    )

            if best_run is None:
                log.info(
                    f"Warning: No runs found with {metric_name} metric. Falling back to run_from_last selection."
                )
                run = runs[len(runs) - run_from_last]
            else:
                run = best_run
                log.info(
                    f"Selected run with highest {metric_name} ({best_metric_value}): {run.name} (ID: {run.id})"
                )
        else:
            log.info(f"Using run_from_last={run_from_last} to select run.")
            run = runs[len(runs) - run_from_last]  # Most recent run first
            log.info(f"Selected run: {run.name} (ID: {run.id})")

        run_id = run.id

    # Check for cached checkpoint
    expected_checkpoint_path = None
    if run_id:
        local_artifact_dir = os.path.join(download_dir, f"run_{run_id}")
        expected_checkpoint_path = os.path.join(local_artifact_dir, f"{filename}.{filetype}")
        log.info(f"Checking for local checkpoint at: {expected_checkpoint_path}")

        # Check if we should use local cache
        if use_cache and not force_download and os.path.exists(expected_checkpoint_path):
            log.info(f"Found cached checkpoint for run {run_id}. Loading config from disk.")
            try:
                api = wandb.Api()
                run_config = api.run(f"{entity}/{project}/{run_id}").config
                config = OmegaConf.create(run_config)

                return config, expected_checkpoint_path, run_id
            except Exception as e:
                log.info(
                    f"Error loading config from local checkpoint {expected_checkpoint_path}: {e}"
                )
                log.info("Proceeding to download from WandB.")
        elif force_download:
            log.info("Force download enabled, skipping local cache.")
        elif not use_cache:
            log.info("Cache disabled, downloading from WandB.")

    # Get artifacts and find best model
    log.info("Retrieving artifacts...")
    artifacts = run.logged_artifacts()

    # Print all available artifacts for debugging
    log.info("Available artifacts:")
    for a in artifacts:
        log.info(f"  - {a.name}")

    best_models = [a for a in artifacts if filename in a.name]

    if not best_models:
        log.info(f"No artifacts found matching '{filename}'")
        # Fallback: try latest checkpoint artifacts
        latest_ckpts = [a for a in artifacts if "latest_checkpoint" in a.name]
        if latest_ckpts:
            # Pick the most recent one (artifacts are usually in chronological order)
            selected_artifact = latest_ckpts[-1]
            log.info(f"Falling back to latest checkpoint artifact: {selected_artifact.name}")
        else:
            # Try to find any model artifacts for better error reporting
            model_artifacts = [a for a in artifacts if "model" in a.name.lower()]
            if model_artifacts:
                log.info("Found other model artifacts:")
                for a in model_artifacts:
                    log.info(f"  - {a.name}")
                raise ValueError(
                    f"No {filename} or latest_checkpoint artifacts found for run {run.id},"
                    " but found other model artifacts. Please check the artifact names."
                )
            else:
                raise ValueError(f"No model artifacts found for run {run.id}")
    else:
        # Intelligent selection of best model artifact
        if len(best_models) == 1:
            selected_artifact = best_models[0]
            log.info(f"Found single best model artifact: {selected_artifact.name}")
        else:
            log.info(f"Found {len(best_models)} best model artifacts:")
            for a in best_models:
                log.info(f"  - {a.name}")

            selected_artifact = _select_best_artifact(best_models, prefer_metric)
            log.info(f"Selected best model artifact: {selected_artifact.name}")

    latest_best = selected_artifact

    # Create download directory if it doesn't exist
    download_path = os.path.join(download_dir, f"run_{run.id}")
    os.makedirs(download_path, exist_ok=True)

    # Download the artifact
    log.info(f"Downloading artifact to {download_path}")
    artifact_dir = latest_best.download(root=download_path)
    checkpoint_path = os.path.join(artifact_dir, f"{filename}.{filetype}")

    # Get config from run
    config = OmegaConf.create(run.config)

    return config, checkpoint_path, run.id


def load_model_from_config_and_checkpoint(
    config: Any,
    checkpoint_path: str,
    run_id: str,
    seed: int = 0,
) -> tuple[Any, dict[str, Any]]:
    """
    Load a model from config and checkpoint file.

    This function takes a config (which can be modified) and loads the model
    from the specified checkpoint path.

    Args:
        config: The hydra config (can be modified from original)
        checkpoint_path: Path to the checkpoint file
        run_id: The WandB run ID
        seed: Seed for RNG initialization

    Returns:
        Tuple of (loaded_model, loaded_dict) containing the instantiated model
        and full loaded state
    """

    # Load the saved state using unified loading
    log.info(f"Loading model from {checkpoint_path}")
    try:
        loaded_dict = load_checkpoint_with_compatibility(checkpoint_path)
    except Exception as e:
        log.error(f"Error loading checkpoint from {checkpoint_path}: {e}")
        raise

    # Instantiate model using the reusable function
    model = instantiate_model_from_config(config, seed=seed)

    # Update model with loaded state (compatibility handled during loading)
    nnx.update(model, loaded_dict["model"])

    # Ensure essential keys are present in loaded_dict
    if "run_id" not in loaded_dict:
        loaded_dict["run_id"] = run_id
    if "config" not in loaded_dict:
        loaded_dict["config"] = config

    return model, loaded_dict


def load_best_model_from_wandb(
    run_id: str | None = None,
    filters: dict[str, Any] | None = None,
    seed: int = 0,
    project: str = "boolean-nca-cc",
    entity: str = "marcello-barylli-growai",
    download_dir: str = "saves",
    filename: str = "best_model",
    filetype: str = "pkl",
    run_from_last: int = 1,
    use_cache: bool = True,
    force_download: bool = False,
    select_by_best_metric: bool = False,
    metric_name: str = "eval_ko_in/final_hard_accuracy",
    prefer_metric: str | None = None,
) -> tuple[Any, dict[str, Any], Any]:
    """
    Load the best model from WandB artifacts with full backward compatibility.

    This function handles both old pickle-based artifacts and new Orbax-based
    checkpoints seamlessly, providing a unified interface for model loading.

    This is now a convenience wrapper around the split functions load_config_from_wandb
    and load_model_from_config_and_checkpoint.

    Args:
        run_id: Optional specific run ID to load from
        filters: Optional dictionary of filters to find runs
        seed: Seed for RNG initialization
        project: WandB project name
        entity: WandB entity/username
        download_dir: Directory to download artifacts to
        filename: Filename of the best model
        filetype: Filetype of the best model
        run_from_last: Index from the end of runs list to select
        use_cache: Whether to use locally cached artifacts if available
        force_download: If True, always download from wandb even if local cache exists
        select_by_best_metric: If True, select run with highest metric
        metric_name: Name of the metric to maximize when select_by_best_metric is True

    Returns:
        Tuple of (loaded_model, loaded_dict, config) containing the instantiated model,
        full loaded state, and the complete hydra config used during training
    """

    # For backward compatibility, if filename is the old style, update it to be more generic
    if filename == "best_model_hard_accuracy":
        log.info("Detected old-style filename, updating to generic 'best_model' for compatibility")
        filename = "best_model"

    # Load config and checkpoint information
    config, checkpoint_path, run_id = load_config_from_wandb(
        run_id=run_id,
        filters=filters,
        project=project,
        entity=entity,
        download_dir=download_dir,
        filename=filename,
        filetype=filetype,
        run_from_last=run_from_last,
        use_cache=use_cache,
        force_download=force_download,
        select_by_best_metric=select_by_best_metric,
        metric_name=metric_name,
        prefer_metric=prefer_metric,
    )

    # Load model from config and checkpoint
    model, loaded_dict = load_model_from_config_and_checkpoint(
        config=config,
        checkpoint_path=checkpoint_path,
        run_id=run_id,
        seed=seed,
    )

    return model, loaded_dict, config


# Best Model Tracking and Checkpointing Functions


class BestModelTracker:
    """
    Unified best model tracker that can handle multiple metrics and save the best models.

    This class tracks the best models for different metrics (e.g., in-distribution vs out-of-distribution)
    and saves checkpoints when improvements are detected.
    """

    def __init__(self):
        self.best_metrics = {}  # metric_key -> best_value
        self.best_epochs = {}  # metric_key -> epoch when best was achieved

    def add_metric(self, metric_key: str, initial_value: float = None):
        """Add a new metric to track."""
        if initial_value is None:
            # Default to worst possible value based on metric type
            initial_value = float("-inf") if "accuracy" in metric_key.lower() else float("inf")

        self.best_metrics[metric_key] = initial_value
        self.best_epochs[metric_key] = -1

    def is_better(self, metric_key: str, current_value: float) -> bool:
        """Check if current value is better than the best for this metric."""
        if metric_key not in self.best_metrics:
            self.add_metric(metric_key)

        best_value = self.best_metrics[metric_key]

        if "accuracy" in metric_key.lower():
            return current_value > best_value  # Higher is better
        else:
            return current_value < best_value  # Lower is better

    def update(self, metric_key: str, current_value: float, epoch: int) -> bool:
        """Update the best value if current is better. Returns True if updated."""
        if self.is_better(metric_key, current_value):
            self.best_metrics[metric_key] = current_value
            self.best_epochs[metric_key] = epoch
            return True
        return False

    def get_best_value(self, metric_key: str) -> float:
        """Get the best value for a metric."""
        return self.best_metrics.get(
            metric_key, float("-inf") if "accuracy" in metric_key.lower() else float("inf")
        )

    def get_best_epoch(self, metric_key: str) -> int:
        """Get the epoch when the best value was achieved."""
        return self.best_epochs.get(metric_key, -1)


def get_metric_value(
    metric_name: str,
    metric_source: str,
    training_metrics: dict,
    eval_metrics: dict | None = None,
) -> float:
    """
    Get metric value from the appropriate source.

    Args:
        metric_name: Name of the metric ('loss', 'hard_loss', 'accuracy', 'hard_accuracy')
        metric_source: Source of the metric ('training' or 'eval')
        training_metrics: Dictionary with training metrics
        eval_metrics: Dictionary with evaluation metrics (optional)

    Returns:
        The metric value as a float
    """
    if metric_source == "training":
        return training_metrics[metric_name]
    elif metric_source == "eval":
        if eval_metrics is None:
            raise ValueError("Evaluation metrics not available for eval source")

        # Map to evaluation metric keys (use IN-distribution evaluation for consistency)
        eval_key_map = {
            "loss": "eval_ko_in/final_loss",
            "hard_loss": "eval_ko_in/final_hard_loss",
            "accuracy": "eval_ko_in/final_accuracy",
            "hard_accuracy": "eval_ko_in/final_hard_accuracy",
        }

        # Fallback map to OUT-of-distribution evaluation metrics
        eval_out_key_map = {
            "loss": "eval_ko_out/final_loss",
            "hard_loss": "eval_ko_out/final_hard_loss",
            "accuracy": "eval_ko_out/final_accuracy",
            "hard_accuracy": "eval_ko_out/final_hard_accuracy",
        }

        # Try IN-distribution metrics first, fallback to OUT-of-distribution if not available
        primary_key = eval_key_map[metric_name]
        fallback_key = eval_out_key_map[metric_name]

        if primary_key in eval_metrics:
            return eval_metrics[primary_key]
        elif fallback_key in eval_metrics:
            return eval_metrics[fallback_key]
        else:
            raise KeyError(f"Neither {primary_key} nor {fallback_key} found in evaluation metrics")
    elif metric_source == "eval_ko_in":
        if eval_metrics is None:
            raise ValueError("Knockout evaluation metrics not available for eval_ko_in source")
        # Map to knockout evaluation metric keys
        eval_key_map = {
            "loss": "eval_ko_in/final_loss",
            "hard_loss": "eval_ko_in/final_hard_loss",
            "accuracy": "eval_ko_in/final_accuracy",
            "hard_accuracy": "eval_ko_in/final_hard_accuracy",
        }
        return eval_metrics[eval_key_map[metric_name]]
    else:
        raise ValueError(f"Unknown metric source: {metric_source}")


def setup_checkpoint_dir(checkpoint_dir: str | None, wandb_id: str | None) -> str | None:
    """Setup checkpoint directory with unique identifier."""
    if checkpoint_dir is None:
        return None

    # Create unique checkpoint directory using wandb ID or timestamp
    from datetime import datetime
    unique_id = wandb_id if wandb_id else datetime.now().strftime("%Y%m%d_%H%M%S")

    checkpoint_path = os.path.join(checkpoint_dir, f"run_{unique_id}")
    os.makedirs(checkpoint_path, exist_ok=True)
    log.info(f"Checkpoints will be saved to: {checkpoint_path}")

    return checkpoint_path


def save_periodic_checkpoint(
    checkpoint_path: str,
    model,
    optimizer,
    metrics: dict,
    epoch: int,
    checkpoint_interval: int,
    wandb_run=None,
) -> None:
    """Save periodic checkpoint if interval allows."""
    if checkpoint_path is None or epoch == 0 or epoch % checkpoint_interval != 0:
        return

    ckpt_filename = "latest_checkpoint.pkl"

    try:
        save_checkpoint(
            model,
            optimizer,
            metrics,
            {"epoch": epoch},
            epoch,
            checkpoint_path,
            filename=ckpt_filename,
        )

        # Log to wandb if enabled
        if wandb_run:
            wandb_run.save(os.path.join(checkpoint_path, ckpt_filename))

            # Also log this as an artifact for better tracking in wandb
            try:
                artifact = wandb_run.Artifact("latest_checkpoint", type="model")
                artifact.add_file(os.path.join(checkpoint_path, ckpt_filename))
                wandb_run.log_artifact(artifact)
            except Exception as e:
                log.warning(f"Error logging checkpoint as artifact: {e}")

    except Exception as e:
        log.warning(f"Error saving checkpoint: {e}")


def save_best_checkpoint(
    checkpoint_path: str,
    is_best: bool,
    save_best: bool,
    model,
    optimizer,
    metrics: dict,
    epoch: int,
    best_metric: str,
    current_metric_value: float,
    wandb_run=None,
) -> None:
    """Save best checkpoint if enabled and is best."""
    if not (checkpoint_path and save_best and is_best):
        return

    # Use a consistent filename for the best model regardless of metric
    # This ensures compatibility with loading logic that expects "best_model.pkl"
    best_filename = "best_model.pkl"
    log.info(
        f"Saving best model at epoch {epoch} with {best_metric}={current_metric_value:.4f}"
    )

    try:
        save_checkpoint(
            model,
            optimizer,
            metrics,
            {"epoch": epoch, f"best_{best_metric}": current_metric_value},
            epoch,
            checkpoint_path,
            filename=best_filename,
        )

        # Log to wandb if enabled
        if wandb_run:
            wandb_run.log({f"best/{best_metric}": current_metric_value, "best/epoch": epoch})

            # Save the best model to wandb (will overwrite the previous best)
            wandb_run.save(os.path.join(checkpoint_path, best_filename))

            # Also log this as an artifact for better tracking in wandb
            try:
                artifact = wandb_run.Artifact("best_model", type="model")
                artifact.add_file(os.path.join(checkpoint_path, best_filename))
                wandb_run.log_artifact(artifact)
            except Exception as e:
                log.warning(f"Error logging best model as artifact: {e}")
    except Exception as e:
        log.warning(f"Error saving best checkpoint: {e}")


def save_stable_state(
    checkpoint_path: str,
    save_stable_states: bool,
    last_stable_state: dict,
    epoch: int,
    wandb_run=None,
) -> None:
    """Save the last stable state before NaN loss."""
    if not (checkpoint_path and save_stable_states):
        return

    try:
        stable_path = os.path.join(checkpoint_path, f"stable_state_epoch_{epoch - 1}.pkl")
        save_checkpoint(
            last_stable_state["model"],
            last_stable_state["optimizer"],
            last_stable_state["metrics"],
            {"epoch": epoch - 1},
            epoch - 1,
            os.path.dirname(stable_path),
            filename=os.path.basename(stable_path),
        )

        # Log to wandb if enabled
        if wandb_run:
            wandb_run.log({"training/early_stop_epoch": epoch - 1})
            wandb_run.alert(
                title="Training Stopped - NaN Loss",
                text=f"Training stopped at epoch {epoch} due to NaN loss. Last stable state saved.",
                level=wandb_run.AlertLevel.WARN,
            )
    except Exception as e:
        log.warning(f"Error saving stable state: {e}")


def track_and_save_best_models(
    best_model_tracker: BestModelTracker,
    checkpoint_path: str | None,
    save_best: bool,
    model,
    optimizer,
    metrics: dict,
    epoch: int,
    training_metrics: dict | None = None,
    eval_metrics: dict | None = None,
    wandb_run=None,
    track_metrics: list[str] | None = None,
) -> dict:
    """
    Unified function to track and save best models for multiple metrics.

    This function can handle both training and evaluation metrics, and track
    the best models for both in-distribution and out-of-distribution evaluations.

    Args:
        best_model_tracker: BestModelTracker instance
        checkpoint_path: Path to save checkpoints
        save_best: Whether to save best models
        model: Model to save
        optimizer: Optimizer to save
        metrics: Training metrics to save with checkpoint
        epoch: Current epoch
        training_metrics: Training metrics dict (optional)
        eval_metrics: Evaluation metrics dict (optional)
        wandb_run: WandB run object (optional)
        track_metrics: List of specific metrics to track in format "source_metric"
                      (e.g., ["eval_ko_in_hard_accuracy", "eval_ko_out_hard_accuracy"]).
                      If None, tracks all available metrics.

    Returns:
        Dictionary with information about which models were updated
    """
    updates = {}

    # Collect all available metrics
    available_metrics = []

    # Add training metrics if available
    if training_metrics:
        for metric in ["hard_accuracy", "accuracy", "hard_loss", "loss"]:
            if metric in training_metrics:
                available_metrics.append(("training", metric, training_metrics[metric]))

    # Add evaluation metrics if available
    if eval_metrics:
        # In-distribution metrics
        for metric in ["hard_accuracy", "accuracy", "hard_loss", "loss"]:
            eval_key = f"eval_ko_in/final_{metric}"
            if eval_key in eval_metrics:
                available_metrics.append(("eval_ko_in", metric, eval_metrics[eval_key]))

        # Out-of-distribution metrics
        for metric in ["hard_accuracy", "accuracy", "hard_loss", "loss"]:
            eval_key = f"eval_ko_out/final_{metric}"
            if eval_key in eval_metrics:
                available_metrics.append(("eval_ko_out", metric, eval_metrics[eval_key]))

    # Filter metrics to track based on configuration
    if track_metrics is not None:
        # Only track specified metrics
        metrics_to_track = []
        for source, metric, value in available_metrics:
            metric_key = f"{source}_{metric}"
            if metric_key in track_metrics:
                metrics_to_track.append((source, metric, value))
    else:
        # Track all available metrics (backward compatibility)
        metrics_to_track = available_metrics

    # Track and save best models for each specified metric
    for source, metric, value in metrics_to_track:
        metric_key = f"{source}_{metric}"

        # Check if this is a new best
        if best_model_tracker.update(metric_key, value, epoch):
            updates[metric_key] = {
                "value": value,
                "epoch": epoch,
                "previous_best": best_model_tracker.get_best_value(metric_key),
            }

            # Save best checkpoint
            save_best_checkpoint(
                checkpoint_path=checkpoint_path,
                is_best=True,
                save_best=save_best,
                model=model,
                optimizer=optimizer,
                metrics=metrics,
                epoch=epoch,
                best_metric=metric_key,
                current_metric_value=value,
                wandb_run=wandb_run,
            )

            log.info(f"New best {metric_key}: {value:.4f} at epoch {epoch}")

    return updates


def check_early_stopping(
    stop_accuracy_enabled: bool,
    epoch: int,
    stop_accuracy_min_epochs: int,
    early_stop_triggered: bool,
    stop_accuracy_metric: str,
    stop_accuracy_source: str,
    training_metrics: dict,
    current_eval_metrics: dict | None,
    stop_accuracy_threshold: float,
    first_threshold_epoch: int | None,
    epochs_above_threshold: int,
    stop_accuracy_patience: int,
    rng: jax.random.PRNGKey,
) -> tuple[bool, bool, int, int | None, dict | None, jax.random.PRNGKey]:
    """
    Check early stopping conditions and handle early stopping logic.

    Returns:
        Tuple of (should_break, early_stop_triggered, epochs_above_threshold,
                 first_threshold_epoch, updated_current_eval_metrics, updated_rng)
    """
    if not stop_accuracy_enabled or early_stop_triggered:
        return (
            False,
            early_stop_triggered,
            epochs_above_threshold,
            first_threshold_epoch,
            current_eval_metrics,
            rng,
        )

    # Get the accuracy value for early stopping
    try:
        stop_accuracy_value = get_metric_value(
            stop_accuracy_metric,
            stop_accuracy_source,
            training_metrics,
            current_eval_metrics,
        )
    except (ValueError, KeyError):
        if stop_accuracy_source == "eval" and current_eval_metrics is None:
            # Evaluation metrics not available, skip early stopping check this epoch
            stop_accuracy_value = None
        else:
            # Fallback to training metrics if eval not available
            stop_accuracy_value = get_metric_value(
                stop_accuracy_metric,
                "training",
                training_metrics,
                current_eval_metrics,
            )

    if stop_accuracy_value is None:
        return (
            False,
            early_stop_triggered,
            epochs_above_threshold,
            first_threshold_epoch,
            current_eval_metrics,
            rng,
        )

    if stop_accuracy_value >= stop_accuracy_threshold:
        if first_threshold_epoch is None:
            first_threshold_epoch = epoch
            log.info(
                f"Reached accuracy threshold {stop_accuracy_threshold:.4f} "
                f"({stop_accuracy_source}_{stop_accuracy_metric}={stop_accuracy_value:.4f}) "
                f"at epoch {epoch}. Starting patience countdown."
            )
        epochs_above_threshold += 1

        # Check if we should stop (only after minimum epochs requirement is met)
        if epochs_above_threshold >= stop_accuracy_patience and epoch >= stop_accuracy_min_epochs:
            early_stop_triggered = True
            log.info(
                f"Early stopping triggered! "
                f"Accuracy {stop_accuracy_source}_{stop_accuracy_metric}={stop_accuracy_value:.4f} "
                f"has been above threshold {stop_accuracy_threshold:.4f} "
                f"for {stop_accuracy_patience} epochs. "
                f"Stopping at epoch {epoch}."
            )

            return (
                True,
                early_stop_triggered,
                epochs_above_threshold,
                first_threshold_epoch,
                current_eval_metrics,
                rng,
            )
        elif epochs_above_threshold >= stop_accuracy_patience and epoch < stop_accuracy_min_epochs:
            # Would stop but waiting for minimum epochs
            pass
    else:
        # Reset counter if accuracy drops below threshold
        if epochs_above_threshold > 0:
            log.info("Accuracy dropped below threshold. Resetting early stopping counter.")
        epochs_above_threshold = 0
        first_threshold_epoch = None

    return (
        False,
        early_stop_triggered,
        epochs_above_threshold,
        first_threshold_epoch,
        current_eval_metrics,
        rng,
    )


# WandB integration functions
def _select_best_artifact(artifacts: list, prefer_metric: str | None = None):
    """
    Intelligently select the best artifact from multiple best model artifacts.

    Args:
        artifacts: List of WandB artifacts to choose from
        prefer_metric: Optional specific metric to prefer (e.g., "eval_ko_in_hard_accuracy")

    Returns:
        Selected artifact
    """
    if not artifacts:
        raise ValueError("No artifacts provided for selection")

    if len(artifacts) == 1:
        return artifacts[0]

    # Extract metric names from artifact names
    artifact_metrics = []
    for artifact in artifacts:
        # Extract metric from artifact name (e.g., "best_model_eval_ko_in_hard_accuracy" -> "eval_ko_in_hard_accuracy")
        name_parts = artifact.name.split("best_model_")
        if len(name_parts) > 1:
            metric_part = name_parts[1]
            # Remove version suffix if present (e.g., ":v0")
            if ":" in metric_part:
                metric_part = metric_part.split(":")[0]
            artifact_metrics.append((artifact, metric_part))
        else:
            # Fallback for artifacts that don't follow the new naming scheme
            artifact_metrics.append((artifact, "unknown"))

    # If a specific metric is preferred, try to find it
    if prefer_metric:
        for artifact, metric in artifact_metrics:
            if metric == prefer_metric:
                log.info(f"Found preferred metric '{prefer_metric}' in artifact: {artifact.name}")
                return artifact
        log.warning(f"Preferred metric '{prefer_metric}' not found, using intelligent selection")

    # Intelligent selection priority:
    # 1. Eval metrics over training metrics
    # 2. Hard metrics over soft metrics
    # 3. In-distribution over out-of-distribution
    # 4. Accuracy over loss

    def metric_priority(metric: str) -> tuple[int, int, int, int]:
        """Calculate priority score for a metric (lower is better)."""
        # Eval vs training (0 = eval, 1 = training)
        eval_score = 0 if metric.startswith("eval") else 1

        # Hard vs soft (0 = hard, 1 = soft)
        hard_score = 0 if "hard" in metric else 1

        # In vs out distribution (0 = in, 1 = out, 2 = neither)
        if "eval_ko_in" in metric:
            dist_score = 0
        elif "eval_ko_out" in metric:
            dist_score = 1
        else:
            dist_score = 2

        # Accuracy vs loss (0 = accuracy, 1 = loss)
        acc_score = 0 if "accuracy" in metric else 1

        return (eval_score, hard_score, dist_score, acc_score)

    # Sort artifacts by priority
    artifact_metrics.sort(key=lambda x: metric_priority(x[1]))

    selected_artifact, selected_metric = artifact_metrics[0]
    log.info(f"Selected artifact with metric '{selected_metric}' using intelligent priority")

    return selected_artifact
