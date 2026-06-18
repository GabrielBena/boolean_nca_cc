#!/usr/bin/env python3
"""
Main training script for Boolean Circuit NCA optimization using GNNs and Self-Attention.

This script trains models that learn to simulate the inner loop of backpropagation
when training boolean circuits, using either Graph Neural Networks or Self-Attention.
"""

import os

# Redirect JAX/XLA temp + compilation cache off the near-full root partition (95% used)
# onto /mnt/8TB_HDD (2.9T free). Avoids "No space left on device" during PTX writes.
# The per-user dir must be pre-created by an admin (we're not in the hddaccess group):
#   sudo mkdir -p /mnt/8TB_HDD/$USER && sudo chown -R $USER:$USER /mnt/8TB_HDD/$USER
_user_disk = os.path.join("/mnt/8TB_HDD", os.environ.get("USER", ""))
if os.access(_user_disk, os.W_OK):
    _tmpdir = os.path.join(_user_disk, "tmp")
    _jax_cache = os.path.join(_user_disk, "jax_cache")
    os.makedirs(_tmpdir, exist_ok=True)
    os.makedirs(_jax_cache, exist_ok=True)
    os.environ.setdefault("TMPDIR", _tmpdir)
    os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", _jax_cache)

# === Optional CPU-only run ===
# To run on CPU only, set the below environment variable before JAX import.
# You can set this via an environment variable or command line argument:
#   $ JAX_PLATFORM_NAME=cpu python train.py

# Or uncomment the following line to force CPU from script and create 8 virtual devices:
# os.environ["JAX_PLATFORM_NAME"] = "cpu"
# os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"

# Configure JAX/XLA GPU memory allocation BEFORE importing JAX.
# Default = BFC allocator (fast, but pools/fragments). On a tight card it can fail
# to find a CONTIGUOUS block for a large buffer requested AFTER a checkpoint load:
# this is why T32 RESUME OOM'd the 32GB V100 (a 13GiB contiguous request into a heap
# fragmented by the just-restored pool/optimizer) while the same config ran fine from
# a fresh start. See HANDOFF (2026-06-10).
#   Fix for tight-memory resumes: export XLA_PYTHON_CLIENT_ALLOCATOR=platform
#   (cudaMalloc, no fragmentation, ~1.5x slower) — or "cuda_async" (faster) — via the
#   env / HSM pre_script. NB: TF_GPU_ALLOCATOR is a TensorFlow var that JAX IGNORES;
#   the previous setting here was a no-op (the allocator stayed BFC, hence the OOM).
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "default")  # override -> "platform"/"cuda_async" for tight resumes

# GPU VISIBILITY
# Pin a specific GPU only for LOCAL dev. Under SLURM (or any launcher that sets
# CUDA_VISIBLE_DEVICES / a GPU cgroup), respect the allocation — hardcoding "3"
# here made cluster jobs silently fall back to CPU (allocated GPU is index 0).
if "SLURM_JOB_ID" not in os.environ and "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import logging

import hydra
import jax
import optax
import pandas as pd
from flax import nnx
from omegaconf import DictConfig, OmegaConf, open_dict

import wandb
from boolean_nca_cc import generate_layer_sizes
from boolean_nca_cc.circuits.model import gen_circuit
from boolean_nca_cc.circuits.tasks import get_task_data
from boolean_nca_cc.circuits.train import LossConfig
from boolean_nca_cc.circuits.viz import plot_wandb_stepwise_results
from boolean_nca_cc.training.checkpointing import (
    EarlyStopping,
    load_checkpoint_with_compatibility,
    save_checkpoint,
)
from boolean_nca_cc.training.demo_probe import demo_probe, format_demo_metrics
from boolean_nca_cc.training.eval_datasets import (
    create_unified_evaluation_datasets,
)
from boolean_nca_cc.training.evaluation import run_bp_scan
from boolean_nca_cc.training.pool.structural_perturbation import (
    compute_damage_params,
    compute_shuffle_params,
)
from boolean_nca_cc.training.train_loop import (
    run_unified_periodic_evaluation,
    train_model,
)
from boolean_nca_cc.training.utils import (
    cleanup_redundant_wandb_artifacts,
)
from boolean_nca_cc.utils.configured_graph_builder import (
    configure_build_graph,
    configured_build_graph,
    is_configured,
)
from boolean_nca_cc.utils.pool_stats import (
    calculate_expected_pool_updates,
    compute_pool_parameter,
)

# Configure logging
log = logging.getLogger(__name__)

if os.path.exists("/mnt/storage/gb21"):
    wandb_cache_dir = "/mnt/storage/gb21/wandb/cache"
    wandb_artifact_dir = "/mnt/storage/gb21/wandb/artifacts"
    os.environ["WANDB_CACHE_DIR"] = wandb_cache_dir
    os.environ["WANDB_ARTIFACT_DIR"] = wandb_artifact_dir
    os.makedirs(wandb_cache_dir, exist_ok=True)
    os.makedirs(wandb_artifact_dir, exist_ok=True)


def extract_track_metrics_config(cfg) -> list[str] | None:
    """
    Extract track_metrics configuration from config using filter-based approach.

    Filters (omit or empty list = include all):
    - metric: [hard_accuracy, accuracy, hard_loss, loss]
    - data: [test, train]
    - wiring: [in, out]
    - damaged: [true, false]

    Returns:
        List of metrics to track, or None to auto-detect all active evaluation metrics.
    """
    track_cfg = cfg.checkpoint.get("track_best_metrics", {})
    if not track_cfg.get("enabled", False):
        return None

    # Get filters (empty list = include all)
    metrics = track_cfg.get("metric", []) or ["hard_accuracy", "accuracy", "hard_loss", "loss"]
    data_splits = track_cfg.get("data", []) or ["test", "train"]
    wirings = track_cfg.get("wiring", []) or ["in", "out"]
    damaged_opts = track_cfg.get("damaged", [])
    if not damaged_opts:
        damaged_prefixes = ["", "damaged_"]
    else:
        damaged_prefixes = ["damaged_" if d else "" for d in damaged_opts]

    # Generate all metric keys from filter combinations
    result = []
    for dp in damaged_prefixes:
        for w in wirings:
            for d in data_splits:
                for m in metrics:
                    result.append(f"eval_{dp}{w}_{d}_{m}")

    if result:
        log.info(f"Tracking metrics: {' | '.join(result)}")
    else:
        log.info("No metrics to track")

    return result if result else None


def _parse_n_rounds_schedule(raw):
    """Normalize the inner-loop n_rounds curriculum config to a list of [frac, N] (or None).

    Robust to how Hydra/hsm pass it: ``None``/empty → None; a string (sweep override, e.g.
    ``"[[0.0,8],[0.5,32]]"``) → parsed via ast.literal_eval; an OmegaConf list → to_container.
    """
    if raw is None:
        return None
    if isinstance(raw, str):
        raw = raw.strip()
        if not raw or raw.lower() == "null":
            return None
        import ast

        return ast.literal_eval(raw)
    return OmegaConf.to_container(raw, resolve=True)


def _resolve_inner_loop_regime(training_cfg):
    """Map the regime config to the current vocabulary.

    Current: "batch" (legacy/demo via run_model_scan_with_loss) | "unified" (the online↔batch
    continuum via run_inner_loop). The legacy value "online" is folded into "unified" (with
    data_mode defaulting to "stream" at the call site), preserving its former behavior.
    """
    regime = str(training_cfg.get("inner_loop_regime", "batch")).lower()
    return "unified" if regime == "online" else regime


def create_and_save_final_results(
    cfg,
    model_results,
    eval_results,
    layer_sizes,
    output_dir,
    wandb_run=None,
    total_params=None,
    demo_metrics=None,
):
    """
    Create comprehensive final results dictionary, save as CSV, and log results.

    Args:
        cfg: Configuration object
        model_results: Results from model training
        eval_results: Results from evaluation (with final_metrics dict)
        layer_sizes: Circuit layer sizes
        output_dir: Directory to save results
        wandb_run: WandB run object (optional)
        total_params: Total model parameters (optional)

    Returns:
        dict: Final results dictionary
    """
    # Collect comprehensive final results
    final_results = {
        # Model and training configuration
        "model_type": cfg.model.type,
        "wiring_mode": cfg.training.wiring_mode,
        "loss_cfg": dict(cfg.loss),
        "learning_rate": cfg.training.learning_rate,
        "epochs_completed": len(model_results["losses"]),
        "total_epochs_planned": cfg.training.epochs or 2**cfg.training.epochs_power_of_2,
        "early_stopped": model_results.get("early_stopped", False),
        "early_stop_epoch": model_results.get("early_stop_epoch", None),
        "first_threshold_epoch": model_results.get("first_threshold_epoch", None),
        # Meta-learning metrics (final training values)
        "meta_loss": model_results["losses"][-1],
        "meta_hard_loss": model_results["hard_losses"][-1],
        "meta_accuracy": model_results["accuracies"][-1],
        "meta_hard_accuracy": model_results["hard_accuracies"][-1],
        # Best model performance
        "best_metric": model_results.get("best_metric", None),
        "best_metric_value": model_results.get("best_metric_value", None),
        # Circuit and model architecture
        "input_bits": cfg.circuit.input_bits,
        "output_bits": cfg.circuit.output_bits,
        "num_layers": len(layer_sizes),
        "total_nodes": sum(size[0] for size in layer_sizes),
        "circuit_hidden_dim": cfg.model.circuit_hidden_dim,
        "attention_dim": cfg.model.get("attention_dim", None),
        "mlp_dim_multiplier": cfg.model.get("mlp_dim_multiplier", None),
        "mlp_dim": cfg.model.get("mlp_dim", None),
        "total_parameters": total_params,
        "message_steps": cfg.training.n_message_steps,
        "eval_target_batch_size_in": cfg.eval.batch_size_in,
        "eval_target_batch_size_out": cfg.eval.batch_size_out,
        "pool_size": cfg.pool.size,
        "pool_initial_diversity": cfg.pool.initial_diversity,
    }

    # Add evaluation metrics from all scenarios (wiring x data)
    if eval_results:
        final_metrics = eval_results.get("final_metrics", {})

        # Extract metrics for all evaluation scenarios
        for full_key, metrics in final_metrics.items():
            if metrics is None:
                continue
            # full_key is e.g. "in_test", "out_train", "damaged_in_test"
            prefix = f"eval_{full_key}"
            for metric_name in [
                "final_loss",
                "final_hard_loss",
                "final_accuracy",
                "final_hard_accuracy",
            ]:
                wandb_key = f"{prefix}/{metric_name}"
                result_key = f"{full_key}_{metric_name}".replace("/", "_")
                final_results[result_key] = metrics.get(wandb_key)

        # Add datasets info
        datasets_info = eval_results.get("datasets_info", {})
        if datasets_info:
            final_results.update(
                {
                    "eval_in_actual_batch_size": datasets_info.get("in_actual_batch_size"),
                    "eval_out_actual_batch_size": datasets_info.get("out_actual_batch_size"),
                    "eval_in_used_chunking": datasets_info.get("in_used_chunking", False),
                    "eval_out_used_chunking": datasets_info.get("out_used_chunking", False),
                    "eval_training_wiring_mode": datasets_info.get("training_wiring_mode"),
                    "eval_training_initial_diversity": datasets_info.get(
                        "training_initial_diversity"
                    ),
                }
            )

    # Demo-probe metrics (batched post-training probe — see training/demo_probe.py)
    if demo_metrics:
        final_results.update(demo_metrics)

    # Save to CSV
    results_df = pd.DataFrame([final_results])
    results_csv_path = os.path.join(output_dir, "final_results.csv")
    results_df.to_csv(results_csv_path, index=False)
    log.info(f"Final results saved to: {results_csv_path}")

    # Display summary
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    log.info("Final Results Summary:\n" + str(results_df.T))

    # Log to wandb
    if cfg.wandb.enabled and wandb_run:
        wandb.log({"final_results_table": wandb.Table(dataframe=results_df)})
        wandb.log({f"final/{k}": v for k, v in final_results.items() if v is not None})

    # Console summary
    log.info("Training complete. Final results:")
    log.info(
        f"  Meta: Loss={model_results['losses'][-1]:.4f}, "
        f"Hard Acc={model_results['hard_accuracies'][-1]:.4f}"
    )

    if eval_results:
        final_metrics = eval_results.get("final_metrics", {})
        for key in ["in_test", "out_test", "in_train", "out_train"]:
            m = final_metrics.get(key)
            if m:
                prefix = f"eval_{key}"
                desc = key.replace("_", " ").upper()
                log.info(
                    f"  {desc}: Loss={m.get(f'{prefix}/final_loss', 0):.4f}, "
                    f"Hard Acc={m.get(f'{prefix}/final_hard_accuracy', 0):.4f}"
                )

    if cfg.checkpoint.save_best and "best_metric_value" in model_results:
        log.info(
            f"  Best {model_results.get('best_metric', 'metric')}: "
            f"{model_results['best_metric_value']:.4f}"
        )

    # Display early stopping and threshold information if applicable
    first_threshold_epoch = model_results.get("first_threshold_epoch", None)
    if first_threshold_epoch is not None:
        log.info(
            f"  Accuracy threshold {cfg.stop_accuracy.threshold} first reached at epoch {first_threshold_epoch}"
        )

    if model_results.get("early_stopped", False):
        log.info(
            f"  Training stopped early at epoch {model_results.get('early_stop_epoch', 'unknown')}"
        )
        log.info(
            f"  Early stopping was triggered by {cfg.stop_accuracy.source}_{cfg.stop_accuracy.metric} >= {cfg.stop_accuracy.threshold}"
        )
    else:
        log.info(
            f"  Training completed all {cfg.training.epochs or 2**cfg.training.epochs_power_of_2} epochs"
        )

    return final_results


def process_pool_configuration(cfg):
    """
    Process pool configuration to automatically compute missing parameters based on expected updates.

    Args:
        cfg: Configuration object with pool settings

    Returns:
        Updated configuration with computed parameters

    Raises:
        ValueError: If configuration is underspecified or invalid
    """

    # Per-visit NCA step count feeding the pool curriculum (expected_updates / reset).
    # For the UNIFIED inner loop a pool visit runs n_rounds × steps_per_round message steps,
    # NOT the legacy batch n_message_steps — so the 256-expected-updates target (and the
    # solved reset_interval) stay correct at any (n_rounds, steps_per_round). Using the
    # legacy n_message_steps here mis-tuned the reset whenever n_rounds·steps_per_round ≠
    # n_message_steps (e.g. the long-horizon W=1 cells). (2026-06-17)
    _regime = str(cfg.training.get("inner_loop_regime", "batch")).lower()
    if _regime in ("unified", "online"):
        _base_n_rounds = int(cfg.training.get("n_rounds", 64))
        _sched = cfg.training.get("n_rounds_schedule", None)
        if _sched:
            try:
                _base_n_rounds = max(_base_n_rounds, max(int(s[1]) for s in _sched))
            except Exception:
                pass
        n_message_steps_effective = _base_n_rounds * int(cfg.training.get("steps_per_round", 1))
    elif cfg.training.random_loss_step:
        n_message_steps_effective = (
            cfg.training.n_message_steps + cfg.training.random_loss_step_min
        ) // 2
    else:
        n_message_steps_effective = cfg.training.n_message_steps

    if cfg.pool.expected_updates is None:
        # No automatic computation requested, validate current config
        log.info("Using explicit pool configuration (no expected_updates specified)")
        return cfg, n_message_steps_effective

    # Check which parameters need to be computed
    pool_params = {
        "pool_size": cfg.pool.size,
        "batch_size": cfg.training.meta_batch_size,
        "n_message_steps": n_message_steps_effective,
        "reset_interval": cfg.pool.reset_interval,
        "reset_fraction": cfg.pool.reset_fraction,
    }

    # Count None/null parameters
    none_params = [key for key, value in pool_params.items() if value is None]

    if len(none_params) == 0:
        # All parameters specified, verify the configuration matches expected updates
        log.info(
            f"Verifying pool configuration matches target expected updates: {cfg.pool.expected_updates}"
        )
        stats = calculate_expected_pool_updates(**pool_params)
        actual_updates = stats.expected_updates

        if abs(actual_updates - cfg.pool.expected_updates) > 0.1:
            log.warning(
                f"Configuration mismatch: Expected {cfg.pool.expected_updates:.2f} updates, "
                f"but configuration yields {actual_updates:.2f} updates"
            )
        else:
            log.info(f"Configuration verified: {actual_updates:.2f} expected updates per circuit")

        return cfg, n_message_steps_effective

    elif len(none_params) == 1:
        # Exactly one parameter to compute
        param_to_solve = none_params[0]
        log.info(
            f"Computing {param_to_solve} for target expected updates: {cfg.pool.expected_updates}"
        )

        # Prepare arguments for computation
        compute_kwargs = {k: v for k, v in pool_params.items() if v is not None}

        try:
            computed_value = compute_pool_parameter(
                target_expected_updates=cfg.pool.expected_updates,
                solve_for=param_to_solve,
                **compute_kwargs,
            )

            # Update configuration with computed value
            with open_dict(cfg):
                if param_to_solve == "pool_size":
                    cfg.pool.size = int(computed_value)
                elif param_to_solve == "batch_size":
                    cfg.training.meta_batch_size = int(computed_value)
                elif param_to_solve == "n_message_steps":
                    cfg.training.n_message_steps = int(computed_value)
                elif param_to_solve == "reset_interval":
                    cfg.pool.reset_interval = int(computed_value)
                elif param_to_solve == "reset_fraction":
                    cfg.pool.reset_fraction = float(computed_value)

            log.info(f"Computed {param_to_solve} = {computed_value:.4f}")

            # Verify the computation
            updated_params = {
                "pool_size": cfg.pool.size,
                "batch_size": cfg.training.meta_batch_size,
                "n_message_steps": n_message_steps_effective,
                "reset_interval": cfg.pool.reset_interval,
                "reset_fraction": cfg.pool.reset_fraction,
            }
            stats = calculate_expected_pool_updates(**updated_params)
            log.info(f"Verification: {stats.expected_updates:.2f} expected updates per circuit")

        except ValueError as e:
            raise ValueError(
                f"Cannot compute {param_to_solve} for target {cfg.pool.expected_updates} updates: {e}"
            )

    else:
        # Multiple parameters are None - underspecified
        raise ValueError(
            f"Pool configuration is underspecified. Cannot compute multiple parameters: {none_params}. "
            f"Please specify all but one of: pool.size, training.meta_batch_size, "
            f"training.n_message_steps, pool.reset_interval, pool.reset_fraction"
        )

    return cfg, n_message_steps_effective


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Main training function using Hydra for configuration.

    Args:
        cfg: Hydra configuration object
    """
    # Print configuration
    log.info(OmegaConf.to_yaml(cfg))

    # Process pool configuration for automatic parameter computation
    cfg, effective_n_message_steps = process_pool_configuration(cfg)

    # Log final pool configuration and expected updates
    pool_params = {
        "pool_size": cfg.pool.size,
        "batch_size": cfg.training.meta_batch_size,
        "n_message_steps": effective_n_message_steps,
        "reset_interval": cfg.pool.reset_interval,
        "reset_fraction": cfg.pool.reset_fraction,
    }
    stats = calculate_expected_pool_updates(**pool_params)

    # Configure global build_graph function with settings from config
    log.info(
        f"Configuring graph builder: neighboring_connections={cfg.graph.neighboring_connections}, bidirectional_edges={cfg.graph.bidirectional_edges}"
    )
    configure_build_graph(
        neighboring_connections=cfg.graph.neighboring_connections,
        bidirectional_edges=cfg.graph.bidirectional_edges,
        use_dist_pe=cfg.graph.get("use_dist_pe", False),
        use_rwse=cfg.graph.get("use_rwse", False),
        rwse_k=cfg.graph.get("rwse_k", 8),
    )

    # Verify configuration was applied
    if is_configured():
        log.info("✅ Graph builder successfully configured with explicit settings")
    else:
        log.warning("⚠️ Graph builder configuration failed, will use defaults")

    log.info("Final Pool Configuration:")
    log.info(f"  Pool Size: {cfg.pool.size}")
    log.info(f"  Batch Size: {cfg.training.meta_batch_size}")
    log.info(f"  Message Steps: {effective_n_message_steps}")
    log.info(f"  Reset Interval: {cfg.pool.reset_interval}")
    log.info(f"  Reset Fraction: {cfg.pool.reset_fraction:.4f}")
    log.info(f"  Expected Updates per Circuit: {stats.expected_updates:.2f}")
    log.info(f"  Selection Probability: {stats.selection_probability:.4f}")
    log.info(f"  Expected Lifetime: {stats.expected_lifetime_epochs:.1f} epochs")

    # Log damage configuration if enabled
    if cfg.damage.enabled:
        target_frac = cfg.damage.get("target_damage_fraction", 0.0)
        log.info(f"Damage Configuration: target {target_frac:.1%} of hidden gates")
        log.info("  (p_fault and knockouts will be auto-computed before training)")
    else:
        log.info("Damage Configuration: DISABLED")

    # Set random seed
    train_key = jax.random.PRNGKey(cfg.seed)

    if cfg.eval_seed is not None:
        eval_key = jax.random.PRNGKey(cfg.eval_seed)
    else:
        eval_key = jax.random.fold_in(train_key, 1)

    # Create output directory
    if cfg.output.dir is not None:
        output_dir = cfg.output.dir
        os.makedirs(output_dir, exist_ok=True)
    else:
        output_dir = os.getcwd()

    # HSM resumable-chain CONTRACT [CONTRACT 1: consume the resume pointer]
    # (HPC-Sweep-Manager issue #12). In `hsm sweep run --resumable` mode HSM exports
    # HSM_RESUME_FROM (non-empty => resume; empty on chunk 1) and HSM_RESUME_TO (the
    # persistent dir holding the checkpoint). We resolve them into
    # cfg.training.resume_from so the rest of the pipeline (wandb peek + train_model)
    # is untouched. Env-only by default; an explicit cfg.training.resume_from wins.
    _hsm_from = os.environ.get("HSM_RESUME_FROM")
    _hsm_to = os.environ.get("HSM_RESUME_TO")
    if not cfg.training.get("resume_from", None) and _hsm_from:
        _cand = None
        if os.path.isfile(_hsm_from):
            _cand = _hsm_from  # FROM points directly at the checkpoint file
        else:
            _base = _hsm_to or (_hsm_from if os.path.isdir(_hsm_from) else None)
            if _base:
                _cand = os.path.join(_base, "latest_checkpoint.pkl")
        if _cand and os.path.exists(_cand):
            with open_dict(cfg):
                cfg.training.resume_from = _cand
            log.info(f"HSM resumable chain: resuming from {_cand}")
        else:
            log.info(
                f"HSM_RESUME_FROM set ({_hsm_from!r}) but no checkpoint found yet "
                "— starting fresh (treating as chunk 1)."
            )

    # Initialize wandb if enabled
    wandb_run = None
    if cfg.wandb.enabled:
        # ── WandB resume continuity ───────────────────────────────────────
        # If this is a mid-training resume, peek the checkpoint for the wandb
        # run id saved by the original run and pass it as TOP-LEVEL init args
        # (id=, resume="must") so the continued run logs to the SAME wandb run
        # with a continuous curve — instead of spawning a fresh run. The id
        # lives in the checkpoint's resume block (resume.wandb_run_id); we read
        # it here, BEFORE wandb.init, because train_model only discovers it much
        # later (after it loads the checkpoint). See wandb resuming docs.
        resume_from = cfg.training.get("resume_from", None)
        resume_wandb_id = None
        if resume_from is not None:
            try:
                _peek = load_checkpoint_with_compatibility(resume_from)
                resume_wandb_id = (_peek.get("resume") or {}).get("wandb_run_id")
                del _peek
            except Exception as e:  # noqa: BLE001
                log.warning(
                    f"Could not peek wandb_run_id from resume checkpoint {resume_from!r}: {e}. "
                    "wandb will start a fresh run for the resumed half."
                )
            if resume_wandb_id:
                log.info(f"Resuming wandb run id={resume_wandb_id} (resume='must').")
            else:
                log.warning(
                    "resume_from is set but no wandb_run_id was found in the checkpoint; "
                    "wandb will start a fresh run for the resumed half."
                )

        init_kwargs = dict(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            name=cfg.wandb.run_name,
            dir=output_dir,
            config=OmegaConf.to_container(cfg, resolve=True),
            group=cfg.wandb.group,
            reinit="finish_previous",
        )
        if resume_wandb_id:
            # Top-level resume args (NOT inside config=) — this is what actually
            # makes wandb continue the existing run rather than create a new one.
            init_kwargs["id"] = resume_wandb_id
            init_kwargs["resume"] = "must"

        wandb.init(**init_kwargs)
        wandb_run = wandb

        # X-axis continuity across the kill+resume seam. wandb's internal _step
        # counter restarts to 0 on a resumed run, which would draw the two halves
        # on top of each other. Pin every metric to the monotonic "epoch" field
        # instead, so the web UI overlays the two halves into one continuous
        # curve regardless of _step. Every .log call includes "epoch".
        try:
            wandb.define_metric("epoch")
            wandb.define_metric("*", step_metric="epoch")
        except Exception as e:  # noqa: BLE001
            log.warning(f"Could not define wandb step metric 'epoch': {e}")

    log.info(f"Output directory: {output_dir}")

    # Generate circuit layer sizes
    input_n, output_n = cfg.circuit.input_bits, cfg.circuit.output_bits
    arity = cfg.circuit.arity

    if cfg.circuit.layer_sizes is None:
        layer_sizes = generate_layer_sizes(
            input_n,
            output_n,
            arity,
            layer_n=cfg.circuit.num_layers,
            width_factor=cfg.circuit.width_factor,
        )
        with open_dict(cfg):
            cfg.circuit.layer_sizes = layer_sizes
    else:
        layer_sizes = cfg.circuit.layer_sizes

    # Generate dummy circuit
    wires_key, logits_key = jax.random.split(jax.random.PRNGKey(42))
    wires, logits = gen_circuit(
        wires_key,
        logits_key,
        cfg.circuit.layer_sizes,
        arity=cfg.circuit.arity,
        noise_scale=cfg.pool.noise_scale,
    )

    # Generate dummy graph using globally configured function
    graph = configured_build_graph(
        wires=wires,
        logits=logits,
        input_n=input_n,
        arity=arity,
        circuit_hidden_dim=cfg.model.circuit_hidden_dim,
    )
    n_nodes = int(graph.n_node[0])

    log.info(f"Circuit layer sizes: {layer_sizes}")
    log.info(f"Number of nodes: {n_nodes}")
    if cfg.wandb.enabled:
        wandb.config.update({"layer_sizes": layer_sizes, "n_nodes": n_nodes})

    # Get task data
    case_n = 1 << input_n

    if cfg.training.test_num is not None:
        if isinstance(cfg.training.test_num, float):
            assert cfg.training.test_num >= 0 and cfg.training.test_num <= 1, (
                "test_num must be a float between 0 and 1, or an integer"
            )
            test_ratio = cfg.training.test_num
        else:
            assert isinstance(cfg.training.test_num, int), "test_num must be a float or an integer"
            assert cfg.training.test_num > 0, (
                "test_num must be greater than 0 and less than the total number of cases"
            )
            if cfg.training.test_num > case_n:
                log.warning(
                    f"test_num is greater than the total number of cases, setting test_ratio to 1"
                )
                test_ratio = 1
            else:
                test_ratio = cfg.training.test_num / case_n

        # Max 1/5 of the data for testing
        test_ratio = min(test_ratio, 0.2)
        print(f"Test ratio: {test_ratio}")
    else:
        test_ratio = None

    # ── Dispatch on cfg.tasks.type ──────────────────────────────────────
    # "fixed":   single library task shared across the pool (legacy path).
    # "sampler": per-circuit task sampling — data_dict carries a placeholder,
    #            actual targets come from the pool's y_task at every step.
    tasks_type = cfg.tasks.type
    if tasks_type == "fixed":
        (x_train, y_train), (x_test, y_test), (x_total, y_total) = get_task_data(
            cfg.tasks.name,
            case_n,
            input_bits=input_n,
            output_bits=output_n,
            text=cfg.tasks.get("text", None),
            train_test_split=test_ratio is not None,
            test_ratio=test_ratio,
            seed=eval_key,
        )
    elif tasks_type == "sampler":
        # Placeholder data — train loop ignores these when task_sampler_cfg
        # is set; the per-circuit eval datasets carry the real targets.
        # Setting x_test/y_test to None disables the "test" data_scenario in
        # ``run_unified_periodic_evaluation`` so eval routes through the
        # per-circuit "task" scenario instead.
        from boolean_nca_cc.tasks import build_task_x

        x_total = build_task_x(input_n)
        y_total = jax.numpy.zeros((x_total.shape[0], output_n), dtype=jax.numpy.float32)
        x_train, y_train = x_total, y_total
        x_test, y_test = None, None
    else:
        raise ValueError(f"cfg.tasks.type must be 'fixed' or 'sampler', got {tasks_type!r}")

    data_dict = {
        "x_train": x_train,
        "y_train": y_train,
        "x_test": x_test,
        "y_test": y_test,
        "x_total": x_total,
        "y_total": y_total,
    }

    # Compute data fraction
    n_train = x_train.shape[0]
    print(f"Train data shape : {x_train.shape}")

    data_fraction = (
        min(cfg.training.data_per_batch / n_train, 1.0)
        if cfg.training.data_per_batch is not None
        else 1.0
    )

    log.info(f"Data Fraction: {data_fraction:.4f}")
    if test_ratio is not None:
        if x_test is not None:
            log.info(
                f"Test Ratio: {test_ratio:.4f} | Sizes = {x_train.shape[0]}, {x_test.shape[0]}, {x_total.shape[0]}"
            )
        else:
            log.info(
                f"Test Ratio: {test_ratio:.4f} | Sizes = {x_train.shape[0]}, {x_total.shape[0]}"
            )

    # Initialize model
    train_key, init_key = jax.random.split(train_key)

    # Common overrides for hydra.instantiate
    # These are values computed in train.py or essential for all models
    instantiate_overrides = {"arity": arity, "rngs": nnx.Rngs(params=init_key)}

    # Specific overrides based on model type, which should still be in the YAML.
    # Alternatively, we could inspect cfg.model._target_ if 'type' was removed.
    if cfg.model.type in ["self_attention", "perceiver_attention"]:
        instantiate_overrides["n_node"] = n_nodes

    # Perceiver-specific: inject the input / output layer sizes from the
    # eagerly-computed layer_sizes so the cross-attention slicing optimisation
    # has Python-static K values. ``layer_sizes`` is
    # ``[(input_n, 1), (gate_layer_1_n, gs), ..., (output_n, 1)]`` so the
    # input layer is index 0 and the output (last gate) layer is the final entry.
    if cfg.model.type == "perceiver_attention":
        instantiate_overrides["input_layer_size"] = int(layer_sizes[0][0])
        instantiate_overrides["output_layer_size"] = int(layer_sizes[-1][0])
    # CircuitGNN does not require n_node in its constructor based on original setup.

    # Instantiate the model using Hydra
    try:
        model = hydra.utils.instantiate(cfg.model, **instantiate_overrides)
    except Exception as e:
        log.error(
            f"Error instantiating model {cfg.model._target_ if '_target_' in cfg.model else cfg.model.type}: {e}"
        )
        raise

    # Fail fast on producer/consumer PE mismatch.
    # The model reads nodes["dist_pe"] / nodes["rwse"] only when its own flag is on;
    # those features exist only if build_graph was configured to emit them. Both sides
    # interpolate from cfg.graph.*, so a mismatch means an inconsistent config (or an
    # ad-hoc override) — catch it here instead of as a cryptic KeyError deep in the scan.
    from boolean_nca_cc.utils.configured_graph_builder import get_configured_pe_flags

    _pe_flags = get_configured_pe_flags()
    for _flag, _key in (("use_dist_pe", "dist_pe"), ("use_rwse", "rwse")):
        _model_wants = bool(getattr(model, _flag, False))
        _graph_emits = bool(_pe_flags[_flag])
        if _model_wants and not _graph_emits:
            raise ValueError(
                f"Model has {_flag}=True but build_graph is configured with "
                f"{_flag}=False, so nodes['{_key}'] will never be emitted. "
                f"Set graph.{_flag}=true (cfg.graph is the single source of truth; "
                f"cfg.model.{_flag} interpolates from it)."
            )
        if _graph_emits and not _model_wants:
            # Not fatal — graph carries an unused feature — but always a config smell.
            log.warning(
                f"build_graph emits nodes['{_key}'] (graph.{_flag}=true) but the model "
                f"has {_flag}=False, so it is computed and ignored. Disable graph.{_flag} "
                f"to save the compute, or enable it on the model."
            )

    # Count and log model parameters with detailed breakdown
    log.info(f"Model instantiated: {cfg.model.type}")

    params = nnx.state(model, nnx.Param)
    module_param_counts = {}

    # Count parameters per module
    for module_name, module_params in params.items():
        num_params = jax.tree.reduce(lambda x, y: x + y.size, module_params, 0)
        module_param_counts[module_name] = num_params
        log.info(f"Number of params in {module_name}: {num_params:,}")

    # Calculate total parameters
    total_params = jax.tree.reduce(lambda x, y: x + y.size, params, 0)
    log.info(f"Total number of params: {total_params:,}")

    # Log to wandb if enabled
    if cfg.wandb.enabled:
        wandb_metrics = {
            "model/parameter_count": total_params,
        }
        # Log per-module counts as well
        for module_name, count in module_param_counts.items():
            wandb_metrics[f"model/params_{module_name}"] = count

        wandb.log(wandb_metrics)

    # Prepare checkpoint directory
    if cfg.checkpoint.enabled:
        checkpoint_dir = os.path.join(output_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        if cfg.checkpoint.save_on_mnt_storage:
            checkpoint_dir = checkpoint_dir.replace("/home/", "/mnt/storage/")
    else:
        checkpoint_dir = None

    # Get track_metrics configuration for training
    track_metrics = extract_track_metrics_config(cfg)

    # Compute all damage parameters from target_damage_fraction
    damage_params = compute_damage_params(cfg, layer_sizes, log)

    # Compute probabilistic wire-shuffle parameters from target_shuffles_per_lifecycle
    shuffle_params = compute_shuffle_params(cfg, log)

    # Resolve task-sampler configs (per-circuit meta-learning). cfg.tasks.type
    # is the discriminator: ``fixed`` → legacy global-task path (sampler args
    # all None); ``sampler`` → per-circuit-task plumbing in train/eval.
    if cfg.tasks.type == "fixed":
        task_sampler_cfg_train = None
        task_sampler_cfg_eval_in = None
        task_sampler_cfg_eval_ood = None
    elif cfg.tasks.type == "sampler":
        task_sampler_cfg_train = OmegaConf.to_container(cfg.tasks.train, resolve=True)
        # In-distribution eval reuses the training sampler (held-out RNG seed
        # is handled by create_unified_evaluation_datasets).
        task_sampler_cfg_eval_in = task_sampler_cfg_train
        task_sampler_cfg_eval_ood = (
            OmegaConf.to_container(cfg.tasks.eval_ood, resolve=True)
            if cfg.tasks.get("eval_ood") is not None
            else None
        )
        log.info(
            f"Per-circuit meta-learning tasks enabled "
            f"(train={task_sampler_cfg_train['name']}, "
            f"ood={None if task_sampler_cfg_eval_ood is None else task_sampler_cfg_eval_ood['name']})"
        )
    else:
        raise ValueError(f"cfg.tasks.type must be 'fixed' or 'sampler', got {cfg.tasks.type!r}")

    # Create evaluation datasets
    eval_datasets = create_unified_evaluation_datasets(
        eval_key=eval_key,
        training_wiring_mode=cfg.training.wiring_mode,
        training_initial_diversity=cfg.pool.initial_diversity,
        layer_sizes=layer_sizes,
        arity=cfg.circuit.arity,
        eval_batch_size_in=cfg.eval.batch_size_in
        if cfg.eval.batch_size_in is not None
        else cfg.pool.initial_diversity,
        eval_batch_size_out=cfg.eval.batch_size_out
        if cfg.eval.batch_size_out is not None
        else cfg.training.meta_batch_size,
        do_ood_evaluation=cfg.eval.do_ood_evaluation
        if cfg.eval.do_ood_evaluation is not None
        else cfg.training.wiring_mode == "random",
        pool_noise_scale=cfg.pool.noise_scale,
        init_logits=cfg.circuit.get("init_logits", "soft_wires"),
        random_init_scale=float(cfg.circuit.get("random_init_scale", 1.0)),
        task_sampler_cfg_in=task_sampler_cfg_eval_in,
        task_sampler_cfg_ood=task_sampler_cfg_eval_ood,
    )

    log.info(eval_datasets.get_summary())

    # Run backpropagation training for comparison if enabled
    bp_results = None
    if cfg.backprop.enabled:
        opt = optax.adamw(
            cfg.backprop.learning_rate,
            cfg.backprop.beta1,
            cfg.backprop.beta2,
            weight_decay=cfg.backprop.weight_decay,
        )

        loss_cfg = LossConfig.from_dict(dict(cfg.loss))
        damage_modes = (
            ["none", "probabilistic", "deterministic"] if cfg.damage.enabled else ["none"]
        )
        print(
            f"Running backpropagation for {cfg.eval.inner_steps} steps on {damage_modes} damage modes"
        )
        n_damage_steps = damage_params["n_damage_steps"]
        if n_damage_steps > 0:
            damage_steps = jax.numpy.linspace(
                0,
                cfg.eval.inner_steps,
                n_damage_steps + 1,
                endpoint=False,
            ).astype(int)[1:]
        else:
            damage_steps = None
        bp_results = {
            damage_mode: run_bp_scan(
                # Engine
                opt=opt,
                batch_params=eval_datasets.in_distribution_logits
                if eval_datasets.in_distribution_logits is not None
                else eval_datasets.out_of_distribution_logits,
                batch_wires=eval_datasets.in_distribution_wires
                if eval_datasets.in_distribution_wires is not None
                else eval_datasets.out_of_distribution_wires,
                # Data
                x_data=x_train,
                y_data=y_train,
                x_test=x_test,
                y_test=y_test,
                # Configuration
                n_steps=cfg.eval.inner_steps,
                layer_sizes=layer_sizes,
                input_n=input_n,
                arity=arity,
                loss_cfg=loss_cfg,
                # Damage
                damage_mode=damage_mode,
                permanent=cfg.damage.permanent,
                p_fault=damage_params["p_fault_eval"] if damage_mode == "probabilistic" else None,
                p_fault_onset_step=0,
                damage_steps=damage_steps,
                knockouts_per_event=damage_params["knockouts_per_event"],
                faulty_value=damage_params["faulty_logit_value"],
            )
            for damage_mode in damage_modes
        }

        if cfg.wandb.enabled:
            for damage_mode, (_, result) in bp_results.items():
                key_prefix = f"BP_results/{damage_mode}"
                for metric in ["loss", "hard_loss", "accuracy", "hard_accuracy"]:
                    wandb.log({f"{key_prefix}/{metric}": result[metric][-1]})

                if cfg.backprop.log_stepwise:
                    import matplotlib.pyplot as plt

                    fig = plot_wandb_stepwise_results(
                        result,
                        title=f"BP Results: {damage_mode}",
                    )
                    wandb.log({f"stepwise/{key_prefix}": wandb.Image(fig)})
                    plt.close(fig)

    # Train model
    log.info(f"Starting {cfg.model.type.upper()} training")
    model_results = train_model(
        # ── Data ────────────────────────────────────────────────────────
        data_dict=data_dict,
        data_fraction=data_fraction,
        eval_datasets=eval_datasets,
        # ── Initialization ──────────────────────────────────────────────
        train_key=train_key,
        eval_key=eval_key,
        init_model=model,
        # ── Model architecture ──────────────────────────────────────────
        layer_sizes=layer_sizes,
        arity=arity,
        circuit_hidden_dim=cfg.model.circuit_hidden_dim,
        # ── Training: core loop ─────────────────────────────────────────
        epochs=cfg.training.epochs or 2**cfg.training.epochs_power_of_2,
        n_message_steps=cfg.training.n_message_steps,
        use_scan=cfg.training.use_scan,
        gradient_checkpointing=cfg.training.gradient_checkpointing,
        # ── Training: optimization ──────────────────────────────────────
        learning_rate=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
        grad_clip_norm=cfg.training.grad_clip_norm,
        lr_scheduler=cfg.training.lr_scheduler,
        lr_scheduler_params=cfg.training.lr_scheduler_params,
        # ── Training: loss ──────────────────────────────────────────────
        loss_cfg=LossConfig.from_dict(dict(cfg.loss)),
        random_loss_step=cfg.training.random_loss_step,
        random_loss_step_min=cfg.training.random_loss_step_min,
        use_beta_loss_step=cfg.training.use_beta_loss_step,
        loss_step_mode=cfg.training.get("loss_step_mode", None),
        loss_tail_fraction=float(cfg.training.get("loss_tail_fraction", 0.5)),
        consistency_weight=float(cfg.training.get("consistency_weight", 0.0)),
        pool_return_step=str(cfg.training.get("pool_return_step", "final")),
        # ── Wiring & batching ───────────────────────────────────────────
        wiring_mode=cfg.training.wiring_mode,
        meta_batch_size=cfg.training.meta_batch_size,
        multi_gpu_enabled=cfg.training.multi_gpu.enabled,
        multi_gpu_num_devices=cfg.training.multi_gpu.num_devices,
        # ── Pool ────────────────────────────────────────────────────────
        pool_size=cfg.pool.size,
        reset_pool_fraction=cfg.pool.reset_fraction,
        reset_pool_interval=cfg.pool.reset_interval,
        reset_strategy=cfg.pool.reset_strategy,
        pool_noise_scale=cfg.pool.noise_scale,
        initial_diversity=cfg.pool.initial_diversity,
        genetic_mutation_rate=cfg.pool.mutation_rate,
        genetic_swaps_per_layer=cfg.pool.n_swaps_per_layer,
        # ── Per-circuit-task meta-learning ──────────────────────────────
        task_sampler_cfg=task_sampler_cfg_train,
        # ── Unified inner-loop regime (online↔batch continuum) ──────────
        # Reads the clean unified keys; falls back to the legacy online keys so older
        # configs keep working. Legacy ``inner_loop_regime: online`` maps to the unified
        # regime with data_mode="stream" (its exact former behavior).
        inner_loop_regime=_resolve_inner_loop_regime(cfg.training),
        data_mode=str(cfg.training.get("data_mode", "stream")),
        n_rounds=int(
            cfg.training.get("n_rounds", cfg.training.get("stream_length", 64))
        ),
        n_rounds_schedule=_parse_n_rounds_schedule(
            cfg.training.get("n_rounds_schedule", cfg.training.get("stream_length_schedule", None))
        ),
        steps_per_round=int(
            cfg.training.get("steps_per_round", cfg.training.get("nca_steps_per_example", 1))
        ),
        window_size=int(
            cfg.training.get("window_size", cfg.training.get("online_window_size", 1))
        ),
        loss_reduction=str(
            cfg.training.get("loss_reduction", cfg.training.get("online_loss_reduction", "mean"))
        ),
        periodic_eval_online_chunk_size=(
            int(cfg.eval.get("online_chunk_size"))
            if cfg.eval.get("online_chunk_size") is not None
            else None
        ),
        eval_ood_task_names=(
            list(task_sampler_cfg_eval_ood["task_names"])
            if task_sampler_cfg_eval_ood and task_sampler_cfg_eval_ood.get("task_names")
            else None
        ),
        init_logits=cfg.circuit.get("init_logits", "soft_wires"),
        random_init_scale=float(cfg.circuit.get("random_init_scale", 1.0)),
        # ── Damage / resilience ─────────────────────────────────────────
        damage_enabled=damage_params["enabled"],
        p_fault=damage_params["p_fault_train"],
        p_fault_eval=damage_params["p_fault_eval"],
        permanent_damage=damage_params["permanent"],
        faulty_logit_value=damage_params["faulty_logit_value"],
        n_damage_steps=damage_params["n_damage_steps"],
        knockouts_per_event=damage_params["knockouts_per_event"],
        p_fault_onset_step_train=damage_params["p_fault_onset_step_train"],
        p_fault_onset_step_eval=damage_params["p_fault_onset_step_eval"],
        # Bursty background damage (training-only; p_fault_train above is the
        # quiet base rate when the burst block is enabled)
        p_fault_burst=damage_params["burst"].get("p_fault_burst_train"),
        burst_start_rate=float(damage_params["burst"].get("burst_start_rate_train", 0.0)),
        burst_length=int(damage_params["burst"].get("length", 4)),
        compute_no_repair_baseline=damage_params["compute_no_repair_baseline"],
        # ── Probabilistic wire shuffle (training-only) ──────────────────
        p_shuffle=shuffle_params["p_shuffle_train"],
        shuffle_fraction=shuffle_params["shuffle_fraction"],
        p_shuffle_onset_step_train=shuffle_params["p_shuffle_onset_step_train"],
        # ── Graph topology (used by topology refresh on shuffle events) ──
        bidirectional_edges=bool(cfg.graph.bidirectional_edges),
        neighboring_connections=bool(cfg.graph.neighboring_connections),
        # ── Periodic evaluation ─────────────────────────────────────────
        periodic_eval_enabled=cfg.eval.enabled,
        periodic_eval_interval=cfg.eval.interval,
        periodic_eval_inner_steps=cfg.eval.inner_steps,
        periodic_eval_log_stepwise=cfg.eval.log_stepwise,
        periodic_eval_log_pool_scatter=cfg.eval.log_pool_scatter,
        # Default False keeps periodic eval's memory footprint small. Flip to
        # True via cfg.eval.keep_full_graphs if you need notebook-style access
        # to the per-circuit per-step graphs / all_metrics during training.
        periodic_eval_keep_full_graphs=cfg.eval.get("keep_full_graphs", False),
        # ── Early stopping ──────────────────────────────────────────────
        early_stopping=EarlyStopping(
            metric=cfg.stop_accuracy.metric,
            source=cfg.stop_accuracy.source,
            threshold=cfg.stop_accuracy.threshold,
            patience=cfg.stop_accuracy.patience,
            min_epochs=cfg.stop_accuracy.min_epochs,
            scope=cfg.stop_accuracy.get(
                "scope",
                ["eval_in_test", "eval_damaged_in_test"]
                if cfg.damage.enabled
                else ["eval_in_test"],
            ),
        )
        if cfg.stop_accuracy.enabled
        else None,
        # ── Checkpointing & best model tracking ────────────────────────
        checkpoint_enabled=cfg.checkpoint.enabled,
        checkpoint_dir=checkpoint_dir,
        checkpoint_interval=cfg.checkpoint.interval,
        save_best=cfg.checkpoint.save_best,
        track_metrics=track_metrics,
        # Bit-faithful mid-training resume + full cfg for self-sufficient saves.
        resume_from=cfg.training.get("resume_from", None),
        cfg=cfg,
        # ── Logging & WandB ─────────────────────────────────────────────
        wandb_logging=cfg.wandb.enabled,
        log_interval=cfg.logging.log_interval,
        wandb_run_config=OmegaConf.to_container(cfg, resolve=True),
        # ── Debugging ───────────────────────────────────────────────────
        do_check_gradients=cfg.training.check_gradients,
    )

    # Save final model if checkpointing is enabled
    if cfg.checkpoint.enabled and not cfg.wandb.enabled:
        # If wandb is enabled, checkpoints are already being saved during training
        save_checkpoint(
            model_results["model"],
            model_results["optimizer"],
            {
                "losses": model_results["losses"],
                "hard_losses": model_results["hard_losses"],
                "accuracies": model_results["accuracies"],
                "hard_accuracies": model_results["hard_accuracies"],
                "reset_steps": model_results.get("reset_steps", []),
            },
            cfg,
            cfg.training.epochs or 2**cfg.training.epochs_power_of_2,
            checkpoint_dir,
            filename="final_model.pkl",
        )

    # Run comprehensive evaluation using standardized datasets
    if cfg.eval.enabled:
        # Create standardized evaluation datasets

        n_damage_steps = damage_params["n_damage_steps"]
        if n_damage_steps > 0:
            damage_steps = jax.numpy.linspace(
                0,
                cfg.eval.inner_steps,
                n_damage_steps + 1,
                endpoint=False,
            ).astype(int)[1:]
            damage_key = eval_key
        else:
            damage_steps = None
            damage_key = None

        eval_results = run_unified_periodic_evaluation(
            model=model_results["model"],
            datasets=eval_datasets,
            data_dict={
                "x_train": x_train,
                "y_train": y_train,
                "x_test": x_test,
                "y_test": y_test,
                "x_total": x_total,
                "y_total": y_total,
            },
            input_n=input_n,
            arity=arity,
            circuit_hidden_dim=cfg.model.circuit_hidden_dim,
            n_message_steps=cfg.eval.inner_steps,
            loss_cfg=LossConfig.from_dict(dict(cfg.loss)),
            epoch=-1,  # Final evaluation marker
            wandb_run=wandb_run,
            log_stepwise=cfg.eval.log_stepwise,
            layer_sizes=layer_sizes,
            log_pool_scatter=False,
            # ── Damage / resilience ─────────────────────────────────────────
            permanent_damage=damage_params["permanent"],
            p_fault=damage_params["p_fault_eval"],
            faulty_value=damage_params["faulty_logit_value"],
            damage_steps=damage_steps,
            knockout_per_damage_step=damage_params["knockouts_per_event"],
            damage_key=damage_key,
            p_fault_onset_step=damage_params["p_fault_onset_step_eval"],
            compute_no_repair_baseline=damage_params["compute_no_repair_baseline"],
            # Memory: same gate as periodic eval. Default False is fine for the
            # final eval since downstream consumers (create_and_save_final_results)
            # only read the small final_metrics summary, not the per-step graphs.
            keep_full_graphs=cfg.eval.get("keep_full_graphs", False),
            # ── Pool ────────────────────────────────────────────────────────
            pool=None,
        )
    else:
        eval_results = None

    if "metrics" in model_results:
        model_results.update(model_results["metrics"])

    # ── Post-training demo probe ─────────────────────────────────────────────
    # Batched scoring of the deploy checkpoint on the live-demo objective
    # (settled accuracy / jitter on random topologies + carry-logits shuffle
    # recovery) — see boolean_nca_cc/training/demo_probe.py. Runs on the GPU
    # this job already holds; metrics land in final_results.csv + final/demo_*.
    demo_metrics = None
    if cfg.get("demo_probe") is not None and cfg.demo_probe.enabled:
        try:
            probe_model = model_results["model"]
            probed = "final"
            if str(cfg.demo_probe.which) == "best":
                wandb_id = wandb_run.run.id if (cfg.wandb.enabled and wandb_run) else None
                if wandb_id and checkpoint_dir is not None:
                    best_path = os.path.join(
                        checkpoint_dir,
                        f"run_{wandb_id}",
                        f"best_model_{cfg.demo_probe.best_metric}.pkl",
                    )
                    if os.path.exists(best_path):
                        # In-place update is safe: the final model was already
                        # saved + evaluated above; only metrics are used below.
                        loaded = load_checkpoint_with_compatibility(best_path)
                        nnx.update(probe_model, loaded["model"])
                        probed = f"best ({cfg.demo_probe.best_metric})"
                    else:
                        log.warning(
                            f"Demo probe: best checkpoint not found at {best_path}; "
                            "probing final model instead"
                        )
            log.info(f"Running demo probe on the {probed} model ...")
            demo_metrics = demo_probe(
                probe_model,
                cfg,
                x_test,
                y_test,
                layer_sizes,
                key=jax.random.fold_in(eval_key, 0xD3),
                n_topologies=int(cfg.demo_probe.n_topologies),
                n_pairs=int(cfg.demo_probe.n_pairs),
                steps=int(cfg.demo_probe.steps),
                settle_window=int(cfg.demo_probe.settle_window),
                chunk_size=int(cfg.demo_probe.chunk_size),
                shotgun_gates=(
                    None
                    if cfg.demo_probe.get("shotgun_gates", None) is None
                    else int(cfg.demo_probe.shotgun_gates)
                ),
                shotgun_volleys=int(cfg.demo_probe.get("shotgun_volleys", 2)),
            )
            log.info("Demo probe results:\n" + format_demo_metrics(demo_metrics))
        except Exception:  # noqa: BLE001 — never let the probe kill a finished run
            log.exception("Demo probe failed; continuing without demo metrics")
            demo_metrics = None

    # Collect comprehensive final results
    final_results = create_and_save_final_results(
        cfg,
        model_results,
        eval_results,
        layer_sizes,
        output_dir,
        wandb_run,
        total_params,
        demo_metrics=demo_metrics,
    )

    # Close wandb if enabled
    if cfg.wandb.enabled:
        cleanup_redundant_wandb_artifacts(
            run_id=wandb_run.run.id,
            dry_run=False,
            verbose=True,
            entity=cfg.wandb.entity,
            project=cfg.wandb.project,
        )
        wandb.finish()

    return final_results


if __name__ == "__main__":
    main()
