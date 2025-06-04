#!/usr/bin/env python3
"""
Safe training script that launches train.py with automatic failure recovery.

This script monitors train.py execution and automatically adjusts parameters
when specific failures occur:
- NaN loss: divide learning rate by 1.5
- Out of memory: divide batch size by 1.5
"""

import os
import sys
import subprocess
import re
import time
import logging
import argparse
import signal
import atexit
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import yaml
from omegaconf import OmegaConf, open_dict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("safe_train.log")],
)
log = logging.getLogger(__name__)

# Global variable to track the current training process
current_process = None

# Global variable to track current wandb run ID for cleanup
current_wandb_run_id = None

# Global variable to track the monitor for cleanup
current_monitor = None


def signal_handler(signum, frame):
    """Handle termination signals and cleanup child processes."""
    log.info(f"Received signal {signum}, cleaning up...")
    cleanup_process()
    cleanup_temp_configs()
    sys.exit(0)


def cleanup_process():
    """Terminate the current training process if it exists."""
    global current_process
    if current_process is not None:
        log.info("Terminating training process...")
        try:
            # For Unix systems, terminate the entire process group
            if os.name != "nt" and hasattr(current_process, "pid"):
                try:
                    # Get the process group ID
                    pgid = os.getpgid(current_process.pid)
                    log.info(f"Terminating process group {pgid}")

                    # Send SIGTERM to the entire process group
                    os.killpg(pgid, signal.SIGTERM)

                    # Wait up to 10 seconds for graceful termination
                    try:
                        current_process.wait(timeout=10)
                        log.info("Training process group terminated gracefully")
                    except subprocess.TimeoutExpired:
                        # If graceful termination fails, force kill the process group
                        log.warning(
                            "Graceful termination failed, force killing process group..."
                        )
                        os.killpg(pgid, signal.SIGKILL)
                        current_process.wait()
                        log.info("Training process group force killed")

                except (ProcessLookupError, OSError) as e:
                    log.warning(
                        f"Process group cleanup failed: {e}, falling back to single process cleanup"
                    )
                    # Fall back to single process cleanup
                    current_process.terminate()
                    try:
                        current_process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        current_process.kill()
                        current_process.wait()
            else:
                # Windows or fallback: terminate single process
                current_process.terminate()
                try:
                    current_process.wait(timeout=10)
                    log.info("Training process terminated gracefully")
                except subprocess.TimeoutExpired:
                    log.warning("Graceful termination failed, force killing process...")
                    current_process.kill()
                    current_process.wait()
                    log.info("Training process force killed")

        except Exception as e:
            log.error(f"Error terminating process: {e}")
        finally:
            current_process = None


def cleanup_temp_configs():
    """Clean up temporary config files if monitor exists."""
    global current_monitor
    if current_monitor is not None:
        try:
            current_monitor.cleanup_temp_files()
            log.info("Cleaned up temporary config files")
        except Exception as e:
            log.warning(f"Error cleaning up temp configs: {e}")


def delete_wandb_run(run_id: str) -> bool:
    """
    Delete a wandb run by ID.

    Returns:
        True if deletion was successful or run doesn't exist, False if error occurred
    """
    if not run_id:
        return True

    try:
        import wandb

        # Try to get the API
        api = wandb.Api()

        # Get the project name from environment or use default
        project = os.environ.get("WANDB_PROJECT", "boolean-nca-cc")
        entity = os.environ.get("WANDB_ENTITY", "m2snn")

        try:
            # Get the run
            run = api.run(f"{entity}/{project}/{run_id}")

            # Delete the run
            run.delete()
            log.info(f"Deleted failed wandb run: {run_id}")
            return True

        except wandb.errors.CommError as e:
            if "not found" in str(e).lower():
                log.info(
                    f"Wandb run {run_id} not found (may not have been created yet)"
                )
                return True
            else:
                log.warning(f"Failed to delete wandb run {run_id}: {e}")
                return False

    except ImportError:
        log.warning("wandb not available, cannot delete runs")
        return True
    except Exception as e:
        log.warning(f"Error deleting wandb run {run_id}: {e}")
        return False


def extract_wandb_run_id(stdout: str, stderr: str) -> Optional[str]:
    """
    Extract wandb run ID from training output.

    Returns:
        The run ID if found, None otherwise
    """
    combined_output = stdout + stderr

    # Look for wandb run URL or ID patterns
    patterns = [
        r"wandb: (?:🚀 )?View run at https://wandb\.ai/.+/runs/([a-zA-Z0-9]+)",
        r"wandb: Run data is saved locally in .+/([a-zA-Z0-9]+)",
        r"wandb.*run.*([a-zA-Z0-9]{8})",  # Generic pattern for run IDs
        r"WandB run ID: ([a-zA-Z0-9]{8})",  # WandB run ID pattern
    ]

    for pattern in patterns:
        matches = re.findall(pattern, combined_output)
        if matches:
            return matches[-1]  # Return the last match (most recent)

    return None


# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)  # Ctrl+C
signal.signal(signal.SIGTERM, signal_handler)  # Termination signal
atexit.register(cleanup_process)  # Cleanup on exit


class TrainingMonitor:
    """Monitors training execution and handles failures."""

    def __init__(
        self,
        max_retries: int = 5,
        min_lr: float = 1e-8,
        min_batch_size: int = 1,
        delete_failed_wandb_runs: bool = True,
        alpha: float = 1.5,
    ):
        self.max_retries = max_retries
        self.min_lr = min_lr
        self.min_batch_size = min_batch_size
        self.delete_failed_wandb_runs = delete_failed_wandb_runs
        self.retry_count = 0
        self.adjustments_made = []
        self.temp_config_files = []  # Track temp files for cleanup
        self.failed_wandb_runs = []  # Track failed runs for deletion
        self.alpha = alpha

    def cleanup_temp_files(self):
        """Clean up temporary config files."""
        if not self.temp_config_files:
            return

        log.info(f"Cleaning up {len(self.temp_config_files)} temporary config files...")
        for temp_file in self.temp_config_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                    log.debug(f"Cleaned up temporary config file: {temp_file}")
                else:
                    log.debug(f"Temporary config file already removed: {temp_file}")
            except Exception as e:
                log.warning(f"Failed to clean up temporary file {temp_file}: {e}")

        self.temp_config_files.clear()
        log.info("Temporary config file cleanup completed")

    def detect_failure_type(
        self, stdout: str, stderr: str, returncode: int
    ) -> Optional[str]:
        """
        Detect the type of failure from process output.

        Returns:
            - "nan_loss" if NaN loss detected
            - "oom" if out of memory detected
            - "other" for other failures
            - None if no failure detected
        """
        combined_output = stdout.lower() + stderr.lower()

        # Check for NaN loss patterns - based on actual logging in train.py and train_loop.py
        nan_patterns = [
            r"loss.*nan",  # "Loss=nan", "loss: nan"
            r"nan.*loss",  # "nan loss"
            r"loss:.*nan",  # "Loss: nan" (progress bar)
            r"loss=.*nan",  # "Loss=nan" (BP training logs)
            r"training.*nan",  # "Loss is NaN at epoch", "training/loss: nan"
            r"gradient.*nan",  # NaN gradients
            r"parameter.*nan",  # NaN parameters
            r"loss is nan",  # Exact match from train_loop.py
            r"hard.*loss.*nan",  # "hard_loss: nan"
            r"accuracy.*nan",  # NaN accuracy values
        ]

        for pattern in nan_patterns:
            if re.search(pattern, combined_output):
                return "nan_loss"

        # Check for OOM patterns - including JAX-specific errors
        oom_patterns = [
            r"out of memory",
            r"outofmemoryerror",
            r"cuda out of memory",
            r"memory error",
            r"allocation failed",
            r"not enough memory",
            r"resource exhausted",  # JAX/TensorFlow style
            r"xla.*memory",  # XLA memory errors
            r"device memory exhausted",  # JAX device memory
            r"failed to allocate",  # Common allocation failure
            r"insufficient memory",  # General memory error
            r"memory limit exceeded",  # Slurm/cluster memory limits
        ]

        for pattern in oom_patterns:
            if re.search(pattern, combined_output):
                return "oom"

        # If process failed but we don't recognize the error
        if returncode != 0:
            return "other"

        return None

    def create_temp_config_with_adjustments(
        self, original_config_path: str, failure_type: str
    ) -> Optional[str]:
        """
        Create a temporary config file with adjustments for the failure type.
        Keeps the original config file untouched.

        Returns:
            Path to temporary config file if successful, None if limits reached
        """
        # Load original config
        try:
            cfg = OmegaConf.load(original_config_path)
        except Exception as e:
            log.error(f"Failed to load config {original_config_path}: {e}")
            return None

        with open_dict(cfg):
            if failure_type == "nan_loss":
                current_lr = cfg.training.learning_rate
                new_lr = current_lr / self.alpha

                if new_lr < self.min_lr:
                    log.warning(
                        f"Learning rate would be too low ({new_lr:.2e} < {self.min_lr:.2e}). Cannot retry."
                    )
                    return None

                cfg.training.learning_rate = new_lr
                adjustment = f"learning_rate: {current_lr:.2e} -> {new_lr:.2e}"
                log.info(f"NaN loss detected. Reducing {adjustment}")

            elif failure_type == "oom":
                current_batch = cfg.training.meta_batch_size
                new_batch = max(1, int(current_batch / self.alpha))

                if new_batch < self.min_batch_size:
                    log.warning(
                        f"Batch size would be too small ({new_batch} < {self.min_batch_size}). Cannot retry."
                    )
                    return None

                cfg.training.meta_batch_size = new_batch

                # Also adjust constant_product in message steps scheduler if it exists
                adjustments = [f"meta_batch_size: {current_batch} -> {new_batch}"]

                if (
                    hasattr(cfg.training, "message_steps_schedule")
                    and cfg.training.message_steps_schedule is not None
                    and hasattr(cfg.training.message_steps_schedule, "constant_product")
                    and cfg.training.message_steps_schedule.constant_product is not None
                ):
                    current_constant_product = (
                        cfg.training.message_steps_schedule.constant_product
                    )
                    new_constant_product = max(
                        1, int(current_constant_product / self.alpha)
                    )
                    cfg.training.message_steps_schedule.constant_product = (
                        new_constant_product
                    )
                    adjustments.append(
                        f"constant_product: {current_constant_product} -> {new_constant_product}"
                    )

                adjustment = ", ".join(adjustments)
                log.info(f"Out of memory detected. Reducing {adjustment}")

            else:
                log.warning(
                    f"Unknown failure type: {failure_type}. Cannot adjust config."
                )
                return None

        # Create temporary config file
        try:
            # Create temp file in same directory as original config
            config_dir = os.path.dirname(original_config_path)
            config_name = os.path.basename(original_config_path)
            name_without_ext = os.path.splitext(config_name)[0]

            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".yaml",
                prefix=f"{name_without_ext}_retry_{self.retry_count + 1}_",
                dir=config_dir,
                delete=False,
            ) as temp_file:
                temp_config_path = temp_file.name

            # Save adjusted config to temp file
            OmegaConf.save(cfg, temp_config_path)
            self.temp_config_files.append(temp_config_path)
            self.adjustments_made.append(adjustment)

            log.info(f"Created temporary config: {temp_config_path}")
            return temp_config_path

        except Exception as e:
            log.error(f"Failed to create temporary config: {e}")
            return None

    def run_training(
        self, train_cmd: list, config_path: str
    ) -> Tuple[bool, str, str, int]:
        """
        Run training command and capture output.

        Returns:
            (success, stdout, stderr, returncode)
        """
        global current_process, current_wandb_run_id

        log.info(
            f"Starting training attempt {self.retry_count + 1}/{self.max_retries + 1}"
        )
        log.info(f"Command: {' '.join(train_cmd)}")
        log.info(f"Using config: {config_path}")

        try:
            # Start new process group to ensure proper cleanup
            # Let stdout go directly to terminal to preserve tqdm, capture stderr for errors
            process = subprocess.Popen(
                train_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,  # Merge stderr into stdout
                text=True,
                bufsize=1,
                universal_newlines=True,
                preexec_fn=os.setsid
                if os.name != "nt"
                else None,  # New process group on Unix
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
                if os.name == "nt"
                else 0,  # Windows equivalent
            )

            # Track the current process globally for signal handling
            current_process = process

            # Capture output while preserving real-time display
            output_lines = []

            # Read output in real-time and forward to terminal
            while True:
                output = process.stdout.read(1024)  # Read in chunks
                if not output and process.poll() is not None:
                    break

                if output:
                    # Print to terminal immediately (preserves tqdm formatting)
                    sys.stdout.write(output)
                    sys.stdout.flush()

                    # Store for failure detection
                    output_lines.append(output)

                    # Quick check for NaN in real-time
                    if "nan" in output.lower() and "loss" in output.lower():
                        log.warning("NaN loss detected in real-time!")

            # Wait for process to complete and get final output
            remaining_output, _ = process.communicate()
            if remaining_output:
                sys.stdout.write(remaining_output)
                sys.stdout.flush()
                output_lines.append(remaining_output)

            # Combine all output
            combined_output = "".join(output_lines)
            stdout = combined_output
            stderr = ""  # Since we merged stderr into stdout
            returncode = process.returncode

            # Extract wandb run ID for potential cleanup
            current_wandb_run_id = extract_wandb_run_id(stdout, stderr)
            if current_wandb_run_id:
                log.info(f"Detected wandb run ID: {current_wandb_run_id}")

            success = returncode == 0
            if success:
                log.info("Training completed successfully!")
                current_wandb_run_id = None  # Don't delete successful runs
            else:
                log.warning(f"Training failed with return code {returncode}")
                if current_wandb_run_id and self.delete_failed_wandb_runs:
                    self.failed_wandb_runs.append(current_wandb_run_id)

            return success, stdout, stderr, returncode

        except Exception as e:
            log.error(f"Failed to run training command: {e}")
            return False, "", str(e), -1
        finally:
            # Clear the global process reference when done
            current_process = None

    def cleanup_failed_wandb_runs(self):
        """Delete all failed wandb runs."""
        if not self.delete_failed_wandb_runs or not self.failed_wandb_runs:
            return

        log.info(f"Cleaning up {len(self.failed_wandb_runs)} failed wandb runs...")
        for run_id in self.failed_wandb_runs:
            delete_wandb_run(run_id)
        self.failed_wandb_runs.clear()


def parse_hydra_overrides(args: list) -> Dict[str, Any]:
    """Parse Hydra-style command line overrides."""
    overrides = {}
    for arg in args:
        if "=" in arg:
            key, value = arg.split("=", 1)
            # Try to convert to appropriate type
            try:
                if value.lower() in ["true", "false"]:
                    value = value.lower() == "true"
                elif "." in value:
                    value = float(value)
                else:
                    value = int(value)
            except ValueError:
                pass  # Keep as string
            overrides[key] = value
    return overrides


def main():
    # Add global monitor reference to enable cleanup in signal handler
    global current_monitor

    parser = argparse.ArgumentParser(
        description="Safe training script with automatic failure recovery"
    )
    parser.add_argument(
        "--config-path", default="configs", help="Path to config directory"
    )
    parser.add_argument(
        "--config-name", default="config", help="Config file name (without .yaml)"
    )
    parser.add_argument(
        "--max-retries", type=int, default=10, help="Maximum number of retries"
    )
    parser.add_argument(
        "--min-lr", type=float, default=1e-6, help="Minimum learning rate"
    )
    parser.add_argument(
        "--min-batch-size", type=int, default=1, help="Minimum batch size"
    )
    parser.add_argument(
        "--train-script", default="train.py", help="Path to training script"
    )
    parser.add_argument(
        "--no-delete-wandb-runs",
        action="store_false",
        help="Disable automatic deletion of failed wandb runs",
    )

    parser.add_argument(
        "--alpha", type=float, default=1.5, help="Alpha for parameter reduction"
    )

    # Parse known args to separate hydra overrides
    args, hydra_args = parser.parse_known_args()

    # Setup paths
    config_dir = Path(args.config_path)
    original_config_file = config_dir / f"{args.config_name}.yaml"

    if not original_config_file.exists():
        log.error(f"Config file not found: {original_config_file}")
        sys.exit(1)

    if not Path(args.train_script).exists():
        log.error(f"Training script not found: {args.train_script}")
        sys.exit(1)

    # Initialize monitor
    monitor = TrainingMonitor(
        max_retries=args.max_retries,
        min_lr=args.min_lr,
        min_batch_size=args.min_batch_size,
        delete_failed_wandb_runs=not args.no_delete_wandb_runs,
        alpha=args.alpha,
    )

    # Set global monitor reference for cleanup on interruption
    current_monitor = monitor

    log.info(f"Safe training started with config: {original_config_file}")
    log.info(f"Max retries: {args.max_retries}")
    log.info(f"Min learning rate: {args.min_lr:.2e}")
    log.info(f"Min batch size: {args.min_batch_size}")
    log.info(f"Delete failed wandb runs: {not args.no_delete_wandb_runs}")

    # Use original config for first attempt
    current_config = str(original_config_file)

    # Training loop with retries
    try:
        while monitor.retry_count <= args.max_retries:
            # Build training command with current config
            train_cmd = [
                sys.executable,
                args.train_script,
                "--config-path",
                os.path.dirname(current_config),
                "--config-name",
                os.path.splitext(os.path.basename(current_config))[0],
            ] + hydra_args

            success, stdout, stderr, returncode = monitor.run_training(
                train_cmd, current_config
            )

            if success:
                log.info("Training completed successfully!")
                if monitor.adjustments_made:
                    log.info("Adjustments made during training:")
                    for adj in monitor.adjustments_made:
                        log.info(f"  - {adj}")
                break

            # Detect failure type
            failure_type = monitor.detect_failure_type(stdout, stderr, returncode)
            log.warning(f"Training failed. Detected failure type: {failure_type}")

            if failure_type in ["nan_loss", "oom"]:
                if monitor.retry_count < args.max_retries:
                    # Create temp config with adjustments and retry
                    temp_config = monitor.create_temp_config_with_adjustments(
                        current_config, failure_type
                    )
                    if temp_config:
                        current_config = temp_config
                        monitor.retry_count += 1
                        log.info(f"Retrying training with adjusted parameters...")
                        time.sleep(2)  # Brief pause before retry
                        continue
                    else:
                        log.error("Cannot adjust parameters further. Stopping.")
                        break
                else:
                    log.error(
                        f"Maximum retries ({args.max_retries}) reached. Stopping."
                    )
                    break
            else:
                log.error(f"Unrecoverable failure or unknown error. Stopping.")
                if stderr:
                    log.error(f"Error output: {stderr}")
                break

    finally:
        # Clean up temporary config files
        monitor.cleanup_temp_files()

        # Clean up failed wandb runs
        monitor.cleanup_failed_wandb_runs()

    # Final summary
    if monitor.adjustments_made:
        log.info("\nFinal summary of adjustments made:")
        for adj in monitor.adjustments_made:
            log.info(f"  - {adj}")
    else:
        log.info("No parameter adjustments were needed.")

    log.info(f"Safe training completed after {monitor.retry_count} retries.")
    log.info(f"Original config file unchanged: {original_config_file}")


if __name__ == "__main__":
    main()
