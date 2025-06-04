# Safe Training Script

The `safe_train.py` script provides automatic failure recovery for training runs. It monitors `train.py` execution and automatically adjusts parameters when specific failures occur.

## Features

- **NaN Loss Recovery**: Automatically reduces learning rate by 1.5x when NaN losses are detected
- **Out-of-Memory Recovery**: Automatically reduces batch size by 1.5x when OOM errors occur
- **Temporary Config Files**: Creates temporary config files for retries, keeping the original config untouched
- **Failed Run Cleanup**: Automatically deletes failed wandb runs to keep your workspace clean
- **Real-time Monitoring**: Shows training output in real-time while monitoring for failures
- **Progress Bar Support**: Preserves tqdm progress bars and terminal formatting
- **Configurable Limits**: Set minimum learning rate and batch size limits
- **Comprehensive Logging**: Logs all adjustments and provides detailed failure analysis
- **Signal Handling**: Properly handles termination signals (Ctrl+C, SIGTERM) and cleans up child processes
- **Process Group Management**: Uses process groups to ensure complete cleanup of all child processes

## Usage

### Basic Usage

```bash
# Use default config
python safe_train.py

# Use specific config
python safe_train.py --config-name experiment_config

# Set custom limits
python safe_train.py --max-retries 10 --min-lr 1e-10 --min-batch-size 1

# Disable wandb run deletion
python safe_train.py --no-delete-wandb-runs
```

### With Hydra Overrides

```bash
# Override training parameters
python safe_train.py training.learning_rate=1e-3 training.meta_batch_size=32

# Override model type
python safe_train.py model=gnn training.epochs_power_of_2=15

# Multiple overrides
python safe_train.py \
    model=self_attention \
    training.learning_rate=5e-4 \
    training.meta_batch_size=16 \
    pool.size=4096
```

## Command Line Arguments

- `--config-path`: Path to config directory (default: "configs")
- `--config-name`: Config file name without .yaml (default: "config")
- `--max-retries`: Maximum number of retries (default: 10)
- `--min-lr`: Minimum learning rate threshold (default: 1e-6)
- `--min-batch-size`: Minimum batch size threshold (default: 1)
- `--train-script`: Path to training script (default: "train.py")
- `--no-delete-wandb-runs`: Disable automatic deletion of failed wandb runs

## Failure Detection

### NaN Loss Detection
Detects patterns like:
- `"Loss is NaN at epoch 123"`
- `"Loss=nan"`
- `"training/loss: nan"`
- `"hard_loss: nan"`

### Out-of-Memory Detection
Detects patterns like:
- `"CUDA out of memory"`
- `"Resource exhausted"`
- `"XLA memory error"`
- `"Device memory exhausted"`

## Automatic Adjustments

### NaN Loss → Reduce Learning Rate
```
Original LR: 2e-4 → Adjusted LR: 1.33e-4 (÷ 1.5)
```

### OOM Error → Reduce Batch Size and Scheduler Constant Product
```
Original Batch: 32 → Adjusted Batch: 21 (÷ 1.5)
Original Constant Product: 500 → Adjusted Constant Product: 333 (÷ 1.5)
```

*Note: The `constant_product` in the message steps scheduler is also adjusted proportionally to maintain curriculum learning consistency when it's configured in the schedule.*

## Config File Management

### Original Config Protection
- **Original config files are never modified**
- Temporary config files are created for each retry attempt
- Temporary files are automatically cleaned up after training
- Original config remains available for future runs

### Temporary Config Files
```bash
# Example temporary files created:
configs/config_retry_1_tmp123.yaml  # First retry
configs/config_retry_2_tmp456.yaml  # Second retry
# etc.
```

## Wandb Run Management

### Automatic Cleanup
- **Failed runs are automatically deleted** from wandb
- Successful runs are preserved
- Run IDs are extracted from training output
- Deletion happens at the end of training

### Wandb Run Detection
The script detects wandb runs from patterns like:
```
wandb: 🚀 View run at https://wandb.ai/team/project/runs/abc123def
wandb: Run data is saved locally in wandb/run-20240115_103045-abc123def
```

### Environment Variables
Set these to override default wandb settings:
```bash
export WANDB_PROJECT="your-project-name"
export WANDB_ENTITY="your-team-name"
```

## Signal Handling & Process Management

The script properly handles termination signals to ensure clean shutdown:

### Supported Signals
- **SIGINT**: Ctrl+C from terminal
- **SIGTERM**: Termination signal from system/scheduler
- **Process Exit**: Automatic cleanup on script exit

### Cleanup Process
1. **Graceful Termination**: First attempts to terminate child processes gracefully (10-second timeout)
2. **Force Kill**: If graceful termination fails, force kills the process group
3. **Process Groups**: Uses process groups on Unix systems to ensure all child processes are terminated
4. **Cross-Platform**: Works on both Unix/Linux and Windows systems
5. **File Cleanup**: Removes temporary config files
6. **Wandb Cleanup**: Deletes failed wandb runs

### Usage with Job Schedulers
```bash
# SLURM example - the script will properly handle job cancellation
sbatch --wrap="python safe_train.py training.epochs_power_of_2=20"

# When you cancel the job, all processes will be cleaned up properly
scancel <job_id>
```

## Example Session

```bash
$ python safe_train.py --max-retries 3

2024-01-15 10:30:00 - INFO - Safe training started with config: configs/config.yaml
2024-01-15 10:30:00 - INFO - Max retries: 3
2024-01-15 10:30:00 - INFO - Min learning rate: 1.00e-06
2024-01-15 10:30:00 - INFO - Min batch size: 1
2024-01-15 10:30:00 - INFO - Delete failed wandb runs: True
2024-01-15 10:30:00 - INFO - Starting training attempt 1/4
2024-01-15 10:30:00 - INFO - Using config: configs/config.yaml

# Training output...
Training GNN: 45%|████▌     | 9000/20000 [15:23<18:42,  9.80it/s, Loss=nan, Accuracy=0.8456]

2024-01-15 10:45:23 - WARNING - NaN loss detected in real-time!
2024-01-15 10:45:23 - INFO - Detected wandb run ID: abc123def
2024-01-15 10:45:23 - WARNING - Training failed with return code 0
2024-01-15 10:45:23 - WARNING - Training failed. Detected failure type: nan_loss
2024-01-15 10:45:23 - INFO - NaN loss detected. Reducing learning_rate: 2.00e-04 -> 1.33e-04
2024-01-15 10:45:23 - INFO - Created temporary config: configs/config_retry_1_tmp456.yaml
2024-01-15 10:45:23 - INFO - Retrying training with adjusted parameters...
2024-01-15 10:45:25 - INFO - Starting training attempt 2/4
2024-01-15 10:45:25 - INFO - Using config: configs/config_retry_1_tmp456.yaml

# Continues with adjusted parameters...
# If successful:
2024-01-15 11:30:00 - INFO - Training completed successfully!
2024-01-15 11:30:00 - INFO - Cleaning up 1 failed wandb runs...
2024-01-15 11:30:00 - INFO - Deleted failed wandb run: abc123def
2024-01-15 11:30:00 - INFO - Original config file unchanged: configs/config.yaml
```

## Output Files

- `safe_train.log`: Detailed log of all safe training activities
- Standard train.py outputs (checkpoints, plots, etc.)
- Temporary config files (automatically cleaned up)
- Original config file (unchanged)

## Tips

1. **Set appropriate limits**: Use `--min-lr` and `--min-batch-size` based on your hardware
2. **Monitor the log**: Check `safe_train.log` for detailed failure analysis
3. **Use with wandb**: The script preserves wandb logging for successful runs and cleans up failed ones
4. **Cluster usage**: Particularly useful for long-running cluster jobs where manual intervention isn't possible
5. **Config preservation**: Your original config files are never modified, so you can safely use version control
6. **Wandb workspace**: Failed runs are automatically deleted to keep your wandb workspace clean