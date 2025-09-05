# Boolean Circuit Training with Cross-Validation

A simplified single experiment training script with k-fold cross-validation and WandB integration for hyperparameter optimization.

## Quick Start

```bash
# Default configuration (5-fold CV, L1 sparsity, 8-bit multiplication)
python train_single.py

# Override specific parameters
python train_single.py optimizer.learning_rate=0.1 sparsity.weight=1e-2

# Different tasks
python train_single.py task=binary_multiply_6bit
python train_single.py task=xor_4bit

# Different sparsity configurations
python train_single.py sparsity=l1_strong
python train_single.py sparsity=none

# Different cross-validation configurations
python train_single.py cross_validation=single  # No CV (1 split)
python train_single.py cross_validation=cv3     # 3-fold CV
python train_single.py cross_validation=cv10    # 10-fold CV

# Disable WandB logging
python train_single.py wandb=disabled
```

## Configuration Structure

The script uses Hydra's compositional configuration system:

```
configs/
├── config.yaml              # Main configuration
├── task/                    # Task definitions
│   ├── binary_multiply_8bit.yaml
│   ├── binary_multiply_6bit.yaml
│   └── xor_4bit.yaml
├── optimizer/               # Optimizer configurations  
│   ├── adam_default.yaml
│   ├── adam_fast.yaml
│   └── adamw_default.yaml
├── sparsity/               # Sparsity regularization
│   ├── none.yaml
│   ├── l1_light.yaml
│   ├── l1_medium.yaml
│   ├── l1_strong.yaml
│   ├── binary_medium.yaml
│   └── entropy_medium.yaml
├── cross_validation/        # CV configurations
│   ├── single.yaml         # 1 split
│   ├── cv3.yaml           # 3-fold
│   ├── cv5.yaml           # 5-fold
│   └── cv10.yaml          # 10-fold
└── wandb/                  # WandB settings
    ├── enabled.yaml
    └── disabled.yaml
```

## Cross-Validation Features

- **Robust evaluation**: Multiple train/test splits for statistical reliability
- **Deterministic splits**: Reproducible results from base seed
- **Comprehensive metrics**: Individual split results + aggregated statistics
- **WandB integration**: Both per-split and summary metrics logged

### Metrics Logged

**Per Split:**
- Training/test accuracy and loss curves
- Generalization gap over time
- Sparsity metrics (if enabled)
- Final performance metrics

**Aggregated (Cross-Validation Summary):**
- Mean ± std for all metrics across splits
- Min/max ranges for key metrics
- Statistical significance indicators

## WandB Integration

- **Project**: `boolean_bp_sweep`
- **Tags**: Automatic tagging by task, sparsity type, optimizer, CV folds
- **Config tracking**: Full Hydra configuration logged
- **Artifacts**: Optional model checkpointing (can be added)

Perfect for hyperparameter optimization with WandB Sweeps!

## Example Usage for Hyperparameter Optimization

```bash
# Single experiment with specific hyperparameters
python train_single.py \
    optimizer.learning_rate=0.5 \
    sparsity.type=l1 \
    sparsity.weight=1e-3 \
    training.num_steps=512 \
    cross_validation.num_splits=10

# Quick testing with minimal CV
python train_single.py cross_validation=single training.num_steps=64
```

## Architecture

The script maintains the same circuit generation and training logic as the original study but:
- Focuses on a single hyperparameter configuration per run
- Uses cross-validation for robust evaluation within that configuration
- Offloads sweep logic to external tools (WandB, Optuna, etc.)
- Provides comprehensive logging for analysis