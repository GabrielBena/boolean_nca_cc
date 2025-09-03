# Boolean Circuit Sparsity Studies

This directory contains Hydra-based experiments for studying sparsity regularization in boolean circuits.

## Structure

```
BP sweep/
├── configs/                    # Hydra configuration files
│   ├── config.yaml            # Main config
│   ├── experiment/            # Experiment types
│   │   ├── quick_study.yaml   # Fast testing
│   │   ├── full_study.yaml    # Comprehensive study
│   │   └── ablation_study.yaml # Focused analysis
│   ├── task/                  # Task configurations
│   │   ├── binary_multiply_8bit.yaml
│   │   ├── binary_multiply_6bit.yaml
│   │   └── xor_4bit.yaml
│   ├── optimizer/             # Optimizer settings
│   │   ├── adam_default.yaml
│   │   ├── adam_fast.yaml
│   │   └── adamw_default.yaml
│   └── sparsity/              # Sparsity configurations
│       ├── none.yaml          # No sparsity (baseline)
│       ├── l1_light.yaml      # Light L1 regularization
│       ├── l1_medium.yaml     # Medium L1 regularization
│       └── ...
├── run_study_hydra.py         # Main Hydra-based runner
├── requirements.txt           # Python dependencies
└── README.md                  # This documentation
```

## Usage

### Basic Usage

```bash
# Default configuration (quick study)
python run_study_hydra.py

# Full comprehensive study
python run_study_hydra.py experiment=full_study

# Different task
python run_study_hydra.py task=binary_multiply_6bit

# Different sparsity configuration
python run_study_hydra.py sparsity=l1_strong

# Combine multiple overrides
python run_study_hydra.py experiment=full_study task=xor_4bit optimizer=adam_fast
```

### Advanced Usage

```bash
# Override specific parameters
python run_study_hydra.py experiment.num_steps=512 task.input_n=10

# Multiple runs with different configurations
python run_study_hydra.py --multirun experiment=quick_study,full_study

# Sweep over multiple sparsity weights
python run_study_hydra.py --multirun sparsity.weight=1e-4,1e-3,1e-2

# Custom results directory
python run_study_hydra.py results_dir=custom_results/my_experiment
```

### Configuration Composition

The system uses Hydra's composition feature:

```yaml
# configs/config.yaml
defaults:
  - experiment: quick_study      # Choose experiment type
  - task: binary_multiply_8bit   # Choose task
  - optimizer: adam_default      # Choose optimizer
  - sparsity: l1_medium         # Choose sparsity
  - _self_                      # Apply overrides last
```

## Configuration Files

### Experiment Types

- **`quick_study`**: Fast experiments for testing (64 steps, 2 seeds)
- **`full_study`**: Comprehensive study (256 steps, 5 seeds each)
- **`ablation_study`**: Focused analysis on specific configurations

### Tasks

- **`binary_multiply_8bit`**: 8-bit multiplication (challenging)
- **`binary_multiply_6bit`**: 6-bit multiplication (faster)
- **`xor_4bit`**: Simple XOR task (debugging)

### Sparsity Types

- **`none`**: No sparsity (baseline)
- **`l1_light`**: Light L1 regularization (1e-4)
- **`l1_medium`**: Medium L1 regularization (1e-3)
- **`l1_strong`**: Strong L1 regularization (1e-2)
- **`binary_medium`**: Binary sparsity regularization
- **`entropy_medium`**: Entropy-based sparsity regularization

## Output

Results are automatically saved to timestamped directories:

```
outputs/sparsity_comparison/YYYY-MM-DD_HH-MM-SS/
├── detailed_results.csv       # All experimental results
├── summary.json              # Statistical summary
├── config.yaml               # Configuration used
└── .hydra/                   # Hydra metadata
```

## Example Workflows

### Quick Testing
```bash
# Test new sparsity configuration
python run_study_hydra.py experiment=quick_study sparsity=l1_strong

# Test on smaller problem
python run_study_hydra.py task=xor_4bit experiment=quick_study
```

### Production Studies
```bash
# Comprehensive 8-bit multiplication study
python run_study_hydra.py experiment=full_study task=binary_multiply_8bit

# Compare all sparsity types
python run_study_hydra.py --multirun experiment=full_study sparsity=l1_medium,binary_medium,entropy_medium
```

### Ablation Studies
```bash
# Fine-grained sparsity weight search
python run_study_hydra.py experiment=ablation_study --multirun sparsity.weight=1e-4,5e-4,1e-3,2e-3,5e-3
```

## Benefits of Hydra System

1. **Clean Configuration**: Separate concerns into modular config files
2. **Easy Composition**: Mix and match different components
3. **Reproducibility**: All configurations are logged automatically
4. **Parameter Sweeps**: Built-in multirun support for hyperparameter search
5. **Override System**: Easy command-line parameter overrides
6. **Timestamped Outputs**: Automatic experiment tracking
7. **Type Safety**: Configuration validation and type checking

## Clean Hydra-Based System

This directory now contains only the modern Hydra-based configuration system for clean, reproducible experiments. The legacy scripts have been removed to maintain a focused codebase.
