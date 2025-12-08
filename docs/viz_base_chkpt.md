# Visualization Base Checkpoint Loading Strategy

## Overview

Base script pattern for loading model checkpoints and recreating evaluation scenarios for downstream visualization scripts. Ensures consistency with training loop evaluation patterns.

## Core Components

### 1. Model Loading

**Function**: `load_best_model_from_wandb()` or `load_config_from_wandb()` + `load_model_from_config_and_checkpoint()`

**Location**: `boolean_nca_cc/training/checkpointing.py`

**Key Pattern** (from `GUI_minimal.py` lines 863-994):
```python
from boolean_nca_cc.training.checkpointing import (
    load_config_from_wandb,
    load_model_from_config_and_checkpoint,
    derive_checkpoint_metric_from_config,
)

# For best model (recommended):
loaded_config, checkpoint_path, run_id = load_config_from_wandb(
    run_id=run_id,
    filename="best_model",  # or "latest_checkpoint"
    select_by_best_metric=True,
    metric_name="eval_ko_in/final_hard_accuracy",  # or derive from config
    prefer_metric="eval_ko_in_hard_accuracy",
)

model, loaded_dict = load_model_from_config_and_checkpoint(
    config=loaded_config,
    checkpoint_path=checkpoint_path,
    run_id=run_id,
    seed=0,
)
```

**Critical**: Use `derive_checkpoint_metric_from_config()` to get correct metric name from config's checkpoint settings (see `tests/test_gui_vs_eval_conditions.py` lines 103-105).

### 2. Data Generation and Splitting

**Function**: `get_task_data()` + `split_input_combinations()`

**Locations**: 
- `boolean_nca_cc/circuits/tasks.py` (get_task_data)
- `boolean_nca_cc/circuits/data_split.py` (split_input_combinations)

**Pattern** (from `train_loop.py` lines 817-832):
```python
from boolean_nca_cc.circuits.tasks import get_task_data
from boolean_nca_cc.circuits.data_split import split_input_combinations

# Generate all input combinations
x_data, y_data = get_task_data(
    task=config.circuit.task,
    n_combinations=2**config.circuit.input_bits,
    input_bits=config.circuit.input_bits,
)

# Split if enabled (matches training)
if config.eval.input_split_enabled:
    x_train, y_train, x_test, y_test = split_input_combinations(
        x_data=x_data,
        y_data=y_data,
        train_fraction=config.eval.input_train_fraction,
        seed=config.eval.input_split_seed,
    )
else:
    x_train, y_train = x_data, y_data
    x_test, y_test = x_data, y_data
```

### 3. Base Circuit Generation

**Function**: `gen_circuit()` or `preconfigure_circuit_logits()`

**Locations**:
- `boolean_nca_cc/circuits/model.py` (gen_circuit)
- `boolean_nca_cc/training/preconfigure.py` (preconfigure_circuit_logits)

**Pattern** (from `train_loop.py` lines 1065-1082):
```python
from boolean_nca_cc.circuits.model import gen_circuit
from boolean_nca_cc.training.preconfigure import preconfigure_circuit_logits

# For growth mode: random circuit
if config.training.training_mode == "growth":
    wiring_key = jax.random.PRNGKey(config.get("wiring_fixed_key", 42))
    base_wires, base_logits = gen_circuit(
        wiring_key, 
        layer_sizes, 
        arity=config.circuit.arity
    )

# For repair mode: preconfigured circuit
elif config.training.training_mode == "repair":
    base_wires, base_logits = preconfigure_circuit_logits(
        wiring_key=wiring_key,
        layer_sizes=layer_sizes,
        arity=config.circuit.arity,
        x_data=x_data,  # Use full data for preconfig
        y_data=y_data,
        loss_type=config.training.loss_type,
        steps=config.training.preconfig_steps,
        lr=config.training.preconfig_lr,
        # ... other backprop params from config.backprop
    )
```

### 4. Evaluation Functions

**Functions**: `evaluate_model_stepwise_batched()`, `evaluate_circuits_in_chunks()`

**Location**: `boolean_nca_cc/training/evaluation.py`

**Pattern for `eval_no_damage`** (from `train_loop.py` lines 1678-1743):
```python
from boolean_nca_cc.training.evaluation import evaluate_circuits_in_chunks, evaluate_model_stepwise_batched

# Replicate base circuit for batch
eval_batch_size = config.eval.periodic_eval_batch_size
eval_wires = jax.tree.map(
    lambda x: jp.repeat(x[None, ...], eval_batch_size, axis=0), 
    base_wires
)
eval_logits = jax.tree.map(
    lambda x: jp.repeat(x[None, ...], eval_batch_size, axis=0), 
    base_logits
)

# Run evaluation (no damage)
step_metrics = evaluate_circuits_in_chunks(
    eval_fn=evaluate_model_stepwise_batched,
    wires=eval_wires,
    logits=eval_logits,
    knockout_patterns=None,  # No damage
    target_chunk_size=eval_batch_size,
    model=model,
    x_data=x_test,  # or x_train for train metrics
    y_data=y_test,  # or y_train
    input_n=config.circuit.input_bits,
    arity=config.circuit.arity,
    circuit_hidden_dim=config.circuit.circuit_hidden_dim,
    n_message_steps=config.eval.periodic_eval_inner_steps,
    loss_type=config.training.loss_type,
    layer_sizes=layer_sizes,
    return_per_pattern=False,
    layer_neighbors=config.training.layer_neighbors,
    # Disable damage injection
    damage_mode="greedy",  # Won't matter
    damage_injection_mode="single",
    max_damage_per_circuit=1,
    greedy_ordered_indices=None,
    knockout_vocabulary=None,
)

# Extract metrics
final_hard_accuracy = step_metrics["hard_accuracy"][-1]
final_loss = step_metrics["soft_loss"][-1]
```

**Pattern for `eval_ko_in`/`eval_ko_out`** (from `train_loop.py` lines 1587-1615):
```python
# Requires knockout vocabulary (from config.pool.damage_knockout_diversity)
from boolean_nca_cc.training.pool.structural_perturbation import create_knockout_vocabulary

# Generate vocabulary if needed (matches training)
if config.pool.damage_knockout_diversity > 0:
    vocab_rng = jax.random.PRNGKey(config.damage_seed)
    knockout_vocabulary = create_knockout_vocabulary(
        rng=vocab_rng,
        vocabulary_size=config.pool.damage_knockout_diversity,
        layer_sizes=layer_sizes,
        damage_prob=config.pool.damage_prob,
        damage_mode=config.pool.damage_mode,
        ordered_indices=config.pool.greedy_ordered_indices,
    )

# For eval_ko_in: use vocabulary
# For eval_ko_out: set knockout_vocabulary=None (generates fresh patterns)
```

### 5. Layer Sizes Generation

**Function**: `generate_layer_sizes()`

**Location**: `boolean_nca_cc/circuits/model.py`

**Pattern**:
```python
from boolean_nca_cc.circuits.model import generate_layer_sizes

if config.circuit.layer_sizes is None:
    layer_sizes = generate_layer_sizes(
        input_n=config.circuit.input_bits,
        output_n=config.circuit.output_bits,
        arity=config.circuit.arity,
        layer_n=config.circuit.num_layers,
    )
else:
    layer_sizes = config.circuit.layer_sizes
```

## Base Script Template

```python
"""Base script for loading checkpoints and running evaluations."""

import jax
import jax.numpy as jp
from omegaconf import OmegaConf

from boolean_nca_cc.circuits.tasks import get_task_data
from boolean_nca_cc.circuits.data_split import split_input_combinations
from boolean_nca_cc.circuits.model import gen_circuit, generate_layer_sizes
from boolean_nca_cc.training.checkpointing import (
    load_config_from_wandb,
    load_model_from_config_and_checkpoint,
    derive_checkpoint_metric_from_config,
)
from boolean_nca_cc.training.evaluation import evaluate_circuits_in_chunks, evaluate_model_stepwise_batched

def load_model_and_data(run_id: str, use_best_model: bool = True):
    """Load model, config, and prepare data splits."""
    # 1. Load config and model
    if use_best_model:
        temp_config, _, _ = load_config_from_wandb(run_id=run_id, filename="latest_checkpoint")
        metric_name, prefer_metric = derive_checkpoint_metric_from_config(temp_config)
        config, checkpoint_path, run_id = load_config_from_wandb(
            run_id=run_id,
            filename="best_model",
            select_by_best_metric=True,
            metric_name=metric_name,
            prefer_metric=prefer_metric,
        )
    else:
        config, checkpoint_path, run_id = load_config_from_wandb(
            run_id=run_id,
            filename="latest_checkpoint",
        )
    
    model, loaded_dict = load_model_from_config_and_checkpoint(
        config=config,
        checkpoint_path=checkpoint_path,
        run_id=run_id,
    )
    
    # 2. Generate data and splits
    x_data, y_data = get_task_data(...)
    if config.eval.input_split_enabled:
        x_train, y_train, x_test, y_test = split_input_combinations(...)
    else:
        x_train, y_train = x_data, y_data
        x_test, y_test = x_data, y_data
    
    # 3. Generate base circuit
    layer_sizes = generate_layer_sizes(...) if config.circuit.layer_sizes is None else config.circuit.layer_sizes
    base_wires, base_logits = gen_circuit(...)  # or preconfigure_circuit_logits(...)
    
    return model, config, x_train, y_train, x_test, y_test, base_wires, base_logits, layer_sizes

def run_eval_no_damage(model, base_wires, base_logits, x_data, y_data, config, layer_sizes):
    """Run no-damage evaluation matching train_loop.py pattern."""
    # Replicate circuit, run evaluation, extract metrics
    # See pattern in Section 4 above
    pass
```

## Key Configuration Parameters

Extract from `config` object:
- `config.circuit.*` - Circuit parameters
- `config.training.*` - Training parameters (loss_type, training_mode, etc.)
- `config.eval.*` - Evaluation parameters (periodic_eval_inner_steps, input_split_enabled, etc.)
- `config.pool.*` - Pool/damage parameters (damage_mode, damage_knockout_diversity, etc.)

## Common Pitfalls

1. **Wrong checkpoint**: Always use `derive_checkpoint_metric_from_config()` to get correct metric name
2. **Data mismatch**: Ensure data generation uses same seeds/parameters as training
3. **Base circuit mismatch**: Repair mode requires preconfigured circuit, growth mode uses random
4. **Vocabulary mismatch**: Knockout vocabulary must use same `damage_seed` as training
5. **Layer sizes**: Must match exactly (use `generate_layer_sizes()` if `config.circuit.layer_sizes` is None)

## References

- Model loading: `boolean_nca_cc/training/checkpointing.py` (lines 775-850)
- Data splitting: `boolean_nca_cc/circuits/data_split.py`
- Evaluation: `boolean_nca_cc/training/evaluation.py`
- Training loop patterns: `boolean_nca_cc/training/train_loop.py` (lines 817-1750)
- Working example: `GUI_minimal.py` (lines 863-994)


