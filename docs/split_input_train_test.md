# Strategy: Input Combination Train/Test Split

## Overview

This document outlines the strategy for implementing train/test splits on input combinations, enabling evaluation of circuit generalization to held-out input patterns. This complements the existing knockout pattern in/out-of-distribution evaluation framework.

## Motivation

Currently, the system evaluates:
- **Knockout pattern generalization**: Seen vs unseen damage patterns (`eval_ko_in`/`eval_ko_out`)
- **Input combination generalization**: Not yet implemented

Adding input combination splits enables:
1. Testing whether circuits generalize to unseen input combinations
2. Measuring overfitting to specific input patterns
3. More rigorous evaluation of circuit robustness

## Current Architecture

### Data Flow

1. **Data Generation** (`boolean_nca_cc/circuits/tasks.py`):
   - `get_task_data()` generates all `2^input_bits` input combinations
   - For 8-bit inputs: 256 combinations (complete truth table)
   - `x_data`: `[256, 8]` - all input combinations
   - `y_data`: `[256, 8]` - corresponding outputs

2. **Training** (`boolean_nca_cc/training/train_loop.py`):
   - `x_data` and `y_data` passed directly to `train_model()`
   - Used in `pool_train_step()` via `get_loss_from_wires_logits()`
   - All combinations used for both training and evaluation

3. **Evaluation** (`boolean_nca_cc/training/evaluation.py`):
   - `evaluate_model_stepwise_batched()` accepts `x_data`/`y_data`
   - Currently uses same data for all evaluations
   - `run_knockout_periodic_evaluation()` evaluates with knockout patterns

### Current Naming Convention

- `eval_ko_in`: In-distribution knockout patterns (from vocabulary)
- `eval_ko_out`: Out-of-distribution knockout patterns (fresh generation)
- **Issue**: Names don't distinguish knockout patterns from input combinations

## Proposed Changes

### 1. New Components

#### A. Data Splitting Utility (`boolean_nca_cc/circuits/data_split.py`)

```python
def split_input_combinations(
    x_data: jp.ndarray,
    y_data: jp.ndarray,
    train_fraction: float = 0.8,
    seed: int = 42,
    shuffle: bool = True,
) -> Tuple[jp.ndarray, jp.ndarray, jp.ndarray, jp.ndarray]:
    """
    Split input combinations into train and test sets.
    
    Args:
        x_data: Input combinations [n_combinations, input_bits]
        y_data: Target outputs [n_combinations, output_bits]
        train_fraction: Fraction of combinations for training (default 0.8)
        seed: Random seed for reproducible shuffling
        shuffle: Whether to shuffle before splitting
        
    Returns:
        Tuple of (x_train, y_train, x_test, y_test)
    """
```

**Key Features**:
- Reproducible via seed
- Optional shuffling (useful for debugging)
- Preserves input-output correspondence

#### B. Configuration Parameters (`configs/config.yaml`)

```yaml
eval:
  # Input combination splitting
  input_split_enabled: false  # Enable train/test split on inputs
  input_train_fraction: 0.8   # Fraction of combinations for training
  input_split_seed: 42        # Seed for reproducible split
  
  # Existing knockout evaluation (unchanged)
  periodic_eval_enabled: true
  periodic_eval_inner_steps: 200
  # ...
```

#### C. Training Loop Updates (`boolean_nca_cc/training/train_loop.py`)

**Changes**:
1. Add data splitting parameters to `train_model()`
2. Split data if `input_split_enabled=True`
3. Use training data for loss computation
4. Pass test data to evaluation functions
5. Update evaluation calls to include input combination metrics

**Minimal Changes**:
- Split happens once at start of training
- Training uses `x_train`/`y_train`
- Evaluation receives both train and test sets
- Existing functions unchanged (they accept any `x_data`/`y_data`)

#### D. Evaluation Updates (`boolean_nca_cc/training/evaluation.py`)

**Changes** (implemented in Phase 2):
1. Update `run_knockout_periodic_evaluation()` to accept test input combinations
2. Add evaluation metrics for input combination splits
3. Implement new naming convention to reflect both dimensions (see Naming Convention section below)

### 2. Naming Convention Update (Implemented in Phase 2, Not Phase 1)

**Knockout Patterns** (structural damage):
- `eval_ko_pattern_seen`: Knockout patterns from training vocabulary
- `eval_ko_pattern_unseen`: Fresh knockout patterns (same distribution)

**Input Combinations** (functional inputs):
- `eval_input_train`: Training input combinations (used during training)
- `eval_input_test`: Test input combinations (held out during training)

**Combined Evaluations**:
- `eval_ko_pattern_seen_input_train`: Seen patterns + training inputs
- `eval_ko_pattern_seen_input_test`: Seen patterns + test inputs
- `eval_ko_pattern_unseen_input_train`: Unseen patterns + training inputs
- `eval_ko_pattern_unseen_input_test`: Unseen patterns + test inputs

**Backward Compatibility**:
- Keep `eval_ko_in`/`eval_ko_out` as aliases for `eval_ko_pattern_seen`/`eval_ko_pattern_unseen`
- Document that these refer to knockout patterns, not input combinations

**Evaluation Matrix** (implemented in Phase 2):

| Knockout Pattern | Input Combination | Metric Name |
|-----------------|-------------------|-------------|
| Seen (vocab) | Train | `eval_ko_pattern_seen_input_train` |
| Seen (vocab) | Test | `eval_ko_pattern_seen_input_test` |
| Unseen (fresh) | Train | `eval_ko_pattern_unseen_input_train` |
| Unseen (fresh) | Test | `eval_ko_pattern_unseen_input_test` |

**Note**: This naming convention update is implemented in Phase 2 when evaluation functions are updated. Phase 1 focuses on infrastructure only and preserves existing `eval_ko_in`/`eval_ko_out` naming.

### 3. Implementation Strategy

#### Phase 1: Core Infrastructure (Minimal, Backward Compatible)

1. **Create data splitting utility**
   - `boolean_nca_cc/circuits/data_split.py`
   - Simple, well-tested function
   - No dependencies on other changes

2. **Add configuration parameters**
   - Add to `configs/config.yaml`
   - Default: `input_split_enabled: false` (backward compatible)

3. **Update training loop signature**
   - Add optional parameters for input splitting
   - Default values preserve existing behavior
   - **No naming convention changes** - keep existing `eval_ko_in`/`eval_ko_out` naming

#### Phase 2: Integration (Opt-in Feature)

1. **Integrate splitting into training loop**
   - Split data if enabled
   - Use training data for loss computation
   - Store test data for evaluation

2. **Update evaluation functions**
   - Accept test input combinations as optional parameters
   - Compute metrics for both train and test inputs

3. **Implement naming convention update**
   - Introduce new metric names: `eval_ko_pattern_seen`/`eval_ko_pattern_unseen` (replacing `eval_ko_in`/`eval_ko_out`)
   - When input split enabled, use combined names: `eval_ko_pattern_seen_input_train`, `eval_ko_pattern_seen_input_test`, etc.
   - Keep `eval_ko_in`/`eval_ko_out` as aliases for backward compatibility
   - When input split disabled, continue using `eval_ko_in`/`eval_ko_out` only

4. **Update metric logging**
   - Log all four combinations (seen/unseen × train/test) when input split enabled
   - Maintain backward compatibility with existing metrics (`eval_ko_in`/`eval_ko_out`)

#### Phase 3: Documentation and Testing

1. **Update documentation**
   - Design doc (`docs/design_doc_ko_selfatt.md`)
   - Function docstrings
   - Configuration examples

2. **Add tests**
   - Data splitting correctness
   - Evaluation with splits
   - Backward compatibility

## Implementation Details

### Data Splitting Logic

```python
# Example: 8-bit inputs, 256 combinations, 80% train
n_combinations = 256
n_train = int(256 * 0.8) = 204
n_test = 256 - 204 = 52

# Shuffle indices
indices = [0, 1, 2, ..., 255]  # shuffled
train_indices = indices[:204]
test_indices = indices[204:]

# Split data
x_train = x_data[train_indices]  # [204, 8]
y_train = y_data[train_indices]  # [204, 8]
x_test = x_data[test_indices]    # [52, 8]
y_test = y_data[test_indices]    # [52, 8]
```

### Training Flow

```
train.py
  ↓
get_task_data() → (x_data, y_data) [256, 8]
  ↓
split_input_combinations() → (x_train, y_train, x_test, y_test)
  ↓
train_model(x_train, y_train, ...)
  ↓
pool_train_step() uses x_train, y_train for loss
  ↓
run_knockout_periodic_evaluation(x_test, y_test, ...)
  ↓
evaluate_model_stepwise_batched(x_test, y_test, ...)
```

### Evaluation Flow

```
run_knockout_periodic_evaluation():
  ├─ eval_ko_pattern_seen_input_train:
  │   └─ evaluate_model_stepwise_batched(
  │       knockout_patterns=vocab_patterns,
  │       x_data=x_train, y_data=y_train
  │   )
  ├─ eval_ko_pattern_seen_input_test:
  │   └─ evaluate_model_stepwise_batched(
  │       knockout_patterns=vocab_patterns,
  │       x_data=x_test, y_data=y_test
  │   )
  ├─ eval_ko_pattern_unseen_input_train:
  │   └─ evaluate_model_stepwise_batched(
  │       knockout_patterns=fresh_patterns,
  │       x_data=x_train, y_data=y_train
  │   )
  └─ eval_ko_pattern_unseen_input_test:
      └─ evaluate_model_stepwise_batched(
          knockout_patterns=fresh_patterns,
          x_data=x_test, y_data=y_test
      )
```

## Backward Compatibility

### Default Behavior (No Changes)

- `input_split_enabled: false` → Uses all data for training and evaluation
- Existing metrics (`eval_ko_in`/`eval_ko_out`) continue to work throughout all phases
- No breaking changes to function signatures (all new parameters optional)
- **Phase 1**: No naming convention changes - keep `eval_ko_in`/`eval_ko_out` as-is
- **Phase 2**: New naming convention introduced, with `eval_ko_in`/`eval_ko_out` as aliases for backward compatibility
- Phase 1: No naming convention changes - existing names preserved
- Phase 2: New naming convention added alongside existing names (backward compatible)

### Migration Path

1. **Existing experiments**: Continue working unchanged
2. **New experiments**: Enable `input_split_enabled: true` to get additional metrics
3. **Gradual adoption**: Can enable/disable per experiment

## Configuration Examples

### Example 1: Disabled (Current Behavior)

```yaml
eval:
  input_split_enabled: false
  periodic_eval_enabled: true
```

**Result**: All 256 input combinations used for both training and evaluation.

### Example 2: Enabled with 80/20 Split

```yaml
eval:
  input_split_enabled: true
  input_train_fraction: 0.8
  input_split_seed: 42
  periodic_eval_enabled: true
```

**Result**: 
- Training: 204 combinations
- Test: 52 combinations
- Evaluation metrics for both sets

### Example 3: Enabled with 50/50 Split

```yaml
eval:
  input_split_enabled: true
  input_train_fraction: 0.5
  input_split_seed: 42
```

**Result**: 
- Training: 128 combinations
- Test: 128 combinations

## Testing Strategy

### Unit Tests

1. **Data Splitting**:
   - Correct train/test sizes
   - Reproducibility with same seed
   - Input-output correspondence preserved
   - Edge cases (fraction=0.0, 1.0)

2. **Evaluation**:
   - Metrics computed correctly for both sets
   - Naming convention followed
   - Backward compatibility maintained

### Integration Tests

1. **End-to-end training**:
   - Training uses train set only
   - Evaluation uses both train and test sets
   - Metrics logged correctly

2. **Backward compatibility**:
   - Existing configs work unchanged
   - Disabled split preserves current behavior

## Metrics and Logging

### WandB Metrics

**When input splits are enabled** (Phase 2), new metrics will be logged:
- `eval_ko_pattern_seen_input_train/final_hard_accuracy`
- `eval_ko_pattern_seen_input_test/final_hard_accuracy`
- `eval_ko_pattern_unseen_input_train/final_hard_accuracy`
- `eval_ko_pattern_unseen_input_test/final_hard_accuracy`

Plus corresponding metrics for:
- `final_loss`
- `final_hard_loss`
- `final_accuracy`

**Backward compatibility**: Existing metrics (`eval_ko_in`/`eval_ko_out`) continue to work unchanged when splits are disabled.

### Metric Interpretation

- **Train vs Test gap**: Measures overfitting to training inputs
- **Seen vs Unseen gap**: Measures generalization to new damage patterns
- **Combined analysis**: Identifies whether circuits generalize better to new inputs or new damage patterns

## Future Enhancements

1. **Stratified splitting**: Ensure balanced representation of input patterns
2. **Cross-validation**: K-fold evaluation across input combinations
3. **Adaptive splitting**: Focus test set on difficult input combinations
4. **Visualization**: Plot train/test performance over time

## Summary

This strategy provides a minimal, backward-compatible implementation that:
- ✅ Adds input combination train/test splits
- ✅ Maintains backward compatibility
- ✅ Uses clear naming conventions
- ✅ Maximally reuses existing components
- ✅ Enables comprehensive evaluation across both dimensions

The implementation is opt-in via configuration, allowing gradual adoption without disrupting existing experiments.

