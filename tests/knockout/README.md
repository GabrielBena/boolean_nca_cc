# Gate Knockout Testing Suite

This directory contains comprehensive tests for the gate knockout functionality in the boolean circuit training system.

## Overview

Gate knockout is a mechanism for permanently disabling specific gates in a boolean circuit during training. This simulates hardware failure or structural damage where certain gates become non-functional and cannot be recovered through normal training processes.

## ⚡ Performance Optimization

**Important**: The `update_params` function has been significantly optimized based on the key insight that gradients are automatically zero for knocked out gates. This results in **55.9% faster performance** with identical correctness. See `PERFORMANCE_OPTIMIZATION.md` for details.

## Key Features Tested

### 1. Permanent Structural Damage
- **Gate Knockout**: Gates are permanently "broken" and cannot be updated by the model
- **Faulty Logits**: Knocked out gates have their logits set to values that produce zero output
- **Update Prevention**: Knocked out gates are prevented from receiving model updates during training

### 2. Gradient Masking
- **Zero Gradients**: Knocked out gates have zero gradients during backpropagation
- **Parameter Preservation**: Original faulty logit values are preserved despite optimization updates
- **Weight Decay Protection**: Weight decay doesn't affect knocked out gates

### 3. Circuit Computation
- **Output Zeroing**: Knocked out gates always produce zero output regardless of input
- **Logit Independence**: Output is the same regardless of the actual logit values for knocked out gates

## Test Files

### `test_gate_knockout.py`
Comprehensive tests for core gate knockout functionality:

- `test_knockout_pattern_creation()`: Tests creation of reproducible knockout patterns
- `test_knocked_out_gates_zero_output()`: Verifies knocked out gates produce zero output
- `test_knockout_independence_from_logits()`: Tests output independence from logit values
- `test_knockout_gradient_masking()`: Verifies zero gradients for knocked out gates
- `test_knockout_prevents_parameter_updates()`: Tests parameter preservation during training
- `test_weight_decay_handling()`: Verifies weight decay doesn't affect knocked out gates
- `test_reproducible_knockout_patterns()`: Tests pattern reproducibility
- `test_edge_cases()`: Tests edge cases (zero knockouts, excessive knockouts, etc.)
- `test_faulty_gate_logits_creation()`: Tests creation of faulty logit values

### `test_gradient_masking.py`
Tests for parameter update integration and optimizer compatibility:

- `test_gradient_masking_in_update_params()`: Tests parameter preservation in updates
- `test_weight_decay_with_masked_updates()`: Tests weight decay handling with masking
- `test_train_step_integration()`: Tests integration with the training step
- `test_different_optimizer_types()`: Tests compatibility with different optimizers
- `test_gradient_computation_with_knockout()`: Tests gradient computation with knockouts
- `test_multiple_training_steps_preserve_knockouts()`: Tests preservation over multiple steps

### `test_gradient_assumption.py`
Verification that gradients are automatically zero for knocked out gates:

- Confirms that no gradient masking is needed in `update_params`
- Validates the optimization insight that led to 55.9% performance improvement
- Demonstrates gradient flow isolation for knocked out gates

## Key Functions Modified

### `update_params()` in `circuits/train.py`
**Optimized** to handle gate masks efficiently:
- ✅ **No gradient masking needed** (gradients are automatically zero)
- ✅ **Only preserves parameter values** after optimizer updates
- ✅ **Prevents optimizer internal state** (momentum, weight decay) from affecting knocked out gates
- ✅ **55.9% performance improvement** over previous implementation
- ✅ **JIT-compiled** parameter preservation for maximum speed

### `train_step()` in `circuits/train.py`
Updated to pass gate masks to `update_params()` function.

## Usage Example

```python
import jax
from boolean_nca_cc.circuits.model import gen_circuit, generate_layer_sizes
from boolean_nca_cc.circuits.train import train_step, TrainState
from boolean_nca_cc.training.pool.structural_perturbation import (
    apply_gate_knockout_to_pool_element
)

# Create circuit
key = jax.random.PRNGKey(42)
layer_sizes = generate_layer_sizes(input_n=8, output_n=4, arity=4, layer_n=2)
wires, logits = gen_circuit(key, layer_sizes, arity=4)

# Apply gate knockout
faulty_logits, _, knockout_masks = apply_gate_knockout_to_pool_element(
    key, logits, wires, layer_sizes, number_knockouts=3
)

# Train with knockout protection
optimizer = optax.adam(learning_rate=0.01)
train_state = TrainState(faulty_logits, optimizer.init(faulty_logits))

# Training step automatically preserves knocked out gates
loss, aux, new_state = train_step(
    train_state, optimizer, wires, x, y0, 
    gate_mask=knockout_masks  # This protects knocked out gates
)
```

## Running the Tests

### Option 1: Using the test runner script
```bash
cd tests/knockout
python run_tests.py
```

### Option 2: Using pytest directly
```bash
cd boolean_nca_cc
python -m pytest tests/knockout/ -v
```

### Option 3: Running specific test files
```bash
python -m pytest tests/knockout/test_gate_knockout.py -v
python -m pytest tests/knockout/test_gradient_masking.py -v
```

## Test Requirements

The tests require the following packages:
- `jax`
- `jax.numpy` 
- `optax`
- `pytest`
- `numpy`

Make sure to activate the appropriate conda environment before running tests:
```bash
conda activate nca
```

## Expected Behavior

All tests should pass, demonstrating that:

1. **Structural Integrity**: Knocked out gates maintain their faulty state throughout training
2. **Gradient Safety**: No gradients flow through knocked out gates
3. **Parameter Safety**: Optimizer updates don't affect knocked out gates
4. **Weight Decay Safety**: Weight decay doesn't modify knocked out gate parameters
5. **Computational Correctness**: Knocked out gates always produce zero output

## Troubleshooting

If tests fail, check:

1. **Environment**: Ensure the `nca` conda environment is activated
2. **Dependencies**: Verify all required packages are installed
3. **JAX Version**: Ensure compatible JAX version is installed
4. **Path Issues**: Run tests from the project root directory

## Implementation Notes

### Gate Mask Format
Gate masks use a layered format where each layer has its own mask array:
- Input layer: Always all 1.0 (never knocked out)
- Hidden layers: 0.0 for knocked out gates, 1.0 for active gates  
- Output layer: Always all 1.0 (never knocked out)

### Faulty Logit Values
Knocked out gates have their logits set to large negative values (default: -10.0) so that after sigmoid transformation, they produce values close to 0.0.

### Gradient Masking Strategy
The `update_params()` function implements a two-stage masking strategy:
1. **Gradient Masking**: Zero out gradients for knocked out gates before optimizer update
2. **Parameter Preservation**: Restore original values for knocked out gates after optimizer update

This ensures that neither the gradients nor the optimizer's internal state (momentum, etc.) can affect knocked out gates.
