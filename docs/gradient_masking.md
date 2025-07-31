# Gradient Masking Analysis in Boolean Circuits

## Executive Summary

This report analyzes two approaches to gradient masking in differentiable boolean circuits: explicit gradient masking versus natural gradient flow through gate masking. We demonstrate that explicit gradient masking is redundant and potentially problematic, as the natural gradient flow through `gate_mask` already produces identical results with better numerical properties.

## Background

Boolean circuits in this framework consist of:
- **Gates**: Lookup tables (LUTs) parameterized by logits
- **Wires**: Connections between gates across layers
- **Gate Masking**: Element-wise multiplication that zeros out specific gate outputs

## Two Approaches to Gradient Masking

### Approach 1: Explicit Gradient Masking
```python
def apply_knockout_mask_to_gradients(grads, knockout_pattern, layer_sizes):
    """Explicitly zero out gradients for knocked out parameters"""
    masked_grads = []
    for grad_layer, (total_gates, group_size) in zip(grads, layer_sizes):
        active_mask = ~knockout_pattern
        masked_grad_layer = grad_layer * active_mask[:, None, None]
        masked_grads.append(masked_grad_layer)
    return masked_grads
```

### Approach 2: Natural Gradient Flow (Gate Masking)
```python
def run_circuit(logits, wires, x, gate_mask=None, hard=False):
    """Circuit execution with natural gate masking"""
    for ws, lgt, mask in zip(wires, logits, gate_mask[1:]):
        luts = jax.nn.sigmoid(lgt)
        x = run_layer(luts, [x[..., w] for w in ws]) * mask  # Gate mask applied
    return acts
```

## Mathematical Analysis

### Natural Gradient Flow

When a gate is masked to `0.0`:
```
Forward Pass:
gate_output = sigmoid(logit) * f(inputs) * mask  # mask = 0.0 for disabled gates

Backward Pass (Automatic):
∂gate_output/∂logit = sigmoid'(logit) * f(inputs) * mask = 0.0  # Naturally zero
```

### Explicit Gradient Masking

```python
# Forward pass (same as natural)
gate_output = sigmoid(logit) * f(inputs) * mask

# Backward pass (automatic + explicit)
grad = ∂Loss/∂logit  # JAX computes this
grad = grad * active_mask  # Explicitly zero it
```

**Result**: Both approaches produce `grad = 0.0` for disabled gates.

## Chain Rule Analysis

For a 3-layer circuit with gate `g1` disabled:

```
Normal Flow:
Loss → o0 → g0 → logit0  ✓ (gets gradient)
Loss → o0 → g1 → logit1  ✓ (gets gradient)  
Loss → o1 → g1 → logit1  ✓ (gets gradient)
Loss → o1 → g2 → logit2  ✓ (gets gradient)

With g1 disabled (gate_mask):
Loss → o0 → g0 → logit0  ✓ (gets gradient)
Loss → o0 → g1 → logit1  ✗ (gradient = 0)
Loss → o1 → g1 → logit1  ✗ (gradient = 0)  
Loss → o1 → g2 → logit2  ✓ (gets gradient)
```

## Key Findings

### 1. Mathematical Equivalence
Both approaches produce identical results: zero gradients for disabled gates.

### 2. Performance Comparison

| Aspect | Natural Flow | Explicit Masking |
|--------|--------------|------------------|
| Computational Overhead | None | Extra operation per gradient |
| Numerical Stability | Better | Potential precision issues |
| Optimizer Compatibility | Full | May affect momentum/statistics |
| Code Complexity | Lower | Higher |

### 3. Gradient Flow Characteristics

**Natural Flow**:
- Gradients are computed simultaneously for all parameters
- Zero gradients arise naturally from the computational graph
- No physical "flow" of gradients through the network

**Explicit Masking**:
- Additional post-processing step
- Redundant computation of already-zero gradients
- Potential interference with optimizer state tracking

## Implementation Recommendations

### Recommended Approach: Natural Gradient Flow

```python
def train_step(state, opt, wires, x, y0, gate_mask=None):
    """Simplified training step using natural gradient flow"""
    logits, opt_state = state
    
    # Compute loss and gradients with gate masking
    (loss, aux), grad = grad_loss_f(logits, wires, x, y0, gate_mask)
    
    # NO explicit gradient masking needed - JAX handles it automatically
    
    # Update parameters
    upd, new_opt_state = opt.update(grad, opt_state, logits)
    new_logits = optax.apply_updates(logits, upd)
    
    return loss, aux, TrainState(new_logits, new_opt_state)
```

### When Explicit Masking Might Be Useful

1. **Non-differentiable operations** that bypass automatic differentiation
2. **Custom gradient functions** that don't respect masking
3. **Debugging and verification** of natural gradient behavior

## Conclusion

Explicit gradient masking is redundant in differentiable boolean circuits. The natural gradient flow through `gate_mask` provides:

- ✅ **Mathematical equivalence** to explicit masking
- ✅ **Better performance** (no extra operations)
- ✅ **Improved numerical properties** (no precision issues)
- ✅ **Full optimizer compatibility** (no interference with state)
- ✅ **Cleaner code** (less complexity)

**Recommendation**: Remove explicit gradient masking and rely on JAX's natural automatic differentiation behavior through gate masking.

## References

- JAX Automatic Differentiation: https://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html
- Boolean Circuit Framework: `boolean_nca_cc/circuits/model.py`
- Training Framework: `boolean_nca_cc/circuits/train.py` 