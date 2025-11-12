# Align GUI PERTURB Button with Evaluation Loop Damage Handling

This document outlines the specific changes needed to align the GUI's PERTURB button with the evaluation loop's multi-inject damage handling in `evaluation.py`.

---

## Goal

Make the PERTURB button use **exactly the same perturbation injection and state management** as the periodic evaluation loop, enabling proper reversible mode activation and recovery behavior.

---

## Current Mismatch

| Aspect | Evaluation Loop (`_evaluate_with_loop`) | GUI PERTURB Button |
|--------|----------------------------------------|-------------------|
| **Damage injection** | Passes `knockout_pattern` to model | Bakes damage into logits |
| **Pattern handling** | Model receives pattern via `knockout_pattern` parameter | Model never sees pattern |
| **Step counter** | Resets `update_steps` to 0 on injection | No step counter manipulation |
| **Reversible mode** | Model applies one-shot bias when `update_steps == 0` | Reversible mode never activates |
| **State source** | Damage applied to current state | Damage applied to `logits0` (baseline) |

**Result**: GUI uses a completely different damage mechanism, so recovery behavior cannot match.

---

## Required Changes

### 1. Extend `evaluate_model_stepwise_generator()` in `evaluation.py`

**File**: `boolean_nca_cc/training/evaluation.py`  
**Function**: `evaluate_model_stepwise_generator()` (lines 135-304)

**Add Parameters**:
```python
knockout_pattern: Optional[jp.ndarray] = None,
reset_step_counter_on_init: bool = False,
```

**Modify Model Call** (line ~247-255):
```python
# BEFORE:
if isinstance(model, CircuitSelfAttention):
    updated_graph = model(
        graph,
        layer_neighbors=layer_neighbors,
        layer_sizes=layer_sizes,
    )

# AFTER:
if isinstance(model, CircuitSelfAttention):
    updated_graph = model(
        graph,
        knockout_pattern=knockout_pattern,  # ← Pass pattern to model
        layer_neighbors=layer_neighbors,
        layer_sizes=layer_sizes,
    )
```

**Add Step Counter Reset Logic** (line ~202-206):
```python
# BEFORE:
current_update_steps = 0
graph = graph._replace(
    globals=jp.array([initial_loss, current_update_steps], dtype=jp.float32)
)

# AFTER:
current_update_steps = 0
if reset_step_counter_on_init:
    current_update_steps = 0  # Force reset to 0 for reversible mode
graph = graph._replace(
    globals=jp.array([initial_loss, current_update_steps], dtype=jp.float32)
)
```

---

### 2. Update `_apply_gate_damage_perturbation()` in `GUI_minimal.py`

**File**: `GUI_minimal.py`  
**Function**: `_apply_gate_damage_perturbation()` (lines 1367-1447)

**Remove Logit Baking** (lines 1405-1422):
```python
# DELETE THIS ENTIRE SECTION:
# 3) Bake damage into current logits (reversible)
# Start from logits0 to ensure we don't accumulate damage
damaged_logits = [l.copy() for l in self.logits0]
# ... (all the logit baking code) ...
self.logits = damaged_logits
```

**Store Pattern Instead**:
```python
# AFTER pattern generation (line ~1399):
pattern = create_greedy_subset_random_pattern(...)

# Store pattern for generator (ADD THIS):
self._current_knockout_pattern = pattern

# Keep visualization mask (lines 1400-1403):
layer_gate_masks = create_gate_mask_from_knockout_pattern(pattern, layer_sizes_list)
self._viz_damage_mask = [m.astype(np.float32) for m in layer_gate_masks]
```

**Update Generator Initialization** (line ~1436):
```python
# BEFORE:
self.model_generator = evaluate_model_stepwise_generator(
    model=self.frozen_model,
    wires=self.wires,
    logits=self.logits,
    # ... other params ...
)

# AFTER:
self.model_generator = evaluate_model_stepwise_generator(
    model=self.frozen_model,
    wires=self.wires,
    logits=self.logits,  # Current logits (NOT damaged, NOT logits0)
    x_data=self.input_x,
    y_data=self.y0,
    input_n=self.input_n,
    arity=self.arity,
    circuit_hidden_dim=hidden_dim_for_graph,
    max_steps=None,
    loss_type=self.loss_type,
    bidirectional_edges=True,
    layer_sizes=self.layer_sizes,
    layer_neighbors=False,
    knockout_pattern=self._current_knockout_pattern,  # ← ADD THIS
    reset_step_counter_on_init=True,  # ← ADD THIS
)
```

---

### 3. Add Pattern Storage to GUI State

**File**: `GUI_minimal.py`

**In `__init__` or initialization section** (~line 300):
```python
# ADD:
self._current_knockout_pattern = None
```

**In `_apply_gate_damage_perturbation()`** (~line 1399):
```python
# After pattern generation:
self._current_knockout_pattern = pattern
```

**In `reset_circuit()`** (~line 1459):
```python
# ADD:
self._current_knockout_pattern = None
```

---

### 4. Update `initialize_model_generator()` in `GUI_minimal.py`

**File**: `GUI_minimal.py`  
**Function**: `initialize_model_generator()` (lines 800-843)

**Modify Generator Initialization** (line ~815):
```python
# BEFORE:
self.model_generator = evaluate_model_stepwise_generator(
    model=self.frozen_model,
    wires=self.wires,
    logits=self.logits,
    # ... other params ...
)

# AFTER:
self.model_generator = evaluate_model_stepwise_generator(
    model=self.frozen_model,
    wires=self.wires,
    logits=self.logits,
    x_data=self.input_x,
    y_data=self.y0,
    input_n=self.input_n,
    arity=self.arity,
    circuit_hidden_dim=hidden_dim_for_graph,
    max_steps=None,
    loss_type=self.loss_type,
    bidirectional_edges=True,
    layer_sizes=self.layer_sizes,
    layer_neighbors=False,
    knockout_pattern=getattr(self, '_current_knockout_pattern', None),  # ← ADD THIS
    reset_step_counter_on_init=(getattr(self, '_current_knockout_pattern', None) is not None),  # ← ADD THIS
)
```

---

## Implementation Order

1. **Step 1**: Extend generator in `evaluation.py`
   - Add `knockout_pattern` and `reset_step_counter_on_init` parameters
   - Pass pattern to model call
   - Implement step counter reset logic

2. **Step 2**: Update GUI damage function in `GUI_minimal.py`
   - Remove logit baking code
   - Store pattern instead of baking
   - Update generator initialization to pass pattern

3. **Step 3**: Add pattern storage
   - Add `_current_knockout_pattern` attribute
   - Store pattern when damage applied
   - Clear pattern on reset

4. **Step 4**: Update generator initialization
   - Make `initialize_model_generator()` use stored pattern

5. **Step 5**: Test and verify
   - Verify reversible mode activates (check `step_count == 0.0`)
   - Verify recovery occurs after damage
   - Verify trajectory matches eval loop

---

## Success Criteria

- [ ] Generator accepts `knockout_pattern` parameter
- [ ] Generator passes pattern to model
- [ ] Step counter resets to 0 when damage injected
- [ ] GUI stores pattern instead of baking into logits
- [ ] GUI uses current logits (not `logits0`)
- [ ] Model receives pattern and activates reversible mode
- [ ] Recovery visible after damage (accuracy increases over steps)
- [ ] Recovery trajectory matches eval loop behavior

---

## Key State Management Alignment Points

1. **Graph globals format**: `[loss, update_steps]` ✓ (already correct)
2. **Step counter reset**: Must be 0 when damage injected (for reversible mode)
3. **Pattern passing**: Model must receive `knockout_pattern` parameter
4. **Logit state**: Damage applied to current logits, not baseline
5. **Graph initialization**: Uses `build_graph()` with proper loss/step initialization ✓ (already correct)

---

## Related Documents

- [`GUI_damage_recovery_mismatch.md`](GUI_damage_recovery_mismatch.md): Detailed investigation and analysis
- [`Simple_SEU.md`](Simple_SEU.md): Reversible damage mode specification
- [`self_attention_accuracy_drift.md`](self_attention_accuracy_drift.md): Accuracy drift investigation

