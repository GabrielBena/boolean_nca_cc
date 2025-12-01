# Permanent Damage Mode Validation Test Plan

## Overview

This document outlines a comprehensive test suite to verify that permanent damage mode correctly ensures that once a damage pattern is applied, the involved nodes are truly permanently damaged. This requires verification across multiple components: damage accumulation, model forward pass, attention masking, circuit extraction, training loop integration, pool management, and evaluation.

## Goal

Ensure that permanent damage mode:
1. **Truly permanently clamps damaged nodes to -10** in the graph, which results in gate outputs being zeroed in circuit mode
2. **Prevents damaged nodes from changing features** at any message passing step
3. **Produces zeroed LUTs** for damaged gates when circuits are converted from graphs
4. **Preserves damage patterns** across perturbations, both in the pool during training and in eval

---

## Critical Verification Points

### 1. Damage Pattern Accumulation (`boolean_nca_cc/training/pool/pool.py`)

**Location**: `GraphPool.apply_knockouts()` (lines 379-426)

**Purpose**: Verify that damage patterns accumulate correctly in permanent mode and that the accumulation logic works as expected.

**Test Cases**:

- **Test 1.1**: Verify `accumulate=True` uses bitwise OR (`|`) to merge patterns
  - Apply pattern A to a circuit
  - Apply pattern B with `accumulate=True`
  - Assert: Final pattern = A | B
  
- **Test 1.2**: Verify `accumulate=False` replaces patterns (low priory, bump to bottom)
  - Apply pattern A to a circuit
  - Apply pattern B with `accumulate=False`
  - Assert: Final pattern = B (not A | B)
  
- **Test 1.3**: Verify `perturb_counter` increments correctly
  - Apply damage to a circuit
  - Assert: `perturb_counter` increases by 1
  - Apply damage again
  - Assert: `perturb_counter` increases by 1 again
  
- **Test 1.4**: Verify accumulation works across multiple damage events
  - Apply pattern A
  - Apply pattern B with `accumulate=True`
  - Apply pattern C with `accumulate=True`
  - Assert: Final pattern = A | B | C
  
- **Test 1.5**: Verify patterns persist after pool updates
  - Apply damage to circuits in pool
  - Update pool with new graphs
  - Assert: Knockout patterns remain unchanged

**Code Reference**:
```python
# boolean_nca_cc/training/pool/pool.py:379-426
@partial(jax.jit, static_argnames=("accumulate",))
def apply_knockouts(
    self,
    idxs: Array,
    new_knockout_patterns: Array,
    accumulate: bool = False,
) -> "GraphPool":
```

---

### 2. Damage Application in Model Forward Pass (`boolean_nca_cc/models/self_attention.py`)

**Location**: `CircuitSelfAttention.__call__()` (lines 466-485)

**Purpose**: Verify that damaged nodes are correctly clamped to -10.0 and that their updates are zeroed out.

**Test Cases**:

- **Test 2.1**: Verify damaged nodes have logits clamped to -10.0
  - Create graph with damage pattern
  - Apply model forward pass
  - Assert: Damaged nodes have `logits == -10.0`
  
- **Test 2.2**: Verify damaged nodes have hidden features zeroed
  - Create graph with damage pattern
  - Apply model forward pass
  - Assert: Damaged nodes have `hidden == 0.0`
  
- **Test 2.3**: Verify logit/hidden updates are zeroed for damaged nodes
  - Create graph with damage pattern
  - Apply model forward pass
  - Assert: Updates for damaged nodes are zero
  
- **Test 2.4**: Verify damaged nodes remain clamped after multiple message passing steps
  - Create graph with damage pattern
  - Apply model N times (N > 1)
  - Assert: Damaged nodes remain at -10.0 logits across all steps
  
- **Test 2.5**: Verify clamping persists even if updates try to modify damaged nodes
  - Create graph with damage pattern
  - Force large positive updates (simulate gradient flow)
  - Apply model forward pass
  - Assert: Damaged nodes remain at -10.0 despite large updates

**Code Reference**:
```python
# boolean_nca_cc/models/self_attention.py:466-485
if knockout_pattern is not None:
    active_mask = ~knockout_pattern
    if self.damage_behavior == "permanent":
        large_negative_value = -10.0
        current_logits = jp.where(
            active_mask[:, None],
            nodes["logits"],
            large_negative_value,
        )
        current_hidden = nodes["hidden"] * active_mask[:, None]
        logit_updates = logit_updates * active_mask[:, None]
        hidden_updates = hidden_updates * active_mask[:, None]
        updated_logits = current_logits + self.logit_scale * logit_updates
        updated_hidden = current_hidden + self.hidden_scale * hidden_updates
```

---

### 3. Attention Mask Exclusion (`boolean_nca_cc/models/self_attention.py`)

**Location**: `CircuitSelfAttention._create_attention_mask()` (lines 290-347)

**Purpose**: Verify that damaged nodes are excluded from attention computation (cannot send or receive attention).

**Test Cases**:

- **Test 3.1**: Verify damaged nodes cannot send attention (rows masked)
  - Create attention mask with damage pattern
  - Assert: Mask rows for damaged nodes are all False
  
- **Test 3.2**: Verify damaged nodes cannot receive attention (columns masked)
  - Create attention mask with damage pattern
  - Assert: Mask columns for damaged nodes are all False
  
- **Test 3.3**: Verify attention mask is applied consistently across all attention layers
  - Create model with multiple attention layers
  - Apply damage pattern
  - Assert: All layers use the same mask
  
- **Test 3.4**: Verify mask persists across multiple message passing steps
  - Create graph with damage pattern
  - Apply model N times
  - Assert: Attention mask remains consistent across steps

**Code Reference**:
```python
# boolean_nca_cc/models/self_attention.py:332-343
if knockout_pattern is not None:
    active_nodes_mask = ~knockout_pattern
    knockout_mask = jp.outer(active_nodes_mask, active_nodes_mask)
    mask = mask & knockout_mask
```

---

### 4. Circuit Extraction from Damaged Graphs (`boolean_nca_cc/utils/extraction.py`)

**Location**: `extract_logits_from_graph()` (lines 13-70)

**Purpose**: Verify that logits extracted from damaged graphs maintain the -10.0 clamping and convert correctly to LUTs.

**Test Cases**:

- **Test 4.1**: Verify extracted logits for damaged nodes are -10.0
  - Create graph with damage pattern
  - Extract logits using `extract_logits_from_graph`
  - Assert: Extracted logits for damaged nodes == -10.0
  
- **Test 4.2**: Verify extraction works correctly with accumulated damage patterns
  - Create graph with accumulated damage (multiple patterns OR'd)
  - Extract logits
  - Assert: All damaged nodes have -10.0 logits
  
- **Test 4.3**: Verify damaged nodes maintain -10.0 across multiple extraction calls
  - Create graph with damage pattern
  - Extract logits multiple times
  - Assert: Damaged nodes consistently have -10.0 logits
  
- **Test 4.4**: Verify LUT conversion (sigmoid of -10.0) produces values ≈ 0
  - Extract logits from damaged graph
  - Convert to LUTs: `luts = sigmoid(logits)`
  - Assert: LUTs for damaged nodes ≈ 0.0 (within numerical precision)

**Code Reference**:
```python
# boolean_nca_cc/utils/extraction.py:13-70
def extract_logits_from_graph(
    graph: jraph.GraphsTuple, logits_original_shapes: list[tuple[int, int, int]]
) -> list[jp.ndarray]:
```

---

### 5. Training Loop Damage Integration (`boolean_nca_cc/training/train_loop.py`)

**Location**: Damage application (lines 1840-1849) and pattern sampling (line 1708)

**Purpose**: Verify that damage patterns are correctly integrated into the training loop and that accumulation behavior is properly determined.

**Test Cases**:

- **Test 5.1**: Verify `damage_behavior` is correctly read from model
  - Create model with `damage_behavior="permanent"`
  - Check behavior in training loop
  - Assert: `damage_behavior == "permanent"`
  
- **Test 5.2**: Verify `accumulate=True` is passed when `damage_behavior == "permanent"`
  - Create model with `damage_behavior="permanent"`
  - Apply damage in training loop
  - Assert: `accumulate=True` is passed to `apply_knockouts`
  
- **Test 5.3**: Verify knockout patterns are sampled from pool and passed to training step
  - Sample batch from pool
  - Assert: Knockout patterns are included in sample
  - Assert: Patterns are passed to `pool_train_step`
  
- **Test 5.4**: Verify patterns persist across training epochs
  - Apply damage at epoch N
  - Train for multiple epochs
  - Sample same circuits at epoch N+M
  - Assert: Patterns remain unchanged
  
- **Test 5.5**: Verify patterns are preserved when graphs are updated in pool
  - Apply damage to circuits
  - Update pool with new graph states
  - Assert: Knockout patterns remain unchanged

**Code Reference**:
```python
# boolean_nca_cc/training/train_loop.py:1840-1849
damage_behavior = getattr(model, "damage_behavior", "permanent")
should_accumulate = (damage_behavior == "permanent")
circuit_pool = circuit_pool.apply_knockouts(
    damaged_idxs, 
    new_patterns, 
    accumulate=should_accumulate
)
```

---

### 6. Pool Update Preservation (`boolean_nca_cc/training/pool/pool.py`)

**Location**: `GraphPool.update()` (lines 109-228)

**Purpose**: Verify that knockout patterns are preserved during pool update operations.

**Test Cases**:

- **Test 6.1**: Verify knockout patterns are preserved when graphs are updated
  - Apply damage to circuits in pool
  - Update pool with new graph states
  - Assert: Knockout patterns remain unchanged
  
- **Test 6.2**: Verify patterns are not overwritten during pool updates
  - Apply damage pattern A
  - Update pool with graphs (no new patterns)
  - Assert: Pattern A remains
  
- **Test 6.3**: Verify patterns persist through reset operations (unless explicitly reset)
  - Apply damage to circuits
  - Reset fraction of pool
  - Assert: Non-reset circuits retain damage patterns
  - Assert: Reset circuits have no damage (if `new_knockout_patterns=None`)
  
- **Test 6.4**: Verify patterns are correctly sampled with graphs
  - Apply damage to circuits
  - Sample batch from pool
  - Assert: Sampled patterns match pool patterns for sampled indices

**Code Reference**:
```python
# boolean_nca_cc/training/pool/pool.py:205-214
updated_knockout_patterns = self.knockout_patterns
if (
    batch_of_knockout_patterns is not None
    and self.knockout_patterns is not None
):
    updated_knockout_patterns = self.knockout_patterns.at[idxs].set(
        batch_of_knockout_patterns
    )
```

---

### 7. Evaluation Damage Preservation (`boolean_nca_cc/training/evaluation.py`)

**Location**: `_evaluate_with_loop()` (lines 663-881)

**Purpose**: Verify that damage patterns persist correctly during evaluation.

**Test Cases**:

- **Test 7.1**: Verify knockout patterns persist across evaluation steps
  - Create evaluation with damage patterns
  - Run multiple evaluation steps
  - Assert: Patterns remain consistent
  
- **Test 7.2**: Verify damaged nodes remain clamped during evaluation
  - Run evaluation with damage patterns
  - Extract logits at each step
  - Assert: Damaged nodes remain at -10.0
  
- **Test 7.3**: Verify patterns are correctly applied in eval mode
  - Run evaluation with damage patterns
  - Assert: Model receives correct patterns
  - Assert: Attention masks exclude damaged nodes
  
- **Test 7.4**: Verify accumulated patterns work correctly in eval
  - Create accumulated damage pattern (A | B)
  - Run evaluation
  - Assert: All damaged nodes (from A and B) remain clamped

---

### 8. LUT Conversion and Circuit Execution (`boolean_nca_cc/circuits/model.py`)

**Location**: `run_circuit()` (lines 163-194) and LUT conversion

**Purpose**: Verify that damaged gates produce zero outputs in circuit execution.

**Test Cases**:

- **Test 8.1**: Verify logits of -10.0 convert to LUT values ≈ 0 (via sigmoid)
  - Create logits with -10.0 for damaged gates
  - Convert: `luts = sigmoid(logits)`
  - Assert: LUTs for damaged gates ≈ 0.0
  
- **Test 8.2**: Verify damaged gates produce zero outputs in circuit execution
  - Create circuit with damaged gates (logits = -10.0)
  - Run circuit with various inputs
  - Assert: Damaged gates output 0.0
  
- **Test 8.3**: Verify `gate_mask` correctly zeros damaged gates if used
  - Create circuit with `gate_mask` for damaged gates
  - Run circuit
  - Assert: Masked gates produce zero outputs
  
- **Test 8.4**: Verify circuit outputs reflect permanent damage
  - Create circuit with damage pattern
  - Extract logits → convert to LUTs → run circuit
  - Assert: Circuit outputs reflect zeroed damaged gates

**Code Reference**:
```python
# boolean_nca_cc/circuits/model.py:163-194
def run_circuit(logits, wires, x, gate_mask=None, hard=False):
```

---

## Integration Test Scenarios

### Scenario 1: Single Damage Event Permanence

**Steps**:
1. Initialize pool with circuits
2. Apply damage pattern to selected circuits
3. Run N message passing steps (N > 1)
4. Extract logits and convert to LUTs
5. Run circuit execution

**Assertions**:
- Damaged nodes remain at -10.0 logits across all steps
- Damaged nodes produce zero LUT outputs
- Attention mask excludes damaged nodes at all steps
- Circuit outputs reflect zeroed damaged gates

---

### Scenario 2: Multi-Damage Accumulation

**Steps**:
1. Initialize pool with circuits
2. Apply damage pattern A to circuits
3. Apply damage pattern B with `accumulate=True`
4. Run multiple message passing steps
5. Extract logits and verify

**Assertions**:
- Final damage pattern = A | B (union)
- All nodes in A ∪ B remain clamped to -10.0
- Accumulation works correctly across multiple events
- `perturb_counter` increments correctly

---

### Scenario 3: Training Loop Persistence

**Steps**:
1. Initialize pool and model
2. Apply damage to circuits in pool
3. Sample and train for multiple epochs
4. Verify patterns persist
5. Check node states during training

**Assertions**:
- Patterns persist in pool across epochs
- Damaged nodes remain clamped during training steps
- Patterns survive pool updates
- Accumulation works correctly during training

---

### Scenario 4: Evaluation Persistence

**Steps**:
1. Damage circuits during training
2. Run evaluation with same patterns
3. Verify node states during eval
4. Check circuit outputs

**Assertions**:
- Damaged nodes remain clamped in eval mode
- Circuit outputs reflect damage
- Patterns persist across eval steps
- Evaluation matches training behavior

---

### Scenario 5: Pool Reset Preservation

**Steps**:
1. Apply damage to circuits
2. Reset a fraction of pool
3. Verify pattern preservation
4. Apply additional damage with accumulation

**Assertions**:
- Damaged circuits retain patterns (unless reset)
- Reset circuits start with no damage
- Accumulation continues correctly after resets
- Patterns correctly preserved/cleared based on reset

---

### Scenario 6: Perturbation Across Pool Operations

**Steps**:
1. Apply initial damage
2. Perform pool updates, resets, sampling
3. Apply additional damage with accumulation
4. Verify all damage persists
5. Check `perturb_counter`

**Assertions**:
- All damage persists across pool operations
- Accumulation works correctly
- `perturb_counter` tracks correctly
- Patterns remain consistent

---

## Edge Cases

### Edge Case 1: Damage Applied Before First Training Step
- Apply damage immediately after pool initialization
- Verify patterns persist through first training step
- Verify clamping works from the start

### Edge Case 2: Damage Applied During Training
- Apply damage mid-training
- Verify patterns integrate correctly
- Verify accumulation works

### Edge Case 3: Damage Applied After Pool Reset
- Reset pool, then apply damage
- Verify patterns apply correctly to reset circuits
- Verify accumulation works

### Edge Case 4: Multiple Damage Events on Same Circuit
- Apply damage multiple times to same circuit
- Verify accumulation works correctly
- Verify `perturb_counter` increments

### Edge Case 5: Damage Pattern Overlaps (Accumulation)
- Apply overlapping patterns
- Verify union is correct
- Verify all damaged nodes remain clamped

### Edge Case 6: Empty Damage Patterns
- Apply empty pattern (all False)
- Verify no nodes are damaged
- Verify system handles gracefully

### Edge Case 7: All Nodes Damaged
- Apply pattern that damages all nodes
- Verify all nodes clamped
- Verify circuit still executes (with zeros)

### Edge Case 8: Damage During Evaluation vs Training
- Compare damage behavior in eval vs training
- Verify consistency
- Verify patterns work in both modes

---

## Implementation Checklist

- [ ] **Unit Test**: `apply_knockouts` accumulation logic
- [ ] **Unit Test**: Model forward pass clamping
- [ ] **Unit Test**: Attention mask exclusion
- [ ] **Unit Test**: Logit extraction from damaged graphs
- [ ] **Integration Test**: Training loop damage flow
- [ ] **Integration Test**: Multi-step message passing persistence
- [ ] **Integration Test**: Pool update pattern preservation
- [ ] **Integration Test**: Evaluation damage persistence
- [ ] **End-to-End Test**: Full training + eval with permanent damage
- [ ] **End-to-End Test**: Multi-damage accumulation across epochs

---

## Test File Structure

```
tests/
├── validation_damage.md (this file)
├── test_damage_accumulation.py
├── test_damage_model_forward.py
├── test_damage_attention_mask.py
├── test_damage_extraction.py
├── test_damage_training_integration.py
├── test_damage_pool_preservation.py
├── test_damage_evaluation.py
└── test_damage_circuit_execution.py
```

---

## Notes

- All tests should verify that `damage_behavior == "permanent"` is correctly configured
- Tests should verify numerical precision: -10.0 logits → sigmoid ≈ 0.0 (within acceptable tolerance)
- Tests should verify that damage patterns are boolean arrays with correct shapes
- Tests should verify that `perturb_counter` tracks damage events correctly
- Tests should verify that accumulation uses bitwise OR (`|`) correctly
- Tests should verify that patterns persist across all pool operations

---

## Related Documentation

- `accumulation.md` - Implementation details for damage accumulation
- `configs/model/self_attention.yaml` - Model configuration including `damage_behavior`
- `boolean_nca_cc/models/self_attention.py` - Model implementation
- `boolean_nca_cc/training/pool/pool.py` - Pool implementation
- `boolean_nca_cc/training/train_loop.py` - Training loop integration

