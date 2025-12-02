# Permanent Damage Mode Validation Test Plan

## Overview

This document outlines validation tests for permanent damage mode, focusing on **direct tracking of node vectors during the evaluation loop**. This is the fundamental priority underpinning the entire experiment.

## Goal

Ensure that permanent damage mode:
1. **Keeps damaged node vectors (logits, hidden features) unchanged** across all message passing steps in the eval loop
2. **Accumulates damage patterns correctly** when multiple damage events occur
3. **Preserves accumulated damage** across all evaluation steps

---

## Priority 1: Direct Node Vector Tracking During Eval Loop

**Location**: `_evaluate_with_loop()` in `boolean_nca_cc/training/evaluation.py` (lines 663-881)

**Purpose**: Add tracking/checks directly into the stepwise evaluation function to verify that damaged nodes' logits and hidden features remain constant across all steps.

### Implementation Requirements

Add tracking to `_evaluate_with_loop()` to:

1. **Track node vectors for damaged nodes at each step**:
   - Extract logits for damaged nodes: `nodes["logits"][knockout_pattern]`
   - Extract hidden features for damaged nodes: `nodes["hidden"][knockout_pattern]`
   - Store these values per step in `per_pattern_metrics` or a new tracking structure

2. **Verify permanence**:
   - After first damage injection, record initial values for damaged nodes
   - At each subsequent step, assert that damaged nodes have identical values:
     - `logits[damaged_nodes] == -10.0` (must remain constant)
     - `hidden[damaged_nodes] == 0.0` (must remain constant)

3. **Track across multiple damage events**:
   - When new damage is injected, track which nodes were newly damaged
   - Verify previously damaged nodes remain unchanged
   - Verify newly damaged nodes are clamped correctly

### Test Cases

- **Test 1.1**: Single damage event permanence
  - Inject damage pattern at step N
  - Track logits and hidden features for damaged nodes
  - Run M additional steps (M > 1)
  - Assert: Damaged nodes have identical logits/hidden at steps N, N+1, ..., N+M
  
- **Test 1.2**: Multiple damage events - accumulation tracking
  - Inject damage pattern A at step N1
  - Track nodes damaged by A
  - Inject damage pattern B at step N2 (with accumulation)
  - Track nodes damaged by B
  - Assert: Nodes from A remain unchanged
  - Assert: Nodes from B are clamped correctly
  - Assert: Final pattern = A | B
  
- **Test 1.3**: Verify values match expected permanent damage behavior
  - Assert: `logits[damaged_nodes] == -10.0` (exact match)
  - Assert: `hidden[damaged_nodes] == 0.0` (exact match)
  - Assert: Updates for damaged nodes are zeroed (verify in model forward pass)

### Code Reference

```466:485:boolean_nca_cc/models/self_attention.py
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

**Tracking location**: Add to `_evaluate_with_loop()` after line 874 (after extracting logits) and after line 862 (after model forward pass).

---

## Priority 2: Accumulation Verification

**Location**: `_evaluate_with_loop()` - damage injection logic (lines 755-850)

**Purpose**: Verify that damage patterns accumulate correctly when multiple damage events occur during evaluation.

### Implementation Requirements

1. **Track damage history**:
   - Maintain cumulative damage pattern: `cumulative_pattern = pattern_A | pattern_B | ...`
   - Track which nodes were damaged at which step
   - Verify cumulative pattern is passed to model correctly

2. **Verify accumulation logic**:
   - When `damage_behavior == "permanent"`, patterns should accumulate (OR operation)
   - Previously damaged nodes should remain damaged
   - Newly damaged nodes should be added to the cumulative pattern

### Test Cases

- **Test 2.1**: First damage event
  - Inject pattern A at step N1
  - Track: `damaged_nodes_A = nodes where pattern_A == True`
  - Verify: These nodes are clamped to -10.0 logits, 0.0 hidden
  
- **Test 2.2**: Second damage event with accumulation
  - Inject pattern B at step N2
  - Track: `damaged_nodes_B = nodes where pattern_B == True`
  - Verify: `cumulative_pattern = pattern_A | pattern_B`
  - Verify: Nodes from A remain clamped (unchanged)
  - Verify: Nodes from B are now clamped
  - Verify: Overlapping nodes remain clamped
  
- **Test 2.3**: Multiple sequential damage events
  - Inject patterns A, B, C at steps N1, N2, N3
  - Verify: Final cumulative pattern = A | B | C
  - Verify: All nodes from A, B, C remain clamped across remaining steps

### Code Reference

```755:850:boolean_nca_cc/training/evaluation.py
for step in range(1, n_message_steps + 1):
    # ... damage injection logic ...
    # step_knockout_patterns is computed per step
    # For accumulation, need to track cumulative pattern
```

**Note**: The eval loop currently computes `step_knockout_patterns` per step. For permanent damage with accumulation, we need to maintain a cumulative pattern that persists across steps.

---

## Priority 3: Pool Training Alignment (Future)

**Status**: Deferred until Priority 1 and 2 are verified.

**Location**: `boolean_nca_cc/training/pool/pool.py` and `boolean_nca_cc/training/train_loop.py`

**Purpose**: Once eval loop tracking is verified, align pool training to ensure it respects permanent damage in the same way.

---

## Implementation Checklist

### Phase 1: Eval Loop Tracking (Priority 1)
- [ ] Add node vector tracking to `_evaluate_with_loop()`
  - [ ] Track logits for damaged nodes at each step
  - [ ] Track hidden features for damaged nodes at each step
  - [ ] Store tracking data in `per_pattern_metrics` or new structure
- [ ] Add assertions to verify permanence
  - [ ] Assert logits remain -10.0 for damaged nodes
  - [ ] Assert hidden features remain 0.0 for damaged nodes
  - [ ] Assert values are identical across steps
- [ ] Test single damage event permanence
- [ ] Test multiple damage events with accumulation

### Phase 2: Accumulation Verification (Priority 2)
- [ ] Implement cumulative pattern tracking in eval loop
- [ ] Verify pattern accumulation (OR operation)
- [ ] Test sequential damage events
- [ ] Verify previously damaged nodes remain unchanged

### Phase 3: Pool Training Alignment (Priority 3 - Future)
- [ ] Verify pool respects permanent damage
- [ ] Align pool training with eval behavior
- [ ] Test end-to-end training + eval consistency

---

## Test File Structure

```
tests/
├── validation_damage.md (this file)
├── test_eval_damage_tracking.py  # Priority 1: Direct node vector tracking
└── test_eval_damage_accumulation.py  # Priority 2: Accumulation verification
```

---

## Notes

- All tracking should happen directly in `_evaluate_with_loop()` - this is the function called during training
- Track exact values: logits should be exactly -10.0, hidden should be exactly 0.0
- For accumulation, track which nodes were damaged at which step to verify correct accumulation
- Numerical precision: Use exact equality checks (not approximate) for permanent damage verification
- Focus on eval loop first - pool training alignment comes later

---

## Related Documentation

- `boolean_nca_cc/training/evaluation.py` - Evaluation loop implementation
- `boolean_nca_cc/models/self_attention.py` - Model forward pass with damage handling
- `configs/model/self_attention.yaml` - Model configuration including `damage_behavior`
