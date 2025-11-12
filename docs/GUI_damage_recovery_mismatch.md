## GUI Damage Recovery Mismatch Investigation

This document addresses the critical issue where the GUI shows stable accuracy but **no recovery** after applying GAMMA RAYS damage, despite the evaluation loop showing successful recovery in wandb.

---

## 🔴 Core Problem: No Recovery After Damage

**Observed Behavior**:
- Accuracy remains stable at ~1.0 before damage (matches eval)
- When GAMMA RAYS is applied, accuracy drops slightly
- **Accuracy then stays perfectly stable at the dropped level - no recovery occurs**
- **Perfect stability is suspicious** - suggests updates might not be applied at all
- This contrasts with eval loop which shows clear recovery trajectories

**Expected Behavior**:
- After damage injection, model should apply updates to recover accuracy
- Recovery should be visible over multiple steps (similar to eval loop)

**Additional Context**:
- GUI defaults to `n_message_steps = 1` (line 208 in `GUI_minimal.py`)
- Training uses `n_message_steps: 5` (from config)
- This means GUI applies 1/5th the updates per tick compared to training
- However, this alone wouldn't prevent recovery entirely - it would just slow it down

## Note on `logit_scale`

**Status**: ⚠️ **Double-check in future** - `logit_scale` magnitude may affect recovery, but current focus is on state management alignment.

---

## Issue 3: Logits Reset to `logits0` Before Damage

### Current Implementation

In `GUI_minimal.py`, `_apply_gate_damage_perturbation()` (line 1107):
```python
# Start from logits0 to ensure we don't accumulate damage
damaged_logits = [l.copy() for l in self.logits0]
```

**Problem**:
- Damage is always applied starting from `logits0` (preconfigured state)
- If current logits have drifted, resetting to `logits0` restores better state
- This **masks the drift issue** rather than showing true damage effect
- Should preserve current logits and apply damage incrementally

**Impact**:
- When accuracy drifts to ~0.6 and PERTURB is clicked, accuracy jumps back to ~0.8
- This suggests `logits0` is better than current drifted state
- Makes it impossible to test damage recovery from a drifted state

### Proposed Fix

**Option A: Apply damage to current logits** (preserve drift, show true damage effect)
```python
# Apply damage to CURRENT logits (preserve any drift)
damaged_logits = [l.copy() for l in self.logits]  # Use current, not logits0
```

**Option B: Track both states** (allow explicit reset)
- Keep `logits0` as preconfigured baseline
- Track `logits_before_damage` when damage is applied
- Add UI control: "Reset to baseline" vs "Preserve current state"

**Recommendation**: Start with Option A to match eval behavior (damage applied to current state).

---

## 🎯 PRIMARY FOCUS: Align PERTURB Button with Periodic Eval Conditions

**Goal**: Make the PERTURB button in the GUI use **exactly the same perturbation injection and state management** as the periodic evaluation loop in `train_loop.py`, which uses functions from `evaluation.py`.

### Current Mismatch

The GUI's PERTURB button uses a completely different damage mechanism than periodic evaluation:

| Aspect | Periodic Eval (`train_loop.py` → `evaluation.py`) | GUI PERTURB Button |
|--------|---------------------------------------------------|-------------------|
| **Damage injection** | Uses `evaluate_model_stepwise_batched()` with `knockout_pattern` parameter | Bakes damage directly into logits |
| **Pattern handling** | Passes `knockout_pattern` to model via `_evaluate_with_loop()` | No pattern passed to model |
| **State management** | Uses `build_graph()` and graph globals `[loss, update_steps]` | Reinitializes generator from baked logits |
| **Step counter** | Resets `update_steps` to 0 on injection (line 826-830 in `evaluation.py`) | No step counter manipulation |
| **Reversible mode** | Model receives pattern, applies one-shot bias when `update_steps == 0` | Model never sees pattern, reversible mode never activates |
| **Graph initialization** | Uses `build_graph()` with proper loss/step initialization | Rebuilds graph from damaged logits without proper state |

### Required Alignment

**The PERTURB button must**:
1. **Use `evaluate_model_stepwise_generator()` from `evaluation.py`** (already used, but needs pattern support)
2. **Pass `knockout_pattern` to the generator** (currently not supported)
3. **Match graph state initialization** from `build_graph()` in `evaluation.py`
4. **Reset `update_steps` to 0** when damage is injected (matches line 826-830 in `evaluation.py`)
5. **Use the same damage pattern generation** as periodic eval (vocabulary-based or greedy patterns)

### Key Functions from `evaluation.py` to Use

1. **`evaluate_model_stepwise_generator()`**: Generator for step-by-step evaluation (already used, needs pattern support)
2. **`build_graph()`**: Graph initialization with proper state (from `utils/graph_builder.py`, used in `evaluation.py`)
3. **`_evaluate_with_loop()`**: Core evaluation loop with pattern handling (reference for state management)
4. **Pattern generation**: Use same functions as `train_loop.py`:
   - `create_knockout_vocabulary()` for vocabulary-based patterns
   - `create_group_greedy_pattern()` for greedy patterns
   - `create_greedy_subset_random_pattern()` for greedy vocabulary sampling

### State Management Alignment

**Critical**: Graph and circuit state must be managed identically between GUI and periodic eval:

1. **Graph globals**: Must be `[loss, update_steps]` format (matches `evaluation.py` line 502-504)
2. **Step counter reset**: Must zero `update_steps` on injection steps (matches `evaluation.py` line 826-830)
3. **Graph initialization**: Must use `build_graph()` with proper loss calculation (matches `evaluation.py` line 468-510)
4. **Logit extraction**: Must use `extract_logits_from_graph()` for consistency (matches `evaluation.py` line 116)
5. **Loss computation**: Must use `get_loss_from_wires_logits()` (already used, good)

### Reference Implementation

**Periodic Eval Flow** (`train_loop.py` → `evaluation.py`):
```python
# In run_knockout_periodic_evaluation() (train_loop.py line 298-329):
step_metrics_in = evaluate_circuits_in_chunks(
    eval_fn=evaluate_model_stepwise_batched,  # Uses _evaluate_with_loop()
    wires=in_wires,
    logits=in_logits,
    knockout_patterns=in_knockout_patterns,  # ← Pattern passed here
    ...
)

# In _evaluate_with_loop() (evaluation.py line 648-906):
# 1. Build graphs with proper state (line 468-510)
# 2. For each step, determine injection timing (line 738-833)
# 3. Reset step counter on injection (line 826-830)
# 4. Pass pattern to model (line 836-845)
# 5. Model applies reversible bias when step_count == 0 (in self_attention.py)
```

**GUI PERTURB Flow** (target):
```python
# In _apply_gate_damage_perturbation() (GUI_minimal.py):
# 1. Generate damage pattern (same as periodic eval)
knockout_pattern = create_damage_pattern(...)  # Match eval pattern generation

# 2. Initialize generator with pattern (NOT by baking into logits)
self.model_generator = evaluate_model_stepwise_generator(
    model=self.frozen_model,
    wires=self.wires,
    logits=self.logits,  # Current logits, NOT damaged
    knockout_pattern=knockout_pattern,  # ← Pass pattern here
    ...
)

# 3. Generator handles pattern internally:
#    - build_graph() initializes with current logits
#    - First step: model sees pattern, applies reversible bias
#    - Subsequent steps: normal recovery updates
```

## 🔴 Critical Mismatch: Evaluation Loop vs GUI Generator

### Evaluation Loop (`_evaluate_with_loop` in `evaluation.py`)

**Key Characteristics**:
1. **Passes `knockout_pattern` to model** (line 836-845):
   ```python
   vmap_model = jax.vmap(
       lambda g, k: model(
           g,
           knockout_pattern=k,  # ← Pattern passed to model
           layer_neighbors=layer_neighbors,
           layer_sizes=layer_sizes,
       )
   )
   updated_graphs = vmap_model(current_graphs, step_knockout_patterns)
   ```

2. **Reversible mode handling** (line 823-830):
   ```python
   damage_behavior = getattr(model, "damage_behavior", "permanent")
   if damage_behavior == "reversible":
       # Zero step counter on injection steps to trigger one-shot bias
       steps_before = current_graphs.globals[:, 1]
       steps_after = jp.where(inject_now_mask, jp.zeros_like(steps_before), steps_before)
       current_graphs = current_graphs._replace(
           globals=jp.stack([current_graphs.globals[:, 0], steps_after], axis=1)
       )
   ```

3. **Model's reversible behavior** (from `Simple_SEU.md`):
   - When `knockout_pattern` is provided AND `damage_behavior == "reversible"`:
     - At first step (`globals[..., 1] == 0`), model applies one-shot logit bias to damaged nodes
     - Subsequent steps apply normal residual updates (recovery path)
     - Attention mask ignores knockout (keeps connectivity)

### GUI Generator (`evaluate_model_stepwise_generator` in `evaluation.py`)

**Key Characteristics**:
1. **Does NOT pass `knockout_pattern` to model** (line 247-255):
   ```python
   if isinstance(model, CircuitSelfAttention):
       updated_graph = model(
           graph,
           layer_neighbors=layer_neighbors,
           layer_sizes=layer_sizes,
           # ← NO knockout_pattern parameter!
       )
   ```

2. **GUI bakes damage into logits directly** (`GUI_minimal.py` line 1117-1122):
   ```python
   # Apply damage bias where mask is 0.0 (damaged)
   damaged_logits[logits_idx] = np.where(
       mask_reshaped[..., None] == 0.0, 
       bias,  # -10.0 bias baked into logits
       damaged_logits[logits_idx]
   )
   ```

3. **Model never sees knockout pattern**:
   - Model cannot use reversible mode behavior
   - Model treats all gates equally (no special handling for damaged gates)
   - Model applies normal residual updates: `updated_logits = nodes["logits"] + logit_scale * logit_updates`

### The Mismatch

| Aspect | Evaluation Loop | GUI Generator |
|--------|----------------|---------------|
| **Damage mechanism** | Passes `knockout_pattern` to model | Bakes damage into logits |
| **Model awareness** | Model knows which gates are damaged | Model doesn't know about damage |
| **Reversible mode** | Model applies one-shot bias at step 0 | Model never applies reversible bias |
| **Step counter reset** | Zeros `update_steps` on injection | No step counter manipulation |
| **Recovery behavior** | Model explicitly handles recovery | Model treats damaged gates as normal |

**Result**: GUI uses a completely different damage mechanism than training/eval, so recovery behavior cannot match.

---

## Why No Recovery in GUI?

### Hypothesis 1: Model Doesn't Recognize Damage

The model was trained to recover from damage when it receives a `knockout_pattern`. In the GUI:
- Model never receives `knockout_pattern`
- Model doesn't know which gates are damaged
- Model applies the same updates to all gates (damaged and healthy)
- If damaged gates are at -10.0, normal residual updates may not be sufficient to recover

### Hypothesis 2: Logits Reset Masks Recovery

If damage is applied to `logits0` (good state) instead of current logits:
- Initial damage effect is smaller (starting from good state)
- Recovery may appear to "work" but is actually just maintaining good state
- True recovery from damaged state is never tested

### Hypothesis 3: Step Counter Mismatch

The eval loop zeros `update_steps` on injection steps to trigger reversible mode's one-shot bias. The GUI:
- Never resets `update_steps`
- Model's reversible mode never activates
- Model checks `step_count == 0.0` to apply one-shot bias (line 496 in `self_attention.py`)
- Since `update_steps` is never reset to 0, the condition `step_count == 0.0` is never true
- One-shot bias is never applied, so recovery never starts

**Model Code** (reversible mode implementation):
```489:499:boolean_nca_cc/models/self_attention.py
step_count = jp.array(0.0, dtype=jp.float32)
if globals_ is not None and globals_.shape[-1] > 1:
    step_count = globals_[..., 1]

current_logits = nodes["logits"]
# Introduce perturbation at the first model application of the episode
# Aligns with eval: step 0 is baseline (no model call), step 1 injects
apply_bias = jp.where(step_count == 0.0, 1.0, 0.0)
bias_mask = knockout_pattern[:, None] * self.reversible_bias
# Only add bias on the exact injection step
current_logits = current_logits + (apply_bias * bias_mask)
```

**Critical**: The model requires BOTH:
1. `knockout_pattern` to be passed (to know which gates are damaged)
2. `update_steps == 0` in graph globals (to trigger one-shot bias application)

The GUI provides neither, so reversible mode never activates.

---

## Success Criteria

### Current Focus: PERTURB Button Alignment

- [ ] GUI passes `knockout_pattern` to generator (matches eval loop)
- [ ] Generator passes pattern to model (matches `_evaluate_with_loop()`)
- [ ] Model receives pattern and activates reversible mode
- [ ] Step counter reset on damage injection (triggers one-shot bias at `update_steps == 0`)
- [ ] Damage applied to current logits (not `logits0`)
- [ ] Graph state management matches periodic eval (globals format, initialization)
- [ ] Recovery visible after damage (accuracy increases over steps)
- [ ] Recovery trajectory matches eval loop behavior

### Future Work

- [ ] Vocabulary-based pattern matching (exact `eval_ko_in` / `eval_ko_out` conditions)
- [ ] Automated sequence button (mirrors multi-inject loop timing)

---

## Related Documents

- [`GUI_reversible_damage.md`](GUI_reversible_damage.md): Original implementation details
- [`self_attention_accuracy_drift.md`](self_attention_accuracy_drift.md): Accuracy drift investigation
- [`Simple_SEU.md`](Simple_SEU.md): Reversible damage mode specification
- [`re_zero_update_mechanism.md`](re_zero_update_mechanism.md): Detailed explanation of `re_zero_update` and why it's necessary for training stability

---

## Implementation Plan

### Priority 1: Align Graph State Management (HIGHEST PRIORITY)

**Goal**: Ensure graph and circuit state management matches periodic eval exactly.

**Tasks**:
1. **Align graph state management**
   - Ensure `build_graph()` is called with proper loss/step initialization (matches `evaluation.py` line 468-510)
   - Graph globals must be `[loss, update_steps]` format (matches `evaluation.py` line 502-504)
   - Preserve graph state when reinitializing generator (don't rebuild from scratch unnecessarily)
   - Use `extract_logits_from_graph()` for consistency (matches `evaluation.py` line 116)

2. **Fix logits reset issue**
   - Change `damaged_logits = [l.copy() for l in self.logits0]` to use current logits
   - Apply damage to current state (preserves any drift)
   - Matches eval behavior: damage applied to current state, not baseline

3. **Update PERTURB button to use pattern-based damage**
   - Generate damage pattern using same logic as periodic eval:
     - If vocabulary available: sample from `knockout_vocabulary` (matches `train_loop.py` line 1795-1812)
     - Otherwise: use `create_greedy_subset_random_pattern()` or `create_reproducible_knockout_pattern()`
   - Pass pattern to generator instead of baking into logits
   - Remove logit baking code from `_apply_gate_damage_perturbation()`
   - Extend `evaluate_model_stepwise_generator()` to support `knockout_pattern` parameter
   - Handle step counter reset: when pattern provided and damage injected, reset `update_steps` to 0 (matches `evaluation.py` line 826-830)

**Files to Modify**:
- `boolean_nca_cc/training/evaluation.py`: Add `knockout_pattern` parameter to `evaluate_model_stepwise_generator()`
- `GUI_minimal.py`: Update `_apply_gate_damage_perturbation()` to use pattern-based damage and fix logits reset

**Success Criteria**:
- [ ] Graph state management matches periodic eval exactly (globals format, initialization)
- [ ] Damage applied to current logits (not `logits0`)
- [ ] PERTURB button passes `knockout_pattern` to generator (matches eval loop)
- [ ] Step counter reset on damage injection (triggers one-shot bias)
- [ ] Model receives pattern and activates reversible mode
- [ ] Recovery visible after damage (accuracy increases over steps)
- [ ] Recovery trajectory matches eval loop behavior

### Priority 2: Exact Perturbation Matching with Vocabulary

**Goal**: Use greedy knockout vocabulary for exact perturbation matching with `eval_ko_in` and `eval_ko_out`. This should be implemented before the automated sequence button.

**Tasks**:
1. **Load knockout vocabulary from training config**
   - Extract `knockout_vocabulary` from loaded WandB config
   - Store vocabulary in GUI state for pattern sampling

2. **Add vocabulary-based pattern selection**
   - Sample patterns from vocabulary (matches `train_loop.py` line 1795-1812)
   - Support both IN-distribution (from vocabulary) and OUT-distribution (fresh patterns)
   - Add UI controls: "Use Vocabulary" vs "Fresh Pattern"

3. **Match eval_ko_in and eval_ko_out conditions**
   - IN-distribution: Sample from vocabulary (seen during training)
   - OUT-distribution: Generate fresh patterns (unseen during training)
   - Use same pattern generation parameters as periodic eval

**Files to Modify**:
- `GUI_minimal.py`: Add vocabulary loading and pattern selection
- `boolean_nca_cc/training/evaluation.py`: Ensure generator supports vocabulary patterns

### Priority 3: Automated Sequence Button

**Goal**: Add automated sequence button that exactly mirrors batched eval multi-inject loop timing and conditions. This builds on vocabulary matching (Priority 2).

**Tasks**:
1. **Implement multi-inject sequence**
   - Mirror `_evaluate_with_loop()` multi-damage injection schedule (line 738-833 in `evaluation.py`)
   - Support `damage_start_offset`, `greedy_injection_recover_steps`, `max_damage_per_circuit`
   - Handle periodic injection timing: first damage after offset, then every `recover_steps + 1` steps
   - Use vocabulary-based patterns (from Priority 2)

2. **Add sequence controls**
   - Button: "Run Automated Sequence"
   - Parameters: damage count, recovery steps, start offset
   - Visualize injection timing in plot

3. **Match batched eval exactly**
   - Use same damage injection schedule as `_evaluate_with_loop()`
   - Use same pattern generation (vocabulary or greedy)
   - Use same state management (graph globals, step counter resets)

**Files to Modify**:
- `GUI_minimal.py`: Add automated sequence button and logic
- `boolean_nca_cc/training/evaluation.py`: May need to expose injection schedule logic

**Note**: This feature depends on vocabulary matching (Priority 2) being implemented first.

### Priority 4: Verify Updates Are Being Applied (LOWEST PRIORITY)

**Note**: ⚠️ Double-check in future - verify that logit updates are actually being computed and applied, but this is lowest priority compared to state management alignment.

---

## Next Steps (Immediate)

1. **Align graph state management** - Ensure graph globals format and initialization match `evaluation.py`
2. **Fix logits reset** - Change PERTURB to use current logits (not `logits0`)
3. **Update PERTURB button** - Generate and pass pattern to generator instead of baking into logits
4. **Extend generator** - Add `knockout_pattern` parameter support and step counter reset
5. **Test recovery behavior** - Verify matches eval loop trajectory

