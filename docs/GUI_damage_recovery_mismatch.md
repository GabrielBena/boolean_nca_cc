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

## 🔴 CRITICAL FINDING: `logit_scale` is Extremely Small (Not Zero, But Negligible)

**The Perfect Stability Problem**:
- Updates ARE being applied, but they're too small to matter
- The update formula is: `updated_logits = current_logits + self.logit_scale * logit_updates`
- **Observed**: `logit_scale = 0.000001` (1e-6) - essentially zero in practice
- **Observed**: `logit_updates` magnitude ~0.000003
- **Effective change**: `0.000001 * 0.000003 = 3e-9` - completely negligible!

**Evidence from Logs**:
```
logit_scale: 0.000001
logit_change_mean: 0.000003
logit_change_max: 0.000014
Loss: 133.419952 → 133.411606 (only 0.008 reduction over 300 steps)
Accuracy: 0.9160 (completely stable, no change)
```

**Root Cause**:
- Model config uses `re_zero_update: true` (line 15 in `self_attention.yaml`)
- When `re_zero_update=True`, `logit_scale` is initialized to `0.0` (line 259-266 in `self_attention.py`)
- `logit_scale` must be **learned during training** via gradient descent
- **The model learned a scale that's too small** (0.000001 instead of 0.01-0.1)
- This is a **training issue** - the optimizer didn't learn an appropriate scale value

**Verification Needed**:
1. ✅ **COMPLETED**: Check the actual value of `model.logit_scale` in the loaded checkpoint
2. ✅ **FOUND**: `logit_scale = 0.000001` (extremely small, not zero but negligible)
3. ✅ **CONFIRMED**: Updates are being applied but too small to matter
4. **Root cause identified**: Model learned a scale that's too small during training

**How to Verify**:
```python
# In GUI, after loading model:
print(f"logit_scale value: {model.logit_scale}")
if hasattr(model.logit_scale, 'value'):
    print(f"logit_scale.value: {model.logit_scale.value}")
else:
    print(f"logit_scale (scalar): {model.logit_scale}")

# Add logging in generator to track update magnitudes:
logit_updates = self.logit_proj(x)[0]  # After model call
print(f"logit_updates magnitude: {jp.abs(logit_updates).mean():.6f}")
print(f"logit_updates max: {jp.abs(logit_updates).max():.6f}")

# Track actual logit changes:
logit_change = updated_logits - current_logits
print(f"Actual logit change magnitude: {jp.abs(logit_change).mean():.6f}")
print(f"Actual logit change max: {jp.abs(logit_change).max():.6f}")
```

**If `logit_scale = 0.0`**:
- This is a training issue - the model didn't learn to update logits
- Check training logs to see if `logit_scale` was actually updated during training
- May need to retrain or check if optimizer was updating this parameter

**If `logit_scale` is extremely small (e.g., 0.000001)**:
- **This is what we observed!** Updates are applied but too small to matter
- The model learned a scale that's essentially zero in practice
- **Training issue**: Optimizer didn't learn an appropriate scale value (should be 0.01-0.1)
- Possible causes:
  - Learning rate too small for scale parameter
  - Scale parameter not receiving sufficient gradients
  - Training stopped before scale could learn
  - Weight decay or regularization penalizing scale too much
- **Solution**: Need to investigate why scale didn't learn properly during training

**If `logit_scale` is reasonable (0.01-0.1) but updates are still zero**:
- Check if `logit_updates` are being computed correctly
- Check if there's a bug in the update application
- Verify that the model is actually being called (not cached/stale)

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

## 🔴 Critical Mismatch: Evaluation Loop vs GUI Generator

### Key Question: Does the Model Need the Pattern or Just the Effect?

This is an important question. Let's examine what actually differs:

**When `knockout_pattern` is passed (eval loop)**:
- Model applies one-shot bias at step 0: `current_logits = current_logits + (apply_bias * bias_mask)`
- Then applies residual updates: `updated_logits = current_logits + self.logit_scale * logit_updates`
- Attention mask keeps damaged nodes connected (reversible mode)

**When damage is baked into logits (GUI)**:
- Logits are already at -10.0 for damaged gates
- Model applies residual updates: `updated_logits = nodes["logits"] + self.logit_scale * logit_updates`
- Attention mask computed normally (no special handling)

**Theoretically**: The model should be able to recover from logits at -10.0 using residual updates, regardless of whether it "knows" about the pattern. The residual updates should work the same way.

**However**: There's a critical timing difference:
- **Eval loop**: Bias applied at step 0, then updates applied at step 1, 2, 3...
- **GUI**: Damage baked into logits BEFORE generator initialization, then updates applied at step 1, 2, 3...

The model's reversible mode checks `step_count == 0.0` to apply the one-shot bias. If damage is already baked into logits when the generator starts, the model never gets a chance to apply its own bias at step 0.

**But wait**: If the model can recover from damaged logits using residual updates, why does it matter when the damage is applied?

**Analysis**:

Looking at the code more carefully:

1. **Attention mask** (line 434-446 in `self_attention.py`):
   - When `knockout_pattern` is provided AND `damage_behavior == "reversible"`: `knockout_for_mask = None` (line 436-437)
   - This means damaged nodes **stay connected** in the attention graph
   - When `knockout_pattern` is None: attention mask computed normally
   - **Result**: Attention mask is the SAME in both cases (damaged nodes stay connected)

2. **Logit updates computation** (line 458-464):
   - Updates are computed from attention output: `logit_updates = self.logit_proj(x)`
   - This depends on the attention mechanism, which processes all nodes
   - **Result**: Updates should be computed the same way regardless of how damage was applied

3. **Update application**:
   - When pattern passed: `updated_logits = current_logits + self.logit_scale * logit_updates` (after applying bias)
   - When damage baked: `updated_logits = nodes["logits"] + self.logit_scale * logit_updates`
   - **Result**: Same update formula, just different starting point

**Conclusion**: Theoretically, the model SHOULD be able to recover from damaged logits without needing the pattern. The residual updates should work the same way.

**So why isn't it recovering?**

Possible explanations:
1. **Step counter issue**: The model's reversible mode checks `step_count == 0.0` to apply bias. If damage is baked before generator init, the model never sees `step_count == 0.0` with damaged logits. But this shouldn't prevent recovery - it just means the model doesn't apply its own bias.

2. **Hidden state mismatch**: The model's hidden state might be initialized assuming normal logits. When damage is baked, the hidden state might be inconsistent with the damaged logits.

3. **Update magnitude**: Residual updates might be too small to recover from -10.0 in a single step. But over multiple steps, they should accumulate.

4. **Attention computation**: The attention mechanism might compute different updates when it sees logits at -10.0 vs when it sees normal logits that get biased. This could affect the update direction/magnitude.

**Most likely**: The model CAN recover from damaged logits, but there's a subtle initialization or state management issue preventing it. The pattern might not be strictly necessary, but passing it ensures the model's internal state is consistent with the damage.

**Key Insight**: Looking at `build_graph` (line 66 in `graph_builder.py`), hidden features are initialized to zeros regardless of logit values. So hidden state initialization is the same in both cases.

**Conclusion**: Theoretically, the model should be able to recover from damaged logits without needing the pattern. The fact that it's not recovering suggests:
1. Updates might be too small to overcome -10.0 bias
2. There might be a bug in how updates are applied when damage is baked
3. The model might need the pattern for proper state management (even if not strictly necessary for recovery)

**Recommendation**: Test both approaches:
- Try passing `knockout_pattern` to the generator (match eval loop)
- Also verify that residual updates are actually being applied when damage is baked into logits
- Check if update magnitudes are sufficient to recover from -10.0

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

## Investigation Plan

### Priority 0: Verify Updates Are Actually Being Applied (CRITICAL)

**Task**: Confirm that logit updates are actually being computed and applied.

**Steps**:
1. **Check `logit_scale` value** in loaded model:
   ```python
   # In GUI, after model is loaded:
   print(f"Model logit_scale: {model.logit_scale}")
   # If it's a Param, access .value
   if isinstance(model.logit_scale, nnx.Param):
       print(f"logit_scale.value: {model.logit_scale.value}")
   ```

2. **Add logging to generator** to track update magnitudes:
   - Log `logit_updates` magnitude before scaling
   - Log `logit_scale * logit_updates` magnitude (scaled updates)
   - Log actual logit changes: `updated_logits - current_logits`
   - Log logit values before and after update

3. **Verify update application**:
   - If `logit_scale = 0.0`: Root cause found - no updates applied
   - If `logit_scale > 0.0` but logit changes are zero: Bug in update application
   - If logit changes are non-zero but small: Updates too small to recover

**Expected Findings**:
- If `logit_scale = 0.0`: Model didn't learn to update (training issue)
- If updates are zero despite non-zero scale: Bug in code
- If updates are non-zero but small: Need to investigate why they're not accumulating

**Implementation Location**: 
- Add logging to `evaluate_model_stepwise_generator` in `evaluation.py`
- Add model inspection to GUI initialization in `GUI_minimal.py`

### Priority 1: Compare Damage Mechanisms

**Task**: Verify that eval loop passes `knockout_pattern` while GUI does not.

**Steps**:
1. Add logging to `evaluate_model_stepwise_generator` to show if `knockout_pattern` is passed
2. Add logging to model's `__call__` to show if `knockout_pattern` is received
3. Compare logs between eval loop and GUI generator

**Expected Finding**: GUI generator never passes `knockout_pattern`, confirming mismatch.

### Priority 2: Extend Generator to Support Knockout Patterns

**Task**: Add `knockout_pattern` parameter to `evaluate_model_stepwise_generator`.

**Implementation**:
```python
def evaluate_model_stepwise_generator(
    model: CircuitGNN | CircuitSelfAttention,
    wires: List[jp.ndarray],
    logits: List[jp.ndarray],
    x_data: jp.ndarray,
    y_data: jp.ndarray,
    input_n: int,
    arity: int = 2,
    circuit_hidden_dim: int = 16,
    max_steps: int = None,
    loss_type: str = "l4",
    bidirectional_edges: bool = True,
    layer_sizes: List[Tuple[int, int]] = None,
    layer_neighbors: bool = False,
    knockout_pattern: Optional[jp.ndarray] = None,  # ← NEW
) -> Generator[StepResult, None, None]:
```

**Model call update**:
```python
if isinstance(model, CircuitSelfAttention):
    updated_graph = model(
        graph,
        layer_neighbors=layer_neighbors,
        layer_sizes=layer_sizes,
        knockout_pattern=knockout_pattern,  # ← Pass pattern
    )
```

**GUI update**: Pass damage pattern when initializing generator after damage injection.

### Priority 3: Fix Logits Reset Issue

**Task**: Apply damage to current logits instead of `logits0`.

**Change** (`GUI_minimal.py` line 1107):
```python
# BEFORE:
damaged_logits = [l.copy() for l in self.logits0]

# AFTER:
damaged_logits = [l.copy() for l in self.logits]  # Use current state
```

**Impact**: 
- Damage applied to actual current state (may include drift)
- True recovery behavior visible
- Matches eval loop behavior (damage applied to current state)

### Priority 4: Implement Step Counter Reset

**Task**: Reset `update_steps` to 0 when damage is injected (to trigger reversible mode).

**Implementation**:
- After damage injection, reinitialize generator with `update_steps=0` in graph globals
- This matches eval loop behavior (line 826-830 in `evaluation.py`)

**Note**: This may require modifying `build_graph` or graph globals initialization in generator.

### Priority 5: Verify Reversible Mode Configuration

**Task**: Ensure model is configured with `damage_behavior="reversible"` when loaded in GUI.

**Check**:
- Model config from checkpoint
- Default model parameters
- Verify `reversible_bias` matches training config (-10.0)

---

## Expected Behavior After Fixes

### Before Fixes (Current)
1. Accuracy stable at 1.0
2. GAMMA RAYS applied → accuracy drops slightly
3. **Accuracy stays flat** (no recovery)

### After Fixes (Expected)
1. Accuracy stable at 1.0
2. GAMMA RAYS applied → accuracy drops
3. **Accuracy recovers over multiple steps** (similar to eval loop)
4. Recovery trajectory visible in plot

---

## Key Files to Modify

1. **`boolean_nca_cc/training/evaluation.py`**:
   - Add `knockout_pattern` parameter to `evaluate_model_stepwise_generator`
   - Pass pattern to model call
   - Handle step counter reset for reversible mode

2. **`GUI_minimal.py`** (and `GUI.py`):
   - Change damage application to use current logits (not `logits0`)
   - Pass `knockout_pattern` when initializing generator after damage
   - Reset `update_steps` to 0 on damage injection

3. **Model verification**:
   - Ensure `damage_behavior="reversible"` in model config
   - Verify `reversible_bias=-10.0` matches training

---

## Success Criteria

- [x] **CRITICAL**: `logit_scale` is non-zero in loaded model (updates are enabled) - **FOUND: 0.000001 (too small!)**
- [x] Logit updates are actually being computed (non-zero `logit_updates`) - **CONFIRMED: ~0.000003**
- [x] Logit changes are visible between steps (non-zero `updated_logits - current_logits`) - **CONFIRMED: but negligible (3e-9)**
- [x] **NEW ISSUE**: `logit_scale` is too small (0.000001) - need to investigate why training didn't learn proper scale
- [ ] GUI passes `knockout_pattern` to model (matches eval loop)
- [ ] Model receives pattern and activates reversible mode
- [ ] Step counter reset on damage injection (triggers one-shot bias)
- [ ] Damage applied to current logits (not `logits0`)
- [ ] Recovery visible after damage (accuracy increases over steps)
- [ ] Recovery trajectory matches eval loop behavior

---

## Training Run Comparison: Problematic vs Successful

### Problematic Training Run (Documented Issue)
- **`logit_scale`**: `0.000001` (1e-6) - **extremely small, positive**
- **Magnitude**: 1000x smaller than expected (should be 0.01-0.1)
- **Effective updates**: `0.000001 * 0.000003 = 3e-9` - negligible
- **Result**: No recovery in GUI, updates too small to matter

### Successful Training Run (Current)
- **`logit_scale`**: `-0.01` (magnitude ~0.01) - **reasonable magnitude, negative sign**
- **Magnitude**: ~10,000x larger than problematic run, within expected range (0.01-0.1)
- **Effective updates**: `0.01 * logit_updates` - substantial updates possible
- **Result**: Model perfectly recovering, updates are effective
- **Observations**:
  - Scale magnitude is in correct range (0.01-0.1) ✓
  - Negative sign is unusual but model compensates ✓
  - Oscillations after epoch 1400 may indicate instability
  - `hidden_scale` collapses to near 0 at end of training

### Key Insight
The **magnitude** of `logit_scale` is critical, not just whether it's non-zero. The problematic run had `|logit_scale| = 0.000001` (too small), while the successful run has `|logit_scale| ≈ 0.01` (appropriate). The negative sign in the successful run suggests the model learned to invert the update direction, which works as long as the magnitude is sufficient.

---

## Related Documents

- [`GUI_reversible_damage.md`](GUI_reversible_damage.md): Original implementation details
- [`self_attention_accuracy_drift.md`](self_attention_accuracy_drift.md): Accuracy drift investigation
- [`Simple_SEU.md`](Simple_SEU.md): Reversible damage mode specification
- [`re_zero_update_mechanism.md`](re_zero_update_mechanism.md): Detailed explanation of `re_zero_update` and why it's necessary for training stability

---

## Next Steps

1. **Immediate**: Add logging to verify `knockout_pattern` is not passed in GUI
2. **Short-term**: Extend generator to support `knockout_pattern` parameter
3. **Short-term**: Fix logits reset issue (use current logits)
4. **Medium-term**: Implement step counter reset for reversible mode
5. **Testing**: Verify recovery behavior matches eval loop

