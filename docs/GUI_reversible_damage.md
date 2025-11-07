## GUI reversible damage (GAMMA RAYS) – implementation reference

This document captures the functional requirements and timing details for adding random, reversible damage to the GUI so that the damage truly affects circuit execution and must be healed by the model (self‑attention) to recover accuracy.

### Goals
- Inject random, reversible damage at runtime that immediately degrades performance (visible in accuracy logs) and requires the model to heal by updating logits.
- Keep execution and visualization consistent with the current logit state as healing proceeds.
- Preserve intuitive reset behavior.

---

## Current behavior summary (GUI.py)

- Damage injection (GAMMA RAYS):
  - Baked into logits by setting damaged gates’ logits to a negative bias (e.g., −10.0).
  - `gate_mask` is set from the damage pattern (0.0 for damaged, 1.0 for active) and is used in `run_circuit(..., gate_mask=self.gate_mask)` for visualization.
  - Model generator (`evaluate_model_stepwise_generator`) does NOT receive a `knockout_pattern`; the model does standard updates from the baked-in logits.

- Accuracy computation path:
  - The generator’s `get_loss_and_update_graph(...)` → `get_loss_from_wires_logits(...)` runs `run_circuit(logits, wires, x)` WITHOUT gate_mask.
  - Therefore, accuracy reflects the current logits only (damage appears if and only if logits were baked with bias before the accuracy tick).

- Visualization path:
  - `run_circuit(..., gate_mask=self.gate_mask)` is used to compute per-layer activations.
  - If a gate heals (logits recover), but `gate_mask` remains 0.0, visualization can incorrectly zero a gate that has effectively healed (timing mismatch).

- Reset behavior:
  - `reset_circuit()` resets logits to `logits0` (pre‑damage), resets step and logs, does not reset wires.
  - After wire shuffle: `reset_circuit()` restores initial logits while keeping mutated wires.
  - After GAMMA RAYS: `reset_circuit()` loses damage (restores pre‑damage logits).

---

## Requirements for reversible damage in the GUI

1) Damage injection semantics
- Apply a random knockout pattern that immediately degrades accuracy for at least one tick.
- Damage is reversible: the model can heal gates by updating logits away from the bias.
- Do NOT alter `logits0` (keep pre‑damage baseline). Store damage only in the current `self.logits`.

2) Execution accuracy vs visualization
- Accuracy logging uses logits-only (no `gate_mask`). To guarantee a first-tick drop, bake damage into `self.logits` BEFORE the next generator step and rebuild the generator state so it begins from the damaged logits.
- Mask is visualization-only and must not affect any computation that feeds accuracy/loss.

3) Gate mask policy (chosen: simplest)
- Use a visualization-only mask with a single-tick red flash:
  - For one tick after injection, draw the circuit with the damage mask.
  - Never pass any mask into execution or accuracy paths; execution is logits-only at all times.

4) State to track
- `self._viz_damage_mask`: per-layer masks (1.0 active, 0.0 damaged) for visualization only.
- `self._viz_flash_ticks`: small integer counter (e.g., 1) for the one-tick red flash.

5) Reset semantics
- Soft reset: preserve current `self.logits` (and thus damage) and only reset counters/logs.
- Full reset (optional): restore `self.logits = self.logits0`, clear viz state, reset counters/logs, optionally reset wires.

---

## Timing and control flow

### On damage injection (GAMMA RAYS)
1. Sample damage pattern (skip inputs and outputs).
2. Build per-layer masks; set `self._viz_damage_mask` for the upcoming visualization flash.
3. Bake damage into current logits (e.g., set −10.0 for damaged gates).
4. Initialize viz flash: `self._viz_flash_ticks = 1`.
5. Optionally reset counters/logs; re-init the generator state from CURRENT logits so the very next logged tick reflects damage.

### Each optimization tick (optimize_with_unified_model)
Order of operations:
1. Advance the generator → returns updated logits; assign `self.logits = self.last_step_result.logits`.
2. Accuracy/logging: already correct (generator path is logits-only).
3. Visualization:
   - If `self._viz_flash_ticks > 0`: call `run_circuit(..., gate_mask=self._viz_damage_mask)`; then decrement to 0.
   - Else: call `run_circuit(..., gate_mask=None)`.

### Healing detection
Not required for the first implementation. Healing is reflected in logits (and thus in accuracy) automatically. Visualization does not need to track healing dynamically.

---

## Integration details and gotchas

- Accuracy path in generator:  
  Accuracy is computed from logits only (no `gate_mask`). To guarantee an immediate accuracy drop, ensure `self.logits` are already damaged BEFORE the next generator step. Reinitializing the generator state right after damage injection ensures the very next logged tick reflects damage.

- Mask vs logits for execution:  
  Masks zero gate outputs regardless of logits; relying on masks for execution blocks healing. For reversible damage, prefer logit bias for execution and keep masks for visualization or at most one tick for a forced drop/visual cue.

- Input/Output layers:  
  Patterns should exclude inputs and outputs; only interior gate layers are damaged. `create_gate_mask_from_knockout_pattern` already aligns mask layout with `layer_sizes`.

- Shape safety and re-init:  
  After injection, clear any cached predictions that depend on shapes and re-init the generator so that step 0/1 states reflect the damaged logits.

---

## Minimal pseudocode drop-in

Damage injection:
```python
# 1) Derive pattern and per-layer masks
self._viz_damage_mask = layer_gate_masks  # 1.0 active, 0.0 damaged (viz-only)
self._viz_flash_ticks = 1  # one-tick red flash (viz-only)

# 2) Bake damage into current logits (reversible)
damaged_logits = [l.copy() for l in self.logits]
for li, layer_mask in enumerate(layer_gate_masks[:-1], start=0):  # skip outputs
    damaged_logits[li] = np.where(layer_mask[..., None] == 0.0, damage_bias, damaged_logits[li])
self.logits = damaged_logits

# 3) Reset counters if desired and re-init generator
self.step_i = 0
self.loss_log[:] = 0; self.hard_log[:] = 0
self.accuracy_log[:] = 0; self.hard_accuracy_log[:] = 0
self.model_generator = None
self.last_step_result = None
self.initialize_model_generator()  # rebuild state from CURRENT logits
```

Per tick after model update:
```python
# After: self.logits = self.last_step_result.logits

# Accuracy is taken from generator (logits-only) – already correct

# Visualize circuit activations (viz-only mask, single tick)
if self._viz_flash_ticks > 0:
    self.act = run_circuit(self.logits, self.wires, self.input_x, hard=False, gate_mask=self._viz_damage_mask)
    self._viz_flash_ticks -= 1
else:
    self.act = run_circuit(self.logits, self.wires, self.input_x, hard=False, gate_mask=None)
```

---

## Future convergence with training/eval reversible mode

The attention model already supports “permanent” and “reversible” damage when a `knockout_pattern` is provided:
- Permanent: removes attention edges for damaged nodes, clamps logits negative, zeros updates.
- Reversible: keeps connectivity, applies a one‑shot bias on step 1 using `globals[..., 1]` to gate timing, and allows recovery afterward.

To converge with training/eval:
- Extend the single‑circuit generator to accept a `knockout_pattern` and forward it to the model for the first tick (reversible one‑shot bias) rather than baking into logits.
- Use the same step counter semantics to time the one‑shot bias.

---

## Checklist for implementation

- [x] Add state: `_viz_damage_mask`, `_viz_flash_ticks`, `damage_bias`.
  - ✅ Implemented in both `GUI.py` and `GUI_minimal.py`
  - State initialized in `__init__`: `_viz_damage_mask = []`, `_viz_flash_ticks = 0`, `damage_bias = -10.0`

- [x] On injection: bake damage into current logits; set `_viz_flash_ticks = 1`; store `_viz_damage_mask`; re-init generator state from current logits.
  - ✅ Implemented in `_apply_gate_damage_perturbation()` in both versions
  - Damage baked into logits starting from `logits0` (prevents accumulation)
  - `_viz_flash_ticks` set to 3 (extended flash for visibility)
  - `_viz_damage_mask` stored from damage pattern
  - Generator re-initialized from damaged logits via `initialize_model_generator()`
  - Plot history preserved (step_i and log arrays not reset)

- [x] On each tick: update `self.logits` from generator; do not use any execution mask; for visualization, use `_viz_damage_mask` only for the next 1 tick.
  - ✅ Logits update correctly from generator (no execution mask used)
  - ✅ Visualization flash logic implemented - uses `_viz_damage_mask` when `_viz_flash_ticks > 0`, then decrements
  - ✅ Updated both `optimize_backprop()` and `optimize_with_unified_model()` to use flash logic

- [x] Ensure accuracy path sees damaged logits at least on the first post‑injection tick.
  - ✅ Generator re-initialized from damaged logits immediately after injection
  - Accuracy computation uses logits-only path (no gate_mask), so damage is visible

- [x] Decide on reset variants: soft reset (preserve damage) vs full reset (clear damage and restore `logits0`).
  - ✅ Full reset implemented in `reset_circuit()`
  - Resets `logits = logits0` (clears damage)
  - Clears visualization state: `_viz_damage_mask = []`, `_viz_flash_ticks = 0`
  - Soft reset not implemented (can be added if needed)

- [ ] Optionally, add controls: "Single‑tick mask", "Preserve damage on reset".
  - ❌ Not implemented - can be added as future enhancement

## Additional Implementation Notes

### Minimal Version (`GUI_minimal.py`)
- ✅ Created focused version with only Self-Attention, Backprop, and GAMMA RAYS
- ✅ Removed GNN mode and wire shuffling functionality
- ✅ Simplified perturbation to GAMMA RAYS only
- ✅ Preserves preconfigure logic: uses `preconfigure_circuit_logits` in repair mode (matches training setup)

### Plot History Preservation
- ✅ Plot history (curves and red line position) preserved when applying GAMMA RAYS
- ✅ `step_i` and log arrays not reset on perturbation
- ✅ Damage appears as continuation of existing plot curve

---

## Implementation Achievements

### ✅ Random Damage Pattern Generation
- **Randomized seed on each PERTURB click**: Each button click now generates a unique damage pattern
- **Implementation**: Uses `np.random.randint(0, 1_000_000)` to create a random seed for pattern generation
- **Location**: `_apply_gate_damage_perturbation()` line 1025
- **Result**: No two damage injections produce the same pattern, enabling diverse testing scenarios

### ✅ Visualization Flash System with Red Gates
- **Red flash visualization**: Damaged gates are visually highlighted in red for 3 frames after injection
- **Implementation Details**:
  - `_viz_damage_mask` stores the damage pattern for visualization (per-layer masks: 1.0=active, 0.0=damaged)
  - `_viz_flash_ticks` counter controls flash duration (set to 3 frames)
  - `draw_circuit()` checks `_viz_damage_mask` when `_viz_flash_ticks > 0` and draws damaged gates in red (`0xA00000FF`)
  - Flash ticks decremented at the end of `draw_circuit()` to ensure proper timing
  - Both `optimize_backprop()` and `optimize_with_unified_model()` use `_viz_damage_mask` for activations during flash period
  - After flash period, visualization returns to normal (no mask) showing actual circuit state
- **Visual Effect**: Same red color (`0xA00000FF`) as when manually clicking gates - clear visual feedback
- **Result**: Clear visual feedback showing which gates were damaged, then smooth transition to showing healing progress

### ✅ Live Loop Behavior in Self-Attention Mode

**When PERTURB is clicked:**
1. Random damage pattern is generated (different each time via randomized seed)
2. Damage is baked into logits: damaged gates set to `-10.0` bias (starting from `logits0` to prevent accumulation)
3. Generator is re-initialized from damaged logits via `initialize_model_generator()`
4. Visualization flash is activated (`_viz_flash_ticks = 3`, `_viz_damage_mask` stored)

**During live optimization loop:**
1. **Logits update**: Each tick, `self.logits = self.last_step_result.logits` - model applies residual updates
2. **Model behavior**: Model treats all gates equally (no `knockout_pattern` passed to generator), applying normal residual updates: `updated_logits = nodes["logits"] + logit_scale * logit_updates`
3. **Healing mechanism**: Damaged gates (at -10.0) receive updates like normal gates, allowing the model to heal them by moving logits away from the bias
4. **Accuracy tracking**: Uses logits-only path (no mask), so accuracy immediately reflects damage and shows recovery as gates heal
5. **Visualization**:
   - First 3 frames: Shows red flash using `_viz_damage_mask` in `draw_circuit()` to highlight damaged gates
   - After flash: Shows actual circuit state based on current logits (healing visible in real-time)

**Key characteristics:**
- ✅ Damage is truly reversible - model can heal gates by updating logits
- ✅ Accuracy drops immediately on first tick after damage
- ✅ Accuracy recovers as model heals damaged gates
- ✅ Visualization provides clear feedback without interfering with execution
- ✅ Plot history preserved - damage appears as continuation of existing curve
- ✅ Random patterns enable diverse testing scenarios

### ✅ Code Quality Improvements
- Consistent implementation across both `optimize_backprop()` and `optimize_with_unified_model()`
- Proper state management for visualization flash (decremented in `draw_circuit()` for correct timing)
- Clean separation between execution (logits-only) and visualization (mask during flash only)
- No interference with accuracy/loss computation paths
- Robust mask value handling (converts to float for reliable comparison)

### Implementation Locations
- **Damage injection**: `_apply_gate_damage_perturbation()` (lines 1005-1084)
- **Visualization flash in circuit drawing**: `draw_circuit()` (lines 1234-1248, 1317-1319)
- **Visualization flash in activations**: `optimize_backprop()` (lines 837-841), `optimize_with_unified_model()` (lines 905-909)
- **State initialization**: `__init__()` (lines 276-278)

---

## Next Order of Operations: Critical Issues to Resolve

### 🔴 Issue 2: Self-Attention Model Accuracy Drift
**Problem**: When loading a self-attention model, accuracy slowly drifts downwards over time, eventually hitting ~0.6. However, during training, the eval loop shows perfect accuracy (>1.0) for >150 inner loop steps.

**Observations**:
- Backprop works fine, converging to accuracy 1.0
- Self-attention model shows stable performance in training eval loop
- Live GUI loop shows progressive accuracy degradation
- Mismatch between training eval and live GUI behavior

**Potential Causes**:
1. **Mismatch in live loop vs. eval loop conditions**:
   - Different initialization (graph state, globals, step counters)
   - Different number of message steps per iteration
   - Different loss computation timing
   - Missing or incorrect parameters in generator initialization

2. **Checkpointing issue**:
   - Model saved might not be the one showing stable performance
   - Model state incomplete (missing optimizer state, graph state, etc.)
   - Version mismatch between training and loading code

3. **Generator state management**:
   - Generator not properly reinitialized after damage
   - Graph globals (update_steps counter) accumulating incorrectly
   - Hidden state not properly reset between iterations

4. **Model behavior differences**:
   - Training uses batched evaluation, GUI uses single circuit
   - Training uses different loss computation path
   - Training eval might use different damage injection timing

**Investigation Needed**:
- Compare exact initialization between `evaluate_model_stepwise_generator` (GUI) and `evaluate_model_stepwise_batched` (training eval). Specifically, in @train_loop.py, eval_ko_in_steps/hard_accuracy gives stepwise accuracy readings that show great and stable accuracy.
- Check if graph globals are properly initialized (especially `update_steps` counter)
- Verify checkpoint loading includes all necessary state
- Compare step-by-step logits updates between training eval and GUI
- Check if `n_message_steps` in GUI matches training eval inner steps
- Verify loss computation uses same path in both cases


**Critical Evidence from Terminal Logs:
```
Initializing model generator with:
  - hidden_dim: 64
  - use_globals: True
  - model type: CircuitSelfAttention
Initialized model generator with initial loss: 0.0266
Step 0: Loss = 0.2349, Hard Loss = 4.0000
Step 100: Loss = 29.7506, Hard Loss = 110.0000
Step 200: Loss = 88.3226, Hard Loss = 248.0000
```

**Key Findings**:
initial loss upon generator initilisation of 0.23 is not terrible, but not great either

2. **🔴 Loss still increases over time**:, loss continues to drift upward:
   - Step 0: 0.2349 (9x increase from initial)
   - Step 100: 29.7506 (127x increase from initial)
   - Step 200: 88.3226 (332x increase from initial)
   - This confirms the drift issue persists even with correct preconfiguration.

3. **First generator step increases loss**: The first step increases loss from 0.0266 → 0.2349 (9x increase). While this is much better than the previous 2x increase from a high baseline, it still suggests the first generator step is applying updates that degrade performance.

**Most Likely Root Cause**:
- **Generator initialization using wrong state**: Graph globals (especially `update_steps` counter), hidden features, or initial graph representation don't match training conditions
- **First step applying incorrect updates**: Possibly due to wrong `update_steps` counter causing the model to think it's at a different step than it actually is
- **Mismatch between training eval and GUI generator initialization**: Training eval may initialize graph state differently than GUI generator
- **Progressive degradation**: Each step continues to worsen performance, suggesting systematic issues with how model updates are being applied

**Action Items Based on Logs**:
Most important reference: conditions in @train_loop.py which generate eval_ko_in_steps/hard_accuracy as logged to wandb. These track the stepwise accuracy readings of an inner loop, and show great performance.
2. Compare generator initialization between training eval and GUI (especially graph globals initialization)
3. Inspect first generator step: why does it increase loss from 0.0266 → 0.2349 (9x)? Check `update_steps` counter and graph globals
4. Verify graph state (globals, hidden features) matches training eval initialization exactly
5. Investigate why loss continues to drift upward over 200 steps despite correct initialization

---

### 🔴 Issue 3: PERTURB Resets Logits to Previous State
**Problem**: When accuracy drifts to ~0.6 and PERTURB is clicked, accuracy jumps back to ~0.8. This suggests PERTURB is resetting logits to a previous state.

**Observations**:
- PERTURB should only apply damage, not reset logits
- Current implementation: `damaged_logits = [l.copy() for l in self.logits0]` (line 1044)
- This resets to `logits0` (preconfigured state) before applying damage
- If `logits0` is from preconfigure, it should be at accuracy 1.0, not 0.8

**Root Cause**:
- `_apply_gate_damage_perturbation()` resets logits to `logits0` before applying damage
- This is intentional to prevent damage accumulation, but:
  - If current logits have drifted, resetting to `logits0` restores better state
  - This masks the drift issue rather than fixing it
  - Should preserve current logits and apply damage on top

**Investigation Needed**:
- Verify what `logits0` actually contains (should be preconfigured, accuracy 1.0)
- Check if `logits0` is being modified during optimization (shouldn't be)
- Determine if reset to `logits0` is desired behavior or bug -> conclusion: bug, we want to let state accumulate
- Consider preserving current logits and applying damage incrementally

**Potential Fix**:
- Option A: Apply damage to current logits (preserve drift, show true damage effect)
- Option B: Reset to `logits0` only if explicitly requested (separate "Reset + Damage" action)
- Option C: Track both `logits0` (preconfigured) and `logits_before_damage` (last good state)

---

## Investigation Priority

1. **High Priority**: Issue 2 (Self-Attention drift) - Core functionality broken
2. **High Priority**: Issue 3 (PERTURB reset) - Masks underlying drift issue

## Next Steps

### Immediate Priority (Based on Log Evidence)
1. **Inspect first generator step**: Why does loss increase from 0.0266 → 0.2349 (9x)? Check `update_steps` counter and graph globals initialization
2. **Compare graph globals initialization** between training eval and GUI generator (especially `update_steps` counter)
3. **Investigate progressive drift**: Why does loss continue to increase from 0.2349 → 29.7506 → 88.3226 over 200 steps?

### High Priority
4. **Compare exact initialization and step execution** between `evaluate_model_stepwise_generator` (GUI) and `evaluate_model_stepwise_batched` (training eval)
5. **Verify graph state (globals, hidden features)** matches training eval initialization exactly
6. **Add debug logging** to track logits state, accuracy, and graph globals at each step

### Medium Priority
7. **Verify checkpoint loading** includes all necessary model state
8. **Fix PERTURB to preserve current logits** (or make reset explicit) (Issue 3)
9. **Cross-reference wandb eval panel** for run `pljk5kp6` (twilight-butterfly-686) to compare training eval metrics with GUI behavior


