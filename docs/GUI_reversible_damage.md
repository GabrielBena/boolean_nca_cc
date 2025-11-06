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

- [ ] Add state: `_viz_damage_mask`, `_viz_flash_ticks`, `damage_bias`.
- [ ] On injection: bake damage into current logits; set `_viz_flash_ticks = 1`; store `_viz_damage_mask`; re-init generator state from current logits.
- [ ] On each tick: update `self.logits` from generator; do not use any execution mask; for visualization, use `_viz_damage_mask` only for the next 1 tick.
- [ ] Ensure accuracy path sees damaged logits at least on the first post‑injection tick.
- [ ] Decide on reset variants: soft reset (preserve damage) vs full reset (clear damage and restore `logits0`).
- [ ] Optionally, add controls: “Single‑tick mask”, “Preserve damage on reset”.


