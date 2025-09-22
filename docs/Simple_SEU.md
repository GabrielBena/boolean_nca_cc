## Simple SEU (Reversible Bit Perturbation) Mode

This document summarizes the minimal reversible damage implementation (SEU-like), how it integrates with the existing training/eval pipeline, and how to extend greedy knockout patterns to cycle with a per-circuit perturbation counter.

### What “reversible” means here
- Nodes are NOT removed from the attention graph and are NOT zeroed out of residual updates.
- A one-shot logit bias is applied to damaged nodes at the first model step of the episode. Subsequent steps apply normal residual updates so the gate can be “healed.”
- Evaluation remains unchanged and still reports: step 0 baseline (pre-damage), step 1 damage application, then recovery over following steps.

### Where it’s implemented
- Model toggle and behavior:
  - `boolean_nca_cc/models/self_attention.py` (class `CircuitSelfAttention`)
    - Constructor now accepts `damage_behavior: "permanent" | "reversible"` and `reversible_bias: float`.
    - In `__call__`:
      - Permanent mode: unchanged (prune attention, clamp logits to large negative, zero residual updates for damaged nodes).
      - Reversible mode: attention mask ignores knockout (keep connectivity), no zeroing; at first step (`globals[..., 1] == 0`), add a logit bias to damaged nodes. All steps after that are clean updates (recovery path).

- Config knobs:
  - `configs/model/self_attention.yaml`
    - `damage_behavior: "reversible"`
    - `reversible_bias: -10.0` (tunable)

- Pool damage tracking:
  - `boolean_nca_cc/training/pool/pool.py`
    - The pool stores `knockout_patterns` per circuit slot.
    - New `perturb_counter: jnp.int32[pool_size]` tracks how many times each circuit has been damaged.
    - Incremented automatically in `GraphPool.apply_knockouts(idxs, new_patterns)`.

### How training and evaluation align
- Training (loop path): sampled circuits receive their stored `knockout_pattern`; the model applies the one-shot bias on the first model call of the episode and then recovers.
- Evaluation (`evaluation.py` batched stepwise): unchanged. Step 0 metrics are computed before the first model call; from step 1 onward, the model sees the `knockout_pattern`, applies the one-shot bias (reversible), and the plotted trajectory shows recovery.

### Pattern lifecycle
- A circuit’s pattern is just an array stored in the pool for that slot. New damage events can overwrite the old pattern at any time (via `apply_knockouts` or `damage_fraction`). Each overwrite increments `perturb_counter` for that slot.

---

## Greedy knockout patterns that cycle with perturb_counter

Goal: For greedy damage, cycle through a fixed, ordered list of gate indices such that:
- First perturbation of a circuit uses `greedy_indices[0]` (knock out that gate only).
- Second perturbation uses `greedy_indices[1]`, and so on; wrap around with modulo when reaching the end.

### Current utilities
- `boolean_nca_cc/training/pool/structural_perturbation.py`
  - `DEFAULT_GREEDY_ORDERED_INDICES`: default ordering.
  - `create_greedy_knockout_pattern(ordered_indices, layer_sizes, max_gates)`
  - `create_knockout_vocabulary(rng, vocabulary_size, layer_sizes, damage_prob, damage_mode, ordered_indices)`
    - Today, in `damage_mode == "greedy"`, it creates a single deterministic pattern with the first `int(damage_prob)` indices and repeats it across the vocabulary.

### Implemented: Greedy rolling-window cycling (ties to perturb_counter)

We implemented a rolling-window greedy perturbation that advances with each circuit’s `perturb_counter` without modifying the existing `damage_fraction(...)` function.

- structural_perturbation.py
  - Added `create_group_greedy_pattern(ordered_indices, layer_sizes, start, size)`
    - Builds a boolean mask for a wrap-around window over `ordered_indices`.
    - JAX-friendly (no Python int coercions on tracers); uses mod/arange/gather.

- pool/pool.py
  - `GraphPool.apply_knockouts(idxs, new_knockout_patterns)` increments `perturb_counter` for affected indices.
  - `damage_fraction(...)` left unchanged; we compose greedy masks in the train loop instead.

- training/train_loop.py
  - At configured damage epochs, we select `damaged_idxs` via `get_reset_indices(...)` (same selection logic as before).
  - For each damaged circuit, we compute:
    - `count = perturb_counter[idx]`
    - `start = (count * greedy_window_size) % len(greedy_ordered_indices)`
    - `size = greedy_window_size` (configured as 5 via damage_prob linkage)
  - Build masks via `create_group_greedy_pattern(...)` (vmapped) and apply with `apply_knockouts(...)`.
  - This rotates the window per circuit over time and increments `perturb_counter` automatically.

### Config cheat sheet
- Training
  - `pool.greedy_ordered_indices: [...]`  # absolute node indices (non-input/output)
  - `pool.greedy_window_size: ${pool.damage_prob}`  # Links to damage_prob, actual value is 5
  - `pool.damage_pool_enabled: true`
  - `pool.damage_pool_interval, pool.damage_pool_fraction, pool.damage_strategy`: as before
  - `pool.damage_prob: 5`  # Controls both window size and fallback pattern generation

- Evaluation  
  - `eval.knockout_eval.damage_prob: ${pool.damage_prob}` # Inherits from pool config (5)
  - `eval.knockout_eval.greedy_eval_enabled: true` # Enables periodic greedy re-damage during eval
  - `eval.knockout_eval.greedy_window_size: ${pool.greedy_window_size}` # Reuses pool's window size
  - `eval.knockout_eval.greedy_injection_recover_steps: 10` # Recovery steps between injections
  - `eval.knockout_eval.greedy_num_injections: 10` # Number of injections then damage-free tail

### Interaction: damage_prob vs greedy_window_size
- **Unified Parameter**: `damage_prob` controls the window size for greedy mode in both training and evaluation (value: 5)
- **Training**: `greedy_window_size: ${pool.damage_prob}` creates a 5-gate rolling window
- **Evaluation**: Can use the same patterns via vocabulary OR enable `greedy_eval_enabled` for periodic greedy re-damage during stepwise evaluation