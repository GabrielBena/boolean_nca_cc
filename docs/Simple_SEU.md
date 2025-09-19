## Simple SEU (Reversible Bit Perturbation) Mode

This document summarizes the minimal reversible damage implementation (SEU-like), how it integrates with the existing training/eval pipeline, and how to extend greedy knockout patterns to cycle with a per-circuit perturbation counter.

### What “reversible” means here
- Nodes are NOT removed from the attention graph and are NOT zeroed out of residual updates.
- A one-shot logit bias is applied to damaged nodes at the first model step of the episode. Subsequent steps apply normal residual updates so the gate can be “healed.”
- Evaluation remains unchanged and still reports: step 0 baseline (pre-damage), step 1 damage application, then recovery over following steps.

### Where it’s implemented
- Model toggle and behavior:
  - `boolean_nca_cc/models/self_attention.py` (class `CircuitSelfAttention`)
    - Constructor now accepts `damage_behavior: "hard" | "reversible"` and `reversible_bias: float`.
    - In `__call__`:
      - Hard mode: unchanged (prune attention, clamp logits to large negative, zero residual updates for damaged nodes).
      - Reversible mode: attention mask ignores knockout (keep connectivity), no zeroing; at first step (`globals[..., 1] == 0`), add a logit bias to damaged nodes. All steps after that are clean updates (recovery path).

- Config knobs:
  - `configs/model/self_attention.yaml`
    - `damage_behavior: "reversible"`
    - `reversible_bias: -4.0` (tunable)

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

## Next step: Greedy knockout patterns that cycle with perturb_counter

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
    - `size = greedy_window_size` (defaults to 1 for single-index iteration)
  - Build masks via `create_group_greedy_pattern(...)` (vmapped) and apply with `apply_knockouts(...)`.
  - This rotates the window per circuit over time and increments `perturb_counter` automatically.

### Config cheat sheet
- Training
  - `pool.greedy_ordered_indices: [...]`  # absolute node indices (non-input/output)
  - `pool.greedy_window_size: 1|5|...`    # 1 = single-index cycling; k = k-wide rolling window
  - `pool.damage_pool_enabled: true`
  - `pool.damage_pool_interval, pool.damage_pool_fraction, pool.damage_strategy`: as before
  - Note: when `greedy_ordered_indices` is set, the training damage path uses the rolling-window greedy logic and ignores `pool.damage_prob` (still used for random/vocab fallbacks only).

- Evaluation
  - Unchanged. To test a fixed greedy window of width k, set `eval.knockout_eval.damage_prob: k` and `pool.damage_mode: "greedy"` (this yields the first k indices, not sliding windows).
  - To mirror the true sliding-window behavior in eval, provide a vocabulary of k-wide windows over the greedy list and evaluate on that vocabulary (or add a small generator in eval).

### Interaction: damage_prob vs greedy_window_size
- Training (greedy-window active): `greedy_window_size` determines how many greedy indices are perturbed per damage event; `damage_prob` is ignored in this branch.
- Evaluation (greedy mode): `damage_prob` controls how many greedy indices are included (e.g., 5 = first 5). It does not automatically slide; sliding requires a custom vocabulary.