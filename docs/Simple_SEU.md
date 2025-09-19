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

### Desired changes (where to implement)
1) structural_perturbation.py
   - Add a helper to build a singleton-gate mask for a single greedy index:
     - `create_singleton_greedy_pattern(index: int, layer_sizes) -> jnp.bool_[num_nodes]`
       - Returns a boolean mask with exactly one gate True at the specified absolute node index (validated to be a gate, not input/output).
   - Option A (vocabulary approach): update `create_knockout_vocabulary(..., damage_mode="greedy_singleton")` to return a vocabulary where each entry knocks out exactly one gate from the `ordered_indices` list. Vocabulary size should match `len(ordered_indices)` or the desired subset.
   - Option B (on-the-fly approach – recommended): keep `create_knockout_vocabulary` unchanged and generate singleton greedy masks inside the pool at damage time (see below). This avoids large vocabularies and ties selection directly to `perturb_counter`.

2) GraphPool.damage_fraction(...) in `boolean_nca_cc/training/pool/pool.py`
   - Extend the signature to accept `damage_mode: str` and `greedy_ordered_indices: Optional[List[int]]`.
   - Add a branch for greedy-cycling mode (e.g., `damage_mode in {"greedy", "greedy_cycle"}`):
     - For the selected `damaged_idxs`, compute per-circuit position:
       - `k = perturb_counter[idx] % len(greedy_ordered_indices)`
       - `gate_idx = greedy_ordered_indices[k]`
     - Build a singleton pattern per selected index using the new helper (`create_singleton_greedy_pattern(gate_idx, layer_sizes)`).
     - Stack those masks into `new_patterns` and call `apply_knockouts(damaged_idxs, new_patterns)`.
     - This naturally increments `perturb_counter` and rotates the greedy index per circuit over time.
   - Keep existing random/vocabulary paths unchanged for other `damage_mode` values.

3) train_loop.py (call site)
   - When calling `circuit_pool.damage_fraction(...)`, pass through `damage_mode` and `greedy_ordered_indices` from config. For greedy cycling, do NOT pass a vocabulary (let `damage_fraction` generate masks on the fly).
   - Config already contains `pool.greedy_ordered_indices`; ensure it is plumbed.

4) Configs (optional)
   - In `configs/config.yaml`, clarify available `pool.damage_mode` values:
     - `"shotgun" | "strip" | "greedy" | "greedy_cycle"`
   - For `greedy_cycle`, rely on pool-level cycling with `perturb_counter` as described above.

### Evaluation impact
- No changes required. Periodic knockout evaluation uses either fresh patterns or vocabulary-based sampling. If you want evaluation to also exercise the greedy cycle deterministically per pattern id, you can generate singleton greedy masks for the eval batch (optional).

### Summary of code touchpoints for greedy cycling
- New helper: `structural_perturbation.create_singleton_greedy_pattern(...)`.
- Extend: `GraphPool.damage_fraction(...)` to accept `damage_mode` + `greedy_ordered_indices` and generate singleton masks using `perturb_counter`.
- Pass-through: `train_loop.py` to provide `damage_mode` and `greedy_ordered_indices` when applying pool damage.
- Keep `evaluation.py` unchanged.


