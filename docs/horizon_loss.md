# Horizon Loss and Loss-Step Selection

This doc describes **horizon loss** (window loss) and **loss-step selection** (random / beta). For low-footprint implementation details (no time-stacking, static gather, snapshot at `loss_step`), see **`docs/low_footprint_loss.md`**.

## Implemented

- **Random loss step**: `loss_step` sampled once per meta-batch in `[random_loss_step_min, n_message_steps)` — see low_footprint_loss Step 1.
- **Horizon loss**: Single-step vs window; window semantics depend on random vs fixed:
  - **Single-step**: loss at `loss_step` only.
  - **Horizon + random**: mean over `[loss_step .. n_message_steps - 1]` (anchor to end).
  - **Horizon + fixed**: mean over last H steps ending at `(n_message_steps - 1)`.
- **Config**: `training.random_loss_step`, `training.random_loss_step_min`, `training.long_horizon_enabled`, `training.long_horizon_size`. `random_loss_step` is passed into `pool_train_step` (static) so horizon window definition branches correctly.

**Code**: `train_loop.py` → `pool_train_step` → `loss_fn_no_scan` (final loss block); `train.py` passes config into `train_model`.

---

## Next Step: Beta Distribution for Loss-Step Selection

**Implementation plan:** See **`docs/low_footprint_loss.md`** → **Step 5** (inserted after scan).

**Goal:** When using random `loss_step`, optionally sample it from a **Beta distribution** that shifts from early steps (early training) to late steps (late training), instead of uniform over `[min_step, T)`.

**Why:** Early on, encourage low loss earlier in the rollout; later, focus on the final steps. Single anchor (`loss_step`) still drives both single-step loss and horizon window (random-horizon = `[loss_step .. T-1]`).

**API (already exists):** `boolean_nca_cc.training.schedulers.get_step_beta`:

```python
def get_step_beta(
    loss_key: jax.random.PRNGKey,
    n_message_steps: int,
    training_progress: float = 0.0,  # e.g. epoch / max(epochs - 1, 1)
    beta_max: float = 10,
    beta_min: float = 0.1,
    min_step: int = 1,
) -> jp.ndarray:  # returns int step index
```

- **Early** (`training_progress ≈ 0`): left-skewed Beta → favors **early** steps.
- **Late** (`training_progress ≈ 1`): right-skewed Beta → favors **late** steps.
- Returns step in `[min_step, n_message_steps - 1]`.

**Integration (to implement):**

1. **Config**: Add `training.use_beta_loss_step: bool` (default `False`). Optional: `training.beta_loss_step_beta_max`, `training.beta_loss_step_beta_min`, `training.beta_loss_step_min_step` (or reuse `random_loss_step_min`).
2. **Train loop**: When sampling `loss_step` (once per meta-batch, before `pool_train_step`):
   - If `random_loss_step` and `use_beta_loss_step`: call `get_step_beta(loss_key, n_message_steps, training_progress=epoch / max(epochs - 1, 1), min_step=random_loss_step_min, ...)` and use the returned step (cast to int for static).
   - Else if `random_loss_step`: keep current `jax.random.randint(..., random_loss_step_min, n_message_steps)`.
   - Else: `loss_step = n_message_steps - 1`.
3. **Static args**: `use_beta_loss_step` (and any new beta params used inside a JIT boundary) must be in `static_argnames` if they affect control flow; `loss_step` is already sampled outside JIT, so no change there.

**Combined with horizon:**

- Beta only chooses **which** step is the anchor (`loss_step`).
- Single-step: loss at that step.
- Horizon + random: window from that step to end.
- Horizon + fixed: unchanged (last H steps at final step); beta does not apply when `random_loss_step=False`.

**JAX:** Sampling happens **before** `pool_train_step` (outside JIT), so no traced-value issues. Return value of `get_step_beta` is converted to Python int for use as static `loss_step`.
