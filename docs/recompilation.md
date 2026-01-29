# Recompilation strategy for message-passing curriculum (JAX)

**Summary**: Optional toggle to grow `n_message_steps` over training in a fixed number of phases (e.g. 3 or 10 recompilations up to T_max), with a single schedule driving both T and beta-compatible loss-step selection.

This note outlines an **optional** strategy to vary `n_message_steps` over training by recompiling the jitted training step at a small number of milestones. It is designed to work with the beta schedule (early → late loss-step focus) and the low-footprint loss in `low_footprint_loss.md`.

**Implementation**: We use the existing schedulers in `boolean_nca_cc/training/schedulers.py` — `get_message_steps_scheduler` and `get_current_message_steps_and_batch_size` — rather than introducing a new function. When the message-steps schedule is disabled, behaviour falls back to a single fixed `n_message_steps` (current behaviour).

## Motivation

- **Single compile (current default)**: One value of `n_message_steps` for the whole run. Always run T steps; only *which* step’s loss is used can vary (e.g. beta, horizon).
- **Recompilation strategy**: Increase T over training in phases (e.g. 5 → 20 → 50 → 100). Fewer steps early saves compute; later phases use more steps. JAX recompiles only when `n_message_steps` (a static arg) changes, so we control how often that happens.

## Toggle and parameters

- **Toggle**: Use the existing message-steps schedule: `training.message_steps_schedule.enabled` (or a single `training.recompilation_schedule: bool` that is mapped to it). When `False`, use a single fixed `n_message_steps` for the entire run (current behaviour; `base_steps` from config). When `True`, derive `n_message_steps` per epoch from the schedule.
- **Parameters** (when enabled): For the recompilation use case we use the existing **step** schedule type:
  - `max_n_message_steps: int` — upper bound T_max (e.g. 100). Maps to `final_steps` and to the last entry of `step_values`.
  - `num_recompile_phases: int` — number of distinct T values (recompilations) over the run (e.g. 3 or 10). This defines the length of `step_values` and the number of phase boundaries.
- **Derived (for step schedule)**: Phase values `T_p` can be evenly spaced: `T_p = round((p + 1) / num_recompile_phases * max_n_message_steps)` for `p = 0 .. num_recompile_phases - 1`, with minimum 1 and last phase exactly `max_n_message_steps`. Then `step_values = [T_0, T_1, ..., T_{num_recompile_phases-1}]`. Phase boundaries: `step_intervals = [total_epochs * (p + 1) // num_recompile_phases - 1 for p in range(num_recompile_phases - 1)]` so that epochs `0 .. step_intervals[0]-1` use T_0, `step_intervals[0] .. step_intervals[1]-1` use T_1, etc. (scheduler switches when `epoch >= interval_epoch`).

**Alternative**: Explicit `recompile_milestones: List[int]` (e.g. `[5, 20, 50, 100]`) can be used directly as `step_values`; then `step_intervals` are the epoch indices at which to switch (e.g. evenly spaced over total_epochs).

## Epoch → n_message_steps mapping (existing API)

- **Single source of truth**: Use `get_message_steps_scheduler(schedule_config, total_epochs)` to obtain a callable `scheduler(epoch) -> int`. The existing `get_current_message_steps_and_batch_size(epoch, schedule_config, total_epochs, base_steps, base_batch_size)` wraps this: when schedule is disabled it returns `(base_steps, base_batch_size)`; when enabled it returns `(scheduler(epoch), current_batch_size)` (with optional constant-product batch scaling).
- **Rule**: For recompilation we use `type: "step"` with `step_intervals` and `step_values` as above (or derived from `num_recompile_phases` and `max_n_message_steps` in a small adapter).
- **Usage**: At the start of each epoch, obtain `n_message_steps` (and optionally batch size) via `get_current_message_steps_and_batch_size(epoch, schedule_config, total_epochs, base_steps=cfg.training.n_message_steps, base_batch_size=...)` and pass `n_message_steps` into every `pool_train_step` in that epoch. JIT will compile on first use of each new T and reuse the cache thereafter.
- **Pre-compute (recommended)**: At startup, build the scheduler once and pre-compute `n_message_steps_per_epoch = [get_current_message_steps_and_batch_size(e, schedule_config, total_epochs, base_steps, base_batch_size)[0] for e in range(total_epochs)]`. Then each epoch use `n_message_steps = n_message_steps_per_epoch[epoch]`. This avoids per-epoch work and makes warm-up recompilation easy (collect unique T’s from the list and trigger one compile per T before the training loop).

## Synchronisation with beta schedule

- **Same progress**: Use the same notion of “training progress” for both the recompilation phase and the beta schedule (e.g. `epoch / max(total_epochs - 1, 1)`).
- **Loss step**: When using random/beta loss-step selection, pass the **current** `n_message_steps` into `get_step_beta(..., n_message_steps=n_message_steps, training_progress=...)` so the anchor step always lies in `[0, n_message_steps - 1]`. No separate sync logic: the schedule that chooses T also bounds loss_step.
- **Horizon**: If using a horizon window, its indices are already clamped to `[0, n_message_steps - 1]` by the current implementation; no change needed.

## JIT timing

- Recompilation is **lazy**: the first call to `pool_train_step` with a new `n_message_steps` triggers a compile; subsequent calls with the same value reuse the cache.
- To avoid a slow step when entering a new phase, optionally **warm up** at startup: after constructing the jitted step, call it once with each T that will be used (e.g. for each milestone), so all compiles happen before the training loop.

## Config sketch

Use the existing message-steps schedule config shape. For recompilation, set `enabled: true` and `type: "step"`; step values can be derived from recompilation params or set explicitly:

```yaml
training:
  n_message_steps: 6   # Used as base_steps when schedule is disabled
  # Message steps schedule (schedulers.get_message_steps_scheduler / get_current_message_steps_and_batch_size)
  message_steps_schedule:
    enabled: false     # When false: fixed n_message_steps for whole run (current behaviour)
    type: "step"       # When enabled, use "step" for phase-wise T
    initial_steps: 5   # Ignored for step type; can match first phase
    final_steps: 100   # max_n_message_steps (last phase)
    # For recompilation: derive from num_recompile_phases and total_epochs, or set explicitly:
    step_intervals: [2730, 5460]   # Epoch boundaries (e.g. total_epochs=8193, 3 phases → 8193//3-1, 2*8193//3-1)
    step_values: [33, 66, 100]     # T per phase (e.g. num_recompile_phases=3, max=100 → ~33, ~66, 100). If null, scheduler auto-generates from initial_steps and final_steps with len(step_intervals)+1 phases.
    constant_product: null        # Optional: meta_batch_size * n_message_steps = constant for memory
```

- If `message_steps_schedule.enabled` is `False`, `n_message_steps` is taken from `training.n_message_steps` (base_steps) and used for all epochs.
- If `True`, `n_message_steps` is **overridden** per epoch by the scheduler; `step_values` and `step_intervals` (or values derived from `max_n_message_steps` and `num_recompile_phases`) define the schedule. A thin adapter can build `step_intervals` / `step_values` from `num_recompile_phases` and `max_n_message_steps` so callers can configure recompilation without hand-writing the lists.

## Relation to existing schedulers (`schedulers.py`)

- **`get_message_steps_scheduler(schedule_config, total_epochs)`**: Returns a callable `epoch -> n_message_steps`. For recompilation we use `type: "step"` with `step_intervals` (epoch boundaries) and `step_values` (T per phase). When `enabled: false`, the factory returns a constant callable using `initial_steps`.
- **`get_current_message_steps_and_batch_size(epoch, schedule_config, total_epochs, base_steps, base_batch_size)`**: When schedule is disabled, returns `(base_steps, base_batch_size)`. When enabled, calls the scheduler for the current epoch and optionally scales batch size via `constant_product`. This is the single entry point for “T (and optionally batch size) for this epoch”.
- **Adaptations for recompilation**: No signature change required. Config must provide `message_steps_schedule` with `enabled`, `type: "step"`, and either (a) explicit `step_intervals` / `step_values`, or (b) a small helper that builds those from `max_n_message_steps` and `num_recompile_phases` and injects them into the config before creating the scheduler.

## Relation to other docs

- **low_footprint_loss.md**: Unchanged. The low-footprint contract (no time-stacking of large objects, scan, loss at one step or window) remains; we only change **how** `n_message_steps` is chosen each epoch (fixed vs schedule).
- **Beta (Step 5 in low_footprint_loss.md)**: Recompilation and beta are complementary: recompilation grows the **number** of steps computed; beta shifts **which** step (or window) is used for the loss within each phase.
