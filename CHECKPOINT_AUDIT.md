# Checkpoint Audit — Bit-Faithful Mid-Training Resume

**Scope.** Audit of all mutable training state in the `boolean_nca_cc` training stack
to enable preemption-safe, *bit-faithful* mid-training resume. Target use case:
a V100 lowprio job on the UZH cluster is preempted at epoch N; we relaunch on
another GPU and want the resumed run to be as close as possible to one that never
stopped.

**Status quo.** `save_checkpoint` exists and is exercised, but only to reload a
*finished* model for the demo probe / web demo. **Nobody has resumed-to-continue.**
The current schema saves model + optimizer + a metrics dict + config + step. It
does **not** save the circuit pool, the RNG key stream, or the best-metric tracker.
This document is the spec for closing those gaps.

**Files audited (canonical line numbers used throughout):**
- `train.py` — entry point; `train_model(...)` call at `train.py:880`; post-train save at `train.py:993-1010`; demo-probe reload at `train.py:1078-1124`.
- `boolean_nca_cc/training/train_loop.py` — `train_model` (`:633`), the epoch loop (`:1447-1796`), `_compute_loss_and_gradients` / `_pool_train_step` (`:1044`, `:1304`).
- `boolean_nca_cc/training/checkpointing.py` — `save_checkpoint` (`:392`), loaders (`:262`, `:446`), `BestModelTracker` (`:460`), `EarlyStopping` (`:767`), `setup_checkpoint_dir` (`:517`), `save_periodic_checkpoint` (`:532`), `save_best_checkpoint` (`:574`).
- `boolean_nca_cc/training/pool/pool.py` — `GraphPool` struct (`:68`), `initialize_graph_pool` (`:734`).
- `boolean_nca_cc/training/schedulers.py` — `get_learning_rate_schedule` (`:26`).
- `boolean_nca_cc/training/evaluation.py` — `run_model_scan_with_loss` scan-key derivation (`:627-808`).

---

## 1. Inventory of mutable training state

Everything that *evolves* during `train_model`'s epoch loop and would need
restoring for a faithful resume. Grouped by subsystem.

### 1.1 Model parameters
- **Name:** `model` (nnx params).
- **Lives in:** the `nnx.Module` (`CircuitGNN` / `CircuitGatheredAttention` /
  `CircuitSelfAttention` / `PerceiverCircuitAttention`), instantiated in
  `train.py:674` and passed as `init_model`; bound to `model` at `train_loop.py:930`.
- **Structure/dtype:** `nnx.state(model, nnx.Param)` — a nested `nnx.State` of
  `f32` arrays. Updated every step by `optimizer.update(model, grads)` (`train_loop.py:1389`).
- **Init:** Hydra `instantiate` with `rngs=nnx.Rngs(params=init_key)` where
  `init_key` is split from `train_key` at `train.py:651`.
- **Currently saved:** YES (`nnx.state(model)`).

### 1.2 Optimizer state (nnx.Optimizer)
- **Name:** `optimizer`.
- **Optimizer:** **AdamW**, wrapped in an `optax.chain`. Built at
  `train_loop.py:957-965` (static-schedule path):
  ```python
  opt_fn = optax.chain(
      optax.clip_by_global_norm(1.0),
      optax.zero_nans(),
      optax.adamw(learning_rate=schedule, weight_decay=weight_decay),
  )
  optimizer = nnx.Optimizer(model, opt_fn, wrt=nnx.Param)
  ```
  (Adaptive path at `:947-956` uses `optax.inject_hyperparams(optax.adamw)` — see §1.4.)
- **State contents (verified empirically with a toy model):** `nnx.state(optimizer)`
  has top-level keys `['opt_state', 'step']`.
  - `opt_state` is the chain's per-transform state tuple:
    `(EmptyState, ZeroNansState, <adamw inner>)`. The AdamW inner carries the
    **first and second moment estimates** (`mu`, `nu`) — one pytree each, same
    shape as the params — plus its own scalar `count`.
  - `step` is a scalar **`uint32`** counter on the `nnx.Optimizer` itself. It
    **increments by 1 on every `optimizer.update`** (verified: 0 → 1 → 2). This
    is the count that an optax LR `schedule` would be evaluated at *if the
    optimizer drove the schedule* — but here it does not (see §1.4 — the
    schedule is evaluated at the Python `epoch`, not `optimizer.step`). Still,
    `step` is part of the saved optimizer state and must round-trip.
- **Init:** fresh AdamW moments = zeros; `step = 0`.
- **Currently saved:** YES (`nnx.state(optimizer)`). The Adam moments and `step`
  are inside that state and therefore captured.
- **Gotcha:** the optimizer holds a *reference* to `model` and a *static*
  `opt_fn` (the optax GradientTransformation, including the baked-in schedule).
  `nnx.state(optimizer)` captures the **arrays** (moments, step) but NOT the
  `opt_fn` closure or the schedule. On resume we must reconstruct `opt_fn` with
  the **identical schedule** (same `lr_scheduler`, `learning_rate`, `epochs`,
  `lr_scheduler_params`) before `nnx.update`-ing the saved arrays in. See §5.

### 1.3 LR scheduler
- **Mechanism:** for `constant`/`exponential`/`cosine`/`linear_warmup`, the
  scheduler is an **optax `optax.Schedule`** — a pure, **stateless** function of
  a step integer (verified: `cosine_decay_schedule(...)(500)` returns the same
  value on repeated calls; it is a plain Python function/closure).
- **How LR is applied:** the schedule is *baked into* `opt_fn` at construction
  (`adamw(learning_rate=schedule, ...)`, `train_loop.py:963`). At call time optax
  evaluates `schedule(opt_state_count)` internally.
  **Important subtlety:** the schedule passed to `adamw` is driven by AdamW's own
  internal `count`, which equals the number of `update` calls = number of epochs
  elapsed (one update per epoch). Separately, for *logging only*, the loop
  recomputes `schedule_value = schedule(epoch)` at `train_loop.py:1636-1637`.
- **Where the "current LR" / step counter lives:** there is **no separate LR
  state object** for the static schedulers. The effective step is the AdamW
  `count` inside `opt_state` (§1.2), which is restored with the optimizer state.
  So for static schedules, restoring optimizer state + reconstructing the same
  `opt_fn` is sufficient — the LR auto-resumes at the correct value.
- **Adaptive / reduce_on_plateau path:** `lr_scheduler in {"adaptive",
  "reduce_on_plateau"}` returns a tuple `(constant_schedule(1.0),
  adaptive_scheduler)` (`schedulers.py:96-135`). Here `adaptive_scheduler` is a
  **stateful Python object** (`AdaptiveLRScheduler` / `ReduceOnPlateauScheduler`)
  that tracks loss EMAs, reduction counts, cooldown, patience, etc., and is
  updated every epoch (`train_loop.py:1603-1618`). Its LR is injected into the
  optimizer via `optax.inject_hyperparams`. **This object's internal state is
  NOT in `nnx.state` and is NOT saved.** (Cluster runs use cosine/constant by
  default — `configs/config.yaml` — so this path is likely inactive for the
  preemption use case, but must be flagged.)
- **Currently saved:** PARTIAL. Static schedule: effectively yes (via optimizer
  `count`). Adaptive scheduler object: NO.

### 1.4 The circuit POOL (`GraphPool`)
- **Name:** `circuit_pool`. Built at `train_loop.py:980` via
  `initialize_graph_pool`; reassigned every epoch by `pool.update` /
  `reset_fraction` (`train_loop.py:1395`, `:1576`).
- **It is a `flax.struct.PyTreeNode`** (`pool.py:68`) — a registered JAX pytree,
  so it can be `nnx.state`-free-pickled / tree-mapped directly. Fields:
  - `size: int` — `pytree_node=False` (static). `cfg.pool.size` (default **4096**).
  - `graphs: jraph.GraphsTuple` — the batched graph, leading axis = `pool_size`.
    This is the bulk of the memory. Per-slot it carries:
    - `nodes` dict: includes the **circuit hidden state** (`circuit_hidden_dim`
      f32 per node — the NCA "carry"), `layer`, `gate_knockout_mask`, optionally
      `dist_pe` / `rwse`. Shapes `[pool, n_node, ...]`.
    - `edges`, `senders`, `receivers` — topology (refreshed on wire-shuffle events).
    - `globals` — `loss` (f32 `[pool]`) and `update_steps` (the per-circuit
      **age/maturity** counter, f32/int `[pool]`); see `_log_pool_scatter`
      `train_loop.py:118-122` and `globals.update_steps` reads in `pool.py`.
  - `wires: PyTree` — list of per-layer wire index arrays, shape
    `[pool, arity, group_n]`, integer indices (from `gen_wires`,
    `model.py:170-186`). Evolves only on wire-shuffle (topology refresh) and on
    pool reset.
  - `logits: PyTree` — list of per-layer gate logits, `[pool, gate_n//gs, gs, 1<<arity]`,
    **f32**. These are the *evolved circuit parameters* — the inner-loop solution
    each circuit has reached. Updated every step (`pool.update` extracts updated
    logits from the stepped graphs).
  - `gate_masks: [pool, total_gates]` f32 (0=knocked out, 1=active). Evolves under
    probabilistic damage; written back via `updated_graphs.nodes["gate_knockout_mask"]`
    (`train_loop.py:1392`).
  - `damage_count: [pool]` int32 — knockouts per circuit. Derived from gate_masks
    on `pool.update` (`pool.py:224-227`).
  - `reset_counter: [pool]` int32 — epochs-since-reset bookkeeping
    (`pool.py:134-135`, incremented on reset `pool.py:522-524`).
  - `y_task: [pool, 2^input_n, output_n]` f32 or `None` — per-circuit task target
    table (per-circuit-task meta-learning; `pool.py:103`). Bound to slot lifetime;
    replaced on reset. `None` on the legacy fixed-task path.
- **How mutated each step:** sample a batch of `idxs` (`train_loop.py:1452`),
  run `n_message_steps` of the model inner loop, then `pool.update(idxs, ...)`
  writes back the stepped graphs/logits/gate_masks for *those slots only*
  (`train_loop.py:1395-1400`). Every `reset_interval` epochs a fraction is reset
  with fresh circuits (`train_loop.py:1513-1589`).
- **This is the meta-curriculum.** The pool's distribution of maturities, losses,
  damage, and (in task mode) tasks is the *training environment* the model sees.
  It is path-dependent and cannot be reconstructed from the epoch number alone.
- **Currently saved:** **NO.** This is the single biggest faithfulness gap.

### 1.5 RNG / PRNG key streams
There are two root keys (`train.py:489-494`):
- `train_key = jax.random.PRNGKey(cfg.seed)` (default seed **0**).
- `eval_key = jax.random.PRNGKey(cfg.eval_seed)` or `fold_in(train_key, 1)` if
  `eval_seed is None` (default null → derived).

`eval_key` is effectively **constant** for the whole run (used for OOD/eval
datasets, fixed-mode pool keys, discrete damage). `train_key` is the one that
**mutates statefully** across epochs. Inside `train_model`, before the loop,
`train_key` is consumed/split several times (model isn't — that's in `train.py`;
but pool init at `:974`, task init at `:898`). Then in the loop, **every epoch**:
- `train_key, sample_key, loss_key = jax.random.split(train_key, 3)` (`train_loop.py:1451`)
  — pool sampling key + the loss/scan key.
- On reset epochs, additional splits: `train_key, pool_key` (`:1536`),
  `train_key, fresh_task_key` (`:1547`), `train_key, reset_key` (`:1574`); genetic
  mode uses `train_key, reset_key` (`:1515`).

Crucially **`train_key` is carried and re-split each iteration** (`train_key, ... = jax.random.split(train_key, ...)`), so it is a *stateful chain*, not
`fold_in(root, epoch)`. The downstream `loss_key` is then split inside the JIT
step to derive ALL per-step randomness: in `run_model_scan_with_loss`,
`data_key, damage_key, subsample_key = jax.random.split(scan_key, 3)`
(`evaluation.py:627-628`), and per-step damage/shuffle keys are
`split`/`fold_in`-derived from those (`:691`, `:698`, `:758`). So per-step
randomness is a pure function of that epoch's `loss_key`. See §4.

- **Currently saved:** **NO.** `train_key`'s evolving value is lost on restart.

### 1.6 Epoch/step counters, best-metric trackers, schedules
- **Epoch counter:** the Python loop variable `epoch` in `for epoch in pbar:`
  (`train_loop.py:1448`). Drives: beta-loss-step progress (`:1104`), reset
  scheduling via `last_reset_epoch` (`:1440`, `:1513`), periodic eval gating
  (`:1710`), LR logging (`:1637`), early-stopping `min_epochs` (`:894`).
  **Saved:** the `step` arg to `save_checkpoint` is set to `epoch` in
  `save_periodic_checkpoint` (`checkpointing.py:553`), so the epoch number *is*
  in the checkpoint dict under `"step"`. But nothing in the loop reads it back —
  there is no `start_epoch`/resume plumbing.
- **`last_reset_epoch`:** initialized `-1` (`train_loop.py:1440`), updated to
  `epoch` on each reset (`:1589`). Governs *when the next pool reset fires*.
  **NOT saved.** If not restored, the reset phase shifts on resume.
- **`BestModelTracker`** (`checkpointing.py:460`): `best_metrics` (metric_key →
  best value) and `best_epochs` (metric_key → epoch). Built fresh at
  `train_loop.py:1433`. Drives which periodic eval triggers a `best_model_*.pkl`
  save. **NOT saved** — on resume it restarts at ±inf, so the first post-resume
  eval will spuriously "improve" and overwrite the best checkpoint even if the
  pre-preemption model was better.
- **`EarlyStopping`** (`checkpointing.py:767`): mutable fields `count`
  (consecutive evals above threshold), `first_epoch`, `triggered`
  (`checkpointing.py:816-818`). Constructed in `train.py:964`. **NOT saved** — on
  resume the patience counter restarts at 0, so a run about to early-stop will
  run extra epochs.
- **Metrics history lists:** `losses`, `hard_losses`, `accuracies`,
  `hard_accuracies`, `reset_steps` (`train_loop.py:911-924`). `train_model`
  *can* re-seed them from `initial_metrics` (`:918-924`), but `train.py` never
  passes `initial_metrics`. These are saved in the metrics dict but not re-read.
- **Curriculum / scheduled quantities:** `reset_interval`, `reset_fraction`,
  damage `p_fault`/`p_fault_onset`, `p_shuffle`, burst params — all **static
  scalars** computed once before the loop (in `train.py` via
  `compute_damage_params` / `compute_shuffle_params`, and in
  `process_pool_configuration`). They are pure functions of config, so they
  recompute identically on resume *provided the same config is reloaded*. The
  only epoch-dependent quantity here is the beta-loss-step distribution
  (`get_step_beta`, depends on `epoch/(epochs-1)`) — restored for free by
  restoring `epoch`.
- **Currently saved:** epoch via `"step"` (but unused on load); trackers NO;
  curriculum recomputable from config.

### 1.7 WandB run id / step
- **Run id:** in `train.py` wandb is initialized in `main` (`train.py:506-514`)
  with `reinit="finish_previous"`; in `train_model` `_init_wandb` calls
  `wandb.init(..., resume="allow")` (`train_loop.py:65-68`). The run id
  (`wandb.run.id`) is used for the checkpoint subdir (`run_<id>`, `setup_checkpoint_dir`
  `checkpointing.py:517-529`) and for the best-model artifact path the demo probe
  reloads (`train.py:1084-1089`).
- **WandB step:** wandb's internal monotonic step; `wandb_run.log(...)` is called
  with bare metric dicts (no explicit `step=`), so on a fresh `wandb.init` the
  step restarts at 0 even though `training/epoch` is logged as a field.
- **Currently saved:** NO (run id not persisted to checkpoint). COSMETIC for
  training correctness; matters for logging continuity / artifact paths.

---

## 2. What `save_checkpoint` currently captures vs. the inventory

`save_checkpoint` (`checkpointing.py:410-418`) writes exactly this dict:

```python
checkpoint = {
    "model":     nnx.state(model),
    "optimizer": nnx.state(optimizer),
    "metrics":   metrics,          # caller-supplied dict
    "config":    OmegaConf.to_container(cfg, resolve=True),  # or raw cfg
    "step":      step,
}
```

At the call sites the `metrics` dict is `{losses, hard_losses, accuracies,
hard_accuracies, reset_steps}` (`train_loop.py:1779-1785` periodic;
`train.py:1000-1005` final) and `cfg` is sometimes just `{"epoch": epoch}`
(`checkpointing.py:552`, `:597`, `:639`) — i.e. the *periodic* checkpoint
overwrites the real config with `{"epoch": ...}`, so `latest_checkpoint.pkl`
does **not** carry the full hydra config. Only the final/best saves do.

| State item (from §1) | Saved? | Notes |
|---|---|---|
| Model params (`nnx.state(model)`) | **YES** | key `"model"`. |
| AdamW moments `mu`/`nu` | **YES** | inside `nnx.state(optimizer)` → `opt_state`. |
| Optimizer `step`/`count` (uint32) | **YES** | inside `nnx.state(optimizer)`; this is what drives the static LR schedule. |
| `opt_fn` / schedule closure | **NO** | not picklable as state; must be reconstructed from config on load. |
| Adaptive/plateau scheduler object | **NO** | only relevant for `lr_scheduler in {adaptive, reduce_on_plateau}`. |
| Pool `graphs` (hidden state, globals, topology) | **NO** | the meta-curriculum; **biggest gap**. |
| Pool `wires` | **NO** | |
| Pool `logits` (evolved circuit params) | **NO** | |
| Pool `gate_masks` / `damage_count` | **NO** | |
| Pool `reset_counter` | **NO** | |
| Pool `y_task` (per-circuit-task mode) | **NO** | |
| `train_key` (evolving PRNG) | **NO** | |
| `eval_key` | recomputable | pure function of `cfg.seed`/`cfg.eval_seed`. |
| Epoch counter | partial | stored under `"step"` but **never re-read** on resume. |
| `last_reset_epoch` | **NO** | controls reset phase. |
| `BestModelTracker.best_metrics`/`best_epochs` | **NO** | causes spurious best overwrite on resume. |
| `EarlyStopping.count`/`first_epoch`/`triggered` | **NO** | patience restarts. |
| Metrics history lists | YES (in `"metrics"`) | but `train.py` never feeds them back as `initial_metrics`. |
| Curriculum scalars (reset_interval, p_fault, p_shuffle, burst) | via `"config"` | recomputed deterministically from config (full config only on final/best saves). |
| WandB run id / step | **NO** | logging continuity / artifact path only. |

**Not saved, full list:** pool (all fields), `train_key`, `last_reset_epoch`,
`BestModelTracker`, `EarlyStopping` mutable state, adaptive scheduler object,
wandb run id. Plus: epoch is saved but ignored, and the *periodic* checkpoint
clobbers the config with `{"epoch": ...}`.

---

## 3. Faithfulness gaps, ranked

### CRITICAL — resume diverges from a never-stopped run

1. **Circuit pool not saved (§1.4).** Restarting reinitializes the pool from
   scratch (`initialize_graph_pool`, `train_loop.py:980`): all circuits revert to
   fresh logits/hidden state, `update_steps=0`, `damage_count=0`, fresh `y_task`.
   The model has been *meta-trained against a matured, partially-damaged,
   heterogeneous pool*; feeding it a brand-new uniform pool is a large
   distribution shift in its own training input. The reset schedule, the
   maturity distribution, and (in task mode) the bound per-circuit tasks all
   reset. **Mechanism:** the inner-loop optimization target the GNN sees at
   epoch N+1 depends on what state the pool reached at epoch N; losing it
   restarts the meta-curriculum mid-flight. This alone makes resume materially
   different from continuous training.

2. **`train_key` not saved (§1.5).** Because `train_key` is a *carried, re-split*
   stateful chain (not `fold_in(root, epoch)`), re-deriving it from `cfg.seed`
   would reproduce epoch-0 randomness, not epoch-N randomness — unless we replay
   N splits. Without restoring it, the pool-sample indices, the per-step
   damage/shuffle draws, and reset selections all differ from the never-stopped
   run. **Mechanism:** see §4 — epoch N's randomness is *not* a pure function of
   `(seed, N)` under the current code, so the only faithful options are (a) save
   `train_key`, or (b) refactor the loop to `fold_in(root_key, epoch)`.

3. **`last_reset_epoch` not saved (§1.6).** Reset cadence
   (`should_reset_pool(epoch, interval, last_reset_epoch)`, `schedulers.py:293`)
   depends on it. Fresh start sets it to `-1`, so the *first* reset after resume
   fires at a different absolute epoch than it would have. Compounds gap #1.
   (Moot only if you also re-seed the pool — but then you have a worse problem.)

### MODERATE — converges but with statistical drift / wrong bookkeeping

4. **`BestModelTracker` not saved (§1.6).** On resume `best_metrics` start at
   ±inf, so the first periodic eval is recorded as a "new best" and overwrites
   `best_model_<metric>.pkl` even if the pre-preemption model scored higher.
   Risk: losing the genuinely-best checkpoint that the demo probe later reloads
   (`train.py:1082-1095`). Training itself converges, but the *selected* artifact
   may regress.

5. **`EarlyStopping` mutable state not saved (§1.6).** `count`/`first_epoch`
   restart at 0/None. A run with patience 6 that had accumulated 5 consecutive
   passing evals will need 6 more after resume → trains longer than it should
   (or, symmetrically, a marginal run gets extra epochs). Does not corrupt the
   model; affects stopping time.

6. **Adaptive/plateau scheduler object not saved (§1.3).** Only if
   `lr_scheduler ∈ {adaptive, reduce_on_plateau}`. Its loss EMAs / reduction
   count / cooldown reset, so the LR trajectory after resume differs. **Verify
   the cluster config**: `configs/config.yaml` defaults to cosine/constant, in
   which case this is inert (the static schedule resumes correctly via the
   optimizer `count`). Flag, don't block.

7. **Metrics history not fed back (§1.6).** `train.py` doesn't pass
   `initial_metrics`, so the resumed run's `losses[...]` lists start empty;
   `final_results.csv`'s `epochs_completed = len(losses)` (`train.py:180`) and
   the `final/` curves will reflect only post-resume epochs. Cosmetic for model
   quality, wrong for reporting.

### COSMETIC — logging continuity only

8. **WandB run id / step not saved (§1.7).** A fresh `wandb.init` starts a new run
   (or, with `resume="allow"` + same id, continues — but the id isn't persisted
   so it can't be matched). WandB step restarts at 0; `training/epoch` is logged
   as a field so curves are still plottable against epoch. Also affects the
   `run_<id>` checkpoint subdir path. No effect on training dynamics.

---

## 4. RNG determinism analysis

**Question:** is epoch N's randomness a pure function of `(root_seed, N)` (so
restoring just the epoch suffices), or does it depend on carried key state?

**Answer: it depends on carried, mutating key state.** It is **NOT**
`fold_in(root, N)`.

The loop carries `train_key` and re-derives it by `split` every iteration
(`train_loop.py:1451`):

```python
train_key, sample_key, loss_key = jax.random.split(train_key, 3)
```

`train_key` at epoch N is the result of N (plus reset-dependent extra) sequential
`jax.random.split` calls starting from the post-init key. On reset epochs the loop
consumes *additional* splits (`train_loop.py:1515`, `:1536`, `:1547`, `:1574`),
so the number of splits before epoch N is itself **path-dependent** (it depends
on how many resets fired, which depends on `reset_interval` and `epochs`). This
means you cannot reproduce epoch N's `train_key` from `(seed, N)` alone without
faithfully replaying the exact reset schedule.

Per-step randomness *below* `loss_key` is, by contrast, a **pure function of that
epoch's `loss_key`** — confirmed in `run_model_scan_with_loss`:

```python
# evaluation.py:627-628
data_key, damage_key, subsample_key = jax.random.split(scan_key, 3)
```
with per-step damage/shuffle keys derived by `split`/`fold_in` from those
(`evaluation.py:691` `prob_damage_keys = jax.random.split(prob_key, num_steps)`;
`:698` `sched_key = jax.random.fold_in(prob_key, 0xB5)`;
`:758` `wire_shuffle_keys = jax.random.split(wire_shuffle_key, ...)`). The
subsample key also folds per-step: "constant base key; per-step keys derived via
fold_in" (`evaluation.py:173`, `:663`). So once `loss_key` for epoch N is fixed,
the entire inner scan is deterministic.

**Consequences for faithful resume:**
- **Save `train_key`** as it stands at the *start* of the epoch you checkpoint at
  (i.e. before the `split` on `train_loop.py:1451` for that epoch), and restore
  it before entering the loop at `start_epoch`. That reproduces every subsequent
  draw bit-for-bit.
- `eval_key` needs no saving — it is a pure function of config and never mutates.
- **Alternative (cleaner long-term):** refactor the loop to derive each epoch's
  key as `epoch_key = jax.random.fold_in(root_train_key, epoch)` and split
  *that* for sample/loss/reset keys. Then epoch N's randomness becomes a pure
  function of `(seed, N)` and resume needs only the epoch number — no key
  blob. This is a behavior change (different random stream than today), so it
  cannot be applied to an in-flight run, only to new runs.

---

## 5. Recommended checkpoint schema for faithful resume

### 5.1 Proposed dict (superset of the current one)

```python
checkpoint = {
    # --- already present ---
    "model":      nnx.state(model),
    "optimizer":  nnx.state(optimizer),     # carries AdamW mu/nu + uint32 step/count
    "metrics":    {losses, hard_losses, accuracies, hard_accuracies, reset_steps},
    "config":     OmegaConf.to_container(cfg, resolve=True),   # FULL config, always
    "step":       epoch,                    # keep for back-compat

    # --- NEW: resume-critical ---
    "epoch":      epoch,                    # explicit; the loop should resume at epoch+1
    "pool":       circuit_pool,             # flax.struct.PyTreeNode -> picklable pytree
    "train_key":  train_key,               # the carried PRNG, snapshotted at epoch start
    "last_reset_epoch": last_reset_epoch,

    # --- NEW: bookkeeping faithfulness ---
    "best_model_tracker": {
        "best_metrics": best_model_tracker.best_metrics,
        "best_epochs":  best_model_tracker.best_epochs,
    },
    "early_stopping": {
        "count":       early_stopping.count,
        "first_epoch": early_stopping.first_epoch,
        "triggered":   early_stopping.triggered,
    } if early_stopping is not None else None,

    # --- NEW: logging continuity (cosmetic) ---
    "wandb_run_id": wandb_id,

    # --- NEW: only if adaptive LR is used ---
    "adaptive_scheduler_state": adaptive_scheduler.state_dict()  # needs a getter; or pickle the object
        if adaptive_scheduler is not None else None,
}
```

Notes:
- `circuit_pool` is a `flax.struct.PyTreeNode`; `pickle.dump` handles it because
  its leaves are jax arrays and `size` is a static field. Round-trips cleanly.
  (If cross-version safety is a concern, store `jax.tree.map(np.asarray, pool)`
  and rebuild via `GraphPool` on load.)
- Save the **full config** even in the periodic path — the current
  `save_periodic_checkpoint` passes `{"epoch": epoch}` as `cfg`
  (`checkpointing.py:552`), which loses everything needed to reconstruct
  `opt_fn`/schedule and the curriculum. Fix that call to pass the real cfg.
- `train_key` must be snapshotted **at the top of the epoch loop body**, before
  the `split` at `train_loop.py:1451`, so the restored key reproduces that
  epoch's draws.

### 5.2 Save cadence / size

Pool dominates the file size. Rough estimate for the default config
(`pool.size = 4096`, `circuit_hidden_dim`, multi-layer graph):
- Logits: `Σ_layers pool × (gate_n//gs) × gs × 2^arity × 4 bytes`. For a 12-bit
  task / typical `layer_sizes`, total gate logit elements per circuit is on the
  order of 10^3–10^4 → **tens of MB** across the pool.
- Node hidden state in `graphs.nodes`: `pool × n_node × circuit_hidden_dim × 4`.
  With `pool=4096`, `n_node` a few hundred, `hidden=16` → also **tens of MB**.
- Wires (int indices) and globals are comparatively small.

**Action:** before implementing, measure exactly:
```python
import jax
sz = sum(x.size * x.dtype.itemsize
         for x in jax.tree_util.tree_leaves(circuit_pool))
print(sz / 1e6, "MB")   # run inside a real training process at the target config
```
Expect roughly **50–200 MB** per checkpoint at `pool=4096`. That argues for:
- A dedicated **`resume_checkpoint.pkl`** written every `checkpoint.interval`
  (default **1031**) and **overwritten** (not versioned) — keep exactly one.
- Do **not** push the resume checkpoint as a wandb artifact every interval
  (bandwidth). The existing `best_model`/`latest_checkpoint` artifact logging
  (`checkpointing.py:559-568`, `:604-617`) stays for the small model-only saves.
- For preemption safety specifically, also write on `KeyboardInterrupt` /
  SIGTERM (the loop already catches `KeyboardInterrupt` at `train_loop.py:1797`).
  Slurm preemption sends SIGTERM — add a signal handler that flips a flag and
  writes a final resume checkpoint before exit.

### 5.3 Restore sequence in `train_model`

Add an `init_pool` / `resume_state` parameter (or a `resume_checkpoint_path`).
Sequence, before the `for epoch in pbar` loop:

1. **Instantiate model and optimizer normally** (need the live `opt_fn` with the
   identical schedule — reconstruct from the *saved config*'s `lr_scheduler`,
   `learning_rate`, `epochs`, `lr_scheduler_params`). Do not skip building
   `opt_fn`; only the *state arrays* come from the checkpoint.
2. `nnx.update(model, ckpt["model"])` and `nnx.update(optimizer, ckpt["optimizer"])`.
   This restores params + AdamW moments + the `step`/`count` that drives the LR
   schedule. **Gotcha:** `nnx.Optimizer` must be reconstructed *first* (so it
   references the restored `model` and the rebuilt `opt_fn`), then state-updated.
   Update `model` and `optimizer` state in a consistent order so the optimizer's
   internal model reference matches.
3. Replace the freshly-built `circuit_pool` with `ckpt["pool"]` (skip
   `initialize_graph_pool` when resuming).
4. `train_key = ckpt["train_key"]`; `last_reset_epoch = ckpt["last_reset_epoch"]`.
5. Restore `best_model_tracker.best_metrics/best_epochs` and `early_stopping`
   `count/first_epoch/triggered` from the checkpoint.
6. Re-seed metrics lists from `ckpt["metrics"]` (pass through `initial_metrics`,
   which `train_model` already supports at `train_loop.py:918-924`).
7. Set `start_epoch = ckpt["epoch"] + 1` and iterate `for epoch in range(start_epoch, epochs)`
   (replace `tqdm(range(epochs))` at `train_loop.py:1436`). The beta-loss-step
   `epoch/(epochs-1)` term and reset gating then line up automatically.

### 5.4 nnx-specific gotchas

- **`nnx.state` / `nnx.update` round-trip:** `nnx.update(obj, saved_state)`
  requires `obj` to be a structurally-identical freshly-constructed module/optimizer.
  The whole `load_checkpoint_with_compatibility` machinery
  (`checkpointing.py:21-277`) exists to paper over Flax version drift in pickled
  `VariableState` objects — reuse it for the model/optimizer state on resume too,
  not just for the demo-probe model load.
- **Optimizer reconstruction:** `nnx.Optimizer(model, opt_fn, wrt=nnx.Param)`
  bakes the schedule into `opt_fn`. You cannot pickle the closure; you must
  rebuild `opt_fn` from config. If the resumed run uses a *different* `epochs`
  value than the original (e.g. extending a run), the cosine/linear_warmup
  schedule will be re-parameterized and the LR trajectory will differ — keep
  `epochs` identical for faithful resume.
- **`step` is uint32:** wraps at 2^32; irrelevant at these epoch counts but note
  it is the LR-schedule clock for static schedules.
- **Static vs traced:** the pool's `size` field is `pytree_node=False` (static);
  it survives pickle as a Python int. The JIT-compiled `pool_train_step` and
  `pool.update`/`pool.sample` (`@jax.jit`, `pool.py:154`, `:249`) recompile on
  the resumed process — fine, just a warmup cost. Multi-GPU sharding
  (`train_loop.py:1024-1039`) re-replicates from the restored (single-device)
  arrays; restore *before* `sharding_ctx.replicate`.

---

## 6. Concrete file:line touchpoint list

### Save side
- `boolean_nca_cc/training/checkpointing.py:392` — `save_checkpoint(...)`: the
  one writer. Extend the dict here (`:410-418`).
- `boolean_nca_cc/training/checkpointing.py:532` — `save_periodic_checkpoint`:
  the per-interval writer (`latest_checkpoint.pkl`). **Bug to fix:** passes
  `{"epoch": epoch}` as cfg (`:552`) → no full config saved. This is where a new
  resume-checkpoint write belongs (needs pool/train_key/trackers in scope, so
  signature must be widened or a new function added).
- `boolean_nca_cc/training/checkpointing.py:574` — `save_best_checkpoint` (also
  passes `{"epoch": ...}` cfg, `:597`); model-only, leave as-is.
- `boolean_nca_cc/training/checkpointing.py:622` — `save_stable_state` (NaN
  rescue; same pattern).
- `train.py:993-1010` — final-model save (`save_checkpoint(... "final_model.pkl")`).
- `boolean_nca_cc/training/train_loop.py:1773-1789` — the periodic-checkpoint
  call site inside the loop; `model`, `optimizer`, `circuit_pool`, `train_key`,
  `last_reset_epoch`, `best_model_tracker`, `early_stopping` are all in scope here.
- `boolean_nca_cc/training/train_loop.py:1797-1816` — `KeyboardInterrupt` handler
  + result dict (already returns `pool`, `optimizer`, `best_model_tracker`).
  Hook a final resume-save here; add a SIGTERM handler for Slurm preemption.

### Load side
- `boolean_nca_cc/training/checkpointing.py:262` / `:446` —
  `load_checkpoint_with_compatibility` / `load_checkpoint`: the loaders. Reuse
  for resume (handles Flax pickled-state compat).
- `boolean_nca_cc/training/checkpointing.py:1273` —
  `load_model_from_config_and_checkpoint`: shows the canonical
  instantiate → `migrate_checkpoint_state` → `nnx.update(model, ...)` flow
  (`:1306-1317`). Mirror this for the optimizer state.
- `train.py:1093-1094` — current only consumer of a loaded checkpoint for
  continuation-of-sorts (`load_checkpoint_with_compatibility` + `nnx.update`),
  but for the demo probe, not resume.

### Init / state-creation touchpoints (what to override on resume)
- `train.py:489-494` — root keys (`train_key`, `eval_key`).
- `train.py:651` / `:674` — model init (`init_key`, Hydra instantiate).
- `boolean_nca_cc/training/train_loop.py:930` — `model = init_model`.
- `boolean_nca_cc/training/train_loop.py:935-969` — optimizer + schedule build
  (`opt_fn`, `nnx.Optimizer(...)`); the schedule/`opt_fn` must be rebuilt on resume.
- `boolean_nca_cc/training/train_loop.py:937-939` — `get_learning_rate_schedule`.
- `boolean_nca_cc/training/train_loop.py:973-995` — pool init
  (`initialize_graph_pool`); **skip / replace with restored pool on resume**.
- `boolean_nca_cc/training/train_loop.py:898-901` — task-sampler init (`y_task`);
  restored implicitly via the saved pool.
- `boolean_nca_cc/training/train_loop.py:911-924` — metrics lists +
  `initial_metrics` re-seed path (already exists; just feed it).
- `boolean_nca_cc/training/train_loop.py:1433` — `BestModelTracker()` (restore fields).
- `boolean_nca_cc/training/train_loop.py:1440` — `last_reset_epoch = -1` (restore).
- `boolean_nca_cc/training/train_loop.py:1436` / `:1448` —
  `tqdm(range(epochs))` / `for epoch in pbar`: change to start at `start_epoch`.
- `train.py:964-978` — `EarlyStopping(...)` construction (restore mutable fields).

### RNG touchpoints
- `boolean_nca_cc/training/train_loop.py:1451` — per-epoch `split(train_key, 3)`.
- `boolean_nca_cc/training/train_loop.py:1515`, `:1536`, `:1547`, `:1574` —
  reset-epoch extra splits (the source of path-dependent key advancement, §4).
- `boolean_nca_cc/training/evaluation.py:627-628`, `:663`, `:691`, `:698`,
  `:758`, `:793` — in-scan derivation of all per-step damage/shuffle/data keys
  from `scan_key` (pure function of epoch `loss_key`).

### Config
- `configs/config.yaml:13-14` — `seed`, `eval_seed`.
- `configs/config.yaml:325-331` — `checkpoint` block: `enabled`, `interval`
  (1031), `save_best`, `save_latest`, `save_on_mnt_storage`. Add a
  `resume`/`resume_path` knob here.
- `configs/config.yaml:154-172` — `pool` block (`size: 4096`, etc.) — determines
  resume-checkpoint file size.

---

## Open items to verify before implementing
- [ ] **Measure pool pickle size** at the real target config (snippet in §5.2).
- [ ] **Confirm the cluster `lr_scheduler` value** (cosine/constant → adaptive
      scheduler state is moot; otherwise it must be saved). Check the launched
      sweep configs, not just `config.yaml` defaults.
- [ ] Confirm `nnx.update(optimizer, saved_opt_state)` cleanly restores the
      `step`/`count` *and* the AdamW moments on the installed Flax version (the
      compat unpickler in `checkpointing.py` was written for model state; test it
      on optimizer state too).
- [ ] Decide save-on-preemption mechanism: SIGTERM handler vs Slurm
      `--signal=B:TERM@<sec>` requeue. The loop already handles `KeyboardInterrupt`
      (`train_loop.py:1797`) but Slurm sends SIGTERM, not SIGINT.
