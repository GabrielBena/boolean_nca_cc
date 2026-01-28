# Low-footprint loss computation for BPTT (JAX)

This note describes a memory-efficient way to compute the training (“meta”) loss in the unrolled message-passing loop used in `boolean_nca_cc/training/train_loop.py`.

## Implementation approach

**All codebase edits should be incremental and stepwise.** Each step should be implemented, tested, and verified before moving to the next. This ensures:
- Each change is isolated and debuggable
- The codebase remains functional at each stage
- Memory improvements can be measured incrementally

## Implementation status

- ✅ **Step 0**: Remove loss from graph state — **COMPLETE** (using `blind_mode=True` by default)
- ✅ **Step 1**: Sample `loss_step` once per meta-batch — **COMPLETE**
- ✅ **Step 2**: Stop stacking large per-step objects — **COMPLETE**
- ✅ **Step 3**: Window loss with static horizon size — **COMPLETE** (`long_horizon_enabled`, `long_horizon_size`)
- ✅ **Step 4**: Replace unrolled Python loop with `lax.scan` — **COMPLETE** (`loss_fn_scan` in `train_loop.py`, gated by `training.use_scan`; same contract as `loss_fn_no_scan`, low-footprint carry + loss-only output).
- ⏸️ **Step 5**: Beta distribution for loss-step selection — see Step 5 below (optional `get_step_beta` when `random_loss_step`; curriculum from early to late steps).
- ⏸️ **Step 6**: Add gradient checkpointing (remat)

## Low-footprint checklist (the non-negotiables)

- **Do not write loss into graph state**: treat `graph` as model state only; do not store loss in `graph.globals` (or similar) during training.
- **Never time-stack large objects**: do not build/stack `[T, ...]` tensors of `graph`, `logits`, or large `aux` PyTrees.
- **Only emit scalars over time**: per-step outputs should be small (typically just `loss_t`).
- **Snapshot selected step with `lax.cond`**: keep `(graph, logits, aux)` only for the selected `loss_step`.
- **Prefer `lax.scan` over Python loops**: avoids unrolled HLO and makes checkpointing straightforward.
- **Use remat/checkpointing when T is large**: reduces activation memory for BPTT at extra compute cost.
- **Window loss must be static**: horizon size `H` should be static; use fixed-length gather + masking (not variable-size slicing).

## What Step 2 does and does *not* affect

**Model parameters are always in memory.** The `CircuitSelfAttention` model (and any GNN) has a fixed set of parameters (attention layers, MLPs, projections, etc.). Those weights live in memory for the whole training run. **Step 2 does not change that.** It does not remove or shrink the model.

**Step 2 only reduces activation/state memory** inside the message-passing loop:

- **Before Step 2**: For each of the T message steps we stored a full copy of `(graph, logits, aux)`. After the loop we stacked them → tensors of shape `[T, batch, n_nodes, ...]`. So we had T copies of the graph state and logits in memory at once.
- **After Step 2**: We still run all T steps (so the computational graph for BPTT is unchanged), but we only *keep* one snapshot at `loss_step`. We accumulate scalar losses only. So we have 1 copy of graph/logits instead of T.

So in short:
- **Parameters** (the model’s weights): always in memory, unchanged by Step 2.
- **Activations / state** (graph, logits, aux at each step): we used to stack T copies; Step 2 keeps a single snapshot → big savings when T is large.

## Problem: why the rolled-back loop is high-memory

The rolled-back `loss_fn_no_scan` (unrolled Python `for`) collects per-step outputs as tuples and then stacks them:

- `all_results.append((loss, aux, graph, logits))`
- `stacked_results = tree.map(stack, *all_results)`
- then indexes `stacked_results[loss_step]`

This **materializes time-major tensors for large objects** (`graph`, `logits`, often parts of `aux`) of shape roughly `[T, ...]`. That’s avoidable because training only needs:

- a scalar loss (single step or a window), and
- optionally a *single* snapshot of `(graph, logits, aux)` at the chosen step for pool update / logging.

## Goal

Build a unified loss computation that:

- supports **fixed** or **random** `loss_step`
- supports **single-step** loss or **fixed-size horizon/window** loss
- avoids time-stacking large objects
- is compatible with `jit` + batching
- optionally enables **gradient checkpointing** (remat) for further memory reduction

## Strategy (stepwise, implementation-ready)

### Step 0 — Remove loss from graph state (prerequisite) ✅ COMPLETE

**Status**: Implemented using `blind_mode=True` by default.

**Implementation**: The loop conditionally excludes loss from `graph.globals` based on `blind_mode`:
- When `blind_mode=True` (default): stores `[0.0, step_count]` — loss is not part of state
- When `blind_mode=False`: stores `[loss, step_count]` — legacy loss-feedback mode

This decouples the recurrence from loss computation and enables the footprint optimizations below.

### Step 1 — Sample `loss_step` once per meta-batch (recommended) ✅ COMPLETE

**Status**: Implemented with configurable random/fixed mode.

**Implementation**:
- Added `random_loss_step: bool` and `random_loss_step_min: int` parameters to `train_model` and `pool_train_step`
- Added both to `static_argnames` in `pool_train_step` (enables Python `if` checks in JIT)
- Modified `get_loss_step()` to support:
  - Random mode: `randint(min_step, T)` when `random_loss_step=True`
  - Fixed mode: `T - 1` when `random_loss_step=False`
- `loss_step` is sampled **once per meta-batch** (before `vmap`) and captured as closure variable in `loss_fn_no_scan`
- Configurable via `cfg.training.random_loss_step` and `cfg.training.random_loss_step_min`

**Why per-batch**: Per-example random steps (inside a `vmap`) reduce compiler optimization and complicate control flow. Per-batch randomness gives most of the training benefit with a simpler and faster graph.

### Step 2 — Stop stacking large per-step objects (biggest immediate win) ✅ COMPLETE

**Status**: Implemented in `loss_fn_no_scan` within `pool_train_step`.

**Implementation**: During the loop:

- **always** collect only `loss_t` (cheap) into a length-`T` array
- **never** collect `(graph, logits, aux)` for all steps
- instead, **conditionally snapshot** `(graph, logits, aux)` only when `t == loss_step`

**Critical JAX requirement**: Use `jax.lax.cond` (not Python `if`) for the conditional snapshot, because the loop runs inside `jit`/`vmap` where traced values cannot be used in Python conditionals:

```python
# ❌ Won't work in traced context:
if i == loss_step:
    selected = (aux, graph, logits)

# ✅ Correct:
selected = jax.lax.cond(
    i == loss_step,
    lambda _: (aux, graph, logits),
    lambda _: selected,
    None
)
```

**Actual implementation pattern**:

```python
losses = []  # Accumulate T scalar losses
selected_snapshot = None  # Will hold (graph, logits, aux) at loss_step

for i in range(n_message_steps):
    graph = model(...)
    graph, loss, logits, aux = get_loss_and_update_graph(...)
    losses.append(loss)  # Cheap: only scalar
    
    # Initialize on first iteration, then conditionally update at loss_step
    if i == 0:
        selected_snapshot = (graph, logits, aux)  # Python if OK for loop index
    else:
        selected_snapshot = jax.lax.cond(
            i == loss_step,
            lambda _: (graph, logits, aux),  # True: save at loss_step
            lambda _: selected_snapshot,      # False: keep previous
            None
        )

final_loss = losses[loss_step]
final_graph, final_logits, final_aux = selected_snapshot
```

**Key implementation details**:

1. **BPTT preservation**: All T steps still execute (preserves computational graph). Both branches of `lax.cond` are traced, ensuring gradient flow is preserved from `loss_step` back to step 0.

2. **Initialization**: Snapshot is initialized on first iteration (`i == 0`) using Python `if`, which is safe since the loop index is a Python int, not a traced value. This avoids needing an extra forward pass before the loop.

3. **Edge cases**: Handles `loss_step == 0` correctly (snapshot initialized on step 0, which is the desired step).

4. **Memory savings**: 
   - **Before**: `[T, batch, n_nodes, logit_dim]` for logits, `[T, batch, n_nodes, hidden_dim]` for graph nodes
   - **After**: `[batch, n_nodes, logit_dim]` and `[batch, n_nodes, hidden_dim]` (single snapshot)
   - **Reduction**: Approximately `(T-1)` times less memory for large objects
   - **Example**: For T=100, batch=64, n_nodes=1000, this saves ~99× memory for graphs/logits

5. **What's still stored**: 
   - `losses`: `[T]` array of scalars (cheap, ~400 bytes for T=100)
   - Single snapshot: `(graph, logits, aux)` at `loss_step` (needed for pool updates and final outputs)

**Removed code**:
- `all_results.append((loss, aux, graph, logits))` → replaced with `losses.append(loss)` + conditional snapshot
- `stacked_results = jax.tree.map(lambda *args: jp.stack(args), *all_results)` → removed entirely
- `final_loss, final_aux, final_graph, final_logits = jax.tree.map(lambda x: x[loss_step], stacked_results)` → replaced with direct extraction

**Future optimization**: If you must still execute all `T_max` steps (static shapes), you can additionally **stop gradient flow after `loss_step`** using `jax.lax.stop_gradient` (memory win for the backward pass), but note it does **not** reduce forward compute.

### Step 3 — Window loss with static horizon size ✅ COMPLETE

**Status**: Implemented. Horizon semantics depend on **random vs fixed** `loss_step` (driven by `random_loss_step`).

**Behaviour**:

- **Single-step** (no horizon or `long_horizon_size=1`): `L = losses[loss_step]` — loss at the chosen step only (random or fixed).
- **Horizon + random** (`random_loss_step=True`, `long_horizon_enabled=True`): window = **from** `loss_step` **to final step** — mean over `[loss_step, ..., T-1]`. Variable-length window; implemented with static-length gather (length `T`) and mask.
- **Horizon + fixed** (`random_loss_step=False`, `long_horizon_enabled=True`): window = **last H steps ending at final step** — mean over `[(T-1)-(H-1), ..., T-1]`. Fixed length `H`; implemented with static gather + mask.

**Static-gather patterns** (JAX requires static shapes; use fixed-length gather + masking):

- **Random + horizon**: `indices = loss_step + arange(T)`; `valid = (indices >= 0) & (indices < T)`; clamp indices to `[0, T-1]`; `L = sum(take(losses, idx_clipped) * valid) / max(sum(valid), 1)`.
- **Fixed + horizon**: `start = (T-1) - (H-1)`; `indices = start + arange(H)`; mask and mean as above.

**Config**: `training.long_horizon_enabled`, `training.long_horizon_size`; `random_loss_step` is passed into `pool_train_step` (static) so the JIT branches on the correct window definition.

### Step 4 — Replace unrolled Python loop with `lax.scan` ✅ COMPLETE

**Status**: Implemented in `train_loop.py`. `loss_fn_scan` lives in the same scope as `loss_fn_no_scan` inside `pool_train_step`. Selection is gated by `use_scan` (static arg): `batch_loss_fn` uses `loss_fn_scan` when `use_scan=True`, else `loss_fn_no_scan`. Config: `training.use_scan` (default `false`); overridable via `configs/config.yaml` or CLI. Wired from `train.py` as `use_scan=cfg.training.use_scan` into `train_model`, then into `pool_train_step`.

**Implementation summary**:
- **First step outside scan**: Step 0 is run once before the scan to obtain initial `(graph_0, loss_0, logits_0, aux_0)` and avoid a placeholder for `selected_snapshot` in the carry.
- **Scan**: `lax.scan(scan_body, init_carry, jp.arange(1, n_message_steps), length=n_message_steps-1)`. Carry = `(graph, selected_snapshot)`; output = scalar loss per step only. Body: one model step, `get_loss_and_update_graph`, globals update (blind_mode), `selected_snapshot = lax.cond(step_idx == loss_step, current, previous)`.
- **After scan**: `losses_arr = jp.concatenate([loss_0[None], scan_losses])`; `final_loss` and snapshot extraction use the same logic as `loss_fn_no_scan` (single-step or horizon window, same static gather + mask).
- **Edge case**: When `n_message_steps == 1`, scan length is 0; only the pre-scan step runs; behaviour matches no-scan.

**Why scan matters at large T**

With the current implementation, `n_message_steps` is a **static** argument, so JAX **unrolls** the Python loop at compile time: the compiled program contains `T` copies of the loop body. Consequences at large T (e.g. 30–100 steps):

- **Compile time** grows roughly with T (more operations to compile).
- **Compiled code size** grows with T (100× more HLO for T=100), which hurts compiler optimizations and runtime.
- **Runtime** can degrade (code bloat, worse cache behavior), so the slowdown when bumping `n_steps` to 100 is severe.

`lax.scan` compiles to a **single** loop body executed `T` times:

- **Compile time** is effectively independent of T (one body to compile).
- **Compiled program** stays compact (one loop, not T copies).
- **Runtime** at large T is typically better (same FLOPs, but cleaner code generation).

So scan does **not** reduce the amount of work (we still run T steps for BPTT and horizon), but it makes **large T feasible** and avoids the immense slowdown from unrolling. It also allows **hoisting** work that is identical every step (e.g. attention mask creation in the self-attention model): compute once before the scan and pass into the step body, instead of recomputing T times inside the loop.

**Scan and the beta-horizon strategy**

The planned **beta** strategy uses a **static** `T_max` (e.g. 100) and a **dynamic** effective step: `loss_step = f(epoch)` (e.g. via `get_step_beta`), so we always run `T_max` steps but only the step used for loss and pool update changes over training. Horizon and single-step loss still use the same contract (window ending at `loss_step`, snapshot at `loss_step`). Implementing this with an **unrolled** loop would mean compiling and running a 100-copy program every time; with **scan**, we compile once for `T_max` and run one loop body `T_max` times, so beta curriculum (early step → late step) is practical. The scan-based loss function keeps the same interface (`loss_step`, horizon, snapshot at `loss_step`), so beta and horizon logic remain entirely in the caller—no duplicate logic between scan and no-scan paths.

**Design principle (unchanged)**

- **scan carry** contains the evolving graph state and the “selected snapshot” containers (low-footprint: do not emit full graph/logits/aux per step).
- **scan output** should be **small** (e.g. scalar loss per step only); after the scan, compute final loss from the `[T]` losses (single-step or horizon) and take the snapshot from the carry.

**Relation to existing scan helpers**

The existing `run_self_attention_scan_with_loss` in `boolean_nca_cc/models/self_attention.py` **does** scan along MP steps (model + `get_loss_and_update_graph` per step), but it **emits** `(graph, loss, logits, aux)` every step, so the scan output is `[T, ...]` for graphs and logits — **not** low-footprint. Do **not** use it as-is for training. Implement `loss_fn_scan` with the low-footprint contract below (carry holds snapshot; output is loss-only). Optionally, add a low-footprint variant in the model module and call it from the train loop; the logic is the same.

**Implementation strategy for `loss_fn_scan`**

1. **Contract (same as `loss_fn_no_scan`)**  
   Inputs: `model`, graph, logits, wires, `loss_step`, `n_message_steps`, horizon params, etc.  
   Outputs: `(final_loss, (aux, graph, logits, loss_step))` — scalar loss (single-step or horizon mean) and one snapshot at `loss_step` for pool update.

2. **Scan shape (low-footprint)**  
   - **Carry**: `(graph, selected_snapshot)` where `selected_snapshot = (graph, logits, aux)` at the chosen step. Updated only when `step_idx == loss_step` via `lax.cond`.  
   - **Output**: scalar loss per step only → after scan, stacked as `losses` of shape `[T]`.  
   - **Do not** emit graph, logits, or aux from the scan body; that would stack `[T, ...]` and defeat low-footprint.

3. **Scan body (one step)**  
   - Input: `(carry, _)` with `carry = (graph, selected_snapshot)`; step index is implicit (scan iteration).  
   - Compute: `graph = model(graph, ...)`; then `graph, loss, logits, aux = get_loss_and_update_graph(graph, ...)`; set graph globals for blind_mode as in no_scan.  
   - Snapshot: `selected_snapshot = lax.cond(step_idx == loss_step, lambda _: (graph, logits, aux), lambda _: selected_snapshot, None)`. Use the scan's iteration index (e.g. pass `jax.lax.iota` or an arange as `xs` so the body receives `step_idx`).  
   - Return: `(new_carry, output)` with `new_carry = (graph, selected_snapshot)` and `output = loss` (scalar).

4. **After the scan**  
   - `losses` = stacked scan outputs, shape `[T]`.  
   - Compute `final_loss` from `losses` and `loss_step` exactly as in `loss_fn_no_scan` (single-step: `losses[loss_step]`; horizon: same static gather + mask as Step 3).  
   - Snapshot = `selected_snapshot` from the final carry.  
   - Return `(final_loss, (aux, graph, logits, loss_step))` with `(graph, logits, aux) = selected_snapshot`.

5. **Where implemented**  
   `loss_fn_scan` is implemented in `train_loop.py` (same scope as `loss_fn_no_scan`). It is invoked from `pool_train_step` when `use_scan=True`. Horizon and loss_step logic are identical to no_scan — no duplicate logic.

6. **Static arguments**  
   `n_message_steps` (or `T_max`) and `loss_step` remain static for JIT; the scan `length` is `n_message_steps`. Beta strategy only changes `loss_step` per epoch in the caller; the scan body does not change.

### Step 5 — Beta distribution for loss-step selection

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

### Step 6 — Add gradient checkpointing (remat) for long unrolls

Once the computation is in a `scan`, wrap the step function with `jax.remat` / `jax.checkpoint` (or `nnx.remat`) to reduce BPTT activation memory.

Trade-off:

- lower memory (often substantially)
- higher compute (recomputes some forward work during backward)

## Recommended unified function contract

Implement a runner that returns:

- `final_loss`: scalar used for gradients (single-step or horizon mean)
- `selected_aux, selected_graph, selected_logits`: snapshot at `loss_step` for pool update / logging
- optionally `losses[T]` only for debugging (avoid returning in training unless needed)

## Notes / assumptions for this design

- We explicitly **forego “loss as part of state”**: do not write loss into `graph.globals` for training dynamics unless you intentionally want loss-feedback.
- Memory savings come primarily from:
  - not time-stacking large objects, and
  - (optionally) remat for activations needed for BPTT up to `loss_step`.

## Why “writing loss into graph state” is a key constraint

Sometimes the loop writes the current loss into the recurrent state (e.g. `graph.globals = [loss, step_count]`). This matters because it changes the unrolled system from:

- **state-only recurrence**: \(s_{t+1} = f_\theta(s_t)\)

to:

- **loss-feedback recurrence**: \(s_{t+1} = f_\theta(s_t, \ell_t)\), where \(\ell_t = g(s_t)\)

Consequences:

- **You can’t skip loss computation on “non-selected” steps** without changing the forward dynamics, because \(\ell_t\) is now an input to later steps.
- Even if you only want the loss at `loss_step` (or a window), the state at `loss_step` depends on all earlier per-step losses, so you must compute them (and their intermediates).
- This blocks the simplest footprint strategy of “compute loss only at the selected step/window” and pushes you toward either (a) computing loss each step, or (b) redesigning the state so it does not depend on loss.

