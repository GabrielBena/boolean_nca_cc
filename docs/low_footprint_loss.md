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
- ⏸️ **Step 4**: Replace unrolled Python loop with `lax.scan`
- ⏸️ **Step 5**: Add gradient checkpointing (remat)

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

### Step 3 — Window loss with static horizon size (no dynamic slice sizes)

JAX requires static shapes for most slicing/gather patterns. For “loss over a selected window”, keep the horizon size `H` **static**.

Compute:

- single-step loss: `L = losses[loss_step]`  (equivalent to `H=1`)
- window loss (length `H`): mean over `[loss_step - (H-1), ..., loss_step]`, implemented with **static-length gather** and a mask.

Static-gather pattern:

1. `idx = loss_step - (H - 1) + arange(H)`  (length `H`)
2. mask out indices outside `[0, T-1]` (or clamp + mask)
3. `window = take(losses, clip(idx, 0, T-1))`
4. `L = sum(window * mask) / sum(mask)`

This avoids `dynamic_slice` with a traced `slice_sizes`.

### Step 4 — Replace unrolled Python loop with `lax.scan`

Switch the time loop to `jax.lax.scan` to:

- keep the compiled program compact (avoid unrolling `T` copies into HLO)
- reduce compile time and improve runtime stability
- create a single step function suitable for rematerialization

Design principle:

- **scan carry** contains the evolving graph state and the “selected snapshot” containers
- **scan output** should be **small** (e.g. scalar loss per step)

### Step 5 — Add gradient checkpointing (remat) for long unrolls

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

