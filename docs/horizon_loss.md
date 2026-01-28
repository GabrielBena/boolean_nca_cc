# Horizon Loss Implementation Strategy

## Current State

**Loss Computation Flow:**

1. Sample circuits from pool → `graphs`, `wires`, `logits`
2. Run `n_message_steps` message passing steps (always runs all steps)
3. Compute loss at each step, store in `all_results`
4. Select loss from step `loss_step = n_message_steps - 1` (always final step)
5. Use selected loss for gradient computation
6. Update pool with graph/logits from selected step

**Key Constraint:** JAX requires static shapes. We must run all `T_max` steps but can mask/slice to compute loss from subset.

## Implementation Order

### Step 1: Random Loss Step (JAX-friendly)

**Goal:** Select random step `t ∈ [min_step, T_max)` for loss computation.

**Implementation:**

- Sample `loss_step` before forward pass (static during computation)
- Run all `T_max` steps (maintains static shape)
- Use `jax.lax.dynamic_slice` to extract loss from `loss_step` (works with traced values)
- Pool update uses graph/logits from `loss_step` (not final step)

**Code location:** `pool_train_step` → `get_loss_step()` (nested function)

**Changes:**

```python
def get_loss_step(loss_key, n_message_steps, random_loss_step_min):
    if random_loss_step:
        return jax.random.randint(
            loss_key, (1,), random_loss_step_min, n_message_steps
        )[0]
    else:
        return n_message_steps - 1

# Extract single step using dynamic_slice (JAX-friendly with traced values)
def extract_step_output(step_outputs, loss_step):
    return jax.tree.map(
        lambda arr: jax.lax.dynamic_slice(arr, (loss_step,), (1,))[0],
        step_outputs
    )
```

**JAX compatibility:** ✅ `loss_step` is sampled before traced computation, `dynamic_slice` works with traced values.

---

### Step 2: Loss Window (Horizon Loss)

**Goal:** Aggregate loss over N steps before `loss_step` to encourage consistent low loss over horizon.

**Window Definition:**

- Start: `max(0, loss_step - horizon_size + 1)`
- End: `loss_step + 1` (inclusive)
- Window size: `min(horizon_size, loss_step + 1)`

**Aggregation:** Mean of losses in window.

**Implementation:**

```python
def compute_horizon_loss(all_losses, loss_step, horizon_size):
    horizon_start = jp.maximum(0, loss_step - horizon_size + 1)
    window_size = loss_step - horizon_start + 1
  
    # Extract window using dynamic_slice (same mechanism as single-step extraction)
    window_losses = jax.lax.dynamic_slice(
        all_losses,
        start_indices=(horizon_start,),
        slice_sizes=(window_size,)
    )
  
    return jp.mean(window_losses)
```

**Unified extraction:** Both single-step and window extraction use `jax.lax.dynamic_slice`:

- Single step: `slice_sizes=(1,)` then index `[0]`
- Window: `slice_sizes=(window_size,)` then compute mean

**Integration:**

- Extract `all_losses` from `stacked_results[0]` (first element is loss, shape: `[n_message_steps]`)
- Compute `horizon_loss` from window using `dynamic_slice`
- Use `horizon_loss` for gradient computation
- Still extract graph/logits/aux from `loss_step` for pool update (from `stacked_results[1:]`)

**JAX compatibility:** ✅ Uses `dynamic_slice` for traced values, maintains static shapes.

---

### Step 3: Beta Distribution Step Selection

**Goal:** Progressively shift from early to late steps during training using Beta distribution.

**Beta Distribution:**

- Early training: `alpha > beta` → left-skewed (favors early steps)
- Late training: `alpha < beta` → right-skewed (favors late steps)
- Parameters evolve with `training_progress = epoch / (epochs - 1)`

**Implementation:**

```python
def get_loss_step(loss_key, n_message_steps, epoch, epochs, 
                  random_loss_step, use_beta_loss_step, random_loss_step_min):
    if random_loss_step:
        if use_beta_loss_step:
            from boolean_nca_cc.training.schedulers import get_step_beta
            return get_step_beta(
                loss_key,
                n_message_steps,
                training_progress=epoch / max(epochs - 1, 1),
            )
        else:
            return jax.random.randint(...)
    else:
        return n_message_steps - 1
```

**Combined with Horizon:**

- Beta selects `loss_step` (target step)
- Horizon aggregates losses from `[loss_step - horizon_size + 1, loss_step]`
- Horizon size can be progressive: `horizon_size = f(epoch)` (optional)

**JAX compatibility:** ✅ Beta sampling happens before traced computation.

---

## Integration Points

**Function:** `pool_train_step()` in `train_loop.py` (nested inside `train_model()`)

**Current flow:**

1. `loss_fn_no_scan` runs all `n_message_steps` in a loop
2. Each step: apply model → compute loss → store in `all_results`
3. Stack all results: `stacked_results = jax.tree.map(lambda *args: jp.stack(args), *all_results)`
4. `get_loss_step()` selects step (currently always `n_message_steps - 1`)
5. Index into `stacked_results` at `loss_step`: `final_loss, final_aux, final_graph, final_logits = jax.tree.map(lambda x: x[loss_step], stacked_results)`
6. `batch_loss_fn` vmaps over batch, computes gradients
7. Update optimizer and pool

**Modified flow:**

1. Run all steps (unchanged)
2. `get_loss_step()` selects step (Step 1: random, Step 3: beta)
3. Extract `all_losses` from `stacked_results[0]` (loss is first element)
4. If horizon enabled: compute `horizon_loss` from window (Step 2)
5. Else: extract `final_loss` from `loss_step` (Step 1)
6. Extract graph/logits/aux from `loss_step` for pool update (unchanged)
7. Return aggregated loss for gradients

**Code structure:**

```python
def pool_train_step(...):
    def get_loss_step(loss_key):
        # Step 1: Random, Step 3: Beta
  
    def loss_fn_no_scan(model, graph, logits, wires, loss_key, knockout_pattern):
        # Run all steps, store in all_results
        # Stack: stacked_results
        # Extract all_losses = stacked_results[0]  # shape: [n_message_steps]
      
        if long_horizon_enabled:
            final_loss = compute_horizon_loss(all_losses, loss_step, long_horizon_size)
        else:
            final_loss = all_losses[loss_step]
      
        # Extract graph/logits from loss_step (unchanged)
        final_aux, final_graph, final_logits = jax.tree.map(
            lambda x: x[loss_step], stacked_results[1:]  # Skip loss, get aux/graph/logits
        )
```

**Key Variables:**

- `random_loss_step: bool` - Enable random step selection
- `random_loss_step_min: int` - Minimum step for random selection
- `use_beta_loss_step: bool` - Use beta distribution (requires `random_loss_step=True`)
- `long_horizon_enabled: bool` - Enable horizon aggregation
- `long_horizon_size: int` - Number of steps in horizon window

**Pool Update:** Always uses graph/logits from `loss_step` (not from horizon computation), ensuring pool state consistency.
