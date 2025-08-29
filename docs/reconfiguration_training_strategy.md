### Reconfiguration Training Mode Strategy

#### Purpose
Introduce a second training mode alongside existing growth (nops → configured) that trains the model to reconfigure an already working circuit after structural damage. This mirrors the "optimize fixed wiring → perturb → reoptimize" workflow demonstrated in `@distances_reconfigured` while integrating with the pool-based training pipeline.

---

### Modes

- **growth** (current):
  - Pool initializes with randomly wired NOPs circuits (single wiring in 'fixed' wiring mode).
  - GNN/Self-Attention learns to configure fresh circuits from scratch.
  - Resets reintroduce fresh, unconfigured circuits.

- **reconfig** (new):
  - Perform one-time backprop optimization on a fixed wiring to obtain a working configuration `(base_wires, base_logits)`.
  - Initialize the entire pool by cloning `(base_wires, base_logits)` (optionally with small noise on logits).
  - Resets also reintroduce clones of the preconfigured circuit.
  - Damage logic is unchanged; training focuses on recovery from damage rather than growth from NOPs.

---

### High-Level Flow (reconfig)

1) Preconfigure once on fixed wiring:
   - Build fixed-wiring graph via `gen_circuit(wiring_fixed_key, layer_sizes, arity)`.
   - Optimize logits only (no GNN parameter update) using same loss path used in training (`get_loss_and_update_graph`).
   - Save resulting `(base_wires, base_logits)`.

2) Pool initialization:
   - Create pool by replicating `(base_wires, base_logits)` to `pool_size` and building graphs via `build_graph`.

3) Training loop:
   - Unchanged core training step; pool entries are already configured at start.
   - Periodic damage evaluation and sample-based damage execution remain unchanged.

4) Resets:
   - When resetting a fraction of the pool, reinsert clones of `(base_wires, base_logits)` (optionally with small noise) instead of fresh NOPs.

---

### API Changes ✅ IMPLEMENTED

- ✅ `boolean_nca_cc/training/train_loop.py::train_model`
  - ✅ Add: `training_mode: str = "growth"` (passed from config)
  - ✅ Add: `preconfig_steps: int = 200` (uses `cfg.backprop.epochs`)
  - ✅ Add: `preconfig_lr: float = 1e-2` (uses `cfg.backprop.learning_rate`)
  - ⚠️ `preconfig_logit_noise` not implemented (not needed initially)
  - ✅ Log `training/mode` to WandB.

- ✅ `boolean_nca_cc/training/pool/pool.py::initialize_graph_pool`
  - ✅ Extend signature to accept `base_wires: Optional[PyTree] = None`, `base_logits: Optional[PyTree] = None`.
  - ✅ If provided, replicate and construct graphs from these instead of generating fresh.
  - ✅ Preserve current behavior when `None`.

- ✅ New utility: `preconfigure_circuit_logits(...) -> Tuple[PyTree, PyTree]`
  - ✅ Location: `boolean_nca_cc/training/preconfigure.py`.
  - ✅ Inputs: `wiring_key, layer_sizes, arity, x_data, y_data, loss_type, steps, lr`.
  - ✅ Output: `(base_wires, base_logits)`.
  - ✅ Reuses existing `circuits.train.train_step` for consistency.

---

### Preconfiguration Procedure ✅ IMPLEMENTED

- ✅ Build single circuit using `gen_circuit(wiring_fixed_key, layer_sizes, arity)`.
- ✅ Treat logits as trainable parameters; optimize with `optax.adam` for `preconfig_steps` using `preconfig_lr`.
- ✅ Compute loss using existing `circuits.train.train_step`:
  - ✅ Reuses existing training step for consistency.
  - ✅ `knockout_pattern = None` (no damage during preconfiguration).
- ✅ Validate loss is finite; if NaN, raise and abort reconfig mode.
- ✅ Store `(base_wires, base_logits)` for pool init and resets.

---

### Pool Initialization Semantics ✅ IMPLEMENTED

- ✅ growth: current `initialize_graph_pool(...)` behavior unchanged.
- ✅ reconfig: call `initialize_graph_pool(..., base_wires=..., base_logits=...)`.

---

### Reset Behavior ✅ IMPLEMENTED

- ✅ growth: unchanged; resets generate fresh unconfigured circuits.
- ✅ reconfig: resets insert clones of `(base_wires, base_logits)` from preconfiguration.
- ⚠️ optional noise not implemented (not needed initially).

---

### Damage Application ✅ IMPLEMENTED

- ✅ No change to `damage_fraction` logic. Reuse existing parameters and selection strategies.
- ✅ In reconfig mode, the pool begins configured; damage trajectories train recovery rather than initial growth.

---

### Evaluation

- Replace event-driven damage evaluation with periodic knockout evaluation that starts from a configured, damaged circuit.
- For reconfig: expect strong baseline at epoch 0 (no-damage); log one baseline eval to verify preconfig success if desired.

#### Setup

- Ensure preconfiguration produced `(base_wires_preconfig, base_logits_preconfig)`.
- Set `knockout_eval = {"enabled": True, ...}` in training config.
- In reconfig mode, set `knockout_eval_base_circuit = (base_wires_preconfig, base_logits_preconfig)`.
- Provide a fixed `knockout_vocabulary` (via `knockout_diversity > 0`) for stable in-distribution sampling.
- Keep `periodic_eval_test_seed` constant for determinism.

#### Scheduled periodic evaluation (use existing function)

Use the existing scheduled call (already in `train_model`) to evaluate stepwise trajectories from configured-but-damaged start states:

```python
ko_eval_results, accumulated_pattern_data = run_knockout_periodic_evaluation(
    model=model,
    knockout_vocabulary=knockout_vocabulary,          # fixed vocab for stable ID sampling
    base_wires=base_wires_preconfig,                  # from preconfigure
    base_logits=base_logits_preconfig,                # from preconfigure
    knockout_config=knockout_eval,
    periodic_eval_test_seed=periodic_eval_test_seed,  # fixed seed for determinism
    x_data=x_data,
    y_data=y_data,
    input_n=input_n,
    arity=arity,
    circuit_hidden_dim=circuit_hidden_dim,
    n_message_steps=periodic_eval_inner_steps,        # trajectory length
    loss_type=loss_type,
    epoch=epoch,
    wandb_run=wandb_run,
    eval_batch_size=periodic_eval_batch_size,         # number of patterns per eval
    accumulated_pattern_data=accumulated_pattern_data,
    log_stepwise=True,                                # enable per-step logging
    layer_sizes=layer_sizes,
    use_scan=use_scan,
    knockout_diversity=knockout_diversity,
)
```

Notes:
- `run_knockout_periodic_evaluation` internally replicates `base_wires/base_logits` for the batch and applies both ID (vocabulary or fresh) and OOD patterns.
- Set `periodic_eval_interval` to control cadence; trajectories are comparable since all runs start from the same configured state.
- Keep `periodic_eval_inner_steps` high enough to capture recovery dynamics.

#### Remove event-driven evaluation ✅ IMPLEMENTED

- ✅ Delete the event-driven damage evaluation block executed immediately after applying damage. Trajectories were incomparable due to varying optimization points at damage time.
- ✅ Rely exclusively on the scheduled periodic evaluation, which provides consistent starting states and stable comparisons across epochs.

#### Pattern tracking and logging

- Continue using `accumulated_pattern_data` for persistent scatter plots; it already records per-pattern hard accuracy by epoch.
- With a fixed vocabulary and seed, you can track the same pattern indices across epochs.
- Enable `periodic_eval_log_stepwise=True` to log per-step series for ID and OOD patterns in separate panels.

#### Growth mode

- Growth mode may continue using the same periodic evaluation function by providing any fixed base circuit (e.g., from `gen_circuit`) and its logits. The trajectories will reflect performance from an unconfigured or partially configured start if not preconfigured.

---

### Determinism ✅ IMPLEMENTED

- ✅ Use `wiring_fixed_key` for preconfig and pool init/reset in reconfig mode.
- ✅ Seed any vocabulary sampling and OOD evaluation with `periodic_eval_test_seed` as already present.

---

### Logging (WandB) ✅ IMPLEMENTED

- ✅ `training/mode`: "growth" or "repair".
- ✅ Preconfig-only metrics: `preconfig/steps`.
- ✅ Pool events unchanged: damage counts, fractions, eval trajectories, etc.

---

### Failure Handling ✅ IMPLEMENTED

- ✅ If preconfiguration yields NaN or fails to reduce loss, raise and stop reconfig path; do not silently fall back to growth.
- ✅ Surface error with succinct diagnostics (final loss).

---

### Backward Compatibility ✅ IMPLEMENTED

- ✅ Default `training_mode="growth"`; no behavior change for existing runs and configs.
- ✅ New params ignored in growth mode.

---

### Configuration Summary ✅ IMPLEMENTED

```bash
# Growth mode (default)
python train.py

# Reconfig mode (uses backprop settings for preconfig)
python train.py training.training_mode=reconfig

# Override preconfig settings if needed
python train.py training.training_mode=reconfig backprop.epochs=500 backprop.learning_rate=5e-3
```

---

### Implementation Status

✅ **FULLY IMPLEMENTED**:
- Training mode flag and WandB logging
- Preconfiguration utility using existing `train_step`
- Pool initialization with base wires/logits
- Reconfig flow: preconfigure → init pool → resets use base
- Event-driven eval removed, periodic eval remains
- Config integration (reuses backprop settings)

⚠️ **NOT IMPLEMENTED** (optional for later):
- `preconfig_logit_noise` for clone diversity
- Advanced noise/diversity features

🚀 **READY FOR TESTING**