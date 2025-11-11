## Self-Attention Model Accuracy Drift Issue

This document focuses on investigating and resolving the accuracy drift issue when loading a self-attention model in the GUI. The goal is to maintain accuracy ~1.0 when the circuit is initialized to accuracy 1.0, matching the stable performance observed during training evaluation.

---

## Background: GUI Model Execution

### Accuracy Computation Path
- The generator's `get_loss_and_update_graph(...)` → `get_loss_from_wires_logits(...)` runs `run_circuit(logits, wires, x)` WITHOUT gate_mask.
- Accuracy reflects the current logits only (no masking in execution path).
- The model generator (`evaluate_model_stepwise_generator`) does NOT receive a `knockout_pattern`; the model does standard updates from the current logits.

### Generator Initialization
- Generator is initialized via `initialize_model_generator()` which creates `evaluate_model_stepwise_generator`.
- Generator state includes graph representation, hidden features, and graph globals (including `update_steps` counter).
- Each optimization tick: `self.logits = self.last_step_result.logits` (model applies residual updates).

### Training vs GUI Context
- **Training eval**: Uses `evaluate_model_stepwise_batched` with batched evaluation, shows stable accuracy (>1.0) for >150 inner loop steps.
- **GUI**: Uses `evaluate_model_stepwise_generator` with single circuit, shows progressive accuracy degradation.
- **Reference**: In `train_loop.py`, `eval_ko_in_steps/hard_accuracy` tracks stepwise accuracy readings that show great and stable performance.

---

## 🔴 Issue: Self-Attention Model Accuracy Drift

**Problem**: When loading a self-attention model, accuracy slowly drifts downwards over time, eventually hitting ~0.6. However, during training, the eval loop shows perfect accuracy (>1.0) for >150 inner loop steps.

**Observations**:
- Backprop works fine, converging to accuracy 1.0
- Self-attention model shows stable performance in training eval loop
- Live GUI loop shows progressive accuracy degradation
- Mismatch between training eval and live GUI behavior

**Expected Behavior**:
- When circuit is initialized to accuracy 1.0, it should maintain that accuracy for a while at least
- Model should maintain stable performance similar to training eval loop

**Critical Discovery: Multi-Inject Evaluation Loop Timing**

**Investigation Finding**: The multi-inject evaluation loop does NOT run a warm-up period before applying damage. Damage is injected at step 1, immediately after the initial evaluation (step 0).

**Current Evaluation Behavior**:
- **Step 0**: Initial evaluation (pre-damage baseline, accuracy ~1.0) - logged before loop starts
- **Step 1**: First damage injection + first model update
- **Step 11, 21, 31, ...**: Subsequent damage injections (every `recover_steps + 1` steps, where `recover_steps=10`)

**Key Insight**: In wandb eval curves, accuracy stays stable at 1.0 for the first N steps (before damage is applied). This suggests:
1. **The model IS capable of maintaining a well-configured circuit** (accuracy 1.0) without degrading it
2. **The GUI starts with the same preconfigured state as eval step 0** (same logits, same circuit configuration)
3. **The GUI's immediate degradation suggests a mismatch** in how the model processes updates in GUI vs. eval context

**This challenges the hypothesis that "the model always applies updates"** - if that were true, eval would also degrade immediately. The fact that eval maintains accuracy suggests the model CAN handle good circuits, but something in the GUI context is different.

**Solution Implemented: Damage Start Offset**

Added `damage_start_offset` parameter to allow warm-up period before first damage:
- **`damage_start_offset: int = 0`**: Number of steps to run before first damage injection
- **`damage_start_offset_random: bool = False`**: If True, randomize offset per circuit (0 to `damage_start_offset`)
- **`damage_start_offset_seed: int = 42`**: Seed for random offset generation

**Usage Examples**:
- `damage_start_offset: 10` → Run 10 steps (1-10) before first damage at step 11
- `damage_start_offset: 10, damage_start_offset_random: true` → Random offset 0-10 per circuit
- `damage_start_offset: 0` → Current behavior (damage at step 1)

**Implementation Location**: `boolean_nca_cc/training/evaluation.py` in `_evaluate_with_loop()`

**Next Steps**: Test with `damage_start_offset: 10` to see if eval maintains accuracy during warm-up period, then compare with GUI behavior during the same period. This will help isolate whether the issue is:
- Model behavior (if both degrade during warm-up)
- GUI-specific initialization (if eval maintains but GUI degrades)

---

## Potential Root Causes

### 1. Mismatch in Live Loop vs. Eval Loop Conditions
- Different initialization (graph state, globals, step counters)
- Different number of message steps per iteration
- Different loss computation timing
- Missing or incorrect parameters in generator initialization

### 2. Checkpointing Issue
- Model saved might not be the one showing stable performance
- Model state incomplete (missing optimizer state, graph state, etc.)
- Version mismatch between training and loading code

### 3. Generator State Management
- Generator not properly initialized
- Graph globals (update_steps counter) accumulating incorrectly
- Hidden state not properly reset between iterations
- Graph state initialization doesn't match training conditions

### 4. Model Behavior Differences
- Training uses batched evaluation, GUI uses single circuit
- Training uses different loss computation path
- Training eval might use different initialization timing

---

## Critical Evidence from Terminal Logs

```
Initializing model generator with:
  - hidden_dim: 64
  - use_globals: True
  - model type: CircuitSelfAttention
Initialized model generator with initial loss: 0.0266
Step 0: Loss = 0.2349, Hard Loss = 4.0000
Step 100: Loss = 29.7506, Hard Loss = 110.0000
Step 200: Loss = 88.3226, Hard Loss = 248.0000
```

**Key Findings**:

1. **Initial loss is reasonable**: Initial loss upon generator initialization of 0.0266 is not terrible, but not great either.

2. **🔴 Loss increases dramatically on first step**: The first step increases loss from 0.0266 → 0.2349 (9x increase). This suggests the first generator step is applying updates that degrade performance.

3. **🔴 Loss continues to drift upward**: Loss continues to drift upward over time:
   - Step 0: 0.2349 (9x increase from initial)
   - Step 100: 29.7506 (127x increase from initial)
   - Step 200: 88.3226 (332x increase from initial)
   - This confirms the drift issue persists even with correct preconfiguration.

**Most Likely Root Cause**:
- **Generator initialization using wrong state**: Graph globals (especially `update_steps` counter), hidden features, or initial graph representation don't match training conditions
- **First step applying incorrect updates**: Possibly due to wrong `update_steps` counter causing the model to think it's at a different step than it actually is
- **Mismatch between training eval and GUI generator initialization**: Training eval may initialize graph state differently than GUI generator
- **Progressive degradation**: Each step continues to worsen performance, suggesting systematic issues with how model updates are being applied

---

## Investigation Plan

### ✅ FIXED (But Does Not Resolve Drift): Model Call Parameter Mismatch

**Issue Found**: The model call in `evaluate_model_stepwise_generator` was missing critical parameters that affect attention mask creation.

**Details**: 
- GUI generator called `model(graph)` without `layer_sizes` and `layer_neighbors` parameters
- Training eval calls `model(graph, layer_neighbors=layer_neighbors, layer_sizes=layer_sizes)`
- These parameters affect how attention masks are created, which directly impacts model behavior

**Fix Applied**:
1. Added `layer_neighbors` parameter to `evaluate_model_stepwise_generator` (defaults to `False` to match training)
2. Updated model call to pass `layer_sizes` and `layer_neighbors` for `CircuitSelfAttention` models
3. Updated both `GUI.py` and `GUI_minimal.py` to pass `layer_neighbors=False` when initializing generator

**Files Changed**:
- `boolean_nca_cc/training/evaluation.py`: Added `layer_neighbors` parameter and updated model call
- `GUI_minimal.py`: Added `layer_neighbors=False` to generator initialization
- `GUI.py`: Added `layer_neighbors=False` to generator initialization

**Status**: ✅ Fix applied, but **does NOT resolve the accuracy drift issue**. The drift persists, indicating there are additional root causes to investigate.

### 🔴 HYPOTHESIS CHALLENGED: Model Always Applies Updates

**Critical Finding from Logs**:
- Preconfigured circuit: `loss=0.026637, hard_accuracy=0.9971` (not perfect, but very good)
- Generator initializes correctly with same loss: `globals=[loss=0.026637, update_steps=0]`
- **First model step**: Loss increases 9x (`0.026637` → `0.234872`) while accuracy stays `1.0000`
- **Progressive drift**: Loss continues increasing over 200 steps

**Original Hypothesis**:
The model was trained to optimize circuits from a **worse starting state**. It always applies learned residual updates:
```python
updated_logits = nodes["logits"] + self.logit_scale * logit_updates
```

When given a well-configured circuit (via preconfigure), the model:
1. Doesn't recognize the circuit is already good
2. Still applies updates optimized for improving worse circuits
3. These updates degrade performance because they're inappropriate for the current state

**Challenge to Hypothesis**:
- **Eval loop shows model CAN maintain accuracy 1.0** for multiple steps before damage is applied
- This suggests the model IS capable of handling well-configured circuits
- The GUI's immediate degradation suggests a **GUI-specific issue**, not a fundamental model limitation

**Revised Hypothesis**:
The issue may not be that "the model always applies updates" but rather:
1. **GUI initialization mismatch**: Graph state (globals, hidden features, `update_steps` counter) doesn't match training eval initialization
2. **Context difference**: Single-circuit generator vs. batched evaluation behaves differently
3. **State accumulation**: Graph globals or hidden state accumulating incorrectly in GUI context

**Potential Solutions** (if hypothesis is correct):
1. **Fix GUI initialization**: Ensure graph globals match training eval exactly
2. **Fix state management**: Ensure `update_steps` counter and hidden state match training conditions
3. **Compare step-by-step**: Use `damage_start_offset: 10` to compare eval vs GUI during warm-up period

### Testing Strategy: Zero-Damage Training Run

**Goal**: Test if the model degrades well-configured circuits even when trained without damage.

**Setup**:
1. **Disable training damage**: Set `pool.damage_pool_enabled: false` in config (or via command line: `pool.damage_pool_enabled=false`)
2. **Keep repair mode**: Training mode should remain "repair" to use preconfigured circuits
3. **Add simple no-damage evaluation**: Evaluate model on preconfigured circuits without any damage

**Expected Behavior**:
- If model maintains good performance: Hypothesis is wrong - model can handle good circuits, issue is damage-related
- If model still degrades: Confirms hypothesis - model always applies updates even to good circuits

**Implementation for No-Damage Evaluation**:
Add to `train_loop.py` around line 1816 (in the `else` branch when `knockout_eval.enabled=false`):

```python
# Simple no-damage evaluation: just evaluate model on base circuit
if knockout_eval_base_circuit is not None and epoch % periodic_eval_interval == 0:
    base_wires, base_logits = knockout_eval_base_circuit
    
    # Replicate base circuit for batch
    eval_batch_size = 16  # Small batch for simple eval
    eval_wires = jax.tree.map(
        lambda x: jp.repeat(x[None, ...], eval_batch_size, axis=0), base_wires
    )
    eval_logits = jax.tree.map(
        lambda x: jp.repeat(x[None, ...], eval_batch_size, axis=0), base_logits
    )
    
    # Evaluate without any damage (knockout_patterns=None)
    step_metrics = evaluate_circuits_in_chunks(
        eval_fn=evaluate_model_stepwise_batched,
        wires=eval_wires,
        logits=eval_logits,
        knockout_patterns=None,  # No damage!
        target_chunk_size=eval_batch_size,
        model=model,
        x_data=x_data,
        y_data=y_data,
        input_n=input_n,
        arity=arity,
        circuit_hidden_dim=circuit_hidden_dim,
        n_message_steps=periodic_eval_inner_steps,
        loss_type=loss_type,
        layer_sizes=layer_sizes,
        return_per_pattern=False,
        layer_neighbors=layer_neighbors,
        # Disable damage injection
        damage_mode="greedy",  # Won't matter since no patterns
        damage_injection_mode="single",
        max_damage_per_circuit=0,  # No damage
    )
    
    # Log metrics similar to knockout eval
    final_metrics = {
        "eval_no_damage/final_loss": step_metrics["soft_loss"][-1],
        "eval_no_damage/final_hard_loss": step_metrics["hard_loss"][-1],
        "eval_no_damage/final_accuracy": step_metrics["soft_accuracy"][-1],
        "eval_no_damage/final_hard_accuracy": step_metrics["hard_accuracy"][-1],
        "eval_no_damage/epoch": epoch,
    }
    
    if wandb_run:
        wandb_run.log(final_metrics)
    
    # Log stepwise if enabled
    if periodic_eval_log_stepwise:
        for step_idx in range(len(step_metrics["step"])):
            wandb_run.log({
                "eval_no_damage_steps/step": step_metrics["step"][step_idx],
                "eval_no_damage_steps/loss": step_metrics["soft_loss"][step_idx],
                "eval_no_damage_steps/hard_loss": step_metrics["hard_loss"][step_idx],
                "eval_no_damage_steps/accuracy": step_metrics["soft_accuracy"][step_idx],
                "eval_no_damage_steps/hard_accuracy": step_metrics["hard_accuracy"][step_idx],
                "eval_no_damage_steps/epoch": epoch,
            })
    
    log.info(
        f"No-Damage Eval (epoch {epoch}): "
        f"Loss={final_metrics['eval_no_damage/final_loss']:.4f}, "
        f"Acc={final_metrics['eval_no_damage/final_accuracy']:.4f}, "
        f"Hard Acc={final_metrics['eval_no_damage/final_hard_accuracy']:.4f}"
    )
    
    current_eval_metrics = final_metrics
```

**Checkpoint Configuration**:
When running zero-damage training, the checkpoint system needs to be configured to use the no-damage evaluation metrics:

**Option 1: Use `best_metric_source: "eval"` (Recommended)**
- The `"eval"` source automatically falls back to `eval_no_damage` metrics when `eval_ko_in` is not available
- This works automatically: `get_metric_value` checks `eval_ko_in` → `eval_ko_out` → `eval_no_damage` in order
- Config example:
  ```yaml
  checkpoint:
    best_metric: "hard_accuracy"
    best_metric_source: "eval"  # Will use eval_no_damage when knockout eval is disabled
  ```

**Option 2: Use `best_metric_source: "eval_no_damage"` (Explicit)**
- Explicitly use no-damage evaluation metrics
- Config example:
  ```yaml
  checkpoint:
    best_metric: "hard_accuracy"
    best_metric_source: "eval_no_damage"  # Explicitly use no-damage eval
  ```

**Note**: The current config uses `best_metric_source: "eval_ko_in"`, which will fail if knockout eval is disabled. Either:
- Change to `"eval"` (automatic fallback), or
- Change to `"eval_no_damage"` (explicit), or
- Keep knockout eval enabled but disable damage (`pool.damage_pool_enabled: false`)

**Early Stopping**: Early stopping also uses `get_metric_value`, so it will automatically work with no-damage evaluation when using `stop_accuracy_source: "eval"`. The same fallback logic applies: `eval_ko_in` → `eval_ko_out` → `eval_no_damage`.

**What to Monitor**:
- `eval_no_damage_steps/hard_accuracy`: Should maintain ~1.0 if model handles good circuits well
- `eval_no_damage_steps/loss`: Should remain low (similar to initial loss ~0.026)
- Compare with GUI behavior: If training eval shows drift but GUI doesn't (or vice versa), indicates environment-specific issue
- Checkpoint saving: Verify best model is saved based on `eval_no_damage/final_hard_accuracy` when using `best_metric_source: "eval"`

### Active Investigation Items

1. **✅ COMPLETED: Multi-Inject Evaluation Loop Investigation**
   - **Finding**: Damage is injected at step 1 (no warm-up period)
   - **Finding**: Eval shows model CAN maintain accuracy 1.0 for multiple steps before damage
   - **Solution**: Implemented `damage_start_offset` parameter to allow warm-up period
   - **Next**: Test with `damage_start_offset: 10` to compare eval vs GUI during warm-up period
   - **Priority**: High - This will help isolate whether issue is model behavior or GUI-specific

2. **Compare graph globals initialization** between training eval and GUI generator (especially `update_steps` counter)
   - Verify initial values match exactly
   - Check if counter is being reset or initialized incorrectly
   - **Priority**: High - `update_steps` counter affects model behavior via graph globals

3. **Investigate first generator step degradation**: Why does loss increase from 0.0266 → 0.2349 (9x) on first step?
   - Check if `update_steps` counter is being read correctly on first step
   - Verify graph globals are properly initialized before first model call
   - Compare first step behavior between GUI and training eval

4. **Investigate progressive drift**: Why does loss continue to increase from 0.2349 → 29.7506 → 88.3226 over 200 steps?
   - Check if graph globals are accumulating incorrectly
   - Verify hidden state updates are correct
   - Compare update mechanism between GUI and training eval
   - Check if `update_steps` counter increments correctly

5. **Compare graph state initialization** between training eval and GUI generator
   - Verify graph construction matches exactly (especially positional encodings)
   - Compare initial hidden features
   - Check if build_graph parameters match

### High Priority (Continued)

6. **Compare exact initialization and step execution** between `evaluate_model_stepwise_generator` (GUI) and `evaluate_model_stepwise_batched` (training eval)
   - Most important reference: conditions in `train_loop.py` which generate `eval_ko_in_steps/hard_accuracy` as logged to wandb
   - These track the stepwise accuracy readings of an inner loop, and show great performance
   - **Key difference**: Training eval uses batched evaluation, GUI uses single circuit - need to verify this doesn't cause issues

7. **Add debug logging** to track logits state, accuracy, and graph globals at each step
   - ✅ **COMPLETED**: Added debug logging to `evaluate_model_stepwise_generator`
   - Logs initial graph globals (step 0)
   - Logs graph globals before and after model call for first 5 steps and every 50th step
   - Logs `update_steps` counter, loss, and accuracy at each logged step
   - **Next**: Run GUI and analyze logs to identify differences
   - **Priority**: High - needed to diagnose the drift

### Medium Priority

7. **Verify checkpoint loading** includes all necessary model state
   - Check if optimizer state is needed (probably not for inference)
   - Verify model weights are loaded correctly

8. **Cross-reference wandb eval panel** for run `pljk5kp6` (twilight-butterfly-686) to compare training eval metrics with GUI behavior

---

## Key Files to Investigate

- `GUI.py` / `GUI_minimal.py`: Generator initialization (`initialize_model_generator()`)
- `training/train_loop.py`: Training eval loop (`evaluate_model_stepwise_batched`)
- `circuits/tasks.py`: Generator functions (`evaluate_model_stepwise_generator`)
- `models/self_attention.py`: Model forward pass and update mechanism
- `utils/graph_builder.py`: Graph construction and initialization

---

## Success Criteria

- [ ] GUI maintains accuracy ~1.0 when initialized to accuracy 1.0
- [ ] First generator step does not degrade performance (loss should remain stable or improve)
- [ ] Loss/accuracy remains stable over 200+ steps (matching training eval behavior)
- [ ] Graph globals initialization matches training eval exactly
- [ ] `update_steps` counter behavior matches training eval

## Investigation Status

- ✅ **Fixed**: Model call parameter mismatch (layer_sizes/layer_neighbors) - but does NOT resolve drift
- ✅ **Identified**: Preconfigured circuit has `loss=0.026637, hard_accuracy=0.9971` (not perfect, but close)
- ✅ **Investigated**: Multi-inject evaluation loop timing - damage injected at step 1 (no warm-up)
- ✅ **Implemented**: `damage_start_offset` parameter to allow warm-up period before first damage
- 🔴 **Hypothesis Challenged**: Original hypothesis that "model always applies updates" is challenged by:
  - Eval loop shows model CAN maintain accuracy 1.0 for multiple steps before damage
  - This suggests model IS capable of handling well-configured circuits
  - GUI's immediate degradation suggests GUI-specific initialization or state management issue
- 🔴 **Active**: Need to test with `damage_start_offset: 10` to compare eval vs GUI during warm-up period
  - If both degrade: Confirms model behavior issue
  - If eval maintains but GUI degrades: Confirms GUI-specific initialization issue

