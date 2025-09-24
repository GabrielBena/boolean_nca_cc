## Simple SEU (Reversible Bit Perturbation) Mode

This document summarizes the minimal reversible damage implementation (SEU-like), how it integrates with the existing training/eval pipeline, and how to extend greedy knockout patterns to cycle with a per-circuit perturbation counter.

### What “reversible” means here

- Nodes are NOT removed from the attention graph and are NOT zeroed out of residual updates.
- A one-shot logit bias is applied to damaged nodes at the first model step of the episode. Subsequent steps apply normal residual updates so the gate can be “healed.”
- Evaluation remains unchanged and still reports: step 0 baseline (pre-damage), step 1 damage application, then recovery over following steps.

### Where it’s implemented

- Model toggle and behavior:

  - `boolean_nca_cc/models/self_attention.py` (class `CircuitSelfAttention`)
    - Constructor now accepts `damage_behavior: "permanent" | "reversible"` and `reversible_bias: float`.
    - In `__call__`:
      - Permanent mode: unchanged (prune attention, clamp logits to large negative, zero residual updates for damaged nodes).
      - Reversible mode: attention mask ignores knockout (keep connectivity), no zeroing; at first step (`globals[..., 1] == 0`), add a logit bias to damaged nodes. All steps after that are clean updates (recovery path).
- Config knobs:

  - `configs/model/self_attention.yaml`
    - `damage_behavior: "reversible"`
    - `reversible_bias: -10.0` (tunable)
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

## Greedy knockout patterns: Vocabulary-based and deterministic modes

### Primary Approach: Greedy Vocabulary with Statistical Sampling

**Goal**: Create diverse damage patterns from greedy indices while maintaining seen/unseen distinction and statistical robustness.

**Design**:

- Generate a vocabulary of subset patterns from `greedy_ordered_indices`
- Training: Random sampling from vocabulary for pool injections
- Evaluation: Multiple patterns per injection for statistical confidence
- Clear seen/unseen distinction through vocabulary reuse vs fresh generation

### Legacy Approach: Deterministic Rolling Window

**Goal**: Cycle through a fixed, ordered list of gate indices deterministically:

- First perturbation of a circuit uses `greedy_indices[0:window_size]`
- Second perturbation uses `greedy_indices[window_size:2*window_size]`, etc.
- Wrap around with modulo when reaching the end

### Implementation Files

- `boolean_nca_cc/training/pool/structural_perturbation.py`
  - `DEFAULT_GREEDY_ORDERED_INDICES`: Default ordering of critical gate indices
  - `create_greedy_knockout_pattern()`: Creates deterministic patterns from ordered indices
  - `create_knockout_vocabulary()`: Extended to support vocabulary generation from greedy subsets
  - `create_group_greedy_pattern()`: Rolling window pattern creation (legacy mode)

### Implementation: Vocabulary-based Greedy Patterns

**Greedy Vocabulary Mode** (`damage_mode: "greedy_vocabulary"`):

- **Vocabulary Generation**: ✅ **IMPLEMENTED & TESTED**

  - Create `damage_knockout_diversity` subset patterns from `greedy_ordered_indices`
  - Each subset has `damage_prob` gates (e.g., 5 gates per pattern)
  - Uses random sampling from greedy indices only (no strip mode needed)
  - Added `create_greedy_subset_random_pattern()` helper function
  - Extended `create_knockout_vocabulary()` with new mode
- **Training Injections**:

  - `damage_pool_fraction` (15%) of circuits get damaged per injection epoch
  - Each selected circuit receives one randomly sampled pattern from vocabulary
  - Pool tracking via `perturb_counter` remains unchanged
- **Evaluation Injections**:

  - **Seen**: Sample `patterns_per_injection` patterns from training vocabulary
  - **Unseen**: Generate `patterns_per_injection` fresh random subsets from greedy indices (no vocabulary reuse)
  - Multiple patterns per injection enable statistical analysis

### Legacy Implementation: Rolling Window Mode

**Deterministic Rolling Window** (`damage_mode: "greedy"`):

- At damage epochs, select `damaged_idxs` via `get_reset_indices(...)`
- For each damaged circuit:
  - `count = perturb_counter[idx]`
  - `start = (count * greedy_window_size) % len(greedy_ordered_indices)`
  - Build pattern via `create_group_greedy_pattern(ordered_indices, layer_sizes, start, window_size)`
- Apply patterns and increment `perturb_counter` automatically

### Config cheat sheet

#### Greedy Vocabulary Mode (Recommended)

```yaml
pool:
  damage_mode: "greedy_vocabulary"           # Enable vocabulary-based sampling
  damage_knockout_diversity: 100            # Vocabulary size (number of subset patterns)
  damage_prob: 5                           # Gates per subset pattern  
  greedy_ordered_indices: [48,17,52,...]   # Greedy indices to sample from
  damage_pool_enabled: true
  damage_pool_fraction: 0.15               # 15% of circuits damaged per training injection
  damage_pool_interval: 128                # Damage every 128 epochs

eval:
  knockout_eval:
    patterns_per_injection: 8              # Multiple patterns per eval injection (statistical robustness)
    unseen_mode: true                      # Generate fresh patterns vs reuse training vocabulary
    greedy_injection_recover_steps: 10     # Recovery steps between injections
    greedy_num_injections: 10              # Number of injections then damage-free tail
```

#### Legacy Rolling Window Mode

```yaml
pool:
  damage_mode: "greedy"                    # Deterministic rolling window
  greedy_window_size: ${pool.damage_prob}  # Window size (5)
  damage_prob: 5                          # Gates per window
  greedy_ordered_indices: [48,17,52,...]  # Ordered indices for rolling window
```

### Parameter Responsibilities

- **`damage_pool_fraction`**: How many circuits get damaged per training injection
- **`patterns_per_injection`**: How many patterns sampled per evaluation injection
- **`damage_knockout_diversity`**: Vocabulary size (number of subset patterns)
- **`damage_prob`**: Pattern size (gates per pattern)

### Seen vs Unseen Patterns

- **Seen**: Evaluation samples from the same vocabulary used during training
- **Unseen**: Evaluation generates fresh random subsets from greedy indices (no vocabulary reuse)

---

## Advantages of Vocabulary Approach

### Statistical Robustness

- **Multiple patterns per injection**: `patterns_per_injection` enables confidence intervals and statistical analysis
- **Diverse training exposure**: Circuits experience varied damage patterns rather than predictable sequences
- **Reproducible randomness**: Vocabulary ensures consistent seen/unseen distinction across runs

### Experimental Control

- **Clear semantics**: "Seen" = trained on this pattern subset, "Unseen" = novel combination
- **Scalable evaluation**: Adjustable `patterns_per_injection` balances thoroughness vs computational cost
- **Parameter separation**: Clear roles for `damage_pool_fraction` (training) vs `patterns_per_injection` (eval)

### Backward Compatibility

- **Legacy preservation**: Deterministic rolling window mode remains available
- **Parameter reuse**: Leverages existing `damage_knockout_diversity` for vocabulary size
- **Evaluation integration**: Works with existing periodic injection infrastructure

---

## Implementation Status

### ✅ **Completed: Step 1 - Vocabulary Generation**

- **Added `create_greedy_subset_random_pattern()` function**: Randomly samples from greedy indices
- **Extended `create_knockout_vocabulary()` with `"greedy_vocabulary"` mode**: Creates diverse pattern vocabularies
- **Comprehensive testing validated**: Generates correct patterns, enforces greedy-only constraint, produces diversity
- **Backward compatibility maintained**: All existing modes (`"shotgun"`, `"strip"`, `"greedy"`) unchanged

### ✅ **Completed: Step 2 - Training Loop Integration**

- **Integrated `"greedy_vocabulary"` mode with training loop**: Modified damage injection logic to support vocabulary sampling
- **Updated damage mode selection**: Legacy `"greedy"` uses rolling window, `"greedy_vocabulary"` uses vocabulary sampling  
- **Maintained backward compatibility**: All existing modes remain functional
- **Updated documentation**: Function signatures and docstrings reflect new capabilities

### ✅ **Completed: Step 3 - Evaluation Extension**

- **Extended evaluation with vocabulary support**: Added `damage_mode`, `patterns_per_injection`, `unseen_mode`, `knockout_vocabulary` parameters
- **Maintained greedy injection structure**: Reused existing `greedy_injection_recover_steps`, `greedy_num_injections` parameters and injection schedule
- **Implemented seen/unseen modes**: Vocabulary sampling vs fresh pattern generation for comprehensive evaluation
- **Preserved backward compatibility**: Legacy `"greedy"` rolling window mode remains unchanged

### ✅ **Completed: Step 4 - Config Integration**

- **Updated config schema**: Added `damage_mode`, `patterns_per_injection`, `unseen_mode` to `knockout_eval` section
- **Enhanced parameter flow**: Config parameters now flow through `train.py` → `train_loop.py` → `evaluation.py`
- **Backward compatibility**: Existing configs work with sensible defaults for new parameters
- **Example config provided**: Complete greedy vocabulary configuration in `config.yaml`

### 🎉 **Complete Implementation Summary**

**Greedy Vocabulary Mode** is now fully integrated into the Boolean NCA training pipeline:

1. **Pattern Generation**: `create_greedy_subset_random_pattern()` samples diverse subsets from greedy indices
2. **Vocabulary Creation**: `create_knockout_vocabulary(..., damage_mode="greedy_vocabulary")` builds pattern libraries  
3. **Training Integration**: Pool damage injection uses vocabulary sampling vs rolling window based on `damage_mode`
4. **Evaluation Extension**: Stepwise evaluation supports seen/unseen patterns with statistical robustness
5. **Config Framework**: Complete parameter flow from YAML configs through training to evaluation

The system now supports sophisticated damage/recovery analysis with controlled vocabulary diversity while maintaining full backward compatibility with existing approaches.

---

## Unified Damage Control System

### **Current Implementation Analysis**

**✅ What Works**:
- **Training**: `perturb_counter` tracks damage events per circuit, automatically incremented
- **Evaluation IN (`eval_ko_in`)**: Multi-damage supported via `event_count` with `greedy_num_injections` capping
- **Pattern Generation**: Both rolling window and vocabulary modes functional
- **Reversible Damage**: One-shot bias implementation working correctly

**❌ Critical Gaps**:
- **Training-Evaluation Mismatch**: `perturb_counter` (training) vs `event_count` (eval) are separate systems
- **No Single-Damage Mode**: No mechanism to limit circuits to exactly one damage event
- **Evaluation OUT (`eval_ko_out`)**: Uses static patterns only, no multi-damage support
- **Unused Features**: `patterns_per_injection` parameter exists but not properly utilized

### **Counter System Clarification**

**Training Pool `perturb_counter`**:
- Tracks cumulative damage events per circuit across all training epochs
- Incremented by `pool.apply_knockouts()` during training
- Persists across training epochs and is used for rolling window patterns
- **Not relevant for evaluation** - each evaluation trajectory starts fresh

**Evaluation `eval_perturb_counter`** (formerly `event_count`):
- Tracks damage events per circuit **within a single evaluation trajectory**
- Resets to zero for each new evaluation run
- Used to cap damage injections per circuit (`eval_perturb_counter < max_damage_per_circuit`)
- Used for rolling window start positions and random seed generation
- **Independent of training pool state**

### **Unified Damage Control Design**

#### **Mode A: Single Damage Per Circuit**
```yaml
damage_mode: "greedy_vocabulary"  # or "greedy"
damage_injection_mode: "single"   # NEW: single vs multi
max_damage_per_circuit: 1         # NEW: unified control
```

**Training**: Only damage circuits where `perturb_counter == 0`
**Evaluation**: Only damage circuits where `perturb_counter == 0`

#### **Mode B: Multi-Damage Per Circuit**
```yaml
damage_mode: "greedy_vocabulary"  # or "greedy"  
damage_injection_mode: "multi"    # NEW: single vs multi
max_damage_per_circuit: 10        # NEW: unified control (replaces greedy_num_injections)
```

**Training**: Damage circuits where `perturb_counter < max_damage_per_circuit`
**Evaluation**: Damage circuits where `perturb_counter < max_damage_per_circuit`

### **Implementation Strategy**

#### **Phase 1: Add Unified Configuration** ✅ **COMPLETED**
- Add `damage_injection_mode: "single" | "multi"` parameter
- Add `max_damage_per_circuit: int` parameter (replaces `greedy_num_injections`)
- Update config schema and parameter flow

#### **Phase 2: Unify Counter Systems** ✅ **COMPLETED**
- **Rename `event_count` → `eval_perturb_counter`** in evaluation for clarity
- **Keep evaluation counter independent** from training pool's `perturb_counter` (they serve different purposes)
- **Maintain same increment logic** - evaluation tracks per-trajectory damage, training tracks cumulative damage
- **No functional changes needed** - the evaluation counter already properly caps damage injections per circuit

#### **Phase 3: Implement Single-Damage Mode** ✅ **COMPLETED**
- **Training**: Filter damaged circuits by `perturb_counter == 0`
- **Evaluation**: Filter damaged circuits by `perturb_counter == 0`
- Add debug logging for single-damage enforcement

#### **Phase 4: Unify Multi-Damage Control** ✅ **COMPLETED**
- **Training**: Use `perturb_counter < max_damage_per_circuit` filtering
- **Evaluation**: Use `perturb_counter < max_damage_per_circuit` filtering
- Ensure both modes respect the same damage limits

#### **Phase 5: Fix Evaluation OUT Multi-Damage** ✅ **COMPLETED**
- **Current**: `eval_ko_out` uses static patterns only
- **Target**: Add multi-damage support to `eval_ko_out` using same `perturb_counter` logic
- Enable `patterns_per_injection` for statistical robustness
- **Implementation**: 
  - Removed `unseen_mode` toggle - evaluation now always runs both seen (IN) and unseen (OUT)
  - OUT evaluation uses `knockout_vocabulary=None` to force unseen pattern generation
  - Added random seed offset (+1000) for OUT evaluation to ensure different patterns from IN
  - Both IN and OUT evaluations now support unified damage control system
  - Fixed critical bug where OUT evaluation was using same patterns as IN due to identical random seeds

#### **Phase 6: Fix Vocabulary Mode Consistency** ✅ **COMPLETED**
- **Training**: Keep random sampling but respect damage limits
- **Evaluation**: Use vocabulary sampling but respect damage limits
- Ensure `patterns_per_injection` works correctly
- **Implementation**: 
  - Vocabulary mode consistency achieved through unified damage control system
  - Both training and evaluation respect `max_damage_per_circuit` limits
  - `patterns_per_injection` parameter properly utilized for statistical robustness
  - Clean separation between seen (vocabulary) and unseen (fresh) pattern generation

### **Configuration Matrix**

| Mode | Training Control | Eval IN Control | Eval OUT Control | Pattern Type |
|------|------------------|-----------------|------------------|--------------|
| **Single Rolling** | `perturb_counter == 0` | `eval_perturb_counter == 0` | `eval_perturb_counter == 0` | Deterministic sequence |
| **Multi Rolling** | `perturb_counter < max` | `eval_perturb_counter < max` | `eval_perturb_counter < max` | Deterministic sequence |
| **Single Vocab** | `perturb_counter == 0` | `eval_perturb_counter == 0` | `eval_perturb_counter == 0` | Random from vocabulary |
| **Multi Vocab** | `perturb_counter < max` | `eval_perturb_counter < max` | `eval_perturb_counter < max` | Random from vocabulary |

**Note**: Training uses pool's `perturb_counter` (cumulative across epochs), evaluation uses `eval_perturb_counter` (per-trajectory).

### **Key Benefits**

1. **Unified Control**: Same damage limits across training and evaluation (via `max_damage_per_circuit`)
2. **Clear Naming**: `eval_perturb_counter` clearly indicates evaluation-specific counter (no confusion with training pool)
3. **Single-Damage Mode**: Clean one-shot damage testing
4. **Multi-Damage Support**: Both `eval_ko_in` and `eval_ko_out` support multi-damage
5. **Statistical Robustness**: Proper `patterns_per_injection` utilization
6. **Independent Systems**: Training and evaluation counters serve their respective purposes without interference

---

## Future Improvements

### 📋 **TODO: Strategic Config Refactoring (Option 3)**

**Goal**: Streamline damage mode configuration and eliminate redundant `greedy_eval_enabled` parameter.

**Planned Changes**:
1. **Rename `greedy_eval_enabled` → `multi_injections`**: Single boolean to control static vs periodic injection behavior
2. **Add missing static modes**: 
   - Static `"greedy_vocabulary"` (replaces `"shotgun"` - no wasted indices)
   - Static `"greedy"` (for completeness)
3. **Unified config semantics**:
   ```yaml
   knockout_eval:
     damage_mode: "greedy_vocabulary"  # Pattern type
     multi_injections: true           # Static (false) vs Periodic (true)
   ```

**Complete Matrix Target**:
| `damage_mode` | `multi_injections: false` | `multi_injections: true` |
|---------------|---------------------------|--------------------------|
| `"shotgun"` | ✅ Vocab sampling | 🔄 *Future: periodic shotgun* |
| `"strip"` | ✅ Vocab sampling | 🔄 *Future: periodic strip* |  
| `"greedy"` | 🔨 **Add: static greedy** | ✅ Rolling window |
| `"greedy_vocabulary"` | 🔨 **Add: static vocab** | ✅ Fresh/vocab patterns |

**Benefits**: 
- **Clearer semantics**: One parameter controls injection timing, another controls pattern type
- **Better static modes**: Static greedy_vocabulary is superior to shotgun (no wasted indices)
- **Future extensibility**: Framework for adding periodic shotgun/strip modes later
- **Reduced config confusion**: Eliminates `greedy_eval_enabled` redundancy

**Estimated Effort**: 2-3 hours

---

### ✅ **Completed: `greedy_eval_enabled` Disentanglement**

**Problem**: Redundant configuration with `greedy_eval_enabled` and `damage_mode` controlling overlapping behavior.

**Solution**: 
- **`damage_mode` now controls both pattern type AND injection timing**:
  - `"greedy"` and `"greedy_vocabulary"` → Periodic injections enabled
  - `"shotgun"` and `"strip"` → Static patterns only (no periodic injections)
- **`greedy_eval_enabled` marked as DEPRECATED** with legacy fallback support
- **Updated config comments** to clarify the unified control mechanism

**Result**: Cleaner, more intuitive configuration where `damage_mode` is the single source of truth for evaluation behavior.
