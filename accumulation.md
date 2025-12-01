# Damage Accumulation Strategy

## Goal
Ensure that "permanent" damage in multi-damage settings behaves correctly by accumulating damage patterns, while "reversible" damage remains transient (no accumulation).

## Problem
Currently, `GraphPool.apply_knockouts` replaces the existing knockout pattern with the new one. 
- **Permanent Mode**: This effectively "heals" previously damaged nodes when new damage is applied, violating the "permanent" constraint in multi-damage scenarios.
- **Reversible Mode**: Replacement is the correct behavior, as the model is expected to recover fully between damage events.

## Strategy

### 1. `GraphPool.apply_knockouts` Modification
Modify `apply_knockouts` to accept an `accumulate` boolean flag.
- **If `accumulate=True`**: `updated_patterns = current_patterns | new_patterns` (Bitwise OR).
- **If `accumulate=False`**: `updated_patterns = new_patterns` (Replacement).

### 2. Training Loop Integration (`train_loop.py`)
In the training loop, determines whether to accumulate based on the model's `damage_behavior`.
- Access `model.damage_behavior` (attribute of `CircuitSelfAttention`).
- If `model.damage_behavior == "permanent"`, set `accumulate=True`.
- Otherwise (e.g., "reversible"), set `accumulate=False`.
- Pass this flag to `circuit_pool.apply_knockouts`.

### 3. Reversible Mode Investigation (Future)
- Confirm if `reversible` mode logic correctly handles step counters during training (currently it relies on `step_count == 0`).
- Ensure `apply_knockouts` or the training step handles any necessary state resets for reversible mode if needed (though user stated "no memory... is necessary").

## Implementation Details

### File: `boolean_nca_cc/training/pool/pool.py`
```python
def apply_knockouts(self, idxs, new_knockout_patterns, accumulate=False):
    # ...
    if accumulate:
         updated = current | new
    else:
         updated = new
    # ...
```

### File: `boolean_nca_cc/training/train_loop.py`
```python
# Inside damage pool block
damage_behavior = getattr(model, "damage_behavior", "permanent")
should_accumulate = (damage_behavior == "permanent")

circuit_pool = circuit_pool.apply_knockouts(
    damaged_idxs, 
    new_patterns, 
    accumulate=should_accumulate
)
```

