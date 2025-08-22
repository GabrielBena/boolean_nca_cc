## Sample-based Damage

Status: core functionality implemented; evaluation upgrades pending.

### Goal

- Apply knockout (KO) masks to arbitrary pool entries during training, independent of resets.
- Keep implementation minimal and aligned with existing reset flow.

### Scope

- Touchpoints: `boolean_nca_cc/training/pool/pool.py`, `boolean_nca_cc/training/train_loop.py`.
- Reuse: `boolean_nca_cc/training/pool/structural_perturbation.py` and existing selection logic.

### Insertion Points

- `train_loop.py` (`train_model`): post `pool_train_step`; first reset, then damage

### Pool API Additions (minimal) [implemented]

- In `GraphPool` (in `pool.py`):
  - `apply_knockouts(idxs, new_knockout_patterns) -> GraphPool` [done]
    - Updates only `self.knockout_patterns.at[idxs]`; leaves `graphs`, `wires`, `logits`, `reset_counter` unchanged. JIT-wrapped.
  - `damage_fraction(key, fraction, layer_sizes, damage_prob, selection_strategy="uniform", combined_weights=(0.5, 0.5), knockout_vocabulary=None) -> (GraphPool, idxs)` [done]
    - Selects indices via `get_reset_indices(...)` (uniform/steps_biased/loss_biased/combined); returns `(updated_pool, damaged_idxs)`.
    - KO masks: from `knockout_vocabulary` if provided, else `vmap(create_reproducible_knockout_pattern(...))`.
    - Applies via `apply_knockouts(...)`. JIT-wrapped with `static_argnames=("layer_sizes",)`.

Signatures (reference):

```python
def apply_knockouts(self, idxs: Array, new_knockout_patterns: Array) -> "GraphPool"

def damage_fraction(
    self,
    key: Array,
    fraction: float,
    layer_sizes: List[Tuple[int, int]],
    damage_prob: float,
    selection_strategy: str = "uniform",
    combined_weights: Tuple[float, float] = (0.5, 0.5),
    knockout_vocabulary: Optional[Array] = None,
) -> Tuple["GraphPool", Array]
```

### Index Selection Semantics and Mirrored Structure

- `get_reset_indices(...)`: unique indices (`replace=False`); count = `round(fraction * pool_size)` with floor=1; strategies: `uniform`, `steps_biased`, `loss_biased`, `combined`; returns `(idxs, avg_steps_of_selected)`.
- Current behavior: `damage_fraction(...)` reuses `get_reset_indices(...)` selection semantics as-is.

IMPORTANT: for damage we ultimately want lowest-loss selection when a loss-driven policy is used. Today `loss_biased` selects higher-loss entries (reset intent). A future change should invert this for damage (e.g., introduce a `loss_lowest` strategy or an `invert_loss=True` flag for damage paths).

- Mutations:

  - `damage_fraction(...)`: only updates `knockout_patterns` via `apply_knockouts(...)`. It does not touch `graphs`, `wires`, `logits`, or `reset_counter`.
  - `reset_fraction(...)`: replaces graphs/wires/logits (and optionally patterns) at selected indices and increments `reset_counter`.

### Train Loop Integration [implemented]

- Config knobs (in `configs/config.yaml` under `pool`, decoupled from legacy):
  - `damage_pool_enabled`, `damage_pool_interval`, `damage_pool_fraction`
  - `damage_strategy`, `damage_combined_weights`
  - `damage_prob` (gates per pattern for fresh masks)
  - `damage_knockout_diversity` (vocabulary size; if >0, builds vocabulary)
- In `train_loop.train_model` (post `pool_train_step(...)`):
  - If enabled and interval hit, and (`damage_prob` > 0 or vocabulary present):
    - Call `circuit_pool.damage_fraction(...)`; capture `damaged_idxs`.
    - Log `damaged_count` and `damaged_fraction`.
- Resets: `reset_fraction(...)` now passes `new_knockout_patterns=None` so resets never inject or remove KOs; damage is applied strictly via the new path.

Pseudo:

```python
if (
    damage_pool_enabled
    and damage_pool_interval > 0
    and damage_pool_fraction > 0
    and (damage_prob > 0 or knockout_vocabulary is not None)
    and epoch % damage_pool_interval == 0
):
    rng, damage_key = jax.random.split(rng)
    circuit_pool, damaged_idxs = circuit_pool.damage_fraction(
        key=damage_key,
        fraction=damage_pool_fraction,
        layer_sizes=layer_sizes,
        damage_prob=damage_prob,  # from pool.damage_prob
        selection_strategy=damage_strategy,
        combined_weights=damage_combined_weights,
        knockout_vocabulary=knockout_vocabulary,
    )
```

### Reused Components [implemented]

- `structural_perturbation.py`:
  - `create_reproducible_knockout_pattern(...)` for fresh KO masks.
  - `create_knockout_vocabulary(...)` when a shared vocabulary is configured.
- `pool.py`:
  - `get_reset_indices(...)` for selection policies (uniform/steps_biased/loss_biased/combined).
- `self_attention.py`:
  - KO masks clamp logits to large negative and zero-out updates each forward. No graph/logit mutation required at damage time.

### Behavior Guarantees [implemented]

- KO state is carried in `GraphPool.knockout_patterns` only; pool graphs/logits are not rewritten.
- When a damaged entry is sampled for train/eval, the model applies isolation and clamping; knocked-out nodes remain non-updating thereafter.
- Damage events are independent of resets; both can coexist with separate intervals.

### Notes

- Keep all mutations post-step to preserve step input consistency and match reset placement.
- JIT-wrap new pool methods; avoid Python loops (`vmap` for pattern generation).
- Evaluation upgrades will consume pool state; KO masks are applied whenever entries are evaluated.

---

### Next Steps

- Event-driven damage evaluation
  - Immediately after `damage_fraction(...)` returns `damaged_idxs`, sample those entries’ `wires/logits/knockout_patterns`, run `evaluate_circuits_in_chunks` for N inner steps, and log under `eval_damage_pool/*` (loss, hard_loss, accuracy, hard_accuracy).
  - No pool mutation; uses the true damaged pool state.
  - this should eventually replace current version of run_knockout_periodic_evaluation. Note: Since eval will become event driven (ie on patterns sampled from knockout vocabulary), it will correspond to in-distribution eval mode within run_knockout_periodic_evaluation. We have to consider how to also allow for an OOD even-driven eval, perhaps sampling damage patterns randomly (no vocab). Not top priority for now, as in-distribution is first goal.

- Selection semantics for damage
  - Implement a damage-specific lowest-loss policy: either a new `loss_lowest` strategy or an inversion flag when `selection_strategy == "loss_biased"` in damage mode. When in doubt, favour minimal code changes / simplest implementation.

- Later: delta-style evaluation
  - Baseline delta: evaluate the same entries undamaged vs damaged after recovery steps to compute accuracy drop and recovery.
  - Do-nothing delta: measure metrics immediately after damage (0-step recovery) as a lower bound.
