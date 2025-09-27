## Minimal Perturbation GUI (Simplified)

### Required imports ✅ COMPLETED

- Simple Damage (reset + pre-bake logits):

```python
from boolean_nca_cc.training.pool.structural_perturbation import (
    create_greedy_subset_random_pattern, DEFAULT_GREEDY_ORDERED_INDICES,
)
from boolean_nca_cc.circuits.train import create_gate_mask_from_knockout_pattern
import jax
import jax.numpy as jp
import numpy as np
```

- Hybrid (reset + generator hook) and Training‑faithful (hook without resets):

```python
from boolean_nca_cc.training.pool.structural_perturbation import (
    create_greedy_subset_random_pattern, DEFAULT_GREEDY_ORDERED_INDICES,
)
from boolean_nca_cc.training.evaluation import evaluate_model_stepwise_generator
import jax
import jax.numpy as jp
import numpy as np
```

### Goal
Add a second perturbation type ("GAMMA RAYS") to `GUI.py` that feels like wire shuffle in UX and code flow, without changing the single‑circuit generator or evaluation path.

### Core idea
Reuse the existing "reset → reinit generator" flow. For damage, "bake" the knockout directly into the current logits before reinitializing the generator (no mask/plumbing changes).

### One-button flow ✅ PARTIALLY COMPLETED
- Dropdown: `perturbation_types = ["Wire Shuffle", "GAMMA RAYS"]` ✅ COMPLETED
- Damage baking path (`_apply_gate_damage_perturbation`) ✅ COMPLETED
  - Resets logs and restores `logits0` ✅ COMPLETED
  - Samples pattern and builds masks ✅ COMPLETED
  - Applies bias into damaged logits ✅ COMPLETED
  - Reinitializes generator and activations ✅ COMPLETED
  - Reads `damage_prob`, `greedy_ordered_indices`, and `damage_seed` from config ✅ COMPLETED
- Button: "PERTURB" ⏳ PENDING
  - Wire Shuffle: call existing mutation (unchanged)
  - GAMMA RAYS: call `_apply_gate_damage_perturbation` (pending one-button hookup)

### Implementation sketch (GUI-only)
```python
from boolean_nca_cc.training.pool.structural_perturbation import (
    create_greedy_subset_random_pattern, DEFAULT_GREEDY_ORDERED_INDICES,
)
from boolean_nca_cc.circuits.train import create_gate_mask_from_knockout_pattern

def _apply_gate_damage_perturbation(self, damage_prob: int = 8, bias: float = -5.0):
    # 1) Reset logs and current logits like wire shuffle
    self.step_i = 0
    self.loss_log[:] = 0; self.hard_log[:] = 0
    self.accuracy_log[:] = 0; self.hard_accuracy_log[:] = 0
    self.logits = self.logits0  # do NOT mutate logits0

    # 2) Sample a flat knockout pattern (seen-like, minimal)
    key = jax.random.PRNGKey(np.random.randint(0, 1_000_000))
    pattern = create_greedy_subset_random_pattern(
        key, self.layer_sizes, int(damage_prob), DEFAULT_GREEDY_ORDERED_INDICES
    )

    # 3) Build per-layer masks for viz and for shaping logits
    layer_gate_masks = create_gate_mask_from_knockout_pattern(self.layer_sizes, pattern)
    self.gate_mask = [m.astype(np.float32) for m in layer_gate_masks]  # for draw_circuit()

    # 4) Apply damage into logits at masked gates
    damaged_logits = [l.copy() for l in self.logits]
    for li in range(1, len(self.layer_sizes) - 1):  # skip input(0) and output(-1)
        gate_n, group_size = self.layer_sizes[li]
        group_n = gate_n // group_size
        mask = np.array(layer_gate_masks[li]).reshape(group_n, group_size)  # True = KO
        damaged_logits[li - 1] = np.where(mask[..., None], bias, damaged_logits[li - 1])
    self.logits = damaged_logits

    # 5) Reinit generator identically to wire shuffle path
    self.model_generator = None
    self.last_step_result = None
    self.initialize_optimization_method()
    self.initialize_activations()
```

Notes:
- Keep the same reset signatures and ordering as wire shuffle to avoid state drift.
- Use a strong negative bias (e.g., -5.0) if you want damaged gates to be effectively "off"; use 0.0 to make them uncertain (≈0.5).
- Do not modify `self.logits0`. Always derive a damaged copy (`self.logits`).

Reversible bias equivalence (GUI context):
- A strong negative logit yields `sigmoid(logit) ≈ 0`, making LUT outputs effectively 0 at damaged gates. This approximates the outcome of a large negative reversible bias (e.g., -10.0) applied at the model level.
- The simple GUI path “bakes” this into logits before step 1. It captures the immediate effect at injection time, but it does not perform attention masking or the exact one‑shot timing used inside the model.

### Pitfalls (and how this sketch avoids them)
- Layer indexing: `self.layer_sizes` includes inputs; `self.logits` starts at the first hidden layer. Hence `damaged_logits[li - 1]` for `li ≥ 1`.
- Mask shape: per-layer mask is flat over gates; reshape to `[group_n, group_size]` before broadcasting over LUT dimension.
- Inputs/outputs: pattern creation already excludes input/output; we still skip `li = 0` and `li = last` for safety.
- Model sees the damage: because logits are pre-damaged before generator init, the frozen model optimizes from that state (no need to pass a pattern).

### Why no generator/eval change is needed here
The live demo is allowed to be simpler than batched eval. Batched eval passes a pattern to the model to implement permanent/reversible behavior inside attention. For the GUI, we simulate damage by directly altering logits and then reuse the existing generator unchanged.

### Hidden channels vs logits (GUI context)
- Training/eval `CircuitSelfAttention` maintains hidden node features in the graph and can apply reversible biases at the right step using the knockout pattern.
- In the GUI, we do not pass a pattern to the model; we only alter logits before reinit. Hidden features are rederived by the model from the graph at step 0 and then updated normally. We do not explicitly mask hidden channels; the "damage" is expressed through the starting logits.
- This is acceptable for a minimal live visualization and preserves the current GUI architecture. If we need reversible semantics and true hidden-state perturbations later, we can add a pattern hook to the single‑circuit generator.

---

## Optional: Generator hook modes

Add a minimal hook to the single‑circuit generator to pass a pattern into the model at step 1 (single‑shot). This enables two advanced modes beyond the default:
- Hybrid (Reset + Hook): stability of resets with model‑accurate injection.
- Training‑faithful (Hook, no resets during trajectory): matches training/batched‑eval semantics more closely.

### Generator API change (single‑circuit)
Extend `evaluate_model_stepwise_generator` to accept an optional pattern and behavior:

```python
def evaluate_model_stepwise_generator(
    model: CircuitGNN | CircuitSelfAttention,
    wires: List[jp.ndarray],
    logits: List[jp.ndarray],
    x_data: jp.ndarray,
    y_data: jp.ndarray,
    input_n: int,
    arity: int = 2,
    circuit_hidden_dim: int = 16,
    max_steps: int = None,
    loss_type: str = "l4",
    bidirectional_edges: bool = True,
    layer_sizes: List[Tuple[int, int]] = None,
    # New (optional)
    pending_knockout_pattern: Optional[jp.ndarray] = None,
    damage_behavior: str = "reversible",  # or "permanent"
):
    ...
```

Injection logic (high‑level):
- Build graph and yield step 0 (baseline) as today.
- At step 1, call the model with `knockout_pattern=pending_knockout_pattern`:
  ```python
  updated_graph = model(
      graph,
      knockout_pattern=pending_knockout_pattern,
      layer_neighbors=False,
      layer_sizes=layer_sizes,
  )
  ```
- For reversible behavior, zero the step counter in `graph.globals[..., 1]` on the injection step (aligns with batched eval’s reversible logic) before updating it.
- After step 1, continue without re‑injecting (single‑shot). If `pending_knockout_pattern is None`, behave exactly as today.

This mirrors the batched path (which already supports per‑step patterns and reversible behavior), but for the single‑circuit generator.

### GUI integration modes

1) Hybrid (Reset + Hook)
- Goal: show an immediate accuracy drop and realistic recovery while keeping GUI state stable.
- Behavior: clear GUI logs/state like the simple path, then pass the pattern to the generator so the model applies masking/one‑shot bias at step 1.

```python
self.model_generator = evaluate_model_stepwise_generator(
    model=self.frozen_model,
    wires=self.wires,
    logits=self.logits0,   # fresh logits
    x_data=self.input_x,
    y_data=self.y0,
    input_n=self.input_n,
    arity=self.arity,
    circuit_hidden_dim=hidden_dim_for_graph,
    max_steps=None,
    loss_type=self.loss_type,
    bidirectional_edges=True,
    layer_sizes=self.layer_sizes,
    pending_knockout_pattern=pattern,
    damage_behavior="reversible",
)
```

Notes:
- Resets avoid long‑horizon drift in the GUI. The model still receives the pattern for masking/one‑shot bias.
- Prefer this when you want a stable demo with more faithful damage semantics than the simple path.

2) Training‑faithful (Hook, no resets during trajectory)
- Goal: replicate training/eval step semantics (attention masking and one‑shot reversible bias timing) across a continuous trajectory.
- Behavior: do not bake logits and do not clear/reset mid‑trajectory; pass the pattern once and let the inner loop evolve.

```python
# No pre-baked damage, no GUI-level resets between steps
self.model_generator = evaluate_model_stepwise_generator(
    model=self.frozen_model,
    wires=self.wires,
    logits=self.logits,  # continue current logits/trajectory
    x_data=self.input_x,
    y_data=self.y0,
    input_n=self.input_n,
    arity=self.arity,
    circuit_hidden_dim=hidden_dim_for_graph,
    max_steps=None,
    loss_type=self.loss_type,
    bidirectional_edges=True,
    layer_sizes=self.layer_sizes,
    pending_knockout_pattern=pattern,
    damage_behavior="reversible",
)
```

### Mode toggle (recommended)
- Simple Damage (default): bake into logits before generator init (no hook). Stable and minimal; approximates reversible effects by driving LUT outputs to ~0 via strong negative logits.
- Hybrid: Reset + generator hook (pattern passed to model at step 1). Stable with more faithful semantics (masking, one‑shot bias).
- Training‑faithful: Generator hook without resets across the trajectory (closest to training/batched‑eval behavior; less stable for very long runs).

This lets us keep the minimal path while enabling perfect semantic fidelity when desired.
