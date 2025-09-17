## Sequential SEU Evaluation (In-loop Scheduling)

This document specifies Option B: performing sequential SEU damage inside the evaluation loop, to minimize compute overhead by keeping graphs resident and reusing compiled primitives.

### Goal

Evaluate “sequential recovery” trajectories where a greedy list of gates is damaged one-by-one, with a fixed recovery window between damages, without rebuilding graphs or reinitializing state between segments:

- Step 0: baseline on base circuits (no damage)
- Apply damage to gate g0, recover for R steps
- Apply damage to gate g1, recover for R steps
- … until K gates applied

This differs from current “single-shot recovery” (apply a fixed SEU mask at t=0 and recover for N steps). We must inject new SEU flips mid-trajectory.

### Current Architecture Reference

- `boolean_nca_cc/training/evaluation.py`
  - `evaluate_model_stepwise_generator` and `evaluate_model_stepwise`: single-circuit step tracing. Baseline step 0 is yielded, then repeated application of `model(graph)` and loss computation per step.
  - `evaluate_model_stepwise_batched` → `_evaluate_with_loop`: batch evaluation that builds graphs once and loops steps. Per-step: apply model (optionally with `knockout_pattern`), extract logits, compute loss/acc, update output-node loss feature and `globals`.
- `boolean_nca_cc/training/pool/perturbation.py`
  - Greedy ordering `DEFAULT_GREEDY_ORDERED_INDICES`.
  - `sample_seu_gates` and `build_flip_masks_from_indices`: construct dense boolean masks aligned to per-layer logits. For SEU, masks are aligned to logits-bearing layers `layer_sizes[1:]`.
  - `flip_logits_with_masks`: apply invert flips to logits.
- `boolean_nca_cc/training/train_loop.py`
  - `run_seu_periodic_evaluation`: builds/chooses SEU masks, applies them to initial logits, then calls `evaluate_model_stepwise_batched` (optionally chunked) to trace recovery. SEU is applied up-front, not mid-trajectory.

### Design Overview

Add an in-loop SEU scheduling mechanism to `boolean_nca_cc/training/evaluation.py` so we can inject new logit flips at specified step indices while the loop runs. The mechanism should:

- Allow one continuous trajectory: single graph build, single loop
- Mutate logits at configured steps before model application
- Preserve existing functionality and signatures by adding optional arguments
- Reuse `boolean_nca_cc/training/pool/perturbation.py` for mask generation and keep mask application centralized via `flip_logits_with_masks`

### API Changes

1) `boolean_nca_cc/training/evaluation.py`

- Add an optional schedule argument to both public batched API and internal loop:

```python
def evaluate_model_stepwise_batched(
    model: CircuitSelfAttention,
    batch_wires: List[jp.ndarray],
    batch_logits: List[jp.ndarray],
    ...,
    seu_schedule: Optional[Dict] = None,  # new
    ...,
) -> Dict:
    ...

def _evaluate_with_loop(
    model,
    batch_graphs: jraph.GraphsTuple,
    batch_wires: List[jp.ndarray],
    ...,
    seu_schedule: Optional[Dict],  # new
    ...,
) -> Dict:
    ...
```

- Schedule structure (global or per-batch):

```python
# Option A: global schedule applied to all graphs
seu_schedule = {
  "events": [  # applied before model at the given step index
    {"step": 1,  "masks": List[jp.ndarray]},  # per-layer masks, logits-aligned
    {"step": 6,  "masks": List[jp.ndarray]},
    ...
  ]
}

# Option B: per-graph schedule
seu_schedule = {
  "per_graph": True,
  "events": [  # length = batch_size
    [ {"step": 1,  "masks": List[jp.ndarray]}, {"step": 6, ...}, ... ],  # for graph 0
    [ {"step": 1,  "masks": List[jp.ndarray]}, ... ],                      # for graph 1
    ...
  ]
}
```

2) `boolean_nca_cc/utils` (new small helper)

- Add `inject_logits_into_graph(graph, logits_per_layer)` that mirrors `extract_logits_from_graph` but writes a new `nodes["logits"]` tensor built from the provided per-layer tensors. This helper must:
  - Respect the packing convention used by `build_graph` for `nodes["logits"]`
  - Return a new `GraphsTuple` via `_replace(nodes=updated_nodes)`

This isolates the writeback logic and keeps `boolean_nca_cc/training/evaluation.py` clean.

### Loop Integration

In `boolean_nca_cc/training/evaluation.py::_evaluate_with_loop`:

1) Maintain a step counter identical to current logic (1..n_message_steps). Baseline step 0 remains unchanged and is already logged.
2) At the top of each loop iteration, before calling `model(...)`, check if the schedule has an event for `step`.
3) If an event exists:
   - Extract current logits via existing `extract_logits_from_graph(updated_graphs, logits_original_shapes)` using a `vmap` across batch.
   - Apply SEU flips: for each graph that has an event at this step, apply `flip_logits_with_masks` with that event’s `masks`; otherwise keep logits unchanged.
   - Write the mutated logits back into `updated_graphs` via `inject_logits_into_graph` (batched with `vmap`).
4) Proceed with the current step: apply `model(...)`, compute loss/accuracy, update output-node loss and globals, record metrics, advance.

Placement of intervention: “pre-model” is consistent with semantics “damage, then recover for R steps”. Events at steps [1, 1+R, 1+2R, ...] produce the required cadence.

### Mask Generation and Alignment

- Sequential scenario is driven by a greedy list of gate indices. For each single-gate event:
  - Use `build_flip_masks_from_indices` with:
    - `layer_sizes_seq = layer_sizes[1:]` to align with logits-bearing layers
    - `selected_gate_indices` containing exactly one global gate offset within `layer_sizes_seq`
    - `flips_per_gate` and `arity` from config
  - `flip_logits_with_masks` expects per-layer masks matching logits shapes `(group_n, group_size, 2**arity)`
- For batch: either apply the same event to all graphs (global schedule), or construct per-graph masks depending on the chosen sequence per sample.

### JAX Considerations

- Keep the loop device-resident. All schedule data should be JAX arrays or pytrees; avoid Python branching per-item where possible. Recommended:
  - Normalize schedule into per-step boolean selectors and mask payloads so per-step intervention is a pure `where`/`select` operation under `vmap`.
  - For global schedule: build a dict mapping `step -> masks` in host, then pass masks as a list indexed by step (static Python list of JAX arrays is acceptable if length is small and fixed). Alternatively, prepack into a dense structure `[n_steps, ...]` with zeros for non-event steps to allow simple selection without host checks.
- Maintain compatibility with the existing optional `knockout_patterns` argument. SEU flips (logit-level) are orthogonal to knockouts (topology-level). Apply SEU pre-model; then call `model(graph, knockout_pattern=...)` as done today.
- Keep pure functional updates: no in-place mutation; use `_replace` on `GraphsTuple`.

### Backward Compatibility

- New arguments are optional and default to `None`; existing callers unchanged.
- If `seu_schedule is None`, the loop behaves exactly as today.

### Data Collected

- Stepwise metrics already recorded remain unchanged.
- No changes required to per-pattern outputs; sequential application is reflected naturally in the trajectory.

### Implementation Steps

1) Add `inject_logits_into_graph` in `boolean_nca_cc/utils` (where `extract_logits_from_graph` lives) and expose it in the package `__init__` if needed. (Completed: implemented in `boolean_nca_cc/utils/extraction.py` and exported via `boolean_nca_cc/utils/__init__.py`)
2) Extend `evaluate_model_stepwise_batched` and `_evaluate_with_loop` signatures to accept `seu_schedule` and thread it through. (Completed: added `seu_schedule` parameter and global schedule support with pre-model injection)
3) In `boolean_nca_cc/training/evaluation.py::_evaluate_with_loop`:
   - Normalize schedule (global or per-graph) pre-loop into a form that is indexable per step
   - At each step, conditionally apply flips before the model call using `vmap_extract_logits` → mask application → `vmap_inject_logits`. (Completed: implemented global schedule normalization and pre-model injection with mask merging)
4) Add a small builder in `boolean_nca_cc/training/train_loop.py` to translate a greedy list and recovery window R into a schedule: (Completed: implemented `build_sequential_seu_schedule` function with comprehensive docstring and tested successfully)

```python
def build_sequential_seu_schedule(ordered_indices, layer_sizes, flips_per_gate, arity, recovery_steps):
    # steps: 1, 1+R, 1+2R, ...
    events = []
    gate_layer_sizes = layer_sizes[1:]
    for j, gate_idx in enumerate(ordered_indices):
        masks = build_flip_masks_from_indices(
            layer_sizes=gate_layer_sizes,
            selected_gate_indices=jp.array([gate_idx], dtype=jp.int32),
            flips_per_gate=flips_per_gate,
            arity=arity,
            key=jax.random.PRNGKey(0),  # deterministic in greedy case
        )
        events.append({"step": 1 + j * recovery_steps, "masks": masks})
    return {"events": events}
```

5) Wire a new mode in `boolean_nca_cc/training/train_loop.py::run_seu_periodic_evaluation` (optional) to trigger sequential evaluation by constructing the schedule and calling `evaluate_model_stepwise_batched(..., seu_schedule=schedule)` with `batch_logits=base_logits` (no initial flips). (Completed: implemented sequential mode with config integration)

6) **TODO: Add control knob for gates per sequential trigger** - Currently sequential mode flips exactly one gate per trigger (from `ordered_indices`). Need to add parameter to control how many gates to flip per sequential event, potentially reusing `gates_per_circuit` or adding `sequential_gates_per_trigger` parameter.

### Configuration Integration

**Parameters:**
- `eval.seu.sequential: bool` - Enable sequential SEU mode (default: false)
- `eval.seu.recovery_steps: int` - Recovery steps between damages (default: 5)
- `eval.periodic_eval_inner_steps: int` - **Total evaluation length** (controls max sequential damages)

**Parameter Relationship:**
- Sequential events occur at steps: 1, 1+R, 1+2R, ... (where R = recovery_steps)
- Max sequential damages = min(len(greedy_indices), floor((periodic_eval_inner_steps-1)/recovery_steps) + 1)
- Example: `periodic_eval_inner_steps=20`, `recovery_steps=5` → events at [1, 6, 11, 16]

**Usage:**
```bash
# Enable via command line
python train.py eval.seu.sequential=true eval.seu.recovery_steps=3

# Or modify config.yaml
eval:
  seu:
    sequential: true
    recovery_steps: 3
```

### Edge Cases

- Multiple events at same step: merge masks by logical OR per-layer before application.
- Events beyond `n_message_steps`: ignored.
- R=0: back-to-back damages at consecutive steps.
- Batched per-graph heterogeneous schedules: ensure `events` length matches `batch_size` when `per_graph=True`.

### Testing

- Unit: round-trip `extract_logits_from_graph` → `inject_logits_into_graph` on synthetic graphs; ensure logits unchanged.
- Unit: single-event schedule reproduces current single-shot behavior when event at `step=1` and initial logits are unmodified.
- Integration: sequential greedy 20-gate case with R=5 on a small circuit; verify stepwise hard accuracy drops and recovery windows align with event steps.

### Performance

- Single graph build, single loop; no re-packing across segments.
- Mask application is cheap (elementwise multiply via `flip_logits_with_masks`).
- Compatible with future `lax.scan` conversion of the loop; schedule can be indexed by `step` inside the scan body.


