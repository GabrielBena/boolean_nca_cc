# SPEC.md — boolean-NCA web port

This document is the **contract** the TypeScript port must satisfy. It is authoritative: TS modules, fixtures, tests, and commits reference sections by stable ID. When discrepancies emerge between TS behavior and Python, **edit this file first** (see plan §9 — Spec change protocol).

Each section provides:
- **Signature** — input/output shapes & dtypes
- **Algorithm** — equation or pseudocode
- **Invariants** — must-hold properties
- **Fixtures** — test IDs that exercise this section
- **Tolerance** — parity threshold vs Python reference
- **jax-js op deps** — primitives the implementation requires

The Python reference is treated as the ground truth on numerical behavior; any TS port that disagrees is wrong unless this spec has been updated to declare otherwise.

---

## §CIRCUIT-FORWARD

**Source of truth:** `boolean_nca_cc/circuits/model.py`, functions `run_layer` (L46) and `run_circuit` (L163).

### Signature

```
run_circuit(x, wires, logits, hard=false, gate_masks=null) -> activations[]
```

| Parameter | Shape | Dtype | Notes |
|---|---|---|---|
| `x` | `(case_n, input_n)` | f32 | Input bits (already unpacked, values in [0,1]) |
| `wires[ℓ]` | `(arity, groups)` | i32 | Wire indices from layer ℓ-1 outputs |
| `logits[ℓ]` | `(groups, group_size, 2^arity)` | f32 | Per-gate LUT logits |
| `gate_masks[ℓ]` | `(groups * group_size,)` | f32 | Optional; 1.0=active, 0.0=knocked |
| `hard` | bool | — | If true, round sigmoid to {0,1} |

For the default config (`input_n=8, output_n=8, arity=4, layer_n=3, width_factor=2`), layer logit shapes are:
```
[(64, 4, 16), (64, 4, 16), (32, 2, 16), (8, 1, 16)]
```
and `case_n = 2^input_n = 256`.

### Algorithm

For each layer ℓ:

```
1. gathered = x[..., wires[ℓ]]                          # shape (case_n, arity, groups)
2. luts = sigmoid(logits[ℓ])  if not hard else round(sigmoid(logits[ℓ]))
3. For each input axis i in 0..arity-1:
     xᵢ = gathered along axis i, expanded as (..., 1, 1)
     luts = (1 - xᵢ) * luts[..., ::2] + xᵢ * luts[..., 1::2]
   # After arity halvings, last dim is 1.
4. y = reshape(luts, (case_n, groups * group_size))
5. If gate_masks[ℓ] is provided: y = y * gate_masks[ℓ]    # multiplicative, AFTER sigmoid
6. x ← y, continue to next layer.
```

The 2^arity = 16 LUT entries encode a binary decision tree over the `arity` gathered inputs; the iterative `(1-x)·even + x·odd` reduction collapses one axis per input.

### Invariants

- **`§CIRCUIT-FORWARD-I1`** — Output bounded: every layer activation ∈ `[0, 1]`.
- **`§CIRCUIT-FORWARD-I2`** — Gradient finite for finite logits: `∂y/∂logits` is finite when logits are finite.
- **`§CIRCUIT-FORWARD-I3`** — Determinism: same `(x, wires, logits)` ⇒ same output (no randomness in the forward pass).
- **`§CIRCUIT-FORWARD-I4`** — Damage idempotence: applying `gate_masks=1` is identical to passing no mask.

### Fixtures

| Test ID | Description |
|---|---|
| `§CIRCUIT-FORWARD-T001` | Default circuit, fixed seed, 256 cases — full activation list per layer |
| `§CIRCUIT-FORWARD-T002` | Hard mode (`hard=true`) on the same inputs |
| `§CIRCUIT-FORWARD-T003` | With a gate_mask zeroing two interior gates |
| `§CIRCUIT-FORWARD-T004` | Random extreme logits (±50) — invariant I1 (bounded output) |

### Tolerance

Soft outputs: `|Δ| < 1e-4` element-wise vs Python `run_circuit` (float32). Hard outputs: exact match.

### jax-js op deps

`sigmoid`, `take` / fancy indexing along axis, slicing with stride 2 (`::2`, `1::2`), elementwise add/mul, reshape.

---

## §LOSS-L4

**Source of truth:** `boolean_nca_cc/training/evaluation.py:get_loss_from_wires_logits` (L49) with `loss_type="l4"`.

### Signature

```
get_loss(wires, logits, x, y, hard_outputs=false, loss_type="l4", power=4)
  -> (loss: scalar f32, aux: tuple)
```

`aux = (hard_loss, pred, pred_hard, accuracy, hard_accuracy, full_map_accuracy, residuals_soft, residuals_hard)`.

### Algorithm

```
1. activations = run_circuit(x, wires, logits, hard=false)
2. pred       = activations[-1]                     # shape (case_n, output_n)
3. pred_hard  = round(activations_hard[-1])          # second pass with hard=true
4. residuals_soft = pred - y
5. residuals_hard = pred_hard - y
6. loss      = mean(|residuals_soft|^power)          # default power=4
7. hard_loss = mean(|residuals_hard|^power)
8. accuracy           = mean(round(pred) == y)
9. hard_accuracy      = mean(pred_hard == y)
10. full_map_accuracy = mean( all(round(pred) == y, axis=-1) )    # stricter: every output bit correct
```

### Invariants

- **`§LOSS-L4-I1`** — Loss is non-negative.
- **`§LOSS-L4-I2`** — Loss = 0 iff `pred == y` exactly.
- **`§LOSS-L4-I3`** — `full_map_accuracy ≤ accuracy` (every-bit-correct is stricter than mean-bits-correct).
- **`§LOSS-L4-I4`** — `accuracy ∈ [0, 1]`.
- **`§LOSS-L4-I5`** — Gradient finite for finite logits and finite `y ∈ [0,1]`.

### Fixtures

| Test ID | Description |
|---|---|
| `§LOSS-L4-T001` | xor task, default circuit, fixed seed — verify scalar loss + all aux fields |
| `§LOSS-L4-T002` | Random `y` with NaN-free logits — invariant checks I1, I3, I4 |

### Tolerance

`|Δloss| < 1e-5`; aux f32 fields `|Δ| < 1e-4`; integer/bool fields exact.

### jax-js op deps

All `§CIRCUIT-FORWARD` deps plus: `pow`, `mean`, comparison (`==`), `round`, reduction `all` along an axis.

---

## §LOSS-BCE

**Source of truth:** same file with `loss_type="bce"`.

### Algorithm

```
loss = mean( sigmoid_binary_cross_entropy(logits=logit(pred), labels=y) )
     = mean( y * softplus(-z) + (1-y) * softplus(z) )    where z = logit(pred)
```

Aux fields identical to `§LOSS-L4`.

### Invariants

- **`§LOSS-BCE-I1`** — Loss ≥ 0.
- **`§LOSS-BCE-I2`** — Numerically stable for `pred ∈ [0,1]` (softplus form avoids overflow at extreme logits).

### Fixtures

| Test ID | Description |
|---|---|
| `§LOSS-BCE-T001` | xor task, fixed seed — scalar loss + aux |

### Tolerance

`|Δloss| < 1e-5`.

### jax-js op deps

All `§LOSS-L4` deps plus `softplus` (or stable `log1p(exp(-|z|)) + max(z,0)`).

---

## §ADAMW

**Source of truth:** `optax.adamw` (`lr=1.0, b1=0.8, b2=0.8, weight_decay=0.1, eps=1e-8`) — initialized in `GUI_minimal.py:_reset_backprop_optimizer` (L852). Loshchilov & Hutter decoupled weight decay.

### Signature

```
init_state(params) -> {m: zeros, v: zeros, t: 0}
update(grads, state, params) -> (updates, new_state)
apply(params, updates) -> new_params
```

`params` is a list of f32 tensors matching the per-layer logits shapes from `§CIRCUIT-FORWARD`.

### Algorithm (per parameter tensor)

```
t       ← t + 1
m_t     ← β₁·m_{t-1} + (1-β₁)·g_t
v_t     ← β₂·v_{t-1} + (1-β₂)·g_t²
m̂_t     ← m_t / (1 - β₁^t)
v̂_t     ← v_t / (1 - β₂^t)
update  ← -α·( m̂_t / (√v̂_t + ε) ) - α·λ·θ_{t-1}
θ_t     ← θ_{t-1} + update
```

with `α=lr=1.0, β₁=β₂=0.8, λ=wd=0.1, ε=1e-8`.

### Invariants

- **`§ADAMW-I1`** — At `t=1` with `β₁=β₂=0.8`, bias-corrected `m̂` equals `g` exactly.
- **`§ADAMW-I2`** — Determinism: same `(grads, state)` ⇒ same `(updates, new_state)`.
- **`§ADAMW-I3`** — Convergence: on a quadratic with finite optimum, the iterates remain bounded.

### Fixtures

| Test ID | Description |
|---|---|
| `§ADAMW-T001` | One step on a fixed `(params, grads)` tuple — bit-identical to optax (within f32) |
| `§ADAMW-T002` | 100 steps on a synthetic quadratic — assert decreasing loss |

### Implementation note

**Try `@jax-js/optax` first.** A subset of optax has been ported (per [jax-js README](https://github.com/ekzhang/jax-js)). If it exposes `adamw(lr, b1, b2, weight_decay, eps)`, use it. Otherwise hand-roll the ~30 lines above.

### Tolerance

`|Δ| < 1e-5` per step vs Python reference.

### jax-js op deps

`grad`, elementwise add/mul/sub/div, sqrt, scalar broadcast.

---

## §DAMAGE

**Source of truth:** `GUI_minimal.py:_apply_gate_damage_perturbation` (L1398), `boolean_nca_cc/circuits/train.py:create_gate_mask_from_knockout_pattern` (~L174–222), `boolean_nca_cc/training/pool/structural_perturbation.py:create_greedy_subset_random_pattern` (~L204–222).

### Algorithm

**1. Sample knockout pattern** (`create_greedy_subset_random_pattern`):

```
greedy = DEFAULT_GREEDY_ORDERED_INDICES   # pre-ranked vulnerable nodes; never includes input/output
n_knock = min(damage_prob, len(greedy))
indices = random_choice(seed, greedy, size=n_knock, replace=False)
pattern = zeros(total_nodes, dtype=bool)
pattern[indices] = True                   # True = knocked
return pattern
```

**2. Build per-layer gate masks** (`create_gate_mask_from_knockout_pattern`):

```
For each layer ℓ with n_gates_ℓ nodes:
  layer_mask[ℓ] = where(pattern[offset:offset+n_gates_ℓ], 0.0, 1.0)
  offset += n_gates_ℓ
gate_masks: list of f32 vectors, one per layer, shape (groups[ℓ] * group_size[ℓ],)
```

**3. Apply** during forward (already in `§CIRCUIT-FORWARD` step 5): multiplicatively, AFTER sigmoid.

### Invariants

- **`§DAMAGE-I1`** — Input and output layer gates are never knocked (`DEFAULT_GREEDY_ORDERED_INDICES` excludes them).
- **`§DAMAGE-I2`** — `damage_prob=0` ⇒ all-1 masks ⇒ identical forward pass to undamaged circuit.
- **`§DAMAGE-I3`** — Reversibility: storing `gate_masks=null` after the perturbation restores the undamaged forward pass *for permanent mode* (the logits themselves are not modified by the mask path).

### Fixtures

| Test ID | Description |
|---|---|
| `§DAMAGE-T001` | Fixed seed, `damage_prob=5` — verify pattern + per-layer mask shapes/values |
| `§DAMAGE-T002` | `damage_prob=0` — invariant I2 (no-op) |

### Tolerance

Pattern and masks are exact (boolean / float exact match).

### jax-js op deps

`random.choice`-equivalent (fixed-seed shuffle + slice is acceptable), boolean indexing, `where`.

---

## §TRAINER-INTERFACE

The TS contract exposed to the UI. Both `BackpropTrainer` and (Phase 2) `SATrainer` implement it.

```ts
type Tensor = jaxjs.Tensor;          // jax-js array handle
type GateMask = Float32Array[];       // one per layer, see §DAMAGE

interface StepMetrics {
  step: number;
  loss: number;
  hardLoss: number;
  accuracy: number;
  hardAccuracy: number;
  fullMapAccuracy: number;
}

interface CircuitState {
  logits: Tensor[];                   // per-layer, see §CIRCUIT-FORWARD shapes
  wires: Int32Array[];                // per-layer (arity, groups), kept outside autodiff
}

interface Trainer {
  readonly mode: 'backprop' | 'sa';
  reset(opts: { seed: number; taskId: string }): void;
  step(): StepMetrics;
  getCircuitState(): CircuitState;
  applyDamage(mask: GateMask): void;
  loadWeights?(weights: WeightBundle): Promise<void>;   // SA only
}
```

### Invariants

- **`§TRAINER-INTERFACE-I1`** — `step()` is deterministic given `(seed, taskId)` and the sequence of prior calls.
- **`§TRAINER-INTERFACE-I2`** — `applyDamage(mask)` does **not** modify `logits` (mask is multiplicative at forward time).
- **`§TRAINER-INTERFACE-I3`** — `getCircuitState()` returns handles that survive subsequent `step()` calls (no mutation in place from the caller's perspective).

---

## §SA-FORWARD (Phase 2)

**Source of truth:** `boolean_nca_cc/models/self_attention.py:CircuitSelfAttention` (L158).

### Confirmed architecture: dense full-graph self-attention with masking

This is the dominant Phase-2 finding: **NOT jraph scatter/segment**. Every node attends to every other node, with a mask derived from the circuit's `senders/receivers` (and from `knockout_pattern` during damage runs). Standard dense matmul + softmax suffices — no graph-library polyfill needed.

### Architecture

```
Input features per node (concat, dim = 16 + 64 + 64 + 64 + 1 = 209):
  [logits (16), hidden (64), layer_pe (64), intra_layer_pe (64), loss (1)]

x ← feature_proj(features)                        # nnx.Linear: 209 → 128

Repeat 3 times (num_layers=3):
  # Self-attention sub-block (pre-norm)
  x ← x + MultiHeadAttention(LayerNorm(x), mask=attention_mask)
       # 4 heads × 32 dim each = 128, softmax over key axis
  # Feed-forward sub-block (pre-norm)
  x ← x + Sequential(LayerNorm, Linear(128→256), gelu, Linear(256→128))(x)

logit_delta  ← logit_proj(x)                      # nnx.Linear: 128 → 16
hidden_delta ← hidden_proj(x)                     # nnx.Linear: 128 → 64
```

`attention_mask` is a dense `(n_nodes, n_nodes)` boolean: `True` where edge exists in the circuit graph (sender→receiver) plus the diagonal (self-attention). For damage runs, knocked rows/cols are masked out.

### Hyperparameters

| Name | Value |
|---|---|
| `num_layers` | 3 |
| `num_heads` | 4 |
| `attention_dim` | 128 |
| `mlp_dim` | 256 |
| `circuit_hidden_dim` | 64 |
| Activation | GELU |
| Norm placement | Pre-norm |

### Weight export convention

`scripts/export_sa_weights.py` (Phase 2) flattens `nnx.state(model)` into a dict keyed by dotted path:

```
feature_proj.kernel          (209, 128)    f32
feature_proj.bias            (128,)        f32
blocks.0.attn.LayerNorm.scale       (128,)
blocks.0.attn.LayerNorm.bias        (128,)
blocks.0.attn.q_proj.kernel  (128, 128)    # or (128, 4, 32)
blocks.0.attn.k_proj.kernel  ...
blocks.0.attn.v_proj.kernel  ...
blocks.0.attn.out_proj.kernel ...
blocks.0.mlp.LayerNorm.scale ...
blocks.0.mlp.dense_0.kernel  (128, 256)
blocks.0.mlp.dense_1.kernel  (256, 128)
... (blocks 1, 2)
logit_proj.kernel            (128, 16)
hidden_proj.kernel           (128, 64)
```

Exact key paths to be confirmed by the export script in Phase 2 (read `nnx.state(model)` and dump the actual tree). Stored as **safetensors**; sidecar `sa_manifest.json` carries the hyperparams above.

### Invariants

- **`§SA-FORWARD-I1`** — Output deltas are finite for finite inputs.
- **`§SA-FORWARD-I2`** — Permutation equivariance over the node order is broken by `intra_layer_pe` and `layer_pe` (positional encodings), so node order must match the export-time order exactly.
- **`§SA-FORWARD-I3`** — With `knockout_pattern=null`, attention mask is the pure circuit-graph mask (no node deletions).

### Fixtures (Phase 2)

| Test ID | Description |
|---|---|
| `§SA-FORWARD-T001` | One forward pass on default circuit, fixed weights — match Python output deltas |
| `§SA-FORWARD-T002` | Same with `damage_prob=5` knockout pattern |

### Tolerance

`|Δlogit_delta| < 1e-3`, `|Δhidden_delta| < 1e-3`. Looser than circuit-forward because of accumulated float drift across 3 attention layers.

### jax-js op deps

dense matmul (einsum), `softmax` along key axis, `LayerNorm`, GELU, residual add. **No** scatter/segment/gather-add.

---

## §FIXTURES

Fixtures live in `web/tests/fixtures/<spec_id>_<case>.json`. Each file is a JSON object with this shape:

```json
{
  "spec_id": "§CIRCUIT-FORWARD-T001",
  "description": "Default circuit, fixed seed, 256 cases, full activations.",
  "tolerance": 1e-4,
  "inputs": {
    "x":      { "shape": [256, 8],   "dtype": "f32", "data_b64": "..." },
    "wires":  [ { "shape": [4, 16],  "dtype": "i32", "data_b64": "..." }, ... ],
    "logits": [ { "shape": [64,4,16],"dtype": "f32", "data_b64": "..." }, ... ]
  },
  "expected": {
    "activations": [ { "shape": [256, 64], "dtype": "f32", "data_b64": "..." }, ... ]
  }
}
```

`data_b64` is little-endian raw bytes, base64-encoded. The TS test loader is in `web/src/test_utils/fixture.ts` (Phase 1).

Fixtures are produced once by `scripts/export_fixtures.py` (Phase 1, before any TS porting). The `§FIXTURES` index here is updated as new fixtures are added.

### Index

| Test ID | File | Owner section |
|---|---|---|
| `§CIRCUIT-FORWARD-T001` | `circuit_forward_001.json` | `§CIRCUIT-FORWARD` |
| `§CIRCUIT-FORWARD-T002` | `circuit_forward_002.json` | `§CIRCUIT-FORWARD` |
| `§CIRCUIT-FORWARD-T003` | `circuit_forward_003.json` | `§CIRCUIT-FORWARD` |
| `§CIRCUIT-FORWARD-T004` | `circuit_forward_004.json` | `§CIRCUIT-FORWARD` |
| `§LOSS-L4-T001` | `loss_l4_001.json` | `§LOSS-L4` |
| `§LOSS-L4-T002` | `loss_l4_002.json` | `§LOSS-L4` |
| `§LOSS-BCE-T001` | `loss_bce_001.json` | `§LOSS-BCE` |
| `§ADAMW-T001` | `adamw_001.json` | `§ADAMW` |
| `§ADAMW-T002` | `adamw_002.json` | `§ADAMW` |
| `§DAMAGE-T001` | `damage_001.json` | `§DAMAGE` |
| `§DAMAGE-T002` | `damage_002.json` | `§DAMAGE` |
| `§SA-FORWARD-T001` | `sa_forward_001.json` | `§SA-FORWARD` (Phase 2) |
| `§SA-FORWARD-T002` | `sa_forward_002.json` | `§SA-FORWARD` (Phase 2) |
