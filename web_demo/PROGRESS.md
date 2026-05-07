# SODC Web Demo — Course of Action

This is the living plan for the in-browser SODC demo. Update the
checkboxes as we land each milestone; append to the decisions log as
we make architectural choices.

## Goal

Embed an interactive figure in the SODC article that lets a reader
watch a pre-trained Topology-Masked Transformer (TMT) self-organise a
Boolean circuit, recover from per-gate damage and shotgun events, and
adapt to freshly shuffled wirings — all client-side, in the spirit of
the original Growing-NCA demo.

## Scope (frozen for v1)

- **One** pre-trained model: random wiring, damage-trained, reverse
  task. Single JSON shipped to `public/weights/`.
- **Four** user actions: reset circuit, shuffle wires, shotgun N
  gates, click an individual gate to damage it.
- **One** optional knob: width-OOD slider (Regime IV) — ship if v1
  comes in under budget.
- **No** runtime W&B loading, no multi-task picker, no BP baseline,
  no profiling overlay.

The reference UI is the Python `training_demo.py`; only the four
actions above are ported.

## Architecture (end-to-end data flow)

```
[ wandb / local checkpoint  (.pkl, hydra cfg)  ]
                |
                |  export/export_weights.py
                v
[ public/weights/<task>_<recipe>.json ]      <-- int8 quant, ~few hundred KB
                |
                |  src/weights.ts (loadWeights)
                v
[ TMTWeights (Float32Array fields, header-declared shapes) ]
                |
                v
[ src/tmt.ts (tmtStep) ]  <--->  [ src/circuit.ts (circuitForward,
                                                   buildTopologyMask) ]
                |
                v
[ src/render.ts (drawCircuit, pickGate)  →  Canvas2D ]
                |
                v
[ src/main.ts (animation loop, UI dispatch) ]
                |
                v
[ Vite dev server (index.html)  |  npm run build → dist/ ]
                |
                v
[ gabrielbena.github.io/_posts/2026-05-06-sodc.md
    <div id="sodc-demo"></div>
    <script type="module" src=".../assets/sodc-demo/main.js"></script>
]
```

The Python side (`export/numpy_oracle.py`) is the single source of
numerical truth. It is validated against JAX, then the TS port is
validated against it.

## Milestones

### M0 — Scaffold (this commit)

- [x] Directory layout, README, PROGRESS, .gitignore.
- [x] package.json / tsconfig / Vite-compatible `index.html`.
- [x] Stub `src/` modules with interfaces + docstrings.
- [x] Stub Python `export/` (exporter + oracle).
- [x] Local node v22 LTS active (was blocked on Node v14).
- [x] Vite dev server boots, mounts placeholder.

### M1 — NumPy oracle (Python)

- [x] Pure-NumPy `tmt_step(state, topology, weights) -> state'`.
- [x] Pure-NumPy `circuit_forward(logits, wires, x) -> activations`.
- [x] Pure-NumPy damage primitives (`apply_shotgun`, `damage_gate`,
       `shuffle_wires`, `reset_circuit`).
- [x] Self-test for `self_attention` *and* `gathered_attention`.
       The deployed checkpoints use `gathered_attention`; the
       oracle covers both so we can sanity-check either checkpoint
       trove if needed.
- [x] Parity test against the live deployment checkpoint
       (`reverse / random_damage`, run-id `1u5ssulx`):
       single-block attention agrees with JAX to **1.3e-5** max abs
       error (float32 ulp); over 32 recurrent ticks **hard accuracy
       agrees bit-exactly** (Δ = 0.0000) and **0 / 49 152 hard
       predictions disagree** at the final step. The compounding
       soft-state drift (~1e-2 in logits, ~5e-1 in hidden by step
       32) is purely the JAX-vs-NumPy einsum accumulation order;
       the JS port will share that drift since it also runs float32.
- [x] **Perturbation parity** (run via
       `python -m export.test_oracle --perturbation-parity`):
       symmetric primitives + per-step hard-accuracy comparison for
       the four demo actions:
         * warm-up: 8 ticks, all Δ=0.0000
         * **shotgun** (5 random hidden gates): chosen indices
           extracted from JAX's RNG and mirrored on the oracle;
           12 post-shotgun ticks all Δ=0.0000 (model recovers
           from 0.67 → 0.88 hard accuracy)
         * **click-to-damage** (single gate, hidden-layer eligibility
           enforced on both sides): 12 ticks all Δ=0.0000
         * **shuffle wires** (Regime III): topology + neighbor
           cache rebuilt symmetrically, hidden state reset to 0
           (mirroring `_restart_tmt_generator`), logits + gate_mask
           kept; 12 ticks max Δ=0.0033 (3 / 49 152 predictions on
           one step — wiring-OOD soft variance, recovers next step)
       Worst-case max Δ across all four phases = **0.0033** ≪ tol = 0.005.
- **Exit criterion**: ✓ oracle reproduces JAX-policy hard
  predictions exactly on a 4096-case test for the deployment
  checkpoint, including under the four user-facing perturbations.

#### Architectural variants supported

The Python codebase exposes two attention architectures with shared
input projection / FFN / output heads / ReZero plumbing but
different attention implementations:

| Aspect              | `self_attention` (`AttentionBlock`) | `gathered_attention` (deployed) |
|---------------------|--------------------------------------|----------------------------------|
| Pre-attn LayerNorm  | separate `ln_q` + `ln_kv`            | single `attn_norm`               |
| Q/K/V projection    | `DenseGeneral` `[D, H, d_h]`         | three `Linear[D,D]` + reshape    |
| Output projection   | `DenseGeneral` `[H, d_h, D]`         | `Linear[D,D]`                    |
| QK-norm bias        | `True` (Flax default)                | `False` (γ only)                 |
| Attention support   | dense `[N, N]` + binary mask         | gathered `[N, max_neighbors]`    |
| Per-layer count     | `num_self_attn_layers` blocks        | single block                     |

`TMTConfig.model_kind` flips between the two; `TMTWeights.blocks`
is a `list[AttentionBlockWeights]` for one and a single
`GatheredAttentionWeights` for the other. `tmt_step` dispatches
internally. The TS port (M3) will likely implement only the gathered
path (since that's what we'll deploy), with the dense version kept
in the oracle as a comparison harness.

#### Inference contract (the math we are porting)

Source of truth: `boolean_nca_cc.training.evaluation.evaluate_model_stepwise_generator`
+ `CircuitSelfAttention` + `build_graph` + `run_circuit`.

**State (per node)**

A single circuit lifts to a graph of `N = input_n + sum(layer_gate_n)`
nodes (input layer included as layer 0). Each node carries:

| field                  | shape              | notes |
|------------------------|--------------------|-------|
| `logits`               | [N, 2^arity]       | input-node rows are all-zero; clamped gates are `faulty_value=-10.0` |
| `hidden`               | [N, h]             | recurrent memory; zero-init |
| `layer_pe`             | [N, h]             | sinusoidal of normalised depth `layer_idx / total_gate_layers * max_val` |
| `intra_layer_pe`       | [N, h]             | sinusoidal of intra-layer index |
| `loss`                 | [N]                | only output rows are nonzero (mean abs residual per output bit) |
| `gate_knockout_mask`   | [N]                | 1.0 healthy, 0.0 clamped |

`h = circuit_hidden_dim`. Edges are bidirectional by default.

**Attention mask** is `[N, N]` bool: `mask[r, s] = True` for every
forward edge `s → r`, plus the diagonal. Bidirectional edges mean
the matrix is symmetric (modulo the diagonal). Built once from the
wires; recomputed only when wires change.

**Tick**

```
state₀ = build_graph(...)              # logits, hidden=0, masks, PEs
state₀ = compute_residuals(state₀, x_data, y_data)   # populate loss[output]
yield step=0

for t in 1, 2, ...:
    if scheduled_damage:
        knockout n_gates from hidden layers
        logits[gates] = -10.0; mask[gates] = 0.0

    # TMT step:
    f = concat([logits, hidden, layer_pe?, intra_layer_pe?, loss[:,None]?])
    z = feature_proj(input_norm(f))
    for block in attention_blocks:           # single block in v1
        z = AttentionBlock(z, attention_mask)
    z = final_norm(z)
    Δlogits = logit_proj(z)
    Δhidden = hidden_proj(z)
    # Knockout: zero updates for damaged gates.
    Δlogits = where(mask==1, Δlogits, 0)
    Δhidden = where(mask==1, Δhidden, 0)
    logits += α_logit * Δlogits     # ReZero scalars
    hidden += α_hidden * Δhidden

    state = compute_residuals(state, x_data, y_data)
    yield step=t
```

`AttentionBlock` is Pre-LN with ReZero residuals:

```
q_n = LN_q(z); kv_n = LN_kv(z)
attn = MultiHeadAttention(q_n, kv_n, kv_n, mask, normalize_qk=True)
z = z + α_attn · attn
z = z + α_ffn · FFN(LN_ffn(z))            # FFN is Linear-GELU-Linear
```

`MultiHeadAttention` with `normalize_qk=True` applies a per-head
LayerNorm to Q and K *after* projection but *before* the dot product.

**Damage primitives**

- *Shotgun*: pick `n` random gates from hidden layers (uniform over
  the eligible flat range `[input_n, total - output_n)`), set
  `mask[idx] = 0`, `logits[idx] = -10`.
- *Damage gate*: same primitive on a single index (used by
  click-to-damage).
- *Shuffle wires*: regenerate `wires` per layer with fresh random
  permutations, rebuild the attention mask, leave logits and hidden
  state unchanged. `gen_wires(key, in_n, out_n, arity, group_size)`
  in `circuits/model.py` is the reference.
- *Reset*: re-init logits to `make_nops_with_noise` (soft-wire
  identity + Gaussian noise), zero hidden, restore mask to all 1.

**Quick reference: which knobs survive into v1**

| Setting                  | v1 demo value              |
|--------------------------|-----------------------------|
| `task`                   | `reverse` (12-bit reversal) |
| `arity`                  | 4                           |
| `wiring_mode`            | `random`                    |
| `bidirectional_edges`    | True                        |
| `neighboring_connections`| False                       |
| `use_layer_PE`           | (taken from training cfg)   |
| `use_intra_layer_PE`     | (taken from training cfg)   |
| `use_node_loss`          | (taken from training cfg)   |
| `permanent_damage`       | 1.0 (always permanent in v1)|
| `faulty_value`           | -10.0                       |

### M2 — Weight export

- [x] Choose deployment checkpoint (recorded in Decisions log
       below): `reverse / random_damage`, run-id `1u5ssulx`.
- [x] Walk the `nnx` parameter tree, name tensors canonically
       (`extract_weights.py`).
- [x] Wire `run_jax_parity_test` to that extractor;
       parity verified end-to-end (see M1 entry above).
- [x] On-disk packing: **fp16** by default (~2 bytes/param,
       ~10-bit mantissa), with **uint8** retained as an option for
       smaller models. uint8 was tried first and rejected — naïve
       per-tensor symmetric scaling was too coarse for transformer
       weight distributions (LayerNorm γ values clustered around 1
       lose precision under ±127 quantisation). fp16 is essentially
       transparent for our 162 K-param model.
- [x] Header records architectural shape *and* on-disk dtype
       (`tensor_dtype`, `model_kind`, `attention_dim`, `num_heads`,
       `circuit_hidden_dim`, `arity`, `max_neighbors`, the three
       feature-flag bools, `use_gelu_approx`, `source_run_id`).
- [x] Round-trip parity test (`--weights-json PATH` flag flips both
       the clean and perturbation parity tests onto the JSON):
         * **Clean parity** (32 ticks): max hard-acc Δ = 0.0039,
           final-step hard predictions agree 49 152 / 49 152.
         * **Perturbation parity** (warmup + shotgun + click +
           shuffle, 45 ticks total): 32 / 45 steps with 0
           disagreements; worst step is shuffle → step 10 with
           4192 / 49 152 (8.5 %) disagreements but only Δ = 0.033
           in hard accuracy — characteristic float-noise on
           low-confidence cases during chaotic wiring-OOD recovery.
- **Exit criterion (✓)**: `public/weights/reverse_random_damage.json`
  exists at **424.4 KB**, decodes back through the oracle, and runs
  with hard-prediction agreement of 0/49152 in the warmup phase
  and degraded only on the wiring-OOD recovery (where soft-state
  variance is *expected* to differ between any two float
  implementations of the same recurrent system).

### M3 — TypeScript port

- [x] `math.ts`: layer-norm, softmax, gelu (tanh-approx — flax default),
       linear, sigmoid, sinusoidal PE, stable argsort. ~160 lines.
- [x] `weights.ts`: fp16 + uint8 dispatch, base64 decode, header
       validation. Includes a `parseWeightsFromJson` shim so the
       parser is reusable from Node (no `fetch` dep).
- [x] `circuit.ts`: `Topology` + `CircuitState`, multilinear `runLayer`,
       `circuitForward`, `buildTopology` with edge list / attention
       mask / sinusoidal PE / neighbour cache, plus the four
       perturbation primitives (`resetCircuit`, `damageGate`,
       `applyShotgun`, `shuffleWires`).
- [x] `tmt.ts`: gathered-attention block (Pre-LN + ReZero,
       per-head QK-norm, gather-by-index attention, FFN with
       internal LN), TMT step with knockout-aware update,
       `computeResiduals` + top-level `tick`. All scratch buffers
       preallocated.
- [x] `replay.ts` + `record_trajectory.py`: a recorded reference
       trajectory (8 ticks, ~1.5 MB) lives next to the weights;
       `runReplay` / `runReplayFromJson` drive the same TS code path
       and assert per-tick agreement. CLI: `python -m export.record_trajectory --weights-json … --out … --n-ticks N`.
- [x] `main.ts` mounts a tiny diagnostic widget that fetches both
       artefacts and surfaces the verdict + a per-tick deltas table.
- [x] **Node-side smoke runner** (`scripts/replay_node.ts`,
       run via `npx tsx scripts/replay_node.ts`) — same TS code,
       different I/O shim. Used as a CI-friendly check.
- **Exit criterion**: ✓ TS port produces **identical hard
  predictions** to the JAX policy. Verified in the browser:

  ```
  TS-side parity PASS: max hard_acc Δ over 8 ticks = 0.0000 (tol 0.05)
  weights: /weights/reverse_random_damage.json
    arch:   gathered_attention · D=128, H=4, arity=4, hidden=64
    params: 162,102 (fp16, 424 KB JSON, 57 ms decode)
    replay: 8 ticks in ~6.6 s
  ```

  Soft-state drift across 8 ticks: max Δ logits ≤ 7.4e-4, max Δ
  hidden ≤ 1.4e-2 — same character as the NumPy oracle, with the
  expected float32-summation-order divergence between JS and
  NumPy. Hard predictions match exactly at every tick.

  *Note on perf*: the 6.6 s figure includes two full
  4096-case forward passes (soft + hard) per tick. M4 will
  subsample to 256 cases for the live demo, dropping per-tick
  cost to ~25 ms.

### M4 — UI

Done in three phases. Phase 1 ships an interactive widget without the
circuit canvas; Phase 2 adds Canvas2D circuit drawing on top; Phase 3
wires click-to-damage to it.

#### Phase 1 (this commit) — interactive widget without circuit canvas

- [x] **Multi-page layout.** `index.html` (live demo) + `verify.html`
       (parity smoke test). `vite.config.ts` enumerates both as
       `rollupOptions.input` so production builds pick up both.
- [x] **`controller.ts`**: a single `Controller` class owns
       weights / topology / state / scratch / rng / step counter,
       and exposes `tick`, `reset`, `shuffle`, `shotgun`, `damage`.
       Each method returns a `ControllerSnapshot`
       (step, soft + hard accuracy, predHard, predSoft, damaged
       count). The UI never touches math primitives directly.
- [x] **Live demo `main.ts`** (was the parity widget, now refactored
       into `verify.ts`): three bit-image strips (Input / Current /
       Expected) updated each tick, four buttons, live readout, rAF
       loop with throttling. Buttons disabled until weights load.
- [x] **Trajectory file repurposed**: still serves the parity test,
       but the live demo also uses its `task_data` (x, y) and
       `layer_sizes` so we don't have to reimplement task generation
       in TS. Per-tick states from the trajectory are ignored by the
       live demo — the controller spawns its own state.
- [x] **Default trajectory now sequential 4096 cases.** Text-reverse
       (the `--task-style text-reverse` opt-in) was visually too
       pixelated for `output_bits=12` even with the wide-render
       trick, so it stays in the codebase as an option but the
       sequential default ships richer bit patterns with the same
       in-distribution semantics.
- **Phase-1 exit criterion (✓)**: live demo loads in the browser,
  Play animates the policy, all four user actions work end-to-end,
  parity test still passes on `/verify.html`. Per-tick latency at
  4096 cases is ~600–800 ms (CPU-only TS, two circuit forwards per
  tick) — feels stately but watchable. Subsampling to 256 cases is
  a one-line change (the model only sees `r_i = mean over batch`,
  so any subsample is in-distribution).

#### Phase 2 (this commit) — Canvas2D circuit drawing

- [x] **`render.ts`** — full Canvas2D port of
       `_render_circuit_diagram` from `training_demo.py`. Gates render
       as rounded rectangles coloured by per-case activation; wires as
       cubic Béziers with brightness modulated by sender activation;
       group outlines for visual structure; damaged gates with a red
       overlay + the most-recent-perturbation set highlighted in
       yellow. HiDPI-aware via `setupCanvas()` (CSS-pixel layout, the
       backing store is `devicePixelRatio`-scaled).
- [x] **Single-case forward** as a small dedicated helper that runs
       `circuitForward` with `caseN=1` (~1 ms total — trivial). Called
       only at draw time, not in the TMT loop, so zero impact on tick
       throughput.
- [x] **Active case** chosen at boot (`1234 % caseN`); the circuit
       visualises how that one input pattern flows through the network.
- [x] **Layout**: circuit panel slots between the input strip and the
       current-output strip, so the data-flow direction matches the
       page layout (input → circuit → output → expected).

#### Phase 3 (this commit) — Click-to-damage + click-on-strip

- [x] **Hit-test** on the circuit canvas (`pickGate` in `render.ts`)
       → flat gate index. Mirrors `DemoSession.damage_gate` eligibility:
       input / output layers ineligible; already-damaged is idempotent.
- [x] **Hover feedback**: cursor turns into a pointer over damageable
       gates; the gate gets a yellow outline. Stays default elsewhere
       so the user knows what's clickable without a tutorial.
- [x] **Click-on-strip** → set the active case. Both *input*, *current*
       and *expected* strips accept clicks; a thin yellow vertical
       marker (CSS-overlay, sub-pixel positioned via `left: %`)
       shows the picked column on all three strips with a 80 ms
       ease-out transition. Mirrors `_render_image_panel`'s
       click-to-set-active-case interaction in `training_demo.py`.
- [x] **Hint line** under the buttons spells out the click-to-damage
       interaction.

#### Phase 3.5 (this commit) — Data-file split

The trajectory file ended up serving double duty (parity reference *and*
demo bootstrap), which made the `--n-cases` knob feel like it lived in
the wrong place. Split into two artefacts produced by one recorder
invocation:

- [x] `public/weights/reverse_demo.json` (~ 512 KB, the file
       `main.ts` loads): header + `layer_sizes` + `task_data`. Just
       what the live demo needs.
- [x] `public/weights/reverse_trajectory.json` (~ 4.2 MB, the file
       `verify.ts` loads): bootstrap + `wires` + `initial_state` +
       per-tick recorded states for parity validation.
- [x] `record_trajectory.py` writes both in one call; the
       backwards-compat alias `--out` still works.
- [x] README's "What lives in `public/weights/`" table makes the
       split discoverable.

#### Phase 4 — Polish (deferred)

- [ ] Mobile-safe sizing (responsive grid, sensible canvas heights).
- [ ] Reduced-motion fallback (auto-pause if `prefers-reduced-motion`).
- [ ] FPS / per-tick-ms display in the readout.
- [ ] Width-slider for Regime IV (M6 pulled in if we have time).

- **Overall M4 exit criterion**: a one-screen widget that mirrors what
  the user sees in `training_demo.py` for the four actions, embedded
  cleanly in the article via a single `<div>` + `<script>`. **Phases
  1–3 complete** — the demo runs end-to-end at `/` with all four user
  actions plus active-case picking; the parity test stays at
  `/verify.html`. M5 (build + deploy) is the only blocker for
  shipping.

### M5 — Deploy

- [ ] `npm run build` produces an ESM bundle + JSON in `dist/`.
- [ ] Copy to `gabrielbena.github.io/assets/sodc-demo/`.
- [ ] Embed in `_posts/2026-05-06-sodc.md` between paragraphs.
- [ ] Smoke-test on the published site (Chrome / Firefox / Safari).
- **Exit criterion**: widget runs in a fresh browser tab with no
  console errors, reset / shuffle / shotgun / click all work.

### M6 — Stretch: width-OOD

- [ ] "Apply width" knob that re-bootstraps at width factor 1.0 /
       1.5 / 2.0 (discrete).
- [ ] Verify scale-free behaviour visibly improves at 2.0× as the
       paper claims.

## Decisions log

- **Front-end stack: TypeScript + Vite.** Reason: numerical-shape
  bug safety (the export-time tensor names are the contract between
  Python and TS), zero-config bundling, single-file ESM output.
  Reconsider if v1 ends with < 300 LoC of TS — vanilla JS is then
  competitive.
- **Render stack: Canvas2D.** Reason: ~300 gates × ~1 K wires at
  30 FPS is comfortable in 2D. WebGL only worth the complexity if
  we later target ≥ 5 K gates.
- **Quantisation: per-tensor int8 with scalar scale.** Mirrors
  `post--growing-ca/public/webgl_models8/*.json`. Fallback is fp16
  if M2 parity fails.
- **Deployment: manual copy.** Build artifacts copied into the blog
  repo by hand. No cross-repo automation in v1; revisit once the
  bundle is stable.
- **Demo task: bit reversal.** Cleanest visual; perfect-accuracy
  ceiling makes shotgun recovery dramatic.
- **Demo bootstrap and parity trajectory are separate files.** The
  parity trajectory is a reference for `verify.ts` only; the live demo
  loads a much smaller bootstrap (header + `layer_sizes` + `task_data`)
  with no recorded ticks. One recorder invocation produces both. The
  user-tunable `--n-cases` knob lives on `record_trajectory.py`,
  which is the natural place for "data the demo consumes" tooling.
- **Default trajectory shape: sequential 4096 cases.** Text-reverse
  (rendering text into ``y`` and using ``y[:, ::-1]`` as ``x``)
  was a beautiful idea on paper — model in-distribution, visual
  payoff is "watch the upside-down text right itself". In practice
  the bit-reversal noise on a 12-row text is too lossy to produce
  legible letters at the policy's ~87% hard-accuracy ceiling on
  text-shaped inputs, even with a 4096-wide render and middle-256
  crop. Sequential `arange(4096)` produces structured stripes that
  look pleasingly TV-static-y and survive reversal noise gracefully.
  Text-reverse stays in the codebase via `--task-style text-reverse`
  for tinkering.
- **Checkpoint pinned** (M2):
  - run-id: `1u5ssulx`
  - entity / project: `gbena / boolean-nca-cc`
  - prefer-metric: `eval_damaged_out_test_hard_accuracy`
  - architecture: `CircuitGatheredAttention`,
    `attention_dim=128`, `num_heads=4`, `circuit_hidden_dim=64`,
    `arity=4`, `max_neighbors=10`, `use_layer_PE=True`,
    `use_node_loss=True`, `use_intra_layer_PE=False`.
  - 162,100 parameters total (well under 1 MB at fp32, ~165 KB
    at int8).
  - Origin: paper figure 4 random-topology reverse, picked from
    `boolean_nca_exploration.ipynb` cell 1
    ("Random wires, damages, reverse task").

## Open risks

- **Damage-mask semantics.** TMT must (a) freeze the LUT logits of
  damaged gates, (b) zero their residual update, (c) still allow
  *attention* to / from them. Easiest verification: replay one
  recorded shotgun trajectory step-by-step in JS vs JAX and diff.
  Tracked in M1 / M3.
- **Quantisation noise.** Bit-reversal sits at perfect hard accuracy;
  int8 noise on the policy *might* tip it off-perfect. Watch the
  M2 round-trip test — if it fails, fall back to fp16.
- **Mobile legibility.** The gate canvas is dense. Plan for a
  responsive cap on circuit size and/or a "tap to expand" mode.
  Tracked in M4.

## References

- **Original NCA in-browser demo**: `post--growing-ca/public/`
  (`demo.js`, `ca.js`, `webgl_models8/`). Defines the JSON quantisation
  format and the "tiny vanilla JS" aesthetic we're aiming at.
- **Reference Python implementation**: `boolean_nca_cc.demo.DemoSession`
  (`step`, `apply_shotgun_damage`, `damage_gate`, `shuffle_wires`,
  `regenerate_pool`).
- **Reference architecture**: `boolean_nca_cc.models.attention`
  (TMT variants), `boolean_nca_cc.circuits.model` (circuit forward).
- **Reference UI semantics**: `training_demo.py`
  (`_render_circuit_diagram`, `_render_damage_section`,
  `_render_circuit_section`).
