# SODC Web Demo

Interactive browser demo of the Topology-Masked Transformer (TMT) policy from
*Self-Organising Digital Circuits*, embedded in the article webpage.

---

## Pipeline overview

```
configs/demo_models.yaml          ← single source of truth: which W&B runs to show
         │
         ▼
export/export_gallery.py          ← batch export: downloads runs, extracts weights,
         │                           records bootstrap files, writes gallery.json
         │
         ├──► public/weights/<id>.json          fp16 TMT weights  (~424 KB each)
         ├──► public/weights/<id>_demo.json     live-demo bootstrap (~514 KB each)
         └──► public/weights/gallery.json       manifest consumed by the TS demo
                      │
                      ▼
              src/main.ts                       fetches gallery.json at startup,
                                                builds model-picker pill buttons,
                                                loads weights + bootstrap on demand
```

The Python side runs once (or whenever you update a run). The TS demo is
read-only at runtime — it only fetches the pre-baked JSON files.

---

## Step 1 — Declare models in `configs/demo_models.yaml`

This is the only file you edit to change which models appear in the demo.

```yaml
recipes:                        # display metadata, keyed by recipe id
  fixed_no_damage:
    label: "Fixed wires · clean"
    description: "Regime I …"
  random_damage:
    label: "Random wires · damage-trained"
    description: "Regime III …"

models:
  - task: reverse
    recipe: fixed_no_damage
    run_id: vt9awu7h             # W&B run id (null = greyed-out placeholder)
    prefer_metric: eval_in_test_hard_accuracy

  - task: reverse
    recipe: random_damage
    run_id: 1u5ssulx
    prefer_metric: eval_damaged_out_test_hard_accuracy
```

**`run_id`** is the W&B short run ID (the 8-character hex in the run URL).
Setting it to `null` keeps the entry in the manifest as a greyed-out button
so you can reserve a slot before the run finishes.

**`prefer_metric`** controls which checkpoint is downloaded — the export
script picks the checkpoint that maximises this metric.

---

## Step 2 — Export everything with `export_gallery.py`

Run from inside `web_demo/` using the `cc_nca` conda environment:

```bash
# Export all tasks declared in demo_models.yaml
conda run -n cc_nca python -m export.export_gallery

# Export only the reverse task
conda run -n cc_nca python -m export.export_gallery --task reverse

# Skip files that already exist (useful when adding new runs)
conda run -n cc_nca python -m export.export_gallery --task reverse --skip-existing
```

For each model entry the script:

1. Downloads the best checkpoint from W&B (`prefer_metric` decides which one).
2. Extracts and fp16-quantises the weights →
   `public/weights/{task}_{recipe}.json`
3. Records the live-demo bootstrap →
   `public/weights/{task}_{recipe}_demo.json`
4. Writes the gallery manifest →
   `public/weights/gallery.json`

### What "recording the bootstrap" does

`record_trajectory.py:record_for_model` runs 8 ticks of the TMT on the
full task dataset and writes the bootstrap file that the browser loads.
The bootstrap contains:

| Field         | Content                                            |
|---------------|----------------------------------------------------|
| `header`      | task name, bit widths, arity, seed, run id         |
| `layer_sizes` | `[[nodes, group_size], …]` for each layer          |
| `wires`       | per-layer wire arrays (see note below)             |
| `task_data`   | full `(x, y)` dataset as base64 float32            |

**Why wires are in the bootstrap** — fixed-wires models (`recipe =
fixed_no_damage` or `fixed_damage`) are trained on one specific circuit
topology. Using any other topology at eval time is out-of-distribution and
performance collapses to chance. The export script derives the correct wires
from the training config via the same JAX PRNG path as the training loop
(`derive_pool_keys(cfg, distribution="auto")`), serialises them, and stores
them in the bootstrap. The TS `Controller` loads these wires and uses them
verbatim instead of generating random ones. Random-wires models work with
any topology, so their bootstrap omits the wires field.

---

## Step 3 — `gallery.json` schema

`export_gallery.py` writes `public/weights/gallery.json`. The TS demo reads
this at startup to build the model picker.

```json
{
  "models": [
    {
      "id":            "reverse_fixed_no_damage",
      "task":          "reverse",
      "recipe":        "fixed_no_damage",
      "label":         "Fixed wires · clean",
      "description":   "Regime I, no damage during training.",
      "runId":         "vt9awu7h",
      "weightsPath":   "reverse_fixed_no_damage.json",
      "bootstrapPath": "reverse_fixed_no_damage_demo.json"
    }
  ]
}
```

`weightsPath` and `bootstrapPath` are relative to `public/weights/`.

---

## Step 4 — How the TS demo uses the files

On page load `src/main.ts`:

1. Fetches `gallery.json` (falls back to a single-entry fallback if missing).
2. Defaults to the **last** model in the list (most capable, shown first).
3. Fetches `{weightsPath}` and `{bootstrapPath}` in parallel.
4. Builds a `Controller` with the decoded weights, task data, and
   **pre-loaded wires** from the bootstrap (for fixed-wires models).
5. Renders pill buttons for each gallery entry.

Clicking a pill calls `switchModel()` which loads the new model's files and
replaces the Controller in-place without rebuilding the DOM.

---

## What lives in `public/weights/`

| File                                  | Purpose                          | Size     |
|---------------------------------------|----------------------------------|----------|
| `gallery.json`                        | Model manifest (read by TS)      | ~1 KB    |
| `reverse_fixed_no_damage.json`        | fp16 TMT weights                 | ~424 KB  |
| `reverse_fixed_no_damage_demo.json`   | Live-demo bootstrap + wires      | ~514 KB  |
| `reverse_fixed_damage.json`           | fp16 TMT weights                 | ~424 KB  |
| `reverse_fixed_damage_demo.json`      | Live-demo bootstrap + wires      | ~514 KB  |
| `reverse_random_damage.json`          | fp16 TMT weights                 | ~424 KB  |
| `reverse_random_damage_demo.json`     | Live-demo bootstrap (no wires)   | ~514 KB  |
| `reverse_trajectory.json`            | Parity-test reference (32 ticks) | ~4.2 MB  |

The trajectory file is only used by `verify.html` (TS↔Python parity test)
and is not needed for the live demo.

---

## Source layout

```
web_demo/
├── README.md
├── PROGRESS.md
├── index.html            /  → live interactive demo
├── verify.html           /verify.html  → TS↔JAX parity smoke test
├── src/
│   ├── main.ts           UI + controller + animation loop
│   ├── controller.ts     state lifecycle (tick / reset / shuffle / damage)
│   ├── circuit.ts        multilinear LUT forward + topology + perturbations
│   ├── tmt.ts            gathered-attention TMT step + residual recompute
│   ├── render.ts         Canvas2D circuit drawing + hit-testing
│   ├── replay.ts         parity replay over a recorded trajectory
│   ├── weights.ts        fp16 / uint8 JSON loader
│   ├── math.ts           layerNorm / softmax / gelu / linear / PE / argsort
│   └── verify.ts         parity-test entry point
├── scripts/
│   └── replay_node.ts    CI-friendly node-side smoke runner
├── export/               Python tooling (cc_nca conda env)
│   ├── export_gallery.py     batch export from demo_models.yaml
│   ├── record_trajectory.py  write bootstrap + parity trajectory for one model
│   ├── extract_weights.py    nnx checkpoint → fp16 JSON
│   ├── numpy_oracle.py       ground-truth NumPy reference inference
│   └── test_oracle.py        oracle self-test + JAX/TS parity checks
├── public/weights/       served verbatim by Vite, copied into dist/
├── vite.config.ts
├── tsconfig.json
└── package.json
```

---

## Quick-start

```bash
cd web_demo
npm install
npm run dev        # http://localhost:5173/
```

---

## Changing what the demo shows

All the knobs that affect the browser experience are flags to
`export_gallery.py`. Run from `web_demo/` with the `cc_nca` conda env.

### `--task-style` — sequential vs text-reverse

```bash
# Default: show the full ordered bit-reversal dataset (4096 cases)
conda run -n cc_nca python -m export.export_gallery --task reverse --bootstrap-only

# Text-reverse: render a phrase as binary image; policy "rights" the upside-down text
conda run -n cc_nca python -m export.export_gallery --task reverse --bootstrap-only \
    --task-style text-reverse \
    --text "Self Organising Circuits"
```

### `--n-cases` — number of input columns shown in the strips

```bash
# 256 columns — fast ticks, compact display
conda run -n cc_nca python -m export.export_gallery --task reverse --bootstrap-only \
    --n-cases 256

# Full 4096 — richest visual, ~600 ms/tick in browser
conda run -n cc_nca python -m export.export_gallery --task reverse --bootstrap-only \
    --n-cases 0
```

### Combined: text-reverse with custom text and column count

```bash
conda run -n cc_nca python -m export.export_gallery --task reverse --bootstrap-only \
    --task-style text-reverse \
    --text "NCA Workshop 2025" \
    --n-cases 512
```

### `--bootstrap-only`

Skips re-downloading or re-serialising weights (uses the local W&B cache for
model config). Only re-records the `_demo.json` bootstrap files. This is fast
because the checkpoint is already cached after the first `export_gallery` run.

Without `--bootstrap-only`, the script also re-serialises the fp16 weights
(needed when you change the run ID or want a fresh weight file).

---

### Flag summary

| Flag | Default | Effect |
|------|---------|--------|
| `--task` | all | Only export this task (`reverse`, `add`, …) |
| `--bootstrap-only` | off | Skip weights, re-record bootstraps only |
| `--skip-existing` | off | Skip models whose files already exist |
| `--task-style` | `sequential` | `sequential` or `text-reverse` |
| `--text` | long welcome phrase | Text rendered for `text-reverse` |
| `--n-cases` | `0` (full) | Cases per strip; `0` = full task (4096 for reverse) |
| `--n-ticks` | `8` | Ticks recorded in the bootstrap parity header |
| `--with-trajectories` | off | Also write `_trajectory.json` for parity tests |

---

## Other operations

Run numerical parity tests:

```bash
conda run -n cc_nca python -m export.test_oracle --all-parity \
    --weights-json public/weights/reverse_random_damage.json
npx tsx scripts/replay_node.ts
```

Build for deployment:

```bash
npm run build
cp -r dist/* ../../gabrielbena.github.io/assets/sodc-demo/
```

Embed in the article (al-folio distill layout):

```html
<div id="sodc-demo"></div>
<script type="module" src="{{ '/assets/sodc-demo/main.js' | relative_url }}"></script>
```

---

## Adding a new model to the gallery

1. Add an entry to `configs/demo_models.yaml` with the W&B `run_id` and the
   correct `prefer_metric` for that recipe.
2. Run `export_gallery.py --task <task> --skip-existing`.
3. Check the bootstrap tick log — a fixed-wires model should reach
   `hard_acc ≈ 1.0` within a few ticks; if it stays at 0.5 the wiring seed
   is wrong (see `derive_pool_keys` in `export/record_trajectory.py`).
4. Start the dev server and verify the new pill button appears and the model
   loads correctly.

---

## Wiring seed details (fixed-wires models)

Fixed-wires models are trained on one specific circuit topology. The topology
is determined by:

```
train_key  = PRNGKey(cfg.seed)
eval_key   = PRNGKey(cfg.eval_seed)  # or fold_in(train_key, 1) if unset
pool_key   = eval_key                # for fixed / genetic wiring modes
wires_key, _ = split(pool_key)
wires = gen_wires(wires_key, ...)    # JAX Threefry RNG
```

The export script calls `boolean_nca_cc.demo.derive_pool_keys(cfg,
distribution="auto")` which replicates this exactly. The resulting numpy
arrays are stored in the bootstrap file under `"wires"`.

The TS `Controller` reads `precomputedWires` from the bootstrap and passes
the raw `Int32Array` buffers directly to `buildTopology`, bypassing its own
RNG-based wire generation. This is why fixed-wires models converge in 2–3
ticks instead of never converging.
