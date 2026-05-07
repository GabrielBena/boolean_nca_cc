# SODC Web Demo

A minimal in-browser, interactive demo of the Topology-Masked Transformer
(TMT) policy from *Self-Organising Digital Circuits*. Designed to be
embedded in the article webpage, in the spirit of the original
Growing Neural Cellular Automata demo (`post--growing-ca/`).

## Status

M4 Phase 1–3 done. The live demo runs end-to-end with all four user
actions (Play/Reset/Shuffle/Shotgun + click-to-damage on the circuit
canvas + click-on-strip to pick the active case). M5 (build + deploy
into the blog post) is next. See `PROGRESS.md` for the milestone
breakdown, exit criteria and decisions log.

## Layout

```
web_demo/
├── README.md             # this file
├── PROGRESS.md           # living course-of-action doc
├── index.html            # /  → live interactive demo
├── verify.html           # /verify.html  → TS↔JAX parity smoke test
├── src/                  # TypeScript front-end
│   ├── main.ts           # live demo entry: UI + controller + animation
│   ├── verify.ts         # parity-test entry (loads recorded trajectory)
│   ├── controller.ts     # state lifecycle (tick / reset / shuffle / …)
│   ├── circuit.ts        # multilinear LUT forward + topology + perturbations
│   ├── tmt.ts            # gathered-attention TMT step + residual recompute
│   ├── render.ts         # Canvas2D circuit drawing + hit-testing
│   ├── replay.ts         # parity replay over a recorded trajectory
│   ├── weights.ts        # fp16 / uint8 JSON loader
│   └── math.ts           # layerNorm / softmax / gelu / linear / PE / argsort
├── scripts/
│   └── replay_node.ts    # CI-friendly node-side smoke runner (parity)
├── export/               # Python tooling (run with the cc_nca conda env)
│   ├── numpy_oracle.py       # ground-truth NumPy reference inference
│   ├── extract_weights.py    # nnx checkpoint  → fp16 JSON
│   ├── record_trajectory.py  # writes the demo bootstrap + parity trajectory
│   └── test_oracle.py        # self-test + JAX parity + perturbation parity
├── public/weights/       # served verbatim by Vite, copied into dist/
│   ├── reverse_random_damage.json   # fp16 weights      ~ 424 KB
│   ├── reverse_demo.json            # live-demo bootstrap ~ 512 KB
│   └── reverse_trajectory.json      # parity reference   ~ 4.2 MB
├── vite.config.ts        # multi-page (index + verify) build config
├── tsconfig.json
└── package.json
```

## What lives in `public/weights/`

| File                            | Purpose                                                 | Size  |
|---------------------------------|---------------------------------------------------------|-------|
| `reverse_random_damage.json`    | fp16 TMT weights (gathered attention, 162 K params)     | 424 KB |
| `reverse_demo.json`             | Live demo bootstrap (header + `layer_sizes` + `task_data`) | 512 KB |
| `reverse_trajectory.json`       | Parity-test reference (everything above + 32 recorded ticks) | 4.2 MB |

The demo file and trajectory are produced together by one invocation
of `record_trajectory.py`. `main.ts` reads `reverse_demo.json`;
`verify.ts` reads `reverse_trajectory.json`.

## Quick start

Front-end dev server (live reload):

```bash
cd web_demo
npm install
npm run dev   # then open http://127.0.0.1:5173/
```

Re-export weights from a different W&B run (Python side, requires the
`boolean_nca_cc` package on the path — activate the `cc_nca` conda env):

```bash
python -m export.extract_weights \
    --run-id <wandb_run_id> \
    --out public/weights/reverse_random_damage.json
```

Re-record the demo bootstrap + parity trajectory (this is also the knob
to tune the live demo's per-tick batch size):

```bash
# Default: full 4096-case batch — richest visual, ~600 ms / tick.
python -m export.record_trajectory \
    --weights-json public/weights/reverse_random_damage.json \
    --n-ticks 32

# Smaller batch — faster ticks, sparser bit-image strips.
python -m export.record_trajectory \
    --weights-json public/weights/reverse_random_damage.json \
    --n-cases 1024 --n-ticks 32
```

Numerical parity tests (oracle self-test + JAX parity + perturbation parity):

```bash
python -m export.test_oracle --all-parity \
    --weights-json public/weights/reverse_random_damage.json
```

Node-side smoke run of the TS port (no browser, same code as the page):

```bash
npx tsx scripts/replay_node.ts
```

Build a single-file bundle for the blog:

```bash
npm run build
cp -r dist/* ../../gabrielbena.github.io/assets/sodc-demo/
```

Embedding in the article (al-folio's distill layout passes raw HTML
through):

```html
<div id="sodc-demo"></div>
<script type="module" src="{{ '/assets/sodc-demo/main.js' | relative_url }}"></script>
```

## Scope (v1)

- One frozen TMT checkpoint: random wiring, damage-trained, reverse task.
- Four user actions: reset circuit, shuffle wires, shotgun, click-to-damage.
- One bonus: click-on-strip to pick the "active case" the circuit visualises.
- Optional stretch: width-OOD knob (Regime IV).

Everything beyond that — multi-task picker, BP comparison, gallery
loading, profiling — is intentionally out of scope. See `PROGRESS.md`
for the rationale and milestone breakdown.
