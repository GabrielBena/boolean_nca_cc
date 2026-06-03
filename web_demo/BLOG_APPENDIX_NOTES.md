# Live-demo appendix — notes for the SODC blog post

📌 PINNED (Gabriel): the final blog post should carry an **honest appendix**
documenting everything the live demo does under the hood. This file is the
running source for it — assembly, not archaeology. Last updated 2026-06-03.

---

## 1. The deployed model & provenance

- **`reverse_random_damage`** (the headline model): run **`1u5ssulx`**, a 12×12
  reverse-task Topology-Masked-Transformer (gathered attention, `layer_PE`,
  damage-trained, random wiring). We ship the **undamaged-best** checkpoint
  (best held-out `eval_out_test_hard_accuracy`): **0.931 clean / 0.887 damaged**
  (topology-averaged held-out test). 162,102 params, fp16 JSON.
- **Two `reverse_fixed_*` models** (Regime I, fixed topology): clean and
  damage-trained. Perfect-reverse on their fixed wiring; used to show zero-shot
  damage resilience.
- **Why `1u5ssulx` and not a fresh sweep model.** A June re-sweep at 12×12
  (`layer_PE`/`dist_pe`/`dist_pe+rwse`) came in **~3% below** `1u5ssulx` on raw
  accuracy. The gap traces to **hyperparameters, not architecture**: the sweep
  used `data_per_batch=256, lr=2e-4` vs `1u5ssulx`'s `64, 4e-4`. A retrain at the
  original recipe is the planned upgrade path (see §8); until it lands, the
  best-accuracy paper checkpoint ships.
- Provenance is recoverable: configs were pulled from the **W&B run records**
  and verified by rebuilding to the exact parameter count.

## 2. What the demo computes each tick (the batch ≠ the display)

- Every tick the TMT runs the message-passing rule over a **fixed, diverse
  batch of 256 reverse cases** — this drives the per-output-node residual
  (`r_i = mean_batch |pred − target|`, a TMT input feature) and hence the
  circuit's evolution. **This batch is the same for every model and never
  changes**, so per-tick cost is constant.
- **The display is decoupled from the batch.** The strips render the *current
  circuit forwarded on whatever case-set is selected* — a cheap ~1 ms boolean
  forward, not another TMT pass. "All cases" shows the diverse batch; "Text"
  shows the text columns (§4). Same evolved circuit, visualized two ways.
- **Why this matters (residual-starvation finding).** `r_i` is a batch *mean*.
  A sparse or structured batch (e.g. a thin text bitmap, or a sequential middle
  slice) drives `r_i → 0` → no error signal → the circuit never settles
  (accuracy flat from step 0; output bits flicker). Feeding a **diverse** batch
  keeps the residual informative. So we run the residual on diverse cases and
  treat text purely as a display — the demo never feeds sparse text into the loss.

## 3. Topology curation ("Shuffle wires")

- A trained TMT solves random wirings with **varying success** (12×12: settled
  hard-acc spans ~0.75–1.00 across random topologies; a minority are exact,
  flicker-free fixed points). The demo shows **one** topology at a time, so we
  curate.
- **`shuffle` walks a pre-screened pool, best-first.** Every wiring shown is a
  *real* random topology the model genuinely solves — just ordered by quality,
  so the first impression is crisp and there are no repeats until the pool is
  spent (then it falls back to fresh random wirings). One live rollout; FPS
  unchanged.
- **Ranking is shuffle-aware, not pure clean accuracy.** A razor-perfect (sharp)
  fixed point recovers *worst* from a mid-flight shuffle (carrying the evolved
  logits onto a new wiring traps them: a 1.0 start → ~0.83 recovery, while a
  ~0.94 start → ~0.95). So we rank by `min(pre, post)` — clean accuracy AND
  carry-logits transfer — favouring topologies that are good *and* shuffle-robust.
- **Curated on the demo's exact display cases.** A topology "perfect" on one
  random case-sample can flicker on another; the pool is screened on the same
  256 cases the demo renders, so rank-0 is genuinely clean on screen.

## 4. The reversed-text trick

- Text is rendered into the height-12 target bitmap `y`; `x = y[:, ::-1]` is the
  per-bit reversal. The reverse policy's job is exactly `x → y`, so a good
  circuit "rights" the mirrored text.
- It's a **pure display**: the circuit evolves on the diverse batch (§2), and we
  forward that *same* circuit on the text columns. No filler, no re-settle, no
  FPS hit — toggling "All cases ⇄ Text" is instant. On a near-true-reverse
  topology the recovered text is pixel-perfect.
- All models ship the *same* text columns ("Welcome to Self Organising
  Circuits! The Future is Now!") and a "Text" toggle; the fixed models default
  to the text view.

## 5. Accuracy readout & chart

- Readout shows three numbers: **hard_acc** and **soft_acc** (the base inference
  on the diverse batch) and, in text mode, **text_acc** (the text reconstruction
  quality). hard/soft always reflect the circuit's reverse capability; text_acc
  is a separate display metric.
- The chart traces all three, colour-blind-safe and style-differentiated:
  **hard = blue solid, text = orange dashed, soft = grey dim**.

## 6. Honesty caveats to state plainly

- The displayed topology is **curated** (shuffle-aware, best-first), not a random
  draw — say so.
- **Mid-flight shuffle and shotgun damage are pure OOD** perturbations (never
  trained). Every model craters to ~50% for a tick or two before recovering —
  that dip is honest, not a bug.
- **Jitter is real but small**: it's *per-bit prediction churn* (a few output
  bits flickering), not an accuracy swing — visible at imperfect fixed points,
  ~zero at perfect ones. Curation to high-accuracy topologies minimizes it.
- Hard accuracy is a *per-topology* number; the headline "~0.93" is a
  topology-average. Don't print a slice-inflated ceiling in UI copy.
- The reverse policy is trained at a short horizon (T=5) but deployed to settle
  over more steps; it settles to a near-fixed-point on in-distribution batches.

## 7. Findings worth disclosing (the "what we learned")

- **Residual starvation** — the node-loss feature is a batch mean; a non-diverse
  batch starves it (§2).
- **Perfection locks in** — a sharper/cleaner fixed point transfers *worse*
  across a carry-logits shuffle; mild residual jitter aids escape.
- **`dist_pe` is wiring-aware** (recomputed per topology), so a dist-PE model
  *senses* a rewiring and re-solves cleanly under carry-logits; `layer_PE` is
  wiring-blind. (Why the dist retrain is the principled robustness upgrade.)
- **Jitter = per-bit churn**, not accuracy std — the metric that matches what the
  eye sees.

## 8. Numerical fidelity & deployment

- Python (JAX) is the source of truth → NumPy oracle → TS port, validated by a
  headless replay (`scripts/replay_node.ts`): TS hard predictions match the JAX
  policy **bit-for-bit** (Δ=0.0000/tick). fp16 weights are transparent at the
  hard-prediction level.
- Planned upgrade (no-promises, running in parallel): retrain `dist`/`dist+rwse`
  at the original recipe (`data_per_batch=64, lr=4e-4`) to recover the +3% raw
  accuracy *and* keep carry-logits shuffle robustness + no jitter.

## Build log (shipped, in order)
- [x] Weights → `1u5ssulx` undamaged-best (best raw accuracy)
- [x] Residual on a diverse 256-case batch (every model); display decoupled
- [x] Shuffle-aware topology pool, curated on the demo's exact display cases
- [x] Text-reverse as a display-forward (no filler) + "All cases ⇄ Text" toggle
- [x] 3-accuracy readout (hard/soft/text) + colour-blind-safe chart traces
- [x] Fixed models made architecturally identical (diverse residual + text toggle)
- [x] All models default to the text view
- [ ] `dist` retrain @ 64/4e-4 (parallel experiment) → swap in if it wins

## Tooling (for the methods-curious)
Probes under `web_demo/export/` (env `bool_nca`): `ceiling_probe`,
`curate_topologies` / `curate_shuffle_aware`, `probe_churn`(+`_screen`),
`probe_shuffle_models`, `probe_recovery_vs_perfection`, `probe_salvage_1u5`,
`probe_density`, `probe_text_{reverse,filler}`. Export/record:
`extract_weights`, `export_local_weights`, `record_local`. Web build/test runs
on anahita (conda env `webdemo`: node + vite + `npx tsx` parity). HSM
sync-brittleness write-up is separate (`HPC-Sweep-Manager/FIRST_USE_FEEDBACK_S3IT.md`).
