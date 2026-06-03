# SODC live-demo — NON-TRAINING (deploy-side) optimizations

Levers that improve demo crispness **without retraining** — complementary to the
training sweep (`sweeps/sweep_demo_12.yaml`, tracked in W&B). These are not
logged by W&B because they're inference-time / offline-curation choices.
Numbers from the 2026-06-02 diagnosis; probe harness:
`web_demo/export/ceiling_probe.py` (conda env `bool_nca`).

---

## The reframe (why these matter)

- The headline "~91%" is a **step-256, topology-AVERAGED, held-out-test** number.
  The demo shows **one** topology at a time → the average is pessimistic.
- Two *distinct* defects, often conflated:
  - **(a) Jitter** = a shuffle-triggered **short-period limit cycle** on the
    deployed 12-bit model. A *clean* topology reaches an exact hard fixed point
    (0 flips); the OOD wire-shuffle trips it into a ~T-periodic cycle.
  - **(b) Defaults** = a random-topology **capability** gap — but it's
    *per-topology*, not uniform, so curation dodges most of it.

## Measured facts (the evidence the levers rest on)

- **12-bit `reverse_random_damage` (run 1u5ssulx):** clean → exact fixed point
  (flip ≈ 3e-4); shuffle → limit cycle (~38× flip rate, never settles);
  ~6–7% bits confidently wrong **but measured on a non-representative
  text-bitmap middle slice → likely inflated**. Logit/temperature sharpening
  does **not** reduce flips (hard bit = `round(sigmoid(·))`, a sign-crossing,
  gain-invariant).
- **14×14 `dist_pe+rwse` (run 6mo8q61y):** step-256 topology-avg = **0.925**;
  **3/24 random topologies hit EXACTLY 1.0000 and hold** (tail-std 0); the model
  is **stable** (tail-std 0.001–0.003, peak−final ≈ 1%);
  **E[best-of-N topology]: 2→0.957, 4→0.977, 8→0.991, 16→0.999**;
  peak step ≈ 65 (≫ training horizon 10). GPU peak VRAM ≈ 10 GB (tiny).

---

## Optimizations (ranked by leverage, all zero-retrain)

### 1. Offline topology curation — HIGHEST leverage
- **Pre-screen random wirings** with `ceiling_probe.py`; ship a **vetted, crisp
  starting topology** in the bootstrap (fixed-wires models already ship `wires`
  this way) so the first impression is ~perfect.
- **`shuffle` draws from a small pool of pre-vetted good topologies** → every
  shuffle lands crisp, **one** live rollout (FPS unchanged), and shuffle stays
  genuinely **OOD to the model** (it's a real random wiring, just pre-screened).
- Payoff: best-of-8 ≈ 0.99; perfect topologies exist. This is *the* fix for the
  "defaults," and it's honest.

### 2. Settle / freeze — kills the 12-bit jitter
- Stop ticking (or damp) once the hard prediction is stable, using a
  **plateau / majority-vote over a late window** — NOT exact-equality (a
  period-3 cycle never goes bit-identical). **Reset the settle clock on EVERY
  perturbation** (shuffle / shotgun / click-damage), not just shuffle.
- Optional companion: **late-window logit-space averaging** before hard-rounding
  (the deploy mirror of the commented-out mean-over-steps training loss).

### 3. Display / readout (cosmetic but honest)
- **Soft-prediction EMA** before rounding (low-pass; ~41–91% flip reduction,
  seed-dependent — attenuates, doesn't kill, a periodic signal).
- "Converged" indicator + honest framing of the residual as a known capability
  limit (don't print a slice-inflated ceiling number in UI copy).

### 4. Representative case set
- The demo currently displays a **text-bitmap middle slice** (`task_style=
  text-reverse`, `_subsample_middle`) — re-sample to representative/stratified
  reverse cases and re-measure; this likely shrinks the *apparent* defaults for
  free (the per-node residual `r_i` over a structured slice is mildly OOD).

### 5. Checkpoint selection
- Ship **clean-best** vs **damaged-best** to match what the card showcases; score
  candidates by a **stability metric** (mean late-window hard-acc − flip-rate)
  over a long rollout, not a single-horizon endpoint.

### 6. Test-time tricks (optional)
- Multi-seed / multi-rollout **voting** on shuffle (caps at 2–3 rollouts for FPS).
- Logit-space temporal mean over a late window.

---

## Deploy dependency (gated on the sweep)
- **Port `dist_pe` + `rwse` to the TS demo** (`tmt.ts:extractFeatures`,
  `weights.ts` header, `circuit.ts:buildTopology`) — **only if** the PE ablation
  (sweep runs 1–4) shows `dist_pe+rwse` beats `layer_PE` at 12×12. If `layer_PE`
  is competitive, this work evaporates (the TS port already supports it).
  → tracked as task #8.

## Out of scope / proven not to help
- Logit / temperature sharpening (sign-crossing, gain-invariant).
- Training on shuffle/shotgun (removes the OOD that makes the demo compelling).
- Perceiver / cross-attention (explicitly out of scope).

## Tooling
- `web_demo/export/ceiling_probe.py` — per-topology best/final, peak-step,
  per-bit error, best-of-N (env `bool_nca`).
- Training sweep `demo_sweep_12` → best PE config + the deployable checkpoint;
  feeds the curation in #1 and the decision in the deploy-dependency above.

(Actionable post-sweep work is tracked as task #7 "probe + curate topologies".)
