# KIRSCH-PROBE PLAN — mapping SODC's memorise→generalise transition

Companion to `HANDOFF.md` (§C/§E). Created 2026-06-09. The arithmetic arc, reframed as
**Kirsch et al. 2022 GPICL's memorise→generalise transition** and the campaign to probe it.
Wiki: `wiki/refs/kirsch-2022-general-purpose-icl` (Gabriel's deep-read); synthesis deposit
`~/.claude/wiki-deposits/2026-06-09-gpicl-sodc-add-plateau-transition.md`.

═══════════════════════════════════════════════════════════════════════════════
## 1. THE FRAME (one paragraph)
A meta-learner (our TMT/NCA optimising boolean circuits over random wirings) undergoes a
**memorise → identify → generalise** transition, gated by **task diversity**, **capacity
(model size AND accessible state)**, and **meta-optimisation stability** (§4.3 — an extended
loss plateau then a SHARP drop into learning-to-learn; explicitly grokking). Mapping:
`add` stuck at the coarse 0.73 plateau = **task memorisation**; `reverse` at near-exact = **generalised**.
Kirsch's bottleneck is **accessible state** (memory), not raw params (Insight 4) — but model SIZE
also drives the within-architecture transition (Fig 2). For us: state ≈ `circuit_hidden_dim × n_nodes × T`,
params ≈ `attention_dim`/mlp width. **Task diversity is likely NOT our bottleneck** (we sample fresh
random wirings = effectively unbounded ≫ Kirsch's ~2¹³≈8192) — caveat: our wirings vary the *routing*
of a FIXED function, not the dataset, so the diversity mapping is a rhyme to test.

═══════════════════════════════════════════════════════════════════════════════
## 2. WHAT THE CURRENT GRID + PROBE ALREADY COVER
- **`crossed_rev_add`** (array 3874375): {add,reverse} × T{8,16,32} × lr{2e-4,3e-4} + reservoir(h128)
  + −damage, 12-bit. Spans the **state(hidden/T) × lr** corner at 12-bit. Result so far (~64%):
  reverse generalises (0.93, reservoir-arm best), **add uniformly stuck at 0.73**, high-T arms unstable.
- **`warmupclip_probe`** (array 3878363): rev+add T16 with warmup_factor=5 + grad_clip=0.5 — testing
  whether STABILITY unlocks the high-state arms. **Verdict pending; warmup=5 too slow (try 10–12).**
- **NOT covered yet:** the **params axis** (attention_dim held at 128), a **fresh diversity sweep**
  (Gabriel's old one predates working OOD — re-run), **multiply** (the hard subject), **curriculum**.

═══════════════════════════════════════════════════════════════════════════════
## 3. SUPPORTING FINDINGS (this session) — read before designing
- **Eval-rollout limit cycle (metric confound).** The 256-step eval develops a PEAK-THEN-DRIFT limit
  cycle as training "plateaus": settles cleanly early (fixed point ~0.73 held), later peaks ~rollout-
  step 30 then drifts down + oscillates. ⇒ `eval_out_test/final_hard_accuracy` (step-256) **understates
  + wanders**. **Use the peak / settle-window (~step 30)** — re-enable a settled-acc metric (cf.
  demo_probe `settle_window`) for the campaign. Stepwise data lives in wandb as PNG plots
  (`stepwise/eval_out_test`), downloadable + viewable.
- **High-T instability.** Reverse arms lift off then peel back (rev/T16 −0.10 to −0.18, rev/T32 −0.15),
  worse with hot lr; `add` arms stay flat (stuck, nothing to destabilise). Long-T = expensive AND
  destabilising AND (per Kirsch) the wrong capacity axis (T = compute depth, not state size).
  **⇒ STARVE T; push state via hidden (parallel, cheap), params via attention_dim (parallel, cheap).**
- **Capacity = params AND state.** Don't push only hidden; attention_dim is the untested Fig-2 lever.

═══════════════════════════════════════════════════════════════════════════════
## 4. THE TOTAL KIRSCH SWEEP (design; subjects = add + multiply; reverse = anchor only)
Reverse is past the transition → it can't *show* the crossing; use it as a cheap **anchor** (the
"generalised" reference + a config sanity check), NOT a subject.

| Kirsch axis | knob | levels |
|---|---|---|
| task difficulty | task | reverse *(anchor)* · **add** · **multiply** |
| model size (Fig 2) | `model.attention_dim` (±mlp) | 128 · 256 · 512 |
| accessible state (Insight 4) | `circuit.circuit_hidden_dim` | 64 · 128 · 256 |
| task diversity (Insight 1/3) | wiring_mode=fixed `initial_diversity` vs random | 64 · 512 · 4k · 16k · ∞ |
| compute depth | `training.n_message_steps` (T) | 8 · 16 · 32 |
| meta-opt stability (§4.3) | meta_batch · warmup/clip · optimiser | 128/512 · ±long-warmup+tight-clip · adamw/sign |
| curriculum (§4.3) | memorise-first | none · few-wirings→expand |
| (grok is seed-noisy) | seed | ≥2–3 at boundary cells |

Full cross = thousands (infeasible). **Run as additive SLICES around a reference** (add, attn128,
h128, ∞-wirings, T8, +warmup/clip, settle-window metric):
- **Capacity plane** (headline params×state): add × attn{128,256,512} × hidden{64,128,256} — 9
- **Diversity** (fresh re-run): add × wirings{64,512,4k,16k,∞} — 5
- **Stability**: add × meta_batch{128,512} × ±warmup-clip × opt{adamw,sign} — ~8
- **Depth**: add × T{8,16,32} (stability on) — 3
- **Curriculum**: add × {none, memorise-first} — 2
- **Hard end**: multiply × best-capacity × best-stability — ~4
- **Anchor**: reverse × {reference, best-capacity} — 2
≈ 30–35 cells × 2–3 seeds at the boundary ⇒ **~50–80 runs.**

═══════════════════════════════════════════════════════════════════════════════
## 5. COST + STAGING (the part that keeps it sane)
- **Most cells = T8 + parallel knobs (attn/hidden)** → V100-affordable; T8 runs parallelise across
  the ~40-card pool → ~1–2 days WALL for ~50 of them, NOT 200h serial.
- **Expensive corners (attn512 / hidden256 / T32)** → A100 (40–80GB), NOT H200, a handful, staged.
  (h256@T8 ≈ ~40GB; drop batch or use A100. T64 is NOT needed — T is the wrong axis to max.)
- **DO NOT scale the circuit DOWN to save compute** — n_nodes is part of accessible state (the
  bottleneck under test); shrinking it removes the resource we're probing (Gabriel's call, correct).
- **Playbook:** cheap V100 capacity+diversity+stability slices → locate the boundary → spend a few
  A100 runs only at the corners that matter + multiply. Never fire all 80 at once.

═══════════════════════════════════════════════════════════════════════════════
## 6. MEASUREMENT
- Per cell: does add/multiply cross from the 0.73 memorise plateau to precise generalisation, read on
  the **settle-window** (NOT step-256). Plus the **grokking signature** (loss/acc plateau → sharp drop)
  and the **Kirsch Table-1 regime** (learns × generalises).
- Output = a **phase diagram** of the transition over (capacity × diversity × stability), per task.
- This is paper-shaped on its own: "where does a topology-agnostic circuit optimiser cross from
  memorising to generalising, and which capacity (params vs state) controls it" = the SODC↔GPICL bridge.

═══════════════════════════════════════════════════════════════════════════════
## 7. OPEN CHOICES / TODO before launching
- **Re-enable a settled/peak eval metric** (the final-step confound) — needed for clean read.
- **Probe warmup_factor** 5 → ~10–12 (5 crawls). grad_clip 0.5 seems fine.
- **Diversity mechanism:** wiring_mode=fixed + initial_diversity=N for the training set; eval_out OOD
  already = held-out random wirings. Confirm the fixed-N path samples a *fixed* set (not re-randomising).
- **Seeds:** budget ≥2–3 at boundary cells — grok timing is run-variable.
- **Don't launch until the current grid gives its verdict** (does add grok on its own?) — that sets
  the campaign's reference + frees the V100 pool.
