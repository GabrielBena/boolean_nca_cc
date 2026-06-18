# STRUCTURAL DIRECTION — past the `add` memorisation plateau (2026-06-15)

Companion to `HANDOFF.md`. The arithmetic arc proved (capacity sweep, 2026-06-14) that **capacity is
NOT the lever** for random-topology `add` — it's flat at 0.73–0.75 across params×state. So the gap is
**structural**. This doc is the design sketch + experiment plan for the structural fix. Source threads:
this is a distilled design conversation grounded in the code (file:line below) + Gabriel's wiki.

═══════════════════════════════════════════════════════════════════════════════
## 0. TL;DR — the thesis
`add`'s 0.73 plateau is most likely an artifact of the **training REGIME**, not of `add`:
- The inner loop is a **BATCH meta-gradient** — every NCA step sees a 256-input batch and an
  **aggregated residual**. That regime (a) averages away per-input carry structure and (b) keeps
  internal nodes **value-blind**, so the meta-learner can satisfy a coarse 0.73 average without ever
  resolving carry chains. Single task + batch ⇒ **memorisation** (Kirsch's first phase).
- The fix that unifies every diagnosis: **diverse tasks (task-sampler) + an ONLINE/sequential inner
  loop** (learn the function one example at a time). This is the **VSML recipe** (Kirsch 2021), which
  is *proven* to yield task/topology-agnostic generalization — exactly what `add` lacks. SODC already
  has the **weight-sharing** half (shared TMT/NCA rule) and a **task-sampler** (diversity half, batch);
  it is missing the **online** half.
- Prediction (the readout that adjudicates): **online should lift `add` (value-dependent compute) and
  leave `reverse` (pure routing) flat.** And **state should become load-bearing online** where it was
  inert in batch — re-opening the capacity null as regime-specific.

═══════════════════════════════════════════════════════════════════════════════
## 1. THE DIAGNOSIS (why `add` fails) — grounded in code
1. **Internal nodes are value-blind.** A node's update vector is `[LUT logits, hidden memory, depth/
   dist PE, (optional) loss scalar]` — `boolean_nca_cc/models/attention/base.py:545-591`
   (`extract_node_features`). It **never** sees the circuit activations (the values flowing under the
   data); those reach the meta-learner only via the loss gradient (`run_circuit` is a separate forward).
2. **Feedback is a coarse scalar.** `use_node_loss` feeds *one number* (|residual|) into **output**
   nodes only (`training/utils.py:42`, broadcast in `train_loop.py:164`). Internal carry-chain nodes
   get nothing. (Gabriel's own wiki `concepts/topology-masked-transformer` already flags: *"per-node
   feedback is a coarse scalar — cross-attention to task data could improve reasoning. Arithmetic task
   generalisation to random topologies remains incomplete."*)
3. **Loss is a flat sum over output bits** — `generalized_bce → jp.sum(pos+neg)`, `circuits/train.py:232`.
   No carry-depth / worst-bit / position weighting. Bit-11 (carry-depth 11) weighted == bit-0.
4. **Batch averaging (the deepest one).** The residual is `|residuals|.mean(axis=0)` over the 256-input
   batch (`evaluation.py` scan body). Per-input carry structure is destroyed before the meta-learner
   sees it. **This is the same averaging as (2)/(3) but over INPUTS — and it's the crux of the online
   reframe.**

**Why `reverse` survives all of this:** reverse is a *topological permutation* (output[i]=input[n-1-i]) —
value-INDEPENDENT routing. The meta-learner only needs "route input→output per the graph," which the
batch/value-blind regime supports. `add` needs value-dependent carry compute, which it doesn't.

═══════════════════════════════════════════════════════════════════════════════
## 2. THE THESIS — online + task-sampler = task-agnostic = past memorisation
- **VSML** (`wiki/refs/kirsch-2021-meta-learning-backprop`, conf 0.78; Gabriel flags §4.3): a shared
  local rule meta-trained to minimise **online prediction loss**, learning **one example at a time**,
  **generalises to datasets/IO-sizes outside meta-training, invariant to input order & random
  projections** → "task/topology agnostic… beyond transferring representations." This is the exact
  property `add` needs, achieved by an ONLINE weight-shared rule.
- **The two routes to a general (non-memorised) rule** (`wiki/concepts/in-context-learning`):
  **weight-sharing** (SODC has it — the NCA/TMT) + **diversity** (the task-sampler). SODC runs the
  weight-shared rule in a regime (single-task batch) that defeats generality → memorisation.
- **Synthesis:** single task + batch → memorise a coarse 0.73. Diverse tasks + online → forced to learn
  a general "infer the function from its examples" algorithm → must resolve carry structure → precise
  `add`. The capacity null and the plateau may both be **single-task-memorisation artifacts**.

═══════════════════════════════════════════════════════════════════════════════
## 3. WHAT EXISTS — the task-sampler (first version, RUNNABLE, but BATCH)
(Full map from an Explore pass; cite file:line.)
- **Entry/config:** `tasks.type == "sampler"`; example `configs/tasks/k_junta.yaml`; gate
  `use_task_sampler = task_sampler_cfg is not None` (`training/train_loop.py:907`). Fully wired,
  tests pass (`tests/test_meta_tasks.py`). Incompatible with `wiring_mode=genetic` (explicit raise).
- **What it samples:** per-circuit **boolean FUNCTIONS** (not wirings/inputs). Two samplers
  (`tasks/samplers.py`, dispatch `tasks/registry.py:78 sample_task_batch`): **k-junta** (train — each
  output bit = random k-subset + random LUT) and **library** (OOD eval — {add, parity, reverse,
  binary_multiply}). Task = full truth table `[2^input_n, output_n]` in `GraphPool.y_task`
  (`pool.py:103`). Inputs are the fixed full enumeration (`build_task_x`, `samplers.py:26`). Fresh task
  per pool slot, re-sampled on pool reset (`train_loop.py:1772`). **Train k-junta vs OOD library = a
  real held-out generalization split.**
- **Regime = BATCH (the gap).** All inputs flow per NCA step; loss `(pred-y).mean(axis=0)`; one
  aggregated residual per node per step (`evaluation.py` ~634-660, scan ~995-1110). Meta-objective =
  final/mean-tail loss (BPTT through the rollout). **Not** per-sample, **not** online.
- **State:** per-node scalar `accumulation`, carried across the inner loop, shared GNN rule across all
  tasks/slots. Logits/hidden fresh per sampled task.

═══════════════════════════════════════════════════════════════════════════════
## 4. THE DESIGN — going online (minimal change on top of the task-sampler)
The task-sampler already gives diversity + task-agnostic objective + topology-agnostic wirings. To add
the **online** half (est. ~200 LOC, a config flag `inner_loop_regime: batch|online`):
1. **Per-sample inner loop:** in `evaluation.py` (`run_model_scan_with_loss` / `apply_model_and_compute_loss`),
   iterate/scan over the input STREAM one (or a small window) at a time instead of the full batch.
2. **Per-sample residual to the NCA:** feed the per-sample error (not `mean(axis=0)`) into the node
   feedback each step (`train_loop.py` aggregation site ~164).
3. **Online meta-objective:** minimise the **sum of online prediction losses** over the stream (predict
   example t after adapting on 0..t-1), VSML-style — not the final batch loss.
4. **Memory as the running estimate:** the per-node hidden state becomes the accumulator of the
   in-context-learned function (this is where STATE should finally matter — test it).

**Feature-level alternatives / complements** (if the full regime change is too big, or to stack):
- **Value-aware node feedback** — concat per-node circuit activations into `extract_node_features`
  (`base.py:588`; source `aux_data["act"]`). Aggregation choice (lightest→richest): mean firing →
  moments → **functional fingerprint** (eval each node on K≈8–16 probe inputs → per-node K-dim
  signature of *what it computes*; ties to FGNN2 functionality reps). This *smuggles* per-input info
  past the batch average — a feature-level approximation of online.
- **Loss reshaping** (cheap, `circuits/train.py:193/302`, extend `LossConfig`): carry-depth/MSB-weighted
  BCE, **worst-bit** term, or **exact-match** (all-bits-correct) bonus → concentrates gradient on the
  broken carry chain. Complementary, not sufficient alone.
- **Output→input cross-attention (Perceiver-lite):** one cross-attn layer, output nodes (Q) → input
  nodes (K/V). The roadmap direction (`wiki/projects/self-organising-boolean-circuits`: Perceiver/
  cross-attention scaling) minus the latent bottleneck. Heaviest; fallback if online underdelivers.

═══════════════════════════════════════════════════════════════════════════════
## 5. EXPERIMENT PLAN (staged, cheap → committed)
0. **Cheap hypothesis probes (V100, days):** (a) **batch-granularity sweep** — `data_per_batch`
   256→16→4→1 in the CURRENT regime; if `add` climbs as batch shrinks, per-instance signal matters
   (supports online); if flat, the benefit needs the memory/stream machinery. (b) **mini-stream** — a
   few inputs sequentially within the existing rollout, memory carrying across. Both are config/small
   changes; run on idle V100s. NB: smaller batch ≠ true online (no stream memory) — it's a directional
   probe only.
1. **Online inner loop on the task-sampler (the headline):** implement §4.1–4.4; meta-train on diverse
   k-junta tasks ONLINE; **eval OOD on `add`** (already in the library sampler). Question: does
   diverse-task online meta-learning cross the memorisation plateau on `add`?
2. **Ablations:** state size (predict it matters online ↔ inert in batch); diversity level (k-junta k,
   library size); stream length; ± loss reshaping; ± value-aware feedback.
3. **Fallbacks:** value-aware feedback / cross-attention if (1) stalls.

**Readout / controls (carry the whole session's discipline):**
- **`reverse` is the control:** online should lift `add` (value-dependent), leave `reverse` (routing)
  flat. add↔reverse gap = the signal. If online lifts both or neither, the mechanism isn't what we think.
- Metric: eval_out hard-acc on the OOD subject; step-256 is FAITHFUL for add (fork-#1). Watch for the
  Kirsch grokking signature (long plateau → sharp drop).

═══════════════════════════════════════════════════════════════════════════════
## 6. INJECTION POINTS (concrete file:line)
| change | where |
|---|---|
| online inner loop (per-sample scan + online loss) | `training/evaluation.py` (`run_model_scan_with_loss`, `apply_model_and_compute_loss`, scan ~995-1110) |
| per-sample residual feed (drop the mean) | `training/train_loop.py` ~164; `training/utils.py:42` |
| `inner_loop_regime: batch\|online` flag | `configs/config.yaml`, threaded into the above |
| value-aware node feedback (activations) | `models/attention/base.py:588` (+ source `aux_data["act"]` in `evaluation.py:151`) |
| loss reshaping (focal/worst-bit/exact-match/weighted) | `circuits/train.py:193` (`generalized_bce`) / `:302` (`compute_loss_from_predictions`); extend `LossConfig` `:22` |
| task-sampler (extend, don't reinvent) | `tasks/samplers.py`, `tasks/registry.py`, `train_loop.py:903-1810`, `configs/tasks/k_junta.yaml` |

═══════════════════════════════════════════════════════════════════════════════
## 7. WIKI REFS (prior thinking — trust tags noted; these are Gabriel's curated notes)
- `wiki/refs/kirsch-2021-meta-learning-backprop` (VSML, draft 0.78) — **the recipe.** Online, weight-
  shared → topology-agnostic. §4.3 is the generalization result Gabriel wants.
- `wiki/refs/kirsch-2022-general-purpose-icl` (GPICL, draft 0.78) — memorise→identify→generalise via
  diversity + state, past a transition. The session's frame.
- `wiki/concepts/in-context-learning` (draft, fresh) — two routes to a general rule = weight-sharing +
  diversity; ICL as forward dynamics.
- `wiki/concepts/topology-masked-transformer` (draft 0.67) — the TMT; "coarse scalar feedback,
  cross-attention to task data, arithmetic incomplete."
- `wiki/refs/guichard-2025-arc-nca` (0.78) — test-time training / per-instance adaptation (the online/
  TTT angle); SODC pool-meta vs per-task TTT.
- `wiki/refs/von-oswald-2022-transformers-icl-gd` (0.82) — inner-loop-as-optimizer (mesa-opt); the
  outer-trained/inner-optimises pattern SODC instantiates.
- `wiki/projects/self-organising-boolean-circuits` (living) — Perceiver/cross-attention scaling +
  the **task-sampler redesign for the agnostic regime** = this direction.

═══════════════════════════════════════════════════════════════════════════════
## 8. OPEN QUESTIONS / RISKS
- **Outer-loop training of a long online inner loop:** BPTT through a stream is harder (vanishing
  credit, cost); VSML used ES or backprop-through. May need stream-length curriculum.
- **What's the stream?** 2^24 input space for 12-bit add — the circuit sees a sample stream and must
  generalize to unseen inputs (the actual hard part). Stream construction matters.
- **Does diversity (k-junta) transfer to `add`?** k-juntas are low-order; add is a specific structured
  function. The OOD-library eval tests this directly, but the train distribution may need add-like
  structure (parametric arithmetic family?) to induce the right algorithm.
- **Memory bottleneck:** if state must hold the running function, how big? (predicted load-bearing —
  but measure).
- Keep the reverse control honest; don't over-read 1-seed signals (grok is seed-noisy).
