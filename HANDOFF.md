# HANDOFF — boolean_nca_cc (as of 2026-06-17)

Single entry point for the next session. Read this first, then the linked docs only
as needed. Two arcs: **demo (DONE, shipped)** and **arithmetic (the live frontier)**.

═══════════════════════════════════════════════════════════════════════════════
## ⟢⟢⟢⟢⟢⟢⟢ 2026-06-17 — REFACTOR+LOSS-FIX DONE → CONTINUUM RAN (VERDICT) → CLEANUP → W=1 HORIZON SWEEP LIVE
**STATUS (latest): continuum_v1 COMPLETE (16/16). VERDICT: `eval_online_out_reverse` RECOVERS with W (stream
.58/.79/.81/.78, fixed .50/.52/.82/.85 over W{1,8,64,256}; tops ~0.85, NOT the old full-table batch 0.92);
`eval_online_out_add` FLAT CHANCE all W/modes → add bottleneck = STRUCTURAL / feedback-channel (next lever =
value-aware feedback / Perceiver). Eval note: `eval_out_task` (batch-adapt OOD) ~0.60 vs `eval_online_out_reverse`
(stream-adapt) 0.81 on the SAME full-table data (x_task "task" scenario) = pure batch-vs-stream regime gap.
CLEANUP DONE: removed dead `loss_fn_no_scan`; pool `expected_updates` fix (feeds n_rounds×steps_per_round, was 32× off
on deep cells); KEPT `run_model_scan_with_loss` as the damage/shuffle engine; 45/45 tests pass. DEFERRED cleanup
(needs review): route no-damage batch through run_inner_loop to make scan damage-only.
NOW LIVE: **W=1 HORIZON sweep** — array `4277117` (slurm `sweep_20260617_210343`, group `w1_horizon_v1`), 10 cells
uzh-v100, tmux `hsm_horizon` = {reverse,add}×(n_rounds,steps_per_round){(8,1),(64,1),(256,1),(8,8),(32,8)}, mb=64 for
T=256. Q: does pure-online W=1 reverse train with longer total T (n_rounds×steps_per_round)? Readout
`eval_online_out_reverse` vs T.**
`UNIFIED_REFACTOR_PLAN.md` = the (executed) refactor spec. The pre-refactor verdict that motivated it is below.

### THE VERDICT (online VSML, pre-refactor — the WHY)
- **online learned `reverse` only to ~0.60** (COARSE, like `add`'s 0.73 plateau), NOT batch's ~0.92; **`add`
  stayed chance.** Depth(S=64)/k=8/un-averaged grads/data-batching all RULED OUT; deep BPTT was stable.
  ⇒ regime SOUND but caps COARSE → suspected bottleneck = the **FEEDBACK CHANNEL** (value-blind nodes + one
  output-only scalar residual). The continuum sweep tests this directly.

### WHAT SHIPPED (this session — all validated)
1. **Unified inner loop** `evaluation.py::run_inner_loop` — one body (`m message steps → get_loss_and_update_graph
   on the round's W-subset`), dials {`window_size` W, `data_mode` fixed|stream, `steps_per_round` m, `n_rounds` N,
   `loss_reduction`}. `predict_before_adapt` DROPPED (emerges from data_mode=stream). Reuses `get_loss_and_update_graph`
   (its `jp.abs(res).mean(axis=0)` IS the mean-over-W) → `fixed,W=full,m=1` is **byte-identical** to the batch scan's
   no-damage core (regression test `test_batch_preset_matches_scan` PASSES). Online eval → `evaluate_model_inner_loop_batched`
   (+`chunk_size` over OOD circuits = the OOM fix for 256-round-from-fresh).
2. **Loss fix** `circuits/train.py`: `generalized_bce`/`res2loss` `jp.sum`→`jp.mean` → W-invariant loss magnitude.
   **No lr/clip retune needed**: meta-opt is `adamw`(scale-invariant)+`clip_by_global_norm`(uniform)+decoupled wd →
   lr=2e-4 transfers (confirmed by smokes: stream W=1 and fixed W=32 both ~O(0.9) loss).
3. **Wiring** (`train_loop.py`/`train.py`/`config.yaml`): regime `batch`(legacy/demo, run_model_scan_with_loss
   UNTOUCHED) | `unified`(continuum). Dials renamed clean (`stream_length→n_rounds`, `nca_steps_per_example→
   steps_per_round`, `online_window_size→window_size`, `online_loss_reduction→loss_reduction`, +`data_mode`). train.py
   keeps legacy-key fallbacks (`inner_loop_regime: online`→unified+stream). `eval.online_chunk_size` added.
4. **Validated**: 45/45 `tests/test_meta_tasks.py` (incl. batch regression); 3 train.py smokes (stream/fixed/legacy);
   VRAM probe heaviest cell (twelve,W=256,stream,mb128) = **16.4 GiB** (V100-safe); readout hook `[online OOD]`
   fires under unified (eval_online_out_{reverse,add}).

### THE LIVE SWEEP — monitor this
- **`sweep_continuum_v1.yaml` → array `4256926` = slurm `sweep_20260617_093719`, wandb group `continuum_v1`,
  16 cells RUNNING on uzh-v100** (tmux `hsm_continuum`, resumable, chunk-walltime 19h). Cells =
  {single_reverse(canary), single_add(target)} × W{1,8,64,256} × {stream→mean, fixed→mean_tail}.
- **READOUT** (wandb group continuum_v1, every eval.interval=2048): `eval_online_out_reverse/final_hard_accuracy`
  vs (W, data_mode). **Does the canary recover ~0.92 at high W / fixed?** → per-step gradient RICHNESS was the
  0.60 cap (continuum validated). **Stay ~0.60 in stream at all W?** → the **FEEDBACK CHANNEL** is the bottleneck
  → next lever = value-aware node feedback / Perceiver cross-attn-to-data (a MODEL change, STRUCTURAL_DIRECTION §4).
  Plus `eval_online_out_add` (the prize). Eval = 256-NCA-step-from-fresh.
- OPS: logs `/scratch/gbena/hsm-runs/boolean_nca_cc/sweeps/sweep_20260617_093719/logs/`. Monitor `squeue -u gbena`
  (name `sweep_20260617_093719_array`). To stop: `kill -9` the tmux `hsm_continuum` launcher FIRST, then `scancel 4256926`.
- **NEVER touch Comp-PVR** (`4251807`/073214, `4227133`/172406) — different project, leave alone.
- v2/bptt/W-probe all STOPPED before the refactor (clean). Built earlier + KEPT: arith_family sampler,
  `binary_subtract`("sub"), `configs/tasks/{arith_family,single_reverse,single_add}.yaml`.

═══════════════════════════════════════════════════════════════════════════════
## ⟢⟢⟢⟢⟢⟢ 2026-06-16 — ONLINE (VSML) inner loop BUILT, TESTED, RUNNING. Task-agnostic SETUP achieved; capability NOT yet demonstrated.
The STRUCTURAL_DIRECTION online half is implemented (~all additive, batch path byte-for-byte intact,
44/44 tests green) and a longer campaign is live. **Read the conceptual frame below before the ops.**

### CONCEPTUAL STATUS — what regime are we actually in? (answers the obvious question)
- **We MOVED from single-task to the task-agnostic SETUP.** Original SODC = one fixed function per run
  (add / reverse), batch inner loop → the shared rule could *bake in* that one function = MEMORISE.
  Now the **task-sampler** binds a DIFFERENT sampled boolean function to each pool slot (re-sampled on
  reset), and a meta-batch step sees `meta_batch_size` DISTINCT tasks. Plus the **online** inner loop:
  each circuit learns its task from a STREAM of examples (predict-before-adapt). Diverse tasks +
  weight-shared rule + online adaptation = **the VSML recipe** (Kirsch 2021). So in FORM this is genuine
  task-agnostic meta-learning — a big step up, not single-task-overfit.
- **BUT we have NOT yet demonstrated task-agnostic CAPABILITY.** Two distributions:
  - `arith_family` = add under a random INPUT-BIT PERMUTATION → a NARROW family. The rule trains on it
    (~0.70 in-dist) but "learn the add family" ≈ an add-ish prior, closer to family-specialisation than
    general "infer any function." Its genuinely held-out OOD = `sub` / `binary_multiply` (different ops,
    never trained); canonical `add` is the *identity-permutation member* = WITHIN-family transfer, NOT OOD.
  - `k_junta` = a fresh random k-junta per circuit → HIGH-ENTROPY, the genuine task-agnostic test. **It
    does NOT train (dead-flat chance) in ANY regime/budget tried.** ⇒ true agnostic capability not achieved.
  - Verdict so far: the agnostic *machinery* runs; whether online *develops* a general optimiser is the
    open question v2 tests. Early v2 reads (50%): arith held-out OOD (sub/multiply) = chance, flat. The
    user's intuition is exactly right — online *could* develop agnostic optimisation; we don't see it yet.

### THE STREAMING SETUP (how the online inner loop actually works) — `evaluation.py:run_model_online_scan_with_loss`
- A task = a boolean FUNCTION stored as its FULL truth table: `x_task` = all 2^input_n input patterns
  (`twelve` → 4096), `y_task[i]` = the function's output per input. (`tasks/samplers.py`.)
- The **stream** = S example indices drawn uniform-with-replacement from the truth table. Per position s:
  (a) **PREDICT** — run the circuit on x_s with the CURRENT LUT logits, record online loss L_s (predict
  BEFORE adapting on x_s = the VSML generalisation signal); (b) **ADAPT** — feed the PER-EXAMPLE residual
  `[output_n]` (NOT the batch-mean — the original regime's `jp.abs(residuals).mean(axis=0)`) to the output
  nodes, then run `k = nca_steps_per_example` NCA message-passing steps. The per-node hidden state carries
  the running "in-context-learned function" across positions.
- **Meta-objective** = mean (default) of the S online prediction losses; BPTT through the whole stream.
- **Few-shot generalisation is the hard part:** S=64 examples seen out of 4096 inputs → the circuit must
  generalise to the ~98% of inputs it never streamed. OOD eval (`evaluate_model_online_batched`) adapts a
  fresh random-wiring circuit on the subject's stream (at full S), then reads **hard-acc on the FULL table**.
- Both inputs AND targets stream (supervised online learning): predict y, get true y as feedback, adapt.

### THE CURRICULUM (`stream_length_schedule`) — a TRAINING aid, not a task curriculum
- `stream_length_schedule=[[epoch_frac,S],...]` ramps the inner-loop stream length over training, e.g.
  S: 8→16→32→64. Short streams early are CHEAP (less BPTT depth/compute) and easier (fewer examples to
  integrate); lengthen as it learns to integrate more → longer in-context horizon. Mitigates the
  vanishing-credit / cost of BPTT through long streams (STRUCTURAL_DIRECTION §8). Stepped (few levels) →
  the jitted train step recompiles only a handful of times (threaded as a static arg, like n_message_steps).
  EVAL always uses the MAX S (full horizon) for comparable readouts. Resume-safe (epoch-based).

### v1 RESULT (2026-06-15 eve, `sweep_20260615_224502`, COMPLETE) — NEGATIVE, with diagnosis
4 cells {online,batch}×{k_junta,arith_family}, 2^15, S=64, k=1, seed 0. Read:
- online·arith relearned add to **0.72 = the plateau** (but add ∈ its train family → within-family, not OOD);
  **reverse collapsed to chance** (arith-trained rule = narrow add-prior, can't route).
- online·k_junta = **chance in-dist** (learned nothing). Stage-0 `data_per_batch` 256→1 = FLAT 0.70–0.74
  (smaller batch ≠ the lever). Metric was padding-confounded (parity/multiply inflated by zero-pad bits).
- Causes: under-power (2^15), compromised OOD (add in arith family), padding-confounded metric.

### THE 5 FIXES (this session — all additive; default `inner_loop_regime: batch` = byte-for-byte unchanged)
1. **Longer + stream-length CURRICULUM** — `stream_length_schedule` (above); v2 runs 2^17 (resumable).
2. **k_junta diagnosed** (local A6000): batch k=1 AND k=4 AND online k=1 are ALL **dead-flat chance, zero
   creep** → NOT under-training; batch structurally can't in-context-learn a *resampled* per-task target
   from coarse feedback (reverse/add/arith "work" only because fixed/narrow). ⇒ tempers the k_junta prior
   (may stay flat even at 2^17); arith+held-out-OOD is the robust test.
3. **Genuine held-out arithmetic OOD** — new `sub` library task (`circuits/tasks.py`) + arith_family `op`
   set (add/sub); π-identity bit-matches the library for BOTH. arith eval_ood = {add=within-family,
   **sub + binary_multiply = TRUE held-out OOD**, reverse=routing control}. (`configs/tasks/arith_family.yaml`.)
4. **Recipe knobs** — `nca_steps_per_example` k>1 (carry depth/example), larger S via curriculum, lr /
   `online_loss_reduction` (mean|sum) exposed.
5. **Padding-corrected metric** — `evaluate_model_online_batched(score_output_bits=...)` scores only the
   native (real) output bits per subject; `reverse` (full width) = the honest anchor. (`train_loop.py:_native_output_width`.)
- Files: `training/evaluation.py` (`run_model_online_scan_with_loss`, `evaluate_model_online_batched`),
  `training/train_loop.py` (`loss_fn_online`, `_resolve_stream_length`, `_native_output_width`, per-subject
  online OOD eval block, stream_length threaded as a static arg), `tasks/samplers.py` + `tasks/registry.py`
  + `circuits/tasks.py` (arith op-set + `sub`), `train.py` (`_parse_stream_length_schedule` + threading),
  `configs/config.yaml`, `configs/tasks/{k_junta,arith_family}.yaml`, `tests/test_meta_tasks.py` (44 green).

### v2 CAMPAIGN — RUNNING (the real test) — `sweeps/sweep_online_v2.yaml`
- Slurm array **4207603** (`sweep_20260616_083618`), W&B group **`online_v2`**, uzh-v100, **RESUMABLE
  (19h chunks)**, tmux launcher **`hsm_online_v2`** on anahita. 4 cells {online,batch}×{arith_family(op=add),
  k_junta(k=4)}, circuit=twelve/h64/attn128/dist_pe, **2^17**, curriculum S 8→16→32→64, **k=2**, damage off,
  eval (the per-subject online OOD readout fires every periodic-eval).
- **As of 2026-06-16 ~17:05 (~50% / epoch ~65536 on the arith-online cell):** healthy, progressing,
  checkpointing. arith-online held-out OOD `sub`=0.50 / `binary_multiply`=0.53 / `add`(within-family)=0.53 /
  `reverse`=0.55 — **all ~chance, FLAT since the first eval point.** No breakout yet.
- **READOUT (the adjudicator): `eval_online_out_{sub,binary_multiply}/final_hard_accuracy`** = does
  add-structured online meta-learning generalise to NEW arithmetic? k_junta arms = does online+curriculum+
  long train generic diversity at all? Reproduce any breakout (>~0.6) at seeds 1,2.
- **GOTCHA — read PROGRESS from the LOGS, not the tqdm bar:** the tqdm `N/131072` line is `\r`-garbled in
  the redirected file (showed a stale `4096` while the run was actually at epoch 65536 — nearly caused a
  false "hung" kill). Authoritative epoch = the latest `Unified Eval (epoch N)` / `Saved resume checkpoint
  at epoch N` log lines under `tasks/task_*/wandb/*/files/output.log`.

### NEXT (if v2 confirms the early flat read — i.e., online doesn't cross via budget/curriculum)
The diagnosis points past "more epochs" to the **FEEDBACK CHANNEL** (coarse scalar residual to OUTPUT nodes
only; internal carry nodes value-blind). The Stage-3 fallbacks in STRUCTURAL_DIRECTION §4 become the lever:
**value-aware node feedback** (concat per-node circuit activations / functional fingerprint into
`extract_node_features` `base.py:588`), loss reshaping, output→input cross-attention. These are NOT built yet.
If a held-out-OOD breakout DOES appear → reproduce at seeds 1,2, then ablate (state size, diversity, S, k).

### OPS / LIVE
- v2 (array 4207603) on uzh-v100, resumable — **the tmux `hsm_online_v2` launcher must advance the chain at
  the ~19h chunk boundary** (job started 08:36 → ~03:36 next day) or `hsm sweep advance sweep_20260616_083618`.
- Stage-0 `data_granularity` (`sweep_20260615_220930`) + v1 `online_v1` (`sweep_20260615_224502`): COMPLETE
  (v1 in `/shares/payvand.ini.uzh/hsm-archive/`). Co-project **Comp-PVR** also runs under `gbena` — leave alone.
- A monitor cadence is armed for v2 (reads logs not tqdm; reports on breakout/failure/chunk-transition/done).
- **Stale pre-existing tests** (NOT from this work) block `pytest tests/` whole-dir collection:
  `test_gradient_fix.py`, `test_multi_step_gradients.py` (import removed `get_loss_from_graph`),
  `tests/knockout/test_*`. Run `pytest tests/test_meta_tasks.py -o addopts=""` directly (green).

═══════════════════════════════════════════════════════════════════════════════
## ⟢⟢⟢⟢⟢ 2026-06-15 — ARITHMETIC ARC: capacity ruled out → STRUCTURAL pivot (online + task-sampler)
Clean-transmission handoff. **Read `STRUCTURAL_DIRECTION.md` next — the full design + experiment plan
for the new direction.** Nothing is running on the cluster; this is a design-state handoff.

### WHERE WE ARE (the arithmetic arc, one breath)
`add` on random wirings is stuck at a ~0.73 coarse plateau. This session systematically ruled out the
within-architecture knob-space and located the gap as **structural**:
- **Crossed grid** (T8/T16, full 2^18 budget): add flat 0.73–0.74 at every T/lr/hidden/±damage.
- **Fork-#1:** the plateau is REAL, not a step-256 metric artifact (within-rollout curve is flat).
- **Stability axis:** DEAD END (warmup/clip don't help; instability self-resolves with training).
- **Capacity plane** (add × attn{128,256,512} × h{64,128,256}, 2026-06-14): FLAT 0.73–0.75, max 0.749 —
  **capacity is NOT the lever.** Params and state each move add ~0.000.
- ⇒ add's plateau ≠ measurement ≠ stability ≠ state ≠ capacity → **STRUCTURAL.** (Detail: the 2026-06-14
  + 2026-06-10 blocks below.)

### THE DIRECTION (do this next) → `STRUCTURAL_DIRECTION.md`
**Thesis:** the plateau is a REGIME artifact. The inner loop is a BATCH meta-gradient (256-input
aggregated residual per NCA step) → value-blind nodes + averaged-away carry structure + single-task
MEMORISATION. The unifying fix: **diverse tasks (task-sampler) + an ONLINE/sequential inner loop**
(learn the function one example at a time) = the **VSML recipe** (Kirsch 2021, proven topology-
agnostic). SODC already has the weight-sharing half + a task-sampler (diversity, batch); it is missing
the **online** half.
- **The task-sampler EXISTS and is runnable** (`configs/tasks/k_junta.yaml`; samples boolean FUNCTIONS
  — k-junta train / library OOD incl. `add`; tests pass) **but is BATCH.** Going online ≈ **~200 LOC**
  (per-sample inner loop + online prediction loss + a `inner_loop_regime: batch|online` flag).
- **Prediction / control:** online should lift `add` (value-dependent compute) and leave `reverse`
  (routing) flat; **state should become load-bearing online** (re-opening the capacity null as
  regime-specific). reverse↔add is still the adjudicating readout.
- **Staged plan:** (0) cheap probes — `data_per_batch` 256→1 granularity sweep + mini-stream; (1) the
  online inner loop on the task-sampler, eval OOD on `add`; (2) ablations (state, diversity, stream
  length); (3) feature-level fallbacks (value-aware node feedback / loss reshaping / output→input
  cross-attention). Full design + injection-point table in `STRUCTURAL_DIRECTION.md`.

### LIVE / OPS STATE (no regression)
- **NOTHING running.** The capacity sweep (`sweep_20260611_094613`) is COMPLETE + closed out (launcher
  killed, monitor `bc4l3q7kv` stopped, tmux gone). No monitors armed; cluster idle for us.
- **Cluster:** `ssh uzh`. Remote `uzh` = A100 `standard/medium` (CONTENDS with co-project Comp-PVR —
  hundreds of jobs; expect queueing). Remote `uzh-v100` = V100 `lowprio` (FREE fallback, T≤8, 20h chunks).
- **train.py allocator FIX (in place):** `XLA_PYTHON_CLIENT_ALLOCATOR` is now env-controllable
  (`setdefault(...,"default")`); the old ineffective `TF_GPU_ALLOCATOR` line is gone. The uzh-v100
  pre_script in `.hsm/config.yaml` has `export XLA_PYTHON_CLIENT_ALLOCATOR=platform` ADDED (for tight
  V100 resumes; **revert it for non-tight V100 sweeps** — ~1.5× slower).
- **GOTCHAS (don't rediscover):** (1) VRAM-probe with the SAME allocator you'll LAUNCH with (BFC≠platform
  cost a relaunch). (2) h256 OOMs on 32GB V100 at LONG-TRAINING pool-reset spikes even with platform (a
  short probe misses it) → h256 needs A100 or `gradient_checkpointing` for full budget. (3) the cluster
  monitor reads only the latest run-dir → chunk-boundary blind spot; verify breakouts via the wandb API
  across chunks. (4) HSM launchers ignore SIGTERM / resubmit on scancel → `kill -9` the launcher (match
  the python proc, NOT your shell) BEFORE scancel. (5) HSM's no-progress heuristic correctly fails a
  deterministic-crash chain after 2 chunks. HSM issue #16 filed.

### DOC MAP
- **`STRUCTURAL_DIRECTION.md`** (NEW) — the next direction: diagnosis (grounded file:line), online/VSML
  thesis, task-sampler state+gap, design, staged experiments, injection points, wiki refs. **READ AFTER THIS.**
- `KIRSCH_PROBE_PLAN.md` — the GPICL frame + the (now-answered) capacity campaign design.
- `sweeps/HSM_NOTE_resume_oom.md` — resume-OOM write-up + HSM issue #16.
- the 2026-06-14 / 2026-06-10 blocks below — capacity verdict, crossed-grid verdict, fork-#1, stability
  dead-end, resume-OOM fix, the train.py allocator details.

═══════════════════════════════════════════════════════════════════════════════
## ⟢⟢⟢⟢ 2026-06-14 EOD — CAPACITY PLANE RAN → capacity is NOT the lever (add's gap is STRUCTURAL)
The headline experiment of the arithmetic arc completed. Clean NEGATIVE result: more capacity
does not unlock random-topology `add`.

### VERDICT: capacity is NOT the lever for `add`.
Sweep `sweep_20260611_094613` (V100), add × attention_dim{128,256,512} × hidden{64,128,256}, T8,
seed 0. Final add plane (peak eval_out hard-acc; breakout line 0.76):
```
         h64     h128    h256
attn128  0.736   0.735   0.735
attn256  0.739   0.739   0.735
attn512  0.749   0.740   0.731   (h256 cell ~30%, rest full/80% budget)
```
- **FLAT at 0.73–0.75, max 0.749 — no cell near 0.76.** Quadrupling params (attn 128→512) buys a
  marginal coarse bump (~+0.01 at h64) that washes out at high state; state (h64→256) does ~nothing.
  Neither unlocks precise arithmetic.
- Reverse anchors generalise regardless of capacity: ref(attn128/h64)=0.924, max(attn512/h256)=0.877.
  The reverse↔add gap (~0.92 vs ~0.74) is huge and capacity-INDEPENDENT.
- **⇒ The whole within-architecture knob-space is ruled out** for add's plateau: NOT measurement
  (fork-#1), NOT stability (dead-end), NOT state-alone, NOT params/capacity. The missing ingredient
  is **STRUCTURAL** — what lets BP solve add exactly on a fixed topology isn't a capacity/compute
  deficit the meta-learner can close with more resources.

### NEXT (the pivot — your design call):
- **Diversity + curriculum** axes (`KIRSCH_PROBE_PLAN.md` §4): the last untested GPICL gates
  (task-diversity, memorise-first curriculum). OR
- **A structural rethink**: random-topology arithmetic may need an architectural change (explicit
  carry-routing, a different readout, a wiring curriculum), not a knob. Diagnosis now points here —
  worth a design session before more sweeps.
- `multiply` remains the harder untouched subject.

### LAUNCH SAGA — durable ops lessons (the campaign took 3 relaunches):
1. **A100 `standard/medium` contends with co-project Comp-PVR** (same gbena user, 100s of jobs) →
   capacity sweep queues behind it. **V100 `lowprio` is the free fallback** (idle nodes), remote `uzh-v100`.
2. **VRAM-probe with the SAME allocator you'll launch with.** Probed attn512×h256 at 22.9GiB with
   `platform` but launched BFC → BFC's ~7GiB pooling overhead OOM'd h256 on the 32GB V100. Added
   `XLA_PYTHON_CLIENT_ALLOCATOR=platform` to the uzh-v100 pre_script (`.hsm/config.yaml`).
3. **Even platform isn't enough for h256 on V100 over LONG training** — a pool-reset memory spike
   transiently exceeds 32GB ~80% in; a 16-epoch probe never reaches it. h256 needs **A100 or
   `gradient_checkpointing`** for full budget (80% data already lands on the plateau, so the verdict
   didn't need them).
4. **Monitor chunk-boundary blind spot:** `~/monitor_capacity.sh` reads only the latest run-dir, so a
   breakout in a PAST chunk could be missed → verify add-breakouts via the **wandb API across chunks**
   (`/tmp/cap_plane.py` pattern), not just the live HB.

### LIVE / OPS:
- **`sweep_20260611_094613` essentially COMPLETE:** 8/9 add cells at full/80% budget, last
  (attn512×h256) running ~30% at plateau (may OOM). Both reverse anchors done. Group `capacity_plane`,
  durable launcher tmux `hsm_capacity`, harness Monitor `bc4l3q7kv` (session-bound, re-arm via
  `~/monitor_capacity.sh`; QUIET/event-only, scoped).
- W&B `capacity_plane` group has old BFC/OOM crash dupes (n=3–6) — cosmetic, deletable.
- The 2026-06-10 block below has the crossed-grid verdict, fork-#1, stability dead-end, resume-OOM
  fix (train.py allocator), and HSM issue #16.

═══════════════════════════════════════════════════════════════════════════════
## ⟢⟢⟢ 2026-06-10 EOD — crossed grid COMPLETE; verdict + 3 findings; capacity campaign drafted
Supersedes the 2026-06-09 blocks. The crossed {reverse,add} grid ran to completion;
add's verdict is in; the resume-OOM was diagnosed+fixed; capacity-plane sweep drafted (not launched).

### VERDICT (headline): `add` does NOT grok.
- **T8 (8 arms) + T16 (4 arms) trained the FULL 2^18 budget — add stuck at 0.73–0.74
  everywhere** (every T, lr, hidden, ±damage). Reverse generalises (T8-reservoir 0.94;
  T16 0.87→0.80 held). The reverse↔add gap is large and robust.
- **Fork-#1 (settle-window re-read): the plateau is REAL, not a step-256 artifact.** add's
  within-rollout curve is FLAT across all 256 steps (viewed the stepwise PNGs) — no hidden
  settle peak. So step-256 is faithful for add; it's a LEARNING failure, not measurement.
- **Stability axis is a DEAD END.** probe (warmup×5+clip0.5) vs grid (default), rev/T16:
  liftoff 2× SLOWER (idx 64 vs 32), peak LOWER (0.849 vs 0.868), SAME oscillation (42
  big-drops each). Reverse-T16 instability is INTRINSIC and SELF-RESOLVES with training
  (grid end-std 0.002); add has ~0 oscillation (stable-but-stuck). ⇒ don't chase stability
  with warmup/clip — starve T + push CAPACITY.
- ⇒ Only untested lever = **PARAMS (`attention_dim`, held at 128 all along)**. State alone
  (h64→h128) moved add 0.000. Next experiment = the capacity plane (below).

### RESUME-OOM (diagnosed + FIXED) — the chunk-boundary incident.
- chunk-1→2 transition WORKED (faithful resume verified, correct epochs). But **all 4 T32
  arms OOM'd on RESUME** (RESOURCE_EXHAUSTED, 13.08GiB contiguous) though the same config ran
  fine FRESH 23h. Cause: **BFC fragmentation** — on resume the 1.3GB checkpoint (pool+optim)
  is on-device before the first step's 13GB rollout buffer is requested → no contiguous hole.
- **Smoking gun (FIXED in train.py):** the intended fix was misconfigured — `TF_GPU_ALLOCATOR`
  is a TF var JAX IGNORES (allocator stayed `GPU_0_bfc`); `XLA_PYTHON_CLIENT_ALLOCATOR=platform`
  was commented out. train.py now env-controls it (`setdefault(...,"default")`) → set
  `=platform` (or `cuda_async`) in the pre_script for tight V100 resumes. **Cluster-specific:**
  V100-32GB × T32 × resume only (fits on A100). 2nd lever: `gradient_checkpointing` (default False).
- **HSM behaved CORRECTLY** (corrected an over-claim): it detected "2 consecutive chunks made no
  progress … likely deterministic crash" and FAILED the chain after 4 chunks (not an infinite
  loop). Residual nit only: OOM masked as SLURM COMPLETED (inner exit not propagated) → caught by
  the 2-chunk no-progress heuristic, not a direct FAILED. → `sweeps/HSM_NOTE_resume_oom.md`.
- T8+T16 results safe; **T32 abandoned at epoch 93,822 (checkpoints intact** in
  tasks/task_{5,6,11,12}/resume/) — recoverable via fresh relaunch + allocator fix (low value).
  12 OOM .err files archived to logs/_oom_chunk2/.

### NEXT: capacity-plane campaign (drafted, NOT launched).
- **`sweeps/sweep_capacity_plane_v1.yaml`** — add × attention_dim{128,256,512} ×
  circuit_hidden_dim{64,128,256} = 9 + 2 reverse anchors, T8, proven recipe, seed 0 (screen).
  **Before launch:** VRAM-probe corners (h256/attn512 exceed 32GB V100 → A100), decide V100/A100
  routing, set allocator=platform for V100 resumable arms, add seeds 1,2 at any cell >0.76.
  Header + KIRSCH_PROBE_PLAN.md §4–5 have the full design + staging.

### LIVE / OPS.
- **Grid chain ENDED** (launcher dead, FAILED-after-4-chunks — correct). **Probe still RUNNING**
  (sweep_20260609_115537, array 3907085, T16, low-value warmup×5 — fine to finish or kill).
- **Monitor** scripts (~/monitor_crossed.sh + ~/wandb_instab.py) still hardcode the crossed+probe
  sweep IDs — EDIT for the capacity sweep at launch; re-arm a session Monitor (session-bound). Two
  monitors were doubled-up this session (bvyz4iodt + sibling bpjeb1ynm) — consolidate to one.

═══════════════════════════════════════════════════════════════════════════════
## ⟢⟢ 2026-06-09 EOD — FULL SESSION HANDOFF (latest; supersedes the ⟢ block below)
Launched the controlled experiment, hit+fixed infra, reframed the whole arc via Kirsch
GPICL. **Read §A (ops) FIRST — there is live compute that needs tending.**

### A. OPERATIONAL STATE — what's live + how to keep it alive
- **RUNNING (cluster uzh-v100):**
  - **Crossed grid** — Slurm array **3874375** (sweep `sweep_20260609_052548`, W&B group
    `crossed_rev_add`), 16 tasks, ~12h into chunk-1 (23h cap). 12-bit add+reverse × T{8,16,32}
    × lr{2e-4,3e-4} + reservoir(h128)/−damage ablations. YAML: `sweeps/sweep_crossed_reverse_add_v1.yaml`.
  - **Warmup+clip probe** — array **3878363** (sweep `sweep_20260609_115537`, group
    `warmupclip_probe`), 2 tasks (rev+add T16, warmup_factor=5 + grad_clip=0.5). YAML:
    `sweeps/sweep_warmupclip_probe.yaml`.
- **DRIVEN BY tmux launchers ON ANAHITA** (not the agent session): `tmux attach -t hsm_crossed` /
  `hsm_probe`. **These persist across agent sessions** and advance the resumable chunks; they die
  only if anahita reboots or tmux is killed. **The chains need a live driver at the ~T+23h chunk
  boundary or chunk-2 won't submit.** If stalled: `conda activate bool_nca && hsm sweep advance
  <sweep_id>` (re-drives; run under tmux/nohup). NB HSM launchers ignore SIGTERM and resubmit on
  scancel → `kill -9` the launcher BEFORE scancel to actually stop a chain.
- **MONITOR is session-bound — RE-ARM IT (first action).** Scripts persist on the cluster:
  `~/monitor_crossed.sh` (event-driven: job states, NaN, fails, soft-instability, add-breakout>0.76,
  reverse>0.90, hourly pulse) + `~/wandb_instab.py` (wandb-API eval-drop detector, both groups).
  Re-arm = a persistent Monitor running `ssh uzh 'bash ~/monitor_crossed.sh'` every ~20min.
  **Both scripts hardcode the two sweep IDs above — edit them for new sweeps.**
- **GPUs:** anahita GPU0 off-limits for auto-work; A6000s (CUDA2/3) had co-tenants (tom/christian);
  CUDA1 (4090) free for probes. Cluster: V100 lowprio 32GB (~40 idle), A100 40–80GB contended.
  **Local wandb API works** (used to download stepwise PNGs).

### B. SCIENCE SNAPSHOT (eval_out OOD hard-acc, peak→latest, ~64% budget)
- **Reverse (CONTROL) generalises to near-exact:** best = rev/T8/**h128** (reservoir) **0.93** peak.
  The reservoir arm is highest → first in-data hint **accessible state helps** (Kirsch Insight 4).
- **High-T reverse PEAKS THEN PEELS BACK (instability):** rev/T16 0.83–0.87→0.69–0.73, rev/T32
  0.835→0.685. Lift off but can't hold — train-loss instability + rollout drift; worse at high T / hot lr.
- **ALL `add` arms stuck at ~0.73 coarse plateau** — no breakout, uniform across capacity/T/lr. THE
  open question: does add grok past it? Not yet at 64% budget.
- **Probe pending + a finding:** warmup_factor=5 (20% warmup) made probe-rev crawl (still 0.60 at 5h,
  pre-liftoff). **factor 5 is too aggressive — try ~10–12 next.**

### C. KEY REFRAMES this session
1. **Whole arc = Kirsch GPICL memorise→generalise transition** (you deep-read it; wiki ref
   `wiki/refs/kirsch-2022-general-purpose-icl`; synthesis captured to staging
   `2026-06-09-gpicl-sodc-add-plateau-transition.md`). add=memorised, reverse=generalised; transition
   gated by capacity (params AND state), task diversity, meta-opt STABILITY (§4.3 = grokking).
   **Full frame + the "total Kirsch sweep" design → `KIRSCH_PROBE_PLAN.md`.**
2. **"damage off for learnability" was WRONG** — burst-damage is load-bearing; grid uses damage-on
   anchor + −damage ablation.
3. **Eval-rollout dynamics (real):** the 256-step eval develops a PEAK-THEN-DRIFT LIMIT CYCLE as
   training plateaus (settles early, loses the fixed point late). ⇒ **`eval_out_test/final_hard_accuracy`
   (step-256) is CONFOUNDED — use peak/settle-window (~rollout-step 30), cf. demo_probe settled-acc.**
   Stepwise eval is logged as PNG plots (`stepwise/eval_out_test`); download+view them.
4. **Capacity = params AND state** (Fig 2 model-size + Insight 4 state). attention_dim held fixed =
   untested lever. Clean test: attention_dim (params) × hidden (state), orthogonal.

### D. GOTCHAS (don't rediscover)
- **OOM = BFC fragmentation** (not true OOM): train.py forces `XLA_PYTHON_CLIENT_PREALLOCATE=false` →
  fragments on 32GB V100 for big contiguous allocs (T32×h128, ~13GB). Avoid that corner OR set
  `XLA_PYTHON_CLIENT_ALLOCATOR=platform` (verified, ~1.5× slower). **Size VRAM with a hard
  `XLA_PYTHON_CLIENT_MEM_FRACTION` cap, NOT nvidia-smi@1s (misses the transient compile peak).**
- **W&B 150GB bloat FIXED** (`checkpointing.py`): resume `latest_checkpoint.pkl` (~2.5GB pool) now
  LOCAL-only (filename-gated); only small `best_model_*` upload. Verified offline.
- **HSM issue #15 filed** (GabrielBena/HPC-Sweep-Manager): crash-reported-as-COMPLETED; pre_script
  `module` no-op in non-login shell; launcher-must-stay-alive.
- **New config knobs:** `training.grad_clip_norm` (default 1.0) and warmup via
  `training.lr_scheduler_params.cosine.warmup_steps_factor` (default 50; LOWER = longer warmup).

### E. NEXT ACTIONS (in order)
1. **Re-arm the monitor** (§A).
2. **Confirm the chains advanced** past the ~T+23h chunk boundary; `hsm sweep advance <id>` if stalled.
3. **Watch:** any add arm >0.76 = THE result (grok); the probe (likely relaunch with warmup_factor ~10–12).
4. **Re-read the metric on the settle-window**, not step-256 (limit-cycle confound).
5. **Kirsch-probe campaign** (`KIRSCH_PROBE_PLAN.md`): launch the cheap V100 slices once the current
   grid gives its verdict. Subjects = add + multiply; reverse = anchor only.



═══════════════════════════════════════════════════════════════════════════════
## ⟢ 2026-06-09 UPDATE — READ FIRST; supersedes parts of §1
The arithmetic arc launched, then was redesigned into a CONTROLLED experiment.

**RUNNING NOW:** a crossed **{reverse, add}** grid — `sweeps/sweep_crossed_reverse_add_v1.yaml`,
Slurm array **3874375** (`sweep_20260609_052548`), W&B group `crossed_rev_add`, on uzh-v100,
resumable chains, driven from a **tmux** session `hsm_crossed` (durable, not session-bound).
16 runs = {reverse,add} × T{8,16,32} × lr{2e-4,3e-4} + reservoir(h128) & −damage ablations.

**WHY crossed (supersedes §1's add-only plan):** add-only-at-new-params is uninterpretable.
REVERSE is the positive control (it generalises across random wirings) → the ceiling at each
setting; the **reverse↔add gap** is the readout. Every run is ONE lever off the PROVEN demo
recipe (t12/seed10: T8, mean_tail tail=.25, dist_pe, solar-burst damage 0.10, lr 2e-4, h64).
This also finally runs the long-T reverse the cancelled A100 ladder was for.

**§1's "damage OFF for learnability" was WRONG (G caught it — do not revert).** The demo
reports show training damage — esp. SOLAR-BURST — is LOAD-BEARING for settled acc + carry
(= random-topology transfer): carry 0.89 with damage vs 0.59 without. So the anchor has
burst-damage ON; **−damage is the ablation**, not the default.

**FIXES this session:**
- **OOM:** T32×h128@b128 OOMs the 32GB V100 — not a true >32GB need but **BFC fragmentation**
  (`train.py:36` forces `XLA_PYTHON_CLIENT_PREALLOCATE=false`; true peak ~17-19GB). The
  one-lever design avoids it (no T32×h128 combo) → fast BFC. If a future arm needs that corner:
  `XLA_PYTHON_CLIENT_ALLOCATOR=platform` in the pre_script fixes it (verified, ~1.5× slower).
  Probe lesson: nvidia-smi@1s MISSES the transient compile peak — size with a hard
  `XLA_PYTHON_CLIENT_MEM_FRACTION` cap instead.
- **W&B 150GB bloat (FIXED in `checkpointing.py`):** the resumable `latest_checkpoint.pkl`
  (~2.5GB full pool) was uploaded to W&B every save. Now gated by filename → latest_checkpoint
  is LOCAL-ONLY (hsm pulls it for resume); only small `best_model_*.pkl` go to W&B. Verified
  offline. **TODO: delete the existing ~150GB of `latest_checkpoint` artifacts from W&B.**
- **HSM issue #15 filed.** NEW op-lesson: HSM launchers IGNORE SIGTERM and RESUBMIT the next
  chunk on scancel → to stop a chain, `kill -9` the launcher FIRST, then scancel. Run in tmux.

**FIRST SIGNAL (now-scrapped add-only runs; suggestive only):** OOD `add` hard-acc plateaued
~0.71–0.73 (the paper's coarse-pattern regime, reproduced); NO collapse at T=32; faint
reservoir signal (h128 0.735 > h64 0.730 at T16). 1 seed, ~8% budget — not a verdict.

**OPEN:** monitor via wandb/checkpoints (slurm logs are block-BUFFERED — useless live);
analyse reverse↔add gap + lr×T; clean the W&B 150GB; spot-check the checkpoint fix on-cluster
at the first checkpoint (epoch 1031).

═══════════════════════════════════════════════════════════════════════════════
## 0. TL;DR — where we are
- **Demo arc: COMPLETE & LIVE.** The SODC live demo ships the self-healing model
  `seed10` (random wiring + solar-burst damage). Pushed to GitHub Pages
  (`gabrielbena.github.io`, commit 63c15e7); slidev embed serves it too. Do not
  re-litigate — it's done.
- **Resumable training: BUILT & VALIDATED (committed 7920087).** A >24h config can
  complete as auto-resubmitting ≤24h chunks on the preemptible V100 pool, as ONE
  continuous wandb run. Faithful resume is bit-identical (CPU A/B); the HSM
  resumable-chain contract (issue #12) is honored and 2-chunk-smoke-passed.
- **NEXT ARC (pressing issue): topology-agnostic ARITHMETIC.** Get `add`/`multiply`
  to learn PRECISE bit patterns on RANDOM wirings. This is the open problem in the
  paper itself. A first experiment is dimensioned below (§3) — not yet launched.
- **Cluster: idle.** All demo sweeps terminal; the stuck A100 ladder (3779117) was
  cancelled (superseded). No monitors running. Next session arms its own.

═══════════════════════════════════════════════════════════════════════════════
## 1. THE PRESSING ISSUE — random-topology arithmetic (the new arc)

### The problem (from the paper, `~/code/writing/gabrielbena.github.io/_posts/2026-05-06-sodc.md`)
- **Fixed topology: arithmetic already works** — add 0.96, multiply 0.84,
  ≈ BP baseline. So representational capacity is NOT the issue (circuit depth was
  dimensioned so BP solves all tasks with room to spare; don't grow depth — it
  only costs n_nodes → attention cost + V100 memory; cap at 4 hidden if ever).
- **Random topology is the frontier, and arithmetic is where it BREAKS** (paper
  §"Random Topologies", ~line 269): reverse generalizes across random graphs, but
  add/multiply "capture coarse statistical patterns… lack the precision required
  for exact arithmetic" — stuck in a coarse-loss local minimum. THIS is the target.
- It's a **LEARNABILITY/optimization problem, not a capacity one.** The NCA
  meta-learner can't FIND the config that BP finds trivially, on arbitrary wiring.
- Pool persistence / T cannot manufacture compute depth — the circuit forward pass
  is fixed-depth and the readout uses only the final logits. T/persistence are
  OPTIMIZATION budget (finding the config), not representation. (Settled with G.)

### The bet (why this isn't a rerun of the paper's failed attempt)
The paper blames the failure on the **normalised-depth-fraction PE** ("changes
resolution as layers are added, disrupting depth perception", ~line 297). We then
BUILT `dist_pe` (absolute, directional, scale-free DAG-distance PE) **specifically
to fix that** — and the paper's failing random-arithmetic runs predate it. So:
- **Lever 1 (headline): `dist_pe`** — directional carry-chain-position perception,
  the principled fix for the diagnosed cause.
- **Lever 2: long T + `loss_step_mode=mean_tail`** — carry propagation needs
  settling iterations; tail-mean is OUR stabilizer for long horizons (the v3.1 win)
  and likely the difference between T=64 collapsing and converging. Now affordable
  via resumable chains.
- **Lever 3: reservoir via `circuit_hidden_dim` (64→128)** — more NCA working
  memory PER NODE (n_nodes/attention cost unchanged → V100-friendly). G's framing:
  "more LUTs = more reservoir"; hidden_dim is the cheaper knob than width.

### DEPRIORITIZED (with reasons — don't waste arms on these first)
- **rwse**: no lift on reverse; no principled link to arithmetic (it's node
  *disambiguation*, arithmetic needs *position-in-carry-chain* = dist_pe). Off.
- **width** (more nodes): scale-free in width per paper, but grows attention +
  V100 memory; reservoir-via-hidden_dim is the cheaper test of the same idea.
- **depth**: capacity isn't the bottleneck (BP solves it); costs n_nodes. ≤4 hidden.
- **genetic wiring**: an old unproven curriculum idea; revisit only with spare time.
- **damage/shuffle**: OFF for the learnability phase (resilience regularizer = pure
  friction here; add back only once it learns).

### The dimensioned FIRST experiment (NOT yet launched — get G's nod + the VRAM probe first)
- Task `add` (the tractable arithmetic; multiply only after add moves), **random
  wiring**, **damage off**.
- Axes: `dist_pe` ON (vs the paper's no-dist_pe baseline) · **T ∈ {16,32,64} ×
  loss_step_mode=mean_tail** · **circuit_hidden_dim ∈ {64,128}**. Width held at the
  trained 264 nodes (`twelve` config). ~6–12 runs.
- **Run the long-T arms as resumable chains** (`hsm sweep run --resumable
  --chunk-walltime 23:00:00 --remote uzh-v100`) — this is the first real on-cluster
  chain test AND the experiment, in one.
- **Open dimensioning checks BEFORE launch:** (1) probe peak VRAM at the T=64 ×
  hidden_dim=128 corner on a V100 to size batch (remat defers but doesn't remove
  the T×n_nodes×batch wall — G confirmed remat has a ceiling); (2) decide whether
  to build the soft-collapse detector first (see §5 — G said OFF-SCOPE for now, so
  hand-watch the first probe).
- **Success = precise arithmetic on random wirings** (not coarse output stats). Any
  real lift over the paper's coarse-pattern result is, in G's words, "huge results."

═══════════════════════════════════════════════════════════════════════════════
## 2. WHAT'S DONE (demo arc — archival, don't redo)
- Shipped model: `seed10` (gallery `reverse_random_damage_v33`, runId `to6sec2g`),
  R2×pure-solar-burst recipe, deep 0.93/0.92, heals 64% of shotgun damage.
  Checkpoint+config durably at `/mnt/8TB_HDD/gbena/hsm-sweeps/SHIPPED_seed10_20260608/`.
- Curation: `web_demo/export/curate_shuffle_aware.py` — ALL-ROUNDER composite score
  `min(boot,shuffle,shotgun)+heal_w*climb-churn_w*churn`, robust-averaged over
  gate-sets × shuffle-targets × click-timing. This is the canonical ship pipeline.
- Demo dose: `DEFAULT_SHOTGUN_GATES=12` (= one eval volley; 2 clicks = paper 2×12).
- Campaign sweeps (all terminal, archival): v3 (3703585), v3.1 (3778421), v3.2
  burst (3820013), v3.3 harvest (3823089), calibration (3820042, FAILED rank-
  fidelity → no cheap 2^16 screening). Results on cluster /scratch + /mnt HDD.

═══════════════════════════════════════════════════════════════════════════════
## 3. WHAT'S BUILT (resumable training — committed 7920087, ready to use)
- `training.resume_from=<latest_checkpoint.pkl>` → faithful continue from epoch+1
  (model+optimizer+pool+train_key+trackers+wandb id). Back-compatible.
- wandb continuity: same run id across resume, epoch-pinned x-axis.
- HSM contract (issue #12): consumes `HSM_RESUME_FROM`, saves to `HSM_RESUME_TO`
  (SIGTERM + periodic), writes `HSM_DONE_SENTINEL` on full-budget completion.
- HSM itself shipped the orchestration (`hsm sweep run --resumable --chunk-walltime`
  + `hsm sweep advance <id>`). Our side honors the contract — verified by a 2-chunk
  smoke. Spec/audit: `CHECKPOINT_AUDIT.md`; plan+contract: `RESUME_IMPLEMENTATION_PLAN.md`.

═══════════════════════════════════════════════════════════════════════════════
## 4. DOC MAP — what each file is, when it was useful, current relevance
| File | What / when | Status |
|---|---|---|
| **HANDOFF.md** (this) | session boundary 2026-06-08 | **START HERE** |
| `RESUME_IMPLEMENTATION_PLAN.md` | resume build plan + HSM contract + Phase-2 chains | **current** (resume reference) |
| `CHECKPOINT_AUDIT.md` | what state resume must save | current (resume reference) |
| `sweeps/HSM_ISSUE_resumable_chains.md` | copy of HSM issue #12 (filed) | reference |
| `REPORT_demo_campaign_2026-06-08.md` | full demo-campaign catch-up | archival (demo arc) |
| `sweeps/ANALYSIS_demo12_v3.md` / `_v31.md` / `_v32_v33.md` | per-sweep demo analyses | archival (demo arc) |
| `SESSION_2026-06-0{3,6}.md` | detailed session logs | archival; ignore unless digging |
| `sweeps/sweep_demo_12_*.yaml` | demo campaign sweep configs | archival (demo done) |
| `sweeps/check_sweep_v3.sh`, `sweep_v3_report_*.md` | demo cron/reports | archival; **safe to delete** |
| paper: `~/code/writing/gabrielbena.github.io/_posts/2026-05-06-sodc.md` | the article + baselines | **current** (arithmetic arc context) |

**Safe to ignore for the new arc:** everything tagged "archival/demo arc" above —
it's the resilience/demo work, orthogonal to the arithmetic learnability problem.

═══════════════════════════════════════════════════════════════════════════════
## 5. KNOWN-REMAINING / OFF-SCOPE (don't start without G's go)
- **Soft-collapse detector — OFF-SCOPE per G, but a real prerequisite for the BIG
  chained sweeps.** Pushing T causes silent loss/acc COLLAPSE (not hard errors). A
  collapsed run exits clean with no done-sentinel → an HSM chain would resume a
  DEAD run for 48h. Needed: in-train health check (NaN/inf loss, or eval-acc
  sustained-drop below running-best for K evals) → mark FAILED + stop the chain (a
  marker distinct from done-sentinel; HSM caps at max_chunks but that wastes
  budget). For the FIRST (short, hand-watched) probe this isn't needed.
- **Real on-cluster `hsm --resumable` chain test** — only local + manual-env smokes
  so far. The first long-T add run doubles as this test.
- Cosmetic: train.py logs "Training completed all N epochs" even on SIGTERM
  interrupt (doesn't affect the sentinel contract; tidy when convenient).

═══════════════════════════════════════════════════════════════════════════════
## 6. INFRA CHEAT-SHEET (so you don't rediscover)
- **Env:** conda `bool_nca` (python at `/home/gbena/miniconda3/envs/bool_nca/bin/python`).
  Web demo node lives in conda env `webdemo` (node v26); web_demo src uses
  extensionless imports → esbuild-bundle for node scripts.
- **anahita GPUs (shared, multi-user):** NEVER GPU:0 for auto work. CUDA order ≠
  nvidia-smi: CUDA 0,1=4090s(smi #0,#3); CUDA 2,3=A6000s(smi #1,#2). Use CUDA 2/3
  (A6000, 48GB) for probes; ALWAYS check residency + `ps -o user= -p <pid>` before
  launching; never kill non-gbena PIDs. Co-tenants seen: christian, laura, jimmy.
- **Cluster (uzh / S3IT):** `ssh uzh`; Slurm; account payvand.ini.uzh. V100 pool =
  `lowprio`/qos `normal`, 24h cap, ~48 idle cards (the resumable target). A100 =
  `standard`/qos `medium`, 48h, contended. HSM remotes `uzh` (A100) + `uzh-v100`
  in `.hsm/config.yaml`. XLA cuda-graph flake (~3-4%) mitigated by
  `XLA_FLAGS=--xla_gpu_enable_command_buffer=` in both pre_scripts.
- **HSM:** editable at `/home/gbena/code/packages/HPC-Sweep-Manager` (G maintains;
  GabrielBena/HPC-Sweep-Manager, gh works). `hsm sweep run -c <yaml> --mode array
  --remote uzh-v100 [--resumable --chunk-walltime 23:00:00]`; cross-check via sacct
  over ssh (hsm once reported FAILED as COMPLETED). Results pull to
  `/mnt/8TB_HDD/gbena/hsm-sweeps/` + cluster `/scratch/gbena/hsm-runs/`.
- **Self-scoring:** every training run runs the in-train demo_probe → demo_* in
  final_results.csv (selection = read CSVs). For arithmetic, the relevant metric is
  plain eval hard-accuracy (precise bit patterns), not the demo_* resilience suite.
- **First action for the next session:** confirm the §1 arithmetic plan with G,
  VRAM-probe the T=64×hd=128 corner, then draft the Phase-0 `add` sweep YAML and
  eyeball it before launch. Arm a fresh persistent monitor for the new sweeps.
