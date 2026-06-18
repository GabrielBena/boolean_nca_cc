# UNIFIED INNER-LOOP REFACTOR + LOSS FIX — execution plan (2026-06-17)

Self-contained spec for the next session (context will have auto-compacted). Goal: collapse the
batch and online inner loops into ONE parameterized loop, fix the W-dependent loss-magnitude
artifact, then relaunch a clean online↔batch continuum sweep. Born from the 2026-06-16/17 online-VSML
results + a design conversation with Gabriel. **Read HANDOFF.md top block first for the verdicts/ops.**

═══════════════════════════════════════════════════════════════════════════════
## 0. WHY (the results that motivate this)
The online (VSML) inner loop was built + run (v1 sweep, bptt depth/k ablation, v2 diverse-task, a W
probe). Verdicts:
- **online learns `reverse` only to ~0.60** (COARSE — same flavor as `add`'s 0.73 plateau), NOT batch's
  ~0.92. **`add` stays at chance.** Depth (S=64), k=8, and un-averaged per-example gradients DON'T move
  it; deep BPTT did NOT destabilize (online tolerates long BPTT, unlike batch high-T). bptt (8,1) cells
  COMPLETED: reverse=0.60, add=chance.
- **v2 diverse-task online:** held-out arithmetic OOD (`sub`,`binary_multiply`) FLAT at chance; `k_junta`
  chance; arith `add`≈0.72 (within-family = the plateau).
- **Stage-0 data_per_batch 256→1:** flat 0.70–0.74 (smaller batch ≠ lever).
- ⇒ The online regime is SOUND but caps COARSE. Budget/depth/k/data-batching are ruled out → the
  bottleneck is most likely the **FEEDBACK CHANNEL** (value-blind internal nodes + a single output-only
  scalar residual), NOT the inner-loop regime. The reverse-cap-at-0.60 is the key clue.

Two design conclusions from the conversation:
1. **online↔batch is ONE continuum** parameterized by W = inputs-per-step. Batch was already a 256/4096
   (6%) subset committed PER-ROLLOUT; online commits PER-STEP with predict-before-adapt. So W up to 256 is
   still a 6% subset (still few-shot/online), not batch. The real online↔batch axis is **per-step vs
   per-rollout data commitment**, NOT W.
2. **The loss magnitude scales with W** (`jp.sum` over inputs×bits) → confounds cross-W comparison (loss
   curves + gradient-norm×grad-clip → W-dependent effective LR). Must normalize to a per-input MEAN.

═══════════════════════════════════════════════════════════════════════════════
## 1. THE UNIFIED DESIGN (Gabriel's refined version — simpler than the first draft)
ONE inner-loop function. Body of each ROUND:
```
message-steps (m) → run circuit on the data subset → feed feedback (residual to nodes + stash subset in globals)
```
Dials:
- **W** = inputs per round/eval. Unifies batch `data_per_batch`, online `window_size`, Perceiver attended-count.
- **data_mode** ∈ {`fixed` (draw the W-subset ONCE per rollout → BATCH), `stream` (fresh W-subset per round → ONLINE)}.
- **steps_per_round (m)**, **n_rounds (N)**, **loss_reduction** (`mean_tail`|`mean`|`final`) — on the NORMALIZED loss.

KEY SIMPLIFICATIONS vs the first online implementation:
- **DROP the `predict_before_adapt` flag.** It EMERGES from `data_mode`: with `stream`, round s's loss
  evaluates x_s using only x_0..x_{s-1}'s feedback (the message steps ran on the PRIOR round's feedback) =
  predict-before-adapt for free; with `fixed`, it's batch (fit the same data). Use BATCH's order
  (`message → circuit/loss → feed`) uniformly so both regimes share an identical body. (My online code did
  `circuit/loss → feed → message`, which is why I thought a flag was needed — it isn't.)
- **feedback_mode is the MODEL's concern, NOT the scan.** The scan only feeds the residual + stashes the
  data subset in globals. Whether a node consumes the scalar residual (GNN / gathered_attention) or attends
  the actual input/output VALUES (Perceiver = value-aware = the feedback-channel fix) lives in the model's
  `__call__`/config. Do NOT put a feedback flag in the unified loop.
- **batch = preset** `{data_mode=fixed, m=1, loss_reduction=mean_tail}`; **online = preset**
  `{data_mode=stream, m=k, loss_reduction=mean}`. Perceiver = a model swap, orthogonal.

═══════════════════════════════════════════════════════════════════════════════
## 2. THE LOSS FIX (the artifact)
- `generalized_bce` (`circuits/train.py` ~:193–232) returns `jp.sum(pos+neg)`; `res2loss` (L4) same →
  loss ∝ (n_inputs × n_bits) → scales with W → confound.
- **Fix:** normalize to a per-input(-per-bit) **MEAN** (divide by n_inputs, and ideally n_bits).
- **CAVEAT (must handle together):** sum→mean drops magnitude ~N× → the proven recipe (`lr=2e-4`,
  `grad_clip_norm=1.0`, tuned on the SUM) needs an **lr/clip RE-TUNE** (≈ ×N on lr, or rescale clip). Do
  the loss-norm + retune as one unit. Adam is ~scale-invariant but grad-clip is NOT — that's the real confound.
- **Regression:** the BATCH preset must reproduce today's batch behavior (modulo the documented loss-norm +
  lr-retune). Add a test asserting batch-preset == old `run_model_scan_with_loss` up to the loss scaling.

═══════════════════════════════════════════════════════════════════════════════
## 3. FILE-BY-FILE CHANGES
- `boolean_nca_cc/training/evaluation.py`: unify `run_model_scan_with_loss` (batch) + `run_model_online_scan_with_loss`
  (online) → one `run_inner_loop(..., W, data_mode, steps_per_round, n_rounds, loss_reduction)`. Batch-mean
  residual site is `:165` (`jp.abs(residuals).mean(axis=0)`) — becomes "mean over the W-subset." Also unify
  `evaluate_model_stepwise_batched` (batch eval) + `evaluate_model_online_batched` (online eval); ADD chunking
  to the online eval (the deferred TODO — it OOM'd at 256-step×64 circuits, currently capped at batch_size_out=16).
- `boolean_nca_cc/circuits/train.py`: `generalized_bce`/`res2loss` sum→mean (the loss fix); `LossConfig` (:22).
- `boolean_nca_cc/training/train_loop.py`: collapse `loss_fn_scan`/`loss_fn_no_scan`/`loss_fn_online` → one;
  the meta-batch vmap (`batch_loss_fn`, vmap over meta_batch_size circuits — KEEP, it's fine); replace the
  threaded params (`inner_loop_regime, stream_length, nca_steps_per_example, online_window_size,
  online_loss_reduction, stream_length_schedule`) with the unified dials (`data_mode, W, steps_per_round,
  n_rounds, loss_reduction`); the online per-subject OOD eval hook + `_native_output_width` + the
  `eval_stream_length = inner_steps//k` budget (KEEP the 256-from-fresh eval idea); `_resolve_stream_length`
  (curriculum — fold into the unified loop or keep as a W/N schedule).
- `train.py`: `_parse_stream_length_schedule` + the kwarg threading → update for the unified dials + lr-retune.
- `configs/config.yaml`: replace the `training.*` online keys with the unified dials.
- `tests/test_meta_tasks.py`: update the online tests; ADD a **batch-preset byte-identical regression** test.
- Built THIS session, KEEP: `tasks/samplers.py::sample_arith_family_y` (+ registry), `circuits/tasks.py::binary_subtract`
  ("sub"), `configs/tasks/{arith_family,single_reverse,single_add}.yaml`.

═══════════════════════════════════════════════════════════════════════════════
## 4. RELAUNCH — clean online↔batch continuum (after refactor + loss fix + smokes)
One sweep over the continuum, normalized loss, retuned lr:
- axes: `data_mode ∈ {stream, fixed}` × `W ∈ {1, 8, 64, 256}` (× maybe `m`), on **reverse (canary)** + **add (target)**.
- circuit=twelve, h64, attn128, dist_pe, S/N sized for 8-step-equiv BPTT, 2^17, resumable, V100 (VRAM ~16GB — W doesn't blow it up; probed).
- READOUT: `eval_online_out_reverse` vs (W, data_mode) — does reverse recover ~0.92 at high W / fixed
  (batch)? Does it stay ~0.60 in stream mode at all W? + `eval_online_out_add`. Eval = 256-step-from-fresh.
- DECISION: if reverse recovers with W (richness was the cap) → continuum validated. If reverse caps ~0.60
  across the continuum (stream mode) → it's the FEEDBACK CHANNEL → pivot to **value-aware node feedback /
  Perceiver cross-attn-to-data** (Stage-3, STRUCTURAL_DIRECTION §4) — a MODEL change, not a loop change.

═══════════════════════════════════════════════════════════════════════════════
## 5. OPS — DO THIS BEFORE EDITING SHARED CODE
- **W probe: STOPPED** (sweeps `sweep_20260617_080248` + `sweep_20260617_082610` cancelled; launchers killed; tmux `hsm_window`/`hsm_window_hi` gone).
- **STILL RUNNING (stop before the refactor edits shared inner-loop code — resumable chains re-rsync the
  local tree on chunk-advance and would pick up half-refactored code):**
  - v2 = `sweep_20260616_083618` chunk-2 array `4251726_1/_2` (online arith/k_junta cells ~81%), tmux `hsm_online_v2`.
  - bptt = `sweep_20260616_185953` deeper (64,1)/(8,8) cells (~35%), tmux `hsm_bptt`.
  - Their verdicts are ALREADY clear (§0) — either let them complete OR `kill -9` the launcher PIDs (pgrep -af
    "online_v2|bptt"), then `scancel` the arrays. Verify gbena ownership; NEVER touch Comp-PVR (`073214`,`172406`).
- Cluster: `ssh uzh`, V100 lowprio (free), A100 standard/medium (contends w/ Comp-PVR). hsm editable at
  `/home/gbena/code/packages/HPC-Sweep-Manager`. conda `bool_nca`. anahita A6000 (CUDA 2/3) for VRAM probes — never GPU 0.
- Already-in-code from this session: cosmetic [online OOD] log fix; online eval batch_size_out=16 OOM-guard.
- Stale tests (pre-existing, NOT this work): `test_gradient_fix.py`, `test_multi_step_gradients.py`,
  `tests/knockout/*` fail to import → `pytest tests/test_meta_tasks.py -o addopts=""` runs the live suite (was 44 green).

═══════════════════════════════════════════════════════════════════════════════
## 6. VERIFICATION (before relaunch)
1. `pytest tests/test_meta_tasks.py -o addopts=""` green, incl. the new batch-preset regression.
2. CPU smokes: `data_mode=fixed` (batch) reproduces old batch numbers; `data_mode=stream` (online) runs;
   both samplers; the normalized loss is W-invariant in magnitude.
3. VRAM-probe the heaviest cell (W=256, stream) on an A6000 (CUDA 2/3) before the cluster relaunch.
4. Dry-run the sweep (`hsm sweep run -c ... --dry-run`) → confirm cell count + overrides.
