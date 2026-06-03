# demo_sweep_12 — RESULTS + resume kit (2026-06-03)

Sweep `sweep_20260602_172806` (W&B group `demo_sweep_12`), 12×12 reverse, T=5,
data_per_batch=256, meta_batch=128, seed=0, on S3IT A100. **7/8 completed**
(exit 0); run 006 (max_neighbors=33) deferred to post-maintenance.

## Final results — `eval_out_test` UNDAMAGED hard acc (topo-avg, 32 random circuits)

| # | recipe | wandb run | epochs | **hard acc** | task | notes |
|---|---|---|---|---|---|---|
| 1 | layerPE  · random+dmg | a7neu6pl | 2¹⁸ | **0.880** | task_001 | TS-ready PE |
| 2 | dist     · random+dmg | 2ir2kerg | 2¹⁸ | **0.891** | task_002 | best |
| 3 | rwse     · random+dmg | kdhk1hq8 | 2¹⁸ | **0.776** | task_003 | weak — don't use |
| 4 | dist+rwse· random+dmg | filut2ch | 2¹⁸ | **0.889** | task_004 | "lead" — no gain vs dist |
| 5 | dist+rwse· random CLEAN | w2afqtqg | 2¹⁸ | **0.876** | task_005 | clean ≈ dmg-trained |
| 7 | dist+rwse· FIXED clean | y6batwm9 | 2¹⁶ | **1.000** | task_007 | Regime I (eval_in_test) |
| 8 | dist+rwse· FIXED dmg   | a1sfx8j1 | 2¹⁶ | **1.000** | task_008 | Regime I |
| 6 | dist+rwse· random+dmg mn=33 | — | — | **deferred** | task_006 | post-maintenance, expected-null |

## Conclusions
1. **layerPE (0.880) ≈ dist (0.891) ≈ dist+rwse (0.889)** — the dist_pe/rwse PEs buy
   ~nothing at 12×12. Since layerPE is **already supported in the TS demo**, the
   TS port (task #8) is **very likely unnecessary** — deploy the 12×12 layerPE (or
   dist) checkpoint as-is. `rwse` alone is the clear loser (0.776).
2. These are **topology averages**; per-topology best is far higher → with
   best-of-N curation the *displayed* demo topology reaches ~0.99 regardless of PE
   (see `web_demo/DEPLOY_OPTIMIZATIONS.md`).
3. Damage-training doesn't cost the clean ceiling (0.889 dmg vs 0.876 clean) → one
   damage-trained model serves both clean display and repair.
4. Fixed-topology reverse = perfect 1.0 (Regime I).

## Checkpoints (pulled local; also on cluster /scratch + group archive when sweep closes)
- Local: `/mnt/8TB_HDD/gbena/hsm-sweeps/sweep_20260602_172806/tasks/<task>/checkpoints/run_<id>/`
  - `best_model_eval_out_test_hard_accuracy.pkl` (random) / `..._in_test_...` (fixed),
    `best_model_eval_damaged_*`, `latest_checkpoint.pkl`.
- Demo candidates (random+damage): **task_002/run_2ir2kerg (dist, best)**,
  task_001/run_a7neu6pl (layerPE, TS-ready), task_004/run_filut2ch (dist+rwse).
- Cluster source: `/scratch/gbena/hsm-runs/boolean_nca_cc/sweeps/sweep_20260602_172806/tasks/...`.

## RESUME CHECKLIST (new session)
- **Env:** conda `bool_nca` (local + cluster); JAX 0.10 + cuda12; pandas added.
- **Cluster:** S3IT via hsm remote `uzh` (`.hsm/config.yaml`); the 3 gotchas live in
  memory `sodc-s3it-hsm-training.md` (CVD=3 fix in `train.py`, `~/miniforge3`
  symlink, hsm-COMPLETED-≠-success). W&B logged in on cluster. Always read the real
  epoch via `training/epoch`, NOT `_step`; monitor via W&B (SLURM files flush only at job end).
- **Probe/curate (task #7):** `web_demo/export/ceiling_probe.py --ckpt <best_model> --cfg <run .hydra/config.yaml>`
  (compose the cfg from this sweep's overrides or pull the run's hydra config from /scratch);
  measure per-topology best → build the curated topology pool for the demo.
- **Deploy (task #8):** likely SKIP the dist_pe+rwse TS port — deploy the 12×12 layerPE/dist
  checkpoint via the existing TS path; export via `web_demo/export/`.
- **Pending:** 006 (max_neighbors) runs post-maintenance (~18:00 2026-06-03); the launch
  process `bfla3ek13` polls it and will pull+archive the whole sweep when it finishes.
- **Docs:** diagnosis+levers `web_demo/DEPLOY_OPTIMIZATIONS.md`; hsm feedback
  `HPC-Sweep-Manager/FIRST_USE_FEEDBACK_S3IT.md`; session log `SESSION_2026-06-02.md`.
