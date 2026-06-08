# Checkpoint-resume — implementation handoff (fresh-session ready)

Spec source: **`CHECKPOINT_AUDIT.md`** (read it first — full state inventory,
saved-vs-needed table, RNG analysis, file:line touchpoints). This doc is the
step-by-step build plan + acceptance test so a cold session can execute without
re-deriving anything.

## Goal
Kill a training run (or get Slurm-preempted) and resume on another GPU so the
resumed run is faithful to one that never stopped. Motivation: the V100 lowprio
pool is fast but preemptible; this makes it the workhorse and lets us stop
depending on the (3-day-stuck) A100 ladder.

## What's already true (from the audit)
- `save_checkpoint` (`boolean_nca_cc/training/checkpointing.py:392`) already saves
  **model + optimizer state (incl. AdamW moments + uint32 step) + step + metrics
  + cfg**. `nnx.state(optimizer)` round-trips the LR schedule for free (optax
  cosine/constant are STATELESS pure functions of the optimizer step).
- The CRITICAL gaps (resume diverges without these):
  1. **circuit pool** not saved — evolved logits/hidden/damage_count/maturity/
     per-circuit y_task. Restarting it restarts the meta-learning curriculum.
  2. **`train_key`** not saved — it's a CARRIED, re-split chain (not
     `fold_in(root,epoch)`), so epoch-N randomness is path-dependent; must save
     the live key.
  3. **`last_reset_epoch`** / pool-reset phase not saved (reset cadence shifts).
- MODERATE: BestModelTracker/EarlyStopping reset to ±inf (first post-resume eval
  spuriously overwrites the deploy "best"); adaptive/plateau LR scheduler state
  (config-gated — confirm cfg.training.lr_scheduler != adaptive on the cluster).
- COSMETIC: wandb run id/step.
- BUG to fix in passing: `save_periodic_checkpoint` (`checkpointing.py:552`) passes
  `{"epoch": epoch}` as cfg, so `latest_checkpoint.pkl` loses the full cfg needed
  to rebuild the optimizer. Save the real cfg.

## Implementation steps
1. **Extend the checkpoint dict** (`save_checkpoint`): add
   `pool` (full nnx.state or the pool's arrays), `train_key`, `epoch`,
   `last_reset_epoch`, `best_tracker` (best_values + best_epochs),
   `early_stop` (count), `wandb_run_id`. Keep back-compat: old checkpoints
   missing these still load for the demo-probe path (guard with `.get`).
   - Pool size check first: `pool_size(4096) × per-circuit(logits+hidden+...)`.
     Measure the pickle MB; if heavy, save pool every Nth checkpoint or to a
     sidecar file. (Audit estimate ~50–200 MB.)
2. **Add `training.resume_from: <path>`** to `configs/config.yaml` (null default).
3. **Restore sequence in `train_model`** (`train_loop.py`, before the epoch loop):
   if `resume_from` set, load ckpt → `nnx.update(model, ...)` + rebuild/refresh
   Optimizer state + set `train_key`, `start_epoch`, `last_reset_epoch`, seed the
   BestModelTracker/EarlyStopping from saved values, restore pool. Start the loop
   at `start_epoch` (not 0). nnx gotchas in audit §5.
4. **Slurm preemption hook**: the loop only catches `KeyboardInterrupt`
   (`train_loop.py:1797`). Add a SIGTERM handler that triggers a final
   `save_checkpoint` (Slurm sends SIGTERM before SIGKILL on preempt). Pair with
   `#SBATCH --signal=B:TERM@120` in the HSM array template for a 120 s grace save.
5. **wandb resume**: pass `id=wandb_run_id, resume="allow"` when present.

## Acceptance test (the smoke — DO run before trusting)
Faithful-resume A/B on a TINY run (2^11, T=5, pool 256), GPU etiquette per
CLAUDE.md (never GPU:0; verify residency; this box is shared):
1. Run 200 epochs uninterrupted → record loss/acc curve + final best metric.
2. Re-run: 100 epochs, checkpoint, kill, `resume_from` that ckpt, 100 more.
3. **PASS** = the resumed curve matches the uninterrupted one to within
   float-noise from epoch 100 onward (same data/damage draws → near-identical
   loss), AND the pool/optimizer/LR are continuous (no curriculum restart spike).
   A divergence at the resume seam = a piece of state still unsaved → back to §1.

Cheaper structural pre-checks (no GPU): checkpoint dict round-trips
(save→load→compare every leaf); `nnx.state(optimizer)` restores the step;
pool arrays survive pickle.

## Files to touch
- `boolean_nca_cc/training/checkpointing.py` (save dict + the periodic-cfg bug)
- `boolean_nca_cc/training/train_loop.py` (restore sequence + start_epoch + SIGTERM)
- `boolean_nca_cc/training/pool/pool.py` (pool serialize/deserialize helpers if not
  already round-trippable via nnx.state)
- `configs/config.yaml` (`training.resume_from`)
- `train.py` (thread resume_from into train_model; wandb id/resume)
- HSM array template (`--signal=B:TERM@120`) — optional, for true preemption.

## Do NOT
- Don't refactor `train_key` to `fold_in(root,epoch)` to "simplify" resume — it
  changes the RNG stream for ALL runs (breaks reproducibility of in-flight work).
  Save the live key instead.
- Don't merge/deploy from the resume branch without the smoke passing.

## Context pointers
- Shipped demo model is settled (seed10) — resume work is independent of it.
- The A100 ladder (array 3779117) is the thing to kill once resume works; until
  then it sits PENDING harmlessly.
- Sweep monitoring lives in the main session (do not duplicate).
