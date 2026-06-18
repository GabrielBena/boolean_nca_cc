# HSM NOTE — resume-OOM crash masked as COMPLETED (2026-06-10)

For GabrielBena/HPC-Sweep-Manager. **HSM's resumable-chain protection WORKED here** — this is a
minor enhancement + a record of the failure mode, NOT a correctness bug. (Corrects an earlier
over-claim that it would crash-loop to `max_chunks`.)

## What happened
Resumable chain `sweep_20260609_052548` (16-task array, uzh-v100). At the chunk-1→2 boundary,
faithful resume worked, but the 4 T32 tasks (`n_message_steps=32`, h64) hit a CUDA OOM on the
**first training step after checkpoint load**:

```
RESOURCE_EXHAUSTED: Out of memory while trying to allocate 13.08GiB   (allocator GPU_0_bfc)
```

BFC fragmentation on the 32 GB V100: the restored 1.3 GB checkpoint (pool + optimizer) is resident
before the first step requests its 13 GB contiguous rollout buffer, so no contiguous block remains.
The same config ran fine **fresh** for 23 h. Cluster-side root cause + the train.py allocator fix
are in HANDOFF (2026-06-10). **Cluster-specific:** V100-32 GB only (the 13 GB request fits on A100).

## The HSM-relevant part
- The crashing task exited via `conda run python` failing, **but the SLURM array task was recorded
  `COMPLETED 0:0`**, not `OUT_OF_MEMORY`/`FAILED` — the inner non-zero exit was not propagated (the
  resumable job-script must exit 0 on a legitimate SIGTERM chunk-cut, and that also masks a real
  crash). So `parse_sacct_state` (the issue-#15 fix) cannot catch it.
- **HSM caught it anyway, correctly:** the orchestrator's no-progress heuristic fired —
  *"2 consecutive chunk(s) made no progress (no new .hsm_done sentinel and no checkpoint written) —
  likely a deterministic crash"* — and **FAILED the chain after 4 chunks** (bounded to 2 crash-cycles,
  not a run-to-`max_chunks` loop). The launcher exited gracefully.

## Suggested enhancement (minor, optional)
Propagate the inner command's exit code in the resumable job-script while still exiting 0 on a
*trapped* SIGTERM, so a genuine crash surfaces as SLURM `FAILED`/`OUT_OF_MEMORY`. Then
`parse_sacct_state` catches it in **1** chunk instead of needing the 2-chunk no-progress heuristic.
(Today's 2-cycle latency was harmless, so this is purely a small optimization.)

## Repro sketch
Resumable chain; chunk-2 resume of a job that OOMs on the first post-load step (32 GB V100, T=32).
chunk-1 `COMPLETED` (walltime) and chunk-2 `COMPLETED` (OOM) are indistinguishable at the sacct
level; the no-progress heuristic is what disambiguates.
