# Feature: resumable chained runs — complete jobs that exceed the walltime cap as auto-resubmitting, checkpoint-chained chunks (with experiment-tracker continuity)

## Motivation / use-case (precise)

Some GPU pools are **fast and abundant but walltime-capped or preemptible**. Concrete
case driving this: the UZH S3IT **V100 `lowprio` pool** — ~48 idle cards, but its only
QOS caps walltime at **24 h**. The alternative (A100 `standard`, 48 h cap) is heavily
contended (we've seen a job sit `PENDING` for 3 days). So the fast resource is exactly
the one we can't run long jobs on.

We want to: **launch an arbitrary config (any `epochs`, `n_timesteps`, etc.) that may
need 48 h+, and have HSM complete it transparently as a chain of ≤24 h chunks on the
capped pool** — each chunk resumes the previous one from a checkpoint, and the
experiment-tracker (W&B) shows **one continuous run** across all the chunk seams.

The ask in one line: `hsm sweep run --resumable --chunk-walltime 23:00:00 -c sweep.yaml`
and a 48 h job just *finishes*, as 2–3 V100 chunks, without me babysitting resubmissions.

This is the natural complement to the hetero-GPU scheduling in #7: that spreads a sweep
*across* GPU types; this lets a *single* long task survive a capped/preemptible pool.

## The generality concern (raised explicitly — and I think it's the crux)

The danger is baking training-framework semantics (checkpoint format, W&B, "epochs")
into HSM and turning a general Slurm orchestrator into something coupled to one project.
**Proposed guardrail: HSM owns ONLY orchestration; the resume/checkpoint/tracker
semantics stay in the user's training script, behind a small documented contract.**

HSM's responsibilities (general, project-agnostic):
1. Submit chunk *k+1* depending on chunk *k* (`--dependency=afterany`), within the
   workdir, until the job signals "done".
2. Give each chunk a **resume pointer** (where the previous chunk left its checkpoint).
3. Arrange the **pre-walltime signal** so the script can save before being killed.
4. Detect **done vs. needs-another-chunk**, with a retry cap so a *crashing* job can't
   resubmit forever.
5. Track the whole chain as **one logical run** in `hsm status` / archive its outputs.

What HSM must stay ignorant of (the script's job):
- the checkpoint *format/contents*, how "faithful" the resume is, the optimizer/RNG/pool
  state — HSM only knows a *path*.
- the experiment tracker — W&B-run continuity is achieved by the script reusing the run
  id it stored in its own checkpoint; HSM just preserves the workdir so that id persists.
- what "done" means semantically (reached target epochs / converged / etc.).

So the feature is, generically: **"a chunked dependency-chain executor with a
resume-pointer convention and a done-protocol."** Any training script that implements a
~10-line contract can use it; HSM never grows a dependency on a specific framework.

### The contract (the only thing HSM imposes on a script)
1. **Consume a resume pointer.** HSM passes `HSM_RESUME_FROM=<path>` (env) and/or a
   CLI override (`--resume-arg training.resume_from`, configurable) on chunks ≥2. The
   script resumes if it's set/non-empty, else starts fresh. (Our `train.py` already does
   this via `training.resume_from`.)
2. **Save on the pre-walltime signal.** HSM ensures `--signal=B:TERM@<grace>` (default
   ~120 s) so the script gets SIGTERM before SIGKILL; the script saves a
   resume-complete checkpoint to the **HSM-provided checkpoint path** (`HSM_RESUME_TO`,
   under the persistent workdir) and exits. (Ours has a SIGTERM handler doing this.)
3. **Signal done.** Pick ONE protocol (see options) — e.g. the script writes a
   `${HSM_WORKDIR}/.hsm_done` sentinel when the work is actually complete; its absence
   after a chunk = "submit another". HSM never inspects epochs.

## Proposed implementations (several — maintainer's call which to land)

**A. Script-driven self-resubmission (thinnest HSM change).** HSM ships a small helper
`hsm chunk-resubmit` that a job calls from its SIGTERM/exit path; the helper resubmits
the next chunk with the same chain id + `afterany` + the resume pointer, unless `.hsm_done`
exists. *Pro:* minimal HSM logic, naturally unbounded length. *Con:* resubmission logic
lives partly in the job wrapper; HSM's view of the chain is reconstructed from chain id.

**B. HSM-driver chain (HSM owns the loop).** The HSM client, after a chunk reaches a
terminal Slurm state, checks the done-protocol and submits the next chunk itself
(threading the resume pointer), repeating until done or retry-cap. *Pro:* HSM fully owns
+ tracks the chain; clean `status`. *Con:* needs the HSM driver to be alive or
re-invokable to advance the chain — fine for the SSH-Slurm backend (a `hsm chain advance`
cron/daemon, or advance-on-`hsm status` poll). Probably the **cleanest fit** for the
existing SSH-Slurm model.

**C. Pre-submitted dependency chain.** Submit N chunks up front, each `afterany` the
previous, where N = ceil(estimated_total / chunk_walltime). *Pro:* one `sbatch` burst,
no live driver. *Con:* must estimate N; the tail chunks waste a queue slot if the job
finishes early (mitigate: a chunk that finds `.hsm_done` exits immediately as a no-op,
and `scancel`s its successors).

**D. `--requeue` + same job id.** Lean on Slurm requeue-on-preempt/timeout; the script
always resumes from the last checkpoint. *Pro:* a single job id, simplest `status`.
*Con:* requeue-on-*timeout* semantics are inconsistent across clusters and harder to cap;
less portable than explicit dependency chains.

**Recommendation:** B (HSM-driven, advance-on-poll) as the principled core, with A's
`afterany` dependency so a chunk's successor is queued immediately (no idle gap waiting
for the driver). But that's the maintainer's final call — all four satisfy the contract.

## Config surface (sketch — adjust to taste)
```yaml
# per-remote spec or sweep-level
resumable:
  enabled: true
  chunk_walltime: "23:00:00"     # < the pool's QOS cap
  signal_grace: 120              # seconds before walltime -> SIGTERM (=> --signal=B:TERM@120)
  resume_arg: "training.resume_from"   # CLI override HSM sets on chunks >=2 (or env-only)
  done_sentinel: ".hsm_done"     # script writes this when complete
  max_chunks: 10                 # runaway guard (also a per-chunk failure retry cap)
  checkpoint_subdir: "resume"    # under the persistent workdir; HSM passes HSM_RESUME_{FROM,TO}
```
`hsm queue`/`status` shows a chain as one row: `chain <id>  chunk 2/?  jobid <N>  RUNNING`.

## Edge cases / gotchas (from our real runs)
- **Hard crash without SIGTERM** (e.g. we hit intermittent
  `CUDA_ERROR_STREAM_CAPTURE_INVALIDATED` ~3–4% of runs): the chunk dies with no clean
  save. The chain should still resume — from the last *periodic* checkpoint — so the
  contract should let the script also checkpoint periodically, and HSM should resume on
  *any* terminal state (`afterany`), NOT only success. But cap retries so a
  *deterministically* crashing job (bad config) doesn't resubmit forever → `max_chunks`
  / consecutive-failure cap, then mark the chain FAILED.
- **Distinguish "timed out, needs resume" from "errored/done".** Timeout is a non-zero
  exit, so exit code alone is ambiguous → the `.hsm_done` sentinel (or a structured
  status file) is the reliable done-signal, not exit code.
- **Persistent checkpoint path.** Must live where it survives between chunks: `/scratch`
  (30-day purge — fine within a run) or `/shares` (durable). HSM already manages
  `workdir`/`archive_dir`; the resume ckpt goes under `workdir`.
- **Tracker continuity is the script's job.** HSM preserving the workdir is *sufficient*
  for it (the script reads its stored run id from the resumed checkpoint). HSM should not
  touch W&B.
- **Faithful schedule.** The script must be told the *total* budget each chunk (so LR
  schedules etc. span the whole run, not the chunk) — i.e. HSM passes the same
  `epochs`/budget every chunk and only varies the resume pointer. Worth a doc note.

## Prerequisites already met on the training side (so HSM can assume the contract is implementable)
- Faithful `resume_from` (model + optimizer + LR-schedule-via-step + full pool +
  carried RNG + best/early-stop trackers), validated bit-identical on an A/B resume.
- SIGTERM handler that writes a resume-complete checkpoint.
- W&B same-run continuity on resume (reuses the stored run id).
These prove the contract is light to implement for a real project; HSM only needs the
orchestration around it.

## Relationship to existing work
- Complements #7 (hetero-GPU scheduling): #7 = spread a sweep across GPU *types*; this =
  let one long task survive a *capped/preemptible* pool. Together they make the cheap,
  abundant, capped pools (V100 lowprio) first-class for long jobs.
