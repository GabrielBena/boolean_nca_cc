# HSM beta-test report — `--resumable` chained sweeps (S3IT/SSH-Slurm)

**Repo:** GabrielBena/HPC-Sweep-Manager · **Version:** `hsm 0.1.0+g5f87299`
**Date:** 2026-06-08 · **Reporter:** boolean_nca_cc sweep campaign
**Relates to:** issue #12 (resumable chained runs) — this is field feedback from its first real on-cluster use.

> Draft for filing as a GitHub issue (or three — Finding 1 is independently severe).
> Findings ranked by severity. Each has Context · Repro · Evidence · Impact · Suggested fix.

---

## Use-case / context (so the friction makes sense)

- **Driver:** a local workstation drives the **S3IT (UZH) Slurm cluster over SSH** (`backend: slurm`,
  remote alias `uzh-v100`). No HSM install on the cluster; jobs are `sbatch`ed over SSH.
- **Why `--resumable`:** JAX meta-learning runs (`train.py`, 2^18 epochs) exceed the V100 `lowprio`
  **24h walltime cap**. A single run is split into a checkpoint-chained sequence of ≤23h chunks
  (`--resumable --chunk-walltime 23:00:00`). Our training script honors the issue-#12 contract
  (`HSM_RESUME_FROM` / `.hsm_done` sentinel / SIGTERM checkpoint).
- **The sweep that surfaced this:** `sweep_20260608_193154`, Slurm array `3867286`, 5 tasks, V100.
  One task (`_4`) hit a **genuine OOM at startup** (our config was mis-sized — a 32×128 corner that
  needs >32 GB). That OOM is *our* bug; what it **revealed about HSM's chunk wrapper is the report.**

---

## Finding 1 — [HIGH] A crashed chunk is reported `COMPLETED`, not `FAILED`

A hard crash (OOM, uncaught exception) inside a resumable chunk surfaces to Slurm — and to
`hsm sweep status` — as a clean **`COMPLETED` / ExitCode `0:0`**, with no checkpoint and no
`.hsm_done` sentinel, and (so far) is **not resubmitted**. A failed run masquerades as a finished one.

**Repro:**
1. Launch a resumable array where one task's first chunk crashes fast (we OOM'd; any uncaught
   exception reproduces it): `hsm sweep run -c <yaml> --mode array --remote uzh-v100 --resumable --chunk-walltime 23:00:00`.
2. `sacct -j <arrayjob> -X` →
   ```
   3867286_4   COMPLETED   00:01:27   0:0      # crashed in 87s, shown COMPLETED/0:0
   3867286_1   RUNNING     00:16:11   0:0      # healthy siblings
   ```
3. The task dir has **no checkpoint** (`tasks/task_4/checkpoints/run_*/` empty) and **no
   `.hsm_done`**. `squeue` 16 min later: `_4` is **not** resubmitted.

**Evidence (`...array_3867286_4.err`):**
```
... RESOURCE_EXHAUSTED: Out of memory while trying to allocate 13.33GiB.
ERROR conda.cli.main_run:execute(125): `conda run python train.py ...` failed. (See above for error)
```
So `conda run` (and python) exited **non-zero**, yet the Slurm job exited **0 → COMPLETED**.

**Root cause (the generated `..._array.slurm` wrapper, chunk-end logic ~L215–255):**
The wrapper deliberately `exit 0`s at chunk end, with this rationale (verbatim comment):
> *"We exit 0 on an incomplete chunk because a non-zero mid-budget exit is EXPECTED — the chain
> driver decides done/advance/fail from the sentinel + checkpoint progress, never from this chunk's
> exit code."*

This is sound **for the timeout case** (a 23h SIGTERM makes python exit non-zero, which is normal and
must not be FAILED). The bug: the wrapper **cannot distinguish a timeout-SIGTERM from a real crash** —
it `exit 0`s for *both*. A `_term` trap (`trap _hsm_term TERM`) already exists, so the wrapper *knows*
whether a SIGTERM arrived; it just doesn't use that to gate the exit code.

**Impact:**
- `hsm sweep status` reports a crashed run as **done** (this is the user-visible "1/5 done" that
  started this investigation — a silent data-loss-shaped failure).
- For the chain: a crasher exits 0/COMPLETED with **no progress and no sentinel**. Either the driver
  (a) trusts Slurm `COMPLETED` → **silently drops** a never-trained task (worst case), or (b)
  resubmits → it re-crashes → the "2 no-progress strikes" guard eventually marks FAILED after
  **wasting 2 scheduler slots**. Both are bad; which one happens should be deterministic and correct.

**Suggested fix:**
- In the wrapper, gate the final exit on the *cause* of the non-zero exit, which the `_term` trap
  already disambiguates:
  - `.hsm_done` present → `exit 0` (done). ✅ (already)
  - SIGTERM received (timeout) **and** checkpoint/`HSM_RESUME_TO` advanced this chunk → `exit 0`
    (legit mid-budget). ✅
  - **else (non-zero exit, no SIGTERM, no progress) → surface the crash**: `exit $exit_code`
    (or write a `FAILED` marker the driver checks), so Slurm shows `FAILED` and the chain stops on
    strike 1 instead of masking.
- Independently, the **chain driver's done-detection must require the `.hsm_done` sentinel** for
  "done" — never infer done from Slurm `COMPLETED` alone (since the wrapper can emit `COMPLETED` for
  a crash). Treat `COMPLETED && no sentinel && no progress` as a failure, not a completion.

---

## Finding 2 — [MED] `pre_script` `module load` silently no-ops (non-login shell)

`pre_script` / `modules` entries that call `module` fail in the `sbatch` shell because the `module`
function isn't initialized in a non-login, non-interactive shell.

**Evidence (`..._4.err`, very first line):**
```
/var/spool/slurmd/job3867342/slurm_script: line 15: module: command not found
```
Line 15 of the wrapper is our configured `pre_script: ["module load miniforge3/25.3.0-3", ...]`.

**Why it didn't take down the run:** HSM's *robust conda/micromamba fallback* block (well done — it
probes PATH, then `*/etc/profile.d/conda.sh`, then a micromamba bridge) found conda another way, so
`conda run -n bool_nca` still worked. But:
- A user relying on `module load` for **anything other than conda** (CUDA toolkit, a compiler, MPI)
  would be **silently broken**, with only a confusing stderr line.
- The S3IT "known-good" recipe (`module load miniforge3`) is in fact a no-op here.

**Suggested fix:** before executing `pre_script`/`modules` commands, initialize the module system if
present, e.g. source the first existing of `/etc/profile.d/lmod.sh`, `/etc/profile.d/modules.sh`,
`/etc/profile.d/z00_lmod.sh`; or detect that `module` is undefined and emit a clear warning
("`module` not available in the batch shell — modules will not load; init Lmod in pre_script").
At minimum, document that `pre_script` runs in a non-login shell.

---

## Finding 3 — [MED] Resumable chain requires the launcher to stay alive locally

`hsm sweep run --resumable` prints:
> *"The launcher drives the chain while alive (run it under tmux/nohup); if it dies, resume with
> `hsm sweep advance sweep_20260608_193154`."*

**Friction:** for multi-day chains (our T=32 run ≈ 3–4 × 23h chunks ≈ 3–4 days), a **local process
must persist for days** on the driving machine, or the chain stalls until a human runs
`hsm sweep advance`. There is no `--detach`/daemon. If the laptop sleeps / SSH drops / the session
ends, chunk N+1 never submits and **nothing warns you** — the chain just stops advancing.

**Suggested fix (in rough order of robustness):**
1. **Server-side self-chaining (best, HPC-native):** submit chunk N+1 from chunk N's epilog via
   `sbatch --dependency=afterany:$SLURM_JOB_ID` (or `--dependency=afternotok`/`afterok` keyed on the
   sentinel). The chain then advances **entirely on the cluster** with **no live local process**.
2. `--detach`: background the launcher robustly (nohup + pidfile + logfile), with `hsm sweep status`
   able to find/report it.
3. Document a cron/systemd recipe around `hsm sweep advance` for unattended advancing.

---

## Minor notes & positives (for balance)

- **Guard interaction:** the "up to 10 chunks, 2 no-progress strikes → FAILED" backstop is good, but
  the Finding-1 fix would let a hard crasher fail on **strike 1** (or immediately) instead of burning
  2 slots + scheduler latency.
- **Nice prior fixes worth keeping:** (a) params extraction via a tempfile run *by path* (the comment
  cites the 2026-06-03 "heredoc-stdin swallowed → silently ran DEFAULT config" field report — exactly
  the right fix); (b) the conda/micromamba fallback is thorough and well-commented; (c) the
  sentinel-based done contract is the right design — Finding 1 is about *honoring it over Slurm
  state*, not replacing it.
- **`--dry-run` / `--count-only` are great** for pre-flighting an array; they caught nothing wrong
  here because the failure was VRAM, which HSM can't know.

---

## Environment / artifacts for repro

- `hsm 0.1.0+g5f87299`; remote `uzh-v100` (`backend: slurm`, host `uzh` = cluster.s3it.uzh.ch),
  partition `lowprio`, qos `normal`, `gpu_type: V100` (32 GB), 24h MaxWall.
- Sweep id `sweep_20260608_193154`; Slurm array `3867286` (tasks `_1.._5`); node `u24-chiivm0-603`.
- Generated wrapper: `<remote_sweep_dir>/scripts/sweep_20260608_193154_array.slurm`
  (`remote_sweep_dir=/scratch/gbena/hsm-runs/boolean_nca_cc/sweeps/sweep_20260608_193154`).
- Crash log: `<...>/logs/sweep_20260608_193154_array_3867286_4.err`.
