# HANDOFF — boolean_nca_cc (as of 2026-06-08)

Single entry point for the next session. Read this first, then the linked docs only
as needed. Two arcs: **demo (DONE, shipped)** and **arithmetic (the NEW frontier)**.

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
