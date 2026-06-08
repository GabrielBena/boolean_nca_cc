# SODC demo-model campaign — state of play (2026-06-08)

A single catch-up report after a weekend of sweeps. Written to be read cold.
Goal throughout: pick the live-demo model(s) for the 12×12 reverse task — maximize
settled accuracy, pre-perturbation crispness (no bit-flicker), and post-perturbation
recovery (shuffle + shotgun buttons), with stable training.

--------------------------------------------------------------------------------
## 0. TL;DR

- We built a **self-scoring training pipeline**: every run now reports its own
  demo metrics (clean accuracy, jitter, shuffle-recovery, shotgun-healing) into
  its results CSV. Model selection is now reading a table, not hand-probing.
- We ran **4 sweeps** (~60 models): v3 (landscape), v3.1 (tail-mean loss),
  v3.2 (solar-burst damage), v3.3 (16-seed harvest of the best recipe).
- **Three things shipped to the web demo** already (all reversible, old models
  kept): a new graph-PE runtime, and two candidate models in the gallery.
- **The decision now in front of you is narrative, not technical** — see §5.
  Short version: we have a clean, data-backed **2-model story** (a *redundant*
  model and a genuinely *self-healing* model) OR a **single best-quality** model.
- **Infra**: the A100 "ladder" sweep has been stuck in the cluster queue for
  3 days while the V100 pool flies. Recommend killing it and adding
  resume-from-checkpoint to train.py (§6).

--------------------------------------------------------------------------------
## 1. Glossary — run names → plain meaning

Sweeps (Slurm array → what it was):
- **v3** (3703585, 22 runs): the broad landscape. Found T=8 horizon helps carry;
  shuffle-aware training is a dead end; *no-damage training is the worst of all*.
- **v3.1** (3778421, 10 runs): **tail-mean loss** (grade the rollout's tail, not
  just its last step) on T=8. Produced the crispest models.
- **v3.2** (3820013, 12 runs): **solar-burst damage** training (see §3) crossed
  on the 3 best recipes. Produced both the quality champion and the first real
  healers.
- **v3.3** (3823089, 16 runs): **harvest** — 16 fresh seeds of the single best
  recipe from v3.2, to mine the seed lottery.
- **calibration** (3820042, 10 runs): tested whether cheap ¼-budget runs rank
  models like full runs. **They don't** — killed the idea of cheap screening.

Model nicknames you'll see:
- **1u5ssulx** — the original deployed model (flat damage, random wires). Baseline.
- **B3s0 / "v31"** — v3.1 winner: flat damage + tail-mean, T=8. Crispest model
  (flicker 0.002). Shipped to gallery as `reverse_random_damage_v31`.
- **t12 / "v32"** — v3.2 champion: solar-burst + endpoint loss. Best raw quality
  (settled 0.97, carry 0.96). Shipped as `reverse_random_damage_v32`, current
  default boot.
- **seed10** — v3.3 harvest: solar-burst, the best **healer** (recovers 64% of
  shotgun damage). NOT yet shipped — the §5 decision.

Metric cheat-sheet (all "demo_*" in the CSVs):
- **settled** = steady accuracy after the circuit settles (higher better).
- **flip** = fraction of output bits flickering per tick at the fixed point
  (the demo "jitter"; want ≤0.01-0.02).
- **carry** = accuracy after the *shuffle* button (evolve on topology A, drop the
  logits onto a fresh topology B). The recovery-from-rewiring number.
- **healed_frac** = of the accuracy lost to the *shotgun* button (permanent gate
  knockouts), what fraction is clawed back. THE adaptive-resilience metric, added
  this weekend after you noticed the shotgun damage wasn't recovering.

--------------------------------------------------------------------------------
## 2. What was built (code — committed on branch sodc-demo-salvage)

1. **Batched demo-probe** (`boolean_nca_cc/training/demo_probe.py`): a fast,
   GPU-batched scorer that runs *inside every training job* after training.
   Outputs the whole demo_* suite to `final_results.csv`. Three families:
   clean-settle, carry/shuffle (2×2 logits×hidden quadrants), and shotgun-recovery.
2. **Tail-mean loss** (`training.loss_step_mode=mean_tail`): grade the rollout's
   last K steps, so the model must reach *and hold* the solution. Gave the
   crispest models we've seen.
3. **Solar-burst damage** (`damage.burst.*`): doubly-stochastic gate failure —
   quiet background rate + random "radiation burst" windows (§3). Same 10% total
   damage budget as flat training; verified by Monte-Carlo.
4. **dist_pe web runtime**: the TS demo couldn't run graph-distance-PE models;
   now it computes them in-browser (line-for-line ported + bit-exact parity).
   This is what unblocked shipping the new models.
5. **CLI + curation** updated to the batched engine; full JAX↔NumPy↔TS parity
   harness extended to local checkpoints.

--------------------------------------------------------------------------------
## 3. What "solar-burst damage" is (the new idea this weekend)

Standard training damages gates at a flat low probability every tick. Real
question: the demo's *shotgun* button knocks out a clump of gates at once on an
already-settled circuit — and flat-trained models had never seen that, so they
just absorbed it without repairing.

Solar-burst training keeps the same 10% lifetime damage budget but delivers it as
a quiet drizzle punctuated by short high-rate windows ("solar events"): ~6 gates
over a 4-tick window, ~twice per circuit lifetime. It's a Cox process — three
nested layers of randomness (when bursts arrive, which gates, whether damage
sticks). It teaches "a settled circuit suddenly loses a clump" *without* ever
showing the literal shotgun event (which stays out-of-distribution for eval).

--------------------------------------------------------------------------------
## 4. What was learned (the science)

1. **Training damage is load-bearing**, not just for resilience: the no-damage
   control was worst on *every* axis (carry 0.59 vs 0.89, flicker 10× worse).
2. **Crispness vs adaptation is a real trade-off.** Crisp, "redundant" models
   boot perfect and barely flicker but *absorb* shotgun damage. "Adaptive" models
   flicker more but *re-grow* function around dead gates.
3. **Carry and crispness are seed lotteries** whose *distribution* the recipe
   sets — which is why we harvest seeds and select on measured metrics, never on
   recipe alone. The quality champion (t12) is a 1-in-18 jackpot draw.
4. **Solar-burst training genuinely improves shotgun healing** — confirmed, not
   marginal: best burst model heals 64% vs 18-29% for flat-damage models. This is
   the evidence behind a possible 2nd "self-healing" demo model.
5. Dead ends, cleanly buried: shuffle-aware training, consistency-penalty loss,
   ¼-budget screening, no-damage, high margin loss. All documented in the
   ANALYSIS_*.md files.

--------------------------------------------------------------------------------
## 5. THE DECISION — which model(s) go on the front

The three deep-probed candidates (256 topologies / 32 pairs, paper shotgun dose):

| model    | regime              | settled | carry | flip  | shotgun healed | character          |
|----------|---------------------|---------|-------|-------|----------------|--------------------|
| **t12**  | solar-burst         | 0.971   | 0.961 | 0.019 | 18%            | best numbers; redundant |
| **B3s0** | flat damage         | 0.923   | 0.872 | 0.002 | 19%            | crispest; redundant |
| **seed10**| solar-burst        | 0.927   | 0.923 | 0.068 | **64%**        | adaptive; self-heals |

Curated demo pools (what users actually click through):
- t12: boot 0.965 / shuffle-recovery 0.962 / 43 perfect topologies / top-12 ≈1.0.
- seed10: keeps p90=1.0 and best-of-8 0.98, so curation can pull clean topologies;
  pool not yet built (would build on ship).

### Reconciling with your stated goal
You want **one** front model, or **two** only if there's a real narrative —
specifically "(A) random wires + fixed damage" vs "(B) random wires + solar burst
*that heals shotguns for real*." We can now support that narrative with data:

- **Option 1 — single best model: ship t12 alone.** Cleanest, highest numbers,
  already the default. Caveat for honesty: t12 is *technically* solar-burst-trained
  but behaves redundant (heals only 18%), so its damage-training story is just
  "robust," not "self-healing." Fine if the demo isn't selling healing.

- **Option 2 — the 2-model narrative you sketched, and it holds up:**
    - **A "robust / redundant"** = **B3s0** (flat damage): boots crisp, near-zero
      flicker, but *absorbs* the shotgun (heals 19%).
    - **B "adaptive / self-healing"** = **seed10** (solar burst): comparable
      quality (settled 0.93), but genuinely *re-grows* after the shotgun
      (heals 64% — visibly climbs back where A flatlines).
  Both ~0.93 settled, same architecture/task → a clean controlled contrast where
  the *only* narrative variable is the damage regime, and the shotgun button
  *demonstrates the difference live*. This is the "resilience vs redundancy"
  story end-to-end. **The healing difference is real and deep-confirmed (19%→64%).**

- **Note on t12 in Option 2:** t12 has the best raw numbers but muddies the clean
  A/B story (it's burst-trained yet redundant). Choices: keep it as a 3rd "best
  overall" gallery entry, or drop it for narrative clarity. My lean: if you go
  2-model, lead with B3s0 vs seed10 and keep t12 as an optional "highest accuracy"
  bonus.

**Recommendation:** Option 2 if the demo's message is the science (adaptive
resilience); Option 1 (t12) if the message is just "look how good/crisp it is."
Either way nothing new needs training — all three models exist and are probed.

--------------------------------------------------------------------------------
## 6. Infra: the stuck A100 ladder + train.py resume

- **A100 "ladder" sweep (3779117, 6 runs, T=12/16 tail-mean)** has been PENDING
  since 2026-06-05 — 3 days behind cluster priority, may never schedule soon.
  It would only test "does longer horizon buy more carry/healing" — a curiosity
  now, since t12 already exceeds any plausible ladder quality. **Recommend
  cancelling it** (frees the slot; we can re-run on V100 if we ever want it).
- **Your resume request is well-founded.** `save_checkpoint` already stores model
  + optimizer + step, but train.py has no "continue from here" path, and crucially
  the **circuit pool state is not checkpointed** — so a naive resume restarts the
  meta-learning curriculum. Proposed change (bounded, ~half a day):
    1. Extend the checkpoint to also save pool (evolved logits/hidden/damage_count)
       + RNG key + epoch.
    2. Add `training.resume_from=<ckpt>` to train.py: restore all of the above and
       start the loop at `start_epoch`.
    3. This makes V100 runs preemption-safe and lets us split a long run across
       short windows — turning the fast-but-volatile V100 pool into the workhorse.
  Want this? It's a clean standalone task; I'd smoke it on a 2^11 run before trusting it.

--------------------------------------------------------------------------------
## 7. Where the artifacts live
- Analyses: `sweeps/ANALYSIS_demo12_v3.md`, `_v31.md`, `_v32_v33.md`.
- Session logs: `SESSION_2026-06-06.md` (the detailed thread).
- Deep-probe data + checkpoints: `/mnt/8TB_HDD/gbena/hsm-sweeps/v3{1,2}_deep_*`,
  `/tmp/v3{1,2,3}_deep/` (tmp = volatile, HDD = durable).
- Shipped demo files: `web_demo/public/weights/reverse_random_damage_v3{1,2}*`.
- Code committed on branch `sodc-demo-salvage` (3 commits, not pushed).
