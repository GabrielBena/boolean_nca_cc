# paper_figures — reproducing the SODC ALIFE figures

Config-driven, no ad-hoc plotting cells. Everything here either pulls **logged W&B
history/summary** (no model re-evaluation) or re-runs a specific checkpoint's
evaluation (documented per-figure below). Two figures are a documented, honest gap —
see **Known Gaps** at the bottom.

Install with `pip install -e .` from the repo root first — this declares
`paper_figures` itself as an importable package, so `python -m paper_figures.<name>`
works from any working directory (including a neutral one like `/tmp`, which avoids
the repo's local `wandb/` dir shadowing things during a pull).

All scripts write into `data/` (cached CSVs, wandb-pulled or eval-generated) and
`out/` (rendered figures) — both git-ignored, regenerate them locally.

## W&B access

`gbena/boolean-nca-cc` (the W&B project every script below pulls from) is
**private** — you will not be able to run these scripts against live W&B unless
you have your own credentials for that project. Two figures don't need this at all:

- **Fig 2** and **random-wiring**: pull only *logged summary scalars*, no model
  weights — still need W&B access, but there's nothing to work around beyond that.
- **Fig 4, Fig 10, PCA trajectories** additionally need to instantiate a model from
  a specific checkpoint. For these three, [`paper_figures/checkpoints/`](checkpoints/)
  archives the small set of checkpoints they target (4 runs, 13MB total, config +
  weights) directly in this repo. `eval_fig4_resilience_isn1.py`,
  `eval_fig10_scalefree_isn1.py`, and `eval_pca_trajectories.py` all load through
  [`local_checkpoints.py`](local_checkpoints.py), which checks this local archive
  **first** and only falls back to `load_config_from_wandb` (i.e. only needs W&B
  credentials) for a run_id that isn't archived — e.g. if you point one of these
  scripts at your own new run. **Confirmed working with zero W&B credentials
  configured** (fresh `$HOME`, no API key) for all three archived-checkpoint
  scripts.

**Verification note**: every script below (Fig 2, Fig 4, PCA trajectories,
random-wiring, Fig 10) has been run end-to-end from a fresh `git clone` + fresh env
+ real W&B checkpoints, confirming the pipeline mechanics genuinely work — not just
that the code parses. That pass also surfaced and fixed a config-schema trap: some
checkpoints (e.g. `yu6kojmx`, from before the `tasks` refactor) store the task under
`circuit.task`/`circuit.text`, while newer ones (e.g. `6mo8q61y`) use
`tasks.name`/`tasks.text`. The `eval_*_isn1.py` / `eval_pca_trajectories.py` scripts
now detect and handle both automatically — if you point one at a different run and
hit a `Missing key` error on `tasks` or `circuit.task`, that's the schema boundary.

## Palette

Every `fig_*.py` script respects the `UNIFORM_PALETTE` env var: unset/`0`/`false`
uses the default Okabe-Ito colour-blind-safe palette (`style.py`); set to any other
value switches to the co-author's uniform 2-colour scheme (green/indigo) and appends
a `_uniform` suffix to the output filename, leaving the default version intact. The
actual submitted figures (`figures/*_uniform.pdf` in the manuscript) used this
toggle enabled.

```bash
UNIFORM_PALETTE=1 python -m paper_figures.fig2_fixed_wiring
```

## Figures

### Fig 2 / Regime I — Fixed-Topology TMT vs. BP
`fig2_fixed_wiring.py`. **Ready.** Sweep of record: `gbena/boolean-nca-cc`, group
**`sweep_20260212_141940`** — 30 runs = 3 tasks × damage-train{on,off} × 5 seeds
(`[0,1,2,4,5]`), `model=gathered_attention`, fixed wiring, `test_num=256`. This is
the sweep the published figure used. Being *after* the Feb-11 "always eval damaged"
change, it logs the damaged-eval ("green") branch as a real scalar plus a
config-matched BP baseline. (The earlier `sweep_20260209_231530` predates damaged
logging — do not use it.)

```bash
python -m paper_figures.wandb_data --no-cache   # -> data/fixed_wiring_sweep.csv
python -m paper_figures.bp_from_history         # -> data/fixed_wiring_bp.csv
python -m paper_figures.fig2_fixed_wiring       # -> out/fig2_fixed_wiring.pdf
```
- `test_num = 256` → 256/4096 held out. Damage mode = **probabilistic** (matches
  the published figure); the sweep also logs a discrete-damage variant
  (`eval_discrete_damaged_in_test/*`) if ever needed.
- `bp_baseline.py` *recomputes* BP via `run_bp_scan` (needs JAX) — kept only as a
  cross-check.
- `reeval_fig2.py` (full live re-eval) is **deprecated for Fig 2**: HEAD's model
  code has drifted from the Feb checkpoints, so it loads them to chance accuracy.

### Fig 4 / Regime I — Resilience to Permanent Damage
`eval_fig4_resilience_isn1.py` → `fig_resilience.py`. Needs a live checkpoint:
wandb run **`gbena/boolean-nca-cc/yu6kojmx`** (fixed wires, damage-trained,
reverse task).

```bash
python -m paper_figures.eval_fig4_resilience_isn1   # -> data/fig4_stepwise.csv (JAX, ~minutes)
python -m paper_figures.fig_resilience               # -> out/fig_resilience.pdf
```
`eval_fig4_resilience_isn1.py` re-runs both the TMT and BP stepwise scans against
that checkpoint (full JAX eval, not just logged scalars). Confirmed working
end-to-end against HEAD's model code (unlike Fig 2's `reeval_fig2.py`, this
checkpoint has *not* drifted to chance accuracy) — a smoke run recovers TMT hard
accuracy to ~0.95 after both shotgun and stochastic damage, versus a ~1.0 BP
ceiling and a ~0.92-0.94 no-repair baseline.

### Fig 3 / Regime I — PCA Trajectories of Circuit Optimisation
`eval_pca_trajectories.py` → `fig_pca_trajectories.py`. Gabriel's own run —
`gbena/boolean-nca-cc/6mo8q61y` (random wiring, reverse task, `gathered_attention`).

```bash
python -m paper_figures.eval_pca_trajectories   # -> data/pca_trajectories.csv (JAX, ~minutes)
python -m paper_figures.fig_pca_trajectories    # -> out/fig_pca_trajectories.pdf
```
Ported from an exploratory notebook (`trained_models.ipynb`, no longer in the
repo — its logic now lives entirely in these two scripts). **Honesty note**: the
original notebook exploration used placeholder knobs that don't match the
submitted figure (it coloured by accuracy, and used a `permanent=0.5` value that
doesn't correspond to a real "permanent vs recoverable" split). `eval_pca_trajectories.py`
reconstructs the *published caption* instead — three conditions (no damage /
recoverable shotgun / permanent shotgun), trajectories coloured by step — as a
faithful best-effort reconstruction, not a byte-identical replay of whatever
produced the exact submitted PDF.

### Random-Topology / Regime III
`fig_random_wiring.py`. **Ready** (wandb-only, no checkpoint download). Group
`random_wiring_sweep`, filtered on `model.type=gathered_attention`,
`width_factor=2`, `circuit_hidden_dim=64`, `damage.target_damage_fraction=0.1`.
Single seed, damage-trained only — bars are point values, no per-seed error.

```bash
python -m paper_figures.fig_random_wiring   # -> out/fig_random_wiring.pdf
```

### Fig 10 / Regime IV — Scale-Free Generalisation
`eval_fig10_scalefree_isn1.py` → `fig_scale_free.py`. Needs two live checkpoints
— one per wiring mode, run **twice**:

```bash
FIG10_RUN_ID=cdjkgrod FIG10_MODE=w python -m paper_figures.eval_fig10_scalefree_isn1   # fixed-wiring-trained, writes header
FIG10_RUN_ID=1u5ssulx FIG10_MODE=a python -m paper_figures.eval_fig10_scalefree_isn1   # random-wiring-trained, appends
python -m paper_figures.fig_scale_free   # -> out/fig_scale_free.pdf
```
Both write to `data/fig10_scalefree.csv` directly (no manual copy step). Compute is
heavy: 7 width factors × batch 64, damage on/off, per run id — expect several
minutes to an hour depending on hardware; `SMOKE_WIDTHS`/`SMOKE_BATCH` env vars
shrink it for a quick smoke test. **The fixed-vs-random mapping above is confirmed**
(directly, via each run's `loaded_config.training.wiring_mode`, not just inferred):
`cdjkgrod` prints `wiring_mode resolved to: fixed`, `1u5ssulx` prints
`wiring_mode resolved to: random`.

## Known Gaps

Two Regime II figures — the combined Soft-Error-Recovery/Hamming panel and the
UMAP degenerate-solution-space embedding — have **no reproducible script in this
branch's history**. They're Marcello Barylli's original work (per the manuscript's
Author Contributions), built and only ever committed on `origin/mergello`, a branch
that diverged before this codebase's refactor and was never merged. Porting them to
the current API is real, non-trivial work (old `boolean_nca_cc.models.self_attention`
/ `boolean_nca_cc.training.preconfigure` / `.training.backprop` module paths that
no longer exist) — not done here.

The original, unmodified source is archived at
[`paper_figures/archive/mergello_regime2/`](archive/mergello_regime2/) (with its own
README) so the pointer survives without needing git archaeology, and the `mergello`
branch itself is being kept alive on `origin` specifically to preserve this. If you
want these two figures, that archive folder is the starting point.

## Modules

| module | role |
|---|---|
| `style.py` | Okabe-Ito CVD-safe palette (+ `UNIFORM_PALETTE` toggle), rcParams, task/label maps |
| `stats.py` | Mann-Whitney U + Cliff's delta significance tests |
| `wandb_data.py` | pull TMT final-eval metrics (undamaged + damaged) from Fig 2's sweep history |
| `bp_from_history.py` | BP ceiling from each Fig 2 run's logged `BP_results/*` summary (exact, no JAX) — **canonical** for Fig 2 |
| `bp_baseline.py` | recomputes BP via live `run_bp_scan` — cross-check only |
| `fig2_fixed_wiring.py` | Fig 2: TMT vs BP bars, eval-damage × train-regime × task |
| `eval_fig4_resilience_isn1.py` / `fig_resilience.py` | Fig 4: resilience stepwise loss/accuracy under permanent damage |
| `eval_pca_trajectories.py` / `fig_pca_trajectories.py` | PCA trajectories of LUT-logit configs under damage |
| `fig_random_wiring.py` | Random-topology TMT accuracy, undamaged vs damaged |
| `eval_fig10_scalefree_isn1.py` / `fig_scale_free.py` | Fig 10: accuracy vs circuit width, fixed vs random wiring |
| `reeval_fig2.py` | deprecated full live re-eval of Fig 2 (drifted from Feb checkpoints) |
