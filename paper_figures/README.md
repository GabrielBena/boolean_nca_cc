# paper_figures — reproducible Fig 2 for the SODC ALIFE paper

A config-driven replacement for the ad-hoc plotting cells in the (uncommitted)
`boolean_nca_exploration.ipynb`. Everything for Fig 2 comes from the **logged
W&B history** of the sweep of record — no model re-evaluation.

## Sweep of record
`gbena/boolean-nca-cc`, group **`sweep_20260212_141940`** (Feb-12) — 30 runs =
3 tasks × damage-train{on,off} × 5 seeds (`[0,1,2,4,5]`), `model=gathered_attention`,
fixed wiring, `test_num=256`. This is the sweep the published `_both` figure used.
Being *after* the Feb-11 "always eval damaged" change, it logs the damaged-eval
("green") branch as a real scalar (`eval_damaged_in_test/final_hard_accuracy`) plus a
config-matched BP baseline (`BP_results/{none,probabilistic}/*`). The earlier
`sweep_20260209_231530` predates damaged logging (NaN green) — do not use it.

## Rebuild Fig 2 (run with the `nca` conda env + W&B creds, from a neutral cwd)
```bash
python -m paper_figures.wandb_data --no-cache   # TMT undamaged + green -> data/fixed_wiring_sweep.csv
python -m paper_figures.bp_from_history         # BP ceiling (logged)   -> data/fixed_wiring_bp.csv
python -m paper_figures.fig2_fixed_wiring       # plot + stats -> out/fig2_fixed_wiring.pdf, out/fig2_stats.csv
```
Steps 1–2 hit W&B and cache to `data/`; step 3 reads the caches (no W&B), so
re-styling never needs W&B again. Run from e.g. `/tmp` so the repo's local `wandb/`
dir doesn't shadow the package. `data/` and `out/` are git-ignored.

## Modules
| module | role |
|---|---|
| `style.py` | Okabe–Ito CVD-safe palette, rcParams, task/label maps (TMT not NCA; "Binary Addition") — R1f, R2 |
| `stats.py` | Mann–Whitney U + Cliff's delta — R1g |
| `wandb_data.py` | pull TMT final-eval metrics (undamaged + damaged "green") from history; reads `training.test_num` per run — R1c |
| `bp_from_history.py` | BP ceiling from each run's logged `BP_results/*` summary (exact, no JAX) — **canonical** |
| `fig2_fixed_wiring.py` | the figure: TMT and BP as side-by-side mean±std bars, eval-damage on x, train-regime rows, task cols, per-panel stats |

## Notes
- `test_num = 256` → 256/4096 held out (R1c); printed by `wandb_data`.
- Damage mode = **probabilistic** (matches the published figure); the sweep also logs a
  discrete-damage variant (`eval_discrete_damaged_in_test/*`) if ever needed.
- `bp_baseline.py` *recomputes* BP via `run_bp_scan` (needs JAX) — kept only as a cross-check.
- `reeval_fig2.py` (full live re-eval) is **deprecated for Fig 2**: HEAD's model code drifted
  ~1085 lines from the Feb checkpoints, so it loads them to chance (TMT ≈ 0.5).
- Variance bands for Figs 3/8 are not pulled here yet.
