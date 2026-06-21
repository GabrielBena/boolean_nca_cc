"""Principled, reproducible figure generation for the SODC ALIFE paper.

Replaces the ad-hoc notebook cells (trained_models.ipynb) with a small,
config-driven package:

  style       - colour-blind-safe palette, consistent rcParams, task/label maps
  stats        - Mann-Whitney U + Cliff's delta (no plotting)
  wandb_data   - pull a sweep group -> tidy long-form DataFrame (config-driven),
                 reading each run's *actual* config (resolves the test-split size)
  bp_baseline  - config-matched BP ceiling via run_bp_scan (fixes the notebook's
                 leaked-global BP cell)
  fig2_fixed_wiring - the redesigned Fig 2 (boxplots separated from BP, CVD-safe)
                      + significance / effect sizes

Run from the repo root, e.g.:
    python -m paper_figures.wandb_data            # pull + cache + report test_num
    python -m paper_figures.bp_baseline           # compute BP ceiling (needs JAX)
    python -m paper_figures.fig2_fixed_wiring      # build the figure + stats
"""
