# Archive: Regime II figures (Marcello's original work, unported)

These files produce two of the paper's Regime II figures:

- **Soft-Error Recovery / combined Hamming+accuracy panel**
  (`combined_hamming_accuracy_REplot_multi.png` in the manuscript) —
  `experiments/combined_damage_stepwise_plot.py`, using
  `experiments/visualization/plot_perturbation_utils.py` and
  `experiments/visualization/plot_stepwise_metrics.py`.
- **UMAP of the degenerate solution space** (`UMAP 2 Panel Horizontal...pdf`) —
  `experiments/visualize_umap.py`, built on the DFS/BFS exploration in
  `experiments/explore_degenerate_solutions.py`.

## Provenance

Copied verbatim (unmodified) from `origin/mergello` — the disjoint branch
containing Marcello Barylli's original work on this half of the paper (see the
manuscript's Author Contributions: "Marcello Barylli conducted the primary
experiments regarding functional maintenance"). `mergello`'s root commit
shares no common ancestor with `main`/`gabi` — it was never merged, so this
code doesn't otherwise exist anywhere in this branch's history.

## Status: known gap, not ported

This code targets a **pre-refactor API** — e.g.
`boolean_nca_cc.models.self_attention` (now
`boolean_nca_cc.models.attention.self_attention`),
`boolean_nca_cc.training.preconfigure`, `boolean_nca_cc.training.backprop`
(module paths that no longer exist) — and will not run against the current
codebase without adaptation. It's kept here as a reference/pointer, not as
working code. If you want to reproduce these two figures, this is the
starting point; expect to need to port the DFS exploration and plotting logic
onto the current `boolean_nca_cc.training.evaluation` /
`boolean_nca_cc.training.pool` APIs.

The `mergello` branch itself is being kept alive on origin specifically so
this pointer stays resolvable even after this archive copy — see
`paper_figures/README.md`'s Known Gaps section.
