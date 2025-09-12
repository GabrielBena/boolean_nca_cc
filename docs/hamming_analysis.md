## Hamming Distance Analysis: Post-Training, Script-First Strategy

### Overview

Perform Hamming-based analysis and plotting after training. Use `experiments/hamming_distance.py` as the single entry point to load a trained GNN checkpoint, evaluate on reproducible knockout vocabularies, compute truth-table Hamming distances against a baseline, and generate unified visualizations. Avoid modifying `training/train_loop.py` and avoid introducing analyzer classes.

### Goals

- **Post-training analysis**: All Hamming analysis runs outside the training loop.
- **Method comparison**: Compare GNN evaluation to Backpropagation (BP) on the same KO patterns.
- **Damage modes**: Support both `shotgun` and `strip` via existing utilities.
- **Unified plots**: Reuse existing visualization to show accuracy vs distance and circular plots.
- **Zero duplication**: Reuse existing functions in `perturbation.py`, `circular_plot.py`, `model.py`, and `checkpointing.py`. No new classes.

### Principles

- Keep training core free of analysis/plotting dependencies.
- Use existing helpers already present in `experiments/hamming_distance.py` for truth-table conversion, masks, and distance metrics.
- Reuse knockout vocabulary generation from `training/pool/perturbation.py` (already implemented for hamming distance analysis of backprop mode).
- Reuse plotting from `experiments/visualization/circular_plot.py`.
- Reuse circuit utilities from `boolean_nca_cc/circuits/model.py`.
- Leverage existing checkpoint loading from `boolean_nca_cc/training/checkpointing.py` for GNNs.

### Inputs and Artifacts

- Trained GNN checkpoint produced by `training/train_loop.py` (model state, optionally wires/logits if saved).
- The config used for the run (`configs/config.yaml`) to reconstruct `layer_sizes`, `arity`, and seeds.
- Optional: the same seed/key used for evaluation wiring to reproduce base circuits.

### Script Enhancements (`experiments/hamming_distance.py`)

Do not introduce new classes. Extend the existing script with minimal additions:

- **CLI arguments**:

  - `--checkpoint`: path to a saved GNN checkpoint
  - `--methods`: `gnn`, `bp`, or `both` (default: `gnn`)
  - `--damage-modes`: comma-list of `shotgun,strip` (default: config value)
  - `--vocab-size`: size of KO vocabulary (default: config)
  - `--damage-prob`: number of knockouts (default: config)
  - `--n-message-steps`: inner steps for GNN evaluation (default: e.g., 50–100)
  - `--loss-type`: `l4` or `bce` (default: config)
  - `--output`: results directory (existing default supported)

- **GNN loading and baseline (leverage existing checkpointing)**:

  - Use `load_best_model_from_wandb(...)` from `boolean_nca_cc/training/checkpointing.py` to load the trained model, config, and full state in a single call.
  - Reconstruct or load the base circuit `(wires, logits)` using the same seed/key as evaluation.
  - Run a baseline forward pass (no knockouts) to get baseline logits and convert to hard tables using existing helpers in the script.

- **Knockout vocabulary (reuse, no redefinition)**:

  - `create_knockout_vocabulary(...)` from `training/pool/perturbation.py` for both `shotgun` and `strip`.
  - Use identical seeds across methods to ensure pattern parity.

- **GNN per-pattern evaluation**:

  - Forward the loaded GNN on base wiring/logits with each KO pattern, without parameter updates.
  - Convert final logits to hard tables and compute distances vs baseline using the script's existing helpers.

- **BP comparison (already present)**:

  - Keep baseline BP and per-pattern BP training/evaluation as implemented.
  - Aggregate results alongside GNN using the same metrics.

### Reused Functionality (no duplication)

- **Checkpoint loading**: `load_best_model_from_wandb` from `boolean_nca_cc/training/checkpointing.py` for seamless one-step model restoration from WandB.
- **Damage pattern generation**: `create_knockout_vocabulary` (and underlying `create_reproducible_knockout_pattern` / `create_strip_knockout_pattern`) from `training/pool/perturbation.py`.
- **Circuit utilities**: `generate_layer_sizes`, `gen_circuit` from `boolean_nca_cc/circuits/model.py`.
- **Visualization**: `plot_circular_knockout_distances`, `plot_accuracy_vs_distance` from `experiments/visualization/circular_plot.py`.
- **Truth tables, masks, and distance math**: use the already-implemented helpers in `experiments/hamming_distance.py`.

### Workflow

1. Load config and resolve `layer_sizes` with `generate_layer_sizes` if missing.
2. **Load GNN checkpoint using existing checkpointing**: Use `load_best_model_from_wandb(run_id, filters, seed, ...)` to restore the trained model, config, and full state in a single call.
3. Construct or load base circuit `(wires, logits)` using the same evaluation seed/key.
4. For each selected `damage_mode`:
   - Build a KO vocabulary with `create_knockout_vocabulary`.
   - Compute GNN baseline hard tables.
   - For each KO pattern:
     - Forward GNN without updates; compute hard tables; compute Hamming distances vs baseline; record final accuracy/loss.
   - If BP is enabled, run the existing BP baseline and per-pattern training/evaluation and compute the same metrics.
5. Aggregate all rows (patterns × methods × modes) into a single DataFrame and write `summary.csv`.
6. Optionally compute and save `pairwise_matrix.csv` using existing helpers.
7. Generate plots using `circular_plot.py` functions; if both methods/modes are present, produce unified comparison plots.

### Data Model

Each row in `summary.csv` contains:

- `pattern_idx`, `damage_mode` in {`shotgun`, `strip`}
- `method` in {`gnn`, `bp`} (BP rows present only if enabled)
- `overall_bitwise_fraction_diff`, `per_gate_mean_hamming`
- `counted_bits_total`, `counted_gates_total`
- `final_accuracy`, `final_hard_loss`
- `per_layer_bitwise_fraction_diff` serialized as JSON

### Outputs

- `summary.csv`, `pairwise_matrix.csv`
- `circular_distances.png` (per mode or unified)
- `accuracy_vs_distance.png` (unified comparison)
- `report.json` with means and artifact paths

### Configuration & Reproducibility

- Read seeds from the config; reuse the same `wiring_fixed_key` or equivalent seed to reproduce base circuits.
- Use the same RNG seed for KO vocabulary across methods to ensure identical patterns.
- Leverage the existing checkpoint loading infrastructure for consistent model restoration.

### Non-Goals

- No train-loop integration for this analysis.
- No analyzer classes.
- No redefinition of functionality from `perturbation.py`, `circular_plot.py`, `model.py`, or `checkpointing.py`.

### Rationale

This approach keeps core training fast, stable, and free of heavy plotting/dataframe dependencies while enabling fair, reproducible comparisons across methods and damage modes using a single, script-first entry point. By leveraging existing checkpointing utilities, we ensure consistent model loading and avoid duplicating checkpoint restoration logic.

### Future Extensions (optional)

- Add additional damage modes (targeted, progressive) by extending KO generation in `perturbation.py` only if needed.
- Add method markers/colors in plots by lightly extending `plot_accuracy_vs_distance` without changing defaults.
