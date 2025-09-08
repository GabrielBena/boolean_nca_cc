## Visualization suite (critical, minimal, ML best-practice)

### Data sources (already logged)

- **training**: `training/loss`, `training/hard_loss`, `training/accuracy`, `training/hard_accuracy`, `training/epoch`, `scheduler/learning_rate`, `pool/reset_steps`, `pool/loss_steps`
- **knockout IN (seen)**: `eval_ko_in/final_loss`, `eval_ko_in/final_hard_loss`, `eval_ko_in/final_accuracy`, `eval_ko_in/final_hard_accuracy`, `eval_ko_in/epoch`
- **knockout OUT (unseen)**: `eval_ko_out/final_loss`, `eval_ko_out/final_hard_loss`, `eval_ko_out/final_accuracy`, `eval_ko_out/final_hard_accuracy`, `eval_ko_out/epoch`
- **per-step (optional enable)**: `eval_ko_in_steps/*`, `eval_ko_out_steps/*` for stepwise curves
- **damage events**: `pool/damaged_count`, `pool/damaged_fraction` per damage application event
- **pattern table (persistent)**: WandB table with columns [`epoch`, `pattern_id`, `hard_accuracy`, `knockout_diversity`] enabling epoch-wise aggregation (mean/std) and drift over time
- **hamming analysis (optional)**: CSVs saved in `hamming_analysis_dir`, fields include `per_gate_mean_hamming`, `final_hard_accuracy`, `method`

### Figure 1 — Training progress tracking (growth vs regeneration) ✅ COMPLETE

- **Goal**: Track generalization under persistent knockouts on seen patterns, with support for both single-mode and combined growth/regeneration visualization.
- **X**: periodic evaluation epoch (`eval_ko_in/epoch`).
- **Y**: accuracy. Two bands:
  - soft accuracy: mean ± std across patterns
  - hard accuracy: mean ± std across patterns
- **Aggregation**: compute per-epoch mean/std from the persistent pattern table for IN-distribution patterns. If the table is unavailable, fall back to the logged `eval_ko_in/final_accuracy`/`eval_ko_in/final_hard_accuracy` as point estimates without error bands.
- **Design**:
  - **Single mode**: Mode-aware colors (growth=blue, repair=red) with mode-specific titles
  - **Combined mode**: Dual curves with error bands (growth=blue, repair=red)
  - Consistent colors: soft (blue), hard (black); shaded std bands; no dual axes.

#### Implementation Details

**Core Module**: `experiments/visualization/figure1_training_progress.py`

**Key Functions**:

- `create_eval_plot_prog()` - Mode-aware single run visualization
- `create_combined_growth_regeneration_plot()` - WandB run aggregation for dual-mode comparison
- `plot_combined_growth_regeneration()` - Low-level dual-mode plotting
- `fetch_wandb_run_data()` - WandB data extraction

**Integration Points**:

- **Training Loop**: `boolean_nca_cc/training/train_loop.py` (lines ~1976-1995)
  - Automatic generation at training completion with mode-aware labeling
  - Final plot: `training_progress_final_{training_mode}.png`
- **CLI Tool**: `experiments/visualization/create_combined_plot.py`
  - Manual combination of WandB runs: `python create_combined_plot.py --growth-run-id X --repair-run-id Y`

**Data Sources**:

- **Primary**: Pattern data `[epoch, pattern_id, hard_accuracy, knockout_diversity]` with mean ± std aggregation
- **Fallback**: WandB logged metrics `eval_ko_in/final_hard_accuracy`, `eval_ko_in/final_accuracy`
- **Mode Detection**: `training_mode` parameter ("growth" or "repair")

**Output Locations**:

- **Single runs**: `reports/figures/training_progress_final_{training_mode}.png`
- **Combined runs**: `reports/figures/combined_growth_regeneration.png`
- **WandB**: Automatic logging of final plot only

**Single Run Walkthrough**:

When you run a single training session (e.g., `python train.py training.training_mode=growth`), here's exactly what gets plotted:

**During Training** (every `periodic_eval_interval` epochs, default=1024):

1. **Knockout Evaluation**: Runs IN-distribution and OUT-distribution knockout tests
2. **Pattern Data Accumulation**: Collects `[epoch, pattern_id, hard_accuracy, knockout_diversity]` for each pattern
3. **Hamming Analysis** (if enabled): Creates `accuracy_vs_distance_epoch_XXXX.png`

**At Training Completion**:

1. **Final Figure 1**: Calls `create_eval_plot_prog()` with all accumulated data
   - `filename`: `training_progress_final_{training_mode}.png` (e.g., `training_progress_final_growth.png`, `training_progress_final_repair.png`)
2. **WandB Logging**: Uploads final plot as `training_progress_final`

**Plot Content**:

- **Growth Mode**: Blue curves, "Training Progress - Growth Mode" title
- **Repair Mode**: Red curves, "Training Progress - Repair Mode" title
- **Error Bands**: Standard deviation across patterns (if pattern data available)
- **Fallback**: Point estimates without error bands (if only WandB metrics available)

**Usage Examples**:

```bash
# During training (automatic)
python train.py training.training_mode=growth  # → Blue plots, "Growth Mode" title
python train.py training.training_mode=repair  # → Red plots, "Repair Mode" title

# Post-training combination (manual)
cd experiments/visualization
python create_combined_plot.py --growth-run-id abc123 --repair-run-id def456
```

### Figure 2 — Performance vs training damage exposure (seen vs unseen)

- **Hypothesis**: As damaged examples in training increase, performance on seen decreases; performance on unseen increases.
- **X**: Damage patterns encountered during training. This is set in @config.yaml via the damage_knockout_diversity parameter. we should include 4 x ticks: 1, 16, 64, 256 (it should be easy to change these values and number of values to investigate.
- **Y**: hard accuracy with some visualisation of the standard deviation.
  - currently, we only care about the eval performance at the end of training. can we re-use final/hard_accuracy? Is this the eval of in or out? If it is one of the two, we should have an equivalent final eval for the other. They should both be plotted for each X (pattern number encountered during training)
- To implement, we will need to access results saved to wandb for multiple runs, where each run ha a different setting for damage_knockout_diversity. I already have sweep architecture in place, so don't worry about that. All we need to implement is the functionality surrounding data retrieval from wandb and the subsequent plotting functions as described above. Only implement one step at a time and report back to me in between

# Figure 2.5

- IMPORTANT FOLLOW UP FIGURE: STATE SIZE SCALING EFFECT
- Since our meta-learner has to handle a multitude of damage patterns, whereas backprop is optimised for a single pattern only, the demands on the network controlling updates is much higher.
- IN line with Kirsch results, we show that scaling the state size of the attention model allows it to perform better on higher sets of tasks, as well as improving its generalisation capabilities

### Figure 3 — Damage recovery and growth trajectories (stepwise) ✅ COMPLETE

- **Goal**: Show recovery dynamics over message-passing steps; only seen needed here
- **X**: message-passing step index (from `eval_ko_in_steps/step`).
- **Y**: accuracy. For each distribution plot mean ± std hard accuracy across evaluated patterns at each step.
- **Curves**: two panels or overlay with distinct color families. Optionally add soft accuracy as faint complement.
- **Notes**: Align step ranges (truncate longer series to the shorter, as handled in `plot_combined_bp_sa_stepwise_performance`).

#### Implementation Details

**Core Module**: `boolean_nca_cc/training/train_loop.py`

**Key Function**: `plot_combined_bp_sa_stepwise_performance()`

**Parameters**:

- `show_bp_trajectory=False`: Enables Figure 3 mode (BP as reference line)
- `show_bp_trajectory=True`: Original mode (full BP trajectory)

#### Standalone Implementation

**Core Module**: `experiments/visualization/figure3_standalone.py`

**Key Functions**:

- `create_figure3_from_wandb_run()` - Main function to generate Figure 3 from WandB run data
- `fetch_stepwise_data_from_wandb()` - Extract stepwise evaluation data from WandB
- `aggregate_stepwise_data_by_epoch()` - Compute mean ± std across patterns per step
- `create_figure3_plot()` - Generate the actual plot

**Usage**:

```bash
# Command line usage
python experiments/visualization/figure3_standalone.py --run-id abc123 --output-dir reports/figures

# With BP reference
python experiments/visualization/figure3_standalone.py --run-id abc123 --bp-run-id def456 --output-dir reports/figures

# Programmatic usage
from experiments.visualization.figure3_standalone import create_figure3_from_wandb_run
output_path = create_figure3_from_wandb_run(
    run_id="abc123",
    bp_run_id="def456",  # optional
    output_dir="reports/figures"
)
```

**Data Sources**:

- **Primary**: WandB logged metrics `eval_ko_in_steps/step`, `eval_ko_in_steps/hard_accuracy`, `eval_ko_in_steps/epoch`
- **BP Reference**: `eval_ko_in/final_hard_accuracy` from BP run (optional)
- **Aggregation**: Mean ± std across patterns per step

**Output**: `damage_recovery_trajectories_{training_mode}_standalone.png`

**Benefits**:

- No need to rerun training loops for plot prototyping
- Reuses existing WandB logged data
- Supports BP reference line from separate runs
- Identical visualization to integrated training loop version

**Integration Points**:

- **Training Loop**: `boolean_nca_cc/training/train_loop.py` (lines ~1983-2050)
  - Automatic generation at end of training
  - Uses knockout vocabulary and base circuit from training
  - Creates proper config object for backprop evaluation
  - Saves to `reports/figures/damage_recovery_trajectories_{training_mode}.png`

**Data Sources**:

- **SA Trajectory**: Per-pattern hard accuracies with mean ± std error bands
- **BP Reference**: Final BP accuracy as horizontal dotted line
- **Pattern Data**: Uses knockout vocabulary from training configuration
- **Base Circuit**: Uses knockout evaluation base circuit

**Visual Design**:

- **SA Trajectory**: Cyan solid line with semi-transparent error bands (α=0.2)
- **BP Reference**: Orange horizontal dotted line with accuracy value in label
- **Error Bands**: Standard deviation across patterns at each step
- **Title**: "Damage Recovery and Growth Trajectories (SA vs BP Reference)"
- **Legend**: Positioned outside plot area to avoid overlap

**Output Locations**:

- **File**: `reports/figures/damage_recovery_trajectories_{training_mode}.png`
- **WandB**: Logged as `"figure3_damage_recovery_trajectories"`
- **Mode-Aware**: Filename includes training mode (growth/repair)

**Technical Features**:

- **Dimension Handling**: Automatic alignment of step arrays and accuracy data
- **Config Management**: Creates proper config object from training parameters and backprop config
- **Error Handling**: Graceful fallback to simple line plot if per-pattern data unavailable
- **Resource Management**: Proper matplotlib figure cleanup after saving

**Usage**:

```python
# Automatic generation at end of training (when conditions met)
# Manual usage:
fig = plot_combined_bp_sa_stepwise_performance(
    cfg=config_object,
    x_data=x_data,
    y_data=y_data,
    loss_type="l4",
    knockout_patterns=knockout_vocabulary,
    model=trained_model,
    base_circuit=base_circuit,
    n_message_steps=30,
    show_bp_trajectory=False  # Figure 3 mode
)
```

### Figure 4 — Accuracy vs. Hamming distance (by method)

- **Goal**: Connect structural deviation to task performance; compare SA vs backprop if available.
- **X**: `per_gate_mean_hamming`
- **Y**: `final_hard_accuracy`
- **Grouping**: color by `method` in CSVs (e.g., `gnn`, `bp`). Optionally annotate mean ± std of distance per method.
- **Source**: CSVs emitted in `hamming_analysis_dir`; PNG already saved via `plot_accuracy_vs_distance`.

### Figure 5 — Training dynamics (loss/accuracy, resets, LR)

- **Panels**: 3 aligned time series with shared X (epoch):
  - training losses: `training/loss`, `training/hard_loss` (lines)
  - training accuracies: `training/accuracy`, `training/hard_accuracy` (lines)
  - learning rate: `scheduler/learning_rate` (line), plus `pool/reset_steps` (bar or stem for reset magnitude)
- **Purpose**: Diagnose stability, overfitting, and influence of resets/schedule.

## Implementation notes

- **Epoch alignment**: Evaluate-only metrics update at `periodic_eval_interval`. Use the logged `eval_ko_* / epoch` values for X; when cross-relating with damage counts, interpolate cumulative damage to nearest prior eval epoch.
- **Error bands**: Prefer std across patterns (from the persistent table). If using standard error, show as lighter band, still keep std as main band if needed.
- **Color map**: hard accuracy (black), soft accuracy (blue). IN (cyan accents), OUT (orange accents). Methods in Figure 4: SA (cyan), BP (orange).
- **Smoothing**: only for Figure 2; use window=3 running median; no smoothing for others.
- **Downsampling**: if periodic eval is very frequent, keep every k-th point (k chosen to retain temporal shape without clutter).
- **CI vs std**: std preferred; CI optional via bootstrap if sample size per eval is small; keep consistent presentation across figures.

## Minimal data transformations

- **Cumulative damage**: `cum_damaged_examples[epoch] = sum(pool/damaged_count[:=epoch])`.
- **Per-epoch means/std**: group pattern table by `epoch`; compute mean and std for `hard_accuracy` (and soft if available in table in future).
- **Join logic**: for Figure 2, left-join per-epoch accuracy onto cumulative damage at the nearest prior epoch.

## Output and storage

- **Static artifacts**: save figures under `reports/figures/` with deterministic names: `knockout_generalization.png`, `damage_vs_accuracy.png`, `recovery_stepwise.png`, `hamming_accuracy.png`, `training_dynamics.png`.
- **WandB**: log the same figures; reuse existing WandB table for pattern-level aggregation; maintain identical titles and legends.

## Implementation Status

### ✅ Completed Figures

**Figure 1 — Training progress tracking (growth vs regeneration)**

- **Status**: Complete and integrated
- **Module**: `experiments/visualization/figure1_training_progress.py`
- **Integration**: Automatic generation during training and at completion
- **Output**: `training_progress_final_{training_mode}.png`

**Figure 3 — Damage recovery and growth trajectories (stepwise)**

- **Status**: Complete and integrated
- **Module**: `boolean_nca_cc/training/train_loop.py`
- **Function**: `plot_combined_bp_sa_stepwise_performance(show_bp_trajectory=False)`
- **Integration**: Automatic generation at end of training
- **Output**: `damage_recovery_trajectories_{training_mode}.png`

### 🔄 Pending Figures

**Figure 2 — Performance vs training damage exposure (seen vs unseen)**

- **Status**: Not implemented
- **Requirements**: Cumulative damage tracking, interpolation to eval epochs

**Figure 4 — Accuracy vs. Hamming distance (by method)**

- **Status**: Partially implemented (CSV generation exists)
- **Requirements**: Visualization integration with existing hamming analysis

**Figure 5 — Training dynamics (loss/accuracy, resets, LR)**

- **Status**: Not implemented
- **Requirements**: Multi-panel time series visualization

## Rationale

- Focus on generalization under structural perturbations (Figure 1 ✅, 2, 3 ✅).
- Link structure to function (Figure 4).
- Track optimization and stability (Figure 5).
