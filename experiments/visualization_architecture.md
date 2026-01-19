# Visualization Architecture Overview

This document describes the structure and data flow patterns of the visualization/figure generation codebase.

## Architecture Layers

### Layer 1: Core Plotting Functions (Pure Visualization)

**Location**: `experiments/visualization/`

Pure plotting functions that take pre-computed data and generate figures. No data loading or checkpoint access.

#### `plot_trajectory.py`
- `plot_inner_loop_trajectory()` - Unified trajectory plotting
  - Supports: `"boolean_discovery"` (train/test generalization) or `"damage_response"` (damage recovery)
  - Input: Pre-computed metrics dictionaries
  - Output: matplotlib Figure
- `plot_combined_bp_sa_stepwise_performance()` - Legacy wrapper (backward compatibility)

#### `plot_perturbation_utils.py`
- `plot_accuracy_vs_distance()` - Scatter plot (accuracy vs hamming distance)
- `plot_damage_size_vs_hamming()` - Line plot with error bars (damage size vs hamming)
- Input: pandas DataFrame
- Output: Saved image file

#### `figure_config.py`
- Centralized styling (colors, fonts, sizes, DPI)
- Used by all plotting functions for consistency
- Paper-format optimized (double-column width)

### Layer 2: Data Loading & Evaluation

**Location**: `experiments/`

Utilities for loading models, running evaluations, and preparing data.

#### `viz_base_checkpoint_test.py`
- `load_model_and_data()` - Loads model from WandB checkpoint + prepares data splits
- `run_eval_no_damage()` - Runs evaluation matching training loop patterns
- **Pattern**: Single-run checkpoint loading
- **Used by**: `fig_boolean_discovery.py` (single-run mode)

#### `hamming_distance_plot.py`
- Full experiment script (not just visualization)
- Loads GNN model from checkpoint
- Runs BP training per knockout pattern
- Runs GNN evaluation per knockout pattern
- Computes hamming distances
- **Calls**: `plot_perturbation_utils.py` functions
- **Output**: CSV summary + plots

### Layer 3: Figure Generation Scripts (Orchestration)

**Location**: `experiments/visualization/`

High-level scripts that orchestrate data loading, processing, and plotting.

#### `fig_boolean_discovery.py`
**Two modes:**
1. **Single-run mode**: 
   - Loads checkpoint → runs eval → plots
   - Uses `viz_base_checkpoint_test.load_model_and_data()`
2. **Sweep mode**: 
   - Loads metrics from WandB/local storage → aggregates → plots
   - Uses `aggregate_stepwise_metrics_from_sweep()`

**Data sources:**
- WandB checkpoints (via `viz_base_checkpoint_test.py`)
- WandB metrics API (for stepwise metrics)
- Local storage (`results/metrics/`) for sweep aggregation

**Calls**: `plot_trajectory.plot_inner_loop_trajectory()`

#### `fig_backprop_discovery.py`
- Trains BP model from scratch (no checkpoint)
- Evaluates on train/test splits
- Plots generalization trajectory
- **Self-contained** - generates its own data

#### `plot_eval_no_damage_training.py`
- Loads metrics from local storage (sweep-organized)
- Plots training vs eval performance over epochs
- **Data source**: Local storage only (`results/metrics/sweep_{id}/`)

## Data Flow Patterns

### Pattern A: Checkpoint → Evaluation → Plot
```
WandB Checkpoint 
  → load_model_and_data() 
  → run_eval_no_damage() 
  → plot_trajectory.plot_inner_loop_trajectory()
```
**Example**: `fig_boolean_discovery.py` (single-run mode)

### Pattern B: Stored Metrics → Aggregate → Plot
```
Local Storage / WandB API
  → aggregate_stepwise_metrics_from_sweep()
  → plot_trajectory.plot_inner_loop_trajectory()
```
**Example**: `fig_boolean_discovery.py` (sweep mode), `plot_eval_no_damage_training.py`

### Pattern C: Experiment → Compute → Plot
```
Checkpoint + Config
  → Run experiments (BP + GNN evaluations)
  → Compute metrics (hamming distances, accuracies)
  → plot_perturbation_utils.plot_*()
```
**Example**: `hamming_distance_plot.py`

## Current Structure Map

```
experiments/
├── visualization/
│   ├── core/                    # Pure plotting functions
│   │   ├── plot_trajectory.py
│   │   ├── plot_perturbation_utils.py
│   │   └── figure_config.py
│   ├── data_loading/            # Data loading utilities
│   │   ├── checkpoint_loader.py  # Consolidate checkpoint loading
│   │   ├── metrics_loader.py     # Consolidate metrics loading
│   │   └── evaluation_runner.py  # Standardized evaluation patterns
│   └── figures/                 # Figure generation scripts
│       ├── fig_boolean_discovery.py
│       ├── fig_backprop_discovery.py
│       └── fig_hamming_analysis.py  # Move from experiments/
└── hamming_distance_plot.py     # Keep as experiment script
```

## Script Connections

```
hamming_distance_plot.py
  └─→ plot_perturbation_utils.py 
      (plot_accuracy_vs_distance, plot_damage_size_vs_hamming)

fig_boolean_discovery.py
  ├─→ viz_base_checkpoint_test.py 
      (load_model_and_data, run_eval_no_damage)
  ├─→ wandb_loader.py (find_runs)
  └─→ plot_trajectory.py (plot_inner_loop_trajectory)

fig_backprop_discovery.py
  └─→ (self-contained, generates own data)

plot_eval_no_damage_training.py
  ├─→ metrics_storage.py (load_sweep_metrics)
  └─→ figure_config.py (styling)

plot_trajectory.py
  └─→ figure_config.py (styling)
```

## Key Insights

### Strengths
1. **Clear separation**: Plotting functions are pure (no side effects, no data loading)
2. **Centralized styling**: `figure_config.py` ensures consistency
3. **Flexible data sources**: Supports checkpoints, WandB API, and local storage

### Inconsistencies
1. **Mixed locations**: Some scripts in `experiments/`, others in `experiments/visualization/`
2. **Duplicate patterns**: Checkpoint loading appears in multiple places
3. **Inconsistent naming**: `fig_*.py` vs `plot_*.py` vs experiment scripts
4. **Scattered data loading**: `viz_base_checkpoint_test.py`, `wandb_loader.py`, `metrics_storage.py`

### Refactoring Opportunities

**Proposed Structure**:
```
experiments/
├── visualization/
│   ├── core/                    # Pure plotting functions
│   │   ├── plot_trajectory.py
│   │   ├── plot_perturbation_utils.py
│   │   └── figure_config.py
│   ├── data_loading/            # Data loading utilities
│   │   ├── checkpoint_loader.py  # Consolidate checkpoint loading
│   │   ├── metrics_loader.py     # Consolidate metrics loading
│   │   └── evaluation_runner.py  # Standardized evaluation patterns
│   └── figures/                 # Figure generation scripts
│       ├── fig_boolean_discovery.py
│       ├── fig_backprop_discovery.py
│       └── fig_hamming_analysis.py
└── hamming_distance_plot.py     # Keep as experiment script
```

**Key Improvements**:
1. **Consolidate checkpoint loading** into one module
2. **Standardize metrics loading** (unified interface for WandB vs local storage)
3. **Create evaluation runner patterns** (reusable evaluation workflows)
4. **Clear naming convention**: 
   - `core/` = pure plotting
   - `figures/` = orchestration
   - `data_loading/` = data access

## Recommendations

1. **Keep separation**: Plotting functions should remain pure (no data loading)
2. **Consolidate data loading**: Create unified `data_loading/` module
3. **Standardize figure scripts**: All `fig_*.py` should follow same pattern
4. **Document data flow**: Add docstrings explaining data source → processing → plot
5. **Consider base class**: `FigureGenerator` with standardized `load_data()`, `process_data()`, `plot()` methods

## Data Sources Summary

| Source | Location | Used By | Purpose |
|--------|----------|---------|---------|
| WandB Checkpoints | `checkpointing.py` | `viz_base_checkpoint_test.py`, `hamming_distance_plot.py` | Load trained models |
| WandB Metrics API | `wandb_loader.py` | `fig_boolean_discovery.py` | Load logged metrics |
| Local Storage | `metrics_storage.py` | `fig_boolean_discovery.py`, `plot_eval_no_damage_training.py` | Load sweep metrics (organized by sweep_id) |
| Generated On-the-fly | Various | `fig_backprop_discovery.py`, `hamming_distance_plot.py` | Run experiments to generate data |
