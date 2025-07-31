# BP vs SA Comparison Framework

## Overview

Framework to compare Backpropagation (BP) vs Self-Attention (SA) performance on knockout patterns from a shared vocabulary.

## Data Structures

### Backprop Results (`bp_results`)

- **Location**: `train.py` → `_run_backpropagation_training_with_knockouts()`
- **Structure**: `bp_results["patterns_performance"]` - list of pattern results
- **Indexing**: Sequential 0-based indexing matching vocabulary order
- **Key data**: `{"pattern_idx": idx, "final_hard_accuracy": acc, "knockout_pattern": pattern}`

### Self-Attention Results

- **Location**: `train_loop.py` → `run_knockout_periodic_evaluation()`
- **Structure**: `step_metrics_in["per_pattern"]["pattern_hard_accuracies"][-1]`
- **Indexing**: Array indices corresponding to sampled vocabulary patterns

## Implementation Strategy

### 1. Final Evaluation Point

- **Rationale**: SA samples different patterns during resets; comprehensive eval needed at end
- **BP results**: Already complete from training start

### 2. Vocabulary-Based Direct Mapping

- **Approach**: Use vocabulary indices directly instead of random sampling
- **Benefit**: Guaranteed pattern-to-pattern pairing between BP and SA
- **Implementation**: Replace random sampling with `jp.arange(len(knockout_vocabulary))`

### 3. Comparison Function (function defined in in utils.py)

```python
def compare_bp_sa_performance(bp_results, sa_pattern_results, vocabulary_patterns):
    # Extract BP results by pattern index
    bp_pattern_accuracies = {p["pattern_idx"]: p["final_hard_accuracy"] 
                           for p in bp_results["patterns_performance"]}
  
    # SA results already indexed by vocabulary position
    sa_pattern_accuracies = sa_pattern_results
  
    # Prepare data for custom visualization
    pattern_indices = []
    bp_accuracies = []
    sa_accuracies = []
  
    for pattern_idx in range(len(vocabulary_patterns)):
        bp_acc = bp_pattern_accuracies[pattern_idx]
        sa_acc = sa_pattern_accuracies[pattern_idx]
      
        pattern_indices.append(pattern_idx)
        bp_accuracies.append(bp_acc)
        sa_accuracies.append(sa_acc)
  
    # Calculate aggregate metrics
    comparisons = []
    for i, pattern_idx in enumerate(pattern_indices):
        bp_acc = bp_accuracies[i]
        sa_acc = sa_accuracies[i]
      
        comparisons.append({
            "pattern_idx": pattern_idx,
            "bp_hard_accuracy": bp_acc,
            "sa_hard_accuracy": sa_acc,
            "accuracy_difference": sa_acc - bp_acc,
            "relative_improvement": (sa_acc - bp_acc) / bp_acc if bp_acc > 0 else 0
        })
  
    return {
        # Raw data for custom plotting
        "plot_data": {
            "pattern_indices": pattern_indices,
            "bp_accuracies": bp_accuracies,
            "sa_accuracies": sa_accuracies
        },
        # Aggregate metrics
        "comparison/patterns": comparisons,
        "comparison/mean_bp_accuracy": np.mean(bp_accuracies),
        "comparison/mean_sa_accuracy": np.mean(sa_accuracies),
        "comparison/mean_improvement": np.mean([c["accuracy_difference"] for c in comparisons]),
        "comparison/patterns_better_than_bp": sum(1 for c in comparisons if c["accuracy_difference"] > 0),
        "comparison/total_patterns": len(comparisons)
    }
```

### 4. Custom Visualization (in function defined in utils.py)

```python
def plot_bp_vs_sa_comparison(comparison_results):
    """Create custom plot: x=pattern_index, y=hard_accuracy, red=BP, green=SA"""
    import matplotlib.pyplot as plt
  
    plot_data = comparison_results["plot_data"]
  
    plt.figure(figsize=(12, 6))
    plt.scatter(plot_data["pattern_indices"], plot_data["bp_accuracies"], 
               c='red', s=50, alpha=0.7, label='Backpropagation')
    plt.scatter(plot_data["pattern_indices"], plot_data["sa_accuracies"], 
               c='green', s=50, alpha=0.7, label='Self-Attention')
  
    plt.xlabel('Pattern Index')
    plt.ylabel('Hard Accuracy')
    plt.title('BP vs SA Performance by Pattern')
    plt.legend()
    plt.grid(True, alpha=0.3)
  
    # Add mean lines
    mean_bp = comparison_results["comparison/mean_bp_accuracy"]
    mean_sa = comparison_results["comparison/mean_sa_accuracy"]
    plt.axhline(y=mean_bp, color='red', linestyle='--', alpha=0.5, label=f'BP Mean: {mean_bp:.3f}')
    plt.axhline(y=mean_sa, color='green', linestyle='--', alpha=0.5, label=f'SA Mean: {mean_sa:.3f}')
  
    return plt.gcf()
```

### 5. Integration Points

- **TODO**

## Key Metrics

- Pattern-by-pattern hard accuracy comparison
- Mean performance across vocabulary
- Number of patterns where SA outperforms BP
- Relative improvement percentages
- Distribution of performance differences

## Configuration

- **Vocabulary size**: inherits `knockout_diversity` in config (default: 16)
- **Evaluation batch**: since comparison is at end, it should be on entire vocabulary (which is of size knockout_diversity)
- **Final eval**: Triggered at last periodic evaluation epoch
