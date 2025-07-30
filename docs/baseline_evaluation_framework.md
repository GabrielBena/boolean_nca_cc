# Baseline Evaluation Framework: GNN vs Backpropagation

## Current State Analysis

**Experimental Setup:**

- **GNN Approach**: Self-attention model learns to optimize boolean circuit LUTs under knockout patterns
- **Knockout Patterns**: Structural perturbations creating robustness challenges
- **Key Metric**: `eval_ko_in/final_hard_accuracy` (ID knockout evaluation)
- **Backprop Baseline**: Implemented in `run_backpropagation_training()` but unused

**Current WandB Visualizations:**

- Training curves: `training/hard_accuracy`, `training/loss`
- Knockout evaluation: `eval_ko_in/final_hard_accuracy`, `eval_ko_out/final_hard_accuracy`
- Pattern-level analysis: Scatter plots showing per-pattern performance over epochs

## Baseline Evaluation Framework Design

### 1. **Core Comparison Metrics**

**Performance Metrics:**

- **Final Hard Accuracy**: Direct comparison on knockout robustness
- **Sample Efficiency**: `n_message_steps` vs backprop iterations to reach target accuracy ✅ **EXISTING** - corresponds to "greater sample efficiency"
- **Network size** - parameter counts
- **Stability**: Variance in performance across knockout patterns
- **Generalization Gap**: ID vs OOD knockout pattern performance difference

**Robustness Metrics:**

- **Knockout Pattern Coverage**: % of patterns where method achieves >90% accuracy
- **Damage Resilience**: Performance degradation vs knockout damage probability ✅ **EXISTING** - corresponds to "fault tolerance at inference time"
- **Recovery Rate**: How quickly each method adapts to new knockout patterns ✅ **EXISTING** - corresponds to "adapts to damage"

### 2. **Experimental Design Options**

**A. Fixed Budget Comparison:**

```python
# Equal computation budget across methods
gnn_epochs = 8192  # Current setting
bp_equivalent_steps = gnn_epochs * n_message_steps * meta_batch_size
```

**CRITICAL NUANCE: The comparison should be `n_message_steps` vs backprop, not epochs vs epochs.**

The meta-learner's key advantage is that it learns to optimize circuits in `n_message_steps` (typically 5-20 steps), while backpropagation requires many more iterations to achieve the same performance. This is the core "greater sample efficiency" claim.

**Batch Dimension Clarification:**
- **GNN `meta_batch_size`**: Number of different circuits sampled from the pool, each with potentially different knockout patterns
- **Backprop batch**: Would correspond to multiple input-output pairs for the same circuit, or multiple circuits with different knockout patterns
- **Fair comparison**: Backprop should use equivalent batch size to process the same number of "circuit instances" per training step

**B. Knockout Pattern Evaluation Protocol:**

- **Shared Vocabulary**: Both methods evaluated on identical knockout patterns
- **Progressive Difficulty**: Start with easy patterns, increase damage probability
- **Cross-Validation**: Multiple random seeds for robust comparison

### 3. **Advanced Visualization Suite**

**A. Performance Dashboards:**

```python
# Comprehensive comparison panels
"comparison/final_hard_accuracy_bp_vs_gnn"
"comparison/convergence_speed_ratio"  
"comparison/robustness_score_difference"
```

**B. Robustness Analysis:**

- **Damage Tolerance Curves**: Accuracy vs knockout damage probability
- **Pattern Difficulty Heatmaps**: Success rate per pattern type
- **Learning Efficiency Plots**: Accuracy vs training steps (log scale)

**C. Statistical Analysis:**

- **Confidence Intervals**: Bootstrap resampling for robust comparisons
- **Significance Testing**: Statistical tests for performance differences
- **Effect Size Quantification**: Cohen's d for practical significance

### 4. **Benchmarking Best Practices**

**A. Fairness Considerations:**

```python
# Hyperparameter optimization for both methods
bp_config = {
    "learning_rates": [1e-3, 2e-4, 5e-4, 1e-4],
    "optimizers": ["adamw", "adam"],
    "batch_sizes": [32, 64, 128],  # Equivalent to meta_batch_size for fair comparison
    "epochs": [1000, 2000, 5000]  # Equivalent budget
}

# CRITICAL: Backprop should be compared on equivalent "circuit instances per step"
# GNN processes meta_batch_size circuits per training step
# Backprop should process equivalent number of circuit instances per step
```

**B. Evaluation Protocols:**

- **Early Stopping**: Prevent overfitting bias
- **Multiple Runs**: Statistical significance (n≥5 seeds)
- **Cross-Validation**: Pattern-level train/test splits

**C. Computational Efficiency:**

- **FLOPs Comparison**: Actual computational cost analysis
- **Memory Usage**: Peak memory consumption tracking
- **Wall-Clock Time**: Real-world efficiency metrics

### 5. **Implementation Strategy**

**Phase 1: Basic Comparison**

1. Integrate backprop baseline into main training loop
2. Run both methods on identical knockout vocabularies
3. Create side-by-side comparison dashboards

**Phase 2: Advanced Analysis**

1. Implement progressive difficulty evaluation
2. Add statistical significance testing
3. Create robustness benchmark suite

**Phase 3: Comprehensive Benchmarking**

1. Multi-task evaluation across circuit types
2. Computational efficiency analysis
3. Publication-ready comparison framework
