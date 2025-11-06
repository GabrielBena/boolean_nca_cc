# Paper Results and Analysis Plan

## Core Results Already Demonstrated ✅

1. **Perfect Recovery for Reversible LUT Damage** - Including multiple consecutive perturbation events
2. **Multi-damage Robustness** - Circuits can recover from sequential damage events
3. **Seen vs Unseen Pattern Generalization** - Evaluation distinguishes IN/OUT distribution performance

---

## Essential Results for Paper (Must-Have)

### 1. **Recovery Trajectories Across Damage Modes**

**What to Show:**
- Recovery curves for permanent vs reversible damage modes
- Recovery speed (steps to recovery) for different damage magnitudes
- Success rate (% of patterns achieving perfect recovery) by damage type

**Metrics:**
- `recovery_speed`: Steps until accuracy ≥ threshold (e.g., 0.95)
- `final_recovery_rate`: Fraction of patterns achieving target accuracy
- `recovery_trajectory`: Step-by-step accuracy curve with error bands

**Visualization:**
- Time-series plots showing accuracy over message steps
- Bar charts comparing recovery rates across modes
- Heatmaps showing recovery success by damage magnitude × pattern type

**Implementation Notes:**
- Already have stepwise metrics (`step`, `hard_accuracy`) in `evaluate_model_stepwise_batched`
- Need to compute recovery speed from stepwise trajectories
- Need to aggregate across damage modes

### 2. **Damage Magnitude Sensitivity Analysis**

**What to Show:**
- How recovery performance scales with number of damaged gates
- Critical damage thresholds where recovery becomes unlikely
- Relationship between damage probability and final accuracy

**Metrics:**
- `damage_magnitude`: Number of gates knocked out per pattern
- `recovery_threshold_curve`: Damage magnitude vs recovery success rate
- `damage_robustness_metric`: Maximum damage before recovery failure

**Visualization:**
- Scatter plot: Damage magnitude (x) vs Final Accuracy (y)
- Threshold analysis showing success rate by damage bins
- Critical damage visualization (circuit diagram highlighting critical gates)

**Implementation Notes:**
- `damage_prob` controls damage magnitude - vary this systematically
- Already track per-pattern accuracies - need to add damage count per pattern

### 3. **Generalization: Seen vs Unseen Patterns**

**What to Show:**
- IN-distribution (vocabulary) vs OUT-distribution (fresh patterns) performance
- Overfitting analysis - difference between seen and unseen
- Pattern diversity impact on generalization

**Metrics:**
- `generalization_gap`: `eval_ko_in/final_hard_accuracy - eval_ko_out/final_hard_accuracy`
- `overfitting_ratio`: Ratio of IN to OUT accuracy
- `vocabulary_coverage`: Impact of vocabulary size on generalization

**Visualization:**
- Side-by-side comparison of IN vs OUT trajectories (already partially implemented)
- Scatter plot: IN accuracy vs OUT accuracy (ideal: diagonal line)
- Generalization gap over training epochs

**Implementation Notes:**
- Already tracking `eval_ko_in` and `eval_ko_out` separately
- Need to compute gap metric and plot over time

### 4. **Self-Attention vs Backpropagation Comparison**

**What to Show:**
- Recovery trajectories side-by-side
- Sample efficiency (steps to recovery)
- Final accuracy comparison
- Configuration change magnitude (Hamming distance)

**Metrics:**
- `recovery_time_sa_vs_bp`: Steps to reach threshold for each method
- `final_accuracy_difference`: `sa_final_acc - bp_final_acc`
- `config_change_magnitude`: Hamming distance from baseline (already computed!)
- `convergence_rate`: Rate of accuracy improvement per step

**Visualization:**
- Combined trajectory plots (already partially implemented in `plot_combined_bp_sa_stepwise_performance`)
- Scatter plot: SA accuracy vs BP accuracy for same patterns
- Hamming distance comparison: SA vs BP (already have data structure!)
- Convergence speed comparison (derivative of accuracy curves)

**Implementation Notes:**
- Already have BP results structure in `_run_backpropagation_training_with_knockouts`
- Already computing hamming distances separately for GNN and BP in `_analyze_knockout_hamming_distances`
- Need to add side-by-side hamming distance comparison visualization

---

## Advanced Analysis (Should-Have)

### 5. **Detailed Perturbation Response Metrics**

**Perturbation Response Characteristics:**

#### A. **Recovery Dynamics**
- `initial_drop`: Accuracy drop immediately after damage (step 1)
- `recovery_rate`: Rate of accuracy improvement (d²accuracy/dt²)
- `overshoot`: Whether accuracy exceeds pre-damage baseline
- `stability`: Variance in accuracy during recovery

#### B. **Configuration Adaptation**
- `gate_change_distribution`: Which gates change most (layer-wise, position-wise)
- `local_vs_global_changes`: Nearby gates vs distant gates changing
- `change_magnitude_by_layer`: Hamming distance per layer
- `critical_path_preservation`: Whether gates on critical paths remain stable

#### C. **Damage Pattern Sensitivity**
- `greedy_vs_random_response`: How response differs for greedy vs shotgun damage
- `layer_sensitivity`: Which layers are most/least sensitive to damage
- `spatial_locality`: Does damage response respect circuit topology?

**Metrics to Track:**
```python
{
    "recovery_dynamics": {
        "initial_drop": float,          # Accuracy drop at damage point
        "recovery_steps": int,          # Steps to recovery
        "recovery_rate": float,        # Slope of recovery curve
        "final_overshoot": float,       # Final accuracy - pre_damage accuracy
        "recovery_variance": float,     # Variance during recovery phase
    },
    "config_adaptation": {
        "total_hamming_distance": float,  # Already tracked!
        "per_layer_hamming": List[float],  # Already tracked!
        "gate_change_count": int,          # Number of gates with changed LUTs
        "max_single_gate_change": float,   # Max hamming for any single gate
        "layer_change_ratio": List[float], # Per-layer change ratios
    },
    "damage_sensitivity": {
        "damage_magnitude": int,        # Number of gates damaged
        "damage_layer_distribution": List[int],  # Damage per layer
        "response_magnitude": float,   # Total config change magnitude
        "spatial_spread": float,       # Spread of changed gates
    }
}
```

**Visualization:**
- Heatmaps showing gate-level changes (circuit visualization)
- Layer-wise change magnitude bar charts
- Recovery dynamics curves (derivatives, variance bands)
- Damage pattern overlay on circuit diagrams

**Implementation Notes:**
- Need to extract stepwise logits to compute recovery dynamics
- Already have hamming distance analysis - can extend for gate-level detail
- Need circuit visualization utilities for gate-level changes

### 6. **Functional Invariance Analysis** (Critical for Paper!)

**Core Argument:** Function computation remains invariant even when LUT configurations change

#### A. **Truth Table Verification**
- Verify that input-output mapping is preserved across all inputs
- Compare baseline vs recovered circuit outputs for all possible inputs
- Measure functional equivalence despite config changes

**Metrics:**
- `functional_equivalence_score`: Fraction of inputs producing identical outputs
- `output_difference_distribution`: Histogram of output bit differences
- `truth_table_distance`: Hamming distance of complete truth tables (not LUT tables)
- `functional_recovery_rate`: % of circuits with perfect functional recovery

#### B. **Task-Specific Functional Analysis**
- For binary multiply task: Verify product computation correctness
- For bit reversal: Verify reversal correctness
- Task-specific correctness metrics

**Metrics:**
```python
{
    "functional_equivalence": {
        "output_bitwise_accuracy": float,      # Fraction of output bits matching
        "complete_truth_table_match": bool,     # All inputs → same outputs?
        "task_specific_correctness": float,     # Task-specific metric (e.g., multiply correctness)
        "output_hamming_distance": int,        # Hamming distance of output patterns
    },
    "functional_stability": {
        "pre_damage_outputs": np.ndarray,      # Outputs before damage
        "post_damage_outputs": np.ndarray,      # Outputs after recovery
        "output_change_rate": float,           # Fraction of outputs that changed
        "critical_output_preservation": float,  # Critical outputs preserved?
    }
}
```

**Visualization:**
- Truth table comparison heatmaps (baseline vs recovered)
- Input-output mapping diagrams
- Functional correctness by input pattern (which inputs maintain correctness)
- Task-specific visualization (e.g., binary multiply results grid)

**Implementation:**
```python
def compute_functional_equivalence(
    base_logits, recovered_logits, wires, x_data_all, y_data_all
):
    """
    Compute functional equivalence between baseline and recovered circuits.
    
    Args:
        base_logits: Baseline circuit logits
        recovered_logits: Recovered circuit logits  
        wires: Circuit wiring (same for both)
        x_data_all: All possible input patterns (or test set)
        y_data_all: Expected outputs
        
    Returns:
        Dict with functional equivalence metrics
    """
    # Run both circuits on all inputs
    base_outputs = run_circuit(base_logits, wires, x_data_all, hard=True)
    recovered_outputs = run_circuit(recovered_logits, wires, x_data_all, hard=True)
    
    # Compute equivalence metrics
    output_match = (base_outputs == recovered_outputs).all(axis=-1)  # Per-sample
    functional_score = output_match.mean()
    
    # Output bitwise differences
    output_diff = (base_outputs != recovered_outputs).sum(axis=-1)
    
    # Task-specific correctness (e.g., for binary multiply)
    # task_correct = verify_task_specific(base_outputs, recovered_outputs, y_data_all)
    
    return {
        "functional_equivalence_score": functional_score,
        "output_bitwise_accuracy": (base_outputs == recovered_outputs).mean(),
        "output_hamming_distances": output_diff,
        "perfect_functional_match": functional_score == 1.0,
    }
```

**Visualization:**
- **Figure: Functional Invariance Visualization**
  - Subplot 1: Input-output mapping diagram (showing preserved mappings)
  - Subplot 2: Truth table comparison (baseline vs recovered)
  - Subplot 3: Functional correctness heatmap (by input pattern)
  - Subplot 4: Task-specific verification (e.g., binary multiply results grid)

### 7. **SA vs BP: Response Nature Comparison**

**Deep Dive into How Methods Differ:**

#### A. **Configuration Change Strategies**
- **SA**: Which gates change? More local or global?
- **BP**: Which gates change? Gradient-driven vs attention-driven?

**Metrics:**
- `change_localization`: Mean distance between changed gates
- `change_magnitude_distribution`: Histogram of per-gate hamming distances
- `layer_preference`: Which layers see most changes (SA vs BP)
- `critical_gate_preservation`: Do critical gates change more in one method?

#### B. **Recovery Trajectory Characteristics**
- **SA**: Smooth recovery? Oscillatory? Rapid then plateau?
- **BP**: Different trajectory shape? Faster initial recovery?

**Metrics:**
- `trajectory_smoothness`: Variance of first/second derivatives
- `convergence_type`: Exponential, linear, or other?
- `recovery_phase_duration`: Time in each recovery phase

#### C. **Sample Efficiency & Computational Cost**
- Steps to recovery
- Total computation (message passes vs gradient steps)
- Memory footprint comparison

**Visualization:**
- Side-by-side gate change heatmaps (SA vs BP)
- Recovery trajectory overlays with derivatives
- Computational cost comparison (steps × operations)

---

## Visualization Wishlist

### Essential Figures:

1. **Figure 1: Training Progress** ✅ (Already implemented)
   - Epoch vs accuracy for seen patterns
   - Growth vs Repair mode comparison

2. **Figure 2: Damage vs Exposure** (Could enhance)
   - Damage magnitude vs recovery success
   - Multi-damage trajectories

3. **Figure 3: Recovery Trajectories** ✅ (Already implemented)
   - SA vs BP stepwise comparison
   - Seen vs Unseen trajectories

4. **Figure 4: Functional Invariance** ⭐ **NEW - CRITICAL**
   - Truth table preservation visualization
   - Input-output mapping preservation
   - Task-specific correctness (binary multiply grid)

5. **Figure 5: Configuration Adaptation Analysis** ⭐ **NEW**
   - Hamming distance: SA vs BP (side-by-side scatter)
   - Gate-level change heatmaps
   - Layer-wise change magnitude

6. **Figure 6: Perturbation Response Metrics** ⭐ **NEW**
   - Recovery dynamics (initial drop, recovery rate)
   - Damage magnitude sensitivity
   - Pattern type sensitivity (greedy vs shotgun)

---

## Implementation Priorities

### Phase 1: Essential Extensions (Immediate)
1. **Recovery Speed Calculation**
   - Add function to compute steps to recovery threshold
   - Track in evaluation pipeline

2. **Damage Magnitude Tracking**
   - Add damage count per pattern to metrics
   - Include in hamming analysis output

3. **Generalization Gap Tracking**
   - Compute IN - OUT accuracy difference
   - Plot over training epochs

4. **Functional Equivalence Function**
   - Implement `compute_functional_equivalence` function
   - Integrate into evaluation pipeline

### Phase 2: Advanced Analysis (Next)
5. **Recovery Dynamics Extraction**
   - Compute derivatives from stepwise trajectories
   - Track recovery phases

6. **Gate-Level Change Analysis**
   - Extend hamming analysis for gate-level detail
   - Create change heatmap visualizations

7. **SA vs BP Deep Comparison**
   - Side-by-side hamming distance scatter
   - Trajectory characteristic analysis

### Phase 3: Polish (Later)
8. **Circuit Visualization Utilities**
   - Gate-level change overlays
   - Critical path highlighting
   - Damage pattern visualization

9. **Comprehensive Analysis Scripts**
   - Post-training analysis pipeline
   - Automated figure generation

---

## Key Insights to Highlight

1. **Functional Invariance Despite Structural Change**
   - Circuits maintain function even with different LUT configurations
   - Multiple valid solutions for same function (degeneracy)

2. **Adaptive Configuration Recovery**
   - SA finds alternative circuit configurations
   - Not just restoring original config, but adapting

3. **Efficient Recovery via Attention**
   - SA achieves recovery faster than BP in many cases
   - Attention mechanism enables targeted adaptation

4. **Robustness to Multiple Damage Types**
   - Works for permanent and reversible damage
   - Handles multiple sequential damages

5. **Generalization Beyond Training Patterns**
   - Good performance on unseen damage patterns
   - Vocabulary-based training enables generalization

---

## Metrics Summary Table

| Metric Category | Metric Name | Current Status | Priority |
|----------------|-------------|----------------|----------|
| Recovery | Recovery speed (steps) | ❌ Not computed | **HIGH** |
| Recovery | Final recovery rate | ✅ Partial (final accuracy) | Medium |
| Generalization | IN-OUT accuracy gap | ❌ Not computed | **HIGH** |
| Comparison | SA vs BP accuracy diff | ✅ Partial (separate metrics) | Medium |
| Functional | Functional equivalence | ❌ Not implemented | **CRITICAL** |
| Configuration | Hamming distance | ✅ Implemented | High |
| Configuration | Gate-level changes | ❌ Not detailed | Medium |
| Dynamics | Recovery rate | ❌ Not computed | Medium |
| Dynamics | Initial drop magnitude | ❌ Not computed | Medium |

---

## Next Steps

1. **Implement functional equivalence analysis** - This is critical for the paper's core argument
2. **Add recovery speed computation** - Essential for quantitative comparison
3. **Create side-by-side SA vs BP visualizations** - Key differentiator
4. **Enhance hamming distance analysis** - Already have foundation, extend for gate-level detail
5. **Implement damage magnitude sensitivity** - Show robustness scaling

Would you like me to start implementing any of these analyses?

