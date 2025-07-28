3. Pattern Similarity Clustering
   What: Measure similarity between knockout patterns (Hamming distance, overlap in knocked-out gates)
   Tests 5: Should show decreasing inter-pattern distances as diversity increases
   Tests M: Should correlate pattern similarity with performance similarity

6. Optimization Trajectory Analysis
   What: Track how circuits evolve during optimization steps for different patterns
   Tests 4: Different meta-strategies should show different optimization paths
   Tests 1: Specialized patterns should show direct paths, general patterns more exploratory

7. Learning Curve Decomposition by Pattern Type
   What: Separate training curves for "easy", "hard", and "broken" patterns
   Tests M: Broken patterns should plateau at low performance
   Tests 1: Easy patterns should show rapid learning with low diversity
   Implementation: Classify patterns by final achievable accuracy, track separately

8. Gradient Flow Analysis
   What: Analyze gradient magnitudes and directions for different knockout patterns
   Tests 4: Should show different meta-learning signals
   Tests 1: Should show pattern-specific vs general gradient patterns

# Training Dynamics Analysis

9. Pool Evolution Tracking
   What: Monitor how circuit solutions in the pool evolve with different knockout exposures
   Tests 5: Should show convergence to similar solutions over time
   Implementation: Track LUT entropy, solution diversity metrics across training

# Meta-Learning Analysis

12. Attention Pattern Visualization
    What: Visualize attention maps for different knockout patterns and diversity levels
    Tests 1: Should show specialized vs general attention strategies
    Tests 4: Should reveal meta-learning attention patterns
    Experimental Design Approaches
13. Controlled Knockout Experiments
    What: Systematically vary knockout locations to isolate critical vs non-critical gates
    Tests M: Should clearly identify which patterns are fundamentally unsolvable
    Implementation: Grid search over knockout combinations, measure theoretical solvability
14. Damage Probability Scaling Study
    What: Vary damage_prob while keeping diversity constant
    Tests M: Convergence ceiling should correlate with damage probability
    Tests 5: Solution space should contract with higher damage
15. Synthetic Pattern Design
    What: Create hand-designed patterns with known solvability properties
    Tests M: Validate that unsolvable patterns drag down averages
    Tests 1: Test memorization on perfectly learnable vs challenging patterns
    Implementation Priority Ranking
    High Priority (Easy to implement, high insight potential):
    Individual Pattern Performance Distribution (#1)
    Solution Space Topology Mapping (#4)
    Learning Curve Decomposition (#7)
    Medium Priority (Moderate effort, good insights):
    Pattern Solvability Validation (#2)
    Critical Path Analysis (#5)
    Attention Pattern Visualization (#12)
    Research Extensions (Higher effort, deep insights):
    Transfer Learning Experiments (#10)
    Controlled Knockout Experiments (#13)
    Gradient Flow Analysis (#8)
    The Individual Pattern Performance Distribution analysis would be the most immediately revealing - if we see a bimodal distribution, it strongly supports hypothesis M. If we see smooth degradation, it supports hypotheses 1/4/5.