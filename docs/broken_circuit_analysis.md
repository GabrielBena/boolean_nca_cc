As we train on different knockout_diversity settings, we see the following:
1 pattern: ID = 100% accuract, OOD = 30%
16 patterns: ID = 90%, OOD = 60%
512 patterns: ID = 86%, OOD = 86%

We have 4 hypotheses explaining this meeting in the middle of ID and OOD:

**H1: Memorization vs Generalization Trade-off**
With low knockout diversity, the model overfits to specific knockout patterns, achieving near-perfect ID performance but failing to generalize to new patterns. As diversity increases, the model learns more robust, generalizable strategies that work across patterns, trading off pattern-specific optimization for broader capability.

**H4: Meta-Learning Dynamics** 
The training process is essentially meta-learning across knockout patterns. With single patterns, the model learns a very specific meta-strategy. With diverse patterns, it learns a more general meta-strategy that transfers better to unseen patterns but performs worse on any individual known pattern.

**H5: Solution Space Convergence**
Different knockout patterns may require fundamentally different circuit configurations. With high diversity, the model converges toward a "consensus" solution space that works adequately for all patterns but optimally for none. This explains why ID and OOD performance converge - the model isn't really pattern-specific anymore.

**HM: Circuit Breaking (Pattern Solvability)**
Some knockout patterns are fundamentally unsolvable due to critical path destruction. With higher diversity, there's a higher chance of sampling these "breaking" patterns. The ~0.8 convergence value represents the solvability fraction - approximately 20% of randomly generated patterns are unsolvable, creating a performance ceiling regardless of model capability.


# Pattern-Level Analysis

1. Individual Pattern Performance Distribution
   What: Log accuracy for each individual pattern during evaluation (not just the mean)
   Tests 1: Should show increasing variance with low diversity (specialist vs poor generalization)
   Implementation: Modify evaluation to store per-pattern metrics, not just averages


