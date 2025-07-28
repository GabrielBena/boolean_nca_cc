2. Pattern Solvability Validation
   What: Theoretical analysis of which patterns are actually solvable
   Tests M: Validate if ~20% of patterns are genuinely unsolvable
   Implementation: Circuit reachability analysis - can any LUT configuration produce target function? TO do this, we need to keep track of lowest performing circuits. We have a backprop functionality in another codebase -> if even backprop cannot optimise, then chances are the circuit is unsolvable.

# Circuit-Level Analysis

3. Circuit Configuration Landscape Mapping
    Status: MERGED (analysis.md #3 + your Options A,B)
    What: Multi-approach analysis of LUT configuration space:
    3a. LUT Distribution Comparison: Compare LUT value distributions between knockout patterns, test reordering equivalence
    3b. PCA/UMAP Visualization: Dimensionality reduction on final logit configurations, with performance coloring to show "red/blue regions". Clustering analysis?
    3c. Graph Embedding: Embed circuits as graphs for similarity analysis (NOte: graphs and circuits are interchangeable, we already have a graph for each circuit. HOw to embed these graphs themselves though for measuring their distance, ie cosine similarity in embedding space)
    Tests: H5 (convergence patterns), H1 (specialization vs generalization)
    Implementation: Multiple visualization approaches for cross-validation

4. Sub-Circuit Functional Analysis
    Status: NEW (your key insight)
    What: Identify and measure functional sub-circuit overlap between knockout patterns
    Tests: H5 (pattern similarity through sub-circuit sharing), H1 (specialization on sub-circuits)
    Implementation:
    Graph motif detection for functional blocks
    Measure knockout impact on shared vs unique sub-circuits
    Quantify "close enough" pattern overlap
    NOTE: This metric depends on pt. 1, individual patternperformance. These 2 metrics combined: If we track for a given high-overlap pair, is performance also correlated? This would bridge ID/OOD perforamnce, via their overlapping sub-circuits. Other tests?

5. Perturbation Response Locality Analysis
    Status: NEW (your idea)
    What: Measure how local vs global the circuit changes are in response to knockouts
    Tests: H1 (specialized vs general responses), H4 (meta-learning locality)
    Implementation:
    Distance metrics between pre/post knockout configurations
    Propagation analysis of changes through circuit layers

6. Critical Path Analysis
   What: Identify which gate combinations break essential circuit paths
   Tests M: Map knockout patterns to circuit functionality loss
   Implementation: Synergy with 5. shared sub-circtuis and critical sub-circtuis amenable to joint analysis


# Phase 2

 Elias suggesting letting it configure the circuit no KO -> knocking out and letting it re-train. IN other words, instead of NOPs initialisation, have functioning circuit initialisation and make the inner loop 'reconfigure' instead of grow. 
-> perhaps then differences are more quantifiable 

Another point: OOD should have a different damage_prob, actually changing the number of knockouts not just the patterns

Once Elias' KO retraining is set-up, we could compare: optimised circuit -> KO -> num samples needed to reconfigure. For this, we can compare sample number required between backprop and GNN MP steps.

**more visual stuff**
- Coloured gate perturbations
- Track perturbation as Number of gates vs gate change magnitude (2D?) + spread of magnitude (1 gate huge mag change or more uniform)
