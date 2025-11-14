# Exploring Degenerate Circuit Solution Spaces

## Overview

This document outlines strategies for systematically exploring the space of degenerate circuit configurations—circuits that implement the same boolean function (e.g., binary multiply) but differ in their internal LUT configurations. Through reversible damage perturbations and recovery, we can discover alternative circuit implementations and map the solution space topology.

## Core Concept: Degenerate Solutions

### The Problem

Given a boolean function (e.g., 8-bit binary multiply), there exist many different circuit configurations that produce identical outputs. These **degenerate solutions** arise because:

1. **Multiple LUT configurations** can implement equivalent logic functions
2. **Reversible damage** can push circuits into different basins of attraction during recovery
3. **Optimization paths** may converge to different local minima with equivalent performance

### The Exploration Goal

Starting from a single preconfigured circuit, we want to:
- **Discover** how many unique circuit configurations exist for the same function
- **Map** the connectivity between configurations (which perturbations lead to which solutions)
- **Characterize** the solution space topology (clusters, paths, distances)
- **Understand** the diversity and robustness of alternative implementations

## Architecture Overview

### Circuit Representation

Circuits are represented as differentiable boolean networks with:
- **LUT logits**: `logits ∈ R^(n_gates × 2^arity)` defining lookup table configurations (`boolean_nca_cc/circuits/model.py`)
- **Wiring**: Fixed connection patterns between layers (`gen_wires`, `gen_circuit`)
- **Layer structure**: Defined by `layer_sizes` tuples `(total_gates, group_size)` per layer

### Damage Mechanisms

The codebase provides multiple perturbation strategies via `boolean_nca_cc/training/pool/structural_perturbation.py`:

1. **`shotgun`**: Random sampling from all eligible gates (`create_reproducible_knockout_pattern`)
2. **`strip`**: Localized damage within radius (`create_strip_knockout_pattern`)
3. **`greedy`**: Deterministic pattern from ordered importance indices (`create_greedy_knockout_pattern`)
4. **`greedy_vocabulary`**: Diverse patterns sampled from greedy indices (`create_greedy_subset_random_pattern`)

All patterns produce boolean arrays where `True` indicates knocked-out gates.

### Recovery Mechanisms

Two optimization approaches are available:

#### 1. Self-Attention Recovery (`boolean_nca_cc/models/self_attention.py`)

**Reversible mode** (`damage_behavior="reversible"`):
- Applies one-shot logit bias (`reversible_bias`, default `-4.0`) at first model step
- Keeps damaged gates in attention graph (connectivity preserved)
- Allows recovery through residual updates in subsequent steps
- Uses step counter (`globals[..., 1]`) to gate bias application

**Permanent mode** (`damage_behavior="permanent"`):
- Removes attention edges for damaged nodes
- Clamps logits to large negative value (`-10.0`)
- Zeros residual updates for damaged gates
- Used for ablation studies

**Recovery process**: Iterative message passing via `run_self_attention_scan` or `evaluate_model_stepwise_generator`, with loss computation at each step (`boolean_nca_cc/training/evaluation.py`).

#### 2. Backpropagation Recovery (`boolean_nca_cc/circuits/train.py`, `boolean_nca_cc/training/backprop.py`)

**Reversible mode** (`damage_behavior="reversible"`):
- Applies one-shot logit bias (`reversible_bias`, default `-10.0`) at first training step (`step_count=0`)
- Zeros LUT outputs via sigmoid of biased logits (equivalent to masking effect)
- Allows recovery through normal gradient updates in subsequent steps
- No `gate_mask` used—gates remain in computational graph

**Permanent mode** (`damage_behavior="permanent"`):
- Uses permanent masking via `gate_mask` (`create_gate_mask_from_knockout_pattern`)
- Damaged gates are zeroed during circuit execution (`run_circuit` with `gate_mask`)
- Gradients naturally become zero for damaged gates (via chain rule)
- Damaged gates remain disabled throughout training

**Recovery process**: Standard gradient descent (`train_step`) with optax optimizers (Adam/AdamW). Can train on vocabulary of patterns in parallel via `jax.vmap`. Both modes support the same API with `damage_behavior` parameter.

## Graph Representation

### Two Complementary Data Structures

Exploration generates two complementary structures:

1. **Exploration Graph** (with cycles): Full trajectory of perturbation-recovery sequences
   - Tracks every edge traversed during exploration
   - **Allows cycles**: Solution A → Solution B → Solution A (same solution revisited)
   - Captures exploration dynamics and convergence behavior
   - Used for trajectory analysis, cycle detection, robustness metrics

2. **Unique Solutions Set**: Distinct circuit configurations discovered
   - One entry per unique circuit (by hash/fingerprint)
   - No duplicates, regardless of how many times visited
   - Used for diversity metrics, UMAP/t-SNE embeddings, solution space size estimation

**Key Insight**: The same solution can be reached via multiple paths, creating cycles in the exploration graph. This is valuable information—cycles indicate:
- **Robustness**: Solutions that can recover to themselves after perturbation
- **Convergence**: Multiple perturbation paths leading to the same solution
- **Stability**: Consistent recovery behavior

### Nodes: Circuit Configurations

Each **node** in the exploration graph represents a circuit configuration:
- Defined by its LUT logits: `logits ∈ R^(n_gates × 2^arity)`
- Characterized by its functional equivalence (same input/output mapping)
- Identified by a hash or fingerprint of its logits
- **Can appear multiple times** in the exploration graph (when revisited via different paths)

### Edges: Perturbation → Recovery Trajectories

Each **edge** represents a perturbation-recovery sequence:
- **Source**: Parent circuit configuration
- **Perturbation**: Damage pattern applied (via `create_knockout_vocabulary` or individual pattern creators)
- **Recovery**: Optimization steps that recover functionality
  - **Self-attention**: Iterative message passing (`CircuitSelfAttention` with `knockout_pattern`, `damage_behavior="reversible"` or `"permanent"`)
  - **Backprop**: Gradient descent (`train_step` with `knockout_pattern`, `damage_behavior="reversible"` or `"permanent"`)
    - **Reversible**: One-shot logit bias at step 0, then normal gradient updates
    - **Permanent**: `gate_mask` applied throughout training
- **Target**: Resulting circuit configuration after recovery
- **Metadata**: Recovery time, perturbation pattern ID, success/failure

**Key Properties**: 
- Multiple perturbations from the same parent may lead to:
  - The same child (convergence to known solution)
  - Different children (discovery of new solutions)
  - Failed recovery (circuit that doesn't implement the function)
- **Cycles are allowed**: Solution A → Solution B → Solution A creates a cycle
- **Multiple paths to same solution**: Different perturbation sequences can converge to the same circuit

### Graph Structure

**Example exploration graph (with cycles)**:
```
Root Circuit (preconfigured)
├── Pattern 1 → Recovery → Circuit A
├── Pattern 2 → Recovery → Circuit B
├── Pattern 3 → Recovery → Circuit A (revisit via different path)
└── Pattern 4 → Recovery → Circuit C

Circuit A
├── Pattern 5 → Recovery → Circuit D
├── Pattern 6 → Recovery → Circuit B
└── Pattern 7 → Recovery → Circuit A (CYCLE: self-recovery)

Circuit B
├── Pattern 8 → Recovery → Circuit E
└── Pattern 9 → Recovery → Circuit A (CYCLE: A↔B bidirectional)

Circuit C
└── Pattern 10 → Recovery → Circuit A (convergence: multiple paths to A)
```

**Unique solutions set** (for diversity analysis):
```
{
  Root Circuit,
  Circuit A,  # Appears once despite multiple visits
  Circuit B,
  Circuit C,
  Circuit D,
  Circuit E
}
```

**Cycle examples**:
- **Self-recovery cycle**: A → Pattern 7 → A (robustness indicator)
- **Bidirectional cycle**: A → Pattern 6 → B → Pattern 9 → A (mutual recovery)
- **Convergence**: Multiple paths leading to A (stability indicator)

## Exploration Strategies

### 1. Breadth-First Search (BFS)

#### Strategy

**Level-by-level exploration**: Explore all circuits at depth `d` before moving to depth `d+1`.

**Algorithm**:
1. Initialize queue with root circuit (depth 0)
2. While queue not empty:
   - Dequeue circuit `C` at depth `d`
   - Apply all perturbation patterns to `C` (via `create_knockout_vocabulary`)
   - For each pattern:
     - Apply damage → recovery → check if unique
     - If unique, add to queue at depth `d+1`
   - Mark `C` as explored

**Implementation notes**:
- Use `evaluate_model_stepwise_generator` for self-attention recovery (with `damage_behavior="reversible"`)
- Use `_train_single_knockout_pattern` for backprop recovery (with `damage_behavior="reversible"` or `"permanent"`)
- Track discovered circuits via logit hashes

#### Advantages

- **Comprehensive coverage**: All circuits within `k` perturbations are discovered
- **Uniform exploration**: No bias toward specific branches
- **Natural clustering**: Circuits at same depth share similar perturbation distance
- **Complete mapping**: Full graph structure up to depth limit

#### Disadvantages

- **Memory intensive**: Must store all circuits at current depth
- **Computational cost**: Explores all neighbors before going deeper
- **May miss deep solutions**: If solution space is deep, BFS is expensive

#### Use Case

**Best for**: Mapping the immediate neighborhood of the root circuit, discovering all solutions within 1-3 perturbation steps.

**Example**:
```
Depth 0: [Root Circuit]
Depth 1: [Circuit A, Circuit B, Circuit C]  (600 unique from 1000 perturbations)
Depth 2: [Circuit D, Circuit E, ..., Circuit Z]  (explore all 600 × 1000 perturbations)
```

### 2. Depth-First Search (DFS)

#### Strategy

**Deep exploration**: Follow one perturbation chain as far as possible before backtracking.

**Algorithm**:
1. Initialize stack with root circuit
2. While stack not empty:
   - Pop circuit `C`
   - Apply perturbation pattern → recovery → check if unique
   - If unique and not explored:
     - Push to stack
     - Continue exploring this branch
   - If duplicate or max depth reached:
     - Backtrack (pop from stack)

**Implementation notes**:
- Sample patterns from vocabulary using `jax.random.choice`
- Use same recovery mechanisms as BFS
- Maintain path history for trajectory analysis

#### Advantages

- **Memory efficient**: Only stores current path (O(depth) vs O(breadth^depth))
- **Deep exploration**: Can discover solutions far from root
- **Fast discovery**: Quickly finds distant solutions
- **Natural for chains**: Good for exploring long perturbation sequences

#### Disadvantages

- **Incomplete coverage**: May miss solutions in unexplored branches
- **Biased exploration**: Depends on perturbation pattern ordering
- **No depth guarantees**: Doesn't ensure all solutions at depth `k` are found

#### Use Case

**Best for**: Exploring deep perturbation chains, discovering distant solutions, memory-constrained scenarios.

**Example**:
```
Root → Pattern 1 → Circuit A → Pattern 2 → Circuit B → Pattern 3 → Circuit C → ...
  (backtrack)
Root → Pattern 4 → Circuit D → Pattern 5 → Circuit E → ...
```

### 3. Iterative Deepening Depth-First Search (IDDFS)

#### Strategy

**Hybrid approach**: Combine BFS completeness with DFS memory efficiency.

**Algorithm**:
1. For depth limit `d = 1, 2, 3, ...`:
   - Perform DFS with depth limit `d`
   - Discover all circuits within `d` perturbations
   - Increment depth limit and repeat

#### Advantages

- **Complete coverage**: Like BFS, finds all solutions up to depth `d`
- **Memory efficient**: Like DFS, only stores current path
- **Balanced**: Good trade-off between completeness and efficiency

#### Disadvantages

- **Redundant work**: Re-explores shallow nodes at each iteration
- **Slower than pure BFS**: For shallow graphs, BFS is faster

#### Use Case

**Best for**: When you want complete coverage but have memory constraints, or when solution depth is unknown.

### 4. Random Walk / Stochastic Exploration

#### Strategy

**Probabilistic exploration**: Randomly select perturbation patterns and circuits to explore.

**Algorithm**:
1. Maintain set of discovered circuits
2. Randomly select circuit `C` from discovered set
3. Randomly select perturbation pattern `P` (from vocabulary or generate fresh)
4. Apply `P` → recovery → add if unique
5. Repeat for `N` iterations

**Implementation notes**:
- Use `create_knockout_vocabulary` to generate pattern sets
- Sample circuits uniformly or by diversity metrics
- Parallelize pattern application via `jax.vmap`

#### Advantages

- **Simple implementation**: No complex data structures
- **Unbiased sampling**: Can discover unexpected solutions
- **Scalable**: Easy to parallelize
- **Good for large spaces**: Doesn't require full graph structure

#### Disadvantages

- **No guarantees**: May miss important regions
- **Redundant exploration**: May revisit same circuits
- **No structure discovery**: Doesn't reveal graph topology

#### Use Case

**Best for**: Initial exploration, discovering solution diversity, when full mapping is too expensive.

### 5. Beam Search

#### Strategy

**Focused exploration**: Maintain `k` best circuits at each depth, explore only from those.

**Algorithm**:
1. Initialize beam with root circuit
2. For each depth:
   - For each circuit in beam:
     - Apply perturbation patterns → recovery
   - Select top `k` unique circuits (by some metric: diversity, performance, etc.)
   - Replace beam with top `k`
3. Continue until convergence or max depth

#### Advantages

- **Focused exploration**: Prioritizes promising regions
- **Memory efficient**: Only stores `k` circuits per depth
- **Quality over quantity**: Finds diverse, high-quality solutions

#### Disadvantages

- **May miss solutions**: Pruning can eliminate important branches
- **Requires scoring**: Need metric to rank circuits
- **Biased**: Depends on scoring function

#### Use Case

**Best for**: Finding diverse, high-quality solutions when full exploration is too expensive.

**Scoring Metrics**:
- **Hamming distance** from other circuits (diversity)
- **Recovery time** (robustness)
- **Perturbation sensitivity** (stability)

### 6. Adaptive Exploration Strategies

#### Hybrid BFS-DFS

**Strategy**: Use BFS for shallow exploration, DFS for deep chains.

**Algorithm**:
1. Perform BFS up to depth `d_bfs` (e.g., 2)
2. For each circuit at depth `d_bfs`:
   - Perform DFS with depth limit `d_dfs` (e.g., 5)
   - This explores chains starting from BFS frontier

#### Cluster-Based Exploration

**Strategy**: Identify clusters of similar circuits, explore from cluster representatives.

**Algorithm**:
1. Discover initial set of circuits (random walk or BFS)
2. Cluster circuits by similarity (e.g., LUT logit distance)
3. Select cluster representatives
4. Explore from representatives (BFS or DFS)

#### Importance Sampling

**Strategy**: Weight perturbation patterns by their likelihood of discovering new solutions.

**Algorithm**:
1. Track success rate of each perturbation pattern
2. Sample patterns with probability proportional to success rate
3. Update weights as exploration progresses

## Implementation Considerations

### Circuit Uniqueness Detection

**Challenge**: Determine if two circuits are functionally equivalent but have different LUT configurations.

**Approaches**:
1. **Functional equivalence**: Test on exhaustive input set (for small circuits)
2. **Logit distance**: L2/L1 distance between logit vectors (threshold-based)
3. **Hamming distance**: Binary distance after quantization
4. **Hash fingerprint**: Hash of logits (fast but may have collisions)

**Recommendation**: Use combination:
- Fast hash for initial filtering
- Functional test for verification
- Distance metrics for similarity analysis

**Implementation**: Extract logits via `extract_logits_from_graph` (`boolean_nca_cc/utils.py`), compute distances, hash for fast lookup.

**Dual tracking strategy**:
- **Unique solutions set**: Use hash-based deduplication (one entry per unique circuit)
- **Exploration graph**: Track all edges, including revisits (allows cycles)
- **Mapping**: Maintain `circuit_hash → [edge_list]` to track all paths to each solution

### Stopping Criteria

**When to stop exploration**:
1. **Depth limit**: Maximum perturbation distance from root
2. **Circuit limit**: Maximum number of unique circuits discovered
3. **Convergence**: No new circuits discovered in last `N` iterations
4. **Diversity threshold**: All discovered circuits are within similarity threshold
5. **Time/compute budget**: Exploration time limit

### Memory Management

**Challenges**:
- Storing full logit vectors for each circuit (large memory footprint)
- Maintaining graph structure (edges, connectivity)
- Tracking exploration state (visited nodes, queue/stack)

**Solutions**:
1. **Lazy evaluation**: Store circuit identifiers, compute logits on-demand
2. **Compression**: Store only unique circuits, use references for duplicates
3. **Checkpointing**: Save exploration state periodically
4. **Distributed storage**: Use database or file system for large sets

**Implementation**: Use JAX arrays for efficient storage, leverage `jax.tree_util` for PyTree structures.

### Parallelization

**Opportunities**:
- **Independent perturbations**: Apply different patterns to same circuit in parallel
- **Batch recovery**: Process multiple perturbation-recovery sequences simultaneously
- **Distributed exploration**: Multiple workers explore different branches

**Considerations**:
- Synchronization of discovered circuits (avoid duplicates)
- Load balancing across workers
- Communication overhead for sharing discoveries

**Implementation**: Use `jax.vmap` for batch processing, `jax.pmap` for multi-device parallelism.

## Metrics and Analysis

### Solution Space Characterization

#### 1. **Diversity Metrics**

- **Unique circuits discovered**: Total count of functionally equivalent but distinct circuits
- **Perturbation efficiency**: Unique circuits / total perturbations (e.g., 600/1000 = 0.6)
- **Solution density**: Unique circuits per perturbation depth

#### 2. **Graph Topology Metrics**

- **Connectivity**: Average degree (how many neighbors per circuit)
- **Clustering coefficient**: How tightly circuits cluster together
- **Path lengths**: Average shortest path between circuits
- **Centrality**: Which circuits are most "central" (many paths pass through)
- **Cycle detection**: 
  - Self-recovery cycles (A → A): Robustness metric
  - Bidirectional cycles (A ↔ B): Mutual recovery pairs
  - Cycle length distribution: How long are recovery cycles?
- **Convergence patterns**: How many distinct paths lead to each solution?
- **Revisit frequency**: How often is each solution revisited?

#### 3. **Distance Metrics**

- **LUT logit distance**: L2/L1 distance between circuit configurations
- **Functional distance**: Hamming distance on output patterns
- **Perturbation distance**: Minimum number of perturbations to transform one circuit to another

#### 4. **Robustness Metrics**

- **Recovery success rate**: Fraction of perturbations that lead to functional recovery
- **Recovery time**: Number of optimization steps to recover
- **Stability**: How often same perturbation leads to same recovered circuit
- **Self-recovery rate**: Fraction of solutions that can recover to themselves (cycle A → A)
- **Cycle robustness**: Average number of distinct perturbation paths that lead back to a solution
- **Convergence diversity**: Number of unique paths leading to each solution

### Visualization Strategies

#### 1. **Graph Visualization**

- **Node**: Circuit configuration (colored by cluster, depth, or revisit frequency)
- **Edge**: Perturbation-recovery trajectory (weighted by recovery time, colored by cycle type)
- **Layout**: Force-directed or hierarchical (by depth)
- **Cycle highlighting**: 
  - Self-recovery cycles (A → A): Highlight with bold edges
  - Bidirectional cycles (A ↔ B): Use bidirectional arrows
  - Convergence paths: Show multiple edges converging to same node

#### 2. **Embedding Visualization**

- **t-SNE/UMAP**: Embed circuits in 2D by logit similarity
  - **Input**: Unique solutions set only (one point per unique circuit)
  - **Color by**: Depth, cluster, recovery time, revisit frequency, cycle participation
  - **Animate**: Show exploration progression over time
  - **Separate views**:
    - **Diversity view**: UMAP of unique solutions (no duplicates)
    - **Trajectory view**: Graph visualization showing full exploration paths (with cycles)
  - **Future enhancement**: Trajectory-overlaid UMAP (see Future Directions) - overlay exploration graph edges and cycles directly onto UMAP embeddings to combine diversity and dynamics in a single visualization

**Implementation Plan** (Next Script):
- Load exploration results using `load_exploration_results()`
- Extract logits from `unique_solutions` dictionary
- Flatten logits into feature vectors for UMAP
- **Depth Coloring**: Color points by minimum perturbation distance from root
  - **Color gradient**: Continuous gradient from root (e.g., blue) to distant solutions (e.g., red)
  - **Discretized gradient**: Discrete color bands for each depth level (depth 0 = root, depth 1, depth 2, etc.)
  - **Implementation**: Compute shortest path from root to each solution using exploration graph, or use BFS depth metadata
  - **Visual benefit**: Reveals solution space structure - shows how solutions cluster by perturbation distance from root

#### 2.1. **Dual Connectivity Visualization: Proximity vs. Trajectory**

**Key Insight**: UMAP's connectivity plotting and our exploration graph represent fundamentally different types of connectivity, and comparing them reveals important insights about the solution space structure.

**Two Types of Connectivity**:

1. **UMAP Connectivity (Proximity-Based)**:
   - Based on **k-nearest neighbors** in logit feature space
   - Reflects **similarity** in circuit configurations (LUT logit values)
   - Computed by UMAP during embedding: `umap.plot.connectivity()` shows UMAP's internal k-NN graph
   - **What it captures**: Circuits that are similar in their internal representation
   - **Edge meaning**: "These circuits have similar logit configurations"

2. **Exploration Graph Connectivity (Trajectory-Based)**:
   - Based on **actual perturbation-recovery sequences** during exploration
   - Reflects **reachability** via the exploration process
   - Captured in `results["exploration_graph"]` and `results["edges"]`
   - **What it captures**: Circuits that were actually reached via perturbation-recovery paths
   - **Edge meaning**: "This circuit was reached by perturbing that circuit and recovering"

**Why This Comparison Matters**:

The correspondence (or lack thereof) between these two connectivity types reveals important properties:

- **Proximity ≠ Reachability**: Circuits that are similar in logit space may not be reachable via perturbations (e.g., separated by high-energy barriers in the optimization landscape)
- **Reachability ≠ Proximity**: Circuits reachable via perturbations may be quite different in logit space (e.g., different basins of attraction that are connected via exploration paths)
- **Convergence Patterns**: Multiple perturbation trajectories converging to similar solutions (high trajectory connectivity) vs. solutions that are naturally similar (high proximity connectivity)
- **Exploration Efficiency**: Whether exploration finds diverse solutions (low trajectory-proximity correlation) or clusters around similar configurations (high correlation)
- **Solution Space Topology**: Reveals whether the solution space is "smooth" (proximity ≈ reachability) or "rugged" (proximity ≠ reachability)

**Visualization Strategy**:

1. **Side-by-side comparison**:
   - Left: UMAP embedding with UMAP's k-NN connectivity (`umap.plot.connectivity()`)
   - Right: Same UMAP embedding with exploration graph edges overlaid
   - Compare edge density, clustering patterns, and structural differences

2. **Overlaid visualization**:
   - UMAP embedding as base
   - UMAP connectivity edges: Light gray, thin lines
   - Exploration graph edges: Colored by phase (BFS=blue, Random Walk=green), thicker lines
   - Highlight edges that exist in both graphs (strong correspondence)
   - Highlight edges that exist in only one graph (divergence)

3. **Edge overlap analysis**:
   - Compute intersection: edges that exist in both UMAP k-NN graph and exploration graph
   - Compute divergence: edges unique to each graph type
   - Visualize overlap ratio per node (how many neighbors are shared vs. unique)
   - Identify "bridge" circuits: high trajectory connectivity but low proximity connectivity (or vice versa)

4. **Path length comparison**:
   - Graph distance in exploration graph (trajectory distance)
   - Euclidean distance in UMAP embedding (proximity distance)
   - Compare: Do short trajectory paths correspond to short embedding distances?
   - Identify cases where trajectory paths are long but embedding distance is short (or vice versa)

**Implementation Notes**:

- UMAP's `umap.plot.connectivity()` uses UMAP's internal graph (not customizable)
- To overlay exploration graph edges, manually draw edges between UMAP embedding coordinates
- Map circuit hashes to embedding indices to connect exploration graph edges to UMAP points
- Use different visual styles (color, thickness, opacity) to distinguish connectivity types
- Consider edge bundling for exploration graph edges to reduce visual clutter (similar to `umap.plot.connectivity(edge_bundling='hammer')`)

**Research Questions**:

- How correlated are proximity and trajectory connectivity?
- Do exploration paths follow "natural" similarity gradients, or do they traverse dissimilar regions?
- Are there "shortcuts" in trajectory space that don't exist in proximity space (or vice versa)?
- How do cycles in the exploration graph relate to proximity clusters in UMAP?

#### 3. **Trajectory Plots**

- **Perturbation chains**: Show sequences of circuits discovered
- **Recovery curves**: Loss/accuracy over recovery steps (from `evaluate_model_stepwise`)
- **Diversity over depth**: Number of unique circuits vs perturbation depth
- **Cycle visualization**: 
  - Highlight cycles in trajectory plots
  - Show convergence paths (multiple trajectories leading to same solution)
  - Animate cycle discovery over exploration time

#### 4. **Cycle Analysis**

**Cycle detection algorithms**:
- **Self-recovery cycles**: Find edges where `source == target` (A → A)
- **Bidirectional cycles**: Find pairs (A → B) and (B → A) within exploration graph
- **Long cycles**: Use DFS to detect cycles of length > 2 (A → B → C → A)
- **Cycle statistics**: 
  - Cycle length distribution
  - Cycle frequency per solution
  - Cycle robustness (how many distinct perturbation patterns create cycles)

**Implementation**:
```python
def detect_cycles(exploration_graph):
    """Detect all cycles in exploration graph."""
    cycles = []
    visited = set()
    
    def dfs(node, path):
        if node in path:
            # Found cycle: path[path.index(node):] + [node]
            cycle = path[path.index(node):] + [node]
            cycles.append(cycle)
            return
        
        if node in visited:
            return
        
        visited.add(node)
        path.append(node)
        
        for neighbor, _, _ in exploration_graph.get(node, []):
            dfs(neighbor, path.copy())
        
        path.pop()
    
    for node in exploration_graph:
        if node not in visited:
            dfs(node, [])
    
    return cycles
```

## Implementation Status

### Current Implementation (`experiments/explore_degenerate_solutions.py`)

**Status**: ✅ **Implemented**

The exploration script implements a **hybrid BFS + Random Walk strategy** optimized for UMAP visualization:

#### Features

1. **Multiple Exploration Strategies**:
   - `hybrid` (default): BFS phase (depth 1-2) followed by Random Walk
   - `bfs`: Breadth-first search only
   - `random_walk`: Random walk only
   - `single_root`: Legacy mode (all perturbations from root)

2. **BFS Phase**:
   - Explores immediate neighborhood of root circuit
   - Configurable depth (default: 2)
   - Configurable perturbations per level (default: 100)
   - Tracks circuits by depth to avoid revisiting

3. **Random Walk Phase**:
   - Randomly samples from discovered circuits
   - Randomly selects perturbation patterns
   - Configurable iterations (default: 500)
   - Maximizes diversity for UMAP visualization

4. **Exploration Graph Tracking**:
   - `exploration_graph`: Adjacency structure (allows cycles)
   - `edges`: List of all edges with metadata
   - Tracks cycles, revisits, and convergence patterns
   - Supports future cycle analysis and visualization

5. **Save/Load Functionality**:
   - Saves results to disk automatically (default: `exploration_results/exploration_{timestamp}/`)
   - Files saved:
     - `solutions.npz`: All unique solution logits
     - `metadata.pkl`: Complete metadata (hashes, graph, results)
     - `wires.npz`: Circuit wiring (for functional testing)
     - `config.json`: Human-readable configuration
   - `load_exploration_results()` function for loading in separate visualization scripts

#### Usage

```bash
# Run hybrid exploration (recommended for UMAP)
python experiments/explore_degenerate_solutions.py \
    --exploration-strategy hybrid \
    --bfs-depth 2 \
    --bfs-perturbations-per-level 100 \
    --random-walk-iterations 500 \
    --output-dir ./my_exploration_results

# Load results in separate script
from experiments.explore_degenerate_solutions import load_exploration_results
results = load_exploration_results(Path("./my_exploration_results"))
```

## Practical Workflow

### Phase 1: Initial Exploration (Random Walk)

**Goal**: Discover initial set of solutions and estimate solution space size.

**Steps**:
1. Start with preconfigured root circuit
2. Generate vocabulary of 1000 random perturbations (`create_knockout_vocabulary` with `damage_mode="shotgun"` or `"greedy_vocabulary"`)
3. Apply each pattern → recovery (self-attention reversible mode or backprop)
4. Track unique circuits discovered
5. Estimate diversity: unique / total perturbations

**Expected Output**: ~600 unique circuits from 1000 perturbations (example from user)

**Implementation**: ✅ **Implemented** - Use `explore_degenerate_solutions()` with `exploration_strategy="random_walk"` or `"hybrid"`. Uses `_train_single_knockout_pattern` for backprop recovery (with `damage_behavior="reversible"`).

### Phase 2: Systematic Mapping (BFS)

**Goal**: Map immediate neighborhood of root circuit.

**Steps**:
1. Perform BFS up to depth 2-3
2. Apply all perturbation patterns to each discovered circuit
3. Build graph structure (nodes and edges)
4. Analyze connectivity and clustering

**Expected Output**: Complete graph of solutions within 2-3 perturbations

**Implementation**: ✅ **Implemented** - Use `explore_degenerate_solutions()` with `exploration_strategy="bfs"` or `"hybrid"`. BFS phase maintains discovered set, queues circuits by depth, tracks exploration graph structure.

### Phase 3: Deep Exploration (DFS)

**Goal**: Discover distant solutions and long perturbation chains.

**Steps**:
1. Select promising circuits from Phase 2
2. Perform DFS with depth limit 5-10
3. Track longest chains discovered
4. Identify "hub" circuits (many paths converge)

**Expected Output**: Deep chains and distant solution clusters

**Implementation**: Use stack-based DFS, track path history, sample patterns stochastically.

### Phase 4: Analysis and Characterization

**Goal**: Understand solution space structure and properties.

**Steps**:
1. Compute graph metrics (connectivity, clustering, paths)
2. Identify clusters and communities
3. Analyze distance distributions
4. Visualize solution space topology
5. Characterize robustness and diversity

## Example Usage

### Running Exploration

```python
from experiments.explore_degenerate_solutions import (
    explore_degenerate_solutions,
    load_exploration_results,
    load_preconfigured_circuit,
)
from boolean_nca_cc.circuits.tasks import get_task_data
from pathlib import Path
import jax

# Load preconfigured circuit
wires, logits, layer_sizes = load_preconfigured_circuit(
    logits_file="preconfigured_circuits/preconfigured_logits.npz",
    wires_file="preconfigured_circuits/wires.npz",
)

# Get task data
x_data, y_data = get_task_data("binary_multiply", case_n=256, input_bits=8, output_bits=8)

# Run hybrid exploration
results = explore_degenerate_solutions(
    root_wires=wires,
    root_logits=logits,
    x_data=x_data,
    y_data=y_data,
    layer_sizes=layer_sizes,
    exploration_strategy="hybrid",
    bfs_depth=2,
    bfs_perturbations_per_level=100,
    random_walk_iterations=500,
    damage_prob=5.0,
    epochs=200,
)

# Results are automatically saved to exploration_results/exploration_{timestamp}/
# Or load previously saved results:
results = load_exploration_results(Path("exploration_results/exploration_20240101_120000"))
```

### Loading Results for UMAP Visualization

```python
from experiments.explore_degenerate_solutions import load_exploration_results
from pathlib import Path
import jax.numpy as jp
import numpy as np

# Load saved exploration results
results = load_exploration_results(Path("exploration_results/exploration_20240101_120000"))

# Extract logits for UMAP
unique_solutions = results["unique_solutions"]
feature_vectors = []
circuit_hashes = []
depths = []  # For depth coloring

# Compute shortest path distances from root (for depth coloring)
root_hash = results["root_hash"]
exploration_graph = results["exploration_graph"]

# BFS to compute distances from root
from collections import deque
distances = {root_hash: 0}
queue = deque([root_hash])
while queue:
    current = queue.popleft()
    for target_hash, _, metadata in exploration_graph.get(current, []):
        if target_hash not in distances:
            distances[target_hash] = distances[current] + 1
            queue.append(target_hash)

# Flatten logits into feature vectors
for circuit_hash, logits in unique_solutions.items():
    flat_logits = jp.concatenate([l.flatten() for l in logits])
    feature_vectors.append(np.array(flat_logits))
    circuit_hashes.append(circuit_hash)
    depths.append(distances.get(circuit_hash, float('inf')))  # Use inf for unreachable circuits

feature_matrix = np.array(feature_vectors)

# Now ready for UMAP with depth coloring
import umap
import matplotlib.pyplot as plt

reducer = umap.UMAP()
embedding = reducer.fit_transform(feature_matrix)

# Plot with depth coloring (continuous gradient)
plt.scatter(embedding[:, 0], embedding[:, 1], c=depths, cmap='viridis', alpha=0.6)
plt.colorbar(label='Distance from root')
plt.title('UMAP of Circuit Solutions (colored by depth from root)')
plt.show()
```

## Example Implementation Sketch (Legacy - for reference)

```python
from boolean_nca_cc.training.pool.structural_perturbation import (
    create_knockout_vocabulary,
    create_greedy_subset_random_pattern,
)
from boolean_nca_cc.training.evaluation import evaluate_model_stepwise_generator
from boolean_nca_cc.training.backprop import _train_single_knockout_pattern
from boolean_nca_cc.utils import extract_logits_from_graph
import jax
import jax.numpy as jp

class CircuitExplorer:
    def __init__(self, root_circuit, perturbation_patterns, recovery_mode="self_attention"):
        self.root = root_circuit
        self.patterns = perturbation_patterns
        self.recovery_mode = recovery_mode
        # Unique solutions set (for diversity analysis)
        self.unique_solutions = {}  # circuit_hash -> circuit_config
        # Exploration graph (with cycles, for trajectory analysis)
        self.exploration_graph = {}  # circuit_hash -> [Edge(neighbor_hash, pattern_id, metadata)]
        # Track all edges for cycle detection
        self.edges = []  # List of (source_hash, target_hash, pattern_id, metadata)
        
    def _hash_circuit(self, logits):
        """Fast hash for uniqueness detection."""
        # Flatten and hash logits
        flat = jp.concatenate([l.flatten() for l in logits])
        return hash(tuple(flat))
    
    def _is_functional(self, logits, wires, x_data, y_data, threshold=0.95):
        """Check if circuit implements target function."""
        from boolean_nca_cc.circuits.model import run_circuit
        from boolean_nca_cc.circuits.train import compute_accuracy
        
        pred = run_circuit(logits, wires, x_data, hard=True)[-1]
        accuracy = compute_accuracy(pred, y_data)
        return accuracy >= threshold
    
    def _perturb_and_recover(self, circuit, pattern, model=None, wires=None, 
                            x_data=None, y_data=None, layer_sizes=None):
        """Apply perturbation and recover."""
        if self.recovery_mode == "self_attention":
            # Use self-attention recovery
            generator = evaluate_model_stepwise_generator(
                model=model,
                wires=wires,
                logits=circuit,
                x_data=x_data,
                y_data=y_data,
                knockout_pattern=pattern,
                layer_sizes=layer_sizes,
                max_steps=100,
            )
            # Run to convergence
            final_result = None
            for result in generator:
                final_result = result
                if result.accuracy >= 0.95:
                    break
            return final_result.logits if final_result else None
            
        else:  # backprop
            # Use backprop recovery (reversible mode by default)
            from boolean_nca_cc.training.backprop import _train_single_knockout_pattern
            from boolean_nca_cc.circuits.train import TrainState
            import optax
            
            opt = optax.adam(1e-3)
            result = _train_single_knockout_pattern(
                initial_logits=circuit,
                knockout_pattern=pattern,
                opt=opt,
                wires=wires,
                x_data=x_data,
                y_data=y_data,
                loss_type="l4",
                layer_sizes=layer_sizes,
                epochs=200,
                damage_behavior="reversible",  # Use reversible mode for recovery
                reversible_bias=-10.0,  # Bias value to zero damaged gates
            )
            return result["params"] if result["final_accuracy"] >= 0.95 else None
        
    def bfs_explore(self, max_depth=3, model=None, wires=None, x_data=None, 
                   y_data=None, layer_sizes=None):
        """Breadth-first exploration."""
        queue = [(self.root, 0)]  # (circuit, depth)
        visited = set()
        
        while queue:
            circuit, depth = queue.pop(0)
            circuit_hash = self._hash_circuit(circuit)
            
            if circuit_hash in visited or depth > max_depth:
                continue
                
            visited.add(circuit_hash)
            # Add to unique solutions set (deduplicated)
            if circuit_hash not in self.unique_solutions:
                self.unique_solutions[circuit_hash] = circuit
            
            # Apply all perturbation patterns
            for pattern_idx, pattern in enumerate(self.patterns):
                recovered = self._perturb_and_recover(
                    circuit, pattern, model, wires, x_data, y_data, layer_sizes
                )
                if recovered and self._is_functional(recovered, wires, x_data, y_data):
                    child_hash = self._hash_circuit(recovered)
                    
                    # Track edge in exploration graph (allows cycles)
                    edge_metadata = {
                        "pattern_idx": pattern_idx,
                        "depth": depth,
                        "is_cycle": (child_hash == circuit_hash),  # Self-recovery cycle
                        "is_revisit": (child_hash in self.unique_solutions),  # Revisiting known solution
                    }
                    self.edges.append((circuit_hash, child_hash, pattern_idx, edge_metadata))
                    
                    # Track graph structure (allows cycles and revisits)
                    if circuit_hash not in self.exploration_graph:
                        self.exploration_graph[circuit_hash] = []
                    self.exploration_graph[circuit_hash].append((child_hash, pattern_idx, edge_metadata))
                    
                    # Add to unique solutions set
                    if child_hash not in self.unique_solutions:
                        self.unique_solutions[child_hash] = recovered
                    
                    # Add to queue if not visited at this depth (allows revisits via different paths)
                    if child_hash not in visited:
                        queue.append((recovered, depth + 1))
```

## Future Directions

### Advanced Exploration

1. **Multi-root exploration**: Start from multiple preconfigured circuits, merge solution spaces
2. **Adaptive pattern selection**: Learn which perturbation patterns are most effective
3. **Solution space navigation**: Use graph structure to navigate between solutions
4. **Quality-aware exploration**: Prioritize circuits with desirable properties (robustness, efficiency)

### Enhanced Visualization

1. **Dual Connectivity Visualization**: Compare proximity-based (UMAP k-NN) vs. trajectory-based (exploration graph) connectivity
   - **Side-by-side comparison**: UMAP connectivity vs. exploration graph edges on same embedding
   - **Overlaid visualization**: Both connectivity types on single plot with different visual styles
   - **Edge overlap analysis**: Identify edges shared vs. unique to each connectivity type
   - **Path length comparison**: Compare graph distance (trajectory) vs. embedding distance (proximity)
   - **Research insights**: Understand when proximity ≠ reachability, identify "bridge" circuits, analyze solution space topology
   - See section 2.1 "Dual Connectivity Visualization: Proximity vs. Trajectory" for detailed discussion

2. **Trajectory-overlaid UMAP**: Overlay exploration graph trajectories (edges, cycles, paths) onto UMAP embeddings
   - **Diversity + Dynamics**: Combine unique solutions (UMAP points) with exploration trajectories (edges/arrows)
   - **Cycle visualization**: Highlight cycles directly in UMAP space (e.g., curved arrows showing A → B → A)
   - **Path highlighting**: Show perturbation-recovery paths as directed edges between UMAP points
   - **Convergence visualization**: Display multiple paths converging to the same solution with different colored edges
   - **Temporal animation**: Animate exploration progression in UMAP space, showing how trajectories evolve
   - **Benefits**: 
     - See both solution diversity (UMAP layout) and exploration dynamics (trajectory overlays) in one view
     - Identify clusters with high cycle activity (robust solutions)
     - Visualize convergence patterns (multiple paths to same solution)
     - Understand exploration efficiency (trajectory density vs solution diversity)

### Applications

1. **Robustness analysis**: Understand which circuits are most resilient to damage
2. **Diversity benchmarking**: Measure solution space richness for different functions
3. **Circuit design**: Use discovered solutions as starting points for optimization
4. **Theoretical insights**: Understand degeneracy in boolean circuit optimization

## Conclusion

Exploring degenerate circuit solution spaces requires systematic graph traversal strategies. BFS provides comprehensive coverage of the immediate neighborhood, while DFS enables deep exploration of perturbation chains. Hybrid approaches like IDDFS and adaptive strategies balance completeness with efficiency. The choice of strategy depends on exploration goals, computational resources, and desired insights into solution space structure.

The key insight is that reversible damage perturbations create a natural exploration mechanism: each perturbation-recovery sequence is an edge in the solution space graph, and systematic traversal reveals the topology of functionally equivalent circuit configurations. The codebase provides robust infrastructure for both perturbation generation (`structural_perturbation.py`) and recovery mechanisms (self-attention and backprop, both supporting reversible and permanent damage modes), enabling comprehensive exploration of degenerate solution spaces.

## Current Implementation Summary

**Status**: ✅ **Core exploration implemented**

The `experiments/explore_degenerate_solutions.py` script implements:

1. **Hybrid BFS + Random Walk strategy** (recommended for UMAP visualization)
2. **Exploration graph tracking** (allows cycles, tracks all edges)
3. **Save/load functionality** (separates exploration from visualization)
4. **Multiple exploration strategies** (hybrid, BFS, random_walk, single_root)

**Next Steps**:
- ✅ Exploration script: **Complete**
- 🔄 UMAP visualization script: **In progress** (load results, compute embeddings, color by depth from root)
- ⏳ Cycle analysis: **Planned**
- ⏳ Trajectory visualization: **Planned**



RUN TEMPLATE
python experiments/explore_degenerate_solutions.py \
    --exploration-strategy hybrid \
    --bfs-depth 2 \
    --bfs-perturbations-per-level 3 \
    --random-walk-iterations 15 