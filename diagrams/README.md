# Architecture Diagrams

This folder contains Manim animations that visualize the different architectures used in the Boolean Neural Cellular Automata project.

## Available Diagrams

### 1. Meta-Learning Overview (`meta_learning_diagram_final.py`)

**Description**: High-level overview of the meta-learning framework for circuit optimization.

**Components**:
- Circuit Pool: Collection of circuits with different optimization states
- Meta-Learner GNN: The neural network that learns optimization policies
- Inner Loop Optimization: The process of applying the learned policy to circuits
- Loss Computation & Evaluation: How the system evaluates performance and updates
- Process Flow: Numbered arrows showing the complete training cycle

**Key Concepts Illustrated**:
- Pool-based meta-learning approach
- Circuit sampling and updating
- Policy application through GNN parameters θ
- Feedback loop for meta-gradient updates

---

### 2. GNN Architecture (`gnn_architecture.py`)

**Description**: Detailed view of the Graph Neural Network message passing mechanism.

**Components**:
- Input Graph: Circuit representation with node features (logits, hidden, positional encodings)
- Message Computation: Edge update MLP that generates messages between connected nodes
- Message Aggregation: Sum or attention-based aggregation of incoming messages
- Node Update: Node update MLP that combines node features with aggregated messages
- Output Graph: Updated circuit with new node features
- Edge Update: Optional edge feature updates

**Key Concepts Illustrated**:
- Message passing paradigm: `m = MLP(fs, fr, e)`
- Aggregation: `m_agg = Σ(messages)` or attention-based
- Node updates: `f' = MLP(f, m_agg)`
- Residual connections and feature propagation

---

### 3. Self-Attention Architecture (`self_attention_architecture.py`)

**Description**: Detailed view of the self-attention mechanism for circuit optimization.

**Components**:
- Input Features: Concatenation of logits, hidden features, positional encodings, and globals
- Feature Projection: Linear transformation to attention dimension
- Attention Mask: Circuit connectivity-based masking matrix
- Multi-Layer Self-Attention: Stack of attention blocks with Q, K, V computations
- Output Projections: Separate projections for logit and hidden feature updates
- Updated Graph: Circuit with residual updates applied

**Key Concepts Illustrated**:
- Feature concatenation and projection
- Masked self-attention based on circuit topology
- Multi-head attention with residual connections
- Parallel output projections: `Δlogits` and `Δhidden`
- Residual updates: `f' = f + Δf`

## Rendering Instructions

To render any of the diagrams, you need to have Manim installed:

```bash
pip install manim
```

Then render the diagrams using:

```bash
# Meta-learning overview
manim diagrams/meta_learning_diagram_final.py MetaLearningCircuitOptimization -p -ql

# GNN architecture
manim diagrams/gnn_architecture.py GNNArchitectureDiagram -p -ql

# Self-attention architecture  
manim diagrams/self_attention_architecture.py SelfAttentionArchitectureDiagram -p -ql
```

### Rendering Options:
- `-p`: Preview the animation after rendering
- `-ql`: Quick, low quality rendering (faster)
- `-qh`: High quality rendering (slower, better for presentations)
- `-qm`: Medium quality rendering

## Output

The rendered animations will be saved as MP4 files in the `media/videos/` directory with the following structure:
```
media/
└── videos/
    ├── meta_learning_diagram_final/
    ├── gnn_architecture/
    └── self_attention_architecture/
```

## Customization

Each diagram can be customized by modifying:
- Colors and styling in the class constructors
- Component positioning in the `create_main_layout()` methods
- Animation timing and effects (if animations are added)
- Text labels and mathematical formulas

The diagrams use a consistent dark theme (`#1e1e1e` background) and color coding:
- **Blue**: Input/raw features
- **Purple**: Projections and transformations  
- **Green**: Aggregation and final outputs
- **Orange**: Updates and modifications
- **Red**: Core processing (attention/message passing)
- **Yellow**: Masks and constraints

## Architecture Comparison

| Aspect | GNN | Self-Attention |
|--------|-----|----------------|
| **Connectivity** | Explicit message passing along edges | Implicit through attention mask |
| **Computation** | Local neighborhood updates | Global context with masking |
| **Scalability** | O(E) with number of edges | O(N²) with number of nodes |
| **Expressivity** | Limited to graph topology | Can attend to any allowed connection |
| **Parallelization** | Sequential message aggregation | Fully parallel attention computation |

Both architectures achieve similar goals but with different computational trade-offs and modeling assumptions. 