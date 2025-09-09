# Boolean Neural Cellular Automata with Circuit Communication

This package provides a framework for optimizing boolean circuits using Graph Neural Networks (GNNs). By representing circuits as graphs and applying message passing neural networks, we can optimize circuit parameters without requiring gradient access to the underlying circuits.

![Circuit Visualisation](assets/Circuit.gif)

## Overview

The `boolean_nca_cc` package provides:

1. **Graph Representation** - Convert boolean circuits to graph structures
2. **GNN Models** - Neural network models for circuit optimization
3. **Training** - Functions for training and evaluating GNN models

The package uses JAX and Flax for efficient, differentiable computations.

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd boolean_nca_cc

# Install the package and its dependencies
pip install -e .

# For development
pip install -e ".[dev]"
```

## Dependencies

- JAX and JAXlib
- Flax (nnx modules)
- Jraph (graph neural networks)
- Optax (optimizers)
- NumPy
- Matplotlib (for visualization)
- tqdm (for progress reporting)

## Usage

Here's a simple example of how to use the package to optimize a boolean circuit:

```python
import jax
import jax.numpy as jp
from boolean_nca_cc import CircuitGNN, build_graph, train_gnn, run_gnn_scan
from model import gen_circuit, generate_layer_sizes

# XOR dataset
x_data = jp.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=jp.float32)
y_data = jp.array([[0], [1], [1], [0]], dtype=jp.float32)

# Circuit parameters
arity = 2
layer_sizes = generate_layer_sizes(input_n=2, output_n=1, arity=arity, layer_n=2)

# Generate random circuit
key = jax.random.PRNGKey(42)
wires, logits = gen_circuit(key, layer_sizes, arity=arity)

# Train GNN to optimize the circuit
results = train_gnn(
    layer_sizes=layer_sizes,
    x_data=x_data,
    y_data=y_data,
    arity=arity,
    hidden_dim=16,
    epochs=50,
    n_message_steps=20,
    key=42,
)

# Extract trained GNN
trained_gnn = results["gnn"]
```

See the `example.py` file for a complete example.

## Package Structure

- `boolean_nca_cc/`
  - `models/`: GNN model implementations
    - `gnn.py`: Main CircuitGNN model
    - `node_update.py`: Node update module
    - `edge_update.py`: Edge update module
    - `aggregation.py`: Message aggregation functions
  - `utils/`: Utility functions
    - `graph_builder.py`: Functions to build graph from circuit
    - `extraction.py`: Functions to extract circuit parameters from graph
    - `positional_encoding.py`: Positional encoding utilities
  - `training/`: Training and evaluation
    - `train_step.py`: Single training step implementation
    - `train_loop.py`: Full training loop
    - `evaluation.py`: Evaluation functions

## Key Concepts

### Circuit Representation

Boolean circuits are represented as directed graphs, where:
- Nodes represent gates
- Edges represent connections between gates
- Node features include gate parameters (logits) and hidden state
- Positional encodings capture structural information

### Message Passing

The GNN optimizes circuits through message passing:
1. Nodes (gates) exchange messages with their neighbors
2. Each node updates its parameters based on received messages
3. After multiple iterations, gate parameters are updated to improve circuit performance

### Training

The package supports two training modes:
- **Direct optimization**: Train a GNN to optimize a specific circuit
- **Meta-learning**: Train a GNN to optimize any circuit of a given architecture
