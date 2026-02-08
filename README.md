# Boolean Neural Cellular Automata with Circuit Communication

This package provides a framework for optimizing boolean circuits using Graph Neural Networks (GNNs), Self-Attention, and Perceiver-style architectures. By representing circuits as graphs and applying message passing neural networks, we can optimize circuit parameters without requiring gradient access to the underlying circuits.

![Circuit Visualisation](assets/reverse_wire_shuffle.gif)

This GIF shows the adaptive reconfiguration capacity of our Graph Transformer as it updates the circuit Lookup Tables (green cells) to perform a target computation using message passing through the graph's functional edges. This can be seen by the alignment of 'Current Output' to 'Expected Output' in the output bit visualisation. Precise wiring of the circuit is shuffled multiple times, and we can see that the model is able to adapt to the new wiring, recovering overall functionality of the boolean circuit.

## Overview

See 'Manuscript.pdf' for model details.

The `boolean_nca_cc` package provides:

1. **Graph Representation** - Convert boolean circuits to graph structures with configurable topology
2. **Multiple Model Architectures** - GNN, Self-Attention, and Perceiver-style models for circuit optimization
3. **Unified Training Framework** - Meta-learning approach with pool-based training and consistent evaluation
4. **Multiple Boolean Tasks** - Various boolean circuit tasks for training and evaluation
5. **Configuration Management** - Hydra-based configuration system
6. **Experiment Tracking** - Weights & Biases integration
7. **Interactive Demo** - Live circuit optimization visualization

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

# For notebooks and visualization
pip install -e ".[notebooks]"

# Install everything
pip install -e ".[all]"
```

## Dependencies

### Core Dependencies
- **JAX ecosystem**: JAX, JAXlib, Flax (nnx modules), Jraph, Optax
- **Scientific computing**: NumPy
- **Configuration**: Hydra-core, OmegaConf
- **Experiment tracking**: Weights & Biases
- **Utilities**: tqdm, IPython

### Optional Dependencies
- **Visualization**: matplotlib, seaborn, Pillow, imgui-bundle
- **Development**: pytest, black, isort, mypy
- **Notebooks**: Jupyter

## Quick Start

### Training with Configuration Files

The easiest way to train models is using the Hydra configuration system:

```bash
# Train with default configuration (Self-Attention model)
python train.py

# Train a GNN model
python train.py model=gnn

# Train a Perceiver model
python train.py model=perceiver_attention

# Train on a specific task
python train.py circuit.task=binary_multiply

# Override specific parameters
python train.py training.learning_rate=1e-4 training.epochs=1000
```

### Programmatic Usage

```python
import jax
from flax import nnx
from boolean_nca_cc import generate_layer_sizes
from boolean_nca_cc.circuits.tasks import get_task_data
from boolean_nca_cc.training import train_model
from boolean_nca_cc.models import CircuitSelfAttention

# Setup task data
x_data, y_data = get_task_data("binary_multiply", case_n=256, input_bits=8, output_bits=8)

# Generate circuit architecture
layer_sizes = generate_layer_sizes(input_n=8, output_n=8, arity=4, layer_n=3)
n_node = sum(size[0] for size in layer_sizes)

# Initialize model
model = CircuitSelfAttention(
    n_node=n_node,
    circuit_hidden_dim=16,
    attention_dim=128,
    arity=4,
    rngs=nnx.Rngs(0)
)

# Train the model
results = train_model(
    key=42,
    init_model=model,
    x_data=x_data,
    y_data=y_data,
    layer_sizes=layer_sizes,
    hidden_dim=16,
    arity=4,
    learning_rate=1e-3,
    epochs=1000,
    n_message_steps=8,
    wiring_mode="random",
    meta_batch_size=256,
)
```

### Evaluation

The package provides a unified evaluation API:

```python
from boolean_nca_cc.training import (
    evaluate_model_stepwise_batched,
    evaluate_model_stepwise_generator,
    run_model_scan_with_loss,
)

# Batched evaluation with automatic chunking for memory efficiency
final_graphs, step_metrics = evaluate_model_stepwise_batched(
    model=model,
    batch_wires=wires,
    batch_logits=logits,
    x_data=x_data,
    y_data=y_data,
    input_n=8,
    arity=4,
    circuit_hidden_dim=16,
    n_message_steps=100,
    layer_sizes=layer_sizes,
    chunk_size=32,  # Process in chunks for large batches
)

# For live demos and streaming results
generator = evaluate_model_stepwise_generator(
    model=model,
    wires=single_wires,
    logits=single_logits,
    x_data=x_data,
    y_data=y_data,
    input_n=8,
    max_steps=None,  # Run indefinitely
    layer_sizes=layer_sizes,
)
for step_result in generator:
    print(f"Step {step_result.step}: Accuracy {step_result.accuracy:.4f}")
```

### Interactive Demo

Run the interactive circuit optimization demo:

```bash
python training_demo.py
```

This provides a GUI for real-time circuit optimization visualization.

## Available Models

### Graph Neural Network (GNN)
- Message passing between circuit gates
- Configurable node and edge update functions  
- Attention-based message aggregation
- Configuration: `model=gnn`

### Self-Attention
- Transformer-style self-attention mechanism
- Masked attention based on circuit connectivity
- Multi-head attention with configurable layers
- ReZero initialization for stable training
- Configuration: `model=self_attention`

### Perceiver-style Attention
- Cross-attention to input data patterns and output residuals
- Provides direct access to the information backpropagation uses
- Layer-restricted attention for structural constraints
- Configuration: `model=perceiver_attention`

## Boolean Circuit Tasks

The package includes several predefined boolean circuit tasks:

| Task | Description |
|------|-------------|
| `binary_multiply` | Multiply two halves of input |
| `add` | Binary addition of input halves |
| `and` | Bitwise AND operation |
| `xor` | Bitwise XOR operation |
| `parity` | Compute input parity |
| `reverse` | Reverse bit order |
| `copy` / `identity` | Identity function |
| `gray` | Gray code conversion |
| `popcount` | Count number of 1 bits |
| `text` | Text rendering to binary pattern |
| `noise` | Random binary patterns |

## Configuration System

The training system uses Hydra for configuration management. Configuration files are located in `configs/`:

- `config.yaml`: Main configuration file
- `model/`: Model-specific configurations
  - `gnn.yaml`: GNN model settings
  - `self_attention.yaml`: Self-attention model settings
  - `perceiver_attention.yaml`: Perceiver model settings

### Key Configuration Sections

```yaml
# Circuit configuration
circuit:
  task: "binary_multiply"
  input_bits: 8
  output_bits: 8
  arity: 4
  num_layers: 3

# Training parameters
training:
  learning_rate: 1e-3
  epochs: 1000
  loss_type: "l4"  # or "bce", "l2"
  wiring_mode: "random"  # or "fixed", "genetic"
  n_message_steps: 8

# Pool-based meta-learning
pool:
  size: 1024
  reset_fraction: 0.05
  reset_interval: 32

# Experiment tracking
wandb:
  enabled: true
  project: "boolean-nca-cc"
  entity: "your-entity"
```

## Package Structure

```
boolean_nca_cc/
├── circuits/                      # Circuit generation and tasks
│   ├── model.py                  # Circuit creation and execution
│   ├── tasks.py                  # Boolean task definitions
│   ├── train.py                  # Circuit training utilities
│   └── viz.py                    # Circuit visualization helpers
├── models/                        # Neural network models
│   ├── attention/                # Attention-based models
│   │   ├── base.py              # Shared attention components
│   │   ├── self_attention.py    # Self-attention model
│   │   └── perceiver_attention.py # Perceiver-style model
│   └── gnn/                      # Graph neural network
│       ├── model.py             # Main GNN model
│       ├── node_update.py       # Node update modules
│       ├── edge_update.py       # Edge update modules
│       └── aggregation.py       # Message aggregation
├── training/                      # Training infrastructure
│   ├── train_loop.py            # Main training loop
│   ├── evaluation.py            # Unified evaluation functions
│   ├── eval_datasets.py         # Evaluation dataset creation
│   ├── checkpointing.py         # Model checkpointing
│   ├── schedulers.py            # Learning rate schedules
│   └── pool/                    # Pool-based meta-learning
│       ├── pool.py              # Circuit pool management
│       └── structural_perturbation.py  # Gate knockout/damage
└── utils/                         # Utility functions
    ├── graph_builder.py         # Graph construction
    ├── extraction.py            # Parameter extraction
    └── positional_encoding.py   # Positional encodings
```

## Key Features

### Meta-Learning with Circuit Pools
- Maintains a pool of circuits with varying optimization states
- Periodic reset strategies to maintain diversity
- Supports different wiring modes (fixed, random, genetic)

### Unified Evaluation System
- Single `run_model_scan_with_loss` function for all model types
- Consistent GraphGlobals across all models
- Support for structural damage/knockout during evaluation
- Memory-efficient chunked processing for large batches

### Advanced Training Features
- Learning rate scheduling (linear warmup, cosine, exponential)
- Checkpointing with best model tracking
- Multiple loss functions (L4, L2, Binary Cross-Entropy)
- Gradient checkpointing for memory efficiency
- Random loss step sampling for robust training

### Evaluation and Visualization
- Stepwise evaluation of inner optimization loops
- Training curve plotting
- Comparison with backpropagation baselines
- Interactive visualization tools

### Experiment Management
- Weights & Biases integration for experiment tracking
- Hydra configuration management
- Automatic hyperparameter logging
- Model checkpointing and restoration

## Citation

If you use this code in your research, please cite:

```bibtex
@software{boolean_nca,
  title={Self-Organising Digital Circuits},
  author={Gabriel Béna, Marcello Barylli, Alexander Mordvintsev},
  year={2025},
  url={https://github.com/GabrielBena/boolean_nca_cc}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
