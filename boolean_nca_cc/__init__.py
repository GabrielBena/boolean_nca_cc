"""
Boolean Neural Cellular Automata with Circuit Communication package.

This package provides a framework for optimizing boolean circuits using
graph neural networks (GNNs).

The package is organized into the following submodules:
- models: GNN models for circuit optimization
- utils: Utility functions for graph construction and processing
- training: Training and evaluation functions
"""

from boolean_nca_cc.circuits.model import (
    generate_layer_sizes,
    gen_circuit,
    run_circuit,
    run_layer,
)
from boolean_nca_cc.circuits.train import TrainState, train_step
from boolean_nca_cc.utils import build_graph, extract_logits_from_graph
from boolean_nca_cc.models import CircuitGNN, run_gnn_scan
from boolean_nca_cc.training import (
    train_model,
    evaluate_model_stepwise,
)

__version__ = "0.1.0"

__all__ = [
    # Circuits
    "generate_layer_sizes",
    "gen_circuit",
    "run_circuit",
    "run_layer",
    "TrainState",
    "train_step",
    # Models
    "CircuitGNN",
    "run_gnn_scan",
    # Utils
    "build_graph",
    "extract_logits_from_graph",
    # Training
    "train_model",
    "evaluate_model_stepwise",
    # Version
    "__version__",
]
