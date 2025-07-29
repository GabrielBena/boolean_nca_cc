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
    gen_circuit,
    generate_layer_sizes,
    run_circuit,
    run_layer,
)
from boolean_nca_cc.circuits.train import TrainState, train_step
from boolean_nca_cc.models import CircuitGNN, run_gnn_scan
from boolean_nca_cc.training import (
    evaluate_model_stepwise,
    train_model,
)
from boolean_nca_cc.utils import build_graph, extract_logits_from_graph

__version__ = "0.1.0"

__all__ = [
    # Models
    "CircuitGNN",
    "TrainState",
    # Version
    "__version__",
    # Utils
    "build_graph",
    "evaluate_model_stepwise",
    "extract_logits_from_graph",
    "gen_circuit",
    # Circuits
    "generate_layer_sizes",
    "run_circuit",
    "run_gnn_scan",
    "run_layer",
    # Training
    "train_model",
    "train_step",
]
