# Knockout Self-Attention Experiment

## Overview

This experiment tests the robustness of self-attention-based optimization of boolean circuits under structural perturbations. The core question is: how well can a self-attention model learn to optimize circuit lookup tables (LUTs) when different gates are "knocked out" through attention masking?

## Architecture Flow

The experimental pipeline follows this sequence:

**Circuit Creation → Graph Representation → Self-Attention Optimization → Knockout Evaluation**

## Core Components

this codebase uses the 'metabool' conda environment. IMPORTANT: conda activate metabool before running any testing script

### 1. Boolean Circuit Foundation
- **`boolean_nca_cc/circuits/model.py`**
  - Implements differentiable boolean circuits with lookup tables
  - Provides circuit generation (`gen_circuit`) and execution (`run_circuit`)
  - Core functions: `make_nops`, `run_layer`, `gen_wires`

- **`boolean_nca_cc/circuits/train.py`**
  - **CRITICAL FOR KNOCKOUT TESTING**: Provides fundamental loss functions that measure knockout impact
  - Implements L4 norm and binary cross-entropy loss computation
  - Key functions: `loss_f_l4`, `loss_f_bce`, `compute_accuracy`, `binary_cross_entropy`
  - Pre-compiled gradient functions for efficient training

### 2. Graph Representation
- **`boolean_nca_cc/utils/graph_builder.py`**
  - Converts boolean circuits into graph structures compatible with JAX/Jraph
  - Creates node features (logits, hidden states, positional encodings)
  - Establishes connectivity patterns for attention mechanisms
  - Main function: `build_graph`

- **`boolean_nca_cc/utils/extraction.py`**
  - **CRITICAL FOR KNOCKOUT TESTING**: Handles graph-to-circuit conversion for evaluation
  - Extracts logits from graph representations after model updates
  - Key functions: `extract_logits_from_graph`, `update_output_node_loss`, `get_output_node_indices`
  - Ensures knockout evaluation can properly measure circuit performance

### 3. Circuit Pool Management
- **`boolean_nca_cc/training/pool/pool.py`**
  - Central component managing collections of circuits during training
  - `GraphPool` class: Batched storage and manipulation of graph structures
  - Supports sampling, updating, and resetting circuit populations
  - Integrates knockout pattern management for persistent perturbations
  - Implements multiple reset strategies (uniform, loss-biased, steps-biased)
  - Key functions: `sample`, `update`, `reset_fraction`, `initialize_graph_pool`

### 4. Self-Attention Model
- **`boolean_nca_cc/models/self_attention.py`**
  - Implements masked self-attention for circuit optimization
  - Key classes: `CircuitSelfAttention`, `SelfAttentionLayer`, `SelfAttentionBlock`
  - Handles attention masking for knockout patterns
  - Provides scan-based iterative optimization functions

### 5. Training Infrastructure
- **`boolean_nca_cc/training/train_loop.py`**
  - Main training loop with pool-based circuit management
  - Integrates knockout evaluation during training
  - Handles metrics tracking and wandb logging
  - Key function: `train_model`

- **`boolean_nca_cc/training/evaluation.py`**
  - **CRITICAL FOR KNOCKOUT TESTING**: Central evaluation pipeline for knockout experiments
  - Processes knockout patterns and computes performance metrics
  - Key functions: `evaluate_model_stepwise_batched`, `get_loss_and_update_graph`, `evaluate_circuits_in_chunks`
  - Handles batched circuit evaluation with knockout pattern application

- **`train.py`**
  - Entry point script using Hydra configuration
  - Orchestrates model instantiation and training
  - Manages experiment configuration and logging

### 6. Configuration Management
- **`configs/config.yaml`**
  - Central hub controlling all experiment parameters
  - **Critical Training Parameters**:
    - `training`: Batch sizes, learning rates, optimizer settings
    - `pool`: Circuit population management and knockout injection settings
    - `eval`: Knockout evaluation frequency and damage parameters
  - **Key Knockout Controls**:
    - `persistent_knockout.fraction`: Percentage of circuits receiving knockouts
    - `persistent_knockout.damage_prob`: Knockout density per circuit
    - `knockout_diversity`: Number of unique knockout patterns
  - **Model Selection**: Swappable model architectures via Hydra config

- **`configs/model/self_attention.yaml`**
  - Model-specific hyperparameters:
    - Self-attention layer dimensions
    - Number of attention heads
    - Masking strategies for knockout nodes
    - Transformer architecture variants
  - Optimization parameters for attention-based updates

### 7. Knockout Mechanisms
>
- **`boolean_nca_cc/training/pool/structural_perturbation.py`**
  - Defines knockout pattern generation
  - Creates reproducible perturbations for evaluation
  - Functions: `create_reproducible_knockout_pattern`, `create_knockout_vocabulary`

## Key Experimental Features

### Knockout Pattern Management
- **Vocabulary-based patterns**: Fixed sets of knockout configurations for consistent evaluation
- **In/Out-of-distribution testing**: Separate evaluation on seen vs unseen knockout patterns
- **Attention masking**: Prevents knocked-out gates from participating in message passing

### Training Pool System
- Dynamic circuit population with periodic resets
- Integration of knockout patterns during training
- Performance tracking across different structural perturbations

### Evaluation Metrics
- Standard circuit performance (loss, accuracy)
- Robustness measures under knockout conditions
- Generalization assessment (in-dist vs out-of-dist patterns)

## Configuration

The experiment uses Hydra configuration management through `configs/` directory, allowing systematic parameter sweeps and reproducible experiments.

## Key JAX Design Patterns

- **Functional programming**: Pure functions with explicit random keys
- **JIT compilation**: Performance optimization for training loops
- **Vectorization**: Batch processing of circuit evaluations
- **Structural typing**: Consistent graph representations across components

## Navigation Tips

1. Start with `train.py` to understand the overall experiment flow
2. Examine `model.py` for boolean circuit fundamentals
3. Study `circuits/train.py` for loss functions that measure knockout impact
4. Review `graph_builder.py` and `extraction.py` for circuit-to-graph conversion
5. Understand `pool.py` for circuit population management and sampling strategies
6. Analyze `self_attention.py` for the core optimization mechanism
7. Check `structural_perturbation.py` for knockout pattern logic
8. Explore `training/evaluation.py` for the knockout testing pipeline
9. Review `train_loop.py` for training dynamics and evaluation integration
