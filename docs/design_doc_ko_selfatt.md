# Knockout Self-Attention Experiment

## Overview

This experiment tests the robustness of self-attention-based optimization of boolean circuits under structural perturbations. The core question is: how well can a graph self-attention model learn to optimize circuit lookup tables (LUTs) when different gates are "knocked out"?

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

### 2. Graph Representation

- **`boolean_nca_cc/utils/graph_builder.py`**

  - Converts boolean circuits into graph structures compatible with JAX/Jraph
  - Creates node features (logits, hidden states, positional encodings)
  - Establishes connectivity patterns for attention mechanisms
  - Main function: `build_graph`
- **`boolean_nca_cc/utils/extraction.py`**

  - Handles graph-to-circuit conversion for evaluation
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
  - Implements masked self-attention for circuit optimization (mask corresponds to circuit topology)
  - Key classes: `CircuitSelfAttention`, `SelfAttentionLayer`, `SelfAttentionBlock`
  - Handles additional attention masking for knockout patterns
  - Provides scan-based iterative optimization functions

### 5. Training Infrastructure

- **`boolean_nca_cc/training/train_loop.py`**

  - Main training loop with pool-based circuit management
  - Integrates knockout evaluation during training
  - Handles metrics tracking and wandb logging
  - keeps track of inner loop eval and performance during training
  - Key function: `train_model`
- **`boolean_nca_cc/training/evaluation.py`**

  - Central evaluation pipeline for knockout experiments
  - Processes knockout patterns and computes performance metrics
  - Key functions: `evaluate_model_stepwise_batched`, `get_loss_and_update_graph`, `evaluate_circuits_in_chunks`
  - Handles batched circuit evaluation with knockout pattern application
- **`train.py`**

  - Entry point script using Hydra configuration
  - Orchestrates model instantiation and training
  - Manages experiment configuration and logging

### 8. Knockout Mechanisms

- **`boolean_nca_cc/training/pool/structural_perturbation.py`**
  - Defines knockout pattern generation
  - Creates reproducible perturbations for evaluation
  - Functions: `create_reproducible_knockout_pattern`, `create_knockout_vocabulary`
  - vocabulary contains set of possible knockout patterns applied to attention mask / graph

### 7. Configuration Management

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

  The primary experiment is the effect of knocking out some gates and introducing these as part of the periodic reset to the pool. The model is then trained on meta-batches, so that some of the sampled circuits in a given batch will have a knockout pattern associated with them. The eval is then on N inner loop steps, starting from NOPs circuits with knockout patterns to see if a configuration can be found that produces the target output bits. The exact configuration is expected to differ for different knockout masks, meaning that the exact LUT configurations of a circuit vary depending on the exact knockout pattern applied to the circuit.
