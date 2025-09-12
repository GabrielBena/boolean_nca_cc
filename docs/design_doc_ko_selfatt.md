# Knockout Self-Attention Experiment

## Overview

This experiment tests the robustness of Graph Transformer-based optimization of boolean circuits under structural perturbations. The core question is: how well can a graph self-attention model learn to optimize circuit lookup tables (LUTs) when different gates are "knocked out"?

The framework now supports multiple training modes, damage types, and advanced attention mechanisms for comprehensive circuit optimization and damage recovery studies.

## Architecture Flow

The experimental pipeline follows this sequence:

**Circuit Creation → Graph Representation → Self-Attention Optimization → Knockout Evaluation**

## Core Components

this codebase uses the 'metabool' conda environment. IMPORTANT: conda activate metabool before running any testing script

### 1. Boolean Circuit Foundation

Circuits are layered networks of LUTs as nodes, with N inputs and M outputs. Circuit-level tasks are defined as eg binary multiply, reverse of input bit processing. Loss is the element-wise difference between circuit output and target pattern at output bits.

- **`boolean_nca_cc/circuits/model.py`**
  - Implements differentiable boolean circuits with lookup tables as gates
  - Provides circuit generation (`gen_circuit`) and execution (`run_circuit`)
  - Core functions: `make_nops`, `run_layer`, `gen_wires`

### 2. Graph Representation

- **`boolean_nca_cc/utils/graph_builder.py`**

  - Converts boolean circuits into graph structures compatible with JAX/Jraph
  - Creates node features (logits, hidden states, positional encodings)
  - Establishes connectivity patterns for graph attention mechanisms
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
  - **Advanced Features**:
    - **Damage Emission**: When enabled, knocked-out nodes cannot receive attention but can still send information to other nodes
    - **Layer Neighbors**: Allows attention between adjacent nodes within the same layer for enhanced local communication
    - **Multiple Attention Heads**: Configurable multi-head attention for parallel processing
    - **Residual Connections**: ReZero-style learnable scaling for update rates

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

- **`boolean_nca_cc/training/pool/perturbation.py`**
  - Defines knockout pattern generation
  - Creates reproducible perturbations for evaluation
  - Functions: `create_reproducible_knockout_pattern`, `create_knockout_vocabulary`
  - vocabulary contains set of possible knockout patterns applied to attention mask / graph
  - **Damage Types**:
    - **Shotgun**: Random knockout patterns across eligible gates
    - **Strip**: Localized damage within radius of randomly selected centers
    - **Greedy**: Deterministic patterns based on predefined ordered gate indices. Selects gate whose knockout results in largest performance drop (gate 1). Selects gate 2 by seeing which second gate, in combination with gate 1, results in the largest performance drop.
      - Possible next feature: Since Greedy only allows for a single damage sequence, ie most damaging, look for alternatives that allow a vocabulary of several damage sizes. Eg: Propose single gate damage, if performance drop is > 0, see if unordered set of gates (1,2,3 gates etc) is already in vocabulary, accept into vocabulary if it is not.
  - **Sample-Based Damage**: Independent damage application to pool entries during training (decoupled from resets)

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
  - **Training Mode Configuration**:
    - `training.training_mode`: "growth" or "repair" mode selection
    - `training.damage_emission`: Enable/disable damage emission attention masking
    - `training.layer_neighbors`: Enable/disable layer neighbor attention
  - **Sample-Based Damage Configuration**:
    - `pool.damage_pool_enabled`: Enable sample-based damage
    - `pool.damage_mode`: "shotgun", "strip", or "greedy" damage patterns
    - `pool.damage_prob`: Number of gates to knock out per pattern
    - `pool.damage_knockout_diversity`: Vocabulary size for shared patterns
    - `pool.greedy_ordered_indices`: Predefined gate order for greedy damage
- **`configs/model/self_attention.yaml`**

  - Model-specific hyperparameters:
    - Self-attention layer dimensions
    - Number of attention heads
    - Masking strategies for knockout nodes
    - Transformer architecture variants
  - Optimization parameters for attention-based updates
  - **Advanced Configuration**:
    - `damage_emission`: Attention masking policy for knocked-out nodes
    - `layer_neighbors`: Enable adjacent node attention within layers
    - `zero_init`: Initialize weights to zero for stable training
    - `re_zero_update`: Learnable scaling for residual connections

  The primary experiment is the effect of knocking out some gates and introducing these as part of the periodic reset to the pool. The model is then trained on meta-batches, so that some of the sampled circuits in a given batch will have a knockout pattern associated with them. The eval is then on K inner loop steps, starting from NOPs circuits with knockout patterns to see if a configuration can be found that produces the target output bits. The exact configuration is expected to differ for different knockout masks, meaning that the exact LUT configurations of a circuit vary depending on the exact knockout pattern applied to the circuit.

## Training Modes

### Growth Mode (Default)

- Pool initializes with randomly wired NOPs circuits
- Self-attention learns to configure fresh circuits from scratch
- Resets reintroduce fresh, unconfigured circuits
- Focus: Learning to optimize circuits from initial state

### Repair Mode

- Performs one-time backprop optimization on fixed wiring to obtain working configuration
- Initializes entire pool by cloning the preconfigured circuit
- Resets also reintroduce clones of the preconfigured circuit
- Focus: Learning to recover from damage to already-functioning circuits
- Enables study of reconfiguration strategies after structural damage

## Damage Application Strategies

### Sample-Based Damage (Current)

- **Independent of resets**: Damage applied to existing pool entries during training
- **Flexible timing**: Can occur at any point during circuit optimization
- **Pool update filtering**: Prevents damage to over-optimized circuits
- **Vocabulary-based**: Reuses shared knockout patterns for consistent evaluation
- **Selection strategies**: Uniform, loss-biased, steps-biased, or combined selection
- **Late-stage reconfiguration**: Enables study of functioning circuit → damaged → recovery

## Advanced Attention Mechanisms

### Damage Emission

- **Standard mode**: Knocked-out nodes cannot send or receive attention (symmetric masking)
- **Emission mode**: Knocked-out nodes cannot receive attention but can still send information
- **Use case**: Allows damaged nodes to influence recovery while preventing interference

### Layer Neighbors

- Enables attention between adjacent nodes within the same layer
- Enhances local communication patterns
- Configurable per-layer neighbor connections
- **Use case**: Improves local circuit optimization and damage recovery

## Evaluation Framework

### In-Distribution vs Out-of-Distribution

- **IN-distribution**: Uses shared knockout vocabulary from training
- **OUT-of-distribution**: Generates fresh patterns with different random seeds
- **Comparative analysis**: Tests generalization to unseen damage patterns

### Hamming Distance Analysis

- Measures circuit reconfiguration distance after damage
- Compares baseline vs damaged circuit truth tables
- Tracks per-gate and per-layer changes
- **Visualization**: Accuracy vs distance plots for damage recovery analysis

## Experimental Workflow

### Basic Usage

```bash
# Activate environment
conda activate metabool

# Growth mode (default) - learn to optimize from scratch
python train.py

# Repair mode - learn to recover from damage
python train.py training.training_mode=repair

# Enable sample-based damage
python train.py pool.damage_pool_enabled=true

# Configure damage type
python train.py pool.damage_mode=strip pool.damage_prob=10

# Enable advanced attention features
python train.py training.damage_emission=true training.layer_neighbors=true
```

### Sweep Configuration

The framework supports comprehensive hyperparameter sweeps via `sweep.yaml`:

- Training modes: growth vs repair
- Damage modes: shotgun vs strip vs greedy
- Attention mechanisms: damage emission, layer neighbors
- Damage parameters: probability, diversity, selection strategies

### Key Experimental Questions

1. **Robustness**: How well do different attention mechanisms handle structural damage?
2. **Reconfiguration**: Can models learn to recover from damage in repair mode?
3. **Generalization**: How do models perform on unseen damage patterns (OOD evaluation)?
4. **Damage Types**: Which damage patterns are most challenging for recovery?
5. **Attention Design**: How do damage emission and layer neighbors affect performance?
