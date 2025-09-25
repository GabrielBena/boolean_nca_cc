# Knockout Self-Attention Experiment

## Overview

This experiment tests the robustness of self-attention-based optimization of boolean circuits under structural perturbations. The core question is: how well can a graph self-attention model learn to optimize circuit lookup tables (LUTs) when different gates are "knocked out" or damaged?

The system now supports a comprehensive range of damage modes, evaluation trajectories, and recovery mechanisms, enabling sophisticated analysis of circuit robustness and adaptation under various perturbation scenarios.

## Architecture Flow

The experimental pipeline follows this sequence:

**Circuit Creation → Graph Representation → Self-Attention Optimization → Damage Application → Recovery Analysis**

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
  - **Damage Behavior Support**:
    - **Permanent Mode**: Traditional knockout behavior (prune attention, clamp logits, zero residual updates)
    - **Reversible Mode**: SEU-like damage with one-shot logit bias and recovery potential
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

### 6. Damage and Perturbation Mechanisms

- **`boolean_nca_cc/training/pool/structural_perturbation.py`**
  - Defines comprehensive knockout pattern generation
  - **Pattern Generation Modes**:
    - **Shotgun**: Random uniform sampling across all gates
    - **Strip**: Systematic layer-by-layer damage patterns
    - **Greedy**: Deterministic rolling window through critical gate indices
    - **Greedy Vocabulary**: Statistical sampling from greedy-ordered critical indices
  - **Key Functions**:
    - `create_reproducible_knockout_pattern`: Basic pattern generation
    - `create_knockout_vocabulary`: Vocabulary-based pattern libraries
    - `create_greedy_subset_random_pattern`: Greedy-based statistical sampling
    - `create_group_greedy_pattern`: Deterministic rolling window patterns
  - **Vocabulary Management**: Creates diverse pattern libraries for statistical robustness

### 7. Configuration Management

- **`configs/config.yaml`**

  - Central hub controlling all experiment parameters
  - **Critical Training Parameters**:
    - `training`: Batch sizes, learning rates, optimizer settings
    - `pool`: Circuit population management and damage injection settings
    - `eval`: Knockout evaluation frequency and damage parameters
  - **Unified Damage Control System**:
    - `damage_mode`: Pattern generation strategy ("shotgun", "strip", "greedy", "greedy_vocabulary")
    - `damage_injection_mode`: Single vs multi-damage per circuit ("single", "multi")
    - `max_damage_per_circuit`: Maximum damage events per circuit
    - `damage_pool_fraction`: Percentage of circuits damaged per training injection
    - `damage_knockout_diversity`: Vocabulary size for pattern generation
    - `damage_prob`: Number of gates per damage pattern
  - **Model Selection**: Swappable model architectures via Hydra config
- **`configs/model/self_attention.yaml`**

  - Model-specific hyperparameters:
    - Self-attention layer dimensions
    - Number of attention heads
    - **Damage Behavior**: `damage_behavior` ("permanent" vs "reversible")
    - **Reversible Damage**: `reversible_bias` parameter for SEU-like damage
    - Masking strategies for knockout nodes
    - Transformer architecture variants
  - Optimization parameters for attention-based updates

## Experimental Framework

### Core Experiment Design

The primary experiment investigates the effect of structural damage on circuit optimization. The model is trained on meta-batches where some circuits have damage patterns applied, then evaluated on K inner loop steps to see if alternative configurations can be found that produce the target output despite damage.

**Key Insight**: The exact LUT configurations of a circuit should vary depending on the damage pattern applied, demonstrating the model's ability to find alternative solutions under constraints.

### Damage Application Modes

#### 1. **Permanent Damage (Stuck-at Faults)**

- **Behavior**: Gates are completely removed from attention graph and zeroed out
- **Use Case**: Testing circuit robustness under permanent structural loss
- **Recovery**: No recovery possible - circuits must work around damaged gates

#### 2. **Reversible Damage (SEU-like)**

- **Behavior**: One-shot logit bias applied to damaged gates, then normal updates resume
- **Use Case**: Testing circuit recovery from temporary perturbations
- **Recovery**: Gates can "heal" through normal optimization updates
- **Implementation**: `damage_behavior: "reversible"` with tunable `reversible_bias`

### Pattern Generation Strategies

#### 1. **Shotgun Mode**

- **Pattern**: Random uniform sampling across all gates
- **Use Case**: Baseline random damage testing
- **Statistical Properties**: Uniform distribution, no bias toward critical gates

#### 2. **Strip Mode**

- **Pattern**: Systematic layer-by-layer damage
- **Use Case**: Testing sensitivity to different circuit depths
- **Statistical Properties**: Structured damage with spatial locality

#### 3. **Greedy Mode (Rolling Window)**

- **Pattern**: Deterministic cycling through critical gate indices
- **Use Case**: Targeted damage to most important gates
- **Statistical Properties**: Deterministic, reproducible sequences
- **Implementation**: `damage_mode: "greedy"` with `greedy_window_size`

#### 4. **Greedy Vocabulary Mode (Statistical)**

- **Pattern**: Random sampling from critical gate indices
- **Use Case**: Statistical robustness testing with targeted damage
- **Statistical Properties**: Biased toward critical gates but with diversity
- **Implementation**: `damage_mode: "greedy_vocabulary"` with vocabulary generation

### Training Modes

#### **1. Growth Mode**
- **Objective**: Learn to configure circuits from scratch (NOPs → configured)
- **Pool Initialization**: Randomly wired NOPs circuits (unconfigured)
- **Training Focus**: Self-attention learns to configure fresh circuits from scratch
- **Resets**: Reintroduce fresh, unconfigured circuits
- **Use Case**: Testing circuit optimization and configuration learning
- **Configuration**: `training_mode: "growth"`

#### **2. Repair Mode**
- **Objective**: Learn to reconfigure already working circuits after damage
- **Preconfiguration**: One-time backprop optimization creates working base circuit `(base_wires, base_logits)`
- **Pool Initialization**: Clone preconfigured working circuits (with optional noise)
- **Training Focus**: Recovery from damage rather than growth from NOPs
- **Resets**: Reintroduce clones of preconfigured circuit
- **Use Case**: Testing damage recovery and circuit reconfiguration capabilities
- **Configuration**: `training_mode: "repair"` with `preconfig_steps` and `preconfig_lr`

### Damage Injection Control

#### **Single Damage Mode**

- **Training**: Each circuit damaged at most once (`perturb_counter == 0`)
- **Evaluation**: Each circuit damaged at most once per trajectory
- **Use Case**: Testing one-shot damage recovery
- **Configuration**: `damage_injection_mode: "single"`

#### **Multi-Damage Mode**

- **Training**: Circuits can be damaged multiple times up to limit
- **Evaluation**: Multiple damage events per trajectory with recovery steps
- **Use Case**: Testing cumulative damage effects and adaptation
- **Configuration**: `damage_injection_mode: "multi"` with `max_damage_per_circuit`

### Evaluation Trajectories

#### **Seen vs Unseen Patterns**

- **Seen (IN)**: Evaluation uses patterns from training vocabulary
- **Unseen (OUT)**: Evaluation generates fresh patterns from same distribution
- **Purpose**: Testing generalization vs memorization of damage patterns

#### **Statistical Robustness**

- **Batch Evaluation**: Multiple patterns per injection for confidence intervals
- **Error Bands**: Mean ± std across patterns at each recovery step
- **Trajectory Plotting**: Damage/recovery curves with statistical significance

### Pool Management Evolution

#### **Current Implementation**

- **Reset-based Damage**: Damage applied only to newly reset circuits (NOPs initialization)
- **Pool Tracking**: `perturb_counter` tracks cumulative damage events per circuit
- **Damage Injection**: Periodic damage application during training

#### **Future Enhancement: Sample-based Damage**

- **Goal**: Damage existing optimized circuits in the pool
- **Challenge**: Functioning circuit → damaged → **reconfiguration** to alternative working circuit
- **Benefit**: Tests adaptation under damage rather than just initialization robustness

## Experimental Capabilities Matrix

### Training Mode Combinations

| Training Mode | Damage Mode         | Injection Mode | Damage Type          | Pattern Source            | Use Case                           |
| --------------- | --------------------- | ---------------- | ---------------------- | --------------------------- | ----------------------------------- |
| **Growth**     | `shotgun`           | single/multi   | permanent/reversible | random uniform            | Circuit optimization + robustness  |
| **Growth**     | `strip`             | single/multi   | permanent/reversible | layer-based               | Circuit optimization + depth sensitivity |
| **Growth**     | `greedy`            | single/multi   | permanent/reversible | deterministic sequence    | Circuit optimization + targeted damage |
| **Growth**     | `greedy_vocabulary` | single/multi   | permanent/reversible | statistical from critical | Circuit optimization + robust damage |
| **Repair**     | `shotgun`           | single/multi   | permanent/reversible | random uniform            | Damage recovery + robustness        |
| **Repair**     | `strip`             | single/multi   | permanent/reversible | layer-based               | Damage recovery + depth sensitivity |
| **Repair**     | `greedy`            | single/multi   | permanent/reversible | deterministic sequence    | Damage recovery + targeted damage   |
| **Repair**     | `greedy_vocabulary` | single/multi   | permanent/reversible | statistical from critical | Damage recovery + robust damage     |

## Configuration Examples

### Growth Mode: Circuit Optimization + Damage Robustness

```yaml
training:
  training_mode: "growth"  # Learn to configure circuits from scratch

pool:
  damage_mode: "greedy_vocabulary"
  damage_injection_mode: "multi"
  max_damage_per_circuit: 5
  damage_pool_fraction: 0.15

model:
  damage_behavior: "reversible"
  reversible_bias: -10.0
```

### Repair Mode: Damage Recovery + Reconfiguration

```yaml
training:
  training_mode: "repair"  # Learn to recover from damage
  preconfig_steps: 200
  preconfig_lr: 1e-2

pool:
  damage_mode: "greedy"
  damage_injection_mode: "single"
  max_damage_per_circuit: 1
  damage_pool_fraction: 0.15

model:
  damage_behavior: "permanent"
```

### Single Damage with Reversible Recovery

```yaml
pool:
  damage_mode: "greedy_vocabulary"
  damage_injection_mode: "single"
  max_damage_per_circuit: 1
  damage_pool_fraction: 0.15

model:
  damage_behavior: "reversible"
  reversible_bias: -10.0
```

### Multi-Damage with Permanent Knockouts

```yaml
pool:
  damage_mode: "greedy"
  damage_injection_mode: "multi"
  max_damage_per_circuit: 10
  greedy_window_size: 5

model:
  damage_behavior: "permanent"
```

This comprehensive framework enables sophisticated analysis of circuit robustness, recovery mechanisms, and adaptation strategies under various damage scenarios.

## Implementation Status

### ✅ **Completed Features**

#### **1. Training Modes (Growth vs Repair)**

- **Growth Mode**: Learn to configure circuits from scratch (NOPs → configured)
- **Repair Mode**: Learn to recover from damage (configured → damaged → recovered)
- **Preconfiguration**: Repair mode uses one-time backprop to create working base circuits
- **Pool Management**: Different initialization strategies for each mode
- **Configuration**: `training_mode: "growth"` or `training_mode: "repair"`

#### **2. Reversible Damage (SEU-like)**

- **Model Integration**: `CircuitSelfAttention` supports both permanent and reversible damage modes
- **One-shot Bias**: Configurable `reversible_bias` parameter for damage strength
- **Recovery Mechanism**: Normal residual updates allow gates to "heal" after initial damage
- **Evaluation Support**: Stepwise evaluation shows damage application and recovery trajectories

#### **3. Greedy Vocabulary Mode**

- **Pattern Generation**: Statistical sampling from critical gate indices
- **Vocabulary Management**: Diverse pattern libraries for robust evaluation
- **Training Integration**: Pool damage injection uses vocabulary sampling
- **Evaluation Extension**: Both seen (vocabulary) and unseen (fresh) pattern testing

#### **4. Unified Damage Control System**

- **Single vs Multi-Damage**: Configurable damage limits per circuit
- **Counter Systems**: Separate tracking for training (`perturb_counter`) and evaluation (`eval_perturb_counter`)
- **Parameter Flow**: Complete configuration from YAML through training to evaluation
- **Backward Compatibility**: All existing modes remain functional

#### **5. Multi-Damage Evaluation**

- **Trajectory Plotting**: Proper damage/recovery curves with statistical error bands
- **Batch Processing**: Multiple patterns per injection for confidence intervals
- **Parameter Alignment**: Consistent damage injection parameters across training and evaluation
- **Statistical Robustness**: Mean ± std aggregation across patterns

#### **6. Configuration Framework**

- **Unified Control**: `damage_mode` controls both pattern type and injection timing
- **Parameter Separation**: Clear roles for different configuration parameters
- **Example Configs**: Complete configuration examples for all modes
- **Validation**: Comprehensive parameter validation and error handling

### **Key Benefits**

#### **1. Experimental Flexibility**

- **32 Mode Combinations**: 2 training modes × 4 damage modes × 2 injection modes × 2 damage types
- **Dual Training Paradigms**: Growth (optimization) vs Repair (recovery) training
- **Statistical Robustness**: Vocabulary-based sampling with confidence intervals
- **Reproducible Results**: Deterministic patterns for consistent evaluation
- **Scalable Evaluation**: Adjustable batch sizes for thoroughness vs computational cost

#### **2. Scientific Rigor**

- **Seen vs Unseen**: Clear distinction between training and generalization testing
- **Recovery Analysis**: Both permanent and reversible damage for different recovery scenarios
- **Targeted Damage**: Greedy modes focus on most critical circuit components
- **Multi-Damage Testing**: Cumulative damage effects and adaptation capabilities

#### **3. Implementation Quality**

- **Clean Architecture**: Modular design with clear separation of concerns
- **Comprehensive Testing**: All modes validated with extensive test coverage
- **Documentation**: Complete parameter documentation and usage examples
- **Maintainability**: Backward compatibility ensures existing experiments continue working

### **Future Enhancements**

Reversible bits flips instead of bias-based perturbation in reversible mode
Feature modularity: eg allow multi-damage accumulation in permanent mode, both training and eval trajectories
