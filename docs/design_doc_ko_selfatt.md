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

### 6. Knockout Mechanisms
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

## Testing Strategy

we are creating a testing suite to get a comprehensive understanding of each component that could affect the knockout experiment results logged to wandb. This means we have to start from the fundamentals and work our way through the codebase. The detailed tests are broken down below.

**NOTE**: All results from each testing stage should be documented in `tests/testing_suite_results.md` for systematic tracking and validation.

### Ground-Up Component Isolation Testing
All testing scripts should be in tests/

The tests should reflect the functions as they are used in the actual codebase / training and eval run. 

IMPORTANT: For each testing script, the functions that are being tested might have dependencies on other parts of the codebase. So we need to read the files that are being imported into the test scripts as well to get a clear understanding of the methods. Before writing any test script, please identify the substructure of the codebase it represents and how the scripts and functions fit together. Then, see how the specific implementation can be tested according to our plan by consulting with me on this structure and proposed tests. 

IMPORTANT: CONSULT WITH ME, do not implement unless I say so. I want part of the report to give me an overview of the structures being tested, as well as the rationale behind each test.

#### Level 1: Foundational Components (Boolean Circuit Core)

**1.1 Boolean Circuit Operations**
- **`circuits/model.py`** basic functions:
  - Test `make_nops` produces valid no-op gates
  - Test `run_layer` executes boolean operations correctly
  - Test `gen_wires` creates proper connectivity
  - Test `gen_circuit` produces valid circuit structures
  - Test `run_circuit` executes full circuit evaluation

**1.2 Loss Function Infrastructure** ⭐ **CRITICAL**
- **`circuits/train.py`** fundamental metrics:
  - Test `loss_f_l4` computes L4 norm correctly
  - Test `loss_f_bce` computes binary cross-entropy accurately
  - Test `compute_accuracy` measures circuit performance
  - Test `binary_cross_entropy` handles edge cases
  - Verify pre-compiled gradient functions work correctly
  - Test loss functions with identical vs different circuit outputs

#### Level 2: Representation Layer

**2.1 Graph Construction**
- **`utils/graph_builder.py`** validation:
  - Test `build_graph` creates valid Jraph graphs
  - Verify node features (logits, hidden states, positional encodings)
  - Test connectivity patterns match circuit structure
  - Validate graph dimensions and shapes

**2.2 Graph-Circuit Conversion** ⭐ **CRITICAL**
- **`utils/extraction.py`** round-trip validation:
  - Test `extract_logits_from_graph` retrieves correct values
  - Test `update_output_node_loss` modifies graphs properly
  - Test `get_output_node_indices` identifies outputs correctly
  - Verify circuit → graph → circuit round-trip preserves functionality
  - Test extraction with modified vs unmodified graphs

#### Level 3: Knockout Pattern Infrastructure

**3.1 Pattern Generation**
- **`training/pool/structural_perturbation.py`** isolation:
  - Test `create_reproducible_knockout_pattern` with fixed seeds
  - Test `create_knockout_vocabulary` produces diverse patterns
  - Verify reproducibility across calls with same seed
  - Test IN-dist vs OUT-dist pattern differentiation
  - Validate pattern format and structure


**3.2 Pattern Application**
- Test knockout patterns correctly modify graph structures
- Verify attention masks prevent message passing to knocked-out nodes
- Test pattern persistence through optimization steps
- Validate pattern application doesn't corrupt non-target nodes

#### Level 4: Self-Attention Mechanics

**4.1 Attention Masking**
- **`models/self_attention.py`** core functionality:
  - Test attention masks correctly zero out knocked-out connections
  - Test `CircuitSelfAttention` respects knockout patterns
  - Test `SelfAttentionLayer` propagates masks correctly
  - Verify scan-based optimization handles masked attention
  - Verify patterns reach the model's attention mechanism
  - Validate attention masking actually prevents updates

**4.2 Model State Updates**
- Test knocked-out nodes remain unchanged during optimization
- Test non-knocked-out nodes update normally
- Verify gradient flow is properly blocked for masked nodes
- Test model convergence with vs without knockouts
- CRITICAL: revisit 2.2 - GRAPH CONVERSION in conjuction with knockouts to investigate interaction between these 2 components. Use same knockout functionality as it is implemented in the actual training run.

#### Level 5: Pool Management System

**5.1 Circuit Population Management**
- **`training/pool/pool.py`** `GraphPool` validation:
  - Test circuit sampling from pool
  - Test pool updating with new circuits
  - Test reset strategies (uniform, loss-biased, steps-biased)
  - Verify knockout pattern integration in pool operations

**5.2 Pool State Consistency**
- Test pool maintains valid graph structures
- Test knockout patterns persist correctly across pool operations
- Verify pool sampling respects knockout constraints
- Test pool reset preserves knockout pattern assignments

#### Level 6: Evaluation Pipeline ⭐ **CRITICAL**

**6.1 Knockout Evaluation Core**
- **`training/evaluation.py`** critical path testing:
  - Test `evaluate_model_stepwise_batched` with knockout patterns
  - Test `get_loss_and_update_graph` handles knockouts correctly
  - Test `evaluate_circuits_in_chunks` processes batches properly
  - Verify evaluation metrics reflect knockout impact
  - **ADDENDUM: Evaluation Function Chain Tests**
  - Test knockout implementation as part of the evaluation, where eval consists of multiple functions itself
  - Test knockout patterns flow through evaluation pipeline

**6.2 Evaluation Data Flow**
- Test knockout patterns flow from generation → application → evaluation
- Test IN-dist vs OUT-dist evaluation produces different metrics
- Verify evaluation doesn't modify original circuits
- Test batched vs individual circuit evaluation consistency

#### Level 7: Training Integration

**7.1 Training Loop Integration**
- **`training/train_loop.py`** with knockout evaluation:
  - Test periodic knockout evaluation triggers correctly
  - Test wandb logging receives knockout metrics
  - Test training continues normally after knockout evaluation
  - Verify knockout evaluation doesn't interfere with training state

**7.2 Configuration & Orchestration**
- **`train.py`** end-to-end validation:
  - Test Hydra configuration loads knockout parameters
  - Test experiment orchestration with knockout evaluation enabled
  - Verify all components integrate without configuration conflicts
  - **ADDENDUM: Parameter Flow Tests**
    - Test parameter passing from configs/config.yaml through the chain of training and eval
  - **MISC A: Configuration & Setup Tests**
    - Verify knockout evaluation is properly enabled
    - Validate configuration parameter flow
    - Check periodic evaluation timing

#### Level 8: End-to-End Validation

**8.05 Model behaviour tests**
Model Behavior Tests
Test identical circuits with/without knockouts show different performance
Verify knocked-out nodes don't update during forward pass
Check attention masks are correctly computed and applied

**8.1 Minimal Working Examples**
End-to-end test of knockout evaluation pipeline
Validate wandb logging receives different values for IN/OUT evaluation
Test with minimal examples to isolate behavior (Create minimal circuits with known knockout behavior)
- Test complete pipeline: generation → training → knockout evaluation
- Verify knockout evaluation produces expected different metrics
- Test reproducibility across runs with same configuration

**8.2 Integration Stress Tests**
- Test with large circuit populations
- Test with complex knockout pattern vocabularies
- Test memory and performance under knockout evaluation load
- Verify system stability under extended knockout evaluation runs

#### LEVEL 9
***Reflection on newly-found knowledge**
Review the updated test result markdown file, review all of the tests. Is there any critical test we missed? Did we achieve our goal of an airtight deconstruction of the entire knockout program structure to make sure that knockout patterns are created, transferred to the graph, maintained and evaluated, as well as differentiated between in-distribution and OOD?

### Key Testing Principles

**Isolation**: Each level can be tested independently of higher levels
**Dependency Validation**: Higher levels assume lower levels work correctly
**Critical Path Focus**: Special emphasis on loss functions, extraction, and evaluation pipeline
**JAX Compatibility**: All tests verify proper handling of JAX functional programming patterns
**Reproducibility**: All tests use explicit random keys for deterministic behavior

This ground-up structure ensures that if a test fails at any level, the issue is isolated to that specific component without requiring investigation of dependent systems.

## JAX Fundamentals

- **Functional Programming**: All knockout functions are pure with explicit PRNG keys for reproducibility
- **vmap**: Batch processing of knockout pattern generation and model application
- **JIT Compilation**: Attention masking and model updates are JIT compiled for performance
- **Immutable Data**: GraphsTuple replacement pattern maintains functional paradigm
