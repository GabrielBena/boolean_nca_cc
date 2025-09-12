## Testing Strategy

we are creating a testing suite to get a comprehensive understanding of each component that could affect the knockout experiment results logged to wandb. This means we have to start from the fundamentals and work our way through the codebase. The detailed tests are broken down below.

**NOTE**: All results from each testing stage should be documented in `tests/testing_suite_results.md` for systematic tracking and validation.

### Ground-Up Component Isolation Testing

All testing scripts should be in tests/

The tests should reflect the functions as they are used in the actual codebase / training and eval run. THis also means that each test script should import from config.yaml, if it uses parameters defined there.

IMPORTANT: For each testing script, the functions that are being tested might have dependencies on other parts of the codebase. So we need to read the files that are being imported into the test scripts as well to get a clear understanding of the methods. Before writing any test script, please identify the substructure of the codebase it represents and how the scripts and functions fit together. Read the exact functions that are being imported, do not make assumptions. Then, see how the specific implementation can be tested according to our plan by consulting with me on this structure and proposed tests.

IMPORTANT: To re-iterate: The design of our test scripts should reflect the functions AS THEY ARE USED in the overall run. This means not only do we hae to import the correct functional dependencies for the functions we are testing, we also have to check where the functions we are testing are being used in the actual program to mirror this use with comprehensive testing.

MOST IMPORTANT!!!!!: To the first order of action should be to point me to all the relevant locations in boolean_nca_cc/training/train_loop.py. These locations should then point to other places in the codebase where the functions are defined. Please also point me to those. DON'T JUST IMPLEMENT. FIRST STEP IS TO HELP ME PINPOINT LOCATIONS.

Example:
the results section shows that there was an imprtant change in 'required test fix'. Please look exactly at how this test was structured. This is important because it should reflect how run_self_attention_scan is used in the actual training run. However, when looking at the actual train_loop.py, run_self_attention_scan was never called. Instead, loss_fn_no_scan was used. The test script should ONLY test this function, and design should be DIRECTLY informed by the actual train loop.

Either:

- The test doesn't reflect the actual codebase, only the test is wrong and therefore providing nothing of value.
- The codebase itself is wrong, and the test correctly reflects this error in our code, thereby providing immense value.

This means that if during the test script run if there is an error, your first goal should be to look at the appropriate location in the actual codebase to find the corresponding implementation and mirror.

IMPORTANT: CONSULT WITH ME, do not implement unless I say so. I want part of the report to give me an overview of the structures being tested, as well as the rationale behind each test.

**General script instructions**
Always 'conda activate metabool' before running
Always check configs/config.yaml to see which parameter settings are ctually used and mirror in the test script. For model-specific configs, check configs/model/self_attention.yaml

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

- **`training/pool/perturbation.py`** isolation:
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
  - Validate that knocked out nodes do not get upddated in any way, which includes message passing, residual updates or any other update

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

#### Level 6: Evaluation Pipeline ⭐ **YOU ARE HERE**

**6.1 Knockout Evaluation Core**

- **`training/evaluation.py`** critical path testing:
  - Test `evaluate_model_stepwise_batched` with knockout patterns
  - Test `get_loss_and_update_graph` handles knockouts correctly
  - Test `evaluate_circuits_in_chunks` processes batches properly
  - Verify evaluation metrics reflect knockout impact
  - **ADDENDUM: Evaluation Function Chain Tests**
  - Test knockout implementation as part of the evaluation, where eval consists of multiple functions itself
  - Test knockout patterns flow through evaluation pipeline

Notes:

1. Critical Bugs Are at Interfaces: The most likely failure modes (knockout patterns not flowing through, attention masking not working, graphs not updating correctly) are integration issues between components.
2. Tight Coupling by Design: get_loss_and_update_graph is specifically designed as a pipeline: extract_logits → compute_loss → update_graph. Testing these in isolation misses the critical data transformations.
3. Mirror Actual Usage: The training loop calls these functions in specific combinations - that's what we need to verify works.
4. Knockout Flow is End-to-End: Knockout patterns must flow correctly through: pattern generation → graph building → attention masking → evaluation → metrics. This is inherently an integration test.

Proposed Testing Strategy:

## Phase 1: Critical Path Integration Tests

Test 1: Core Training Step Integration* Mirror exact usage from loss_fn_no_scan in training loop

* build_graph → model(graph, knockout_pattern) → get_loss_and_update_graph
* Verify knockout patterns prevent node updates throughout entire chain
* Note: Once the attention mask is created with the knockout pattern incorporated, it should work as usual via attention masking

Test 2: Knockout Evaluation Pipeline Integration* Mirror exact usage from run_knockout_periodic_evaluation

* vocabulary sampling → evaluate_circuits_in_chunks → evaluate_model_stepwise_batched
* Verify IN-dist vs OUT-dist produce different metrics end-to-end

Test 3: Batch Processing Integration* Mirror exact usage from evaluate_model_stepwise_batched

* batch_graphs → vmap(model) → vmap(get_loss_and_update_graph)
* Verify knockout patterns work correctly in batched operations

## Phase 2: Targeted Isolation Tests

Test 4: get_loss_and_update_graph Edge Cases* Test with malformed graphs, edge case loss values, empty knockouts

* Verify error handling and boundary conditions

Test 5: Chunking Logic Isolation* Test evaluate_circuits_in_chunks with various batch sizes, empty chunks

* Verify averaging logic works correctly

This approach gives us:1. High confidence that the critical paths work as intended

1. Fast failure detection when interfaces break
2. Realistic test conditions that match actual usage
3. Complete coverage through subsequent isolation tests

**6.2 Evaluation Data Flow**

- Test knockout patterns flow from generation → application → evaluation
- Test IN-dist vs OUT-dist evaluation produces different metrics
- Verify evaluation doesn't modify original circuits
- Test batched vs individual circuit evaluation consistency

#### Level 7: Training Integration

**7.1 Configuration & Orchestration**

- **`train.py`** end-to-end validation:
  - Test Hydra configuration loads knockout parameters
  - Test parameter passing from configs/config.yaml through the chain of training and eval
  - Test experiment orchestration with knockout evaluation enabled
  - Verify all components integrate without configuration conflicts
  -

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
Has our goal been met to verify: circuit construction -> knockout pattern generation -> self-attention mask creation with knockouts -> transfer knocked out graph back to circuit to ensure circuit wasn't affected unexpecetdly -> ???

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
