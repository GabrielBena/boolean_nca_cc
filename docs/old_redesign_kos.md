The overview of this codebase is in docs/design_doc_ko_selfatt.md

Once the testing suite is concluded, we need to rework how knockouts are introduced into the training flow / pool. Currently, knockouts are introduced as part of the periodic pool reset, therefore knocked out circuits will always start as NOPs. This means that the network will only ever learn to optimise knocked out circuits starting from NOPs.

An alternative design is to instead periodically sample any given circuit present in the pool, apply a knockout mask to it in the same way that currently fresh circuits are knocked out, and to then proceed as usual with pool optimisation. This means that knockouts are introduced randomly to any kinds of circuits along their optimisation trajctory, simulating damage during lifetime.

The question remains as to what to do with the gates that have been removed from the circuits that are knocked out, as their LUTs will be partially optimised and functional. Our options are:

- stuck-at-faults: reset their LUT fields to all 0 after the gate is knocked out. SInce it is also removed from message passing (masked attention), the fields will remain at 0 and not be updated. Perhaps all of the hiddenchannels could be zeroed, but this is a variable design choice to vary and compare.
- gate-removal: If a node is removed from the self-attention process via the attention mask (as is currently implemented), the corresponding gate should also be removed from the circuit, thereby no longer participating in circuit function. Perhaps we could look at how the gate_mask in boolean_nca_cc/circuits/model.py works and leverage this. Or there could be another way of removing gates form the circutis themselves

For deciding which of the above to implement, we must weigh which one requires less invasive changes to the existing program, but also which one seems more feasible to be learned by the network. introducing knockouts during pool optimisation could pose a significant challenge to the learning process.

In any case, the actual evaluation we care about is on circuits that were not knocked out (their LUTs are configured for good hard_accuracy), then the circuit with those already configured LUTs is knocked out, then the self-attention GNN has to reconfigure the remaining gates to recover the circuit function. I believe that this is already done if we implement the above knockout design choices, since at some point the opol will be saturated with circuits that are fully functinoal, so the sampling-based approach will provide examples of knockouts on functional circuits. Then we just have to match the eval function to not create a fresh circuit using gen_circuit, applying knockout and evaluating on N message steps, but to instead sample any circuit from the pool, apply the knock out, and then evaluate after N message passing steps. Therefore, the only difference is the source of circuit for evaluation.

Perhaps we first have to make a decision between the following:
A. First implement the gate removal / stuck at faults to the circuits, before changing the overall strategy from reset-based to sampling-based. This would mean making the network learn to optimise from fresh NOPs, but with actual circuit damage
- In the case of gate removal, maybe we have to rethink our strategy. If we are implementing gate removal at a circuit level, let's see how graph_builder.py creates a graph FROM the circuit itself. Then the downstream structure would be: gen_circuit (now knocked out circuit) -> graph_builder (no change necessary as it is built from a damaged circuit automatically) -> self-attention mask (can be reverted to not include a damage update, as it will simply be generated from the already damaged graph upstream).
B. Implement the gate removal alongside the sampling strategy. It makes little sense to implement only the sampling strategy without affecting the gates themselves, as this will leave them in a partially optimised state when their circuit is sampled for knockout.

Given all the considerations above, we need to plan a course of iterative changes to be made and tested at each stage.

In the case where we first do A. and then the sampling strategy: If gates are modified at the circuit generation level, with graphs and attention masks automatically generated from the damaged circuit, we have to consider what this means for the sampling strategy. Because the circuit up to that point in the pool had a certain structure, which would be modified by gate removal. How to best work with JAX immutability? Is there a padding strategy or something else we can do?

# TEST (DO NOT IMPLEMENT BEFORE WE ARE DONE WITH ABOVE)

Simplest test for first change: Look at where the gate removal is applied in the train loop, how it interacts with the attention mask perturbation and look at the N-message passing loop. This loop is critical: In theory, if we remove a gate and decouple the node from the mask, we can track the nodes state throughout the loop. The state should not change. If the gate removal is implemented in such a way that the node is gone from the graph altogether, then I guess ...

---

# STRATEGIC ANALYSIS: INCREMENTAL IMPLEMENTATION PATHS

## Goal Clarification

**Final Goal**: Remove gates entirely from sampled circuits in the pool, enabling the network to learn recovery from realistic mid-optimization damage rather than just NOP-based damage. Since the pool will be eventually saturated with optimised circuits, the model will also learn recovery from damage to fully functional circuits during this late training stage, as they will be sampled and damaged at this point.

## Implementation Path Analysis

### Path A: Damage Type First, Then Sampling Strategy

**Advantages**:

- Incremental and testable at each stage
- Can validate gate removal mechanism before introducing sampling complexity
- Maintains current reset-based flow initially
- JAX immutability challenges are deferred until sampling phase

**Disadvantages**:

- Still trains on damaged NOPs initially (but with proper gate removal)
- Requires two separate implementation phases

### Path B: Simultaneous Change (Damage Type + Sampling)

**Advantages**:

- Reaches final goal faster
- Avoids intermediate state that still has some limitations

**Disadvantages**:

- More complex to implement and debug
- Harder to isolate issues (gate removal vs sampling vs their interaction)
- JAX immutability challenges must be solved upfront

## Recommended Path: A (Incremental Implementation)

### Phase 1: Implement Gate Removal with Reset-Based Knockouts

#### Step A1: Circuit-Level Gate Removal Implementation

**Location**: `boolean_nca_cc/circuits/model.py`
**Functions to Modify**:

- `gen_circuit()` - Add post-generation gate removal capability
- `run_circuit()` - Leverage existing `gate_mask` parameter for gate removal
- New function: `apply_gate_removal(wires, logits, knockout_pattern)`

**Key Changes**:

```python
# In gen_circuit() - add optional knockout application
def gen_circuit(key, layer_sizes, arity=4, knockout_pattern=None, ...):
    # Generate circuit normally
    wires, logits = standard_generation(...)
  
    # Apply gate removal if pattern provided
    if knockout_pattern is not None:
        wires, logits = apply_gate_removal(wires, logits, knockout_pattern)
  
    return wires, logits

# New function for gate removal
def apply_gate_removal(wires, logits, knockout_pattern):
    # Option 1: Stuck-at-faults - zero out LUT fields
    # Option 2: True gate removal - modify circuit structure
    # Return modified wires, logits
```

#### Step A2: Graph Builder Adaptation

**Location**: `boolean_nca_cc/utils/graph_builder.py`
**Function to Verify**: `build_graph()`

**Analysis**: If gate removal is implemented at circuit level, `build_graph()` should automatically handle the damaged circuit correctly without modification, since it builds graphs from the provided `logits` and `wires`.

#### Step A3: Attention Mask Strategy Update

**Location**: `boolean_nca_cc/models/self_attention.py`
**Function to Modify**: `CircuitSelfAttention.__call__()`

**Decision Point**:

- **Option 1**: Remove attention mask damage entirely since circuit is pre-damaged
- **Option 2**: Keep attention mask as additional protection layer
- **Recommendation**: Start with Option 1 for cleaner separation of concerns

#### Step A4: Pool Reset Integration

**Location**: `boolean_nca_cc/training/pool/pool.py`
**Function to Modify**: `initialize_graph_pool()`

**Changes**:

```python
# Modify to generate pre-damaged circuits
damaged_pool = initialize_graph_pool(
    # ... existing parameters ...
    apply_gate_removal=True,  # New parameter
    knockout_patterns=sampled_knockout_patterns
)
```

### Phase 2: Transition to Sampling-Based Knockouts

#### Step B1: Pool Sampling Enhancement

**Location**: `boolean_nca_cc/training/pool/pool.py`
**Function to Modify**: `GraphPool.sample()`

**New Functionality**:

```python
def sample_for_knockout_application(self, key, batch_size, min_optimization_steps=50):
    # Identify functional circuits (update_steps > threshold)
    functional_mask = self.graphs.globals[:, 1] > min_optimization_steps
    functional_indices = jp.where(functional_mask)[0]
  
    # Sample from functional circuits
    # Apply gate removal to sampled circuits
    # Return damaged versions
```

#### Step B2: JAX Immutability Strategy

**Challenge**: Pool contains circuits with fixed structure, but gate removal changes structure.

**Solutions**:

Note: These 3 assumptions and propositions are outdated, refer to Thinking Ahead... below.


1. **Padding Strategy**: Pre-allocate maximum circuit size, use masking for removed gates
2. **Separate Pool**: Maintain separate pool of damaged circuits
3. **On-the-fly Generation**: Generate damaged variants during sampling without storing

**Recommendation**: Start with Solution 3 (on-the-fly) for simplicity.

#### Step B3: Training Loop Integration

**Location**: `boolean_nca_cc/training/train_loop.py`
**Function to Modify**: Main training loop

**Changes**:

```python
# Replace current pool sampling with knockout-enhanced sampling
if enable_runtime_knockouts:
    idxs, graphs, wires, logits, knockout_patterns = circuit_pool.sample_and_damage(
        sample_key, meta_batch_size, knockout_config
    )
else:
    # Fallback to current behavior
    idxs, graphs, wires, logits, knockout_patterns = circuit_pool.sample(
        sample_key, meta_batch_size
    )
```

### Phase 3: Evaluation Pipeline Update

#### Step C1: Evaluation Circuit Source

**Location**: `boolean_nca_cc/training/train_loop.py`
**Function to Modify**: `run_knockout_periodic_evaluation()`

**Changes**:

```python
# Replace: base_circuit = gen_circuit(wiring_fixed_key, layer_sizes, arity=arity)
# With: base_circuit = circuit_pool.sample_functional_circuit_for_eval(eval_key)
```

## Implementation Timeline & Testing Strategy

### Phase 1 Testing:

1. **Unit Tests**: Verify gate removal produces expected circuit modifications
2. **Integration Tests**: Ensure graph building works with damaged circuits
3. **Training Tests**: Confirm reset-based training still works with gate removal

### Phase 2 Testing:

1. **Sampling Tests**: Verify functional circuit identification and sampling
2. **Damage Application Tests**: Ensure on-the-fly gate removal works correctly
3. **Performance Tests**: Check that sampling doesn't significantly slow training

### Phase 3 Testing:

1. **Evaluation Consistency**: Compare pool-sampled vs generated circuit evaluation
2. **End-to-End Validation**: Verify complete pipeline produces meaningful knockout training

## Key Technical Decisions to Resolve

1. **Gate Removal Method**: Stuck-at-faults vs true structural removal
2. **Attention Mask Strategy**: Remove entirely or keep as backup
3. **JAX Immutability**: Padding vs separate pools vs on-the-fly generation
4. **Functional Circuit Threshold**: How to identify "optimized enough" circuits

## Risk Mitigation

- **Gradual Rollout**: Each phase can be feature-flagged and compared against baseline
- **Fallback Mechanisms**: Maintain ability to revert to current behavior
- **Comprehensive Testing**: Each phase has clear success criteria before proceeding
- **Performance Monitoring**: Ensure changes don't significantly impact training speed

---

# THINKING AHEAD: JAX-Compatible Strategies for Sample → Damage → Reinsert

Based on our analysis of JAX compilation boundaries and structural constraints, we need to choose between two viable approaches for implementing the "sample existing functional circuits → damage → reinsert to pool" workflow.

## Option 1: Padding Strategy (Uniform Structure Maintenance)

### Core Concept

Maintain identical circuit structure across all pool entries, but use masking to functionally "remove" gates without changing the underlying data structure.

### Technical Implementation

#### Circuit Structure

```python
# All circuits maintain maximum possible structure
class PaddedCircuit:
    wires: List[jp.ndarray]      # Full connectivity, never changes
    logits: List[jp.ndarray]     # Full LUT tables, never changes  
    gate_mask: jp.ndarray        # True = active, False = "removed"
    knockout_pattern: jp.ndarray # Original knockout specification

def apply_gate_removal_padding(circuit, knockout_pattern):
    """Apply gate removal via masking, preserving structure"""
    # Don't modify wires or logits - just update masks
    gate_mask = ~knockout_pattern  # False for "removed" gates
  
    return circuit._replace(
        gate_mask=gate_mask,
        knockout_pattern=knockout_pattern
    )
```

#### Pool Integration

```python
# Enhanced GraphPool with uniform structure
class GraphPool(struct.PyTreeNode):
    graphs: jraph.GraphsTuple    # All same structure [pool_size, max_nodes, features]
    wires: PyTree               # All same structure [pool_size, max_wires, ...]
    logits: PyTree              # All same structure [pool_size, max_gates, max_lut_size]
    gate_masks: jp.ndarray      # [pool_size, max_nodes] - True = active
    knockout_patterns: jp.ndarray # [pool_size, max_nodes] - True = knocked out
  
    def sample_and_damage_functional(self, key, batch_size, min_steps=50):
        """Sample functional circuits and apply damage via masking"""
        # Identify functional circuits
        functional_mask = self.graphs.globals[:, 1] > min_steps
        functional_idxs = jp.where(functional_mask)[0]
      
        if len(functional_idxs) == 0:
            # Fallback to normal sampling if no functional circuits
            return self.sample(key, batch_size)
      
        # Sample from functional circuits
        selected_idxs = jax.random.choice(key, functional_idxs, shape=(batch_size,))
      
        # Generate new knockout patterns
        damage_key = jax.random.fold_in(key, 1)
        new_patterns = generate_knockout_patterns(damage_key, batch_size)
      
        # Apply damage via masking (structure preserved)
        damaged_masks = ~new_patterns
      
        # Update pool with new masks (structure unchanged)
        updated_gate_masks = self.gate_masks.at[selected_idxs].set(damaged_masks)
        updated_knockout_patterns = self.knockout_patterns.at[selected_idxs].set(new_patterns)
      
        updated_pool = self.replace(
            gate_masks=updated_gate_masks,
            knockout_patterns=updated_knockout_patterns
        )
      
        return selected_idxs, updated_pool
```

#### Circuit Execution Integration

```python
# Modify run_circuit to respect gate_mask
def run_circuit(logits, wires, x, gate_mask=None, knockout_pattern=None, hard=False):
    """Enhanced to handle both gate_mask and knockout_pattern"""
    # Combine both masking mechanisms
    if gate_mask is None:
        gate_mask = [jp.ones_like(x)]
        for lgt in logits:
            gate_mask.append(jp.ones(lgt.shape[0] * lgt.shape[1]))
  
    # Apply knockout pattern if provided (attention masking)
    attention_mask = None
    if knockout_pattern is not None:
        attention_mask = create_attention_mask(knockout_pattern)
  
    # Apply input mask
    x = x * gate_mask[0]
    acts = [x]
  
    for ws, lgt, mask in zip(wires, logits, gate_mask[1:]):
        luts = jax.nn.sigmoid(lgt)
        if hard:
            luts = jp.round(luts)
      
        # Gate-level masking: zero out LUTs for removed gates
        luts = luts * mask[..., None]  # Broadcast mask over LUT dimension
      
        x = run_layer(luts, [x[..., w] for w in ws]) * mask
        acts.append(x)
  
    return acts
```

### Advantages

- ✅ **JAX Compatible**: No structural changes during training
- ✅ **Flexible Timing**: Can apply damage anytime outside JIT
- ✅ **Incremental**: Easy to implement step by step
- ✅ **Testable**: Clear separation between structure and function

### Disadvantages

- ❌ **Memory Overhead**: All circuits store maximum structure
- ❌ **Computational Waste**: Processes "dead" gates
- ❌ **Complexity**: Dual masking system (gate_mask + knockout_pattern)

---

## Option 2: Reset Timing Strategy (Structural Evolution)

### Core Concept

Allow circuit structure to change between training epochs, but only during reset windows when JAX compilation boundaries permit structural modifications.

### Technical Implementation

#### Circuit Structure Evolution

```python
# Circuits can have different structures across reset cycles
def apply_structural_gate_removal(wires, logits, knockout_pattern):
    """Actually remove gates from circuit structure"""
    active_gates = ~knockout_pattern
  
    # Remove knocked-out gates from logits
    filtered_logits = []
    for layer_logits in logits:
        layer_active = active_gates[get_layer_indices(layer)]
        filtered_logits.append(layer_logits[layer_active])
  
    # Update wiring to skip removed gates
    filtered_wires = update_wires_for_removed_gates(wires, active_gates)
  
    return filtered_wires, filtered_logits

def intelligent_reset_strategy(pool, epoch, reset_key):
    """Reset strategy that evolves from NOPs to functional damage"""
  
    if epoch < FUNCTIONAL_THRESHOLD:
        # Phase 1: Traditional reset with damaged NOPs
        return reset_with_damaged_fresh_circuits(pool, reset_key)
    else:
        # Phase 2: Reset with structurally damaged functional circuits
        return reset_with_damaged_functional_circuits(pool, reset_key)

def reset_with_damaged_functional_circuits(pool, reset_key):
    """Sample existing functional circuits, damage them, use for reset"""
  
    # Sample high-performing circuits from current pool
    performance_threshold = jp.percentile(pool.graphs.globals[:, 1], 75)
    functional_candidates = pool.graphs.globals[:, 1] > performance_threshold
    functional_idxs = jp.where(functional_candidates)[0]
  
    if len(functional_idxs) == 0:
        # Fallback to traditional reset if no functional circuits
        return reset_with_damaged_fresh_circuits(pool, reset_key)
  
    # Extract functional circuits for damage
    num_to_reset = max(1, round(pool.size * RESET_FRACTION))
    sample_idxs = jax.random.choice(
        reset_key, functional_idxs, shape=(num_to_reset,), replace=True
    )
  
    # Extract circuit data
    functional_circuits = extract_circuits_at_indices(pool, sample_idxs)
  
    # Generate knockout patterns and apply structural damage
    damage_key = jax.random.fold_in(reset_key, 42)
    knockout_patterns = generate_knockout_patterns(damage_key, num_to_reset)
  
    damaged_circuits = []
    for circuit, pattern in zip(functional_circuits, knockout_patterns):
        damaged_wires, damaged_logits = apply_structural_gate_removal(
            circuit.wires, circuit.logits, pattern
        )
        damaged_circuits.append((damaged_wires, damaged_logits, pattern))
  
    # Create reset pool from damaged functional circuits
    reset_pool = initialize_graph_pool_from_damaged_circuits(
        damaged_circuits, layer_sizes, input_n, arity, circuit_hidden_dim
    )
  
    # Perform reset using damaged functional circuits
    return pool.reset_fraction_with_custom_pool(reset_pool)
```

#### Pool Reset Integration

```python
def initialize_graph_pool_from_damaged_circuits(
    damaged_circuits, layer_sizes, input_n, arity, circuit_hidden_dim
):
    """Create pool from pre-damaged circuits with potentially different structures"""
  
    all_graphs = []
    all_wires = []
    all_logits = []
    all_patterns = []
  
    for damaged_wires, damaged_logits, pattern in damaged_circuits:
        # Build graph from damaged circuit
        graph = build_graph(
            logits=damaged_logits,
            wires=damaged_wires,
            input_n=input_n,
            arity=arity,
            circuit_hidden_dim=circuit_hidden_dim,
            loss_value=0.0,
            update_steps=0,
        )
      
        all_graphs.append(graph)
        all_wires.append(damaged_wires)
        all_logits.append(damaged_logits)
        all_patterns.append(pattern)
  
    # Batch all circuits (they should have consistent structure within this reset batch)
    batched_graphs = jraph.batch(all_graphs)
    batched_wires = jax.tree.map(lambda *x: jp.stack(x), *all_wires)
    batched_logits = jax.tree.map(lambda *x: jp.stack(x), *all_logits)
    batched_patterns = jp.stack(all_patterns)
  
    return GraphPool.create(
        batched_graphs, batched_wires, batched_logits, 
        reset_counter=jp.zeros(len(damaged_circuits)), 
        knockout_patterns=batched_patterns
    )
```

#### Training Loop Integration

```python
def enhanced_training_loop_with_functional_damage():
    """Training loop with evolving damage strategy"""
  
    for epoch in range(epochs):
        # Check if we should apply functional damage via reset
        if should_reset_pool(epoch, reset_pool_interval, last_reset_epoch):
            # Evolution: Early epochs damage NOPs, later epochs damage functional circuits
            circuit_pool = intelligent_reset_strategy(circuit_pool, epoch, reset_key)
            last_reset_epoch = epoch
      
        # Normal training on current pool structure
        idxs, graphs, wires, logits, knockout_patterns = circuit_pool.sample(
            sample_key, meta_batch_size
        )
      
        # JIT compiled training step (structure is uniform within this epoch)
        loss, (aux, circuit_pool, loss_steps) = pool_train_step(
            model, optimizer, circuit_pool, idxs, graphs, wires, logits,
            x_data, y_data, tuple(layer_sizes), n_message_steps,
            loss_type, loss_key, epoch, knockout_patterns
        )
```

### Advantages

- ✅ **True Gate Removal**: Actually removes gates from circuits
- ✅ **Memory Efficient**: No padding overhead
- ✅ **Conceptually Clean**: Matches real-world damage scenarios
- ✅ **JAX Compatible**: Structural changes only during allowed windows

### Disadvantages

- ❌ **Limited Timing**: Can only damage during reset windows
- ❌ **Implementation Complexity**: Need to handle structure evolution
- ❌ **Testing Complexity**: Must validate across structural changes

---

## Recommendation: Start with Option 2, Fallback to Option 1

### Phase 1: Implement Option 2 (Reset Timing)

**Rationale**:

- Achieves the core goal most directly
- True gate removal matches real-world scenarios
- Natural evolution from NOPs → functional damage

### Phase 2: Add Option 1 (Padding) as Enhancement

**Rationale**:

- Provides more flexible timing if needed
- Can coexist with Option 2 for different damage scenarios
- Useful for fine-grained damage experiments

### Implementation Priority

1. **First**: Implement `intelligent_reset_strategy()` in reset timing
2. **Second**: Add structural gate removal functions
3. **Third**: Test evolution from NOP damage to functional damage
4. **Fourth**: Consider padding strategy for additional flexibility

This approach gives us the most direct path to the goal while maintaining fallback options and incremental testability.
