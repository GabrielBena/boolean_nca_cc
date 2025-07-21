# Knockout System Redesign: From Reset-Based to Sampling-Based Damage

## Overview

This document outlines the redesign of knockout mechanisms from reset-based damage (knockouts applied only to fresh NOPs) to sampling-based damage (knockouts applied to existing functional circuits during optimization). The goal is to enable the network to learn recovery from realistic mid-optimization damage rather than just NOP-based damage.

## Current System Analysis

### Existing Infrastructure

- **Gate Masking**: `run_circuit()` supports `gate_mask` parameter for enabling/disabling gates
- **Attention Masking**: `CircuitSelfAttention._create_attention_mask()` applies knockout patterns to attention mechanisms
- **Pool Management**: `GraphPool` stores `knockout_patterns` and applies them during initialization
- **Reset Integration**: Knockouts currently applied only during pool reset operations

### Current Limitations

- Knockouts only applied to fresh NOPs during reset
- Network never learns to recover functional circuits from damage
- Missed opportunity for realistic damage scenarios during optimization trajectory
- Attention masking only prevents message passing, but knocked-out nodes retain their existing feature values
- No automatic translation of attention-level knockouts to circuit-level execution

## Recommended Approach: Node Feature Zeroing Strategy

### Why Node Feature Zeroing?

1. **System Flow Alignment**: Works with the natural `gen_circuit()` → `build_graph()` → message passing → `extract_logits_from_graph()` → `run_circuit()` flow
2. **Automatic Coordination**: Ensures perfect alignment between attention masking and circuit execution without complex coordination
3. **Simplicity**: Single point of change achieves both attention isolation and circuit-level gate removal
4. **Leverages Existing Infrastructure**: Uses current knockout patterns and attention masking systems
5. **True Gate Removal**: Achieves complete functional isolation through feature zeroing

### Key Insight: Natural Flow Coordination

The existing system already has a perfect coordination mechanism:

```
Circuit LUTs → Graph Node Features → Message Passing Updates → Extracted LUTs → Circuit Execution
```

By zeroing node features for knocked-out gates:

1. **Attention masking** prevents new updates from reaching knocked-out nodes
2. **Feature zeroing** ensures knocked-out nodes have zero values
3. **Extraction** pulls zero logits for knocked-out gates
4. **Circuit execution** automatically gets zero LUTs, achieving complete gate removal

### Current Infrastructure Capabilities

The existing codebase already provides the foundation:

```python
# 1. Attention Isolation (prevents message passing)
knockout_mask = jp.outer(active_nodes_mask, active_nodes_mask)
mask = mask & knockout_mask

# 2. Parameter Update Prevention (prevents learning)
logit_updates = logit_updates * active_mask[:, None]
hidden_updates = hidden_updates * active_mask[:, None]

# 3. Feature Zeroing (NEW - ensures zero LUTs in circuit execution)
current_logits = nodes["logits"] * active_mask[:, None]
current_hidden = nodes["hidden"] * active_mask[:, None]
```

**Complete gate removal is achieved when all three mechanisms work together automatically.**

## Implementation Strategy

**Key Insight**: The implementation follows a simple progression: **feature zeroing enhancement** → **pool initialization consistency** → **sampling-based damage application**. Each phase builds on the previous one and can be tested independently.

### Phase 1: Node Feature Zeroing Enhancement

**Goal**: Enhance the existing attention masking to include node feature zeroing, achieving automatic coordination between attention and circuit execution.

#### Step 1.1: Enhanced Self-Attention Integration

**Target**: `boolean_nca_cc/models/self_attention.py`

```python
def __call__(
    self,
    graph: jraph.GraphsTuple,
    attention_mask: Optional[jp.ndarray] = None,
    knockout_pattern: Optional[jp.ndarray] = None,
) -> jraph.GraphsTuple:
    """Enhanced to support complete gate removal via feature zeroing."""
    # ... existing code for attention processing ...
  
    # Apply knockout pattern for complete gate removal
    if knockout_pattern is not None:
        active_mask = ~knockout_pattern
  
        # ZERO OUT existing node features for knocked-out nodes (NEW)
        current_logits = nodes["logits"] * active_mask[:, None]
        current_hidden = nodes["hidden"] * active_mask[:, None]
  
        # Zero out updates for knocked-out nodes (EXISTING)
        logit_updates = logit_updates * active_mask[:, None]
        hidden_updates = hidden_updates * active_mask[:, None]
  
        # Apply updates to zeroed features for complete isolation
        updated_logits = current_logits + self.logit_scale * logit_updates
        updated_hidden = current_hidden + self.hidden_scale * hidden_updates
    else:
        # No knockouts applied - normal residual updates
        updated_logits = nodes["logits"] + self.logit_scale * logit_updates
        updated_hidden = nodes["hidden"] + self.hidden_scale * hidden_updates
  
    # ... rest of existing code ...
```

#### Step 1.2: Testing and Validation

**Testing Strategy for Phase 1**:

- Verify knocked-out nodes have zero logits after message passing
- Confirm extracted logits are zero for knocked-out gates
- Test that circuit execution receives zero LUTs for knocked-out gates
- Validate that attention masking still prevents message passing to/from knocked-out nodes

### Phase 2: Pool Initialization Consistency

**Goal**: Ensure pool initialization properly zeros node features for circuits with knockout patterns, maintaining consistency throughout the system.

#### Step 2.1: Pool Initialization Feature Zeroing

**Target**: `boolean_nca_cc/training/pool/pool.py`

```python
def initialize_graph_pool(
    rng: jax.random.PRNGKey,
    layer_sizes: List[Tuple[int, int]],
    pool_size: int,
    input_n: int,
    arity: int = 2,
    circuit_hidden_dim: int = 16,
    loss_value: float = 0.0,
    knockout_config: Optional[Dict[str, Any]] = None,
    knockout_patterns: Optional[Array] = None,
) -> GraphPool:
    """Initialize pool with consistent feature zeroing for knocked-out nodes."""
  
    # ... existing pool initialization code ...
  
    # After applying knockout patterns, zero corresponding node features
    if knockout_config or knockout_patterns is not None:
        # Zero out features for knocked-out nodes in initial graphs
        updated_nodes = dict(graphs.nodes)
      
        for i in range(pool_size):
            if jp.any(pool_knockout_patterns[i]):
                active_mask = ~pool_knockout_patterns[i]
              
                # Zero out logits for knocked-out nodes
                updated_nodes["logits"] = updated_nodes["logits"].at[i].set(
                    updated_nodes["logits"][i] * active_mask[:, None]
                )
              
                # Zero out hidden features for knocked-out nodes  
                updated_nodes["hidden"] = updated_nodes["hidden"].at[i].set(
                    updated_nodes["hidden"][i] * active_mask[:, None]
                )
      
        # Update graphs with zeroed features
        graphs = graphs._replace(nodes=updated_nodes)
  
    return GraphPool.create(
        graphs, all_wires, all_logits, reset_counter, pool_knockout_patterns
    )
```

#### Step 2.2: Testing and Validation

**Testing Strategy for Phase 2**:

- Verify pool initialization zeros features for knocked-out nodes
- Confirm consistency between knockout patterns and zeroed features
- Test that pool circuits maintain zero features throughout training
- Validate reset operations preserve knockout consistency

### Phase 3: Sampling-Based Damage Application

**Goal**: Transition from reset-based damage to sampling-based damage, where knockouts are applied to existing functional circuits during training, leveraging the automatic feature zeroing from Phase 1.

#### Step 3.1: Pool Sampling Enhancement

**Target**: `boolean_nca_cc/training/pool/pool.py`

```python
def sample_and_apply_knockouts(
    self, 
    key: Array, 
    batch_size: int,
    knockout_config: Optional[Dict] = None
) -> Tuple[Array, jraph.GraphsTuple, PyTree, PyTree, Array]:
    """
    Sample circuits from pool and apply knockouts with configurable targeting.
  
    Note: Feature zeroing happens automatically in CircuitSelfAttention when
    knockout patterns are applied, so no manual coordination is needed.
  
    Args:
        key: Random key
        batch_size: Size of batch to sample
        knockout_config: Configuration for knockout application
            - 'target_functional': If True, prefer functional circuits for knockouts
            - 'min_steps_threshold': Minimum steps to consider circuit functional
            - 'knockout_probability': Probability of applying knockouts to sampled circuits
  
    Returns:
        Tuple of (idxs, graphs, wires, logits, knockout_patterns)
    """
    sample_key, knockout_key = jax.random.split(key)
  
    # Sample from pool as normal
    idxs, graphs, wires, logits, existing_patterns = self.sample(sample_key, batch_size)
  
    if knockout_config is None:
        return idxs, graphs, wires, logits, existing_patterns
  
    # Determine which circuits to apply new knockouts to
    knockout_prob = knockout_config.get('knockout_probability', 0.0)
    apply_knockout_mask = jax.random.uniform(knockout_key, (batch_size,)) < knockout_prob
  
    if knockout_config.get('target_functional', False):
        # Prefer functional circuits for knockout application
        min_steps = knockout_config.get('min_steps_threshold', 50)
        if graphs.globals is not None:
            update_steps = graphs.globals[..., 1]
            functional_mask = update_steps > min_steps
            # Only apply knockouts to functional circuits
            apply_knockout_mask = apply_knockout_mask & functional_mask
  
    # Generate new knockout patterns for selected circuits
    new_patterns = existing_patterns.copy()  # Start with existing patterns
    num_to_knockout = jp.sum(apply_knockout_mask)
  
    if num_to_knockout > 0:
        pattern_keys = jax.random.split(knockout_key, num_to_knockout)
  
        # Extract layer info for pattern generation
        if batch_size > 0:
            single_graph = jax.tree.map(lambda x: x[0], graphs)
            layer_sizes = extract_layer_info_from_graph(single_graph, input_n=None)
      
            vmapped_pattern_creator = jax.vmap(
                lambda k: create_reproducible_knockout_pattern(
                    key=k,
                    layer_sizes=layer_sizes,
                    damage_prob=knockout_config.get('damage_prob', 0.1),
                    input_n=graphs.nodes['logits'].shape[1] - sum(ls[0]*ls[1] for ls in layer_sizes)
                )
            )
      
            generated_patterns = vmapped_pattern_creator(pattern_keys)
            knockout_indices = jp.where(apply_knockout_mask)[0]
            new_patterns = new_patterns.at[knockout_indices].set(generated_patterns)
  
    return idxs, graphs, wires, logits, new_patterns
```

#### Step 3.2: Training Loop Integration

**Target**: `boolean_nca_cc/training/train_loop.py`

```python
def enhanced_pool_train_step(
    model, optimizer, circuit_pool, sample_key, meta_batch_size,
    x_data, y_data, layer_sizes, n_message_steps, loss_type, loss_key, epoch,
    knockout_config=None
):
    """Enhanced training step with runtime knockout application using automatic feature zeroing."""
  
    # Sample circuits and apply knockouts
    idxs, graphs, wires, logits, knockout_patterns = circuit_pool.sample_and_apply_knockouts(
        sample_key, meta_batch_size, knockout_config
    )
  
    # No manual logit zeroing needed - CircuitSelfAttention handles feature zeroing automatically
    # when knockout_patterns are provided. The extracted logits will automatically be zero
    # for knocked-out gates due to the feature zeroing in the attention mechanism.
  
    # Continue with normal training step using knockout_patterns
    # The model will automatically zero features and circuit execution will get zero LUTs
    # ... rest of training logic ...
```

### Phase 4: Evaluation Pipeline Update

#### Step 4.1: Evaluation Circuit Source

**Target**: `boolean_nca_cc/training/train_loop.py`

```python
def run_knockout_periodic_evaluation(
    model, circuit_pool, eval_config, epoch, key, layer_sizes, x_data, y_data, loss_type
):
    """Enhanced evaluation using pool-sampled circuits with automatic feature zeroing."""
  
    eval_sample_key, eval_knockout_key = jax.random.split(key)
  
    # Sample functional circuits from pool for evaluation
    min_steps_for_eval = eval_config.get('min_steps_for_eval', 100)
    eval_batch_size = eval_config.get('eval_batch_size', 32)
  
    # Sample high-performing circuits
    if circuit_pool.graphs.globals is not None:
        update_steps = circuit_pool.graphs.globals[..., 1]
        functional_mask = update_steps > min_steps_for_eval
        functional_indices = jp.where(functional_mask)[0]
  
        if len(functional_indices) >= eval_batch_size:
            selected_indices = jax.random.choice(
                eval_sample_key, functional_indices, shape=(eval_batch_size,), replace=False
            )
            eval_circuits = extract_circuits_at_indices(circuit_pool, selected_indices)
        else:
            # Fallback to normal sampling if insufficient functional circuits
            idxs, graphs, wires, logits, _ = circuit_pool.sample(eval_sample_key, eval_batch_size)
            eval_circuits = list(zip(wires, logits))
    else:
        # Fallback to normal sampling
        idxs, graphs, wires, logits, _ = circuit_pool.sample(eval_sample_key, eval_batch_size)
        eval_circuits = list(zip(wires, logits))
  
    # Apply knockouts and evaluate recovery using automatic feature zeroing
    input_n = x_data.shape[-1]
    for circuit_wires, circuit_logits in eval_circuits:
        # Generate knockout pattern
        knockout_pattern = create_reproducible_knockout_pattern(
            eval_knockout_key, layer_sizes, 
            damage_prob=eval_config.get('damage_prob', 0.1),
            input_n=input_n
        )
  
        # Evaluate recovery after N message passing steps
        # CircuitSelfAttention will automatically zero features and provide zero LUTs to circuit execution
        # ... rest of evaluation logic with automatic feature zeroing ...
```

## Configuration Integration

### Simplified Config Structure

**Target**: `configs/config.yaml`

```yaml
knockout_system:
  # Runtime knockout application during training
  training_knockouts:
    enabled: true
    probability: 0.3  # Probability of applying knockouts to sampled circuits
    target_functional: true  # Prefer functional circuits for knockouts
    min_steps_threshold: 50  # Minimum steps to consider circuit functional
    damage_prob: 0.1  # Knockout density per circuit
    feature_zeroing: true  # Enable automatic feature zeroing (recommended: always true)
  
  # Pool reset behavior (legacy support)
  reset_knockouts:
    enabled: true  # Keep some reset-based knockouts for early training
    fraction: 0.2  # Fraction of pool to apply knockouts during reset
    damage_prob: 0.1
    feature_zeroing: true  # Apply feature zeroing during pool initialization
  
  # Evaluation configuration
  evaluation:
    use_pool_circuits: true  # Use pool circuits instead of fresh generation
    min_steps_for_eval: 100  # Minimum optimization steps for evaluation candidates
    eval_batch_size: 32
    damage_prob: 0.15  # Knockout density for evaluation
    feature_zeroing: true  # Use automatic feature zeroing in evaluation
```

## Key Design Decisions

### 1. Feature Zeroing Strategy: Natural Flow Alignment

- **Decision**: Zero node features for knocked-out gates in addition to attention masking
- **Rationale**: Leverages the natural circuit → graph → circuit flow to achieve automatic coordination without complex infrastructure

### 2. Single Point of Change: CircuitSelfAttention Enhancement

- **Decision**: Implement feature zeroing in the existing attention mechanism
- **Rationale**: Minimal code changes achieve both attention isolation and circuit-level gate removal automatically

### 3. Automatic Coordination: No Manual Conversion

- **Decision**: Let the system's natural flow handle coordination instead of manual conversion functions
- **Rationale**: Simpler, more reliable, and eliminates potential coordination bugs

### 4. JAX Immutability: Feature Masking Approach

- **Decision**: Use feature masking to achieve gate removal while maintaining uniform structure
- **Rationale**: Compatible with JAX compilation while achieving complete functional isolation

### 5. Incremental Implementation: Build on Existing Infrastructure

- **Decision**: Enhance existing systems rather than replacing them
- **Rationale**: Lower risk, easier testing, and maintains backward compatibility

## Risk Mitigation

- **Feature Flags**: Feature zeroing can be disabled via configuration for fallback to attention-only masking
- **Fallback Mechanisms**: System gracefully handles cases where feature zeroing is disabled
- **Gradual Rollout**: Each phase independently testable before proceeding to next phase
- **Performance Monitoring**: Benchmark against baseline to ensure feature zeroing doesn't cause regression
- **Automatic Validation**: System automatically ensures zeroed features translate to zero LUTs in circuit execution

## Expected Outcomes

1. **Automatic Coordination**: Perfect alignment between attention masking and circuit execution without manual coordination
2. **Enhanced Learning**: Network learns to recover from realistic gate removal scenarios during optimization
3. **Improved Robustness**: Better performance on functional circuit knockouts with complete gate isolation
4. **Maintained Performance**: No degradation on existing knockout scenarios
5. **System Compatibility**: Full backward compatibility with existing experiments
6. **Simplified Implementation**: Significantly reduced complexity compared to manual coordination approaches
