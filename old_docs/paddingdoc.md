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
# Use large negative values (-10.0) so sigmoid(-10) ≈ 0.0
large_negative_value = -10.0
current_logits = jp.where(
    active_mask[:, None], 
    nodes["logits"], 
    large_negative_value
)
current_hidden = nodes["hidden"] * active_mask[:, None]
```

**Complete gate removal is achieved when all three mechanisms work together automatically.**

## Implementation Strategy

**Key Insight**: The implementation follows a simple progression: **feature zeroing enhancement** → **pool initialization consistency** → **sampling-based damage application**. Each phase builds on the previous one and can be tested independently.

### Phase 1: Node Feature Zeroing Enhancement

**Goal**: Enhance the existing attention masking to include node feature zeroing, achieving automatic coordination between attention and circuit execution.

#### Step 1.1: Enhanced Self-Attention Integration (COMPLETE)

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
  
                    # SET TO LARGE NEGATIVE VALUES for knocked-out nodes (NEW)
            # Use -10.0 which gives sigmoid(-10) ≈ 4.5e-5 ≈ 0.0
            large_negative_value = -10.0
            current_logits = jp.where(
                active_mask[:, None], 
                nodes["logits"], 
                large_negative_value
            )
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

#### Step 1.2: Testing and Validation (COMPLETE)

**Testing Strategy for Phase 1**:

- Verify knocked-out nodes have zero logits after message passing
- Confirm extracted logits are zero for knocked-out gates
- Test that circuit execution receives zero LUTs for knocked-out gates
- Validate that attention masking still prevents message passing to/from knocked-out nodes

