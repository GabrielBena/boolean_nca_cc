# Loss Propagation Strategy: Implicit vs Explicit

## Executive Summary

This document analyzes whether the current "implicit" loss propagation mechanism (loss as read-only input) is sufficient, or whether we should implement "explicit" loss propagation (loss as a model output that gets written back to the graph).

**Key Finding:** Fresh loss is injected after **every message step**, not just between outer loops. This significantly mitigates the concern about "lost" information between steps.

---

## Current Architecture: Implicit Loss Propagation

### How Loss Flows Through the System

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      SINGLE MESSAGE STEP                                │
│                                                                         │
│  1. READ: Model extracts features including loss                        │
│     features = [logits, hidden, layer_pe, intra_layer_pe, loss]         │
│                                                                         │
│  2. PROCESS: Attention layers propagate information                     │
│     x = feature_proj(features)                                          │
│     for layer in attention_layers:                                      │
│         x = layer(x, attention_mask)  ← Loss info propagates HERE       │
│                                                                         │
│  3. WRITE: Model outputs updates to logits and hidden ONLY              │
│     updated_logits = logits + logit_proj(x)                             │
│     updated_hidden = hidden + hidden_proj(x)                            │
│     # NOTE: loss is NOT written by the model!                           │
│                                                                         │
│  4. EXTERNAL: Circuit executes, fresh loss computed                     │
│     residuals = circuit_output - target                                 │
│     graph.nodes["loss"][output_nodes] = abs(residuals).mean(axis=0)     │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### What IS and IS NOT Preserved

| Feature | Read by Model | Written by Model | Preserved Between Steps |
|---------|--------------|------------------|------------------------|
| `logits` | ✅ Yes | ✅ Yes | ✅ Yes (in graph) |
| `hidden` | ✅ Yes | ✅ Yes | ✅ Yes (in graph) |
| `layer_pe` | ✅ Yes | ❌ No | ✅ Yes (static) |
| `intra_layer_pe` | ✅ Yes | ❌ No | ✅ Yes (static) |
| `loss` | ✅ Yes | ❌ No | ⚠️ Recomputed externally |
| Internal `x` | N/A | N/A | ❌ No (ephemeral) |

---

## Precise Loss Injection Points

### During Training (`train_loop.py`)

**Location:** `pool_train_step` → `loss_fn_no_scan` (lines 968-1009)

```python
for i in range(n_message_steps):
    # Step 1: Model forward pass
    graph = model(graph, ...)
    
    # Step 2: LOSS INJECTION - happens EVERY step
    graph, loss, logits, aux = get_loss_and_update_graph(
        graph=graph,
        ...
    )
    # This calls update_output_node_loss() internally
```

**Key Insight:** Loss is injected after **every single message step**, not just at the end.

### During Evaluation (`evaluation.py`)

**Location 1:** Initial injection before step 0 (line 224)
```python
graph = update_output_node_loss(graph, layer_sizes, initial_res_for_update.mean(axis=0))
```

**Location 2:** After every step in the loop (line 893)
```python
for step in range(1, n_message_steps + 1):
    # Model forward pass
    updated_graphs = vmap_model(current_graphs, ...)
    
    # Extract logits and compute loss
    current_losses, current_aux = vmap_get_loss(current_batch_logits, batch_wires)
    
    # LOSS INJECTION - happens EVERY step
    updated_graphs = vmap_update_loss(updated_graphs, graph_update_res)
```

### During Generator-based Evaluation (`evaluate_model_stepwise_generator`)

**Location 1:** Initial injection (line 224)
```python
graph = update_output_node_loss(graph, layer_sizes, initial_res_for_update.mean(axis=0))
```

**Location 2:** Inside `get_loss_and_update_graph` called every step (line 262-270)
```python
updated_graph, loss, current_logits, aux = get_loss_and_update_graph(...)
# This internally calls update_output_node_loss()
```

---

## The Multi-Hop Propagation Question

### Two Levels of Iteration

| Level | Config Parameter | What Iterates | Loss Injection |
|-------|-----------------|---------------|----------------|
| **Outer (Message Steps)** | `n_message_steps` | Entire GNN call + circuit execution | ✅ After EVERY step |
| **Inner (Attention Layers)** | `num_layers` | Attention blocks within one GNN call | ❌ No injection between |

### Information Propagation

**Within one GNN call (`num_layers` attention blocks):**
- Loss info at output nodes propagates via attention mechanism
- Each attention block = 1 hop along circuit edges
- With `num_layers=2` and a 4-layer circuit, output loss can reach hidden layer 1 in one GNN call

**Across GNN calls (`n_message_steps` outer steps):**
- Fresh loss is computed and injected after each step
- `hidden` state persists and can carry information
- Each step builds on the previous circuit state

### The "Lost Information" Concern

**Initial Concern:** Internal activations `x` that carry propagated loss through attention layers are ephemeral—they don't persist between message steps.

**Mitigating Factor:** Fresh loss is injected after EVERY message step. The model doesn't need to "remember" the loss from step t-1 because it gets the updated loss at step t.

**Remaining Gap:** If `num_layers < circuit_depth`, a hidden layer gate may not "see" the output loss within a single GNN call. It must wait until:
1. The loss propagates partially in GNN call 1
2. Gets encoded into `hidden` states
3. Those `hidden` states influence the next GNN call
4. Fresh loss comes in again

---

## Option A: Increase `num_layers` to Match Circuit Depth

### Implementation

```yaml
# configs/model/self_attention.yaml
num_layers: 4  # Match circuit depth (input + 2 hidden + output)
```

### Pros
- Simple config change, no code modifications
- Guarantees full propagation within one GNN call
- Model can see loss from all output nodes in every step

### Cons
- Computational cost: O(num_layers × num_nodes²) for attention
- Memory overhead for storing attention matrices
- May be overkill if `hidden` state is sufficient for cross-step memory

### Recommendation
**Set `num_layers ≥ circuit_depth - 1`** as a baseline. For a 4-layer circuit, use `num_layers: 3`.

---

## Option B: Explicit Loss Propagation (VSML-style)

### Inspiration: Variable Shared Meta Learning (VSML)

VSML implements bidirectional message passing:
```
s(k)_ab ← f_RNN(s(k)_ab, →m(k)_a, ←m(k)_b)
```

Where `←m` (backward message) explicitly carries error signals from outputs to inputs.

### Proposed Implementation

**Architecture Change:**
```python
# In CircuitSelfAttention.__init__():
self.loss_proj = nnx.Linear(
    self.attention_dim,
    1,  # Loss is scalar per node
    kernel_init=nnx.initializers.zeros,  # Start with no updates
    bias_init=nnx.initializers.zeros,
    rngs=rngs,
)
self.loss_scale = nnx.Param(jp.zeros(1)) if re_zero_update else 1.0

# In CircuitSelfAttention.__call__():
logit_updates = self.logit_proj(x)
hidden_updates = self.hidden_proj(x)
loss_updates = self.loss_proj(x)  # NEW

updated_logits = nodes["logits"] + self.logit_scale * logit_updates
updated_hidden = nodes["hidden"] + self.hidden_scale * hidden_updates
updated_loss = nodes["loss"] + self.loss_scale * loss_updates.squeeze(-1)  # NEW

updated_nodes = {
    **nodes, 
    "logits": updated_logits, 
    "hidden": updated_hidden,
    "loss": updated_loss  # NEW
}
```

### Pros
- Explicit credit assignment pathway
- Model can learn sophisticated error propagation strategies
- Aligns with proven VSML approach
- Could enable novel credit assignment beyond simple backpropagation

### Cons
- Requires retraining from scratch
- Adds parameters and computational cost
- Interaction with external loss injection needs careful design:
  - Should model-predicted loss ADD to or REPLACE external loss?
  - Should external loss only be injected at output nodes while model propagates to hidden?

### Design Questions
1. **Additive vs Replacement:** Should the model's loss updates be added to fresh external loss, or should the model fully control loss propagation?
2. **Node Scope:** Should loss updates be allowed for ALL nodes, or only hidden layers (since outputs already get fresh loss)?
3. **Supervision:** Should we add an auxiliary loss to encourage the model to propagate error signals correctly?

---

## Comparison Table

| Aspect | Option A (Increase `num_layers`) | Option B (Explicit `loss_proj`) |
|--------|----------------------------------|--------------------------------|
| Implementation Effort | Config change only | Architecture + retraining |
| Computational Cost | Higher attention cost | Higher projection cost |
| Flexibility | Fixed heuristic | Learnable strategy |
| Interpretability | Clear multi-hop semantics | Model decides propagation |
| VSML Alignment | Partial | Full |
| Risk | Low (simple change) | Medium (new architecture) |

---

## Recommendation

### Short-Term (Immediate)
1. **Increase `num_layers`** to match circuit depth:
   ```yaml
   num_layers: 4  # For 4-layer circuits
   ```
2. **Verify with `blind_mode`** ablation that the model isn't over-relying on fresh loss injection.

### Medium-Term (If Performance Plateaus)
1. **Implement explicit `loss_proj`** as described in Option B.
2. Start with **additive mode**: model loss updates ADD to external loss.
3. Use **re_zero initialization** so the model starts by ignoring loss updates and gradually learns to use them.

### Long-Term (Research Direction)
1. **Full VSML-style architecture:** Replace the separate "circuit execution + loss computation" with a unified forward-only dynamics.
2. This would allow meta-learning the credit assignment algorithm itself, not just the circuit configuration policy.

---

## Appendix: Code Locations for Loss Injection

| File | Function | Line(s) | Description |
|------|----------|---------|-------------|
| `evaluation.py` | `get_loss_and_update_graph` | 130 | Core loss injection function |
| `evaluation.py` | `evaluate_model_stepwise_generator` | 224 | Initial loss before step 0 |
| `evaluation.py` | `evaluate_model_stepwise_generator` | 290-293 | Blind mode override |
| `evaluation.py` | `evaluate_model_stepwise_batched` | 526-528 | Batch initial loss |
| `evaluation.py` | `_evaluate_with_loop` | 732-735, 893 | Loop loss injection |
| `train_loop.py` | `pool_train_step.loss_fn_no_scan` | 983-991 | Training loop injection |
| `extraction.py` | `update_output_node_loss` | 73-123 | Actual injection implementation |


