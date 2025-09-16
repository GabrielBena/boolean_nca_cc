## SEU and Progressive Damage: Minimal, Composable Design

### Goals

- **SEU (Single Event Upset)**: Model stable but reversible LUT bit flips. Persist in circuit state until actively reconfigured by the Graph Transformer (GT).
- **Progressive damage**: Allow incremental addition of damage events (e.g., greedy gate order), composing with SEU and hard knockouts without invasive changes.
- **Evaluation-ready**: Support tracing baseline performance, immediate post-flip drop, and recovery after GT steps using existing evaluation utilities.
- **JAX implementation**: All SEU utilities and operations must be implemented using JAX for compatibility with the existing codebase and JIT compilation support.

### Principles

- **Separation of concerns**:
  - Hard knockouts = topology/attention masking and feature clamping (already implemented in `models/self_attention.py` via `_create_attention_mask` and knockout handling).
  - SEU = mutate circuit logits (LUT entries) only; leave attention fully enabled so GT can repair.
  - Attenuation (optional, non-SEU) = multiplicative gate output masks inside `circuits/model.run_circuit` through the `gate_mask` argument.
- **Persist at the circuit state**: Store SEU effects in the pool’s `logits`; graphs are built from these mutated logits via `utils/graph_builder.build_graph`.
- **Maximize reuse**: Leverage `training/evaluation.py` for loss computation and stepwise evaluation, `utils/extraction.py` for logits extraction, and `training/pool/pool.py` for persistence and updates.

---

## Data Model and Existing Persistence

- `training/pool/pool.py::GraphPool` persists:
  - `graphs`: current batched graph representations (built by `utils/graph_builder.build_graph`)
  - `wires`: canonical wiring
  - `logits`: canonical LUT parameters per layer (per circuit)
  - `knockout_patterns`: hard knockouts (boolean mask; separate from logits)
- SEU requires only mutating `logits` for selected circuits and rebuilding their corresponding `graphs` from updated logits and existing `wires`.

---

## Integration Touchpoints (existing functions to reuse)

- Graph build: `utils/graph_builder.build_graph(logits, wires, input_n, arity, circuit_hidden_dim, ...)`
- Logit extraction: `utils/extraction.extract_logits_from_graph(graph, logits_original_shapes)`
- Circuit execution + loss: `training/evaluation.get_loss_from_wires_logits(logits, wires, x, y, loss_type)`
- Unified loss + graph update: `training/evaluation.get_loss_and_update_graph(graph, logits_original_shapes, wires, x, y, loss_type, layer_sizes)`
- Stepwise eval (single): `training/evaluation.evaluate_model_stepwise_generator(...)`
- Stepwise eval (batched): `training/evaluation.evaluate_model_stepwise_batched(...)`
- Training losses (with optional gate masks): `circuits/train.loss_f_l4`, `circuits/train.loss_f_bce`
- Gate-mask creation from knockouts (for attenuation or hard): `circuits/train.create_gate_mask_from_knockout_pattern(knockout_pattern, layer_sizes)`
- Self-attention scan: `models/self_attention.run_self_attention_scan_with_loss(...)` (already integrates loss and graph update per step)
- Pool persistence/update: `training/pool/pool.GraphPool.update(...)`, `GraphPool.initialize_graph_pool(...)`

---

## API Additions (No GT changes)

### In `training/pool/perturbation.py`

Add utilities to operate on logits tensors using dense masks (pure JAX, JIT-friendly):

1) `compute_layer_offsets(layer_sizes) -> (layer_start_indices, gates_per_layer)`
   - Compute per-layer gate counts and cumulative start indices for mapping global gate indices without coupling to extraction internals.

2) `flip_logits_with_masks(logits_per_layer, flip_masks_per_layer, mode="invert") -> logits_per_layer`
   - `logits_per_layer`: list[(group_n, group_size, 2^arity)].
   - `flip_masks_per_layer`: list of boolean masks with same shapes; True entries are flipped.
   - `mode="invert"` only (negate selected logits). Defer `set0`/`set1` to future work.

3) Sampling and mask construction helpers:
   - `sample_seu_gates(key, layer_sizes, num_gates, strategy="greedy"|"random", ordered_indices=None)` (reuse `DEFAULT_GREEDY_ORDERED_INDICES`).
   - `build_flip_masks_from_indices(layer_sizes, selected_gate_indices, flips_per_gate, arity, key) -> flip_masks_per_layer` (samples LUT entry indices per selected gate and returns dense masks aligned to logits shapes).

### In `training/pool/pool.py`

Add method to apply SEUs to a subset of circuits and rebuild graphs using existing builders:

- `apply_seu(idxs, flip_masks_per_circuit, layer_sizes, input_n, arity, circuit_hidden_dim) -> GraphPool`
  - For each `idx` (vectorized via `vmap`):
    - Read `self.logits[idx]` (list of layer logits) and corresponding `flip_masks_per_layer`.
    - Apply `flip_logits_with_masks(...)` (invert-only) to obtain mutated logits.
    - Preserve globals: extract `[loss, update_steps] = self.graphs.globals[idx]` and pass as `loss_value`/`update_steps` to `build_graph(...)`.
    - Rebuild graph via `build_graph(logits=mutated_logits, wires=self.wires[idx], input_n=input_n, arity=arity, circuit_hidden_dim=circuit_hidden_dim, ...)`.
  - Commit via `update(idxs, batch_of_graphs=new_graphs, batch_of_logits=mutated_logits)`; leave `knockout_patterns` unchanged.

Convenience sampler mirroring `damage_fraction` but for SEU:

- `seu_fraction(key, fraction, layer_sizes, gates_per_circuit, flips_per_gate, selection_strategy, gate_selection, greedy_ordered_indices=None) -> (GraphPool, Array)`
  - Reuse `get_reset_indices(...)` to choose circuits.
  - For selected circuits, sample exactly `gates_per_circuit` gates via `sample_seu_gates(...)` and build dense LUT flip masks with `flips_per_gate` entries per gate via `build_flip_masks_from_indices(...)`.
  - Call `apply_seu(...)` with constructed flip masks.

---

## Execution Pipelines (using existing functions)

### Hard Knockout (existing; unchanged)
1) Generate/assign `knockout_pattern`s (e.g., `training/pool/perturbation.create_reproducible_knockout_pattern`)
2) GT step uses attention pruning and feature clamping for knocked-out nodes (`models/self_attention.CircuitSelfAttention` knockout handling)
3) Evaluation applies hard removal semantics (optionally via gate masks in `circuits/train.create_gate_mask_from_knockout_pattern` if using circuit-only execution)

### SEU Pipeline (stable but repairable)
1) Baseline measurement (before SEU):
   - Loss/acc: `training/evaluation.get_loss_from_wires_logits(logits, wires, x, y, loss_type)`
   - Or stepwise: `training/evaluation.evaluate_model_stepwise_batched(...)` step 0
2) Apply SEU (invert-only):
   - Select circuits and gates (greedy or random)
   - Build dense flip masks and mutate `logits` using `flip_logits_with_masks(...)`
   - Rebuild `graphs` from mutated `logits` using `utils/graph_builder.build_graph`
   - Persist via `GraphPool.update(...)` if operating on the pool
3) Immediate post-flip measurement:
   - Reuse `get_loss_from_wires_logits(...)` or the stepwise evaluators (no GT steps)
4) Recovery via GT:
   - `models/self_attention.run_self_attention_scan_with_loss(...)` for K steps
   - Internally uses `training/evaluation.get_loss_and_update_graph(...)` which calls `utils/extraction.extract_logits_from_graph` and updates loss features
5) Post-recovery measurement:
   - Use returned per-step losses/aux to read recovery; or run `evaluate_model_stepwise_batched(...)` for a uniform eval surface

This ensures SEU impact is persistent and only recovers through actual GT updates. No attention mask changes for SEU.

### Optional Attenuation Mode (non-SEU)

- Construct per-layer `gate_mask` (alpha in (0,1)) and pass to `circuits/model.run_circuit(logits, wires, x, gate_mask=...)`.
- For training or analysis, the loss wrappers in `circuits/train.py` (`loss_f_l4`, `loss_f_bce`) already accept `gate_mask`.
- Keep separate from SEU to avoid conflation of semantics.

---

## Progressive Damage (Future, Composes Cleanly)

- Represent progressive damage as repeated SEU applications or increasing `damage_count` for greedy gate prefixes:
  - For greedy: at `damage_count = k`, the damaged gate set is `ordered_indices[:k]`
  - Each increment triggers `apply_seu(...)` with flips for the newly added gate(s)
- Coexists with existing hard knockouts (orthogonal state in `knockout_patterns`).
- Eval can interleave progressive events with the SEU pipeline measurements using `evaluate_model_stepwise_batched(...)`.

---

## Evaluation Considerations (Design-Aware)

### Pool-Free Evaluation Architecture

SEU evaluation should mirror the knockout evaluation pattern to avoid pool dependencies:

- **No pool usage** - Use `base_wires` and `base_logits` directly (like `run_knockout_periodic_evaluation`)
- **SEU pattern vocabulary** - Generate/sample SEU flip mask patterns similar to knockout patterns
- **Circuit replication** - Replicate base circuit for batch evaluation
- **Stepwise integration** - Use `evaluate_model_stepwise_batched` with SEU-modified logits

### SEU Pattern System

Create SEU pattern system that mirrors knockout patterns:

1. **SEU Pattern Generation**:
   ```python
   def create_seu_vocabulary(
       rng: jax.random.PRNGKey,
       vocabulary_size: int,
       layer_sizes: List[Tuple[int, int]],
       seu_config: Dict,
   ) -> List[List[jp.ndarray]]:
       """Generate vocabulary of SEU flip mask patterns per circuit"""
   ```

2. **SEU Pattern Application**:
   - Apply flip masks to logits **before** GNN processing (upstream)
   - Unlike knockouts which are applied **during** GNN forward pass via attention masking
   - GNN can freely repair the flipped logits through message passing

3. **Modified Evaluation Flow**:
   ```python
   def run_seu_periodic_evaluation(
       model,
       seu_vocabulary: Optional[List[List[jp.ndarray]]],
       base_wires: PyTree,
       base_logits: PyTree,
       seu_config: Dict,
       # ... other params
   ) -> Dict:
       # 1. Generate/sample SEU patterns
       # 2. Replicate base circuit for batch
       # 3. Apply SEU patterns to logits
       # 4. Call evaluate_model_stepwise_batched with SEU-modified logits
   ```

### Logit Storage and Persistence

**Key Insight**: SEU uses the same logit storage system as normal training:

- **Pool Storage**: `self.logits` stores logits separately from `self.graphs`
- **Graph Building**: `build_graph()` embeds logits into graph node features
- **GNN Processing**: GNN updates logits in graph nodes during message passing
- **Logit Extraction**: `extract_logits_from_graph()` extracts updated logits from graphs
- **Pool Update**: Pool's `update()` method can store extracted logits back to `self.logits`

**SEU Flow**:
1. Apply SEU flips to logits and store in pool (temporary)
2. Build graphs with flipped logits
3. GNN repairs flipped logits through message passing
4. Extract repaired logits from graphs
5. Update pool storage with repaired logits

### Stepwise Evaluation Integration

Modify `_evaluate_with_loop` to handle SEU patterns:

```python
def _evaluate_with_loop(
    # ... existing params
    seu_patterns: Optional[List[List[jp.ndarray]]] = None,  # New parameter
):
    # Apply SEU patterns to logits before GNN processing
    if seu_patterns is not None:
        modified_logits = apply_seu_to_logits(batch_logits, seu_patterns)
    else:
        modified_logits = batch_logits
    
    # Continue with normal GNN processing
```

### Evaluation Protocol

- **Baseline measurement**: Record metrics on original logits
- **SEU application**: Apply flip masks to logits (upstream of GNN)
- **Immediate measurement**: Record metrics on flipped logits (0 GNN steps)
- **Recovery measurement**: Run K GNN steps and record recovery trajectory
- **Final measurement**: Record final metrics after GNN repair

### Implementation Requirements

- Use existing evaluation stack: `get_loss_from_wires_logits`, `get_loss_and_update_graph`, `extract_logits_from_graph`
- Support both soft and hard accuracy measurements
- Enable stepwise analysis of SEU recovery dynamics
- Optional: persist SEU event log (gate indices and LUT entry indices) for audit/replay

---

## Configuration Hooks (Non-invasive)

- `eval.seu.enabled: bool`
- `eval.seu.flips_per_gate: int` (default 1)
- `eval.seu.selection_strategy: str` (uniform|steps_biased|loss_biased|combined)
- `eval.seu.gate_selection: str` (greedy|random)
- `eval.seu.greedy_ordered_indices: list[int]` (fallback to default constant)
- `eval.progressive.max_flips: int` (if testing progressive SEUs)

Training-side analogs can mirror these if applying SEUs during training phases.

---

## Rationale and Trade-offs

- **Why mutate logits for SEU?** Ensures stability of the upset until active reconfiguration; avoids conflating with attention/topology; minimal changes to existing model/eval.
- **Why keep attenuation separate?** Attenuation captures analog degradation, not persistent bit flips; distinct enforcement (runtime mask) vs. parameter corruption (SEU).
- **Why use pool `update` flow?** Centralizes persistence and batching; reuses `build_graph` and `extract_logits_from_graph` pathways; aligns with current damage/selection mechanics in `GraphPool`.
 - **Why dense masks over ragged flip specs?** Boolean masks are JAX- and JIT-friendly, enabling vectorized flips with `vmap` and avoiding Python-side loops and shape polymorphism.

---

## Minimal Implementation Checklist

1) Add SEU utilities in `training/pool/perturbation.py`:
   - layer offset computation, mask-based LUT flipping (invert-only), sampling + mask builders  [DONE]
2) Add `GraphPool.apply_seu(...)` (mask-based, globals-preserving) and `GraphPool.seu_fraction(...)` wrappers  [DONE]
3) Create SEU pattern vocabulary system:
   - `create_seu_vocabulary()` function for generating SEU flip mask patterns  [DONE]
   - Pattern sampling and replication for batch evaluation  [DONE]
4) Modify `run_seu_periodic_evaluation()` to mirror knockout evaluation:
   - Remove pool dependency, use `base_wires` and `base_logits` directly  [DONE]
   - Generate/sample SEU patterns similar to knockout patterns  [DONE]
   - Apply SEU patterns to logits before GNN processing  [DONE]
   - Use `evaluate_model_stepwise_batched` for stepwise evaluation  [DONE]
5) ~~Extend `_evaluate_with_loop()` to handle SEU patterns~~ [NOT NEEDED]
   - ~~Add `seu_patterns` parameter~~
   - ~~Apply SEU flip masks to logits upstream of GNN processing~~
   - ~~Enable stepwise SEU recovery analysis~~
   - **Note**: No changes needed to `_evaluate_with_loop()` - existing evaluation stack works perfectly with SEU-modified logits
6) Integrate SEU evaluation into main training loop:
   - Update SEU evaluation call to match knockout evaluation pattern  [DONE]
   - Use `base_wires` and `base_logits` instead of pool sampling  [DONE]
   - Proper metrics handling and WandB logging  [DONE]

---

## Implementation Status: COMPLETE ✅

**All SEU functionality has been successfully implemented and integrated!**

### Key Achievements:

1. **SEU Utilities** (`training/pool/perturbation.py`):
   - Complete set of JAX-compatible SEU functions
   - Greedy and random gate selection strategies
   - Dense mask-based LUT flipping (invert-only mode)
   - SEU pattern vocabulary generation

2. **Pool Integration** (`training/pool/pool.py`):
   - `GraphPool.apply_seu()` for applying SEU to selected circuits
   - `GraphPool.seu_fraction()` for pool-based SEU application during training
   - Preserves graph globals and maintains pool consistency

3. **Evaluation System** (`training/train_loop.py`):
   - `run_seu_periodic_evaluation()` mirrors knockout evaluation pattern exactly
   - Supports both vocabulary-based and fresh pattern generation
   - IN-distribution and OUT-of-distribution evaluation
   - Complete WandB logging and metrics handling

4. **Training Integration**:
   - SEU evaluation fully integrated into main training loop
   - Uses `base_wires` and `base_logits` (no pool dependency)
   - Consistent with knockout evaluation architecture

### Design Principles Achieved:

- **Upstream SEU Application**: LUT flips applied before GNN processing
- **Training-Evaluation Consistency**: Both use identical SEU application pattern
- **Code Reuse**: Leverages existing evaluation stack without modification
- **Knockout Pattern Mirroring**: Identical architecture and function signatures

### Ready for Production Use:

The SEU evaluation system is now ready to:
- Generate SEU patterns using greedy or random gate selection
- Apply LUT bit flips upstream of GNN processing
- Measure GNN repair capabilities through stepwise evaluation
- Log comprehensive metrics to WandB
- Support both vocabulary-based and fresh pattern generation

**The implementation is complete and follows the exact same pattern as knockout evaluation, ensuring consistency and maintainability!**
