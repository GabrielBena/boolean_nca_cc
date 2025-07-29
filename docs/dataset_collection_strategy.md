# Circuit Configuration Collection Strategy

## Objective

Create a unified dataset collection system to store optimized circuit configurations for both unperturbed and perturbed (knockout) circuits, enabling analysis of reconfiguration behavior under structural damage.

## Current Infrastructure (Reuse)

- **Pool System**: `GraphPool` already manages mixed perturbed/unperturbed circuits
- **Pattern Storage**: `knockout_patterns` field stores damage masks
- **Config Extraction**: `extract_logits_from_graph()` extracts LUT configurations
- **Performance Tracking**: `accumulated_pattern_data` tracks pattern performance over time
- **Evaluation Pipeline**: Existing knockout evaluation infrastructure

## Data Collection Strategy

### 1. Unified Collection Function

```python
def collect_circuit_configuration(
    graph: jraph.GraphsTuple,
    wires: PyTree, 
    logits: PyTree,
    knockout_pattern: Optional[jp.ndarray],
    performance_metrics: Dict[str, float],
    metadata: Dict[str, Any]
) -> Dict[str, Any]
```

**Handles both cases:**

- **Unperturbed**: `knockout_pattern=None`, stores baseline LUT config + performance
- **Perturbed**: `knockout_pattern` present, stores reconfigured LUT + damage pattern + performance

### 2. Collection Trigger Points

- **Pool Updates**: During `pool.update()` when performance exceeds threshold
- **Evaluation**: During periodic knockout evaluation for successful reconfigurations
- **Training**: At specified intervals during optimization steps
- QUESTION: before implementing anything, consult with me to see where exactly collection could and should take place

### 3. Performance Threshold Filter

- **Hard Accuracy**: Primary filter (e.g., `hard_accuracy >= 0.95`)

## Dataset Structure

```python
CircuitRecord = {
    'circuit_id': str,           # Unique identifier
    'lut_config': List[Array],   # Extracted logit configurations  
    'wires': PyTree,             # Circuit topology
    'knockout_pattern': Optional[Array],  # Damage mask (None for unperturbed)
    'performance': Dict,         # All metrics (hard_acc, soft_acc, loss)
    'optimization_mode': str,    # 'baseline' | 'from_nops' | 'reconfigured'
    'epoch': int,                # When collected
    'pattern_id': Optional[int], # Index in knockout vocabulary (if applicable)
}
```

## Implementation Phases

### Phase 1: Extend Pool Collection

- Add `collect_configurations()` method to `GraphPool`
- Integrate with existing `update()` method
- Use performance thresholding in pool sampling

### Phase 2: Evaluation Integration

- Extend `run_knockout_periodic_evaluation()` to collect high-performing configs
- Leverage existing `accumulated_pattern_data` pipeline
- Store both in-distribution and out-of-distribution results

### Phase 3: Mode A vs B Comparison

- **Mode A**: Optimize unperturbed → apply damage → re-optimize → collect both configs
- **Mode B**: Start from NOPs + damage → optimize → collect final config
- Compare endpoint configurations for identical damage patterns

## Key Research Question

**Convergence Analysis**: Do Mode A and Mode B reach the same final LUT configuration C for identical damage patterns?

## Storage Backend

- **File Format**: HDF5 or JAX-native pickle for array efficiency
- **Metadata**: JSON sidecar files for searchability
- **Organization**: Separate collections for unperturbed vs perturbed circuits
- **Indexing**: By performance, damage pattern, optimization mode

## Efficiency Considerations

- **Lazy Collection**: Only store when performance threshold met
- **Batch Processing**: Collect multiple configurations per evaluation cycle
- **Memory Management**: Stream to disk during long training runs
- **Deduplication**: Hash-based checking for identical configurations

## Integration Points

- `train_loop.py`: Add collection calls during training
- `evaluation.py`: Extend evaluation to include collection
- `pool.py`: Add collection methods to `GraphPool`
- New module: `boolean_nca_cc/training/config_collection.py`
