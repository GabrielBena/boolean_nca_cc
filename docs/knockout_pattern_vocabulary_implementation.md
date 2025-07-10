# Knockout Pattern Vocabulary Implementation Strategy

## Overview

This document outlines the implementation strategy for creating a controlled vocabulary of knockout patterns for training, enabling true in-distribution (ID) evaluation based on patterns actually encountered during training.

## Problem Statement

Currently, knockout patterns are generated randomly from an astronomical space of possibilities during training, while evaluation uses completely separate pattern sets. This means:

1.  **No true memorization testing**: ID evaluation patterns are never actually seen during training.
2.  **Infinite pattern space**: Training draws from unlimited combinations, making reproducibility difficult.
3.  **Evaluation mismatch**: The "ID" patterns in `knockout_eval.py` are not truly in-distribution.

## Solution: Fixed Knockout Vocabulary

We will create a bounded vocabulary of knockout patterns shared between training and evaluation. This ensures that in-distribution (ID) evaluation accurately tests for the memorization of patterns seen during training. The implementation will use a shared random seed to generate a master set of patterns, from which both processes will randomly sample.

## Implementation Strategy

### Step 1: Update Configuration

In `configs/config.yaml`, add a `knockout_diversity` parameter under the `pool.persistent_knockout` section. This integer will define the size of the shared knockout pattern vocabulary. If this parameter is not set, the system will default to the original behavior of generating random patterns on-the-fly.

### Step 2: Create the Vocabulary Generation Function

In `boolean_nca_cc/training/knockout_eval.py`, add a new function named `create_knockout_vocabulary`. This function will reuse the existing `create_reproducible_knockout_pattern` logic, using `jax.vmap` to efficiently generate the entire vocabulary from a single base seed passed to it.

### Step 3: Integrate Vocabulary into the Training Loop

In `boolean_nca_cc/training/train_loop.py`:
1.  **Initialization**: At the beginning of `train_model`, if `knockout_diversity` is configured, call `create_knockout_vocabulary` to generate the vocabulary. This should use the same `periodic_eval_test_seed` that evaluation uses to ensure the pattern space is identical.
2.  **Sampling**: In the pool reset logic (inside the `should_reset_pool` block), when new knockout patterns are needed, randomly sample the required number of patterns **with replacement** from the in-memory vocabulary.

### Step 4: Adapt Pool Initialization

Modify the existing `initialize_graph_pool` function in `boolean_nca_cc/training/pool/pool.py`. It should be updated to accept an optional `knockout_patterns` array.
- If the array is provided, the function will use these patterns directly for the new circuits.
- If the array is `None`, the function will fall back to its original behavior of generating new random patterns based on the `knockout_config`.

The training loop will then pass the patterns sampled in Step 3 to this modified function.

### Step 5: Eliminate Separate Evaluation Dataset Creation

**CRITICAL ARCHITECTURAL CHANGE**: Remove the separate evaluation dataset creation that generates independent patterns.

In `boolean_nca_cc/training/train_loop.py`, modify the knockout evaluation logic:

1. **Remove `create_knockout_evaluation_datasets` calls**: This function creates independent patterns, breaking the vocabulary sharing.

2. **Direct vocabulary sampling for evaluation**: 
   - **In-Distribution (ID)**: Sample patterns directly from the same `knockout_vocabulary` created in Step 3
   - **Out-of-Distribution (OOD)**: Generate fresh patterns using a different seed (`periodic_eval_test_seed + 1`)

3. **Modify `run_knockout_periodic_evaluation`**: 
   - Accept the vocabulary and base circuit directly instead of pre-created datasets
   - Sample ID patterns from vocabulary on-demand
   - Generate OOD patterns on-demand
   - Use the same base circuit for both evaluations

**Why this matters**: The current `create_knockout_evaluation_datasets` function generates its own patterns independently from the training vocabulary, making "ID" evaluation actually test on unseen patterns. True ID evaluation requires sampling from the exact same vocabulary used during training.

## Recommended Configuration

```yaml
pool:
  persistent_knockout:
    fraction: 0.8
    damage_prob: 70.0
    target_layer: null
    knockout_diversity: 1024  # Vocabulary size for knockout patterns
```

## Expected Benefits

1.  **True ID Evaluation**: ID patterns are sampled from the exact same vocabulary seen during training.
2.  **Reproducible Training**: The same seed produces the same knockout vocabulary across runs.
3.  **Tunable Memorization Pressure**: The `knockout_diversity` and training length control the frequency of pattern reuse.
4.  **Simpler Architecture**: Eliminates duplicate pattern generation and ensures single source of truth for knockout patterns.
5.  **Simplified State Management**: Relies on standard JAX RNG key management without extra state variables.

## Testing Strategy

1.  **Pattern Identity Verification**: Assert that patterns in the ID evaluation are sampled from the training vocabulary and that OOD patterns are freshly generated.
2.  **Performance Comparison**: Compare ID vs. OOD evaluation performance to measure generalization vs. memorization.
3.  **Seed Reproducibility**: Verify that identical seeds produce identical knockout vocabularies.
4.  **Checkpoint Compatibility**: Ensure that the training `rng` state is properly saved and restored from checkpoints.

## Files to Modify

1.  `configs/config.yaml` ✓ **COMPLETE**
2.  `boolean_nca_cc/training/knockout_eval.py` ✓ **COMPLETE**
3.  `boolean_nca_cc/training/train_loop.py` ✓ **COMPLETE**
4.  `boolean_nca_cc/training/pool/pool.py` ✓ **COMPLETE**

## Implementation Priority

1.  **High**: Steps 1-4 (Core vocabulary system for training) ✓ **COMPLETE**
2.  **Medium**: Step 5 (Evaluation architecture fix) ✓ **COMPLETE**

## Implementation Status

**ALL STEPS COMPLETED** ✓

All components of the knockout pattern vocabulary implementation have been successfully implemented:

- ✅ **Step 1**: Configuration parameter added (`knockout_diversity`)
- ✅ **Step 2**: Vocabulary generation function created (`create_knockout_vocabulary`)
- ✅ **Step 3**: Vocabulary integrated into training loop with sampling logic
- ✅ **Step 4**: Pool initialization adapted to accept pre-generated patterns
- ✅ **Step 5**: Evaluation architecture restructured to use vocabulary directly

The system now provides true in-distribution (ID) evaluation by sampling from the same knockout pattern vocabulary used during training, while generating fresh patterns for out-of-distribution (OOD) evaluation.

## Final Implementation Summary

The knockout pattern vocabulary system has been successfully implemented with the following key features:

### Core Architecture
- **Shared Vocabulary**: A fixed set of knockout patterns (`knockout_diversity` size) shared between training and evaluation
- **Deterministic Generation**: Uses `periodic_eval_test_seed` to ensure reproducible pattern vocabularies
- **Memory Efficient**: Patterns generated once and reused throughout training

### Training Integration
- **Pool Reset Sampling**: Training samples patterns with replacement from the vocabulary during pool resets
- **Fallback Compatibility**: Maintains backward compatibility with original random pattern generation when `knockout_diversity` is not configured
- **Consistent Layer Sizes**: Uses actual graph layer sizes for accurate pattern generation

### Evaluation Architecture
- **True ID Evaluation**: Samples from the same vocabulary used in training (seed: `periodic_eval_test_seed`)
- **True OOD Evaluation**: Generates fresh patterns with different seed (`periodic_eval_test_seed + 1`)
- **On-Demand Generation**: Creates evaluation circuits only when needed, eliminating pre-computed datasets
- **Vocabulary-Independent**: Functions correctly whether vocabulary is available or not

### Key Benefits Achieved
1. **Accurate Memorization Testing**: ID evaluation truly tests patterns seen during training
2. **Reproducible Research**: Same seeds produce identical pattern vocabularies across runs
3. **Tunable Complexity**: `knockout_diversity` parameter controls memorization pressure
4. **Clean Architecture**: Single source of truth for knockout patterns eliminates duplication 