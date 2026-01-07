# Seed Control Reference

This document maps which configuration seeds control which stochastic components.

## Seed Parameters

| Parameter | Default | Config Location |
|-----------|---------|-----------------|
| `seed` | `${test_seed}` | `config.yaml:9` |
| `test_seed` | `34` | `config.yaml:10` |
| `damage_seed` | `481` | `config.yaml:11` |
| `input_split_seed` | `${test_seed}` | `config.yaml:109` |

---

## What Each Seed Controls

### `seed` (Model Initialization Seed)

**Varied in statistical sweeps to measure model variance.**

| Component | Effect |
|-----------|--------|
| Model weight initialization | All transformer/attention/MLP weights |
| Pool batch sampling | Which circuits sampled each epoch |
| Pool reset selection | Which circuits chosen for periodic resets |
| Damage application order | Which circuits receive damage (if enabled) |

### `test_seed` (Circuit Structure Seed)

**Fixed to ensure same problem across runs.**

| Component | Effect |
|-----------|--------|
| Circuit wiring pattern | Gate connection topology |
| Pool initialization wiring | All pool circuits share this wiring |
| Pool reset wiring | Reset circuits use same wiring |
| Evaluation circuit | Base circuit for periodic evaluation |

### `input_split_seed` (Data Split Seed)

**References `test_seed` by default (fixed at 34).**

| Component | Effect |
|-----------|--------|
| Train/test index shuffling | Which input combinations go to train vs test |

### `damage_seed` (Knockout Pattern Seed)

**Fixed to ensure reproducible damage patterns.**

| Component | Effect |
|-----------|--------|
| Knockout vocabulary generation | Which gates are in damage patterns |

---

## Sweep Design Implications

### Current Design: Vary `seed` Only

- ✅ Measures **model initialization variance**
- ✅ Same problem (circuit + data split) across all runs
- ✅ Clean interpretation: "How consistently can different models solve this specific problem?"

### Why Train ≈ Test Curves Match

For algorithmic tasks (add, xor, etc.):
- Correct gate configuration implements the **general algorithm**
- Algorithm works for ALL inputs, not just training inputs
- Matching curves indicate **good generalization**, not data leakage

### If You Need Different Experiments

| Goal | What to Vary |
|------|--------------|
| Model robustness | `seed` only (current) |
| Data split robustness | `input_split_seed` only |
| Circuit structure robustness | `test_seed` only |
| Full stochasticity | All seeds together |

---

## Code Locations

```
seed usage:
  train.py:193          → Master RNG
  train.py:282-286      → Model initialization (nnx.Rngs)
  train_loop.py:777     → Training loop RNG
  train_loop.py:1147    → Batch sampling
  train_loop.py:1218    → Pool reset selection

test_seed usage:
  train.py:229,360      → Circuit wiring
  train_loop.py:914     → Pool initialization
  train_loop.py:1225    → Pool reset wiring

input_split_seed usage:
  data_split.py:65      → Index shuffling

damage_seed usage:
  train.py:260          → Backprop knockout vocab
  train_loop.py:1099    → Training knockout vocab
```

