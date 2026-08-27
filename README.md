# Self-Organising Digital Circuits (SODC)

**Barylli\*, Béna\*, Mordvintsev, Nisioti, Risi — ALIFE 2026** — [Paper](https://direct.mit.edu/isal/proceedings/isal2026/38/41/138130) · [Blog post + live demo](https://self-organising-circuits.github.io/) · [PhD thesis context](https://gabrielbena.github.io/phd/)

We extend the Neural Cellular Automata paradigm from pattern formation on grids to
functional logic generation and self-repair on arbitrary graphs. A **Topology-Masked
Transformer (TMT)** — a single shared-weight attention block, applied recurrently
under a binary wiring mask — configures the Look-Up Tables (LUTs) of a Boolean
circuit's gates. It self-assembles functional circuits from unconfigured "soft
wires," and re-routes logic around previously-unseen hardware faults by exploiting
degeneracies of the Boolean solution space, without any global backpropagation at
deployment time.

This repository is the training/evaluation code, the exact configs behind every
reported number, and the source of the paper's figures.

## See it run

The clearest way to understand what this repo trains is to watch it: the trained
policy runs live in your browser, settling a circuit from scratch and re-healing it
after damage.

- **[Live demo, in the blog post](https://self-organising-circuits.github.io/#interactive-demo)**
  — no install required.
- **Run it locally** (same demo, driven by the pretrained weights checked into this
  repo):
  ```bash
  cd web_demo
  npm install
  npm run dev        # http://localhost:5173/
  ```
  See [`web_demo/README.md`](web_demo/README.md) for the export pipeline that
  produces the weight bundles it loads, and for how to add your own trained runs to
  the gallery.

## Installation

```bash
git clone https://github.com/GabrielBena/boolean_nca_cc.git
cd boolean_nca_cc

pip install -e .            # core (JAX/Flax training + eval)
pip install -e ".[notebooks]"  # + Jupyter/seaborn for exploratory analysis
pip install -e ".[dev]"        # + pytest/ruff/mypy
pip install -e ".[all]"        # everything
```

Requires Python ≥3.11. The package uses JAX/Flax (`nnx`) for training, Hydra for
configuration, and Weights & Biases for experiment tracking (the paper's exact runs
are public under the `gbena/boolean-nca-cc` W&B project, referenced by run ID
throughout `paper_figures/`).

## Quick Start

```bash
# Train the paper's model (TMT / gathered_attention, the Hydra default) on the
# default task (bit reversal) with random wiring
python train.py

# Fixed-topology training (Regime I/II)
python train.py training.wiring_mode=fixed

# Switch task (paper's three: reverse, add, binary_multiply)
python train.py circuit.task=add

# Ablation models, kept for comparison/testing — not what the paper reports
python train.py model=self_attention   # dense O(N^2) masked attention, no gathering
python train.py model=gnn              # sparse message-passing GNN
python train.py model=perceiver_attention  # + cross-attention to task data (superseded design)
```

### Programmatic usage

```python
from flax import nnx
from boolean_nca_cc import generate_layer_sizes
from boolean_nca_cc.circuits.tasks import get_task_data
from boolean_nca_cc.models import CircuitGatheredAttention
from boolean_nca_cc.training import train_model

x_data, y_data = get_task_data("reverse", case_n=4096, input_bits=12, output_bits=12)
layer_sizes = generate_layer_sizes(input_n=12, output_n=12, arity=4, layer_n=3)
n_node = sum(size[0] for size in layer_sizes)

model = CircuitGatheredAttention(
    circuit_hidden_dim=64, attention_dim=128, arity=4, rngs=nnx.Rngs(0)
)

results = train_model(
    key=42, init_model=model, x_data=x_data, y_data=y_data,
    layer_sizes=layer_sizes, hidden_dim=64, arity=4,
    learning_rate=2e-4, n_message_steps=5, wiring_mode="random",
)
```

Full config reference: `configs/config.yaml` (the single source of truth — every
Hydra group interpolates from it).

## The Model — Topology-Masked Transformer (TMT)

Code: [`boolean_nca_cc/models/attention/gathered_attention.py`](boolean_nca_cc/models/attention/gathered_attention.py)
(`CircuitGatheredAttention`), config [`configs/model/gathered_attention.yaml`](configs/model/gathered_attention.yaml)
— this is the Hydra default (`model=gathered_attention`) and the only architecture
trained for every reported result and shipped in the live demo.

A single-block Transformer, applied recurrently for `T` steps, whose attention is
restricted to each gate's wired neighbours via a binary topology mask:

- **Node state** — LUT logits (the gate's programmable logic), a recurrent latent
  memory, a positional encoding (normalised depth; optionally a directional
  DAG-distance encoding, `graph.use_dist_pe`), and (for random wirings) a per-node
  scalar error-feedback signal (`graph.use_node_loss`).
- **Gathered masked attention** — neighbour features are gathered into a fixed-width
  padded tensor per node (`max_neighbors` in the model config) instead of a dense
  N×N matrix — sub-quadratic in circuit size while attending only to actual wired
  neighbours.
- **Pre-LN + QK-normalisation + ReZero** — stabilises the block across the many
  recurrent applications training requires (residual gates initialised at zero, so
  the block starts as an identity).
- **Weight-tied, size-independent** — the same parameters apply whether the circuit
  has 20 gates or 500, giving the scale-freedom explored in Regime IV.

Full derivation and equations: the [manuscript](https://direct.mit.edu/isal/proceedings/isal2026/38/41/138130).

### Other architectures in this repo (not the paper's headline model)

`boolean_nca_cc/models/` also contains three earlier/alternative designs, kept
because the test suite exercises them as correctness/ablation baselines — they are
**not** what any reported figure or the demo uses:

| Class | Config | Role |
|---|---|---|
| `CircuitSelfAttention` | `model=self_attention` | Dense O(N²) masked attention — the reference implementation `CircuitGatheredAttention` is validated against (`tests/test_gathered_correctness.py`) |
| `CircuitGNN` | `model=gnn` | Sparse message-passing GNN — an explicit non-attention ablation |
| `PerceiverCircuitAttention` | `model=perceiver_attention` | Adds cross-attention to input/output data — an earlier, richer design point superseded by the simpler gathered-attention TMT before the final campaign |

## Experiments and Results — the Four Regimes

Each regime in the paper maps to a training config/sweep and a `paper_figures/`
script. See [`paper_figures/README.md`](paper_figures/README.md) for exact rebuild
commands, W&B run IDs, and — for the two figures not yet reproducible from this
repo — where their original code lives.

| Regime | Paper section | What it shows | Config / sweep | Figure script |
|---|---|---|---|---|
| I — Growth, Persistence, Repair | Fixed topologies | Self-assembly from soft-wires; recovery from stochastic damage | `training.wiring_mode=fixed`, `sweeps/sweep_demo_12*.yaml` | `paper_figures/fig2_fixed_wiring.py`, `fig_resilience.py` |
| I — PCA trajectories | Fixed topologies | Degenerate re-routing under damage, visualised via PCA on LUT-logit configs | (eval-only; run `eval_fig4_resilience_isn1.py`-style rollout) | `paper_figures/fig_pca_trajectories.py` |
| II — Self-Healing & Degenerate Solutions | Reversible soft-errors | Near-perfect OOD recovery at damage scales beyond training; degenerate solution clusters (UMAP) | — | *known gap, see below* |
| III — Random-Topology Generalisation | Random wiring | Wiring-agnostic policy generalising to unseen graphs | `training.wiring_mode=random`, `sweeps/sweep_random.yaml` | `paper_figures/fig_random_wiring.py` |
| IV — Scale-Free Optimisation | Circuit-width scaling | Zero-retrain generalisation to circuits 1.7× the training width | (same trained checkpoints as Regime I/III, re-evaluated at other widths) | `paper_figures/fig_scale_free.py` |

**Known gap**: the Regime II combined Soft-Error-Recovery/Hamming panel and the UMAP
solution-space figure were built by a co-author (Marcello Barylli) against a
pre-refactor API and were never merged into this branch's history. Their original
source is archived at [`paper_figures/archive/`](paper_figures/archive/) and on the
`mergello` branch, kept alive specifically for this — see the Known Gaps section of
`paper_figures/README.md`.

## Boolean Tasks

The paper's three 12-bit tasks (all $2^{12}=4{,}096$ input–output pairs, 256 held
out for testing):

| Task | Config | Description |
|---|---|---|
| `reverse` | `configs/tasks/reverse.yaml` | Bit Reversal — pure routing, maps bit `i` to position `11-i` |
| `add` | `configs/tasks/add.yaml` | Split Addition — two 6-bit integers added |
| `binary_multiply` | `configs/tasks/binary_multiply.yaml` | Split Multiplication — two 6-bit integers multiplied |

`configs/tasks/` also has extras (`identity`, `parity`, `text`) and a later,
separate research thread on per-circuit sampled tasks (`k_junta`, `arith_family`,
`single_add`, `single_reverse`, driving the "unified" online/batch inner-loop
continuum in `training.inner_loop_regime`) — these are exploratory extensions built
on top of the same framework, not part of the published results.

## Package Structure

```
boolean_nca_cc/
├── circuits/                      # Circuit generation, tasks, differentiable execution
│   ├── model.py                   # Circuit creation and (soft/hard) execution
│   ├── tasks.py                   # Boolean task definitions
│   └── train.py                   # Direct-BP baseline training utilities
├── models/
│   ├── attention/
│   │   ├── base.py                # Shared building blocks: ReZero, Pre-LN, QK-norm,
│   │   │                          #  gathered/dense masked attention
│   │   ├── gathered_attention.py  # CircuitGatheredAttention — the paper's TMT
│   │   ├── self_attention.py      # CircuitSelfAttention — dense reference impl
│   │   └── perceiver_attention.py # PerceiverCircuitAttention — superseded design
│   └── gnn/                       # CircuitGNN — sparse message-passing ablation
├── training/
│   ├── train_loop.py              # Main meta-training loop (pool + BPTT)
│   ├── evaluation.py              # Unified in/out-of-distribution evaluation
│   ├── demo_probe.py              # Post-training deploy-checkpoint scoring
│   └── pool/
│       ├── pool.py                 # Circuit pool (persistent meta-learning population)
│       ├── structural_perturbation.py  # Gate knockout / damage injection
│       └── perturbation.py         # Wire-level mutation / shuffle
└── utils/
    ├── graph_builder.py            # Circuit → graph construction
    ├── extraction.py               # LUT logit extraction, per-node error feedback
    └── positional_encoding.py      # Normalised-depth, DAG-distance, RWSE encodings
```

## Reproducing the Paper's Figures

See [`paper_figures/README.md`](paper_figures/README.md) — every figure's data
source (logged W&B scalars vs. a specific checkpoint), exact rebuild commands, and
the one honest "known gap" (above).

## Configuration System

Hydra-driven; `configs/config.yaml` is the single source of truth (model configs
interpolate from its `graph:` block so graph construction and the model's feature
dimensions always agree). Groups under `configs/`: `model/` (architecture),
`circuit/` (topology size), `tasks/` (target function), `loss/`.

## Citation

```bibtex
@article{barylli2026sodc,
    title   = {Self-Organising Digital Circuits},
    author  = {Barylli, Marcello and B\'{e}na, Gabriel and Mordvintsev, Alexander and Nisioti, Eleni and Risi, Sebastian},
    year    = {2026},
    journal = {Artificial Life Conference (ALIFE 2026)}
}
```

## License

MIT — see [LICENSE](LICENSE).
