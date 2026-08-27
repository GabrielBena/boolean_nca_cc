# ============================================================================
# PCA-trajectories runner: reconstructs "Fig 3" (fig:pca_trajectories) --
# ported from Gabriel's own trained_models.ipynb (cells 8, 9, 11, 23, 24,
# 36-42), which explored this on wandb run 6mo8q61y (random wiring, reverse
# task, gathered_attention/TMT, dist_pe -- confirmed via the run's cached
# config). Unlike Marcello's mergello-only figures (see paper_figures/README.md
# "Known Gaps"), this one is Gabriel's own model/run, so it's ported here as a
# real, reproducible script rather than left as a gap.
#
# HONESTY NOTE: the notebook's own exploration used placeholder knobs that
# don't match the submitted figure (damage_permanence=[0.5], hue=hard_accuracy
# instead of step). This script reconstructs the published figure's caption
# instead: three conditions -- none / recoverable-shotgun (permanent=0.0) /
# permanent-shotgun (permanent=1.0) -- with trajectories coloured by step, not
# accuracy. Treat this as a faithful best-effort reconstruction of the
# published figure, not a byte-identical replay.
# ============================================================================
import csv
import os

import numpy as np

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
force_cpu = False
if force_cpu:
    os.environ["JAX_PLATFORM_NAME"] = "cpu"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

import jax
import jax.numpy as jp

print(jax.devices())

import logging

log = logging.getLogger(__name__)
from boolean_nca_cc.training.checkpointing import configure_notebook_logging

configure_notebook_logging()

from boolean_nca_cc.circuits.model import generate_layer_sizes
from boolean_nca_cc.circuits.tasks import get_task_data
from boolean_nca_cc.circuits.train import LossConfig
from boolean_nca_cc.training.checkpointing import load_model_from_config_and_checkpoint
from boolean_nca_cc.training.evaluation import evaluate_model_stepwise_batched
from boolean_nca_cc.training.pool.pool import initialize_graph_pool
from boolean_nca_cc.training.pool.structural_perturbation import (
    count_eligible_gates,
)
from boolean_nca_cc.utils.configured_graph_builder import configure_build_graph
from paper_figures.local_checkpoints import load_config_and_checkpoint

RUN_ID_SHORT = os.environ.get("PCA_RUN_ID", "6mo8q61y")  # random wires 14x14, reverse

loaded_config, checkpoint_path, run_id = load_config_and_checkpoint(
    run_id=RUN_ID_SHORT,
    entity="gbena",
    filename="best_model_eval_damaged_out_test_hard_accuracy",
    use_cache=False,
)

# Must match the checkpoint's graph feature config exactly (dist_pe/rwse change the
# model's feature_proj input dim) -- this run trained with use_dist_pe=True.
configure_build_graph(
    neighboring_connections=loaded_config.graph.neighboring_connections,
    bidirectional_edges=loaded_config.graph.bidirectional_edges,
    use_dist_pe=loaded_config.graph.get("use_dist_pe", False),
    use_rwse=loaded_config.graph.get("use_rwse", False),
    rwse_k=loaded_config.graph.get("rwse_k", 8),
)

case_n = 2**loaded_config.circuit.input_bits
base_key = jax.random.PRNGKey(loaded_config.seed)
eval_key = jax.random.fold_in(base_key, 1)
hidden_dim = loaded_config.circuit.circuit_hidden_dim
input_n, output_n = loaded_config.circuit.input_bits, loaded_config.circuit.output_bits
arity = loaded_config.circuit.arity
layer_n = loaded_config.circuit.num_layers

layer_sizes = generate_layer_sizes(
    input_n, output_n, arity, layer_n=layer_n, width_factor=loaded_config.circuit.width_factor
)
n_gates = sum(group_n for group_n, _ in layer_sizes)
print(f"n_gates: {n_gates}")

pool_key = eval_key if loaded_config.training.wiring_mode == "fixed" else jax.random.fold_in(eval_key, 1)
wires_key, logits_key = jax.random.split(pool_key, 2)

batch_size = 16
_smoke = os.environ.get("SMOKE_BATCH")
if _smoke:
    batch_size = int(_smoke)

pool = initialize_graph_pool(
    wires_key=wires_key,
    logits_key=logits_key,
    pool_size=batch_size,
    layer_sizes=layer_sizes,
    arity=arity,
    input_n=input_n,
    circuit_hidden_dim=hidden_dim,
    wiring_mode=loaded_config.training.wiring_mode,
    noise_scale=0.0,
)
print(f"Pool of size {batch_size} initialised! Circuits are: {pool.graphs.n_node[0]} nodes")

# Schema migration: task moved from circuit.task/circuit.text to tasks.name/
# tasks.text partway through the project's history -- handle both eras.
if "tasks" in loaded_config:
    task_name, task_text = loaded_config.tasks.name, loaded_config.tasks.get("text", None)
else:
    task_name, task_text = loaded_config.circuit.task, loaded_config.circuit.get("text", None)

(x_train, y_train), (x_test, y_test), (x, y0) = get_task_data(
    task_name,
    case_n,
    input_bits=loaded_config.circuit.input_bits,
    output_bits=loaded_config.circuit.output_bits,
    text=task_text,
    train_test_split=True,
    test_ratio=loaded_config.training.test_num / case_n,
    seed=eval_key,
)

n_eligible_gates = count_eligible_gates(layer_sizes)
aim_failures = 0.10  # matches the paper's 10%-of-gates "shotgun" protocol
n_steps = 256
n_damage_steps = 2
damage_steps = jp.linspace(0, n_steps, n_damage_steps + 1, endpoint=False).astype(int)[1:]
knockouts_per_event = int(n_eligible_gates * aim_failures / n_damage_steps)
p_fault_onset_step = 32

best_attn_model, _, _ = load_model_from_config_and_checkpoint(
    config=loaded_config,
    checkpoint_path=checkpoint_path,
    run_id=run_id,
    seed=loaded_config.seed,
    n_node=int(pool.graphs.n_node[0].item()),
)

# Three conditions, matching the published caption: no damage / recoverable
# shotgun (permanent=0.0) / permanent shotgun (permanent=1.0). All three use
# the same deterministic "shotgun" delivery for a controlled comparison.
CONDITIONS = {
    "none": {"damage_mode": "none", "permanent": 1.0},
    "recoverable": {"damage_mode": "deterministic", "permanent": 0.0},
    "permanent": {"damage_mode": "deterministic", "permanent": 1.0},
}

eval_results = {}
for label, cfg in CONDITIONS.items():
    damage_mode = cfg["damage_mode"]
    eval_results[label] = evaluate_model_stepwise_batched(
        model=best_attn_model,
        batch_wires=pool.wires,
        batch_logits=pool.logits,
        x_data=x_test,
        y_data=y_test,
        input_n=input_n,
        n_message_steps=n_steps,
        arity=arity,
        circuit_hidden_dim=hidden_dim,
        loss_cfg=LossConfig(dict(loaded_config.loss)),
        layer_sizes=layer_sizes,
        damage_steps=damage_steps if damage_mode == "deterministic" else None,
        knockout_per_damage_step=knockouts_per_event if damage_mode == "deterministic" else None,
        p_fault=None,
        bidirectional_edges=True,
        permanent_damage=cfg["permanent"],
        verbose=True,
        compute_no_repair_baseline=False,
        p_fault_onset_step=p_fault_onset_step,
    )

# ============================================================================
# Build a tidy (batch, step, condition, logits_vec) table, fit joint PCA,
# write PC1/PC2 + step + condition to CSV (small; the fig script just plots).
# ============================================================================
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

rows = []
for label, (_, step_metrics) in eval_results.items():
    logits_traj = np.array(step_metrics["graphs"].nodes["logits"])
    b_size, n_s, _, _ = logits_traj.shape
    logits_flat = logits_traj.reshape(b_size, n_s, -1)
    for b in range(b_size):
        for t in range(n_s):
            rows.append({"condition": label, "batch": b, "step": t, "logits_vec": logits_flat[b, t]})

X = np.stack([r["logits_vec"] for r in rows])
X_scaled = StandardScaler().fit_transform(X)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
print(f"PCA on {X.shape[0]} points, feature_dim={X.shape[1]}")
print(f"Explained variance: {pca.explained_variance_ratio_} (total {pca.explained_variance_ratio_.sum():.2%})")

out_path = os.path.join(os.path.dirname(__file__), "data", "pca_trajectories.csv")
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["condition", "batch", "step", "PC1", "PC2"])
    w.writeheader()
    for r, (pc1, pc2) in zip(rows, X_pca, strict=True):
        w.writerow({"condition": r["condition"], "batch": r["batch"], "step": r["step"],
                    "PC1": float(pc1), "PC2": float(pc2)})

print(f"WROTE {out_path} with {len(rows)} rows")
