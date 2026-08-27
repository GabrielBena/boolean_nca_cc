# ============================================================================
# Fig 10 runner: scale-free experiment. Lifted from boolean_nca_exploration.ipynb
#   cells 41,42 (imports + load config/ckpt) -> 43 (layer_sizes + pools per width)
#   -> 44 (task data) -> 45 (load model per pool) -> 46 (eval per width,damage)
#   -> 47 (assemble per-batch final-step hard_accuracy).
# Runs once per run_id; appends rows tagged with `wiring`. The CSV is written
# fresh by the FIRST run_id and appended-to by subsequent ones (mode controlled
# by env FIG10_MODE = "w" | "a").
#
# Env knobs:
#   FIG10_RUN_ID   : wandb run id. REQUIRED. Run twice, once per wiring mode:
#     1u5ssulx -- random-wiring-trained (cross-referenced against
#                 web_demo/configs/demo_models.yaml's "random_damage"/Regime III
#                 recipe entry, which pins this same run id; high confidence but
#                 not directly confirmed against the wandb config itself)
#     cdjkgrod -- fixed-wiring-trained (the other candidate; by elimination)
#   FIG10_MODE     : "w" (write+header, use for the FIRST run) or "a" (append,
#                    use for the SECOND run). default "w".
#   SMOKE_WIDTHS   : optional comma list to restrict widths (smoke test).
#   SMOKE_BATCH    : optional int to shrink batch_size (smoke test).
# ============================================================================
import os
import copy

import numpy as np

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
force_cpu = False
if force_cpu:
    os.environ["JAX_PLATFORM_NAME"] = "cpu"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

# ---- cell 41 ----
import pandas as pd
import jax
import jax.numpy as jp

print(jax.devices())

# logging + imports (mirrors cells 6, 8 needed in this namespace)
import logging
log = logging.getLogger(__name__)
from boolean_nca_cc.training.checkpointing import configure_notebook_logging
configure_notebook_logging()

from boolean_nca_cc.circuits.model import generate_layer_sizes
from boolean_nca_cc.circuits.tasks import get_task_data
from boolean_nca_cc.circuits.train import LossConfig
from boolean_nca_cc.training.checkpointing import (
    load_config_from_wandb,
    load_model_from_config_and_checkpoint,
)
from boolean_nca_cc.training.evaluation import evaluate_model_stepwise_batched
from boolean_nca_cc.training.pool.pool import initialize_graph_pool
from boolean_nca_cc.training.pool.structural_perturbation import compute_damage_params

# ---- cell 42 ----
RUN_ID_SHORT = os.environ["FIG10_RUN_ID"]
run_id = f"gbena/boolean-nca-cc/{RUN_ID_SHORT}"
loaded_config, checkpoint_path, run_id = load_config_from_wandb(
    run_id=run_id.split("/")[-1],
    entity="gbena",
    filename="eval_damaged",
    use_cache=False,
)

# ---- cell 43 ----
case_n = 2**loaded_config.circuit.input_bits
base_key = jax.random.PRNGKey(loaded_config.seed)
eval_key = jax.random.fold_in(base_key, 1)
hidden_dim = loaded_config.circuit.circuit_hidden_dim
n_message_steps = loaded_config.training.n_message_steps
input_n, output_n = loaded_config.circuit.input_bits, loaded_config.circuit.output_bits
arity = loaded_config.circuit.arity

layer_widths = [1, 1.5, 2, 2.5, 3, 3.5, 4]
_smoke_w = os.environ.get("SMOKE_WIDTHS")
if _smoke_w:
    layer_widths = [float(x) for x in _smoke_w.split(",")]
# Memory-isolation knob: run a single width per process so GPU memory is
# reclaimed on process exit between widths. Does NOT change any eval logic.
_single_w = os.environ.get("FIG10_WIDTH")
if _single_w:
    layer_widths = [float(_single_w)]

layer_sizes = {
    width_factor: generate_layer_sizes(
        input_n,
        output_n,
        arity,
        layer_n=loaded_config.circuit.num_layers,
        width_factor=width_factor,
    )
    for width_factor in layer_widths
}

if loaded_config.training.wiring_mode == "fixed":
    pool_key = eval_key
else:
    pool_key = jax.random.fold_in(eval_key, 1)

wires_key, logits_key = jax.random.split(pool_key, 2)

batch_size = 64
_smoke_b = os.environ.get("SMOKE_BATCH")
if _smoke_b:
    batch_size = int(_smoke_b)

pools = {
    width_factor: initialize_graph_pool(
        wires_key=wires_key,
        logits_key=logits_key,
        pool_size=batch_size,
        layer_sizes=layer_sizes[width_factor],
        arity=arity,
        input_n=input_n,
        circuit_hidden_dim=hidden_dim,
        wiring_mode=loaded_config.training.wiring_mode,
        noise_scale=0.1,
    )
    for width_factor in layer_widths
}

# Damage params are defined w.r.t. the TRAINING width_factor; compute its
# layer_sizes standalone so this never depends on which widths we iterate.
_train_layer_sizes = generate_layer_sizes(
    input_n,
    output_n,
    arity,
    layer_n=loaded_config.circuit.num_layers,
    width_factor=loaded_config.circuit.width_factor,
)
damage_params = compute_damage_params(loaded_config, _train_layer_sizes, None)

# ---- cell 44 ----
(x_train, y_train), (x_test, y_test), (x, y0) = get_task_data(
    loaded_config.circuit.task,
    case_n,
    input_bits=loaded_config.circuit.input_bits,
    output_bits=loaded_config.circuit.output_bits,
    text=loaded_config.circuit.text,
    train_test_split=True,
    test_ratio=loaded_config.training.test_num / case_n,
    seed=eval_key,
)

# ---- cell 45 ----
best_attn_models = {
    key: load_model_from_config_and_checkpoint(
        config=loaded_config,
        checkpoint_path=checkpoint_path,
        run_id=run_id,
        seed=loaded_config.seed,
        n_node=int(pool.graphs.n_node[0].item()),
    )[0]
    for key, pool in pools.items()
}

# ---- cells 46 + 47 (fused, memory-frugal streaming) ----
# Original notebook built attn_eval_results as a dict over ALL (width,damage),
# holding every heavy result simultaneously, then cell 47 read final-step
# all_metrics. For large circuits that OOMs the GPU. Here we eval ONE
# (width, damage) at a time, immediately pull the FINAL-STEP metrics to host
# (identical to cell 47's step_metrics["all_metrics"][m][batch_idx, -1]), then
# free the heavy result + clear JAX caches before the next eval. Eval call and
# extraction semantics are byte-for-byte the same.
all_results = []
for key in pools:
    for damage in [True, False]:
        final_graphs, step_metrics = evaluate_model_stepwise_batched(
            model=best_attn_models[key],
            batch_wires=pools[key].wires,
            batch_logits=pools[key].logits,
            x_data=x_test,
            y_data=y_test,
            input_n=input_n,
            n_message_steps=256,
            arity=arity,
            circuit_hidden_dim=hidden_dim,
            loss_cfg=LossConfig(dict(loaded_config.loss)),
            layer_sizes=layer_sizes[key],
            damage_steps=None,
            knockout_per_damage_step=None,
            p_fault=damage_params["p_fault_eval"] if damage else None,
            bidirectional_edges=True,
            permanent_damage=True,
            verbose=True,
            compute_no_repair_baseline=False,
            p_fault_onset_step=0,
        )
        width_factor = key
        damage_active = damage
        n_node = int(final_graphs.n_node[0].item())
        # Pull FINAL-STEP metrics to host numpy once (shape [batch_size]).
        am = step_metrics["all_metrics"]
        final_vals = {
            m: np.asarray(am[m][:, -1]) for m in ["hard_accuracy", "hard_loss", "accuracy", "loss"]
        }
        for batch_idx in range(batch_size):
            row = {
                "width_factor": width_factor,
                "n_node": n_node,
                "damage_active": damage_active,
                "batch_idx": batch_idx,
            }
            for m in ["hard_accuracy", "hard_loss", "accuracy", "loss"]:
                row[m] = float(final_vals[m][batch_idx])
            all_results.append(row)
        # Free heavy JAX-resident result and reclaim device memory.
        del final_graphs, step_metrics, am, final_vals
        jax.clear_caches()

# ============================================================================
# CSV writer (appended; not from notebook)
# ============================================================================
import csv

wiring = "random" if loaded_config.training.wiring_mode == "random" else "fixed"
print(f"wiring_mode resolved to: {loaded_config.training.wiring_mode} -> column '{wiring}'")

out_path = os.path.join(os.path.dirname(__file__), "data", "fig10_scalefree.csv")
os.makedirs(os.path.dirname(out_path), exist_ok=True)
mode = os.environ.get("FIG10_MODE", "w")
fieldnames = ["wiring", "width_factor", "n_node", "damage_active", "batch_idx", "hard_accuracy"]

write_header = (mode == "w")
with open(out_path, mode, newline="") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    if write_header:
        w.writeheader()
    for r in all_results:
        w.writerow({
            "wiring": wiring,
            "width_factor": r["width_factor"],
            "n_node": r["n_node"],
            "damage_active": r["damage_active"],
            "batch_idx": r["batch_idx"],
            "hard_accuracy": r["hard_accuracy"],
        })

print(f"\n{'WROTE' if write_header else 'APPENDED'} {len(all_results)} rows to {out_path} (mode={mode}, wiring={wiring})")

# Sanity: per (n_node, damage) mean hard_accuracy
import collections
agg = collections.defaultdict(list)
for r in all_results:
    agg[(r["n_node"], r["damage_active"])].append(r["hard_accuracy"])
print(f"--- {wiring} mean hard_accuracy by (n_node, damage_active) ---")
for (nn, da), vals in sorted(agg.items()):
    print(f"  n_node={nn:>4} damage={da}: mean={np.mean(vals):.3f} (n={len(vals)})")
