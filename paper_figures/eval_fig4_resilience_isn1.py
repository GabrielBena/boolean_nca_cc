# ============================================================================
# Fig 4 runner: resilience / stepwise trajectories for run_id yu6kojmx (fixed wires)
# Lifted verbatim from boolean_nca_exploration.ipynb cells:
#   2,3,4,6,8 (imports) -> 9 (load config+ckpt) -> 11 (pool) ->
#   13,14 (task data + damage params) -> 15 (damage config) ->
#   17,18 (BP eval) -> 21 (load model) -> 22 (TMT eval)
# then append CSV writer.
# ============================================================================

# ---- cell 2 ----
import copy
import os

import numpy as np

# ---- cell 3 ----
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
force_cpu = False
if force_cpu:
    os.environ["JAX_PLATFORM_NAME"] = "cpu"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

# ---- cell 4 ----
import jax
import jax.numpy as jp
import jraph
import optax
from flax import nnx

print(jax.devices())

# ---- cell 6 ----
import logging

log = logging.getLogger(__name__)
from boolean_nca_cc.training.checkpointing import configure_notebook_logging

configure_notebook_logging()

# ---- cell 8 ----
from boolean_nca_cc.circuits.model import generate_layer_sizes
from boolean_nca_cc.circuits.tasks import get_task_data
from boolean_nca_cc.circuits.train import LossConfig
from boolean_nca_cc.circuits.viz import plot_wandb_stepwise_results
from boolean_nca_cc.training.checkpointing import (
    load_config_from_wandb,
    load_model_from_config_and_checkpoint,
)
from boolean_nca_cc.training.evaluation import (
    evaluate_model_stepwise_batched,
)
from boolean_nca_cc.training.pool.pool import initialize_graph_pool
from boolean_nca_cc.training.pool.structural_perturbation import compute_damage_params

# ---- cell 9 ----
run_id = "gbena/boolean-nca-cc/yu6kojmx"  # Fixed wires, damages, reverse task, big model
loaded_config, checkpoint_path, run_id = load_config_from_wandb(
    run_id=run_id.split("/")[-1],
    entity="gbena",
    filename="best_model_eval",
    use_cache=False,
)

# ---- cell 11 ----
case_n = 2**loaded_config.circuit.input_bits
base_key = jax.random.PRNGKey(loaded_config.seed)
eval_key = jax.random.fold_in(base_key, 1)
hidden_dim = loaded_config.circuit.circuit_hidden_dim
n_message_steps = loaded_config.training.n_message_steps
input_n, output_n = loaded_config.circuit.input_bits, loaded_config.circuit.output_bits
arity = loaded_config.circuit.arity
layer_n = loaded_config.circuit.num_layers

layer_sizes = generate_layer_sizes(
    input_n, output_n, arity, layer_n=layer_n, width_factor=loaded_config.circuit.width_factor
)
n_gates = sum(group_n for group_n, _ in layer_sizes)
print(f"n_gates: {n_gates}")

if loaded_config.training.wiring_mode == "fixed":
    pool_key = eval_key
else:
    pool_key = jax.random.fold_in(eval_key, 1)

wires_key, logits_key = jax.random.split(pool_key, 2)

batch_size = 32  # SMOKE_BATCH override below if env set

import os as _os
_smoke = _os.environ.get("SMOKE_BATCH")
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
print(f"Pool of size {batch_size} initialised ! Circuits are: {pool.graphs.n_node[0]} nodes")

# ---- cell 13 ----
from boolean_nca_cc.training.pool.structural_perturbation import (
    compute_p_fault_from_expected,
    count_eligible_gates,
)

# ---- cell 14 ----
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
mid_point = x.shape[0] // 2
x_plot, y_plot = (
    x[max(0, mid_point - 128) : min(x.shape[0], mid_point + 128)],
    y0[max(0, mid_point - 128) : min(x.shape[0], mid_point + 128)],
)
damage_params = compute_damage_params(loaded_config, layer_sizes, None)

# ---- cell 15 ----
damage_modes = ["none", "probabilistic", "deterministic"]
damage_permanence = [0.0, 1.0]
n_eligible_gates = count_eligible_gates(layer_sizes)
aim_failures = 0.1
n_steps = 256

n_damage_steps = 2
damage_steps = jp.linspace(0, n_steps, n_damage_steps + 1, endpoint=False).astype(int)[1:]
knockouts_per_event = int(n_eligible_gates * aim_failures / n_damage_steps)

p_fault_onset_step = 32
p_fault_eval = compute_p_fault_from_expected(
    int(n_eligible_gates * aim_failures), n_eligible_gates, (n_steps - p_fault_onset_step)
)

shuffle_wires = [False]
shuffle_wire_steps = damage_steps
shuffle_wire_key = jax.random.fold_in(eval_key, 2)
wire_shuffle_fraction = 0.05

# ---- cell 17 ----
from boolean_nca_cc.training.evaluation import run_bp_scan

# ---- cell 18 ----
run_bp_eval = True
if run_bp_eval:
    opt = optax.adamw(1, 0.8, 0.8, weight_decay=1e-1)
    bp_eval_results = {
        (damage_mode, permanent): run_bp_scan(
            opt=opt,
            batch_wires=pool.wires,
            batch_params=pool.logits,
            x_data=x_train,
            y_data=y_train,
            x_test=x_test,
            y_test=y_test,
            n_steps=n_steps,
            layer_sizes=layer_sizes,
            input_n=input_n,
            arity=arity,
            loss_cfg=LossConfig(loaded_config.loss),
            p_fault=damage_params["p_fault_eval"] if damage_mode == "probabilistic" else None,
            p_fault_onset_step=0,
            damage_mode=damage_mode,
            damage_steps=damage_steps,
            knockouts_per_event=damage_params["knockouts_per_event"],
            faulty_value=damage_params["faulty_logit_value"],
            permanent=permanent,
            use_scan=not force_cpu,
        )
        for damage_mode in damage_modes
        for permanent in damage_permanence
        if damage_mode != "none" or permanent != 1.0
    }
else:
    bp_eval_results = None

# ---- cell 21 ----
best_attn_model, loaded_dict, init_attn_model = load_model_from_config_and_checkpoint(
    config=loaded_config,
    checkpoint_path=checkpoint_path,
    run_id=run_id,
    seed=loaded_config.seed,
    n_node=int(pool.graphs.n_node[0].item()),
)

# ---- cell 22 ----
print(
    f"Starting eval with {n_damage_steps} damage steps, {knockouts_per_event} knockouts per step | failure rate: {p_fault_eval:1e}"
)
attn_eval_results = {
    (damage_mode, permanent, shuffle): evaluate_model_stepwise_batched(
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
        p_fault=p_fault_eval if damage_mode == "probabilistic" else None,
        bidirectional_edges=True,
        permanent_damage=permanent,
        verbose=True,
        compute_no_repair_baseline=True,
        p_fault_onset_step=p_fault_onset_step,
        wire_shuffle_steps=shuffle_wire_steps if shuffle else None,
        wire_shuffle_fraction=wire_shuffle_fraction if shuffle else None,
        wire_shuffle_key=shuffle_wire_key,
    )
    for damage_mode in damage_modes
    for permanent in damage_permanence
    for shuffle in shuffle_wires
    if (damage_mode != "none" or permanent != 1.0) and not (shuffle and damage_mode != "none")
}

# ============================================================================
# CSV writer (appended; not from notebook)
# ============================================================================
import csv

# Inspect step_metrics structure for the two damage kinds the paper shows
def _describe(d, label):
    print(f"\n=== step_metrics keys for {label} ===")
    for k, v in d.items():
        if isinstance(v, list):
            arr = np.asarray(v)
            print(f"  {k}: list len={len(v)} shape={arr.shape} dtype={arr.dtype}")
        elif isinstance(v, dict):
            print(f"  {k}: dict keys={list(v.keys())}")
            for kk, vv in v.items():
                try:
                    print(f"      {kk}: shape={np.asarray(vv).shape}")
                except Exception:
                    print(f"      {kk}: <{type(vv).__name__}>")
        else:
            print(f"  {k}: <{type(v).__name__}>")

tmt_det = attn_eval_results[("deterministic", 1.0, False)][1]
tmt_prob = attn_eval_results[("probabilistic", 1.0, False)][1]
bp_det = bp_eval_results[("deterministic", 1.0)][1]
bp_prob = bp_eval_results[("probabilistic", 1.0)][1]

_describe(tmt_det, "TMT deterministic(shotgun) permanent")
_describe(bp_det, "BP deterministic(shotgun) permanent")

# damage_kind -> (tmt_metrics, bp_metrics)
mapping = {
    "shotgun": (tmt_det, bp_det),
    "stochastic": (tmt_prob, bp_prob),
}

rows = []
for damage_kind, (tmt, bp) in mapping.items():
    n = len(tmt["step"])  # 256
    # TMT (NCA) series: plotted hard_loss + hard_accuracy; also emit soft loss
    for step in range(n):
        rows.append(dict(
            damage_kind=damage_kind, series="TMT", step=step,
            loss=float(tmt["loss"][step]),
            hard_loss=float(tmt["hard_loss"][step]),
            hard_accuracy=float(tmt["hard_accuracy"][step]),
        ))
    # No-repair baseline (lives inside the TMT dict)
    if "no_repair_hard_accuracy" in tmt:
        for step in range(n):
            rows.append(dict(
                damage_kind=damage_kind, series="no_repair", step=step,
                loss=float(tmt["no_repair_loss"][step]),
                hard_loss=float(tmt["no_repair_hard_loss"][step]),
                hard_accuracy=float(tmt["no_repair_hard_accuracy"][step]),
            ))
    else:
        print(f"WARNING: no no_repair baseline in TMT dict for {damage_kind}")
    # BP series
    nb = len(bp["step"])
    for step in range(nb):
        rows.append(dict(
            damage_kind=damage_kind, series="BP", step=step,
            loss=float(bp["loss"][step]),
            hard_loss=float(bp["hard_loss"][step]),
            hard_accuracy=float(bp["hard_accuracy"][step]),
        ))

out_path = os.path.join(os.path.dirname(__file__), "data", "fig4_stepwise.csv")
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["damage_kind", "series", "step", "loss", "hard_loss", "hard_accuracy"])
    w.writeheader()
    w.writerows(rows)

print(f"\nWROTE {out_path} with {len(rows)} rows")
# Sanity printout
for damage_kind, (tmt, bp) in mapping.items():
    print(f"[{damage_kind}] TMT hard_acc: start={tmt['hard_accuracy'][0]:.3f} "
          f"min={min(tmt['hard_accuracy']):.3f} final={tmt['hard_accuracy'][-1]:.3f}")
    print(f"[{damage_kind}] BP  hard_acc: final={bp['hard_accuracy'][-1]:.3f}")
    if 'no_repair_hard_accuracy' in tmt:
        print(f"[{damage_kind}] no_repair hard_acc: final={tmt['no_repair_hard_accuracy'][-1]:.3f}")
