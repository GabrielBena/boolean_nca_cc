"""
Batched demo-probe: score a trained model on the LIVE-DEMO objective.

The web_demo/export probes (ceiling_probe / probe_shuffle_models /
curate_topologies) loop one topology at a time through
``evaluate_model_stepwise_generator`` — minutes per checkpoint. This module
reproduces their semantics with batched/vmapped rollouts (the same
``nnx.vmap(run_model_scan_with_loss)`` pattern as
``evaluate_model_stepwise_batched``), so a full probe takes seconds on GPU and
can run inside the training job itself (see the ``demo_probe`` config block +
the post-training hook in train.py).

Two metric families:

1. CLEAN SETTLE over N random topologies (the demo's boot regime):
   - ``demo_settled_*``   tail-window mean hard accuracy (the "settled" acc)
   - ``demo_tail_std``    tail accuracy std — a limit cycle shows up here
   - ``demo_flip_rate``   per-step output-bit churn in the tail (visible jitter)
   - ``demo_best_of_K``   order-stats curation headroom (best-of-K topologies)
   - ``demo_peak_step_*`` where in the rollout accuracy peaks (peak >> train
                          horizon means the model never reaches a fixed point)

2. CARRY RECOVERY over P topology pairs — the full 2x2 of WHAT SURVIVES the
   shuffle (logits x hidden), settling on fresh topology B after settling on A:

   | quadrant     | logits  | hidden | meaning                                  |
   |--------------|---------|--------|------------------------------------------|
   | carry_none   | fresh   | zero   | B's own ceiling (``demo_fresh_*``)       |
   | carry_logits | carried | zero   | the DEMO BUTTON semantics (legacy        |
   |              |         |        | ``demo_carry_*`` keys, kept as aliases)  |
   | carry_hidden | fresh   | carried| is working memory alone useful?          |
   | carry_both   | carried | carried| persistence semantics — what training-   |
   |              |         |        | time shuffle experiences (nothing reset) |

   The carry_logits-vs-carry_both gap decomposes each model's recovery into
   logits-borne vs hidden-borne — e.g. shuffle-trained models are expected to
   lean on hidden persistence (which the demo button never grants them).
   - ``demo_carry_dip/settled/std``    carry_logits quadrant (legacy names)
   - ``demo_carry_both_*``             persistence quadrant
   - ``demo_carry_hidden_*``           hidden-only quadrant
   - ``demo_fresh_settled``            carry_none ceiling
   - ``demo_penalty``                  fresh - carry_logits (legacy)
   - ``demo_carry_both_penalty``       fresh - carry_both

3. SHOTGUN RECOVERY over the same N topologies — the demo's OTHER big OOD
   button (added 2026-06-06 after it turned out the suite was shuffle-only).
   PAPER PROTOCOL: ``shotgun_volleys`` (2) volleys totalling
   ``damage.target_damage_fraction`` (10%) of the hidden gates — 2 x 12 on the
   12-bit circuit — knocked out PERMANENTLY (logits clamped faulty + updates
   masked, exactly the demo button) after a clean ``steps``-tick settle, with
   ``steps`` more ticks to re-route around the dead gates.
   - ``demo_shotgun_pre``        settled acc just before the volley
   - ``demo_shotgun_dip``        worst acc after the volley
   - ``demo_shotgun_settled``    recovered settled acc (tail window)
   - ``demo_shotgun_penalty``    pre - settled = the UN-recovered residual
   - ``demo_shotgun_flip_rate``  churn at the post-damage fixed point

All metrics are plain floats so they drop straight into final_results.csv and
``final/demo_*`` wandb keys.

Parity (verified 2026-06-03 on demo_12 task_004/filut2ch): ``batched_settle``
is BIT-IDENTICAL to the unbatched ``run_model_scan_with_loss`` (accs and
logits, vmap introduces zero difference). The legacy generator probes diverge
from the scan path by recurrent float drift only (identical first ~7 steps,
then hard-acc diffs <=~0.002 over 20 steps; logit max-drift grows with rollout
length) — same dynamics, different XLA fusion. Rankings produced by this
module are therefore self-consistent and demo-faithful.
"""

from __future__ import annotations

import logging
from math import comb

import jax
import jax.numpy as jp
import numpy as np
from flax import nnx

from boolean_nca_cc.circuits.train import LossConfig
from boolean_nca_cc.training.evaluation import run_model_scan_with_loss
from boolean_nca_cc.training.pool.pool import get_wires_and_logits
from boolean_nca_cc.utils.configured_graph_builder import configured_build_graph

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Batched rollout
# ---------------------------------------------------------------------------


def batched_settle(
    model,
    batch_wires: list[jp.ndarray],
    batch_logits: list[jp.ndarray],
    x_data: jp.ndarray,
    y_data: jp.ndarray,
    *,
    steps: int,
    layer_sizes: list[tuple[int, int]],
    input_n: int,
    arity: int,
    circuit_hidden_dim: int,
    loss_cfg: LossConfig,
    bidirectional_edges: bool = True,
    settle_window: int = 8,
    chunk_size: int | None = 32,
    initial_hidden: jp.ndarray | None = None,
    damage_steps: list[int] | None = None,
    knockout_per_damage_step: int = 0,
    scan_keys: jp.ndarray | None = None,
) -> dict:
    """Roll out ``steps`` NCA ticks on B circuits at once.

    Mirrors ``evaluate_model_stepwise_batched`` (graph build + nnx.vmap over
    ``run_model_scan_with_loss``) but keeps PER-TOPOLOGY data instead of
    batch-averaging, and slices the tail window needed for jitter metrics.
    Clean rollout by default; ``damage_steps`` + ``knockout_per_damage_step``
    arm the scan's discrete-volley machinery (PERMANENT knockouts — the demo's
    shotgun button semantics) with per-topology ``scan_keys`` so each circuit
    draws its own gate set.

    NOTE: graphs are built with ``configured_build_graph`` — callers must have
    configured the global builder (train.py does this at startup; standalone
    callers go through ``ceiling_probe.load_model_and_cfg``) or dist_pe/rwse
    models will see mismatched node features.

    ``initial_hidden`` ([B, n_nodes, hidden_dim]) overrides the zero-initialized
    hidden state after the graph build (for the carry_hidden/carry_both probe
    quadrants — node identity is preserved under rewiring, so the carry-over
    mapping is per-node identity).

    Returns dict with:
        hard_acc:     np [B, S]   per-step hard accuracy per topology
        tail_preds:   np [B, W, cases, bits] hard predictions over the tail
        final_logits: list of jp [B, ...layer shape] last-step evolved logits
        final_hidden: jp [B, n_nodes, hidden_dim] last-step hidden state
    """
    batch_size = int(batch_logits[0].shape[0])
    settle_window = min(settle_window, steps)

    # --- chunk for memory (forward-only, but the scan stacks per-step outputs)
    if chunk_size is not None and batch_size > chunk_size:
        outs = []
        for i in range(0, batch_size, chunk_size):
            end = min(i + chunk_size, batch_size)
            outs.append(
                batched_settle(
                    model,
                    [w[i:end] for w in batch_wires],
                    [lg[i:end] for lg in batch_logits],
                    x_data,
                    y_data,
                    steps=steps,
                    layer_sizes=layer_sizes,
                    input_n=input_n,
                    arity=arity,
                    circuit_hidden_dim=circuit_hidden_dim,
                    loss_cfg=loss_cfg,
                    bidirectional_edges=bidirectional_edges,
                    settle_window=settle_window,
                    chunk_size=None,
                    initial_hidden=None if initial_hidden is None else initial_hidden[i:end],
                    damage_steps=damage_steps,
                    knockout_per_damage_step=knockout_per_damage_step,
                    scan_keys=None if scan_keys is None else scan_keys[i:end],
                )
            )
        return {
            "hard_acc": np.concatenate([o["hard_acc"] for o in outs], axis=0),
            "tail_preds": np.concatenate([o["tail_preds"] for o in outs], axis=0),
            "final_logits": [
                jp.concatenate([o["final_logits"][i] for o in outs], axis=0)
                for i in range(len(outs[0]["final_logits"]))
            ],
            "final_hidden": jp.concatenate([o["final_hidden"] for o in outs], axis=0),
        }

    # --- build initial graphs (vectorized; same call shape as
    #     evaluate_model_stepwise_batched, which the periodic eval has proven
    #     correct for dist_pe/rwse models)
    vmap_build_graph = jax.vmap(
        lambda logits, wires: configured_build_graph(
            logits,
            wires,
            input_n,
            arity,
            circuit_hidden_dim,
            loss_value=0.0,
            bidirectional_edges=bidirectional_edges,
        )
    )
    batch_graphs = vmap_build_graph(batch_logits, batch_wires)
    if initial_hidden is not None:
        # carry_hidden / carry_both quadrants: override the zero-init hidden
        batch_graphs = batch_graphs._replace(
            nodes={**batch_graphs.nodes, "hidden": initial_hidden}
        )

    damage_on = damage_steps is not None and knockout_per_damage_step > 0

    def run_single(graph, wires, logits, scan_key):
        final_graph, step_outputs = run_model_scan_with_loss(
            model=model,
            graph=graph,
            num_steps=steps,
            logits_original_shapes=[tuple(lg.shape) for lg in logits],
            wires=wires,
            x_data=x_data,
            y_data=y_data,
            loss_cfg=loss_cfg,
            layer_sizes=layer_sizes,
            data_fraction=1.0,
            # data_fraction=1.0 leaves the data path untouched: scan_key only
            # seeds the (per-topology) damage key on the shotgun path.
            scan_key=scan_key,
            gradient_checkpointing=False,
            damage_steps=damage_steps if damage_on else None,
            knockout_per_damage_step=knockout_per_damage_step if damage_on else 1,
            permanent_damage=1.0,  # demo shotgun: gates stay dead
        )
        _graphs, _losses, step_logits, aux = step_outputs
        hard_acc = aux["hard_accuracy"]  # [S]
        tail_preds = aux["hard_predictions"][-settle_window:]  # [W, cases, bits]
        final_logits = [sl[-1] for sl in step_logits]  # evolved logits, last step
        final_hidden = _graphs.nodes["hidden"][-1]  # [n_nodes, hidden_dim]
        return hard_acc, tail_preds, final_logits, final_hidden

    if damage_on:
        if scan_keys is None:
            raise ValueError("damage_steps requires per-topology scan_keys")
        hard_acc, tail_preds, final_logits, final_hidden = nnx.vmap(
            run_single, in_axes=(0, 0, 0, 0)
        )(batch_graphs, batch_wires, batch_logits, scan_keys)
    else:
        hard_acc, tail_preds, final_logits, final_hidden = nnx.vmap(
            run_single, in_axes=(0, 0, 0, None)
        )(batch_graphs, batch_wires, batch_logits, None)

    return {
        "hard_acc": np.asarray(hard_acc),
        "tail_preds": np.asarray(tail_preds),
        "final_logits": list(final_logits),
        "final_hidden": final_hidden,
    }


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------


def _expected_best_of_k(settled: np.ndarray, k: int) -> float:
    """E[max of a random k-subset] via order statistics on the empirical sample
    (same math as curate_topologies.py / ceiling_probe.py)."""
    n = len(settled)
    if k > n:
        return float("nan")
    srt = np.sort(settled)[::-1]
    return float(sum(srt[i] * (comb(n - 1 - i, k - 1) / comb(n, k)) for i in range(n - k + 1)))


def _tail_flip_rate(tail_preds: np.ndarray) -> np.ndarray:
    """Mean fraction of (case, bit) hard predictions flipping between
    consecutive tail steps, per topology. tail_preds: [B, W, cases, bits]."""
    if tail_preds.shape[1] < 2:
        return np.zeros(tail_preds.shape[0])
    flips = tail_preds[:, 1:] != tail_preds[:, :-1]
    return flips.mean(axis=(1, 2, 3))


# ---------------------------------------------------------------------------
# The probe
# ---------------------------------------------------------------------------


def demo_probe(
    model,
    cfg,
    x_data: jp.ndarray,
    y_data: jp.ndarray,
    layer_sizes: list[tuple[int, int]],
    *,
    key: jax.random.PRNGKey,
    n_topologies: int = 64,
    n_pairs: int = 16,
    steps: int = 64,
    settle_window: int = 8,
    chunk_size: int | None = 32,
    shotgun_gates: int | None = None,
    shotgun_volleys: int = 2,
) -> dict[str, float]:
    """Compute the demo metric suite for ``model`` (see module docstring).

    The first ``n_pairs`` clean topologies double as the "A" side of the
    recovery pairs (their evolved final logits are carried into fresh "B"
    topologies), so the clean screen and the recovery probe share rollouts.

    Args:
        model: trained nnx model (graph builder must already be configured).
        cfg: the run's Hydra config (circuit/pool/loss params are read here).
        x_data, y_data: evaluation cases (e.g. the held-out test split).
        layer_sizes: circuit layer sizes incl. input layer.
        key: PRNG key (topology generation only — rollouts are deterministic).

    Returns:
        Flat dict of float metrics, all keys prefixed ``demo_``.
    """
    n_pairs = min(n_pairs, n_topologies)
    input_n = int(cfg.circuit.input_bits)
    arity = int(cfg.circuit.arity)
    circuit_hidden_dim = int(cfg.model.circuit_hidden_dim)
    loss_cfg = LossConfig.from_dict(dict(cfg.loss))
    bidirectional_edges = bool(cfg.graph.bidirectional_edges)
    noise_scale = float(cfg.pool.noise_scale)
    init_logits = str(cfg.circuit.get("init_logits", "soft_wires"))
    random_init_scale = float(cfg.circuit.get("random_init_scale", 1.0))

    settle_kwargs = dict(
        steps=steps,
        layer_sizes=layer_sizes,
        input_n=input_n,
        arity=arity,
        circuit_hidden_dim=circuit_hidden_dim,
        loss_cfg=loss_cfg,
        bidirectional_edges=bidirectional_edges,
        settle_window=settle_window,
        chunk_size=chunk_size,
    )

    # --- generate topologies: N clean (first P double as pair-side A) + P "B"
    key_a, key_b, lkey_a, lkey_b = jax.random.split(key, 4)
    gen_kwargs = dict(
        layer_sizes=layer_sizes,
        arity=arity,
        noise_scale=noise_scale,
        wiring_mode="random",
        init_logits=init_logits,
        random_init_scale=random_init_scale,
    )
    wires_a, logits_a = get_wires_and_logits(key_a, lkey_a, pool_size=n_topologies, **gen_kwargs)
    wires_b, logits_b = get_wires_and_logits(key_b, lkey_b, pool_size=n_pairs, **gen_kwargs)

    # --- clean settle over the N topologies ---------------------------------
    clean = batched_settle(model, wires_a, logits_a, x_data, y_data, **settle_kwargs)
    acc = clean["hard_acc"]  # [N, S]
    tail_acc = acc[:, -settle_window:]
    settled = tail_acc.mean(axis=1)  # [N]
    tail_std = tail_acc.std(axis=1)  # [N]
    flip_rate = _tail_flip_rate(clean["tail_preds"])  # [N]
    peak_step = acc.argmax(axis=1) + 1  # 1-indexed like StepResult

    metrics: dict[str, float] = {
        "demo_settled_mean": float(settled.mean()),
        "demo_settled_median": float(np.median(settled)),
        "demo_settled_min": float(settled.min()),
        "demo_settled_max": float(settled.max()),
        "demo_settled_p90": float(np.percentile(settled, 90)),
        "demo_tail_std": float(tail_std.mean()),
        "demo_flip_rate": float(flip_rate.mean()),
        "demo_final_mean": float(acc[:, -1].mean()),
        "demo_peak_step_mean": float(peak_step.mean()),
        "demo_peak_step_max": float(peak_step.max()),
    }
    for k in (2, 4, 8, 16):
        if k <= n_topologies:
            metrics[f"demo_best_of_{k}"] = _expected_best_of_k(settled, k)

    # --- shotgun recovery over the N topologies (the demo's 2nd OOD button) --
    # PAPER PROTOCOL: ``shotgun_volleys`` volleys of PERMANENT knockouts
    # totalling ``damage.target_damage_fraction`` of the hidden gates
    # (default 2 x ceil(0.10 * 240 / 2) = 2 x 12 on the 12-bit circuit).
    # Settle clean for ``steps`` ticks first (the demo regime), then volley at
    # ``steps`` and evenly through the recovery half; gates are clamped faulty
    # + update-masked exactly like the demo button.
    if shotgun_gates is None:
        n_eligible = sum(g for g, _ in layer_sizes[1:-1])
        target_frac = float(cfg.damage.get("target_damage_fraction", 0.10))
        n_volleys = max(1, int(shotgun_volleys))
        shotgun_gates = max(1, int(np.ceil(target_frac * n_eligible / n_volleys)))
    if shotgun_gates > 0 and shotgun_volleys > 0:
        # Volleys spaced ``steps // (2 * volleys)`` apart from ``steps`` on
        # (2 volleys @ steps=64 -> ticks 64 and 80): partial recovery is visible
        # between volleys and the final tail reflects the full damage budget.
        volley_steps = [steps + (v * steps) // (2 * shotgun_volleys) for v in range(shotgun_volleys)]
        D = volley_steps[0]
        skey = jax.random.fold_in(key, 0x506)
        scan_keys = jax.random.split(skey, n_topologies)
        shot = batched_settle(
            model, wires_a, logits_a, x_data, y_data,
            **{**settle_kwargs, "steps": 2 * steps},
            damage_steps=volley_steps,
            knockout_per_damage_step=shotgun_gates,
            scan_keys=scan_keys,
        )
        shot_acc = shot["hard_acc"]  # [N, 2S]
        # Window means are insensitive to the ±1 step alignment of the volley.
        shot_pre = shot_acc[:, D - 1 - settle_window : D - 1].mean(axis=1)  # [N]
        shot_dip = shot_acc[:, D - 1 :].min(axis=1)  # [N]
        shot_settled = shot_acc[:, -settle_window:].mean(axis=1)  # [N]
        # Tanking vs adapting ("adaptive resilience vs redundancy"): a redundant
        # model shows a SHALLOW wound and no climb (the volley never mattered);
        # an adaptive one shows a real wound and heals it. Same penalty can hide
        # either — so report the wound and the healed fraction separately.
        wound = shot_pre - shot_dip  # [N] how much the volley hurt
        climb = shot_settled - shot_dip  # [N] how much was clawed back
        healed_frac = np.where(wound > 1e-3, climb / np.maximum(wound, 1e-3), np.nan)
        metrics.update(
            {
                "demo_shotgun_pre": float(shot_pre.mean()),
                "demo_shotgun_dip": float(shot_dip.mean()),
                "demo_shotgun_settled": float(shot_settled.mean()),
                "demo_shotgun_settled_min": float(shot_settled.min()),
                "demo_shotgun_penalty": float((shot_pre - shot_settled).mean()),
                "demo_shotgun_wound": float(wound.mean()),
                "demo_shotgun_climb": float(climb.mean()),
                "demo_shotgun_healed_frac": float(np.nanmean(healed_frac)),
                "demo_shotgun_flip_rate": float(_tail_flip_rate(shot["tail_preds"]).mean()),
                "demo_shotgun_gates": float(shotgun_gates),
                "demo_shotgun_volleys": float(shotgun_volleys),
            }
        )

    # --- carry recovery over P pairs: the 2x2 quadrants (see module docstring)
    if n_pairs > 0:
        carried_logits = [fl[:n_pairs] for fl in clean["final_logits"]]
        carried_hidden = clean["final_hidden"][:n_pairs]

        # carry_logits = demo-button semantics (legacy demo_carry_* keys)
        carry = batched_settle(model, wires_b, carried_logits, x_data, y_data, **settle_kwargs)
        # carry_none = B's own ceiling
        fresh = batched_settle(model, wires_b, logits_b, x_data, y_data, **settle_kwargs)
        # carry_both = persistence semantics (what training-time shuffle experiences)
        both = batched_settle(
            model, wires_b, carried_logits, x_data, y_data,
            initial_hidden=carried_hidden, **settle_kwargs,
        )
        # carry_hidden = working memory alone (fresh logits, carried hidden)
        hid = batched_settle(
            model, wires_b, logits_b, x_data, y_data,
            initial_hidden=carried_hidden, **settle_kwargs,
        )

        carry_acc = carry["hard_acc"]  # [P, S]
        carry_tail = carry_acc[:, -settle_window:]
        fresh_tail = fresh["hard_acc"][:, -settle_window:]
        both_acc = both["hard_acc"]
        both_tail = both_acc[:, -settle_window:]
        hid_acc = hid["hard_acc"]
        clean_a_settled = settled[:n_pairs]

        carry_settled = carry_tail.mean(axis=1)
        fresh_settled = fresh_tail.mean(axis=1)
        both_settled = both_tail.mean(axis=1)

        metrics.update(
            {
                "demo_cleanA_settled": float(clean_a_settled.mean()),
                # carry_logits quadrant (legacy names = demo button)
                "demo_carry_dip": float(carry_acc.min(axis=1).mean()),
                "demo_carry_settled": float(carry_settled.mean()),
                "demo_carry_std": float(carry_tail.std(axis=1).mean()),
                "demo_carry_flip_rate": float(_tail_flip_rate(carry["tail_preds"]).mean()),
                # carry_none quadrant
                "demo_fresh_settled": float(fresh_settled.mean()),
                "demo_penalty": float((fresh_settled - carry_settled).mean()),
                # carry_both quadrant (persistence)
                "demo_carry_both_dip": float(both_acc.min(axis=1).mean()),
                "demo_carry_both_settled": float(both_settled.mean()),
                "demo_carry_both_std": float(both_tail.std(axis=1).mean()),
                "demo_carry_both_flip_rate": float(_tail_flip_rate(both["tail_preds"]).mean()),
                "demo_carry_both_penalty": float((fresh_settled - both_settled).mean()),
                # carry_hidden quadrant
                "demo_carry_hidden_settled": float(hid_acc[:, -settle_window:].mean()),
                "demo_carry_hidden_dip": float(hid_acc.min(axis=1).mean()),
            }
        )

    # --- provenance -----------------------------------------------------------
    metrics.update(
        {
            "demo_n_topologies": float(n_topologies),
            "demo_n_pairs": float(n_pairs),
            "demo_steps": float(steps),
            "demo_settle_window": float(settle_window),
            "demo_cases": float(x_data.shape[0]),
        }
    )
    return metrics


def format_demo_metrics(metrics: dict[str, float]) -> str:
    """Human-readable one-glance summary (for logs / CLI)."""
    g = metrics.get
    lines = [
        f"clean : settled={g('demo_settled_mean', float('nan')):.4f} "
        f"(min={g('demo_settled_min', float('nan')):.4f} "
        f"p90={g('demo_settled_p90', float('nan')):.4f} "
        f"best-of-8={g('demo_best_of_8', float('nan')):.4f})  "
        f"tail_std={g('demo_tail_std', float('nan')):.4f}  "
        f"flip={g('demo_flip_rate', float('nan')):.4f}  "
        f"peak_step={g('demo_peak_step_mean', float('nan')):.1f}",
    ]
    if "demo_shotgun_settled" in metrics:
        lines.append(
            f"shotgun: pre={g('demo_shotgun_pre', float('nan')):.4f} "
            f"dip={g('demo_shotgun_dip', float('nan')):.4f} -> "
            f"settled={g('demo_shotgun_settled', float('nan')):.4f} "
            f"(penalty={g('demo_shotgun_penalty', float('nan')):.4f}, "
            f"healed={g('demo_shotgun_healed_frac', float('nan')):.0%} of wound, "
            f"flip={g('demo_shotgun_flip_rate', float('nan')):.4f}, "
            f"k={g('demo_shotgun_gates', 0):.0f} gates)"
        )
    if "demo_carry_settled" in metrics:
        lines.append(
            f"shuffle: cleanA={g('demo_cleanA_settled', float('nan')):.4f} "
            f"dip={g('demo_carry_dip', float('nan')):.4f} -> "
            f"carry={g('demo_carry_settled', float('nan')):.4f}"
            f"±{g('demo_carry_std', float('nan')):.4f} "
            f"(fresh={g('demo_fresh_settled', float('nan')):.4f}, "
            f"penalty={g('demo_penalty', float('nan')):.4f})"
        )
    if "demo_carry_both_settled" in metrics:
        lines.append(
            f"quadrants: logits={g('demo_carry_settled', float('nan')):.4f} "
            f"both={g('demo_carry_both_settled', float('nan')):.4f} "
            f"hidden={g('demo_carry_hidden_settled', float('nan')):.4f} "
            f"none(fresh)={g('demo_fresh_settled', float('nan')):.4f}"
        )
    return "\n".join(lines)
