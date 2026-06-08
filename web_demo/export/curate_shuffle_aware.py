"""
Shuffle-AWARE topology curation with an ALL-ROUNDER composite score.

A demo topology is hit by three buttons; ranking on any single metric just trades
one off against another (razor-100% boot topologies recover WORST from shuffle and
shotgun). So we screen all three states and rank by ONE composite:
  pre(T)      = clean settled hard-acc on T            (boot / target ceiling)
  post(T)     = settled acc after carrying T's EVOLVED logits onto a reference
                topology R                              (SHUFFLE-button recovery)
  shot_set(T) = settled acc after a 12-gate permanent volley on a settled T, with
                steps to re-route                       (SHOTGUN final accuracy)
  shot_dip(T) = worst accuracy right after the volley   (how deep the wound is)
  climb(T)    = shot_set - shot_dip                      (accuracy actually REGROWN)
  churn(T)    = per-step output-bit flicker at the fixed point (the visible jitter)

  allround    = min(pre, post, shot_set) + heal_weight * climb - churn_weight * churn

i.e. "the worst final accuracy across ANY button (so it always LANDS high),
REWARDED for visibly re-growing accuracy after the shotgun, penalized for jitter."
This deliberately ranks a topology that dips ~8% and heals back to 1.0 (visible
self-repair) ABOVE one that's so robust it never dips — the dip+recovery IS the
adaptive exhibit. final accuracy stays in the min() floor so a deep-dip /
low-recovery topology can't win on climb alone. The 12-gate volley is
eval-consistent (one paper volley); pair it with the bumped demo
DEFAULT_SHOTGUN_GATES=12 so the demo and the curation see the same dose.

Legacy rank-by modes (min/mean/pre/post over just pre,post) kept for the crisp
models; allround is the default.

Exports the demo pool ranked by score, with pre/post/shot/churn metadata.

Both screening passes run BATCHED on GPU via
boolean_nca_cc.training.demo_probe.batched_settle (one vmapped rollout per pass
instead of one generator rollout per topology — seconds instead of minutes).

Run: conda run -n bool_nca python -m web_demo.export.curate_shuffle_aware \
        --tag 1u5ssulx --n-screen 160 --top-k 96 --steps 44 --cases 256 \
        --out web_demo/public/weights/reverse_random_topology_pool.json
     (or point at a local checkpoint: --ckpt <best_model>.pkl --cfg <.hydra/config.yaml>)
"""

from __future__ import annotations

import argparse
import json

import jax
import jax.numpy as jp
import numpy as np

from boolean_nca_cc.circuits.train import LossConfig
from boolean_nca_cc.training.demo_probe import _tail_flip_rate, batched_settle
from boolean_nca_cc.training.pool.pool import get_wires_and_logits
from web_demo.export.probe_jitter_batch import diverse_batch
from web_demo.export.probe_shuffle_models import MODELS, load_any
from web_demo.export.record_trajectory import _b64_array


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="1u5ssulx")
    ap.add_argument("--ckpt", default=None, help="local checkpoint .pkl (with --cfg) instead of --tag")
    ap.add_argument("--cfg", default=None, help="the run's .hydra/config.yaml (with --ckpt)")
    ap.add_argument("--n-screen", type=int, default=160)
    ap.add_argument("--top-k", type=int, default=96)
    ap.add_argument("--steps", type=int, default=44)
    ap.add_argument("--cases", type=int, default=256)
    ap.add_argument("--tail", type=int, default=16)
    ap.add_argument("--chunk", type=int, default=32, help="topologies per batched rollout")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--rank-by", default="allround",
                    choices=["allround", "min", "mean", "post", "pre"])
    ap.add_argument("--shotgun-gates", type=int, default=12,
                    help="permanent knockouts in the shotgun-recovery screen (one eval volley)")
    ap.add_argument("--churn-weight", type=float, default=1.0,
                    help="penalty per unit of tail flip-rate in the allround score")
    ap.add_argument("--heal-weight", type=float, default=0.5,
                    help="reward per unit of shotgun climb (settled-dip) in the allround score; "
                         "keep <1 so final accuracy (the min floor) still dominates")
    ap.add_argument("--shot-draws", type=int, default=6,
                    help="draws averaged in the shotgun screen; each draw uses a fresh gate set "
                         "AND a random click-step (see --shot-timing-jitter), so we average over "
                         "(gate set x click timing) jointly — the EXPECTED dip/recovery a user sees")
    ap.add_argument("--shot-timing-jitter", type=int, default=None,
                    help="+/- steps the shotgun click-step is jittered around the settle point, "
                         "per draw (default steps//3). Models the user clicking at any settle "
                         "state — the dominant uncontrolled demo factor. 0 = fixed timing.")
    ap.add_argument("--shuffle-targets", type=int, default=6,
                    help="distinct target topologies the shuffle-recovery is averaged over — "
                         "the demo shuffles T's logits onto a DIFFERENT pool topology each click, "
                         "so a single reference is noisy (same reason as shot-draws)")
    ap.add_argument("--cases-from", default=None,
                    help="rank on the EXACT cases the demo displays (a bootstrap JSON's task_data) "
                         "— avoids the curate/display case-set mismatch")
    ap.add_argument("--out", default="web_demo/public/weights/reverse_random_topology_pool.json")
    args = ap.parse_args()

    if args.ckpt:
        from web_demo.export.ceiling_probe import load_model_and_cfg

        model, cfg, layer_sizes, _ = load_model_and_cfg(args.ckpt, args.cfg, seed=0)
        tag = args.ckpt
    else:
        spec = [m for m in MODELS if args.tag in m[0]][0]
        model, cfg, layer_sizes = load_any(spec)
        tag = spec[0]
    arity = int(cfg.circuit.arity); bits = int(cfg.circuit.output_bits)
    if args.cases_from:
        import base64
        td = json.load(open(args.cases_from))["task_data"]
        dec = lambda e: np.frombuffer(base64.b64decode(e["data_b64"]), np.float32).reshape(e["shape"])
        xn, yn = dec(td["x"]), dec(td["y"])
        print(f"ranking on {xn.shape[0]} cases from {args.cases_from} (matches the demo display)")
    else:
        xn, yn = diverse_batch(bits, args.cases, 7)
    x, y = jax.numpy.asarray(xn), jax.numpy.asarray(yn)

    settle_kwargs = dict(
        steps=args.steps,
        layer_sizes=layer_sizes,
        input_n=int(cfg.circuit.input_bits),
        arity=arity,
        circuit_hidden_dim=int(cfg.model.circuit_hidden_dim),
        loss_cfg=LossConfig.from_dict(dict(cfg.loss)),
        bidirectional_edges=bool(cfg.graph.bidirectional_edges),
        settle_window=args.tail,
        chunk_size=args.chunk,
    )

    # --- pass 1 (batched): screen N topologies — pre / churn / evolved logits
    wkey, lkey = jax.random.split(jax.random.PRNGKey(args.seed))
    batch_wires, batch_logits = get_wires_and_logits(
        wkey, lkey, layer_sizes, pool_size=args.n_screen, arity=arity,
        noise_scale=float(cfg.pool.noise_scale), wiring_mode="random",
    )
    clean = batched_settle(model, batch_wires, batch_logits, x, y, **settle_kwargs)
    pre_arr = clean["hard_acc"][:, -args.tail:].mean(axis=1)          # [N]
    churn_arr = _tail_flip_rate(clean["tail_preds"])                  # [N]

    n = args.n_screen

    # --- pass 2 (batched): post = SHUFFLE-button recovery, AVERAGED over K target
    #     topologies. The demo carries T's evolved logits onto a DIFFERENT pool
    #     topology each shuffle, so a single reference is noisy (same reason as the
    #     shotgun gate-draw averaging). Targets = top-K by clean acc (where the
    #     best-first demo pool actually shuffles to). carry-onto-self is masked.
    target_idxs = [int(i) for i in np.argsort(-pre_arr)[:args.shuffle_targets]]
    post_draws, post_churn_draws = [], []
    for ref_idx in target_idxs:
        ref_wires = [jp.stack([w[ref_idx]] * n) for w in batch_wires]
        carry = batched_settle(model, ref_wires, clean["final_logits"], x, y, **settle_kwargs)
        p = carry["hard_acc"][:, -args.tail:].mean(axis=1).copy()    # [N]
        pc = _tail_flip_rate(carry["tail_preds"]).copy()
        p[ref_idx] = np.nan; pc[ref_idx] = np.nan                    # mask carry-onto-self
        post_draws.append(p); post_churn_draws.append(pc)
    post_arr = np.nanmean(post_draws, axis=0)                        # E[shuffle recovery]
    post_churn_arr = np.nanmean(post_churn_draws, axis=0)

    # --- pass 3 (batched): shot = SHOTGUN-button recovery. Settle each T for
    #     `steps`, fire a `shotgun_gates` PERMANENT volley, then `steps` more to
    #     re-route. Per-topology scan keys so each draws its own gate set.
    #     (Skipped for the legacy non-allround modes to save the 2x rollout.)
    if args.rank_by == "allround":
        # Average over ``shot_draws`` distinct volley gate sets — healing is
        # gate-set-dependent and the demo fires a fresh set per click, so a single
        # draw mis-estimates what the user sees. Report the EXPECTED dip/recovery.
        J = args.shot_timing_jitter if args.shot_timing_jitter is not None else args.steps // 3
        # rollout long enough that even the latest click leaves >= `steps` to recover
        total = args.steps + J + args.steps
        onset_rng = np.random.default_rng(args.seed)
        shot_settled_draws, shot_dip_draws = [], []
        for d in range(args.shot_draws):
            onset = int(np.clip(args.steps + onset_rng.integers(-J, J + 1), args.tail, total - args.tail))
            shot_keys = jax.random.split(jax.random.fold_in(jax.random.PRNGKey(args.seed), 0x506 + d), n)
            shot = batched_settle(
                model, batch_wires, batch_logits, x, y,
                **{**settle_kwargs, "steps": total},
                damage_steps=[onset],                                # click timing varies per draw
                knockout_per_damage_step=args.shotgun_gates,
                scan_keys=shot_keys,
            )
            sf = shot["hard_acc"]                                     # [N, total]
            shot_settled_draws.append(sf[:, -args.tail:].mean(axis=1))
            shot_dip_draws.append(sf[:, onset:].min(axis=1))
        shot_arr = np.mean(shot_settled_draws, axis=0)              # E[final settled]
        shot_dip_arr = np.mean(shot_dip_draws, axis=0)             # E[worst post-volley]
        climb_arr = np.maximum(shot_arr - shot_dip_arr, 0.0)       # E[accuracy regrown]
    else:
        shot_arr = shot_dip_arr = climb_arr = np.full(n, np.nan)

    if args.rank_by == "allround":
        # land high on every button (floor) + reward visible shotgun re-growth - jitter
        score_arr = np.minimum(np.minimum(pre_arr, post_arr), shot_arr) \
            + args.heal_weight * climb_arr \
            - args.churn_weight * churn_arr
    else:
        score_fn = {
            "min": lambda p, q: np.minimum(p, q),
            "mean": lambda p, q: 0.5 * (p + q),
            "post": lambda p, q: q,
            "pre": lambda p, q: p,
        }[args.rank_by]
        score_arr = score_fn(pre_arr, post_arr)

    rows = [
        dict(
            pre=float(pre_arr[t]), post=float(post_arr[t]), shot=float(shot_arr[t]),
            shot_dip=float(shot_dip_arr[t]), climb=float(climb_arr[t]),
            churn=float(churn_arr[t]), post_churn=float(post_churn_arr[t]),
            score=float(score_arr[t]),
            wires=[np.asarray(w[t]) for w in batch_wires],
        )
        for t in range(n)
    ]
    rows.sort(key=lambda r: -r["score"])

    pre_sorted = np.array([r["pre"] for r in rows])
    post_sorted = np.array([r["post"] for r in rows])
    shot_sorted = np.array([r["shot"] for r in rows])
    allround = args.rank_by == "allround"
    print(f"model={tag} n={len(rows)} rank-by={args.rank_by}"
          + (f" = min(pre,post,shot) + {args.heal_weight}*climb - {args.churn_weight}*churn"
             if allround else "(pre,post)"))
    print(f"pre:  mean={pre_sorted.mean():.3f} max={pre_sorted.max():.3f} #(>=.99)={int((pre_sorted>=.99).sum())}")
    print(f"post: mean={post_sorted.mean():.3f} max={post_sorted.max():.3f}")
    if allround:
        print(f"shot: mean={np.nanmean(shot_sorted):.3f} max={np.nanmax(shot_sorted):.3f}  (shotgun {args.shotgun_gates}-gate final acc)")
    print("\nrank  pre    post   dip    shot   climb  churn   score  (top-12)")
    for i in range(min(12, len(rows))):
        r = rows[i]
        if allround:
            print(f"  {i:3d} {r['pre']:.3f}  {r['post']:.3f}  {r['shot_dip']:.3f}  {r['shot']:.3f}  "
                  f"{r['climb']:+.3f} {r['churn']:.4f}  {r['score']:.3f}")
        else:
            print(f"  {i:3d} {r['pre']:.3f}  {r['post']:.3f}    -      -       -    {r['churn']:.4f}  {r['score']:.3f}")
    # where do the razor-perfect (pre>=.999) topologies land?
    perf_ranks = [i for i, r in enumerate(rows) if r["pre"] >= 0.999]
    print(f"\nrazor-perfect (pre>=.999) topologies now sit at ranks: {perf_ranks[:10]}"
          f"{' ...' if len(perf_ranks) > 10 else ''}  (n={len(perf_ranks)})")

    # --- export top-K in the demo pool schema ---
    k = min(args.top_k, len(rows))
    topos = [{
        "rank": i, "pre_acc": round(rows[i]["pre"], 4), "post_acc": round(rows[i]["post"], 4),
        "shot_acc": (round(rows[i]["shot"], 4) if allround else None),
        "shot_dip": (round(rows[i]["shot_dip"], 4) if allround else None),
        "climb": (round(rows[i]["climb"], 4) if allround else None),
        "churn": round(rows[i]["churn"], 5), "score": round(rows[i]["score"], 4),
        "wires": [_b64_array(np.asarray(w, np.int32), np.int32) for w in rows[i]["wires"]],
    } for i in range(k)]
    ranked_by = (f"allround=min(pre,post,shot{args.shotgun_gates})-{args.churn_weight}churn"
                 if allround else f"{args.rank_by}(pre,post)_shuffle_aware")
    payload = {
        "header": {"kind": "ranked_topology_pool", "source_tag": tag,
                   "ranked_by": ranked_by, "n_screened": args.n_screen,
                   "n_exported": k, "steps": args.steps, "cases": args.cases, "arity": arity,
                   "shotgun_gates": (args.shotgun_gates if allround else None)},
        "layer_sizes": [[int(a), int(b)] for (a, b) in layer_sizes],
        "topologies": topos,
    }
    json.dump(payload, open(args.out, "w"), separators=(",", ":"))
    import os
    print(f"\nexported top-{k} (shuffle-aware) -> {args.out} ({os.path.getsize(args.out)/1024:.0f} KB)")
    print(f"  rank0: pre={topos[0]['pre_acc']} post={topos[0]['post_acc']} churn={topos[0]['churn']}")


if __name__ == "__main__":
    main()
