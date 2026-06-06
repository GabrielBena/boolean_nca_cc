"""
Batched demo-probe CLI — score any checkpoint on the live-demo objective in
seconds on GPU, instead of the minutes-per-checkpoint generator probes.

Thin veneer over ``boolean_nca_cc.training.demo_probe`` using the same loaders
as the other export probes (``load_model_and_cfg`` configures the global graph
builder, which the engine requires for dist_pe/rwse models).

Run:  conda run -n bool_nca python -m web_demo.export.demo_probe_cli \
          --ckpt <run>/best_model_eval_out_test_hard_accuracy.pkl \
          --cfg <run>/.hydra/config.yaml \
          [--n-topo 64 --pairs 16 --steps 64 --window 8 --chunk 32 --seed 0]

NOTE: unlike train.py this does NOT pin CUDA_VISIBLE_DEVICES — set it yourself
(GPU 0 is reserved on anahita; use 1/2/3 as free).
"""

from __future__ import annotations

import argparse
import json

import jax

from boolean_nca_cc.training.demo_probe import demo_probe, format_demo_metrics
from web_demo.export.ceiling_probe import load_model_and_cfg, make_task


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--cfg", default=None, help="path to the run's .hydra/config.yaml")
    ap.add_argument("--tag", default=None, help="load via probe_shuffle_models.MODELS (e.g. 1u5ssulx) instead of --ckpt/--cfg")
    ap.add_argument("--n-topo", type=int, default=64)
    ap.add_argument("--pairs", type=int, default=16)
    ap.add_argument("--steps", type=int, default=64)
    ap.add_argument("--window", type=int, default=8, help="settle window (tail steps)")
    ap.add_argument("--chunk", type=int, default=32, help="topologies per batched rollout")
    ap.add_argument("--cases", type=int, default=512)
    ap.add_argument("--shotgun-gates", type=int, default=None,
                    help="per-volley knockouts (default: auto = paper protocol 10%% / volleys; 0 = skip)")
    ap.add_argument("--shotgun-volleys", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", default=None, help="optional path to dump the metrics dict")
    args = ap.parse_args()

    if args.tag:
        from web_demo.export.probe_shuffle_models import MODELS, load_any

        spec = [m for m in MODELS if args.tag in m[0]][0]
        model, cfg, layer_sizes = load_any(spec)
        snapshot = {}
    else:
        assert args.ckpt and args.cfg, "--ckpt/--cfg required unless --tag is given"
        model, cfg, layer_sizes, snapshot = load_model_and_cfg(args.ckpt, args.cfg, seed=args.seed)
    x, y, task_name = make_task(cfg, args.cases)
    print(
        f"task={task_name} in={int(cfg.circuit.input_bits)} out={int(cfg.circuit.output_bits)} "
        f"cases={x.shape[0]}  n_topo={args.n_topo} pairs={args.pairs} "
        f"steps={args.steps} window={args.window}"
    )
    if snapshot:
        snap = {k: round(float(v), 4) for k, v in snapshot.items() if "hard_acc" in k and "out_test" in k}
        print(f"checkpoint metric snapshot: {snap}")

    metrics = demo_probe(
        model,
        cfg,
        x,
        y,
        layer_sizes,
        key=jax.random.PRNGKey(args.seed + 0xD3),
        n_topologies=args.n_topo,
        n_pairs=args.pairs,
        steps=args.steps,
        settle_window=args.window,
        chunk_size=args.chunk,
        shotgun_gates=args.shotgun_gates,
        shotgun_volleys=args.shotgun_volleys,
    )

    print("\n" + format_demo_metrics(metrics) + "\n")
    for k in sorted(metrics):
        print(f"  {k} = {metrics[k]:.4f}")

    if args.json:
        with open(args.json, "w") as f:
            json.dump(metrics, f, indent=2, sort_keys=True)
        print(f"\nmetrics written to {args.json}")


if __name__ == "__main__":
    main()
