"""
Drive ``record_for_model`` from a LOCAL weights JSON + resolved config — no W&B
load. Regenerates the demo bootstrap + parity trajectory for a checkpoint that
was exported via ``export_local_weights``.

Defaults to ``subsample=random`` so the batch (= the demo's residual batch) is
diverse, not the residual-starving sequential middle slice.

Run:  conda run -n bool_nca python -m web_demo.export.record_local \
          --weights-json web_demo/public/weights/reverse_random_damage.json \
          --cfg web_demo/export/run_configs/task_001_layerPE_a7neu6pl.yaml \
          --run-id a7neu6pl \
          --out-bootstrap web_demo/public/weights/reverse_random_damage_demo.json \
          --out-trajectory web_demo/public/weights/reverse_trajectory.json \
          --n-cases 256 --subsample random
"""

from __future__ import annotations

import argparse

from omegaconf import OmegaConf

from web_demo.export.extract_weights import deserialize_weights
from web_demo.export.record_trajectory import record_for_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights-json", required=True)
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--run-id", default="local")
    ap.add_argument("--out-bootstrap", default="")
    ap.add_argument("--out-trajectory", default="")
    ap.add_argument("--n-ticks", type=int, default=8)
    ap.add_argument("--n-cases", type=int, default=256)
    ap.add_argument("--subsample", default="random", choices=["random", "middle"])
    ap.add_argument("--task-style", default="sequential")
    ap.add_argument("--text", default="Welcome to Self Organising Circuits! The Future is Now!")
    ap.add_argument("--render-width", type=int, default=None)
    ap.add_argument("--seed", type=int, default=44)
    args = ap.parse_args()

    cfg = OmegaConf.load(args.cfg)
    weights = deserialize_weights(args.weights_json)
    print(f"[record-local] weights={args.weights_json} cfg={args.cfg} "
          f"subsample={args.subsample} n_cases={args.n_cases}")
    record_for_model(
        weights, cfg,
        run_id=args.run_id,
        out_bootstrap=args.out_bootstrap or None,
        out_trajectory=args.out_trajectory or None,
        n_ticks=args.n_ticks,
        seed=args.seed,
        task_style=args.task_style,
        text=args.text,
        n_cases=args.n_cases,
        render_width=args.render_width,
        subsample=args.subsample,
    )


if __name__ == "__main__":
    main()
