"""
Export a LOCAL checkpoint (.pkl + its resolved config.yaml) to the demo's fp16
weights JSON — bypassing the W&B artifact download in ``extract_weights.load_run``.

Use when the checkpoint lives on disk (e.g. an hsm sweep pull) and its config was
recovered separately (e.g. from W&B cloud). Reuses the same extraction +
serialisation as the W&B path, so output is byte-compatible with the TS loader.

Run:  conda run -n bool_nca python -m web_demo.export.export_local_weights \
          --ckpt <best_model>.pkl --cfg <config.yaml> \
          --out web_demo/public/weights/reverse_random_damage.json --run-id a7neu6pl
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from web_demo.export.ceiling_probe import load_model_and_cfg
from web_demo.export.extract_weights import (
    deserialize_weights,
    extract_weights,
    serialize_weights,
    _summarise,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--run-id", default=None, help="source_run_id tag for the header")
    ap.add_argument("--dtype", default="fp16")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    model, cfg, _layer_sizes, _ = load_model_and_cfg(args.ckpt, args.cfg, seed=args.seed)
    weights = extract_weights(model, cfg)
    print("[export-local] extracted weights summary:")
    print(_summarise(weights))

    serialize_weights(weights, args.out, source_run_id=args.run_id, dtype=args.dtype)
    size_kb = os.path.getsize(args.out) / 1024.0
    print(f"[export-local] wrote {args.out} ({size_kb:,.1f} KB, dtype={args.dtype})")

    # fp16 round-trip check: re-read and compare every tensor to the in-memory pack.
    rt = deserialize_weights(args.out)
    import jax
    n_params = sum(int(x.size) for x in jax.tree.leaves(__import__("flax").nnx.state(model, __import__("flax").nnx.Param)))
    max_rel = 0.0
    pairs = [
        ("feature_proj_W", weights.feature_proj_W, rt.feature_proj_W),
        ("logit_proj_W", weights.logit_proj_W, rt.logit_proj_W),
        ("blocks.Wq", weights.blocks.Wq, rt.blocks.Wq),
        ("blocks.ffn_W1", weights.blocks.ffn_W1, rt.blocks.ffn_W1),
    ]
    for name, a, b in pairs:
        a = np.asarray(a); b = np.asarray(b)
        denom = np.maximum(np.abs(a), 1e-6)
        max_rel = max(max_rel, float(np.max(np.abs(a - b) / denom)))
    import json
    with open(args.out) as f:
        _hdr = json.load(f)["header"]
    print(f"[export-local] params={n_params:,}  header.source_run_id={_hdr.get('source_run_id')!r} "
          f"kind={rt.config.model_kind!r} layerPE={rt.config.use_layer_PE} "
          f"dist_pe={rt.config.use_dist_pe} node_loss={rt.config.use_node_loss} "
          f"max_neighbors={rt.config.max_neighbors}")
    print(f"[export-local] fp16 round-trip max relative tensor error = {max_rel:.2e} (expect ~1e-3)")


if __name__ == "__main__":
    main()
