"""
Export a frozen TMT checkpoint to a self-contained JSON consumed by
the web demo.

Pipeline (finalised in M2):

    1. Resolve a W&B run-id (or local checkpoint path).
    2. Load the nnx model + Hydra config via
       ``boolean_nca_cc.training.checkpointing``.
    3. Walk the parameter tree, name each tensor canonically.
    4. Quantise each tensor to int8 with a per-tensor scalar scale
       (mirrors the format in
       ``post--growing-ca/public/webgl_models8/*.json``).
    5. Base64-pack and emit a single JSON next to ``public/weights/``.

The JSON schema is documented inline in ``src/weights.ts``.

This module is deliberately separate from the rest of the training
stack so that the demo build can be reproduced from a single
checkpoint without invoking the full training pipeline.

Usage (placeholder, finalised in M2)::

    python -m export.export_weights \\
        --run-id <wandb_run_id> \\
        --task reverse \\
        --recipe random_damage \\
        --out public/weights/reverse_random_damage.json
"""

from __future__ import annotations


def main() -> None:
    raise NotImplementedError("Implement in M2 — weight export.")


if __name__ == "__main__":
    main()
