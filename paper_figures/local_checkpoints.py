"""Local-first checkpoint loading for paper_figures/'s eval_*_isn1.py scripts.

gbena/boolean-nca-cc is a private W&B project, so a reproducer without access to it
cannot otherwise run eval_fig4_resilience_isn1.py / eval_fig10_scalefree_isn1.py /
eval_pca_trajectories.py -- they'd fail on `wandb.Api()` before even reaching the
model weights. paper_figures/checkpoints/<run_id>/ archives the small set of
checkpoints (+ their config, since the model can't be instantiated without it)
those three scripts need -- 13MB total across 4 runs -- so figure reproduction
doesn't depend on live W&B access at all. Falls back to load_config_from_wandb for
any run_id not archived here (e.g. if you re-point a script at a new run of your
own).
"""
from __future__ import annotations

import os
from glob import glob

from omegaconf import OmegaConf

_HERE = os.path.dirname(__file__)
ARCHIVE_DIR = os.path.join(_HERE, "checkpoints")


def load_config_and_checkpoint(run_id: str, **wandb_kwargs):
    """Drop-in replacement for load_config_from_wandb(run_id=..., **wandb_kwargs).

    Returns (config, checkpoint_path, run_id). Checks the local archive first;
    falls back to W&B if this run_id isn't archived.
    """
    run_dir = os.path.join(ARCHIVE_DIR, run_id)
    config_path = os.path.join(run_dir, "config.yaml")
    if os.path.exists(config_path):
        pkls = glob(os.path.join(run_dir, "*.pkl"))
        if len(pkls) == 1:
            print(f"[local_checkpoints] loading archived checkpoint for {run_id} (no W&B needed)")
            return OmegaConf.load(config_path), pkls[0], run_id
        print(
            f"[local_checkpoints] {run_dir} has config.yaml but "
            f"{len(pkls)} .pkl files (expected 1) -- falling back to W&B."
        )

    from boolean_nca_cc.training.checkpointing import load_config_from_wandb

    return load_config_from_wandb(run_id=run_id, **wandb_kwargs)
