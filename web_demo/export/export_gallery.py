"""
Batch-export all gallery models from configs/demo_models.yaml.

For each model entry (filtered by --task, default = all entries with a run_id):

  1. Download from W&B and extract / quantise weights →
       public/weights/{task}_{recipe}.json

  2. Record the live-demo bootstrap →
       public/weights/{task}_{recipe}_demo.json

  3. (Optional) Record the parity trajectory →
       public/weights/{task}_{recipe}_trajectory.json

  4. Write the manifest used by the TS demo →
       public/weights/gallery.json

Usage (from web_demo/):
  python -m export.export_gallery --task reverse
  python -m export.export_gallery --task reverse --skip-existing
  python -m export.export_gallery --task reverse --with-trajectories

The TS demo loads gallery.json at startup and builds the model picker from it.
The schema is documented in ``GalleryModel`` below.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import yaml

from .extract_weights import (
    DEFAULT_ENTITY,
    DEFAULT_PROJECT,
    extract_weights,
    load_run,
    serialize_weights,
)
from .record_trajectory import record_for_model

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# configs/ lives two levels above web_demo/
_HERE = Path(__file__).parent  # …/web_demo/export/
_DEMO_YAML = _HERE.parent.parent / "configs" / "demo_models.yaml"
_WEIGHTS_DIR = _HERE.parent / "public" / "weights"
_GALLERY_JSON = _WEIGHTS_DIR / "gallery.json"


# ---------------------------------------------------------------------------
# gallery.json schema (written here, read by main.ts)
# ---------------------------------------------------------------------------
#
#   {
#     "models": [
#       {
#         "id":            "reverse_fixed_no_damage",
#         "task":          "reverse",
#         "recipe":        "fixed_no_damage",
#         "label":         "Fixed · clean",
#         "description":   "Regime I …",
#         "runId":         "vt9awu7h",
#         "weightsPath":   "reverse_fixed_no_damage.json",
#         "bootstrapPath": "reverse_fixed_no_damage_demo.json"
#       },
#       …
#     ]
#   }
#
# weightsPath / bootstrapPath are relative to public/weights/ so the TS side
# prepends BASE_URL + "weights/".


def _model_id(task: str, recipe: str) -> str:
    return f"{task}_{recipe}"


def _weights_path(model_id: str) -> str:
    return str(_WEIGHTS_DIR / f"{model_id}.json")


def _bootstrap_path(model_id: str) -> str:
    return str(_WEIGHTS_DIR / f"{model_id}_demo.json")


def _trajectory_path(model_id: str) -> str:
    return str(_WEIGHTS_DIR / f"{model_id}_trajectory.json")


# ---------------------------------------------------------------------------
# Per-model export
# ---------------------------------------------------------------------------


def export_model(
    entry: dict,
    recipe_meta: dict,
    *,
    skip_existing: bool,
    bootstrap_only: bool,
    with_trajectories: bool,
    n_ticks: int,
    seed: int,
    n_cases: int,
    task_style: str,
    text: str,
    entity: str,
    project: str,
) -> dict:
    """Extract weights + record bootstrap for one gallery entry.

    Returns the gallery.json fragment for this model.
    """
    task = entry["task"]
    recipe = entry["recipe"]
    run_id = entry["run_id"]
    prefer_metric = entry.get("prefer_metric", "eval_in_test_hard_accuracy")
    mid = _model_id(task, recipe)

    weights_out = _weights_path(mid)
    bootstrap_out = _bootstrap_path(mid)
    trajectory_out = _trajectory_path(mid) if with_trajectories else None

    # --bootstrap-only: always re-record bootstrap, never re-serialise weights.
    if bootstrap_only:
        need_weights = False
        need_bootstrap = True
        need_trajectory = trajectory_out is not None
    else:
        already_have_weights = skip_existing and Path(weights_out).exists()
        already_have_bootstrap = skip_existing and Path(bootstrap_out).exists()
        already_have_trajectory = (
            trajectory_out is not None and skip_existing and Path(trajectory_out).exists()
        )
        need_weights = not already_have_weights
        need_bootstrap = not already_have_bootstrap
        need_trajectory = trajectory_out is not None and not already_have_trajectory

    if not need_weights and not need_bootstrap and not need_trajectory:
        print(f"[gallery] {mid}: all files present, skipping (--skip-existing)")
        return _gallery_fragment(mid, task, recipe, run_id, recipe_meta)

    print(f"\n[gallery] ── {mid} (run {run_id}) ──────────────────────────────────")

    # Always load from W&B (uses local cache after first download).
    model, cfg, _ = load_run(
        run_id=run_id,
        entity=entity,
        project=project,
        prefer_metric=prefer_metric,
    )
    weights = extract_weights(model, cfg)

    if need_weights:
        print(f"[gallery] {mid}: writing weights → {weights_out}")
        serialize_weights(weights, weights_out, source_run_id=run_id)
    else:
        print(f"[gallery] {mid}: weights exist, skipping serialise")

    if need_bootstrap or need_trajectory:
        print(
            f"[gallery] {mid}: recording bootstrap{' + trajectory' if need_trajectory else ''} …"
        )
        record_for_model(
            weights,
            cfg,
            run_id=run_id,
            out_bootstrap=bootstrap_out if need_bootstrap else None,
            out_trajectory=trajectory_out if need_trajectory else None,
            n_ticks=n_ticks,
            seed=seed,
            n_cases=n_cases,
            task_style=task_style,
            text=text,
        )

    return _gallery_fragment(mid, task, recipe, run_id, recipe_meta)


def _gallery_fragment(
    mid: str, task: str, recipe: str, run_id: str, recipe_meta: dict
) -> dict:
    return {
        "id": mid,
        "task": task,
        "recipe": recipe,
        "label": recipe_meta.get("label", recipe),
        "description": recipe_meta.get("description", ""),
        "runId": run_id,
        "weightsPath": f"{mid}.json",
        "bootstrapPath": f"{mid}_demo.json",
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task",
        default=None,
        help=(
            "Only export entries for this task (e.g. 'reverse'). "
            "Omit to export all tasks that have a run_id."
        ),
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip models whose weight + bootstrap files already exist.",
    )
    parser.add_argument(
        "--bootstrap-only",
        action="store_true",
        help=(
            "Re-record bootstrap files for all selected models without "
            "re-downloading or re-serialising weights. Uses the local W&B "
            "cache for cfg. Useful when changing --task-style, --text, or "
            "--n-cases without touching weights."
        ),
    )
    parser.add_argument(
        "--with-trajectories",
        action="store_true",
        help="Also write the parity-test trajectory for each model.",
    )
    parser.add_argument("--n-ticks", type=int, default=8)
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument(
        "--n-cases",
        type=int,
        default=0,
        help=(
            "Number of cases in the bootstrap (0 = full task, e.g. 4096 for "
            "12-bit reverse). Use 256 for a fast/compact demo."
        ),
    )
    parser.add_argument(
        "--task-style",
        choices=["sequential", "text-reverse"],
        default="sequential",
        help=(
            "'sequential' = middle-N cases of the task dataset (default). "
            "'text-reverse' = render --text into a binary image; the policy "
            "visually 'rights' the upside-down text. Reverse task only."
        ),
    )
    parser.add_argument(
        "--text",
        default="Welcome to Self Organising Circuits! The Future is Now!",
        help="Text rendered into the y image when --task-style=text-reverse.",
    )
    parser.add_argument("--entity", default=DEFAULT_ENTITY)
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    args = parser.parse_args()

    # Load demo_models.yaml
    if not _DEMO_YAML.exists():
        raise FileNotFoundError(
            f"demo_models.yaml not found at {_DEMO_YAML}. "
            "Run this script from the web_demo/ directory."
        )
    with open(_DEMO_YAML) as f:
        gallery_cfg = yaml.safe_load(f)

    recipes: dict[str, dict] = gallery_cfg.get("recipes", {})
    entries: list[dict] = gallery_cfg.get("models", [])

    # Filter
    selected = [
        e
        for e in entries
        if e.get("run_id")  # skip placeholder entries with run_id: null
        and (args.task is None or e["task"] == args.task)
    ]

    if not selected:
        print(f"[gallery] No entries to export (task={args.task!r}).")
        return

    mode_str = "bootstrap-only" if args.bootstrap_only else "full export"
    text_str = f" · text='{args.text}'" if args.task_style == "text-reverse" else ""
    print(
        f"[gallery] {mode_str} · {len(selected)} model(s) · "
        f"task-style={args.task_style}{text_str} · "
        f"n-cases={args.n_cases or 'full'} · n-ticks={args.n_ticks}"
    )
    print(f"[gallery] Models: {[_model_id(e['task'], e['recipe']) for e in selected]}")

    os.makedirs(_WEIGHTS_DIR, exist_ok=True)

    gallery_models = []
    for entry in selected:
        recipe_meta = recipes.get(entry["recipe"], {})
        fragment = export_model(
            entry,
            recipe_meta,
            skip_existing=args.skip_existing,
            bootstrap_only=args.bootstrap_only,
            with_trajectories=args.with_trajectories,
            n_ticks=args.n_ticks,
            seed=args.seed,
            n_cases=args.n_cases,
            task_style=args.task_style,
            text=args.text,
            entity=args.entity,
            project=args.project,
        )
        gallery_models.append(fragment)

    # Write gallery.json
    gallery_doc = {"models": gallery_models}
    with open(_GALLERY_JSON, "w") as f:
        json.dump(gallery_doc, f, indent=2)
    print(f"\n[gallery] wrote {_GALLERY_JSON} ({len(gallery_models)} model(s))")

    # Print summary table
    print("\n── Gallery summary ──────────────────────────────────────────────────")
    for m in gallery_models:
        w = _weights_path(m["id"])
        b = _bootstrap_path(m["id"])
        w_kb = Path(w).stat().st_size / 1024 if Path(w).exists() else 0
        b_kb = Path(b).stat().st_size / 1024 if Path(b).exists() else 0
        print(f"  {m['id']:<35}  weights {w_kb:6.0f} KB  bootstrap {b_kb:6.0f} KB")


if __name__ == "__main__":
    main()
