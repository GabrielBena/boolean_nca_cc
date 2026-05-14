"""
Generate the JSON data files the web demo consumes.

This script produces *two* artefacts in one invocation:

1. ``--out-bootstrap`` (default ``public/weights/reverse_demo.json``):
   the **live demo bootstrap** — header + ``layer_sizes`` + ``task_data``
   ``(x, y)``. ``src/main.ts`` loads this and spawns its own controller
   state on top.

2. ``--out-trajectory`` (default ``public/weights/reverse_trajectory.json``):
   the **parity smoke-test trajectory** — everything the bootstrap has,
   plus ``initial_state``, ``wires`` and a per-tick log of
   ``logits / hidden / loss / hard_accuracy``. ``src/replay.ts``
   (mounted by ``verify.html``) replays this in the browser and asserts
   the TS port matches the NumPy oracle bit-exactly on hard predictions.

The two files share most of their content; they're split because the
two pages have different concerns. The live demo only needs the task
data; the parity test needs the recorded reference trajectory.

The ``--n-cases`` flag is the user-tunable knob: it controls the size
of the (x, y) batch shipped to the browser, and therefore both the
bit-image strip width and the per-tick compute cost in the live demo.

Schema (mirrored on the TS side in ``src/replay.ts`` /
``src/main.ts:fetchBootstrap``)::

    {
        # ----- shared by both files --------------------------------
        "header": {
            "task": "reverse",
            "task_style": "sequential" | "text-reverse",
            "text": "..." | None,
            "input_bits": 12,
            "output_bits": 12,
            "arity": 4,
            "circuit_hidden_dim": 64,
            "max_neighbors": 10,
            "case_n": 4096,
            "n_ticks": 32 | 0,           # 0 in the bootstrap file
            "seed": 44,
            "source_run_id": "1u5ssulx",
            "weights_path": "reverse_random_damage.json",
        },
        "layer_sizes": [[12, 1], [96, 4], [96, 4], [48, 2], [12, 1]],
        "task_data": {
            "x": {"shape": [case_n, input_bits], "data_b64": "<float32-LE>"},
            "y": {"shape": [case_n, output_bits], "data_b64": "<float32-LE>"},
        },

        # ----- trajectory-only (omitted from the demo bootstrap) ---
        "wires": [{"shape": [arity, group_n], "data_b64": "<int32-LE>"}, ...],
        "initial_state": {
            "logits": {...}, "hidden": {...}, "loss": {...}, "gate_mask": {...},
        },
        "ticks": [
            {"step": 1, "hard_accuracy": 0.5417, "logits": ..., "hidden": ..., "loss": ...},
            ...,
        ],
    }

Bytes are little-endian (web-friendly). The Float32Array / Int32Array
JS views over the base64-decoded buffers reproduce the NumPy arrays
exactly.
"""

from __future__ import annotations

import argparse
import base64
import json
import os

import numpy as np

from .extract_weights import (
    DEFAULT_RUN_ID,
    deserialize_weights,
    extract_weights,
    load_run,
    to_np,
)
from .numpy_oracle import (
    CircuitState,
    build_topology,
    compute_residuals,
    gen_wires,
    init_state,
    tmt_step,
)


def _b64_array(arr: np.ndarray, dtype) -> dict:
    arr = np.ascontiguousarray(np.asarray(arr, dtype=dtype))
    return {
        "shape": list(arr.shape),
        "data_b64": base64.b64encode(arr.tobytes()).decode("ascii"),
    }


def _generate_text_reverse_data(
    text: str, n_cases: int, output_bits: int, render_width: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Build a (x, y) test set whose bit-reverse mapping coincides with a visual flip.

    The text is rendered into a *wide* binary canvas (``render_width`` columns,
    ``output_bits`` rows) — wide enough for letters to have real horizontal
    resolution — then the middle ``n_cases`` columns are cropped and used as
    ``y``. ``x = y[:, ::-1]`` is the per-case bit reversal, visually the same
    image flipped vertically.

    The deployed reverse-task policy maps ``x -> y`` (its trained job). Feeding
    the policy this special pair makes the visible behaviour "right the
    upside-down text", in-distribution since the underlying function is
    unchanged. This mirrors the (render-then-slice-middle) pattern from
    ``boolean_nca_cc.circuits.tasks.text_task`` + the exploration notebooks.

    Default ``render_width`` is ``max(n_cases, 4096)`` so a long phrase rendered
    at 12 px tall is still readable in the cropped 256-wide slice.
    """
    import PIL.Image
    import PIL.ImageDraw

    if render_width is None:
        render_width = max(n_cases, 4096)
    if render_width < n_cases:
        raise ValueError(
            f"render_width ({render_width}) must be >= n_cases ({n_cases})"
        )

    im = PIL.Image.new("L", (render_width, output_bits))
    draw = PIL.ImageDraw.Draw(im)
    draw.text((render_width // 2, output_bits // 2), text, fill=255, anchor="mm")
    full = (np.array(im) > 100).astype(np.float32)  # [output_bits, render_width]

    mid = render_width // 2
    half = n_cases // 2
    sl = slice(mid - half, mid - half + n_cases)
    cropped = full[:, sl]  # [output_bits, n_cases]
    y = np.ascontiguousarray(cropped.T)  # [n_cases, output_bits]
    x = np.ascontiguousarray(y[:, ::-1])  # per-case bit reversal
    return x, y


def _subsample_middle(
    x_full: np.ndarray, y_full: np.ndarray, n_cases: int
) -> tuple[np.ndarray, np.ndarray]:
    """Pick the middle ``n_cases`` rows of (x_full, y_full) so the bits aren't
    all-zero (low x) or all-one (high x). Cheap perf knob for the live demo."""
    full = x_full.shape[0]
    if n_cases >= full:
        return x_full, y_full
    mid = full // 2
    half = n_cases // 2
    sl = slice(mid - half, mid - half + n_cases)
    return np.ascontiguousarray(x_full[sl]), np.ascontiguousarray(y_full[sl])


def record_for_model(
    weights: "TMTWeights",
    cfg: "DictConfig",
    *,
    run_id: str = DEFAULT_RUN_ID,
    out_bootstrap: str | None = None,
    out_trajectory: str | None = None,
    n_ticks: int = 8,
    seed: int = 44,
    task_style: str = "sequential",
    text: str = "Welcome to Self Organising Circuits! The Future is Now!",
    n_cases: int = 0,
    render_width: int | None = None,
) -> None:
    """Core record logic — callable from export_gallery or main().

    Exactly one of ``out_bootstrap`` / ``out_trajectory`` must be non-empty
    (or both). Writes the same JSON schemas as before.
    """
    from boolean_nca_cc.circuits.model import generate_layer_sizes
    from boolean_nca_cc.circuits.tasks import get_task_data

    arity = int(cfg.circuit.arity)
    input_bits = int(cfg.circuit.input_bits)
    output_bits = int(cfg.circuit.output_bits)
    hidden_dim = int(cfg.model.circuit_hidden_dim)
    case_n = 1 << input_bits

    layer_sizes = generate_layer_sizes(
        input_bits,
        output_bits,
        arity,
        layer_n=int(cfg.circuit.num_layers),
        width_factor=float(cfg.circuit.get("width_factor", 2)),
    )

    rng = np.random.default_rng(seed)  # always needed for init_state noise
    wiring_mode = str(cfg.training.get("wiring_mode", "random"))
    if wiring_mode in ("fixed", "genetic"):
        # Derive the exact in-distribution wires the model was trained on.
        # Mirrors DemoSession.derive_pool_keys(cfg, distribution="auto").
        import jax
        from boolean_nca_cc.circuits.model import gen_wires as jax_gen_wires
        from boolean_nca_cc.demo import derive_pool_keys

        jax_wires_key, _ = derive_pool_keys(cfg, distribution="auto")
        wires_np = []
        in_n = layer_sizes[0][0]
        for out_n, group_size in layer_sizes[1:]:
            w = jax_gen_wires(jax_wires_key, in_n, out_n, arity, group_size)
            wires_np.append(np.asarray(w, dtype=np.int32))
            in_n = out_n
        print(f"  wiring: fixed (in-distribution, JAX seed from cfg.seed={cfg.seed})")
    else:
        wires_np = []
        for li in range(1, len(layer_sizes)):
            in_n = layer_sizes[li - 1][0]
            out_n, group_size = layer_sizes[li]
            wires_np.append(gen_wires(rng, in_n, out_n, arity, group_size))
        print(f"  wiring: random (numpy seed={seed})")

    topology = build_topology(
        layer_sizes,
        wires_np,
        arity,
        hidden_dim,
        max_neighbors=int(cfg.model.max_neighbors),
    )
    state, _ = init_state(
        topology, weights.config, rng, noise_scale=float(cfg.pool.noise_scale)
    )

    from boolean_nca_cc.utils.task_compat import get_fixed_task_name

    task_name = get_fixed_task_name(cfg)
    if task_name is None:
        raise ValueError(
            "record_trajectory requires a fixed task in the checkpoint cfg "
            "(cfg.tasks.name or legacy cfg.circuit.task)."
        )

    if task_style == "text-reverse":
        if task_name != "reverse":
            raise ValueError(
                f"--task-style=text-reverse only makes sense for the reverse task, "
                f"got task={task_name!r}"
            )
        if input_bits != output_bits:
            raise ValueError(
                f"text-reverse requires input_bits == output_bits, got "
                f"{input_bits} != {output_bits}"
            )
        n_cases_actual = n_cases if n_cases > 0 else 256
        x_np, y_np = _generate_text_reverse_data(
            text, n_cases_actual, output_bits, render_width=render_width
        )
        case_n = n_cases_actual
    else:
        (x_full, y_full), _, _ = get_task_data(
            task_name,
            case_n=case_n,
            input_bits=input_bits,
            output_bits=output_bits,
            train_test_split=False,
        )
        x_np = to_np(x_full)
        y_np = to_np(y_full)
        if 0 < n_cases < case_n:
            x_np, y_np = _subsample_middle(x_np, y_np, n_cases)
            case_n = x_np.shape[0]
            print(f"  task-style: sequential (middle subsample, n_cases={case_n})")
        else:
            print(f"  task-style: sequential (full, n_cases={case_n})")

    state, _aux0 = compute_residuals(state, topology, x_np, y_np)

    initial_state = {
        "logits": _b64_array(state.logits, np.float32),
        "hidden": _b64_array(state.hidden, np.float32),
        "loss": _b64_array(state.loss, np.float32),
        "gate_mask": _b64_array(
            state.gateMask if hasattr(state, "gateMask") else state.gate_mask,
            np.float32,
        ),
    }

    ticks: list[dict] = []
    for t in range(1, n_ticks + 1):
        state = tmt_step(state, topology, weights)
        state, aux = compute_residuals(state, topology, x_np, y_np)
        ticks.append(
            {
                "step": t,
                "hard_accuracy": float(aux["hard_accuracy"]),
                "soft_accuracy": float(aux["accuracy"]),
                "logits": _b64_array(state.logits, np.float32),
                "hidden": _b64_array(state.hidden, np.float32),
                "loss": _b64_array(state.loss, np.float32),
            }
        )
        print(f"  tick {t:2d}: hard_acc={aux['hard_accuracy']:.4f}")

    doc = {
        "header": {
            "task": str(task_name),
            "task_style": task_style,
            "text": text if task_style == "text-reverse" else None,
            "input_bits": input_bits,
            "output_bits": output_bits,
            "arity": arity,
            "circuit_hidden_dim": hidden_dim,
            "max_neighbors": int(cfg.model.max_neighbors),
            "case_n": case_n,
            "n_ticks": n_ticks,
            "seed": seed,
            "source_run_id": run_id,
            "weights_path": None,
        },
        "layer_sizes": [list(ls) for ls in layer_sizes],
        "wires": [_b64_array(w, np.int32) for w in wires_np],
        "initial_state": initial_state,
        "task_data": {
            "x": _b64_array(x_np, np.float32),
            "y": _b64_array(y_np, np.float32),
        },
        "ticks": ticks,
    }

    def _write(path: str, payload: dict, what: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(payload, f, separators=(",", ":"))
        size_kb = os.path.getsize(path) / 1024.0
        print(f"[record-traj] wrote {path} ({size_kb:,.1f} KB, {what})")

    if out_trajectory:
        _write(out_trajectory, doc, f"{n_ticks} ticks · parity reference")

    if out_bootstrap:
        bootstrap_doc = {
            "header": {**doc["header"], "n_ticks": 0},
            "layer_sizes": doc["layer_sizes"],
            # Include wires so the TS Controller uses the exact same topology
            # as the training run — critical for fixed-wires models.
            "wires": doc["wires"],
            "task_data": doc["task_data"],
        }
        _write(out_bootstrap, bootstrap_doc, "live-demo bootstrap")

    if not out_trajectory and not out_bootstrap:
        raise ValueError("Nothing to write: both out_bootstrap and out_trajectory are empty.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument(
        "--weights-json",
        default=None,
        help="If provided, dequantise weights from this JSON; otherwise extract from W&B.",
    )
    parser.add_argument(
        "--out-bootstrap",
        default="public/weights/reverse_demo.json",
        help=(
            "Output path for the live-demo bootstrap (header + layer_sizes + "
            "task_data, no per-tick states). Pass an empty string to skip."
        ),
    )
    parser.add_argument(
        "--out-trajectory",
        default="public/weights/reverse_trajectory.json",
        help=(
            "Output path for the parity-test trajectory (full doc with "
            "initial_state + wires + per-tick states). Pass an empty string "
            "to skip."
        ),
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Deprecated alias for --out-trajectory (kept for backwards compat).",
    )
    parser.add_argument("--n-ticks", type=int, default=8)
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument(
        "--task-style",
        choices=["sequential", "text-reverse"],
        default="sequential",
        help=(
            "Test-set construction: 'sequential' = middle-N cases of the regular "
            "task (cheap subsample for perf); 'text-reverse' = render TEXT into y "
            "and use bit-reversed y as x (visual demo where the policy 'rights' "
            "an upside-down text). Reverse task only."
        ),
    )
    parser.add_argument(
        "--text",
        default="Welcome to Self Organising Circuits! The Future is Now!",
        help="Text rendered into the y image when --task-style=text-reverse.",
    )
    parser.add_argument(
        "--n-cases",
        type=int,
        default=0,
        help=(
            "Batch size used for circuit forward + residual computation. "
            "0 (default) means the *full* task (2^input_bits = 4096 for the "
            "12-bit reverse task) — gives the visualisation rich bit patterns. "
            "Use a smaller value to subsample the middle of the dataset for "
            "perf (e.g. 256) if you want to keep per-tick cost under 100 ms."
        ),
    )
    parser.add_argument(
        "--render-width",
        type=int,
        default=None,
        help=(
            "Width of the canvas the text is rendered into for --task-style="
            "text-reverse. The middle n_cases columns are then cropped out. "
            "Default = max(n_cases, 4096) — keeps long phrases legible at 12 "
            "px tall."
        ),
    )
    args = parser.parse_args()

    print(f"[record-traj] loading run {args.run_id} ...")
    model, cfg, _ = load_run(run_id=args.run_id)
    if args.weights_json:
        print(f"[record-traj] loading quantised weights from {args.weights_json}")
        weights = deserialize_weights(args.weights_json)
    else:
        weights = extract_weights(model, cfg)

    # Resolve output paths (--out is a backwards-compat alias for --out-trajectory).
    out_trajectory = args.out or args.out_trajectory
    out_bootstrap = args.out_bootstrap

    record_for_model(
        weights,
        cfg,
        run_id=args.run_id,
        out_bootstrap=out_bootstrap or None,
        out_trajectory=out_trajectory or None,
        n_ticks=args.n_ticks,
        seed=args.seed,
        task_style=args.task_style,
        text=args.text,
        n_cases=args.n_cases,
        render_width=args.render_width,
    )


if __name__ == "__main__":
    main()
