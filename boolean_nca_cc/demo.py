"""
Live demo session for Self-Organising Digital Circuits.

Shared (GUI-free) infrastructure for the live demo, mirroring train.py
exactly so that what trains/evaluates well in the training script also
runs faithfully in the live demo.

The four paper regimes are exposed as one-click presets that flip a few
high-level toggles. The underlying state always lives in a single
``DemoSession`` object whose configuration mirrors the training config
schema (Hydra DictConfig), so ground truth stays in one place.

This module is deliberately import-cheap and does not pull in any GUI
dependencies, so it can be smoke-tested headlessly.
"""

from __future__ import annotations

import copy
import logging
import os
import random
from collections.abc import Generator
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import jax
import jax.numpy as jp
import numpy as np
import optax
from flax import nnx
from omegaconf import DictConfig, OmegaConf, open_dict

from boolean_nca_cc.circuits.model import gen_circuit, gen_wires, generate_layer_sizes, run_circuit
from boolean_nca_cc.circuits.tasks import TASKS, get_task_data
from boolean_nca_cc.circuits.train import LossConfig
from boolean_nca_cc.training.checkpointing import (
    instantiate_model_from_config,
    load_checkpoint_with_compatibility,
    load_config_from_wandb,
)
from boolean_nca_cc.training.evaluation import (
    StepResult,
    evaluate_model_stepwise_generator,
    get_loss_from_wires_logits,
)
from boolean_nca_cc.training.pool.pool import GraphPool, initialize_graph_pool
from boolean_nca_cc.training.pool.structural_perturbation import (
    apply_knockout_to_circuit,
    compute_damage_params,
    create_eligible_gate_mask,
)
from boolean_nca_cc.utils.configured_graph_builder import (
    configure_build_graph,
    is_configured,
)

log = logging.getLogger(__name__)


class _NullRegion:
    """No-op ``contextmanager``-shaped sentinel used when no profiler is attached."""

    def __enter__(self) -> None:
        return None

    def __exit__(self, *exc: object) -> None:
        return None


_NULL_REGION = _NullRegion()


# ---------------------------------------------------------------------------
# Constants & enums
# ---------------------------------------------------------------------------


class Method(StrEnum):
    BACKPROP = "Backprop"
    TMT = "TMT (loaded model)"


class WiringMode(StrEnum):
    FIXED = "fixed"
    RANDOM = "random"
    GENETIC = "genetic"


class Regime(StrEnum):
    """Evaluation-time regime presets.

    These describe *how* a loaded model is exercised, not what it was
    trained for. The training regime is encoded by the gallery card the
    user picks; these regimes are orthogonal and can be applied to any
    model to recreate the corresponding paper figure.
    """

    LIVE = "I. Live optimisation"
    SELF_HEALING = "II. Self-healing (BP-grown + damage)"
    WIRING_OOD = "III. Wiring-OOD evaluation"
    WIDTH_SWEEP = "IV. Scale-free width sweep"


# Display labels (separate from enum value for GUI flexibility).
REGIME_LABELS: dict[Regime, str] = {
    Regime.LIVE: "I. Live optimisation",
    Regime.SELF_HEALING: "II. Self-healing (BP-grown + damage)",
    Regime.WIRING_OOD: "III. Wiring-OOD evaluation",
    Regime.WIDTH_SWEEP: "IV. Scale-free width sweep",
}

REGIME_DESCRIPTIONS: dict[Regime, str] = {
    Regime.LIVE: (
        "Default. Optimiser runs continuously on the trained-distribution "
        "circuits. Toggle damage to recreate Regime I figure 2."
    ),
    Regime.SELF_HEALING: (
        "Pre-grow with BP to perfect accuracy, then trigger Shotgun and "
        "watch the TMT find a new functional basin (paper §4.2)."
    ),
    Regime.WIRING_OOD: (
        "Flip the eval wiring mode opposite to training. Tests how a "
        "fixed-trained model handles random wires, and vice versa."
    ),
    Regime.WIDTH_SWEEP: (
        "Apply the same model to circuits of different widths (paper "
        "§4.4). The width slider becomes the primary control."
    ),
}


# Default width factors used for the scale-free sweep, matching the
# paper figure 4 (Regime IV).
SCALE_FREE_WIDTH_FACTORS: tuple[float, ...] = (1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0)

# How many steps of metric history to keep for plotting.
DEFAULT_LOG_LENGTH = 1024


# ---------------------------------------------------------------------------
# Pre-trained model gallery (configs/demo_models.yaml)
# ---------------------------------------------------------------------------

#: Canonical recipe IDs. Order = display order in the GUI dropdown.
RECIPE_ORDER: tuple[str, ...] = (
    "fixed_no_damage",
    "fixed_damage",
    "random_no_damage",
    "random_damage",
)


@dataclass
class ModelEntry:
    """One row of the gallery YAML — a curated W&B run."""

    task: str
    recipe: str
    run_id: str | None
    entity: str
    project: str
    prefer_metric: str | None = None
    notes: str | None = None
    label: str = ""
    description: str = ""

    @property
    def is_available(self) -> bool:
        return self.run_id is not None and self.run_id != ""

    @property
    def wiring_mode(self) -> str:
        return "random" if self.recipe.startswith("random") else "fixed"

    @property
    def damage_training(self) -> bool:
        return self.recipe.endswith("_damage")

    def display_name(self) -> str:
        return f"{self.task} · {self.label or self.recipe}"


@dataclass
class ModelGallery:
    """The full curated registry, indexed by (task, recipe)."""

    entries: list[ModelEntry]
    recipe_labels: dict[str, str] = field(default_factory=dict)
    recipe_descriptions: dict[str, str] = field(default_factory=dict)
    source_path: str | None = None

    @classmethod
    def from_yaml(cls, path: str | None = None) -> ModelGallery:
        """
        Load the gallery from a YAML file (default: ``configs/demo_models.yaml``).

        Missing file → empty gallery (the GUI shows the manual run-ID input
        as the only option). Malformed entries log a warning and are skipped.
        """
        if path is None:
            path = os.path.join(_project_root(), "configs", "demo_models.yaml")
        if not os.path.exists(path):
            log.warning(f"Model gallery file not found at {path}; gallery will be empty.")
            return cls(entries=[], source_path=path)

        try:
            doc = OmegaConf.load(path)
        except Exception as e:
            log.error(f"Failed to parse {path}: {e}")
            return cls(entries=[], source_path=path)

        defaults = doc.get("defaults", {}) or {}
        recipe_labels = {k: v.get("label", k) for k, v in (doc.get("recipes", {}) or {}).items()}
        recipe_descs = {
            k: v.get("description", "") for k, v in (doc.get("recipes", {}) or {}).items()
        }

        entries: list[ModelEntry] = []
        for raw in doc.get("models", []) or []:
            try:
                recipe = raw["recipe"]
                entry = ModelEntry(
                    task=raw["task"],
                    recipe=recipe,
                    run_id=raw.get("run_id"),
                    entity=raw.get("entity", defaults.get("entity", "gbena")),
                    project=raw.get("project", defaults.get("project", "boolean-nca-cc")),
                    prefer_metric=raw.get("prefer_metric"),
                    notes=raw.get("notes"),
                    label=recipe_labels.get(recipe, recipe),
                    description=recipe_descs.get(recipe, ""),
                )
                entries.append(entry)
            except KeyError as e:
                log.warning(f"Skipping malformed gallery entry {raw}: missing {e}")
        return cls(
            entries=entries,
            recipe_labels=recipe_labels,
            recipe_descriptions=recipe_descs,
            source_path=path,
        )

    # ---- Lookup helpers used by the GUI ----

    def tasks(self) -> list[str]:
        """Tasks that have at least one entry (regardless of availability)."""
        seen: list[str] = []
        for e in self.entries:
            if e.task not in seen:
                seen.append(e.task)
        return seen

    def available_tasks(self) -> list[str]:
        """Tasks that have at least one *available* (non-null run_id) entry."""
        return [t for t in self.tasks() if any(self.lookup(t, r) for r in RECIPE_ORDER)]

    def recipes_for_task(self, task: str) -> list[str]:
        """Recipe IDs declared for this task (in canonical order)."""
        declared = {e.recipe for e in self.entries if e.task == task}
        return [r for r in RECIPE_ORDER if r in declared]

    def lookup(self, task: str, recipe: str) -> ModelEntry | None:
        """Return the entry for (task, recipe) iff it has a valid run_id."""
        for e in self.entries:
            if e.task == task and e.recipe == recipe and e.is_available:
                return e
        return None

    def get(self, task: str, recipe: str) -> ModelEntry | None:
        """Return the entry for (task, recipe) regardless of availability."""
        for e in self.entries:
            if e.task == task and e.recipe == recipe:
                return e
        return None

    def first_available(self) -> ModelEntry | None:
        for e in self.entries:
            if e.is_available:
                return e
        return None


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def _project_root() -> str:
    """Absolute path to the repo root (parent of the boolean_nca_cc package)."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))


def load_default_config(config_name: str = "config") -> DictConfig:
    """
    Load a Hydra config from ``configs/`` without going through ``@hydra.main``.

    This gives the demo a self-consistent default config (matching
    train.py defaults) when no W&B run has been loaded yet, so the user
    can play with Backprop on a sensible task before bringing in a
    trained TMT.
    """
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    config_dir = os.path.join(_project_root(), "configs")
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name=config_name)

    # Disable W&B for demo bootstrap; the user can opt in later.
    with open_dict(cfg):
        cfg.wandb.enabled = False

    return cfg


# ---------------------------------------------------------------------------
# Random key derivation — matches train.py / eval_datasets.py exactly
# ---------------------------------------------------------------------------


def derive_pool_keys(cfg: DictConfig, distribution: str = "auto") -> tuple[jax.Array, jax.Array]:
    """
    Derive ``(wires_key, logits_key)`` for the demo pool.

    The conventions exactly mirror what train.py and
    ``create_unified_evaluation_datasets`` do:

    * ``train_key = PRNGKey(cfg.seed)``
    * ``eval_key = PRNGKey(cfg.eval_seed)`` if set, else ``fold_in(train_key, 1)``

    Three modes are exposed:

    ``distribution="auto"`` (default):
        Match how the training script samples its initial pool:

        * fixed / genetic → IN-distribution eval key (= ``eval_key``)
        * random          → first split off ``train_key``
          (matches ``train_key, pool_key = split(train_key)``
          in ``train_loop.py``)

    ``distribution="in_distribution"``:
        Always use the eval IN-distribution key. For random wiring this
        falls back to OOD because there is no notion of in-distribution
        for random training.

    ``distribution="ood"``:
        Always use the eval OOD key (``fold_in(eval_key, 42)``), giving
        fresh random circuits drawn from the same distribution but
        distinct from any seen during training.
    """
    train_key = jax.random.PRNGKey(int(cfg.seed))

    eval_seed = cfg.get("eval_seed", None)
    eval_key = (
        jax.random.PRNGKey(int(eval_seed))
        if eval_seed is not None
        else jax.random.fold_in(train_key, 1)
    )

    wiring_mode = cfg.training.wiring_mode

    if distribution == "ood":
        pool_key = jax.random.fold_in(eval_key, 42)
    elif distribution == "in_distribution":
        if wiring_mode in ("fixed", "genetic"):
            pool_key = eval_key
        else:
            pool_key = jax.random.fold_in(eval_key, 42)
    elif distribution == "auto":
        if wiring_mode in ("fixed", "genetic"):
            pool_key = eval_key
        else:
            # Match train_loop.py: ``train_key, pool_key = split(train_key)``.
            # Note: JAX's threefry hash makes ``split(key)[1] == fold_in(key, 1)``
            # for any seed, so the pool wires we sample here are byte-identical
            # to those a fixed-mode run would land on with the same seed.
            _, pool_key = jax.random.split(train_key)
    else:
        raise ValueError(f"Unknown distribution mode: {distribution!r}")

    wires_key, logits_key = jax.random.split(pool_key)
    return wires_key, logits_key


# ---------------------------------------------------------------------------
# Pool / model assembly
# ---------------------------------------------------------------------------


def build_layer_sizes(cfg: DictConfig, width_factor: float | None = None) -> list[tuple[int, int]]:
    """Build layer sizes from config, optionally overriding the width factor."""
    wf = float(cfg.circuit.width_factor) if width_factor is None else float(width_factor)
    return generate_layer_sizes(
        int(cfg.circuit.input_bits),
        int(cfg.circuit.output_bits),
        int(cfg.circuit.arity),
        layer_n=int(cfg.circuit.num_layers),
        width_factor=wf,
    )


def configure_global_graph_builder(cfg: DictConfig) -> None:
    """Configure the global graph builder once per session, like train.py does."""
    configure_build_graph(
        neighboring_connections=cfg.graph.neighboring_connections,
        bidirectional_edges=cfg.graph.bidirectional_edges,
        use_dist_pe=cfg.graph.get("use_dist_pe", False),
        use_rwse=cfg.graph.get("use_rwse", False),
        rwse_k=cfg.graph.get("rwse_k", 8),
    )



def initialize_demo_pool(
    cfg: DictConfig,
    layer_sizes: list[tuple[int, int]],
    pool_size: int,
    distribution: str = "auto",
    noise_scale: float | None = None,
) -> GraphPool:
    """
    Build a small evaluation pool that matches train.py's distribution.

    Args:
        cfg: Loaded config (Hydra DictConfig).
        layer_sizes: Output of ``build_layer_sizes`` (may differ from cfg
            for the scale-free sweep).
        pool_size: Number of circuits in the demo pool. Typically much
            smaller than ``cfg.pool.size`` to keep the GUI responsive.
        distribution: ``"auto" | "in_distribution" | "ood"``; see
            ``derive_pool_keys`` for semantics.
        noise_scale: Override for ``cfg.pool.noise_scale`` (None = use cfg).
    """
    if not is_configured():
        configure_global_graph_builder(cfg)

    wires_key, logits_key = derive_pool_keys(cfg, distribution=distribution)
    ns = float(cfg.pool.noise_scale) if noise_scale is None else float(noise_scale)

    return initialize_graph_pool(
        wires_key=wires_key,
        logits_key=logits_key,
        pool_size=pool_size,
        layer_sizes=layer_sizes,
        arity=int(cfg.circuit.arity),
        input_n=int(cfg.circuit.input_bits),
        circuit_hidden_dim=int(cfg.model.circuit_hidden_dim),
        wiring_mode=cfg.training.wiring_mode,
        initial_diversity=int(cfg.training.initial_diversity),
        noise_scale=ns,
    )


def load_task_data(
    cfg: DictConfig,
) -> tuple[jp.ndarray, jp.ndarray, jp.ndarray | None, jp.ndarray | None, jp.ndarray, jp.ndarray]:
    """Get the task data tuple ``(x_train, y_train, x_test, y_test, x_total, y_total)``."""
    case_n = 1 << int(cfg.circuit.input_bits)

    test_num = cfg.training.get("test_num", None)
    if test_num is None:
        test_ratio = None
    elif isinstance(test_num, float):
        test_ratio = min(test_num, 0.2)
    else:
        test_ratio = min(float(test_num) / case_n, 0.2)

    eval_seed = cfg.get("eval_seed", None)
    train_key = jax.random.PRNGKey(int(cfg.seed))
    eval_key = (
        jax.random.PRNGKey(int(eval_seed))
        if eval_seed is not None
        else jax.random.fold_in(train_key, 1)
    )

    from boolean_nca_cc.utils.task_compat import get_fixed_task_name, get_task_text

    task_name = get_fixed_task_name(cfg)
    if task_name is None:
        raise ValueError(
            "demo.py requires a fixed task; cfg has no cfg.tasks.name (new) "
            "or cfg.circuit.task (legacy). Sampler-mode configs are not "
            "supported here."
        )
    (x_train, y_train), (x_test, y_test), (x_total, y_total) = get_task_data(
        task_name,
        case_n,
        input_bits=int(cfg.circuit.input_bits),
        output_bits=int(cfg.circuit.output_bits),
        text=get_task_text(cfg),
        train_test_split=test_ratio is not None,
        test_ratio=test_ratio,
        seed=eval_key,
    )
    return x_train, y_train, x_test, y_test, x_total, y_total


def resolve_permanence(value: Any) -> float:
    """
    Coerce ``cfg.damage.permanent`` to a float in ``[0, 1]``.

    The training config schema (and ``compute_damage_params``) accept
    three flavours of value: a float, a bool, or the string ``"random"``
    (≡ 0.5). Mirrors the resolver inside
    ``evaluate_model_stepwise_batched`` so the live demo handles loaded
    runs identically to the eval scan.
    """
    if isinstance(value, str):
        if value.strip().lower() == "random":
            return 0.5
        try:
            return float(value)
        except ValueError as e:
            raise ValueError(f"Unrecognised damage.permanent value: {value!r}") from e
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if value is None:
        return 1.0
    return float(value)


def reload_model_for_layer_sizes(
    cfg: DictConfig,
    checkpoint_path: str,
    layer_sizes: list[tuple[int, int]],
    seed: int = 0,
) -> Any:
    """
    Re-instantiate a model on a (possibly resized) graph and load weights.

    For the scale-free sweep (Regime IV) we want to apply the *same
    learned weights* to circuits of varying width. The TMT is parameter-
    free with respect to size: only the attention mask depends on the
    node count. This helper builds a dummy graph from ``layer_sizes`` to
    get the correct ``n_node``, instantiates a fresh model, then copies
    the trained weights from the checkpoint.
    """
    if not is_configured():
        configure_global_graph_builder(cfg)

    from boolean_nca_cc.utils.configured_graph_builder import (
        configured_build_graph as build_graph,
    )

    wires_key, logits_key = jax.random.split(jax.random.PRNGKey(int(seed)))
    wires, logits = gen_circuit(
        wires_key,
        logits_key,
        layer_sizes,
        arity=int(cfg.circuit.arity),
        noise_scale=float(cfg.pool.noise_scale),
    )
    graph = build_graph(
        wires=wires,
        logits=logits,
        input_n=int(cfg.circuit.input_bits),
        arity=int(cfg.circuit.arity),
        circuit_hidden_dim=int(cfg.model.circuit_hidden_dim),
    )
    n_node = int(graph.n_node[0])

    init_model = instantiate_model_from_config(cfg, seed=seed, n_node=n_node)
    model = copy.deepcopy(init_model)

    loaded = load_checkpoint_with_compatibility(checkpoint_path)
    nnx.update(model, loaded["model"])
    return model


# ---------------------------------------------------------------------------
# UI-state helpers (small enough that a dataclass is overkill, but nice
# to keep the GUI input contract documented in one place).
# ---------------------------------------------------------------------------


@dataclass
class WandbCoordinates:
    """Where to look on W&B when the user clicks 'Load model'."""

    entity: str = "gbena"
    project: str = "boolean-nca-cc"
    download_dir: str = "saves"
    run_id: str | None = None
    prefer_metric: str | None = "eval_damaged_in_test_hard_accuracy"


# ---------------------------------------------------------------------------
# DemoSession — the single source of truth for the demo
# ---------------------------------------------------------------------------


@dataclass
class DemoSession:
    """
    All non-GUI state for the live demo.

    A session is the runtime equivalent of ``train.py``'s ``main(cfg)``
    function: it owns a config, a pool, optionally a model, a metric
    history, and exposes step/regenerate/damage methods that the GUI
    drives.

    The session is intentionally side-effect heavy (mutates its own
    state in place) because the GUI loop is the natural owner of the
    state — we don't want to ferry an updated copy around every frame.
    """

    # --- Core config ----------------------------------------------------
    cfg: DictConfig
    method: Method = Method.BACKPROP

    # Snapshot of the config we *trained* under (the loaded W&B run, or
    # the default Hydra config at boot). The live ``cfg`` may diverge
    # from this when the user pushes OOD knobs; ``ood_diffs()`` reports
    # the divergence for the GUI to surface.
    training_cfg: DictConfig | None = None
    training_width_factor: float | None = None  # snapshot of ``width_factor`` at load

    # --- Pool / model dimensioning -------------------------------------
    pool_size: int = 16
    width_factor: float | None = None  # None = use cfg.circuit.width_factor

    # --- Mutable state (populated by ``bootstrap``) --------------------
    layer_sizes: list[tuple[int, int]] = field(default_factory=list)
    pool: GraphPool | None = None
    current_circuit_idx: int = 0
    wires: list[jp.ndarray] = field(default_factory=list)
    logits: list[jp.ndarray] = field(default_factory=list)
    logits0: list[jp.ndarray] = field(default_factory=list)  # initial state for reset
    gate_mask_flat: jp.ndarray | None = None  # damage mask, flat per-circuit

    # --- Task data ------------------------------------------------------
    x_train: jp.ndarray | None = None
    y_train: jp.ndarray | None = None
    x_test: jp.ndarray | None = None
    y_test: jp.ndarray | None = None
    x_total: jp.ndarray | None = None
    y_total: jp.ndarray | None = None

    # --- Damage ---------------------------------------------------------
    damage_params: dict = field(default_factory=dict)
    damage_enabled: bool = False  # User toggle (mirrors cfg.damage.enabled by default)
    permanent_damage: float = 1.0

    # --- Model & step driver -------------------------------------------
    model: Any | None = None  # nnx Module; None = no TMT loaded yet
    model_run_id: str | None = None
    checkpoint_path: str | None = None
    loaded_entry: ModelEntry | None = None  # gallery card the user picked, if any
    bp_optimizer: optax.GradientTransformation | None = None
    bp_opt_state: Any | None = None
    bp_learning_rate: float = 1.0
    tmt_generator: Generator[StepResult, None, None] | None = None
    last_step_result: StepResult | None = None

    # Curated gallery loaded once at session start (configs/demo_models.yaml).
    gallery: ModelGallery = field(default_factory=ModelGallery.from_yaml)

    # --- Metric logs ----------------------------------------------------
    step_count: int = 0
    log_length: int = DEFAULT_LOG_LENGTH
    loss_log: np.ndarray = field(default_factory=lambda: np.zeros(DEFAULT_LOG_LENGTH, np.float32))
    hard_loss_log: np.ndarray = field(
        default_factory=lambda: np.zeros(DEFAULT_LOG_LENGTH, np.float32)
    )
    accuracy_log: np.ndarray = field(
        default_factory=lambda: np.zeros(DEFAULT_LOG_LENGTH, np.float32)
    )
    hard_accuracy_log: np.ndarray = field(
        default_factory=lambda: np.zeros(DEFAULT_LOG_LENGTH, np.float32)
    )

    # --- Visualization-only fields (cached predictions / activations) --
    activations: list[jp.ndarray] = field(default_factory=list)
    err_mask: jp.ndarray | None = None
    current_pred: jp.ndarray | None = None
    current_pred_hard: jp.ndarray | None = None

    # --- Probabilistic damage state (for streaming damage in the demo) -
    _prob_damage_key: jax.Array = field(default_factory=lambda: jax.random.PRNGKey(99), repr=False)
    _perm_damage_key: jax.Array = field(default_factory=lambda: jax.random.PRNGKey(77), repr=False)
    # Cached damage settings the active TMT generator was built with;
    # used to detect ``damage_enabled``/``permanent_damage`` toggles and
    # restart the generator so the new settings take effect.
    _generator_p_fault: float | None = field(default=None, repr=False)
    _generator_permanent: float | None = field(default=None, repr=False)

    # --- Profiling --------------------------------------------------------
    # Optional ``boolean_nca_cc.perf.FrameProfiler``. The GUI sets it so
    # we can break ``step()`` into BP forward/grad/update + viz forward
    # passes. ``None`` ⇒ profiling is a noop.
    prof: Any = field(default=None, repr=False)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_default_config(cls, **overrides: Any) -> DemoSession:
        """Build a session from the default Hydra config with optional overrides.

        Note: ``training_cfg`` stays ``None`` until a model is actually
        loaded from W&B. Until then we are running pure Backprop and
        nothing is "out of distribution" in any meaningful sense — the
        OOD panel stays dormant.
        """
        cfg = load_default_config()
        session = cls(cfg=cfg, **overrides)
        session.bootstrap()
        return session

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def bootstrap(self) -> None:
        """
        (Re)build everything: layer sizes, pool, task data, damage params.

        Safe to call after any change to ``cfg`` or ``width_factor``.
        Preserves ``method`` and (if compatible) the loaded model.
        """
        configure_global_graph_builder(self.cfg)

        self.layer_sizes = build_layer_sizes(self.cfg, width_factor=self.width_factor)

        # Compute damage parameters from cfg, like train.py does.
        self.damage_params = compute_damage_params(self.cfg, self.layer_sizes, log)
        # ``permanent`` may be a float, a bool, or the string ``"random"``
        # in W&B-loaded configs — normalise via the shared resolver so
        # everything downstream sees a clean float.
        self.permanent_damage = resolve_permanence(self.damage_params.get("permanent", 1.0))
        # if self.damage_enabled is False:
        #     # Default to whatever the cfg says (so loading a damage-trained
        #     # run shows damage live, like the paper figure 2).
        #     self.damage_enabled = bool(self.cfg.damage.get("enabled", False))

        # Build the demo pool.
        self.pool = initialize_demo_pool(
            self.cfg,
            self.layer_sizes,
            pool_size=self.pool_size,
        )

        # Pick a circuit out of the pool to visualize.
        self.current_circuit_idx = min(self.current_circuit_idx, self.pool_size - 1)
        self._sync_current_circuit()

        # Get task data.
        self.x_train, self.y_train, self.x_test, self.y_test, self.x_total, self.y_total = (
            load_task_data(self.cfg)
        )

        # If a model was previously loaded, re-instantiate it on the new
        # graph (only matters when the width_factor changed).
        if self.checkpoint_path is not None and self.method == Method.TMT:
            try:
                self.model = reload_model_for_layer_sizes(
                    self.cfg, self.checkpoint_path, self.layer_sizes, seed=int(self.cfg.seed)
                )
            except Exception as e:
                log.warning(f"Could not re-instantiate TMT for new layer sizes: {e}")
                self.model = None
                self.method = Method.BACKPROP

        self._reset_optimizers()
        self._reset_logs()
        self._refresh_visualization()

    def apply_cfg_changes(self, **changes: Any) -> None:
        """
        Patch nested config keys and re-bootstrap.

        Example::

            session.apply_cfg_changes(
                **{
                    "circuit.task": "add",
                    "training.wiring_mode": "random",
                }
            )
        """
        if not changes:
            return
        with open_dict(self.cfg):
            for dotted_key, value in changes.items():
                node = self.cfg
                parts = dotted_key.split(".")
                for p in parts[:-1]:
                    node = node[p]
                node[parts[-1]] = value
        self.bootstrap()

    # ---------- Pool / circuit selection ----------

    def select_circuit_idx(self, idx: int) -> None:
        """Switch to a different circuit in the pool (cheap, no rebuild)."""
        if self.pool is None:
            return
        idx = int(np.clip(idx, 0, self.pool_size - 1))
        self.current_circuit_idx = idx
        self._sync_current_circuit()
        self._reset_optimizers()
        self._reset_logs()
        self._refresh_visualization()

    def regenerate_pool(self, distribution: str = "auto") -> None:
        """Resample the pool (e.g. after ``Shuffle Wires``)."""
        # Re-derive keys after fold_in'ing a small random salt so we get
        # a *new* pool while still matching the wiring distribution.
        salt = random.randint(0, 1 << 30)
        with open_dict(self.cfg):
            self.cfg.seed = int(self.cfg.seed) ^ salt
        self.pool = initialize_demo_pool(
            self.cfg,
            self.layer_sizes,
            pool_size=self.pool_size,
            distribution=distribution,
        )
        self.current_circuit_idx = 0
        self._sync_current_circuit()
        self._reset_optimizers()
        self._reset_logs()
        self._refresh_visualization()

    def reset_circuit(self) -> None:
        """Reset the current circuit's logits to its initial (soft-wire) state."""
        self.logits = list(self.logits0)
        self.gate_mask_flat = None
        self._reset_optimizers()
        self._reset_logs()
        self._refresh_visualization()

    # ---------- Wire shuffle (Regime III demonstration) ----------

    def shuffle_wires(self) -> None:
        """Replace the current circuit's wires with freshly sampled random ones.

        This is the live-demo handle for Regime III in the manuscript: a
        wiring-agnostic TMT (i.e. one trained on a random-wire pool)
        should be able to recover good accuracy on a brand-new topology
        it has never seen, even though we keep the *optimised logits*
        from the previous run intact.

        Deliberately:

        * **Logits are kept** — the OOD test is "drop a different wiring
          on top of an already-trained circuit" and watch the TMT (or BP)
          re-route. Resetting logits would conflate two effects.
        * **Metric logs are kept** — the whole point is to see the
          accuracy *drop* the moment the wires change and (hopefully) a
          *recovery* over the next few hundred steps. Resetting the log
          would erase the comparison.
        * **Damage state (``gate_mask_flat``) is kept** — same
          rationale: we want to test wiring OOD in isolation.
        """
        salt = random.randint(0, 1 << 30)
        key = jax.random.PRNGKey(salt)
        arity = int(self.cfg.circuit.arity)
        n_layers = len(self.layer_sizes)
        if n_layers <= 1:
            return
        keys = jax.random.split(key, n_layers - 1)
        new_wires: list[jp.ndarray] = []
        in_n = self.layer_sizes[0][0]
        for i, (out_n, group_size) in enumerate(self.layer_sizes[1:]):
            new_wires.append(gen_wires(keys[i], in_n, out_n, arity, group_size))
            in_n = out_n
        self.wires = new_wires
        # Generator (TMT) / optimiser (BP) need to pick up the new topology.
        if self.method == Method.TMT:
            self._restart_tmt_generator()
        else:
            self._reset_optimizers()
        self._refresh_visualization()

    # ---------- Damage ----------

    def apply_shotgun_damage(self, num_knockouts: int | None = None) -> None:
        """
        Apply a single discrete-knockout volley (the 'shotgun' from §4.1).

        With no argument, defaults to ``damage_params['knockouts_per_event']``,
        which is the value derived from cfg.damage.target_damage_fraction.
        """
        if num_knockouts is None:
            num_knockouts = int(self.damage_params.get("knockouts_per_event", 1) or 1)
        if num_knockouts <= 0:
            return
        seed_val = random.randint(0, 1 << 30)
        key = jax.random.PRNGKey(seed_val)
        modified_logits, knockout_mask_flat = apply_knockout_to_circuit(
            key,
            self._stacked_logits(),
            self.layer_sizes,
            num_knockouts=num_knockouts,
            faulty_value=float(self.damage_params.get("faulty_logit_value", -10.0)),
            flat=True,
        )
        self.logits = self._unstack_logits(modified_logits)
        if self.gate_mask_flat is None:
            self.gate_mask_flat = jp.ones(modified_logits.shape[0], dtype=jp.float32)
        self.gate_mask_flat = self.gate_mask_flat * knockout_mask_flat

        # If the TMT generator is running, restart it so the next step
        # picks up the damaged logits.
        self._restart_tmt_generator()
        self._refresh_visualization()

    def damage_gate(self, flat_idx: int) -> None:
        """Permanently knock out a single gate by its flat-circuit index.

        This is the click-to-damage handle: the user clicks a gate in the
        circuit diagram and we add it to the existing damage set
        (``gate_mask_flat``), exactly like ``apply_shotgun_damage`` would
        for a randomly-chosen gate.

        Input/output gates are silently ignored — only hidden-layer
        gates are eligible for damage (matching the training-time
        eligibility mask).

        Args:
            flat_idx: Flat index into the per-circuit gate vector
                (layout: ``[input | hidden_1 | ... | hidden_k | output]``).
        """
        flat_idx = int(flat_idx)
        # Eligibility: skip input + output layers.
        input_n = int(self.layer_sizes[0][0])
        output_n = int(self.layer_sizes[-1][0])
        total_gates = sum(g for g, _ in self.layer_sizes)
        if flat_idx < input_n or flat_idx >= total_gates - output_n:
            return

        if self.gate_mask_flat is None:
            self.gate_mask_flat = jp.ones(total_gates, dtype=jp.float32)

        # Already damaged → nothing to do.
        if float(self.gate_mask_flat[flat_idx]) == 0.0:
            return

        faulty_value = float(self.damage_params.get("faulty_logit_value", -10.0))
        self.gate_mask_flat = self.gate_mask_flat.at[flat_idx].set(0.0)
        # Push the faulty logit into the stacked layout, then unstack.
        flat_logits = self._stacked_logits().at[flat_idx, :].set(faulty_value)
        self.logits = self._unstack_logits(flat_logits)

        # The TMT generator carries its own graph state; rebuild so it
        # picks up the new mask + logits via ``initial_gate_knockout_mask``.
        self._restart_tmt_generator()
        self._refresh_visualization()

    # ---------- Pre-grow with BP (Regime II) ----------

    def pre_grow_with_bp(self, n_steps: int = 300, learning_rate: float = 1.0) -> None:
        """
        Run ``n_steps`` of BP on the current circuit, then return control.

        This is used for Regime II: pre-converge a circuit with
        backpropagation, then hand off to the TMT to demonstrate
        pure-repair behaviour. Damage and probabilistic faults are
        *disabled* during pre-growth to avoid contaminating the
        starting state.
        """
        opt = optax.adamw(learning_rate, 0.8, 0.8, weight_decay=1e-1)
        opt_state = opt.init(self.logits)
        loss_cfg = self._loss_cfg()
        x_data, y_data = self._train_xy()

        @jax.jit
        def update(logits, wires, opt_state):
            def loss_fn(lg):
                loss, _ = get_loss_from_wires_logits(lg, wires, x_data, y_data, loss_cfg)
                return loss

            grads = jax.grad(loss_fn)(logits)
            updates, new_opt_state = opt.update(grads, opt_state, logits)
            new_logits = optax.apply_updates(logits, updates)
            return new_logits, new_opt_state

        for _ in range(int(n_steps)):
            self.logits, opt_state = update(self.logits, self.wires, opt_state)

        # The freshly-grown circuit becomes the new "initial state" so
        # that ``Reset Circuit`` returns here instead of the soft wires.
        self.logits0 = list(self.logits)
        self._reset_optimizers()
        self._reset_logs()
        self._restart_tmt_generator()
        self._refresh_visualization()

    # ---------- Optimization step (one frame's worth) ----------

    def step(self, n_message_steps: int = 1) -> None:
        """
        Run one optimisation step (BP) or ``n_message_steps`` TMT steps.

        Returns silently. Updates ``self.logits``, ``self.last_step_result``,
        the metric logs, and the visualization caches.

        Damage is applied **before** the model step in both methods, so
        the TMT/BP gets a chance to repair the freshly-injected fault
        within the same frame — this matches the ordering used inside
        ``run_model_scan_with_loss`` during training.
        """
        # Make sure the TMT generator's damage settings match the live
        # GUI knobs (``damage_enabled`` / ``permanent_damage``). This is
        # the cheap path: when nothing has changed, the comparison is a
        # no-op; when the user flips a toggle we rebuild the generator
        # so the new fault stream takes effect.
        if self.method == Method.TMT and self.tmt_generator is not None:
            self._sync_generator_damage_settings()

        with self._region("step.optimise"):
            if self.method == Method.BACKPROP:
                # BP path: damage is applied externally because BP has
                # no inner loop to fold it into.
                if self.damage_enabled and self.damage_params.get("p_fault_eval"):
                    with self._region("step.damage"):
                        self._apply_step_probabilistic_damage()
                self._step_bp()
            else:
                # TMT path: damage is applied inside the generator
                # (``evaluate_model_stepwise_generator`` was built with
                # the live ``p_fault`` / ``permanent_damage`` settings).
                self._step_tmt(n_message_steps)
        with self._region("step.metrics"):
            self._record_metrics()
        self.step_count += 1
        with self._region("step.viz_forward"):
            self._refresh_visualization()

    def _sync_generator_damage_settings(self) -> None:
        """Restart the TMT generator if the live damage knobs have drifted.

        We snapshot ``p_fault`` and ``permanent_damage`` whenever we
        rebuild the generator (in ``_restart_tmt_generator``); if the
        user has since toggled them via the GUI, we rebuild so the new
        fault stream takes effect.
        """
        live_p_fault = (
            float(self.damage_params.get("p_fault_eval"))
            if self.damage_enabled and self.damage_params.get("p_fault_eval")
            else None
        )
        live_perm = float(self.permanent_damage)
        if live_p_fault != self._generator_p_fault or live_perm != self._generator_permanent:
            self._restart_tmt_generator()

    # ------------------------------------------------------------------
    # Profiling helper
    # ------------------------------------------------------------------

    def _region(self, name: str):
        """Return ``self.prof.region(name)`` or a no-op context manager."""
        if self.prof is None:
            return _NULL_REGION
        return self.prof.region(name)

    # ---------- W&B model loading ----------

    def load_model_from_wandb(
        self, coords: WandbCoordinates, entry: ModelEntry | None = None
    ) -> bool:
        """
        Pull a config + checkpoint from W&B and re-bootstrap the session.

        After this, ``self.cfg`` and ``self.training_cfg`` both hold the
        loaded run's training config. Subsequent OOD knob changes mutate
        ``cfg`` only; ``training_cfg`` stays pinned so ``ood_diffs()``
        can flag the divergence.

        ``entry`` is the gallery card that was selected (if any), so the
        GUI can display its description / notes; pass ``None`` for
        manual run-ID loads.
        """
        try:
            loaded_cfg, checkpoint_path, run_id = load_config_from_wandb(
                run_id=coords.run_id,
                entity=coords.entity,
                project=coords.project,
                download_dir=coords.download_dir,
                # filename="best_model_eval_damaged_in_test_hard_accuracy",
                filename=f"best_model_{coords.prefer_metric}",
                use_cache=True,
                force_download=False,
                select_by_best_metric=False,
            )
        except Exception as e:
            log.error(f"Failed to load run from W&B: {e}")
            return False

        self.cfg = loaded_cfg
        # Snapshot the training config; from this point on, anything
        # the user changes counts as an OOD override.
        self.training_cfg = OmegaConf.create(OmegaConf.to_container(loaded_cfg, resolve=True))
        self.training_width_factor = float(loaded_cfg.circuit.width_factor)
        self.checkpoint_path = checkpoint_path
        self.model_run_id = run_id
        self.loaded_entry = entry
        self.method = Method.TMT
        # Reset any leftover OOD overrides from a previous load: the
        # session tracks ``width_factor`` separately from cfg.
        self.width_factor = None
        # ``bootstrap`` will re-instantiate the model on the freshly built
        # graph (and load the checkpoint weights) because we just set
        # ``checkpoint_path`` and switched the method to TMT.
        self.bootstrap()
        return True

    def load_model_from_gallery(self, task: str, recipe: str) -> bool:
        """
        Look up a gallery entry by ``(task, recipe)`` and load it.

        Returns ``False`` if the entry is missing / unavailable / fails
        to download.
        """
        entry = self.gallery.lookup(task, recipe)
        if entry is None:
            log.warning(f"No gallery entry available for ({task}, {recipe})")
            return False
        coords = WandbCoordinates(
            entity=entry.entity,
            project=entry.project,
            run_id=entry.run_id,
            prefer_metric=entry.prefer_metric,
        )
        return self.load_model_from_wandb(coords, entry=entry)

    # ---------- OOD diff tracking ----------

    def ood_diffs(self) -> dict[str, tuple[Any, Any]]:
        """
        Compare the live config to the training snapshot.

        Returns ``{knob_name: (trained_value, current_value)}`` for every
        knob that has diverged from the loaded run's training config.
        Empty dict ⇒ in-distribution evaluation.

        Knobs surfaced (matching the OOD playground panel in the GUI):
        ``task``, ``wiring_mode``, ``num_layers``, ``width_factor``,
        ``damage_enabled``.
        """
        if self.training_cfg is None:
            return {}

        diffs: dict[str, tuple[Any, Any]] = {}
        t = self.training_cfg
        c = self.cfg

        if c.circuit.task != t.circuit.task:
            diffs["task"] = (str(t.circuit.task), str(c.circuit.task))
        if c.training.wiring_mode != t.training.wiring_mode:
            diffs["wiring_mode"] = (str(t.training.wiring_mode), str(c.training.wiring_mode))
        if int(c.circuit.num_layers) != int(t.circuit.num_layers):
            diffs["num_layers"] = (int(t.circuit.num_layers), int(c.circuit.num_layers))

        wf_now = (
            float(self.width_factor)
            if self.width_factor is not None
            else float(c.circuit.width_factor)
        )
        wf_trained = float(self.training_width_factor or t.circuit.width_factor)
        if abs(wf_now - wf_trained) > 1e-3:
            diffs["width_factor"] = (round(wf_trained, 3), round(wf_now, 3))

        damage_trained = bool(t.damage.get("enabled", False))
        if bool(self.damage_enabled) != damage_trained:
            diffs["damage_enabled"] = (damage_trained, bool(self.damage_enabled))

        return diffs

    @property
    def is_ood(self) -> bool:
        return bool(self.ood_diffs())

    def reset_to_training_config(self) -> None:
        """Snap every OOD knob back to the loaded run's training config."""
        if self.training_cfg is None:
            return
        # Restore mutable fields first so bootstrap sees the pinned values.
        self.cfg = OmegaConf.create(OmegaConf.to_container(self.training_cfg, resolve=True))
        self.width_factor = None
        self.damage_enabled = bool(self.cfg.damage.get("enabled", False))
        self.bootstrap()

    # ------------------------------------------------------------------
    # Regime presets (evaluation-time)
    # ------------------------------------------------------------------

    def apply_regime(self, regime: Regime) -> None:
        """
        Apply an *evaluation* regime preset to whichever model is loaded.

        Regimes are orthogonal to the training recipe (the gallery card):
        any model can be exercised in any regime, with corresponding
        OOD overrides flagged in the GUI.

        - ``LIVE``        : default — restore training config, optimiser runs.
        - ``SELF_HEALING``: pre-grow with BP to convergence; user triggers
                            damage to see TMT repair (paper §4.2).
        - ``WIRING_OOD``  : flip wiring at eval time vs training (Regime III).
        - ``WIDTH_SWEEP`` : promote the width slider; same model, varied size.
        """
        if regime == Regime.LIVE:
            # Restore training config; user can still flip damage etc.
            if self.training_cfg is not None:
                self.reset_to_training_config()

        elif regime == Regime.SELF_HEALING:
            # Pre-grow with BP, then await user damage trigger.
            self.method = Method.BACKPROP
            self.pre_grow_with_bp(n_steps=300)
            if self.model is not None:
                self.method = Method.TMT
                self._restart_tmt_generator()

        elif regime == Regime.WIRING_OOD:
            # Flip wiring opposite to training to recreate the OOD
            # evaluation: fixed-trained → eval on random, and vice versa.
            trained_mode = (
                str(self.training_cfg.training.wiring_mode)
                if self.training_cfg is not None
                else str(self.cfg.training.wiring_mode)
            )
            new_mode = "random" if trained_mode == "fixed" else "fixed"
            self.apply_cfg_changes(**{"training.wiring_mode": new_mode})

        elif regime == Regime.WIDTH_SWEEP:
            # Park the width factor at the training default; the GUI
            # exposes the slider that drives the sweep.
            self.width_factor = float(
                self.training_width_factor
                if self.training_width_factor is not None
                else self.cfg.circuit.width_factor
            )
            self.bootstrap()

        else:
            raise ValueError(f"Unknown regime: {regime!r}")

        self._refresh_visualization()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _train_xy(self) -> tuple[jp.ndarray, jp.ndarray]:
        """Pick the training/test data tuple to use for the running step."""
        if self.x_train is not None and self.y_train is not None:
            return self.x_train, self.y_train
        return self.x_total, self.y_total

    def _loss_cfg(self) -> LossConfig:
        return LossConfig.from_dict(dict(self.cfg.loss))

    def _stacked_logits(self) -> jp.ndarray:
        """Concatenate all layers' logits into one ``[N_gates, lut]`` array."""
        # Logits stored per-layer have shape (group_n, group_size, lut).
        # The stacked / "flat" form is what apply_knockout_to_circuit
        # expects when ``flat=True``. We must include input gates as a
        # zero block so flat indices match flat masks.
        ld = self.logits[0].shape[-1]
        parts = [jp.zeros((int(self.cfg.circuit.input_bits), ld), dtype=self.logits[0].dtype)]
        for lg in self.logits:
            gn, gs, ls = lg.shape
            parts.append(lg.reshape(gn * gs, ls))
        return jp.concatenate(parts, axis=0)

    def _unstack_logits(self, flat: jp.ndarray) -> list[jp.ndarray]:
        """Inverse of ``_stacked_logits`` for the per-layer list form."""
        out = []
        idx = int(self.cfg.circuit.input_bits)
        for lg in self.logits:
            gn, gs, ls = lg.shape
            n = gn * gs
            out.append(flat[idx : idx + n].reshape(gn, gs, ls))
            idx += n
        return out

    def _sync_current_circuit(self) -> None:
        """Copy ``self.pool``'s current circuit into the working ``wires/logits0``."""
        assert self.pool is not None
        idx = self.current_circuit_idx
        self.wires = jax.tree.map(lambda x: x[idx], self.pool.wires)
        self.logits = jax.tree.map(lambda x: x[idx], self.pool.logits)
        self.logits0 = list(self.logits)
        self.gate_mask_flat = (
            self.pool.gate_masks[idx] if self.pool.gate_masks is not None else None
        )

    def _reset_optimizers(self) -> None:
        """Rebuild whichever step driver the active method requires."""
        if self.method == Method.BACKPROP:
            self.bp_optimizer = optax.adamw(self.bp_learning_rate, 0.8, 0.8, weight_decay=1e-1)
            self.bp_opt_state = self.bp_optimizer.init(self.logits)
            self.tmt_generator = None
        else:  # TMT
            self.bp_optimizer = None
            self.bp_opt_state = None
            self._restart_tmt_generator()

    def _restart_tmt_generator(self) -> None:
        """Reset the TMT step generator so the next ``step`` reads fresh logits.

        Damage settings (``p_fault``, ``permanent_damage``) are baked
        into the generator at construction time, exactly as
        ``run_model_scan_with_loss`` does in training: damage is applied
        *inside* the generator's loop, before each model step. This is
        the fix for the "models flicker on/off and never converge" bug
        where damage used to be applied externally and the generator
        was restarted every frame.

        Any pre-existing damage on the current circuit (e.g. from
        clicking *Shotgun*) is forwarded via ``initial_gate_knockout_mask``
        so the generator's graph starts in the right state.
        """
        if self.method != Method.TMT or self.model is None:
            self.tmt_generator = None
            self._generator_p_fault = None
            self._generator_permanent = None
            return

        p_fault = (
            float(self.damage_params.get("p_fault_eval"))
            if self.damage_enabled and self.damage_params.get("p_fault_eval")
            else None
        )
        permanent = float(self.permanent_damage)
        faulty_value = float(self.damage_params.get("faulty_logit_value", -10.0))

        x_data, y_data = self._train_xy()
        self.tmt_generator = evaluate_model_stepwise_generator(
            model=self.model,
            wires=self.wires,
            logits=self.logits,
            x_data=x_data,
            y_data=y_data,
            input_n=int(self.cfg.circuit.input_bits),
            arity=int(self.cfg.circuit.arity),
            circuit_hidden_dim=int(self.cfg.model.circuit_hidden_dim),
            max_steps=None,
            loss_cfg=self._loss_cfg(),
            bidirectional_edges=bool(self.cfg.graph.bidirectional_edges),
            layer_sizes=self.layer_sizes,
            p_fault=p_fault,
            faulty_value=faulty_value,
            permanent_damage=permanent,
            initial_gate_knockout_mask=self.gate_mask_flat,
        )
        # Cache so ``_sync_generator_damage_settings`` can detect drift.
        self._generator_p_fault = p_fault
        self._generator_permanent = permanent
        # Consume the initial (step 0) result so the generator is ready
        # to yield post-update results from step()'s point of view.
        self.last_step_result = next(self.tmt_generator)

    def _reset_logs(self) -> None:
        self.step_count = 0
        self.loss_log = np.zeros(self.log_length, np.float32)
        self.hard_loss_log = np.zeros(self.log_length, np.float32)
        self.accuracy_log = np.zeros(self.log_length, np.float32)
        self.hard_accuracy_log = np.zeros(self.log_length, np.float32)

    def _record_metrics(self) -> None:
        if self.last_step_result is None:
            return
        i = self.step_count % self.log_length
        # Clamp to the same range the GUI expects for log-scale plotting.
        self.loss_log[i] = float(np.clip(self.last_step_result.loss, 1e-6, 10.0))
        self.hard_loss_log[i] = float(np.clip(self.last_step_result.hard_loss, 1e-6, 10.0))
        self.accuracy_log[i] = float(np.clip(self.last_step_result.accuracy, 0.0, 1.0))
        self.hard_accuracy_log[i] = float(np.clip(self.last_step_result.hard_accuracy, 0.0, 1.0))

    # ------------- Method-specific step bodies -------------

    def _step_bp(self) -> None:
        x_data, y_data = self._train_xy()
        loss_cfg = self._loss_cfg()

        with self._region("bp.forward"):
            loss, aux = get_loss_from_wires_logits(
                self.logits, self.wires, x_data, y_data, loss_cfg
            )

        with self._region("bp.grad"):
            # Gradient step.
            def loss_fn(lg):
                loss_val, _ = get_loss_from_wires_logits(lg, self.wires, x_data, y_data, loss_cfg)
                return loss_val

            grads = jax.grad(loss_fn)(self.logits)

        with self._region("bp.update"):
            updates, new_opt_state = self.bp_optimizer.update(grads, self.bp_opt_state, self.logits)
            new_logits = optax.apply_updates(self.logits, updates)
            # Preserve damaged gates — once a gate is knocked out we must not
            # let the optimiser revive it by walking the LUT back up.
            if self.gate_mask_flat is not None:
                new_logits = self._unstack_logits(
                    jp.where(
                        self.gate_mask_flat[:, None] == 0.0,
                        self._stacked_logits(),
                        self._stack(new_logits),
                    )
                )
            self.logits = new_logits
            self.bp_opt_state = new_opt_state

        with self._region("bp.host_sync"):
            # ``float(...)`` blocks on the device, so this is the natural
            # place to attribute the dispatch latency.
            self.last_step_result = StepResult(
                step=self.step_count + 1,
                loss=float(loss),
                hard_loss=float(aux["hard_loss"]),
                accuracy=float(aux["accuracy"]),
                hard_accuracy=float(aux["hard_accuracy"]),
                predictions=aux["predictions"],
                hard_predictions=aux["hard_predictions"],
                residuals=aux["residuals"],
                hard_residuals=aux["hard_residuals"],
                logits=self.logits,
                graph=None,
            )

    def _stack(self, logits_list: list[jp.ndarray]) -> jp.ndarray:
        """Stack a list of per-layer logits into the same flat form as ``_stacked_logits``."""
        ld = logits_list[0].shape[-1]
        parts = [jp.zeros((int(self.cfg.circuit.input_bits), ld), dtype=logits_list[0].dtype)]
        for lg in logits_list:
            gn, gs, ls = lg.shape
            parts.append(lg.reshape(gn * gs, ls))
        return jp.concatenate(parts, axis=0)

    def _step_tmt(self, n_message_steps: int) -> None:
        if self.tmt_generator is None or self.model is None:
            log.warning("step_tmt called with no generator/model — falling back to BP")
            self._step_bp()
            return
        try:
            for _ in range(max(1, int(n_message_steps))):
                self.last_step_result = next(self.tmt_generator)
        except StopIteration:
            self._restart_tmt_generator()
            return
        self.logits = self.last_step_result.logits
        # Mirror the generator's internal damage mask back into the
        # session so the GUI's red-overlay reflects what's actually
        # happening inside the model loop (probabilistic faults applied
        # by ``evaluate_model_stepwise_generator``).
        graph = self.last_step_result.graph
        if graph is not None and graph.nodes is not None:
            mask = graph.nodes.get("gate_knockout_mask")
            if mask is not None:
                self.gate_mask_flat = mask

    def _apply_step_probabilistic_damage(self) -> None:
        """Apply one step's worth of probabilistic gate failure (BP path only).

        For the TMT path, damage is folded directly into
        ``evaluate_model_stepwise_generator`` (``p_fault`` /
        ``permanent_damage`` arguments) so the model gets a chance to
        repair faults *within* the same scan iteration, exactly as in
        training. This external helper is therefore only invoked from
        the BP step path, which has no internal scan loop.
        """
        from boolean_nca_cc.training.pool.structural_perturbation import (
            apply_probabilistic_gate_failure,
        )

        p_fault = self.damage_params.get("p_fault_eval")
        if not p_fault:
            return
        if self.gate_mask_flat is None:
            self.gate_mask_flat = jp.ones(sum(g for g, _ in self.layer_sizes), dtype=jp.float32)

        eligible_mask = create_eligible_gate_mask(self.layer_sizes)
        self._prob_damage_key, step_key = jax.random.split(self._prob_damage_key)
        flat_logits = self._stacked_logits()
        new_logits, new_mask = apply_probabilistic_gate_failure(
            step_key,
            flat_logits,
            self.gate_mask_flat,
            eligible_mask,
            float(p_fault),
            float(self.damage_params.get("faulty_logit_value", -10.0)),
        )
        # Per-gate coin flip for permanence (matches train.py / eval).
        self._perm_damage_key, perm_key = jax.random.split(self._perm_damage_key)
        perm_mask = (
            jax.random.uniform(perm_key, shape=self.gate_mask_flat.shape) < self.permanent_damage
        )
        self.gate_mask_flat = jp.where(perm_mask, new_mask, self.gate_mask_flat)
        self.logits = self._unstack_logits(new_logits)

    # ------------- Visualization helpers -------------

    def _refresh_visualization(self) -> None:
        """Recompute the activations + predictions used for live drawing."""
        x_for_viz = self.x_total if self.x_total is not None else self.x_train
        if x_for_viz is None:
            return
        try:
            with self._region("viz.run_soft"):
                acts = run_circuit(self.logits, self.wires, x_for_viz, hard=False)
            with self._region("viz.run_hard"):
                acts_hard = run_circuit(self.logits, self.wires, x_for_viz, hard=True)
            self.activations = acts
            self.current_pred = acts[-1]
            self.current_pred_hard = acts_hard[-1]
            if self.y_total is not None:
                self.err_mask = self.current_pred_hard != self.y_total
            else:
                self.err_mask = None
        except Exception as e:
            log.warning(f"Failed to refresh activations: {e}")

    # ------------- Convenience getters for the GUI -------------

    @property
    def task_name(self) -> str:
        from boolean_nca_cc.utils.task_compat import get_fixed_task_name
        name = get_fixed_task_name(self.cfg)
        if name is None:
            raise ValueError(
                "Demo state has no fixed task (sampler-mode cfg not supported here)."
            )
        return str(name)

    @property
    def case_n(self) -> int:
        return 1 << int(self.cfg.circuit.input_bits)

    @property
    def input_n(self) -> int:
        return int(self.cfg.circuit.input_bits)

    @property
    def output_n(self) -> int:
        return int(self.cfg.circuit.output_bits)

    @property
    def arity(self) -> int:
        return int(self.cfg.circuit.arity)

    @property
    def hidden_dim(self) -> int:
        return int(self.cfg.model.circuit_hidden_dim)

    @property
    def wiring_mode(self) -> str:
        return str(self.cfg.training.wiring_mode)

    @property
    def n_message_steps(self) -> int:
        return int(self.cfg.training.n_message_steps)

    def __repr__(self) -> str:
        return (
            f"DemoSession(method={self.method.value}, "
            f"task={self.task_name}, wiring={self.wiring_mode}, "
            f"width_factor={self.width_factor}, "
            f"pool_size={self.pool_size}, step={self.step_count})"
        )


# ---------------------------------------------------------------------------
# Convenience helpers for the GUI layer
# ---------------------------------------------------------------------------


def damage_status_string(damage_params: dict) -> str:
    """Pretty-print the damage configuration for status panels."""
    if not damage_params:
        return "no damage"
    pf = damage_params.get("p_fault_eval")
    ko = damage_params.get("knockouts_per_event") or 0
    target = float(damage_params.get("target_fraction", 0.0))
    if not pf and ko <= 0:
        return f"target {target:.0%} (disabled)"
    pf_txt = f"p_fault={pf:.1e}" if pf else "p_fault=0"
    return f"target {target:.0%} | {pf_txt} | shotgun={ko} per volley"


def list_available_tasks() -> list[str]:
    """Tasks the demo can switch between via cfg overrides."""
    return list(TASKS.keys())
