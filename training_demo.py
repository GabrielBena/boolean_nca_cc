#!/usr/bin/env python3
"""
Live demo for Self-Organising Digital Circuits.

This is the GUI layer on top of ``boolean_nca_cc.demo.DemoSession``,
which carries all the actual logic (config loading, pool init,
optimisation steps, damage, etc.). Everything visible here is purely
about interacting with that session: drawing the circuit, running plots,
exposing the four paper regimes (I-IV) as one-click presets, and a
small set of toggles that map directly onto the training config schema.

Default boot-up: Backprop on the default Hydra config (configs/config.yaml),
no model loaded. Hit *Load Model from W&B* to pull in a trained TMT
policy and explore the four regimes from the manuscript.
"""

from __future__ import annotations

import logging

import IPython
import jax.numpy as jp
import numpy as np
from imgui_bundle import imgui, immapp, immvision, implot

from boolean_nca_cc.demo import (
    REGIME_DESCRIPTIONS,
    REGIME_LABELS,
    SCALE_FREE_WIDTH_FACTORS,
    DemoSession,
    Method,
    Regime,
    WandbCoordinates,
    WiringMode,
    damage_status_string,
    list_available_tasks,
)
from boolean_nca_cc.perf import FrameProfiler

logging.basicConfig(level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s")


# ---------------------------------------------------------------------------
# Lightweight visualisation helpers
# ---------------------------------------------------------------------------


def zoom(a: np.ndarray, k: int = 2) -> np.ndarray:
    """Repeat-pixel zoom for image visualisation."""
    return np.repeat(np.repeat(a, k, 1), k, 0)


# Visualisation budget: cap the texture-side column count so that
# (a) the GPU isn't filtering 32 k pixels down to 700, which produced
# the "mostly black" mid-gray result for binary inputs, and
# (b) we feed immvision a contiguously-sized array we can actually
# read back if we ever need to. 512 keeps the texture cheap while
# preserving enough column resolution to read the case axis.
_MAX_PANEL_COLS = 512


def is_point_in_box(p0, p1, p) -> bool:
    (x0, y0), (x1, y1), (x, y) = p0, p1, p
    return (x0 <= x <= x1) and (y0 <= y <= y1)


def calc_lut_input_use(logits: jp.ndarray) -> jp.ndarray:
    """Boolean mask: which inputs of each LUT actually influence its output."""
    luts = jp.sign(logits) * 0.5 + 0.5
    arity = luts.shape[-1].bit_length() - 1
    luts = luts.reshape(luts.shape[:-1] + (2,) * arity)
    axes_to_flatten = -1 - np.arange(arity - 1)
    input_use = []
    for i in range(1, arity + 1):
        m = luts.take(0, -i) != luts.take(1, -i)
        m = m.any(axes_to_flatten)
        input_use.append(m)
    return jp.stack(input_use)


def propagate_gate_use(input_n, wires, logits, output_use):
    output_use = output_use.reshape(logits.shape[:2])
    gate_input_use = calc_lut_input_use(logits) * output_use
    wire_use_mask = gate_input_use.any(-1)
    used_wires = wires[wire_use_mask]
    prev_gate_use = np.zeros(input_n, np.bool_)
    prev_gate_use[used_wires] = True
    return prev_gate_use, wire_use_mask


def calc_gate_use_masks(input_n, wires, logits):
    """Backwards-traverse the circuit to figure out which gates affect output."""
    layer_sizes = [input_n] + [int(np.prod(log.shape[:2])) for log in logits]
    gate_use_mask = np.ones(layer_sizes[-1], np.bool_)
    gate_masks = [gate_use_mask]
    wire_masks = []
    for i in range(len(logits))[::-1]:
        gate_use_mask, wire_use_mask = propagate_gate_use(
            layer_sizes[i], wires[i], logits[i], gate_use_mask
        )
        wire_masks.append(wire_use_mask)
        gate_masks.append(gate_use_mask)
    return gate_masks[::-1], wire_masks[::-1]


def to_rgb_image(arr: np.ndarray) -> np.ndarray:
    """Convert a 2D scalar (H, W) or 3D RGB (H, W, 3) float array into an RGB uint8 image.

    The previous implementation pre-zoomed the array 8x as a hack to
    make the per-pixel rect renderer give visible cells. Now that
    panels go through ``immvision.image_display`` (which lets the GPU
    scale the texture to any panel size with proper filtering), the 8x
    zoom is wasted memory: e.g. a ``(12, 4096)`` input was inflated to
    ``(96, 32768)`` and then squashed back down to a ~700 px panel,
    where the linear minification merged binary data into a featureless
    mid-grey strip.

    We now:
        * keep the array at its native resolution,
        * stride-sample very-wide arrays down to ``_MAX_PANEL_COLS``
          columns so the GPU isn't filtering across dozens of source
          pixels per output pixel (which is what was producing the
          "mostly black" look on binary inputs),
        * uniformly broadcast 2D data to RGB and pass through 3D RGB.
    """
    arr = np.asarray(arr)
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=2)
    elif arr.ndim == 3:
        if arr.shape[2] == 1:
            arr = np.repeat(arr, 3, axis=2)
        elif arr.shape[2] != 3:
            raise ValueError(f"Expected 2D or RGB-shaped array, got shape {arr.shape}")
    else:
        raise ValueError(f"Expected 2D or 3D array, got ndim={arr.ndim}")

    # Stride-sample columns when the array is dramatically wider than
    # what we can show. Plain striding is fine here: input bits are
    # ordered, so showing every Nth case still preserves the local
    # structure of the bit columns.
    w = arr.shape[1]
    if w > _MAX_PANEL_COLS:
        stride = max(1, w // _MAX_PANEL_COLS)
        arr = arr[:, ::stride]

    return np.ascontiguousarray(np.uint8(np.clip(arr, 0.0, 1.0) * 255))


# ---------------------------------------------------------------------------
# DemoApp: GUI wrapper around DemoSession
# ---------------------------------------------------------------------------


class DemoApp:
    """The GUI controller. Owns a ``DemoSession`` and a small bag of GUI state."""

    PLOT_TYPES = ("Loss", "Accuracy")
    LOSS_DISPLAY_MODES = ("Both", "Soft only", "Hard only")
    METHODS = (Method.BACKPROP.value, Method.TMT.value)
    WIRING_MODES = (WiringMode.FIXED.value, WiringMode.RANDOM.value)
    REGIMES = tuple(r.value for r in Regime)

    def __init__(self) -> None:
        # Build the session with the default Hydra config — Backprop, no
        # model loaded, fixed wiring, reverse task. The rest is driven
        # by the user.
        self.session = DemoSession.from_default_config(pool_size=8)
        self.is_optimizing = True
        self.tmt_message_steps = 1

        # Evaluation regime selector (defaults to Regime I).
        self.regime_idx = 0

        # Gallery selection state. Initialise to whichever (task, recipe)
        # has a usable run-id in the YAML so the dropdowns boot to a
        # working entry rather than a "n/a" one.
        first_entry = self.session.gallery.first_available()
        if first_entry is not None:
            self.gallery_task = first_entry.task
            self.gallery_recipe = first_entry.recipe
        else:
            tasks = self.session.gallery.tasks() or ["reverse"]
            self.gallery_task = tasks[0]
            recipes = self.session.gallery.recipes_for_task(self.gallery_task)
            self.gallery_recipe = recipes[0] if recipes else "fixed_damage"

        # Manual W&B coordinates (used by the "Custom W&B run ID" expander).
        self.wandb = WandbCoordinates()
        self._wandb_run_id_buffer = ""

        # Plotting / display state.
        self.plot_type_idx = 1  # Default to Accuracy (more visually informative)
        self.loss_display_mode_idx = 0

        # Active visualisation case (for circuit preview).
        self.active_case_i = 123 % self.session.case_n

        # Circuit gate-use mask (for visual highlighting only — does NOT
        # touch the underlying logits).
        self._gate_use_mask: list[np.ndarray] = []
        self._wire_use_mask: list[np.ndarray] = []
        self._reset_gate_visual_mask()

        # Cached images for input/ground-truth/output panels.
        self._input_img: np.ndarray | None = None
        self._gt_img: np.ndarray | None = None
        self._output_img: np.ndarray | None = None
        # Texture-upload dirty flags. ``image_display`` only re-uploads
        # to the GPU when ``refresh_image=True``; we set this when the
        # underlying numpy buffer actually changed.
        self._image_dirty: dict[str, bool] = {
            "inputs": True,
            "outputs": True,
            "ground_truth": True,
        }
        self._refresh_image_cache(force=True)

        # Per-frame profiler. Toggle visible in the Performance section.
        self.prof = FrameProfiler(window=120)
        self.show_perf = True
        # Hand it to the session so step() can subdivide BP forward / grad / viz.
        self.session.prof = self.prof

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    @property
    def s(self) -> DemoSession:
        """Short alias used inside render loops."""
        return self.session

    # ------------------------------------------------------------------
    # Visual mask plumbing
    # ------------------------------------------------------------------

    def _reset_gate_visual_mask(self) -> None:
        s = self.session
        self._gate_use_mask = [np.ones(int(gate_n)) for gate_n, _ in s.layer_sizes]
        self._wire_use_mask = [np.ones_like(np.asarray(w), np.bool_) for w in s.wires]

    def _mask_unused_gates(self) -> None:
        s = self.session
        gate_masks, self._wire_use_mask = calc_gate_use_masks(s.input_n, s.wires, s.logits)
        for i, gm in enumerate(gate_masks):
            self._gate_use_mask[i] = np.array(self._gate_use_mask[i] * gm)

    # ------------------------------------------------------------------
    # Image cache (input / ground-truth / output)
    # ------------------------------------------------------------------

    def _refresh_image_cache(self, force: bool = False) -> None:
        s = self.session
        if s.x_total is None or s.y_total is None:
            return
        if force or self._input_img is None:
            self._input_img = to_rgb_image(np.asarray(s.x_total).T)
            self._gt_img = to_rgb_image(np.asarray(s.y_total).T)
            self._image_dirty["inputs"] = True
            self._image_dirty["ground_truth"] = True

        # Output image with red overlay on errors.
        if s.current_pred is not None:
            pred = np.asarray(s.current_pred).T
            oimg = np.dstack([pred] * 3).astype(np.float32)
            if s.current_pred_hard is not None and s.y_total is not None:
                err = (np.asarray(s.current_pred_hard) != np.asarray(s.y_total)).T
                m = err[..., None] * 0.5
                oimg = oimg * (1.0 - m) + m * np.float32([1, 0, 0])
            self._output_img = to_rgb_image(oimg)
            self._image_dirty["outputs"] = True

    # ------------------------------------------------------------------
    # Main step (called every frame while playing)
    # ------------------------------------------------------------------

    def _step(self) -> None:
        if not self.is_optimizing:
            return
        try:
            with self.prof.region("compute.step"):
                self.s.step(n_message_steps=self.tmt_message_steps)
        except Exception as e:
            # Keep the GUI alive even if a step blows up.
            logging.error(f"Optimisation step failed: {e}", exc_info=True)
            self.is_optimizing = False
        with self.prof.region("compute.image_cache"):
            self._refresh_image_cache()

    # ------------------------------------------------------------------
    # GUI: top-level dispatcher
    # ------------------------------------------------------------------

    def gui(self) -> None:
        try:
            with self.prof.frame():
                self._step()

                imgui.begin_child("main", (-340, 0))
                with self.prof.region("render.main_panel"):
                    self._render_main_panel()
                imgui.end_child()
                imgui.same_line()
                imgui.begin_child("controls")
                with self.prof.region("render.control_panel"):
                    self._render_control_panel()
                imgui.end_child()
        except Exception as e:
            logging.exception("Top-level GUI error")
            imgui.text(f"GUI error: {e}")

    # ------------------------------------------------------------------
    # GUI: main panel (plots, circuit, I/O images)
    # ------------------------------------------------------------------

    def _render_main_panel(self) -> None:
        with self.prof.region("render.plot"):
            self._render_progress_plot()

        imgui.separator_text("Inputs")
        with self.prof.region("render.image_panel.inputs"):
            if self._input_img is not None:
                self._render_image_panel("inputs", self._input_img)

        imgui.separator_text("Circuit")
        circuit_height = imgui.get_content_region_avail().y - 360
        with self.prof.region("render.circuit_diagram"):
            self._render_circuit_diagram(height=max(int(circuit_height), 280))

        imgui.separator_text("Current output")
        with self.prof.region("render.image_panel.outputs"):
            if self._output_img is not None:
                self._render_image_panel("outputs", self._output_img)

        imgui.separator_text("Expected output")
        with self.prof.region("render.image_panel.gt"):
            if self._gt_img is not None:
                self._render_image_panel("ground_truth", self._gt_img)

    def _render_progress_plot(self) -> None:
        s = self.session
        plot_type = self.PLOT_TYPES[self.plot_type_idx]
        plot_title = f"Circuit Optimisation — {plot_type}"
        if implot.begin_plot(plot_title, (-1, 220)):
            implot.setup_legend(implot.Location_.north_east.value)
            display_mode = self.LOSS_DISPLAY_MODES[self.loss_display_mode_idx]
            if plot_type == "Loss":
                implot.setup_axis_scale(implot.ImAxis_.y1.value, implot.Scale_.log10.value)
                implot.setup_axes(
                    "Step",
                    "Loss",
                    implot.AxisFlags_.auto_fit.value,
                    implot.AxisFlags_.auto_fit.value,
                )
                implot.setup_axis_limits(implot.ImAxis_.y1.value, 1e-6, 10.0)
                if display_mode in ("Both", "Soft only"):
                    implot.plot_line("soft loss", s.loss_log)
                if display_mode in ("Both", "Hard only"):
                    implot.plot_line("hard loss", s.hard_loss_log)
            else:  # Accuracy
                implot.setup_axes(
                    "Step",
                    "Accuracy",
                    implot.AxisFlags_.auto_fit.value,
                    implot.AxisFlags_.none.value,
                )
                implot.setup_axis_limits(implot.ImAxis_.y1.value, 0.0, 1.05)
                if display_mode in ("Both", "Soft only"):
                    implot.plot_line("soft acc", s.accuracy_log)
                if display_mode in ("Both", "Hard only"):
                    implot.plot_line("hard acc", s.hard_accuracy_log)
            implot.drag_line_x(1, s.step_count % len(s.loss_log), (0.8, 0, 0, 0.5))
            implot.end_plot()

    # ------------------------------------------------------------------
    # GUI: control panel (the right sidebar)
    # ------------------------------------------------------------------

    def _render_control_panel(self) -> None:
        if imgui.button("Python REPL"):
            IPython.embed()

        self._render_play_section()
        self._render_model_section()
        self._render_regime_section()
        self._render_ood_section()
        self._render_circuit_section()
        self._render_damage_section()
        self._render_visualization_section()
        self._render_status_section()
        self._render_performance_section()

    # ---------- Play / pause / reset --------------------------------------

    def _render_play_section(self) -> None:
        imgui.separator_text("Optimisation")
        if self.is_optimizing:
            if imgui.button("⏸ Pause", (120, 0)):
                self.is_optimizing = False
        else:
            if imgui.button("▶ Play", (120, 0)):
                self.is_optimizing = True
        imgui.same_line()
        imgui.text("running" if self.is_optimizing else "paused")
        if imgui.button("Reset circuit"):
            self.session.reset_circuit()
            self._refresh_image_cache(force=True)

    # ---------- Model: gallery + manual run-ID ----------------------------

    def _render_model_section(self) -> None:
        imgui.separator_text("Pre-trained model")
        s = self.session

        # Loaded-model card (status panel).
        if s.loaded_entry is not None:
            imgui.text_colored(
                imgui.ImVec4(0.0, 1.0, 0.0, 1.0),
                f"✓ {s.loaded_entry.task} · {s.loaded_entry.label}",
            )
            imgui.text_colored(
                imgui.ImVec4(0.7, 0.7, 0.7, 1.0),
                f"   run {s.loaded_entry.entity}/{s.loaded_entry.project}/"
                f"{s.loaded_entry.run_id}",
            )
            if s.loaded_entry.description:
                imgui.text_wrapped(s.loaded_entry.description)
        elif s.model_run_id is not None:
            imgui.text_colored(
                imgui.ImVec4(0.0, 1.0, 0.0, 1.0), f"✓ custom run {s.model_run_id}"
            )
        else:
            imgui.text_colored(
                imgui.ImVec4(0.7, 0.7, 0.7, 1.0),
                "No model loaded — running Backprop on default config",
            )

        # Gallery dropdowns: task x recipe.
        gallery = s.gallery
        tasks = gallery.tasks()
        if not tasks:
            imgui.text_colored(
                imgui.ImVec4(1.0, 0.7, 0.0, 1.0),
                "Gallery empty — fill in configs/demo_models.yaml.",
            )
        else:
            self._render_gallery_dropdowns(gallery, tasks)

        # Manual run-ID expander (folded by default).
        if imgui.tree_node("Custom W&B run ID"):
            _, self.wandb.entity = imgui.input_text("Entity", self.wandb.entity)
            _, self.wandb.project = imgui.input_text("Project", self.wandb.project)
            _, self._wandb_run_id_buffer = imgui.input_text(
                "Run ID", self._wandb_run_id_buffer
            )
            if imgui.button("Load custom run"):
                self.wandb.run_id = self._wandb_run_id_buffer or None
                ok = self.session.load_model_from_wandb(self.wandb)
                if ok:
                    self._on_model_loaded()
                else:
                    logging.warning("Could not load model — leaving Backprop active")
            imgui.tree_pop()

        # Method (BP vs TMT) toggle. Only meaningful when a model is loaded.
        method_idx = list(self.METHODS).index(s.method.value)
        changed, method_idx = imgui.combo("Method", method_idx, list(self.METHODS))
        if changed:
            new_method = Method(list(self.METHODS)[method_idx])
            if new_method == Method.TMT and s.model is None:
                imgui.text_colored(
                    imgui.ImVec4(1, 0.4, 0.4, 1),
                    "No TMT model loaded — keeping Backprop",
                )
            else:
                s.method = new_method
                s._reset_optimizers()

        if s.method == Method.BACKPROP:
            _, s.bp_learning_rate = imgui.slider_float(
                "Learning rate",
                s.bp_learning_rate,
                1e-4,
                2.0,
                "%.4f",
                imgui.SliderFlags_.logarithmic.value,
            )
        else:
            _, self.tmt_message_steps = imgui.slider_int(
                "Message steps / frame", self.tmt_message_steps, 1, 16
            )

    def _render_gallery_dropdowns(self, gallery, tasks: list[str]) -> None:
        # Sticky task index.
        try:
            task_idx = tasks.index(self.gallery_task)
        except ValueError:
            task_idx = 0
            self.gallery_task = tasks[0]
        changed, task_idx = imgui.combo("Task", task_idx, tasks)
        if changed:
            self.gallery_task = tasks[task_idx]

        # Build recipe options for the chosen task. Show greyed-out
        # entries for declared-but-unavailable recipes.
        declared_recipes = gallery.recipes_for_task(self.gallery_task)
        if not declared_recipes:
            imgui.text_colored(
                imgui.ImVec4(0.7, 0.7, 0.7, 1.0), "No recipes declared for this task."
            )
            return

        recipe_labels = [
            f"{gallery.recipe_labels.get(r, r)}"
            + ("" if gallery.lookup(self.gallery_task, r) else "  (n/a)")
            for r in declared_recipes
        ]
        try:
            rec_idx = declared_recipes.index(self.gallery_recipe)
        except ValueError:
            rec_idx = 0
            self.gallery_recipe = declared_recipes[0]
        changed, rec_idx = imgui.combo("Recipe", rec_idx, recipe_labels)
        if changed:
            self.gallery_recipe = declared_recipes[rec_idx]

        # Status hint for the currently picked card.
        entry = gallery.lookup(self.gallery_task, self.gallery_recipe)
        if entry is None:
            imgui.text_colored(
                imgui.ImVec4(1.0, 0.7, 0.0, 1.0),
                "✗ Not yet trained — pick another card.",
            )
            imgui.begin_disabled()
            imgui.button("Load")
            imgui.end_disabled()
        else:
            imgui.text_colored(
                imgui.ImVec4(0.0, 1.0, 0.0, 1.0),
                f"✓ {entry.run_id}",
            )
            if imgui.button("Load this model"):
                ok = self.session.load_model_from_gallery(
                    self.gallery_task, self.gallery_recipe
                )
                if ok:
                    self._on_model_loaded()

    def _on_model_loaded(self) -> None:
        """Refresh GUI caches after a successful model load."""
        self._reset_gate_visual_mask()
        self._refresh_image_cache(force=True)
        self.active_case_i = 123 % self.session.case_n

    # ---------- Evaluation regime ----------------------------------------

    def _render_regime_section(self) -> None:
        imgui.separator_text("Evaluation regime")
        regimes = list(Regime)
        for i, r in enumerate(regimes):
            label = REGIME_LABELS[r]
            selected = self.regime_idx == i
            if imgui.radio_button(label, selected):
                self.regime_idx = i
                self.session.apply_regime(r)
                self._on_model_loaded()
        # Active description.
        active = regimes[self.regime_idx]
        imgui.text_colored(
            imgui.ImVec4(0.6, 0.6, 0.6, 1.0), REGIME_DESCRIPTIONS[active]
        )
        # Width slider is *always* shown when Regime IV is active.
        if active == Regime.WIDTH_SWEEP:
            wf_current = float(
                self.session.width_factor
                if self.session.width_factor is not None
                else self.session.cfg.circuit.width_factor
            )
            wf_min, wf_max = SCALE_FREE_WIDTH_FACTORS[0], SCALE_FREE_WIDTH_FACTORS[-1]
            changed, wf_new = imgui.slider_float(
                "Width factor", wf_current, wf_min, wf_max, "%.2f"
            )
            if changed and abs(wf_new - wf_current) > 1e-3:
                self.session.width_factor = float(wf_new)
                self.session.bootstrap()
                self._on_model_loaded()

    # ---------- OOD overrides (advanced) ---------------------------------

    def _render_ood_section(self) -> None:
        s = self.session
        diffs = s.ood_diffs()
        # Prefix the header with a status badge so the user can tell at
        # a glance whether they're inside or outside training distribution.
        if s.training_cfg is None:
            header = "OOD overrides (advanced) — load a model to enable"
        elif diffs:
            header = f"OOD overrides ({len(diffs)} active) ▼"
        else:
            header = "OOD overrides (advanced) — matches training"

        if not imgui.collapsing_header(header):
            return

        if s.training_cfg is None:
            imgui.text_colored(
                imgui.ImVec4(0.7, 0.7, 0.7, 1.0),
                "No model loaded yet — these knobs become active once "
                "you pull a run from the gallery.",
            )
            return

        if imgui.button("Reset all to training config"):
            s.reset_to_training_config()
            self._on_model_loaded()

        amber = imgui.ImVec4(1.0, 0.65, 0.0, 1.0)
        green = imgui.ImVec4(0.6, 0.6, 0.6, 1.0)

        def label_with_status(name: str, knob: str) -> None:
            """Coloured label + (training: X) annotation."""
            if knob in diffs:
                trained, current = diffs[knob]
                imgui.text_colored(amber, f"{name}: trained {trained!r} → now {current!r}")
            else:
                imgui.text_colored(green, f"{name}: matches training")

        # Task override.
        tasks = list_available_tasks()
        try:
            task_idx = tasks.index(s.task_name)
        except ValueError:
            task_idx = 0
        changed, task_idx = imgui.combo("Task##ood", task_idx, tasks)
        if changed:
            s.apply_cfg_changes(**{"circuit.task": tasks[task_idx]})
            self._on_model_loaded()
        label_with_status("Task", "task")

        # Wiring mode override.
        wiring_idx = list(self.WIRING_MODES).index(s.wiring_mode)
        changed, wiring_idx = imgui.combo(
            "Wiring##ood", wiring_idx, list(self.WIRING_MODES)
        )
        if changed:
            s.apply_cfg_changes(
                **{"training.wiring_mode": list(self.WIRING_MODES)[wiring_idx]}
            )
            self._on_model_loaded()
        label_with_status("Wiring", "wiring_mode")

        # Width factor override.
        wf_current = float(
            s.width_factor if s.width_factor is not None else s.cfg.circuit.width_factor
        )
        wf_min, wf_max = SCALE_FREE_WIDTH_FACTORS[0], SCALE_FREE_WIDTH_FACTORS[-1]
        changed, wf_new = imgui.slider_float(
            "Width factor##ood", wf_current, wf_min, wf_max, "%.2f"
        )
        if changed and abs(wf_new - wf_current) > 1e-3:
            s.width_factor = float(wf_new)
            s.bootstrap()
            self._on_model_loaded()
        label_with_status("Width factor", "width_factor")

        # Num layers override.
        nl_current = int(s.cfg.circuit.num_layers)
        changed, nl_new = imgui.slider_int("Num hidden layers##ood", nl_current, 1, 6)
        if changed and nl_new != nl_current:
            s.apply_cfg_changes(**{"circuit.num_layers": int(nl_new)})
            self._on_model_loaded()
        label_with_status("Num layers", "num_layers")

        # Damage override.
        _, dmg_now = imgui.checkbox("Damage at eval##ood", bool(s.damage_enabled))
        if dmg_now != bool(s.damage_enabled):
            s.damage_enabled = dmg_now
        label_with_status("Damage", "damage_enabled")

    # ---------- Circuit / pool knobs -------------------------------------

    def _render_circuit_section(self) -> None:
        imgui.separator_text("Pool")
        s = self.session
        # Pool / circuit index slider.
        if s.pool_size > 1:
            changed, new_idx = imgui.slider_int(
                f"Circuit (1..{s.pool_size})",
                s.current_circuit_idx,
                0,
                s.pool_size - 1,
            )
            if changed:
                s.select_circuit_idx(int(new_idx))
                self._on_model_loaded()

        # Regime III demonstration: replace the wires of the *current*
        # circuit with a fresh random topology, keeping the optimised
        # logits and the metric history. With a wiring-agnostic TMT
        # this should produce a visible accuracy drop followed by a
        # recovery — without ever touching the trained policy.
        if imgui.button("Shuffle wires (current circuit)"):
            s.shuffle_wires()
            # Don't call _on_model_loaded — that would force-reset the
            # gate visual mask and image cache, which is fine, but we
            # also leave the metric log untouched (so the drop+recovery
            # is visible across the shuffle event).
            self._reset_gate_visual_mask()
            self._refresh_image_cache(force=True)
        imgui.text_colored(
            imgui.ImVec4(0.6, 0.6, 0.6, 1.0),
            "Regime III: OOD wire shuffle, keeps logits + log",
        )

        # Full pool reshuffle: rebuild the entire pool from a fresh seed,
        # which *does* reset the logs (intentional — every circuit is
        # different now). Useful for scrubbing through topologies.
        if imgui.button("Reshuffle pool"):
            s.regenerate_pool()
            self._on_model_loaded()

    # ---------- Damage controls ------------------------------------------

    def _render_damage_section(self) -> None:
        imgui.separator_text("Damage")
        s = self.session
        _, s.damage_enabled = imgui.checkbox(
            "Stochastic faults each step", s.damage_enabled
        )
        _, s.permanent_damage = imgui.slider_float(
            "Permanence", s.permanent_damage, 0.0, 1.0, "%.2f"
        )
        if imgui.button("🔫 Shotgun"):
            s.apply_shotgun_damage()
            self._refresh_image_cache(force=True)
        imgui.same_line()
        if imgui.button("Pre-grow with BP"):
            s.pre_grow_with_bp(n_steps=300)
            self._refresh_image_cache(force=True)
        imgui.text_colored(
            imgui.ImVec4(0.7, 0.7, 0.7, 1.0), damage_status_string(s.damage_params)
        )

    # ---------- Visualisation options ------------------------------------

    def _render_visualization_section(self) -> None:
        imgui.separator_text("Visualisation")
        _, self.plot_type_idx = imgui.combo(
            "Plot", self.plot_type_idx, list(self.PLOT_TYPES)
        )
        _, self.loss_display_mode_idx = imgui.combo(
            "Display", self.loss_display_mode_idx, list(self.LOSS_DISPLAY_MODES)
        )
        if imgui.button("Reset gate highlight"):
            self._reset_gate_visual_mask()
        imgui.same_line()
        if imgui.button("Highlight used gates"):
            self._mask_unused_gates()

    # ---------- Performance / profiler ----------------------------------

    def _render_performance_section(self) -> None:
        imgui.separator_text("Performance")
        _, self.show_perf = imgui.checkbox("Show breakdown", self.show_perf)
        imgui.same_line()
        if imgui.small_button("Reset"):
            self.prof.reset()
        if not self.show_perf:
            return

        frame_mean, frame_p95, _frame_last = self.prof.frame_stats_ms()
        fps = self.prof.fps()
        if frame_mean > 0:
            imgui.text(f"Frame: {frame_mean:5.1f} ms (p95 {frame_p95:5.1f}) → {fps:5.1f} FPS")
        else:
            imgui.text("Frame: (warming up)")

        rows = self.prof.summary()
        if not rows:
            return

        # Headers.
        imgui.separator()
        imgui.text(f"{'Region':<26}{'mean':>8}{'p95':>8}{'last':>8}{'%':>6}")
        denom = max(frame_mean, 1e-6)
        red = imgui.ImVec4(1.0, 0.45, 0.45, 1.0)
        amber = imgui.ImVec4(1.0, 0.75, 0.3, 1.0)
        green = imgui.ImVec4(0.55, 0.85, 0.55, 1.0)
        stale = imgui.ImVec4(0.45, 0.45, 0.45, 1.0)
        for name, mean_ms, p95_ms, last_ms, is_stale in rows:
            pct = 100.0 * mean_ms / denom
            if is_stale:
                # Greyed out: e.g. ``bp.*`` rows after the user
                # switches to TMT, or ``compute.step`` while paused.
                color = stale
                tag = " (stale)"
            else:
                if pct >= 35.0:
                    color = red
                elif pct >= 15.0:
                    color = amber
                else:
                    color = green
                tag = ""
            imgui.text_colored(
                color,
                f"{name:<26}{mean_ms:7.2f} {p95_ms:7.2f} {last_ms:7.2f} {pct:5.1f}{tag}",
            )

    # ---------- Status panel ---------------------------------------------

    def _render_status_section(self) -> None:
        imgui.separator_text("Status")
        s = self.session
        imgui.text(f"Method: {s.method.value}")
        imgui.text(f"Task: {s.task_name}  ({s.input_n} → {s.output_n} bits)")
        imgui.text(f"Wiring: {s.wiring_mode}")
        imgui.text(f"Layers: {s.layer_sizes}")
        imgui.text(f"Step: {s.step_count}")
        imgui.text(f"Pool: {s.current_circuit_idx + 1}/{s.pool_size}")
        if s.last_step_result is not None:
            imgui.text(f"Soft acc: {float(s.last_step_result.accuracy):.3f}")
            imgui.text(f"Hard acc: {float(s.last_step_result.hard_accuracy):.3f}")
        if s.gate_mask_flat is not None:
            n_active = int(jp.asarray(s.gate_mask_flat).sum())
            n_total = int(s.gate_mask_flat.shape[0])
            imgui.text(f"Damaged gates: {n_total - n_active} / {n_total}")

    # ------------------------------------------------------------------
    # Circuit diagram (gates + wires)
    # ------------------------------------------------------------------

    def _render_circuit_diagram(self, pad: int = 4, d: int = 24, height: int = 600) -> None:
        s = self.session
        io = imgui.get_io()
        width = imgui.get_content_region_avail().x - pad * 2
        imgui.invisible_button("circuit", (width, height))
        base_x, base_y = imgui.get_item_rect_min()
        base_x += pad

        dl = imgui.get_window_draw_list()
        layer_sizes = s.layer_sizes
        layer_h = (height - d) / max(1, (len(layer_sizes) - 1))

        # Make sure cached masks match the current layer count.
        if len(self._gate_use_mask) != len(layer_sizes):
            self._reset_gate_visual_mask()

        # Pre-fetch per-case activations once (each ``np.asarray`` on a
        # JAX array forces a device→host sync, so doing it per-layer
        # inside the gate loop was paying ``len(layer_sizes)`` syncs
        # *every frame*).
        case = self.active_case_i
        case_activations: list[np.ndarray] = []
        for li in range(len(layer_sizes)):
            if li < len(s.activations) and case < s.activations[li].shape[0]:
                case_activations.append(np.asarray(s.activations[li][case]))
            else:
                case_activations.append(np.zeros(layer_sizes[li][0]))
        # Pre-fetch wires too — they only change on bootstrap/mutate, so
        # a per-frame ``np.asarray`` is wasteful but harmless to cache
        # at this granularity.
        wires_np = [np.asarray(w) for w in s.wires]

        prev_gate_x = None
        prev_y = 0
        prev_act = None
        hover_gate = None

        for li, (gate_n, group_size) in enumerate(layer_sizes):
            group_n = gate_n // max(group_size, 1)
            span_x = width / max(group_n, 1)
            group_w = min(d * group_size, span_x - 6)
            gate_w = group_w / max(group_size, 1)
            group_x = base_x + (np.arange(group_n)[:, None] + 0.5) * span_x
            gate_ofs = (np.arange(group_size) - group_size / 2 + 0.5) * gate_w
            gate_x = (group_x + gate_ofs).ravel()
            y = base_y + li * layer_h + d / 2

            act = case_activations[li]

            for i, x in enumerate(gate_x):
                a = int(act[i] * 0xA0) if i < len(act) else 0
                col = 0xFF202020 + (a << 8)
                p0, p1 = (x - gate_w / 2, y - d / 2), (x + gate_w / 2, y + d / 2)
                dl.add_rect_filled(p0, p1, col, 4)

                if is_point_in_box(p0, p1, io.mouse_pos):
                    dl.add_rect(p0, p1, 0xA00000FF, 4, thickness=2.0)
                    if li > 0:
                        group_idx = i // max(group_size, 1)
                        gate_idx = i % max(group_size, 1)
                        layer_logits = s.logits[li - 1]
                        if (
                            group_idx < layer_logits.shape[0]
                            and gate_idx < layer_logits.shape[1]
                        ):
                            hover_gate = (x, y, layer_logits[group_idx, gate_idx])
                    if io.mouse_clicked[0]:
                        if li == 0:
                            # Input layer: flip the active case bit.
                            self.active_case_i = self.active_case_i ^ (1 << i)
                        elif 0 < li < len(layer_sizes) - 1:
                            # Hidden gate: knock it out and add to the
                            # damage set. Mirrors what shotgun does for a
                            # random gate (logits → faulty value, mask → 0).
                            flat_idx = sum(g for g, _ in layer_sizes[:li]) + i
                            s.damage_gate(flat_idx)
                            self._refresh_image_cache(force=True)

                # Damaged-gate overlay (red).
                if (
                    s.gate_mask_flat is not None
                    and 0 < li < len(layer_sizes) - 1
                ):
                    flat_idx = sum(g for g, _ in layer_sizes[:li]) + i
                    if (
                        flat_idx < s.gate_mask_flat.shape[0]
                        and float(s.gate_mask_flat[flat_idx]) == 0.0
                    ):
                        dl.add_rect_filled(p0, p1, 0xC00000FF, 4)

            # Group boundaries.
            for x in group_x[:, 0]:
                dl.add_rect(
                    (x - group_w / 2, y - d / 2),
                    (x + group_w / 2, y + d / 2),
                    0x80FFFFFF,
                    4,
                )

            # Wires from previous layer.
            if (
                li > 0
                and prev_gate_x is not None
                and (li - 1) < len(wires_np)
                and (li - 1) < len(self._wire_use_mask)
            ):
                wires = wires_np[li - 1].T
                masks = self._wire_use_mask[li - 1].T
                src_x = prev_gate_x[wires]
                dst_x = (
                    group_x
                    + (np.arange(s.arity) + 0.5) / s.arity * group_w
                    - group_w / 2
                )
                my = (prev_y + y) / 2
                for x0, x1, si, m in zip(
                    src_x.ravel(),
                    dst_x.ravel(),
                    wires.ravel(),
                    masks.ravel(),
                    strict=False,
                ):
                    if not m:
                        continue
                    if prev_act is not None and si < len(prev_act):
                        a_i = int(prev_act[si] * 0x60)
                    else:
                        a_i = 0
                    col = 0xFF404040 + (a_i << 8)
                    dl.add_bezier_cubic(
                        (x0, prev_y + d / 2),
                        (x0, my),
                        (x1, my),
                        (x1, y - d / 2),
                        col,
                        1.0,
                    )

            if hover_gate is not None:
                self._draw_gate_lut(*hover_gate)

            prev_gate_x = gate_x
            prev_act = act
            prev_y = y

    def _draw_gate_lut(self, x: float, y: float, logit: jp.ndarray) -> None:
        x0, y0 = x - 20, y - 20 - 36
        dl = imgui.get_window_draw_list()
        from jax.nn import sigmoid

        lut = np.asarray(sigmoid(logit)).reshape(-1, 4)
        col = np.uint32(lut * 255)
        col = (col << 16) | (col << 8) | col | 0xFF000000
        for (i, j), c in np.ndenumerate(col):
            xp, yp = x0 + j * 10, y0 + i * 10
            dl.add_rect_filled((xp, yp), (xp + 10, yp + 10), c)

    # ------------------------------------------------------------------
    # Image panel (input / output / ground-truth)
    # ------------------------------------------------------------------

    def _render_image_panel(self, name: str, img: np.ndarray) -> None:
        """Draw an RGB uint8 image as a single GL texture (immvision).

        This replaces the previous per-pixel ``add_rect_filled`` grid
        which, at default settings, issued ~4096 draw calls per panel,
        across 3 panels (~12k draw calls per frame). That was the
        dominant rendering cost.
        ``image_display`` uploads to the GPU once per change of the
        underlying numpy buffer (``refresh_image=True``) and otherwise just
        re-binds the existing texture (~one draw call).
        """
        try:
            view_w = imgui.get_content_region_avail().x
            img_h, img_w = img.shape[:2]
            disp_w = max(1, int(view_w))
            # Honour the image's natural aspect ratio in the limit, but
            # floor the displayed height generously so very-wide arrays
            # (e.g. 12 x 4096 input grids) still render as a *readable*
            # strip rather than a 30-pixel sliver. Capping above the
            # natural height is intentional: it deliberately stretches
            # the bit axis so each input bit is several rows tall.
            natural_h = round(disp_w * img_h / max(img_w, 1))
            disp_h = int(np.clip(natural_h, 80, 220))

            refresh = self._image_dirty.get(name, True)
            immvision.image_display(
                f"##img_{name}",
                img,
                image_display_size=(disp_w, disp_h),
                refresh_image=refresh,
                show_options_button=False,
            )
            if refresh:
                self._image_dirty[name] = False

            # Overlay the active-case marker on top of the image we just
            # drew. ``imgui.get_item_rect_*`` refers to the immvision item.
            p0 = imgui.get_item_rect_min()
            p1 = imgui.get_item_rect_max()
            dl = imgui.get_window_draw_list()
            case_n = max(self.session.case_n, 1)
            x = p0.x + (p1.x - p0.x) * (self.active_case_i + 0.5) / case_n
            dl.add_line((x, p0.y), (x, p1.y), 0xFF00FF00, 2.0)
            dl.add_rect(p0, p1, 0xFFFFFFFF, 4.0)

            # Click-to-select active case.
            if imgui.is_item_hovered() and imgui.is_mouse_clicked(0):
                mx = imgui.get_io().mouse_pos.x - p0.x
                mx_ratio = mx / max(p1.x - p0.x, 1)
                self.active_case_i = max(
                    0, min(int(mx_ratio * case_n), case_n - 1)
                )
        except Exception as e:
            imgui.text(f"Error rendering {name}: {e}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    print("Self-Organising Digital Circuits — live demo")
    print(
        "Default: Backprop on the default Hydra config. "
        "Use the 'Load TMT from W&B' panel to bring in a trained model."
    )
    # Required by immvision since Oct-2024: declare the color order of
    # the numpy buffers we'll feed to ``image_display``. We build all
    # panel images as RGB uint8 (see ``to_rgb_image``).
    immvision.use_rgb_color_order()
    app = DemoApp()
    immapp.run(
        app.gui,
        window_title="Self-Organising Digital Circuits — live demo",
        window_size=(1400, 900),
        # Idle FPS bumped from 10 → 30: at 10 FPS, dropdown / hover reactions
        # incur up to 100 ms of latency which feels sluggish even though
        # the renderer is fine. 30 FPS is plenty for static UI and barely
        # costs anything since rendering is now texture-based.
        fps_idle=30,
        with_implot=True,
    )


if __name__ == "__main__":
    main()
