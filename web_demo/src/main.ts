/**
 * SODC live demo — entry point.
 *
 * Mounts an interactive widget into the element with id ``sodc-demo``:
 *
 *   - Three bit-image strips (Input / Current TMT output / Expected),
 *     refreshed each tick.
 *   - Four buttons: ▶ Play / ⏸ Pause, ⟲ Reset, ⤬ Shuffle wires,
 *     🔫 Shotgun.
 *   - Live readout: step, hard accuracy, soft accuracy, damaged-gate count.
 *   - rAF animation loop calls ``controller.tick()`` while playing.
 *
 * Math layer is shared with the parity test at ``/verify.html`` (which is
 * the regression check that the TS port matches the JAX policy bit-for-bit
 * on hard predictions).
 */

import { Controller, type ControllerSnapshot } from "./controller";
import type { LayerSize } from "./circuit";
import {
  type RenderTarget,
  activeCaseInput,
  drawCircuit,
  pickGate,
  setupCanvas,
  singleCaseActivations,
} from "./render";
import { countParameters, loadWeights, type TMTWeights } from "./weights";

const MOUNT_ID = "sodc-demo";
const WEIGHTS_URL = "/weights/reverse_random_damage.json";
// Live demo loads the *bootstrap* file (header + layer_sizes + task_data
// only). The full ``reverse_trajectory.json`` is reserved for the parity
// test at /verify.html. Both files are produced by ``record_trajectory.py``
// in one invocation (see ``web_demo/export/record_trajectory.py``).
const BOOTSTRAP_URL = "/weights/reverse_demo.json";

// Default knobs. Tweakable from the URL hash later if needed.
const DEFAULT_SHOTGUN_GATES = 8;
const TICK_THROTTLE_MS = 16; // ≤ 60 FPS — actual ticks gated by compute time

// ---------------------------------------------------------------------------
// DOM helpers
// ---------------------------------------------------------------------------

function el<T extends HTMLElement>(tag: string, cls?: string, text?: string): T {
  const e = document.createElement(tag) as T;
  if (cls) e.className = cls;
  if (text !== undefined) e.textContent = text;
  return e;
}

function clearChildren(node: HTMLElement) {
  while (node.firstChild) node.removeChild(node.firstChild);
}

interface UI {
  status: HTMLElement;
  imagePanel: HTMLDivElement;
  inputCanvas: HTMLCanvasElement;
  inputWrap: HTMLDivElement;
  circuitCanvas: HTMLCanvasElement;
  currentCanvas: HTMLCanvasElement;
  currentWrap: HTMLDivElement;
  expectedCanvas: HTMLCanvasElement;
  expectedWrap: HTMLDivElement;
  tickLabel: HTMLElement;
  controls: HTMLDivElement;
  playBtn: HTMLButtonElement;
  resetBtn: HTMLButtonElement;
  shuffleBtn: HTMLButtonElement;
  shotgunBtn: HTMLButtonElement;
  readout: HTMLDivElement;
}

function buildUI(root: HTMLElement): UI {
  clearChildren(root);
  const wrap = el<HTMLDivElement>("div", "sodc-wrap");
  const status = el<HTMLDivElement>("div", "sodc-status", "Loading weights ...");
  const tickLabel = el<HTMLDivElement>("div", "sodc-tick-label", "tick 0 — hard_acc=…");

  const imagePanel = el<HTMLDivElement>("div", "sodc-image-panel");
  const inputRow = makeBitCanvas("input  (x)");
  const circuitRow = makeCircuitCanvas("circuit (active case)");
  const currentRow = makeBitCanvas("current TMT output");
  const expectedRow = makeBitCanvas("expected (y)");
  imagePanel.append(
    tickLabel,
    inputRow.row,
    circuitRow.row,
    currentRow.row,
    expectedRow.row,
  );

  const controls = el<HTMLDivElement>("div", "sodc-controls");
  const playBtn = el<HTMLButtonElement>("button", "sodc-btn sodc-btn-primary", "▶ Play");
  const resetBtn = el<HTMLButtonElement>("button", "sodc-btn", "⟲ Reset");
  const shuffleBtn = el<HTMLButtonElement>("button", "sodc-btn", "⤬ Shuffle wires");
  const shotgunBtn = el<HTMLButtonElement>("button", "sodc-btn", "🔫 Shotgun");
  controls.append(playBtn, resetBtn, shuffleBtn, shotgunBtn);

  const hint = el<HTMLDivElement>(
    "div",
    "sodc-hint",
    "tip · click any hidden-layer gate in the circuit above to damage it",
  );
  const readout = el<HTMLDivElement>("div", "sodc-readout", "—");

  const style = document.createElement("style");
  style.textContent = `
    .sodc-wrap { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
    .sodc-status { font-size: 1.05em; padding: 0.4em 0; font-weight: 600; }
    .sodc-status.pass { color: #1a7f37; }
    .sodc-status.fail { color: #cf222e; }
    .sodc-image-panel { margin: 0.6em 0; padding: 1em; background: #0d1117;
                        border-radius: 8px; color: #d0d7de; }
    .sodc-tick-label { font-size: 0.9em; opacity: 0.85; margin-bottom: 0.6em;
                       font-variant-numeric: tabular-nums; }
    .sodc-image-row { display: grid; grid-template-columns: 12em 1fr; align-items: center;
                      gap: 1em; margin: 0.4em 0; }
    .sodc-image-label { font-size: 0.85em; opacity: 0.85; text-align: right; }
    .sodc-image-canvas-wrap { position: relative; width: 100%; line-height: 0; }
    .sodc-image-canvas { width: 100%; image-rendering: pixelated;
                         border: 1px solid #30363d; background: #1117;
                         cursor: pointer; display: block; }
    .sodc-active-marker { position: absolute; top: 0; bottom: 0; width: 2px;
                          background: rgba(255, 220, 90, 0.85);
                          box-shadow: 0 0 4px rgba(255, 220, 90, 0.6);
                          transform: translateX(-50%);
                          pointer-events: none;
                          transition: left 80ms ease-out; }
    .sodc-circuit-canvas { width: 100%; height: 280px;
                           border: 1px solid #30363d; background: #0d1117;
                           border-radius: 4px; }
    .sodc-controls { display: flex; flex-wrap: wrap; gap: 0.5em; margin: 0.6em 0; }
    .sodc-btn { font: inherit; cursor: pointer; padding: 0.45em 1em;
                border-radius: 6px; border: 1px solid #d0d7de; background: #f6f8fa;
                color: #1f2328; transition: background 0.1s; }
    .sodc-btn:hover { background: #eaeef2; }
    .sodc-btn:disabled { opacity: 0.5; cursor: not-allowed; }
    .sodc-btn-primary { background: #1f883d; color: #fff; border-color: #1a7f37; }
    .sodc-btn-primary:hover { background: #1a7f37; }
    .sodc-hint { font-size: 0.8em; color: #6e7681; font-style: italic;
                 padding: 0.2em 0.1em 0.6em; }
    .sodc-readout { font-size: 0.9em; padding: 0.5em 0.8em;
                    background: #f6f8fa; border-radius: 6px; color: #57606a;
                    font-variant-numeric: tabular-nums; }
  `;
  wrap.append(style, status, imagePanel, controls, hint, readout);
  root.append(wrap);

  return {
    status,
    imagePanel,
    inputCanvas: inputRow.canvas,
    inputWrap: inputRow.wrap,
    circuitCanvas: circuitRow.canvas,
    currentCanvas: currentRow.canvas,
    currentWrap: currentRow.wrap,
    expectedCanvas: expectedRow.canvas,
    expectedWrap: expectedRow.wrap,
    tickLabel,
    controls,
    playBtn,
    resetBtn,
    shuffleBtn,
    shotgunBtn,
    readout,
  };
}

function makeBitCanvas(labelText: string): {
  row: HTMLDivElement;
  canvas: HTMLCanvasElement;
  wrap: HTMLDivElement;
} {
  const row = el<HTMLDivElement>("div", "sodc-image-row");
  const label = el<HTMLDivElement>("div", "sodc-image-label", labelText);
  // Wrap the canvas in a relatively-positioned container so we can overlay
  // the active-case marker (a thin vertical strip) without redrawing pixels.
  const wrap = el<HTMLDivElement>("div", "sodc-image-canvas-wrap");
  const canvas = el<HTMLCanvasElement>("canvas", "sodc-image-canvas");
  const marker = el<HTMLDivElement>("div", "sodc-active-marker");
  wrap.append(canvas, marker);
  row.append(label, wrap);
  return { row, canvas, wrap };
}

function makeCircuitCanvas(labelText: string): { row: HTMLDivElement; canvas: HTMLCanvasElement } {
  const row = el<HTMLDivElement>("div", "sodc-image-row");
  const label = el<HTMLDivElement>("div", "sodc-image-label", labelText);
  const canvas = el<HTMLCanvasElement>("canvas", "sodc-circuit-canvas");
  row.append(label, canvas);
  return { row, canvas };
}

// ---------------------------------------------------------------------------
// Bit-image rendering
// ---------------------------------------------------------------------------

function drawBitImage(
  canvas: HTMLCanvasElement,
  bits: Float32Array,
  caseN: number,
  nBits: number,
  threshold = 0.5,
): void {
  if (canvas.width !== caseN || canvas.height !== nBits) {
    canvas.width = caseN;
    canvas.height = nBits;
    canvas.style.aspectRatio = `${caseN} / ${nBits}`;
    canvas.style.height = `${Math.max(40, nBits * 4)}px`;
  }
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  const img = ctx.createImageData(caseN, nBits);
  for (let b = 0; b < nBits; b++) {
    for (let c = 0; c < caseN; c++) {
      const v = bits[c * nBits + b] >= threshold ? 255 : 32;
      const px = (b * caseN + c) * 4;
      img.data[px] = v;
      img.data[px + 1] = v;
      img.data[px + 2] = v;
      img.data[px + 3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);
}

// ---------------------------------------------------------------------------
// Demo bootstrap (header + layer_sizes + task_data)
//
// The live demo doesn't need the recorded reference trajectory — only the
// task data (x, y) and the topology shape (layer_sizes). Those live in a
// dedicated bootstrap JSON so the demo's per-tick batch size knob (the
// ``--n-cases`` flag of ``record_trajectory.py``) is decoupled from the
// parity smoke test.
// ---------------------------------------------------------------------------

interface BootstrapData {
  caseN: number;
  inputBits: number;
  outputBits: number;
  layerSizes: LayerSize[];
  xData: Float32Array;
  yData: Float32Array;
  taskStyle: string;
  text: string | null;
}

interface RawB64 {
  shape: number[];
  data_b64: string;
}

function base64ToBytes(b64: string): Uint8Array {
  const bin = atob(b64);
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}

function decodeFloat32(entry: RawB64): Float32Array {
  const bytes = base64ToBytes(entry.data_b64);
  const buf = new ArrayBuffer(bytes.byteLength);
  new Uint8Array(buf).set(bytes);
  return new Float32Array(buf);
}

async function fetchBootstrap(): Promise<BootstrapData> {
  const resp = await fetch(BOOTSTRAP_URL);
  if (!resp.ok) throw new Error(`fetch bootstrap (${BOOTSTRAP_URL}): ${resp.status}`);
  const raw = (await resp.json()) as {
    header: {
      case_n: number;
      input_bits: number;
      output_bits: number;
      task_style?: string;
      text?: string | null;
    };
    layer_sizes: [number, number][];
    task_data: { x: RawB64; y: RawB64 };
  };
  return {
    caseN: raw.header.case_n,
    inputBits: raw.header.input_bits,
    outputBits: raw.header.output_bits,
    layerSizes: raw.layer_sizes.map((p) => [p[0], p[1]]),
    xData: decodeFloat32(raw.task_data.x),
    yData: decodeFloat32(raw.task_data.y),
    taskStyle: raw.header.task_style ?? "sequential",
    text: raw.header.text ?? null,
  };
}

// ---------------------------------------------------------------------------
// Bootstrap
// ---------------------------------------------------------------------------

interface AppState {
  ctrl: Controller;
  ui: UI;
  playing: boolean;
  rafHandle: number | null;
  lastTickTs: number;
  bootstrap: BootstrapData;
  circuitTarget: RenderTarget;
  /** Index of the case fed through the circuit for the activation-coloured render. */
  activeCase: number;
  /** Most recently shotgun-chosen indices (highlighted briefly on the canvas). */
  recentShotgun: Int32Array | null;
  /** Flat index of the gate the mouse is currently over, or -1. */
  hoveredGateIdx: number;
}

function updateReadout(app: AppState, snap: ControllerSnapshot): void {
  const total = app.ctrl.totalHiddenGates();
  app.ui.readout.textContent =
    `step ${snap.step}` +
    `  ·  hard_acc ${snap.hardAccuracy.toFixed(4)}` +
    `  ·  soft_acc ${snap.softAccuracy.toFixed(4)}` +
    `  ·  damaged ${snap.damaged}/${total}` +
    `  ·  case ${app.activeCase}/${app.bootstrap.caseN}`;
  app.ui.tickLabel.textContent =
    `tick ${snap.step} — hard_acc=${snap.hardAccuracy.toFixed(4)}` +
    `  ·  active case ${app.activeCase}`;
}

function refreshFrame(app: AppState, snap: ControllerSnapshot): void {
  drawBitImage(app.ui.currentCanvas, snap.predHard, snap.caseN, snap.outputBits);
  redrawCircuit(app);
  updateReadout(app, snap);
}

/** Move the active-case marker on a wrap to ``activeCase``'s column. */
function updateActiveMarker(wrap: HTMLDivElement, activeCase: number, caseN: number): void {
  const marker = wrap.querySelector(".sodc-active-marker") as HTMLElement | null;
  if (!marker) return;
  const pct = ((activeCase + 0.5) / Math.max(1, caseN)) * 100;
  marker.style.left = `${pct}%`;
}

/** Set the active case (which input pattern flows through the circuit) and
 *  refresh everything that depends on it. */
function setActiveCase(app: AppState, idx: number): void {
  app.activeCase = idx;
  updateActiveMarker(app.ui.inputWrap, idx, app.bootstrap.caseN);
  updateActiveMarker(app.ui.currentWrap, idx, app.bootstrap.caseN);
  updateActiveMarker(app.ui.expectedWrap, idx, app.bootstrap.caseN);
  redrawCircuit(app);
  // Echo the current case in the readout subtitle row.
  app.ui.readout.dataset["activeCase"] = String(idx);
}

/** Compute a case index from a click on a bit-strip canvas. */
function pickCase(canvas: HTMLCanvasElement, ev: MouseEvent, caseN: number): number {
  const rect = canvas.getBoundingClientRect();
  const ratio = (ev.clientX - rect.left) / Math.max(1, rect.width);
  return Math.max(0, Math.min(caseN - 1, Math.floor(ratio * caseN)));
}

function redrawCircuit(app: AppState): void {
  const { ctrl, bootstrap } = app;
  const x1 = activeCaseInput(bootstrap.xData, app.activeCase, bootstrap.inputBits);
  const acts = singleCaseActivations(ctrl.topology, ctrl.state, x1, ctrl.arity);
  // Combine highlights: hovered + recent-shotgun + last-clicked. The renderer
  // only shows which is "current"; we just want the user to see "yes, that's
  // the gate I'm targeting" for hover, and the just-damaged set after each
  // perturbation. Order doesn't matter.
  const highlights: number[] = [];
  if (app.hoveredGateIdx >= 0) highlights.push(app.hoveredGateIdx);
  if (app.recentShotgun) for (let i = 0; i < app.recentShotgun.length; i++) highlights.push(app.recentShotgun[i]);
  if (ctrl.lastDamagedFlatIdx !== null) highlights.push(ctrl.lastDamagedFlatIdx);
  drawCircuit(app.circuitTarget, ctrl.topology, ctrl.state, acts, {
    highlight: highlights.length ? new Int32Array(highlights) : null,
  });
}

/** Hidden-layer eligibility for click-to-damage — same rules as
 *  ``DemoSession.damage_gate`` and ``circuit.ts:damageGate``. */
function isDamageable(ctrl: Controller, flatIdx: number): boolean {
  if (flatIdx < 0) return false;
  const inputN = ctrl.topology.layerSizes[0][0];
  const outputN = ctrl.topology.layerSizes[ctrl.topology.layerSizes.length - 1][0];
  const total = ctrl.topology.nNodes;
  if (flatIdx < inputN || flatIdx >= total - outputN) return false;
  if (ctrl.state.gateMask[flatIdx] === 0) return false;
  return true;
}

/** CSS-pixel coordinates of a mouse event relative to ``canvas``. The
 *  render target uses CSS-px units (HiDPI scaling is in the backing store
 *  only), so this matches what ``pickGate`` expects. */
function canvasMousePos(canvas: HTMLCanvasElement, ev: MouseEvent): { x: number; y: number } {
  const rect = canvas.getBoundingClientRect();
  return { x: ev.clientX - rect.left, y: ev.clientY - rect.top };
}

function setPlaying(app: AppState, on: boolean): void {
  app.playing = on;
  app.ui.playBtn.textContent = on ? "⏸ Pause" : "▶ Play";
  if (on && app.rafHandle === null) scheduleNextTick(app);
}

function scheduleNextTick(app: AppState): void {
  app.rafHandle = requestAnimationFrame((ts) => {
    app.rafHandle = null;
    if (!app.playing) return;
    if (ts - app.lastTickTs >= TICK_THROTTLE_MS) {
      app.lastTickTs = ts;
      const snap = app.ctrl.tick();
      refreshFrame(app, snap);
    }
    scheduleNextTick(app);
  });
}

function wireControls(app: AppState): void {
  app.ui.playBtn.addEventListener("click", () => setPlaying(app, !app.playing));
  app.ui.resetBtn.addEventListener("click", () => {
    setPlaying(app, false);
    app.recentShotgun = null;
    refreshFrame(app, app.ctrl.reset());
  });
  app.ui.shuffleBtn.addEventListener("click", () => {
    app.recentShotgun = null;
    refreshFrame(app, app.ctrl.shuffle());
  });
  app.ui.shotgunBtn.addEventListener("click", () => {
    const { snapshot, chosen } = app.ctrl.shotgun(DEFAULT_SHOTGUN_GATES);
    app.recentShotgun = chosen;
    refreshFrame(app, snapshot);
  });

  // ---- Click-on-strip → set active case ------------------------------
  for (const canvas of [app.ui.inputCanvas, app.ui.currentCanvas, app.ui.expectedCanvas]) {
    canvas.addEventListener("click", (ev) => {
      const idx = pickCase(canvas, ev, app.bootstrap.caseN);
      setActiveCase(app, idx);
    });
  }

  // ---- Click-to-damage on the circuit canvas -------------------------
  const canvas = app.ui.circuitCanvas;
  canvas.addEventListener("mousemove", (ev) => {
    const { x, y } = canvasMousePos(canvas, ev);
    const idx = pickGate(app.circuitTarget, app.ctrl.topology, x, y);
    const eligible = isDamageable(app.ctrl, idx);
    canvas.style.cursor = eligible ? "pointer" : "default";
    const showIdx = eligible ? idx : -1;
    if (showIdx !== app.hoveredGateIdx) {
      app.hoveredGateIdx = showIdx;
      // When paused the rAF loop isn't repainting; do it here. When playing
      // the next animation frame would catch it, but a 1-tick-per-second
      // cadence makes hover feel laggy — so always redraw on hover change.
      redrawCircuit(app);
    }
  });
  canvas.addEventListener("mouseleave", () => {
    if (app.hoveredGateIdx !== -1) {
      app.hoveredGateIdx = -1;
      canvas.style.cursor = "default";
      redrawCircuit(app);
    }
  });
  canvas.addEventListener("click", (ev) => {
    const { x, y } = canvasMousePos(canvas, ev);
    const idx = pickGate(app.circuitTarget, app.ctrl.topology, x, y);
    if (!isDamageable(app.ctrl, idx)) return;
    const snap = app.ctrl.damage(idx);
    if (snap === null) return;
    // After a manual click we want the yellow outline to be the freshly
    // damaged gate, not whatever the previous shotgun set was.
    app.recentShotgun = null;
    refreshFrame(app, snap);
  });
}

function summariseWeights(weights: TMTWeights): string {
  return (
    `${weights.header.modelKind} (gathered) · ` +
    `D=${weights.header.attentionDim}, H=${weights.header.numHeads}, ` +
    `arity=${weights.header.arity}, hidden=${weights.header.circuitHiddenDim} · ` +
    `${countParameters(weights).toLocaleString()} params (${weights.header.tensorDtype})`
  );
}

async function bootstrap(): Promise<void> {
  const root = document.getElementById(MOUNT_ID);
  if (!root) {
    console.error(`[sodc-demo] mount target #${MOUNT_ID} not found`);
    return;
  }
  const ui = buildUI(root);
  // Disable buttons until the controller is up.
  for (const b of [ui.playBtn, ui.resetBtn, ui.shuffleBtn, ui.shotgunBtn]) b.disabled = true;
  try {
    ui.status.textContent = "Loading weights ...";
    const [weights, bs] = await Promise.all([loadWeights(WEIGHTS_URL), fetchBootstrap()]);
    ui.status.textContent = summariseWeights(weights);
    ui.status.classList.add("pass");

    const ctrl = new Controller(weights, {
      caseN: bs.caseN,
      xData: bs.xData,
      yData: bs.yData,
      inputBits: bs.inputBits,
      outputBits: bs.outputBits,
      layerSizes: bs.layerSizes,
      rngSeed: 44,
    });
    // Set up the circuit canvas at the natural size of its container.
    const cssWidth = ui.circuitCanvas.clientWidth || 1000;
    const circuitTarget = setupCanvas(ui.circuitCanvas, cssWidth, 280);
    const app: AppState = {
      ctrl,
      ui,
      playing: false,
      rafHandle: null,
      lastTickTs: 0,
      bootstrap: bs,
      circuitTarget,
      activeCase: 1234 % bs.caseN,
      recentShotgun: null,
      hoveredGateIdx: -1,
    };

    drawBitImage(ui.inputCanvas, bs.xData, bs.caseN, bs.inputBits);
    drawBitImage(ui.expectedCanvas, bs.yData, bs.caseN, bs.outputBits);

    // Render initial frame (post step-0 residual recompute).
    refreshFrame(app, ctrl.reset());
    // Position the active-case markers + ensure the readout shows the case.
    setActiveCase(app, app.activeCase);

    wireControls(app);
    for (const b of [ui.playBtn, ui.resetBtn, ui.shuffleBtn, ui.shotgunBtn]) b.disabled = false;

    // Re-setup canvas on resize — keeps it crisp + responsive.
    window.addEventListener("resize", () => {
      const w = ui.circuitCanvas.clientWidth || 1000;
      app.circuitTarget = setupCanvas(ui.circuitCanvas, w, 280);
      redrawCircuit(app);
    });
  } catch (err) {
    console.error(err);
    ui.status.textContent = `Error: ${(err as Error).message}`;
    ui.status.classList.add("fail");
  }
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", () => void bootstrap());
} else {
  void bootstrap();
}
