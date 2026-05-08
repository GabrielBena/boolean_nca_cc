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
const BASE = import.meta.env.BASE_URL; // "/" in dev, "/assets/sodc-demo/" in prod
const GALLERY_URL = `${BASE}weights/gallery.json`;

// Default knobs. Tweakable from the URL hash later if needed.
const DEFAULT_SHOTGUN_GATES = 8;
const TICK_THROTTLE_MS = 16; // ≤ 60 FPS — actual ticks gated by compute time

// ---------------------------------------------------------------------------
// Gallery types (mirrors gallery.json written by export_gallery.py)
// ---------------------------------------------------------------------------

export interface GalleryEntry {
  id: string;
  task: string;
  recipe: string;
  label: string;
  description: string;
  runId: string;
  weightsPath: string;  // relative to /weights/
  bootstrapPath: string;
}

export interface Gallery {
  models: GalleryEntry[];
}

async function fetchGallery(): Promise<Gallery> {
  const resp = await fetch(GALLERY_URL);
  if (!resp.ok) throw new Error(`fetch gallery (${GALLERY_URL}): ${resp.status}`);
  return (await resp.json()) as Gallery;
}

function weightsUrl(entry: GalleryEntry): string {
  return `${BASE}weights/${entry.weightsPath}`;
}

function bootstrapUrl(entry: GalleryEntry): string {
  return `${BASE}weights/${entry.bootstrapPath}`;
}

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
  modelPicker: HTMLDivElement;
  imagePanel: HTMLDivElement;
  inputCanvas: HTMLCanvasElement;
  inputWrap: HTMLDivElement;
  circuitCanvas: HTMLCanvasElement;
  currentCanvas: HTMLCanvasElement;
  currentWrap: HTMLDivElement;
  expectedCanvas: HTMLCanvasElement;
  expectedWrap: HTMLDivElement;
  chartCanvas: HTMLCanvasElement;
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
  // Status is hidden by default — only shown on load errors.
  const status = el<HTMLDivElement>("div", "sodc-status");
  status.style.display = "none";

  const imagePanel = el<HTMLDivElement>("div", "sodc-image-panel");
  const inputRow = makeBitCanvas("input  (x)");
  const circuitRow = makeCircuitCanvas("circuit (active case)");
  const currentRow = makeBitCanvas("current TMT output");
  const expectedRow = makeBitCanvas("expected (y)");
  const chartCanvas = el<HTMLCanvasElement>("canvas", "sodc-chart-canvas");
  imagePanel.append(
    inputRow.row,
    circuitRow.row,
    currentRow.row,
    expectedRow.row,
    chartCanvas,
  );

  const modelPicker = el<HTMLDivElement>("div", "sodc-model-picker");

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
    .sodc-model-picker { display: flex; flex-wrap: wrap; gap: 0.4em;
                         align-items: center; margin-bottom: 0.5em; }
    .sodc-model-label { font-size: 0.8em; color: #6e7681; margin-right: 0.25em; }
    .sodc-model-btn { font: inherit; font-size: 0.82em; cursor: pointer;
                      padding: 0.3em 0.75em; border-radius: 20px;
                      border: 1px solid #d0d7de; background: #f6f8fa;
                      color: #57606a; transition: all 0.1s; white-space: nowrap; }
    .sodc-model-btn:hover { background: #eaeef2; color: #1f2328; }
    .sodc-model-btn.active { background: #1f883d; color: #fff;
                             border-color: #1a7f37; font-weight: 600; }
    .sodc-model-btn:disabled { opacity: 0.5; cursor: not-allowed; }
    .sodc-chart-canvas { width: 100%; height: 130px; display: block;
                         border: 1px solid #30363d; border-radius: 4px;
                         margin-top: 0.6em; background: #0d1117; }
  `;
  wrap.append(style, status, modelPicker, imagePanel, controls, hint, readout);
  root.append(wrap);

  return {
    status,
    modelPicker,
    imagePanel,
    inputCanvas: inputRow.canvas,
    inputWrap: inputRow.wrap,
    circuitCanvas: circuitRow.canvas,
    currentCanvas: currentRow.canvas,
    currentWrap: currentRow.wrap,
    expectedCanvas: expectedRow.canvas,
    expectedWrap: expectedRow.wrap,
    chartCanvas,
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
  /** Pre-computed wires from the training run. Present for fixed-wires models;
   *  absent for random-wires models (Controller generates its own). */
  wires?: Int32Array[];
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

function decodeInt32(entry: RawB64): Int32Array {
  const bytes = base64ToBytes(entry.data_b64);
  const buf = new ArrayBuffer(bytes.byteLength);
  new Uint8Array(buf).set(bytes);
  return new Int32Array(buf);
}

async function fetchBootstrap(url: string): Promise<BootstrapData> {
  const resp = await fetch(url);
  if (!resp.ok) throw new Error(`fetch bootstrap (${url}): ${resp.status}`);
  const raw = (await resp.json()) as {
    header: {
      case_n: number;
      input_bits: number;
      output_bits: number;
      task_style?: string;
      text?: string | null;
    };
    layer_sizes: [number, number][];
    wires?: RawB64[];
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
    ...(raw.wires ? { wires: raw.wires.map(decodeInt32) } : {}),
  };
}

// ---------------------------------------------------------------------------
// Accuracy history + chart
// ---------------------------------------------------------------------------

type EventKind = "damage" | "shotgun" | "shuffle" | "reset";

interface AccuracyHistory {
  hard: number[];
  soft: number[];
  events: Array<{ tick: number; kind: EventKind }>;
}

function makeHistory(): AccuracyHistory {
  return { hard: [], soft: [], events: [] };
}

const EVENT_COLORS: Record<EventKind, string> = {
  damage: "#f85149",
  shotgun: "#e3b341",
  shuffle: "#388bfd",
  reset: "#848d97",
};

function drawAccuracyChart(canvas: HTMLCanvasElement, history: AccuracyHistory): void {
  const dpr = window.devicePixelRatio || 1;
  const cssW = canvas.clientWidth || 600;
  const cssH = canvas.clientHeight || 130;
  if (canvas.width !== Math.round(cssW * dpr) || canvas.height !== Math.round(cssH * dpr)) {
    canvas.width = Math.round(cssW * dpr);
    canvas.height = Math.round(cssH * dpr);
  }
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  ctx.save();
  ctx.scale(dpr, dpr);

  // Layout: top 24px = legend header, then chart area, then 16px x-axis footer
  const LEGEND_H = 24;
  const PAD_L = 38;
  const PAD_R = 8;
  const PAD_B = 16;
  const cW = cssW - PAD_L - PAD_R;
  const chartTop = LEGEND_H;
  const cH = cssH - LEGEND_H - PAD_B;

  ctx.fillStyle = "#0d1117";
  ctx.fillRect(0, 0, cssW, cssH);

  ctx.font = `10px ui-monospace, monospace`;

  // ---- Legend row (above the plot) ---------------------------------------
  const legY = LEGEND_H / 2 + 3.5; // vertically centred in the header strip

  // Thin separator between legend and plot
  ctx.strokeStyle = "#21262d";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(0, LEGEND_H);
  ctx.lineTo(cssW, LEGEND_H);
  ctx.stroke();

  // "hard acc" swatch
  let legX = PAD_L + 4;
  ctx.strokeStyle = "#3fb950";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(legX, legY);
  ctx.lineTo(legX + 14, legY);
  ctx.stroke();
  ctx.fillStyle = "#d0d7de";
  ctx.textAlign = "left";
  ctx.fillText("hard acc", legX + 18, legY);
  legX += 82;

  // "soft acc" swatch
  ctx.strokeStyle = "#1f6feb";
  ctx.lineWidth = 1;
  ctx.globalAlpha = 0.6;
  ctx.beginPath();
  ctx.moveTo(legX, legY);
  ctx.lineTo(legX + 14, legY);
  ctx.stroke();
  ctx.globalAlpha = 1;
  ctx.fillStyle = "#8b949e";
  ctx.fillText("soft acc", legX + 18, legY);
  legX += 82;

  // Event type swatches (dashed vertical lines)
  const evTypes: [EventKind, string][] = [
    ["shotgun", "shotgun"],
    ["damage", "click"],
    ["shuffle", "shuffle"],
  ];
  for (const [kind, label] of evTypes) {
    ctx.strokeStyle = EVENT_COLORS[kind];
    ctx.lineWidth = 1;
    ctx.setLineDash([3, 3]);
    ctx.beginPath();
    ctx.moveTo(legX + 5, LEGEND_H * 0.15);
    ctx.lineTo(legX + 5, LEGEND_H * 0.85);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = "#8b949e";
    ctx.fillText(label, legX + 11, legY);
    legX += 54;
  }

  // ---- Chart area --------------------------------------------------------
  ctx.strokeStyle = "#21262d";
  ctx.lineWidth = 1;
  ctx.fillStyle = "#6e7681";
  ctx.textAlign = "right";
  for (let v = 0; v <= 1.001; v += 0.25) {
    const y = chartTop + cH * (1 - v);
    ctx.beginPath();
    ctx.moveTo(PAD_L, y);
    ctx.lineTo(PAD_L + cW, y);
    ctx.stroke();
    ctx.fillText(v.toFixed(2), PAD_L - 4, y + 3.5);
  }

  const n = history.hard.length;
  if (n === 0) {
    ctx.restore();
    return;
  }

  const MAX_TICKS = 120;
  const startTick = Math.max(0, n - MAX_TICKS);
  const xOf = (i: number) =>
    PAD_L + ((i - startTick) / Math.max(1, MAX_TICKS - 1)) * cW;
  const yOf = (v: number) => chartTop + cH * (1 - Math.max(0, Math.min(1, v)));

  // Event markers
  for (const ev of history.events) {
    if (ev.tick < startTick) continue;
    const x = xOf(ev.tick);
    ctx.strokeStyle = EVENT_COLORS[ev.kind];
    ctx.lineWidth = 1;
    ctx.setLineDash([3, 3]);
    ctx.beginPath();
    ctx.moveTo(x, chartTop);
    ctx.lineTo(x, chartTop + cH);
    ctx.stroke();
  }
  ctx.setLineDash([]);

  // Soft accuracy (dim blue)
  if (history.soft.length === history.hard.length) {
    ctx.strokeStyle = "#1f6feb";
    ctx.lineWidth = 1;
    ctx.globalAlpha = 0.55;
    ctx.beginPath();
    for (let i = startTick; i < n; i++) {
      const x = xOf(i);
      const y = yOf(history.soft[i]!);
      i === startTick ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    }
    ctx.stroke();
    ctx.globalAlpha = 1;
  }

  // Hard accuracy (bright green)
  ctx.strokeStyle = "#3fb950";
  ctx.lineWidth = 2;
  ctx.beginPath();
  for (let i = startTick; i < n; i++) {
    const x = xOf(i);
    const y = yOf(history.hard[i]!);
    i === startTick ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  }
  ctx.stroke();

  // Dot at last point
  ctx.fillStyle = "#3fb950";
  ctx.beginPath();
  ctx.arc(xOf(n - 1), yOf(history.hard[n - 1]!), 3, 0, Math.PI * 2);
  ctx.fill();

  // X-axis tick labels
  ctx.fillStyle = "#6e7681";
  ctx.textAlign = "left";
  ctx.fillText(`tick ${startTick}`, PAD_L, cssH - 3);
  ctx.textAlign = "right";
  ctx.fillText(`tick ${n - 1}`, PAD_L + cW, cssH - 3);

  ctx.restore();
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
  /** Running accuracy history for the live chart. */
  history: AccuracyHistory;
  /** Available models from gallery.json. */
  gallery: Gallery;
  /** ID of the currently loaded gallery entry. */
  activeModelId: string;
}

function updateReadout(app: AppState, snap: ControllerSnapshot): void {
  const total = app.ctrl.totalHiddenGates();
  app.ui.readout.textContent =
    `step ${snap.step}` +
    `  ·  hard_acc ${snap.hardAccuracy.toFixed(4)}` +
    `  ·  soft_acc ${snap.softAccuracy.toFixed(4)}` +
    `  ·  damaged ${snap.damaged}/${total}` +
    `  ·  case ${app.activeCase}/${app.bootstrap.caseN}`;
}

function refreshFrame(app: AppState, snap: ControllerSnapshot): void {
  drawBitImage(app.ui.currentCanvas, snap.predHard, snap.caseN, snap.outputBits);
  redrawCircuit(app);
  updateReadout(app, snap);
  app.history.hard.push(snap.hardAccuracy);
  app.history.soft.push(snap.softAccuracy);
  drawAccuracyChart(app.ui.chartCanvas, app.history);
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
  if (app.circuitTarget.width <= 1) return; // not yet sized by rAF
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

// ---------------------------------------------------------------------------
// Gallery / model switching
// ---------------------------------------------------------------------------

/** Populate the model picker with buttons from the gallery. */
function populateModelPicker(app: AppState): void {
  const picker = app.ui.modelPicker;
  picker.innerHTML = "";

  for (const entry of app.gallery.models) {
    const btn = el<HTMLButtonElement>("button", "sodc-model-btn", entry.label);
    btn.title = entry.description;
    if (entry.id === app.activeModelId) btn.classList.add("active");
    btn.addEventListener("click", () => {
      if (entry.id === app.activeModelId) return;
      void switchModel(app, entry);
    });
    picker.appendChild(btn);
  }
}

/** Load a new model from the gallery and reinitialise the Controller in place. */
async function switchModel(app: AppState, entry: GalleryEntry): Promise<void> {
  setPlaying(app, false);

  // Disable all controls while loading.
  const allBtns = app.ui.modelPicker.querySelectorAll<HTMLButtonElement>("button");
  allBtns.forEach((b) => (b.disabled = true));
  for (const b of [app.ui.playBtn, app.ui.resetBtn, app.ui.shuffleBtn, app.ui.shotgunBtn])
    b.disabled = true;
  app.ui.status.textContent = `Loading ${entry.label} …`;
  app.ui.status.className = "sodc-status";

  try {
    const [weights, bs] = await Promise.all([
      loadWeights(weightsUrl(entry)),
      fetchBootstrap(bootstrapUrl(entry)),
    ]);

    // Replace the controller with a fresh one.
    app.ctrl = new Controller(weights, {
      caseN: bs.caseN,
      xData: bs.xData,
      yData: bs.yData,
      inputBits: bs.inputBits,
      outputBits: bs.outputBits,
      layerSizes: bs.layerSizes,
      rngSeed: 44,
      ...(bs.wires ? { precomputedWires: bs.wires } : {}),
    });
    app.bootstrap = bs;
    app.activeModelId = entry.id;
    app.history = makeHistory();
    app.recentShotgun = null;
    app.hoveredGateIdx = -1;
    app.activeCase = 1234 % bs.caseN;

    // Redraw static strips.
    drawBitImage(app.ui.inputCanvas, bs.xData, bs.caseN, bs.inputBits);
    drawBitImage(app.ui.expectedCanvas, bs.yData, bs.caseN, bs.outputBits);

    // Reset circuit canvas size (new model may have different architecture).
    const w = Math.floor(app.ui.circuitCanvas.getBoundingClientRect().width) || 1000;
    app.circuitTarget = setupCanvas(app.ui.circuitCanvas, w, 280);

    app.ui.status.textContent = summariseWeights(weights);
    app.ui.status.classList.add("pass");

    refreshFrame(app, app.ctrl.reset());
    setActiveCase(app, app.activeCase);

    // Update picker active state.
    populateModelPicker(app);
  } catch (err) {
    console.error(err);
    app.ui.status.textContent = `Error loading ${entry.label}: ${(err as Error).message}`;
    app.ui.status.classList.add("fail");
    app.ui.status.style.display = "block";
  } finally {
    for (const b of [app.ui.playBtn, app.ui.resetBtn, app.ui.shuffleBtn, app.ui.shotgunBtn])
      b.disabled = false;
    app.ui.modelPicker.querySelectorAll<HTMLButtonElement>("button")
      .forEach((b) => (b.disabled = false));
  }
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
    app.history = makeHistory();
    refreshFrame(app, app.ctrl.reset());
  });
  app.ui.shuffleBtn.addEventListener("click", () => {
    app.recentShotgun = null;
    app.history.events.push({ tick: app.history.hard.length, kind: "shuffle" });
    refreshFrame(app, app.ctrl.shuffle());
  });
  app.ui.shotgunBtn.addEventListener("click", () => {
    const { snapshot, chosen } = app.ctrl.shotgun(DEFAULT_SHOTGUN_GATES);
    app.recentShotgun = chosen;
    app.history.events.push({ tick: app.history.hard.length, kind: "shotgun" });
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
    app.history.events.push({ tick: app.history.hard.length, kind: "damage" });
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

/** Build a single-entry fallback gallery from the legacy per-model files. */
function fallbackGallery(): Gallery {
  return {
    models: [
      {
        id: "reverse_random_damage",
        task: "reverse",
        recipe: "random_damage",
        label: "Random · damage-trained",
        description: "Regime III, random wiring + damage.",
        runId: "1u5ssulx",
        weightsPath: "reverse_random_damage.json",
        bootstrapPath: "reverse_demo.json",
      },
    ],
  };
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
    ui.status.textContent = "Loading …";

    // Fetch gallery + first model in parallel. gallery.json may not exist yet
    // (pre-export), in which case fall back to the legacy single-model files.
    const galleryPromise = fetchGallery().catch(() => fallbackGallery());
    const gallery = await galleryPromise;

    // Default to the last entry (random_damage — most capable) or the only one.
    const defaultEntry = gallery.models[gallery.models.length - 1] ?? gallery.models[0];
    if (!defaultEntry) throw new Error("gallery.json contains no models");

    const [weights, bs] = await Promise.all([
      loadWeights(weightsUrl(defaultEntry)),
      fetchBootstrap(bootstrapUrl(defaultEntry)),
    ]);
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
      ...(bs.wires ? { precomputedWires: bs.wires } : {}),
    });
    // Bootstrap with a 1-px placeholder; the rAF below sets the real size once
    // the DOM has laid out and clientWidth is reliable.
    const app: AppState = {
      ctrl,
      ui,
      playing: false,
      rafHandle: null,
      lastTickTs: 0,
      bootstrap: bs,
      circuitTarget: setupCanvas(ui.circuitCanvas, 1, 280),
      activeCase: 1234 % bs.caseN,
      recentShotgun: null,
      hoveredGateIdx: -1,
      history: makeHistory(),
      gallery,
      activeModelId: defaultEntry.id,
    };

    // Populate model picker (before rendering so it's visible immediately).
    populateModelPicker(app);

    drawBitImage(ui.inputCanvas, bs.xData, bs.caseN, bs.inputBits);
    drawBitImage(ui.expectedCanvas, bs.yData, bs.caseN, bs.outputBits);

    // Render initial frame (post step-0 residual recompute).
    refreshFrame(app, ctrl.reset());
    // Position the active-case markers + ensure the readout shows the case.
    setActiveCase(app, app.activeCase);

    wireControls(app);
    for (const b of [ui.playBtn, ui.resetBtn, ui.shuffleBtn, ui.shotgunBtn]) b.disabled = false;

    // Defer circuit setup to after first paint so getBoundingClientRect() is accurate.
    requestAnimationFrame(() => {
      const w = Math.floor(ui.circuitCanvas.getBoundingClientRect().width) || 1000;
      app.circuitTarget = setupCanvas(ui.circuitCanvas, w, 280);
      redrawCircuit(app);
    });

    // Re-setup canvases on resize — keeps them crisp + responsive.
    window.addEventListener("resize", () => {
      const w = Math.floor(ui.circuitCanvas.getBoundingClientRect().width) || 1000;
      app.circuitTarget = setupCanvas(ui.circuitCanvas, w, 280);
      redrawCircuit(app);
      drawAccuracyChart(ui.chartCanvas, app.history);
    });
  } catch (err) {
    console.error(err);
    ui.status.textContent = `Error: ${(err as Error).message}`;
    ui.status.classList.add("fail");
    ui.status.style.display = "block";
  }
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", () => void bootstrap());
} else {
  void bootstrap();
}
