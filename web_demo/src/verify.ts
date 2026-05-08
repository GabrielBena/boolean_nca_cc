/**
 * Entry point for the SODC live demo.
 *
 * For M3 / the visualisation milestone ``main.ts``:
 *   1. Loads the deployed weights.
 *   2. Loads a recorded reference trajectory.
 *   3. Re-runs the trajectory through the TS port (parity check).
 *   4. Renders three bit-image strips: input (x), TMT current output
 *      (animated through the per-tick pred_hard snapshots), expected
 *      output (y).
 *   5. Surfaces the per-tick parity report.
 *
 * The "current output" strip starts as the post-step-0 baseline (no TMT
 * step yet — input bits routed through the soft-wire identity) and
 * progressively converges to the expected output as the policy iterates.
 * For the text-reverse trajectory style, the user *literally watches the
 * upside-down text get flipped right-side-up*.
 */

import { runReplayFromJson, type PerTickReport, type ReplayResult } from "./replay";
import { countParameters, loadWeights } from "./weights";

const MOUNT_ID = "sodc-demo";
const BASE = import.meta.env.BASE_URL;
const WEIGHTS_URL = `${BASE}weights/reverse_random_damage.json`;
const TRAJECTORY_URL = `${BASE}weights/reverse_trajectory.json`;

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
  log: HTMLPreElement;
  imagePanel: HTMLDivElement;
  inputCanvas: HTMLCanvasElement;
  currentCanvas: HTMLCanvasElement;
  expectedCanvas: HTMLCanvasElement;
  tickLabel: HTMLElement;
  table: HTMLTableElement;
}

function buildUI(root: HTMLElement): UI {
  clearChildren(root);
  const wrap = el<HTMLDivElement>("div", "sodc-wrap");
  const status = el<HTMLDivElement>("div", "sodc-status", "Loading weights ...");
  const log = el<HTMLPreElement>("pre", "sodc-log");

  // Image panel: three labelled canvases stacked vertically.
  const imagePanel = el<HTMLDivElement>("div", "sodc-image-panel");
  const tickLabel = el<HTMLDivElement>("div", "sodc-tick-label", "tick 0");
  const inputCanvas = makeBitCanvas("input  (x)");
  const currentCanvas = makeBitCanvas("current TMT output");
  const expectedCanvas = makeBitCanvas("expected (y)");
  imagePanel.append(tickLabel, inputCanvas.row, currentCanvas.row, expectedCanvas.row);

  const table = el<HTMLTableElement>("table", "sodc-table");

  // Inline style — keeps the widget self-contained when embedded.
  const style = document.createElement("style");
  style.textContent = `
    .sodc-wrap { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
    .sodc-status { font-size: 1.1em; padding: 0.5em 0; font-weight: 600; }
    .sodc-status.pass { color: #1a7f37; }
    .sodc-status.fail { color: #cf222e; }
    .sodc-log { background: #f6f8fa; padding: 0.75em 1em; border-radius: 6px;
                font-size: 0.85em; line-height: 1.4; max-height: 14em; overflow: auto; }
    .sodc-image-panel { margin: 1em 0; padding: 1em; background: #0d1117;
                        border-radius: 8px; color: #d0d7de; }
    .sodc-tick-label { font-size: 0.9em; opacity: 0.85; margin-bottom: 0.5em;
                       font-variant-numeric: tabular-nums; }
    .sodc-image-row { display: grid; grid-template-columns: 12em 1fr; align-items: center;
                      gap: 1em; margin: 0.4em 0; }
    .sodc-image-label { font-size: 0.85em; opacity: 0.85; text-align: right; }
    .sodc-image-canvas { width: 100%; image-rendering: pixelated;
                         border: 1px solid #30363d; background: #1117; }
    .sodc-table { border-collapse: collapse; margin-top: 0.6em; font-size: 0.85em; }
    .sodc-table th, .sodc-table td { padding: 2px 10px; text-align: right;
                                     border-bottom: 1px solid #eaecef;
                                     font-variant-numeric: tabular-nums; }
    .sodc-table th { text-align: right; background: #f6f8fa; }
    .sodc-table th:first-child, .sodc-table td:first-child { text-align: left; }
  `;
  wrap.append(style, status, log, imagePanel, table);
  root.append(wrap);
  return {
    status,
    log,
    imagePanel,
    inputCanvas: inputCanvas.canvas,
    currentCanvas: currentCanvas.canvas,
    expectedCanvas: expectedCanvas.canvas,
    tickLabel,
    table,
  };
}

function makeBitCanvas(labelText: string): { row: HTMLDivElement; canvas: HTMLCanvasElement } {
  const row = el<HTMLDivElement>("div", "sodc-image-row");
  const label = el<HTMLDivElement>("div", "sodc-image-label", labelText);
  const canvas = el<HTMLCanvasElement>("canvas", "sodc-image-canvas");
  // ``aspect-ratio: caseN / outputBits`` will be set when we know the shape.
  row.append(label, canvas);
  return { row, canvas };
}

// ---------------------------------------------------------------------------
// Bit-image rendering
// ---------------------------------------------------------------------------

/** Render ``bits`` (shape ``[caseN, nBits]``, values in {0, 1} or {0..1})
 *  to ``canvas`` as a ``nBits × caseN`` image (bit ``b`` = row ``b``,
 *  case ``c`` = column ``c``). White = 1, black = 0. CSS ``image-rendering:
 *  pixelated`` keeps it crisp under arbitrary scaling. */
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
    // Floor the rendered height so wide-and-short images don't vanish.
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
// Per-tick parity table
// ---------------------------------------------------------------------------

function buildPerTickTable(table: HTMLTableElement, perTick: PerTickReport[]) {
  clearChildren(table);
  const thead = el<HTMLTableSectionElement>("thead");
  const trh = el<HTMLTableRowElement>("tr");
  for (const h of ["step", "hard_acc (jax)", "hard_acc (ts)", "Δ", "Δ logits", "Δ hidden", "Δ loss"]) {
    trh.append(el("th", undefined, h));
  }
  thead.append(trh);
  const tbody = el<HTMLTableSectionElement>("tbody");
  for (const row of perTick) {
    const tr = el<HTMLTableRowElement>("tr");
    for (const cell of [
      String(row.step),
      row.hardAccJax.toFixed(4),
      row.hardAccTs.toFixed(4),
      row.hardAccDelta.toFixed(4),
      row.maxAbsLogitsDelta.toExponential(2),
      row.maxAbsHiddenDelta.toExponential(2),
      row.maxAbsLossDelta.toExponential(2),
    ]) {
      tr.append(el("td", undefined, cell));
    }
    tbody.append(tr);
  }
  table.append(thead, tbody);
}

// ---------------------------------------------------------------------------
// Animate the "current" canvas through per-tick snapshots
// ---------------------------------------------------------------------------

function animateTrajectory(ui: UI, result: ReplayResult, fps = 6): void {
  const frames: { step: number; bits: Float32Array; hardAcc: number }[] = [
    { step: 0, bits: result.initialPredHard, hardAcc: NaN },
    ...result.perTick.map((t) => ({
      step: t.step,
      bits: t.predHard,
      hardAcc: t.hardAccTs,
    })),
  ];
  let i = 0;
  const intervalMs = 1000 / fps;
  const draw = () => {
    const f = frames[i % frames.length];
    drawBitImage(ui.currentCanvas, f.bits, result.caseN, result.outputBits);
    const accStr = Number.isFinite(f.hardAcc) ? `, hard_acc=${f.hardAcc.toFixed(4)}` : "";
    ui.tickLabel.textContent = `tick ${f.step}${accStr}`;
    i++;
  };
  draw();
  setInterval(draw, intervalMs);
}

// ---------------------------------------------------------------------------
// Bootstrap
// ---------------------------------------------------------------------------

function logLine(ui: UI, line: string) {
  ui.log.textContent += line + "\n";
}

async function bootstrap(): Promise<void> {
  const root = document.getElementById(MOUNT_ID);
  if (!root) {
    console.error(`[sodc-demo] mount target #${MOUNT_ID} not found`);
    return;
  }
  const ui = buildUI(root);
  try {
    ui.status.textContent = "Loading weights ...";
    const t0 = performance.now();
    const weights = await loadWeights(WEIGHTS_URL);
    const tLoad = performance.now() - t0;
    logLine(
      ui,
      `weights:  ${WEIGHTS_URL}\n` +
        `  arch:    ${weights.header.modelKind} (gathered) · ` +
        `D=${weights.header.attentionDim}, H=${weights.header.numHeads}, ` +
        `arity=${weights.header.arity}, hidden=${weights.header.circuitHiddenDim}\n` +
        `  params:  ${countParameters(weights).toLocaleString()}\n` +
        `  dtype:   ${weights.header.tensorDtype}\n` +
        `  loaded:  ${tLoad.toFixed(0)} ms`,
    );

    ui.status.textContent = "Fetching reference trajectory ...";
    const trajResp = await fetch(TRAJECTORY_URL);
    if (!trajResp.ok) {
      throw new Error(`Failed to fetch trajectory: ${trajResp.status}`);
    }
    const trajRaw = await trajResp.json();

    ui.status.textContent = "Running parity replay ...";
    const t1 = performance.now();
    const result = runReplayFromJson(weights, trajRaw);
    const tReplay = performance.now() - t1;
    logLine(ui, `replay:   ${result.nTicks} ticks in ${tReplay.toFixed(0)} ms`);
    if (result.text) {
      logLine(ui, `task:     ${result.taskStyle} (text="${result.text}", n_cases=${result.caseN})`);
    } else {
      logLine(ui, `task:     ${result.taskStyle} (n_cases=${result.caseN})`);
    }
    logLine(ui, result.message);
    ui.status.textContent = result.message;
    ui.status.classList.add(result.pass ? "pass" : "fail");

    drawBitImage(ui.inputCanvas, result.taskInputBits, result.caseN, result.inputBits);
    drawBitImage(ui.expectedCanvas, result.taskTargetBits, result.caseN, result.outputBits);
    animateTrajectory(ui, result);

    buildPerTickTable(ui.table, result.perTick);
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
