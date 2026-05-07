/**
 * Canvas2D circuit rendering.
 *
 * Direct port of ``_render_circuit_diagram`` from ``training_demo.py``:
 *   - Gates: rounded rectangles, fill colour modulated by per-case activation.
 *   - Groups: thin outline around the gates inside each "wired group".
 *   - Wires: cubic Bézier curves from previous-layer-gate to current-gate
 *     input slot, brightness modulated by sender activation.
 *   - Damaged gates: red overlay.
 *
 * The activations come from a single "active case" — the policy's prediction
 * for one specific input pattern — so the user sees how that input flows
 * through the circuit, not just batch averages.
 */

import {
  type CircuitState,
  type Topology,
  circuitForward,
} from "./circuit";

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

export interface RenderTarget {
  canvas: HTMLCanvasElement;
  ctx: CanvasRenderingContext2D;
  /** CSS-pixel logical dimensions (separate from the backing-store dimensions
   *  which we scale by ``devicePixelRatio`` for HiDPI crispness). */
  width: number;
  height: number;
  dpr: number;
}

export interface DrawOptions {
  /** Flat index of a gate to highlight (e.g. last shotgun-chosen, hovered). */
  highlight?: Int32Array | number | null;
  /** Compact mode: smaller gates / less padding. Mobile-friendly. */
  compact?: boolean;
}

// ---------------------------------------------------------------------------
// Setup
// ---------------------------------------------------------------------------

/** Initialise / re-size the canvas to ``logicalWidth × logicalHeight`` CSS px,
 *  with the backing store scaled by devicePixelRatio for HiDPI. Call this
 *  once at boot, and again on window resize. */
export function setupCanvas(
  canvas: HTMLCanvasElement,
  logicalWidth: number,
  logicalHeight: number,
): RenderTarget {
  const dpr = window.devicePixelRatio || 1;
  const ctx = canvas.getContext("2d", { alpha: false });
  if (!ctx) throw new Error("2D context unavailable");
  canvas.width = Math.max(1, Math.floor(logicalWidth * dpr));
  canvas.height = Math.max(1, Math.floor(logicalHeight * dpr));
  canvas.style.width = `${logicalWidth}px`;
  canvas.style.height = `${logicalHeight}px`;
  // We do all geometry in CSS-px and let the backing store scale.
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { canvas, ctx, width: logicalWidth, height: logicalHeight, dpr };
}

// ---------------------------------------------------------------------------
// Single-case forward — gives us the per-layer activations for one input
// ---------------------------------------------------------------------------

/** Slice a single case from a flat ``[caseN, inputBits]`` x array. */
export function activeCaseInput(
  xData: Float32Array,
  caseIdx: number,
  inputBits: number,
): Float32Array {
  const out = new Float32Array(inputBits);
  for (let b = 0; b < inputBits; b++) out[b] = xData[caseIdx * inputBits + b];
  return out;
}

/** Run ``circuitForward`` for one input case and return per-layer activations
 *  as ``Float32Array``s of length ``layerSize``. Soft path (sigmoid + interp). */
export function singleCaseActivations(
  topology: Topology,
  state: CircuitState,
  caseInput: Float32Array,
  arity: number,
): Float32Array[] {
  const acts = circuitForward(topology, state, caseInput, /* caseN */ 1, /* hard */ false, arity);
  // ``acts[L]`` is shape ``[1, layerSize]`` — drop the leading 1 so callers
  // can index by gate directly.
  return acts.map((a) => a);
}

// ---------------------------------------------------------------------------
// Geometry — pure functions of the topology, not the state
// ---------------------------------------------------------------------------

interface Layout {
  /** Per-layer arrays. */
  layerY: number[];
  /** Flat per-gate centre x. ``gateX[layerStart[li] + g]`` for layer ``li`` gate ``g``. */
  gateX: number[];
  gateW: number;
  groupW: number;
  /** Per-layer x of the *first* gate of each group (one entry per group). */
  groupXByLayer: number[][];
  /** Per-layer (groupN, groupSize). */
  groupShape: { groupN: number; groupSize: number }[];
  /** Per-layer top y (gate top, not centre). */
  topY: (li: number) => number;
  /** Vertical gate height (= horizontal width in the symmetric layout). */
  d: number;
  pad: number;
}

function computeLayout(
  topology: Topology,
  width: number,
  height: number,
  d: number,
  pad: number,
): Layout {
  const nLayers = topology.layerSizes.length;
  const layerH = (height - 2 * pad - d) / Math.max(1, nLayers - 1);
  const layerY: number[] = [];
  for (let li = 0; li < nLayers; li++) {
    layerY.push(pad + li * layerH + d / 2);
  }
  const gateX: number[] = new Array(topology.nNodes);
  const groupShape: { groupN: number; groupSize: number }[] = [];
  const groupXByLayer: number[][] = [];
  let usableW = width - 2 * pad;
  for (let li = 0; li < nLayers; li++) {
    const [gateN, groupSize] = topology.layerSizes[li];
    const groupN = gateN / Math.max(groupSize, 1);
    const spanX = usableW / Math.max(groupN, 1);
    const groupW = Math.min(d * groupSize, spanX - 6);
    const gateW = groupW / Math.max(groupSize, 1);
    const groupXs: number[] = [];
    for (let gi = 0; gi < groupN; gi++) {
      const groupCenterX = pad + (gi + 0.5) * spanX;
      groupXs.push(groupCenterX);
      for (let gj = 0; gj < groupSize; gj++) {
        const ofs = (gj - groupSize / 2 + 0.5) * gateW;
        gateX[topology.layerStart[li] + gi * groupSize + gj] = groupCenterX + ofs;
      }
    }
    groupShape.push({ groupN, groupSize });
    groupXByLayer.push(groupXs);
    // groupW is per-layer; we only need the layer-0 reference value below for
    // wire-target spacing. Stash via closure capture — simplest path is to
    // recompute inline in the wire pass.
  }
  // For now we expose layer-1's gate width as the canonical gate width.
  // The layout struct keeps per-layer info; the renderer uses layer-local
  // values where it matters (recomputed at draw time).
  const fallbackGateW = (() => {
    const li = 1;
    const [gateN, groupSize] = topology.layerSizes[li] ?? topology.layerSizes[0];
    const groupN = gateN / Math.max(groupSize, 1);
    const spanX = usableW / Math.max(groupN, 1);
    const groupW = Math.min(d * groupSize, spanX - 6);
    return groupW / Math.max(groupSize, 1);
  })();
  void usableW;
  return {
    layerY,
    gateX,
    gateW: fallbackGateW,
    groupW: d,
    groupXByLayer,
    groupShape,
    topY: (li) => layerY[li] - d / 2,
    d,
    pad,
  };
}

// ---------------------------------------------------------------------------
// Colour helpers
// ---------------------------------------------------------------------------

/** Map a soft activation [0, 1] to a bright "signal" colour ramp.
 *  Dark blue-grey at 0 → vivid green at 1, tuned for the dark theme. */
function activationColor(a: number): string {
  const v = Math.max(0, Math.min(1, a));
  // Background base + green channel ramp. Same family as the bit strips so
  // the signal flow feels visually unified across input → circuit → output.
  const r = 32 + Math.round(v * 20);
  const g = 50 + Math.round(v * 180);
  const b = 60 + Math.round(v * 60);
  return `rgb(${r},${g},${b})`;
}

function wireColor(senderAct: number): string {
  const v = Math.max(0, Math.min(1, senderAct));
  const r = 60 + Math.round(v * 40);
  const g = 80 + Math.round(v * 130);
  const b = 95 + Math.round(v * 60);
  return `rgba(${r},${g},${b},${0.35 + v * 0.45})`;
}

// ---------------------------------------------------------------------------
// Rounded-rect helper
// ---------------------------------------------------------------------------

function fillRoundRect(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  w: number,
  h: number,
  r: number,
  fill: string,
): void {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.lineTo(x + w - r, y);
  ctx.quadraticCurveTo(x + w, y, x + w, y + r);
  ctx.lineTo(x + w, y + h - r);
  ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
  ctx.lineTo(x + r, y + h);
  ctx.quadraticCurveTo(x, y + h, x, y + h - r);
  ctx.lineTo(x, y + r);
  ctx.quadraticCurveTo(x, y, x + r, y);
  ctx.closePath();
  ctx.fillStyle = fill;
  ctx.fill();
}

function strokeRoundRect(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  w: number,
  h: number,
  r: number,
  stroke: string,
  width = 1,
): void {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.lineTo(x + w - r, y);
  ctx.quadraticCurveTo(x + w, y, x + w, y + r);
  ctx.lineTo(x + w, y + h - r);
  ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
  ctx.lineTo(x + r, y + h);
  ctx.quadraticCurveTo(x, y + h, x, y + h - r);
  ctx.lineTo(x, y + r);
  ctx.quadraticCurveTo(x, y, x + r, y);
  ctx.closePath();
  ctx.strokeStyle = stroke;
  ctx.lineWidth = width;
  ctx.stroke();
}

// ---------------------------------------------------------------------------
// Main draw pass
// ---------------------------------------------------------------------------

const D_DEFAULT = 18;
const PAD_DEFAULT = 14;
const D_COMPACT = 12;
const PAD_COMPACT = 8;

export function drawCircuit(
  target: RenderTarget,
  topology: Topology,
  state: CircuitState,
  activations: Float32Array[],
  opts: DrawOptions = {},
): void {
  const { ctx, width, height } = target;
  const d = opts.compact ? D_COMPACT : D_DEFAULT;
  const pad = opts.compact ? PAD_COMPACT : PAD_DEFAULT;

  // Background.
  ctx.fillStyle = "#0d1117";
  ctx.fillRect(0, 0, width, height);

  const layout = computeLayout(topology, width, height, d, pad);
  const nLayers = topology.layerSizes.length;
  const arity = topology.arity;
  const usableW = width - 2 * pad;

  // ---- Wires (drawn first so they sit *behind* the gates) ------------
  for (let li = 1; li < nLayers; li++) {
    const [gateN, groupSize] = topology.layerSizes[li];
    const groupN = gateN / Math.max(groupSize, 1);
    const spanX = usableW / Math.max(groupN, 1);
    const groupW = Math.min(d * groupSize, spanX - 6);
    const wires = topology.wires[li - 1]; // shape [arity, groupN] flat
    const prevLayerStart = topology.layerStart[li - 1];
    const prevActs = activations[li - 1];
    const myY = layout.layerY[li];
    const prevY = layout.layerY[li - 1];
    const my = (prevY + myY) / 2;
    for (let gi = 0; gi < groupN; gi++) {
      const groupCenterX = pad + (gi + 0.5) * spanX;
      for (let a = 0; a < arity; a++) {
        const sendLocal = wires[a * groupN + gi];
        const sendGlobal = prevLayerStart + sendLocal;
        const x0 = layout.gateX[sendGlobal];
        const senderAct = prevActs ? prevActs[sendLocal] : 0;
        // Distribute ``arity`` input slots evenly across groupW.
        const x1 = groupCenterX + ((a + 0.5) / arity) * groupW - groupW / 2;
        ctx.beginPath();
        ctx.moveTo(x0, prevY + d / 2);
        ctx.bezierCurveTo(x0, my, x1, my, x1, myY - d / 2);
        ctx.strokeStyle = wireColor(senderAct);
        ctx.lineWidth = 1;
        ctx.stroke();
      }
    }
  }

  // ---- Group outlines + gates --------------------------------------
  for (let li = 0; li < nLayers; li++) {
    const [gateN, groupSize] = topology.layerSizes[li];
    const groupN = gateN / Math.max(groupSize, 1);
    const spanX = usableW / Math.max(groupN, 1);
    const groupW = Math.min(d * groupSize, spanX - 6);
    const gateW = groupW / Math.max(groupSize, 1);
    const y = layout.layerY[li];
    const acts = activations[li];

    // Group outlines (subtle).
    for (let gi = 0; gi < groupN; gi++) {
      const groupCenterX = pad + (gi + 0.5) * spanX;
      strokeRoundRect(
        ctx,
        groupCenterX - groupW / 2,
        y - d / 2,
        groupW,
        d,
        4,
        "rgba(220, 230, 240, 0.18)",
        1,
      );
    }

    // Gates.
    for (let gi = 0; gi < groupN; gi++) {
      for (let gj = 0; gj < groupSize; gj++) {
        const flatIdx = topology.layerStart[li] + gi * groupSize + gj;
        const a = acts ? acts[gi * groupSize + gj] : 0;
        const x = layout.gateX[flatIdx];
        const left = x - gateW / 2;
        const top = y - d / 2;
        fillRoundRect(ctx, left + 0.5, top + 0.5, gateW - 1, d - 1, 3, activationColor(a));

        // Damaged-gate overlay (red, semi-transparent).
        if (state.gateMask[flatIdx] === 0) {
          fillRoundRect(ctx, left + 0.5, top + 0.5, gateW - 1, d - 1, 3, "rgba(207, 34, 46, 0.55)");
          strokeRoundRect(
            ctx,
            left + 0.5,
            top + 0.5,
            gateW - 1,
            d - 1,
            3,
            "rgba(255, 90, 90, 0.95)",
            1.5,
          );
        }

        // Highlight (last damaged / hovered).
        if (isHighlighted(opts.highlight, flatIdx)) {
          strokeRoundRect(
            ctx,
            left - 0.5,
            top - 0.5,
            gateW + 1,
            d + 1,
            4,
            "rgba(255, 220, 90, 0.95)",
            2,
          );
        }
      }
    }
  }
}

function isHighlighted(highlight: DrawOptions["highlight"], flatIdx: number): boolean {
  if (highlight == null) return false;
  if (typeof highlight === "number") return highlight === flatIdx;
  for (let i = 0; i < highlight.length; i++) if (highlight[i] === flatIdx) return true;
  return false;
}

// ---------------------------------------------------------------------------
// Hit-testing — used by Phase 3's click-to-damage
// ---------------------------------------------------------------------------

/** Translate a CSS-pixel mouse position (relative to the canvas) into a flat
 *  gate index, or ``-1`` for a miss. */
export function pickGate(
  target: RenderTarget,
  topology: Topology,
  px: number,
  py: number,
  compact = false,
): number {
  const d = compact ? D_COMPACT : D_DEFAULT;
  const pad = compact ? PAD_COMPACT : PAD_DEFAULT;
  const layout = computeLayout(topology, target.width, target.height, d, pad);
  const usableW = target.width - 2 * pad;
  for (let li = 0; li < topology.layerSizes.length; li++) {
    const [gateN, groupSize] = topology.layerSizes[li];
    const groupN = gateN / Math.max(groupSize, 1);
    const spanX = usableW / Math.max(groupN, 1);
    const groupW = Math.min(d * groupSize, spanX - 6);
    const gateW = groupW / Math.max(groupSize, 1);
    const y = layout.layerY[li];
    if (Math.abs(py - y) > d / 2) continue;
    for (let g = 0; g < gateN; g++) {
      const flatIdx = topology.layerStart[li] + g;
      const x = layout.gateX[flatIdx];
      if (Math.abs(px - x) < gateW / 2) return flatIdx;
    }
  }
  return -1;
}
