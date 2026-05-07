/**
 * TS-side parity replay.
 *
 * Loads the reference trajectory written by ``export/record_trajectory.py``,
 * reconstructs the same initial state in this module's ``CircuitState``,
 * drives ``tick`` for ``n_ticks`` steps, and asserts per-step agreement
 * with the recorded NumPy-oracle values. The browser shows a verdict:
 * "TS-side parity: PASS" or per-step diagnostics on FAIL.
 */

import {
  type CircuitState,
  type Topology,
  buildTopology,
} from "./circuit";
import { type Scratch, computeResiduals, makeScratch, tick } from "./tmt";
import type { TMTWeights } from "./weights";

// ---------------------------------------------------------------------------
// Trajectory schema (mirror of record_trajectory.py)
// ---------------------------------------------------------------------------

interface B64Array {
  shape: number[];
  data_b64: string;
}

interface RawTrajectory {
  header: {
    task: string;
    task_style?: string;
    text?: string | null;
    input_bits: number;
    output_bits: number;
    arity: number;
    circuit_hidden_dim: number;
    max_neighbors: number;
    case_n: number;
    n_ticks: number;
    seed: number;
    source_run_id: string;
    weights_path: string | null;
  };
  layer_sizes: [number, number][];
  wires: B64Array[];
  initial_state: {
    logits: B64Array;
    hidden: B64Array;
    loss: B64Array;
    gate_mask: B64Array;
  };
  task_data: {
    x: B64Array;
    y: B64Array;
  };
  ticks: Array<{
    step: number;
    hard_accuracy: number;
    soft_accuracy: number;
    logits: B64Array;
    hidden: B64Array;
    loss: B64Array;
  }>;
}

function base64ToBytes(b64: string): Uint8Array {
  const bin = atob(b64);
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}

function decodeFloat32(entry: B64Array): Float32Array {
  const bytes = base64ToBytes(entry.data_b64);
  // Aligned views: copy into a fresh ArrayBuffer to guarantee 4-byte alignment.
  const buf = new ArrayBuffer(bytes.byteLength);
  new Uint8Array(buf).set(bytes);
  return new Float32Array(buf);
}

function decodeInt32(entry: B64Array): Int32Array {
  const bytes = base64ToBytes(entry.data_b64);
  const buf = new ArrayBuffer(bytes.byteLength);
  new Uint8Array(buf).set(bytes);
  return new Int32Array(buf);
}

// ---------------------------------------------------------------------------
// Replay
// ---------------------------------------------------------------------------

export interface PerTickReport {
  step: number;
  hardAccJax: number;
  hardAccTs: number;
  hardAccDelta: number;
  maxAbsLogitsDelta: number;
  maxAbsHiddenDelta: number;
  maxAbsLossDelta: number;
  /** ``[caseN * outputBits]`` hard-prediction snapshot the demo animates. */
  predHard: Float32Array;
}

export interface ReplayResult {
  pass: boolean;
  nTicks: number;
  maxHardAccDelta: number;
  perTick: PerTickReport[];
  message: string;
  /** Initial (post step-0 residual recompute) hard prediction — the pre-tick
   *  baseline the visualisation starts from. */
  initialPredHard: Float32Array;
  /** Static x / y bit images decoded from the trajectory, ready to draw. */
  taskInputBits: Float32Array;
  taskTargetBits: Float32Array;
  /** Header echoed from the trajectory so the renderer knows shape + layout. */
  caseN: number;
  inputBits: number;
  outputBits: number;
  taskStyle: string;
  text: string | null;
}

const HARD_ACC_TOL = 0.05;

/** Same as :func:`runReplay` but takes a parsed JSON document directly,
 *  for offline / Node use where ``fetch`` isn't appropriate. */
export function runReplayFromJson(weights: TMTWeights, raw: unknown): ReplayResult {
  const traj = raw as RawTrajectory;
  return _doReplay(weights, traj);
}

export async function runReplay(
  weights: TMTWeights,
  trajectoryUrl: string,
): Promise<ReplayResult> {
  const resp = await fetch(trajectoryUrl);
  if (!resp.ok) {
    throw new Error(`Failed to fetch trajectory at ${trajectoryUrl}: ${resp.status}`);
  }
  return _doReplay(weights, (await resp.json()) as RawTrajectory);
}

function _doReplay(weights: TMTWeights, raw: RawTrajectory): ReplayResult {
  const h = raw.header;

  // Reconstruct topology from recorded wires.
  const wireData: Int32Array[] = raw.wires.map(decodeInt32);
  const wireShapes: [number, number][] = raw.wires.map((w) => [w.shape[0], w.shape[1]]);
  const layerSizes: [number, number][] = raw.layer_sizes.map((p) => [p[0], p[1]]);
  const topology: Topology = buildTopology(layerSizes, wireData, wireShapes, {
    arity: h.arity,
    hiddenDim: h.circuit_hidden_dim,
    maxNeighbors: h.max_neighbors,
  });

  // Hydrate state from the initial-state record (post step-0 residual
  // recompute, matches the JAX side's get_loss_and_update_graph step 0).
  const state: CircuitState = {
    logits: decodeFloat32(raw.initial_state.logits),
    hidden: decodeFloat32(raw.initial_state.hidden),
    loss: decodeFloat32(raw.initial_state.loss),
    gateMask: decodeFloat32(raw.initial_state.gate_mask),
  };
  const expectedLogits = topology.nNodes * (1 << h.arity);
  if (state.logits.length !== expectedLogits) {
    throw new Error(
      `initial_state.logits length ${state.logits.length} ≠ N×lutDim ${expectedLogits}`,
    );
  }

  const xData = decodeFloat32(raw.task_data.x);
  const yData = decodeFloat32(raw.task_data.y);
  const caseN = h.case_n;
  // Initial frame (post step-0 residual recompute) — gives the visualisation a
  // tick-0 baseline before any TMT step runs.
  const aux0 = computeResiduals(state, topology, xData, yData, caseN, h.arity);
  const initialPredHard = new Float32Array(aux0.predHard);

  const scratch: Scratch = makeScratch(weights, topology.nNodes);

  const perTick: PerTickReport[] = [];
  let maxHardAccDelta = 0;

  for (const recorded of raw.ticks) {
    const aux = tick(state, topology, weights, scratch, xData, yData, caseN, h.arity);

    const recordedLogits = decodeFloat32(recorded.logits);
    const recordedHidden = decodeFloat32(recorded.hidden);
    const recordedLoss = decodeFloat32(recorded.loss);

    let mLog = 0;
    for (let i = 0; i < state.logits.length; i++) {
      const d = Math.abs(state.logits[i] - recordedLogits[i]);
      if (d > mLog) mLog = d;
    }
    let mHid = 0;
    for (let i = 0; i < state.hidden.length; i++) {
      const d = Math.abs(state.hidden[i] - recordedHidden[i]);
      if (d > mHid) mHid = d;
    }
    let mLoss = 0;
    for (let i = 0; i < state.loss.length; i++) {
      const d = Math.abs(state.loss[i] - recordedLoss[i]);
      if (d > mLoss) mLoss = d;
    }

    const hardDelta = Math.abs(aux.hardAccuracy - recorded.hard_accuracy);
    if (hardDelta > maxHardAccDelta) maxHardAccDelta = hardDelta;
    perTick.push({
      step: recorded.step,
      hardAccJax: recorded.hard_accuracy,
      hardAccTs: aux.hardAccuracy,
      hardAccDelta: hardDelta,
      maxAbsLogitsDelta: mLog,
      maxAbsHiddenDelta: mHid,
      maxAbsLossDelta: mLoss,
      predHard: new Float32Array(aux.predHard),
    });
  }

  const pass = maxHardAccDelta < HARD_ACC_TOL;
  const message = pass
    ? `TS-side parity PASS: max hard_acc Δ over ${raw.ticks.length} ticks = ${maxHardAccDelta.toFixed(4)} (tol ${HARD_ACC_TOL})`
    : `TS-side parity FAIL: max hard_acc Δ = ${maxHardAccDelta.toFixed(4)} > tol ${HARD_ACC_TOL}`;
  return {
    pass,
    nTicks: raw.ticks.length,
    maxHardAccDelta,
    perTick,
    message,
    initialPredHard,
    taskInputBits: xData,
    taskTargetBits: yData,
    caseN,
    inputBits: h.input_bits,
    outputBits: h.output_bits,
    taskStyle: h.task_style ?? "sequential",
    text: h.text ?? null,
  };
}

