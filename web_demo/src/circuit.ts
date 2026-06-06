/**
 * Boolean circuit primitives, ported line-for-line from
 * ``export/numpy_oracle.py``:
 *
 *  - ``Topology``   — static structure (layer sizes, wires, attention mask,
 *                     positional encodings, gathered-attention neighbour cache).
 *  - ``CircuitState``— mutable per-tick state (logits, hidden, loss, gate mask).
 *  - ``buildTopology`` rebuilds Topology from scratch (e.g. after wire shuffle).
 *  - ``circuitForward`` runs the multilinear LUT forward over a batch of inputs.
 *  - The four user-action perturbations: ``resetCircuit``, ``shuffleWires``,
 *    ``applyShotgun``, ``damageGate``.
 *
 * Numerical contract: hard predictions match the JAX policy on a fixed
 * seed (validated on the Python side; see ``test_oracle.py``).
 */

import { sinusoidalPE, sigmoid, stableArgsortRowsByNotMask } from "./math";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/** ``[gateN, groupSize]`` pair for one circuit layer (input layer included as layer 0). */
export type LayerSize = readonly [number, number];

export interface Topology {
  /** Per-layer (gate count, group size). Layer 0 is the input. */
  layerSizes: LayerSize[];
  /** Per-gate-layer wires, flat int32 of shape ``[arity * groupN]`` row-major (arity-major). */
  wires: Int32Array[];
  /** Original ``[arity, groupN]`` for each wire array — useful for ``circuitForward``. */
  wiresShape: [number, number][];
  arity: number;
  /** Total node count across all layers. */
  nNodes: number;
  /** Cumulative node-index offsets per layer. ``layerStart[li]`` = first node index of layer ``li``. */
  layerStart: number[];
  outputStart: number;
  outputEnd: number;
  /** Dense ``[nNodes, nNodes]`` binary mask, row-major. ``mask[r * N + s] = 1`` if there's an
   *  edge ``s -> r`` (or ``r == s`` for self-loops). Always symmetric for our bidirectional graph. */
  attentionMask: Uint8Array;
  /** Sinusoidal layer-depth PE, ``[nNodes, hiddenDim]``. */
  layerPe: Float32Array;
  /** Sinusoidal intra-layer PE, ``[nNodes, hiddenDim]``. */
  intraLayerPe: Float32Array;
  /** DAG-distance PE ``[nNodes, 2 * (hiddenDim >> 1)]`` — sinusoidal encoding of
   *  (dist_from_input, dist_to_output) over forward edges. Wire-dependent:
   *  rebuilt on shuffle along with the rest of the topology. */
  distPe: Float32Array;
  /** Padded neighbour indices for gathered attention, ``[nNodes, maxNeighbors]``. */
  neighborIndices: Int32Array;
  /** Per-row flag: ``true`` for real neighbours, ``false`` for pad slots. */
  neighborMask: Uint8Array;
  maxNeighbors: number;
}

export interface CircuitState {
  /** Flat ``[nNodes, logitDim]``. Input rows are zeros; clamped gates are ``-10`` everywhere. */
  logits: Float32Array;
  /** Flat ``[nNodes, hiddenDim]``. */
  hidden: Float32Array;
  /** Flat ``[nNodes]``. Only output rows carry the per-output-bit residual signal. */
  loss: Float32Array;
  /** ``1`` for healthy gates, ``0`` for clamped (damaged). */
  gateMask: Float32Array;
}

// ---------------------------------------------------------------------------
// Topology construction
// ---------------------------------------------------------------------------

const FAULTY_VALUE = -10;
const PE_MAX_VAL = 10000;

function fillSenders(
  layerSizes: LayerSize[],
  layerStart: number[],
  wires: Int32Array[],
  wiresShape: [number, number][],
  arity: number,
): { senders: Int32Array; receivers: Int32Array } {
  let totalEdges = 0;
  for (let li = 1; li < layerSizes.length; li++) {
    totalEdges += layerSizes[li][0] * arity;
  }
  const senders = new Int32Array(totalEdges);
  const receivers = new Int32Array(totalEdges);
  let cursor = 0;
  for (let li = 1; li < layerSizes.length; li++) {
    const [gateN, groupSize] = layerSizes[li];
    const groupN = gateN / groupSize;
    const w = wires[li - 1];
    const [arityW, groupNW] = wiresShape[li - 1];
    if (arityW !== arity || groupNW !== groupN) {
      throw new Error(
        `wire shape mismatch: layer ${li} expected (${arity}, ${groupN}) got (${arityW}, ${groupNW})`,
      );
    }
    const recvBase = layerStart[li];
    const sendBase = layerStart[li - 1];
    // Each *gate* receives ``arity`` edges; gates within a group share wires.
    for (let g = 0; g < gateN; g++) {
      const recvNode = recvBase + g;
      const groupIdx = (g / groupSize) | 0;
      for (let a = 0; a < arity; a++) {
        // wires shape is [arity, groupN]; transpose gives groupN x arity.
        const sendLocal = w[a * groupN + groupIdx];
        senders[cursor] = sendBase + sendLocal;
        receivers[cursor] = recvNode;
        cursor++;
      }
    }
  }
  return { senders, receivers };
}

function buildAttentionMask(senders: Int32Array, receivers: Int32Array, nNodes: number): Uint8Array {
  const mask = new Uint8Array(nNodes * nNodes);
  for (let i = 0; i < nNodes; i++) mask[i * nNodes + i] = 1; // self-loop
  for (let e = 0; e < senders.length; e++) {
    const s = senders[e];
    const r = receivers[e];
    mask[r * nNodes + s] = 1; // forward
    mask[s * nNodes + r] = 1; // backward (bidirectional_edges=True default)
  }
  return mask;
}

function buildPositionalEncodings(
  layerSizes: LayerSize[],
  hiddenDim: number,
): { layerPe: Float32Array; intraLayerPe: Float32Array; nNodes: number; layerStart: number[] } {
  let nNodes = 0;
  const layerStart: number[] = [];
  for (const [g] of layerSizes) {
    layerStart.push(nNodes);
    nNodes += g;
  }
  const layerPe = new Float32Array(nNodes * hiddenDim);
  const intraLayerPe = new Float32Array(nNodes * hiddenDim);
  const totalGateLayers = layerSizes.length - 1;
  for (let li = 0; li < layerSizes.length; li++) {
    const [gateN] = layerSizes[li];
    const off = layerStart[li];
    // Per-layer normalised depth (input = 0.0, output = 1.0).
    const normalizedDepth = totalGateLayers > 0 ? li / totalGateLayers : 0;
    const scaled = normalizedDepth * PE_MAX_VAL;
    const layerScaled = new Float32Array(gateN);
    for (let g = 0; g < gateN; g++) layerScaled[g] = scaled;
    sinusoidalPE(layerPe.subarray(off * hiddenDim, (off + gateN) * hiddenDim), layerScaled, hiddenDim);
    const idx = new Float32Array(gateN);
    for (let g = 0; g < gateN; g++) idx[g] = g;
    sinusoidalPE(intraLayerPe.subarray(off * hiddenDim, (off + gateN) * hiddenDim), idx, hiddenDim);
  }
  return { layerPe, intraLayerPe, nNodes, layerStart };
}

/** DAG-distance PE — line-for-line port of the NumPy oracle's
 *  ``compute_dag_distance_pe`` (itself mirroring
 *  ``utils.positional_encoding.compute_dag_distance_pe``).
 *
 *  Bellman-Ford relaxation over the *forward-only* edge list: dist-from-input
 *  seeds layer 0, dist-to-output seeds the last layer and relaxes reversed
 *  edges. Distances saturate at ``nIterations + 1`` for unreachable nodes
 *  (e.g. gates whose output no downstream gate consumes). Each side is
 *  sinusoidally encoded at ``hiddenDim >> 1`` dims and concatenated. */
function buildDistancePe(
  layerSizes: LayerSize[],
  layerStart: number[],
  senders: Int32Array,
  receivers: Int32Array,
  nNodes: number,
  hiddenDim: number,
): Float32Array {
  const nIterations = layerSizes.length;
  const infVal = nIterations + 1;
  const distIn = new Float32Array(nNodes).fill(infVal);
  const distOut = new Float32Array(nNodes).fill(infVal);
  const inputN = layerSizes[0][0];
  for (let i = 0; i < inputN; i++) distIn[i] = 0;
  const outStart = layerStart[layerSizes.length - 1];
  for (let i = outStart; i < nNodes; i++) distOut[i] = 0;

  const E = senders.length;
  for (let it = 0; it < nIterations; it++) {
    // dist[r] = min(dist[r], dist[s] + 1) — forward; mirrored for the output side.
    for (let e = 0; e < E; e++) {
      const s = senders[e];
      const r = receivers[e];
      const candIn = distIn[s] + 1;
      if (candIn < distIn[r]) distIn[r] = candIn;
      const candOut = distOut[r] + 1;
      if (candOut < distOut[s]) distOut[s] = candOut;
    }
  }

  const perSide = hiddenDim >> 1;
  const distPe = new Float32Array(nNodes * 2 * perSide);
  const peIn = new Float32Array(nNodes * perSide);
  const peOut = new Float32Array(nNodes * perSide);
  sinusoidalPE(peIn, distIn, perSide);
  sinusoidalPE(peOut, distOut, perSide);
  // Concatenate [PE(dist_in), PE(dist_out)] per node.
  for (let n = 0; n < nNodes; n++) {
    distPe.set(peIn.subarray(n * perSide, (n + 1) * perSide), n * 2 * perSide);
    distPe.set(peOut.subarray(n * perSide, (n + 1) * perSide), n * 2 * perSide + perSide);
  }
  return distPe;
}

/** Convert the dense ``[N, N]`` attention mask into padded neighbour indices.
 *
 *  Mirrors ``boolean_nca_cc.models.attention.base.build_neighbor_indices`` and
 *  the NumPy oracle's helper of the same name. Real neighbours come first
 *  (sorted by node index ascending — the stable-argsort behaviour); padding
 *  slots point at index 0 and are zero in ``neighborMask``. */
function buildNeighborCache(
  attentionMask: Uint8Array,
  nNodes: number,
  maxNeighbors: number,
): { neighborIndices: Int32Array; neighborMask: Uint8Array; maxNeighbors: number } {
  const cap = Math.min(maxNeighbors, nNodes);
  const sorted = stableArgsortRowsByNotMask(attentionMask, nNodes);
  const neighborIndices = new Int32Array(nNodes * cap);
  const neighborMask = new Uint8Array(nNodes * cap);
  for (let r = 0; r < nNodes; r++) {
    let degree = 0;
    for (let c = 0; c < nNodes; c++) if (attentionMask[r * nNodes + c]) degree++;
    for (let k = 0; k < cap; k++) {
      neighborIndices[r * cap + k] = sorted[r * nNodes + k];
      neighborMask[r * cap + k] = k < degree ? 1 : 0;
    }
  }
  return { neighborIndices, neighborMask, maxNeighbors: cap };
}

export interface BuildTopologyOptions {
  arity: number;
  hiddenDim: number;
  maxNeighbors: number;
}

export function buildTopology(
  layerSizes: LayerSize[],
  wires: Int32Array[],
  wiresShape: [number, number][],
  opts: BuildTopologyOptions,
): Topology {
  const { arity, hiddenDim, maxNeighbors } = opts;
  const { layerPe, intraLayerPe, nNodes, layerStart } = buildPositionalEncodings(layerSizes, hiddenDim);
  const { senders, receivers } = fillSenders(layerSizes, layerStart, wires, wiresShape, arity);
  const attentionMask = buildAttentionMask(senders, receivers, nNodes);
  const distPe = buildDistancePe(layerSizes, layerStart, senders, receivers, nNodes, hiddenDim);
  const cache = buildNeighborCache(attentionMask, nNodes, maxNeighbors);

  const outputStart = layerStart[layerSizes.length - 1];
  const outputEnd = outputStart + layerSizes[layerSizes.length - 1][0];

  return {
    layerSizes: [...layerSizes],
    wires,
    wiresShape,
    arity,
    nNodes,
    layerStart,
    outputStart,
    outputEnd,
    attentionMask,
    layerPe,
    intraLayerPe,
    distPe,
    neighborIndices: cache.neighborIndices,
    neighborMask: cache.neighborMask,
    maxNeighbors: cache.maxNeighbors,
  };
}

// ---------------------------------------------------------------------------
// Initial state (soft-wire identity + Gaussian noise; mirrors gen_circuit)
// ---------------------------------------------------------------------------

/** Mulberry32 PRNG — 32-bit, deterministic, seedable. Used to mirror
 *  the NumPy ``Generator`` we drive on the Python side, although for the
 *  TS port we never need *bit*-exact reproducibility against numpy
 *  (the deployment validates against JAX-extracted weights, not against
 *  the oracle's random init). Seedable + deterministic is enough for tests. */
export function makeRng(seed: number) {
  let s = seed >>> 0;
  return {
    /** Uniform float in [0, 1). */
    next(): number {
      s = (s + 0x6d2b79f5) >>> 0;
      let t = s;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    },
    /** Standard normal via Box-Muller. */
    normal(): number {
      const u = Math.max(this.next(), 1e-12);
      const v = this.next();
      return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
    },
  };
}
export type Rng = ReturnType<typeof makeRng>;

/** Random wiring matching ``circuits.model.gen_wires`` semantically (but not bit-for-bit). */
export function genWires(
  rng: Rng,
  inN: number,
  outN: number,
  arity: number,
  groupSize: number,
): { data: Int32Array; shape: [number, number] } {
  const edgeN = (outN * arity) / groupSize;
  const n = Math.max(inN, edgeN);
  const perm = new Int32Array(n);
  for (let i = 0; i < n; i++) perm[i] = i;
  // Fisher–Yates shuffle.
  for (let i = n - 1; i > 0; i--) {
    const j = (rng.next() * (i + 1)) | 0;
    const tmp = perm[i];
    perm[i] = perm[j];
    perm[j] = tmp;
  }
  const data = new Int32Array(arity * (edgeN / arity));
  for (let i = 0; i < edgeN; i++) data[i] = perm[i] % inN;
  // Reshape: stored as [arity, groupN] row-major.
  const groupN = edgeN / arity;
  return { data, shape: [arity, groupN] };
}

/** Soft-wire identity init + Gaussian noise. Per-gate LUT logits.
 *  Output is flat ``[gateN, lutDim]`` row-major. */
function softWireLogits(
  rng: Rng,
  gateN: number,
  arity: number,
  noiseScale: number,
  out: Float32Array,
  outOffset: number,
): void {
  const lutDim = 1 << arity;
  // bits[a, c] = ((c >> a) & 1) — same indexing as NumPy oracle's make_nops_with_noise.
  for (let g = 0; g < gateN; g++) {
    const a = g % arity;
    for (let c = 0; c < lutDim; c++) {
      const bit = (c >> a) & 1;
      const base = (2 * bit - 1) * 3.0;
      const noise = rng.normal() * noiseScale;
      out[outOffset + g * lutDim + c] = base + noise;
    }
  }
}

export interface InitStateOptions {
  hiddenDim: number;
  arity: number;
  noiseScale?: number;
}

export function initState(
  topology: Topology,
  opts: InitStateOptions,
  rng: Rng,
): CircuitState {
  const { hiddenDim, arity, noiseScale = 0.1 } = opts;
  const lutDim = 1 << arity;
  const N = topology.nNodes;
  const logits = new Float32Array(N * lutDim);
  for (let li = 1; li < topology.layerSizes.length; li++) {
    const [gateN] = topology.layerSizes[li];
    const off = topology.layerStart[li] * lutDim;
    softWireLogits(rng, gateN, arity, noiseScale, logits, off);
  }
  return {
    logits,
    hidden: new Float32Array(N * hiddenDim),
    loss: new Float32Array(N),
    gateMask: new Float32Array(N).fill(1),
  };
}

// ---------------------------------------------------------------------------
// Circuit forward (multilinear LUT interpolation)
// ---------------------------------------------------------------------------

/** ``run_layer`` for a single batch of cases.
 *
 *  ``luts``  shape ``[groupN, groupSize, lutDim]`` — flat input.
 *  ``inputs[a]`` is the full ``[caseN, prevLayerSize]`` activation tensor of the previous layer;
 *  the gather ``inputs[a][..., wires[a]]`` happens on the fly here using ``wires``.
 *
 *  Output ``acts`` is ``[caseN, gateN]`` flat. */
function runLayer(
  luts: Float32Array,
  prevActs: Float32Array,
  prevSize: number,
  wires: Int32Array,
  arity: number,
  caseN: number,
  groupN: number,
  groupSize: number,
  acts: Float32Array,
  hard: boolean,
): void {
  const lutDim = 1 << arity;
  // Working buffer: per-case multilinear-interp LUT, peeled one input at a time.
  const work = new Float32Array(lutDim);
  for (let c = 0; c < caseN; c++) {
    for (let gi = 0; gi < groupN; gi++) {
      // Cache the gathered inputs for this group, one per arity slot.
      // wires shape is [arity, groupN]; slot a -> wires[a * groupN + gi].
      let outOff = gi * groupSize * lutDim;
      // Iterate gates in the group (each has its own LUT but shares wires).
      for (let g = 0; g < groupSize; g++) {
        // Initialise work[] = round(sigmoid(luts[gi, g, :])) if hard, else sigmoid(luts).
        for (let k = 0; k < lutDim; k++) {
          let v = sigmoid(luts[outOff + k]);
          if (hard) v = v >= 0.5 ? 1 : 0;
          work[k] = v;
        }
        // Peel inputs one at a time, halving the working LUT each step.
        let curLen = lutDim;
        for (let a = 0; a < arity; a++) {
          const idx = wires[a * groupN + gi];
          const x = prevActs[c * prevSize + idx];
          const half = curLen >> 1;
          for (let k = 0; k < half; k++) {
            work[k] = (1 - x) * work[2 * k] + x * work[2 * k + 1];
          }
          curLen = half;
        }
        acts[c * (groupN * groupSize) + gi * groupSize + g] = work[0];
        outOff += lutDim;
      }
    }
  }
}

/** Run the full circuit forward over a batch of input cases.
 *
 *  ``state.logits`` provides the per-gate LUT logits (flat ``[N, lutDim]``).
 *  ``x`` is ``[caseN, inputBits]``, values in ``[0, 1]``.
 *  ``acts`` is filled out as a sparse list: ``acts[li]`` is ``[caseN, gateN]`` for layer ``li``.
 *
 *  Output of the last layer = ``acts[acts.length - 1]``.
 */
export function circuitForward(
  topology: Topology,
  state: CircuitState,
  x: Float32Array,
  caseN: number,
  hard: boolean,
  arity: number,
): Float32Array[] {
  const lutDim = 1 << arity;
  const acts: Float32Array[] = [];
  // Input layer: optionally mask, then cache.
  const inputN = topology.layerSizes[0][0];
  const layerMaskInput = state.gateMask.subarray(topology.layerStart[0], topology.layerStart[0] + inputN);
  let curActs = new Float32Array(caseN * inputN);
  for (let c = 0; c < caseN; c++) {
    for (let i = 0; i < inputN; i++) curActs[c * inputN + i] = x[c * inputN + i] * layerMaskInput[i];
  }
  acts.push(curActs);

  let prevSize = inputN;
  for (let li = 1; li < topology.layerSizes.length; li++) {
    const [gateN, groupSize] = topology.layerSizes[li];
    const groupN = gateN / groupSize;
    const wires = topology.wires[li - 1];
    const lgtFlat = state.logits.subarray(
      topology.layerStart[li] * lutDim,
      (topology.layerStart[li] + gateN) * lutDim,
    );
    const layerMask = state.gateMask.subarray(
      topology.layerStart[li],
      topology.layerStart[li] + gateN,
    );
    const out = new Float32Array(caseN * gateN);
    runLayer(lgtFlat, curActs, prevSize, wires, arity, caseN, groupN, groupSize, out, hard);
    // Apply gate mask (zero out clamped gates).
    for (let c = 0; c < caseN; c++) {
      for (let g = 0; g < gateN; g++) {
        out[c * gateN + g] *= layerMask[g];
      }
    }
    acts.push(out);
    curActs = out;
    prevSize = gateN;
  }
  return acts;
}

// ---------------------------------------------------------------------------
// Perturbations (the four user actions)
// ---------------------------------------------------------------------------

/** Re-init logits to soft-wire identity + noise; clear damage. */
export function resetCircuit(
  topology: Topology,
  rng: Rng,
  arity: number,
  hiddenDim: number,
  noiseScale: number = 0.1,
): CircuitState {
  return initState(topology, { arity, hiddenDim, noiseScale }, rng);
}

/** Stamp a single gate as damaged. Hidden-layer eligibility enforced —
 *  input/output indices are silently skipped, mirroring
 *  ``DemoSession.damage_gate``. Returns ``true`` iff the gate was actually
 *  clamped (i.e. eligible *and* not already damaged). */
export function damageGate(
  state: CircuitState,
  topology: Topology,
  flatIdx: number,
  lutDim: number,
): boolean {
  const inputN = topology.layerSizes[0][0];
  const outputN = topology.layerSizes[topology.layerSizes.length - 1][0];
  const total = topology.nNodes;
  if (flatIdx < inputN || flatIdx >= total - outputN) return false;
  if (state.gateMask[flatIdx] === 0) return false;
  state.gateMask[flatIdx] = 0;
  for (let k = 0; k < lutDim; k++) state.logits[flatIdx * lutDim + k] = FAULTY_VALUE;
  return true;
}

/** Knock out ``nGates`` random hidden-layer gates. Returns the chosen flat indices. */
export function applyShotgun(
  state: CircuitState,
  topology: Topology,
  nGates: number,
  rng: Rng,
  lutDim: number,
): Int32Array {
  // Hidden-layer eligibility: skip layer 0 (input) and last layer (output).
  const eligible: number[] = [];
  for (let li = 1; li < topology.layerSizes.length - 1; li++) {
    const [gateN] = topology.layerSizes[li];
    const start = topology.layerStart[li];
    for (let g = 0; g < gateN; g++) {
      if (state.gateMask[start + g] === 1) eligible.push(start + g);
    }
  }
  const k = Math.min(nGates, eligible.length);
  // Reservoir sample / Fisher-Yates partial.
  for (let i = 0; i < k; i++) {
    const j = i + ((rng.next() * (eligible.length - i)) | 0);
    const tmp = eligible[i];
    eligible[i] = eligible[j];
    eligible[j] = tmp;
  }
  const chosen = new Int32Array(k);
  for (let i = 0; i < k; i++) {
    chosen[i] = eligible[i];
    state.gateMask[chosen[i]] = 0;
    for (let l = 0; l < lutDim; l++) state.logits[chosen[i] * lutDim + l] = FAULTY_VALUE;
  }
  return chosen;
}

/** Regenerate per-layer wires + topology. Mirrors ``DemoSession.shuffle_wires``:
 *  logits and ``gateMask`` are kept; the topology, attention mask, neighbour
 *  cache, and hidden state are rebuilt (hidden = 0, matching how the JAX side
 *  rebuilds the graph in ``_restart_tmt_generator``). The returned topology
 *  replaces the old one; the caller mutates ``state.hidden`` to zero too. */
export function shuffleWires(
  topology: Topology,
  rng: Rng,
  arity: number,
  hiddenDim: number,
  maxNeighbors: number,
): Topology {
  const newWires: Int32Array[] = [];
  const newShapes: [number, number][] = [];
  for (let li = 1; li < topology.layerSizes.length; li++) {
    const [outN, groupSize] = topology.layerSizes[li];
    const inN = topology.layerSizes[li - 1][0];
    const w = genWires(rng, inN, outN, arity, groupSize);
    newWires.push(w.data);
    newShapes.push(w.shape);
  }
  return buildTopology(topology.layerSizes, newWires, newShapes, { arity, hiddenDim, maxNeighbors });
}

/** Helper: zero out ``state.hidden`` after a wire shuffle, matching the
 *  JAX-side ``build_graph`` reset. */
export function zeroHidden(state: CircuitState): void {
  state.hidden.fill(0);
}
