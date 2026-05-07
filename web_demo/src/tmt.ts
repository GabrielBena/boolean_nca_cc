/**
 * Topology-Masked Transformer (gathered-attention variant) — single-step
 * inference, the residual recompute, and a top-level ``tick`` helper.
 *
 * Mirrors ``export/numpy_oracle.py``'s ``gathered_attention_block`` +
 * ``tmt_step`` + ``compute_residuals``. The Python side is bit-validated
 * against the JAX policy on the deployment checkpoint; this TS port is
 * validated against a recorded reference trajectory in ``main.ts``.
 *
 * All buffers are preallocated in ``Scratch`` so each tick is allocation-free.
 */

import {
  type CircuitState,
  type Topology,
  circuitForward,
} from "./circuit";
import {
  geluApproxInPlace,
  layerNorm,
  linear,
  softmax,
} from "./math";
import type { GatheredAttentionWeights, TMTHeader, TMTWeights } from "./weights";

// ---------------------------------------------------------------------------
// Scratch buffers
// ---------------------------------------------------------------------------

export interface Scratch {
  /** [N, F] concatenated input features. */
  features: Float32Array;
  /** [N, F] post-input-norm. */
  inputNormed: Float32Array;
  /** [N, D] post feature projection (= attention block input). */
  proj: Float32Array;
  /** [N, D] post-attention-block output (= final norm input). */
  postAttn: Float32Array;
  /** [N, D] post-final-norm. */
  finalNormed: Float32Array;
  /** [N, D] within-block intermediates (Q, K, V, attn_out, attn_out_proj). */
  xNorm: Float32Array;
  Q: Float32Array;
  K: Float32Array;
  V: Float32Array;
  scores: Float32Array; // [N, H, K]
  attnW: Float32Array;  // [N, H, K]
  attnOut: Float32Array;
  attnOutProj: Float32Array;
  ffnNorm: Float32Array;
  ffnHidden: Float32Array; // [N, mlp_dim]
  ffnOut: Float32Array;
  /** [N, lut_dim] / [N, hidden_dim] update buffers. */
  dLogit: Float32Array;
  dHidden: Float32Array;
}

export function makeScratch(weights: TMTWeights, nNodes: number): Scratch {
  const D = weights.header.attentionDim;
  const F = weights.header.featureDim;
  const H = weights.header.numHeads;
  const K = weights.header.maxNeighbors;
  const mlp = weights.block.ffnW1.length / D;
  const lutDim = weights.header.logitDim;
  const hiddenDim = weights.header.circuitHiddenDim;
  return {
    features: new Float32Array(nNodes * F),
    inputNormed: new Float32Array(nNodes * F),
    proj: new Float32Array(nNodes * D),
    postAttn: new Float32Array(nNodes * D),
    finalNormed: new Float32Array(nNodes * D),
    xNorm: new Float32Array(nNodes * D),
    Q: new Float32Array(nNodes * D),
    K: new Float32Array(nNodes * D),
    V: new Float32Array(nNodes * D),
    scores: new Float32Array(nNodes * H * K),
    attnW: new Float32Array(nNodes * H * K),
    attnOut: new Float32Array(nNodes * D),
    attnOutProj: new Float32Array(nNodes * D),
    ffnNorm: new Float32Array(nNodes * D),
    ffnHidden: new Float32Array(nNodes * mlp),
    ffnOut: new Float32Array(nNodes * D),
    dLogit: new Float32Array(nNodes * lutDim),
    dHidden: new Float32Array(nNodes * hiddenDim),
  };
}

// ---------------------------------------------------------------------------
// Feature concatenation
// ---------------------------------------------------------------------------

function extractFeatures(
  out: Float32Array,
  state: CircuitState,
  topology: Topology,
  header: TMTHeader,
): void {
  const { logitDim, circuitHiddenDim, useLayerPE, useIntraLayerPE, useNodeLoss, featureDim } = header;
  const N = topology.nNodes;
  for (let n = 0; n < N; n++) {
    const base = n * featureDim;
    let off = 0;
    // logits [logitDim]
    for (let k = 0; k < logitDim; k++) out[base + off + k] = state.logits[n * logitDim + k];
    off += logitDim;
    // hidden [circuitHiddenDim]
    for (let h = 0; h < circuitHiddenDim; h++) out[base + off + h] = state.hidden[n * circuitHiddenDim + h];
    off += circuitHiddenDim;
    if (useIntraLayerPE) {
      for (let h = 0; h < circuitHiddenDim; h++) out[base + off + h] = topology.intraLayerPe[n * circuitHiddenDim + h];
      off += circuitHiddenDim;
    }
    if (useLayerPE) {
      for (let h = 0; h < circuitHiddenDim; h++) out[base + off + h] = topology.layerPe[n * circuitHiddenDim + h];
      off += circuitHiddenDim;
    }
    if (useNodeLoss) {
      out[base + off] = state.loss[n];
      off += 1;
    }
    // Sanity: invariant about feature_dim consistency.
    if (off !== featureDim) {
      throw new Error(`extractFeatures cursor=${off} ≠ featureDim=${featureDim}`);
    }
  }
}

// ---------------------------------------------------------------------------
// Gathered-neighborhood attention block (Pre-LN + ReZero + FFN)
// ---------------------------------------------------------------------------

function gatheredAttentionBlock(
  x: Float32Array,
  block: GatheredAttentionWeights,
  topology: Topology,
  N: number,
  D: number,
  scratch: Scratch,
): void {
  const H = block.numHeads;
  const headDim = D / H;
  const K = topology.maxNeighbors;
  const mlpDim = scratch.ffnHidden.length / N;

  // Pre-LN.
  layerNorm(scratch.xNorm, x, block.attnNormScale, block.attnNormBias, N, D);

  // Q / K / V projections (plain Linear, then reshape to [N, H, headDim] in-place).
  linear(scratch.Q, scratch.xNorm, block.Wq, block.bq, N, D, D);
  linear(scratch.K, scratch.xNorm, block.Wk, block.bk, N, D, D);
  linear(scratch.V, scratch.xNorm, block.Wv, block.bv, N, D, D);

  // QK normalisation: per-head LayerNorm (no bias), over headDim.
  // Treat Q/K as [N*H, headDim] for this call.
  layerNorm(scratch.Q, scratch.Q, block.qLnGamma, null, N * H, headDim);
  layerNorm(scratch.K, scratch.K, block.kLnGamma, null, N * H, headDim);

  // Attention scores: scores[n, h, m] = (Q[n, h, :] · K[ni[n, m], h, :]) / sqrt(headDim).
  const sqrtD = Math.sqrt(headDim);
  for (let n = 0; n < N; n++) {
    const qBase = n * D;
    for (let h = 0; h < H; h++) {
      const qH = qBase + h * headDim;
      for (let m = 0; m < K; m++) {
        const kIdx = topology.neighborIndices[n * K + m];
        const kH = kIdx * D + h * headDim;
        let dot = 0;
        for (let e = 0; e < headDim; e++) dot += scratch.Q[qH + e] * scratch.K[kH + e];
        let s = dot / sqrtD;
        if (!topology.neighborMask[n * K + m]) s = -1e30;
        scratch.scores[n * H * K + h * K + m] = s;
      }
    }
  }

  // Softmax over the K axis (per [n, h] row).
  softmax(scratch.attnW, scratch.scores, N * H, K);

  // Aggregate: attn_out[n, h, e] = Σ_m attnW[n, h, m] * V[ni[n, m], h, e].
  for (let n = 0; n < N; n++) {
    for (let h = 0; h < H; h++) {
      for (let e = 0; e < headDim; e++) {
        let v = 0;
        for (let m = 0; m < K; m++) {
          const kIdx = topology.neighborIndices[n * K + m];
          v += scratch.attnW[n * H * K + h * K + m] * scratch.V[kIdx * D + h * headDim + e];
        }
        scratch.attnOut[n * D + h * headDim + e] = v;
      }
    }
  }

  // Output projection: attn_out @ Wo + bo.
  linear(scratch.attnOutProj, scratch.attnOut, block.Wo, block.bo, N, D, D);

  // ReZero residual: x += α_attn * attn_out_proj.
  const aAttn = block.attnRezero;
  for (let i = 0; i < N * D; i++) x[i] += aAttn * scratch.attnOutProj[i];

  // FFN with internal Pre-LN.
  layerNorm(scratch.ffnNorm, x, block.ffnLnScale, block.ffnLnBias, N, D);
  linear(scratch.ffnHidden, scratch.ffnNorm, block.ffnW1, block.ffnB1, N, D, mlpDim);
  geluApproxInPlace(scratch.ffnHidden);
  linear(scratch.ffnOut, scratch.ffnHidden, block.ffnW2, block.ffnB2, N, mlpDim, D);

  // ReZero residual: x += α_ffn * ffn_out.
  const aFfn = block.ffnRezero;
  for (let i = 0; i < N * D; i++) x[i] += aFfn * scratch.ffnOut[i];
}

// ---------------------------------------------------------------------------
// Single TMT step (logits + hidden mutated in place)
// ---------------------------------------------------------------------------

export function tmtStep(
  state: CircuitState,
  topology: Topology,
  weights: TMTWeights,
  scratch: Scratch,
): void {
  const { attentionDim: D, featureDim: F, logitDim, circuitHiddenDim } = weights.header;
  const N = topology.nNodes;

  extractFeatures(scratch.features, state, topology, weights.header);
  layerNorm(scratch.inputNormed, scratch.features, weights.inputNormScale, weights.inputNormBias, N, F);
  linear(scratch.proj, scratch.inputNormed, weights.featureProjW, weights.featureProjB, N, F, D);

  gatheredAttentionBlock(scratch.proj, weights.block, topology, N, D, scratch);

  layerNorm(scratch.finalNormed, scratch.proj, weights.finalNormScale, weights.finalNormBias, N, D);
  linear(scratch.dLogit, scratch.finalNormed, weights.logitProjW, weights.logitProjB, N, D, logitDim);
  linear(scratch.dHidden, scratch.finalNormed, weights.hiddenProjW, weights.hiddenProjB, N, D, circuitHiddenDim);

  // Knockout: zero updates for clamped gates.
  for (let n = 0; n < N; n++) {
    if (state.gateMask[n] !== 1) {
      for (let k = 0; k < logitDim; k++) scratch.dLogit[n * logitDim + k] = 0;
      for (let h = 0; h < circuitHiddenDim; h++) scratch.dHidden[n * circuitHiddenDim + h] = 0;
    }
  }

  // ReZero-gated residual updates on logits + hidden.
  const aLogit = weights.logitRezero;
  const aHidden = weights.hiddenRezero;
  for (let i = 0; i < N * logitDim; i++) state.logits[i] += aLogit * scratch.dLogit[i];
  for (let i = 0; i < N * circuitHiddenDim; i++) state.hidden[i] += aHidden * scratch.dHidden[i];
}

// ---------------------------------------------------------------------------
// Residual recompute + accuracy aux
// ---------------------------------------------------------------------------

export interface StepAux {
  /** ``[caseN, outputBits]`` soft predictions from the last layer. */
  predSoft: Float32Array;
  /** ``[caseN, outputBits]`` hard predictions ({0, 1}). */
  predHard: Float32Array;
  /** Mean over (case, output_bit) of ``|y - pred_soft| < 0.5``. */
  softAccuracy: number;
  /** Mean over (case, output_bit) of ``pred_hard == y``. */
  hardAccuracy: number;
}

/** Run circuit forward (soft + hard) and stamp output rows of ``state.loss`` with
 *  ``mean_batch |y - pred_soft|`` per output bit. Returns the aux bag. */
export function computeResiduals(
  state: CircuitState,
  topology: Topology,
  xData: Float32Array,
  yData: Float32Array,
  caseN: number,
  arity: number,
): StepAux {
  const softActs = circuitForward(topology, state, xData, caseN, false, arity);
  const hardActs = circuitForward(topology, state, xData, caseN, true, arity);
  const predSoft = softActs[softActs.length - 1];
  const predHard = hardActs[hardActs.length - 1];

  const outputN = topology.layerSizes[topology.layerSizes.length - 1][0];
  // Refresh state.loss: zero everything, then write |residual|-mean per output bit.
  state.loss.fill(0);
  let softMatches = 0;
  let hardMatches = 0;
  const total = caseN * outputN;
  for (let bit = 0; bit < outputN; bit++) {
    let absSum = 0;
    for (let c = 0; c < caseN; c++) {
      const off = c * outputN + bit;
      const ps = predSoft[off];
      const ph = predHard[off];
      const y = yData[off];
      absSum += Math.abs(ps - y);
      if (Math.abs(ps - y) < 0.5) softMatches++;
      if (ph === y) hardMatches++;
    }
    state.loss[topology.outputStart + bit] = absSum / caseN;
  }
  return {
    predSoft,
    predHard,
    softAccuracy: softMatches / total,
    hardAccuracy: hardMatches / total,
  };
}

// ---------------------------------------------------------------------------
// Top-level helper: one full tick (model step + residual refresh)
// ---------------------------------------------------------------------------

/** One full demo tick: TMT step + residual recompute. Returns the aux from the
 *  fresh forward pass, ready to drive plots / labels in the UI. */
export function tick(
  state: CircuitState,
  topology: Topology,
  weights: TMTWeights,
  scratch: Scratch,
  xData: Float32Array,
  yData: Float32Array,
  caseN: number,
  arity: number,
): StepAux {
  tmtStep(state, topology, weights, scratch);
  return computeResiduals(state, topology, xData, yData, caseN, arity);
}
