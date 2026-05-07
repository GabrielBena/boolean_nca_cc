/**
 * Math primitives shared by ``circuit.ts`` and ``tmt.ts``.
 *
 * All operations work on flat ``Float32Array`` storage with explicit
 * shape arguments — same convention as the NumPy oracle this is
 * being ported from. Read ``export/numpy_oracle.py`` first if you
 * need to reason about correctness; this file is the line-by-line
 * TS counterpart of those primitives.
 *
 * Performance note: the demo does ≤ 264 nodes, attention dim ≤ 128,
 * a few k-ops per tick. A naïve loop-based JS implementation hits
 * 30 FPS comfortably; we don't bother with WebGL or SIMD here.
 */

/** Affine map ``out[m, n] = sum_k x[m, k] * w[k, n] + b[n]`` with optional bias.
 *
 *  ``x``  flat shape ``[M, K]``  ``w`` flat shape ``[K, N]``  ``b`` shape ``[N]`` or null.
 *  ``out`` is filled in place; the caller owns the buffer. */
export function linear(
  out: Float32Array,
  x: Float32Array,
  w: Float32Array,
  b: Float32Array | null,
  M: number,
  K: number,
  N: number,
): void {
  for (let m = 0; m < M; m++) {
    for (let n = 0; n < N; n++) {
      let acc = b ? b[n] : 0;
      const xRow = m * K;
      for (let k = 0; k < K; k++) {
        acc += x[xRow + k] * w[k * N + n];
      }
      out[m * N + n] = acc;
    }
  }
}

/** LayerNorm over the last (innermost) axis.
 *
 *  ``x`` is treated as ``[batch, lastDim]``; ``gamma`` / ``beta`` have shape
 *  ``[lastDim]``. With ``beta = null`` we treat β as zero (matches the
 *  ``use_bias=False`` Q/K-norm in the gathered attention path).
 *  Operates out-of-place: caller passes a separate output buffer. */
export function layerNorm(
  out: Float32Array,
  x: Float32Array,
  gamma: Float32Array,
  beta: Float32Array | null,
  batch: number,
  lastDim: number,
  eps: number = 1e-6,
): void {
  for (let b = 0; b < batch; b++) {
    const off = b * lastDim;
    let mean = 0;
    for (let i = 0; i < lastDim; i++) mean += x[off + i];
    mean /= lastDim;
    let varSum = 0;
    for (let i = 0; i < lastDim; i++) {
      const d = x[off + i] - mean;
      varSum += d * d;
    }
    const invStd = 1 / Math.sqrt(varSum / lastDim + eps);
    for (let i = 0; i < lastDim; i++) {
      const z = (x[off + i] - mean) * invStd * gamma[i];
      out[off + i] = beta ? z + beta[i] : z;
    }
  }
}

/** Numerically-stable softmax along the last axis. ``x`` is ``[batch, lastDim]``. */
export function softmax(out: Float32Array, x: Float32Array, batch: number, lastDim: number): void {
  for (let b = 0; b < batch; b++) {
    const off = b * lastDim;
    let maxV = -Infinity;
    for (let i = 0; i < lastDim; i++) {
      const v = x[off + i];
      if (v > maxV) maxV = v;
    }
    let sum = 0;
    for (let i = 0; i < lastDim; i++) {
      const e = Math.exp(x[off + i] - maxV);
      out[off + i] = e;
      sum += e;
    }
    const inv = 1 / sum;
    for (let i = 0; i < lastDim; i++) out[off + i] *= inv;
  }
}

/** Numerically-stable sigmoid (per-element). */
export function sigmoid(x: number): number {
  if (x >= 0) {
    return 1 / (1 + Math.exp(-x));
  }
  const e = Math.exp(x);
  return e / (1 + e);
}

/** Tanh-based GELU approximation, matching ``jax.nn.gelu(approximate=True)``
 *  / ``flax.nnx.gelu`` (Flax default, validated in the parity tests). */
const GELU_K = Math.sqrt(2 / Math.PI);
export function geluApprox(x: number): number {
  return 0.5 * x * (1 + Math.tanh(GELU_K * (x + 0.044715 * x * x * x)));
}

/** Apply ``geluApprox`` element-wise to a flat array (in place). */
export function geluApproxInPlace(arr: Float32Array): void {
  for (let i = 0; i < arr.length; i++) arr[i] = geluApprox(arr[i]);
}

/** Sinusoidal positional encoding, mirroring
 *  ``utils.positional_encoding.get_positional_encoding``. ``positions`` is
 *  a length-``N`` array of floats; ``out`` is filled with shape ``[N, dim]``.
 *  ``dim`` must be even. */
export function sinusoidalPE(
  out: Float32Array,
  positions: Float32Array,
  dim: number,
  maxVal: number = 10000,
): void {
  if ((dim & 1) !== 0) {
    throw new Error(`sinusoidalPE: dim must be even, got ${dim}`);
  }
  const N = positions.length;
  const half = dim >>> 1;
  // Precompute the exp(... -log(maxVal)/dim) frequencies once.
  const factor = -Math.log(maxVal) / dim;
  const div = new Float32Array(half);
  for (let k = 0; k < half; k++) div[k] = Math.exp(2 * k * factor);
  for (let n = 0; n < N; n++) {
    const p = positions[n];
    const off = n * dim;
    for (let k = 0; k < half; k++) {
      const arg = p * div[k];
      out[off + 2 * k] = Math.sin(arg);
      out[off + 2 * k + 1] = Math.cos(arg);
    }
  }
}

/** Stable argsort of ``~mask`` along the last axis with ``kind='stable'`` —
 *  the same primitive ``build_neighbor_indices`` uses to push real
 *  neighbours to the front of each row. ``mask`` is ``[N, N]`` row-major,
 *  output ``[N, N]`` row-major Int32Array of column indices. */
export function stableArgsortRowsByNotMask(mask: Uint8Array, N: number): Int32Array {
  const out = new Int32Array(N * N);
  for (let r = 0; r < N; r++) {
    // Two-pass: first the True indices in input order, then the False indices.
    let writeIdx = 0;
    for (let c = 0; c < N; c++) {
      if (mask[r * N + c]) out[r * N + writeIdx++] = c;
    }
    for (let c = 0; c < N; c++) {
      if (!mask[r * N + c]) out[r * N + writeIdx++] = c;
    }
  }
  return out;
}
