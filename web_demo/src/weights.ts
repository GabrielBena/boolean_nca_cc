/**
 * Loads quantised TMT weights from the JSON emitted by
 * ``export/extract_weights.py`` and dequantises them into
 * ``Float32Array`` buffers ready for ``tmt.ts``.
 *
 * The on-disk schema is version-tagged (``schema_version`` in the
 * header). Currently only ``"gathered_attention"`` is supported on
 * read — that's the architecture our deployment checkpoint uses.
 */

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

export interface TMTHeader {
  schemaVersion: number;
  modelKind: "gathered_attention";
  arity: number;
  circuitHiddenDim: number;
  attentionDim: number;
  numHeads: number;
  numAttnLayers: number;
  useLayerPE: boolean;
  useIntraLayerPE: boolean;
  useNodeLoss: boolean;
  /** DAG-distance PE (wire-dependent). Absent in pre-dist_pe exports → false. */
  useDistPe: boolean;
  maxNeighbors: number;
  logitDim: number;
  featureDim: number;
  useGeluApprox: boolean;
  sourceRunId: string | null;
  tensorDtype: "fp16" | "uint8";
}

export interface GatheredAttentionWeights {
  attnNormScale: Float32Array;
  attnNormBias: Float32Array;
  Wq: Float32Array;
  bq: Float32Array;
  Wk: Float32Array;
  bk: Float32Array;
  Wv: Float32Array;
  bv: Float32Array;
  Wo: Float32Array;
  bo: Float32Array;
  qLnGamma: Float32Array;
  kLnGamma: Float32Array;
  ffnLnScale: Float32Array;
  ffnLnBias: Float32Array;
  ffnW1: Float32Array;
  ffnB1: Float32Array;
  ffnW2: Float32Array;
  ffnB2: Float32Array;
  attnRezero: number;
  ffnRezero: number;
  numHeads: number;
}

export interface TMTWeights {
  header: TMTHeader;
  inputNormScale: Float32Array;
  inputNormBias: Float32Array;
  featureProjW: Float32Array;
  featureProjB: Float32Array;
  block: GatheredAttentionWeights;
  finalNormScale: Float32Array;
  finalNormBias: Float32Array;
  logitProjW: Float32Array;
  logitProjB: Float32Array;
  hiddenProjW: Float32Array;
  hiddenProjB: Float32Array;
  logitRezero: number;
  hiddenRezero: number;
}

// ---------------------------------------------------------------------------
// Decoders
// ---------------------------------------------------------------------------

function base64ToBytes(b64: string): Uint8Array {
  const bin = atob(b64);
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}

/** IEEE-754 half-precision -> single-precision, manual bit unpack.
 *
 *  Falls back to a hand-coded converter rather than relying on
 *  ``DataView.getFloat16`` which is too new (Chrome ≥ 122). The
 *  per-element cost is irrelevant: the whole 162 K-param weight
 *  trove decodes in well under 50 ms on a typical laptop. */
function fp16ToFp32(h: number): number {
  const sign = (h & 0x8000) >>> 15;
  const exp = (h & 0x7c00) >>> 10;
  const frac = h & 0x03ff;
  if (exp === 0) {
    if (frac === 0) return sign ? -0 : 0;
    // Subnormal: 2^-14 * frac/1024
    const v = frac * Math.pow(2, -24);
    return sign ? -v : v;
  }
  if (exp === 31) {
    if (frac === 0) return sign ? -Infinity : Infinity;
    return NaN;
  }
  const v = (1 + frac / 1024) * Math.pow(2, exp - 15);
  return sign ? -v : v;
}

interface TensorEntry {
  shape: number[];
  dtype: "fp16" | "uint8";
  data_b64: string;
  scale?: number;
  zero_point?: number;
}

function dequantizeTensor(entry: TensorEntry): Float32Array {
  const bytes = base64ToBytes(entry.data_b64);
  const totalElements = entry.shape.reduce((a, b) => a * b, 1);

  if (entry.dtype === "fp16") {
    if (bytes.length !== totalElements * 2) {
      throw new Error(
        `fp16 byte length mismatch: have ${bytes.length}, expected ${totalElements * 2}`,
      );
    }
    // Reinterpret the byte buffer as Uint16 and unpack each fp16 value.
    const u16 = new Uint16Array(bytes.buffer, bytes.byteOffset, totalElements);
    const out = new Float32Array(totalElements);
    for (let i = 0; i < totalElements; i++) out[i] = fp16ToFp32(u16[i]);
    return out;
  }

  if (entry.dtype === "uint8") {
    if (entry.scale === undefined || entry.zero_point === undefined) {
      throw new Error(`uint8 tensor missing scale or zero_point`);
    }
    if (bytes.length !== totalElements) {
      throw new Error(
        `uint8 byte length mismatch: have ${bytes.length}, expected ${totalElements}`,
      );
    }
    const scale = entry.scale;
    const zp = entry.zero_point;
    const out = new Float32Array(totalElements);
    for (let i = 0; i < totalElements; i++) out[i] = (bytes[i] - zp) * scale;
    return out;
  }

  throw new Error(`Unknown tensor dtype: ${(entry as { dtype: string }).dtype}`);
}

// ---------------------------------------------------------------------------
// Top-level loader
// ---------------------------------------------------------------------------

interface RawDoc {
  header: {
    schema_version: number;
    model_kind: string;
    arity: number;
    circuit_hidden_dim: number;
    attention_dim: number;
    num_heads: number;
    num_attn_layers: number;
    use_layer_PE: boolean;
    use_intra_layer_PE: boolean;
    use_node_loss: boolean;
    use_dist_pe?: boolean;
    max_neighbors: number;
    logit_dim: number;
    feature_dim: number;
    use_gelu_approx: boolean;
    source_run_id: string | null;
    tensor_dtype: string;
  };
  scalars: Record<string, number>;
  tensors: Record<string, TensorEntry>;
}

function parseHeader(raw: RawDoc["header"]): TMTHeader {
  if (raw.model_kind !== "gathered_attention") {
    throw new Error(
      `Unsupported model_kind '${raw.model_kind}'; only gathered_attention is wired up.`,
    );
  }
  if (raw.tensor_dtype !== "fp16" && raw.tensor_dtype !== "uint8") {
    throw new Error(`Unsupported tensor_dtype ${raw.tensor_dtype}`);
  }
  return {
    schemaVersion: raw.schema_version,
    modelKind: "gathered_attention",
    arity: raw.arity,
    circuitHiddenDim: raw.circuit_hidden_dim,
    attentionDim: raw.attention_dim,
    numHeads: raw.num_heads,
    numAttnLayers: raw.num_attn_layers,
    useLayerPE: raw.use_layer_PE,
    useIntraLayerPE: raw.use_intra_layer_PE,
    useNodeLoss: raw.use_node_loss,
    useDistPe: raw.use_dist_pe ?? false,
    maxNeighbors: raw.max_neighbors,
    logitDim: raw.logit_dim,
    featureDim: raw.feature_dim,
    useGeluApprox: raw.use_gelu_approx,
    sourceRunId: raw.source_run_id,
    tensorDtype: raw.tensor_dtype,
  };
}

function buildBlock(
  tensors: Record<string, TensorEntry>,
  scalars: Record<string, number>,
  numHeads: number,
): GatheredAttentionWeights {
  const dq = (k: string): Float32Array => {
    const entry = tensors[k];
    if (!entry) throw new Error(`Missing tensor ${k} in weights JSON`);
    return dequantizeTensor(entry);
  };
  const sc = (k: string): number => {
    const v = scalars[k];
    if (v === undefined) throw new Error(`Missing scalar ${k} in weights JSON`);
    return v;
  };
  return {
    attnNormScale: dq("block.attn_norm.scale"),
    attnNormBias: dq("block.attn_norm.bias"),
    Wq: dq("block.Wq"),
    bq: dq("block.bq"),
    Wk: dq("block.Wk"),
    bk: dq("block.bk"),
    Wv: dq("block.Wv"),
    bv: dq("block.bv"),
    Wo: dq("block.Wo"),
    bo: dq("block.bo"),
    qLnGamma: dq("block.q_ln_gamma"),
    kLnGamma: dq("block.k_ln_gamma"),
    ffnLnScale: dq("block.ffn_ln.scale"),
    ffnLnBias: dq("block.ffn_ln.bias"),
    ffnW1: dq("block.ffn_W1"),
    ffnB1: dq("block.ffn_b1"),
    ffnW2: dq("block.ffn_W2"),
    ffnB2: dq("block.ffn_b2"),
    attnRezero: sc("block.attn_rezero"),
    ffnRezero: sc("block.ffn_rezero"),
    numHeads,
  };
}

/** Parse a JSON object (already-decoded) into a fully populated ``TMTWeights``. */
export function parseWeightsFromJson(raw: unknown): TMTWeights {
  const doc = raw as RawDoc;
  const header = parseHeader(doc.header);
  const tensors = doc.tensors;
  const scalars = doc.scalars;
  const dq = (k: string): Float32Array => dequantizeTensor(tensors[k]);
  const sc = (k: string): number => {
    const v = scalars[k];
    if (v === undefined) throw new Error(`Missing scalar ${k} in weights JSON`);
    return v;
  };
  return {
    header,
    inputNormScale: dq("input_norm.scale"),
    inputNormBias: dq("input_norm.bias"),
    featureProjW: dq("feature_proj.kernel"),
    featureProjB: dq("feature_proj.bias"),
    block: buildBlock(tensors, scalars, header.numHeads),
    finalNormScale: dq("final_norm.scale"),
    finalNormBias: dq("final_norm.bias"),
    logitProjW: dq("logit_proj.kernel"),
    logitProjB: dq("logit_proj.bias"),
    hiddenProjW: dq("hidden_proj.kernel"),
    hiddenProjB: dq("hidden_proj.bias"),
    logitRezero: sc("logit_rezero"),
    hiddenRezero: sc("hidden_rezero"),
  };
}

/** Fetch the JSON at ``url``, dequantise every tensor, return a fully
 *  populated ``TMTWeights``. */
export async function loadWeights(url: string): Promise<TMTWeights> {
  const resp = await fetch(url);
  if (!resp.ok) {
    throw new Error(`Failed to fetch weights at ${url}: ${resp.status} ${resp.statusText}`);
  }
  return parseWeightsFromJson(await resp.json());
}

/** Parameter count, useful for log lines / sanity checks. */
export function countParameters(w: TMTWeights): number {
  const arrays: Float32Array[] = [
    w.inputNormScale,
    w.inputNormBias,
    w.featureProjW,
    w.featureProjB,
    w.block.attnNormScale,
    w.block.attnNormBias,
    w.block.Wq,
    w.block.bq,
    w.block.Wk,
    w.block.bk,
    w.block.Wv,
    w.block.bv,
    w.block.Wo,
    w.block.bo,
    w.block.qLnGamma,
    w.block.kLnGamma,
    w.block.ffnLnScale,
    w.block.ffnLnBias,
    w.block.ffnW1,
    w.block.ffnB1,
    w.block.ffnW2,
    w.block.ffnB2,
    w.finalNormScale,
    w.finalNormBias,
    w.logitProjW,
    w.logitProjB,
    w.hiddenProjW,
    w.hiddenProjB,
  ];
  return arrays.reduce((acc, a) => acc + a.length, 0) + 4; // 4 ReZero scalars
}
