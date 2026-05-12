"""
Extract a frozen TMT checkpoint into the oracle's ``TMTWeights`` schema.

Mirrors the gallery-load path used by ``DemoSession.load_model_from_wandb``:

    load_config_from_wandb(run_id, entity, project, ...)
        -> (loaded_cfg, checkpoint_path, run_id)
    instantiate_model_from_config(loaded_cfg, seed=0, n_node=...)
    load_checkpoint_with_compatibility(checkpoint_path)
    nnx.update(model, loaded["model"])

…then walks the loaded ``nnx`` model and packs each parameter into a
``GatheredAttentionWeights`` (or ``AttentionBlockWeights``) instance.

The default coordinates point at the v1 deployment run for the demo:
``reverse / random_damage`` — paper figure 4 random-topology reverse,
``run_id = 1u5ssulx``. Override on the CLI if you want to extract a
different run for sanity checks.
"""

from __future__ import annotations

import argparse
import base64
import copy
import json
import os
from typing import Any

import jax
import numpy as np
from flax import nnx
from omegaconf import DictConfig

from boolean_nca_cc.circuits.model import gen_circuit, generate_layer_sizes
from boolean_nca_cc.training.checkpointing import (
    instantiate_model_from_config,
    load_checkpoint_with_compatibility,
    load_config_from_wandb,
    migrate_checkpoint_state,
)
from boolean_nca_cc.utils.graph_builder import build_graph

from .numpy_oracle import (
    AttentionBlockWeights,
    GatheredAttentionWeights,
    MHAWeights,
    TMTConfig,
    TMTWeights,
)

# Schema version for the on-disk JSON. Bump if the field layout changes
# in a non-backwards-compatible way.
WEIGHTS_SCHEMA_VERSION = 1

# Defaults from configs/demo_models.yaml — paper figure 4 random reverse.
DEFAULT_RUN_ID = "1u5ssulx"
DEFAULT_ENTITY = "gbena"
DEFAULT_PROJECT = "boolean-nca-cc"
DEFAULT_PREFER_METRIC = "eval_damaged_out_test_hard_accuracy"
DEFAULT_DOWNLOAD_DIR = "saves"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def to_np(x: Any) -> np.ndarray:
    """Convert a JAX array (or ``nnx.Param`` value) to a contiguous float32 numpy array."""
    arr = np.asarray(x).astype(np.float32, copy=False)
    return np.ascontiguousarray(arr)


def _scalar(p: Any) -> float:
    """Pull a Python float out of an ``nnx.Param`` whose value is a [1] array (ReZero)."""
    return float(np.asarray(p.value).reshape(-1)[0])


# ---------------------------------------------------------------------------
# W&B → loaded model
# ---------------------------------------------------------------------------


def load_run(
    *,
    run_id: str = DEFAULT_RUN_ID,
    entity: str = DEFAULT_ENTITY,
    project: str = DEFAULT_PROJECT,
    prefer_metric: str = DEFAULT_PREFER_METRIC,
    download_dir: str = DEFAULT_DOWNLOAD_DIR,
    seed: int = 0,
) -> tuple[nnx.Module, DictConfig, str]:
    """Load a frozen run from W&B and return ``(model, cfg, checkpoint_path)``.

    Mirrors the call chain in ``DemoSession.load_model_from_wandb`` +
    ``DemoSession.bootstrap`` so the resulting model is byte-identical
    to what the live demo deploys.
    """
    loaded_cfg, checkpoint_path, _ = load_config_from_wandb(
        run_id=run_id,
        entity=entity,
        project=project,
        download_dir=download_dir,
        filename=f"best_model_{prefer_metric}",
        use_cache=True,
        force_download=False,
        select_by_best_metric=False,
    )

    # Build a dummy graph just to compute n_node — required for
    # ``CircuitSelfAttention``, harmlessly ignored by gathered.
    layer_sizes = generate_layer_sizes(
        int(loaded_cfg.circuit.input_bits),
        int(loaded_cfg.circuit.output_bits),
        int(loaded_cfg.circuit.arity),
        layer_n=int(loaded_cfg.circuit.num_layers),
        width_factor=float(loaded_cfg.circuit.get("width_factor", 2)),
    )
    test_key = jax.random.PRNGKey(int(loaded_cfg.get("test_seed", 42)))
    wires_key, logits_key = jax.random.split(test_key)
    wires, logits = gen_circuit(
        wires_key,
        logits_key,
        layer_sizes,
        arity=int(loaded_cfg.circuit.arity),
        noise_scale=float(loaded_cfg.pool.noise_scale),
    )
    graph = build_graph(
        wires=wires,
        logits=logits,
        input_n=int(loaded_cfg.circuit.input_bits),
        arity=int(loaded_cfg.circuit.arity),
        circuit_hidden_dim=int(loaded_cfg.model.circuit_hidden_dim),
    )
    n_node = int(graph.n_node[0])

    init_model = instantiate_model_from_config(loaded_cfg, seed=seed, n_node=n_node)
    model = copy.deepcopy(init_model)
    loaded = load_checkpoint_with_compatibility(checkpoint_path)
    # Backward-compat for the CircuitGatheredAttention attn_block refactor:
    # pre-refactor checkpoints stored attention submodules at the top level,
    # the new module layout nests them inside `attn_block`. No-op otherwise.
    loaded["model"] = migrate_checkpoint_state(model, loaded["model"])
    nnx.update(model, loaded["model"])
    return model, loaded_cfg, checkpoint_path


# ---------------------------------------------------------------------------
# Config → TMTConfig
# ---------------------------------------------------------------------------


def cfg_to_tmt_config(cfg: DictConfig) -> TMTConfig:
    """Pull the inference-time architectural flags out of the Hydra config."""
    target = str(cfg.model.get("_target_", "")).lower()
    type_field = str(cfg.model.get("type", "")).lower()
    if "gathered" in target or type_field == "gathered_attention":
        kind = "gathered_attention"
    elif "self_attention" in target or type_field == "self_attention":
        kind = "self_attention"
    else:
        raise ValueError(f"Unsupported model in cfg.model._target_={target!r}")

    attention_dim = int(cfg.model.attention_dim)
    mlp_dim = cfg.model.get("mlp_dim", None)
    if mlp_dim is None:
        mlp_dim = int(cfg.model.get("mlp_dim_multiplier", 2)) * attention_dim
    return TMTConfig(
        arity=int(cfg.circuit.arity),
        circuit_hidden_dim=int(cfg.model.circuit_hidden_dim),
        attention_dim=attention_dim,
        num_heads=int(cfg.model.num_heads),
        num_attn_layers=int(cfg.model.get("num_self_attn_layers", 1)),
        use_layer_PE=bool(cfg.model.get("use_layer_PE", True)),
        use_intra_layer_PE=bool(cfg.model.get("use_intra_layer_PE", False)),
        use_node_loss=bool(cfg.model.get("use_node_loss", False)),
        model_kind=kind,
        max_neighbors=int(cfg.model.get("max_neighbors", 16)),
    )


# ---------------------------------------------------------------------------
# Model → oracle weights
# ---------------------------------------------------------------------------


def extract_gathered_block(model: nnx.Module) -> GatheredAttentionWeights:
    """Pack ``CircuitGatheredAttention``'s attention block into ``GatheredAttentionWeights``.

    Attribute mapping (LHS = nnx model, RHS = oracle field). All attention
    submodules live on ``model.attn_block`` after the GatheredAttentionBlock
    refactor — legacy checkpoints are re-nested by ``migrate_checkpoint_state``
    in :func:`load_run` so this path is uniform.

      attn_block.attn_norm        -> attn_norm
      attn_block.query_proj       -> Wq, bq
      attn_block.key_proj         -> Wk, bk
      attn_block.value_proj       -> Wv, bv
      attn_block.output_proj      -> Wo, bo
      attn_block.query_ln (γ only)-> q_ln_gamma
      attn_block.key_ln   (γ only)-> k_ln_gamma
      attn_block.ffn[0] (LN)      -> ffn_ln
      attn_block.ffn[1] (Linear)  -> ffn_W1, ffn_b1
      attn_block.ffn[3] (Linear)  -> ffn_W2, ffn_b2
      attn_block.attn_rezero      -> attn_rezero (scalar)
      attn_block.ffn_rezero       -> ffn_rezero  (scalar)
    """
    block = model.attn_block
    ffn_layers = list(block.ffn.layers)  # nnx.Sequential exposes .layers
    return GatheredAttentionWeights(
        attn_norm=(to_np(block.attn_norm.scale.value), to_np(block.attn_norm.bias.value)),
        Wq=to_np(block.query_proj.kernel.value),
        bq=to_np(block.query_proj.bias.value),
        Wk=to_np(block.key_proj.kernel.value),
        bk=to_np(block.key_proj.bias.value),
        Wv=to_np(block.value_proj.kernel.value),
        bv=to_np(block.value_proj.bias.value),
        Wo=to_np(block.output_proj.kernel.value),
        bo=to_np(block.output_proj.bias.value),
        q_ln_gamma=to_np(block.query_ln.scale.value),
        k_ln_gamma=to_np(block.key_ln.scale.value),
        ffn_ln=(to_np(ffn_layers[0].scale.value), to_np(ffn_layers[0].bias.value)),
        ffn_W1=to_np(ffn_layers[1].kernel.value),
        ffn_b1=to_np(ffn_layers[1].bias.value),
        ffn_W2=to_np(ffn_layers[3].kernel.value),
        ffn_b2=to_np(ffn_layers[3].bias.value),
        attn_rezero=_scalar(block.attn_rezero.scale),
        ffn_rezero=_scalar(block.ffn_rezero.scale),
        num_heads=int(model.num_heads),
        use_gelu_approx=True,  # match nnx.gelu / jax.nn.gelu default
    )


def extract_self_attention_blocks(model: nnx.Module) -> list[AttentionBlockWeights]:
    """Pack ``CircuitSelfAttention``'s ``attention_layers`` into ``AttentionBlockWeights``.

    Validated only at the architectural level until we deploy a self-attention
    checkpoint. Today's deployment uses gathered attention; this exists as
    a fallback for parity-checking older runs.
    """
    blocks: list[AttentionBlockWeights] = []
    for layer in model.attention_layers:
        attn = layer.attention  # nnx.MultiHeadAttention
        ffn_layers = list(layer.ffn.layers)
        mha = MHAWeights(
            Wq=to_np(attn.query.kernel.value),
            bq=to_np(attn.query.bias.value),
            Wk=to_np(attn.key.kernel.value),
            bk=to_np(attn.key.bias.value),
            Wv=to_np(attn.value.kernel.value),
            bv=to_np(attn.value.bias.value),
            Wo=to_np(attn.out.kernel.value),
            bo=to_np(attn.out.bias.value),
            q_ln_gamma=to_np(attn.query_ln.scale.value),
            q_ln_beta=to_np(attn.query_ln.bias.value),
            k_ln_gamma=to_np(attn.key_ln.scale.value),
            k_ln_beta=to_np(attn.key_ln.bias.value),
        )
        blocks.append(
            AttentionBlockWeights(
                ln_q=(to_np(layer.layer_norm_q.scale.value), to_np(layer.layer_norm_q.bias.value)),
                ln_kv=(
                    to_np(layer.layer_norm_kv.scale.value),
                    to_np(layer.layer_norm_kv.bias.value),
                ),
                mha=mha,
                ffn_ln=(to_np(ffn_layers[0].scale.value), to_np(ffn_layers[0].bias.value)),
                ffn_W1=to_np(ffn_layers[1].kernel.value),
                ffn_b1=to_np(ffn_layers[1].bias.value),
                ffn_W2=to_np(ffn_layers[3].kernel.value),
                ffn_b2=to_np(ffn_layers[3].bias.value),
                attn_rezero=_scalar(layer.attn_rezero.scale),
                ffn_rezero=_scalar(layer.ffn_rezero.scale),
                use_gelu_approx=True,
            )
        )
    return blocks


def extract_weights(model: nnx.Module, cfg: DictConfig) -> TMTWeights:
    """Walk a loaded ``nnx`` model and produce the oracle's ``TMTWeights``."""
    tmt_cfg = cfg_to_tmt_config(cfg)
    if tmt_cfg.model_kind == "gathered_attention":
        blocks: list[AttentionBlockWeights] | GatheredAttentionWeights = extract_gathered_block(
            model
        )
    else:
        blocks = extract_self_attention_blocks(model)

    return TMTWeights(
        config=tmt_cfg,
        input_norm=(to_np(model.input_norm.scale.value), to_np(model.input_norm.bias.value)),
        feature_proj_W=to_np(model.feature_proj.kernel.value),
        feature_proj_b=to_np(model.feature_proj.bias.value),
        blocks=blocks,
        final_norm=(to_np(model.final_norm.scale.value), to_np(model.final_norm.bias.value)),
        logit_proj_W=to_np(model.logit_proj.kernel.value),
        logit_proj_b=to_np(model.logit_proj.bias.value),
        hidden_proj_W=to_np(model.hidden_proj.kernel.value),
        hidden_proj_b=to_np(model.hidden_proj.bias.value),
        logit_rezero=_scalar(model.logit_rezero.scale),
        hidden_rezero=_scalar(model.hidden_rezero.scale),
    )


# ---------------------------------------------------------------------------
# CLI: smoke-load + summary
# ---------------------------------------------------------------------------


def _summarise(weights: TMTWeights) -> str:
    cfg = weights.config
    if isinstance(weights.blocks, list):
        n_params_attn = sum(_count_params(b) for b in weights.blocks)
        kind_label = f"self_attention × {len(weights.blocks)}"
    else:
        n_params_attn = _count_params(weights.blocks)
        kind_label = "gathered_attention × 1"
    n_params_total = (
        n_params_attn
        + weights.feature_proj_W.size
        + weights.feature_proj_b.size
        + weights.input_norm[0].size
        + weights.input_norm[1].size
        + weights.final_norm[0].size
        + weights.final_norm[1].size
        + weights.logit_proj_W.size
        + weights.logit_proj_b.size
        + weights.hidden_proj_W.size
        + weights.hidden_proj_b.size
        + 2  # rezero scalars
    )
    return (
        f"  kind:               {kind_label}\n"
        f"  arity:              {cfg.arity}\n"
        f"  circuit_hidden_dim: {cfg.circuit_hidden_dim}\n"
        f"  attention_dim:      {cfg.attention_dim}\n"
        f"  num_heads:          {cfg.num_heads}\n"
        f"  feature_dim:        {cfg.feature_dim}  (logits + hidden + PE/loss flags)\n"
        f"  use_layer_PE:       {cfg.use_layer_PE}\n"
        f"  use_intra_layer_PE: {cfg.use_intra_layer_PE}\n"
        f"  use_node_loss:      {cfg.use_node_loss}\n"
        f"  max_neighbors:      {cfg.max_neighbors}\n"
        f"  total parameters:   {n_params_total:,}\n"
        f"  attn_rezero:        {_block_attn_rezero(weights.blocks):.4f}\n"
        f"  logit_rezero:       {weights.logit_rezero:.4f}\n"
        f"  hidden_rezero:      {weights.hidden_rezero:.4f}"
    )


def _count_params(b: Any) -> int:
    out = 0
    for f in b.__dataclass_fields__:
        v = getattr(b, f)
        if isinstance(v, np.ndarray):
            out += v.size
        elif isinstance(v, tuple):
            out += sum(x.size for x in v if isinstance(x, np.ndarray))
        elif isinstance(v, MHAWeights):
            out += _count_params(v)
    return out


def _block_attn_rezero(blocks: Any) -> float:
    if isinstance(blocks, list):
        return float(blocks[0].attn_rezero) if blocks else 0.0
    return float(blocks.attn_rezero)


# ---------------------------------------------------------------------------
# Tensor packing: fp16 (default) or per-tensor symmetric uint8
# ---------------------------------------------------------------------------
#
# Two on-disk dtypes are supported:
#
# - ``fp16`` (default, ~2 bytes/param): IEEE-754 half-precision, no scale
#   metadata. Enough mantissa bits (10) to be effectively transparent for our
#   164 K-param TMT, while halving the file size vs fp32.
#
# - ``uint8`` (~1 byte/param + scale): symmetric quantisation with zero_point
#   128, mirrors the format used by the Growing-NCA web demo
#   (``post--growing-ca/public/webgl_models8/*.json``). Smaller, but per-tensor
#   symmetric scaling is too coarse for transformer-style weight distributions
#   where LayerNorm γ values cluster around 1 — one outlier dominates the scale
#   and the rest of the tensor loses precision. Available for tiny / well-
#   conditioned models if file size matters more than transparency.
#
# Sidecar scalars (ReZero α's, etc.) live next to the tensors as exact floats
# in ``scalars`` and are never quantised.

QUANT_DTYPES = ("fp16", "uint8")


def quantize_tensor(arr: np.ndarray, dtype: str = "fp16") -> dict[str, Any]:
    """Pack a float32 tensor for on-disk storage. Format is dtype-keyed."""
    arr = np.asarray(arr, dtype=np.float32)
    shape = [int(s) for s in arr.shape]

    if dtype == "fp16":
        packed = arr.astype(np.float16).tobytes()
        return {
            "shape": shape,
            "dtype": "fp16",
            "data_b64": base64.b64encode(packed).decode("ascii"),
        }

    if dtype == "uint8":
        abs_max = float(np.max(np.abs(arr))) if arr.size else 0.0
        if abs_max == 0.0:
            # All-zero: stamp 128's so dequantisation reads as zeros.
            q = np.full(shape, 128, dtype=np.uint8)
            scale = 0.0
        else:
            scale = abs_max / 127.0
            q = np.clip(np.round(arr / scale) + 128.0, 0.0, 255.0).astype(np.uint8)
        return {
            "shape": shape,
            "dtype": "uint8",
            "scale": scale,
            "zero_point": 128,
            "data_b64": base64.b64encode(q.tobytes()).decode("ascii"),
        }

    raise ValueError(f"Unsupported quantisation dtype {dtype!r}; want one of {QUANT_DTYPES}")


def dequantize_tensor(entry: dict[str, Any]) -> np.ndarray:
    """Inverse of :func:`quantize_tensor`. Dispatches on ``entry['dtype']``."""
    raw = base64.b64decode(entry["data_b64"])
    shape = tuple(entry["shape"])
    dtype = entry.get("dtype", "uint8")
    if dtype == "fp16":
        arr = np.frombuffer(raw, dtype=np.float16)
        if shape:
            arr = arr.reshape(shape)
        return arr.astype(np.float32)
    if dtype == "uint8":
        q = np.frombuffer(raw, dtype=np.uint8)
        if shape:
            q = q.reshape(shape)
        return (
            (q.astype(np.float32) - float(entry["zero_point"])) * float(entry["scale"])
        ).astype(np.float32)
    raise ValueError(f"Unsupported on-disk dtype {dtype!r}")


# ---------------------------------------------------------------------------
# Tensor-tree (de)serialisation
# ---------------------------------------------------------------------------


def _gathered_block_tensor_inventory(
    block: GatheredAttentionWeights,
) -> list[tuple[str, np.ndarray]]:
    """Canonical (name, tensor) pairs for one gathered-attention block."""
    return [
        ("block.attn_norm.scale", block.attn_norm[0]),
        ("block.attn_norm.bias", block.attn_norm[1]),
        ("block.Wq", block.Wq),
        ("block.bq", block.bq),
        ("block.Wk", block.Wk),
        ("block.bk", block.bk),
        ("block.Wv", block.Wv),
        ("block.bv", block.bv),
        ("block.Wo", block.Wo),
        ("block.bo", block.bo),
        ("block.q_ln_gamma", block.q_ln_gamma),
        ("block.k_ln_gamma", block.k_ln_gamma),
        ("block.ffn_ln.scale", block.ffn_ln[0]),
        ("block.ffn_ln.bias", block.ffn_ln[1]),
        ("block.ffn_W1", block.ffn_W1),
        ("block.ffn_b1", block.ffn_b1),
        ("block.ffn_W2", block.ffn_W2),
        ("block.ffn_b2", block.ffn_b2),
    ]


def _top_level_tensor_inventory(weights: TMTWeights) -> list[tuple[str, np.ndarray]]:
    """Canonical (name, tensor) pairs for the wrapper around the attention block."""
    return [
        ("input_norm.scale", weights.input_norm[0]),
        ("input_norm.bias", weights.input_norm[1]),
        ("feature_proj.kernel", weights.feature_proj_W),
        ("feature_proj.bias", weights.feature_proj_b),
        ("final_norm.scale", weights.final_norm[0]),
        ("final_norm.bias", weights.final_norm[1]),
        ("logit_proj.kernel", weights.logit_proj_W),
        ("logit_proj.bias", weights.logit_proj_b),
        ("hidden_proj.kernel", weights.hidden_proj_W),
        ("hidden_proj.bias", weights.hidden_proj_b),
    ]


def serialize_weights(
    weights: TMTWeights,
    out_path: str,
    *,
    source_run_id: str | None = None,
    dtype: str = "fp16",
) -> dict[str, Any]:
    """Quantise + base64-pack every tensor and write a single JSON to ``out_path``.

    Returns the ``doc`` dict that was written (handy for tests). The
    enclosing directory must exist; we create it if it doesn't.

    ``dtype`` selects the on-disk packing: ``"fp16"`` (default, transparent
    for transformer-style weights) or ``"uint8"`` (smaller but coarser).
    """
    if dtype not in QUANT_DTYPES:
        raise ValueError(f"dtype must be one of {QUANT_DTYPES}, got {dtype!r}")
    cfg = weights.config
    if cfg.model_kind != "gathered_attention":
        raise NotImplementedError(
            f"v1 ships gathered_attention only; got model_kind={cfg.model_kind!r}"
        )
    if not isinstance(weights.blocks, GatheredAttentionWeights):
        raise TypeError("gathered_attention TMTWeights must carry a GatheredAttentionWeights block")

    header = {
        "schema_version": WEIGHTS_SCHEMA_VERSION,
        "model_kind": cfg.model_kind,
        "arity": cfg.arity,
        "circuit_hidden_dim": cfg.circuit_hidden_dim,
        "attention_dim": cfg.attention_dim,
        "num_heads": cfg.num_heads,
        "num_attn_layers": cfg.num_attn_layers,
        "use_layer_PE": cfg.use_layer_PE,
        "use_intra_layer_PE": cfg.use_intra_layer_PE,
        "use_node_loss": cfg.use_node_loss,
        "max_neighbors": cfg.max_neighbors,
        "logit_dim": cfg.logit_dim,
        "feature_dim": cfg.feature_dim,
        "use_gelu_approx": bool(weights.blocks.use_gelu_approx),
        "source_run_id": source_run_id,
    }

    tensors: dict[str, dict[str, Any]] = {}
    for name, arr in _top_level_tensor_inventory(weights):
        tensors[name] = quantize_tensor(arr, dtype=dtype)
    for name, arr in _gathered_block_tensor_inventory(weights.blocks):
        tensors[name] = quantize_tensor(arr, dtype=dtype)
    header["tensor_dtype"] = dtype

    scalars = {
        "logit_rezero": weights.logit_rezero,
        "hidden_rezero": weights.hidden_rezero,
        "block.attn_rezero": weights.blocks.attn_rezero,
        "block.ffn_rezero": weights.blocks.ffn_rezero,
    }

    doc = {"header": header, "scalars": scalars, "tensors": tensors}
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        # ``separators`` strips spaces; bumps file size down ~10 %.
        json.dump(doc, f, separators=(",", ":"))
    return doc


def deserialize_weights(json_path: str) -> TMTWeights:
    """Load a JSON written by :func:`serialize_weights`. Inverse round-trip."""
    with open(json_path) as f:
        doc = json.load(f)
    header = doc["header"]
    if header.get("schema_version") != WEIGHTS_SCHEMA_VERSION:
        raise ValueError(
            f"weights JSON schema_version={header.get('schema_version')!r} "
            f"!= expected {WEIGHTS_SCHEMA_VERSION}"
        )
    if header["model_kind"] != "gathered_attention":
        raise NotImplementedError(
            f"deserialize_weights only supports gathered_attention; got {header['model_kind']!r}"
        )

    cfg = TMTConfig(
        arity=int(header["arity"]),
        circuit_hidden_dim=int(header["circuit_hidden_dim"]),
        attention_dim=int(header["attention_dim"]),
        num_heads=int(header["num_heads"]),
        num_attn_layers=int(header.get("num_attn_layers", 1)),
        use_layer_PE=bool(header["use_layer_PE"]),
        use_intra_layer_PE=bool(header["use_intra_layer_PE"]),
        use_node_loss=bool(header["use_node_loss"]),
        model_kind=header["model_kind"],
        max_neighbors=int(header["max_neighbors"]),
    )
    t = doc["tensors"]
    s = doc["scalars"]

    def dq(name: str) -> np.ndarray:
        return dequantize_tensor(t[name])

    block = GatheredAttentionWeights(
        attn_norm=(dq("block.attn_norm.scale"), dq("block.attn_norm.bias")),
        Wq=dq("block.Wq"),
        bq=dq("block.bq"),
        Wk=dq("block.Wk"),
        bk=dq("block.bk"),
        Wv=dq("block.Wv"),
        bv=dq("block.bv"),
        Wo=dq("block.Wo"),
        bo=dq("block.bo"),
        q_ln_gamma=dq("block.q_ln_gamma"),
        k_ln_gamma=dq("block.k_ln_gamma"),
        ffn_ln=(dq("block.ffn_ln.scale"), dq("block.ffn_ln.bias")),
        ffn_W1=dq("block.ffn_W1"),
        ffn_b1=dq("block.ffn_b1"),
        ffn_W2=dq("block.ffn_W2"),
        ffn_b2=dq("block.ffn_b2"),
        attn_rezero=float(s["block.attn_rezero"]),
        ffn_rezero=float(s["block.ffn_rezero"]),
        num_heads=int(cfg.num_heads),
        use_gelu_approx=bool(header.get("use_gelu_approx", True)),
    )

    return TMTWeights(
        config=cfg,
        input_norm=(dq("input_norm.scale"), dq("input_norm.bias")),
        feature_proj_W=dq("feature_proj.kernel"),
        feature_proj_b=dq("feature_proj.bias"),
        blocks=block,
        final_norm=(dq("final_norm.scale"), dq("final_norm.bias")),
        logit_proj_W=dq("logit_proj.kernel"),
        logit_proj_b=dq("logit_proj.bias"),
        hidden_proj_W=dq("hidden_proj.kernel"),
        hidden_proj_b=dq("hidden_proj.bias"),
        logit_rezero=float(s["logit_rezero"]),
        hidden_rezero=float(s["hidden_rezero"]),
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract a TMT checkpoint into the oracle schema.")
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--entity", default=DEFAULT_ENTITY)
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument("--prefer-metric", default=DEFAULT_PREFER_METRIC)
    parser.add_argument("--download-dir", default=DEFAULT_DOWNLOAD_DIR)
    parser.add_argument(
        "--out",
        default=None,
        help=(
            "If given, quantise + serialise the extracted weights to this path "
            "(typical: public/weights/<task>_<recipe>.json)."
        ),
    )
    parser.add_argument(
        "--dtype",
        default="fp16",
        choices=QUANT_DTYPES,
        help="On-disk tensor dtype (default fp16; uint8 = smaller but coarser).",
    )
    args = parser.parse_args()

    print(f"[extract] loading run {args.entity}/{args.project}/{args.run_id} ...")
    model, cfg, ckpt_path = load_run(
        run_id=args.run_id,
        entity=args.entity,
        project=args.project,
        prefer_metric=args.prefer_metric,
        download_dir=args.download_dir,
    )
    print(f"[extract] checkpoint: {ckpt_path}")
    weights = extract_weights(model, cfg)
    print("[extract] extracted weights summary:")
    print(_summarise(weights))

    if args.out:
        print(f"[extract] serialising {args.dtype} JSON -> {args.out}")
        serialize_weights(weights, args.out, source_run_id=args.run_id, dtype=args.dtype)
        size_kb = os.path.getsize(args.out) / 1024.0
        print(f"[extract] wrote {args.out} ({size_kb:,.1f} KB, dtype={args.dtype})")


if __name__ == "__main__":
    main()
