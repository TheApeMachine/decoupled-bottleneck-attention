"""
build provides factories for weight modules from config.
"""

from __future__ import annotations

from torch import nn

from caramba.config.weight import (
    DecoupledAttentionWeightConfig,
    DenseWeightConfig,
    LlamaAttentionWeightConfig,
    MultiheadWeightConfig,
    NormWeightConfig,
    RMSNormWeightConfig,
    SwiGLUWeightConfig,
    WeightConfig,
)
from caramba.weight.attention_decoupled import DecoupledAttentionWeight
from caramba.weight.attention_llama import LlamaAttentionWeight
from caramba.weight.dense import DenseWeight
from caramba.weight.layer_norm import LayerNormWeight
from caramba.weight.multihead import MultiheadWeight
from caramba.weight.rms_norm import RMSNormWeight
from caramba.weight.swiglu import SwiGLUWeight


def build_weight(config: WeightConfig) -> nn.Module:
    """
    build_weight builds a weight module from a WeightConfig.
    """
    match config:
        case DenseWeightConfig() as c:
            return DenseWeight(int(c.d_in), int(c.d_out), bias=bool(c.bias))
        case NormWeightConfig() as c:
            return LayerNormWeight(
                int(c.d_model),
                elementwise_affine=bool(c.elementwise_affine),
            )
        case RMSNormWeightConfig() as c:
            return RMSNormWeight(int(c.d_model))
        case SwiGLUWeightConfig() as c:
            return SwiGLUWeight(
                d_model=int(c.d_model),
                d_ff=int(c.d_ff),
                bias=bool(c.bias),
            )
        case MultiheadWeightConfig() as c:
            return MultiheadWeight(
                d_model=int(c.d_model),
                n_heads=int(c.n_heads),
                dropout=float(c.dropout),
            )
        case (
            LlamaAttentionWeightConfig() | DecoupledAttentionWeightConfig()
        ) as c:
            return build_attention_weight(c)
        case _:
            raise ValueError(f"Unsupported weight config: {type(config)!r}")


def build_attention_weight(
    config: LlamaAttentionWeightConfig | DecoupledAttentionWeightConfig,
) -> LlamaAttentionWeight | DecoupledAttentionWeight:
    """
    build_attention_weight builds an attention-weight module from config.
    """
    match config:
        case LlamaAttentionWeightConfig() as c:
            return LlamaAttentionWeight(
                d_model=int(c.d_model),
                n_heads=int(c.n_heads),
                n_kv_heads=int(c.n_kv_heads),
                rope_base=float(c.rope_base),
                rope_dim=int(c.rope_dim),
                bias=bool(c.bias),
            )
        case DecoupledAttentionWeightConfig() as c:
            return DecoupledAttentionWeight(
                d_model=int(c.d_model),
                n_heads=int(c.n_heads),
                n_kv_heads=int(c.n_kv_heads),
                sem_dim=int(c.sem_dim),
                geo_dim=int(c.geo_dim),
                rope_base=float(c.rope_base),
                rope_dim=int(c.rope_dim),
                bias=bool(c.bias),
                gate=bool(c.gate),
            )
        case _:
            raise ValueError(
                "Unsupported attention weight config: "
                f"{type(config)!r}"
            )


def build_dense_weight(config: DenseWeightConfig) -> DenseWeight:
    """
    build_dense_weight builds a dense weight module.
    """
    weight = build_weight(config)
    if not isinstance(weight, DenseWeight):
        raise RuntimeError(f"Expected DenseWeight, got {type(weight)!r}")
    return weight


def build_layer_norm_weight(config: NormWeightConfig) -> LayerNormWeight:
    """
    build_layer_norm_weight builds a layer norm weight module.
    """
    weight = build_weight(config)
    if not isinstance(weight, LayerNormWeight):
        raise RuntimeError(f"Expected LayerNormWeight, got {type(weight)!r}")
    return weight


def build_rms_norm_weight(config: RMSNormWeightConfig) -> RMSNormWeight:
    """
    build_rms_norm_weight builds an RMSNorm weight module.
    """
    weight = build_weight(config)
    if not isinstance(weight, RMSNormWeight):
        raise RuntimeError(f"Expected RMSNormWeight, got {type(weight)!r}")
    return weight


def build_swiglu_weight(config: SwiGLUWeightConfig) -> SwiGLUWeight:
    """
    build_swiglu_weight builds a SwiGLU weight module.
    """
    weight = build_weight(config)
    if not isinstance(weight, SwiGLUWeight):
        raise RuntimeError(f"Expected SwiGLUWeight, got {type(weight)!r}")
    return weight


def build_multihead_weight(config: MultiheadWeightConfig) -> MultiheadWeight:
    """
    build_multihead_weight builds a Multihead weight module.
    """
    weight = build_weight(config)
    if not isinstance(weight, MultiheadWeight):
        raise RuntimeError(f"Expected MultiheadWeight, got {type(weight)!r}")
    return weight
