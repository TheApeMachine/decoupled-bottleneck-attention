"""
validate_manifest provides strict compiler-time validation for manifests.
"""
from __future__ import annotations

from typing import Iterable

from caramba.compiler.validate import validate_topology
from caramba.config.layer import AttentionLayerConfig, LayerConfig, MultiheadLayerConfig
from caramba.config.manifest import Manifest
from caramba.config.topology import (
    BranchingTopologyConfig,
    CyclicTopologyConfig,
    NestedTopologyConfig,
    ParallelTopologyConfig,
    RecurrentTopologyConfig,
    ResidualTopologyConfig,
    SequentialTopologyConfig,
    StackedTopologyConfig,
    TopologyConfig,
    _TopologyConfigBase,
)
from caramba.config.weight import DecoupledAttentionWeightConfig, LlamaAttentionWeightConfig
from caramba.config.weight import MultiheadWeightConfig


def validate_manifest(manifest: Manifest) -> None:
    """
    validate_manifest validates a lowered manifest.
    """
    validate_topology(manifest.model.topology, path="model.topology")

    for layer, layer_path in _iter_layers(manifest.model.topology, path="model.topology"):
        if isinstance(layer, AttentionLayerConfig):
            _validate_attention_weight(
                layer.weight,
                path=f"{layer_path}.weight",
            )
        if isinstance(layer, MultiheadLayerConfig):
            _validate_multihead_weight(
                layer.weight,
                path=f"{layer_path}.weight",
            )


def _iter_layers(
    topology: TopologyConfig,
    *,
    path: str,
) -> Iterable[tuple[LayerConfig, str]]:
    match topology:
        case (
            NestedTopologyConfig()
            | StackedTopologyConfig()
            | ResidualTopologyConfig()
            | SequentialTopologyConfig()
            | ParallelTopologyConfig()
            | BranchingTopologyConfig()
            | CyclicTopologyConfig()
            | RecurrentTopologyConfig()
        ) as c:
            for i, node in enumerate(c.layers):
                node_path = f"{path}.layers[{i}]"
                if isinstance(node, _TopologyConfigBase):
                    yield from _iter_layers(node, path=node_path)
                else:
                    yield node, node_path
        case _:
            raise ValueError(f"Unsupported topology config: {type(topology)!r}")


def _validate_attention_weight(
    weight: LlamaAttentionWeightConfig | DecoupledAttentionWeightConfig,
    *,
    path: str,
) -> None:
    d_model = weight.d_model
    n_heads = weight.n_heads
    n_kv_heads = weight.n_kv_heads

    if d_model % n_heads != 0:
        raise ValueError(
            f"{path}.d_model: expected d_model % n_heads == 0, got "
            f"d_model={d_model}, n_heads={n_heads}. Fix: set d_model to a "
            "multiple of n_heads."
        )

    if n_heads % n_kv_heads != 0:
        raise ValueError(
            f"{path}.n_kv_heads: expected n_heads % n_kv_heads == 0, got "
            f"n_heads={n_heads}, n_kv_heads={n_kv_heads}. Fix: choose n_kv_heads "
            "that divides n_heads."
        )

    match weight:
        case LlamaAttentionWeightConfig() as w:
            head_dim = d_model // n_heads
            rope_dim = w.rope_dim
            if rope_dim % 2 != 0:
                raise ValueError(
                    f"{path}.rope_dim: expected rope_dim to be even, got "
                    f"rope_dim={rope_dim}. Fix: set rope_dim to an even number."
                )
            if rope_dim > head_dim:
                raise ValueError(
                    f"{path}.rope_dim: expected rope_dim <= head_dim, got "
                    f"rope_dim={rope_dim}, head_dim={head_dim}. Fix: lower rope_dim "
                    "or increase head_dim."
                )
        case DecoupledAttentionWeightConfig() as w:
            sem_dim = w.sem_dim
            geo_dim = w.geo_dim

            if sem_dim % n_heads != 0:
                raise ValueError(
                    f"{path}.sem_dim: expected sem_dim % n_heads == 0, got "
                    f"sem_dim={sem_dim}, n_heads={n_heads}. Fix: set sem_dim to a "
                    "multiple of n_heads."
                )

            if geo_dim % n_heads != 0:
                raise ValueError(
                    f"{path}.geo_dim: expected geo_dim % n_heads == 0, got "
                    f"geo_dim={geo_dim}, n_heads={n_heads}. Fix: set geo_dim to a "
                    "multiple of n_heads."
                )

            geo_head_dim = geo_dim // n_heads
            rope_dim = w.rope_dim

            if rope_dim % 2 != 0:
                raise ValueError(
                    f"{path}.rope_dim: expected rope_dim to be even, got "
                    f"rope_dim={rope_dim}. Fix: set rope_dim to an even number."
                )

            if rope_dim > geo_head_dim:
                raise ValueError(
                    f"{path}.rope_dim: expected rope_dim <= geo_head_dim, got "
                    f"rope_dim={rope_dim}, geo_head_dim={geo_head_dim}. Fix: lower "
                    "rope_dim or increase geo_dim."
                )
        case _:
            raise ValueError(f"Unsupported attention weight config: {type(weight)!r}")


def _validate_multihead_weight(weight: MultiheadWeightConfig, *, path: str) -> None:
    d_model = weight.d_model
    n_heads = weight.n_heads
    if d_model % n_heads != 0:
        raise ValueError(
            f"{path}.d_model: expected d_model % n_heads == 0, got "
            f"d_model={d_model}, n_heads={n_heads}. Fix: set d_model to a "
            "multiple of n_heads."
        )

