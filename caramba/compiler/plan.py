"""
plan provides an IR-like printer for lowered manifests.
"""
from __future__ import annotations

from typing import Iterable

from caramba.config.layer import LayerConfig
from caramba.config.manifest import Manifest
from caramba.config.operation import (
    AttentionOperationConfig,
    DropoutOperationConfig,
    LayerNormOperationConfig,
    MatmulOperationConfig,
    MultiheadOperationConfig,
    OperationConfig,
    RMSNormOperationConfig,
    SwiGLUOperationConfig,
)
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
)
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


def format_manifest_plan(manifest: Manifest) -> str:
    """
    format_manifest_plan renders a human-readable plan for a lowered manifest.
    """
    out: list[str] = []
    out.append(f"manifest.version={manifest.version}")
    out.append(f"model.type={manifest.model.type.value}")
    out.append("model.topology:")
    out.extend(_format_topology(manifest.model.topology, indent=2, path="model"))
    return "\n".join(out)


def _format_topology(
    config: TopologyConfig,
    *,
    indent: int,
    path: str,
) -> Iterable[str]:
    pad = " " * indent
    head = (
        f"{pad}- topology={config.type.value} "
        f"repeat={getattr(config, 'repeat', None)} "
        f"path={path}.topology"
    )
    yield head

    match config:
        case NestedTopologyConfig() as c:
            for i, child in enumerate(c.layers):
                yield from _format_topology(
                    child,
                    indent=indent + 2,
                    path=f"{path}.topology.layers[{i}]",
                )
        case (
            StackedTopologyConfig()
            | ResidualTopologyConfig()
            | SequentialTopologyConfig()
            | ParallelTopologyConfig()
            | BranchingTopologyConfig()
            | CyclicTopologyConfig()
            | RecurrentTopologyConfig()
        ) as c:
            for i, layer in enumerate(c.layers):
                yield from _format_layer(
                    layer,
                    indent=indent + 2,
                    path=f"{path}.topology.layers[{i}]",
                )
        case _:
            raise ValueError(f"Unsupported topology config: {type(config)!r}")


def _format_layer(
    config: LayerConfig,
    *,
    indent: int,
    path: str,
) -> Iterable[str]:
    pad = " " * indent
    op = getattr(config, "operation", None)
    weight = getattr(config, "weight", None)

    yield (
        f"{pad}- layer={config.type.value} "
        f"op={_op_type(op)} "
        f"weight={_weight_type(weight)} "
        f"{_fmt_dims(weight)} "
        f"path={path}"
    ).rstrip()

    if op is not None:
        yield from _format_operation(op, indent=indent + 2, path=f"{path}.operation")

    if weight is not None:
        yield from _format_weight(weight, indent=indent + 2, path=f"{path}.weight")


def _format_operation(
    config: OperationConfig,
    *,
    indent: int,
    path: str,
) -> Iterable[str]:
    pad = " " * indent
    match config:
        case MatmulOperationConfig():
            yield f"{pad}- operation=matmul path={path}"
        case LayerNormOperationConfig() as c:
            yield f"{pad}- operation=layer_norm eps={c.eps} path={path}"
        case RMSNormOperationConfig() as c:
            yield f"{pad}- operation=rms_norm eps={c.eps} path={path}"
        case DropoutOperationConfig() as c:
            yield f"{pad}- operation=dropout p={c.p} path={path}"
        case MultiheadOperationConfig():
            yield f"{pad}- operation=multihead path={path}"
        case AttentionOperationConfig() as c:
            yield (
                f"{pad}- operation=attention causal={c.is_causal} "
                f"dropout_p={c.dropout_p} path={path}"
            )
        case SwiGLUOperationConfig():
            yield f"{pad}- operation=swiglu path={path}"
        case _:
            raise ValueError(f"Unsupported operation config: {type(config)!r}")


def _format_weight(
    config: WeightConfig,
    *,
    indent: int,
    path: str,
) -> Iterable[str]:
    pad = " " * indent
    match config:
        case DenseWeightConfig() as c:
            yield (
                f"{pad}- weight=dense d_in={c.d_in} d_out={c.d_out} "
                f"bias={c.bias} path={path}"
            )
        case NormWeightConfig() as c:
            yield (
                f"{pad}- weight=norm d_model={c.d_model} "
                f"affine={c.elementwise_affine} path={path}"
            )
        case RMSNormWeightConfig() as c:
            yield f"{pad}- weight=rms_norm d_model={c.d_model} path={path}"
        case SwiGLUWeightConfig() as c:
            yield (
                f"{pad}- weight=swiglu d_model={c.d_model} d_ff={c.d_ff} "
                f"bias={c.bias} path={path}"
            )
        case MultiheadWeightConfig() as c:
            head_dim = c.d_model // c.n_heads
            yield (
                f"{pad}- weight=multihead d_model={c.d_model} n_heads={c.n_heads} "
                f"head_dim={head_dim} dropout={c.dropout} path={path}"
            )
        case LlamaAttentionWeightConfig() as c:
            head_dim = c.d_model // c.n_heads
            kv_groups = c.n_heads // c.n_kv_heads
            yield (
                f"{pad}- weight=llama_attention d_model={c.d_model} "
                f"n_heads={c.n_heads} n_kv_heads={c.n_kv_heads} "
                f"head_dim={head_dim} kv_groups={kv_groups} "
                f"rope_dim={c.rope_dim} rope_base={c.rope_base} "
                f"bias={c.bias} path={path}"
            )
        case DecoupledAttentionWeightConfig() as c:
            head_dim = c.d_model // c.n_heads
            sem_head_dim = c.sem_dim // c.n_heads
            geo_head_dim = c.geo_dim // c.n_heads
            kv_groups = c.n_heads // c.n_kv_heads
            yield (
                f"{pad}- weight=decoupled_attention d_model={c.d_model} "
                f"n_heads={c.n_heads} n_kv_heads={c.n_kv_heads} "
                f"head_dim={head_dim} sem_dim={c.sem_dim} "
                f"geo_dim={c.geo_dim} sem_head_dim={sem_head_dim} "
                f"geo_head_dim={geo_head_dim} kv_groups={kv_groups} "
                f"rope_dim={c.rope_dim} rope_base={c.rope_base} "
                f"gate={c.gate} bias={c.bias} path={path}"
            )
        case _:
            raise ValueError(f"Unsupported weight config: {type(config)!r}")


def _op_type(op: OperationConfig | None) -> str:
    if op is None:
        return "none"
    return op.type.value


def _weight_type(weight: WeightConfig | None) -> str:
    if weight is None:
        return "none"
    return weight.type.value


def _fmt_dims(weight: WeightConfig | None) -> str:
    if weight is None:
        return ""

    match weight:
        case DenseWeightConfig() as c:
            return f"d_in={c.d_in} d_out={c.d_out}"
        case NormWeightConfig() as c:
            return f"d_model={c.d_model}"
        case RMSNormWeightConfig() as c:
            return f"d_model={c.d_model}"
        case SwiGLUWeightConfig() as c:
            return f"d_model={c.d_model} d_ff={c.d_ff}"
        case MultiheadWeightConfig() as c:
            return f"d_model={c.d_model} n_heads={c.n_heads}"
        case LlamaAttentionWeightConfig() as c:
            return (
                f"d_model={c.d_model} n_heads={c.n_heads} "
                f"n_kv_heads={c.n_kv_heads} rope_dim={c.rope_dim}"
            )
        case DecoupledAttentionWeightConfig() as c:
            return (
                f"d_model={c.d_model} n_heads={c.n_heads} "
                f"sem_dim={c.sem_dim} geo_dim={c.geo_dim}"
            )
        case _:
            raise ValueError(f"Unsupported weight config: {type(weight)!r}")


