"""
validate provides compiler-time validation of lowered configs.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing_extensions import TypeGuard

from caramba.config.layer import (
    AttentionLayerConfig,
    DropoutLayerConfig,
    LayerConfig,
    LayerNormLayerConfig,
    LinearLayerConfig,
    MultiheadLayerConfig,
    RMSNormLayerConfig,
    SwiGLULayerConfig,
    _LayerConfigBase,
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
    _TopologyConfigBase,
)


NodeConfig = LayerConfig | TopologyConfig


@dataclass(frozen=True, slots=True)
class _IO:
    d_in: int | None
    d_out: int | None


def validate_topology(
    config: TopologyConfig,
    *,
    path: str = "model.topology",
) -> None:
    """validate_topology checks basic cross-layer shape invariants."""
    _infer_topology_io(config, path=path)


# -------------------------
# Node → IO (single entry)
# -------------------------

def _is_layer_config(node: NodeConfig) -> TypeGuard[LayerConfig]:
    """
    _is_layer_config checks if node is a layer config instance.
    """
    return isinstance(node, _LayerConfigBase)


def _infer_node_io(node: NodeConfig, *, path: str) -> _IO:
    if isinstance(node, _TopologyConfigBase):
        return _infer_topology_io(node, path=path)
    if _is_layer_config(node):
        return _infer_layer_io(node)
    raise ValueError(f"{path}: unsupported node type: {type(node)!r}")


def _infer_layer_io(config: LayerConfig) -> _IO:
    match config:
        case LinearLayerConfig() as c:
            return _IO(d_in=c.weight.d_in, d_out=c.weight.d_out)

        case (LayerNormLayerConfig() | RMSNormLayerConfig()) as c:
            d = c.weight.d_model
            return _IO(d_in=d, d_out=d)

        case (MultiheadLayerConfig()
              | AttentionLayerConfig()
              | SwiGLULayerConfig()) as c:
            d = c.weight.d_model
            return _IO(d_in=d, d_out=d)

        case DropoutLayerConfig():
            # Dropout is shape-transparent; leave unknown so it does not
            # constrain.
            return _IO(d_in=None, d_out=None)

        case _:
            raise ValueError(f"Unsupported layer config: {type(config)!r}")


def _require_match(cur: int | None, want: int, *, path: str) -> None:
    if cur is not None and cur != want:
        raise ValueError(
            f"{path}: expected d_in={cur}, got d_in={want}. "
            "Fix: make this node's input dim match the previous "
            "node's output dim."
        )


def _infer_seq_io(nodes: list[NodeConfig], *, path: str) -> _IO:
    cur: int | None = None
    for i, node in enumerate(nodes):
        node_path = f"{path}.layers[{i}]"
        io = _infer_node_io(node, path=node_path)

        if io.d_in is not None:
            _require_match(cur, io.d_in, path=node_path)
            cur = io.d_in

        if io.d_out is not None:
            cur = io.d_out

    return _IO(d_in=None, d_out=cur)


def _collect_branch_outs(nodes: list[NodeConfig], *, path: str) -> set[int]:
    outs: set[int] = set()
    for i, node in enumerate(nodes):
        node_path = f"{path}.layers[{i}]"
        io = _infer_node_io(node, path=node_path)
        if io.d_out is not None:
            outs.add(io.d_out)
    return outs


def _require_single_out(outs: set[int], *, path: str, kind: str) -> _IO:
    if len(outs) > 1:
        raise ValueError(
            f"{path}: {kind} requires consistent d_out, got {sorted(outs)}. "
            f"Fix: ensure all {kind} branches produce the same d_out."
        )
    return _IO(d_in=None, d_out=next(iter(outs)) if outs else None)


def _require_shape_preserving(nodes: list[NodeConfig], *, path: str) -> None:
    # Residual adds the block output to the block input.
    # For now we require every node to preserve shape, which is sufficient for
    # standard Transformer blocks (norm/attn/mlp are shape-preserving
    # end-to-end).
    for i, node in enumerate(nodes):
        node_path = f"{path}.layers[{i}]"
        io = _infer_node_io(node, path=node_path)
        if io.d_in is not None and io.d_out is not None and io.d_in != io.d_out:
            raise ValueError(
                f"{node_path}: residual requires shape-preserving nodes, got "
                f"d_in={io.d_in}, d_out={io.d_out}. "
                "Fix: ensure all nodes inside residual preserve d_model."
            )


def _infer_topology_io(config: TopologyConfig, *, path: str) -> _IO:
    match config:
        case (StackedTopologyConfig()
              | SequentialTopologyConfig()
              | NestedTopologyConfig()
              | CyclicTopologyConfig()
              | RecurrentTopologyConfig()) as c:
            return _infer_seq_io(list(c.layers), path=path)

        case ResidualTopologyConfig() as c:
            io = _infer_seq_io(list(c.layers), path=path)
            # If we cannot determine a final d_out, we cannot enforce
            # preservation reliably.
            if io.d_out is not None:
                _require_shape_preserving(list(c.layers), path=path)
            return io

        case ParallelTopologyConfig() as c:
            outs = _collect_branch_outs(list(c.layers), path=path)
            return _require_single_out(outs, path=path, kind="parallel")

        case BranchingTopologyConfig() as c:
            outs = _collect_branch_outs(list(c.layers), path=path)
            return _require_single_out(outs, path=path, kind="branching")

        case _:
            raise ValueError(f"Unsupported topology config: {type(config)!r}")
