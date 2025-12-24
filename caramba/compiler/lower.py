"""
lower provides a small lowering pass from user configs to canonical configs.
"""
from __future__ import annotations

from caramba.config.manifest import Manifest
from caramba.config.model import ModelConfig
from caramba.config.topology import (
    BranchingTopologyConfig,
    CyclicTopologyConfig,
    NestedTopologyConfig,
    NodeConfig,
    ParallelTopologyConfig,
    RecurrentTopologyConfig,
    ResidualTopologyConfig,
    SequentialTopologyConfig,
    StackedTopologyConfig,
    TopologyConfig,
    _TopologyConfigBase,
)


def lower_manifest(manifest: Manifest) -> Manifest:
    """
    lower_manifest lowers a manifest into a canonical form.

    Currently this pass:
    - expands `repeat` at compile-time for all topology nodes
    - resets all `repeat` fields to 1
    """
    lowered_model = lower_model(manifest.model)
    return manifest.model_copy(update={"model": lowered_model})


def lower_model(model: ModelConfig) -> ModelConfig:
    """
    lower_model lowers a model config into a canonical form.
    """
    lowered_topology = lower_topology(model.topology)
    return model.model_copy(update={"topology": lowered_topology})


def lower_topology(config: TopologyConfig) -> TopologyConfig:
    """
    lower_topology expands topology-level `repeat` at compile-time.

    Note: this pass intentionally duplicates layer/topology configs to represent
    repeated structure explicitly. Any weight sharing must be expressed by a
    dedicated topology and is out-of-scope for this pass.
    """
    match config:
        case NestedTopologyConfig() as c:
            lowered = _lower_nodes(list(c.layers))
            layers = _repeat_nodes(lowered, repeat=int(c.repeat))
            return c.model_copy(update={"layers": layers, "repeat": 1})
        case StackedTopologyConfig() as c:
            lowered = _lower_nodes(list(c.layers))
            layers = _repeat_nodes(lowered, repeat=int(c.repeat))
            return c.model_copy(update={"layers": layers, "repeat": 1})
        case ResidualTopologyConfig() as c:
            lowered = _lower_nodes(list(c.layers))
            layers = _repeat_nodes(lowered, repeat=int(c.repeat))
            return c.model_copy(update={"layers": layers, "repeat": 1})
        case SequentialTopologyConfig() as c:
            lowered = _lower_nodes(list(c.layers))
            layers = _repeat_nodes(lowered, repeat=int(c.repeat))
            return c.model_copy(update={"layers": layers, "repeat": 1})
        case ParallelTopologyConfig() as c:
            lowered = _lower_nodes(list(c.layers))
            layers = _repeat_nodes(lowered, repeat=int(c.repeat))
            return c.model_copy(update={"layers": layers, "repeat": 1})
        case BranchingTopologyConfig() as c:
            lowered = _lower_nodes(list(c.layers))
            layers = _repeat_nodes(lowered, repeat=int(c.repeat))
            return c.model_copy(update={"layers": layers, "repeat": 1})
        case CyclicTopologyConfig() as c:
            lowered = _lower_nodes(list(c.layers))
            layers = _repeat_nodes(lowered, repeat=int(c.repeat))
            return c.model_copy(update={"layers": layers, "repeat": 1})
        case RecurrentTopologyConfig() as c:
            lowered = _lower_nodes(list(c.layers))
            layers = _repeat_nodes(lowered, repeat=int(c.repeat))
            return c.model_copy(update={"layers": layers, "repeat": 1})
        case _:
            raise ValueError(f"Unsupported topology config: {type(config)!r}")


def _lower_nodes(nodes: list[NodeConfig]) -> list[NodeConfig]:
    """
    _lower_nodes lowers nested topology nodes.
    """
    lowered: list[NodeConfig] = []
    for node in nodes:
        if isinstance(node, _TopologyConfigBase):
            lowered.append(lower_topology(node))
        else:
            lowered.append(node)
    return lowered


def _repeat_nodes(nodes: list[NodeConfig], *, repeat: int) -> list[NodeConfig]:
    """
    _repeat_nodes repeats nodes by deep-copying each element.
    """
    return [
        node.model_copy(deep=True)
        for _ in range(int(repeat))
        for node in nodes
    ]
