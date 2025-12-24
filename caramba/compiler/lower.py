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
    ParallelTopologyConfig,
    RecurrentTopologyConfig,
    ResidualTopologyConfig,
    SequentialTopologyConfig,
    StackedTopologyConfig,
    TopologyConfig,
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
            lowered_layers = [lower_topology(x) for x in c.layers]
            layers = [
                x.model_copy(deep=True)
                for _ in range(c.repeat)
                for x in lowered_layers
            ]
            return c.model_copy(update={"layers": layers, "repeat": 1})
        case StackedTopologyConfig() as c:
            layers = [
                x.model_copy(deep=True)
                for _ in range(c.repeat)
                for x in c.layers
            ]
            return c.model_copy(update={"layers": layers, "repeat": 1})
        case ResidualTopologyConfig() as c:
            layers = [
                x.model_copy(deep=True)
                for _ in range(c.repeat)
                for x in c.layers
            ]
            return c.model_copy(update={"layers": layers, "repeat": 1})
        case SequentialTopologyConfig() as c:
            layers = [
                x.model_copy(deep=True)
                for _ in range(c.repeat)
                for x in c.layers
            ]
            return c.model_copy(update={"layers": layers, "repeat": 1})
        case ParallelTopologyConfig() as c:
            layers = [
                x.model_copy(deep=True)
                for _ in range(c.repeat)
                for x in c.layers
            ]
            return c.model_copy(update={"layers": layers, "repeat": 1})
        case BranchingTopologyConfig() as c:
            layers = [
                x.model_copy(deep=True)
                for _ in range(c.repeat)
                for x in c.layers
            ]
            return c.model_copy(update={"layers": layers, "repeat": 1})
        case CyclicTopologyConfig() as c:
            layers = [
                x.model_copy(deep=True)
                for _ in range(c.repeat)
                for x in c.layers
            ]
            return c.model_copy(update={"layers": layers, "repeat": 1})
        case RecurrentTopologyConfig() as c:
            layers = [
                x.model_copy(deep=True)
                for _ in range(c.repeat)
                for x in c.layers
            ]
            return c.model_copy(update={"layers": layers, "repeat": 1})
        case _:
            raise ValueError(f"Unsupported topology config: {type(config)!r}")
