"""
topology provides topology modules.
"""
from __future__ import annotations

from typing_extensions import TypeGuard
from torch import nn
from caramba.builder import Builder
from caramba.builder.layer import LayerBuilder
from caramba.config.layer import LayerConfig
from caramba.config.topology import TopologyConfig, TopologyType, _TopologyConfigBase
from caramba.topology.nested import Nested
from caramba.topology.sequential import Sequential
from caramba.topology.parallel import Parallel
from caramba.topology.branching import Branching
from caramba.topology.cyclic import Cyclic
from caramba.topology.recurrent import Recurrent
from caramba.topology.residual import Residual
from caramba.topology.stacked import Stacked


def is_topology_config(config: object) -> TypeGuard[TopologyConfig]:
    """
    is_topology_config checks if config is a topology config instance.
    """
    return isinstance(config, _TopologyConfigBase)


def is_layer_config(config: object) -> TypeGuard[LayerConfig]:
    """
    is_layer_config checks if config is a layer config instance.
    """
    return not isinstance(config, _TopologyConfigBase)


class TopologyBuilder(Builder):
    """
    TopologyBuilder builds topology modules from config.
    """
    def __init__(self) -> None:
        """
        __init__ initializes the topology builder.
        """
        self.layer_builder = LayerBuilder()

    def build(self, config: LayerConfig | TopologyConfig) -> nn.Module:
        """
        build builds a topology module from config.
        """
        if not isinstance(config, _TopologyConfigBase):
            raise ValueError(f"TopologyBuilder only accepts TopologyConfig, got {type(config)!r}")
        layers: list[nn.Module] = []

        match config.type:
            case TopologyType.NESTED:
                for layer in config.layers:
                    layers.append(self.build(layer))
                return Nested(config, layers)
            case (
                TopologyType.STACKED
                | TopologyType.RESIDUAL
                | TopologyType.SEQUENTIAL
                | TopologyType.PARALLEL
                | TopologyType.BRANCHING
                | TopologyType.CYCLIC
                | TopologyType.RECURRENT
            ):
                for layer in config.layers:
                    if is_layer_config(layer):
                        layers.append(self.layer_builder.build(layer))
                    else:
                        raise ValueError(f"Unsupported layer type: {type(layer)}")

                match config.type:
                    case TopologyType.STACKED:
                        return Stacked(config, layers)
                    case TopologyType.RESIDUAL:
                        return Residual(config, layers)
                    case TopologyType.SEQUENTIAL:
                        return Sequential(config, layers)
                    case TopologyType.PARALLEL:
                        return Parallel(config, layers)
                    case TopologyType.BRANCHING:
                        return Branching(config, layers)
                    case TopologyType.CYCLIC:
                        return Cyclic(config, layers)
                    case TopologyType.RECURRENT:
                        return Recurrent(config, layers)
                    case _:
                        raise ValueError(f"Unsupported topology type: {config.type}")
            case _:
                raise ValueError(f"Unsupported topology type: {config.type}")