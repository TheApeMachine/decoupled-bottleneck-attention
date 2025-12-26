"""
topology provides the network topology configuration.
"""
from __future__ import annotations

import enum
from typing import Annotated, Literal, TypeAlias

from pydantic import Field
from caramba.config.layer import LayerConfig
from caramba.config import Config, PositiveInt


class TopologyType(str, enum.Enum):
    """
    TopologyType provides the network topology type.
    """
    BRANCHING = "BranchingTopology"
    CYCLIC = "CyclicTopology"
    NESTED = "NestedTopology"
    PARALLEL = "ParallelTopology"
    RECURRENT = "RecurrentTopology"
    RESIDUAL = "ResidualTopology"
    SEQUENTIAL = "SequentialTopology"
    STACKED = "StackedTopology"

    @staticmethod
    def module_name() -> str:
        """Returns the module name for the topology type."""
        return "caramba.topology"


class NestedTopologyConfig(Config):
    """
    NestedTopologyConfig provides a nested topology.
    """
    type: Literal[TopologyType.NESTED] = TopologyType.NESTED
    layers: list["NodeConfig"]
    repeat: PositiveInt = 1


class StackedTopologyConfig(Config):
    """
    StackedTopologyConfig provides a simple sequential topology.
    """
    type: Literal[TopologyType.STACKED] = TopologyType.STACKED
    layers: list["NodeConfig"]
    repeat: PositiveInt = 1


class ResidualTopologyConfig(Config):
    """
    ResidualTopologyConfig provides a residual topology.
    """
    type: Literal[TopologyType.RESIDUAL] = TopologyType.RESIDUAL
    layers: list["NodeConfig"]
    repeat: PositiveInt = 1


class SequentialTopologyConfig(Config):
    """
    SequentialTopologyConfig provides a sequential topology.
    """
    type: Literal[TopologyType.SEQUENTIAL] = TopologyType.SEQUENTIAL
    layers: list["NodeConfig"]
    repeat: PositiveInt = 1


class ParallelTopologyConfig(Config):
    """
    ParallelTopologyConfig provides a parallel topology.
    """
    type: Literal[TopologyType.PARALLEL] = TopologyType.PARALLEL
    layers: list["NodeConfig"]
    repeat: PositiveInt = 1


class BranchingTopologyConfig(Config):
    """
    BranchingTopologyConfig provides a branching topology.
    """
    type: Literal[TopologyType.BRANCHING] = TopologyType.BRANCHING
    layers: list["NodeConfig"]
    repeat: PositiveInt = 1


class CyclicTopologyConfig(Config):
    """
    CyclicTopologyConfig provides a cyclic topology.
    """
    type: Literal[TopologyType.CYCLIC] = TopologyType.CYCLIC
    layers: list["NodeConfig"]
    repeat: PositiveInt = 1


class RecurrentTopologyConfig(Config):
    """
    RecurrentTopologyConfig provides a recurrent topology.
    """
    type: Literal[TopologyType.RECURRENT] = TopologyType.RECURRENT
    layers: list["NodeConfig"]
    repeat: PositiveInt = 1


TopologyConfig: TypeAlias = Annotated[
    NestedTopologyConfig
    | StackedTopologyConfig
    | ResidualTopologyConfig
    | SequentialTopologyConfig
    | ParallelTopologyConfig
    | BranchingTopologyConfig
    | CyclicTopologyConfig
    | RecurrentTopologyConfig,
    Field(discriminator="type"),
]


NodeConfig: TypeAlias = LayerConfig | TopologyConfig
