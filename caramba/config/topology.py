"""
topology provides the network topology configuration.
"""
from __future__ import annotations

import enum
from typing import Annotated, Literal, TypeAlias

from pydantic import BaseModel, Field
from caramba.config.layer import LayerConfig


class TopologyType(str, enum.Enum):
    """
    TopologyType provides the network topology type.
    """
    NESTED = "nested"
    STACKED = "stacked"
    RESIDUAL = "residual"
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    BRANCHING = "branching"
    CYCLIC = "cyclic"
    RECURRENT = "recurrent"


class _TopologyConfigBase(BaseModel):
    """
    _TopologyConfigBase provides the base type for topology configs.
    """


class NestedTopologyConfig(_TopologyConfigBase):
    """
    NestedTopologyConfig provides a nested topology.
    """
    type: Literal[TopologyType.NESTED] = TopologyType.NESTED
    layers: list["NodeConfig"]
    repeat: int = Field(default=1, ge=1)


class StackedTopologyConfig(_TopologyConfigBase):
    """
    StackedTopologyConfig provides a simple sequential topology.
    """
    type: Literal[TopologyType.STACKED] = TopologyType.STACKED
    layers: list["NodeConfig"]
    repeat: int = Field(default=1, ge=1)


class ResidualTopologyConfig(_TopologyConfigBase):
    """
    ResidualTopologyConfig provides a residual topology.
    """
    type: Literal[TopologyType.RESIDUAL] = TopologyType.RESIDUAL
    layers: list["NodeConfig"]
    repeat: int = Field(default=1, ge=1)


class SequentialTopologyConfig(_TopologyConfigBase):
    """
    SequentialTopologyConfig provides a sequential topology.
    """
    type: Literal[TopologyType.SEQUENTIAL] = TopologyType.SEQUENTIAL
    layers: list["NodeConfig"]
    repeat: int = Field(default=1, ge=1)


class ParallelTopologyConfig(_TopologyConfigBase):
    """
    ParallelTopologyConfig provides a parallel topology.
    """
    type: Literal[TopologyType.PARALLEL] = TopologyType.PARALLEL
    layers: list["NodeConfig"]
    repeat: int = Field(default=1, ge=1)


class BranchingTopologyConfig(_TopologyConfigBase):
    """
    BranchingTopologyConfig provides a branching topology.
    """
    type: Literal[TopologyType.BRANCHING] = TopologyType.BRANCHING
    layers: list["NodeConfig"]
    repeat: int = Field(default=1, ge=1)


class CyclicTopologyConfig(_TopologyConfigBase):
    """
    CyclicTopologyConfig provides a cyclic topology.
    """
    type: Literal[TopologyType.CYCLIC] = TopologyType.CYCLIC
    layers: list["NodeConfig"]
    repeat: int = Field(default=1, ge=1)


class RecurrentTopologyConfig(_TopologyConfigBase):
    """
    RecurrentTopologyConfig provides a recurrent topology.
    """
    type: Literal[TopologyType.RECURRENT] = TopologyType.RECURRENT
    layers: list["NodeConfig"]
    repeat: int = Field(default=1, ge=1)


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
