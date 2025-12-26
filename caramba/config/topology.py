"""Network topology configuration: how layers are composed.

A transformer isn't just a list of layers—layers are grouped into blocks,
blocks have residual connections, etc. Topology configs describe these
composition patterns declaratively, allowing flexible architecture experiments.
"""
from __future__ import annotations

import enum
from typing import Annotated, Literal, TypeAlias

from pydantic import Field

from caramba.config import Config, PositiveInt
from caramba.config.layer import LayerConfig


class TopologyType(str, enum.Enum):
    """Available topology patterns for composing layers.

    STACKED: Simple sequential stack
    RESIDUAL: Layers with residual (skip) connections
    NESTED: Recursive composition (topologies containing topologies)
    PARALLEL: Multiple paths run in parallel
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
        """Return the Python module containing topology implementations."""
        return "caramba.topology"


class NestedTopologyConfig(Config):
    """A topology that contains other topologies or layers.

    Use this for recursive composition, like transformer blocks that
    contain attention and MLP sub-blocks.
    """

    type: Literal[TopologyType.NESTED] = TopologyType.NESTED
    layers: list["NodeConfig"]
    repeat: PositiveInt = 1


class StackedTopologyConfig(Config):
    """A simple sequential stack of layers.

    Layers are run in order: output of layer N is input to layer N+1.
    """

    type: Literal[TopologyType.STACKED] = TopologyType.STACKED
    layers: list["NodeConfig"]
    repeat: PositiveInt = 1


class ResidualTopologyConfig(Config):
    """Layers with residual (skip) connections.

    Output = input + layers(input). This is the standard transformer
    pattern where attention and MLP blocks add to the residual stream.
    """

    type: Literal[TopologyType.RESIDUAL] = TopologyType.RESIDUAL
    layers: list["NodeConfig"]
    repeat: PositiveInt = 1


class SequentialTopologyConfig(Config):
    """Sequential layer composition (similar to STACKED)."""

    type: Literal[TopologyType.SEQUENTIAL] = TopologyType.SEQUENTIAL
    layers: list["NodeConfig"]
    repeat: PositiveInt = 1


class ParallelTopologyConfig(Config):
    """Multiple paths run in parallel, then combined.

    All branches receive the same input; outputs are concatenated or summed.
    """

    type: Literal[TopologyType.PARALLEL] = TopologyType.PARALLEL
    layers: list["NodeConfig"]
    repeat: PositiveInt = 1


class BranchingTopologyConfig(Config):
    """A topology with multiple branching paths."""

    type: Literal[TopologyType.BRANCHING] = TopologyType.BRANCHING
    layers: list["NodeConfig"]
    repeat: PositiveInt = 1


class CyclicTopologyConfig(Config):
    """A topology with cycles (for experimental architectures)."""

    type: Literal[TopologyType.CYCLIC] = TopologyType.CYCLIC
    layers: list["NodeConfig"]
    repeat: PositiveInt = 1


class RecurrentTopologyConfig(Config):
    """A topology with recurrent connections."""

    type: Literal[TopologyType.RECURRENT] = TopologyType.RECURRENT
    layers: list["NodeConfig"]
    repeat: PositiveInt = 1


# Union of all topology types for discriminated parsing
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


# A node in the topology tree can be either a layer or a sub-topology
NodeConfig: TypeAlias = LayerConfig | TopologyConfig
