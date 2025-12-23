"""
topology provides the network topology configuration.
"""
from __future__ import annotations
import enum

from pydantic import BaseModel
from caramba.config.layer import LayerConfig


class TopologyType(str, enum.Enum):
    """
    TopologyType provides the network topology type.
    """
    STACKED = "stacked"
    RESIDUAL = "residual"
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    BRANCHING = "branching"
    CYCLIC = "cyclic"
    RECURRENT = "recurrent"
    

class TopologyConfig(BaseModel):
    """
    TopologyConfig provides the network topology configuration.
    """
    type: TopologyType
    layers: list[LayerConfig]
