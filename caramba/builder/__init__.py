"""
builder provides the builder module.
"""
from __future__ import annotations
from abc import ABC, abstractmethod

from torch import nn

from caramba.config.layer import LayerConfig
from caramba.config.topology import TopologyConfig


class Builder(ABC):
    """
    Builder provides the builder module.
    """
    @abstractmethod
    def build(self, config: LayerConfig | TopologyConfig) -> nn.Module:
        """
        build a module from config.
        """
