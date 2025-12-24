"""
builder provides the builder module.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from torch import nn

TConfig = TypeVar("TConfig")


class Builder(ABC, Generic[TConfig]):
    """
    Builder provides the builder module.
    """
    @abstractmethod
    def build(self, config: TConfig) -> nn.Module:
        """
        build a module from config.
        """
