"""
caramba.model contains model components.
"""

from __future__ import annotations

from torch import Tensor, nn
from typing_extensions import override

from caramba.model.embedder import Embedder
from caramba.config.model import ModelConfig

class Model(nn.Module):
    """
    Model composes the embedder and the network module.
    """
    def __init__(self, config: ModelConfig) -> None:
        """
        __init__ initializes the model.
        """
        super().__init__()
        self.embedder: Embedder = Embedder(config.embedder)
        self.topology: nn.Module = config.topology.build()

    @override
    def forward(self, x: Tensor) -> Tensor:
        """
        forward passes through embedder and network.
        """
        if self.embedder is not None:
            x = self.embedder.forward(x)

        if self.topology is not None:
            x = self.topology.forward(x)

        return x