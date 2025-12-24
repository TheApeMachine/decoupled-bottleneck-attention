"""
model provides the top-level model wrapper.
"""
from __future__ import annotations

from torch import Tensor, nn
from typing_extensions import override

from caramba.config.model import ModelConfig, ModelType
from caramba.model.embedder import Embedder
from caramba.model.transformer import Transformer


class Model(nn.Module):
    """
    Model composes the embedder and the network module.
    """
    def __init__(self, config: ModelConfig) -> None:
        """
        __init__ initializes the model.
        """
        super().__init__()
        self.config: ModelConfig = config
        self.embedder: Embedder = Embedder(config.embedder)
        self.network: nn.Module = self._build_network(config)

    @override
    def forward(self, x: Tensor) -> Tensor:
        """
        forward passes through embedder and network.
        """
        x = self.embedder.forward(x)
        return self.network.forward(x)

    def _build_network(self, config: ModelConfig) -> nn.Module:
        """
        _build_network builds the network module from config.
        """
        match config.type:
            case ModelType.TRANSFORMER:
                return Transformer(config.topology)
            case _:
                raise ValueError(f"Unsupported model type: {config.type}")
