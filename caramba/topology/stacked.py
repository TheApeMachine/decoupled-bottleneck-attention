"""
stacked provides the stacked network.
"""
from __future__ import annotations

from torch import nn, Tensor
from typing_extensions import override

from caramba.config.layer import LayerType
from caramba.config.topology import TopologyConfig
from caramba.layer.linear import Linear
from caramba.layer.normalize import Normalize
from caramba.layer.multihead import Multihead
from caramba.layer.dropout import Dropout
from caramba.layer.sequential import Sequential

class Stacked(nn.Module):
    """
    Stacked provides the stacked network.
    """
    def __init__(self, config: TopologyConfig) -> None:
        super().__init__()
        self.config: TopologyConfig = config
        self.layers: nn.ModuleList = nn.ModuleList([])

    def build(self) -> None:
        """
        build the stacked network.
        """
        for layer in self.config.layers:
            match layer.type:
                case LayerType.LINEAR:
                    self.layers.append(Linear(layer))
                case LayerType.LAYER_NORM:
                    self.layers.append(Normalize(layer))
                case LayerType.MULTIHEAD:
                    self.layers.append(Multihead(layer))
                case LayerType.DROPOUT:
                    self.layers.append(Dropout(layer))
                case LayerType.SEQUENTIAL:
                    self.layers.append(Sequential(layer))
                case _:
                    raise ValueError(f"Unsupported layer type: {layer.type}")

    @override
    def forward(self, x: Tensor) -> Tensor:
        """
        forward pass for the stacked network.
        """
        if len(self.layers) == 0:
            self.build()

        for layer in self.layers:
            x = layer.forward(x)

        return x
