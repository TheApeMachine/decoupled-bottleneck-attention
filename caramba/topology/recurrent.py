"""
recurrent provides the recurrent topology.
"""
from __future__ import annotations

from torch import nn, Tensor
from typing_extensions import override

from caramba.config.topology import RecurrentTopologyConfig


class Recurrent(nn.Module):
    """
    Recurrent provides a recurrent topology.
    """
    def __init__(
        self,
        config: RecurrentTopologyConfig,
        layers: list[nn.Module],
    ) -> None:
        super().__init__()
        self.config: RecurrentTopologyConfig = config
        self.layers: nn.ModuleList = nn.ModuleList(layers)

    @override
    def forward(self, x: Tensor) -> Tensor:
        """
        forward pass for the recurrent topology.
        """
        for layer in self.layers:
            x = layer.forward(x)
        return x