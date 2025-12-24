"""
multihead provides the multihead layer.
"""
from __future__ import annotations

from torch import nn, Tensor
from typing_extensions import override

from caramba.config.layer import MultiheadLayerConfig
from caramba.operation.build import build_multihead_operation
from caramba.operation.multihead import MultiheadOp
from caramba.weight.build import build_multihead_weight
from caramba.weight.multihead import MultiheadWeight


class Multihead(nn.Module):
    """
    Multihead provides the multihead layer.
    """
    def __init__(self, config: MultiheadLayerConfig) -> None:
        super().__init__()
        self.config: MultiheadLayerConfig = config
        self.operation: MultiheadOp = build_multihead_operation(
            config.operation
        )
        self.weight: MultiheadWeight = build_multihead_weight(config.weight)

    @override
    def forward(self, x: Tensor) -> Tensor:
        """
        forward pass for the multihead layer.
        """
        return self.operation.forward(x, attn=self.weight.attn)
