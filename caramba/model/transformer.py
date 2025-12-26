"""
transformer provides the transformer model.
"""
from __future__ import annotations

from torch import nn, Tensor
from caramba.config.topology import TopologyConfig
from caramba.compiler import Compiler

class TransformerModel(nn.Module):
    """
    Transformer provides the transformer model.
    """
    def __init__(self, config: TopologyConfig) -> None:
        super().__init__()

        self.compiler = Compiler()
        config = self.compiler.lowerer.lower_topology(config)
        self.topology: nn.Module = config.build()

    def forward(self, x: Tensor, *, ctx: object | None = None) -> Tensor:
        """
        forward pass for the transformer model.

        Args:
            x: Input tensor (B, T, d_model)
            ctx: Optional InferContext for cached inference
        """
        return self.topology(x, ctx=ctx)  # type: ignore[call-arg]