"""Transformer model wrapper with compiler support.

This module provides a TransformerModel that uses the Compiler to process
topology configurations before building the model. The compiler can apply
optimizations like layer fusion or architecture transformations.
"""
from __future__ import annotations

from torch import Tensor, nn

from caramba.compiler import Compiler
from caramba.config.topology import TopologyConfig


class TransformerModel(nn.Module):
    """A transformer built from a compiled topology configuration.

    Unlike Model (which uses raw configs), TransformerModel runs the
    config through the Compiler first. This enables optimizations and
    transformations that operate on the topology before instantiation.
    """

    def __init__(self, config: TopologyConfig) -> None:
        """Compile the config and build the transformer layers.

        The compiler's lowerer processes the topology config, potentially
        applying optimizations before the actual nn.Module is created.
        """
        super().__init__()

        compiler = Compiler()
        config = compiler.lowerer.lower_topology(config)
        self.topology = config.build()

    def forward(self, x: Tensor, *, ctx: object | None = None) -> Tensor:
        """Run the transformer forward pass.

        Args:
            x: Input embeddings, shape (B, T, d_model)
            ctx: Optional InferContext for KV-cache during generation

        Returns:
            Output features, shape (B, T, d_model)
        """
        return self.topology(x, ctx=ctx)  # type: ignore[call-arg]
