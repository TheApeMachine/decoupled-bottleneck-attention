"""
embedder provides the embedder module.
"""
from __future__ import annotations

from torch import Tensor, nn
from typing_extensions import override
from caramba.config.embedder import EmbedderConfig, EmbedderType


class Embedder(nn.Module):
    """
    Embedder provides a pluggable embedding stage.
    """
    def __init__(self, config: EmbedderConfig) -> None:
        """
        __init__ initializes the embedder module.
        """
        super().__init__()
        self.config: EmbedderConfig = config
        self.token_embedding: nn.Embedding | None = None

        match config.type:
            case EmbedderType.NONE:
                self.token_embedding = None
            case EmbedderType.TOKEN:
                self.token_embedding = nn.Embedding(
                    num_embeddings=config.vocab_size,
                    embedding_dim=config.d_model,
                )
            case _:
                raise ValueError(f"Unknown embedder type: {config.type}")

    @override
    def forward(self, x: Tensor) -> Tensor:
        """
        forward pass for the embedder.
        """
        match self.config.type:
            case EmbedderType.NONE:
                return x
            case EmbedderType.TOKEN:
                if self.token_embedding is None:
                    raise RuntimeError("Token embedder is not initialized.")
                return self.token_embedding(
                    x.to(dtype=self.token_embedding.weight.dtype).long(),
                )
            case _:
                raise ValueError(f"Unknown embedder type: {self.config.type}")
