"""
embedder_test provides tests to validate the
embedder module.
"""
from __future__ import annotations

import unittest

import torch
from caramba.model.embedder import Embedder
from caramba.config.embedder import (
    EmbedderType,
    NoEmbedderConfig,
    TokenEmbedderConfig,
)


class EmbedderTest(unittest.TestCase):
    """
    EmbedderTest provides tests to validate the
    embedder module.
    """
    def test_forward(self) -> None:
        """
        test the forward pass of the embedder.
        """
        embedder = Embedder(
            TokenEmbedderConfig(
                type=EmbedderType.TOKEN,
                vocab_size=128,
                d_model=128,
            )
        )
        x: torch.Tensor = torch.randint(0, 128, (1, 10))
        self.assertEqual(embedder(x).shape, (1, 10, 128))

    def test_forward_none(self) -> None:
        """
        test the forward pass with no embedder.
        """
        embedder = Embedder(NoEmbedderConfig(type=EmbedderType.NONE))
        x: torch.Tensor = torch.randn(1, 10, 16)
        self.assertIs(embedder(x), x)
