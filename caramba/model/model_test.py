"""
model_test provides tests for the top-level model wrapper.
"""
from __future__ import annotations

import unittest

import torch

from caramba.config.embedder import (
    EmbedderType,
    NoEmbedderConfig,
    TokenEmbedderConfig,
)
from caramba.config.layer import LayerType, LinearLayerConfig
from caramba.config.model import ModelConfig, ModelType
from caramba.config.operation import MatmulOperationConfig
from caramba.config.topology import StackedTopologyConfig
from caramba.config.weight import DenseWeightConfig
from caramba.model.model import Model


class ModelTest(unittest.TestCase):
    """
    ModelTest provides tests for the top-level model wrapper.
    """
    def test_forward_with_token_embedder(self) -> None:
        """
        test forward pass with a token embedder.
        """
        d_model = 16
        model = Model(
            ModelConfig(
                type=ModelType.TRANSFORMER,
                embedder=TokenEmbedderConfig(
                    type=EmbedderType.TOKEN,
                    vocab_size=32,
                    d_model=d_model,
                ),
                topology=StackedTopologyConfig(
                    layers=[
                        LinearLayerConfig(
                            type=LayerType.LINEAR,
                            operation=MatmulOperationConfig(),
                            weight=DenseWeightConfig(
                                d_in=d_model,
                                d_out=d_model,
                                bias=False,
                            ),
                        ),
                    ],
                ),
            )
        )

        x: torch.Tensor = torch.randint(0, 32, (2, 4))
        self.assertEqual(model(x).shape, (2, 4, d_model))

    def test_forward_with_no_embedder(self) -> None:
        """
        test forward pass with a no-op embedder.
        """
        d_model = 16
        model = Model(
            ModelConfig(
                type=ModelType.TRANSFORMER,
                embedder=NoEmbedderConfig(type=EmbedderType.NONE),
                topology=StackedTopologyConfig(
                    layers=[
                        LinearLayerConfig(
                            type=LayerType.LINEAR,
                            operation=MatmulOperationConfig(),
                            weight=DenseWeightConfig(
                                d_in=d_model,
                                d_out=d_model,
                                bias=False,
                            ),
                        ),
                    ],
                ),
            )
        )

        x: torch.Tensor = torch.randn(2, 4, d_model)
        self.assertEqual(model(x).shape, (2, 4, d_model))
