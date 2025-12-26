"""
transformer_test provides tests to validate the
transformer model.
"""
from __future__ import annotations

import unittest
from typing import cast

import torch

from caramba.model.transformer import TransformerModel
from caramba.config.topology import StackedTopologyConfig, NodeConfig
from caramba.config.layer import (
    LinearLayerConfig,
    LayerNormLayerConfig,
    DropoutLayerConfig,
    LayerType,
    AttentionLayerConfig,
)


class TransformerTest(unittest.TestCase):
    """
    TransformerTest provides tests to validate the
    transformer model.
    """
    def test_forward(self) -> None:
        """
        test the forward pass of the transformer model.
        """
        layers: list[NodeConfig] = [
            LinearLayerConfig(
                type=LayerType.LINEAR,
                d_in=128,
                d_out=128,
                bias=True,
            ),
            LayerNormLayerConfig(
                type=LayerType.LAYER_NORM,
                d_model=128,
                eps=1e-5,
            ),
            AttentionLayerConfig(
                type=LayerType.ATTENTION,
                d_model=128,
                n_heads=4,
                dropout_p=0.1,
            ),
            DropoutLayerConfig(
                type=LayerType.DROPOUT,
                p=0.1,
            ),
        ]
        transformer = TransformerModel(StackedTopologyConfig(layers=layers))

        x: torch.Tensor = torch.randn(1, 10, 128)
        self.assertEqual(transformer(x).shape, (1, 10, 128))