"""
transformer_test provides tests to validate the
transformer model.
"""
from __future__ import annotations

import unittest
import torch
from caramba.model.transformer import Transformer
from caramba.config.topology import TopologyConfig, TopologyType
from caramba.config.layer import (
    LinearLayerConfig,
    LayerNormLayerConfig,
    MultiheadLayerConfig,
    DropoutLayerConfig,
    LayerType,
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
        transformer = Transformer(TopologyConfig(layers=[
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
                elementwise_affine=True,
            ),
            MultiheadLayerConfig(
                type=LayerType.MULTIHEAD,
                d_model=128,
                n_heads=4,
                dropout=0.1,
            ),
            DropoutLayerConfig(type=LayerType.DROPOUT, p=0.1),
        ],
            type=TopologyType.STACKED,
        ))

        x: torch.Tensor = torch.randn(1, 10, 128)
        self.assertEqual(transformer(x).shape, (1, 10, 128))