"""Test the layer norm layer."""
from __future__ import annotations

import unittest

import torch

from caramba.config.layer import LayerNormLayerConfig, LayerType
from caramba.layer.layer_norm import LayerNormLayer


class LayerNormLayerTest(unittest.TestCase):
    """Test the layer norm layer."""

    def test_forward_shape(self) -> None:
        layer = LayerNormLayer(LayerNormLayerConfig(type=LayerType.LAYER_NORM, d_model=8, eps=1e-5))
        x = torch.randn(2, 3, 8)
        y = layer(x)
        self.assertEqual(tuple(y.shape), (2, 3, 8))


if __name__ == "__main__":
    unittest.main()

