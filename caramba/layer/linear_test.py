"""Test the linear layer."""
from __future__ import annotations

import unittest

import torch

from caramba.config.layer import LayerType, LinearLayerConfig
from caramba.layer.linear import LinearLayer


class LinearLayerTest(unittest.TestCase):
    """Test the linear layer."""

    def test_forward_shape(self) -> None:
        layer = LinearLayer(LinearLayerConfig(type=LayerType.LINEAR, d_in=8, d_out=16, bias=True))
        x = torch.randn(2, 3, 8)
        y = layer(x)
        self.assertEqual(tuple(y.shape), (2, 3, 16))


if __name__ == "__main__":
    unittest.main()

