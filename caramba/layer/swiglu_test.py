"""
swiglu_test provides tests for SwiGLU.
"""
from __future__ import annotations

import unittest
import torch

from caramba.config.layer import LayerType, SwiGLULayerConfig
from caramba.layer.swiglu import SwiGLULayer


class SwiGLULayerTest(unittest.TestCase):
    """
    SwiGLULayerTest provides tests for SwiGLULayer.
    """
    def test_forward_shape(self) -> None:
        """
        test SwiGLULayer output shape.
        """
        cfg = SwiGLULayerConfig(
            type=LayerType.SWIGLU,
            d_model=8,
            d_ff=16,
            bias=True,
        )
        layer = SwiGLULayer(cfg)
        x = torch.randn(2, 3, 8)
        y = layer(x)
        self.assertEqual(tuple(y.shape), (2, 3, 8))


if __name__ == "__main__":
    unittest.main()


