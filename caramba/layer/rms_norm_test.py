"""
rms_norm_test provides tests for RMSNorm.
"""
from __future__ import annotations

import unittest
import torch

from caramba.config.layer import LayerType, RMSNormLayerConfig
from caramba.layer.rms_norm import RMSNormLayer


class RMSNormTest(unittest.TestCase):
    """
    RMSNormTest provides tests for RMSNorm.
    """
    def test_forward_shape(self) -> None:
        """
        test RMSNorm output shape.
        """
        norm = RMSNormLayer(
            RMSNormLayerConfig(
                type=LayerType.RMS_NORM,
                d_model=8,
                eps=1e-5,
                elementwise_affine=True,
            )
        )
        x = torch.randn(2, 3, 8)
        y = norm(x)
        self.assertEqual(tuple(y.shape), (2, 3, 8))


if __name__ == "__main__":
    unittest.main()
