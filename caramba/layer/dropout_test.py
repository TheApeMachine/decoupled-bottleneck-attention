"""Test the dropout layer."""
from __future__ import annotations

import unittest
import torch

from caramba.config.layer import DropoutLayerConfig, LayerType
from caramba.layer.dropout import DropoutLayer

class DropoutLayerTest(unittest.TestCase):
    """Test the dropout layer."""
    def test_forward_shape(self) -> None:
        """Test the forward shape of the dropout layer."""
        dropout = DropoutLayer(
            DropoutLayerConfig(type=LayerType.DROPOUT, p=0.1)
        )
        x = torch.randn(2, 3, 8)
        y = dropout(x)
        self.assertEqual(tuple(y.shape), (2, 3, 8))


if __name__ == "__main__":
    unittest.main()
