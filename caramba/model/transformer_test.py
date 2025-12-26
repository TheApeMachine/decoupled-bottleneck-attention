"""
transformer_test provides tests to validate the
transformer model.
"""
from __future__ import annotations

import unittest

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

    def test_forward_with_activation_checkpointing(self) -> None:
        """Forward works when activation checkpointing is enabled on the topology."""
        layers: list[NodeConfig] = [
            LinearLayerConfig(type=LayerType.LINEAR, d_in=32, d_out=32, bias=True),
            AttentionLayerConfig(type=LayerType.ATTENTION, d_model=32, n_heads=4, dropout_p=0.0),
        ]
        transformer = TransformerModel(StackedTopologyConfig(layers=layers))
        # Topology is a StackedTopology; enable checkpointing.
        topo = transformer.topology
        if hasattr(topo, "activation_checkpointing"):
            setattr(topo, "activation_checkpointing", True)
            # Use a small positive threshold so >0 checks pass.
            setattr(topo, "activation_checkpoint_threshold_mb", 0.1)
        x = torch.randn(1, 8, 32, requires_grad=True)
        y = transformer(x)
        self.assertEqual(y.shape, (1, 8, 32))
        # Perform a backward pass to confirm autograd works with checkpointing.
        y.sum().backward()
        self.assertIsNotNone(x.grad, "Input should have gradients after backward pass")