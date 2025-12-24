"""
lower_test provides tests for the lowering pass.
"""
from __future__ import annotations

import unittest
import torch
from typing import cast

from caramba.compiler.lower import lower_topology
from caramba.config.layer import LayerType, LinearLayerConfig
from caramba.config.layer import LayerNormLayerConfig
from caramba.config.operation import LayerNormOperationConfig, MatmulOperationConfig
from caramba.config.topology import (
    NestedTopologyConfig,
    SequentialTopologyConfig,
    StackedTopologyConfig,
)
from caramba.config.weight import DenseWeightConfig, NormWeightConfig
from caramba.model.transformer import Transformer


class LowerTest(unittest.TestCase):
    """
    LowerTest provides tests for lowering.
    """
    def test_expands_repeat_on_stacked(self) -> None:
        """
        test expanding repeat on stacked topology.
        """
        linear = LinearLayerConfig(
            type=LayerType.LINEAR,
            operation=MatmulOperationConfig(),
            weight=DenseWeightConfig(d_in=4, d_out=4, bias=True),
        )
        topo = StackedTopologyConfig(layers=[linear], repeat=3)
        lowered = lower_topology(topo)

        self.assertEqual(lowered.repeat, 1)
        self.assertEqual(len(lowered.layers), 3)

    def test_expands_repeat_through_nested(self) -> None:
        """
        test expanding repeat through nested topology.
        """
        linear = LinearLayerConfig(
            type=LayerType.LINEAR,
            operation=MatmulOperationConfig(),
            weight=DenseWeightConfig(d_in=4, d_out=4, bias=True),
        )
        inner = StackedTopologyConfig(layers=[linear], repeat=2)
        outer = NestedTopologyConfig(layers=[inner], repeat=3)
        lowered = lower_topology(outer)

        self.assertIsInstance(lowered, NestedTopologyConfig)
        lowered_nested = cast(NestedTopologyConfig, lowered)

        self.assertEqual(lowered.repeat, 1)
        self.assertEqual(len(lowered.layers), 3)
        for child in lowered_nested.layers:
            self.assertIsInstance(child, StackedTopologyConfig)
            child_stacked = cast(StackedTopologyConfig, child)
            self.assertEqual(child_stacked.repeat, 1)
            self.assertEqual(len(child_stacked.layers), 2)

    def test_builds_and_runs_after_lowering(self) -> None:
        """
        test model build/forward after lowering.
        """
        linear = LinearLayerConfig(
            type=LayerType.LINEAR,
            operation=MatmulOperationConfig(),
            weight=DenseWeightConfig(d_in=4, d_out=4, bias=False),
        )
        topo = StackedTopologyConfig(layers=[linear], repeat=2)
        lowered = lower_topology(topo)

        x = torch.randn(2, 3, 4)
        _ = Transformer(lowered).forward(x)

    def test_allows_topology_nodes_inside_layers(self) -> None:
        """
        test allowing topology nodes inside layer lists.
        """
        linear = LinearLayerConfig(
            type=LayerType.LINEAR,
            operation=MatmulOperationConfig(),
            weight=DenseWeightConfig(d_in=4, d_out=4, bias=False),
        )
        inner = SequentialTopologyConfig(layers=[linear], repeat=1)
        outer = StackedTopologyConfig(layers=[inner], repeat=1)
        lowered = lower_topology(outer)

        x = torch.randn(2, 3, 4)
        _ = Transformer(lowered).forward(x)

    def test_rejects_dim_mismatch(self) -> None:
        """
        test rejecting a simple d_model mismatch across layers.
        """
        topo = StackedTopologyConfig(
            layers=[
                LinearLayerConfig(
                    type=LayerType.LINEAR,
                    operation=MatmulOperationConfig(),
                    weight=DenseWeightConfig(d_in=4, d_out=4, bias=False),
                ),
                LayerNormLayerConfig(
                    type=LayerType.LAYER_NORM,
                    operation=LayerNormOperationConfig(eps=1e-5),
                    weight=NormWeightConfig(d_model=8, elementwise_affine=True),
                ),
            ]
        )
        with self.assertRaises(ValueError):
            _ = Transformer(topo)


if __name__ == "__main__":
    unittest.main()

