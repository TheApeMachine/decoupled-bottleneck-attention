"""
lower_test provides tests for the lowering pass.
"""
from __future__ import annotations

import unittest
import torch
from typing import cast

from caramba.compiler import Compiler
from caramba.config.layer import LinearLayerConfig, LayerNormLayerConfig
from caramba.config.topology import (
    NestedTopologyConfig,
    SequentialTopologyConfig,
    StackedTopologyConfig,
)


class LowerTest(unittest.TestCase):
    """
    LowerTest provides tests for lowering.
    """

    def setUp(self) -> None:
        """Create a fresh Compiler instance for each test."""
        self.compiler = Compiler()

    def test_expands_repeat_on_stacked(self) -> None:
        """
        test expanding repeat on stacked topology.
        """
        linear = LinearLayerConfig(d_in=4, d_out=4, bias=True)
        topo = StackedTopologyConfig(layers=[linear], repeat=3)
        lowered = self.compiler.lowerer.lower_topology(topo)

        self.assertEqual(lowered.repeat, 1)
        self.assertEqual(len(lowered.layers), 3)

    def test_expands_repeat_through_nested(self) -> None:
        """
        test expanding repeat through nested topology.
        """
        linear = LinearLayerConfig(d_in=4, d_out=4, bias=True)
        inner = StackedTopologyConfig(layers=[linear], repeat=2)
        outer = NestedTopologyConfig(layers=[inner], repeat=3)
        lowered = self.compiler.lowerer.lower_topology(outer)

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
        linear = LinearLayerConfig(d_in=4, d_out=4, bias=False)
        topo = StackedTopologyConfig(layers=[linear], repeat=2)
        lowered = self.compiler.lowerer.lower_topology(topo)

        model = lowered.build()
        x = torch.randn(2, 3, 4)
        _ = model.forward(x)

    def test_allows_topology_nodes_inside_layers(self) -> None:
        """
        test allowing topology nodes inside layer lists.
        """
        linear = LinearLayerConfig(d_in=4, d_out=4, bias=False)
        inner = SequentialTopologyConfig(layers=[linear], repeat=1)
        outer = StackedTopologyConfig(layers=[inner], repeat=1)
        lowered = self.compiler.lowerer.lower_topology(outer)

        model = lowered.build()
        x = torch.randn(2, 3, 4)
        _ = model.forward(x)

    def test_rejects_dim_mismatch(self) -> None:
        """
        test rejecting a simple d_model mismatch across layers.
        """
        topo = StackedTopologyConfig(
            layers=[
                LinearLayerConfig(d_in=4, d_out=4, bias=False),
                LayerNormLayerConfig(d_model=8),
            ]
        )
        with self.assertRaises(ValueError):
            self.compiler.validator.validate_topology(topo)


if __name__ == "__main__":
    unittest.main()
