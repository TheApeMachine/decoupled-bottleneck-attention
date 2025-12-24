"""
trace_test provides tests for Trace.
"""
from __future__ import annotations

import unittest

import torch
from torch import nn

from caramba.model.trace import Trace


class TraceTest(unittest.TestCase):
    """
    TraceTest provides tests for Trace.
    """
    def test_captures_outputs(self) -> None:
        """
        test capturing outputs from matching modules.
        """
        model = nn.Sequential(
            nn.Linear(4, 4),
            nn.ReLU(),
            nn.Linear(4, 4),
        )
        tracer = Trace(
            model,
            predicate=lambda _name, module: isinstance(module, nn.Linear),
        )
        x = torch.randn(2, 4)
        with tracer:
            _ = model(x)

        self.assertEqual(len(tracer.outputs), 2)
        self.assertEqual(tracer.outputs[0].shape, (2, 4))
        self.assertEqual(tracer.outputs[1].shape, (2, 4))

    def test_rejects_empty_match(self) -> None:
        """
        test rejecting a predicate that matches no modules.
        """
        model = nn.ReLU()
        tracer = Trace(
            model,
            predicate=lambda _name, module: isinstance(module, nn.Linear),
        )
        with self.assertRaises(ValueError):
            tracer.attach()
