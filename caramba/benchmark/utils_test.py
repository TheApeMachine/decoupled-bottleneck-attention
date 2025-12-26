"""
Unit tests for benchmark utility functions.
"""
from __future__ import annotations

import unittest
from dataclasses import dataclass
from unittest.mock import MagicMock

import torch
from torch import nn

from caramba.benchmark.utils import get_model_vocab_size


class SimpleModel(nn.Module):
    """Model with vocab_size attribute."""

    def __init__(self, vocab_size: int = 50000) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, 64)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embed(x)


class ConfigModel(nn.Module):
    """Model with config.vocab_size (HuggingFace style)."""

    def __init__(self, vocab_size: int = 128256) -> None:
        super().__init__()
        self.config = type("Config", (), {"vocab_size": vocab_size})()
        self.embed = nn.Embedding(vocab_size, 64)
        self.head = nn.Linear(64, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.embed(x))


class EmbeddingModel(nn.Module):
    """Model with get_input_embeddings method."""

    def __init__(self, vocab_size: int = 32768) -> None:
        super().__init__()
        self._embedding = nn.Embedding(vocab_size, 64)

    def get_input_embeddings(self) -> nn.Embedding:
        return self._embedding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._embedding(x)


class MinimalModel(nn.Module):
    """Model with no vocab_size indicators."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(64, 64)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class TestGetModelVocabSize(unittest.TestCase):
    """Tests for get_model_vocab_size function."""

    def test_direct_vocab_size_attribute(self) -> None:
        """Gets vocab_size from model.vocab_size attribute."""
        model = SimpleModel(vocab_size=50000)
        result = get_model_vocab_size(model)
        self.assertEqual(result, 50000)

    def test_config_vocab_size(self) -> None:
        """Gets vocab_size from model.config.vocab_size."""
        model = ConfigModel(vocab_size=128256)
        result = get_model_vocab_size(model)
        self.assertEqual(result, 128256)

    def test_embedding_num_embeddings(self) -> None:
        """Gets vocab_size from get_input_embeddings().num_embeddings."""
        model = EmbeddingModel(vocab_size=32768)
        result = get_model_vocab_size(model)
        self.assertEqual(result, 32768)

    def test_fallback_to_default(self) -> None:
        """Falls back to default when no vocab_size found."""
        model = MinimalModel()
        result = get_model_vocab_size(model)
        self.assertEqual(result, 32000)  # Default value

    def test_custom_default(self) -> None:
        """Uses custom default value when provided."""
        model = MinimalModel()
        result = get_model_vocab_size(model, default=100000)
        self.assertEqual(result, 100000)

    def test_priority_direct_over_config(self) -> None:
        """Direct vocab_size takes priority over config."""
        model = SimpleModel(vocab_size=10000)
        object.__setattr__(model, "config", type("Config", (), {"vocab_size": 20000})())
        result = get_model_vocab_size(model)
        self.assertEqual(result, 10000)

    def test_invalid_vocab_size_skipped(self) -> None:
        """Invalid vocab_size values are skipped."""
        model = MinimalModel()
        object.__setattr__(model, "vocab_size", 0)  # Invalid (not > 0)
        result = get_model_vocab_size(model)
        self.assertEqual(result, 32000)  # Falls back to default

    def test_non_int_vocab_size_skipped(self) -> None:
        """Non-integer vocab_size is skipped."""
        model = MinimalModel()
        model.vocab_size = "invalid"  # type: ignore
        result = get_model_vocab_size(model)
        self.assertEqual(result, 32000)

    def test_none_embedding_handled(self) -> None:
        """Handles None return from get_input_embeddings."""
        model = MinimalModel()
        model.get_input_embeddings = lambda: None  # type: ignore
        result = get_model_vocab_size(model)
        self.assertEqual(result, 32000)


class TestGetModelVocabSizeEdgeCases(unittest.TestCase):
    """Edge case tests for get_model_vocab_size."""

    def test_negative_vocab_size_skipped(self) -> None:
        """Negative vocab_size is skipped."""
        model = MinimalModel()
        object.__setattr__(model, "vocab_size", -100)
        result = get_model_vocab_size(model)
        self.assertEqual(result, 32000)

    def test_zero_vocab_size_skipped(self) -> None:
        """Zero vocab_size is skipped."""
        model = SimpleModel(vocab_size=1)
        model.vocab_size = 0
        result = get_model_vocab_size(model)
        self.assertEqual(result, 32000)

    def test_large_vocab_size(self) -> None:
        """Handles large vocab sizes correctly."""
        model = SimpleModel(vocab_size=1000000)
        result = get_model_vocab_size(model)
        self.assertEqual(result, 1000000)

    def test_one_vocab_size(self) -> None:
        """Handles vocab_size of 1 (edge case)."""
        model = MinimalModel()
        object.__setattr__(model, "vocab_size", 1)
        result = get_model_vocab_size(model)
        self.assertEqual(result, 1)


if __name__ == "__main__":
    unittest.main()
