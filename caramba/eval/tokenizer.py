"""Tokenizer abstraction for evaluation.

Evaluation prompts need to be converted to token IDs. This module provides
a pluggable tokenizer interface with implementations for different backends.
Currently supports tiktoken (used by GPT models).
"""
from __future__ import annotations

import abc
import importlib
import importlib.util
from collections.abc import Callable, Sequence

from caramba.config.eval import TiktokenTokenizerConfig, TokenizerConfig


class Tokenizer(abc.ABC):
    """Abstract base class for text-to-token encoding.

    Provides a consistent interface for evaluation code regardless of
    the underlying tokenizer implementation.
    """

    @abc.abstractmethod
    def encode(self, text: str) -> list[int]:
        """Convert text to token IDs."""

    @abc.abstractmethod
    def decode(self, ids: Sequence[int]) -> str:
        """Convert token IDs back to text."""


class _TiktokenTokenizer(Tokenizer):
    """Tiktoken-based tokenizer implementation.

    Wraps OpenAI's tiktoken library for efficient tokenization.
    Used for evaluating models with GPT-compatible tokenizers.
    """

    def __init__(self, *, encoding: str) -> None:
        """Initialize with a specific tiktoken encoding (e.g., 'cl100k_base')."""
        if not encoding:
            raise ValueError("encoding must be non-empty")
        if importlib.util.find_spec("tiktoken") is None:
            raise ImportError("tiktoken is required for tokenizer=tiktoken")
        mod = importlib.import_module("tiktoken")
        get_enc = getattr(mod, "get_encoding", None)
        if not callable(get_enc):
            raise ImportError("tiktoken.get_encoding is not available")
        self._enc = get_enc(str(encoding))

        # Cache and validate encode function
        encode_fn = getattr(self._enc, "encode", None)
        if not callable(encode_fn):
            raise ValueError("tiktoken encoding does not support encode(...)")
        self._encode_fn: Callable[[str], list[int]] = encode_fn  # type: ignore[assignment]

        # Cache and validate decode function
        decode_fn = getattr(self._enc, "decode", None)
        if not callable(decode_fn):
            raise ValueError("tiktoken encoding does not support decode(...)")
        self._decode_fn: Callable[[list[int]], str] = decode_fn  # type: ignore[assignment]

        # Smoke test
        test_ids = self._encode_fn("test")
        if not isinstance(test_ids, list) or (
            test_ids and not isinstance(test_ids[0], int)
        ):
            raise ValueError("tiktoken encode(...) does not return list[int]")

    def encode(self, text: str) -> list[int]:
        """Convert text to token IDs using tiktoken."""
        return self._encode_fn(str(text))

    def decode(self, ids: Sequence[int]) -> str:
        """Convert token IDs to text using tiktoken."""
        return str(self._decode_fn(list(ids)))


def build_tokenizer(cfg: TokenizerConfig) -> Tokenizer:
    """Build a Tokenizer from config.

    Factory function that creates the appropriate tokenizer based on
    the config type.
    """
    if isinstance(cfg, TiktokenTokenizerConfig):
        return _TiktokenTokenizer(encoding=str(cfg.encoding))
    raise ValueError(f"Unsupported tokenizer config: {type(cfg)!r}")
