"""
tokenizer provides text-to-token encoding for evaluation.
"""
from __future__ import annotations

import abc
import importlib
import importlib.util
from collections.abc import Sequence
from typing import Callable

from caramba.config.eval import TiktokenTokenizerConfig, TokenizerConfig


class Tokenizer(abc.ABC):
    """
    Tokenizer provides encode/decode for eval prompts.
    """
    @abc.abstractmethod
    def encode(self, text: str) -> list[int]:
        """
        encode converts text to token ids.
        """

    @abc.abstractmethod
    def decode(self, ids: Sequence[int]) -> str:
        """
        decode converts token ids to text.
        """


class _TiktokenTokenizer(Tokenizer):
    """
    _TiktokenTokenizer wraps a tiktoken Encoding object.
    """

    def __init__(self, *, encoding: str) -> None:
        """
        __init__ initializes the tiktoken tokenizer.
        """
        if not encoding:
            raise ValueError("encoding must be non-empty")
        if importlib.util.find_spec("tiktoken") is None:
            raise ImportError("tiktoken is required for tokenizer=tiktoken")
        mod = importlib.import_module("tiktoken")
        get_enc = getattr(mod, "get_encoding", None)
        if not callable(get_enc):
            raise ImportError("tiktoken.get_encoding is not available")
        self._enc = get_enc(str(encoding))

        # Validate encode is callable and cache it
        encode_fn = getattr(self._enc, "encode", None)
        if not callable(encode_fn):
            raise ValueError("tiktoken encoding does not support encode(...)")
        self._encode_fn: Callable[[str], list[int]] = encode_fn  # type: ignore[assignment]

        # Validate decode is callable and cache it
        decode_fn = getattr(self._enc, "decode", None)
        if not callable(decode_fn):
            raise ValueError("tiktoken encoding does not support decode(...)")
        self._decode_fn: Callable[[list[int]], str] = decode_fn  # type: ignore[assignment]

        # Smoke test: verify encode returns a list of ints
        test_ids = self._encode_fn("test")
        if not isinstance(test_ids, list) or (test_ids and not isinstance(test_ids[0], int)):
            raise ValueError("tiktoken encode(...) does not return list[int]")

    def encode(self, text: str) -> list[int]:
        """
        encode converts text to token ids.
        """
        return self._encode_fn(str(text))

    def decode(self, ids: Sequence[int]) -> str:
        """
        decode converts token ids to text.
        """
        return str(self._decode_fn(list(ids)))


def build_tokenizer(cfg: TokenizerConfig) -> Tokenizer:
    """
    build_tokenizer builds a Tokenizer from config.
    """
    if isinstance(cfg, TiktokenTokenizerConfig):
        return _TiktokenTokenizer(encoding=str(cfg.encoding))
    raise ValueError(f"Unsupported tokenizer config: {type(cfg)!r}")


