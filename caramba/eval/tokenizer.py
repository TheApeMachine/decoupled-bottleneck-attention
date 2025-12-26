"""
tokenizer provides text-to-token encoding for evaluation.
"""
from __future__ import annotations

import importlib
import importlib.util

from collections.abc import Sequence

from caramba.config.eval import TiktokenTokenizerConfig, TokenizerConfig


class Tokenizer:
    """
    Tokenizer provides encode/decode for eval prompts.
    """
    def encode(self, text: str) -> list[int]:
        """
        encode converts text to token ids.
        """
        raise NotImplementedError

    def decode(self, ids: Sequence[int]) -> str:
        """
        decode converts token ids to text.
        """
        raise NotImplementedError


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

    def encode(self, text: str) -> list[int]:
        """
        encode converts text to token ids.
        """
        fn = getattr(self._enc, "encode", None)
        if not callable(fn):
            raise ValueError("tiktoken encoding does not support encode(...)")
        ids = fn(str(text))
        if not isinstance(ids, list) or not all(isinstance(i, int) for i in ids):
            raise ValueError("tiktoken encode(...) returned invalid ids")
        return [int(i) for i in ids]

    def decode(self, ids: Sequence[int]) -> str:
        """
        decode converts token ids to text.
        """
        fn = getattr(self._enc, "decode", None)
        if not callable(fn):
            raise ValueError("tiktoken encoding does not support decode(...)")
        return str(fn(list(ids)))


def build_tokenizer(cfg: TokenizerConfig) -> Tokenizer:
    """
    build_tokenizer builds a Tokenizer from config.
    """
    if isinstance(cfg, TiktokenTokenizerConfig):
        return _TiktokenTokenizer(encoding=str(cfg.encoding))
    raise ValueError(f"Unsupported tokenizer config: {type(cfg)!r}")


