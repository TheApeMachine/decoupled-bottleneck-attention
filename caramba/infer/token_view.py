"""Thread-safe views over token buffers used during generation.

Generation utilities often build up a token buffer incrementally (append,
rollback, slice). In interactive or serving settings, it is common to have
one thread producing tokens while another consumes them (streaming).

This module provides a small, well-tested abstraction that serializes these
mutations, keeping the buffer and its logical length consistent.
"""

from __future__ import annotations

from dataclasses import dataclass
import threading

import torch
from torch import Tensor


@dataclass
class TokenView:
    """A thread-safe token buffer with append/slice/rollback operations.

    Why this exists:
    - Autoregressive decoding builds token sequences incrementally.
    - Speculative decoding may rollback accepted tokens.
    - Streaming consumers need a safe way to read partial results.
    """

    _buf: Tensor
    _length: int
    _lock: threading.Lock

    @staticmethod
    def allocate(*, batch_size: int, max_len: int, device: torch.device, dtype: torch.dtype) -> "TokenView":
        """Allocate an uninitialized token buffer.

        Note: The buffer is uninitialized (torch.empty). Only positions up to
        the current logical length are valid; unused capacity contains garbage.
        """
        buf = torch.empty((int(batch_size), int(max_len)), device=device, dtype=dtype)
        return TokenView(_buf=buf, _length=0, _lock=threading.Lock())

    @property
    def buf(self) -> Tensor:
        """Return the full backing buffer (including unused capacity)."""
        return self._buf

    @property
    def length(self) -> int:
        """Current logical length of the sequence."""
        with self._lock:
            return int(self._length)

    def append(self, tokens: Tensor) -> None:
        """Append tokens (B, T) to the end of the buffer."""
        if tokens.dim() != 2:
            raise ValueError("tokens must be rank-2 (B, T)")
        b, t = int(tokens.size(0)), int(tokens.size(1))
        if b != int(self._buf.size(0)):
            raise ValueError(f"batch mismatch: {b} != {int(self._buf.size(0))}")
        if t <= 0:
            return
        with self._lock:
            end = self._length + t
            if end > int(self._buf.size(1)):
                raise ValueError(f"buffer overflow: end={end} > max_len={int(self._buf.size(1))}")
            self._buf[:, self._length:end] = tokens
            self._length = int(end)

    def rollback(self, n_tokens: int) -> None:
        """Rollback the logical length by n_tokens."""
        n = int(n_tokens)
        if n <= 0:
            return
        with self._lock:
            self._length = int(max(0, self._length - n))

    def slice(self, start: int, end: int) -> Tensor:
        """Return a view of tokens in [start, end) clamped to current length."""
        s = int(start)
        e = int(end)
        with self._lock:
            L = int(self._length)
            s2 = max(0, min(s, L))
            e2 = max(s2, min(e, L))
            return self._buf[:, s2:e2]

    def as_tensor(self) -> Tensor:
        """Return a view of the buffer up to the current length."""
        with self._lock:
            return self._buf[:, : int(self._length)]

