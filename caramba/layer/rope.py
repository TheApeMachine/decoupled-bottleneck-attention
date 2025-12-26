"""Rotary positional embedding (RoPE) with amortized cache.

Caches cos/sin tables per (device, dtype) and grows geometrically
to avoid O(N) cache entries during token-by-token decode.
"""

from __future__ import annotations

import torch
from torch import nn


class RotaryEmbedding(nn.Module):
    """RoPE with cached cos/sin tables.

    Subclasses nn.Module so that inv_freq moves with model.to(device).
    """

    inv_freq: torch.Tensor  # Registered buffer
    rot_dim: int

    def __init__(self, rot_dim: int, base: float = 10000.0) -> None:
        super().__init__()
        if rot_dim % 2 != 0:
            raise ValueError(f"rot_dim must be even, got {rot_dim}")
        self.rot_dim = int(rot_dim)
        inv_freq = 1.0 / (
            base ** (torch.arange(0, self.rot_dim, 2, dtype=torch.float32) / float(self.rot_dim))
        )
        # Register as buffer so it moves with model.to(device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._cos_sin_cache: dict[tuple[str, str], tuple[torch.Tensor, torch.Tensor]] = {}

    @staticmethod
    def _next_pow2(n: int) -> int:
        n = int(n)
        if n <= 0:
            return 0
        return 1 << (n - 1).bit_length()

    def _cos_sin(
        self, seq_len: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        seq_len = int(seq_len)
        key = (str(device), str(dtype))
        cached = self._cos_sin_cache.get(key)

        if cached is None:
            cached_len = 0
            cos_cached = torch.empty((0, self.rot_dim // 2), device=device, dtype=dtype)
            sin_cached = torch.empty((0, self.rot_dim // 2), device=device, dtype=dtype)
        else:
            cos_cached, sin_cached = cached
            cached_len = int(cos_cached.size(0))

        if cached_len < seq_len:
            target_len = self._next_pow2(seq_len)
            # _next_pow2(seq_len) always returns >= seq_len for positive seq_len
            start = cached_len
            t = torch.arange(start, target_len, device=device, dtype=torch.float32)
            inv = self.inv_freq.to(device=device, dtype=torch.float32)
            freqs = torch.outer(t, inv)
            cos_new = torch.cos(freqs).to(dtype=dtype)
            sin_new = torch.sin(freqs).to(dtype=dtype)

            cos_cached = torch.cat([cos_cached, cos_new], dim=0)
            sin_cached = torch.cat([sin_cached, sin_new], dim=0)
            self._cos_sin_cache[key] = (cos_cached, sin_cached)

        return (cos_cached[:seq_len], sin_cached[:seq_len])

    def rotate(self, x: torch.Tensor, pos_offset: int = 0) -> torch.Tensor:
        """Apply RoPE to tensor x of shape (B, H, T, D).

        Rotates the first rot_dim dimensions; passes through the rest.
        """
        _B, _H, T, D = x.shape
        rot = self.rot_dim
        if rot > D:
            raise ValueError(f"rot_dim {rot} > head_dim {D}")

        cos, sin = self._cos_sin(pos_offset + T, x.device, x.dtype)
        cos = cos[pos_offset : pos_offset + T].unsqueeze(0).unsqueeze(0)  # (1,1,T,rot/2)
        sin = sin[pos_offset : pos_offset + T].unsqueeze(0).unsqueeze(0)

        x_rot = x[..., :rot]
        x_pass = x[..., rot:]

        x1 = x_rot[..., : rot // 2]
        x2 = x_rot[..., rot // 2 : rot]
        y1 = x1 * cos - x2 * sin
        y2 = x1 * sin + x2 * cos

        return torch.cat([y1, y2, x_pass], dim=-1)
