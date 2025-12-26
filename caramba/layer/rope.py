"""Rotary positional embeddings (RoPE) with efficient caching.

RoPE encodes position information by rotating query and key vectors.
Unlike absolute positional embeddings, RoPE makes attention scores depend
on relative positions, improving generalization to sequence lengths longer
than training. The rotation is applied in pairs of dimensions.
"""
from __future__ import annotations

import torch
from torch import nn


class RotaryEmbedding(nn.Module):
    """RoPE with cached cos/sin tables.

    Caches are keyed by (device, dtype) and grow geometrically to avoid
    creating O(N) cache entries during token-by-token decoding. This is
    critical for efficient inference.
    """

    inv_freq: torch.Tensor
    rot_dim: int

    def __init__(self, rot_dim: int, base: float = 10000.0) -> None:
        """Initialize RoPE with the given rotation dimension.

        Args:
            rot_dim: Number of dimensions to rotate (must be even)
            base: Base for the frequency calculation (higher = slower decay)
        """
        super().__init__()
        if rot_dim % 2 != 0:
            raise ValueError(f"rot_dim must be even, got {rot_dim}")
        self.rot_dim = int(rot_dim)
        inv_freq = 1.0 / (
            base
            ** (
                torch.arange(0, self.rot_dim, 2, dtype=torch.float32)
                / float(self.rot_dim)
            )
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._cos_sin_cache: dict[tuple[str, str], tuple[torch.Tensor, torch.Tensor]] = (
            {}
        )

    @staticmethod
    def _next_pow2(n: int) -> int:
        """Round up to the next power of 2 for cache sizing."""
        n = int(n)
        if n <= 0:
            return 0
        return 1 << (n - 1).bit_length()

    def _cos_sin(
        self, seq_len: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Get cached cos/sin tables, expanding if needed.

        Uses geometric growth to minimize reallocations during decoding.
        """
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
        """Apply rotary embeddings to query/key tensor.

        Args:
            x: Tensor of shape (B, H, T, D) where D >= rot_dim
            pos_offset: Starting position for RoPE (for cached generation)

        Returns:
            Rotated tensor of same shape

        The first rot_dim dimensions are rotated; the rest pass through unchanged.
        """
        _B, _H, T, D = x.shape
        rot = self.rot_dim
        if rot > D:
            raise ValueError(f"rot_dim {rot} > head_dim {D}")

        cos, sin = self._cos_sin(pos_offset + T, x.device, x.dtype)
        cos = cos[pos_offset : pos_offset + T].unsqueeze(0).unsqueeze(0)
        sin = sin[pos_offset : pos_offset + T].unsqueeze(0).unsqueeze(0)

        x_rot = x[..., :rot]
        x_pass = x[..., rot:]

        x1 = x_rot[..., : rot // 2]
        x2 = x_rot[..., rot // 2 : rot]
        y1 = x1 * cos - x2 * sin
        y2 = x1 * sin + x2 * cos

        return torch.cat([y1, y2, x_pass], dim=-1)
