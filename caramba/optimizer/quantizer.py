"""Quantization for KV-cache memory compression.

Large KV-caches can dominate GPU memory. Quantization reduces memory by
storing keys and values in lower precision formats:
- q8_0: 8-bit integers with per-block scales (1 byte/element)
- q4_0: 4-bit integers packed (0.5 bytes/element + scale overhead)
- nf4: NormalFloat4, QLoRA-style 4-bit with better distribution

The quantizer handles blocking, padding, and scale computation to minimize
quantization error while maintaining fast dequantization.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from caramba.cache import QuantSpec


# NormalFloat4 levels from QLoRA paper
NF4_LEVELS = torch.tensor(
    [
        -1.0,
        -0.6961928009986877,
        -0.5250730514526367,
        -0.39491748809814453,
        -0.28444138169288635,
        -0.18477343022823334,
        -0.09105003625154495,
        0.0,
        0.07958029955625534,
        0.16093020141124725,
        0.24611230194568634,
        0.33791524171829224,
        0.44070982933044434,
        0.5626170039176941,
        0.7229568362236023,
        1.0,
    ],
    dtype=torch.float32,
)
# Midpoints for nearest-neighbor quantization
NF4_BOUNDS = (NF4_LEVELS[:-1] + NF4_LEVELS[1:]) * 0.5


class Quantizer:
    """Quantizes and dequantizes tensors for KV-cache storage.

    Supports q8_0 (8-bit), q4_0 (4-bit linear), and nf4 (4-bit normal float).
    Each format uses per-block scales for better accuracy on varying
    magnitude data.
    """

    _FP16_TINY = float(torch.finfo(torch.float16).tiny)

    def qblock_eff(self, kind: str, dim: int, qblock: int) -> int:
        """Compute effective quantization block size.

        For 4-bit formats, ensures block size is even (required for packing).
        """
        qb = min(qblock if qblock > 0 else 32, dim)
        if kind in ("q4_0", "nf4"):
            if dim < 2:
                raise ValueError(f"{kind} cache requires dim >= 2")
            qb = min(qb, dim - (dim % 2))
            qb = max(qb, 2)
            qb -= qb % 2
        return max(1, qb)

    def make_quantspec(self, kind: str, dim: int, qblock: int) -> QuantSpec:
        """Create a QuantSpec for the given parameters."""
        qb = self.qblock_eff(kind, int(dim), int(qblock))
        pad_dim = ((int(dim) + qb - 1) // qb) * qb
        n_blocks = pad_dim // qb
        return QuantSpec(
            kind=kind, dim=int(dim), qblock=qb, pad_dim=pad_dim, n_blocks=n_blocks
        )

    @staticmethod
    def _pack_nibbles(u: torch.Tensor) -> torch.Tensor:
        """Pack two 4-bit values into one byte."""
        return (u[..., 0::2] << 4) | u[..., 1::2]

    @staticmethod
    def _unpack_nibbles(p: torch.Tensor, pad_dim: int) -> torch.Tensor:
        """Unpack bytes into pairs of 4-bit values."""
        p = p.to(torch.int16)
        out = torch.empty((p.size(0), pad_dim), device=p.device, dtype=torch.int16)
        out[:, 0::2] = (p >> 4) & 0xF
        out[:, 1::2] = p & 0xF
        return out

    @staticmethod
    def _pad_to(x: torch.Tensor, pad_dim: int) -> torch.Tensor:
        """Pad tensor to target dimension."""
        return (
            x
            if x.size(-1) == pad_dim
            else F.pad(x, (0, pad_dim - x.size(-1)), value=0.0)
        )

    def _to_blocks(
        self, x: torch.Tensor, spec: QuantSpec
    ) -> tuple[torch.Tensor, tuple[int, ...]]:
        """Reshape tensor into quantization blocks."""
        if x.size(-1) != spec.dim:
            raise ValueError(f"Expected dim {spec.dim}, got {x.size(-1)}")
        x = self._pad_to(x, spec.pad_dim)
        orig = x.shape[:-1]
        x2 = x.reshape(-1, spec.n_blocks, spec.qblock)
        return x2, orig

    def _scale(self, amax: torch.Tensor, denom: float = 1.0) -> torch.Tensor:
        """Compute per-block scales, avoiding division by zero."""
        return (amax / denom).clamp(min=self._FP16_TINY)

    @staticmethod
    def _check_kind(spec: QuantSpec, kind: str) -> None:
        """Validate quantization kind matches."""
        if spec.kind != kind:
            raise ValueError(f"Expected kind '{kind}', got '{spec.kind}'")

    @staticmethod
    def _check_scale(scale: torch.Tensor, spec: QuantSpec) -> None:
        """Validate scale tensor shape."""
        if scale.size(-1) != spec.n_blocks:
            raise ValueError(
                f"Expected scale n_blocks {spec.n_blocks}, got {scale.size(-1)}"
            )

    def _quant_linear(
        self,
        x: torch.Tensor,
        spec: QuantSpec,
        qmin: int,
        qmax: int,
        denom: float,
        *,
        pack4: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Core linear quantization logic."""
        x2, orig = self._to_blocks(x, spec)
        amax = x2.abs().amax(dim=-1).to(torch.float32)
        scale = self._scale(amax, denom)

        q = torch.round(x2.to(torch.float32) / scale.unsqueeze(-1)).clamp(qmin, qmax)

        if pack4:
            u = (q - qmin).to(torch.uint8)
            packed = self._pack_nibbles(u).reshape(*orig, spec.pad_dim // 2)
            return packed, scale.to(torch.float16).reshape(*orig, spec.n_blocks)

        q = q.to(torch.int8).reshape(*orig, spec.pad_dim)
        return q, scale.to(torch.float16).reshape(*orig, spec.n_blocks)

    def _dequant_linear(
        self,
        q: torch.Tensor,
        scale: torch.Tensor,
        spec: QuantSpec,
        *,
        qmin: int | None = None,
        packed4: bool = False,
    ) -> torch.Tensor:
        """Core linear dequantization logic."""
        self._check_scale(scale, spec)
        nb, qb, pad_dim, dim = spec.n_blocks, spec.qblock, spec.pad_dim, spec.dim
        orig = q.shape[:-1]

        if packed4:
            if q.size(-1) != pad_dim // 2:
                raise ValueError(
                    f"Expected packed last dim {pad_dim // 2}, got {q.size(-1)}"
                )
            u = self._unpack_nibbles(q.reshape(-1, pad_dim // 2), pad_dim).to(
                torch.float32
            )
            q2 = u + (float(qmin) if qmin is not None else 0.0)
            q2 = q2.reshape(-1, nb, qb)
        else:
            if q.size(-1) != pad_dim:
                raise ValueError(f"Expected q pad_dim {pad_dim}, got {q.size(-1)}")
            q2 = q.reshape(-1, nb, qb).to(torch.float32)

        s = scale.reshape(-1, nb).to(torch.float32).unsqueeze(-1)
        x2 = q2 * s
        return x2.reshape(*orig, pad_dim)[..., :dim]

    def quantize_q8_0(
        self, x: torch.Tensor, spec: QuantSpec
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Quantize to q8_0: 8-bit integers with per-block scales."""
        self._check_kind(spec, "q8_0")
        return self._quant_linear(x, spec, qmin=-127, qmax=127, denom=127.0, pack4=False)

    def dequantize_q8_0(
        self, q: torch.Tensor, scale: torch.Tensor, spec: QuantSpec
    ) -> torch.Tensor:
        """Dequantize from q8_0."""
        self._check_kind(spec, "q8_0")
        return self._dequant_linear(q, scale, spec, packed4=False)

    def quantize_q4_0(
        self, x: torch.Tensor, spec: QuantSpec
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Quantize to q4_0: 4-bit integers packed, with per-block scales."""
        self._check_kind(spec, "q4_0")
        return self._quant_linear(x, spec, qmin=-8, qmax=7, denom=7.0, pack4=True)

    def dequantize_q4_0(
        self, packed: torch.Tensor, scale: torch.Tensor, spec: QuantSpec
    ) -> torch.Tensor:
        """Dequantize from q4_0."""
        self._check_kind(spec, "q4_0")
        return self._dequant_linear(packed, scale, spec, qmin=-8, packed4=True)

    def quantize_nf4(
        self, x: torch.Tensor, spec: QuantSpec
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Quantize to nf4: 4-bit NormalFloat with per-block scales.

        NormalFloat uses non-uniform quantization levels optimized for
        normally-distributed data, reducing quantization error.
        """
        self._check_kind(spec, "nf4")

        x2, orig = self._to_blocks(x, spec)
        amax = x2.abs().amax(dim=-1).to(torch.float32)
        scale = self._scale(amax)

        y = (x2.to(torch.float32) / scale.unsqueeze(-1)).clamp(-1.0, 1.0)
        bounds = NF4_BOUNDS.to(device=y.device, dtype=y.dtype)
        idx = torch.bucketize(y, bounds).to(torch.uint8)

        packed = self._pack_nibbles(idx).reshape(*orig, spec.pad_dim // 2)
        return packed, scale.to(torch.float16).reshape(*orig, spec.n_blocks)

    def dequantize_nf4(
        self, packed: torch.Tensor, scale: torch.Tensor, spec: QuantSpec
    ) -> torch.Tensor:
        """Dequantize from nf4."""
        self._check_kind(spec, "nf4")
        self._check_scale(scale, spec)

        nb, qb, pad_dim, dim = spec.n_blocks, spec.qblock, spec.pad_dim, spec.dim
        if packed.size(-1) != pad_dim // 2:
            raise ValueError(
                f"Expected packed last dim {pad_dim // 2}, got {packed.size(-1)}"
            )

        orig = packed.shape[:-1]
        idx = self._unpack_nibbles(packed.reshape(-1, pad_dim // 2), pad_dim).to(
            torch.long
        )

        levels = NF4_LEVELS.to(device=packed.device, dtype=torch.float32)
        q = levels[idx].reshape(-1, nb, qb)

        s = scale.reshape(-1, nb).to(torch.float32).unsqueeze(-1)
        x2 = q * s
        return x2.reshape(*orig, pad_dim)[..., :dim]
