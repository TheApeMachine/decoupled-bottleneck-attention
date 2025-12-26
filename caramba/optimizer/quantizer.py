from __future__ import annotations

import math
import torch
import torch.nn.functional as F

from caramba.cache import QuantSpec

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
# Midpoints between levels => bucketize gives nearest level, with ties going to the lower index
NF4_BOUNDS = (NF4_LEVELS[:-1] + NF4_LEVELS[1:]) * 0.5


class Quantizer:
    """Quantizer provides quantization and dequantization for KV caches."""
    _FP16_TINY = float(torch.finfo(torch.float16).tiny)

    def qblock_eff(self, kind: str, dim: int, qblock: int) -> int:
        qb = min(qblock if qblock > 0 else 32, dim)
        if kind in ("q4_0", "nf4"):
            if dim < 2:
                raise ValueError(f"{kind} cache requires dim >= 2")
            qb = min(qb, dim - (dim % 2))  # <= max even
            qb = max(qb, 2)
            qb -= qb % 2                   # even
        return max(1, qb)

    def make_quantspec(self, kind: str, dim: int, qblock: int) -> QuantSpec:
        qb = self.qblock_eff(kind, int(dim), int(qblock))
        pad_dim = ((int(dim) + qb - 1) // qb) * qb
        n_blocks = pad_dim // qb
        return QuantSpec(kind=kind, dim=int(dim), qblock=qb, pad_dim=pad_dim, n_blocks=n_blocks)

    @staticmethod
    def _pack_nibbles(u: torch.Tensor) -> torch.Tensor:
        # u: (..., even) uint8 in [0, 15]  -> packed (..., even/2) uint8
        return (u[..., 0::2] << 4) | u[..., 1::2]

    @staticmethod
    def _unpack_nibbles(p: torch.Tensor, pad_dim: int) -> torch.Tensor:
        # p: (N, pad_dim//2) uint8/int -> (N, pad_dim) int16 in [0, 15]
        p = p.to(torch.int16)
        out = torch.empty((p.size(0), pad_dim), device=p.device, dtype=torch.int16)
        out[:, 0::2] = (p >> 4) & 0xF
        out[:, 1::2] = p & 0xF
        return out

    @staticmethod
    def _pad_to(x: torch.Tensor, pad_dim: int) -> torch.Tensor:
        return x if x.size(-1) == pad_dim else F.pad(x, (0, pad_dim - x.size(-1)), value=0.0)

    def _to_blocks(self, x: torch.Tensor, spec: QuantSpec) -> tuple[torch.Tensor, tuple[int, ...]]:
        if x.size(-1) != spec.dim:
            raise ValueError(f"Expected dim {spec.dim}, got {x.size(-1)}")
        x = self._pad_to(x, spec.pad_dim)
        orig = x.shape[:-1]
        x2 = x.reshape(-1, spec.n_blocks, spec.qblock)
        return x2, orig

    def _scale(self, amax: torch.Tensor, denom: float = 1.0) -> torch.Tensor:
        return (amax / denom).clamp(min=self._FP16_TINY)

    @staticmethod
    def _check_kind(spec: QuantSpec, kind: str) -> None:
        if spec.kind != kind:
            raise ValueError(f"Expected kind '{kind}', got '{spec.kind}'")

    @staticmethod
    def _check_scale(scale: torch.Tensor, spec: QuantSpec) -> None:
        if scale.size(-1) != spec.n_blocks:
            raise ValueError(f"Expected scale n_blocks {spec.n_blocks}, got {scale.size(-1)}")

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
        x2, orig = self._to_blocks(x, spec)
        amax = x2.abs().amax(dim=-1).to(torch.float32)
        scale = self._scale(amax, denom)

        q = torch.round(x2.to(torch.float32) / scale.unsqueeze(-1)).clamp(qmin, qmax)

        if pack4:
            u = (q - qmin).to(torch.uint8)  # qmin negative => maps to [0, 15]
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
        self._check_scale(scale, spec)
        nb, qb, pad_dim, dim = spec.n_blocks, spec.qblock, spec.pad_dim, spec.dim
        orig = q.shape[:-1]

        if packed4:
            if q.size(-1) != pad_dim // 2:
                raise ValueError(f"Expected packed last dim {pad_dim // 2}, got {q.size(-1)}")
            u = self._unpack_nibbles(q.reshape(-1, pad_dim // 2), pad_dim).to(torch.float32)
            q2 = u + (float(qmin) if qmin is not None else 0.0)
            q2 = q2.reshape(-1, nb, qb)
        else:
            if q.size(-1) != pad_dim:
                raise ValueError(f"Expected q pad_dim {pad_dim}, got {q.size(-1)}")
            q2 = q.reshape(-1, nb, qb).to(torch.float32)

        s = scale.reshape(-1, nb).to(torch.float32).unsqueeze(-1)
        x2 = q2 * s
        return x2.reshape(*orig, pad_dim)[..., :dim]

    def quantize_q8_0(self, x: torch.Tensor, spec: QuantSpec) -> tuple[torch.Tensor, torch.Tensor]:
        self._check_kind(spec, "q8_0")
        return self._quant_linear(x, spec, qmin=-127, qmax=127, denom=127.0, pack4=False)

    def dequantize_q8_0(self, q: torch.Tensor, scale: torch.Tensor, spec: QuantSpec) -> torch.Tensor:
        self._check_kind(spec, "q8_0")
        return self._dequant_linear(q, scale, spec, packed4=False)

    def quantize_q4_0(self, x: torch.Tensor, spec: QuantSpec) -> tuple[torch.Tensor, torch.Tensor]:
        self._check_kind(spec, "q4_0")
        return self._quant_linear(x, spec, qmin=-8, qmax=7, denom=7.0, pack4=True)

    def dequantize_q4_0(self, packed: torch.Tensor, scale: torch.Tensor, spec: QuantSpec) -> torch.Tensor:
        self._check_kind(spec, "q4_0")
        return self._dequant_linear(packed, scale, spec, qmin=-8, packed4=True)

    def quantize_nf4(self, x: torch.Tensor, spec: QuantSpec) -> tuple[torch.Tensor, torch.Tensor]:
        self._check_kind(spec, "nf4")

        x2, orig = self._to_blocks(x, spec)
        amax = x2.abs().amax(dim=-1).to(torch.float32)
        scale = self._scale(amax)

        y = (x2.to(torch.float32) / scale.unsqueeze(-1)).clamp(-1.0, 1.0)
        bounds = NF4_BOUNDS.to(device=y.device, dtype=y.dtype)
        idx = torch.bucketize(y, bounds).to(torch.uint8)  # 0..15

        packed = self._pack_nibbles(idx).reshape(*orig, spec.pad_dim // 2)
        return packed, scale.to(torch.float16).reshape(*orig, spec.n_blocks)

    def dequantize_nf4(self, packed: torch.Tensor, scale: torch.Tensor, spec: QuantSpec) -> torch.Tensor:
        self._check_kind(spec, "nf4")
        self._check_scale(scale, spec)

        nb, qb, pad_dim, dim = spec.n_blocks, spec.qblock, spec.pad_dim, spec.dim
        if packed.size(-1) != pad_dim // 2:
            raise ValueError(f"Expected packed last dim {pad_dim // 2}, got {packed.size(-1)}")

        orig = packed.shape[:-1]
        idx = self._unpack_nibbles(packed.reshape(-1, pad_dim // 2), pad_dim).to(torch.long)

        levels = NF4_LEVELS.to(device=packed.device, dtype=torch.float32)
        q = levels[idx].reshape(-1, nb, qb)

        s = scale.reshape(-1, nb).to(torch.float32).unsqueeze(-1)
        x2 = q * s
        return x2.reshape(*orig, pad_dim)[..., :dim]
