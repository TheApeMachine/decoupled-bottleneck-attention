from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import torch
import torch.nn.functional as F


# -----------------------------
# KV cache quantization (formats + storage)
# -----------------------------

KVCacheKind = Literal["fp16", "fp32", "q8_0", "q4_0", "nf4"]


@dataclass(frozen=True)
class QuantSpec:
    kind: KVCacheKind
    dim: int
    qblock: int
    pad_dim: int
    n_blocks: int


@dataclass(frozen=True)
class KVCacheTensorConfig:
    kind: KVCacheKind = "fp16"
    qblock: int = 32
    residual_len: int = 0  # keep a small fp16 "hot" window for the newest tokens (ring-buffer)


def _qblock_eff(kind: KVCacheKind, dim: int, qblock: int) -> int:
    qb = min(qblock if qblock > 0 else 32, dim)
    if kind in ("q4_0", "nf4"):
        if dim < 2:
            raise ValueError(f"{kind} cache requires dim >= 2")
        max_even = dim if (dim % 2 == 0) else (dim - 1)
        qb = min(qb, max_even)
        if qb < 2:
            qb = 2
        if qb % 2 != 0:
            qb -= 1
    return max(1, qb)


def make_quantspec(kind: KVCacheKind, dim: int, qblock: int) -> QuantSpec:
    qb = _qblock_eff(kind, dim, qblock)
    pad_dim = int(math.ceil(dim / qb) * qb)
    if kind in ("q4_0", "nf4") and (pad_dim % 2 != 0):
        pad_dim += qb
    n_blocks = pad_dim // qb
    return QuantSpec(kind=kind, dim=dim, qblock=qb, pad_dim=pad_dim, n_blocks=n_blocks)


def quantize_q8_0(x: torch.Tensor, spec: QuantSpec) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-block symmetric int8 with fp16 scale (absmax / 127)."""
    if spec.kind != "q8_0":
        raise ValueError(spec.kind)
    dim, pad_dim, qb, nb = spec.dim, spec.pad_dim, spec.qblock, spec.n_blocks
    if x.size(-1) != dim:
        raise ValueError(f"Expected dim {dim}, got {x.size(-1)}")
    if pad_dim != dim:
        x = F.pad(x, (0, pad_dim - dim), value=0.0)

    orig = x.shape[:-1]
    x2 = x.reshape(-1, pad_dim).reshape(-1, nb, qb)
    amax = x2.abs().amax(dim=-1)  # (N, nb)
    scale = (amax / 127.0).clamp(min=1e-8)
    q = torch.round(x2 / scale.unsqueeze(-1)).clamp(-127, 127).to(torch.int8)
    q = q.reshape(*orig, pad_dim)
    return q, scale.to(torch.float16).reshape(*orig, nb)


def dequantize_q8_0(q: torch.Tensor, scale: torch.Tensor, spec: QuantSpec) -> torch.Tensor:
    if spec.kind != "q8_0":
        raise ValueError(spec.kind)
    dim, pad_dim, qb, nb = spec.dim, spec.pad_dim, spec.qblock, spec.n_blocks
    if q.size(-1) != pad_dim:
        raise ValueError(f"Expected q pad_dim {pad_dim}, got {q.size(-1)}")
    if scale.size(-1) != nb:
        raise ValueError(f"Expected scale n_blocks {nb}, got {scale.size(-1)}")
    orig = q.shape[:-1]
    q2 = q.reshape(-1, pad_dim).reshape(-1, nb, qb).to(torch.float32)
    s2 = scale.reshape(-1, nb).to(torch.float32).unsqueeze(-1)
    x2 = q2 * s2
    x = x2.reshape(*orig, pad_dim)[..., :dim]
    return x


def quantize_q4_0(x: torch.Tensor, spec: QuantSpec) -> Tuple[torch.Tensor, torch.Tensor]:
    """Q4_0-like: int4 packed into uint8, with fp16 scale per block."""
    if spec.kind != "q4_0":
        raise ValueError(spec.kind)
    dim, pad_dim, qb, nb = spec.dim, spec.pad_dim, spec.qblock, spec.n_blocks
    if x.size(-1) != dim:
        raise ValueError(f"Expected dim {dim}, got {x.size(-1)}")
    if pad_dim != dim:
        x = F.pad(x, (0, pad_dim - dim), value=0.0)

    orig = x.shape[:-1]
    x2 = x.reshape(-1, pad_dim).reshape(-1, nb, qb)
    amax = x2.abs().amax(dim=-1)
    scale = (amax / 7.0).clamp(min=1e-8)
    q = torch.round(x2 / scale.unsqueeze(-1)).clamp(-8, 7).to(torch.int16)
    u = (q + 8).clamp(0, 15).to(torch.uint8)

    u_even = u[..., 0::2]
    u_odd = u[..., 1::2]
    packed = (u_even * 16) + u_odd  # uint8

    packed = packed.reshape(*orig, pad_dim // 2)
    return packed, scale.to(torch.float16).reshape(*orig, nb)


def dequantize_q4_0(packed: torch.Tensor, scale: torch.Tensor, spec: QuantSpec) -> torch.Tensor:
    if spec.kind != "q4_0":
        raise ValueError(spec.kind)
    dim, pad_dim, qb, nb = spec.dim, spec.pad_dim, spec.qblock, spec.n_blocks
    if packed.size(-1) != pad_dim // 2:
        raise ValueError(f"Expected packed last dim {pad_dim//2}, got {packed.size(-1)}")
    if scale.size(-1) != nb:
        raise ValueError(f"Expected scale n_blocks {nb}, got {scale.size(-1)}")

    orig = packed.shape[:-1]
    p2 = packed.reshape(-1, pad_dim // 2).to(torch.int16)
    hi = p2 // 16
    lo = p2 % 16
    u = torch.stack([hi, lo], dim=-1).reshape(-1, pad_dim)
    q = (u - 8).clamp(-8, 7).to(torch.float32)

    q = q.reshape(-1, nb, qb)
    s = scale.reshape(-1, nb).to(torch.float32).unsqueeze(-1)
    x2 = q * s
    x = x2.reshape(*orig, pad_dim)[..., :dim]
    return x


# NF4 codebook from QLoRA Appendix E / bitsandbytes (normalized to [-1, 1]).
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


def quantize_nf4(x: torch.Tensor, spec: QuantSpec) -> Tuple[torch.Tensor, torch.Tensor]:
    """NF4: non-uniform 4-bit codebook quantization (pure PyTorch research impl)."""
    if spec.kind != "nf4":
        raise ValueError(spec.kind)
    dim, pad_dim, qb, nb = spec.dim, spec.pad_dim, spec.qblock, spec.n_blocks
    if x.size(-1) != dim:
        raise ValueError(f"Expected dim {dim}, got {x.size(-1)}")
    if pad_dim != dim:
        x = F.pad(x, (0, pad_dim - dim), value=0.0)

    orig = x.shape[:-1]
    x2 = x.reshape(-1, pad_dim).reshape(-1, nb, qb)
    amax = x2.abs().amax(dim=-1)
    scale = amax.clamp(min=1e-8)

    y = (x2 / scale.unsqueeze(-1)).clamp(-1.0, 1.0)
    levels = NF4_LEVELS.to(device=y.device, dtype=torch.float32)
    diff = (y.to(torch.float32).unsqueeze(-1) - levels).abs()
    idx = diff.argmin(dim=-1).to(torch.uint8)  # 0..15

    idx_even = idx[..., 0::2]
    idx_odd = idx[..., 1::2]
    packed = (idx_even * 16) + idx_odd
    packed = packed.reshape(*orig, pad_dim // 2)
    return packed, scale.to(torch.float16).reshape(*orig, nb)


def dequantize_nf4(packed: torch.Tensor, scale: torch.Tensor, spec: QuantSpec) -> torch.Tensor:
    if spec.kind != "nf4":
        raise ValueError(spec.kind)
    dim, pad_dim, qb, nb = spec.dim, spec.pad_dim, spec.qblock, spec.n_blocks
    if packed.size(-1) != pad_dim // 2:
        raise ValueError(f"Expected packed last dim {pad_dim//2}, got {packed.size(-1)}")
    if scale.size(-1) != nb:
        raise ValueError(f"Expected scale n_blocks {nb}, got {scale.size(-1)}")

    orig = packed.shape[:-1]
    p2 = packed.reshape(-1, pad_dim // 2).to(torch.int16)
    hi = p2 // 16
    lo = p2 % 16
    idx = torch.stack([hi, lo], dim=-1).reshape(-1, pad_dim).to(torch.long)

    levels = NF4_LEVELS.to(device=packed.device, dtype=torch.float32)
    q = levels[idx]

    q = q.reshape(-1, nb, qb)
    s = scale.reshape(-1, nb).to(torch.float32).unsqueeze(-1)
    x2 = q * s
    x = x2.reshape(*orig, pad_dim)[..., :dim]
    return x


class SeqCacheTensor:
    """A [B, max_seq_len, dim] sequence tensor stored in fp16/fp32/q8_0/q4_0/nf4."""

    def __init__(
        self,
        *,
        batch_size: int,
        max_seq_len: int,
        dim: int,
        cfg: KVCacheTensorConfig,
        device: torch.device,
    ):
        self.kind: KVCacheKind = cfg.kind
        self.device = device
        self.spec = make_quantspec(cfg.kind, dim, cfg.qblock)
        self.pos = 0
        self.max_seq_len = max_seq_len

        self.residual_len = int(max(0, cfg.residual_len))
        self._residual: Optional[torch.Tensor]
        self._residual_len_eff = min(self.residual_len, max_seq_len) if self.residual_len > 0 else 0

        if self.kind in ("fp16", "fp32"):
            dtype = torch.float16 if self.kind == "fp16" else torch.float32
            self.buf = torch.empty((batch_size, max_seq_len, dim), device=device, dtype=dtype)
            self.q = None
            self.s = None
            self._residual = None
        elif self.kind == "q8_0":
            self.buf = None
            self.q = torch.empty((batch_size, max_seq_len, self.spec.pad_dim), device=device, dtype=torch.int8)
            self.s = torch.empty((batch_size, max_seq_len, self.spec.n_blocks), device=device, dtype=torch.float16)
            self._residual = torch.empty((batch_size, self._residual_len_eff, dim), device=device, dtype=torch.float16) if self._residual_len_eff > 0 else None
        elif self.kind in ("q4_0", "nf4"):
            self.buf = None
            self.q = torch.empty((batch_size, max_seq_len, self.spec.pad_dim // 2), device=device, dtype=torch.uint8)
            self.s = torch.empty((batch_size, max_seq_len, self.spec.n_blocks), device=device, dtype=torch.float16)
            self._residual = torch.empty((batch_size, self._residual_len_eff, dim), device=device, dtype=torch.float16) if self._residual_len_eff > 0 else None
        else:
            raise ValueError(self.kind)

    @property
    def is_quantized(self) -> bool:
        return self.kind not in ("fp16", "fp32")

    def _residual_start(self) -> int:
        if self._residual is None:
            return self.pos
        return max(0, self.pos - self._residual_len_eff)

    def _residual_gather(self, start: int, end: int) -> torch.Tensor:
        if self._residual is None:
            raise RuntimeError("No residual buffer allocated")
        if not (0 <= start <= end <= self.pos):
            raise ValueError(f"Invalid residual slice {start}:{end} for pos={self.pos}")
        rlen = self._residual_len_eff
        if rlen <= 0:
            raise RuntimeError("Residual length is 0")
        idx = torch.arange(start, end, device=self.device, dtype=torch.long) % rlen
        return self._residual.index_select(1, idx)

    def append(self, x_new: torch.Tensor) -> int:
        B, Tn, D = x_new.shape
        if D != self.spec.dim:
            raise ValueError(f"dim mismatch: expected {self.spec.dim}, got {D}")
        if self.pos + Tn > self.max_seq_len:
            raise ValueError(f"Cache overflow: pos {self.pos} + {Tn} > max {self.max_seq_len}")
        old = self.pos

        if self.kind in ("fp16", "fp32"):
            self.buf[:, old : old + Tn] = x_new.to(self.buf.dtype)  # type: ignore[index]
        elif self.kind == "q8_0":
            q, s = quantize_q8_0(x_new, self.spec)
            self.q[:, old : old + Tn] = q  # type: ignore[index]
            self.s[:, old : old + Tn] = s  # type: ignore[index]
        elif self.kind == "q4_0":
            q, s = quantize_q4_0(x_new, self.spec)
            self.q[:, old : old + Tn] = q  # type: ignore[index]
            self.s[:, old : old + Tn] = s  # type: ignore[index]
        elif self.kind == "nf4":
            q, s = quantize_nf4(x_new, self.spec)
            self.q[:, old : old + Tn] = q  # type: ignore[index]
            self.s[:, old : old + Tn] = s  # type: ignore[index]
        else:
            raise ValueError(self.kind)

        if self._residual is not None:
            rlen = self._residual_len_eff
            if rlen > 0:
                if Tn >= rlen:
                    x_tail = x_new[:, -rlen:].to(torch.float16)
                    idx = torch.arange(old + Tn - rlen, old + Tn, device=self.device, dtype=torch.long) % rlen
                    self._residual[:, idx] = x_tail
                else:
                    x_fp16 = x_new.to(torch.float16)
                    idx = torch.arange(old, old + Tn, device=self.device, dtype=torch.long) % rlen
                    self._residual[:, idx] = x_fp16

        self.pos += Tn
        return old

    def truncate(self, new_pos: int) -> None:
        """Rollback the logical cache length to `new_pos` (does not clear underlying storage).

        This is used by speculative decoding, which may temporarily append draft tokens and then
        rollback if the verifier rejects them.
        """
        new_pos = int(new_pos)
        if new_pos < 0 or new_pos > int(self.pos):
            raise ValueError(f"Invalid truncate new_pos={new_pos} for pos={self.pos}")
        if new_pos == int(self.pos):
            return
        self.pos = new_pos

        # Rebuild residual ring to match the new tail. This keeps get_slice() correct for newest tokens.
        if self._residual is None:
            return
        rlen = int(self._residual_len_eff)
        if rlen <= 0:
            return
        start = max(0, self.pos - rlen)
        end = self.pos
        if start >= end:
            return

        # Gather tail from authoritative storage (buf or quant buffers) without using the residual fast-path.
        if self.kind in ("fp16", "fp32"):
            tail = self.buf[:, start:end].to(torch.float16)  # type: ignore[index]
        elif self.kind == "q8_0":
            tail = dequantize_q8_0(self.q[:, start:end], self.s[:, start:end], self.spec).to(torch.float16)  # type: ignore[index]
        elif self.kind == "q4_0":
            tail = dequantize_q4_0(self.q[:, start:end], self.s[:, start:end], self.spec).to(torch.float16)  # type: ignore[index]
        elif self.kind == "nf4":
            tail = dequantize_nf4(self.q[:, start:end], self.s[:, start:end], self.spec).to(torch.float16)  # type: ignore[index]
        else:
            raise ValueError(self.kind)

        idx = torch.arange(start, end, device=self.device, dtype=torch.long) % rlen
        self._residual[:, idx] = tail

    def get_slice(self, start: int, end: int, *, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        start = int(start)
        end = int(end)
        if start < 0 or end < start:
            raise ValueError(f"Invalid slice {start}:{end}")
        if end > self.pos:
            raise ValueError(f"Requested end {end} > cached length {self.pos}")
        if start == end:
            B = (self.buf.size(0) if self.buf is not None else self.q.size(0))  # type: ignore[union-attr]
            return torch.empty((B, 0, self.spec.dim), device=self.device, dtype=dtype)

        if self.kind in ("fp16", "fp32"):
            return self.buf[:, start:end].to(dtype)  # type: ignore[index]

        r_start = self._residual_start()
        if self._residual is not None and start >= r_start:
            return self._residual_gather(start, end).to(dtype)

        if self._residual is not None and end > r_start and start < r_start:
            a = self.get_slice(start, r_start, dtype=dtype)
            b = self._residual_gather(r_start, end).to(dtype)
            return torch.cat([a, b], dim=1)

        if self.kind == "q8_0":
            x = dequantize_q8_0(self.q[:, start:end], self.s[:, start:end], self.spec)  # type: ignore[index]
            return x.to(dtype)
        if self.kind == "q4_0":
            x = dequantize_q4_0(self.q[:, start:end], self.s[:, start:end], self.spec)  # type: ignore[index]
            return x.to(dtype)
        if self.kind == "nf4":
            x = dequantize_nf4(self.q[:, start:end], self.s[:, start:end], self.spec)  # type: ignore[index]
            return x.to(dtype)
        raise ValueError(self.kind)

    def get(self, length: Optional[int] = None, *, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        L = self.pos if length is None else int(length)
        return self.get_slice(0, L, dtype=dtype)


class LayerKVCache:
    def __init__(
        self,
        *,
        batch_size: int,
        max_seq_len: int,
        k_dim: int,
        v_dim: int,
        k_cfg: KVCacheTensorConfig,
        v_cfg: KVCacheTensorConfig,
        device: torch.device,
    ):
        self.k = SeqCacheTensor(batch_size=batch_size, max_seq_len=max_seq_len, dim=k_dim, cfg=k_cfg, device=device)
        self.v = SeqCacheTensor(batch_size=batch_size, max_seq_len=max_seq_len, dim=v_dim, cfg=v_cfg, device=device)

    @property
    def pos(self) -> int:
        return self.k.pos

    def append(self, k_new: torch.Tensor, v_new: torch.Tensor) -> int:
        old = self.k.append(k_new)
        old2 = self.v.append(v_new)
        if old != old2:
            raise RuntimeError("K/V cache desync")
        return old

    def get(self, *, dtype: torch.dtype = torch.float32) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.k.get(dtype=dtype), self.v.get(dtype=dtype)

    def truncate(self, new_pos: int) -> None:
        self.k.truncate(new_pos)
        self.v.truncate(new_pos)
        if self.k.pos != self.v.pos:
            raise RuntimeError("K/V cache desync after truncate")


class DecoupledLayerKVCache:
    def __init__(
        self,
        *,
        batch_size: int,
        max_seq_len: int,
        k_sem_dim: int,
        k_geo_dim: int,
        v_dim: int,
        k_sem_cfg: KVCacheTensorConfig,
        k_geo_cfg: KVCacheTensorConfig,
        v_cfg: KVCacheTensorConfig,
        device: torch.device,
    ):
        self.k_sem = SeqCacheTensor(batch_size=batch_size, max_seq_len=max_seq_len, dim=k_sem_dim, cfg=k_sem_cfg, device=device)
        self.k_geo = SeqCacheTensor(batch_size=batch_size, max_seq_len=max_seq_len, dim=k_geo_dim, cfg=k_geo_cfg, device=device)
        self.v = SeqCacheTensor(batch_size=batch_size, max_seq_len=max_seq_len, dim=v_dim, cfg=v_cfg, device=device)

    @property
    def pos(self) -> int:
        return self.k_sem.pos

    def append(self, k_sem_new: torch.Tensor, k_geo_new: torch.Tensor, v_new: torch.Tensor) -> int:
        old = self.k_sem.append(k_sem_new)
        old2 = self.k_geo.append(k_geo_new)
        old3 = self.v.append(v_new)
        if not (old == old2 == old3):
            raise RuntimeError("Decoupled cache desync")
        return old

    def get(self, *, dtype: torch.dtype = torch.float32) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.k_sem.get(dtype=dtype), self.k_geo.get(dtype=dtype), self.v.get(dtype=dtype)

    def truncate(self, new_pos: int) -> None:
        self.k_sem.truncate(new_pos)
        self.k_geo.truncate(new_pos)
        self.v.truncate(new_pos)
        if not (self.k_sem.pos == self.k_geo.pos == self.v.pos):
            raise RuntimeError("Decoupled cache desync after truncate")


