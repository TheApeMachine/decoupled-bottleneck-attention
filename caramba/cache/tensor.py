from __future__ import annotations
from typing import TYPE_CHECKING

import torch

from caramba.cache import QuantSpec, make_quantspec
from caramba.optimizer.quantizer import Quantizer
from caramba.config.kvcache import KVCacheTensorConfig, KVCacheKind


class SeqCacheTensor:
    """
    SeqCacheTensor stores a [B, max_seq_len, dim] tensor in fp16/fp32/q8_0/q4_0/nf4.
    """
    kind: KVCacheKind
    device: torch.device
    spec: QuantSpec
    pos: int
    max_seq_len: int
    residual_len: int
    _residual: torch.Tensor | None
    _residual_len_eff: int
    buf: torch.Tensor | None
    q: torch.Tensor | None
    s: torch.Tensor | None

    def __init__(
        self,
        *,
        batch_size: int,
        max_seq_len: int,
        dim: int,
        cfg: KVCacheTensorConfig,
        device: torch.device,
    ) -> None:
        self.quantizer = Quantizer()
        self.kind = cfg.kind
        self.device = device
        self.spec = make_quantspec(cfg.kind, int(dim), int(cfg.qblock))
        self.pos = 0
        self.max_seq_len = int(max_seq_len)

        self.residual_len = int(max(0, cfg.residual_len))
        self._residual = None
        self._residual_len_eff = (
            min(int(self.residual_len), int(self.max_seq_len)) if self.residual_len > 0 else 0
        )

        if self.kind in ("fp16", "fp32"):
            dtype = torch.float16 if self.kind == "fp16" else torch.float32
            self.buf = torch.empty(
                (int(batch_size), int(self.max_seq_len), int(dim)),
                device=device,
                dtype=dtype,
            )
            self.q = None
            self.s = None
            self._residual = None
        elif self.kind == "q8_0":
            self.buf = None
            self.q = torch.empty(
                (int(batch_size), int(self.max_seq_len), int(self.spec.pad_dim)),
                device=device,
                dtype=torch.int8,
            )
            self.s = torch.empty(
                (int(batch_size), int(self.max_seq_len), int(self.spec.n_blocks)),
                device=device,
                dtype=torch.float16,
            )
            self._residual = (
                torch.empty(
                    (int(batch_size), int(self._residual_len_eff), int(dim)),
                    device=device,
                    dtype=torch.float16,
                )
                if self._residual_len_eff > 0
                else None
            )
        elif self.kind in ("q4_0", "nf4"):
            self.buf = None
            self.q = torch.empty(
                (int(batch_size), int(self.max_seq_len), int(self.spec.pad_dim // 2)),
                device=device,
                dtype=torch.uint8,
            )
            self.s = torch.empty(
                (int(batch_size), int(self.max_seq_len), int(self.spec.n_blocks)),
                device=device,
                dtype=torch.float16,
            )
            self._residual = (
                torch.empty(
                    (int(batch_size), int(self._residual_len_eff), int(dim)),
                    device=device,
                    dtype=torch.float16,
                )
                if self._residual_len_eff > 0
                else None
            )
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
        _b, tn, d = x_new.shape
        if d != self.spec.dim:
            raise ValueError(f"dim mismatch: expected {self.spec.dim}, got {d}")
        if self.pos + tn > self.max_seq_len:
            raise ValueError(
                f"Cache overflow: pos {self.pos} + {tn} > max {self.max_seq_len}"
            )
        old = self.pos

        def apply(op):
            qv, sv = op(x_new, self.spec)
            if self.q is None or self.s is None:
                raise RuntimeError("Expected quant buffers for q8_0 cache")
            self.q[:, old : old + tn] = qv
            self.s[:, old : old + tn] = sv

        if self.kind in ("fp16", "fp32"):
            if self.buf is None:
                raise RuntimeError("Expected fp buffer for fp16/fp32 cache")
            self.buf[:, old : old + tn] = x_new.to(self.buf.dtype)
        elif self.kind == "q8_0":
            apply(self.quantizer.quantize_q8_0)
        elif self.kind == "q4_0":
            apply(self.quantizer.quantize_q4_0)
        elif self.kind == "nf4":
            apply(self.quantizer.quantize_nf4)
        else:
            raise ValueError(self.kind)

        if self._residual is not None:
            rlen = self._residual_len_eff
            if rlen > 0:
                if tn >= rlen:
                    x_tail = x_new[:, -rlen:].to(torch.float16)
                    idx = torch.arange(
                        old + tn - rlen,
                        old + tn,
                        device=self.device,
                        dtype=torch.long,
                    ) % rlen
                    self._residual[:, idx] = x_tail
                else:
                    x_fp16 = x_new.to(torch.float16)
                    idx = torch.arange(old, old + tn, device=self.device, dtype=torch.long) % rlen
                    self._residual[:, idx] = x_fp16

        self.pos += tn
        return old

    def get_slice(
        self,
        start: int,
        end: int,
        *,
        dtype: torch.dtype = torch.float16,
    ) -> torch.Tensor:
        start = int(start)
        end = int(end)
        if start < 0 or end < start:
            raise ValueError(f"Invalid slice {start}:{end}")
        if end > self.pos:
            raise ValueError(f"Requested end {end} > cached length {self.pos}")

        if start == end:
            if self.buf is not None:
                bsz = int(self.buf.size(0))
            elif self.q is not None:
                bsz = int(self.q.size(0))
            else:
                raise RuntimeError("Cache storage is not initialized")
            return torch.empty((bsz, 0, self.spec.dim), device=self.device, dtype=dtype)

        if self.kind in ("fp16", "fp32"):
            if self.buf is None:
                raise RuntimeError("Expected fp buffer for fp16/fp32 cache")
            return self.buf[:, start:end].to(dtype)

        r_start = self._residual_start()
        if self._residual is not None and start >= r_start:
            return self._residual_gather(start, end).to(dtype)

        if self._residual is not None and end > r_start and start < r_start:
            a = self.get_slice(start, r_start, dtype=dtype)
            b = self._residual_gather(r_start, end).to(dtype)
            return torch.cat([a, b], dim=1)

        if self.kind == "q8_0":
            if self.q is None or self.s is None:
                raise RuntimeError("Expected quant buffers for q8_0 cache")
            x = self.quantizer.dequantize_q8_0(self.q[:, start:end], self.s[:, start:end], self.spec)
            return x.to(dtype)
        if self.kind == "q4_0":
            if self.q is None or self.s is None:
                raise RuntimeError("Expected quant buffers for q4_0 cache")
            x = self.quantizer.dequantize_q4_0(self.q[:, start:end], self.s[:, start:end], self.spec)
            return x.to(dtype)
        if self.kind == "nf4":
            if self.q is None or self.s is None:
                raise RuntimeError("Expected quant buffers for nf4 cache")
            x = self.quantizer.dequantize_nf4(self.q[:, start:end], self.s[:, start:end], self.spec)
            return x.to(dtype)
        raise ValueError(self.kind)

    def get(self, *, dtype: torch.dtype = torch.float16) -> torch.Tensor:
        return self.get_slice(0, int(self.pos), dtype=dtype)

    def truncate(self, new_pos: int) -> None:
        """Rollback the logical cache length to new_pos (for speculative decoding)."""
        new_pos = int(new_pos)
        if new_pos < 0 or new_pos > int(self.pos):
            raise ValueError(f"Invalid truncate new_pos={new_pos} for pos={self.pos}")
        if new_pos == int(self.pos):
            return
        self.pos = new_pos

        # Rebuild residual ring to match new tail
        if self._residual is None:
            return
        rlen = int(self._residual_len_eff)
        if rlen <= 0:
            return
        start = max(0, self.pos - rlen)
        end = self.pos
        if start >= end:
            return

        # Gather tail from authoritative storage
        if self.kind in ("fp16", "fp32"):
            if self.buf is None:
                raise RuntimeError("Expected fp buffer for fp16/fp32 cache")
            tail = self.buf[:, start:end].to(torch.float16)
        elif self.kind == "q8_0":
            if self.q is None or self.s is None:
                raise RuntimeError("Expected quant buffers for q8_0 cache")
            tail = self.quantizer.dequantize_q8_0(
                self.q[:, start:end], self.s[:, start:end], self.spec
            ).to(torch.float16)
        elif self.kind == "q4_0":
            if self.q is None or self.s is None:
                raise RuntimeError("Expected quant buffers for q4_0 cache")
            tail = self.quantizer.dequantize_q4_0(
                self.q[:, start:end], self.s[:, start:end], self.spec
            ).to(torch.float16)
        elif self.kind == "nf4":
            if self.q is None or self.s is None:
                raise RuntimeError("Expected quant buffers for nf4 cache")
            tail = self.quantizer.dequantize_nf4(
                self.q[:, start:end], self.s[:, start:end], self.spec
            ).to(torch.float16)
        else:
            raise ValueError(self.kind)

        idx = torch.arange(start, end, device=self.device, dtype=torch.long) % rlen
        self._residual[:, idx] = tail
