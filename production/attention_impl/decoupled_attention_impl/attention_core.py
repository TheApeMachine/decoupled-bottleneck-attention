"""Decoupled bottleneck attention (core implementation).

Why this exists:
- We support multiple attention layouts (baseline/bottleneck/decoupled/GQA) in one module so
  experiments can compare architectures without duplicating code.
- Optional fused Triton kernels live behind runtime guards so the project imports and type-checks
  cleanly on machines without Triton (CPU/MPS).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import TYPE_CHECKING, Protocol, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing_extensions import final, override

from production.attention_impl.decoupled_attention_impl.helpers import (
    decoupled_qk_cat, decoupled_scores_f32, neg_inf
)
from production.attention_impl.decoupled_attention_impl.kernels_q4q8q4 import (
    kv_decode_partition_stats_decoupled_q4q8q4,
    kv_decode_reduce_partitions,
    kv_decode_update_decoupled_q4q8q4,
)
from production.attention_impl.decoupled_attention_impl.triton_runtime import TRITON_AVAILABLE
from production.kvcache_backend import DecoupledLayerKVCache, LayerKVCache, SeqCacheTensor
from production.rope import RotaryEmbedding

if TYPE_CHECKING:
    from production.model.config import ModelConfig


def _normalize_attn_mode(mode: object) -> str:
    """Normalize mode inputs to canonical strings ("standard","gqa","bottleneck","decoupled")."""
    v = getattr(mode, "value", mode)
    # Treat `None` as "unset"; preserve falsy-but-meaningful values (0/False) by not using `v or ""`.
    s = "" if v is None else str(v).strip().lower()
    if s == "":
        raise ValueError(
            f'attn_mode is unset/empty (got {v!r}). Expected one of: "standard" (aliases: "baseline","base"), "gqa", "bottleneck", "decoupled".'
        )
    if s in ("baseline", "standard", "base"):
        return "standard"
    if s in ("gqa", "bottleneck", "decoupled"):
        return s
    raise ValueError(
        f'Unknown attn_mode={v!r} (normalized={s!r}). Accepted aliases: ("standard"/"baseline"/"base"), ("gqa"/"bottleneck"/"decoupled").'
    )


class _Kernel(Protocol):
    """Minimal Triton kernel interface (`kernel[grid](...)`)."""

    def __getitem__(self, grid: tuple[int, ...]) -> Callable[..., object]: ...


_decoupled_qk_cat = decoupled_qk_cat
_decoupled_scores_f32 = decoupled_scores_f32
_kv_decode_update_decoupled_q4q8q4 = cast(_Kernel | None, kv_decode_update_decoupled_q4q8q4)
_kv_decode_partition_stats_decoupled_q4q8q4 = cast(_Kernel | None, kv_decode_partition_stats_decoupled_q4q8q4)
_kv_decode_reduce_partitions = cast(_Kernel | None, kv_decode_reduce_partitions)

def _sdpa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    attn_mask: torch.Tensor | None,
    dropout_p: float,
    is_causal: bool,
) -> torch.Tensor:
    """Why: torch stubs for SDPA vary; we call it via a typed getattr to avoid `Any` leaks."""
    fn_obj = getattr(F, "scaled_dot_product_attention", None)
    if not callable(fn_obj):
        raise RuntimeError("scaled_dot_product_attention is not available in this build of torch")
    def _call(fn: Callable[..., object], /, *args: object, **kwargs: object) -> object:
        """Why: keep the SDPA call typed even when linters can't follow `cast(callable)`."""
        return fn(*args, **kwargs)

    out = _call(
        fn_obj,
        q,
        k,
        v,
        attn_mask=attn_mask,
        dropout_p=float(dropout_p),
        is_causal=bool(is_causal),
    )
    return cast(torch.Tensor, out)


@final
class DecoupledBottleneckAttention(nn.Module):
    def __init__(self, cfg: "ModelConfig"):
        # Why: torch's stubs mark `nn.Module.__init__` as partially unknown in strict mode.
        # - For type-checking, call the base init via a typed cast (avoids `Unknown` call diagnostics).
        # - At runtime, call `super().__init__()` (canonical, preserves nn.Module semantics).
        if TYPE_CHECKING:
            cast(Callable[[nn.Module], None], nn.Module.__init__)(self)
        else:
            super().__init__()
        self.cfg = cfg
        n_head = int(cfg.n_head)
        self.H = n_head
        self.H_kv = n_head
        self.group_size = 1
        self.drop = nn.Dropout(cfg.dropout)

        # Projection modules are mode-dependent; annotate up-front so forward paths can be strict.
        self.q_proj: nn.Linear | None
        self.k_proj: nn.Linear | None
        self.q_sem: nn.Linear | None
        self.k_sem: nn.Linear | None
        self.q_geo: nn.Linear | None
        self.k_geo: nn.Linear | None
        self.v_proj: nn.Linear
        self.out_proj: nn.Linear

        # Head dims are intentionally mode-dependent.
        self.qk_head_dim: int | None
        self.sem_head_dim: int | None
        self.geo_head_dim: int | None
        self.v_head_dim: int

        # Optional mask tokens (null attention).
        self.k_null: torch.Tensor | None
        self.v_null: torch.Tensor | None
        self.k_sem_null: torch.Tensor | None
        self.k_geo_null: torch.Tensor | None

        # (Decoupled) Optional per-head sem/geo mixing gate (created only in decoupled mode).
        self.decoupled_gate_logit: torch.Tensor | None = None
        self.decoupled_gate_proj: nn.Linear | None = None

        self.long_seq_mem_k_linear: nn.Linear | None = None
        self.long_seq_mem_v_linear: nn.Linear | None = None
        self.long_seq_mem_k_conv: nn.Conv1d | None = None
        self.long_seq_mem_v_conv: nn.Conv1d | None = None
        self.long_seq_mem_k_alpha: torch.Tensor | None = None
        self.long_seq_mem_v_alpha: torch.Tensor | None = None

        def must_div(name: str, total: int, denom: int) -> int:
            if total % denom != 0:
                raise ValueError(f"{name} ({total}) must be divisible by {denom}")
            return total // denom

        mode = _normalize_attn_mode(cfg.attn_mode)
        match mode:
            case "standard":
                qk_dim = cfg.d_model
                v_dim = cfg.d_model
                self.qk_head_dim = must_div("d_model", qk_dim, n_head)
                self.v_head_dim = must_div("d_model", v_dim, n_head)

                self.q_proj = nn.Linear(cfg.d_model, qk_dim, bias=False)
                self.k_proj = self.q_proj if cfg.tie_qk else nn.Linear(cfg.d_model, qk_dim, bias=False)
                self.v_proj = nn.Linear(cfg.d_model, v_dim, bias=False)
                self.out_proj = nn.Linear(v_dim, cfg.d_model, bias=False)

                self.q_sem = self.k_sem = self.q_geo = self.k_geo = None
                self.sem_head_dim = self.geo_head_dim = None
                self.rotary = RotaryEmbedding(self.qk_head_dim, base=cfg.rope_base) if cfg.rope else None

                self.k_null = nn.Parameter(torch.zeros(1, 1, qk_dim)) if cfg.null_attn else None
                self.v_null = nn.Parameter(torch.zeros(1, 1, v_dim)) if cfg.null_attn else None
                self.k_sem_null = self.k_geo_null = None

            case "bottleneck":
                qk_dim = cfg.attn_dim
                v_dim = cfg.attn_dim
                self.qk_head_dim = must_div("attn_dim", qk_dim, n_head)
                self.v_head_dim = must_div("attn_dim", v_dim, n_head)
                if cfg.rope and (self.qk_head_dim % 2 != 0):
                    raise ValueError("RoPE needs even head dim; pick attn_dim divisible by 2*n_head")

                self.q_proj = nn.Linear(cfg.d_model, qk_dim, bias=False)
                self.k_proj = self.q_proj if cfg.tie_qk else nn.Linear(cfg.d_model, qk_dim, bias=False)
                self.v_proj = nn.Linear(cfg.d_model, v_dim, bias=False)
                self.out_proj = nn.Linear(v_dim, cfg.d_model, bias=False)

                self.q_sem = self.k_sem = self.q_geo = self.k_geo = None
                self.sem_head_dim = self.geo_head_dim = None
                self.rotary = RotaryEmbedding(self.qk_head_dim, base=cfg.rope_base) if cfg.rope else None

                self.k_null = nn.Parameter(torch.zeros(1, 1, qk_dim)) if cfg.null_attn else None
                self.v_null = nn.Parameter(torch.zeros(1, 1, v_dim)) if cfg.null_attn else None
                self.k_sem_null = self.k_geo_null = None

            case "gqa":
                kv_head = int(cfg.kv_head if cfg.kv_head is not None else n_head)
                if kv_head <= 0:
                    raise ValueError("kv_head must be > 0")
                if n_head % kv_head != 0:
                    raise ValueError(
                        f"gqa requires n_head % kv_head == 0 (got n_head={n_head}, kv_head={kv_head})"
                    )
                self.H_kv = kv_head
                self.group_size = n_head // kv_head

                self.qk_head_dim = must_div("attn_dim", cfg.attn_dim, n_head)
                self.v_head_dim = self.qk_head_dim
                kv_dim = kv_head * self.qk_head_dim

                if cfg.rope and (self.qk_head_dim % 2 != 0):
                    raise ValueError(
                        "RoPE requires an even head dim. Choose attn_dim divisible by 2*n_head."
                    )

                if cfg.tie_qk:
                    raise ValueError(
                        "tie_qk is not supported for gqa unless kv_head == n_head (use attn_mode=standard)."
                    )

                self.q_proj = nn.Linear(cfg.d_model, cfg.attn_dim, bias=False)
                self.k_proj = nn.Linear(cfg.d_model, kv_dim, bias=False)
                self.v_proj = nn.Linear(cfg.d_model, kv_dim, bias=False)
                self.out_proj = nn.Linear(cfg.attn_dim, cfg.d_model, bias=False)

                self.q_sem = self.k_sem = self.q_geo = self.k_geo = None
                self.sem_head_dim = self.geo_head_dim = None
                self.rotary = RotaryEmbedding(self.qk_head_dim, base=cfg.rope_base) if cfg.rope else None

                self.k_null = nn.Parameter(torch.zeros(1, 1, kv_dim)) if cfg.null_attn else None
                self.v_null = nn.Parameter(torch.zeros(1, 1, kv_dim)) if cfg.null_attn else None
                self.k_sem_null = self.k_geo_null = None

            case "decoupled":
                self.sem_head_dim = must_div("sem_dim", cfg.sem_dim, n_head)
                self.geo_head_dim = must_div("geo_dim", cfg.geo_dim, n_head)
                self.v_head_dim = must_div("attn_dim", cfg.attn_dim, n_head)
                if cfg.rope and (self.geo_head_dim % 2 != 0):
                    raise ValueError("RoPE needs even geo_head_dim; pick geo_dim divisible by 2*n_head")

                # Semantic/content path: intentionally NO RoPE. Models lexical/content similarity only.
                self.q_sem = nn.Linear(cfg.d_model, cfg.sem_dim, bias=False)
                self.k_sem = self.q_sem if cfg.tie_qk else nn.Linear(cfg.d_model, cfg.sem_dim, bias=False)
                # Geometric/position path: RoPE applied here ONLY. Models relative position similarity.
                self.q_geo = nn.Linear(cfg.d_model, cfg.geo_dim, bias=False)
                self.k_geo = self.q_geo if cfg.tie_qk else nn.Linear(cfg.d_model, cfg.geo_dim, bias=False)

                self.v_proj = nn.Linear(cfg.d_model, cfg.attn_dim, bias=False)
                self.out_proj = nn.Linear(cfg.attn_dim, cfg.d_model, bias=False)

                self.q_proj = self.k_proj = None
                self.qk_head_dim = None
                self.rotary = RotaryEmbedding(self.geo_head_dim, base=cfg.rope_base) if cfg.rope else None

                self.k_sem_null = nn.Parameter(torch.zeros(1, 1, cfg.sem_dim)) if cfg.null_attn else None
                self.k_geo_null = nn.Parameter(torch.zeros(1, 1, cfg.geo_dim)) if cfg.null_attn else None
                self.v_null = nn.Parameter(torch.zeros(1, 1, cfg.attn_dim)) if cfg.null_attn else None
                self.k_null = None
                if cfg.decoupled_gate:
                    self.decoupled_gate_logit = nn.Parameter(torch.zeros(n_head))
                    if bool(cfg.decoupled_gate_dynamic):
                        self.decoupled_gate_proj = nn.Linear(cfg.d_model, n_head, bias=False)
                        nn.init.zeros_(self.decoupled_gate_proj.weight)

                sem_head_dim = int(self.sem_head_dim)
                v_head_dim = int(self.v_head_dim)
                self.long_seq_mem_k_linear = nn.Linear(sem_head_dim, sem_head_dim, bias=False)
                self.long_seq_mem_v_linear = nn.Linear(v_head_dim, v_head_dim, bias=False)
                self.long_seq_mem_k_conv = nn.Conv1d(
                    sem_head_dim,
                    sem_head_dim,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    groups=sem_head_dim,
                    bias=False,
                )
                self.long_seq_mem_v_conv = nn.Conv1d(
                    v_head_dim,
                    v_head_dim,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    groups=v_head_dim,
                    bias=False,
                )
                self.long_seq_mem_k_alpha = nn.Parameter(torch.zeros(()))
                self.long_seq_mem_v_alpha = nn.Parameter(torch.zeros(()))

            case _:
                # `_normalize_attn_mode` returns a string; keep match exhaustive and fail loud.
                raise ValueError(f"Unknown attn_mode={mode!r}")

        self.logit_scale = nn.Parameter(torch.zeros(n_head)) if cfg.learned_temp else None
        self._flash2_scratch: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None
        self._flash2_scratch_cap = (0, 0, 0)

        # Pre-computed attention scaling factors (avoid repeated sqrt in hot paths).
        self._qk_scale: float | None = None
        self._sem_scale: float | None = None
        self._geo_scale: float | None = None
        if self.qk_head_dim is not None:
            self._qk_scale = float(1.0 / math.sqrt(float(self.qk_head_dim)))
        if self.sem_head_dim is not None:
            self._sem_scale = float(1.0 / math.sqrt(float(self.sem_head_dim)))
        if self.geo_head_dim is not None:
            self._geo_scale = float(1.0 / math.sqrt(float(self.geo_head_dim)))

    def _shape(self, x: torch.Tensor, head_dim: int, n_head: int | None = None) -> torch.Tensor:
        B, T, _D = x.shape
        n_head = self.H if n_head is None else int(n_head)
        return x.view(B, T, n_head, head_dim).transpose(1, 2).contiguous()

    def _merge(self, x: torch.Tensor) -> torch.Tensor:
        B, H, T, hd = x.shape
        return x.transpose(1, 2).contiguous().view(B, T, H * hd)

    def _apply_logit_scale_to_q(self, q: torch.Tensor) -> torch.Tensor:
        if self.logit_scale is None:
            return q
        # Stability guard: learned temperatures can drift and exp() can overflow under mixed precision.
        # Clamp in fp32, then cast to q dtype for the multiply.
        s = self.logit_scale.float().clamp(min=-8.0, max=8.0)
        scale = torch.exp(s).to(dtype=q.dtype).view(1, -1, 1, 1)
        return q * scale

    def _decoupled_gate(self, x: torch.Tensor) -> torch.Tensor | None:
        if self.decoupled_gate_logit is None:
            return None
        gate_bias = self.decoupled_gate_logit.view(1, -1, 1, 1).to(dtype=torch.float32, device=x.device)
        if self.decoupled_gate_proj is None:
            gate_logit = gate_bias
        else:
            dyn = self.decoupled_gate_proj.forward(x).transpose(1, 2).unsqueeze(-1).to(dtype=torch.float32)
            gate_logit = gate_bias + dyn
        return torch.sigmoid(gate_logit).to(dtype=x.dtype)

    def _sdp(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: torch.Tensor | None,
        *,
        scale: float | None = None,
        is_causal: bool | None = None,
    ) -> torch.Tensor:
        dropout_p = self.cfg.dropout if self.training else 0.0
        if is_causal is None:
            is_causal = attn_mask is None

        # SDPA defaults to an implicit scale of 1/sqrt(dk). To request an explicit `scale`, we emulate it
        # by pre-scaling q so we can stay on the most stable/portable SDPA call signature (no `scale=` kwarg).
        if scale is not None:
            dk = int(q.size(-1))
            q = q * (float(scale) * math.sqrt(dk))

        return _sdpa(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=bool(is_causal))

    def _streaming_decode_attn(
        self,
        *,
        q: torch.Tensor,
        k_cache: SeqCacheTensor,
        v_cache: SeqCacheTensor,
        head_dim: int,
        decode_block: int,
        scale: float,
        v_head_dim: int | None = None,
        k_null: torch.Tensor | None = None,
        v_null: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, H, Tq, _hd = q.shape
        assert Tq == 1
        if v_head_dim is None:
            v_head_dim = head_dim

        cache_len = k_cache.pos
        if cache_len != v_cache.pos:
            raise RuntimeError("K/V cache desync in streaming decode")

        compute_dtype = q.dtype
        m = torch.full((B, H, 1), -float("inf"), device=q.device, dtype=torch.float32)
        d = torch.zeros((B, H, 1), device=q.device, dtype=torch.float32)
        o = torch.zeros((B, H, 1, v_head_dim), device=q.device, dtype=torch.float32)

        qh = q.to(compute_dtype)

        def update(scores_f32: torch.Tensor, v_block_f16: torch.Tensor) -> None:
            nonlocal m, d, o
            block_max = scores_f32.amax(dim=-1)
            m_new = torch.maximum(m, block_max)
            exp_m = torch.exp(m - m_new)
            exp_scores = torch.exp(scores_f32 - m_new.unsqueeze(-1))
            exp_scores_f16 = exp_scores.to(compute_dtype)
            d = d * exp_m + exp_scores_f16.sum(dim=-1).to(torch.float32)
            o = o * exp_m.unsqueeze(-1) + torch.matmul(exp_scores_f16, v_block_f16).to(torch.float32)
            m = m_new

        if k_null is not None and v_null is not None:
            s = (qh * k_null.to(compute_dtype)).sum(dim=-1, keepdim=True).to(torch.float32) * scale
            update(s, v_null.to(compute_dtype))

        blk = int(max(1, decode_block))
        for start in range(0, cache_len, blk):
            end = min(cache_len, start + blk)
            k_blk = k_cache.get_slice(start, end, dtype=compute_dtype)
            v_blk = v_cache.get_slice(start, end, dtype=compute_dtype)
            kbh = self._shape(k_blk, head_dim)
            vbh = self._shape(v_blk, v_head_dim)
            scores = torch.matmul(qh, kbh.transpose(-2, -1)) * scale
            update(scores.to(torch.float32), vbh)

        out = o / d.clamp(min=1e-9).unsqueeze(-1)
        return out.to(q.dtype)

    def _streaming_decode_attn_decoupled(
        self,
        *,
        q_sem: torch.Tensor,
        q_geo: torch.Tensor,
        k_sem_cache: SeqCacheTensor,
        k_geo_cache: SeqCacheTensor,
        v_cache: SeqCacheTensor,
        sem_head_dim: int,
        geo_head_dim: int,
        v_head_dim: int,
        decode_block: int,
        sem_scale: float,
        geo_scale: float,
        k_sem_null: torch.Tensor | None = None,
        k_geo_null: torch.Tensor | None = None,
        v_null: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, H, Tq, _ = q_sem.shape
        assert Tq == 1
        cache_len = k_sem_cache.pos
        if not (cache_len == k_geo_cache.pos == v_cache.pos):
            raise RuntimeError("Decoupled cache desync in streaming decode")

        compute_dtype = q_sem.dtype
        m = torch.full((B, H, 1), -float("inf"), device=q_sem.device, dtype=torch.float32)
        d = torch.zeros((B, H, 1), device=q_sem.device, dtype=torch.float32)
        o = torch.zeros((B, H, 1, v_head_dim), device=q_sem.device, dtype=torch.float32)

        qsh = q_sem.to(compute_dtype)
        qgh = q_geo.to(compute_dtype)

        def update(scores_f32: torch.Tensor, v_block_f16: torch.Tensor) -> None:
            nonlocal m, d, o
            block_max = scores_f32.amax(dim=-1)
            m_new = torch.maximum(m, block_max)
            exp_m = torch.exp(m - m_new)
            exp_scores = torch.exp(scores_f32 - m_new.unsqueeze(-1))
            exp_scores_f16 = exp_scores.to(compute_dtype)
            d = d * exp_m + exp_scores_f16.sum(dim=-1).to(torch.float32)
            o = o * exp_m.unsqueeze(-1) + torch.matmul(exp_scores_f16, v_block_f16).to(torch.float32)
            m = m_new

        if k_sem_null is not None and k_geo_null is not None and v_null is not None:
            s = _decoupled_scores_f32(
                q_sem=qsh,
                q_geo=qgh,
                k_sem=k_sem_null.to(compute_dtype),
                k_geo=k_geo_null.to(compute_dtype),
                sem_scale=sem_scale,
                geo_scale=geo_scale,
            )
            update(s, v_null.to(compute_dtype))

        blk = int(max(1, decode_block))
        for start in range(0, cache_len, blk):
            end = min(cache_len, start + blk)
            k_sem_blk = k_sem_cache.get_slice(start, end, dtype=compute_dtype)
            k_geo_blk = k_geo_cache.get_slice(start, end, dtype=compute_dtype)
            v_blk = v_cache.get_slice(start, end, dtype=compute_dtype)

            ksh = self._shape(k_sem_blk, sem_head_dim)
            kgh = self._shape(k_geo_blk, geo_head_dim)
            vbh = self._shape(v_blk, v_head_dim)

            # Score decomposition is explicit:
            # - semantic contribution: content similarity (no RoPE)
            # - geometric contribution: relative position similarity (RoPE)
            s = _decoupled_scores_f32(q_sem=qsh, q_geo=qgh, k_sem=ksh, k_geo=kgh, sem_scale=sem_scale, geo_scale=geo_scale)
            update(s, vbh)

        out = o / d.clamp(min=1e-9).unsqueeze(-1)
        return out.to(q_sem.dtype)

    def _fused_decode_attn_decoupled_q4q8q4(
        self,
        *,
        q_sem: torch.Tensor,
        q_geo: torch.Tensor,
        cache: DecoupledLayerKVCache,
        sem_head_dim: int,
        geo_head_dim: int,
        v_head_dim: int,
        decode_block: int,
        sem_scale: float,
        geo_scale: float,
    ) -> torch.Tensor:
        if not TRITON_AVAILABLE:
            raise RuntimeError("Triton not available")
        if q_sem.device.type != "cuda":
            raise RuntimeError("Fused decode requires CUDA")

        if not (cache.k_sem.kind == "q4_0" and cache.k_geo.kind == "q8_0" and cache.v.kind == "q4_0"):
            raise RuntimeError("Fused decode q4q8q4 requires k_sem=q4_0, k_geo=q8_0, v=q4_0")
        if not (cache.k_sem.spec.qblock == cache.k_geo.spec.qblock == cache.v.spec.qblock == 32):
            raise RuntimeError("Fused decode currently assumes qblock=32")

        B, H, Tq, _ = q_sem.shape
        assert Tq == 1
        cache_len = cache.pos

        k_sem = cache.k_sem
        rlen = int(getattr(k_sem, "_residual_len_eff", 0)) if getattr(k_sem, "_residual", None) is not None else 0
        r_start = max(0, cache_len - rlen) if rlen > 0 else cache_len
        L_prefix = int(r_start)

        q_sem2 = q_sem[:, :, 0, :].contiguous().to(torch.float16)
        q_geo2 = q_geo[:, :, 0, :].contiguous().to(torch.float16)

        has_null = bool(self.cfg.null_attn)
        if has_null:
            ksn = self._shape(cast(torch.Tensor, self.k_sem_null).expand(B, 1, -1), sem_head_dim)[:, :, 0, :].contiguous().to(torch.float16)
            kgn = self._shape(cast(torch.Tensor, self.k_geo_null).expand(B, 1, -1), geo_head_dim)[:, :, 0, :].contiguous().to(torch.float16)
            vn = self._shape(cast(torch.Tensor, self.v_null).expand(B, 1, -1), v_head_dim)[:, :, 0, :].contiguous().to(torch.float16)
        else:
            ksn = kgn = vn = q_sem2

        BH = B * H
        m = torch.full((BH,), -float("inf"), device=q_sem.device, dtype=torch.float32)
        d = torch.zeros((BH,), device=q_sem.device, dtype=torch.float32)
        o = torch.zeros((BH, v_head_dim), device=q_sem.device, dtype=torch.float32)

        if has_null and L_prefix == 0:
            s_null = (q_sem2.float() * ksn.float()).sum(dim=-1) * float(sem_scale) + (q_geo2.float() * kgn.float()).sum(dim=-1) * float(geo_scale)
            m = s_null.reshape(BH)
            d = torch.ones((BH,), device=q_sem.device, dtype=torch.float32)
            o = vn.float().reshape(BH, v_head_dim)

        if L_prefix > 0:
            block_n = int(getattr(cache, "block_n", 128)) or 128
            num_sub = max(1, int(decode_block // block_n))
            step = block_n * num_sub
            num_warps = int(getattr(cache, "num_warps_1pass", 4))
            num_stages = int(getattr(cache, "num_stages_1pass", 2))

            ksq = cache.k_sem.q
            kss = cache.k_sem.s
            kgq = cache.k_geo.q
            kgs = cache.k_geo.s
            vq = cache.v.q
            vs = cache.v.s
            assert ksq is not None and kss is not None and kgq is not None and kgs is not None and vq is not None and vs is not None

            kernel = _kv_decode_update_decoupled_q4q8q4
            if kernel is None:
                raise RuntimeError("Fused decode kernels are unavailable")

            grid = (BH,)
            launch = kernel.__getitem__(grid)
            for start in range(0, L_prefix, step):
                _ = launch(
                    q_sem2,
                    q_geo2,
                    ksn,
                    kgn,
                    vn,
                    ksq,
                    kss,
                    kgq,
                    kgs,
                    vq,
                    vs,
                    m,
                    d,
                    o,
                    start,
                    L_prefix,
                    H=H,
                    HD_SEM=sem_head_dim,
                    HD_GEO=geo_head_dim,
                    HD_V=v_head_dim,
                    QBLOCK_SEM=32,
                    QBLOCK_GEO=32,
                    QBLOCK_V=32,
                    SEM_SCALE=sem_scale,
                    GEO_SCALE=geo_scale,
                    BLOCK_N=block_n,
                    NUM_SUBBLOCKS=num_sub,
                    HAS_NULL=has_null,
                    SEED_NULL=bool(has_null and start == 0),
                    stride_qsem_b=q_sem2.stride(0),
                    stride_qsem_h=q_sem2.stride(1),
                    stride_qgeo_b=q_geo2.stride(0),
                    stride_qgeo_h=q_geo2.stride(1),
                    stride_ksn_b=ksn.stride(0),
                    stride_ksn_h=ksn.stride(1),
                    stride_kgn_b=kgn.stride(0),
                    stride_kgn_h=kgn.stride(1),
                    stride_vn_b=vn.stride(0),
                    stride_vn_h=vn.stride(1),
                    stride_ksq_b=ksq.stride(0),
                    stride_ksq_t=ksq.stride(1),
                    stride_kss_b=kss.stride(0),
                    stride_kss_t=kss.stride(1),
                    stride_kss_c=kss.stride(2),
                    stride_kgq_b=kgq.stride(0),
                    stride_kgq_t=kgq.stride(1),
                    stride_kgs_b=kgs.stride(0),
                    stride_kgs_t=kgs.stride(1),
                    stride_kgs_c=kgs.stride(2),
                    stride_vq_b=vq.stride(0),
                    stride_vq_t=vq.stride(1),
                    stride_vs_b=vs.stride(0),
                    stride_vs_t=vs.stride(1),
                    stride_vs_c=vs.stride(2),
                    stride_o=o.stride(0),
                    num_warps=num_warps,
                    num_stages=num_stages,
                )

        if L_prefix < cache_len:
            qsh = q_sem.to(torch.float16)
            qgh = q_geo.to(torch.float16)

            m_t = m.view(B, H, 1)
            d_t = d.view(B, H, 1)
            o_t = o.view(B, H, 1, v_head_dim)

            def update(scores_f32: torch.Tensor, v_block_f16: torch.Tensor) -> None:
                nonlocal m_t, d_t, o_t
                block_max = scores_f32.amax(dim=-1)
                m_new = torch.maximum(m_t, block_max)
                exp_m = torch.exp(m_t - m_new)
                exp_scores = torch.exp(scores_f32 - m_new.unsqueeze(-1)).to(torch.float16)
                d_t = d_t * exp_m + exp_scores.sum(dim=-1).to(torch.float32)
                o_t = o_t * exp_m.unsqueeze(-1) + torch.matmul(exp_scores, v_block_f16).to(torch.float32)
                m_t = m_new

            k_sem_blk = cache.k_sem.get_slice(L_prefix, cache_len, dtype=torch.float16)
            k_geo_blk = cache.k_geo.get_slice(L_prefix, cache_len, dtype=torch.float16)
            v_blk = cache.v.get_slice(L_prefix, cache_len, dtype=torch.float16)
            ksh = self._shape(k_sem_blk, sem_head_dim)
            kgh = self._shape(k_geo_blk, geo_head_dim)
            vbh = self._shape(v_blk, v_head_dim)
            s = _decoupled_scores_f32(q_sem=qsh, q_geo=qgh, k_sem=ksh, k_geo=kgh, sem_scale=sem_scale, geo_scale=geo_scale)
            update(s, vbh)

            m = m_t.view(BH)
            d = d_t.view(BH)
            o = o_t.view(BH, v_head_dim)

        out = (o / d.clamp(min=1e-9).unsqueeze(-1)).view(B, H, 1, v_head_dim)
        return out.to(q_sem.dtype)

    def _fused_decode_attn_decoupled_q4q8q4_2pass(
        self,
        *,
        q_sem: torch.Tensor,
        q_geo: torch.Tensor,
        cache: DecoupledLayerKVCache,
        sem_head_dim: int,
        geo_head_dim: int,
        v_head_dim: int,
        decode_block: int,
        sem_scale: float,
        geo_scale: float,
    ) -> torch.Tensor:
        if not TRITON_AVAILABLE:
            raise RuntimeError("Triton not available")
        if q_sem.device.type != "cuda":
            raise RuntimeError("Fused decode requires CUDA")
        if not (cache.k_sem.kind == "q4_0" and cache.k_geo.kind == "q8_0" and cache.v.kind == "q4_0"):
            raise RuntimeError("Fused decode q4q8q4 requires k_sem=q4_0, k_geo=q8_0, v=q4_0")
        if not (cache.k_sem.spec.qblock == cache.k_geo.spec.qblock == cache.v.spec.qblock == 32):
            raise RuntimeError("Fused decode currently assumes qblock=32")

        B, H, Tq, _ = q_sem.shape
        assert Tq == 1
        cache_len = cache.pos

        k_sem = cache.k_sem
        rlen = int(getattr(k_sem, "_residual_len_eff", 0)) if getattr(k_sem, "_residual", None) is not None else 0
        r_start = max(0, cache_len - rlen) if rlen > 0 else cache_len
        L_prefix = int(r_start)

        q_sem2 = q_sem[:, :, 0, :].contiguous().to(torch.float16)
        q_geo2 = q_geo[:, :, 0, :].contiguous().to(torch.float16)

        has_null = bool(self.cfg.null_attn)
        if has_null:
            ksn = self._shape(cast(torch.Tensor, self.k_sem_null).expand(B, 1, -1), sem_head_dim)[:, :, 0, :].contiguous().to(torch.float16)
            kgn = self._shape(cast(torch.Tensor, self.k_geo_null).expand(B, 1, -1), geo_head_dim)[:, :, 0, :].contiguous().to(torch.float16)
            vn = self._shape(cast(torch.Tensor, self.v_null).expand(B, 1, -1), v_head_dim)[:, :, 0, :].contiguous().to(torch.float16)
        else:
            ksn = kgn = vn = q_sem2

        BH = B * H

        # Partitioning knobs (tunable via cache attrs)
        block_n = int(getattr(cache, "block_n", 128)) or 128
        num_sub = max(1, int(decode_block // block_n))
        partition_size = block_n * num_sub
        num_parts = int(math.ceil(max(1, L_prefix) / float(partition_size))) if L_prefix > 0 else 0
        num_parts_cap = int(max(1, num_parts))
        P = int(num_parts)

        # scratch buffers (BH, P) / (BH, P, hd_v)
        need = (BH, num_parts_cap, v_head_dim)
        if self._flash2_scratch is None or any(a < b for a, b in zip(self._flash2_scratch_cap, need)):
            m_part = torch.empty((BH, num_parts_cap), device=q_sem.device, dtype=torch.float32)
            d_part = torch.empty((BH, num_parts_cap), device=q_sem.device, dtype=torch.float32)
            o_part = torch.empty((BH, num_parts_cap, v_head_dim), device=q_sem.device, dtype=torch.float32)
            self._flash2_scratch = (m_part, d_part, o_part)
            self._flash2_scratch_cap = need
        else:
            m_part, d_part, o_part = self._flash2_scratch

        m = torch.empty((BH,), device=q_sem.device, dtype=torch.float32)
        d = torch.empty((BH,), device=q_sem.device, dtype=torch.float32)
        o = torch.empty((BH, v_head_dim), device=q_sem.device, dtype=torch.float32)

        num_warps_part = int(getattr(cache, "num_warps_part", 4))
        num_stages_part = int(getattr(cache, "num_stages_part", 2))
        num_warps_reduce = int(getattr(cache, "num_warps_reduce", 1))
        num_stages_reduce = int(getattr(cache, "num_stages_reduce", 1))

        if L_prefix > 0:
            ksq = cache.k_sem.q
            kss = cache.k_sem.s
            kgq = cache.k_geo.q
            kgs = cache.k_geo.s
            vq = cache.v.q
            vs = cache.v.s
            assert ksq is not None and kss is not None and kgq is not None and kgs is not None and vq is not None and vs is not None

            kernel_part = _kv_decode_partition_stats_decoupled_q4q8q4
            if kernel_part is None:
                raise RuntimeError("Fused decode kernels are unavailable")

            grid1 = (BH, P)
            launch_part = kernel_part.__getitem__(grid1)
            _ = launch_part(
                q_sem2,
                q_geo2,
                ksq,
                kss,
                kgq,
                kgs,
                vq,
                vs,
                m_part,
                d_part,
                o_part,
                L_prefix,
                P,
                PARTITION_SIZE=partition_size,
                H=H,
                HD_SEM=sem_head_dim,
                HD_GEO=geo_head_dim,
                HD_V=v_head_dim,
                QBLOCK_SEM=32,
                QBLOCK_GEO=32,
                QBLOCK_V=32,
                SEM_SCALE=sem_scale,
                GEO_SCALE=geo_scale,
                BLOCK_N=block_n,
                NUM_SUBBLOCKS=num_sub,
                stride_qsem_b=q_sem2.stride(0),
                stride_qsem_h=q_sem2.stride(1),
                stride_qgeo_b=q_geo2.stride(0),
                stride_qgeo_h=q_geo2.stride(1),
                stride_ksq_b=ksq.stride(0),
                stride_ksq_t=ksq.stride(1),
                stride_kss_b=kss.stride(0),
                stride_kss_t=kss.stride(1),
                stride_kss_c=kss.stride(2),
                stride_kgq_b=kgq.stride(0),
                stride_kgq_t=kgq.stride(1),
                stride_kgs_b=kgs.stride(0),
                stride_kgs_t=kgs.stride(1),
                stride_kgs_c=kgs.stride(2),
                stride_vq_b=vq.stride(0),
                stride_vq_t=vq.stride(1),
                stride_vs_b=vs.stride(0),
                stride_vs_t=vs.stride(1),
                stride_vs_c=vs.stride(2),
                stride_mp_row=m_part.stride(0),
                stride_mp_part=m_part.stride(1),
                stride_dp_row=d_part.stride(0),
                stride_dp_part=d_part.stride(1),
                stride_op_row=o_part.stride(0),
                stride_op_part=o_part.stride(1),
                num_warps=num_warps_part,
                num_stages=num_stages_part,
            )

        kernel_reduce = _kv_decode_reduce_partitions
        if kernel_reduce is None:
            raise RuntimeError("Fused decode kernels are unavailable")

        num_parts_const = 1
        while num_parts_const < P:
            num_parts_const *= 2
        grid2 = (BH,)
        launch_reduce = kernel_reduce.__getitem__(grid2)
        _ = launch_reduce(
            q_sem2,
            q_geo2,
            ksn,
            kgn,
            vn,
            m_part,
            d_part,
            o_part,
            m,
            d,
            o,
            P,
            NUM_PARTS=num_parts_const,
            HAS_NULL=has_null,
            H=H,
            HD_SEM=sem_head_dim,
            HD_GEO=geo_head_dim,
            HD_V=v_head_dim,
            SEM_SCALE=sem_scale,
            GEO_SCALE=geo_scale,
            stride_qsem_b=q_sem2.stride(0),
            stride_qsem_h=q_sem2.stride(1),
            stride_qgeo_b=q_geo2.stride(0),
            stride_qgeo_h=q_geo2.stride(1),
            stride_ksn_b=ksn.stride(0),
            stride_ksn_h=ksn.stride(1),
            stride_kgn_b=kgn.stride(0),
            stride_kgn_h=kgn.stride(1),
            stride_vn_b=vn.stride(0),
            stride_vn_h=vn.stride(1),
            stride_mp_row=m_part.stride(0),
            stride_mp_part=m_part.stride(1),
            stride_dp_row=d_part.stride(0),
            stride_dp_part=d_part.stride(1),
            stride_op_row=o_part.stride(0),
            stride_op_part=o_part.stride(1),
            stride_o=o.stride(0),
            num_warps=num_warps_reduce,
            num_stages=num_stages_reduce,
        )

        # Residual tail via Python streaming updater (tiny).
        _ = cache_len
        if L_prefix < cache_len:
            qsh = q_sem.to(torch.float16)
            qgh = q_geo.to(torch.float16)
            m_t = m.view(B, H, 1)
            d_t = d.view(B, H, 1)
            o_t = o.view(B, H, 1, v_head_dim)

            def update(scores_f32: torch.Tensor, v_block_f16: torch.Tensor) -> None:
                nonlocal m_t, d_t, o_t
                block_max = scores_f32.amax(dim=-1)
                m_new = torch.maximum(m_t, block_max)
                exp_m = torch.exp(m_t - m_new)
                exp_scores = torch.exp(scores_f32 - m_new.unsqueeze(-1)).to(torch.float16)
                d_t = d_t * exp_m + exp_scores.sum(dim=-1).to(torch.float32)
                o_t = o_t * exp_m.unsqueeze(-1) + torch.matmul(exp_scores, v_block_f16).to(torch.float32)
                m_t = m_new

            k_sem_blk = cache.k_sem.get_slice(L_prefix, cache_len, dtype=torch.float16)
            k_geo_blk = cache.k_geo.get_slice(L_prefix, cache_len, dtype=torch.float16)
            v_blk = cache.v.get_slice(L_prefix, cache_len, dtype=torch.float16)
            ksh = self._shape(k_sem_blk, sem_head_dim)
            kgh = self._shape(k_geo_blk, geo_head_dim)
            vbh = self._shape(v_blk, v_head_dim)
            s = _decoupled_scores_f32(q_sem=qsh, q_geo=qgh, k_sem=ksh, k_geo=kgh, sem_scale=sem_scale, geo_scale=geo_scale)
            update(s, vbh)
            o = o_t.view(BH, v_head_dim)
            d = d_t.view(BH)

        out = (o / d.clamp(min=1e-9).unsqueeze(-1)).view(B, H, 1, v_head_dim)
        return out.to(q_sem.dtype)

    @override
    def forward(
        self,
        x: torch.Tensor,
        *,
        attn_mask: torch.Tensor | None,
        cache: DecoupledLayerKVCache | LayerKVCache | None,
        pos_offset: int,
    ) -> tuple[torch.Tensor, DecoupledLayerKVCache | LayerKVCache | None]:
        cfg = self.cfg
        B, T, _ = x.shape
        ninfty = neg_inf(x.dtype)

        mode = _normalize_attn_mode(cfg.attn_mode)

        if mode in ("standard", "bottleneck"):
            # If tie_qk is enabled, q_proj == k_proj and the projections are identical for self-attention.
            # Compute once and reuse (saves a full matmul + reshape + rotary per layer).
            q_proj = self.q_proj
            k_proj = self.k_proj
            if q_proj is None or k_proj is None or self.qk_head_dim is None:
                raise RuntimeError("baseline/bottleneck mode requires q_proj/k_proj and qk_head_dim")
            v_proj = self.v_proj
            out_proj = self.out_proj
            qk_head_dim = int(self.qk_head_dim)
            v_head_dim = int(self.v_head_dim)

            if k_proj is q_proj:
                qk = q_proj.forward(x)
                qh_base = self._shape(qk, qk_head_dim)
                if self.rotary is not None:
                    qh_base = self.rotary.rotate(qh_base, pos_offset)
                kh = qh_base
            else:
                q = q_proj.forward(x)
                k = k_proj.forward(x)
                qh_base = self._shape(q, qk_head_dim)
                kh = self._shape(k, qk_head_dim)
                if self.rotary is not None:
                    qh_base = self.rotary.rotate(qh_base, pos_offset)
                    kh = self.rotary.rotate(kh, pos_offset)

            v = v_proj.forward(x)
            vh = self._shape(v, v_head_dim)

            qh = self._apply_logit_scale_to_q(qh_base)
            qk_scale = float(self._qk_scale or (1.0 / math.sqrt(float(qk_head_dim))))

            if cache is None:
                if not cfg.null_attn:
                    out = self._sdp(qh, kh, vh, attn_mask=None if attn_mask is None else attn_mask)
                    y = out_proj.forward(self._merge(out))
                    return y, None

                scores = torch.matmul(qh, kh.transpose(-2, -1)) * qk_scale
                if attn_mask is not None:
                    scores = scores.masked_fill(~attn_mask, ninfty)

                k_null = self._shape(cast(torch.Tensor, self.k_null).expand(B, 1, -1), qk_head_dim)
                v_null = self._shape(cast(torch.Tensor, self.v_null).expand(B, 1, -1), v_head_dim)
                s_null = torch.matmul(qh, k_null.transpose(-2, -1)) * qk_scale
                scores = torch.cat([s_null, scores], dim=-1)

                if attn_mask is not None:
                    extra = torch.ones((1, 1, T, 1), device=attn_mask.device, dtype=attn_mask.dtype)
                    keep = torch.cat([extra, attn_mask], dim=-1)
                    scores = scores.masked_fill(~keep, ninfty)

                attn = F.softmax(scores, dim=-1)
                attn = self.drop.forward(attn)
                vals = torch.cat([v_null, vh], dim=-2)
                out = torch.matmul(attn, vals)

                y = out_proj.forward(self._merge(out))
                return y, None

            assert isinstance(cache, LayerKVCache)
            old_len = cache.pos

            if old_len == 0 and T > 1:
                if not cfg.null_attn:
                    out = self._sdp(qh, kh, vh, attn_mask=None if attn_mask is None else attn_mask)
                    y = out_proj.forward(self._merge(out))
                    _ = cache.append(self._merge(kh), self._merge(vh))
                    return y, cache

                scores = torch.matmul(qh, kh.transpose(-2, -1)) * qk_scale
                if attn_mask is not None:
                    scores = scores.masked_fill(~attn_mask, ninfty)

                k_null = self._shape(cast(torch.Tensor, self.k_null).expand(B, 1, -1), qk_head_dim)
                v_null = self._shape(cast(torch.Tensor, self.v_null).expand(B, 1, -1), v_head_dim)
                s_null = torch.matmul(qh, k_null.transpose(-2, -1)) * qk_scale
                scores = torch.cat([s_null, scores], dim=-1)

                if attn_mask is not None:
                    extra = torch.ones((1, 1, T, 1), device=attn_mask.device, dtype=attn_mask.dtype)
                    keep = torch.cat([extra, attn_mask], dim=-1)
                    scores = scores.masked_fill(~keep, ninfty)

                attn = F.softmax(scores, dim=-1)
                attn = self.drop.forward(attn)
                vals = torch.cat([v_null, vh], dim=-2)
                out = torch.matmul(attn, vals)

                y = out_proj.forward(self._merge(out))
                _ = cache.append(self._merge(kh), self._merge(vh))
                return y, cache

            _ = cache.append(self._merge(kh), self._merge(vh))
            cache_len = cache.pos

            if T == 1:
                decode_block = getattr(cache, "decode_block", 1024)
                if cfg.null_attn:
                    k_null = self._shape(cast(torch.Tensor, self.k_null).expand(B, 1, -1), qk_head_dim)
                    v_null = self._shape(cast(torch.Tensor, self.v_null).expand(B, 1, -1), v_head_dim)
                else:
                    k_null = v_null = None

                if cache.k.is_quantized or cache.v.is_quantized or cfg.null_attn:
                    out = self._streaming_decode_attn(
                        q=qh,
                        k_cache=cache.k,
                        v_cache=cache.v,
                        head_dim=qk_head_dim,
                        v_head_dim=v_head_dim,
                        decode_block=decode_block,
                        scale=qk_scale,
                        k_null=k_null,
                        v_null=v_null,
                    )
                else:
                    k_all = self._shape(cache.k.get(dtype=qh.dtype), qk_head_dim)
                    v_all = self._shape(cache.v.get(dtype=qh.dtype), v_head_dim)
                    out = _sdpa(qh, k_all, v_all, attn_mask=None, dropout_p=0.0, is_causal=False)

                y = out_proj.forward(self._merge(out))
                return y, cache

            k_all, v_all = cache.get(dtype=x.dtype)
            kh_all = self._shape(k_all, qk_head_dim)
            vh_all = self._shape(v_all, v_head_dim)

            scores = torch.matmul(qh, kh_all.transpose(-2, -1)) * qk_scale
            if attn_mask is not None:
                scores = scores.masked_fill(~attn_mask, ninfty)
            elif T > 1:
                key_pos = torch.arange(cache_len, device=x.device).view(1, 1, 1, cache_len)
                q_pos = (old_len + torch.arange(T, device=x.device)).view(1, 1, T, 1)
                keep = key_pos <= q_pos
                scores = scores.masked_fill(~keep, ninfty)

            if cfg.null_attn:
                k_null = self._shape(cast(torch.Tensor, self.k_null).expand(B, 1, -1), qk_head_dim)
                v_null = self._shape(cast(torch.Tensor, self.v_null).expand(B, 1, -1), v_head_dim)
                s_null = torch.matmul(qh, k_null.transpose(-2, -1)) * qk_scale
                scores = torch.cat([s_null, scores], dim=-1)
                attn = F.softmax(scores, dim=-1)
                attn = self.drop.forward(attn)
                vals = torch.cat([v_null, vh_all], dim=-2)
                out = torch.matmul(attn, vals)
            else:
                attn = F.softmax(scores, dim=-1)
                attn = self.drop.forward(attn)
                out = torch.matmul(attn, vh_all)

            y = out_proj.forward(self._merge(out))
            return y, cache

        # -----------------------------
        # gqa
        # -----------------------------
        if mode == "gqa":
            q_proj = self.q_proj
            k_proj = self.k_proj
            if q_proj is None or k_proj is None or self.qk_head_dim is None:
                raise RuntimeError("gqa mode requires q_proj/k_proj and qk_head_dim")
            v_proj = self.v_proj
            out_proj = self.out_proj
            qk_head_dim = int(self.qk_head_dim)
            v_head_dim = int(self.v_head_dim)

            q = q_proj.forward(x)
            k = k_proj.forward(x)
            v = v_proj.forward(x)

            qh = self._shape(q, qk_head_dim, n_head=self.H)
            kh = self._shape(k, qk_head_dim, n_head=self.H_kv)
            vh = self._shape(v, v_head_dim, n_head=self.H_kv)

            if self.rotary is not None:
                qh = self.rotary.rotate(qh, pos_offset)
                kh = self.rotary.rotate(kh, pos_offset)

            qh = self._apply_logit_scale_to_q(qh)
            qk_scale = float(self._qk_scale or (1.0 / math.sqrt(float(qk_head_dim))))

            if cache is None:
                kh_rep = kh.repeat_interleave(self.group_size, dim=1)
                vh_rep = vh.repeat_interleave(self.group_size, dim=1)

                if not cfg.null_attn:
                    out = self._sdp(qh, kh_rep, vh_rep, attn_mask=None if attn_mask is None else attn_mask)
                    y = out_proj.forward(self._merge(out))
                    return y, None

                scores = torch.matmul(qh, kh_rep.transpose(-2, -1)) * qk_scale
                if attn_mask is not None:
                    scores = scores.masked_fill(~attn_mask, ninfty)

                k_null = self._shape(cast(torch.Tensor, self.k_null).expand(B, 1, -1), qk_head_dim, n_head=self.H_kv)
                v_null = self._shape(cast(torch.Tensor, self.v_null).expand(B, 1, -1), v_head_dim, n_head=self.H_kv)
                k_null_rep = k_null.repeat_interleave(self.group_size, dim=1)
                v_null_rep = v_null.repeat_interleave(self.group_size, dim=1)

                s_null = torch.matmul(qh, k_null_rep.transpose(-2, -1)) * qk_scale
                scores = torch.cat([s_null, scores], dim=-1)

                if attn_mask is not None:
                    extra = torch.ones((1, 1, T, 1), device=attn_mask.device, dtype=attn_mask.dtype)
                    keep = torch.cat([extra, attn_mask], dim=-1)
                    scores = scores.masked_fill(~keep, ninfty)

                attn = F.softmax(scores, dim=-1)
                attn = self.drop.forward(attn)
                vals = torch.cat([v_null_rep, vh_rep], dim=-2)
                out = torch.matmul(attn, vals)

                y = out_proj.forward(self._merge(out))
                return y, None

            assert isinstance(cache, LayerKVCache)
            old_len = cache.pos

            if old_len == 0 and T > 1:
                kh_rep = kh.repeat_interleave(self.group_size, dim=1)
                vh_rep = vh.repeat_interleave(self.group_size, dim=1)
                if not cfg.null_attn:
                    out = self._sdp(qh, kh_rep, vh_rep, attn_mask=None if attn_mask is None else attn_mask)
                    y = out_proj.forward(self._merge(out))
                    _ = cache.append(self._merge(kh), self._merge(vh))
                    return y, cache

                scores = torch.matmul(qh, kh_rep.transpose(-2, -1)) * qk_scale
                if attn_mask is not None:
                    scores = scores.masked_fill(~attn_mask, ninfty)

                k_null = self._shape(cast(torch.Tensor, self.k_null).expand(B, 1, -1), qk_head_dim, n_head=self.H_kv)
                v_null = self._shape(cast(torch.Tensor, self.v_null).expand(B, 1, -1), v_head_dim, n_head=self.H_kv)
                k_null_rep = k_null.repeat_interleave(self.group_size, dim=1)
                v_null_rep = v_null.repeat_interleave(self.group_size, dim=1)

                s_null = torch.matmul(qh, k_null_rep.transpose(-2, -1)) * qk_scale
                scores = torch.cat([s_null, scores], dim=-1)

                if attn_mask is not None:
                    extra = torch.ones((1, 1, T, 1), device=attn_mask.device, dtype=attn_mask.dtype)
                    keep = torch.cat([extra, attn_mask], dim=-1)
                    scores = scores.masked_fill(~keep, ninfty)

                attn = F.softmax(scores, dim=-1)
                attn = self.drop.forward(attn)
                vals = torch.cat([v_null_rep, vh_rep], dim=-2)
                out = torch.matmul(attn, vals)

                y = out_proj.forward(self._merge(out))
                _ = cache.append(self._merge(kh), self._merge(vh))
                return y, cache

            _ = cache.append(self._merge(kh), self._merge(vh))
            cache_len = cache.pos

            if T == 1:
                decode_block = getattr(cache, "decode_block", 1024)
                compute_dtype = qh.dtype
                qg = qh.view(B, self.H_kv, self.group_size, 1, qk_head_dim).to(compute_dtype)
                m = torch.full((B, self.H, 1), -float("inf"), device=qh.device, dtype=torch.float32)
                d = torch.zeros((B, self.H, 1), device=qh.device, dtype=torch.float32)
                o = torch.zeros((B, self.H, 1, v_head_dim), device=qh.device, dtype=torch.float32)

                def update(scores_f32: torch.Tensor, v_block_f16: torch.Tensor) -> None:
                    nonlocal m, d, o
                    block_max = scores_f32.amax(dim=-1)
                    m_new = torch.maximum(m, block_max)
                    exp_m = torch.exp(m - m_new)
                    exp_scores = torch.exp(scores_f32 - m_new.unsqueeze(-1)).to(compute_dtype)
                    d = d * exp_m + exp_scores.sum(dim=-1).to(torch.float32)
                    o = o * exp_m.unsqueeze(-1) + torch.matmul(exp_scores, v_block_f16).to(torch.float32)
                    m = m_new

                if cfg.null_attn:
                    k_null = self._shape(cast(torch.Tensor, self.k_null).expand(B, 1, -1), qk_head_dim, n_head=self.H_kv)
                    v_null = self._shape(cast(torch.Tensor, self.v_null).expand(B, 1, -1), v_head_dim, n_head=self.H_kv)
                    s = (qg * k_null.to(compute_dtype).unsqueeze(2)).sum(dim=-1, keepdim=True).to(torch.float32) * qk_scale
                    # v_null is stored per KV-head; repeat to match per-query-head scores.
                    v_null_rep = v_null.repeat_interleave(self.group_size, dim=1).to(compute_dtype)  # (B, H, 1, hd)
                    update(s.view(B, self.H, 1, 1), v_null_rep)

                blk = int(max(1, decode_block))
                for start in range(0, cache_len, blk):
                    end = min(cache_len, start + blk)
                    k_blk = cache.k.get_slice(start, end, dtype=compute_dtype)
                    v_blk = cache.v.get_slice(start, end, dtype=compute_dtype)
                    kbh = self._shape(k_blk, qk_head_dim, n_head=self.H_kv)
                    vbh = self._shape(v_blk, v_head_dim, n_head=self.H_kv)
                    s = torch.matmul(qg, kbh.unsqueeze(2).transpose(-2, -1)) * qk_scale
                    # Scores are per-query-head (H); values are per-KV-head (H_kv). Repeat values across group_size.
                    vbh_rep = vbh.repeat_interleave(self.group_size, dim=1)  # (B, H, blk, hd)
                    update(s.view(B, self.H, 1, -1).to(torch.float32), vbh_rep)

                out = o / d.clamp(min=1e-9).unsqueeze(-1)
                y = out_proj.forward(self._merge(out.to(qh.dtype)))
                return y, cache

            # Prefill/chunked (materialize)
            k_all, v_all = cache.get(dtype=x.dtype)
            kh_all = self._shape(k_all, qk_head_dim, n_head=self.H_kv)
            vh_all = self._shape(v_all, v_head_dim, n_head=self.H_kv)
            kh_rep = kh_all.repeat_interleave(self.group_size, dim=1)
            vh_rep = vh_all.repeat_interleave(self.group_size, dim=1)

            scores = torch.matmul(qh, kh_rep.transpose(-2, -1)) * qk_scale
            if attn_mask is not None:
                scores = scores.masked_fill(~attn_mask, ninfty)
            elif T > 1:
                key_pos = torch.arange(cache_len, device=x.device).view(1, 1, 1, cache_len)
                q_pos = (old_len + torch.arange(T, device=x.device)).view(1, 1, T, 1)
                keep = key_pos <= q_pos
                scores = scores.masked_fill(~keep, ninfty)

            if cfg.null_attn:
                k_null = self._shape(cast(torch.Tensor, self.k_null).expand(B, 1, -1), qk_head_dim, n_head=self.H_kv)
                v_null = self._shape(cast(torch.Tensor, self.v_null).expand(B, 1, -1), v_head_dim, n_head=self.H_kv)
                k_null_rep = k_null.repeat_interleave(self.group_size, dim=1)
                v_null_rep = v_null.repeat_interleave(self.group_size, dim=1)
                s_null = torch.matmul(qh, k_null_rep.transpose(-2, -1)) * qk_scale
                scores = torch.cat([s_null, scores], dim=-1)
                attn = F.softmax(scores, dim=-1)
                attn = self.drop.forward(attn)
                vals = torch.cat([v_null_rep, vh_rep], dim=-2)
                out = torch.matmul(attn, vals)
            else:
                attn = F.softmax(scores, dim=-1)
                attn = self.drop.forward(attn)
                out = torch.matmul(attn, vh_rep)

            y = out_proj.forward(self._merge(out))
            return y, cache

        # -----------------------------
        # decoupled
        # -----------------------------
        assert mode == "decoupled"

        q_sem_proj = self.q_sem
        k_sem_proj = self.k_sem
        q_geo_proj = self.q_geo
        k_geo_proj = self.k_geo
        sem_head_dim = self.sem_head_dim
        geo_head_dim = self.geo_head_dim
        v_head_dim = int(self.v_head_dim)
        if (
            q_sem_proj is None
            or k_sem_proj is None
            or q_geo_proj is None
            or k_geo_proj is None
            or sem_head_dim is None
            or geo_head_dim is None
        ):
            raise RuntimeError("decoupled mode requires sem/geo projections and head dims")

        # Conceptually, we split attention into two additive paths:
        #   score = (Q_sem · K_sem^T) + (Q_geo · K_geo^T)
        # with RoPE applied only to the geometric path.
        if k_sem_proj is q_sem_proj:
            sem = q_sem_proj.forward(x)
            sem_h = self._shape(sem, sem_head_dim)
            qsh_base = sem_h
            ksh = sem_h
        else:
            q_sem = q_sem_proj.forward(x)
            k_sem = k_sem_proj.forward(x)
            qsh_base = self._shape(q_sem, sem_head_dim)
            ksh = self._shape(k_sem, sem_head_dim)

        if k_geo_proj is q_geo_proj:
            geo = q_geo_proj.forward(x)
            geo_h = self._shape(geo, geo_head_dim)
            if self.rotary is not None:
                geo_h = self.rotary.rotate(geo_h, pos_offset)
            qgh_base = geo_h
            kgh = geo_h
        else:
            q_geo = q_geo_proj.forward(x)
            k_geo = k_geo_proj.forward(x)
            qgh_base = self._shape(q_geo, geo_head_dim)
            kgh = self._shape(k_geo, geo_head_dim)
            if self.rotary is not None:
                # IMPORTANT: RoPE applies ONLY to geometric/position path.
                qgh_base = self.rotary.rotate(qgh_base, pos_offset)
                kgh = self.rotary.rotate(kgh, pos_offset)

        v = self.v_proj.forward(x)
        vh = self._shape(v, v_head_dim)

        # Per-head learned temperature (applied to both paths by scaling queries).
        qsh = self._apply_logit_scale_to_q(qsh_base)
        qgh = self._apply_logit_scale_to_q(qgh_base)

        sem_scale = float(self._sem_scale or (1.0 / math.sqrt(float(sem_head_dim))))
        geo_scale = float(self._geo_scale or (1.0 / math.sqrt(float(geo_head_dim))))

        # Optional per-head sem/geo mixing gate.
        # Implemented as a query scaling so it works with both streaming decode and fused kernels.
        g = self._decoupled_gate(x)
        if g is not None:
            gq = g.to(dtype=qsh.dtype, device=qsh.device)
            qsh = qsh * (2.0 * gq)
            qgh = qgh * (2.0 - 2.0 * gq)

        if cache is None:
            # Training / full-attn:
            # Prefer SDPA for stability (it is much less prone to bf16 softmax overflow than the manual path).
            if not cfg.null_attn:
                # Long-seq auto-optimization: use a semantic "compressed memory" plus full-res local tail.
                # This reduces the dominant O(T^2) term to ~O(T^2 / mem_block + T*local_window) while keeping
                # the geometric path high-resolution in the recent window.
                #
                # Trigger only at long sequence lengths to minimize behavioral drift at typical training ctx.
                if bool(cfg.train_long_seq_enabled) and self.training and attn_mask is None:
                    dev = x.device.type
                    # Auto-trigger at long seq: start once we approach the configured block_size (but never below 3k).
                    long_seq_threshold = cfg.train_long_seq_threshold
                    if long_seq_threshold is None:
                        long_seq_threshold = max(3072, int(0.75 * cfg.block_size))
                        if dev == "mps":
                            long_seq_threshold = min(int(long_seq_threshold), 3072)
                    if T >= int(long_seq_threshold):
                        try:
                            mem_block = cfg.train_long_seq_mem_block
                            local_window = cfg.train_long_seq_local_window
                            q_chunk = cfg.train_long_seq_q_chunk

                            # Fill any missing knobs with conservative, backend-aware defaults.
                            if mem_block is None:
                                mem_block = 128 if (dev == "cuda" and T >= 8192) else (64 if dev == "cuda" else 64)
                            if local_window is None:
                                local_window = 1024 if (dev == "cuda" and T >= 8192) else (512 if dev == "cuda" else 512)
                            if q_chunk is None:
                                q_chunk = 256 if dev == "cuda" else 128

                            local_window = int(min(max(0, local_window), T))
                            mem_block = int(max(1, mem_block))
                            q_chunk = int(max(1, q_chunk))

                            B0, H0, T0, sem_hd = qsh.shape
                            v_hd = int(vh.size(-1))
                            n_blocks = int(T0 // mem_block)

                            q_cat_full = torch.cat([qsh * sem_scale, qgh * geo_scale], dim=-1)

                            mem_summarizer = str(cfg.train_long_seq_mem_summarizer).strip().lower()
                            if mem_summarizer == "conv" and dev != "cuda":
                                mem_summarizer = "linear"
                            if mem_summarizer not in ("mean", "linear", "conv"):
                                mem_summarizer = "mean"

                            if n_blocks > 0:
                                k_sem_tok = ksh[:, :, : n_blocks * mem_block, :].reshape(B0, H0, n_blocks, mem_block, sem_hd)
                                v_tok = vh[:, :, : n_blocks * mem_block, :].reshape(B0, H0, n_blocks, mem_block, v_hd)
                                k_sem_blocks = k_sem_tok.mean(dim=3)
                                v_blocks = v_tok.mean(dim=3)

                                if mem_summarizer == "linear":
                                    k_alpha = cast(torch.Tensor, self.long_seq_mem_k_alpha).to(dtype=k_sem_blocks.dtype)
                                    v_alpha = cast(torch.Tensor, self.long_seq_mem_v_alpha).to(dtype=v_blocks.dtype)
                                    k_lin = cast(nn.Linear, self.long_seq_mem_k_linear)
                                    v_lin = cast(nn.Linear, self.long_seq_mem_v_linear)
                                    k_sem_blocks = k_sem_blocks + k_alpha * k_lin.forward(k_sem_blocks)
                                    v_blocks = v_blocks + v_alpha * v_lin.forward(v_blocks)
                                elif mem_summarizer == "conv":
                                    k_alpha = cast(torch.Tensor, self.long_seq_mem_k_alpha).to(dtype=k_sem_blocks.dtype)
                                    v_alpha = cast(torch.Tensor, self.long_seq_mem_v_alpha).to(dtype=v_blocks.dtype)
                                    k_conv = cast(nn.Conv1d, self.long_seq_mem_k_conv)
                                    v_conv = cast(nn.Conv1d, self.long_seq_mem_v_conv)
                                    k_in = k_sem_tok.reshape(B0 * H0 * n_blocks, mem_block, sem_hd).transpose(1, 2).contiguous()
                                    v_in = v_tok.reshape(B0 * H0 * n_blocks, mem_block, v_hd).transpose(1, 2).contiguous()
                                    k_conv_sum = k_conv.forward(k_in).mean(dim=2).view(B0, H0, n_blocks, sem_hd)
                                    v_conv_sum = v_conv.forward(v_in).mean(dim=2).view(B0, H0, n_blocks, v_hd)
                                    k_sem_blocks = k_sem_blocks + k_alpha * k_conv_sum
                                    v_blocks = v_blocks + v_alpha * v_conv_sum
                                zeros_geo = k_sem_blocks.new_zeros((B0, H0, n_blocks, geo_head_dim))
                                k_mem_cat_all = torch.cat([k_sem_blocks, zeros_geo], dim=-1)
                            else:
                                v_blocks = None
                                k_mem_cat_all = None

                            out = vh.new_empty((B0, H0, T0, v_hd))
                            for t0 in range(0, T0, q_chunk):
                                t1 = min(T0, t0 + q_chunk)
                                # Define a full-res local band ending at the current chunk end.
                                # Align its start to mem_block so memory summaries never overlap the local band.
                                local_start_raw = max(0, int(t0) - int(local_window))
                                mem_len = int(local_start_raw // mem_block)
                                local_start = int(mem_len * mem_block)

                                q_cat = q_cat_full[:, :, t0:t1, :]
                                k_local_cat = torch.cat([ksh[:, :, local_start:t1, :], kgh[:, :, local_start:t1, :]], dim=-1)
                                v_local = vh[:, :, local_start:t1, :]

                                if mem_len > 0 and k_mem_cat_all is not None and v_blocks is not None:
                                    k_cat = torch.cat([k_mem_cat_all[:, :, :mem_len, :], k_local_cat], dim=2)
                                    v_cat = torch.cat([v_blocks[:, :, :mem_len, :], v_local], dim=2)
                                else:
                                    k_cat = k_local_cat
                                    v_cat = v_local

                                out[:, :, t0:t1, :] = self._sdp(q_cat, k_cat, v_cat, attn_mask=None, scale=1.0)

                            y = self.out_proj.forward(self._merge(out))
                            return y, None
                        except (RuntimeError, ValueError, TypeError):
                            # Safety fallback: if anything goes wrong, use the exact SDPA path.
                            pass

                # Combine sem+geo into a single SDPA call by concatenating along head_dim.
                # q/k last-dim must match; v may have a different last-dim (v_head_dim).
                q_cat, k_cat = _decoupled_qk_cat(q_sem=qsh, q_geo=qgh, k_sem=ksh, k_geo=kgh, sem_scale=sem_scale, geo_scale=geo_scale)
                out = self._sdp(q_cat, k_cat, vh, attn_mask=None if attn_mask is None else attn_mask, scale=1.0, is_causal=True if attn_mask is None else False)
                y = self.out_proj.forward(self._merge(out))
                return y, None

            # Null-attn path (manual).
            # Note: This feature is intentionally optional and is not part of the default decoupled
            # production/paper preset. It adds complexity in the hottest forward path and disables
            # some fused decode fast-paths; prefer keeping it off unless an explicit ablation shows
            # it is required for a particular regime. See `production/ablate_null_attn.py`.
            scores = _decoupled_scores_f32(q_sem=qsh, q_geo=qgh, k_sem=ksh, k_geo=kgh, sem_scale=sem_scale, geo_scale=geo_scale)

            if attn_mask is not None:
                scores = scores.masked_fill(~attn_mask, ninfty)
            elif T > 1:
                keep = torch.tril(torch.ones((T, T), device=x.device, dtype=torch.bool)).view(1, 1, T, T)
                scores = scores.masked_fill(~keep, ninfty)

            ksn = self._shape(cast(torch.Tensor, self.k_sem_null).expand(B, 1, -1), sem_head_dim)
            kgn = self._shape(cast(torch.Tensor, self.k_geo_null).expand(B, 1, -1), geo_head_dim)
            vn = self._shape(cast(torch.Tensor, self.v_null).expand(B, 1, -1), v_head_dim)
            s_null = _decoupled_scores_f32(q_sem=qsh, q_geo=qgh, k_sem=ksn, k_geo=kgn, sem_scale=sem_scale, geo_scale=geo_scale)
            scores = torch.cat([s_null, scores], dim=-1)

            attn = F.softmax(scores, dim=-1)
            attn = self.drop.forward(attn)
            vals = torch.cat([vn, vh], dim=-2)
            out = torch.matmul(attn, vals)

            y = self.out_proj.forward(self._merge(out))
            return y, None

        assert isinstance(cache, DecoupledLayerKVCache)
        old_len = cache.pos

        if old_len == 0 and T > 1:
            # prefill without cache readback
            if cfg.null_attn:
                scores = _decoupled_scores_f32(q_sem=qsh, q_geo=qgh, k_sem=ksh, k_geo=kgh, sem_scale=sem_scale, geo_scale=geo_scale)
                if attn_mask is not None:
                    scores = scores.masked_fill(~attn_mask, ninfty)
                ksn = self._shape(cast(torch.Tensor, self.k_sem_null).expand(B, 1, -1), sem_head_dim)
                kgn = self._shape(cast(torch.Tensor, self.k_geo_null).expand(B, 1, -1), geo_head_dim)
                vn = self._shape(cast(torch.Tensor, self.v_null).expand(B, 1, -1), v_head_dim)
                s_null = (torch.matmul(qsh, ksn.transpose(-2, -1)) * sem_scale + torch.matmul(qgh, kgn.transpose(-2, -1)) * geo_scale)
                scores = torch.cat([s_null, scores], dim=-1)
                if attn_mask is not None:
                    extra = torch.ones((1, 1, T, 1), device=attn_mask.device, dtype=attn_mask.dtype)
                    keep = torch.cat([extra, attn_mask], dim=-1)
                    scores = scores.masked_fill(~keep, ninfty)
                attn = F.softmax(scores, dim=-1)
                attn = self.drop.forward(attn)
                vals = torch.cat([vn, vh], dim=-2)
                out = torch.matmul(attn, vals)
            else:
                q_cat, k_cat = _decoupled_qk_cat(q_sem=qsh, q_geo=qgh, k_sem=ksh, k_geo=kgh, sem_scale=sem_scale, geo_scale=geo_scale)
                out = self._sdp(q_cat, k_cat, vh, attn_mask=None if attn_mask is None else attn_mask, scale=1.0, is_causal=True if attn_mask is None else False)

            y = self.out_proj.forward(self._merge(out))
            _ = cache.append(self._merge(ksh), self._merge(kgh), self._merge(vh))
            return y, cache

        # General cached path:
        # - T==1: decode (streaming or fused)
        # - T>1: prefer a sequential streaming path when caches are quantized (avoids dequantizing the full prefix),
        #        otherwise fall back to the materialized path.

        if T == 1:
            _ = cache.append(self._merge(ksh), self._merge(kgh), self._merge(vh))
            decode_block = getattr(cache, "decode_block", 1024)
            # fp16 cache -> materialize and use SDPA (usually faster than Python streaming)
            if (not cfg.null_attn) and (not cache.k_sem.is_quantized) and (not cache.k_geo.is_quantized) and (not cache.v.is_quantized):
                k_sem_all = self._shape(cache.k_sem.get(dtype=qsh.dtype), sem_head_dim)
                k_geo_all = self._shape(cache.k_geo.get(dtype=qsh.dtype), geo_head_dim)
                v_all = self._shape(cache.v.get(dtype=qsh.dtype), v_head_dim)
                q_cat, k_cat = _decoupled_qk_cat(q_sem=qsh, q_geo=qgh, k_sem=k_sem_all, k_geo=k_geo_all, sem_scale=sem_scale, geo_scale=geo_scale)
                # Important: decode uses is_causal=False here because query length is 1 but its logical
                # position is the *end* of the prefix; SDPA's built-in causal mask would treat it as position 0.
                out = self._sdp(q_cat, k_cat, v_all, attn_mask=None, scale=1.0, is_causal=False)
            else:
                # Quantized/null-attn -> streaming or fused decode.
                if cfg.null_attn:
                    ksn = self._shape(cast(torch.Tensor, self.k_sem_null).expand(B, 1, -1), sem_head_dim)
                    kgn = self._shape(cast(torch.Tensor, self.k_geo_null).expand(B, 1, -1), geo_head_dim)
                    vn = self._shape(cast(torch.Tensor, self.v_null).expand(B, 1, -1), v_head_dim)
                else:
                    ksn = kgn = vn = None

                def fused_ok() -> bool:
                    return bool(
                        TRITON_AVAILABLE
                        and cache.k_sem.kind == "q4_0"
                        and cache.k_geo.kind == "q8_0"
                        and cache.v.kind == "q4_0"
                        and cache.k_sem.spec.qblock == 32
                        and cache.k_geo.spec.qblock == 32
                        and cache.v.spec.qblock == 32
                        and x.device.type == "cuda"
                    )

                fused = getattr(cache, "fused", "none")
                if fused == "auto":
                    if fused_ok():
                        fused = "triton2pass" if int(cache.pos) >= 4 * int(decode_block) else "triton1pass"
                    else:
                        fused = "none"

                if fused in ("triton1pass", "triton2pass") and fused_ok():
                    try:
                        if fused == "triton1pass":
                            out = self._fused_decode_attn_decoupled_q4q8q4(
                                q_sem=qsh,
                                q_geo=qgh,
                                cache=cache,
                                sem_head_dim=sem_head_dim,
                                geo_head_dim=geo_head_dim,
                                v_head_dim=v_head_dim,
                                decode_block=decode_block,
                                sem_scale=float(sem_scale),
                                geo_scale=float(geo_scale),
                            )
                        else:
                            out = self._fused_decode_attn_decoupled_q4q8q4_2pass(
                                q_sem=qsh,
                                q_geo=qgh,
                                cache=cache,
                                sem_head_dim=sem_head_dim,
                                geo_head_dim=geo_head_dim,
                                v_head_dim=v_head_dim,
                                decode_block=decode_block,
                                sem_scale=float(sem_scale),
                                geo_scale=float(geo_scale),
                            )
                    except (RuntimeError, ValueError, TypeError):
                        out = self._streaming_decode_attn_decoupled(
                            q_sem=qsh,
                            q_geo=qgh,
                            k_sem_cache=cache.k_sem,
                            k_geo_cache=cache.k_geo,
                            v_cache=cache.v,
                            sem_head_dim=sem_head_dim,
                            geo_head_dim=geo_head_dim,
                            v_head_dim=v_head_dim,
                            decode_block=decode_block,
                            sem_scale=float(sem_scale),
                            geo_scale=float(geo_scale),
                            k_sem_null=ksn,
                            k_geo_null=kgn,
                            v_null=vn,
                        )
                else:
                    out = self._streaming_decode_attn_decoupled(
                        q_sem=qsh,
                        q_geo=qgh,
                        k_sem_cache=cache.k_sem,
                        k_geo_cache=cache.k_geo,
                        v_cache=cache.v,
                        sem_head_dim=sem_head_dim,
                        geo_head_dim=geo_head_dim,
                        v_head_dim=v_head_dim,
                        decode_block=decode_block,
                        sem_scale=float(sem_scale),
                        geo_scale=float(geo_scale),
                        k_sem_null=ksn,
                        k_geo_null=kgn,
                        v_null=vn,
                    )

            y = self.out_proj.forward(self._merge(out))
            return y, cache

        # T > 1
        decode_block = getattr(cache, "decode_block", 1024)
        quantized_cache = bool(cache.k_sem.is_quantized or cache.k_geo.is_quantized or cache.v.is_quantized)
        if quantized_cache and old_len > 0:
            # Sequential streaming prefill: append token-by-token and use streaming decode for each token.
            # This avoids materializing/dequantizing the full prefix for every token in the chunk.
            if cfg.null_attn:
                ksn = self._shape(cast(torch.Tensor, self.k_sem_null).expand(B, 1, -1), sem_head_dim)
                kgn = self._shape(cast(torch.Tensor, self.k_geo_null).expand(B, 1, -1), geo_head_dim)
                vn = self._shape(cast(torch.Tensor, self.v_null).expand(B, 1, -1), v_head_dim)
            else:
                ksn = kgn = vn = None

            outs: list[torch.Tensor] = []
            for t in range(T):
                # Append one token worth of KV to the cache.
                _ = cache.append(
                    self._merge(ksh[:, :, t : t + 1, :]),
                    self._merge(kgh[:, :, t : t + 1, :]),
                    self._merge(vh[:, :, t : t + 1, :]),
                )
                out_t = self._streaming_decode_attn_decoupled(
                    q_sem=qsh[:, :, t : t + 1, :],
                    q_geo=qgh[:, :, t : t + 1, :],
                    k_sem_cache=cache.k_sem,
                    k_geo_cache=cache.k_geo,
                    v_cache=cache.v,
                    sem_head_dim=sem_head_dim,
                    geo_head_dim=geo_head_dim,
                    v_head_dim=v_head_dim,
                    decode_block=decode_block,
                    sem_scale=float(sem_scale),
                    geo_scale=float(geo_scale),
                    k_sem_null=ksn,
                    k_geo_null=kgn,
                    v_null=vn,
                )
                outs.append(out_t)

            out = torch.cat(outs, dim=2)  # (B,H,T,hd_v)
            y = self.out_proj.forward(self._merge(out))
            return y, cache

        # Prefill/chunked attention (materialize).
        _ = cache.append(self._merge(ksh), self._merge(kgh), self._merge(vh))
        cache_len = cache.pos
        k_sem_all, k_geo_all, v_all = cache.get(dtype=x.dtype)
        ksh_all = self._shape(k_sem_all, sem_head_dim)
        kgh_all = self._shape(k_geo_all, geo_head_dim)
        vh_all = self._shape(v_all, v_head_dim)

        scores = _decoupled_scores_f32(q_sem=qsh, q_geo=qgh, k_sem=ksh_all, k_geo=kgh_all, sem_scale=sem_scale, geo_scale=geo_scale)

        if attn_mask is not None:
            scores = scores.masked_fill(~attn_mask, ninfty)
        elif T > 1:
            key_pos = torch.arange(cache_len, device=x.device).view(1, 1, 1, cache_len)
            q_pos = (old_len + torch.arange(T, device=x.device)).view(1, 1, T, 1)
            keep = key_pos <= q_pos
            scores = scores.masked_fill(~keep, ninfty)

        if cfg.null_attn:
            ksn = self._shape(cast(torch.Tensor, self.k_sem_null).expand(B, 1, -1), sem_head_dim)
            kgn = self._shape(cast(torch.Tensor, self.k_geo_null).expand(B, 1, -1), geo_head_dim)
            vn = self._shape(cast(torch.Tensor, self.v_null).expand(B, 1, -1), v_head_dim)
            s_null = _decoupled_scores_f32(q_sem=qsh, q_geo=qgh, k_sem=ksn, k_geo=kgn, sem_scale=sem_scale, geo_scale=geo_scale)
            scores = torch.cat([s_null, scores], dim=-1)
            attn = F.softmax(scores, dim=-1)
            attn = self.drop.forward(attn)
            vals = torch.cat([vn, vh_all], dim=-2)
            out = torch.matmul(attn, vals)
        else:
            attn = F.softmax(scores, dim=-1)
            attn = self.drop.forward(attn)
            out = torch.matmul(attn, vh_all)

        y = self.out_proj.forward(self._merge(out))
        return y, cache
