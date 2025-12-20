from __future__ import annotations

import math
from typing import Any, List, Optional, Tuple, TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

from production.kvcache_backend import DecoupledLayerKVCache, LayerKVCache, SeqCacheTensor
from production.rope import RotaryEmbedding

try:
    import triton  # type: ignore
    import triton.language as tl  # type: ignore
    TRITON_AVAILABLE = True
except Exception:
    triton = None  # type: ignore
    tl = None  # type: ignore
    TRITON_AVAILABLE = False

if TYPE_CHECKING:
    from production.model import ModelConfig


def neg_inf(dtype: torch.dtype) -> float:
    """
    A large negative "mask" value.

    Important: this must be representable in the *compute dtype* used by attention. On some backends
    (notably torch.compile+inductor on MPS), constant-folding a float32-min sentinel into bf16 can
    error with "cannot be converted to BFloat16 without overflow".
    """
    # Use a conservative finite sentinel that reliably zeroes softmax *and* is safely representable
    # across fp16/bf16/fp32. This avoids torch.compile constant-folding failures when a float32-min
    # sentinel gets cast into bf16/fp16.
    #
    # exp(-1e4) underflows to 0 for all practical purposes.
    return -1.0e4


def _decoupled_qk_cat(
    *,
    q_sem: torch.Tensor,
    q_geo: torch.Tensor,
    k_sem: torch.Tensor,
    k_geo: torch.Tensor,
    sem_scale: float,
    geo_scale: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build composite (q_cat, k_cat) for decoupled attention.

    Guarantees score equivalence:
      (q_cat @ k_cat^T) == (q_sem @ k_sem^T) * sem_scale + (q_geo @ k_geo^T) * geo_scale
    while leaving SDPA's internal softmax/numerics as an implementation detail.
    """
    q_cat = torch.cat([q_sem * float(sem_scale), q_geo * float(geo_scale)], dim=-1)
    k_cat = torch.cat([k_sem, k_geo], dim=-1)
    return q_cat, k_cat


def _decoupled_scores_f32(
    *,
    q_sem: torch.Tensor,
    q_geo: torch.Tensor,
    k_sem: torch.Tensor,
    k_geo: torch.Tensor,
    sem_scale: float,
    geo_scale: float,
) -> torch.Tensor:
    """Single source of truth for decoupled score computation in fp32."""
    sem = torch.matmul(q_sem, k_sem.transpose(-2, -1)).to(torch.float32) * float(sem_scale)
    geo = torch.matmul(q_geo, k_geo.transpose(-2, -1)).to(torch.float32) * float(geo_scale)
    return sem + geo


# -----------------------------
# Optional Triton fused kernels (decode only)
# -----------------------------

if TRITON_AVAILABLE:

    @triton.jit
    def _kv_decode_update_decoupled_q4q8q4(
        q_sem_ptr,
        q_geo_ptr,
        k_sem_q_ptr,
        k_sem_s_ptr,
        k_geo_q_ptr,
        k_geo_s_ptr,
        v_q_ptr,
        v_s_ptr,
        m_ptr,
        d_ptr,
        o_ptr,
        start: tl.int32,
        L_prefix: tl.int32,
        H: tl.constexpr,
        HD_SEM: tl.constexpr,
        HD_GEO: tl.constexpr,
        HD_V: tl.constexpr,
        QBLOCK_SEM: tl.constexpr,
        QBLOCK_GEO: tl.constexpr,
        QBLOCK_V: tl.constexpr,
        SEM_SCALE: tl.constexpr,
        GEO_SCALE: tl.constexpr,
        BLOCK_N: tl.constexpr,
        NUM_SUBBLOCKS: tl.constexpr,
        stride_qsem_b: tl.constexpr,
        stride_qsem_h: tl.constexpr,
        stride_qgeo_b: tl.constexpr,
        stride_qgeo_h: tl.constexpr,
        stride_ksq_b: tl.constexpr,
        stride_ksq_t: tl.constexpr,
        stride_kss_b: tl.constexpr,
        stride_kss_t: tl.constexpr,
        stride_kss_c: tl.constexpr,
        stride_kgq_b: tl.constexpr,
        stride_kgq_t: tl.constexpr,
        stride_kgs_b: tl.constexpr,
        stride_kgs_t: tl.constexpr,
        stride_kgs_c: tl.constexpr,
        stride_vq_b: tl.constexpr,
        stride_vq_t: tl.constexpr,
        stride_vs_b: tl.constexpr,
        stride_vs_t: tl.constexpr,
        stride_vs_c: tl.constexpr,
        stride_o: tl.constexpr,
    ):
        pid = tl.program_id(0)
        b = pid // H
        h = pid - b * H

        m = tl.load(m_ptr + pid).to(tl.float32)
        d = tl.load(d_ptr + pid).to(tl.float32)
        dv = tl.arange(0, HD_V)
        o = tl.load(o_ptr + pid * stride_o + dv, mask=dv < HD_V, other=0.0).to(tl.float32)

        ds = tl.arange(0, HD_SEM)
        dg = tl.arange(0, HD_GEO)
        q_sem = tl.load(q_sem_ptr + b * stride_qsem_b + h * stride_qsem_h + ds, mask=ds < HD_SEM, other=0.0).to(tl.float32)
        q_geo = tl.load(q_geo_ptr + b * stride_qgeo_b + h * stride_qgeo_h + dg, mask=dg < HD_GEO, other=0.0).to(tl.float32)

        for sb in tl.static_range(NUM_SUBBLOCKS):
            t = start + sb * BLOCK_N + tl.arange(0, BLOCK_N)
            tm = t < L_prefix

            ksd = h * HD_SEM + ds
            ks_byte = ksd // 2
            ks_nib = ksd % 2
            ks_ptr = k_sem_q_ptr + b * stride_ksq_b + t[:, None] * stride_ksq_t + ks_byte[None, :]
            ks_p = tl.load(ks_ptr, mask=tm[:, None] & (ds[None, :] < HD_SEM), other=0).to(tl.int32)
            ks_hi = ks_p >> 4
            ks_lo = ks_p & 0xF
            ks_u = tl.where(ks_nib[None, :] == 0, ks_hi, ks_lo)
            ks_q = ks_u.to(tl.int32) - 8
            ks_sb = ksd // QBLOCK_SEM
            ks_s_ptr = k_sem_s_ptr + b * stride_kss_b + t[:, None] * stride_kss_t + ks_sb[None, :] * stride_kss_c
            ks_s = tl.load(ks_s_ptr, mask=tm[:, None] & (ds[None, :] < HD_SEM), other=0.0).to(tl.float32)
            k_sem = ks_q.to(tl.float32) * ks_s
            logit_sem = tl.sum(k_sem * q_sem[None, :], axis=1)

            kgd = h * HD_GEO + dg
            kg_ptr = k_geo_q_ptr + b * stride_kgq_b + t[:, None] * stride_kgq_t + kgd[None, :]
            kg_q = tl.load(kg_ptr, mask=tm[:, None] & (dg[None, :] < HD_GEO), other=0).to(tl.int32)
            kg_sb = kgd // QBLOCK_GEO
            kg_s_ptr = k_geo_s_ptr + b * stride_kgs_b + t[:, None] * stride_kgs_t + kg_sb[None, :] * stride_kgs_c
            kg_s = tl.load(kg_s_ptr, mask=tm[:, None] & (dg[None, :] < HD_GEO), other=0.0).to(tl.float32)
            k_geo = kg_q.to(tl.float32) * kg_s
            logit_geo = tl.sum(k_geo * q_geo[None, :], axis=1)

            logits = logit_sem * SEM_SCALE + logit_geo * GEO_SCALE
            logits = tl.where(tm, logits, -float("inf"))

            block_max = tl.max(logits, axis=0)
            m_new = tl.maximum(m, block_max)
            exp_m = tl.exp(m - m_new)
            exp_logits = tl.exp(logits - m_new)

            vd = h * HD_V + dv
            v_byte = vd // 2
            v_nib = vd % 2
            v_ptr = v_q_ptr + b * stride_vq_b + t[:, None] * stride_vq_t + v_byte[None, :]
            v_p = tl.load(v_ptr, mask=tm[:, None] & (dv[None, :] < HD_V), other=0).to(tl.int32)
            v_hi = v_p >> 4
            v_lo = v_p & 0xF
            v_u = tl.where(v_nib[None, :] == 0, v_hi, v_lo)
            v_q = v_u.to(tl.int32) - 8
            v_sb = vd // QBLOCK_V
            vs_ptr = v_s_ptr + b * stride_vs_b + t[:, None] * stride_vs_t + v_sb[None, :] * stride_vs_c
            v_s = tl.load(vs_ptr, mask=tm[:, None] & (dv[None, :] < HD_V), other=0.0).to(tl.float32)
            v_val = v_q.to(tl.float32) * v_s

            d = d * exp_m + tl.sum(exp_logits, axis=0)
            wv = tl.sum(exp_logits[:, None] * v_val, axis=0)
            o = o * exp_m + wv
            m = m_new

        tl.store(m_ptr + pid, m)
        tl.store(d_ptr + pid, d)
        tl.store(o_ptr + pid * stride_o + dv, o, mask=dv < HD_V)

    @triton.jit
    def _kv_decode_partition_stats_decoupled_q4q8q4(
        q_sem_ptr,
        q_geo_ptr,
        k_sem_q_ptr,
        k_sem_s_ptr,
        k_geo_q_ptr,
        k_geo_s_ptr,
        v_q_ptr,
        v_s_ptr,
        m_part_ptr,
        d_part_ptr,
        o_part_ptr,
        L_prefix: tl.int32,
        P: tl.int32,
        PARTITION_SIZE: tl.constexpr,
        H: tl.constexpr,
        HD_SEM: tl.constexpr,
        HD_GEO: tl.constexpr,
        HD_V: tl.constexpr,
        QBLOCK_SEM: tl.constexpr,
        QBLOCK_GEO: tl.constexpr,
        QBLOCK_V: tl.constexpr,
        SEM_SCALE: tl.constexpr,
        GEO_SCALE: tl.constexpr,
        BLOCK_N: tl.constexpr,
        NUM_SUBBLOCKS: tl.constexpr,
        stride_qsem_b: tl.constexpr,
        stride_qsem_h: tl.constexpr,
        stride_qgeo_b: tl.constexpr,
        stride_qgeo_h: tl.constexpr,
        stride_ksq_b: tl.constexpr,
        stride_ksq_t: tl.constexpr,
        stride_kss_b: tl.constexpr,
        stride_kss_t: tl.constexpr,
        stride_kss_c: tl.constexpr,
        stride_kgq_b: tl.constexpr,
        stride_kgq_t: tl.constexpr,
        stride_kgs_b: tl.constexpr,
        stride_kgs_t: tl.constexpr,
        stride_kgs_c: tl.constexpr,
        stride_vq_b: tl.constexpr,
        stride_vq_t: tl.constexpr,
        stride_vs_b: tl.constexpr,
        stride_vs_t: tl.constexpr,
        stride_vs_c: tl.constexpr,
        stride_mp_row: tl.constexpr,
        stride_mp_part: tl.constexpr,
        stride_dp_row: tl.constexpr,
        stride_dp_part: tl.constexpr,
        stride_op_row: tl.constexpr,
        stride_op_part: tl.constexpr,
    ):
        pid_row = tl.program_id(0)
        pid_part = tl.program_id(1)
        b = pid_row // H
        h = pid_row - b * H

        ds = tl.arange(0, HD_SEM)
        dg = tl.arange(0, HD_GEO)
        dv = tl.arange(0, HD_V)

        qsem = tl.load(q_sem_ptr + b * stride_qsem_b + h * stride_qsem_h + ds, mask=ds < HD_SEM, other=0.0).to(tl.float32)
        qgeo = tl.load(q_geo_ptr + b * stride_qgeo_b + h * stride_qgeo_h + dg, mask=dg < HD_GEO, other=0.0).to(tl.float32)

        m = -float("inf")
        d = 0.0
        o = tl.zeros([HD_V], dtype=tl.float32)

        start = pid_part * PARTITION_SIZE
        for sb in tl.static_range(0, NUM_SUBBLOCKS):
            t = start + sb * BLOCK_N + tl.arange(0, BLOCK_N)
            tm = t < L_prefix

            ksd = h * HD_SEM + ds
            ks_byte = ksd // 2
            ks_nib = ksd % 2
            ks_ptr = k_sem_q_ptr + b * stride_ksq_b + t[:, None] * stride_ksq_t + ks_byte[None, :]
            ks_p = tl.load(ks_ptr, mask=tm[:, None] & (ds[None, :] < HD_SEM), other=0).to(tl.uint8)
            ks_hi = ks_p >> 4
            ks_lo = ks_p & 0xF
            ks_u = tl.where(ks_nib[None, :] == 0, ks_hi, ks_lo)
            ks_q = ks_u.to(tl.int32) - 8

            ks_sb = ksd // QBLOCK_SEM
            ks_s_ptr = k_sem_s_ptr + b * stride_kss_b + t[:, None] * stride_kss_t + ks_sb[None, :] * stride_kss_c
            ks_s = tl.load(ks_s_ptr, mask=tm[:, None] & (ds[None, :] < HD_SEM), other=0.0).to(tl.float32)
            ksem = ks_q.to(tl.float32) * ks_s
            sem = tl.sum(ksem * qsem[None, :], axis=1)

            kgd = h * HD_GEO + dg
            kg_ptr = k_geo_q_ptr + b * stride_kgq_b + t[:, None] * stride_kgq_t + kgd[None, :]
            kg_q = tl.load(kg_ptr, mask=tm[:, None] & (dg[None, :] < HD_GEO), other=0).to(tl.int8).to(tl.float32)

            kg_sb = kgd // QBLOCK_GEO
            kg_s_ptr = k_geo_s_ptr + b * stride_kgs_b + t[:, None] * stride_kgs_t + kg_sb[None, :] * stride_kgs_c
            kg_s = tl.load(kg_s_ptr, mask=tm[:, None] & (dg[None, :] < HD_GEO), other=0.0).to(tl.float32)
            kgeo = kg_q * kg_s
            geo = tl.sum(kgeo * qgeo[None, :], axis=1)

            logits = sem * SEM_SCALE + geo * GEO_SCALE
            logits = tl.where(tm, logits, -float("inf"))

            block_max = tl.max(logits, axis=0)
            m_new = tl.maximum(m, block_max)
            exp_m = tl.exp(m - m_new)
            exp_logits = tl.exp(logits - m_new)

            vd = h * HD_V + dv
            v_byte = vd // 2
            v_nib = vd % 2
            v_ptr = v_q_ptr + b * stride_vq_b + t[:, None] * stride_vq_t + v_byte[None, :]
            v_p = tl.load(v_ptr, mask=tm[:, None] & (dv[None, :] < HD_V), other=0).to(tl.uint8)
            v_hi = v_p >> 4
            v_lo = v_p & 0xF
            v_u = tl.where(v_nib[None, :] == 0, v_hi, v_lo)
            v_q = v_u.to(tl.int32) - 8

            v_sb = vd // QBLOCK_V
            vs_ptr = v_s_ptr + b * stride_vs_b + t[:, None] * stride_vs_t + v_sb[None, :] * stride_vs_c
            v_s = tl.load(vs_ptr, mask=tm[:, None] & (dv[None, :] < HD_V), other=0.0).to(tl.float32)
            v_val = v_q.to(tl.float32) * v_s

            d = d * exp_m + tl.sum(exp_logits, axis=0)
            wv = tl.sum(exp_logits[:, None] * v_val, axis=0)
            o = o * exp_m + wv
            m = m_new

        tl.store(m_part_ptr + pid_row * stride_mp_row + pid_part * stride_mp_part, m)
        tl.store(d_part_ptr + pid_row * stride_dp_row + pid_part * stride_dp_part, d)
        tl.store(o_part_ptr + pid_row * stride_op_row + pid_part * stride_op_part + dv, o, mask=dv < HD_V)

    @triton.jit
    def _kv_decode_reduce_partitions(
        m_part_ptr,
        d_part_ptr,
        o_part_ptr,
        m_ptr,
        d_ptr,
        o_ptr,
        P: tl.int32,
        NUM_PARTS: tl.constexpr,
        HD_V: tl.constexpr,
        stride_mp_row: tl.constexpr,
        stride_mp_part: tl.constexpr,
        stride_dp_row: tl.constexpr,
        stride_dp_part: tl.constexpr,
        stride_op_row: tl.constexpr,
        stride_op_part: tl.constexpr,
        stride_o: tl.constexpr,
    ):
        pid_row = tl.program_id(0)
        dv = tl.arange(0, HD_V)

        m = -float("inf")
        for p in tl.static_range(0, NUM_PARTS):
            p_i = tl.full([], p, tl.int32)
            pm = p_i < P
            mp = tl.load(m_part_ptr + pid_row * stride_mp_row + p_i * stride_mp_part, mask=pm, other=-float("inf"))
            m = tl.maximum(m, mp)

        d = 0.0
        o = tl.zeros([HD_V], dtype=tl.float32)
        for p in tl.static_range(0, NUM_PARTS):
            p_i = tl.full([], p, tl.int32)
            pm = p_i < P
            mp = tl.load(m_part_ptr + pid_row * stride_mp_row + p_i * stride_mp_part, mask=pm, other=-float("inf"))
            dp = tl.load(d_part_ptr + pid_row * stride_dp_row + p_i * stride_dp_part, mask=pm, other=0.0)
            op = tl.load(o_part_ptr + pid_row * stride_op_row + p_i * stride_op_part + dv, mask=pm & (dv < HD_V), other=0.0).to(tl.float32)
            scale = tl.where(pm, tl.exp(mp - m), 0.0)
            d += dp * scale
            o += op * scale

        tl.store(m_ptr + pid_row, m)
        tl.store(d_ptr + pid_row, d)
        tl.store(o_ptr + pid_row * stride_o + dv, o, mask=dv < HD_V)


def _triton_decoupled_q4q8q4_available() -> bool:
    return bool(TRITON_AVAILABLE)


class DecoupledBottleneckAttention(nn.Module):
    def __init__(self, cfg: "ModelConfig"):
        super().__init__()
        self.cfg = cfg
        H = cfg.n_head
        self.H = H
        self.H_kv = H
        self.group_size = 1
        self.drop = nn.Dropout(cfg.dropout)

        # (Decoupled) Optional per-head sem/geo mixing gate (created only in decoupled mode).
        self.decoupled_gate_logit: Optional[torch.Tensor] = None

        def must_div(name: str, total: int, denom: int) -> int:
            if total % denom != 0:
                raise ValueError(f"{name} ({total}) must be divisible by {denom}")
            return total // denom

        if cfg.attn_mode == "standard":
            qk_dim = cfg.d_model
            v_dim = cfg.d_model
            self.qk_head_dim = must_div("d_model", qk_dim, H)
            self.v_head_dim = must_div("d_model", v_dim, H)

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

        elif cfg.attn_mode == "bottleneck":
            qk_dim = cfg.attn_dim
            v_dim = cfg.attn_dim
            self.qk_head_dim = must_div("attn_dim", qk_dim, H)
            self.v_head_dim = must_div("attn_dim", v_dim, H)
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

        elif cfg.attn_mode == "gqa":
            kv_head = cfg.kv_head if cfg.kv_head is not None else H
            if kv_head <= 0:
                raise ValueError("kv_head must be > 0")
            if H % kv_head != 0:
                raise ValueError(f"gqa requires n_head % kv_head == 0 (got n_head={H}, kv_head={kv_head})")
            self.H_kv = kv_head
            self.group_size = H // kv_head

            self.qk_head_dim = must_div("attn_dim", cfg.attn_dim, H)
            self.v_head_dim = self.qk_head_dim
            kv_dim = kv_head * self.qk_head_dim

            if cfg.rope and (self.qk_head_dim % 2 != 0):
                raise ValueError("RoPE requires an even head dim. Choose attn_dim divisible by 2*n_head.")

            if cfg.tie_qk:
                raise ValueError("tie_qk is not supported for gqa unless kv_head == n_head (use --attn-mode standard).")

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

        elif cfg.attn_mode == "decoupled":
            self.sem_head_dim = must_div("sem_dim", cfg.sem_dim, H)
            self.geo_head_dim = must_div("geo_dim", cfg.geo_dim, H)
            self.v_head_dim = must_div("attn_dim", cfg.attn_dim, H)
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
                # Per-head gate (sigmoid) controlling semantic vs geometric score contribution.
                self.decoupled_gate_logit = nn.Parameter(torch.zeros(H))
        else:
            raise ValueError(cfg.attn_mode)

        self.logit_scale = nn.Parameter(torch.zeros(H)) if cfg.learned_temp else None
        self._flash2_scratch = None  # type: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
        self._flash2_scratch_cap = (0, 0, 0)

    def _shape(self, x: torch.Tensor, head_dim: int, H: Optional[int] = None) -> torch.Tensor:
        B, T, _D = x.shape
        H = self.H if H is None else H
        return x.view(B, T, H, head_dim).transpose(1, 2).contiguous()

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

    def _sdp(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
        *,
        scale: Optional[float] = None,
        is_causal: Optional[bool] = None,
    ) -> torch.Tensor:
        dropout_p = self.cfg.dropout if self.training else 0.0
        if is_causal is None:
            is_causal = attn_mask is None

        # SDPA defaults to an implicit scale of 1/sqrt(dk). To request an explicit `scale`, we emulate it
        # by pre-scaling q so we can stay on the most stable/portable SDPA call signature (no `scale=` kwarg).
        if scale is not None:
            dk = int(q.size(-1))
            q = q * (float(scale) * math.sqrt(dk))

        return F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=bool(is_causal))

    def _streaming_decode_attn(
        self,
        *,
        q: torch.Tensor,
        k_cache: SeqCacheTensor,
        v_cache: SeqCacheTensor,
        head_dim: int,
        decode_block: int,
        scale: float,
        v_head_dim: Optional[int] = None,
        k_null: Optional[torch.Tensor] = None,
        v_null: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, H, Tq, _hd = q.shape
        assert Tq == 1
        if v_head_dim is None:
            v_head_dim = head_dim

        L = k_cache.pos
        if L != v_cache.pos:
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
        for start in range(0, L, blk):
            end = min(L, start + blk)
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
        k_sem_null: Optional[torch.Tensor] = None,
        k_geo_null: Optional[torch.Tensor] = None,
        v_null: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, H, Tq, _ = q_sem.shape
        assert Tq == 1
        L = k_sem_cache.pos
        if not (L == k_geo_cache.pos == v_cache.pos):
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
        for start in range(0, L, blk):
            end = min(L, start + blk)
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
        if self.cfg.null_attn:
            raise RuntimeError("Fused decode currently assumes null_attn=False")

        if not (cache.k_sem.kind == "q4_0" and cache.k_geo.kind == "q8_0" and cache.v.kind == "q4_0"):
            raise RuntimeError("Fused decode q4q8q4 requires k_sem=q4_0, k_geo=q8_0, v=q4_0")
        if not (cache.k_sem.spec.qblock == cache.k_geo.spec.qblock == cache.v.spec.qblock == 32):
            raise RuntimeError("Fused decode currently assumes qblock=32")

        B, H, Tq, _ = q_sem.shape
        assert Tq == 1
        L = cache.pos

        rlen = cache.k_sem._residual_len_eff if cache.k_sem._residual is not None else 0
        r_start = max(0, L - rlen) if rlen > 0 else L
        L_prefix = int(r_start)

        q_sem2 = q_sem[:, :, 0, :].contiguous().to(torch.float16)
        q_geo2 = q_geo[:, :, 0, :].contiguous().to(torch.float16)

        BH = B * H
        m = torch.full((BH,), -float("inf"), device=q_sem.device, dtype=torch.float32)
        d = torch.zeros((BH,), device=q_sem.device, dtype=torch.float32)
        o = torch.zeros((BH, v_head_dim), device=q_sem.device, dtype=torch.float32)

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

            grid = (BH,)
            for start in range(0, L_prefix, step):
                _kv_decode_update_decoupled_q4q8q4[grid](
                    q_sem2,
                    q_geo2,
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
                    stride_o=o.stride(0),
                    num_warps=num_warps,
                    num_stages=num_stages,
                )

        if L_prefix < L:
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

            k_sem_blk = cache.k_sem.get_slice(L_prefix, L, dtype=torch.float16)
            k_geo_blk = cache.k_geo.get_slice(L_prefix, L, dtype=torch.float16)
            v_blk = cache.v.get_slice(L_prefix, L, dtype=torch.float16)
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
        if self.cfg.null_attn:
            raise RuntimeError("Fused decode currently assumes null_attn=False")
        if not (cache.k_sem.kind == "q4_0" and cache.k_geo.kind == "q8_0" and cache.v.kind == "q4_0"):
            raise RuntimeError("Fused decode q4q8q4 requires k_sem=q4_0, k_geo=q8_0, v=q4_0")
        if not (cache.k_sem.spec.qblock == cache.k_geo.spec.qblock == cache.v.spec.qblock == 32):
            raise RuntimeError("Fused decode currently assumes qblock=32")

        B, H, Tq, _ = q_sem.shape
        assert Tq == 1
        L = cache.pos

        rlen = cache.k_sem._residual_len_eff if cache.k_sem._residual is not None else 0
        r_start = max(0, L - rlen) if rlen > 0 else L
        L_prefix = int(r_start)

        q_sem2 = q_sem[:, :, 0, :].contiguous().to(torch.float16)
        q_geo2 = q_geo[:, :, 0, :].contiguous().to(torch.float16)

        BH = B * H

        # Partitioning knobs (tunable via cache attrs)
        block_n = int(getattr(cache, "block_n", 128)) or 128
        num_sub = max(1, int(decode_block // block_n))
        partition_size = block_n * num_sub
        P = int(math.ceil(max(1, L_prefix) / float(partition_size))) if L_prefix > 0 else 0
        num_parts_cap = int(max(1, P))

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

        if L_prefix > 0:
            ksq = cache.k_sem.q
            kss = cache.k_sem.s
            kgq = cache.k_geo.q
            kgs = cache.k_geo.s
            vq = cache.v.q
            vs = cache.v.s
            assert ksq is not None and kss is not None and kgq is not None and kgs is not None and vq is not None and vs is not None

            num_warps_part = int(getattr(cache, "num_warps_part", 4))
            num_stages_part = int(getattr(cache, "num_stages_part", 2))
            num_warps_reduce = int(getattr(cache, "num_warps_reduce", 1))
            num_stages_reduce = int(getattr(cache, "num_stages_reduce", 1))

            grid1 = (BH, P)
            _kv_decode_partition_stats_decoupled_q4q8q4[grid1](
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

            # Reduce
            # NUM_PARTS is a constexpr loop bound. Round up to power-of-two-ish small bound.
            num_parts_const = 1
            while num_parts_const < P:
                num_parts_const *= 2
            grid2 = (BH,)
            _kv_decode_reduce_partitions[grid2](
                m_part,
                d_part,
                o_part,
                m,
                d,
                o,
                P,
                NUM_PARTS=num_parts_const,
                HD_V=v_head_dim,
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
        else:
            m.fill_(-float("inf"))
            d.zero_()
            o.zero_()

        # Residual tail via Python streaming updater (tiny).
        if L_prefix < L:
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

            k_sem_blk = cache.k_sem.get_slice(L_prefix, L, dtype=torch.float16)
            k_geo_blk = cache.k_geo.get_slice(L_prefix, L, dtype=torch.float16)
            v_blk = cache.v.get_slice(L_prefix, L, dtype=torch.float16)
            ksh = self._shape(k_sem_blk, sem_head_dim)
            kgh = self._shape(k_geo_blk, geo_head_dim)
            vbh = self._shape(v_blk, v_head_dim)
            s = _decoupled_scores_f32(q_sem=qsh, q_geo=qgh, k_sem=ksh, k_geo=kgh, sem_scale=sem_scale, geo_scale=geo_scale)
            update(s, vbh)
            o = o_t.view(BH, v_head_dim)
            d = d_t.view(BH)

        out = (o / d.clamp(min=1e-9).unsqueeze(-1)).view(B, H, 1, v_head_dim)
        return out.to(q_sem.dtype)

    def forward(
        self,
        x: torch.Tensor,
        *,
        attn_mask: Optional[torch.Tensor],
        cache: Optional[Any],
        pos_offset: int,
    ) -> Tuple[torch.Tensor, Optional[Any]]:
        cfg = self.cfg
        B, T, _ = x.shape
        ninfty = neg_inf(x.dtype)

        if cfg.attn_mode in ("standard", "bottleneck"):
            # If tie_qk is enabled, q_proj == k_proj and the projections are identical for self-attention.
            # Compute once and reuse (saves a full matmul + reshape + rotary per layer).
            if self.k_proj is self.q_proj:
                qk = self.q_proj(x)
                qh_base = self._shape(qk, self.qk_head_dim)
                if self.rotary is not None:
                    qh_base = self.rotary.rotate(qh_base, pos_offset)
                kh = qh_base
            else:
                q = self.q_proj(x)
                k = self.k_proj(x)
                qh_base = self._shape(q, self.qk_head_dim)
                kh = self._shape(k, self.qk_head_dim)
                if self.rotary is not None:
                    qh_base = self.rotary.rotate(qh_base, pos_offset)
                    kh = self.rotary.rotate(kh, pos_offset)

            v = self.v_proj(x)
            vh = self._shape(v, self.v_head_dim)

            qh = self._apply_logit_scale_to_q(qh_base)

            if cache is None:
                if not cfg.null_attn:
                    out = self._sdp(qh, kh, vh, attn_mask=None if attn_mask is None else attn_mask)
                    y = self.out_proj(self._merge(out))
                    return y, None

                scores = torch.matmul(qh, kh.transpose(-2, -1)) / math.sqrt(self.qk_head_dim)
                if attn_mask is not None:
                    scores = scores.masked_fill(~attn_mask, ninfty)

                k_null = self._shape(self.k_null.expand(B, 1, -1), self.qk_head_dim)
                v_null = self._shape(self.v_null.expand(B, 1, -1), self.v_head_dim)
                s_null = torch.matmul(qh, k_null.transpose(-2, -1)) / math.sqrt(self.qk_head_dim)
                scores = torch.cat([s_null, scores], dim=-1)

                if attn_mask is not None:
                    extra = torch.ones((1, 1, T, 1), device=attn_mask.device, dtype=attn_mask.dtype)
                    keep = torch.cat([extra, attn_mask], dim=-1)
                    scores = scores.masked_fill(~keep, ninfty)

                attn = F.softmax(scores, dim=-1)
                attn = self.drop(attn)
                vals = torch.cat([v_null, vh], dim=-2)
                out = torch.matmul(attn, vals)

                y = self.out_proj(self._merge(out))
                return y, None

            assert isinstance(cache, LayerKVCache)
            old_len = cache.pos

            if old_len == 0 and T > 1:
                if not cfg.null_attn:
                    out = self._sdp(qh, kh, vh, attn_mask=None if attn_mask is None else attn_mask)
                    y = self.out_proj(self._merge(out))
                    cache.append(self._merge(kh), self._merge(vh))
                    return y, cache

                scores = torch.matmul(qh, kh.transpose(-2, -1)) / math.sqrt(self.qk_head_dim)
                if attn_mask is not None:
                    scores = scores.masked_fill(~attn_mask, ninfty)

                k_null = self._shape(self.k_null.expand(B, 1, -1), self.qk_head_dim)
                v_null = self._shape(self.v_null.expand(B, 1, -1), self.v_head_dim)
                s_null = torch.matmul(qh, k_null.transpose(-2, -1)) / math.sqrt(self.qk_head_dim)
                scores = torch.cat([s_null, scores], dim=-1)

                if attn_mask is not None:
                    extra = torch.ones((1, 1, T, 1), device=attn_mask.device, dtype=attn_mask.dtype)
                    keep = torch.cat([extra, attn_mask], dim=-1)
                    scores = scores.masked_fill(~keep, ninfty)

                attn = F.softmax(scores, dim=-1)
                attn = self.drop(attn)
                vals = torch.cat([v_null, vh], dim=-2)
                out = torch.matmul(attn, vals)

                y = self.out_proj(self._merge(out))
                cache.append(self._merge(kh), self._merge(vh))
                return y, cache

            cache.append(self._merge(kh), self._merge(vh))
            L = cache.pos

            if T == 1:
                decode_block = getattr(cache, "decode_block", 1024)
                if cfg.null_attn:
                    k_null = self._shape(self.k_null.expand(B, 1, -1), self.qk_head_dim)
                    v_null = self._shape(self.v_null.expand(B, 1, -1), self.v_head_dim)
                else:
                    k_null = v_null = None

                if cache.k.is_quantized or cache.v.is_quantized or cfg.null_attn:
                    out = self._streaming_decode_attn(
                        q=qh,
                        k_cache=cache.k,
                        v_cache=cache.v,
                        head_dim=self.qk_head_dim,
                        v_head_dim=self.v_head_dim,
                        decode_block=decode_block,
                        scale=(1.0 / math.sqrt(self.qk_head_dim)),
                        k_null=k_null,
                        v_null=v_null,
                    )
                else:
                    k_all = self._shape(cache.k.get(dtype=qh.dtype), self.qk_head_dim)
                    v_all = self._shape(cache.v.get(dtype=qh.dtype), self.v_head_dim)
                    out = F.scaled_dot_product_attention(qh, k_all, v_all, attn_mask=None, dropout_p=0.0, is_causal=False)

                y = self.out_proj(self._merge(out))
                return y, cache

            k_all, v_all = cache.get(dtype=x.dtype)
            kh_all = self._shape(k_all, self.qk_head_dim)
            vh_all = self._shape(v_all, self.v_head_dim)

            scores = torch.matmul(qh, kh_all.transpose(-2, -1)) / math.sqrt(self.qk_head_dim)
            if attn_mask is not None:
                scores = scores.masked_fill(~attn_mask, ninfty)
            elif T > 1:
                key_pos = torch.arange(L, device=x.device).view(1, 1, 1, L)
                q_pos = (old_len + torch.arange(T, device=x.device)).view(1, 1, T, 1)
                keep = key_pos <= q_pos
                scores = scores.masked_fill(~keep, ninfty)

            if cfg.null_attn:
                k_null = self._shape(self.k_null.expand(B, 1, -1), self.qk_head_dim)
                v_null = self._shape(self.v_null.expand(B, 1, -1), self.v_head_dim)
                s_null = torch.matmul(qh, k_null.transpose(-2, -1)) / math.sqrt(self.qk_head_dim)
                scores = torch.cat([s_null, scores], dim=-1)
                attn = F.softmax(scores, dim=-1)
                attn = self.drop(attn)
                vals = torch.cat([v_null, vh_all], dim=-2)
                out = torch.matmul(attn, vals)
            else:
                attn = F.softmax(scores, dim=-1)
                attn = self.drop(attn)
                out = torch.matmul(attn, vh_all)

            y = self.out_proj(self._merge(out))
            return y, cache

        # -----------------------------
        # gqa
        # -----------------------------
        if cfg.attn_mode == "gqa":
            q = self.q_proj(x)
            k = self.k_proj(x)
            v = self.v_proj(x)

            qh = self._shape(q, self.qk_head_dim, H=self.H)
            kh = self._shape(k, self.qk_head_dim, H=self.H_kv)
            vh = self._shape(v, self.v_head_dim, H=self.H_kv)

            if self.rotary is not None:
                qh = self.rotary.rotate(qh, pos_offset)
                kh = self.rotary.rotate(kh, pos_offset)

            qh = self._apply_logit_scale_to_q(qh)

            if cache is None:
                kh_rep = kh.repeat_interleave(self.group_size, dim=1)
                vh_rep = vh.repeat_interleave(self.group_size, dim=1)

                if not cfg.null_attn:
                    out = self._sdp(qh, kh_rep, vh_rep, attn_mask=None if attn_mask is None else attn_mask)
                    y = self.out_proj(self._merge(out))
                    return y, None

                scores = torch.matmul(qh, kh_rep.transpose(-2, -1)) / math.sqrt(self.qk_head_dim)
                if attn_mask is not None:
                    scores = scores.masked_fill(~attn_mask, ninfty)

                k_null = self._shape(self.k_null.expand(B, 1, -1), self.qk_head_dim, H=self.H_kv)
                v_null = self._shape(self.v_null.expand(B, 1, -1), self.v_head_dim, H=self.H_kv)
                k_null_rep = k_null.repeat_interleave(self.group_size, dim=1)
                v_null_rep = v_null.repeat_interleave(self.group_size, dim=1)

                s_null = torch.matmul(qh, k_null_rep.transpose(-2, -1)) / math.sqrt(self.qk_head_dim)
                scores = torch.cat([s_null, scores], dim=-1)

                if attn_mask is not None:
                    extra = torch.ones((1, 1, T, 1), device=attn_mask.device, dtype=attn_mask.dtype)
                    keep = torch.cat([extra, attn_mask], dim=-1)
                    scores = scores.masked_fill(~keep, ninfty)

                attn = F.softmax(scores, dim=-1)
                attn = self.drop(attn)
                vals = torch.cat([v_null_rep, vh_rep], dim=-2)
                out = torch.matmul(attn, vals)

                y = self.out_proj(self._merge(out))
                return y, None

            assert isinstance(cache, LayerKVCache)
            old_len = cache.pos

            if old_len == 0 and T > 1:
                kh_rep = kh.repeat_interleave(self.group_size, dim=1)
                vh_rep = vh.repeat_interleave(self.group_size, dim=1)
                if not cfg.null_attn:
                    out = self._sdp(qh, kh_rep, vh_rep, attn_mask=None if attn_mask is None else attn_mask)
                    y = self.out_proj(self._merge(out))
                    cache.append(self._merge(kh), self._merge(vh))
                    return y, cache

                scores = torch.matmul(qh, kh_rep.transpose(-2, -1)) / math.sqrt(self.qk_head_dim)
                if attn_mask is not None:
                    scores = scores.masked_fill(~attn_mask, ninfty)

                k_null = self._shape(self.k_null.expand(B, 1, -1), self.qk_head_dim, H=self.H_kv)
                v_null = self._shape(self.v_null.expand(B, 1, -1), self.v_head_dim, H=self.H_kv)
                k_null_rep = k_null.repeat_interleave(self.group_size, dim=1)
                v_null_rep = v_null.repeat_interleave(self.group_size, dim=1)

                s_null = torch.matmul(qh, k_null_rep.transpose(-2, -1)) / math.sqrt(self.qk_head_dim)
                scores = torch.cat([s_null, scores], dim=-1)

                if attn_mask is not None:
                    extra = torch.ones((1, 1, T, 1), device=attn_mask.device, dtype=attn_mask.dtype)
                    keep = torch.cat([extra, attn_mask], dim=-1)
                    scores = scores.masked_fill(~keep, ninfty)

                attn = F.softmax(scores, dim=-1)
                attn = self.drop(attn)
                vals = torch.cat([v_null_rep, vh_rep], dim=-2)
                out = torch.matmul(attn, vals)

                y = self.out_proj(self._merge(out))
                cache.append(self._merge(kh), self._merge(vh))
                return y, cache

            cache.append(self._merge(kh), self._merge(vh))
            L = cache.pos

            if T == 1:
                decode_block = getattr(cache, "decode_block", 1024)
                compute_dtype = qh.dtype
                qg = qh.view(B, self.H_kv, self.group_size, 1, self.qk_head_dim).to(compute_dtype)
                m = torch.full((B, self.H, 1), -float("inf"), device=qh.device, dtype=torch.float32)
                d = torch.zeros((B, self.H, 1), device=qh.device, dtype=torch.float32)
                o = torch.zeros((B, self.H, 1, self.v_head_dim), device=qh.device, dtype=torch.float32)

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
                    k_null = self._shape(self.k_null.expand(B, 1, -1), self.qk_head_dim, H=self.H_kv)
                    v_null = self._shape(self.v_null.expand(B, 1, -1), self.v_head_dim, H=self.H_kv)
                    s = (qg * k_null.to(compute_dtype).unsqueeze(2)).sum(dim=-1, keepdim=True).to(torch.float32) * (1.0 / math.sqrt(self.qk_head_dim))
                    update(s.view(B, self.H, 1, 1), v_null.to(compute_dtype).unsqueeze(2).view(B, self.H, 1, self.v_head_dim))

                blk = int(max(1, decode_block))
                for start in range(0, L, blk):
                    end = min(L, start + blk)
                    k_blk = cache.k.get_slice(start, end, dtype=compute_dtype)
                    v_blk = cache.v.get_slice(start, end, dtype=compute_dtype)
                    kbh = self._shape(k_blk, self.qk_head_dim, H=self.H_kv)
                    vbh = self._shape(v_blk, self.v_head_dim, H=self.H_kv)
                    s = torch.matmul(qg, kbh.unsqueeze(2).transpose(-2, -1)) * (1.0 / math.sqrt(self.qk_head_dim))
                    update(s.view(B, self.H, 1, -1).to(torch.float32), vbh)

                out = o / d.clamp(min=1e-9).unsqueeze(-1)
                y = self.out_proj(self._merge(out.to(qh.dtype)))
                return y, cache

            # Prefill/chunked (materialize)
            k_all, v_all = cache.get(dtype=x.dtype)
            kh_all = self._shape(k_all, self.qk_head_dim, H=self.H_kv)
            vh_all = self._shape(v_all, self.v_head_dim, H=self.H_kv)
            kh_rep = kh_all.repeat_interleave(self.group_size, dim=1)
            vh_rep = vh_all.repeat_interleave(self.group_size, dim=1)

            scores = torch.matmul(qh, kh_rep.transpose(-2, -1)) / math.sqrt(self.qk_head_dim)
            if attn_mask is not None:
                scores = scores.masked_fill(~attn_mask, ninfty)
            elif T > 1:
                key_pos = torch.arange(L, device=x.device).view(1, 1, 1, L)
                q_pos = (old_len + torch.arange(T, device=x.device)).view(1, 1, T, 1)
                keep = key_pos <= q_pos
                scores = scores.masked_fill(~keep, ninfty)

            if cfg.null_attn:
                k_null = self._shape(self.k_null.expand(B, 1, -1), self.qk_head_dim, H=self.H_kv)
                v_null = self._shape(self.v_null.expand(B, 1, -1), self.v_head_dim, H=self.H_kv)
                k_null_rep = k_null.repeat_interleave(self.group_size, dim=1)
                v_null_rep = v_null.repeat_interleave(self.group_size, dim=1)
                s_null = torch.matmul(qh, k_null_rep.transpose(-2, -1)) / math.sqrt(self.qk_head_dim)
                scores = torch.cat([s_null, scores], dim=-1)
                attn = F.softmax(scores, dim=-1)
                attn = self.drop(attn)
                vals = torch.cat([v_null_rep, vh_rep], dim=-2)
                out = torch.matmul(attn, vals)
            else:
                attn = F.softmax(scores, dim=-1)
                attn = self.drop(attn)
                out = torch.matmul(attn, vh_rep)

            y = self.out_proj(self._merge(out))
            return y, cache

        # -----------------------------
        # decoupled
        # -----------------------------
        assert cfg.attn_mode == "decoupled"

        # Conceptually, we split attention into two additive paths:
        #   score = (Q_sem · K_sem^T) + (Q_geo · K_geo^T)
        # with RoPE applied only to the geometric path.
        if self.k_sem is self.q_sem:
            sem = self.q_sem(x)
            sem_h = self._shape(sem, self.sem_head_dim)
            qsh_base = sem_h
            ksh = sem_h
        else:
            q_sem = self.q_sem(x)
            k_sem = self.k_sem(x)
            qsh_base = self._shape(q_sem, self.sem_head_dim)
            ksh = self._shape(k_sem, self.sem_head_dim)

        if self.k_geo is self.q_geo:
            geo = self.q_geo(x)
            geo_h = self._shape(geo, self.geo_head_dim)
            if self.rotary is not None:
                geo_h = self.rotary.rotate(geo_h, pos_offset)
            qgh_base = geo_h
            kgh = geo_h
        else:
            q_geo = self.q_geo(x)
            k_geo = self.k_geo(x)
            qgh_base = self._shape(q_geo, self.geo_head_dim)
            kgh = self._shape(k_geo, self.geo_head_dim)
            if self.rotary is not None:
                # IMPORTANT: RoPE applies ONLY to geometric/position path.
                qgh_base = self.rotary.rotate(qgh_base, pos_offset)
                kgh = self.rotary.rotate(kgh, pos_offset)

        v = self.v_proj(x)
        vh = self._shape(v, self.v_head_dim)

        # Per-head learned temperature (applied to both paths by scaling queries).
        qsh = self._apply_logit_scale_to_q(qsh_base)
        qgh = self._apply_logit_scale_to_q(qgh_base)

        sem_scale = 1.0 / math.sqrt(self.sem_head_dim)
        geo_scale = 1.0 / math.sqrt(self.geo_head_dim)

        # Optional per-head sem/geo mixing gate.
        # Implemented as a query scaling so it works with both streaming decode and fused kernels.
        if self.decoupled_gate_logit is not None:
            g = torch.sigmoid(self.decoupled_gate_logit).view(1, -1, 1, 1).to(dtype=qsh.dtype, device=qsh.device)
            qsh = qsh * (2.0 * g)  # mean 1.0 at init (g=0.5)
            qgh = qgh * (2.0 * (1.0 - g))

        if cache is None:
            # Training / full-attn:
            # Prefer SDPA for stability (it is much less prone to bf16 softmax overflow than the manual path).
            if not cfg.null_attn:
                # Long-seq auto-optimization: use a semantic "compressed memory" plus full-res local tail.
                # This reduces the dominant O(T^2) term to ~O(T^2 / mem_block + T*local_window) while keeping
                # the geometric path high-resolution in the recent window.
                #
                # Trigger only at long sequence lengths to minimize behavioral drift at typical training ctx.
                if self.training and attn_mask is None:
                    dev = x.device.type
                    # Auto-trigger at long seq: start once we approach the configured block_size (but never below 3k).
                    long_seq_threshold = max(3072, int(0.75 * cfg.block_size))
                    if dev == "mps":
                        long_seq_threshold = min(long_seq_threshold, 3072)
                    if T >= long_seq_threshold:
                        try:
                            if dev == "cuda":
                                mem_block = 128 if T >= 8192 else 64
                                local_window = 1024 if T >= 8192 else 512
                                q_chunk = 256
                            else:
                                mem_block = 64
                                local_window = 512
                                q_chunk = 128

                            local_window = int(min(max(0, local_window), T))
                            mem_block = int(max(1, mem_block))
                            q_chunk = int(max(1, q_chunk))

                            B0, H0, T0, sem_hd = qsh.shape
                            v_hd = int(vh.size(-1))
                            n_blocks = int(T0 // mem_block)

                            q_cat_full = torch.cat([qsh * sem_scale, qgh * geo_scale], dim=-1)

                            if n_blocks > 0:
                                k_sem_blocks = (
                                    ksh[:, :, : n_blocks * mem_block, :]
                                    .reshape(B0, H0, n_blocks, mem_block, sem_hd)
                                    .mean(dim=3)
                                )
                                v_blocks = (
                                    vh[:, :, : n_blocks * mem_block, :]
                                    .reshape(B0, H0, n_blocks, mem_block, v_hd)
                                    .mean(dim=3)
                                )
                                k_mem_cat_all = torch.cat(
                                    [
                                        k_sem_blocks,
                                        k_sem_blocks.new_zeros((B0, H0, n_blocks, self.geo_head_dim)),
                                    ],
                                    dim=-1,
                                )
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

                            y = self.out_proj(self._merge(out))
                            return y, None
                        except Exception:
                            # Safety fallback: if anything goes wrong, use the exact SDPA path.
                            pass

                # Combine sem+geo into a single SDPA call by concatenating along head_dim.
                # q/k last-dim must match; v may have a different last-dim (v_head_dim).
                q_cat, k_cat = _decoupled_qk_cat(q_sem=qsh, q_geo=qgh, k_sem=ksh, k_geo=kgh, sem_scale=sem_scale, geo_scale=geo_scale)
                out = self._sdp(q_cat, k_cat, vh, attn_mask=None if attn_mask is None else attn_mask, scale=1.0, is_causal=True if attn_mask is None else False)
                y = self.out_proj(self._merge(out))
                return y, None

            # Null-attn path (manual).
            scores = _decoupled_scores_f32(q_sem=qsh, q_geo=qgh, k_sem=ksh, k_geo=kgh, sem_scale=sem_scale, geo_scale=geo_scale)

            if attn_mask is not None:
                scores = scores.masked_fill(~attn_mask, ninfty)
            elif T > 1:
                keep = torch.tril(torch.ones((T, T), device=x.device, dtype=torch.bool)).view(1, 1, T, T)
                scores = scores.masked_fill(~keep, ninfty)

            ksn = self._shape(self.k_sem_null.expand(B, 1, -1), self.sem_head_dim)
            kgn = self._shape(self.k_geo_null.expand(B, 1, -1), self.geo_head_dim)
            vn = self._shape(self.v_null.expand(B, 1, -1), self.v_head_dim)
            s_null = _decoupled_scores_f32(q_sem=qsh, q_geo=qgh, k_sem=ksn, k_geo=kgn, sem_scale=sem_scale, geo_scale=geo_scale)
            scores = torch.cat([s_null, scores], dim=-1)

            attn = F.softmax(scores, dim=-1)
            attn = self.drop(attn)
            vals = torch.cat([vn, vh], dim=-2)
            out = torch.matmul(attn, vals)

            y = self.out_proj(self._merge(out))
            return y, None

        assert isinstance(cache, DecoupledLayerKVCache)
        old_len = cache.pos

        if old_len == 0 and T > 1:
            # prefill without cache readback
            if cfg.null_attn:
                scores = _decoupled_scores_f32(q_sem=qsh, q_geo=qgh, k_sem=ksh, k_geo=kgh, sem_scale=sem_scale, geo_scale=geo_scale)
                if attn_mask is not None:
                    scores = scores.masked_fill(~attn_mask, ninfty)
                ksn = self._shape(self.k_sem_null.expand(B, 1, -1), self.sem_head_dim)
                kgn = self._shape(self.k_geo_null.expand(B, 1, -1), self.geo_head_dim)
                vn = self._shape(self.v_null.expand(B, 1, -1), self.v_head_dim)
                s_null = (torch.matmul(qsh, ksn.transpose(-2, -1)) * sem_scale + torch.matmul(qgh, kgn.transpose(-2, -1)) * geo_scale)
                scores = torch.cat([s_null, scores], dim=-1)
                if attn_mask is not None:
                    extra = torch.ones((1, 1, T, 1), device=attn_mask.device, dtype=attn_mask.dtype)
                    keep = torch.cat([extra, attn_mask], dim=-1)
                    scores = scores.masked_fill(~keep, ninfty)
                attn = F.softmax(scores, dim=-1)
                attn = self.drop(attn)
                vals = torch.cat([vn, vh], dim=-2)
                out = torch.matmul(attn, vals)
            else:
                q_cat, k_cat = _decoupled_qk_cat(q_sem=qsh, q_geo=qgh, k_sem=ksh, k_geo=kgh, sem_scale=sem_scale, geo_scale=geo_scale)
                out = self._sdp(q_cat, k_cat, vh, attn_mask=None if attn_mask is None else attn_mask, scale=1.0, is_causal=True if attn_mask is None else False)

            y = self.out_proj(self._merge(out))
            cache.append(self._merge(ksh), self._merge(kgh), self._merge(vh))
            return y, cache

        # General cached path:
        # - T==1: decode (streaming or fused)
        # - T>1: prefer a sequential streaming path when caches are quantized (avoids dequantizing the full prefix),
        #        otherwise fall back to the materialized path.

        if T == 1:
            cache.append(self._merge(ksh), self._merge(kgh), self._merge(vh))
            decode_block = getattr(cache, "decode_block", 1024)
            # fp16 cache -> materialize and use SDPA (usually faster than Python streaming)
            if (not cfg.null_attn) and (not cache.k_sem.is_quantized) and (not cache.k_geo.is_quantized) and (not cache.v.is_quantized):
                k_sem_all = self._shape(cache.k_sem.get(dtype=qsh.dtype), self.sem_head_dim)
                k_geo_all = self._shape(cache.k_geo.get(dtype=qsh.dtype), self.geo_head_dim)
                v_all = self._shape(cache.v.get(dtype=qsh.dtype), self.v_head_dim)
                q_cat, k_cat = _decoupled_qk_cat(q_sem=qsh, q_geo=qgh, k_sem=k_sem_all, k_geo=k_geo_all, sem_scale=sem_scale, geo_scale=geo_scale)
                # Important: decode uses is_causal=False here because query length is 1 but its logical
                # position is the *end* of the prefix; SDPA's built-in causal mask would treat it as position 0.
                out = self._sdp(q_cat, k_cat, v_all, attn_mask=None, scale=1.0, is_causal=False)
            else:
                # Quantized/null-attn -> streaming or fused decode.
                if cfg.null_attn:
                    ksn = self._shape(self.k_sem_null.expand(B, 1, -1), self.sem_head_dim)
                    kgn = self._shape(self.k_geo_null.expand(B, 1, -1), self.geo_head_dim)
                    vn = self._shape(self.v_null.expand(B, 1, -1), self.v_head_dim)
                else:
                    ksn = kgn = vn = None

                def fused_ok() -> bool:
                    return bool(
                        (not cfg.null_attn)
                        and _triton_decoupled_q4q8q4_available()
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
                                sem_head_dim=self.sem_head_dim,
                                geo_head_dim=self.geo_head_dim,
                                v_head_dim=self.v_head_dim,
                                decode_block=decode_block,
                                sem_scale=float(sem_scale),
                                geo_scale=float(geo_scale),
                            )
                        else:
                            out = self._fused_decode_attn_decoupled_q4q8q4_2pass(
                                q_sem=qsh,
                                q_geo=qgh,
                                cache=cache,
                                sem_head_dim=self.sem_head_dim,
                                geo_head_dim=self.geo_head_dim,
                                v_head_dim=self.v_head_dim,
                                decode_block=decode_block,
                                sem_scale=float(sem_scale),
                                geo_scale=float(geo_scale),
                            )
                    except Exception:
                        out = self._streaming_decode_attn_decoupled(
                            q_sem=qsh,
                            q_geo=qgh,
                            k_sem_cache=cache.k_sem,
                            k_geo_cache=cache.k_geo,
                            v_cache=cache.v,
                            sem_head_dim=self.sem_head_dim,
                            geo_head_dim=self.geo_head_dim,
                            v_head_dim=self.v_head_dim,
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
                        sem_head_dim=self.sem_head_dim,
                        geo_head_dim=self.geo_head_dim,
                        v_head_dim=self.v_head_dim,
                        decode_block=decode_block,
                        sem_scale=float(sem_scale),
                        geo_scale=float(geo_scale),
                        k_sem_null=ksn,
                        k_geo_null=kgn,
                        v_null=vn,
                    )

            y = self.out_proj(self._merge(out))
            return y, cache

        # T > 1
        decode_block = getattr(cache, "decode_block", 1024)
        quantized_cache = bool(cache.k_sem.is_quantized or cache.k_geo.is_quantized or cache.v.is_quantized)
        if quantized_cache and old_len > 0:
            # Sequential streaming prefill: append token-by-token and use streaming decode for each token.
            # This avoids materializing/dequantizing the full prefix for every token in the chunk.
            if cfg.null_attn:
                ksn = self._shape(self.k_sem_null.expand(B, 1, -1), self.sem_head_dim)
                kgn = self._shape(self.k_geo_null.expand(B, 1, -1), self.geo_head_dim)
                vn = self._shape(self.v_null.expand(B, 1, -1), self.v_head_dim)
            else:
                ksn = kgn = vn = None

            outs: List[torch.Tensor] = []
            for t in range(T):
                # Append one token worth of KV to the cache.
                cache.append(
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
                    sem_head_dim=self.sem_head_dim,
                    geo_head_dim=self.geo_head_dim,
                    v_head_dim=self.v_head_dim,
                    decode_block=decode_block,
                    sem_scale=float(sem_scale),
                    geo_scale=float(geo_scale),
                    k_sem_null=ksn,
                    k_geo_null=kgn,
                    v_null=vn,
                )
                outs.append(out_t)

            out = torch.cat(outs, dim=2)  # (B,H,T,hd_v)
            y = self.out_proj(self._merge(out))
            return y, cache

        # Prefill/chunked attention (materialize).
        cache.append(self._merge(ksh), self._merge(kgh), self._merge(vh))
        L = cache.pos
        k_sem_all, k_geo_all, v_all = cache.get(dtype=x.dtype)
        ksh_all = self._shape(k_sem_all, self.sem_head_dim)
        kgh_all = self._shape(k_geo_all, self.geo_head_dim)
        vh_all = self._shape(v_all, self.v_head_dim)

        scores = _decoupled_scores_f32(q_sem=qsh, q_geo=qgh, k_sem=ksh_all, k_geo=kgh_all, sem_scale=sem_scale, geo_scale=geo_scale)

        if attn_mask is not None:
            scores = scores.masked_fill(~attn_mask, ninfty)
        elif T > 1:
            key_pos = torch.arange(L, device=x.device).view(1, 1, 1, L)
            q_pos = (old_len + torch.arange(T, device=x.device)).view(1, 1, T, 1)
            keep = key_pos <= q_pos
            scores = scores.masked_fill(~keep, ninfty)

        if cfg.null_attn:
            ksn = self._shape(self.k_sem_null.expand(B, 1, -1), self.sem_head_dim)
            kgn = self._shape(self.k_geo_null.expand(B, 1, -1), self.geo_head_dim)
            vn = self._shape(self.v_null.expand(B, 1, -1), self.v_head_dim)
            s_null = _decoupled_scores_f32(q_sem=qsh, q_geo=qgh, k_sem=ksn, k_geo=kgn, sem_scale=sem_scale, geo_scale=geo_scale)
            scores = torch.cat([s_null, scores], dim=-1)
            attn = F.softmax(scores, dim=-1)
            attn = self.drop(attn)
            vals = torch.cat([vn, vh_all], dim=-2)
            out = torch.matmul(attn, vals)
        else:
            attn = F.softmax(scores, dim=-1)
            attn = self.drop(attn)
            out = torch.matmul(attn, vh_all)

        y = self.out_proj(self._merge(out))
        return y, cache
