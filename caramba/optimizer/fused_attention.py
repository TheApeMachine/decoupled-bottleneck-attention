"""Fused decoupled attention using Triton kernels when available.

This module provides optimized attention implementations that:
1. Fall back to pure PyTorch on MPS/CPU
2. Use FlashAttention-style Triton kernels on CUDA for decode
3. Support quantized KV caches (q4_0 for K_sem/V, q8_0 for K_geo)

The fused decode path provides significant speedups for long sequences
by avoiding Python-loop-based cache slicing and fusing dequant+attention+softmax.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, cast, Callable, Protocol

import torch
import torch.nn.functional as F

from caramba.optimizer.triton_runtime import TRITON_AVAILABLE
from caramba.optimizer.kernels_decoupled import (
    kv_decode_update_decoupled_q4q8q4,
    kv_decode_partition_stats_decoupled_q4q8q4,
    kv_decode_reduce_partitions,
)

if TYPE_CHECKING:
    from caramba.cache.decoupled import DecoupledLayerKVCache


__all__ = [
    "fused_decode_available",
    "fused_decode_decoupled_q4q8q4",
    "fused_decode_decoupled_q4q8q4_2pass",
    "decoupled_scores_f32",
    "decoupled_qk_cat",
]


class _Kernel(Protocol):
    """Minimal Triton kernel interface (`kernel[grid](...)`)."""
    def __getitem__(self, grid: tuple[int, ...]) -> Callable[..., object]: ...


_kv_decode_update_decoupled_q4q8q4 = cast(_Kernel | None, kv_decode_update_decoupled_q4q8q4)
_kv_decode_partition_stats_decoupled_q4q8q4 = cast(_Kernel | None, kv_decode_partition_stats_decoupled_q4q8q4)
_kv_decode_reduce_partitions = cast(_Kernel | None, kv_decode_reduce_partitions)


def fused_decode_available(cache: "DecoupledLayerKVCache", device_type: str) -> bool:
    """Check if fused decode can be used for the given cache and device."""
    if not TRITON_AVAILABLE:
        return False
    if device_type != "cuda":
        return False
    # Check cache quantization format
    k_sem = cache.k_sem
    k_geo = cache.k_geo
    v = cache.v
    if not (k_sem.kind == "q4_0" and k_geo.kind == "q8_0" and v.kind == "q4_0"):
        return False
    if not (k_sem.spec.qblock == 32 and k_geo.spec.qblock == 32 and v.spec.qblock == 32):
        return False
    return True


def decoupled_scores_f32(
    *,
    q_sem: torch.Tensor,
    q_geo: torch.Tensor,
    k_sem: torch.Tensor,
    k_geo: torch.Tensor,
    sem_scale: float,
    geo_scale: float,
) -> torch.Tensor:
    """Single source of truth for decoupled score computation in fp32.

    Keeps score math identical across SDPA/manual/streaming/fused paths.
    fp32 accumulation avoids mixed-precision drift when validating policies.
    """
    sem = torch.matmul(q_sem, k_sem.transpose(-2, -1)).to(torch.float32) * float(sem_scale)
    geo = torch.matmul(q_geo, k_geo.transpose(-2, -1)).to(torch.float32) * float(geo_scale)
    return sem + geo


def decoupled_qk_cat(
    *,
    q_sem: torch.Tensor,
    q_geo: torch.Tensor,
    k_sem: torch.Tensor,
    k_geo: torch.Tensor,
    sem_scale: float,
    geo_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build composite (q_cat, k_cat) for decoupled attention.

    Allows a single SDPA call to implement decoupled scores while delegating
    softmax numerics to the backend.

    Guarantees score equivalence:
        (q_cat @ k_cat^T) == (q_sem @ k_sem^T) * sem_scale + (q_geo @ k_geo^T) * geo_scale
    """
    q_cat = torch.cat([q_sem * float(sem_scale), q_geo * float(geo_scale)], dim=-1)
    k_cat = torch.cat([k_sem, k_geo], dim=-1)
    return q_cat, k_cat


def _shape(x: torch.Tensor, head_dim: int, n_heads: int) -> torch.Tensor:
    """Reshape (B, T, D) -> (B, H, T, head_dim)."""
    B, T, _ = x.shape
    return x.view(B, T, n_heads, head_dim).transpose(1, 2).contiguous()


def fused_decode_decoupled_q4q8q4(
    *,
    q_sem: torch.Tensor,
    q_geo: torch.Tensor,
    cache: "DecoupledLayerKVCache",
    n_heads: int,
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
    """Fused single-pass decode for quantized decoupled cache.

    Args:
        q_sem: Semantic query (B, H, 1, sem_head_dim)
        q_geo: Geometric query (B, H, 1, geo_head_dim)
        cache: Decoupled KV cache with q4_0/q8_0/q4_0 quantization
        n_heads: Number of attention heads
        sem_head_dim: Dimension per head for semantic keys
        geo_head_dim: Dimension per head for geometric keys
        v_head_dim: Dimension per head for values
        decode_block: Block size for streaming decode
        sem_scale: Scaling factor for semantic scores (1/sqrt(sem_head_dim))
        geo_scale: Scaling factor for geometric scores (1/sqrt(geo_head_dim))
        k_sem_null: Optional null semantic key for null attention (B, H, 1, sem_head_dim)
        k_geo_null: Optional null geometric key for null attention (B, H, 1, geo_head_dim)
        v_null: Optional null value for null attention (B, H, 1, v_head_dim)

    Returns:
        Attention output (B, H, 1, v_head_dim)
    """
    if not TRITON_AVAILABLE:
        raise RuntimeError("Triton not available")
    if q_sem.device.type != "cuda":
        raise RuntimeError("Fused decode requires CUDA")

    kernel = _kv_decode_update_decoupled_q4q8q4
    if kernel is None:
        raise RuntimeError("Fused decode kernels are unavailable")

    B, H, Tq, _ = q_sem.shape
    assert Tq == 1, "Fused decode only supports single-token queries"

    cache_len = cache.pos

    # Get residual start position (tokens in fp16 residual buffer)
    k_sem_cache = cache.k_sem
    rlen = int(getattr(k_sem_cache, "_residual_len_eff", 0)) if getattr(k_sem_cache, "_residual", None) is not None else 0
    r_start = max(0, cache_len - rlen) if rlen > 0 else cache_len
    L_prefix = int(r_start)

    # Prepare query tensors (flatten batch and head dims)
    q_sem2 = q_sem[:, :, 0, :].contiguous().to(torch.float16)
    q_geo2 = q_geo[:, :, 0, :].contiguous().to(torch.float16)

    has_null = k_sem_null is not None and k_geo_null is not None and v_null is not None
    if has_null and k_sem_null is not None and k_geo_null is not None and v_null is not None:
        ksn = k_sem_null[:, :, 0, :].contiguous().to(torch.float16)
        kgn = k_geo_null[:, :, 0, :].contiguous().to(torch.float16)
        vn = v_null[:, :, 0, :].contiguous().to(torch.float16)
    else:
        # Placeholder tensors (not used when HAS_NULL=False)
        ksn = q_sem2.new_zeros((B, H, sem_head_dim))
        kgn = q_geo2.new_zeros((B, H, geo_head_dim))
        vn = q_sem2.new_zeros((B, H, v_head_dim))

    BH = B * H
    m = torch.full((BH,), -float("inf"), device=q_sem.device, dtype=torch.float32)
    d = torch.zeros((BH,), device=q_sem.device, dtype=torch.float32)
    o = torch.zeros((BH, v_head_dim), device=q_sem.device, dtype=torch.float32)

    # Handle null token initialization
    if has_null and L_prefix == 0:
        s_null = (q_sem2.float() * ksn.float()).sum(dim=-1) * float(sem_scale) + \
                 (q_geo2.float() * kgn.float()).sum(dim=-1) * float(geo_scale)
        m = s_null.reshape(BH)
        d = torch.ones((BH,), device=q_sem.device, dtype=torch.float32)
        o = vn.float().reshape(BH, v_head_dim)

    # Process quantized prefix with fused kernel
    if L_prefix > 0:
        block_n = 128
        num_sub = max(1, int(decode_block // block_n))
        step = block_n * num_sub
        num_warps = 4
        num_stages = 2

        ksq_ = cache.k_sem.q
        kss_ = cache.k_sem.s
        kgq_ = cache.k_geo.q
        kgs_ = cache.k_geo.s
        vq_ = cache.v.q
        vs_ = cache.v.s

        if ksq_ is None or kss_ is None or kgq_ is None or kgs_ is None or vq_ is None or vs_ is None:
            raise RuntimeError("Cache quantized buffers not initialized")

        # Reassign after None check for type narrowing
        ksq, kss, kgq, kgs, vq, vs = ksq_, kss_, kgq_, kgs_, vq_, vs_

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

    # Handle residual tail (recent tokens in fp16)
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

        ksh = _shape(k_sem_blk, sem_head_dim, n_heads)
        kgh = _shape(k_geo_blk, geo_head_dim, n_heads)
        vbh = _shape(v_blk, v_head_dim, n_heads)

        s = decoupled_scores_f32(
            q_sem=qsh, q_geo=qgh, k_sem=ksh, k_geo=kgh,
            sem_scale=sem_scale, geo_scale=geo_scale
        )
        update(s, vbh)

        m = m_t.view(BH)
        d = d_t.view(BH)
        o = o_t.view(BH, v_head_dim)

    out = (o / d.clamp(min=1e-9).unsqueeze(-1)).view(B, H, 1, v_head_dim)
    return out.to(q_sem.dtype)


def fused_decode_decoupled_q4q8q4_2pass(
    *,
    q_sem: torch.Tensor,
    q_geo: torch.Tensor,
    cache: "DecoupledLayerKVCache",
    n_heads: int,
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
    """Fused two-pass (split-K) decode for very long sequences.

    This uses FlashAttention-2-style parallelism across the sequence length:
    1. Partition kernel: Each partition computes local (m, d, o) stats
    2. Reduce kernel: Combine partition stats using online softmax

    Use this for cache_len > 4 * decode_block for better GPU utilization.
    """
    if not TRITON_AVAILABLE:
        raise RuntimeError("Triton not available")
    if q_sem.device.type != "cuda":
        raise RuntimeError("Fused decode requires CUDA")

    kernel_part = _kv_decode_partition_stats_decoupled_q4q8q4
    kernel_reduce = _kv_decode_reduce_partitions
    if kernel_part is None or kernel_reduce is None:
        raise RuntimeError("Fused decode kernels are unavailable")

    B, H, Tq, _ = q_sem.shape
    assert Tq == 1
    cache_len = cache.pos

    # Get residual positions
    k_sem_cache = cache.k_sem
    rlen = int(getattr(k_sem_cache, "_residual_len_eff", 0)) if getattr(k_sem_cache, "_residual", None) is not None else 0
    r_start = max(0, cache_len - rlen) if rlen > 0 else cache_len
    L_prefix = int(r_start)

    q_sem2 = q_sem[:, :, 0, :].contiguous().to(torch.float16)
    q_geo2 = q_geo[:, :, 0, :].contiguous().to(torch.float16)

    has_null = k_sem_null is not None and k_geo_null is not None and v_null is not None
    if has_null and k_sem_null is not None and k_geo_null is not None and v_null is not None:
        ksn = k_sem_null[:, :, 0, :].contiguous().to(torch.float16)
        kgn = k_geo_null[:, :, 0, :].contiguous().to(torch.float16)
        vn = v_null[:, :, 0, :].contiguous().to(torch.float16)
    else:
        ksn = kgn = vn = q_sem2  # Placeholder

    BH = B * H

    # Partition parameters
    block_n = 128
    num_sub = max(1, int(decode_block // block_n))
    partition_size = block_n * num_sub
    num_parts = int(math.ceil(max(1, L_prefix) / float(partition_size))) if L_prefix > 0 else 0
    num_parts_cap = int(max(1, num_parts))
    P = int(num_parts)

    # Scratch buffers for partition stats
    m_part = torch.empty((BH, num_parts_cap), device=q_sem.device, dtype=torch.float32)
    d_part = torch.empty((BH, num_parts_cap), device=q_sem.device, dtype=torch.float32)
    o_part = torch.empty((BH, num_parts_cap, v_head_dim), device=q_sem.device, dtype=torch.float32)

    m = torch.empty((BH,), device=q_sem.device, dtype=torch.float32)
    d = torch.empty((BH,), device=q_sem.device, dtype=torch.float32)
    o = torch.empty((BH, v_head_dim), device=q_sem.device, dtype=torch.float32)

    # Pass 1: Partition stats
    if L_prefix > 0:
        ksq_ = cache.k_sem.q
        kss_ = cache.k_sem.s
        kgq_ = cache.k_geo.q
        kgs_ = cache.k_geo.s
        vq_ = cache.v.q
        vs_ = cache.v.s

        if ksq_ is None or kss_ is None or kgq_ is None or kgs_ is None or vq_ is None or vs_ is None:
            raise RuntimeError("Cache quantized buffers not initialized")

        ksq, kss, kgq, kgs, vq, vs = ksq_, kss_, kgq_, kgs_, vq_, vs_

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
            num_warps=4,
            num_stages=2,
        )

    # Pass 2: Reduce partitions
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
        num_warps=1,
        num_stages=1,
    )

    # Handle residual tail
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

        ksh = _shape(k_sem_blk, sem_head_dim, n_heads)
        kgh = _shape(k_geo_blk, geo_head_dim, n_heads)
        vbh = _shape(v_blk, v_head_dim, n_heads)

        s = decoupled_scores_f32(
            q_sem=qsh, q_geo=qgh, k_sem=ksh, k_geo=kgh,
            sem_scale=sem_scale, geo_scale=geo_scale
        )
        update(s, vbh)

        o = o_t.view(BH, v_head_dim)
        d = d_t.view(BH)

    out = (o / d.clamp(min=1e-9).unsqueeze(-1)).view(B, H, 1, v_head_dim)
    return out.to(q_sem.dtype)
