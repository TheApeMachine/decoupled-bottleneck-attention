"""
Small pure helpers for decoupled attention.

Used by both the attention layer and optional fused decode paths.
Keeping them pure and separate makes correctness and benchmarking easier.
"""

from __future__ import annotations

import torch

__all__ = ["neg_inf", "decoupled_qk_cat", "decoupled_scores_f32"]


def neg_inf(dtype: torch.dtype) -> float:
    """
    A large negative "mask" value.

    This must be representable in the *compute dtype* used by attention.
    Some backends (notably torch.compile+inductor on MPS) can error when constant-folding
    float32-min sentinels into bf16/fp16.
    """
    _ = dtype
    # exp(-1e4) underflows to 0 for all practical purposes, across fp16/bf16/fp32.
    return -1.0e4


def decoupled_qk_cat(
    *,
    q_sem: torch.Tensor,
    q_geo: torch.Tensor,
    k_sem: torch.Tensor,
    k_geo: torch.Tensor,
    sem_scale: float,
    geo_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build composite (q_cat, k_cat) for decoupled attention.

    Allows a single SDPA call to implement decoupled scores while delegating softmax
    numerics to the backend.
    Guarantees score equivalence:
        (q_cat @ k_cat^T) == (q_sem @ k_sem^T) * sem_scale + (q_geo @ k_geo^T) * geo_scale
    """
    q_cat = torch.cat([q_sem * float(sem_scale), q_geo * float(geo_scale)], dim=-1)
    k_cat = torch.cat([k_sem, k_geo], dim=-1)
    return q_cat, k_cat


def decoupled_scores_f32(
    *,
    q_sem: torch.Tensor,
    q_geo: torch.Tensor,
    k_sem: torch.Tensor,
    k_geo: torch.Tensor,
    sem_scale: float,
    geo_scale: float,
) -> torch.Tensor:
    """
    Single source of truth for decoupled score computation in fp32.

    Keeps score math identical across SDPA/manual/streaming/fused paths.
    fp32 accumulation avoids mixed-precision drift when validating policies.
    """
    sem = torch.matmul(q_sem, k_sem.transpose(-2, -1)).to(torch.float32) * float(sem_scale)
    geo = torch.matmul(q_geo, k_geo.transpose(-2, -1)).to(torch.float32) * float(geo_scale)
    return sem + geo


