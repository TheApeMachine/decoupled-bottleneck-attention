"""
attention_math provides small, pure attention helpers.
"""

from __future__ import annotations

import torch
from torch import Tensor


def shape_heads(x: Tensor, *, n_heads: int, head_dim: int) -> Tensor:
    """
    shape_heads reshapes (B,T,H*D) -> (B,H,T,D).
    """
    if x.ndim != 3:
        raise ValueError(f"Expected rank-3 (B,T,D), got {x.shape}")
    if n_heads <= 0:
        raise ValueError(f"n_heads must be > 0, got {n_heads}")
    if head_dim <= 0:
        raise ValueError(f"head_dim must be > 0, got {head_dim}")
    if x.shape[-1] != n_heads * head_dim:
        raise ValueError(
            "Expected last dim to equal n_heads*head_dim, got "
            f"x={x.shape}, n_heads={n_heads}, head_dim={head_dim}"
        )

    b, t, _ = x.shape
    return x.view(b, t, int(n_heads), int(head_dim)).transpose(1, 2).contiguous()


def decoupled_qk_cat(
    *,
    q_sem: Tensor,
    q_geo: Tensor,
    k_sem: Tensor,
    k_geo: Tensor,
    sem_scale: float,
    geo_scale: float,
) -> tuple[Tensor, Tensor]:
    """
    Build composite (q_cat, k_cat) for decoupled attention.

    Guarantees score equivalence:
        (q_cat @ k_cat^T) == (q_sem @ k_sem^T) * sem_scale
                       + (q_geo @ k_geo^T) * geo_scale
    """
    q_cat = torch.cat(
        [q_sem * float(sem_scale), q_geo * float(geo_scale)],
        dim=-1,
    )
    k_cat = torch.cat([k_sem, k_geo], dim=-1)
    return q_cat, k_cat


