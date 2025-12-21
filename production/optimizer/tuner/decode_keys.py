"""Keying helpers for decode-plan caching."""

from __future__ import annotations

import torch

from production.selfopt_utils import device_sig

from production.optimizer.tuner.buckets import pow2_bucket


def decode_plan_key(*, device: torch.device, attn: object, cache: object, prefix_len: int) -> str:
    """Stable key for caching decode plans."""
    bucket = pow2_bucket(int(prefix_len))
    try:
        ksig = (
            f"ksem={getattr(getattr(cache, 'k_sem'), 'kind')},"
            f"kgeo={getattr(getattr(cache, 'k_geo'), 'kind')},"
            f"v={getattr(getattr(cache, 'v'), 'kind')}"
        )
    except (AttributeError, TypeError):
        ksig = "kv=unknown"
    try:
        dims = (
            f"H={getattr(attn, 'H')},"
            f"hd_sem={getattr(attn, 'sem_head_dim')},"
            f"hd_geo={getattr(attn, 'geo_head_dim')},"
            f"hd_v={getattr(attn, 'v_head_dim')}"
        )
    except (AttributeError, TypeError):
        dims = "dims=unknown"
    return f"{device_sig(device)}|{bucket}|{dims}|{ksig}"


