"""Memory estimation helpers for KV-cache tuning."""

from __future__ import annotations

__all__ = [
    "as_mb",
    "estimate_decoupled_kvcache_bytes",
    "estimate_seq_cache_bytes",
]

from production.kvcache_backend import KVCacheTensorConfig, make_quantspec

from .cache_policy import KVCachePolicy


def estimate_seq_cache_bytes(*, batch_size: int, max_seq_len: int, dim: int, cfg: KVCacheTensorConfig) -> int:
    """Estimate the memory footprint of a single sequence cache tensor."""
    batch_size = int(batch_size)
    max_seq_len = int(max_seq_len)
    dim = int(dim)
    kind = str(cfg.kind)

    if kind == "fp16":
        return batch_size * max_seq_len * dim * 2
    if kind == "fp32":
        return batch_size * max_seq_len * dim * 4

    spec = make_quantspec(cfg.kind, dim, cfg.qblock)

    if kind == "q8_0":
        q_bytes = batch_size * max_seq_len * spec.pad_dim * 1
        s_bytes = batch_size * max_seq_len * spec.n_blocks * 2
    elif kind in ("q4_0", "nf4"):
        q_bytes = batch_size * max_seq_len * (spec.pad_dim // 2) * 1
        s_bytes = batch_size * max_seq_len * spec.n_blocks * 2
    else:
        raise ValueError(kind)

    rlen = int(max(0, cfg.residual_len))
    r_eff = min(rlen, max_seq_len)
    r_bytes = batch_size * r_eff * dim * 2

    return int(q_bytes + s_bytes + r_bytes)


def estimate_decoupled_kvcache_bytes(
    *,
    n_layer: int,
    batch_size: int,
    max_seq_len: int,
    sem_dim: int,
    geo_dim: int,
    v_dim: int,
    policy: KVCachePolicy,
) -> int:
    """Estimate the memory footprint of a decoupled KV cache."""
    k_sem_cfg, k_geo_cfg, v_cfg = policy.to_tensor_cfgs()
    per_layer = (
        estimate_seq_cache_bytes(batch_size=batch_size, max_seq_len=max_seq_len, dim=sem_dim, cfg=k_sem_cfg)
        + estimate_seq_cache_bytes(batch_size=batch_size, max_seq_len=max_seq_len, dim=geo_dim, cfg=k_geo_cfg)
        + estimate_seq_cache_bytes(batch_size=batch_size, max_seq_len=max_seq_len, dim=v_dim, cfg=v_cfg)
    )
    return int(n_layer) * int(per_layer)


def as_mb(n_bytes: int) -> float:
    """Convert bytes to MiB."""
    return float(n_bytes) / (1024.0 * 1024.0)


# Back-compat alias (avoid importing underscore names across modules).
_as_mb = as_mb


