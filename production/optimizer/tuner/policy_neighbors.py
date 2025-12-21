"""Local neighborhood generator for cache-policy tuning."""

from __future__ import annotations

from production.kvcache_backend import KVCacheKind

from production.optimizer.tuner.cache_policy import KVCachePolicy
from production.optimizer.tuner.config import KVSelfOptConfig


def neighbors(cfg: KVSelfOptConfig, p: KVCachePolicy) -> list[KVCachePolicy]:
    """Small local neighborhood around `p` (1-step mutations)."""
    resid_cands = sorted({int(x) for x in cfg.residuals if int(x) >= 0})
    qb_cands = sorted({int(x) for x in cfg.qblocks if int(x) > 0})

    def neigh_num(cur: int, cands: list[int]) -> list[int]:
        if cur not in cands:
            cands = sorted(set(cands + [cur]))
        i = cands.index(cur)
        out: list[int] = []
        if i - 1 >= 0:
            out.append(cands[i - 1])
        if i + 1 < len(cands):
            out.append(cands[i + 1])
        return out

    def neigh_kind(cur: KVCacheKind, cands: tuple[KVCacheKind, ...]) -> list[KVCacheKind]:
        c = list(cands)
        if cur not in c:
            c = list(dict.fromkeys([cur] + c))
        i = c.index(cur)
        out: list[KVCacheKind] = []
        if i - 1 >= 0:
            out.append(c[i - 1])
        if i + 1 < len(c):
            out.append(c[i + 1])
        return out

    out: list[KVCachePolicy] = []

    if p.k_sem_kind != p.k_geo_kind:
        out.append(
            KVCachePolicy(
                k_sem_kind=p.k_geo_kind,
                k_geo_kind=p.k_sem_kind,
                v_kind=p.v_kind,
                k_sem_qblock=p.k_sem_qblock,
                k_geo_qblock=p.k_geo_qblock,
                v_qblock=p.v_qblock,
                residual_len=p.residual_len,
            )
        )

    for r in neigh_num(int(p.residual_len), resid_cands):
        out.append(
            KVCachePolicy(
                k_sem_kind=p.k_sem_kind,
                k_geo_kind=p.k_geo_kind,
                v_kind=p.v_kind,
                k_sem_qblock=p.k_sem_qblock,
                k_geo_qblock=p.k_geo_qblock,
                v_qblock=p.v_qblock,
                residual_len=r,
            )
        )

    for qb in neigh_num(int(p.k_sem_qblock), qb_cands):
        out.append(
            KVCachePolicy(
                k_sem_kind=p.k_sem_kind,
                k_geo_kind=p.k_geo_kind,
                v_kind=p.v_kind,
                k_sem_qblock=qb,
                k_geo_qblock=qb,
                v_qblock=qb,
                residual_len=p.residual_len,
            )
        )

    for k in neigh_kind(p.k_sem_kind, cfg.k_sem_kinds):
        out.append(
            KVCachePolicy(
                k_sem_kind=k,
                k_geo_kind=p.k_geo_kind,
                v_kind=p.v_kind,
                k_sem_qblock=p.k_sem_qblock,
                k_geo_qblock=p.k_geo_qblock,
                v_qblock=p.v_qblock,
                residual_len=p.residual_len,
            )
        )
    for k in neigh_kind(p.k_geo_kind, cfg.k_geo_kinds):
        out.append(
            KVCachePolicy(
                k_sem_kind=p.k_sem_kind,
                k_geo_kind=k,
                v_kind=p.v_kind,
                k_sem_qblock=p.k_sem_qblock,
                k_geo_qblock=p.k_geo_qblock,
                v_qblock=p.v_qblock,
                residual_len=p.residual_len,
            )
        )
    for k in neigh_kind(p.v_kind, cfg.v_kinds):
        out.append(
            KVCachePolicy(
                k_sem_kind=p.k_sem_kind,
                k_geo_kind=p.k_geo_kind,
                v_kind=k,
                k_sem_qblock=p.k_sem_qblock,
                k_geo_qblock=p.k_geo_qblock,
                v_qblock=p.v_qblock,
                residual_len=p.residual_len,
            )
        )

    uniq: list[KVCachePolicy] = []
    seen: set[str] = set()
    for cand in out:
        key = cand.short()
        if key not in seen:
            seen.add(key)
            uniq.append(cand)
    return uniq


