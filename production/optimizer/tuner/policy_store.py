"""Persistence for cache-policy tuning results."""

from __future__ import annotations

from typing import cast

from production.selfopt_cache import get_cache_entry, set_cache_entry

from production.kvcache_backend import KVCacheKind
from production.optimizer.tuner.cache_policy import KVCachePolicy


class PolicyStore:
    """Persist cache policies as a single JSON entry (per cache path)."""

    def __init__(self, cache_path: str | None) -> None:
        self.cache_path: str | None = cache_path
        self._raw: dict[str, dict[str, object]] = self._load_raw()

    def _load_raw(self) -> dict[str, dict[str, object]]:
        if not self.cache_path:
            return {}
        try:
            cp = get_cache_entry(self.cache_path, section="cache_policies", key="__all__")
            if not isinstance(cp, dict):
                return {}
            out: dict[str, dict[str, object]] = {}
            raw_map = cast(dict[object, object], cp)
            for k, v in raw_map.items():
                if isinstance(v, dict):
                    inner = cast(dict[object, object], v)
                    out[str(k)] = {str(kk): vv for kk, vv in inner.items()}
            return out
        except (TypeError, OSError, ValueError):
            return {}

    def get(self, key: str) -> KVCachePolicy | None:
        """Get a cached policy by key."""
        raw = self._raw.get(str(key))
        if raw is None:
            return None

        def _as_kind(x: object) -> KVCacheKind:
            s = str(x)
            if s in ("fp16", "fp32", "q8_0", "q4_0", "nf4"):
                return s
            raise ValueError(f"Unknown KVCacheKind: {s!r}")

        try:
            return KVCachePolicy(
                k_sem_kind=_as_kind(raw["k_sem_kind"]),
                k_geo_kind=_as_kind(raw["k_geo_kind"]),
                v_kind=_as_kind(raw["v_kind"]),
                k_sem_qblock=int(str(raw["k_sem_qblock"])),
                k_geo_qblock=int(str(raw["k_geo_qblock"])),
                v_qblock=int(str(raw["v_qblock"])),
                residual_len=int(str(raw["residual_len"])),
            )
        except (KeyError, TypeError, ValueError):
            return None

    def set(self, key: str, policy: KVCachePolicy) -> None:
        """Set a cached policy (in-memory)."""
        self._raw[str(key)] = {
            "k_sem_kind": policy.k_sem_kind,
            "k_geo_kind": policy.k_geo_kind,
            "v_kind": policy.v_kind,
            "k_sem_qblock": int(policy.k_sem_qblock),
            "k_geo_qblock": int(policy.k_geo_qblock),
            "v_qblock": int(policy.v_qblock),
            "residual_len": int(policy.residual_len),
        }

    def save(self) -> None:
        """Persist all cached policies (best-effort)."""
        if not self.cache_path:
            return
        try:
            set_cache_entry(
                str(self.cache_path),
                section="cache_policies",
                key="__all__",
                value=self._raw,
            )
        except (OSError, ValueError, TypeError):
            pass


