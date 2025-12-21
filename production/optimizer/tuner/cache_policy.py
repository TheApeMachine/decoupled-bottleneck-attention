"""KV-cache policy representation for runtime tuning."""

from __future__ import annotations

from dataclasses import dataclass

from production.kvcache_backend import KVCacheKind, KVCacheTensorConfig


@dataclass(frozen=True)
class KVCachePolicy:
    """A cache policy for a decoupled KV cache."""

    k_sem_kind: KVCacheKind
    k_geo_kind: KVCacheKind
    v_kind: KVCacheKind
    k_sem_qblock: int
    k_geo_qblock: int
    v_qblock: int
    residual_len: int

    def _residual_for(self, kind: KVCacheKind) -> int:
        return int(self.residual_len) if kind not in ("fp16", "fp32") else 0

    def to_tensor_cfgs(self) -> tuple[KVCacheTensorConfig, KVCacheTensorConfig, KVCacheTensorConfig]:
        """Convert the policy into tensor-level configs."""
        k_sem = KVCacheTensorConfig(
            kind=self.k_sem_kind,
            qblock=int(self.k_sem_qblock),
            residual_len=self._residual_for(self.k_sem_kind),
        )
        k_geo = KVCacheTensorConfig(
            kind=self.k_geo_kind,
            qblock=int(self.k_geo_qblock),
            residual_len=self._residual_for(self.k_geo_kind),
        )
        v = KVCacheTensorConfig(
            kind=self.v_kind,
            qblock=int(self.v_qblock),
            residual_len=self._residual_for(self.v_kind),
        )
        return k_sem, k_geo, v

    def short(self) -> str:
        """Return a compact string representation."""
        return (
            f"ksem={self.k_sem_kind}@{self.k_sem_qblock},"
            f"kgeo={self.k_geo_kind}@{self.k_geo_qblock},"
            f"v={self.v_kind}@{self.v_qblock},"
            f"resid={self.residual_len}"
        )

    @classmethod
    def parse(cls, s: str) -> "KVCachePolicy":
        """Parse an atomic cache policy string."""
        raw = str(s).strip()
        if not raw:
            raise ValueError("Empty kv-policy string")

        def norm_key(k: str) -> str:
            return str(k).strip().lower().replace("_", "")

        def parse_kind_qblock(val: str) -> tuple[KVCacheKind, int]:
            v = str(val).strip().lower()
            if "@" in v:
                kind_s, qb_s = v.split("@", 1)
                kind = str(kind_s).strip()
                qb = int(str(qb_s).strip())
            else:
                kind = str(v).strip()
                qb = 32

            if kind not in ("fp16", "fp32", "q8_0", "q4_0", "nf4"):
                raise ValueError(f"Unknown KV cache kind: {kind}")
            if qb <= 0:
                raise ValueError(f"qblock must be > 0 (got {qb})")
            return kind, int(qb)

        items: dict[str, str] = {}
        for part in [p.strip() for p in raw.split(",") if p.strip()]:
            if "=" not in part:
                raise ValueError(f"Invalid kv-policy field (expected key=value): {part!r}")
            k, v = part.split("=", 1)
            items[norm_key(k)] = str(v).strip()

        ksem_s = items.get("ksem") or items.get("ksemkind")
        kgeo_s = items.get("kgeo") or items.get("kgeokind")
        v_s = items.get("v") or items.get("vkind")
        resid_s = items.get("resid") or items.get("residual") or items.get("residuallen")

        missing: list[str] = []
        if ksem_s is None:
            missing.append("ksem")
        if kgeo_s is None:
            missing.append("kgeo")
        if v_s is None:
            missing.append("v")
        if resid_s is None:
            missing.append("resid")
        if missing:
            raise ValueError(f"Missing kv-policy fields: {', '.join(missing)}")

        ksem_kind, ksem_qb = parse_kind_qblock(str(ksem_s))
        kgeo_kind, kgeo_qb = parse_kind_qblock(str(kgeo_s))
        v_kind, v_qb = parse_kind_qblock(str(v_s))
        resid = int(str(resid_s).strip())
        if resid < 0:
            raise ValueError(f"resid must be >= 0 (got {resid})")

        return cls(
            k_sem_kind=ksem_kind,
            k_geo_kind=kgeo_kind,
            v_kind=v_kind,
            k_sem_qblock=int(ksem_qb),
            k_geo_qblock=int(kgeo_qb),
            v_qblock=int(v_qb),
            residual_len=int(resid),
        )


