"""Persistence for decode-plan tuning results."""

from __future__ import annotations

from dataclasses import asdict

from production.selfopt_cache import get_cache_entry, set_cache_entry

from production.optimizer.tuner.decode_plan import KVDecodePlan


class DecodePlanStore:
    """Persist decode plans as a single JSON entry (per cache path)."""

    def __init__(self, cache_path: str | None, *, verbose: bool) -> None:
        self.cache_path: str | None = cache_path
        self.verbose: bool = bool(verbose)

    def load(self) -> dict[str, KVDecodePlan]:
        """Load all cached plans from disk (best-effort)."""
        if not self.cache_path:
            return {}
        try:
            raw = get_cache_entry(self.cache_path, section="decode_plans", key="__all__")
            if not isinstance(raw, dict):
                return {}
            out: dict[str, KVDecodePlan] = {}
            for k, v in raw.items():
                if isinstance(v, dict):
                    out[str(k)] = KVDecodePlan(**v)
            return out
        except (TypeError, OSError, ValueError):
            if self.verbose:
                print(f"[selfopt] Failed to load cache '{self.cache_path}'")
            return {}

    def save(self, plans: dict[str, KVDecodePlan]) -> None:
        """Save all plans to disk (best-effort)."""
        if not self.cache_path:
            return
        try:
            payload = {k: asdict(v) for k, v in plans.items()}
            set_cache_entry(str(self.cache_path), section="decode_plans", key="__all__", value=payload)
        except (OSError, ValueError, TypeError):
            if self.verbose:
                print(f"[selfopt] Failed to save cache '{self.cache_path}'")


