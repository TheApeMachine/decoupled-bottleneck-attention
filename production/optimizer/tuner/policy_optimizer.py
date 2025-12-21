"""KV-cache policy self-optimizer.

This file stays intentionally small: it orchestrates policy selection and
delegates persistence, keying, neighborhood generation, memory estimation, and
benchmarking to helper modules in `production.optimizer.tuner.*`.
"""

from __future__ import annotations

from production.optimizer.tuner.buckets import pow2_bucket
from production.optimizer.tuner.cache_estimates import as_mb
from production.optimizer.tuner.cache_policy import KVCachePolicy
from production.optimizer.tuner.config import KVSelfOptConfig
from production.optimizer.tuner.policy_bench import bench_policy_ms
from production.optimizer.tuner.policy_keys import attn_mode_value, policy_key
from production.optimizer.tuner.policy_memory import budget_bytes, policy_mem_bytes
from production.optimizer.tuner.policy_neighbors import neighbors
from production.optimizer.tuner.policy_store import PolicyStore


class KVCachePolicySelfOptimizer:
    """Pick a cache policy that fits a strict memory budget and improves decode throughput."""

    def __init__(
        self,
        cfg: KVSelfOptConfig,
        *,
        device,
        attn: object,
        model_cfg: object,
        batch_size: int,
        max_seq_len: int,
        base_policy: KVCachePolicy,
        base_decode_block: int,
        base_fused: str,
    ) -> None:
        self.cfg: KVSelfOptConfig = cfg
        self.device = device
        self.attn: object = attn
        self.model_cfg: object = model_cfg
        self.batch_size: int = int(batch_size)
        self.max_seq_len: int = int(max_seq_len)
        self.base_policy: KVCachePolicy = base_policy
        self.base_decode_block: int = int(base_decode_block)
        self.base_fused: str = str(base_fused)

        self._store = PolicyStore(cfg.cache_path)

    def update_cached_policy(self, policy: KVCachePolicy) -> None:
        """Overwrite the persisted cache policy for this hardware/model key."""
        try:
            key = self._key()
            self._store.set(key, policy)
            self._store.save()
        except (OSError, ValueError, TypeError, KeyError, AttributeError):
            pass

    def _key(self) -> str:
        return policy_key(
            device=self.device,
            model_cfg=self.model_cfg,
            batch_size=self.batch_size,
            max_seq_len=self.max_seq_len,
        )

    def _budget_bytes(self) -> int:
        return budget_bytes(
            self.cfg,
            model_cfg=self.model_cfg,
            batch_size=self.batch_size,
            max_seq_len=self.max_seq_len,
            base_policy=self.base_policy,
        )

    def choose_policy(self, *, prompt_len: int) -> KVCachePolicy:
        if self.cfg.mode == "none":
            return self.base_policy
        if attn_mode_value(self.model_cfg) != "decoupled":
            return self.base_policy

        key = self._key()
        budget = self._budget_bytes()

        def mem_bytes(pol: KVCachePolicy) -> int:
            return policy_mem_bytes(
                model_cfg=self.model_cfg,
                batch_size=self.batch_size,
                max_seq_len=self.max_seq_len,
                policy=pol,
            )

        def ok_mem(pol: KVCachePolicy) -> bool:
            return mem_bytes(pol) <= budget

        cached = self._store.get(key)
        if cached is not None:
            try:
                pb = mem_bytes(cached)
                self._print_policy_summary(
                    policy=cached,
                    prefix_len=min(self.max_seq_len - 1, max(1, int(prompt_len))),
                    best_ms=float("nan"),
                    budget_bytes=budget,
                    policy_bytes=pb,
                    note="cached",
                )
            except (ValueError, TypeError, AttributeError):
                pass
            return cached

        if self.cfg.policy_prefix_len is not None:
            prefix_len = int(self.cfg.policy_prefix_len)
        else:
            prefix_len = int(min(self.max_seq_len - 1, max(1024, pow2_bucket(int(prompt_len)))))
        prefix_len = max(1, min(prefix_len, self.max_seq_len - 1))

        cur = self.base_policy
        if not ok_mem(cur):
            cur = KVCachePolicy(
                k_sem_kind=cur.k_sem_kind,
                k_geo_kind=cur.k_geo_kind,
                v_kind=cur.v_kind,
                k_sem_qblock=cur.k_sem_qblock,
                k_geo_qblock=cur.k_geo_qblock,
                v_qblock=cur.v_qblock,
                residual_len=0,
            )

        best = cur
        try:
            best_ms = bench_policy_ms(
                self.cfg,
                device=self.device,
                attn=self.attn,
                model_cfg=self.model_cfg,
                batch_size=self.batch_size,
                base_decode_block=self.base_decode_block,
                base_fused=self.base_fused,
                policy=cur,
                prefix_len=prefix_len,
            )
        except (RuntimeError, ValueError, TypeError, AttributeError):
            best_ms = float("inf")

        improved = True
        while improved:
            improved = False
            candidates = [p for p in neighbors(self.cfg, best) if ok_mem(p)]
            if not candidates:
                break

            scored: list[tuple[float, int, KVCachePolicy]] = []
            for p in candidates:
                try:
                    ms = bench_policy_ms(
                        self.cfg,
                        device=self.device,
                        attn=self.attn,
                        model_cfg=self.model_cfg,
                        batch_size=self.batch_size,
                        base_decode_block=self.base_decode_block,
                        base_fused=self.base_fused,
                        policy=p,
                        prefix_len=prefix_len,
                    )
                except (RuntimeError, ValueError, TypeError, AttributeError):
                    ms = float("inf")
                scored.append((ms, mem_bytes(p), p))

            scored.sort(key=lambda x: (x[0], x[1]))
            for ms, mb, p in scored:
                if ms < best_ms * (1.0 - float(self.cfg.policy_hysteresis)):
                    if self.cfg.verbose:
                        print(
                            f"[selfopt] cache-policy step: {best.short()} -> {p.short()} "
                            f"({best_ms:.3f}ms -> {ms:.3f}ms, mem={as_mb(mb):.1f}MB)"
                        )
                    best = p
                    best_ms = ms
                    improved = True
                    break

        if self.cfg.prefer_lower_mem_within > 0 and best_ms < float("inf"):
            cur_ms = best_ms
            cur_mem = mem_bytes(best)
            for p in [p for p in neighbors(self.cfg, best) if ok_mem(p)]:
                try:
                    ms = bench_policy_ms(
                        self.cfg,
                        device=self.device,
                        attn=self.attn,
                        model_cfg=self.model_cfg,
                        batch_size=self.batch_size,
                        base_decode_block=self.base_decode_block,
                        base_fused=self.base_fused,
                        policy=p,
                        prefix_len=prefix_len,
                    )
                except (RuntimeError, ValueError, TypeError, AttributeError):
                    continue
                if ms <= cur_ms * (1.0 + float(self.cfg.prefer_lower_mem_within)):
                    m = mem_bytes(p)
                    if m < cur_mem:
                        if self.cfg.verbose:
                            print(
                                f"[selfopt] cache-policy tie-break: {best.short()} -> {p.short()} "
                                f"(ms={ms:.3f} within {self.cfg.prefer_lower_mem_within*100:.1f}%, "
                                f"mem {as_mb(cur_mem):.1f}MB -> {as_mb(m):.1f}MB)"
                            )
                        best = p
                        cur_mem = m

        self._store.set(key, best)
        self._store.save()

        self._print_policy_summary(
            policy=best,
            prefix_len=prefix_len,
            best_ms=best_ms,
            budget_bytes=budget,
            policy_bytes=mem_bytes(best),
            note="chosen",
        )
        return best

    def shortlist_policies(self, *, prompt_len: int, max_candidates: int = 8) -> list[KVCachePolicy]:
        k = max(1, int(max_candidates))

        if self.cfg.mode == "none":
            return [self.base_policy]
        if attn_mode_value(self.model_cfg) != "decoupled":
            return [self.base_policy]

        key = self._key()
        budget = self._budget_bytes()

        def mem_bytes(pol: KVCachePolicy) -> int:
            return policy_mem_bytes(
                model_cfg=self.model_cfg,
                batch_size=self.batch_size,
                max_seq_len=self.max_seq_len,
                policy=pol,
            )

        def ok_mem(pol: KVCachePolicy) -> bool:
            try:
                return mem_bytes(pol) <= budget
            except (ValueError, TypeError, AttributeError):
                return False

        if self.cfg.policy_prefix_len is not None:
            prefix_len = int(self.cfg.policy_prefix_len)
        else:
            prefix_len = int(min(self.max_seq_len - 1, max(1024, pow2_bucket(int(prompt_len)))))
        prefix_len = max(1, min(prefix_len, self.max_seq_len - 1))

        cands: list[KVCachePolicy] = []
        cached = self._store.get(key)
        if cached is not None:
            cands.append(cached)
        cands.append(self.base_policy)

        try:
            cands.append(self.choose_policy(prompt_len=int(prompt_len)))
        except (RuntimeError, ValueError, TypeError, AttributeError):
            pass

        try:
            resid_default = 128
            try:
                resid_default = int(max(int(x) for x in self.cfg.residuals))
            except (AttributeError, TypeError, ValueError):
                resid_default = 128
            cands.append(
                KVCachePolicy(
                    k_sem_kind="q4_0",
                    k_geo_kind="q8_0",
                    v_kind="q4_0",
                    k_sem_qblock=32,
                    k_geo_qblock=32,
                    v_qblock=32,
                    residual_len=int(resid_default),
                )
            )
        except (TypeError, ValueError):
            pass

        seeds = list(cands)
        for p in seeds:
            try:
                cands.extend(neighbors(self.cfg, p))
            except (RuntimeError, ValueError, TypeError, AttributeError):
                pass

        uniq: list[KVCachePolicy] = []
        seen: set[str] = set()
        for p in cands:
            keyp = p.short()
            if keyp in seen:
                continue
            seen.add(keyp)
            if ok_mem(p):
                uniq.append(p)

        if not uniq:
            return [self.base_policy]

        scored: list[tuple[float, int, KVCachePolicy]] = []
        for p in uniq:
            try:
                ms = float(
                    bench_policy_ms(
                        self.cfg,
                        device=self.device,
                        attn=self.attn,
                        model_cfg=self.model_cfg,
                        batch_size=self.batch_size,
                        base_decode_block=self.base_decode_block,
                        base_fused=self.base_fused,
                        policy=p,
                        prefix_len=prefix_len,
                    )
                )
            except (RuntimeError, ValueError, TypeError, AttributeError):
                ms = float("inf")
            try:
                mb = int(mem_bytes(p))
            except (ValueError, TypeError, AttributeError):
                mb = 1 << 62
            scored.append((ms, mb, p))

        scored.sort(key=lambda x: (x[0], x[1]))
        out: list[KVCachePolicy] = [p for _ms, _mb, p in scored[:k]]

        if all(p.short() != self.base_policy.short() for p in out):
            out.append(self.base_policy)
        return out

    def _print_policy_summary(
        self,
        *,
        policy: KVCachePolicy,
        prefix_len: int,
        best_ms: float,
        budget_bytes: int,
        policy_bytes: int,
        note: str,
    ) -> None:
        print(
            f"[selfopt] cache-policy {note}: {policy.short()} "
            f"(mem={as_mb(policy_bytes):.1f}MB <= {as_mb(budget_bytes):.1f}MB, "
            f"prefix_len={int(prefix_len)}, best_ms={float(best_ms):.3f})"
        )


