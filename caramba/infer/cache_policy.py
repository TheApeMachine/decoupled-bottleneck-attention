"""KV-cache policy selection for inference.

Why this exists:
- Quantized KV caches are a major lever for long-context inference.
- The best choice depends on (batch, seq_len), model dimensions, and any
  memory budget constraints.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from caramba.cache import make_quantspec
from caramba.cache.decoupled import DecoupledLayerKVCache
from caramba.cache.layer import LayerKVCache
from caramba.config.kvcache import KVCacheKind
from caramba.config.kvcache import KVCacheTensorConfig
from caramba.config.layer import AttentionLayerConfig, AttentionMode
from caramba.infer.context import InferContext
from caramba.layer.attention import AttentionLayer


@dataclass(frozen=True)
class CachePolicyChoice:
    """Chosen cache kind and its estimated memory footprint."""

    kind: KVCacheKind
    estimated_bytes: int


@dataclass(frozen=True)
class FidelityResult:
    """Short-context fidelity metrics for a cache kind vs a baseline."""

    baseline_nll: float
    candidate_nll: float
    delta_nll: float
    ppl_ratio: float


@dataclass(frozen=True)
class NeedleGateResult:
    """Proxy for long-range retrieval fidelity via baseline-vs-candidate logit similarity."""

    mean_kl: float
    max_kl: float


def _bytes_per_cache_tensor(
    *,
    kind: KVCacheKind,
    batch_size: int,
    max_seq_len: int,
    dim: int,
    qblock: int,
    residual_len: int,
) -> int:
    bs = int(batch_size)
    T = int(max_seq_len)
    d = int(dim)
    if kind in (KVCacheKind.FP16, KVCacheKind.FP32):
        bpe = 2 if kind == KVCacheKind.FP16 else 4
        return int(bs * T * d * bpe)

    spec = make_quantspec(kind.value, d, int(qblock))
    residual_eff = min(int(max(0, residual_len)), T)
    # Quant buffers.
    if kind == KVCacheKind.Q8_0:
        q_bytes = int(bs * T * spec.pad_dim * 1)
    else:
        q_bytes = int(bs * T * (spec.pad_dim // 2) * 1)
    s_bytes = int(bs * T * spec.n_blocks * 2)  # fp16 scales
    r_bytes = int(bs * residual_eff * d * 2)  # fp16 residual tail
    return int(q_bytes + s_bytes + r_bytes)


def estimate_kvcache_bytes(
    *,
    model: object,
    batch_size: int,
    max_seq_len: int,
    kind: KVCacheKind,
    qblock: int,
    residual_len: int,
) -> int:
    """Estimate total bytes for all attention-layer KV caches."""

    total = 0
    configs: list[AttentionLayerConfig] = []
    modules_fn = getattr(model, "modules", None)
    if modules_fn is None or not callable(modules_fn):
        return 0
    for m in modules_fn():  # type: ignore[union-attr]
        if isinstance(m, AttentionLayer):
            configs.append(m.config)

    for cfg in configs:
        if cfg.mode == AttentionMode.DECOUPLED:
            sem_dim = int(cfg.sem_dim if cfg.sem_dim is not None else cfg.d_model)
            geo_dim = int(cfg.geo_dim if cfg.geo_dim is not None else cfg.d_model)
            v_dim = int(cfg.v_dim)
            total += _bytes_per_cache_tensor(
                kind=kind,
                batch_size=batch_size,
                max_seq_len=max_seq_len,
                dim=sem_dim,
                qblock=qblock,
                residual_len=residual_len,
            )
            total += _bytes_per_cache_tensor(
                kind=kind,
                batch_size=batch_size,
                max_seq_len=max_seq_len,
                dim=geo_dim,
                qblock=qblock,
                residual_len=residual_len,
            )
            total += _bytes_per_cache_tensor(
                kind=kind,
                batch_size=batch_size,
                max_seq_len=max_seq_len,
                dim=v_dim,
                qblock=qblock,
                residual_len=residual_len,
            )
        else:
            kv_dim = int(cfg.kv_heads * cfg.head_dim)
            total += _bytes_per_cache_tensor(
                kind=kind,
                batch_size=batch_size,
                max_seq_len=max_seq_len,
                dim=kv_dim,
                qblock=qblock,
                residual_len=residual_len,
            )
            total += _bytes_per_cache_tensor(
                kind=kind,
                batch_size=batch_size,
                max_seq_len=max_seq_len,
                dim=kv_dim,
                qblock=qblock,
                residual_len=residual_len,
            )
    return int(total)


def _create_caches_for_kind(
    model: object,
    *,
    batch_size: int,
    max_seq_len: int,
    kind: KVCacheKind,
    qblock: int,
    residual_len: int,
    device: torch.device,
) -> list[LayerKVCache | DecoupledLayerKVCache]:
    tensor_cfg = KVCacheTensorConfig(kind=kind, qblock=int(qblock), residual_len=int(residual_len))
    caches: list[LayerKVCache | DecoupledLayerKVCache] = []
    modules_fn = getattr(model, "modules", None)
    if modules_fn is None or not callable(modules_fn):
        return caches
    for m in modules_fn():  # type: ignore[union-attr]
        if not isinstance(m, AttentionLayer):
            continue
        cfg = m.config
        if cfg.mode == AttentionMode.DECOUPLED:
            sem_dim = int(cfg.sem_dim if cfg.sem_dim is not None else cfg.d_model)
            geo_dim = int(cfg.geo_dim if cfg.geo_dim is not None else cfg.d_model)
            v_dim = int(cfg.v_dim)
            caches.append(
                DecoupledLayerKVCache(
                    batch_size=batch_size,
                    max_seq_len=max_seq_len,
                    k_sem_dim=sem_dim,
                    k_geo_dim=geo_dim,
                    v_dim=v_dim,
                    k_sem_cfg=tensor_cfg,
                    k_geo_cfg=tensor_cfg,
                    v_cfg=tensor_cfg,
                    device=device,
                )
            )
        else:
            kv_dim = int(cfg.kv_heads * cfg.head_dim)
            caches.append(
                LayerKVCache(
                    batch_size=batch_size,
                    max_seq_len=max_seq_len,
                    k_dim=kv_dim,
                    v_dim=kv_dim,
                    k_cfg=tensor_cfg,
                    v_cfg=tensor_cfg,
                    device=device,
                )
            )
    return caches


@torch.inference_mode()
def short_context_fidelity_check(
    *,
    model: torch.nn.Module,
    token_ids: torch.Tensor,
    baseline_kind: KVCacheKind,
    candidate_kind: KVCacheKind,
    max_seq_len: int,
    qblock: int,
    residual_len: int,
    prompt_len: int,
) -> FidelityResult:
    """Compute delta-NLL / PPL ratio for a candidate cache kind vs baseline."""

    if token_ids.dim() != 2:
        raise ValueError("token_ids must be (B, T)")
    B, T = int(token_ids.size(0)), int(token_ids.size(1))
    prompt_len = max(1, min(int(prompt_len), T - 1))

    device = token_ids.device

    def nll_for(kind: KVCacheKind) -> float:
        caches = _create_caches_for_kind(
            model,
            batch_size=B,
            max_seq_len=int(max_seq_len),
            kind=kind,
            qblock=int(qblock),
            residual_len=int(residual_len),
            device=device,
        )
        ctx = InferContext(caches=caches, pos_offset=0)
        # Prefill prompt.
        ctx.begin(pos_offset=0)
        logits = model(token_ids[:, :prompt_len], ctx=ctx)  # type: ignore[call-arg]
        ctx.ensure_consumed()
        nll = 0.0
        n = 0
        # Predict next token after prompt.
        target = token_ids[:, prompt_len]
        nll += float(F.cross_entropy(logits[:, -1, :], target))
        n += 1
        # Decode remaining tokens step-by-step.
        prev = target
        for i in range(prompt_len, T - 1):
            ctx.begin(pos_offset=i)
            logits = model(prev.view(B, 1), ctx=ctx)  # type: ignore[call-arg]
            ctx.ensure_consumed()
            target = token_ids[:, i + 1]
            nll += float(F.cross_entropy(logits[:, -1, :], target))
            n += 1
            prev = target
        return nll / float(max(1, n))

    base = nll_for(baseline_kind)
    cand = nll_for(candidate_kind)
    delta = float(cand - base)
    ppl_ratio = float(torch.exp(torch.tensor(delta)).item())
    return FidelityResult(
        baseline_nll=float(base),
        candidate_nll=float(cand),
        delta_nll=float(delta),
        ppl_ratio=float(ppl_ratio),
    )


@torch.inference_mode()
def long_context_fidelity_check(
    *,
    model: torch.nn.Module,
    token_ids: torch.Tensor,
    baseline_kind: KVCacheKind,
    candidate_kind: KVCacheKind,
    max_seq_len: int,
    qblock: int,
    residual_len: int,
    prompt_len: int = 4096,
) -> FidelityResult:
    """Long-context fidelity check (same metrics, but intended for 4K+ contexts)."""

    if int(token_ids.size(1)) < int(prompt_len) + 1:
        raise ValueError("token_ids length must be > prompt_len for long-context check")
    return short_context_fidelity_check(
        model=model,
        token_ids=token_ids,
        baseline_kind=baseline_kind,
        candidate_kind=candidate_kind,
        max_seq_len=max_seq_len,
        qblock=qblock,
        residual_len=residual_len,
        prompt_len=int(prompt_len),
    )


@torch.inference_mode()
def needle_in_haystack_gate(
    *,
    model: torch.nn.Module,
    token_ids: torch.Tensor,
    baseline_kind: KVCacheKind,
    candidate_kind: KVCacheKind,
    max_seq_len: int,
    qblock: int,
    residual_len: int,
    prompt_len: int,
    decode_steps: int = 8,
) -> NeedleGateResult:
    """Compare baseline vs candidate logits after a long prefill + decode.

    This is a pragmatic "long-range" gate: if quantization harms long-context
    retrieval, the candidate's decode logits tend to drift from the fp16
    baseline, especially after long prefills.
    """

    if token_ids.dim() != 2:
        raise ValueError("token_ids must be (B, T)")
    B, T = int(token_ids.size(0)), int(token_ids.size(1))
    prompt_len = max(1, min(int(prompt_len), T))
    decode_steps = max(1, int(decode_steps))
    device = token_ids.device

    def make_ctx(kind: KVCacheKind) -> InferContext:
        caches = _create_caches_for_kind(
            model,
            batch_size=B,
            max_seq_len=int(max_seq_len),
            kind=kind,
            qblock=int(qblock),
            residual_len=int(residual_len),
            device=device,
        )
        return InferContext(caches=caches, pos_offset=0)

    ctx_base = make_ctx(baseline_kind)
    ctx_cand = make_ctx(candidate_kind)

    # Prefill both contexts.
    ctx_base.begin(pos_offset=0)
    logits_b = model(token_ids[:, :prompt_len], ctx=ctx_base)  # type: ignore[call-arg]
    ctx_base.ensure_consumed()
    ctx_cand.begin(pos_offset=0)
    logits_c = model(token_ids[:, :prompt_len], ctx=ctx_cand)  # type: ignore[call-arg]
    ctx_cand.ensure_consumed()

    # Start decode from the baseline argmax to keep both in the same state trajectory.
    token = logits_b[:, -1, :].argmax(dim=-1)
    kls: list[float] = []
    for i in range(decode_steps):
        pos = prompt_len + i
        ctx_base.begin(pos_offset=pos)
        lb = model(token.view(B, 1), ctx=ctx_base)  # type: ignore[call-arg]
        ctx_base.ensure_consumed()
        ctx_cand.begin(pos_offset=pos)
        lc = model(token.view(B, 1), ctx=ctx_cand)  # type: ignore[call-arg]
        ctx_cand.ensure_consumed()

        # KL(base || cand) over vocab.
        logp_b = torch.log_softmax(lb[:, -1, :].float(), dim=-1)
        p_c = torch.softmax(lc[:, -1, :].float(), dim=-1)
        kl = F.kl_div(logp_b, p_c, reduction="batchmean")
        kls.append(float(kl))
        token = lb[:, -1, :].argmax(dim=-1)

    mean_kl = float(sum(kls) / float(len(kls))) if kls else 0.0
    max_kl = float(max(kls)) if kls else 0.0
    return NeedleGateResult(mean_kl=mean_kl, max_kl=max_kl)


def choose_cache_kind(
    *,
    model: object,
    batch_size: int,
    max_seq_len: int,
    qblock: int,
    residual_len: int,
    budget_mb: float | None,
) -> CachePolicyChoice:
    """Choose a KV cache kind, optionally constrained by a memory budget."""

    # Prefer higher-quality formats first.
    candidates = [KVCacheKind.FP16, KVCacheKind.Q8_0, KVCacheKind.NF4, KVCacheKind.Q4_0]

    if budget_mb is None:
        kind = KVCacheKind.FP16
        return CachePolicyChoice(
            kind=kind,
            estimated_bytes=estimate_kvcache_bytes(
                model=model,
                batch_size=batch_size,
                max_seq_len=max_seq_len,
                kind=kind,
                qblock=qblock,
                residual_len=residual_len,
            ),
        )

    budget_bytes = float(budget_mb) * 1024.0 * 1024.0

    # Find the highest-quality candidate that fits (candidates are ordered high to low quality).
    for k in candidates:
        est = estimate_kvcache_bytes(
            model=model,
            batch_size=batch_size,
            max_seq_len=max_seq_len,
            kind=k,
            qblock=qblock,
            residual_len=residual_len,
        )
        if float(est) <= budget_bytes:
            return CachePolicyChoice(kind=k, estimated_bytes=int(est))

    # Nothing fits; fall back to the most compressed kind.
    fallback_kind = KVCacheKind.Q4_0
    est = estimate_kvcache_bytes(
        model=model,
        batch_size=batch_size,
        max_seq_len=max_seq_len,
        kind=fallback_kind,
        qblock=qblock,
        residual_len=residual_len,
    )
    return CachePolicyChoice(kind=fallback_kind, estimated_bytes=int(est))

