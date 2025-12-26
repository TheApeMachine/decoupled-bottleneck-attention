"""Text generation loop with KV-cache support.

This module provides the core generation loop that takes a model with
attention layers and generates tokens autoregressively. It handles:
- KV-cache creation for all attention layers
- Prefill (process prompt) and decode (generate tokens) phases
- Temperature, top-k, and top-p sampling
- Optional diffusion head sampling
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
from torch import Tensor, nn

from caramba.cache.decoupled import DecoupledLayerKVCache
from caramba.cache.layer import LayerKVCache
from caramba.config.kvcache import KVCacheKind, KVCacheTensorConfig
from caramba.config.layer import AttentionLayerConfig, AttentionMode

from caramba.infer.cache_policy import (
    estimate_kvcache_bytes,
    needle_in_haystack_gate,
    short_context_fidelity_check,
)
from caramba.infer.cache_plan import (
    cache_plan_key,
    cache_plan_payload,
    load_cached_entry,
    load_cached_kind,
    save_cached_kind,
    should_probe_entry,
)
from caramba.infer.context import InferContext
from caramba.layer.attention import AttentionLayer


@dataclass
class GenerateConfig:
    """Configuration for text generation.

    Controls sampling strategy (temperature, top-k, top-p), sequence limits,
    and KV-cache settings. Optionally enables diffusion-based sampling if
    the model has a diffusion head.
    """

    max_new_tokens: int = 64
    temperature: float = 1.0
    top_k: int | None = None
    top_p: float | None = None
    eos_token_id: int | None = None
    max_seq_len: int = 2048
    cache_kind: KVCacheKind | str = KVCacheKind.FP16
    cache_qblock: int = 32
    cache_residual_len: int = 0
    cache_budget_mb: float | None = None
    cache_auto_benchmark: bool = False
    cache_auto_bench_steps: int = 8
    cache_auto_bench_prompt_len: int = 64
    cache_fp16_prefix_layers: int = 0
    cache_quality_max_delta_nll: float | None = None
    cache_quality_max_ppl_ratio: float | None = None
    cache_quality_max_mean_kl: float | None = None
    cache_quality_prompt_len: int = 64
    cache_quality_decode_steps: int = 4
    cache_plan_path: str | None = None
    cache_plan_probe: bool = False
    cache_plan_probe_interval_sec: int = 3600
    use_diffusion: bool = False
    diffusion_guidance_scale: float | None = None

    # Decode-plan heuristics (optional).
    decode_plan: str = "auto"  # auto|fixed|none
    decode_q_chunk: int | None = None
    decode_local_window: int | None = None
    decode_bucket_short: int = 512
    decode_bucket_mid: int = 2048
    decode_q_chunk_mid: int = 128
    decode_q_chunk_long: int = 64
    decode_local_window_long: int = 2048


def _resolve_cache_kind(
    model: nn.Module,
    *,
    batch_size: int,
    max_seq_len: int,
    config: GenerateConfig,
) -> KVCacheKind:
    ck = config.cache_kind
    if isinstance(ck, KVCacheKind):
        return ck
    s = str(ck).strip().lower()
    if s != "auto":
        try:
            return KVCacheKind(s)
        except Exception:
            return KVCacheKind.FP16

    # Candidate set (higher quality first).
    candidates = [KVCacheKind.FP16, KVCacheKind.Q8_0, KVCacheKind.NF4, KVCacheKind.Q4_0]

    try:
        bench_device = next(model.parameters()).device
    except Exception:
        bench_device = torch.device("cpu")

    # Budget filtering (if requested).
    if config.cache_budget_mb is not None:
        budget_bytes = float(config.cache_budget_mb) * 1024.0 * 1024.0
        filtered: list[KVCacheKind] = []
        for k in candidates:
            est = estimate_kvcache_bytes(
                model=model,
                batch_size=batch_size,
                max_seq_len=max_seq_len,
                kind=k,
                qblock=int(config.cache_qblock),
                residual_len=int(config.cache_residual_len),
            )
            if float(est) <= budget_bytes:
                filtered.append(k)
        candidates = filtered if filtered else [KVCacheKind.Q4_0]

    # Cache persistence: if configured, try to reuse a previous decision.
    plan_path = getattr(config, "cache_plan_path", None)
    key: str | None = None
    if isinstance(plan_path, str) and plan_path:
        payload = cache_plan_payload(
            model=model,
            batch_size=batch_size,
            max_seq_len=max_seq_len,
            qblock=int(config.cache_qblock),
            residual_len=int(config.cache_residual_len),
            budget_mb=config.cache_budget_mb,
            quality_max_delta_nll=config.cache_quality_max_delta_nll,
            quality_max_ppl_ratio=config.cache_quality_max_ppl_ratio,
            quality_max_mean_kl=config.cache_quality_max_mean_kl,
            quality_prompt_len=int(config.cache_quality_prompt_len),
            quality_decode_steps=int(config.cache_quality_decode_steps),
            auto_benchmark=bool(config.cache_auto_benchmark),
            auto_bench_steps=int(config.cache_auto_bench_steps),
            auto_bench_prompt_len=int(config.cache_auto_bench_prompt_len),
        )
        key = cache_plan_key(payload)
        cached = load_cached_kind(Path(plan_path), key=key)
        if cached is not None:
            if not bool(getattr(config, "cache_plan_probe", False)):
                return cached
            entry = load_cached_entry(Path(plan_path), key=key)
            if entry is not None and not should_probe_entry(
                entry, interval_sec=int(getattr(config, "cache_plan_probe_interval_sec", 3600))
            ):
                return cached

    # If benchmarking isn't requested (or no steps), pick the highest-quality candidate.
    # Optional short-context quality gate vs fp16 baseline.
    if (
        config.cache_quality_max_delta_nll is not None
        or config.cache_quality_max_ppl_ratio is not None
        or config.cache_quality_max_mean_kl is not None
    ):
        gate_candidates: list[KVCacheKind] = []
        # Random sequence for a cheap approximation (real gates can use dataset-driven tokens).
        seq_len = max(2, min(max_seq_len, int(config.cache_quality_prompt_len) + 16))
        token_ids = torch.randint(0, 1000, (batch_size, seq_len), device=bench_device, dtype=torch.long)
        for k in candidates:
            try:
                res = short_context_fidelity_check(
                    model=model,
                    token_ids=token_ids,
                    baseline_kind=KVCacheKind.FP16,
                    candidate_kind=k,
                    max_seq_len=max_seq_len,
                    qblock=int(config.cache_qblock),
                    residual_len=int(config.cache_residual_len),
                    prompt_len=int(config.cache_quality_prompt_len),
                )
            except Exception:
                continue
            if config.cache_quality_max_delta_nll is not None and res.delta_nll > float(config.cache_quality_max_delta_nll):
                continue
            if config.cache_quality_max_ppl_ratio is not None and res.ppl_ratio > float(config.cache_quality_max_ppl_ratio):
                continue
            if config.cache_quality_max_mean_kl is not None:
                try:
                    needle = needle_in_haystack_gate(
                        model=model,
                        token_ids=token_ids,
                        baseline_kind=KVCacheKind.FP16,
                        candidate_kind=k,
                        max_seq_len=max_seq_len,
                        qblock=int(config.cache_qblock),
                        residual_len=int(config.cache_residual_len),
                        prompt_len=int(config.cache_quality_prompt_len),
                        decode_steps=int(config.cache_quality_decode_steps),
                    )
                except Exception:
                    continue
                if needle.mean_kl > float(config.cache_quality_max_mean_kl):
                    continue
            gate_candidates.append(k)
        if gate_candidates:
            candidates = gate_candidates

    if not bool(config.cache_auto_benchmark) or int(config.cache_auto_bench_steps) <= 0:
        chosen = candidates[0]
        if isinstance(plan_path, str) and plan_path and key is not None:
            try:
                save_cached_kind(Path(plan_path), key=key, kind=chosen, source="default")  # type: ignore[arg-type]
            except Exception:
                pass
        return chosen

    # Best-effort micro-benchmark: prefill + a few decode steps.
    best_kind = candidates[0]
    best_tps = -1.0

    prompt_len = max(1, min(int(config.cache_auto_bench_prompt_len), max_seq_len - 1))
    bench_steps = max(1, int(config.cache_auto_bench_steps))

    # If the model doesn't accept ctx, caches won't be consumed; default to fp16.
    for kind in candidates:
        try:
            caches = create_caches(
                model,
                batch_size=batch_size,
                max_seq_len=max_seq_len,
                device=bench_device,
                cache_kind=kind,
                cache_qblock=int(config.cache_qblock),
                cache_residual_len=int(config.cache_residual_len),
            )
            ctx = InferContext(caches=caches, pos_offset=0)
        except Exception:
            continue

        # Random prompt using model's vocab size if available, otherwise fallback to 1000.
        vocab_size = getattr(model, "vocab_size", None)
        if vocab_size is None:
            config_attr = getattr(model, "config", None)
            if config_attr is not None:
                vocab_size = getattr(config_attr, "vocab_size", None)
        if vocab_size is None or vocab_size < 1:
            vocab_size = 1000  # Safe fallback
        input_ids = torch.randint(
            0, int(vocab_size), (batch_size, prompt_len), dtype=torch.long, device=bench_device
        )
        t0 = time.perf_counter()
        try:
            ctx.begin(pos_offset=0)
            logits = model(input_ids, ctx=ctx)  # type: ignore[call-arg]
            ctx.ensure_consumed()
            token = logits[:, -1, :].argmax(dim=-1)
            for i in range(bench_steps):
                ctx.begin(pos_offset=prompt_len + i)
                logits = model(token.view(batch_size, 1), ctx=ctx)  # type: ignore[call-arg]
                ctx.ensure_consumed()
                token = logits[:, -1, :].argmax(dim=-1)
        except TypeError:
            # Model doesn't support ctx; abort benchmarking.
            chosen = KVCacheKind.FP16
            if isinstance(plan_path, str) and plan_path and key is not None:
                try:
                    save_cached_kind(Path(plan_path), key=key, kind=chosen, source="no_ctx")  # type: ignore[arg-type]
                except Exception:
                    pass
            return chosen
        except Exception:
            continue
        t1 = time.perf_counter()
        dt = max(1e-9, float(t1 - t0))
        tps = float(bench_steps) / dt
        if tps > best_tps:
            best_tps = tps
            best_kind = kind

    if isinstance(plan_path, str) and plan_path and key is not None:
        try:
            save_cached_kind(
                Path(plan_path),
                key=key,
                kind=best_kind,
                tps=float(best_tps) if best_tps >= 0 else None,
                source="bench",
            )  # type: ignore[arg-type]
        except Exception:
            pass
    return best_kind


def count_attention_layers(model: nn.Module) -> int:
    """Count attention layers in a model."""
    count = 0
    for module in model.modules():
        if isinstance(module, AttentionLayer):
            count += 1
    return count


def has_diffusion_head(model: nn.Module) -> bool:
    """Check if a model has an enabled diffusion head."""
    return getattr(model, "diffusion_head", None) is not None


def get_attention_configs(model: nn.Module) -> list[AttentionLayerConfig]:
    """Extract attention layer configs from a model."""
    configs = []
    for module in model.modules():
        if isinstance(module, AttentionLayer):
            configs.append(module.config)
    return configs


def create_caches(
    model: nn.Module,
    *,
    batch_size: int,
    max_seq_len: int,
    device: torch.device,
    cache_kind: KVCacheKind = KVCacheKind.FP16,
    cache_qblock: int = 32,
    cache_residual_len: int = 0,
    fp16_prefix_layers: int = 0,
) -> list[LayerKVCache | DecoupledLayerKVCache]:
    """Create KV caches for all attention layers.

    Inspects the model to find attention layers, then creates the
    appropriate cache type for each: LayerKVCache for standard/GQA,
    DecoupledLayerKVCache for DBA.
    """
    configs = get_attention_configs(model)
    caches: list[LayerKVCache | DecoupledLayerKVCache] = []

    for i, cfg in enumerate(configs):
        kind_i = cache_kind
        if int(fp16_prefix_layers) > 0 and i < int(fp16_prefix_layers):
            kind_i = KVCacheKind.FP16

        if cfg.mode == AttentionMode.DECOUPLED:
            sem_dim = cfg.sem_dim if cfg.sem_dim is not None else cfg.d_model
            geo_dim = cfg.geo_dim if cfg.geo_dim is not None else cfg.d_model
            v_dim = cfg.v_dim

            tensor_cfg_i = KVCacheTensorConfig(
                kind=kind_i,
                qblock=cache_qblock,
                residual_len=cache_residual_len,
            )
            cache = DecoupledLayerKVCache(
                batch_size=batch_size,
                max_seq_len=max_seq_len,
                k_sem_dim=sem_dim,
                k_geo_dim=geo_dim,
                v_dim=v_dim,
                k_sem_cfg=tensor_cfg_i,
                k_geo_cfg=tensor_cfg_i,
                v_cfg=tensor_cfg_i,
                device=device,
            )
        else:
            k_dim = cfg.kv_heads * cfg.head_dim
            v_dim = cfg.kv_heads * cfg.head_dim

            tensor_cfg_i = KVCacheTensorConfig(
                kind=kind_i,
                qblock=cache_qblock,
                residual_len=cache_residual_len,
            )
            cache = LayerKVCache(
                batch_size=batch_size,
                max_seq_len=max_seq_len,
                k_dim=k_dim,
                v_dim=v_dim,
                k_cfg=tensor_cfg_i,
                v_cfg=tensor_cfg_i,
                device=device,
            )
        caches.append(cache)

    return caches


def sample_next_token(
    logits: Tensor,
    *,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
) -> Tensor:
    """Sample the next token from logits.

    Supports greedy (temp=0), temperature scaling, top-k filtering,
    and nucleus (top-p) sampling.
    """
    if temperature <= 0:
        return logits.argmax(dim=-1)

    logits = logits / temperature

    if top_k is not None and top_k > 0:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < v[:, [-1]]] = float("-inf")

    if top_p is not None and 0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
        sorted_indices_to_remove[:, 0] = False
        indices_to_remove = sorted_indices_to_remove.scatter(
            1, sorted_indices, sorted_indices_to_remove
        )
        logits[indices_to_remove] = float("-inf")

    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


@torch.inference_mode()
def generate(
    model: nn.Module,
    input_ids: Tensor,
    *,
    config: GenerateConfig | None = None,
    lm_head: nn.Module | None = None,
) -> Tensor:
    """Generate tokens autoregressively with KV-cache.

    Stateless API: creates fresh caches each call. For persistent caches
    across calls (e.g., multi-turn chat), use the Generator class.
    """
    if config is None:
        config = GenerateConfig()

    device = input_ids.device
    batch_size, seq_len = input_ids.shape

    caches = create_caches(
        model,
        batch_size=batch_size,
        max_seq_len=config.max_seq_len,
        device=device,
        cache_kind=_resolve_cache_kind(
            model,
            batch_size=batch_size,
            max_seq_len=int(config.max_seq_len),
            config=config,
        ),
        cache_qblock=config.cache_qblock,
        cache_residual_len=config.cache_residual_len,
        fp16_prefix_layers=int(config.cache_fp16_prefix_layers),
    )

    ctx = InferContext(caches=caches, pos_offset=0)

    max_gen_len = seq_len + config.max_new_tokens
    generated = torch.empty(
        (batch_size, max_gen_len),
        dtype=input_ids.dtype,
        device=device,
    )
    generated[:, :seq_len] = input_ids
    gen_len = seq_len

    for i in range(config.max_new_tokens):
        if i == 0:
            tokens = generated[:, :gen_len]
            pos_offset = 0
        else:
            tokens = generated[:, gen_len - 1 : gen_len]
            pos_offset = gen_len - 1

        ctx.begin(pos_offset=pos_offset)
        hidden = model(tokens, ctx=ctx)  # type: ignore[call-arg]
        ctx.ensure_consumed()

        if lm_head is not None:
            logits = lm_head(hidden[:, -1, :])
        else:
            logits = hidden[:, -1, :]

        next_token = sample_next_token(
            logits,
            temperature=config.temperature,
            top_k=config.top_k,
            top_p=config.top_p,
        )

        generated[:, gen_len] = next_token
        gen_len += 1

        if config.eos_token_id is not None:
            if (next_token == config.eos_token_id).all():
                break

    return generated[:, :gen_len]


class Generator:
    """Stateful generator with persistent KV-cache.

    Unlike the generate() function, this class keeps caches alive across
    multiple calls, enabling multi-turn conversations or streaming generation.
    Supports both standard sampling and diffusion-based sampling.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        config: GenerateConfig | None = None,
        lm_head: nn.Module | None = None,
        device: torch.device | None = None,
    ) -> None:
        """Set up the generator with model and config."""
        self.model = model
        self.config = config or GenerateConfig()
        self.lm_head = lm_head
        self.device = device or torch.device("cpu")

        self._caches: list[LayerKVCache | DecoupledLayerKVCache] | None = None
        self._ctx: InferContext | None = None
        self._pos: int = 0
        self._has_diffusion = has_diffusion_head(model)
        self._resolved_cache_kind: KVCacheKind | None = None

    def reset(self) -> None:
        """Clear caches and reset position."""
        self._caches = None
        self._ctx = None
        self._pos = 0

    def _ensure_caches(self, batch_size: int) -> None:
        """Allocate caches on first use."""
        if self._caches is not None:
            return

        kind = _resolve_cache_kind(
            self.model,
            batch_size=int(batch_size),
            max_seq_len=int(self.config.max_seq_len),
            config=self.config,
        )
        self._resolved_cache_kind = kind
        self._caches = create_caches(
            self.model,
            batch_size=batch_size,
            max_seq_len=self.config.max_seq_len,
            device=self.device,
            cache_kind=kind,
            cache_qblock=self.config.cache_qblock,
            cache_residual_len=self.config.cache_residual_len,
            fp16_prefix_layers=int(self.config.cache_fp16_prefix_layers),
        )
        self._ctx = InferContext(caches=self._caches)
        self._pos = 0

    def _plan_for_pos(self, pos_offset: int) -> tuple[int | None, int | None]:
        """Choose q_chunk/local_window overrides based on prefix length."""
        plan = str(getattr(self.config, "decode_plan", "auto")).lower()
        if plan in ("none", "off", "disabled"):
            return None, None
        if plan == "fixed":
            return self.config.decode_q_chunk, self.config.decode_local_window
        # auto: simple bucketing.
        p = int(pos_offset)
        if p < int(self.config.decode_bucket_short):
            return None, None
        if p < int(self.config.decode_bucket_mid):
            return int(self.config.decode_q_chunk_mid), None
        return int(self.config.decode_q_chunk_long), int(self.config.decode_local_window_long)

    def _forward_with_features(
        self,
        tokens: Tensor,
        use_diffusion: bool,
    ) -> tuple[Tensor, Tensor | None]:
        """Run forward, optionally extracting features for diffusion."""
        assert self._ctx is not None

        if use_diffusion and hasattr(self.model, "forward"):
            try:
                result = self.model(tokens, ctx=self._ctx, return_features=True)  # type: ignore[call-arg]
                if isinstance(result, tuple) and len(result) == 2:
                    features = result[0]
                    hidden = result[0]
                    return hidden, features
                else:
                    hidden = result if isinstance(result, Tensor) else result[0]  # type: ignore[index]
                    return hidden, None
            except TypeError:
                result2 = self.model(tokens, ctx=self._ctx)  # type: ignore[call-arg]
                hidden = result2 if isinstance(result2, Tensor) else result2[0]  # type: ignore[index]
                return hidden, None
        else:
            result3 = self.model(tokens, ctx=self._ctx)  # type: ignore[call-arg]
            hidden = result3 if isinstance(result3, Tensor) else result3[0]  # type: ignore[index]
            return hidden, None

    @torch.inference_mode()
    def prefill(self, input_ids: Tensor) -> Tensor:
        """Process the prompt and return logits for the last token.

        This is the first phase of generation: run the full prompt through
        the model, populating the KV-cache.
        """
        batch_size = input_ids.size(0)
        self._ensure_caches(batch_size)
        assert self._ctx is not None

        q_chunk, local_window = self._plan_for_pos(0)
        self._ctx.begin(pos_offset=0, q_chunk=q_chunk, local_window=local_window)

        use_diffusion = self.config.use_diffusion and self._has_diffusion
        hidden, self._last_features = self._forward_with_features(
            input_ids, use_diffusion
        )

        self._ctx.ensure_consumed()
        self._pos = input_ids.size(1)

        if use_diffusion and self._last_features is not None:
            features_last = self._last_features.narrow(
                1, self._last_features.size(1) - 1, 1
            )
            return self._sample_with_diffusion(features_last)

        if self.lm_head is not None:
            hidden_last = hidden.narrow(1, hidden.size(1) - 1, 1).squeeze(1)
            return self.lm_head(hidden_last)
        hidden_last = hidden.narrow(1, hidden.size(1) - 1, 1).squeeze(1)
        return hidden_last

    def _sample_with_diffusion(self, features_last: Tensor) -> Tensor:
        """Sample using the diffusion head."""
        if not hasattr(self.model, "sample_with_diffusion"):
            raise RuntimeError("Model does not support sample_with_diffusion method")
        return self.model.sample_with_diffusion(  # type: ignore[attr-defined]
            features_last,
            temperature=self.config.temperature,
            guidance_scale=self.config.diffusion_guidance_scale,
        )

    @torch.inference_mode()
    def decode_step(self, token_ids: Tensor) -> Tensor:
        """Decode one step: given the last token, return logits for next.

        This is the iterative phase of generation: process one token at a
        time, reading from and appending to the KV-cache.
        """
        assert self._ctx is not None

        if token_ids.dim() == 1:
            token_ids = token_ids.unsqueeze(-1)

        q_chunk, local_window = self._plan_for_pos(self._pos)
        self._ctx.begin(pos_offset=self._pos, q_chunk=q_chunk, local_window=local_window)

        use_diffusion = self.config.use_diffusion and self._has_diffusion
        hidden, self._last_features = self._forward_with_features(
            token_ids, use_diffusion
        )

        self._ctx.ensure_consumed()
        self._pos += token_ids.size(1)

        if use_diffusion and self._last_features is not None:
            features_last = self._last_features.narrow(
                1, self._last_features.size(1) - 1, 1
            )
            return self._sample_with_diffusion(features_last)

        if self.lm_head is not None:
            hidden_last = hidden.narrow(1, hidden.size(1) - 1, 1).squeeze(1)
            return self.lm_head(hidden_last)
        hidden_last = hidden.narrow(1, hidden.size(1) - 1, 1).squeeze(1)
        return hidden_last

    @torch.inference_mode()
    def generate(self, input_ids: Tensor) -> Tensor:
        """Full generation loop: prefill then decode until done."""
        self.reset()
        batch_size, seq_len = input_ids.shape
        self._ensure_caches(batch_size)

        max_gen_len = seq_len + self.config.max_new_tokens
        generated = torch.empty(
            (batch_size, max_gen_len),
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        generated[:, :seq_len] = input_ids
        gen_len = seq_len

        logits = self.prefill(input_ids)

        for _ in range(self.config.max_new_tokens):
            next_token = sample_next_token(
                logits,
                temperature=self.config.temperature,
                top_k=self.config.top_k,
                top_p=self.config.top_p,
            )

            generated[:, gen_len] = next_token
            gen_len += 1

            if self.config.eos_token_id is not None:
                if (next_token == self.config.eos_token_id).all():
                    break

            logits = self.decode_step(next_token)

        return generated[:, :gen_len]

    def rollback(self, n_tokens: int) -> None:
        """Rollback the cache by n tokens (for speculative decoding)."""
        if self._caches is None:
            return

        new_pos = max(0, self._pos - n_tokens)
        for cache in self._caches:
            cache.truncate(new_pos)
        self._pos = new_pos
