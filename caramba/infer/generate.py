"""Text generation loop with KV-cache support.

This module provides the core generation loop that takes a model with
attention layers and generates tokens autoregressively. It handles:
- KV-cache creation for all attention layers
- Prefill (process prompt) and decode (generate tokens) phases
- Temperature, top-k, and top-p sampling
- Optional diffusion head sampling
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from torch import Tensor, nn

from caramba.cache.decoupled import DecoupledLayerKVCache
from caramba.cache.layer import LayerKVCache
from caramba.config.kvcache import KVCacheKind, KVCacheTensorConfig
from caramba.config.layer import AttentionLayerConfig, AttentionMode
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
    cache_kind: KVCacheKind = KVCacheKind.FP16
    cache_qblock: int = 32
    cache_residual_len: int = 0
    use_diffusion: bool = False
    diffusion_guidance_scale: float | None = None


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
) -> list[LayerKVCache | DecoupledLayerKVCache]:
    """Create KV caches for all attention layers.

    Inspects the model to find attention layers, then creates the
    appropriate cache type for each: LayerKVCache for standard/GQA,
    DecoupledLayerKVCache for DBA.
    """
    configs = get_attention_configs(model)
    caches: list[LayerKVCache | DecoupledLayerKVCache] = []

    tensor_cfg = KVCacheTensorConfig(
        kind=cache_kind,
        qblock=cache_qblock,
        residual_len=cache_residual_len,
    )

    for cfg in configs:
        if cfg.mode == AttentionMode.DECOUPLED:
            sem_dim = cfg.sem_dim if cfg.sem_dim is not None else cfg.d_model
            geo_dim = cfg.geo_dim if cfg.geo_dim is not None else cfg.d_model
            v_dim = cfg.v_dim

            cache = DecoupledLayerKVCache(
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
        else:
            k_dim = cfg.kv_heads * cfg.head_dim
            v_dim = cfg.kv_heads * cfg.head_dim

            cache = LayerKVCache(
                batch_size=batch_size,
                max_seq_len=max_seq_len,
                k_dim=k_dim,
                v_dim=v_dim,
                k_cfg=tensor_cfg,
                v_cfg=tensor_cfg,
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
        cache_kind=config.cache_kind,
        cache_qblock=config.cache_qblock,
        cache_residual_len=config.cache_residual_len,
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

    def reset(self) -> None:
        """Clear caches and reset position."""
        self._caches = None
        self._ctx = None
        self._pos = 0

    def _ensure_caches(self, batch_size: int) -> None:
        """Allocate caches on first use."""
        if self._caches is not None:
            return

        self._caches = create_caches(
            self.model,
            batch_size=batch_size,
            max_seq_len=self.config.max_seq_len,
            device=self.device,
            cache_kind=self.config.cache_kind,
            cache_qblock=self.config.cache_qblock,
            cache_residual_len=self.config.cache_residual_len,
        )
        self._ctx = InferContext(caches=self._caches)
        self._pos = 0

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

        self._ctx.begin(pos_offset=0)

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

        self._ctx.begin(pos_offset=self._pos)

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
