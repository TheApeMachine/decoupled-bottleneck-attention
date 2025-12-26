"""
generate provides the text generation loop with KV-cache support.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from torch import Tensor, nn

from caramba.cache.layer import LayerKVCache
from caramba.cache.decoupled import DecoupledLayerKVCache
from caramba.config.kvcache import KVCacheTensorConfig, KVCacheKind
from caramba.config.layer import AttentionLayerConfig, AttentionMode
from caramba.infer.context import InferContext
from caramba.layer.attention import AttentionLayer


@dataclass
class GenerateConfig:
    """Configuration for text generation."""
    max_new_tokens: int = 64
    temperature: float = 1.0
    top_k: int | None = None
    top_p: float | None = None
    eos_token_id: int | None = None
    # Cache configuration
    max_seq_len: int = 2048
    cache_kind: KVCacheKind = KVCacheKind.FP16
    cache_qblock: int = 32
    cache_residual_len: int = 0
    # Diffusion head options
    use_diffusion: bool = False  # Use diffusion head for sampling if available
    diffusion_guidance_scale: float | None = None  # Override CFG scale


def count_attention_layers(model: nn.Module) -> int:
    """Count the number of attention layers in a model."""
    count = 0
    for module in model.modules():
        if isinstance(module, AttentionLayer):
            count += 1
    return count


def has_diffusion_head(model: nn.Module) -> bool:
    """Check if a model has an enabled diffusion head.

    Args:
        model: The model to check

    Returns:
        True if the model has a diffusion head attribute that is not None
    """
    return (
        hasattr(model, "diffusion_head")
        and getattr(model, "diffusion_head", None) is not None
    )


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
    """Create KV caches for all attention layers in a model.

    Args:
        model: The transformer model
        batch_size: Batch size for cache allocation
        max_seq_len: Maximum sequence length to cache
        device: Device to allocate caches on
        cache_kind: Quantization kind for cache tensors
        cache_qblock: Quantization block size
        cache_residual_len: Number of recent tokens to keep in fp16

    Returns:
        List of caches in layer order
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
            # Decoupled cache with separate k_sem, k_geo, v
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
            # Standard/GQA cache
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

    Args:
        logits: Logits tensor (B, vocab_size)
        temperature: Sampling temperature (1.0 = no change)
        top_k: If set, only sample from top k tokens
        top_p: If set, nucleus sampling with this probability mass

    Returns:
        Sampled token ids (B,)
    """
    if temperature <= 0:
        # Greedy decoding
        return logits.argmax(dim=-1)

    logits = logits / temperature

    if top_k is not None and top_k > 0:
        # Top-k filtering
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < v[:, [-1]]] = float("-inf")

    if top_p is not None and 0 < top_p < 1.0:
        # Nucleus (top-p) filtering
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

    Args:
        model: The transformer model (expects forward(x, ctx=...))
        input_ids: Initial token ids (B, T)
        config: Generation configuration
        lm_head: Optional language model head for logits (if not part of model)

    Returns:
        Generated token ids (B, T + max_new_tokens)
    """
    if config is None:
        config = GenerateConfig()

    device = input_ids.device
    batch_size, seq_len = input_ids.shape

    # Create caches for all attention layers
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

    # Pre-allocate buffer for generated tokens to avoid O(n^2) torch.cat
    max_gen_len = seq_len + config.max_new_tokens
    generated = torch.empty(
        (batch_size, max_gen_len),
        dtype=input_ids.dtype,
        device=device,
    )
    generated[:, :seq_len] = input_ids
    gen_len = seq_len

    # Prefill phase: process all input tokens at once
    # The model needs an embedding layer to convert token ids to embeddings
    # For now, assume model handles this internally or input is already embedded

    for i in range(config.max_new_tokens):
        # Get the tokens to process
        if i == 0:
            # Prefill: process all input tokens
            tokens = generated[:, :gen_len]
            pos_offset = 0
        else:
            # Decode: process only the last token
            tokens = generated[:, gen_len - 1 : gen_len]
            pos_offset = gen_len - 1

        # Reset context for this forward pass
        ctx.begin(pos_offset=pos_offset)

        # Forward pass
        hidden = model(tokens, ctx=ctx)  # type: ignore[call-arg]

        # Ensure all caches were used
        ctx.ensure_consumed()

        # Get logits for the last token
        if lm_head is not None:
            logits = lm_head(hidden[:, -1, :])
        else:
            # Assume model returns logits directly or has a built-in head
            logits = hidden[:, -1, :]

        # Sample next token
        next_token = sample_next_token(
            logits,
            temperature=config.temperature,
            top_k=config.top_k,
            top_p=config.top_p,
        )

        # Append to generated sequence using pre-allocated buffer
        generated[:, gen_len] = next_token
        gen_len += 1

        # Check for EOS
        if config.eos_token_id is not None:
            if (next_token == config.eos_token_id).all():
                break

    return generated[:, :gen_len]


class Generator:
    """Stateful generator with persistent KV-cache.

    Supports both standard softmax-based sampling and diffusion-based
    sampling when the model has a diffusion head. Diffusion sampling
    is slower (multi-step) but can produce higher quality outputs.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        config: GenerateConfig | None = None,
        lm_head: nn.Module | None = None,
        device: torch.device | None = None,
    ) -> None:
        """Initialize the generator.

        Args:
            model: The transformer model
            config: Generation configuration
            lm_head: Optional language model head
            device: Device for cache allocation
        """
        self.model = model
        self.config = config or GenerateConfig()
        self.lm_head = lm_head
        self.device = device or torch.device("cpu")

        # Caches will be created on first generation
        self._caches: list[LayerKVCache | DecoupledLayerKVCache] | None = None
        self._ctx: InferContext | None = None
        self._pos: int = 0

        # Check for diffusion head
        self._has_diffusion = has_diffusion_head(model)

    def reset(self) -> None:
        """Reset the generator state (clear caches)."""
        self._caches = None
        self._ctx = None
        self._pos = 0

    def _ensure_caches(self, batch_size: int) -> None:
        """Ensure caches are allocated for the given batch size."""
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

    @torch.inference_mode()
    def prefill(self, input_ids: Tensor) -> Tensor:
        """Prefill the cache with input tokens.

        Args:
            input_ids: Input token ids (B, T)

        Returns:
            Logits for the last token (B, vocab_size)
        """
        batch_size = input_ids.size(0)
        self._ensure_caches(batch_size)
        assert self._ctx is not None

        self._ctx.begin(pos_offset=0)

        # Use return_features path if we need diffusion sampling
        use_diffusion = self.config.use_diffusion and self._has_diffusion
        hidden: Tensor
        if use_diffusion and hasattr(self.model, "forward"):
            try:
                result = self.model(input_ids, ctx=self._ctx, return_features=True)  # type: ignore[call-arg]
                if isinstance(result, tuple) and len(result) == 2:
                    self._last_features = result[0]
                    hidden = result[0]  # Use features for diffusion
                else:
                    hidden = result if isinstance(result, Tensor) else result[0]  # type: ignore[index]
                    self._last_features = None
            except TypeError:
                # Model doesn't support return_features
                result2 = self.model(input_ids, ctx=self._ctx)  # type: ignore[call-arg]
                hidden = result2 if isinstance(result2, Tensor) else result2[0]  # type: ignore[index]
                self._last_features = None
        else:
            result3 = self.model(input_ids, ctx=self._ctx)  # type: ignore[call-arg]
            hidden = result3 if isinstance(result3, Tensor) else result3[0]  # type: ignore[index]
            self._last_features = None

        self._ctx.ensure_consumed()
        self._pos = input_ids.size(1)

        # Get logits
        if use_diffusion and self._last_features is not None:
            # Use slicing that works with Tensor
            features_last = self._last_features.narrow(1, self._last_features.size(1) - 1, 1)
            return self._sample_with_diffusion(features_last)

        if self.lm_head is not None:
            # Get last token hidden state
            hidden_last = hidden.narrow(1, hidden.size(1) - 1, 1).squeeze(1)
            return self.lm_head(hidden_last)
        hidden_last = hidden.narrow(1, hidden.size(1) - 1, 1).squeeze(1)
        return hidden_last

    def _sample_with_diffusion(self, features_last: Tensor) -> Tensor:
        """Sample logits using the diffusion head.

        Args:
            features_last: Features at the last position (B, 1, d_model)

        Returns:
            Logits (B, vocab_size)
        """
        if not hasattr(self.model, "sample_with_diffusion"):
            raise RuntimeError(
                "Model does not support sample_with_diffusion method"
            )
        return self.model.sample_with_diffusion(  # type: ignore[attr-defined]
            features_last,
            temperature=self.config.temperature,
            guidance_scale=self.config.diffusion_guidance_scale,
        )

    @torch.inference_mode()
    def decode_step(self, token_ids: Tensor) -> Tensor:
        """Decode one step with the given tokens.

        Args:
            token_ids: Token ids to decode (B, 1) or (B,)

        Returns:
            Logits for the next token (B, vocab_size)
        """
        assert self._ctx is not None

        if token_ids.dim() == 1:
            token_ids = token_ids.unsqueeze(-1)

        self._ctx.begin(pos_offset=self._pos)

        # Use return_features path if we need diffusion sampling
        use_diffusion = self.config.use_diffusion and self._has_diffusion
        hidden: Tensor
        if use_diffusion and hasattr(self.model, "forward"):
            try:
                result = self.model(token_ids, ctx=self._ctx, return_features=True)  # type: ignore[call-arg]
                if isinstance(result, tuple) and len(result) == 2:
                    self._last_features = result[0]
                    hidden = result[0]
                else:
                    hidden = result if isinstance(result, Tensor) else result[0]  # type: ignore[index]
                    self._last_features = None
            except TypeError:
                result2 = self.model(token_ids, ctx=self._ctx)  # type: ignore[call-arg]
                hidden = result2 if isinstance(result2, Tensor) else result2[0]  # type: ignore[index]
                self._last_features = None
        else:
            result3 = self.model(token_ids, ctx=self._ctx)  # type: ignore[call-arg]
            hidden = result3 if isinstance(result3, Tensor) else result3[0]  # type: ignore[index]
            self._last_features = None

        self._ctx.ensure_consumed()
        self._pos += token_ids.size(1)

        # Get logits
        if use_diffusion and self._last_features is not None:
            features_last = self._last_features.narrow(1, self._last_features.size(1) - 1, 1)
            return self._sample_with_diffusion(features_last)

        if self.lm_head is not None:
            hidden_last = hidden.narrow(1, hidden.size(1) - 1, 1).squeeze(1)
            return self.lm_head(hidden_last)
        hidden_last = hidden.narrow(1, hidden.size(1) - 1, 1).squeeze(1)
        return hidden_last

    @torch.inference_mode()
    def generate(self, input_ids: Tensor) -> Tensor:
        """Generate tokens autoregressively.

        Args:
            input_ids: Initial token ids (B, T)

        Returns:
            Generated token ids (B, T + max_new_tokens)
        """
        self.reset()
        batch_size, seq_len = input_ids.shape
        self._ensure_caches(batch_size)

        # Pre-allocate buffer for generated tokens to avoid O(n^2) torch.cat
        max_gen_len = seq_len + self.config.max_new_tokens
        generated = torch.empty(
            (batch_size, max_gen_len),
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        generated[:, :seq_len] = input_ids
        gen_len = seq_len

        # Prefill
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

            # Decode step with the new token
            logits = self.decode_step(next_token)

        return generated[:, :gen_len]

    def rollback(self, n_tokens: int) -> None:
        """Rollback the cache by n tokens (for speculative decoding).

        Args:
            n_tokens: Number of tokens to rollback
        """
        if self._caches is None:
            return

        new_pos = max(0, self._pos - n_tokens)
        for cache in self._caches:
            cache.truncate(new_pos)
        self._pos = new_pos
