"""Unified attention layer supporting standard, GQA, and decoupled (DBA) modes.

Modes:
- standard: Full multi-head attention
- gqa: Grouped-query attention (fewer KV heads than Q heads)
- decoupled: DBA with separate semantic (content) and geometric (position) key paths
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, cast

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from caramba.config.layer import AttentionLayerConfig, AttentionMode
from caramba.layer.rope import RotaryEmbedding

if TYPE_CHECKING:
    from caramba.cache.layer import LayerKVCache
    from caramba.cache.decoupled import DecoupledLayerKVCache


def _neg_inf(dtype: torch.dtype) -> float:
    """Return a large negative value safe for the given dtype."""
    if dtype == torch.float16:
        return -65504.0
    return -1e9


class AttentionLayer(nn.Module):
    """Unified attention with standard/GQA/decoupled (DBA) support."""

    # Declare all attributes with types
    q_proj: nn.Linear | None
    k_proj: nn.Linear | None
    v_proj: nn.Linear
    out_proj: nn.Linear
    q_sem: nn.Linear | None
    k_sem: nn.Linear | None
    q_geo: nn.Linear | None
    k_geo: nn.Linear | None
    rotary: RotaryEmbedding | None
    rotary_geo: RotaryEmbedding | None
    decoupled_gate_logit: nn.Parameter | None
    decoupled_gate_proj: nn.Linear | None
    logit_scale: nn.Parameter | None
    _scale: float | None
    _sem_scale: float | None
    _geo_scale: float | None
    _v_head_dim: int

    def __init__(self, config: AttentionLayerConfig) -> None:
        super().__init__()
        self.config = config
        self.mode = config.mode
        self.n_heads = config.n_heads
        self.n_kv_heads = config.kv_heads
        self.head_dim = config.head_dim
        self.dropout = nn.Dropout(config.dropout_p)

        # Compute group size for GQA
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError(
                f"n_heads ({self.n_heads}) must be divisible by n_kv_heads ({self.n_kv_heads})"
            )
        self.group_size = self.n_heads // self.n_kv_heads

        # Mode-specific initialization
        if self.mode == AttentionMode.DECOUPLED:
            self._init_decoupled(config)
        else:
            self._init_standard(config)

        # Learned temperature per head
        self.logit_scale = None
        if config.learned_temp:
            self.logit_scale = nn.Parameter(torch.zeros(self.n_heads))

    def _init_standard(self, config: AttentionLayerConfig) -> None:
        """Initialize standard/GQA attention projections."""
        d_model = config.d_model
        attn_dim = config.attn_dim if config.attn_dim else d_model
        kv_dim = self.n_kv_heads * self.head_dim

        self.q_proj = nn.Linear(d_model, attn_dim, bias=config.bias)
        self.k_proj = nn.Linear(d_model, kv_dim, bias=config.bias)
        self.v_proj = nn.Linear(d_model, kv_dim, bias=config.bias)
        self.out_proj = nn.Linear(attn_dim, d_model, bias=config.bias)

        # RoPE for standard/GQA
        if config.rope_enabled:
            self.rotary = RotaryEmbedding(self.head_dim, base=config.rope_base)
        else:
            self.rotary = None

        # Pre-computed scale
        self._scale = 1.0 / math.sqrt(float(self.head_dim))

        # Decoupled projections (not used in this mode)
        self.q_sem = None
        self.k_sem = None
        self.q_geo = None
        self.k_geo = None
        self.rotary_geo = None
        self._sem_scale = None
        self._geo_scale = None
        self._v_head_dim = self.head_dim
        self.decoupled_gate_logit = None
        self.decoupled_gate_proj = None

    def _init_decoupled(self, config: AttentionLayerConfig) -> None:
        """Initialize decoupled (DBA) attention projections."""
        d_model = config.d_model

        if config.sem_dim is None or config.geo_dim is None:
            raise ValueError("Decoupled mode requires sem_dim and geo_dim")

        sem_dim = config.sem_dim
        geo_dim = config.geo_dim
        v_dim = config.v_dim

        sem_head_dim = config.sem_head_dim
        geo_head_dim = config.geo_head_dim

        if sem_head_dim is None or geo_head_dim is None:
            raise ValueError("Could not compute sem/geo head dims")

        # Semantic projections (no RoPE - content similarity only)
        self.q_sem = nn.Linear(d_model, sem_dim, bias=config.bias)
        self.k_sem = nn.Linear(d_model, sem_dim, bias=config.bias)

        # Geometric projections (RoPE applied - position similarity)
        self.q_geo = nn.Linear(d_model, geo_dim, bias=config.bias)
        self.k_geo = nn.Linear(d_model, geo_dim, bias=config.bias)

        # Value and output
        self.v_proj = nn.Linear(d_model, v_dim, bias=config.bias)
        self.out_proj = nn.Linear(v_dim, d_model, bias=config.bias)

        # RoPE only for geometric path
        if config.rope_enabled:
            if geo_head_dim % 2 != 0:
                raise ValueError("Decoupled mode with RoPE requires even geo_head_dim")
            self.rotary_geo = RotaryEmbedding(geo_head_dim, base=config.rope_base)
        else:
            self.rotary_geo = None

        # Scales
        self._sem_scale = 1.0 / math.sqrt(float(sem_head_dim))
        self._geo_scale = 1.0 / math.sqrt(float(geo_head_dim))
        self._v_head_dim = v_dim // self.n_heads

        # Optional per-head gating between semantic and geometric
        if config.decoupled_gate:
            self.decoupled_gate_logit = nn.Parameter(torch.zeros(self.n_heads))
            if config.decoupled_gate_dynamic:
                self.decoupled_gate_proj = nn.Linear(d_model, self.n_heads, bias=False)
                nn.init.zeros_(self.decoupled_gate_proj.weight)
            else:
                self.decoupled_gate_proj = None
        else:
            self.decoupled_gate_logit = None
            self.decoupled_gate_proj = None

        # Standard projections (not used in this mode)
        self.q_proj = None
        self.k_proj = None
        self.rotary = None
        self._scale = None

    def _shape(self, x: Tensor, head_dim: int, n_heads: int | None = None) -> Tensor:
        """Reshape (B, T, D) -> (B, H, T, head_dim)."""
        B, T, _ = x.shape
        H = n_heads if n_heads is not None else self.n_heads
        return x.view(B, T, H, head_dim).transpose(1, 2).contiguous()

    def _merge(self, x: Tensor) -> Tensor:
        """Reshape (B, H, T, head_dim) -> (B, T, D)."""
        B, H, T, hd = x.shape
        return x.transpose(1, 2).contiguous().view(B, T, H * hd)

    def _apply_logit_scale(self, q: Tensor) -> Tensor:
        """Apply learned per-head temperature scaling."""
        if self.logit_scale is None:
            return q
        s = self.logit_scale.float().clamp(min=-8.0, max=8.0)
        scale = torch.exp(s).to(dtype=q.dtype).view(1, -1, 1, 1)
        return q * scale

    def _decoupled_gate(self, x: Tensor) -> Tensor | None:
        """Compute per-head semantic/geometric mixing gate."""
        if self.decoupled_gate_logit is None:
            return None
        gate_bias = self.decoupled_gate_logit.view(1, -1, 1, 1).to(
            dtype=torch.float32, device=x.device
        )
        if self.decoupled_gate_proj is None:
            gate_logit = gate_bias
        else:
            dyn = self.decoupled_gate_proj(x).transpose(1, 2).unsqueeze(-1).to(torch.float32)
            gate_logit = gate_bias + dyn
        return torch.sigmoid(gate_logit).to(dtype=x.dtype)

    def forward(
        self,
        x: Tensor,
        *,
        mask: Tensor | None = None,
        cache: "LayerKVCache | DecoupledLayerKVCache | None" = None,
        pos_offset: int = 0,
        ctx: object | None = None,
    ) -> tuple[Tensor, "LayerKVCache | DecoupledLayerKVCache | None"]:
        """Forward pass with optional KV cache.

        Args:
            x: Input tensor (B, T, d_model)
            mask: Optional attention mask (B, 1, T, S) or None for causal
            cache: Optional KV cache for incremental decoding
            pos_offset: Position offset for RoPE (for cached generation)
            ctx: Optional InferContext containing caches and position info

        Returns:
            Tuple of (output, updated_cache)
        """
        # Extract cache and pos_offset from InferContext if provided
        from caramba.infer.context import InferContext
        if ctx is not None and isinstance(ctx, InferContext):
            cache = ctx.next_cache()
            pos_offset = ctx.pos_offset
            if ctx.attn_mask is not None:
                mask = ctx.attn_mask

        if self.mode == AttentionMode.DECOUPLED:
            # Cast to decoupled cache type (runtime check happens in forward)
            decoupled_cache = cast("DecoupledLayerKVCache | None", cache)
            return self._forward_decoupled(x, mask=mask, cache=decoupled_cache, pos_offset=pos_offset)
        # Cast to standard cache type
        standard_cache = cast("LayerKVCache | None", cache)
        return self._forward_standard(x, mask=mask, cache=standard_cache, pos_offset=pos_offset)

    def _forward_standard(
        self,
        x: Tensor,
        *,
        mask: Tensor | None,
        cache: "LayerKVCache | None",
        pos_offset: int,
    ) -> tuple[Tensor, "LayerKVCache | None"]:
        """Standard/GQA attention forward."""
        B, T, _ = x.shape

        if self.q_proj is None or self.k_proj is None:
            raise RuntimeError("Standard mode projections not initialized")

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        qh = self._shape(q, self.head_dim, self.n_heads)
        kh = self._shape(k, self.head_dim, self.n_kv_heads)
        vh = self._shape(v, self.head_dim, self.n_kv_heads)

        # Apply RoPE
        if self.rotary is not None:
            qh = self.rotary.rotate(qh, pos_offset)
            kh = self.rotary.rotate(kh, pos_offset)

        # Apply learned temperature
        qh = self._apply_logit_scale(qh)

        # Handle cache
        if cache is not None:
            old_len = cache.pos
            _ = cache.append(self._merge(kh), self._merge(vh))

            if T == 1 and old_len > 0:
                # Decode: use full cached K/V
                k_all, v_all = cache.get(dtype=qh.dtype)
                kh = self._shape(k_all, self.head_dim, self.n_kv_heads)
                vh = self._shape(v_all, self.head_dim, self.n_kv_heads)
            elif old_len > 0:
                # Prefill with existing cache
                k_all, v_all = cache.get(dtype=qh.dtype)
                kh = self._shape(k_all, self.head_dim, self.n_kv_heads)
                vh = self._shape(v_all, self.head_dim, self.n_kv_heads)

        # Expand KV heads for GQA
        if self.group_size > 1:
            kh = kh.repeat_interleave(self.group_size, dim=1)
            vh = vh.repeat_interleave(self.group_size, dim=1)

        # Compute attention
        is_causal = self.config.is_causal and mask is None and T > 1 and cache is None
        out = F.scaled_dot_product_attention(
            qh, kh, vh,
            attn_mask=mask,
            dropout_p=self.config.dropout_p if self.training else 0.0,
            is_causal=is_causal,
            scale=self._scale,
        )

        y = self.out_proj(self._merge(out))
        return y, cache

    def _forward_decoupled(
        self,
        x: Tensor,
        *,
        mask: Tensor | None,
        cache: "DecoupledLayerKVCache | None",
        pos_offset: int,
    ) -> tuple[Tensor, "DecoupledLayerKVCache | None"]:
        """Decoupled (DBA) attention forward."""
        B, T, _ = x.shape
        ninfty = _neg_inf(x.dtype)

        if self.q_sem is None or self.k_sem is None:
            raise RuntimeError("Decoupled mode projections not initialized")
        if self.q_geo is None or self.k_geo is None:
            raise RuntimeError("Decoupled mode projections not initialized")
        if self._sem_scale is None or self._geo_scale is None:
            raise RuntimeError("Decoupled mode scales not initialized")

        sem_head_dim = self.config.sem_head_dim
        geo_head_dim = self.config.geo_head_dim
        v_head_dim = self._v_head_dim

        if sem_head_dim is None or geo_head_dim is None:
            raise RuntimeError("Head dims not set")

        # Semantic path (no RoPE)
        q_sem = self.q_sem(x)
        k_sem = self.k_sem(x)
        qsh = self._shape(q_sem, sem_head_dim)
        ksh = self._shape(k_sem, sem_head_dim)

        # Geometric path (with RoPE)
        q_geo = self.q_geo(x)
        k_geo = self.k_geo(x)
        qgh = self._shape(q_geo, geo_head_dim)
        kgh = self._shape(k_geo, geo_head_dim)

        if self.rotary_geo is not None:
            qgh = self.rotary_geo.rotate(qgh, pos_offset)
            kgh = self.rotary_geo.rotate(kgh, pos_offset)

        # Value
        v = self.v_proj(x)
        vh = self._shape(v, v_head_dim)

        # Apply learned temperature
        qsh = self._apply_logit_scale(qsh)
        qgh = self._apply_logit_scale(qgh)

        # Apply gating
        g = self._decoupled_gate(x)
        if g is not None:
            qsh = qsh * (2.0 * g)
            qgh = qgh * (2.0 - 2.0 * g)

        # Handle cache
        if cache is not None:
            old_len = cache.pos
            _ = cache.append(self._merge(ksh), self._merge(kgh), self._merge(vh))

            if T == 1 and old_len > 0:
                k_sem_all, k_geo_all, v_all = cache.get(dtype=qsh.dtype)
                ksh = self._shape(k_sem_all, sem_head_dim)
                kgh = self._shape(k_geo_all, geo_head_dim)
                vh = self._shape(v_all, v_head_dim)
            elif old_len > 0:
                k_sem_all, k_geo_all, v_all = cache.get(dtype=qsh.dtype)
                ksh = self._shape(k_sem_all, sem_head_dim)
                kgh = self._shape(k_geo_all, geo_head_dim)
                vh = self._shape(v_all, v_head_dim)

        # Compute decoupled scores: semantic + geometric
        sem_scores = torch.matmul(qsh, ksh.transpose(-2, -1)) * self._sem_scale
        geo_scores = torch.matmul(qgh, kgh.transpose(-2, -1)) * self._geo_scale
        scores = sem_scores + geo_scores

        # Apply mask
        if mask is not None:
            scores = scores.masked_fill(~mask, ninfty)
        elif self.config.is_causal and T > 1 and cache is None:
            # Build causal mask
            causal = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
            scores = scores.masked_fill(~causal.view(1, 1, T, T), ninfty)
        elif self.config.is_causal and cache is not None:
            # Causal mask for cached generation
            cache_len = ksh.size(2)
            key_pos = torch.arange(cache_len, device=x.device).view(1, 1, 1, cache_len)
            q_pos = (cache.pos - T + torch.arange(T, device=x.device)).view(1, 1, T, 1)
            keep = key_pos <= q_pos
            scores = scores.masked_fill(~keep, ninfty)

        # Softmax and dropout
        attn = F.softmax(scores.float(), dim=-1).to(x.dtype)
        attn = self.dropout(attn)

        # Output
        out = torch.matmul(attn, vh)
        y = self.out_proj(self._merge(out))

        return y, cache
