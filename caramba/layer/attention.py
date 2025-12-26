"""Unified attention layer supporting standard, GQA, and DBA modes.

Attention is the core mechanism that lets tokens "look at" each other.
This module supports three modes:
- standard: Full multi-head attention (every head has its own K/V)
- gqa: Grouped-query attention (fewer K/V heads, shared across Q heads)
- decoupled: DBA with separate semantic and geometric key paths

The decoupled (DBA) mode is our research contribution—it splits attention
into content-based (semantic) and position-based (geometric) components,
enabling significant KV-cache compression.
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
    from caramba.cache.decoupled import DecoupledLayerKVCache
    from caramba.cache.layer import LayerKVCache

# Lazy-cached reference to avoid per-call import overhead
_InferContext: type | None = None


def _get_infer_context_type() -> type:
    """Get the InferContext type, caching it on first access."""
    global _InferContext
    if _InferContext is None:
        from caramba.infer.context import InferContext

        _InferContext = InferContext
    return _InferContext


def _neg_inf(dtype: torch.dtype) -> float:
    """Return a large negative value safe for the given dtype.

    Float16 has limited range, so we use -65504 instead of -1e9.
    """
    if dtype == torch.float16:
        return -65504.0
    return -1e9


class AttentionLayer(nn.Module):
    """Multi-head attention with standard/GQA/DBA support.

    The mode determines the attention computation:
    - standard/gqa: Traditional Q·K^T → softmax → V pipeline
    - decoupled: (Q_sem·K_sem^T + Q_geo·K_geo^T) → softmax → V

    DBA's key insight is that content routing (semantic) and position
    patterns (geometric) are separate concerns that can use compressed
    key projections, reducing KV-cache memory dramatically.
    """

    # Type declarations for all attributes
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
    mem_k_proj: nn.Module | None
    mem_v_proj: nn.Module | None

    def __init__(self, config: AttentionLayerConfig) -> None:
        """Initialize attention based on the configured mode.

        The config specifies dimensions, number of heads, RoPE settings,
        and mode-specific parameters (sem_dim, geo_dim for DBA).
        """
        super().__init__()
        self.config = config
        self.mode = config.mode
        self.n_heads = config.n_heads
        self.n_kv_heads = config.kv_heads
        self.head_dim = config.head_dim
        self.dropout = nn.Dropout(config.dropout_p)
        self.mem_k_proj = None
        self.mem_v_proj = None

        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError(
                f"n_heads ({self.n_heads}) must be divisible by "
                f"n_kv_heads ({self.n_kv_heads})"
            )
        self.group_size = self.n_heads // self.n_kv_heads

        if self.mode == AttentionMode.DECOUPLED:
            self._init_decoupled(config)
        else:
            self._init_standard(config)

        # Optional learned temperature scaling per head
        self.logit_scale = None
        if config.learned_temp:
            self.logit_scale = nn.Parameter(torch.zeros(self.n_heads))

        # Optional long-sequence memory summarization modules.
        self._init_memory_summarizer()

    def _init_memory_summarizer(self) -> None:
        """Initialize optional modules for mem_block summarization."""

        kind = str(getattr(self.config, "mem_summarize", "mean")).lower()
        if kind == "linear":
            d = int(self.head_dim)
            self.mem_k_proj = nn.Linear(d, d, bias=False)
            self.mem_v_proj = nn.Linear(d, d, bias=False)
            # Initialize as identity so "linear" starts equivalent to mean pooling.
            nn.init.eye_(self.mem_k_proj.weight)
            nn.init.eye_(self.mem_v_proj.weight)
        elif kind == "conv":
            d = int(self.head_dim)
            # Depthwise conv over sequence dimension; stable and cheap.
            self.mem_k_proj = nn.Conv1d(d, d, kernel_size=3, padding=1, groups=d, bias=False)
            self.mem_v_proj = nn.Conv1d(d, d, kernel_size=3, padding=1, groups=d, bias=False)
            # Initialize to a simple smoothing kernel [0.25, 0.5, 0.25] per channel.
            w = torch.tensor([0.25, 0.5, 0.25], dtype=torch.float32)
            wk = cast(nn.Conv1d, self.mem_k_proj).weight
            wv = cast(nn.Conv1d, self.mem_v_proj).weight
            wk.data.zero_()
            wv.data.zero_()
            wk.data[:, 0, :].copy_(w.to(device=wk.device, dtype=wk.dtype).view(1, 3).expand(d, 3))
            wv.data[:, 0, :].copy_(w.to(device=wv.device, dtype=wv.dtype).view(1, 3).expand(d, 3))
        else:
            self.mem_k_proj = None
            self.mem_v_proj = None

    def _maybe_summarize_kv(
        self,
        *,
        k: Tensor,
        v: Tensor,
        k_pos: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Summarize older KV blocks into memory tokens for long sequences.

        This is an approximation used to reduce compute/memory on long sequences.
        It is inactive unless config.mem_block is set and activation threshold
        (if any) is met.
        """

        mem_block = getattr(self.config, "mem_block", None)
        if mem_block is None:
            return k, v, k_pos
        mb = int(mem_block)
        if mb <= 0:
            return k, v, k_pos

        # Guard for empty k tensor.
        if k.size(2) == 0:
            return k, v, k_pos

        threshold = getattr(self.config, "mem_activation_threshold", None)
        if threshold is not None and int(k.size(2)) < int(threshold):
            return k, v, k_pos

        # If local_window is set, keep that many most-recent tokens uncompressed.
        local_window = getattr(self.config, "local_window", None)
        lw = int(local_window) if local_window is not None else 0
        T = int(k.size(2))
        if lw <= 0 or lw >= T:
            return k, v, k_pos

        remote_len = T - lw
        if remote_len <= 0:
            return k, v, k_pos

        k_remote = k[:, :, :remote_len, :]
        v_remote = v[:, :, :remote_len, :]
        k_local = k[:, :, remote_len:, :]
        v_local = v[:, :, remote_len:, :]
        pos_remote = k_pos[:remote_len]
        pos_local = k_pos[remote_len:]

        # Optional conv preprocessing (per head) before pooling.
        kind = str(getattr(self.config, "mem_summarize", "mean")).lower()
        if kind == "conv" and self.mem_k_proj is not None and self.mem_v_proj is not None:
            # (B,H,T,D) -> (B*H,D,T)
            BH = int(k_remote.size(0) * k_remote.size(1))
            d = int(k_remote.size(-1))
            k_in = k_remote.reshape(BH, remote_len, d).transpose(1, 2)
            v_in = v_remote.reshape(BH, remote_len, d).transpose(1, 2)
            k_f = cast(nn.Conv1d, self.mem_k_proj)(k_in).transpose(1, 2).reshape_as(k_remote)
            v_f = cast(nn.Conv1d, self.mem_v_proj)(v_in).transpose(1, 2).reshape_as(v_remote)
            k_remote = k_f
            v_remote = v_f

        # Pool remote into blocks.
        k_blocks: list[Tensor] = []
        v_blocks: list[Tensor] = []
        pos_blocks: list[int] = []
        for i0 in range(0, remote_len, mb):
            i1 = min(remote_len, i0 + mb)
            k_b = k_remote[:, :, i0:i1, :].mean(dim=2, keepdim=False)
            v_b = v_remote[:, :, i0:i1, :].mean(dim=2, keepdim=False)
            if kind == "linear" and self.mem_k_proj is not None and self.mem_v_proj is not None:
                k_b = cast(nn.Linear, self.mem_k_proj)(k_b)
                v_b = cast(nn.Linear, self.mem_v_proj)(v_b)
            k_blocks.append(k_b)
            v_blocks.append(v_b)
            pos_blocks.append(int(pos_remote[i1 - 1].item()))

        k_mem = torch.stack(k_blocks, dim=2)  # (B,H,M,D)
        v_mem = torch.stack(v_blocks, dim=2)
        pos_mem = torch.tensor(pos_blocks, device=k.device, dtype=pos_remote.dtype)

        k2 = torch.cat([k_mem, k_local], dim=2)
        v2 = torch.cat([v_mem, v_local], dim=2)
        pos2 = torch.cat([pos_mem, pos_local], dim=0)
        return k2, v2, pos2

    def _init_standard(self, config: AttentionLayerConfig) -> None:
        """Set up projections for standard/GQA attention."""
        d_model = config.d_model
        attn_dim = config.attn_dim if config.attn_dim else d_model
        kv_dim = self.n_kv_heads * self.head_dim

        self.q_proj = nn.Linear(d_model, attn_dim, bias=config.bias)
        self.k_proj = nn.Linear(d_model, kv_dim, bias=config.bias)
        self.v_proj = nn.Linear(d_model, kv_dim, bias=config.bias)
        self.out_proj = nn.Linear(attn_dim, d_model, bias=config.bias)

        if config.rope_enabled:
            self.rotary = RotaryEmbedding(self.head_dim, base=config.rope_base)
        else:
            self.rotary = None

        self._scale = 1.0 / math.sqrt(float(self.head_dim))

        # Decoupled projections unused in this mode
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
        """Set up projections for DBA attention.

        DBA has two key paths:
        - Semantic (no RoPE): learns content/topic similarity
        - Geometric (with RoPE): learns position-based patterns

        These are combined before softmax, allowing the model to learn
        which path to emphasize for different attention patterns.
        """
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

        # Semantic projections (content similarity, no position encoding)
        self.q_sem = nn.Linear(d_model, sem_dim, bias=config.bias)
        self.k_sem = nn.Linear(d_model, sem_dim, bias=config.bias)

        # Geometric projections (position patterns, RoPE applied)
        self.q_geo = nn.Linear(d_model, geo_dim, bias=config.bias)
        self.k_geo = nn.Linear(d_model, geo_dim, bias=config.bias)

        self.v_proj = nn.Linear(d_model, v_dim, bias=config.bias)
        self.out_proj = nn.Linear(v_dim, d_model, bias=config.bias)

        if config.rope_enabled:
            if geo_head_dim % 2 != 0:
                raise ValueError("Decoupled mode with RoPE requires even geo_head_dim")
            self.rotary_geo = RotaryEmbedding(geo_head_dim, base=config.rope_base)
        else:
            self.rotary_geo = None

        self._sem_scale = 1.0 / math.sqrt(float(sem_head_dim))
        self._geo_scale = 1.0 / math.sqrt(float(geo_head_dim))
        self._v_head_dim = v_dim // self.n_heads

        # Optional learned gate between semantic and geometric paths
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

        # Standard projections unused in this mode
        self.q_proj = None
        self.k_proj = None
        self.rotary = None
        self._scale = None

    def _shape(self, x: Tensor, head_dim: int, n_heads: int | None = None) -> Tensor:
        """Reshape (B, T, D) → (B, H, T, head_dim) for attention."""
        B, T, _ = x.shape
        H = n_heads if n_heads is not None else self.n_heads
        return x.view(B, T, H, head_dim).transpose(1, 2).contiguous()

    def _merge(self, x: Tensor) -> Tensor:
        """Reshape (B, H, T, head_dim) → (B, T, D) after attention."""
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
        """Compute per-head semantic/geometric mixing weights.

        Returns a gate value in [0, 1] where 1 means fully semantic
        and 0 means fully geometric.
        """
        if self.decoupled_gate_logit is None:
            return None
        gate_bias = self.decoupled_gate_logit.view(1, -1, 1, 1).to(
            dtype=torch.float32, device=x.device
        )
        if self.decoupled_gate_proj is None:
            gate_logit = gate_bias
        else:
            dyn = (
                self.decoupled_gate_proj(x).transpose(1, 2).unsqueeze(-1).to(torch.float32)
            )
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
        """Compute attention and return output with updated cache.

        Args:
            x: Input features (B, T, d_model)
            mask: Optional attention mask (B, 1, T, S)
            cache: Optional KV cache for incremental decoding
            pos_offset: Position offset for RoPE in cached generation
            ctx: Optional InferContext containing caches and position info

        Returns:
            (output, updated_cache) tuple
        """
        InferContextType = _get_infer_context_type()
        q_chunk_override: int | None = None
        local_window_override: int | None = None
        decode_block_override: int | None = None
        if ctx is not None and isinstance(ctx, InferContextType):
            cache = ctx.next_cache()  # type: ignore[union-attr]
            pos_offset = ctx.pos_offset  # type: ignore[union-attr]
            if ctx.attn_mask is not None:  # type: ignore[union-attr]
                mask = ctx.attn_mask  # type: ignore[union-attr]
            q_chunk_override = getattr(ctx, "q_chunk", None)  # type: ignore[assignment]
            local_window_override = getattr(ctx, "local_window", None)  # type: ignore[assignment]
            decode_block_override = getattr(ctx, "decode_block", None)  # type: ignore[assignment]

        if self.mode == AttentionMode.DECOUPLED:
            decoupled_cache = cast("DecoupledLayerKVCache | None", cache)
            return self._forward_decoupled(
                x,
                mask=mask,
                cache=decoupled_cache,
                pos_offset=pos_offset,
                q_chunk_override=q_chunk_override,
                local_window_override=local_window_override,
                decode_block_override=decode_block_override,
            )
        standard_cache = cast("LayerKVCache | None", cache)
        return self._forward_standard(
            x,
            mask=mask,
            cache=standard_cache,
            pos_offset=pos_offset,
            q_chunk_override=q_chunk_override,
            local_window_override=local_window_override,
        )

    def _forward_standard(
        self,
        x: Tensor,
        *,
        mask: Tensor | None,
        cache: "LayerKVCache | None",
        pos_offset: int,
        q_chunk_override: int | None = None,
        local_window_override: int | None = None,
    ) -> tuple[Tensor, "LayerKVCache | None"]:
        """Standard/GQA attention: Q·K^T → softmax → V."""
        B, T, _ = x.shape

        if self.q_proj is None or self.k_proj is None:
            raise RuntimeError("Standard mode projections not initialized")

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        qh = self._shape(q, self.head_dim, self.n_heads)
        kh = self._shape(k, self.head_dim, self.n_kv_heads)
        vh = self._shape(v, self.head_dim, self.n_kv_heads)

        if self.rotary is not None:
            qh = self.rotary.rotate(qh, pos_offset)
            kh = self.rotary.rotate(kh, pos_offset)

        qh = self._apply_logit_scale(qh)

        if cache is not None:
            old_len = cache.pos
            _ = cache.append(self._merge(kh), self._merge(vh))
            if old_len > 0:
                k_all, v_all = cache.get(dtype=qh.dtype)
                kh = self._shape(k_all, self.head_dim, self.n_kv_heads)
                vh = self._shape(v_all, self.head_dim, self.n_kv_heads)

        if self.group_size > 1:
            kh = kh.repeat_interleave(self.group_size, dim=1)
            vh = vh.repeat_interleave(self.group_size, dim=1)

        q_chunk = q_chunk_override if q_chunk_override is not None else self.config.q_chunk
        local_window = (
            local_window_override
            if local_window_override is not None
            else self.config.local_window
        )
        dropout_p = self.config.dropout_p if self.training else 0.0

        # Memory-efficient path: compute attention in query chunks and/or restrict to a window.
        # Also fixes causal masking for cached inference prefill/decode by using explicit masks.
        if mask is None and (q_chunk is not None or local_window is not None or cache is not None):
            out = self._sdp_attention_chunked(
                qh,
                kh,
                vh,
                pos_offset=pos_offset,
                cache=cache,
                q_chunk=int(q_chunk) if q_chunk is not None else T,
                local_window=int(local_window) if local_window is not None else None,
                dropout_p=float(dropout_p),
            )
        else:
            is_causal = self.config.is_causal and mask is None and T > 1 and cache is None
            out = F.scaled_dot_product_attention(
                qh,
                kh,
                vh,
                attn_mask=mask,
                dropout_p=dropout_p,
                is_causal=is_causal,
                scale=self._scale,
            )

        y = self.out_proj(self._merge(out))
        return y, cache

    def _sdp_attention_chunked(
        self,
        qh: Tensor,
        kh: Tensor,
        vh: Tensor,
        *,
        pos_offset: int,
        cache: "LayerKVCache | None",
        q_chunk: int,
        local_window: int | None,
        dropout_p: float,
    ) -> Tensor:
        """Scaled-dot-product attention with chunking/windowing for lower peak memory."""

        B, H, T, D = qh.shape
        kT = kh.size(2)

        # Base positions: in cached mode, q positions are aligned to the global cache index.
        # We mirror the decoupled implementation semantics.
        if cache is not None:
            base_q = int(cache.pos) - int(T)
            q_pos_full = base_q + torch.arange(T, device=qh.device)
            k_pos_full = torch.arange(kT, device=qh.device)
        else:
            q_pos_full = pos_offset + torch.arange(T, device=qh.device)
            k_pos_full = pos_offset + torch.arange(kT, device=qh.device)

        outs: list[Tensor] = []
        q_chunk = max(1, int(q_chunk))
        for i0 in range(0, T, q_chunk):
            i1 = min(T, i0 + q_chunk)

            q_pos = q_pos_full[i0:i1]

            # Key range selection to reduce work further when local_window is set.
            k0 = 0
            k1 = kT
            if local_window is not None:
                w = int(local_window)
                if w > 0:
                    q_min = int(q_pos.min().item())
                    q_max = int(q_pos.max().item())
                    if self.config.is_causal:
                        k0 = max(0, q_min - w + 1)
                        k1 = min(kT, q_max + 1)
                    else:
                        k0 = max(0, q_min - w + 1)
                        k1 = min(kT, q_max + w)

            q_slice = qh[:, :, i0:i1, :]
            k_slice = kh[:, :, k0:k1, :]
            v_slice = vh[:, :, k0:k1, :]

            # Build a boolean "allowed positions" matrix for SDPA.
            if self.config.is_causal or local_window is not None:
                k_pos = k_pos_full[k0:k1]
                # Optional memory summarization over the key/value sequence.
                k_slice, v_slice, k_pos = self._maybe_summarize_kv(k=k_slice, v=v_slice, k_pos=k_pos)
                allowed = torch.ones((q_pos.numel(), k_pos.numel()), device=qh.device, dtype=torch.bool)
                if self.config.is_causal:
                    allowed &= k_pos.view(1, -1) <= q_pos.view(-1, 1)
                if local_window is not None:
                    w = int(local_window)
                    if w > 0:
                        allowed &= k_pos.view(1, -1) >= (q_pos.view(-1, 1) - w + 1)
                        if not self.config.is_causal:
                            allowed &= k_pos.view(1, -1) <= (q_pos.view(-1, 1) + w - 1)
                attn_mask = allowed  # True = allowed for SDPA
            else:
                # Even without causal/window masks, allow optional summarization.
                k_pos = k_pos_full[k0:k1]
                k_slice, v_slice, _k_pos2 = self._maybe_summarize_kv(k=k_slice, v=v_slice, k_pos=k_pos)
                attn_mask = None

            out = F.scaled_dot_product_attention(
                q_slice,
                k_slice,
                v_slice,
                attn_mask=attn_mask,
                dropout_p=dropout_p,
                is_causal=False,
                scale=self._scale,
            )
            outs.append(out)

        return torch.cat(outs, dim=2)

    def _forward_decoupled(
        self,
        x: Tensor,
        *,
        mask: Tensor | None,
        cache: "DecoupledLayerKVCache | None",
        pos_offset: int,
        q_chunk_override: int | None = None,
        local_window_override: int | None = None,
        decode_block_override: int | None = None,
    ) -> tuple[Tensor, "DecoupledLayerKVCache | None"]:
        """DBA attention: (Q_sem·K_sem^T + Q_geo·K_geo^T) → softmax → V.

        The semantic path captures content-based routing (what tokens to
        attend to based on meaning). The geometric path captures position-
        based patterns (local attention, recency bias). Combining them
        before softmax lets the model learn the right balance.
        """
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

        # Semantic path (no RoPE—pure content similarity)
        q_sem = self.q_sem(x)
        k_sem = self.k_sem(x)
        qsh = self._shape(q_sem, sem_head_dim)
        ksh = self._shape(k_sem, sem_head_dim)

        # Geometric path (with RoPE—position patterns)
        q_geo = self.q_geo(x)
        k_geo = self.k_geo(x)
        qgh = self._shape(q_geo, geo_head_dim)
        kgh = self._shape(k_geo, geo_head_dim)

        if self.rotary_geo is not None:
            qgh = self.rotary_geo.rotate(qgh, pos_offset)
            kgh = self.rotary_geo.rotate(kgh, pos_offset)

        v = self.v_proj(x)
        vh = self._shape(v, v_head_dim)

        qsh = self._apply_logit_scale(qsh)
        qgh = self._apply_logit_scale(qgh)

        # Apply learned gating between paths
        g = self._decoupled_gate(x)
        if g is not None:
            qsh = qsh * (2.0 * g)
            qgh = qgh * (2.0 - 2.0 * g)

        if cache is not None:
            old_len = cache.pos
            _ = cache.append(self._merge(ksh), self._merge(kgh), self._merge(vh))

            # Fast-path: fused decode for quantized decoupled caches on CUDA.
            # We only use this for single-token decode with no extra masking/windowing.
            if (
                (not self.training)
                and old_len > 0
                and int(T) == 1
                and mask is None
                and (local_window_override is None and self.config.local_window is None)
                and (q_chunk_override is None and self.config.q_chunk is None)
                and x.device.type == "cuda"
            ):
                try:
                    from caramba.optimizer.fused_attention import (
                        fused_decode_available,
                        fused_decode_decoupled_q4q8q4,
                        fused_decode_decoupled_q4q8q4_2pass,
                    )

                    if fused_decode_available(cache, x.device.type):
                        decode_block = (
                            int(decode_block_override)
                            if decode_block_override is not None
                            else 1024
                        )
                        # Heuristic: for very long prefixes, prefer split-K 2-pass decode.
                        cache_len = int(cache.pos)
                        if cache_len > 4 * int(decode_block):
                            try:
                                out_fused = fused_decode_decoupled_q4q8q4_2pass(
                                    q_sem=qsh,
                                    q_geo=qgh,
                                    cache=cache,
                                    n_heads=int(self.n_heads),
                                    sem_head_dim=int(sem_head_dim),
                                    geo_head_dim=int(geo_head_dim),
                                    v_head_dim=int(v_head_dim),
                                    decode_block=int(decode_block),
                                    sem_scale=float(self._sem_scale),
                                    geo_scale=float(self._geo_scale),
                                )
                            except Exception:
                                out_fused = fused_decode_decoupled_q4q8q4(
                                    q_sem=qsh,
                                    q_geo=qgh,
                                    cache=cache,
                                    n_heads=int(self.n_heads),
                                    sem_head_dim=int(sem_head_dim),
                                    geo_head_dim=int(geo_head_dim),
                                    v_head_dim=int(v_head_dim),
                                    decode_block=int(decode_block),
                                    sem_scale=float(self._sem_scale),
                                    geo_scale=float(self._geo_scale),
                                )
                        else:
                            out_fused = fused_decode_decoupled_q4q8q4(
                                q_sem=qsh,
                                q_geo=qgh,
                                cache=cache,
                                n_heads=int(self.n_heads),
                                sem_head_dim=int(sem_head_dim),
                                geo_head_dim=int(geo_head_dim),
                                v_head_dim=int(v_head_dim),
                                decode_block=int(decode_block),
                                sem_scale=float(self._sem_scale),
                                geo_scale=float(self._geo_scale),
                            )
                        y = self.out_proj(self._merge(out_fused.to(dtype=x.dtype)))
                        return y, cache
                except Exception:
                    # Any failure in optional fused kernels should silently fall back
                    # to the safe PyTorch implementation.
                    pass

            if old_len > 0:
                k_sem_all, k_geo_all, v_all = cache.get(dtype=qsh.dtype)
                ksh = self._shape(k_sem_all, sem_head_dim)
                kgh = self._shape(k_geo_all, geo_head_dim)
                vh = self._shape(v_all, v_head_dim)

        q_chunk = q_chunk_override if q_chunk_override is not None else self.config.q_chunk
        local_window = (
            local_window_override
            if local_window_override is not None
            else self.config.local_window
        )

        if q_chunk is None and local_window is None:
            # Combine semantic and geometric scores
            sem_scores = torch.matmul(qsh, ksh.transpose(-2, -1)) * self._sem_scale
            geo_scores = torch.matmul(qgh, kgh.transpose(-2, -1)) * self._geo_scale
            scores = sem_scores + geo_scores

            # Apply masking (mask semantics: True = keep)
            if mask is not None:
                scores = scores.masked_fill(~mask, ninfty)
            elif self.config.is_causal and T > 1 and cache is None:
                causal = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
                scores = scores.masked_fill(~causal.view(1, 1, T, T), ninfty)
            elif self.config.is_causal and cache is not None:
                cache_len = ksh.size(2)
                key_pos = torch.arange(cache_len, device=x.device).view(1, 1, 1, cache_len)
                q_pos = (cache.pos - T + torch.arange(T, device=x.device)).view(1, 1, T, 1)
                keep = key_pos <= q_pos
                scores = scores.masked_fill(~keep, ninfty)

            attn = F.softmax(scores.float(), dim=-1).to(x.dtype)
            attn = self.dropout(attn)
            out = torch.matmul(attn, vh)
        else:
            out = self._decoupled_attention_chunked(
                qsh=qsh,
                ksh=ksh,
                qgh=qgh,
                kgh=kgh,
                vh=vh,
                ninfty=ninfty,
                mask=mask,
                cache=cache,
                q_chunk=int(q_chunk) if q_chunk is not None else T,
                local_window=int(local_window) if local_window is not None else None,
            )

        y = self.out_proj(self._merge(out))

        return y, cache

    def _decoupled_attention_chunked(
        self,
        *,
        qsh: Tensor,
        ksh: Tensor,
        qgh: Tensor,
        kgh: Tensor,
        vh: Tensor,
        ninfty: float,
        mask: Tensor | None,
        cache: "DecoupledLayerKVCache | None",
        q_chunk: int,
        local_window: int | None,
    ) -> Tensor:
        """Chunked DBA attention to reduce peak memory for long sequences."""

        B, H, T, _ = qsh.shape
        kT = ksh.size(2)
        q_chunk = max(1, int(q_chunk))

        if cache is not None:
            base_q = int(cache.pos) - int(T)
            q_pos_full = base_q + torch.arange(T, device=qsh.device)
            k_pos_full = torch.arange(kT, device=qsh.device)
        else:
            q_pos_full = torch.arange(T, device=qsh.device)
            k_pos_full = torch.arange(kT, device=qsh.device)

        sem_scale = float(self._sem_scale) if self._sem_scale is not None else 1.0
        geo_scale = float(self._geo_scale) if self._geo_scale is not None else 1.0

        outs: list[Tensor] = []
        for i0 in range(0, T, q_chunk):
            i1 = min(T, i0 + q_chunk)
            q_pos = q_pos_full[i0:i1]

            # Key slice range when local_window is set.
            k0 = 0
            k1 = kT
            if local_window is not None:
                w = int(local_window)
                if w > 0:
                    q_min = int(q_pos.min().item())
                    q_max = int(q_pos.max().item())
                    if self.config.is_causal:
                        k0 = max(0, q_min - w + 1)
                        k1 = min(kT, q_max + 1)
                    else:
                        k0 = max(0, q_min - w + 1)
                        k1 = min(kT, q_max + w)

            k_pos = k_pos_full[k0:k1]
            q_slice_sem = qsh[:, :, i0:i1, :]
            q_slice_geo = qgh[:, :, i0:i1, :]
            k_slice_sem = ksh[:, :, k0:k1, :]
            k_slice_geo = kgh[:, :, k0:k1, :]
            v_slice = vh[:, :, k0:k1, :]

            # Optional memory summarization (applied consistently across K/V).
            # We summarize using the semantic K/V tensors as the reference.
            k_slice_sem, v_slice, k_pos = self._maybe_summarize_kv(
                k=k_slice_sem, v=v_slice, k_pos=k_pos
            )
            # Apply the same summarization shape to geometric keys (mean pool in blocks).
            if k_slice_geo.size(2) != k_slice_sem.size(2):
                # Fallback: recompute geometric block means to match.
                # k_pos already corresponds to the summarized sequence.
                # Keep last local_window tokens intact.
                cfg_local_window = getattr(self.config, "local_window", None)
                Tgeo = int(k_slice_geo.size(2))
                Tsem = int(k_slice_sem.size(2))
                # Ensure lw is bounded by actual tensor sizes.
                lw = int(cfg_local_window) if cfg_local_window is not None else 0
                lw = min(lw, Tgeo, Tsem)
                mem_block_val = getattr(self.config, "mem_block", None)
                if lw > 0 and lw < Tgeo and mem_block_val is not None:
                    remote_len = max(0, Tgeo - lw)
                    mb = int(mem_block_val)
                    blocks: list[Tensor] = []
                    for j0 in range(0, remote_len, mb):
                        j1 = min(remote_len, j0 + mb)
                        blocks.append(k_slice_geo[:, :, j0:j1, :].mean(dim=2))
                    if blocks:
                        k_mem_geo = torch.stack(blocks, dim=2)
                        k_local_geo = k_slice_geo[:, :, remote_len:, :]
                        k_slice_geo = torch.cat([k_mem_geo, k_local_geo], dim=2)
                    else:
                        # No blocks to stack; use local portion only.
                        k_slice_geo = k_slice_geo[:, :, remote_len:, :]

            sem_scores = torch.matmul(q_slice_sem, k_slice_sem.transpose(-2, -1)) * sem_scale
            geo_scores = torch.matmul(q_slice_geo, k_slice_geo.transpose(-2, -1)) * geo_scale
            scores = sem_scores + geo_scores

            # Build keep mask (True = attend) and apply.
            if mask is not None:
                # Fall back to provided mask semantics (True=keep).
                try:
                    m = mask[..., i0:i1, k0:k1]
                    scores = scores.masked_fill(~m, ninfty)
                except Exception:
                    pass
            else:
                keep = torch.ones((q_pos.numel(), k_pos.numel()), device=qsh.device, dtype=torch.bool)
                if self.config.is_causal:
                    keep &= k_pos.view(1, -1) <= q_pos.view(-1, 1)
                if local_window is not None:
                    w = int(local_window)
                    if w > 0:
                        keep &= k_pos.view(1, -1) >= (q_pos.view(-1, 1) - w + 1)
                        if not self.config.is_causal:
                            keep &= k_pos.view(1, -1) <= (q_pos.view(-1, 1) + w - 1)
                scores = scores.masked_fill(~keep.view(1, 1, q_pos.numel(), k_pos.numel()), ninfty)

            attn = F.softmax(scores.float(), dim=-1).to(qsh.dtype)
            attn = self.dropout(attn)
            out = torch.matmul(attn, v_slice)
            outs.append(out)

        return torch.cat(outs, dim=2)
