"""
gpt defines the neural architecture and orchestration for generation.
"""
from __future__ import annotations
from typing import TYPE_CHECKING, Callable, Protocol, cast
import torch
from torch import nn
import torch.utils.checkpoint as torch_checkpoint
from typing_extensions import override

from production.kvcache_backend import KVCacheKind
from production.runtime_tuning import KVSelfOptConfig
from .metrics import Metrics
from .runtime import Runtime
from .base import BaseModel
from .block import Block

if TYPE_CHECKING:
    from .config import ModelConfig
    from production.kvcache_backend import DecoupledLayerKVCache, LayerKVCache

class _CacheWithPos(Protocol):
    pos: int

def _checkpoint(fn: Callable[[torch.Tensor], torch.Tensor], x: torch.Tensor) -> torch.Tensor:
    """Typed wrapper around torch's checkpoint (no ignore comments)."""
    cp = cast(Callable[..., torch.Tensor], torch_checkpoint.checkpoint)
    return cp(fn, x, use_reentrant=False)

class GPT(BaseModel):
    """Neural architecture with speculative and incremental support."""
    def __init__(self, cfg: ModelConfig) -> None:
        """
        init the GPT model.
        """
        super().__init__()
        self.cfg: ModelConfig = cfg
        self.tok_emb: nn.Embedding = nn.Embedding(cfg.vocab_size, cfg.embed_dim)
        self.emb_in: nn.Linear | None = (
            nn.Linear(cfg.embed_dim, cfg.d_model, bias=False)
            if cfg.embed_dim != cfg.d_model else None
        )
        self.emb_out: nn.Linear | None = (
            nn.Linear(cfg.d_model, cfg.embed_dim, bias=False)
            if cfg.embed_dim != cfg.d_model else None
        )
        self.drop: nn.Dropout = nn.Dropout(cfg.dropout)
        self.blocks: nn.ModuleList = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f: nn.LayerNorm = nn.LayerNorm(cfg.d_model)

        # Causal mask for prefill
        mask = torch.tril(torch.ones(cfg.block_size, cfg.block_size, dtype=torch.bool))
        self.register_buffer(
            "mask",
            mask.view(1, 1, cfg.block_size, cfg.block_size),
            persistent=False,
        )
        _ = self.apply(self._init_weights)
        self.grad_checkpointing: bool = False

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, (nn.Linear, nn.Embedding)):
            _ = nn.init.normal_(m.weight, mean=0.0, std=0.02)

    @override
    def forward(
        self,
        idx: torch.Tensor,
        *,
        caches: list[DecoupledLayerKVCache | LayerKVCache] | None = None,
        pos_offset: int = 0,
        return_features: bool = False,
    ) -> tuple[torch.Tensor, list[DecoupledLayerKVCache | LayerKVCache] | None]:
        """Forward pass supporting training, prefill, and incremental decode."""
        _, t = idx.shape
        if caches is None and t > self.cfg.block_size:
            raise ValueError(
                f"Sequence length {t} > block_size {self.cfg.block_size}. Increase --block."
            )

        x = cast(torch.Tensor, self.tok_emb(idx))
        if self.emb_in is not None:
            x = cast(torch.Tensor, self.emb_in(x))
        x = cast(torch.Tensor, self.drop(x))

        attn_mask: torch.Tensor | None
        if caches is None:
            if self.cfg.null_attn:
                causal = cast(torch.Tensor, getattr(self, "mask"))
                attn_mask = causal[:, :, :t, :t]
            else:
                attn_mask = None
        else:
            if t > 1:
                prev_len = cast(_CacheWithPos, cast(object, caches[0])).pos if caches else 0
                if prev_len == 0 and (not self.cfg.null_attn):
                    attn_mask = None
                else:
                    seq_len = prev_len + t
                    key_pos = torch.arange(seq_len, device=idx.device).view(1, 1, 1, seq_len)
                    q_pos = (prev_len + torch.arange(t, device=idx.device)).view(1, 1, t, 1)
                    attn_mask = key_pos <= q_pos
            else:
                attn_mask = None

        new_caches: list[DecoupledLayerKVCache | LayerKVCache] | None = [] if caches is not None else None

        if caches is None and self.training and self.grad_checkpointing:
            for m in self.blocks:
                blk = cast(Block, m)

                def _blk_fwd(x_in: torch.Tensor, _blk: Block = blk) -> torch.Tensor:
                    y, _c = cast(
                        tuple[torch.Tensor, object],
                        _blk(x_in, attn_mask=attn_mask, cache=None, pos_offset=pos_offset),
                    )
                    return y

                x = _checkpoint(_blk_fwd, x)
        elif caches is None:
            for m in self.blocks:
                blk = cast(Block, m)
                x, _c = cast(
                    tuple[torch.Tensor, object],
                    blk(x, attn_mask=attn_mask, cache=None, pos_offset=pos_offset),
                )
        else:
            for i, m in enumerate(self.blocks):
                blk = cast(Block, m)
                layer_cache_in = caches[i]
                x, layer_cache_out = cast(
                    tuple[torch.Tensor, DecoupledLayerKVCache | LayerKVCache | None],
                    blk(x, attn_mask=attn_mask, cache=layer_cache_in, pos_offset=pos_offset),
                )
                cast(list[DecoupledLayerKVCache | LayerKVCache], new_caches).append(
                    cast(DecoupledLayerKVCache | LayerKVCache, layer_cache_out)
                )

        x = cast(torch.Tensor, self.ln_f(x))
        x_small = cast(torch.Tensor, self.emb_out(x)) if self.emb_out is not None else x
        if return_features:
            return x_small, new_caches
        logits = x_small @ self.tok_emb.weight.t()
        return logits, new_caches

    def generate_speculative(
        self,
        prompt: torch.Tensor,
        draft: GPT,
        max_new: int,
        *,
        temp: float = 1.0,
        spec_k: int = 4,
        self_opt: KVSelfOptConfig | None = None,
        kv_cache: KVCacheKind = "fp16",
        kv_qblock: int = 32,
        kv_residual: int = 128,
        kv_decode_block: int = 1024,
        kv_fused: str = "auto",
        kv_cache_k: KVCacheKind | None = None,
        kv_cache_v: KVCacheKind | None = None,
        kv_cache_k_sem: KVCacheKind | None = None,
        kv_cache_k_geo: KVCacheKind | None = None,
    ) -> torch.Tensor:
        """Draft proposes, Main verifies. Clean implementation of speculative decode."""
        with torch.no_grad():
            rt = Runtime(self).build_kv(
                prompt,
                max_new,
                self_opt,
                kv_cache=kv_cache,
                kv_qblock=kv_qblock,
                kv_residual=kv_residual,
                kv_decode_block=kv_decode_block,
                kv_fused=kv_fused,
                kv_cache_k=kv_cache_k,
                kv_cache_v=kv_cache_v,
                kv_cache_k_sem=kv_cache_k_sem,
                kv_cache_k_geo=kv_cache_k_geo,
            )
            dr = Runtime(draft).build_kv(
                prompt,
                max_new,
                self_opt,
                kv_cache=kv_cache,
                kv_qblock=kv_qblock,
                kv_residual=kv_residual,
                kv_decode_block=kv_decode_block,
                kv_fused=kv_fused,
                kv_cache_k=kv_cache_k,
                kv_cache_v=kv_cache_v,
                kv_cache_k_sem=kv_cache_k_sem,
                kv_cache_k_geo=kv_cache_k_geo,
            )

        out, pos = prompt, prompt.size(1)
        # Prefill both
        main_logits, _ = self.forward(out, caches=rt.caches, pos_offset=0)
        draft_logits, _ = draft.forward(out, caches=dr.caches, pos_offset=0)

        for _ in range(max_new // (spec_k + 1)):
            proposed: list[torch.Tensor] = []
            q_probs: list[torch.Tensor] = []
            for _ in range(spec_k):
                tok, p = Metrics.sample(draft_logits[:, -1, :], temp)
                proposed.append(tok)
                q_probs.append(p)
                draft_logits, _ = draft.forward(
                    tok,
                    caches=dr.caches,
                    pos_offset=pos + len(proposed) - 1,
                )

            proposed_t = torch.cat(proposed, dim=1)
            main_block, _ = self.forward(
                proposed_t,
                caches=rt.caches,
                pos_offset=pos,
            )

            accepted_k, next_tok = Metrics.verify(
                main_logits[:, -1, :],
                main_block,
                proposed_t,
                q_probs,
            )

            out = torch.cat([out, proposed_t[:, :accepted_k], next_tok], dim=1)
            pos += accepted_k + 1
            main_logits, _ = self.forward(
                next_tok,
                caches=rt.caches,
                pos_offset=pos - 1,
            )
            draft_logits = main_logits

        return out
