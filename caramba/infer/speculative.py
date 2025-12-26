"""Speculative decoding for faster inference.

Standard autoregressive generation is memory-bound: we process one token
at a time, mostly waiting on memory. Speculative decoding uses a smaller
draft model to propose K tokens, then verifies them in parallel with the
target model. When predictions align, we get multiple tokens per forward.

References:
- Leviathan et al. "Fast Inference from Transformers via Speculative Decoding"
- Chen et al. "Accelerating LLM Decoding with Speculative Sampling"
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from torch import Tensor, nn

from caramba.cache.decoupled import DecoupledLayerKVCache
from caramba.cache.layer import LayerKVCache
from caramba.config.kvcache import KVCacheKind
from caramba.infer.cache_policy import choose_cache_kind
from caramba.infer.context import InferContext
from caramba.infer.generate import (
    GenerateConfig,
    create_caches,
    get_attention_configs,
    sample_next_token,
)

__all__ = [
    "SpeculativeConfig",
    "SpeculativeGenerator",
    "speculative_generate",
]


@dataclass
class SpeculativeConfig:
    """Configuration for speculative decoding.

    Key parameter is spec_k: how many tokens to draft before verification.
    Higher K means more potential tokens per forward, but lower acceptance
    rate if draft quality is poor.
    """

    spec_k: int = 4
    spec_method: str = "reject_sampling"
    spec_extra_token: bool = True
    spec_disable_below_accept: float = 0.0
    spec_k_adaptive: bool = False
    spec_k_min: int = 1
    spec_k_max: int = 16
    spec_k_target_accept: float = 0.7
    spec_k_adjust_interval: int = 32
    spec_k_step: int = 1
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


def sampling_probs(
    logits: Tensor,
    *,
    temperature: float,
    top_k: int | None,
    eps: float = 1e-8,
) -> Tensor:
    """Compute sampling probabilities with temperature and top-k.

    Creates a copy to avoid mutating input logits.
    """
    if temperature <= 0:
        idx = logits.argmax(dim=-1, keepdim=True)
        return torch.zeros_like(logits).scatter_(-1, idx, 1.0)

    scaled_logits = logits / temperature

    if top_k is not None and top_k > 0:
        v, _ = torch.topk(scaled_logits, min(top_k, scaled_logits.size(-1)))
        mask = scaled_logits < v[:, [-1]]
        scaled_logits = scaled_logits.masked_fill(mask, float("-inf"))

    p = torch.softmax(scaled_logits, dim=-1)
    p_sum = p.sum(dim=-1, keepdim=True)
    return p / p_sum.clamp_min(float(eps))


def sample_with_probs(
    logits: Tensor,
    *,
    temperature: float,
    top_k: int | None,
) -> tuple[Tensor, Tensor]:
    """Sample a token and return (token, probabilities)."""
    p = sampling_probs(logits, temperature=float(temperature), top_k=top_k)
    if temperature <= 0:
        return p.argmax(dim=-1, keepdim=True), p
    return torch.multinomial(p, 1), p


def verify_speculative(
    main_next: Tensor,
    main_block: Tensor,
    proposed: Tensor,
    q_probs: list[Tensor],
    *,
    temperature: float,
    top_k: int | None,
    eps: float = 1e-8,
) -> tuple[int, Tensor]:
    """Verify draft tokens using rejection sampling.

    For each proposed token, compute acceptance probability p(x)/q(x).
    If rejected, sample from the residual distribution (p - q)+.
    Returns (num_accepted, next_token).
    """
    B = proposed.size(0)
    k = proposed.size(1)
    device = proposed.device

    accepted_mask = torch.ones(B, dtype=torch.bool, device=device)
    reject_pos = torch.full((B,), k, dtype=torch.long, device=device)
    next_tokens = torch.zeros((B, 1), dtype=torch.long, device=device)

    for i in range(k):
        if i == 0:
            p = sampling_probs(main_next, temperature=temperature, top_k=top_k, eps=eps)
        else:
            p = sampling_probs(
                main_block[:, i - 1, :], temperature=temperature, top_k=top_k, eps=eps
            )

        q = q_probs[i].float()

        token = proposed[:, i : i + 1]
        p_tok = p.gather(-1, token).squeeze(-1)
        q_tok = q.gather(-1, token).squeeze(-1).clamp_min(float(eps))

        accept_ratio = (p_tok / q_tok).clamp(max=1.0)
        rand_vals = torch.rand_like(accept_ratio)
        rejected_now = (rand_vals > accept_ratio) & accepted_mask

        if rejected_now.any():
            diff = (p - q).clamp(min=0)
            diff_sum = diff.sum(dim=-1, keepdim=True)
            valid_diff = torch.isfinite(diff_sum.squeeze(-1)) & (
                diff_sum.squeeze(-1) > float(eps)
            )

            for b in range(B):
                if rejected_now[b]:
                    reject_pos[b] = i
                    accepted_mask[b] = False
                    if valid_diff[b]:
                        next_tokens[b] = torch.multinomial(
                            diff[b : b + 1] / diff_sum[b : b + 1], 1
                        )
                    else:
                        next_tokens[b] = torch.multinomial(p[b : b + 1], 1)

    all_accepted_mask = accepted_mask
    if all_accepted_mask.any():
        p_final = sampling_probs(
            main_block[:, -1, :], temperature=temperature, top_k=top_k, eps=eps
        )
        if temperature <= 0:
            final_tokens = p_final.argmax(dim=-1, keepdim=True)
        else:
            final_tokens = torch.multinomial(p_final, 1)
        next_tokens[all_accepted_mask] = final_tokens[all_accepted_mask]

    num_accepted = int(reject_pos.min().item())
    return num_accepted, next_tokens


class SpeculativeGenerator:
    """Stateful speculative generator with draft and target model caches.

    Maintains separate KV-caches for both models, synced to the same
    position. When speculation fails, rolls back both caches.
    """

    def __init__(
        self,
        target_model: nn.Module,
        draft_model: nn.Module,
        *,
        config: SpeculativeConfig | None = None,
        target_lm_head: nn.Module | None = None,
        draft_lm_head: nn.Module | None = None,
        device: torch.device | None = None,
    ) -> None:
        """Set up with target (large) and draft (small) models."""
        self.target_model = target_model
        self.draft_model = draft_model
        self.config = config or SpeculativeConfig()
        self.target_lm_head = target_lm_head
        self.draft_lm_head = draft_lm_head
        self.device = device or torch.device("cpu")

        self._target_caches: list[LayerKVCache | DecoupledLayerKVCache] | None = None
        self._draft_caches: list[LayerKVCache | DecoupledLayerKVCache] | None = None
        self._target_ctx: InferContext | None = None
        self._draft_ctx: InferContext | None = None
        self._pos: int = 0

        self._accept_total: int = 0
        self._propose_total: int = 0
        self._spec_k_current: int = int(self.config.spec_k)
        self._last_k_adjust_at: int = 0

    def reset(self) -> None:
        """Clear caches and stats."""
        self._target_caches = None
        self._draft_caches = None
        self._target_ctx = None
        self._draft_ctx = None
        self._pos = 0
        self._accept_total = 0
        self._propose_total = 0
        self._spec_k_current = int(self.config.spec_k)
        self._last_k_adjust_at = 0

    def _maybe_adjust_spec_k(self) -> None:
        """Adapt spec_k based on observed acceptance rate.

        Why this exists:
        - Higher spec_k increases throughput when acceptance is high.
        - Lower spec_k avoids wasted work when the draft model is diverging.
        """

        cfg = self.config
        if not cfg.spec_k_adaptive:
            return
        interval = max(1, int(cfg.spec_k_adjust_interval))
        if self._propose_total < interval:
            return
        if (self._propose_total - self._last_k_adjust_at) < interval:
            return

        target = float(cfg.spec_k_target_accept)
        step = max(1, int(cfg.spec_k_step))
        k_min = max(1, int(cfg.spec_k_min))
        k_max = max(k_min, int(cfg.spec_k_max))
        r = float(self.acceptance_rate)

        # Deadband to avoid oscillation.
        hi = min(0.99, target + 0.1)
        lo = max(0.0, target - 0.1)

        k = int(self._spec_k_current)
        if r >= hi:
            k = min(k_max, k + step)
        elif r <= lo:
            k = max(k_min, k - step)

        self._spec_k_current = int(k)
        self._last_k_adjust_at = int(self._propose_total)

    @property
    def acceptance_rate(self) -> float:
        """Current acceptance rate (accepted / proposed)."""
        if self._propose_total == 0:
            return 1.0
        return float(self._accept_total) / float(self._propose_total)

    def _ensure_caches(self, batch_size: int) -> None:
        """Allocate caches for both models."""
        if self._target_caches is not None:
            return

        ck = self.config.cache_kind
        if isinstance(ck, str) and ck.strip().lower() == "auto":
            choice = choose_cache_kind(
                model=self.target_model,
                batch_size=int(batch_size),
                max_seq_len=int(self.config.max_seq_len),
                qblock=int(self.config.cache_qblock),
                residual_len=int(self.config.cache_residual_len),
                budget_mb=self.config.cache_budget_mb,
            )
            cache_kind = choice.kind
        else:
            cache_kind = ck if isinstance(ck, KVCacheKind) else KVCacheKind.FP16

        self._target_caches = create_caches(
            self.target_model,
            batch_size=batch_size,
            max_seq_len=self.config.max_seq_len,
            device=self.device,
            cache_kind=cache_kind,
            cache_qblock=self.config.cache_qblock,
            cache_residual_len=self.config.cache_residual_len,
        )
        self._draft_caches = create_caches(
            self.draft_model,
            batch_size=batch_size,
            max_seq_len=self.config.max_seq_len,
            device=self.device,
            cache_kind=cache_kind,
            cache_qblock=self.config.cache_qblock,
            cache_residual_len=self.config.cache_residual_len,
        )
        self._target_ctx = InferContext(caches=self._target_caches)
        self._draft_ctx = InferContext(caches=self._draft_caches)
        self._pos = 0

    def _get_logits(
        self,
        model: nn.Module,
        tokens: Tensor,
        ctx: InferContext,
        lm_head: nn.Module | None,
        pos_offset: int,
    ) -> Tensor:
        """Single token forward pass."""
        ctx.begin(pos_offset=pos_offset)
        hidden = model(tokens, ctx=ctx)  # type: ignore[call-arg]
        ctx.ensure_consumed()

        if lm_head is not None:
            return lm_head(hidden[:, -1:, :])[:, 0, :]
        return hidden[:, -1, :]

    def _get_logits_block(
        self,
        model: nn.Module,
        tokens: Tensor,
        ctx: InferContext,
        lm_head: nn.Module | None,
        pos_offset: int,
    ) -> Tensor:
        """Multi-token forward pass, returns logits for all positions."""
        ctx.begin(pos_offset=pos_offset)
        hidden = model(tokens, ctx=ctx)  # type: ignore[call-arg]
        ctx.ensure_consumed()

        if lm_head is not None:
            return lm_head(hidden)
        return hidden

    def _rollback(
        self, caches: list[LayerKVCache | DecoupledLayerKVCache], new_pos: int
    ) -> None:
        """Truncate all caches to a previous position."""
        for cache in caches:
            cache.truncate(new_pos)

    @torch.inference_mode()
    def generate(self, input_ids: Tensor) -> Tensor:
        """Generate using speculative decoding.

        Alternates between:
        1. Draft K tokens with the draft model
        2. Verify in parallel with the target model
        3. Accept/reject and update caches
        """
        self.reset()
        batch_size, seq_len = input_ids.shape
        self._ensure_caches(batch_size)

        assert self._target_ctx is not None
        assert self._draft_ctx is not None
        assert self._target_caches is not None
        assert self._draft_caches is not None

        cfg = self.config

        max_gen_len = seq_len + cfg.max_new_tokens
        generated = torch.empty(
            (batch_size, max_gen_len),
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        generated[:, :seq_len] = input_ids
        gen_len = seq_len

        # Prefill both models
        self._target_ctx.begin(pos_offset=0)
        target_hidden = self.target_model(input_ids, ctx=self._target_ctx)  # type: ignore[call-arg]
        self._target_ctx.ensure_consumed()

        self._draft_ctx.begin(pos_offset=0)
        draft_hidden = self.draft_model(input_ids, ctx=self._draft_ctx)  # type: ignore[call-arg]
        self._draft_ctx.ensure_consumed()

        self._pos = seq_len

        if self.target_lm_head is not None:
            target_logits = self.target_lm_head(target_hidden[:, -1:, :])[:, 0, :]
        else:
            target_logits = target_hidden[:, -1, :]

        if self.draft_lm_head is not None:
            draft_logits = self.draft_lm_head(draft_hidden[:, -1:, :])[:, 0, :]
        else:
            draft_logits = draft_hidden[:, -1, :]

        while gen_len < seq_len + cfg.max_new_tokens:
            remaining = seq_len + cfg.max_new_tokens - gen_len

            use_speculation = True
            if cfg.spec_disable_below_accept > 0 and self._propose_total > 10:
                if self.acceptance_rate < cfg.spec_disable_below_accept:
                    use_speculation = False

            if not use_speculation or remaining <= 1:
                next_token = sample_next_token(
                    target_logits,
                    temperature=cfg.temperature,
                    top_k=cfg.top_k,
                    top_p=cfg.top_p,
                )
                generated[:, gen_len] = next_token
                gen_len += 1

                if (
                    cfg.eos_token_id is not None
                    and (next_token == cfg.eos_token_id).all()
                ):
                    break

                target_logits = self._get_logits(
                    self.target_model,
                    next_token.unsqueeze(-1),
                    self._target_ctx,
                    self.target_lm_head,
                    self._pos,
                )
                draft_logits = self._get_logits(
                    self.draft_model,
                    next_token.unsqueeze(-1),
                    self._draft_ctx,
                    self.draft_lm_head,
                    self._pos,
                )
                self._pos += 1
                continue

            k = min(int(self._spec_k_current), remaining - 1)
            k = max(1, k)

            target_pos_before = (
                self._target_caches[0].pos if self._target_caches else self._pos
            )
            draft_pos_before = (
                self._draft_caches[0].pos if self._draft_caches else self._pos
            )

            proposed: list[Tensor] = []
            q_probs: list[Tensor] = []

            for _ in range(k):
                tok, p = sample_with_probs(
                    draft_logits,
                    temperature=cfg.temperature,
                    top_k=cfg.top_k,
                )
                proposed.append(tok)
                q_probs.append(p)

                draft_logits = self._get_logits(
                    self.draft_model,
                    tok,
                    self._draft_ctx,
                    self.draft_lm_head,
                    self._pos + len(proposed) - 1,
                )

            proposed_t = torch.cat(proposed, dim=1)

            target_block = self._get_logits_block(
                self.target_model,
                proposed_t,
                self._target_ctx,
                self.target_lm_head,
                self._pos,
            )

            accepted_k, next_tok = verify_speculative(
                target_logits,
                target_block,
                proposed_t,
                q_probs,
                temperature=cfg.temperature,
                top_k=cfg.top_k,
            )

            self._accept_total += accepted_k
            self._propose_total += k
            self._maybe_adjust_spec_k()

            if accepted_k > 0:
                generated[:, gen_len : gen_len + accepted_k] = proposed_t[:, :accepted_k]
                gen_len += accepted_k

            generated[:, gen_len] = next_tok.squeeze(-1)
            gen_len += 1

            correct_pos = target_pos_before + accepted_k + 1
            self._rollback(self._target_caches, correct_pos)
            self._rollback(self._draft_caches, correct_pos)
            self._pos = correct_pos

            if accepted_k < k:
                tokens_to_add = generated[:, target_pos_before:gen_len]

                self._rollback(self._target_caches, target_pos_before)
                self._rollback(self._draft_caches, draft_pos_before)

                target_logits = self._get_logits_block(
                    self.target_model,
                    tokens_to_add,
                    self._target_ctx,
                    self.target_lm_head,
                    target_pos_before,
                )[:, -1, :]
                draft_logits = self._get_logits_block(
                    self.draft_model,
                    tokens_to_add,
                    self._draft_ctx,
                    self.draft_lm_head,
                    draft_pos_before,
                )[:, -1, :]
                self._pos = gen_len
            else:
                target_logits = target_block[:, -1, :]
                draft_logits = self._get_logits(
                    self.draft_model,
                    next_tok,
                    self._draft_ctx,
                    self.draft_lm_head,
                    self._pos - 1,
                )

            if cfg.eos_token_id is not None:
                for i in range(accepted_k + 1):
                    pos = gen_len - accepted_k - 1 + i
                    if pos < gen_len and (generated[:, pos] == cfg.eos_token_id).all():
                        return generated[:, : pos + 1]

        return generated[:, :gen_len]


@torch.inference_mode()
def speculative_generate(
    target_model: nn.Module,
    draft_model: nn.Module,
    input_ids: Tensor,
    *,
    config: SpeculativeConfig | None = None,
    target_lm_head: nn.Module | None = None,
    draft_lm_head: nn.Module | None = None,
    log_callback: Callable[[dict[str, object]], None] | None = None,
) -> Tensor:
    """Stateless API for speculative generation.

    Creates a fresh SpeculativeGenerator, runs generation, and optionally
    reports stats via log_callback.
    """
    generator = SpeculativeGenerator(
        target_model,
        draft_model,
        config=config,
        target_lm_head=target_lm_head,
        draft_lm_head=draft_lm_head,
        device=input_ids.device,
    )

    result = generator.generate(input_ids)

    if log_callback is not None:
        log_callback(
            {
                "accept_total": generator._accept_total,
                "propose_total": generator._propose_total,
                "acceptance_rate": generator.acceptance_rate,
            }
        )

    return result
