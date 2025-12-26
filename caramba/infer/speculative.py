"""Speculative decoding with a draft model for faster inference.

Speculative decoding uses a smaller/faster "draft" model to propose multiple tokens,
then verifies them in parallel with the larger "target" model. When the draft model's
predictions align with the target, we get multiple tokens per forward pass.

References:
- Leviathan et al. "Fast Inference from Transformers via Speculative Decoding" (2023)
- Chen et al. "Accelerating Large Language Model Decoding with Speculative Sampling" (2023)
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
from caramba.infer.generate import (
    create_caches,
    sample_next_token,
    get_attention_configs,
    GenerateConfig,
)
from caramba.layer.attention import AttentionLayer


__all__ = [
    "SpeculativeConfig",
    "SpeculativeGenerator",
    "speculative_generate",
]


@dataclass
class SpeculativeConfig:
    """Configuration for speculative decoding."""
    # Number of tokens to draft per speculation step
    spec_k: int = 4
    # Method: "reject_sampling" or "typical_acceptance"
    spec_method: str = "reject_sampling"
    # Whether to sample an extra token when all drafts are accepted
    spec_extra_token: bool = True
    # Disable speculation when acceptance rate falls below this threshold
    spec_disable_below_accept: float = 0.0
    # Generation parameters (inherited from GenerateConfig)
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


def sampling_probs(
    logits: Tensor,
    *,
    temperature: float,
    top_k: int | None,
    eps: float = 1e-8,
) -> Tensor:
    """Compute sampling probabilities with temperature and top-k."""
    if temperature <= 0:
        # Greedy: one-hot on argmax
        idx = logits.argmax(dim=-1, keepdim=True)
        return torch.zeros_like(logits).scatter_(-1, idx, 1.0)

    logits = logits / temperature

    if top_k is not None and top_k > 0:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < v[:, [-1]]] = float("-inf")

    p = torch.softmax(logits, dim=-1)
    p_sum = p.sum(dim=-1, keepdim=True)
    return p / p_sum.clamp_min(float(eps))


def sample_with_probs(
    logits: Tensor,
    *,
    temperature: float,
    top_k: int | None,
) -> tuple[Tensor, Tensor]:
    """Sample with (temperature, top_k) and return (token, probs)."""
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
    """Parallel verification for speculative decoding (rejection sampling).

    Args:
        main_next: Main model logits for the position before proposed tokens (B, V)
        main_block: Main model logits for proposed positions (B, K, V)
        proposed: Proposed token ids from draft model (B, K)
        q_probs: Draft model probabilities for each proposed token
        temperature: Sampling temperature
        top_k: Top-k filtering
        eps: Small epsilon for numerical stability

    Returns:
        Tuple of (num_accepted, next_token)
        - num_accepted: Number of accepted tokens (0 to K)
        - next_token: The next token to append (B, 1)
    """
    k = proposed.size(1)

    for i in range(k):
        # Get main model distribution for the current step
        if i == 0:
            p = sampling_probs(main_next, temperature=temperature, top_k=top_k, eps=eps)
        else:
            p = sampling_probs(main_block[:, i - 1, :], temperature=temperature, top_k=top_k, eps=eps)

        q = q_probs[i].float()

        # Acceptance probability: p(x)/q(x)
        token = proposed[:, i:i + 1]
        p_tok = p.gather(-1, token)
        q_tok = q.gather(-1, token).clamp_min(float(eps))

        # Rejection check
        if torch.rand_like(p_tok) > (p_tok / q_tok).clamp(max=1.0):
            # Rejected: sample from normalized difference: norm(max(0, p - q))
            diff = (p - q).clamp(min=0)
            diff_sum = diff.sum(dim=-1, keepdim=True)
            ok = torch.isfinite(diff_sum) & (diff_sum > float(eps))
            if not bool(ok.all()):
                next_tok = torch.multinomial(p, 1)
            else:
                next_tok = torch.multinomial(diff / diff_sum, 1)
            return i, next_tok

    # All accepted: sample the next token from the final main distribution
    p_final = sampling_probs(main_block[:, -1, :], temperature=temperature, top_k=top_k, eps=eps)
    if temperature <= 0:
        return k, p_final.argmax(dim=-1, keepdim=True)
    return k, torch.multinomial(p_final, 1)


class SpeculativeGenerator:
    """Stateful speculative generator with draft and target model caches."""

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
        """Initialize the speculative generator.

        Args:
            target_model: The main/target transformer model (larger, more accurate)
            draft_model: The draft transformer model (smaller, faster)
            config: Speculative decoding configuration
            target_lm_head: Optional LM head for target model
            draft_lm_head: Optional LM head for draft model
            device: Device for cache allocation
        """
        self.target_model = target_model
        self.draft_model = draft_model
        self.config = config or SpeculativeConfig()
        self.target_lm_head = target_lm_head
        self.draft_lm_head = draft_lm_head
        self.device = device or torch.device("cpu")

        # Caches will be created on first generation
        self._target_caches: list[LayerKVCache | DecoupledLayerKVCache] | None = None
        self._draft_caches: list[LayerKVCache | DecoupledLayerKVCache] | None = None
        self._target_ctx: InferContext | None = None
        self._draft_ctx: InferContext | None = None
        self._pos: int = 0

        # Stats
        self._accept_total: int = 0
        self._propose_total: int = 0

    def reset(self) -> None:
        """Reset the generator state (clear caches and stats)."""
        self._target_caches = None
        self._draft_caches = None
        self._target_ctx = None
        self._draft_ctx = None
        self._pos = 0
        self._accept_total = 0
        self._propose_total = 0

    @property
    def acceptance_rate(self) -> float:
        """Get the current acceptance rate."""
        if self._propose_total == 0:
            return 1.0
        return float(self._accept_total) / float(self._propose_total)

    def _ensure_caches(self, batch_size: int) -> None:
        """Ensure caches are allocated for the given batch size."""
        if self._target_caches is not None:
            return

        self._target_caches = create_caches(
            self.target_model,
            batch_size=batch_size,
            max_seq_len=self.config.max_seq_len,
            device=self.device,
            cache_kind=self.config.cache_kind,
            cache_qblock=self.config.cache_qblock,
            cache_residual_len=self.config.cache_residual_len,
        )
        self._draft_caches = create_caches(
            self.draft_model,
            batch_size=batch_size,
            max_seq_len=self.config.max_seq_len,
            device=self.device,
            cache_kind=self.config.cache_kind,
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
        """Run forward pass and get logits."""
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
        """Run forward pass for a block of tokens and get all logits."""
        ctx.begin(pos_offset=pos_offset)
        hidden = model(tokens, ctx=ctx)  # type: ignore[call-arg]
        ctx.ensure_consumed()

        if lm_head is not None:
            return lm_head(hidden)
        return hidden

    def _rollback(self, caches: list[LayerKVCache | DecoupledLayerKVCache], new_pos: int) -> None:
        """Rollback caches to a previous position."""
        for cache in caches:
            cache.truncate(new_pos)

    @torch.inference_mode()
    def generate(self, input_ids: Tensor) -> Tensor:
        """Generate tokens using speculative decoding.

        Args:
            input_ids: Initial token ids (B, T)

        Returns:
            Generated token ids (B, T + max_new_tokens)
        """
        self.reset()
        batch_size, seq_len = input_ids.shape
        self._ensure_caches(batch_size)

        assert self._target_ctx is not None
        assert self._draft_ctx is not None
        assert self._target_caches is not None
        assert self._draft_caches is not None

        cfg = self.config

        # Pre-allocate buffer for generated tokens
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

        # Get initial logits
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

            # Check if we should disable speculation due to low acceptance
            use_speculation = True
            if cfg.spec_disable_below_accept > 0 and self._propose_total > 10:
                if self.acceptance_rate < cfg.spec_disable_below_accept:
                    use_speculation = False

            if not use_speculation or remaining <= 1:
                # Fall back to standard decoding
                next_token = sample_next_token(
                    target_logits,
                    temperature=cfg.temperature,
                    top_k=cfg.top_k,
                    top_p=cfg.top_p,
                )
                generated[:, gen_len] = next_token
                gen_len += 1

                if cfg.eos_token_id is not None and (next_token == cfg.eos_token_id).all():
                    break

                # Update both models with the new token
                target_logits = self._get_logits(
                    self.target_model, next_token.unsqueeze(-1),
                    self._target_ctx, self.target_lm_head, self._pos,
                )
                draft_logits = self._get_logits(
                    self.draft_model, next_token.unsqueeze(-1),
                    self._draft_ctx, self.draft_lm_head, self._pos,
                )
                self._pos += 1
                continue

            # Speculative decoding
            k = min(cfg.spec_k, remaining - 1)
            k = max(1, k)

            # Save cache positions for potential rollback
            target_pos_before = self._target_caches[0].pos if self._target_caches else self._pos
            draft_pos_before = self._draft_caches[0].pos if self._draft_caches else self._pos

            # Draft k tokens
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

                # Update draft model with proposed token
                draft_logits = self._get_logits(
                    self.draft_model, tok,
                    self._draft_ctx, self.draft_lm_head, self._pos + len(proposed) - 1,
                )

            proposed_t = torch.cat(proposed, dim=1)  # (B, k)

            # Verify with target model
            target_block = self._get_logits_block(
                self.target_model, proposed_t,
                self._target_ctx, self.target_lm_head, self._pos,
            )

            # Run verification
            accepted_k, next_tok = verify_speculative(
                target_logits,
                target_block,
                proposed_t,
                q_probs,
                temperature=cfg.temperature,
                top_k=cfg.top_k,
            )

            # Update stats
            self._accept_total += accepted_k
            self._propose_total += k

            # Accept tokens and update generated sequence
            if accepted_k > 0:
                generated[:, gen_len:gen_len + accepted_k] = proposed_t[:, :accepted_k]
                gen_len += accepted_k

            # Add the next token (either from rejection sampling or final acceptance)
            generated[:, gen_len] = next_tok.squeeze(-1)
            gen_len += 1

            # Rollback caches to correct position
            correct_pos = target_pos_before + accepted_k + 1
            self._rollback(self._target_caches, correct_pos)
            self._rollback(self._draft_caches, correct_pos)
            self._pos = correct_pos

            # Re-compute caches for the correct token sequence
            # We need to re-append the accepted tokens + next_tok to rebuild cache state
            if accepted_k < k:
                # Some tokens were rejected, need to rebuild cache
                tokens_to_add = generated[:, target_pos_before:gen_len]

                # Clear and re-run forward for correct sequence
                self._rollback(self._target_caches, target_pos_before)
                self._rollback(self._draft_caches, draft_pos_before)

                target_logits = self._get_logits_block(
                    self.target_model, tokens_to_add,
                    self._target_ctx, self.target_lm_head, target_pos_before,
                )[:, -1, :]
                draft_logits = self._get_logits_block(
                    self.draft_model, tokens_to_add,
                    self._draft_ctx, self.draft_lm_head, draft_pos_before,
                )[:, -1, :]
                self._pos = gen_len
            else:
                # All accepted, use last logits from target
                target_logits = target_block[:, -1, :]
                # Need to get draft logits for next iteration
                draft_logits = self._get_logits(
                    self.draft_model, next_tok,
                    self._draft_ctx, self.draft_lm_head, self._pos - 1,
                )

            # Check for EOS
            if cfg.eos_token_id is not None:
                for i in range(accepted_k + 1):
                    pos = gen_len - accepted_k - 1 + i
                    if pos < gen_len and (generated[:, pos] == cfg.eos_token_id).all():
                        return generated[:, :pos + 1]

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
    """Generate tokens using speculative decoding (stateless API).

    Args:
        target_model: The main/target transformer model
        draft_model: The draft transformer model
        input_ids: Initial token ids (B, T)
        config: Speculative decoding configuration
        target_lm_head: Optional LM head for target model
        draft_lm_head: Optional LM head for draft model
        log_callback: Optional callback for logging stats

    Returns:
        Generated token ids (B, T + max_new_tokens)
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
        log_callback({
            "accept_total": generator._accept_total,
            "propose_total": generator._propose_total,
            "acceptance_rate": generator.acceptance_rate,
        })

    return result
