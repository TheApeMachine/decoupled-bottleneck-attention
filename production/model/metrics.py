"""
metrics provides quality math and sampling for generation.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F

class Metrics:
    """Judge and sampler for model outputs."""
    @staticmethod
    def sample(logits: torch.Tensor, temp: float) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample with temperature and return (token, probs)."""
        p = F.softmax(logits / max(1e-8, temp), dim=-1)
        return torch.multinomial(p, 1), p

    @staticmethod
    def verify(
        main_next: torch.Tensor,
        main_block: torch.Tensor,
        proposed: torch.Tensor,
        q_probs: list[torch.Tensor]
    ) -> tuple[int, torch.Tensor]:
        """Parallel verification for speculative decoding (rejection sampling)."""
        k = proposed.size(1)
        for i in range(k):
            # 1. Get main model distribution for the current step
            p = F.softmax((main_next if i == 0 else main_block[:, i-1, :]), dim=-1)
            q = q_probs[i]

            # 2. Acceptance probability: p(x)/q(x)
            token = proposed[:, i : i+1]
            p_tok = p.gather(-1, token)
            q_tok = q.gather(-1, token)

            # 3. Rejection check
            if torch.rand_like(p_tok) > (p_tok / q_tok).clamp(max=1.0):
                # Sample from normalized difference: norm(max(0, p - q))
                diff = (p - q).clamp(min=0)
                next_tok = torch.multinomial(diff / diff.sum(-1, keepdim=True), 1)
                return i, next_tok

        # 4. All accepted: Sample the next token from the final main distribution
        p_final = F.softmax(main_block[:, -1, :], dim=-1)
        return k, torch.multinomial(p_final, 1)

    @staticmethod
    def compare(lb: torch.Tensor, lt: torch.Tensor, tgt: torch.Tensor) -> dict[str, float]:
        """Compare base vs test logits for quality gating."""
        lb, lt = lb[:, -1, :].float(), lt[:, -1, :].float()
        dnll = (F.cross_entropy(lt, tgt) - F.cross_entropy(lb, tgt)).item()
        return {"max_abs_logit": (lt - lb).abs().max().item(), "delta_nll": dnll}
