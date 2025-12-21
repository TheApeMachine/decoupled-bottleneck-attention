"""Quality gates for cache-policy tuning."""

from __future__ import annotations

import math


def policy_quality_reject_reasons(
    metrics: dict[str, float],
    *,
    max_abs_logit_tol: float | None,
    delta_nll_tol: float | None,
    ppl_ratio_tol: float | None,
    kl_tol: float | None,
) -> list[str]:
    """Convert quality metrics into human-readable reject reasons."""
    out: list[str] = []
    if max_abs_logit_tol is not None and "max_abs_logit" in metrics:
        mx = float(metrics["max_abs_logit"])
        if math.isnan(mx):
            out.append("max_abs_logit=nan")
        elif mx > float(max_abs_logit_tol):
            out.append(f"max_abs_logit={mx:.4g} > {float(max_abs_logit_tol):.4g}")
    if delta_nll_tol is not None and "delta_nll" in metrics:
        dnll = float(metrics["delta_nll"])
        if math.isnan(dnll):
            out.append("ΔNLL=nan")
        elif dnll > float(delta_nll_tol):
            out.append(f"ΔNLL={dnll:.4g} > {float(delta_nll_tol):.4g} nats/tok")
    if ppl_ratio_tol is not None and "ppl_ratio" in metrics:
        pr = float(metrics["ppl_ratio"])
        if math.isnan(pr):
            out.append("ppl_ratio=nan")
        elif pr > float(ppl_ratio_tol):
            out.append(f"ppl_ratio={pr:.4g} > {float(ppl_ratio_tol):.4g}")
    if kl_tol is not None and "kl_base_cand" in metrics:
        klv = float(metrics["kl_base_cand"])
        if math.isnan(klv):
            out.append("KL=nan")
        elif klv > float(kl_tol):
            out.append(f"KL={klv:.4g} > {float(kl_tol):.4g} nats/tok")
    return out


def warn_policy_quality_reject(*, chosen: str, fallback: str, reasons: list[str]) -> None:
    """Non-verbose warning when a candidate policy is rejected by quality guardrails."""
    if not reasons:
        return
    msg = "; ".join(reasons)
    print(f"[warn] selfopt cache-policy rejected: {chosen} -> fallback {fallback} ({msg})")


