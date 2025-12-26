"""Short-context fidelity checks for teacher/student model pairs.

We want fast, quantitative quality gates that catch regressions the moment we
change architecture, kernels, or cache policies. Comparing internal activations
is useful, but the most direct signal is language-modeling loss on identical
inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass
class FidelityResult:
    """Aggregated loss-based fidelity metrics for teacher vs student.

    Why this exists:
    - Delta NLL is a stable, unitless per-token measure of quality drift.
    - PPL ratio is a convenient exponential view of the same drift.
    """

    teacher_nll: float
    student_nll: float
    delta_nll: float
    ppl_ratio: float
    batches: int
    tokens: int


@dataclass
class FidelityViolation:
    """Record of a fidelity threshold being exceeded.

    Why this exists:
    - We often want to collect all violations (fail_fast=False) and report them
      together, instead of stopping at the first failure.
    """

    metric: str
    value: float
    threshold: float

    def message(self) -> str:
        """Format a human-readable error message."""
        return (
            f"{self.metric} exceeded threshold: "
            f"value={self.value:.6f}, threshold={self.threshold:.6f}"
        )


def compute_short_context_fidelity(
    *,
    teacher: nn.Module,
    student: nn.Module,
    batches: list[tuple[Tensor, Tensor]],
) -> FidelityResult:
    """Compute teacher/student delta NLL and PPL ratio on fixed batches.

    Why this exists:
    - It is cheap compared to full benchmarking and correlates strongly with
      "did we preserve model quality?"
    - It is model-agnostic: anything that maps token IDs -> logits works.
    """

    if not batches:
        raise ValueError("batches must be non-empty")

    teacher.eval()
    student.eval()

    teacher_sum = 0.0
    student_sum = 0.0
    total_tokens = 0

    with torch.no_grad():
        for x, y in batches:
            if x.ndim != 2 or y.ndim != 2:
                raise ValueError("Expected x and y to be rank-2 (batch, seq)")
            if x.shape != y.shape:
                raise ValueError(f"x/y shape mismatch: {x.shape} vs {y.shape}")

            t_logits = teacher(x)
            s_logits = student(x)

            if not isinstance(t_logits, Tensor) or not isinstance(s_logits, Tensor):
                raise TypeError("teacher/student must return a Tensor of logits")
            if t_logits.shape != s_logits.shape:
                raise ValueError(
                    f"teacher/student logits shape mismatch: {t_logits.shape} vs {s_logits.shape}"
                )
            if t_logits.ndim != 3:
                raise ValueError(f"Expected logits rank-3 (B, T, V), got {t_logits.ndim}")
            if t_logits.shape[:2] != y.shape:
                raise ValueError(
                    f"logits/y shape mismatch: logits={t_logits.shape}, y={y.shape}"
                )

            vocab = int(t_logits.shape[-1])
            tok = int(y.numel())
            total_tokens += tok

            t_loss = F.cross_entropy(
                t_logits.reshape(-1, vocab).float(),
                y.reshape(-1),
                reduction="mean",
            )
            s_loss = F.cross_entropy(
                s_logits.reshape(-1, vocab).float(),
                y.reshape(-1),
                reduction="mean",
            )

            teacher_sum += float(t_loss) * float(tok)
            student_sum += float(s_loss) * float(tok)

    if total_tokens <= 0:
        raise RuntimeError("No tokens were evaluated.")

    teacher_nll = teacher_sum / float(total_tokens)
    student_nll = student_sum / float(total_tokens)
    delta_nll = student_nll - teacher_nll
    # Guard against overflow in math.exp by clamping delta_nll to a safe range.
    clamped_delta = max(-700.0, min(700.0, delta_nll))
    ppl_ratio = math.exp(clamped_delta)

    return FidelityResult(
        teacher_nll=float(teacher_nll),
        student_nll=float(student_nll),
        delta_nll=float(delta_nll),
        ppl_ratio=float(ppl_ratio),
        batches=int(len(batches)),
        tokens=int(total_tokens),
    )


def check_fidelity_thresholds(
    *,
    result: FidelityResult,
    max_delta_nll: float | None,
    max_ppl_ratio: float | None,
) -> list[FidelityViolation]:
    """Check fidelity metrics against thresholds, returning violations.

    Why this exists:
    - The verify pipeline needs both a strict (fail-fast) and permissive mode.
    """

    violations: list[FidelityViolation] = []

    if max_delta_nll is not None and result.delta_nll > float(max_delta_nll):
        violations.append(
            FidelityViolation(
                metric="delta_nll",
                value=float(result.delta_nll),
                threshold=float(max_delta_nll),
            )
        )

    if max_ppl_ratio is not None and result.ppl_ratio > float(max_ppl_ratio):
        violations.append(
            FidelityViolation(
                metric="ppl_ratio",
                value=float(result.ppl_ratio),
                threshold=float(max_ppl_ratio),
            )
        )

    return violations


def assert_fidelity_thresholds(
    *,
    result: FidelityResult,
    max_delta_nll: float | None,
    max_ppl_ratio: float | None,
    fail_fast: bool = True,
) -> list[FidelityViolation]:
    """Check thresholds and optionally raise on the first violation."""

    violations = check_fidelity_thresholds(
        result=result,
        max_delta_nll=max_delta_nll,
        max_ppl_ratio=max_ppl_ratio,
    )

    if violations and fail_fast:
        raise ValueError(f"fidelity failed: {violations[0].message()}")

    return violations

