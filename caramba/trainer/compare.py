"""Teacher/student comparison for verifying distillation quality.

After training, we need to verify that the student model produces outputs
similar to the teacher. This module runs both models on the same inputs
and measures how much their outputs diverge, helping catch training failures
before expensive benchmarking.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from caramba.config.verify import CompareThreshold
from caramba.model.trace import Trace


@dataclass
class CompareResult:
    """Aggregated metrics from comparing teacher and student models.

    We track both mean and max L1 error because they catch different problems:
    mean catches overall drift, max catches catastrophic single-layer failures.
    """

    attention_mean_l1: float | None
    attention_max_l1: float | None
    logits_mean_l1: float | None
    logits_max_l1: float | None
    batches: int

    def __post_init__(self) -> None:
        """Ensure batches is stored as int, not float."""
        self.batches = int(self.batches)


def compare_teacher_student(
    *,
    teacher: nn.Module,
    student: nn.Module,
    batches: list[Tensor],
    predicate: Callable[[str, nn.Module], bool],
    attention: CompareThreshold | None,
    logits: CompareThreshold | None,
) -> CompareResult:
    """Run both models on identical inputs and measure output divergence.

    This is the core verification function. It traces intermediate outputs
    from both models (using the predicate to select which layers to compare)
    and computes L1 error metrics.

    Args:
        teacher: The frozen teacher model (ground truth)
        student: The trained student model to verify
        batches: List of input tensors to test on
        predicate: Function that identifies which layers to trace
        attention: If set, compare traced layer outputs
        logits: If set, compare final model outputs

    Returns:
        CompareResult with mean and max L1 errors
    """
    if not batches:
        raise ValueError("batches must be non-empty")
    if attention is None and logits is None:
        raise ValueError("Expected at least one metric: attention or logits")

    teacher.eval()
    student.eval()

    attn_sum = 0.0
    attn_count = 0
    attn_max = 0.0

    logits_sum = 0.0
    logits_count = 0
    logits_max = 0.0

    t_trace = Trace(teacher, predicate=predicate)
    s_trace = Trace(student, predicate=predicate)

    with torch.no_grad():
        for x in batches:
            t_trace.clear()
            s_trace.clear()

            with t_trace:
                t_logits = teacher(x)
            with s_trace:
                s_logits = student(x)

            # Compare traced layer outputs (typically attention outputs)
            if attention is not None:
                if len(t_trace.outputs) != len(s_trace.outputs):
                    raise ValueError(
                        f"Teacher/student trace outputs mismatch: "
                        f"{len(t_trace.outputs)} vs {len(s_trace.outputs)}"
                    )
                for t_out, s_out in zip(t_trace.outputs, s_trace.outputs):
                    if t_out.shape != s_out.shape:
                        raise ValueError(
                            f"Teacher/student output shapes mismatch: "
                            f"{t_out.shape} vs {s_out.shape}"
                        )
                    err = F.l1_loss(s_out, t_out, reduction="mean")
                    v = float(err)
                    attn_sum += v
                    attn_count += 1
                    if v > attn_max:
                        attn_max = v

            # Compare final model outputs (logits)
            if logits is not None:
                if t_logits.shape != s_logits.shape:
                    raise ValueError(
                        f"Teacher/student logits shapes mismatch: "
                        f"{t_logits.shape} vs {s_logits.shape}"
                    )
                err = F.l1_loss(s_logits, t_logits, reduction="mean")
                v = float(err)
                logits_sum += v
                logits_count += 1
                if v > logits_max:
                    logits_max = v

    # Compute averages
    attn_mean = None
    if attention is not None:
        if attn_count <= 0:
            raise RuntimeError("No attention comparisons were computed.")
        attn_mean = attn_sum / float(attn_count)

    logits_mean = None
    if logits is not None:
        if logits_count <= 0:
            raise RuntimeError("No logits comparisons were computed.")
        logits_mean = logits_sum / float(logits_count)

    return CompareResult(
        attention_mean_l1=attn_mean if attention is not None else None,
        attention_max_l1=attn_max if attention is not None else None,
        logits_mean_l1=logits_mean if logits is not None else None,
        logits_max_l1=logits_max if logits is not None else None,
        batches=len(batches),
    )


@dataclass
class ThresholdViolation:
    """Record of a single threshold being exceeded.

    Used to collect all violations when fail_fast=False, so we can
    report everything that went wrong instead of just the first failure.
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


def check_thresholds(
    *,
    result: CompareResult,
    attention: CompareThreshold | None,
    logits: CompareThreshold | None,
) -> list[ThresholdViolation]:
    """Check if comparison results exceed configured thresholds.

    This is the non-raising version—it returns a list of violations
    rather than throwing on the first failure.

    Args:
        result: Metrics from compare_teacher_student
        attention: Max allowed attention divergence
        logits: Max allowed logits divergence

    Returns:
        List of violations (empty if all thresholds pass)
    """
    violations: list[ThresholdViolation] = []

    if attention is not None:
        if result.attention_mean_l1 is None or result.attention_max_l1 is None:
            raise RuntimeError("Missing attention results for threshold check.")
        if result.attention_mean_l1 > float(attention.max_mean_l1):
            violations.append(ThresholdViolation(
                metric="attention_mean_l1",
                value=result.attention_mean_l1,
                threshold=float(attention.max_mean_l1),
            ))
        if result.attention_max_l1 > float(attention.max_max_l1):
            violations.append(ThresholdViolation(
                metric="attention_max_l1",
                value=result.attention_max_l1,
                threshold=float(attention.max_max_l1),
            ))

    if logits is not None:
        if result.logits_mean_l1 is None or result.logits_max_l1 is None:
            raise RuntimeError("Missing logits results for threshold check.")
        if result.logits_mean_l1 > float(logits.max_mean_l1):
            violations.append(ThresholdViolation(
                metric="logits_mean_l1",
                value=result.logits_mean_l1,
                threshold=float(logits.max_mean_l1),
            ))
        if result.logits_max_l1 > float(logits.max_max_l1):
            violations.append(ThresholdViolation(
                metric="logits_max_l1",
                value=result.logits_max_l1,
                threshold=float(logits.max_max_l1),
            ))

    return violations


def assert_thresholds(
    *,
    result: CompareResult,
    attention: CompareThreshold | None,
    logits: CompareThreshold | None,
    fail_fast: bool = True,
) -> list[ThresholdViolation]:
    """Check thresholds and optionally raise on violations.

    With fail_fast=True (the default), this raises ValueError on the first
    violation—good for CI pipelines that should fail immediately.

    With fail_fast=False, it returns all violations without raising, so
    training can continue to benchmarks even if verification fails.

    Args:
        result: Metrics from compare_teacher_student
        attention: Max allowed attention divergence
        logits: Max allowed logits divergence
        fail_fast: If True, raise on first violation

    Returns:
        List of violations (only useful when fail_fast=False)

    Raises:
        ValueError: If fail_fast=True and any threshold is exceeded
    """
    violations = check_thresholds(result=result, attention=attention, logits=logits)

    if violations and fail_fast:
        raise ValueError(f"compare failed: {violations[0].message()}")

    return violations
