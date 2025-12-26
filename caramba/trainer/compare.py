"""
compare provides teacher/student comparison utilities.
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
    """
    CompareResult holds aggregated comparison metrics.
    """
    attention_mean_l1: float | None
    attention_max_l1: float | None
    logits_mean_l1: float | None
    logits_max_l1: float | None
    batches: int

    def __post_init__(self) -> None:
        """Ensure batches is an int."""
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
    """
    compare_teacher_student compares teacher and student on a batch list.
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

            if attention is not None:
                if len(t_trace.outputs) != len(s_trace.outputs):
                    raise ValueError(
                        "Teacher/student trace outputs mismatch: "
                        f"{len(t_trace.outputs)} vs {len(s_trace.outputs)}"
                    )
                for t_out, s_out in zip(t_trace.outputs, s_trace.outputs):
                    if t_out.shape != s_out.shape:
                        raise ValueError(
                            "Teacher/student output shapes mismatch: "
                            f"{t_out.shape} vs {s_out.shape}"
                        )
                    err = F.l1_loss(s_out, t_out, reduction="mean")
                    v = float(err)
                    attn_sum += v
                    attn_count += 1
                    if v > attn_max:
                        attn_max = v

            if logits is not None:
                if t_logits.shape != s_logits.shape:
                    raise ValueError(
                        "Teacher/student logits shapes mismatch: "
                        f"{t_logits.shape} vs {s_logits.shape}"
                    )
                err = F.l1_loss(s_logits, t_logits, reduction="mean")
                v = float(err)
                logits_sum += v
                logits_count += 1
                if v > logits_max:
                    logits_max = v

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


def assert_thresholds(
    *,
    result: CompareResult,
    attention: CompareThreshold | None,
    logits: CompareThreshold | None,
) -> None:
    """
    assert_thresholds validates results against configured thresholds.
    """
    if attention is not None:
        if result.attention_mean_l1 is None or result.attention_max_l1 is None:
            raise RuntimeError("Missing attention results for threshold check.")
        if result.attention_mean_l1 > float(attention.max_mean_l1):
            raise ValueError(
                "compare failed: attention_mean_l1 exceeded threshold: "
                f"mean={result.attention_mean_l1:.6f}, "
                f"max_mean_l1={float(attention.max_mean_l1):.6f}"
            )
        if result.attention_max_l1 > float(attention.max_max_l1):
            raise ValueError(
                "compare failed: attention_max_l1 exceeded threshold: "
                f"max={result.attention_max_l1:.6f}, "
                f"max_max_l1={float(attention.max_max_l1):.6f}"
            )

    if logits is not None:
        if result.logits_mean_l1 is None or result.logits_max_l1 is None:
            raise RuntimeError("Missing logits results for threshold check.")
        if result.logits_mean_l1 > float(logits.max_mean_l1):
            raise ValueError(
                "compare failed: logits_mean_l1 exceeded threshold: "
                f"mean={result.logits_mean_l1:.6f}, "
                f"max_mean_l1={float(logits.max_mean_l1):.6f}"
            )
        if result.logits_max_l1 > float(logits.max_max_l1):
            raise ValueError(
                "compare failed: logits_max_l1 exceeded threshold: "
                f"max={result.logits_max_l1:.6f}, "
                f"max_max_l1={float(logits.max_max_l1):.6f}"
            )


