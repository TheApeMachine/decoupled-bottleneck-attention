"""
suite provides behavioral evaluation for teacher/student comparison.
"""
from __future__ import annotations

import re

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from caramba.config.eval import EvalCase, EvalThresholds, EvalVerifyConfig
from caramba.eval.tokenizer import Tokenizer, build_tokenizer


class EvalCaseResult:
    """
    EvalCaseResult records per-case outcomes for teacher and student.
    """
    def __init__(
        self,
        *,
        case_id: str,
        teacher_ok: bool,
        student_ok: bool,
        teacher_answer: str,
        student_answer: str,
    ) -> None:
        """
        __init__ initializes the case result.
        """
        self.case_id: str = str(case_id)
        self.teacher_ok: bool = bool(teacher_ok)
        self.student_ok: bool = bool(student_ok)
        self.teacher_answer: str = str(teacher_answer)
        self.student_answer: str = str(student_answer)


class EvalSummary:
    """
    EvalSummary aggregates suite metrics.
    """
    def __init__(self, *, results: list[EvalCaseResult]) -> None:
        """
        __init__ initializes the summary.
        """
        if not results:
            raise ValueError("results must be non-empty")
        self.results: list[EvalCaseResult] = results
        self.teacher_accuracy: float = (
            sum(1 for r in results if r.teacher_ok) / float(len(results))
        )
        self.student_accuracy: float = (
            sum(1 for r in results if r.student_ok) / float(len(results))
        )


def run_eval_verify(
    *,
    teacher: nn.Module,
    student: nn.Module,
    cfg: EvalVerifyConfig,
    device: torch.device,
) -> EvalSummary:
    """
    run_eval_verify runs the configured suite and returns an EvalSummary.
    """
    tokenizer = build_tokenizer(cfg.tokenizer)
    teacher.eval()
    student.eval()

    results: list[EvalCaseResult] = []
    for case in list(cfg.cases):
        results.append(
            _run_case(
                teacher=teacher,
                student=student,
                case=case,
                tokenizer=tokenizer,
                max_new_tokens=int(cfg.max_new_tokens),
                context_window=cfg.context_window,
                device=device,
            )
        )
    return EvalSummary(results=results)


def assert_eval_thresholds(*, summary: EvalSummary, thresholds: EvalThresholds) -> None:
    """
    assert_eval_thresholds validates suite metrics against thresholds.
    """
    if summary.student_accuracy < float(thresholds.min_student_accuracy):
        raise ValueError(
            "eval failed: student accuracy below threshold: "
            f"acc={summary.student_accuracy:.3f}, "
            f"min={float(thresholds.min_student_accuracy):.3f}"
        )
    drop = float(summary.teacher_accuracy - summary.student_accuracy)
    if drop > float(thresholds.max_accuracy_drop):
        raise ValueError(
            "eval failed: accuracy drop exceeded threshold: "
            f"teacher={summary.teacher_accuracy:.3f}, "
            f"student={summary.student_accuracy:.3f}, "
            f"drop={drop:.3f}, "
            f"max_drop={float(thresholds.max_accuracy_drop):.3f}"
        )


def _run_case(
    *,
    teacher: nn.Module,
    student: nn.Module,
    case: EvalCase,
    tokenizer: Tokenizer,
    max_new_tokens: int,
    context_window: int | None,
    device: torch.device,
) -> EvalCaseResult:
    prompt_ids = tokenizer.encode(case.prompt)
    if not prompt_ids:
        raise ValueError(f"Case {case.id!r} encoded to empty prompt ids")

    match case.kind:
        case "choice_logprob":
            assert isinstance(case.answer, str)
            assert case.choices is not None
            t_choice = _pick_choice_by_logprob(
                model=teacher,
                prompt_ids=prompt_ids,
                choices=case.choices,
                tokenizer=tokenizer,
                device=device,
                context_window=context_window,
            )
            s_choice = _pick_choice_by_logprob(
                model=student,
                prompt_ids=prompt_ids,
                choices=case.choices,
                tokenizer=tokenizer,
                device=device,
                context_window=context_window,
            )
            return EvalCaseResult(
                case_id=case.id,
                teacher_ok=(t_choice == case.answer),
                student_ok=(s_choice == case.answer),
                teacher_answer=t_choice,
                student_answer=s_choice,
            )
        case "int_greedy":
            assert isinstance(case.answer, int)
            t_text = _greedy_generate(
                model=teacher,
                prompt_ids=prompt_ids,
                tokenizer=tokenizer,
                device=device,
                max_new_tokens=max_new_tokens,
                context_window=context_window,
            )
            s_text = _greedy_generate(
                model=student,
                prompt_ids=prompt_ids,
                tokenizer=tokenizer,
                device=device,
                max_new_tokens=max_new_tokens,
                context_window=context_window,
            )
            t_int = _extract_first_int(t_text)
            s_int = _extract_first_int(s_text)
            return EvalCaseResult(
                case_id=case.id,
                teacher_ok=(t_int == case.answer),
                student_ok=(s_int == case.answer),
                teacher_answer=str(t_int),
                student_answer=str(s_int),
            )
        case _:
            raise ValueError(f"Unsupported eval kind: {case.kind!r}")


def _extract_first_int(text: str) -> int:
    m = re.search(r"-?\\d+", str(text))
    if m is None:
        return 0
    return int(m.group(0))


def _pick_choice_by_logprob(
    *,
    model: nn.Module,
    prompt_ids: list[int],
    choices: list[str],
    tokenizer: Tokenizer,
    device: torch.device,
    context_window: int | None,
) -> str:
    if not choices:
        raise ValueError("choices must be non-empty")
    best: tuple[float, str] | None = None
    for choice in choices:
        score = _score_completion_logprob(
            model=model,
            prompt_ids=prompt_ids,
            completion_ids=tokenizer.encode(str(choice)),
            device=device,
            context_window=context_window,
        )
        item = (float(score), str(choice))
        best = item if best is None or item[0] > best[0] else best
    if best is None:
        raise RuntimeError("No choices were scored")
    return best[1]


def _score_completion_logprob(
    *,
    model: nn.Module,
    prompt_ids: list[int],
    completion_ids: list[int],
    device: torch.device,
    context_window: int | None,
) -> float:
    if not prompt_ids:
        raise ValueError("prompt_ids must be non-empty")
    if not completion_ids:
        raise ValueError("completion_ids must be non-empty")

    if context_window is not None:
        return _score_completion_logprob_windowed(
            model=model,
            prompt_ids=prompt_ids,
            completion_ids=completion_ids,
            device=device,
            context_window=int(context_window),
        )

    seq = prompt_ids + completion_ids
    x = torch.tensor([seq], device=device, dtype=torch.long)
    logits = model(x)
    if logits.ndim != 3:
        raise ValueError(f"Expected logits (B,T,V), got {logits.shape}")
    if int(logits.shape[1]) != len(seq):
        raise ValueError("Unexpected logits length mismatch")

    logp = F.log_softmax(logits[:, :-1, :], dim=-1)
    target = x[:, 1:]

    start = len(prompt_ids) - 1
    end = start + len(completion_ids)
    tok_logp = logp[0, start:end, :].gather(
        dim=-1,
        index=target[0, start:end].unsqueeze(-1),
    )
    return float(tok_logp.sum())


def _score_completion_logprob_windowed(
    *,
    model: nn.Module,
    prompt_ids: list[int],
    completion_ids: list[int],
    device: torch.device,
    context_window: int,
) -> float:
    if context_window <= 0:
        raise ValueError("context_window must be > 0")

    seq = prompt_ids + completion_ids
    total = 0.0
    start_k = len(prompt_ids)
    for k in range(start_k, len(seq)):
        ctx = seq[max(0, k - context_window) : k]
        if not ctx:
            raise RuntimeError("Empty context during windowed scoring")
        x = torch.tensor([ctx], device=device, dtype=torch.long)
        logits = model(x)
        next_id = int(seq[k])
        lp = F.log_softmax(logits[0, -1, :], dim=-1)
        total += float(lp[next_id])
    return float(total)


def _greedy_generate(
    *,
    model: nn.Module,
    prompt_ids: list[int],
    tokenizer: Tokenizer,
    device: torch.device,
    max_new_tokens: int,
    context_window: int | None,
) -> str:
    ids = list(prompt_ids)
    for _ in range(int(max_new_tokens)):
        ctx = ids
        if context_window is not None:
            ctx = ids[-int(context_window) :]
        x = torch.tensor([ctx], device=device, dtype=torch.long)
        logits = model(x)
        next_id = int(torch.argmax(logits[0, -1, :]).item())
        ids.append(next_id)
    return tokenizer.decode(ids[len(prompt_ids) :])


