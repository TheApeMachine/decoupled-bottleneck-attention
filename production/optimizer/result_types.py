"""Result/experiment naming helpers."""

from __future__ import annotations


class ResultTypeMapper:
    """Maps user-facing result/exp strings into canonical experiment IDs."""

    @staticmethod
    def normalize(exp_or_result: str) -> str:
        s = str(exp_or_result).strip().lower()
        if not s:
            return ""
        if s.startswith("paper_"):
            s = s[len("paper_") :]
        return s

    @classmethod
    def to_exp(cls, result_type: str) -> str:
        rt = cls.normalize(result_type)
        if rt in ("baseline", "standard"):
            return "paper_baseline"
        if rt in ("bottleneck",):
            return "paper_bottleneck"
        if rt in ("decoupled",):
            return "paper_decoupled"
        if rt in ("gqa",):
            return "paper_gqa"
        if rt.startswith("train_"):
            return rt
        return f"paper_{rt}" if rt else ""
