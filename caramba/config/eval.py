"""
eval provides config models for behavioral evaluation suites.
"""
from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field, model_validator

from caramba.config import PositiveInt, Probability


class TiktokenTokenizerConfig(BaseModel):
    """
    TiktokenTokenizerConfig configures a tiktoken tokenizer.
    """
    type: Literal["tiktoken"] = "tiktoken"
    encoding: str


TokenizerConfig = Annotated[
    TiktokenTokenizerConfig,
    Field(discriminator="type"),
]


class EvalCase(BaseModel):
    """
    EvalCase defines a single behavior eval case with a ground truth.
    """
    id: str
    prompt: str
    kind: Literal["choice_logprob", "int_greedy"]

    choices: list[str] | None = None
    answer: str | int | None = None

    @model_validator(mode="after")
    def _validate_case(self) -> "EvalCase":
        """
        _validate_case validates case invariants.
        """
        if not self.id:
            raise ValueError("case.id must be non-empty")
        if not self.prompt:
            raise ValueError("case.prompt must be non-empty")

        match self.kind:
            case "choice_logprob":
                if not self.choices:
                    raise ValueError("choice_logprob requires non-empty choices")
                if not isinstance(self.answer, str) or not self.answer:
                    raise ValueError("choice_logprob requires a string answer")
                if str(self.answer) not in set(self.choices):
                    raise ValueError(
                        "choice_logprob answer must be one of choices"
                    )
            case "int_greedy":
                if not isinstance(self.answer, int):
                    raise ValueError("int_greedy requires an integer answer")
            case _:
                raise ValueError(f"Unsupported eval case kind: {self.kind!r}")

        return self


class EvalThresholds(BaseModel):
    """
    EvalThresholds defines acceptance thresholds for eval results.
    """
    min_student_accuracy: Probability
    max_accuracy_drop: Probability


class EvalVerifyConfig(BaseModel):
    """
    EvalVerifyConfig runs a small behavioral suite against teacher and student.
    """
    type: Literal["eval"] = "eval"
    tokenizer: TokenizerConfig
    max_new_tokens: PositiveInt
    context_window: PositiveInt | None = None
    cases: list[EvalCase] = Field(min_length=1)
    thresholds: EvalThresholds


