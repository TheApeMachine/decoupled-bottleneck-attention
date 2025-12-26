"""
verify provides config models for post-run verification steps.
"""
from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field, model_validator

from caramba.config import PositiveFloat, PositiveInt
from caramba.config.eval import EvalVerifyConfig
from caramba.config.kvcache import KVCachePolicyConfig, KVCachePolicyDecoupledConfig


class CompareThreshold(BaseModel):
    """
    CompareThreshold defines absolute error thresholds for a tensor stream.
    """
    max_mean_l1: PositiveFloat
    max_max_l1: PositiveFloat


class CompareVerifyConfig(BaseModel):
    """
    CompareVerifyConfig verifies teacher vs student agreement after adaptation.
    """
    type: Literal["compare"] = "compare"
    batches: PositiveInt
    attention: CompareThreshold | None = None
    logits: CompareThreshold | None = None

    @model_validator(mode="after")
    def _validate_mode(self) -> "CompareVerifyConfig":
        """
        _validate_mode validates compare configuration invariants.
        """
        if self.attention is None and self.logits is None:
            raise ValueError(
                "compare verify requires at least one of: attention, logits"
            )
        return self


class KVCacheVerifyConfig(BaseModel):
    """
    KVCacheVerifyConfig estimates KV-cache memory for teacher vs student.
    """

    type: Literal["kvcache"] = "kvcache"
    n_layers: PositiveInt
    batch_size: PositiveInt
    max_seq_len: PositiveInt
    teacher: KVCachePolicyConfig
    student: KVCachePolicyDecoupledConfig
    min_reduction_ratio: PositiveFloat | None = None


VerifyConfig = Annotated[
    CompareVerifyConfig | EvalVerifyConfig | KVCacheVerifyConfig,
    Field(discriminator="type"),
]

