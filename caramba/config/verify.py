"""Verification configuration for checking model quality after training.

Verification runs after training to catch failures before expensive
benchmarking. It compares teacher and student outputs, runs behavioral
evaluations, or analyzes KV-cache memory usage.
"""
from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field, model_validator

from caramba.config import PositiveFloat, PositiveInt
from caramba.config.eval import EvalVerifyConfig
from caramba.config.kvcache import KVCachePolicyConfig, KVCachePolicyDecoupledConfig


class CompareThreshold(BaseModel):
    """Maximum allowed divergence between teacher and student outputs.

    We check both mean (overall drift) and max (worst-case layer) to catch
    different failure modes.
    """

    max_mean_l1: PositiveFloat
    max_max_l1: PositiveFloat


class CompareVerifyConfig(BaseModel):
    """Compare teacher and student outputs to verify distillation quality.

    Runs both models on the same batches and measures how much their outputs
    diverge. With fail_fast=False, threshold violations log warnings but
    don't stop the pipeline—useful for getting benchmark results even when
    thresholds are exceeded.
    """

    type: Literal["compare"] = "compare"
    batches: PositiveInt
    attention: CompareThreshold | None = None
    logits: CompareThreshold | None = None
    fail_fast: bool = False

    @model_validator(mode="after")
    def _require_at_least_one_metric(self) -> "CompareVerifyConfig":
        """Ensure at least one comparison metric is configured."""
        if self.attention is None and self.logits is None:
            raise ValueError(
                "compare verify requires at least one of: attention, logits"
            )
        return self


class KVCacheVerifyConfig(BaseModel):
    """Analyze KV-cache memory savings from the new architecture.

    For DBA upcycling, we expect significant memory reduction because
    the semantic and geometric key dimensions are much smaller than
    the original key dimension.
    """

    type: Literal["kvcache"] = "kvcache"
    n_layers: PositiveInt
    batch_size: PositiveInt
    max_seq_len: PositiveInt
    teacher: KVCachePolicyConfig
    student: KVCachePolicyDecoupledConfig
    min_reduction_ratio: PositiveFloat | None = None


# Union type for all verification configs, using Pydantic's discriminator
# pattern for automatic deserialization from YAML.
VerifyConfig = Annotated[
    CompareVerifyConfig | EvalVerifyConfig | KVCacheVerifyConfig,
    Field(discriminator="type"),
]
