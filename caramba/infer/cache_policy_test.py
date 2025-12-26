from __future__ import annotations

import pytest
import torch

from caramba.config.kvcache import KVCacheKind
from caramba.config.layer import AttentionLayerConfig, AttentionMode, LayerType
from caramba.infer.cache_policy import (
    choose_cache_kind,
    estimate_kvcache_bytes,
    long_context_fidelity_check,
    needle_in_haystack_gate,
    short_context_fidelity_check,
)
from caramba.layer.attention import AttentionLayer


class _TinyAttnModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        cfg = AttentionLayerConfig(
            type=LayerType.ATTENTION,
            d_model=32,
            n_heads=4,
            mode=AttentionMode.STANDARD,
        )
        self.attn = AttentionLayer(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y, _ = self.attn(x)
        return y


def test_estimate_kvcache_bytes_monotonic() -> None:
    m = _TinyAttnModel()
    fp16 = estimate_kvcache_bytes(
        model=m, batch_size=1, max_seq_len=128, kind=KVCacheKind.FP16, qblock=32, residual_len=0
    )
    q8 = estimate_kvcache_bytes(
        model=m, batch_size=1, max_seq_len=128, kind=KVCacheKind.Q8_0, qblock=32, residual_len=0
    )
    q4 = estimate_kvcache_bytes(
        model=m, batch_size=1, max_seq_len=128, kind=KVCacheKind.Q4_0, qblock=32, residual_len=0
    )
    assert fp16 > q8 > q4


def test_choose_cache_kind_budget_zero_picks_most_compressed() -> None:
    m = _TinyAttnModel()
    choice = choose_cache_kind(
        model=m,
        batch_size=1,
        max_seq_len=128,
        qblock=32,
        residual_len=0,
        budget_mb=0.0,
    )
    assert choice.kind == KVCacheKind.Q4_0


class _TinyTokenModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed = torch.nn.Embedding(1000, 32)
        self.attn = AttentionLayer(
            AttentionLayerConfig(type=LayerType.ATTENTION, d_model=32, n_heads=4, mode=AttentionMode.STANDARD)
        )
        self.head = torch.nn.Linear(32, 1000)

    def forward(self, ids: torch.Tensor, *, ctx: object | None = None) -> torch.Tensor:
        h = self.embed(ids)
        h, _ = self.attn(h, ctx=ctx)
        return self.head(h)


def test_short_context_fidelity_fp16_vs_fp16_zero_delta() -> None:
    model = _TinyTokenModel()
    tokens = torch.randint(0, 1000, (1, 16), dtype=torch.long)
    res = short_context_fidelity_check(
        model=model,
        token_ids=tokens,
        baseline_kind=KVCacheKind.FP16,
        candidate_kind=KVCacheKind.FP16,
        max_seq_len=64,
        qblock=32,
        residual_len=0,
        prompt_len=8,
    )
    assert abs(res.delta_nll) < 1e-6


def test_long_context_fidelity_requires_long_prompt() -> None:
    model = _TinyTokenModel()
    tokens = torch.randint(0, 1000, (1, 16), dtype=torch.long)
    with pytest.raises(ValueError):
        _ = long_context_fidelity_check(
            model=model,
            token_ids=tokens,
            baseline_kind=KVCacheKind.FP16,
            candidate_kind=KVCacheKind.FP16,
            max_seq_len=64,
            qblock=32,
            residual_len=0,
            prompt_len=32,
        )


def test_needle_gate_fp16_vs_fp16_zero_kl() -> None:
    model = _TinyTokenModel()
    tokens = torch.randint(0, 1000, (1, 32), dtype=torch.long)
    res = needle_in_haystack_gate(
        model=model,
        token_ids=tokens,
        baseline_kind=KVCacheKind.FP16,
        candidate_kind=KVCacheKind.FP16,
        max_seq_len=64,
        qblock=32,
        residual_len=0,
        prompt_len=16,
        decode_steps=2,
    )
    assert res.mean_kl < 1e-6

