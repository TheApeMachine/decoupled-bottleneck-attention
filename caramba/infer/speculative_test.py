from __future__ import annotations

import torch

from caramba.config.kvcache import KVCacheKind
from caramba.config.layer import AttentionLayerConfig, AttentionMode, LayerType
from caramba.infer.speculative import SpeculativeConfig, SpeculativeGenerator
from caramba.layer.attention import AttentionLayer


class _ConstLogitsModel(torch.nn.Module):
    """A tiny token model that always predicts the same token as argmax."""

    def __init__(self, *, vocab_size: int, token_id: int) -> None:
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.token_id = int(token_id)

    def forward(self, ids: torch.Tensor, *, ctx: object | None = None) -> torch.Tensor:
        b, t = ids.shape
        logits = torch.zeros((b, t, self.vocab_size), dtype=torch.float32, device=ids.device)
        logits[..., self.token_id] = 10.0
        return logits


def test_adaptive_spec_k_increases_when_acceptance_is_high() -> None:
    target = _ConstLogitsModel(vocab_size=16, token_id=3)
    draft = _ConstLogitsModel(vocab_size=16, token_id=3)

    cfg = SpeculativeConfig(
        spec_k=2,
        spec_k_adaptive=True,
        spec_k_min=1,
        spec_k_max=4,
        spec_k_target_accept=0.7,
        spec_k_adjust_interval=1,
        spec_k_step=1,
        max_new_tokens=8,
        temperature=0.0,
        top_k=None,
        top_p=None,
    )
    gen = SpeculativeGenerator(target_model=target, draft_model=draft, config=cfg)
    _ = gen.generate(torch.tensor([[1, 2, 3]], dtype=torch.long))
    assert gen._spec_k_current == 4  # type: ignore[attr-defined]


def test_adaptive_spec_k_decreases_when_acceptance_is_low() -> None:
    target = _ConstLogitsModel(vocab_size=16, token_id=2)
    draft = _ConstLogitsModel(vocab_size=16, token_id=1)

    cfg = SpeculativeConfig(
        spec_k=4,
        spec_k_adaptive=True,
        spec_k_min=1,
        spec_k_max=8,
        spec_k_target_accept=0.7,
        spec_k_adjust_interval=1,
        spec_k_step=1,
        max_new_tokens=8,
        temperature=0.0,
        top_k=None,
        top_p=None,
    )
    gen = SpeculativeGenerator(target_model=target, draft_model=draft, config=cfg)
    _ = gen.generate(torch.tensor([[1, 2, 3]], dtype=torch.long))
    assert gen._spec_k_current == 1  # type: ignore[attr-defined]


class _TinyTokenModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed = torch.nn.Embedding(1000, 32)
        self.attn = AttentionLayer(
            AttentionLayerConfig(
                type=LayerType.ATTENTION, d_model=32, n_heads=4, n_kv_heads=4, mode=AttentionMode.STANDARD
            )
        )
        self.head = torch.nn.Linear(32, 1000)

    def forward(self, ids: torch.Tensor, *, ctx: object | None = None) -> torch.Tensor:
        h = self.embed(ids)
        h, _ = self.attn(h, ctx=ctx)
        return self.head(h)


def test_speculative_cache_kind_auto_uses_cache_policy_optimizer() -> None:
    target = _TinyTokenModel()
    draft = _TinyTokenModel()
    cfg = SpeculativeConfig(
        cache_kind="auto",
        cache_budget_mb=0.0,
        max_new_tokens=2,
        temperature=0.0,
    )
    gen = SpeculativeGenerator(target_model=target, draft_model=draft, config=cfg)
    _ = gen.generate(torch.tensor([[1, 2, 3]], dtype=torch.long))

    assert gen._target_caches is not None  # type: ignore[truthy-bool]
    c0 = gen._target_caches[0]  # type: ignore[index]
    # LayerKVCache exposes k/v SeqCacheTensor with .kind.
    # Verify a valid KVCacheKind was selected (not a specific implementation detail).
    assert c0.k.kind is not None  # type: ignore[attr-defined]
    assert isinstance(c0.k.kind, KVCacheKind)  # type: ignore[attr-defined]


class _ConstTokenAttnModel(torch.nn.Module):
    """Attention model whose argmax token is fixed, but still exercises KV-caches."""

    def __init__(self, *, vocab_size: int = 128, token_id: int = 3) -> None:
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.token_id = int(token_id)
        self.embed = torch.nn.Embedding(1000, 32)
        self.attn = AttentionLayer(
            AttentionLayerConfig(
                type=LayerType.ATTENTION,
                d_model=32,
                n_heads=4,
                n_kv_heads=4,
                mode=AttentionMode.STANDARD,
            )
        )
        self.head = torch.nn.Linear(32, self.vocab_size, bias=True)
        with torch.no_grad():
            self.head.weight.zero_()
            self.head.bias.zero_()
            self.head.bias[self.token_id] = 10.0

    def forward(self, ids: torch.Tensor, *, ctx: object | None = None) -> torch.Tensor:
        h = self.embed(ids)
        h, _ = self.attn(h, ctx=ctx)
        return self.head(h)


def test_speculative_rollback_keeps_cache_positions_in_sync_on_rejection() -> None:
    """When draft disagrees, speculative decoding rolls back caches safely."""
    target = _ConstTokenAttnModel(vocab_size=64, token_id=2)
    draft = _ConstTokenAttnModel(vocab_size=64, token_id=1)
    cfg = SpeculativeConfig(
        spec_k=4,
        max_new_tokens=6,
        temperature=0.0,
        cache_kind=KVCacheKind.FP16,
    )
    gen = SpeculativeGenerator(target_model=target, draft_model=draft, config=cfg)
    out = gen.generate(torch.tensor([[1, 2, 3, 4]], dtype=torch.long))

    expected_pos = int(out.size(1))
    assert gen._pos == expected_pos  # type: ignore[attr-defined]

    assert gen._target_caches is not None  # type: ignore[truthy-bool]
    assert gen._draft_caches is not None  # type: ignore[truthy-bool]
    for c in gen._target_caches:  # type: ignore[union-attr]
        assert int(c.pos) == expected_pos  # type: ignore[attr-defined]
    for c in gen._draft_caches:  # type: ignore[union-attr]
        assert int(c.pos) == expected_pos  # type: ignore[attr-defined]

