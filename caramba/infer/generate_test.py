"""
Unit tests for the generate module.
"""
from __future__ import annotations

import unittest

import torch
from torch import nn

from caramba.config.kvcache import KVCacheKind
from caramba.config.layer import AttentionLayerConfig, AttentionMode, LayerType
from caramba.infer.generate import (
    GenerateConfig,
    Generator,
    count_attention_layers,
    create_caches,
    get_attention_configs,
    sample_next_token,
)
from caramba.layer.attention import AttentionLayer


class DummyModel(nn.Module):
    """Simple model without attention."""

    def __init__(self, d_model: int = 64) -> None:
        super().__init__()
        self.linear = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor, *, ctx: object | None = None) -> torch.Tensor:
        return self.linear(x.float())


class AttentionModel(nn.Module):
    """Model with attention layers for cache testing."""

    def __init__(self, n_layers: int = 2, d_model: int = 64, n_heads: int = 4) -> None:
        super().__init__()
        self.embed = nn.Embedding(1000, d_model)
        self.layers = nn.ModuleList([
            AttentionLayer(AttentionLayerConfig(
                type=LayerType.ATTENTION,
                d_model=d_model,
                n_heads=n_heads,
                n_kv_heads=n_heads,
            ))
            for _ in range(n_layers)
        ])
        self.head = nn.Linear(d_model, 1000)

    def forward(self, x: torch.Tensor, *, ctx: object | None = None) -> torch.Tensor:
        h = self.embed(x)
        for layer in self.layers:
            out, _ = layer(h, ctx=ctx)
            h = out
        return self.head(h)


class DBAModel(nn.Module):
    """Model with DBA attention layers."""

    def __init__(
        self,
        n_layers: int = 2,
        d_model: int = 64,
        n_heads: int = 4,
        sem_dim: int = 16,
        geo_dim: int = 32,
    ) -> None:
        super().__init__()
        self.embed = nn.Embedding(1000, d_model)
        self.layers = nn.ModuleList([
            AttentionLayer(AttentionLayerConfig(
                type=LayerType.ATTENTION,
                d_model=d_model,
                n_heads=n_heads,
                n_kv_heads=n_heads,
                mode=AttentionMode.DECOUPLED,
                sem_dim=sem_dim,
                geo_dim=geo_dim,
            ))
            for _ in range(n_layers)
        ])
        self.head = nn.Linear(d_model, 1000)

    def forward(self, x: torch.Tensor, *, ctx: object | None = None) -> torch.Tensor:
        h = self.embed(x)
        for layer in self.layers:
            out, _ = layer(h, ctx=ctx)
            h = out
        return self.head(h)


class TestGenerateConfig(unittest.TestCase):
    """Tests for GenerateConfig."""

    def test_defaults(self) -> None:
        """Default values are applied."""
        cfg = GenerateConfig()
        self.assertEqual(cfg.max_new_tokens, 64)
        self.assertAlmostEqual(cfg.temperature, 1.0)
        self.assertIsNone(cfg.top_k)
        self.assertIsNone(cfg.top_p)
        self.assertIsNone(cfg.eos_token_id)
        self.assertEqual(cfg.max_seq_len, 2048)
        self.assertEqual(cfg.cache_kind, KVCacheKind.FP16)

    def test_custom_values(self) -> None:
        """Custom values override defaults."""
        cfg = GenerateConfig(
            max_new_tokens=128,
            temperature=0.7,
            top_k=50,
            top_p=0.9,
            eos_token_id=2,
            max_seq_len=4096,
            cache_kind=KVCacheKind.Q8_0,
        )
        self.assertEqual(cfg.max_new_tokens, 128)
        self.assertAlmostEqual(cfg.temperature, 0.7)
        self.assertEqual(cfg.top_k, 50)
        self.assertIsNotNone(cfg.top_p)
        assert cfg.top_p is not None
        self.assertAlmostEqual(cfg.top_p, 0.9)
        self.assertEqual(cfg.eos_token_id, 2)
        self.assertEqual(cfg.max_seq_len, 4096)
        self.assertEqual(cfg.cache_kind, KVCacheKind.Q8_0)


class TestCountAttentionLayers(unittest.TestCase):
    """Tests for count_attention_layers."""

    def test_no_attention(self) -> None:
        """Model without attention returns 0."""
        model = DummyModel()
        count = count_attention_layers(model)
        self.assertEqual(count, 0)

    def test_with_attention(self) -> None:
        """Model with attention returns correct count."""
        model = AttentionModel(n_layers=3)
        count = count_attention_layers(model)
        self.assertEqual(count, 3)


class TestGetAttentionConfigs(unittest.TestCase):
    """Tests for get_attention_configs."""

    def test_no_attention(self) -> None:
        """Model without attention returns empty list."""
        model = DummyModel()
        configs = get_attention_configs(model)
        self.assertEqual(len(configs), 0)

    def test_with_attention(self) -> None:
        """Model with attention returns configs."""
        model = AttentionModel(n_layers=2, d_model=64, n_heads=4)
        configs = get_attention_configs(model)
        self.assertEqual(len(configs), 2)
        for cfg in configs:
            self.assertIsInstance(cfg, AttentionLayerConfig)
            self.assertEqual(cfg.d_model, 64)
            self.assertEqual(cfg.n_heads, 4)


class TestCreateCaches(unittest.TestCase):
    """Tests for create_caches."""

    def test_no_attention(self) -> None:
        """Model without attention returns empty list."""
        model = DummyModel()
        caches = create_caches(
            model,
            batch_size=1,
            max_seq_len=512,
            device=torch.device("cpu"),
        )
        self.assertEqual(len(caches), 0)

    def test_standard_attention(self) -> None:
        """Standard attention creates LayerKVCache."""
        model = AttentionModel(n_layers=2)
        caches = create_caches(
            model,
            batch_size=1,
            max_seq_len=512,
            device=torch.device("cpu"),
        )
        self.assertEqual(len(caches), 2)
        from caramba.cache.layer import LayerKVCache
        for cache in caches:
            self.assertIsInstance(cache, LayerKVCache)

    def test_dba_attention(self) -> None:
        """DBA attention creates DecoupledLayerKVCache."""
        model = DBAModel(n_layers=2)
        caches = create_caches(
            model,
            batch_size=1,
            max_seq_len=512,
            device=torch.device("cpu"),
        )
        self.assertEqual(len(caches), 2)
        from caramba.cache.decoupled import DecoupledLayerKVCache
        for cache in caches:
            self.assertIsInstance(cache, DecoupledLayerKVCache)


class TestSampleNextToken(unittest.TestCase):
    """Tests for sample_next_token."""

    def test_greedy_decoding(self) -> None:
        """Temperature 0 gives greedy decoding."""
        logits = torch.tensor([[1.0, 5.0, 2.0, 3.0]])
        token = sample_next_token(logits, temperature=0.0)
        self.assertEqual(token.item(), 1)  # argmax

    def test_temperature_1(self) -> None:
        """Temperature 1 samples from distribution."""
        torch.manual_seed(42)
        logits = torch.tensor([[0.0, 0.0, 10.0, 0.0]])  # Strong preference for idx 2
        tokens = [sample_next_token(logits, temperature=1.0).item() for _ in range(10)]
        # Most should be 2
        self.assertGreater(tokens.count(2), 5)

    def test_top_k_filtering(self) -> None:
        """Top-k filtering limits choices."""
        torch.manual_seed(42)
        logits = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
        tokens = [
            sample_next_token(logits, temperature=1.0, top_k=2).item()
            for _ in range(20)
        ]
        # Only top 2 tokens (idx 3, 4) should appear
        for t in tokens:
            self.assertIn(t, [3, 4])

    def test_output_shape(self) -> None:
        """Output has correct shape."""
        logits = torch.randn(4, 100)  # batch_size=4, vocab_size=100
        tokens = sample_next_token(logits, temperature=1.0)
        self.assertEqual(tokens.shape, (4,))


class TestGenerator(unittest.TestCase):
    """Tests for Generator class."""

    def test_reset_clears_state(self) -> None:
        """Reset clears caches and position."""
        model = AttentionModel(n_layers=2)
        gen = Generator(model, device=torch.device("cpu"))

        # Initialize caches
        gen._ensure_caches(batch_size=1)
        self.assertIsNotNone(gen._caches)
        self.assertIsNotNone(gen._ctx)

        # Reset
        gen.reset()
        self.assertIsNone(gen._caches)
        self.assertIsNone(gen._ctx)
        self.assertEqual(gen._pos, 0)

    def test_ensure_caches_idempotent(self) -> None:
        """Calling _ensure_caches multiple times is safe."""
        model = AttentionModel(n_layers=2)
        gen = Generator(model, device=torch.device("cpu"))

        gen._ensure_caches(batch_size=1)
        caches1 = gen._caches

        gen._ensure_caches(batch_size=1)
        caches2 = gen._caches

        self.assertIs(caches1, caches2)

    def test_prefill_updates_position(self) -> None:
        """Prefill updates position counter."""
        model = AttentionModel(n_layers=2)
        gen = Generator(model, device=torch.device("cpu"))

        input_ids = torch.randint(0, 1000, (1, 10))
        gen.prefill(input_ids)

        self.assertEqual(gen._pos, 10)

    def test_decode_step_updates_position(self) -> None:
        """Decode step increments position."""
        model = AttentionModel(n_layers=2)
        gen = Generator(model, device=torch.device("cpu"))

        input_ids = torch.randint(0, 1000, (1, 10))
        gen.prefill(input_ids)

        token = torch.tensor([[5]])
        gen.decode_step(token)

        self.assertEqual(gen._pos, 11)


if __name__ == "__main__":
    unittest.main()
