"""Tests for the unified attention layer."""
from __future__ import annotations

import unittest
import torch

from caramba.config.layer import AttentionLayerConfig, AttentionMode, LayerType
from caramba.layer.attention import AttentionLayer
from caramba.cache.layer import LayerKVCache
from caramba.cache.decoupled import DecoupledLayerKVCache
from caramba.config.kvcache import KVCacheTensorConfig, KVCacheKind


class TestAttentionLayerStandard(unittest.TestCase):
    """Tests for standard multi-head attention."""

    def setUp(self) -> None:
        self.batch_size = 2
        self.seq_len = 16
        self.d_model = 64
        self.n_heads = 4
        self.device = torch.device("cpu")

    def test_output_shape(self) -> None:
        """Output shape matches input shape."""
        cfg = AttentionLayerConfig(d_model=self.d_model, n_heads=self.n_heads)
        layer = AttentionLayer(cfg)

        x = torch.randn(self.batch_size, self.seq_len, self.d_model)
        y, cache = layer(x)

        self.assertEqual(y.shape, x.shape)
        self.assertIsNone(cache)

    def test_has_output_projection(self) -> None:
        """Layer has an output projection."""
        cfg = AttentionLayerConfig(d_model=self.d_model, n_heads=self.n_heads)
        layer = AttentionLayer(cfg)

        self.assertIsNotNone(layer.out_proj)
        self.assertEqual(layer.out_proj.in_features, self.d_model)
        self.assertEqual(layer.out_proj.out_features, self.d_model)

    def test_rope_applied(self) -> None:
        """RoPE is applied when enabled."""
        cfg = AttentionLayerConfig(
            d_model=self.d_model, n_heads=self.n_heads, rope_enabled=True
        )
        layer = AttentionLayer(cfg)

        self.assertIsNotNone(layer.rotary)
        assert layer.rotary is not None  # type guard for pyright
        self.assertEqual(layer.rotary.rot_dim, cfg.head_dim)

    def test_rope_disabled(self) -> None:
        """RoPE is not applied when disabled."""
        cfg = AttentionLayerConfig(
            d_model=self.d_model, n_heads=self.n_heads, rope_enabled=False
        )
        layer = AttentionLayer(cfg)

        self.assertIsNone(layer.rotary)

    def test_causal_masking(self) -> None:
        """Causal masking prevents attending to future tokens."""
        cfg = AttentionLayerConfig(
            d_model=self.d_model, n_heads=self.n_heads, is_causal=True
        )
        layer = AttentionLayer(cfg)
        layer.eval()

        x = torch.randn(1, 8, self.d_model)
        y1, _ = layer(x)

        # Modifying future tokens shouldn't affect earlier outputs
        x_modified = x.clone()
        x_modified[:, 4:, :] = torch.randn_like(x_modified[:, 4:, :])
        y2, _ = layer(x_modified)

        # First 4 positions should be identical
        torch.testing.assert_close(y1[:, :4, :], y2[:, :4, :])

    def test_deterministic_eval(self) -> None:
        """Same input produces same output in eval mode."""
        cfg = AttentionLayerConfig(
            d_model=self.d_model, n_heads=self.n_heads, dropout_p=0.1
        )
        layer = AttentionLayer(cfg)
        layer.eval()

        x = torch.randn(self.batch_size, self.seq_len, self.d_model)
        y1, _ = layer(x)
        y2, _ = layer(x)

        torch.testing.assert_close(y1, y2)

    def test_q_chunk_matches_full_attention(self) -> None:
        """Chunked SDPA matches full attention in eval mode."""
        cfg_full = AttentionLayerConfig(
            d_model=self.d_model, n_heads=self.n_heads, is_causal=True, dropout_p=0.0
        )
        cfg_chunk = AttentionLayerConfig(
            d_model=self.d_model,
            n_heads=self.n_heads,
            is_causal=True,
            dropout_p=0.0,
            q_chunk=3,
        )
        full = AttentionLayer(cfg_full).eval()
        chunked = AttentionLayer(cfg_chunk).eval()
        chunked.load_state_dict(full.state_dict())

        x = torch.randn(1, 9, self.d_model)
        y_full, _ = full(x)
        y_chunk, _ = chunked(x)
        torch.testing.assert_close(y_full, y_chunk, atol=1e-5, rtol=1e-5)

    def test_local_window_one_only_self(self) -> None:
        """local_window=1 forces each token to attend only to itself."""
        cfg = AttentionLayerConfig(
            d_model=self.d_model,
            n_heads=self.n_heads,
            is_causal=True,
            dropout_p=0.0,
            q_chunk=4,
            local_window=1,
        )
        layer = AttentionLayer(cfg).eval()

        x = torch.randn(2, 7, self.d_model)
        y, _ = layer(x)

        # With window=1, attention output equals V (single key), then out_proj.
        v = layer.v_proj(x)
        expected = layer.out_proj(v)
        torch.testing.assert_close(y, expected, atol=1e-5, rtol=1e-5)

    def test_mem_block_summarization_enabled(self) -> None:
        """mem_block summarization runs and preserves output shape."""
        cfg = AttentionLayerConfig(
            d_model=self.d_model,
            n_heads=self.n_heads,
            is_causal=True,
            dropout_p=0.0,
            q_chunk=4,
            local_window=8,
            mem_block=4,
            mem_summarize="mean",
        )
        layer = AttentionLayer(cfg).eval()
        x = torch.randn(2, 32, self.d_model)
        y, _ = layer(x)
        self.assertEqual(y.shape, x.shape)

    def test_mem_activation_threshold_disables_summarization(self) -> None:
        """mem_activation_threshold can disable summarization for short sequences."""
        cfg_base = AttentionLayerConfig(
            d_model=self.d_model,
            n_heads=self.n_heads,
            is_causal=True,
            dropout_p=0.0,
            q_chunk=4,
            local_window=8,
        )
        cfg_mem_disabled = AttentionLayerConfig(
            d_model=self.d_model,
            n_heads=self.n_heads,
            is_causal=True,
            dropout_p=0.0,
            q_chunk=4,
            local_window=8,
            mem_block=4,
            mem_summarize="mean",
            mem_activation_threshold=10_000,  # disable for small T
        )
        base = AttentionLayer(cfg_base).eval()
        mem = AttentionLayer(cfg_mem_disabled).eval()
        mem.load_state_dict(base.state_dict())
        x = torch.randn(1, 32, self.d_model)
        y_base, _ = base(x)
        y_mem, _ = mem(x)
        torch.testing.assert_close(y_base, y_mem, atol=1e-5, rtol=1e-5)


class TestAttentionLayerGQA(unittest.TestCase):
    """Tests for grouped-query attention."""

    def setUp(self) -> None:
        self.batch_size = 2
        self.seq_len = 16
        self.d_model = 64
        self.n_heads = 8
        self.n_kv_heads = 2
        self.device = torch.device("cpu")

    def test_gqa_output_shape(self) -> None:
        """GQA output shape matches input."""
        cfg = AttentionLayerConfig(
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_kv_heads=self.n_kv_heads,
            mode=AttentionMode.GQA,
        )
        layer = AttentionLayer(cfg)

        x = torch.randn(self.batch_size, self.seq_len, self.d_model)
        y, _ = layer(x)

        self.assertEqual(y.shape, x.shape)

    def test_gqa_fewer_kv_projections(self) -> None:
        """GQA uses fewer KV heads than Q heads."""
        cfg = AttentionLayerConfig(
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_kv_heads=self.n_kv_heads,
            mode=AttentionMode.GQA,
        )
        layer = AttentionLayer(cfg)

        # K and V projections should have fewer output features
        kv_dim = self.n_kv_heads * cfg.head_dim
        q_dim = self.n_heads * cfg.head_dim

        self.assertIsNotNone(layer.k_proj)
        self.assertIsNotNone(layer.q_proj)
        assert layer.k_proj is not None  # type guard for pyright
        assert layer.q_proj is not None  # type guard for pyright
        self.assertEqual(layer.k_proj.out_features, kv_dim)
        self.assertEqual(layer.v_proj.out_features, kv_dim)
        self.assertEqual(layer.q_proj.out_features, q_dim)

    def test_gqa_group_size(self) -> None:
        """Group size is correctly computed."""
        cfg = AttentionLayerConfig(
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_kv_heads=self.n_kv_heads,
            mode=AttentionMode.GQA,
        )
        layer = AttentionLayer(cfg)

        self.assertEqual(layer.group_size, self.n_heads // self.n_kv_heads)


class TestAttentionLayerDecoupled(unittest.TestCase):
    """Tests for decoupled (DBA) attention."""

    def setUp(self) -> None:
        self.batch_size = 2
        self.seq_len = 16
        self.d_model = 64
        self.n_heads = 4
        self.sem_dim = 32
        self.geo_dim = 32
        self.attn_dim = 64
        self.device = torch.device("cpu")

    def test_decoupled_output_shape(self) -> None:
        """Decoupled attention output shape matches input."""
        cfg = AttentionLayerConfig(
            d_model=self.d_model,
            n_heads=self.n_heads,
            mode=AttentionMode.DECOUPLED,
            sem_dim=self.sem_dim,
            geo_dim=self.geo_dim,
            attn_dim=self.attn_dim,
        )
        layer = AttentionLayer(cfg)

        x = torch.randn(self.batch_size, self.seq_len, self.d_model)
        y, _ = layer(x)

        self.assertEqual(y.shape, x.shape)

    def test_decoupled_q_chunk_matches_full(self) -> None:
        """Chunked DBA matches full DBA in eval mode."""
        cfg_full = AttentionLayerConfig(
            d_model=self.d_model,
            n_heads=self.n_heads,
            mode=AttentionMode.DECOUPLED,
            sem_dim=128,
            geo_dim=256,
            dropout_p=0.0,
            is_causal=True,
        )
        cfg_chunk = AttentionLayerConfig(
            d_model=self.d_model,
            n_heads=self.n_heads,
            mode=AttentionMode.DECOUPLED,
            sem_dim=128,
            geo_dim=256,
            dropout_p=0.0,
            is_causal=True,
            q_chunk=4,
        )
        full = AttentionLayer(cfg_full).eval()
        chunked = AttentionLayer(cfg_chunk).eval()
        chunked.load_state_dict(full.state_dict())
        x = torch.randn(1, 10, self.d_model)
        y_full, _ = full(x)
        y_chunk, _ = chunked(x)
        torch.testing.assert_close(y_full, y_chunk, atol=1e-5, rtol=1e-5)

    def test_has_semantic_projections(self) -> None:
        """Decoupled mode has semantic Q/K projections."""
        cfg = AttentionLayerConfig(
            d_model=self.d_model,
            n_heads=self.n_heads,
            mode=AttentionMode.DECOUPLED,
            sem_dim=self.sem_dim,
            geo_dim=self.geo_dim,
            attn_dim=self.attn_dim,
        )
        layer = AttentionLayer(cfg)

        self.assertIsNotNone(layer.q_sem)
        self.assertIsNotNone(layer.k_sem)
        assert layer.q_sem is not None  # type guard for pyright
        assert layer.k_sem is not None  # type guard for pyright
        self.assertEqual(layer.q_sem.out_features, self.sem_dim)
        self.assertEqual(layer.k_sem.out_features, self.sem_dim)

    def test_has_geometric_projections(self) -> None:
        """Decoupled mode has geometric Q/K projections."""
        cfg = AttentionLayerConfig(
            d_model=self.d_model,
            n_heads=self.n_heads,
            mode=AttentionMode.DECOUPLED,
            sem_dim=self.sem_dim,
            geo_dim=self.geo_dim,
            attn_dim=self.attn_dim,
        )
        layer = AttentionLayer(cfg)

        self.assertIsNotNone(layer.q_geo)
        self.assertIsNotNone(layer.k_geo)
        assert layer.q_geo is not None  # type guard for pyright
        assert layer.k_geo is not None  # type guard for pyright
        self.assertEqual(layer.q_geo.out_features, self.geo_dim)
        self.assertEqual(layer.k_geo.out_features, self.geo_dim)

    def test_rope_only_on_geometric(self) -> None:
        """RoPE is only applied to geometric path."""
        cfg = AttentionLayerConfig(
            d_model=self.d_model,
            n_heads=self.n_heads,
            mode=AttentionMode.DECOUPLED,
            sem_dim=self.sem_dim,
            geo_dim=self.geo_dim,
            attn_dim=self.attn_dim,
            rope_enabled=True,
        )
        layer = AttentionLayer(cfg)

        # Should have rotary_geo but not rotary
        self.assertIsNotNone(layer.rotary_geo)
        self.assertIsNone(layer.rotary)
        assert layer.rotary_geo is not None  # type guard for pyright
        self.assertEqual(layer.rotary_geo.rot_dim, cfg.geo_head_dim)

    def test_no_standard_projections(self) -> None:
        """Decoupled mode doesn't use standard Q/K projections."""
        cfg = AttentionLayerConfig(
            d_model=self.d_model,
            n_heads=self.n_heads,
            mode=AttentionMode.DECOUPLED,
            sem_dim=self.sem_dim,
            geo_dim=self.geo_dim,
            attn_dim=self.attn_dim,
        )
        layer = AttentionLayer(cfg)

        self.assertIsNone(layer.q_proj)
        self.assertIsNone(layer.k_proj)

    def test_decoupled_gate(self) -> None:
        """Decoupled gate is created when enabled."""
        cfg = AttentionLayerConfig(
            d_model=self.d_model,
            n_heads=self.n_heads,
            mode=AttentionMode.DECOUPLED,
            sem_dim=self.sem_dim,
            geo_dim=self.geo_dim,
            attn_dim=self.attn_dim,
            decoupled_gate=True,
        )
        layer = AttentionLayer(cfg)

        self.assertIsNotNone(layer.decoupled_gate_logit)
        assert layer.decoupled_gate_logit is not None  # type guard for pyright
        self.assertEqual(layer.decoupled_gate_logit.shape, (self.n_heads,))

    def test_decoupled_gate_dynamic(self) -> None:
        """Dynamic decoupled gate has query projection."""
        cfg = AttentionLayerConfig(
            d_model=self.d_model,
            n_heads=self.n_heads,
            mode=AttentionMode.DECOUPLED,
            sem_dim=self.sem_dim,
            geo_dim=self.geo_dim,
            attn_dim=self.attn_dim,
            decoupled_gate=True,
            decoupled_gate_dynamic=True,
        )
        layer = AttentionLayer(cfg)

        self.assertIsNotNone(layer.decoupled_gate_proj)
        assert layer.decoupled_gate_proj is not None  # type guard for pyright
        self.assertEqual(layer.decoupled_gate_proj.out_features, self.n_heads)

    def test_different_sem_geo_dims(self) -> None:
        """Semantic and geometric can have different dimensions."""
        cfg = AttentionLayerConfig(
            d_model=self.d_model,
            n_heads=self.n_heads,
            mode=AttentionMode.DECOUPLED,
            sem_dim=48,
            geo_dim=16,
            attn_dim=self.attn_dim,
        )
        layer = AttentionLayer(cfg)

        self.assertIsNotNone(layer.q_sem)
        self.assertIsNotNone(layer.q_geo)
        assert layer.q_sem is not None  # type guard for pyright
        assert layer.q_geo is not None  # type guard for pyright
        self.assertEqual(layer.q_sem.out_features, 48)
        self.assertEqual(layer.q_geo.out_features, 16)


class TestAttentionWithCache(unittest.TestCase):
    """Tests for attention with KV cache."""

    def setUp(self) -> None:
        self.batch_size = 2
        self.seq_len = 16
        self.d_model = 64
        self.n_heads = 4
        self.max_seq_len = 128
        self.device = torch.device("cpu")
        self.kv_cfg = KVCacheTensorConfig(kind=KVCacheKind.FP16, qblock=32, residual_len=0)

    def test_standard_cache_prefill(self) -> None:
        """Standard attention populates cache during prefill."""
        cfg = AttentionLayerConfig(d_model=self.d_model, n_heads=self.n_heads)
        layer = AttentionLayer(cfg)

        cache = LayerKVCache(
            batch_size=self.batch_size,
            max_seq_len=self.max_seq_len,
            k_dim=self.d_model,
            v_dim=self.d_model,
            k_cfg=self.kv_cfg,
            v_cfg=self.kv_cfg,
            device=self.device,
        )

        x = torch.randn(self.batch_size, self.seq_len, self.d_model)
        y, updated_cache = layer(x, cache=cache)

        self.assertIsNotNone(updated_cache)
        self.assertEqual(updated_cache.pos, self.seq_len)
        self.assertEqual(y.shape, x.shape)

    def test_standard_cache_decode(self) -> None:
        """Standard attention uses cache during decode."""
        cfg = AttentionLayerConfig(d_model=self.d_model, n_heads=self.n_heads)
        layer = AttentionLayer(cfg)
        layer.eval()

        cache = LayerKVCache(
            batch_size=self.batch_size,
            max_seq_len=self.max_seq_len,
            k_dim=self.d_model,
            v_dim=self.d_model,
            k_cfg=self.kv_cfg,
            v_cfg=self.kv_cfg,
            device=self.device,
        )

        # Prefill
        x_prefill = torch.randn(self.batch_size, self.seq_len, self.d_model)
        _, cache = layer(x_prefill, cache=cache)

        # Decode single token
        x_decode = torch.randn(self.batch_size, 1, self.d_model)
        y_decode, updated_cache = layer(x_decode, cache=cache, pos_offset=self.seq_len)

        self.assertIsNotNone(updated_cache)
        self.assertEqual(updated_cache.pos, self.seq_len + 1)
        self.assertEqual(y_decode.shape, (self.batch_size, 1, self.d_model))

    def test_decoupled_cache_prefill(self) -> None:
        """Decoupled attention populates cache during prefill."""
        cfg = AttentionLayerConfig(
            d_model=self.d_model,
            n_heads=self.n_heads,
            mode=AttentionMode.DECOUPLED,
            sem_dim=32,
            geo_dim=32,
            attn_dim=64,
        )
        layer = AttentionLayer(cfg)

        cache = DecoupledLayerKVCache(
            batch_size=self.batch_size,
            max_seq_len=self.max_seq_len,
            k_sem_dim=32,
            k_geo_dim=32,
            v_dim=64,
            k_sem_cfg=self.kv_cfg,
            k_geo_cfg=self.kv_cfg,
            v_cfg=self.kv_cfg,
            device=self.device,
        )

        x = torch.randn(self.batch_size, self.seq_len, self.d_model)
        y, updated_cache = layer(x, cache=cache)

        self.assertIsNotNone(updated_cache)
        self.assertEqual(updated_cache.pos, self.seq_len)
        self.assertEqual(y.shape, x.shape)

    def test_decoupled_cache_decode(self) -> None:
        """Decoupled attention uses cache during decode."""
        cfg = AttentionLayerConfig(
            d_model=self.d_model,
            n_heads=self.n_heads,
            mode=AttentionMode.DECOUPLED,
            sem_dim=32,
            geo_dim=32,
            attn_dim=64,
        )
        layer = AttentionLayer(cfg)
        layer.eval()

        cache = DecoupledLayerKVCache(
            batch_size=self.batch_size,
            max_seq_len=self.max_seq_len,
            k_sem_dim=32,
            k_geo_dim=32,
            v_dim=64,
            k_sem_cfg=self.kv_cfg,
            k_geo_cfg=self.kv_cfg,
            v_cfg=self.kv_cfg,
            device=self.device,
        )

        # Prefill
        x_prefill = torch.randn(self.batch_size, self.seq_len, self.d_model)
        _, cache = layer(x_prefill, cache=cache)

        # Decode
        x_decode = torch.randn(self.batch_size, 1, self.d_model)
        y_decode, updated_cache = layer(x_decode, cache=cache, pos_offset=self.seq_len)

        self.assertIsNotNone(updated_cache)
        self.assertEqual(updated_cache.pos, self.seq_len + 1)
        self.assertEqual(y_decode.shape, (self.batch_size, 1, self.d_model))

    def test_cache_consistency(self) -> None:
        """Cached decode produces same result as full forward."""
        cfg = AttentionLayerConfig(
            d_model=self.d_model, n_heads=self.n_heads, is_causal=True
        )
        layer = AttentionLayer(cfg)
        layer.eval()

        # Full sequence forward
        x_full = torch.randn(self.batch_size, self.seq_len, self.d_model)
        y_full, _ = layer(x_full)

        # Incremental decode
        cache = LayerKVCache(
            batch_size=self.batch_size,
            max_seq_len=self.max_seq_len,
            k_dim=self.d_model,
            v_dim=self.d_model,
            k_cfg=self.kv_cfg,
            v_cfg=self.kv_cfg,
            device=self.device,
        )

        outputs = []
        for t in range(self.seq_len):
            x_t = x_full[:, t : t + 1, :]
            y_t, cache = layer(x_t, cache=cache, pos_offset=t)
            outputs.append(y_t)

        y_incremental = torch.cat(outputs, dim=1)

        # Results should be close (not exact due to floating point accumulation)
        torch.testing.assert_close(y_full, y_incremental, rtol=1e-3, atol=1e-3)

    def test_ctx_overrides_q_chunk_and_local_window(self) -> None:
        """InferContext can override q_chunk/local_window for long prefixes."""
        from caramba.infer.context import InferContext

        cfg = AttentionLayerConfig(d_model=self.d_model, n_heads=self.n_heads, is_causal=True)
        layer = AttentionLayer(cfg)
        layer.eval()

        cache = LayerKVCache(
            batch_size=self.batch_size,
            max_seq_len=self.max_seq_len,
            k_dim=self.d_model,
            v_dim=self.d_model,
            k_cfg=KVCacheTensorConfig(kind=KVCacheKind.FP16, qblock=32, residual_len=0),
            v_cfg=KVCacheTensorConfig(kind=KVCacheKind.FP16, qblock=32, residual_len=0),
            device=self.device,
        )
        ctx = InferContext(caches=[cache], pos_offset=0)
        # Override plan knobs.
        ctx.begin(pos_offset=0, q_chunk=4, local_window=1)
        x = torch.randn(self.batch_size, 8, self.d_model, device=self.device)
        y, _ = layer(x, ctx=ctx)
        self.assertEqual(y.shape, (self.batch_size, 8, self.d_model))


class TestAttentionMemorySummarization(unittest.TestCase):
    """Tests for mem_block memory summarization helpers."""

    def test_mem_block_reduces_kv_length(self) -> None:
        """mem_block summarizes older tokens into fewer memory tokens."""
        cfg = AttentionLayerConfig(
            d_model=32,
            n_heads=4,
            mode=AttentionMode.STANDARD,
            is_causal=True,
            q_chunk=4,
            local_window=4,
            mem_block=2,
            mem_summarize="mean",
        )
        layer = AttentionLayer(cfg)
        B, T, D = 1, 8, 32
        x = torch.randn(B, T, D)

        q = layer.q_proj(x)  # type: ignore[union-attr]
        k = layer.k_proj(x)  # type: ignore[union-attr]
        v = layer.v_proj(x)
        qh = layer._shape(q, layer.head_dim, layer.n_heads)
        kh = layer._shape(k, layer.head_dim, layer.n_kv_heads)
        vh = layer._shape(v, layer.head_dim, layer.n_kv_heads)
        if layer.group_size > 1:
            kh = kh.repeat_interleave(layer.group_size, dim=1)
            vh = vh.repeat_interleave(layer.group_size, dim=1)

        k_pos = torch.arange(T, device=qh.device)
        k2, v2, pos2 = layer._maybe_summarize_kv(k=kh, v=vh, k_pos=k_pos)
        # remote_len=4, mem_block=2 => 2 mem + 4 local = 6
        self.assertEqual(k2.size(2), 6)
        self.assertEqual(v2.size(2), 6)
        self.assertEqual(pos2.numel(), 6)

    def test_activation_threshold_disables_summarization(self) -> None:
        """mem_activation_threshold prevents summarization below the threshold."""
        cfg = AttentionLayerConfig(
            d_model=32,
            n_heads=4,
            mode=AttentionMode.STANDARD,
            is_causal=True,
            local_window=4,
            mem_block=2,
            mem_activation_threshold=999,
        )
        layer = AttentionLayer(cfg)
        B, T, D = 1, 8, 32
        x = torch.randn(B, T, D)
        k = layer.k_proj(x)  # type: ignore[union-attr]
        v = layer.v_proj(x)
        kh = layer._shape(k, layer.head_dim, layer.n_kv_heads)
        vh = layer._shape(v, layer.head_dim, layer.n_kv_heads)
        if layer.group_size > 1:
            kh = kh.repeat_interleave(layer.group_size, dim=1)
            vh = vh.repeat_interleave(layer.group_size, dim=1)
        k_pos = torch.arange(T, device=kh.device)
        k2, v2, pos2 = layer._maybe_summarize_kv(k=kh, v=vh, k_pos=k_pos)
        self.assertEqual(k2.size(2), T)
        self.assertEqual(v2.size(2), T)
        self.assertEqual(pos2.numel(), T)


class TestLearnedTemperature(unittest.TestCase):
    """Tests for learned per-head temperature."""

    def test_learned_temp_parameter(self) -> None:
        """Learned temperature creates per-head parameters."""
        cfg = AttentionLayerConfig(
            d_model=64, n_heads=4, learned_temp=True
        )
        layer = AttentionLayer(cfg)

        self.assertIsNotNone(layer.logit_scale)
        assert layer.logit_scale is not None  # type guard for pyright
        self.assertEqual(layer.logit_scale.shape, (4,))
        self.assertTrue(layer.logit_scale.requires_grad)

    def test_no_learned_temp(self) -> None:
        """No learned temperature by default."""
        cfg = AttentionLayerConfig(d_model=64, n_heads=4)
        layer = AttentionLayer(cfg)

        self.assertIsNone(layer.logit_scale)


class TestConfigProperties(unittest.TestCase):
    """Tests for config computed properties."""

    def test_head_dim_from_d_model(self) -> None:
        """Head dim computed from d_model when attn_dim not set."""
        cfg = AttentionLayerConfig(d_model=64, n_heads=4)
        self.assertEqual(cfg.head_dim, 16)

    def test_head_dim_from_attn_dim(self) -> None:
        """Head dim computed from attn_dim when set."""
        cfg = AttentionLayerConfig(d_model=64, n_heads=4, attn_dim=32)
        self.assertEqual(cfg.head_dim, 8)

    def test_kv_heads_default(self) -> None:
        """KV heads defaults to n_heads."""
        cfg = AttentionLayerConfig(d_model=64, n_heads=4)
        self.assertEqual(cfg.kv_heads, 4)

    def test_kv_heads_explicit(self) -> None:
        """KV heads uses explicit value when set."""
        cfg = AttentionLayerConfig(d_model=64, n_heads=8, n_kv_heads=2)
        self.assertEqual(cfg.kv_heads, 2)

    def test_sem_geo_head_dims(self) -> None:
        """Semantic and geometric head dims computed correctly."""
        cfg = AttentionLayerConfig(
            d_model=64,
            n_heads=4,
            mode=AttentionMode.DECOUPLED,
            sem_dim=32,
            geo_dim=16,
            attn_dim=64,
        )
        self.assertEqual(cfg.sem_head_dim, 8)
        self.assertEqual(cfg.geo_head_dim, 4)


if __name__ == "__main__":
    unittest.main()
