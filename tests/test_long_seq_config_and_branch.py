import unittest

try:
    import torch  # type: ignore
except Exception as e:  # pragma: no cover
    raise unittest.SkipTest(f"torch is required for these tests but is not available: {e}")

from production.attention_impl.decoupled_attention_impl.attention_core import DecoupledBottleneckAttention
from production.model import ModelConfig


class TestLongSeqConfigAndBranch(unittest.TestCase):
    def test_model_config_from_dict_includes_long_seq_fields(self) -> None:
        cfg = ModelConfig.from_dict(
            {
                "vocab_size": 128,
                "block_size": 32,
                "n_layer": 1,
                "n_head": 4,
                "d_model": 32,
                "d_ff": 64,
                "embed_dim": 32,
                "attn_mode": "decoupled",
                "attn_dim": 32,
                "sem_dim": 16,
                "geo_dim": 16,
                "rope": False,
                "learned_temp": False,
                "dropout": 0.0,
                "null_attn": False,
                "tie_qk": True,
                "train_long_seq_enabled": False,
                "train_long_seq_threshold": 123,
                "train_long_seq_mem_block": 9,
                "train_long_seq_local_window": 456,
                "train_long_seq_q_chunk": 7,
                "train_long_seq_mem_summarizer": "conv",
            }
        )
        self.assertFalse(bool(cfg.train_long_seq_enabled))
        self.assertEqual(cfg.train_long_seq_threshold, 123)
        self.assertEqual(cfg.train_long_seq_mem_block, 9)
        self.assertEqual(cfg.train_long_seq_local_window, 456)
        self.assertEqual(cfg.train_long_seq_q_chunk, 7)
        self.assertEqual(cfg.train_long_seq_mem_summarizer, "conv")

    def test_long_seq_branch_uses_chunked_sdpa_when_enabled(self) -> None:
        cfg = ModelConfig(
            vocab_size=128,
            block_size=32,
            n_layer=1,
            n_head=4,
            d_model=32,
            d_ff=64,
            embed_dim=32,
            attn_mode="decoupled",
            attn_dim=32,
            sem_dim=16,
            geo_dim=16,
            rope=False,
            learned_temp=False,
            dropout=0.0,
            null_attn=False,
            tie_qk=True,
            train_long_seq_enabled=True,
            train_long_seq_threshold=0,
            train_long_seq_mem_block=2,
            train_long_seq_local_window=4,
            train_long_seq_q_chunk=2,
        )

        attn = DecoupledBottleneckAttention(cfg).train()
        call_count = 0
        original_sdp = attn._sdp

        def counted_sdp(
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            attn_mask: torch.Tensor | None,
            *,
            scale: float | None = None,
            is_causal: bool | None = None,
        ) -> torch.Tensor:
            nonlocal call_count
            call_count += 1
            return original_sdp(q, k, v, attn_mask, scale=scale, is_causal=is_causal)

        setattr(attn, "_sdp", counted_sdp)

        x = torch.randn(2, 7, cfg.d_model)
        y, cache = attn(x, attn_mask=None, cache=None, pos_offset=0)
        self.assertIsNone(cache)
        self.assertEqual(tuple(y.shape), (2, 7, cfg.d_model))
        self.assertGreater(call_count, 1)

    def test_learned_mem_summarizers_match_mean_at_init(self) -> None:
        cfg = ModelConfig(
            vocab_size=128,
            block_size=32,
            n_layer=1,
            n_head=4,
            d_model=32,
            d_ff=64,
            embed_dim=32,
            attn_mode="decoupled",
            attn_dim=32,
            sem_dim=16,
            geo_dim=16,
            rope=False,
            learned_temp=False,
            dropout=0.0,
            null_attn=False,
            tie_qk=True,
            train_long_seq_enabled=True,
            train_long_seq_threshold=0,
            train_long_seq_mem_block=2,
            train_long_seq_local_window=4,
            train_long_seq_q_chunk=2,
        )
        attn = DecoupledBottleneckAttention(cfg).train()
        x = torch.randn(2, 7, cfg.d_model)

        cfg.train_long_seq_mem_summarizer = "mean"
        y_mean, _ = attn(x, attn_mask=None, cache=None, pos_offset=0)

        cfg.train_long_seq_mem_summarizer = "linear"
        y_linear, _ = attn(x, attn_mask=None, cache=None, pos_offset=0)

        cfg.train_long_seq_mem_summarizer = "conv"
        y_conv, _ = attn(x, attn_mask=None, cache=None, pos_offset=0)

        self.assertTrue(bool(torch.allclose(y_mean, y_linear, rtol=0.0, atol=0.0)))
        self.assertTrue(bool(torch.allclose(y_mean, y_conv, rtol=0.0, atol=0.0)))


if __name__ == "__main__":
    unittest.main()
