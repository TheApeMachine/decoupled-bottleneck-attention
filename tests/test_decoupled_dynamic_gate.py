import unittest

try:
    import torch  # type: ignore
except Exception as e:  # pragma: no cover
    raise unittest.SkipTest(f"torch is required for these tests but is not available: {e}")

from production.attention_impl.decoupled_attention_impl.attention_core import DecoupledBottleneckAttention
from production.model import ModelConfig


class TestDecoupledDynamicGate(unittest.TestCase):
    def _cfg(self) -> ModelConfig:
        return ModelConfig(
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
            decoupled_gate=True,
            decoupled_gate_dynamic=True,
        )

    def test_gate_is_neutral_at_init(self) -> None:
        attn = DecoupledBottleneckAttention(self._cfg()).eval()
        x = torch.randn(2, 5, attn.cfg.d_model, dtype=torch.float32)
        g = attn._decoupled_gate(x)
        self.assertIsNotNone(g)
        self.assertTrue(bool(torch.allclose(g, g.new_full(g.shape, 0.5), rtol=0.0, atol=0.0)))

    def test_gate_is_token_local(self) -> None:
        attn = DecoupledBottleneckAttention(self._cfg()).eval()
        proj = attn.decoupled_gate_proj
        self.assertIsNotNone(proj)
        assert proj is not None
        with torch.no_grad():
            proj.weight.zero_()
            proj.weight[0, 0] = 1.0

        x1 = torch.zeros(1, 3, attn.cfg.d_model, dtype=torch.float32)
        x2 = x1.clone()
        x2[0, 1, 0] = 1.0

        g1 = attn._decoupled_gate(x1)
        g2 = attn._decoupled_gate(x2)
        self.assertIsNotNone(g1)
        self.assertIsNotNone(g2)
        assert g1 is not None
        assert g2 is not None

        self.assertTrue(bool(torch.allclose(g1[:, :, 0, :], g2[:, :, 0, :], rtol=0.0, atol=0.0)))
        self.assertTrue(bool(torch.allclose(g1[:, :, 2, :], g2[:, :, 2, :], rtol=0.0, atol=0.0)))
        self.assertFalse(bool(torch.allclose(g1[:, 0:1, 1, :], g2[:, 0:1, 1, :], rtol=0.0, atol=0.0)))
        self.assertTrue(bool(torch.allclose(g1[:, 1:, 1, :], g2[:, 1:, 1, :], rtol=0.0, atol=0.0)))


if __name__ == "__main__":
    unittest.main()

