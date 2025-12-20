import math
import unittest

try:
    import torch  # type: ignore
    import torch.nn.functional as F  # type: ignore
except Exception as e:  # pragma: no cover
    raise unittest.SkipTest(f"torch is required for these tests but is not available: {e}")

from production.attention import _decoupled_qk_cat, _decoupled_scores_f32


class TestDecoupledScoreEquivalence(unittest.TestCase):
    def test_cat_dot_product_matches_sem_plus_geo_scores(self) -> None:
        torch.manual_seed(0)
        B, H, Tq, Tk = 2, 3, 4, 5
        sem_hd, geo_hd = 6, 8
        q_sem = torch.randn((B, H, Tq, sem_hd), dtype=torch.float32)
        q_geo = torch.randn((B, H, Tq, geo_hd), dtype=torch.float32)
        k_sem = torch.randn((B, H, Tk, sem_hd), dtype=torch.float32)
        k_geo = torch.randn((B, H, Tk, geo_hd), dtype=torch.float32)
        sem_scale = 1.0 / math.sqrt(float(sem_hd))
        geo_scale = 1.0 / math.sqrt(float(geo_hd))

        scores = _decoupled_scores_f32(q_sem=q_sem, q_geo=q_geo, k_sem=k_sem, k_geo=k_geo, sem_scale=sem_scale, geo_scale=geo_scale)
        q_cat, k_cat = _decoupled_qk_cat(q_sem=q_sem, q_geo=q_geo, k_sem=k_sem, k_geo=k_geo, sem_scale=sem_scale, geo_scale=geo_scale)
        scores2 = torch.matmul(q_cat, k_cat.transpose(-2, -1)).to(torch.float32)

        mx = float((scores - scores2).abs().max().item())
        self.assertLess(mx, 1e-5)

    def test_sdpa_scale_cancellation_matches_manual_attention(self) -> None:
        # Validate the SDPA usage pattern we rely on:
        # pre-scale q by sqrt(dk) to cancel SDPA's implicit 1/sqrt(dk) scaling (i.e., effective scale=1.0).
        torch.manual_seed(0)
        B, H, Tq, Tk = 1, 2, 1, 7
        d = 16
        q = torch.randn((B, H, Tq, d), dtype=torch.float32)
        k = torch.randn((B, H, Tk, d), dtype=torch.float32)
        v = torch.randn((B, H, Tk, 12), dtype=torch.float32)

        dk = int(q.size(-1))
        out_sdpa = F.scaled_dot_product_attention(q * math.sqrt(dk), k, v, attn_mask=None, dropout_p=0.0, is_causal=False)

        scores = torch.matmul(q, k.transpose(-2, -1))
        attn = torch.softmax(scores, dim=-1)
        out_manual = torch.matmul(attn, v)

        mx = float((out_sdpa - out_manual).abs().max().item())
        self.assertLess(mx, 5e-5)


