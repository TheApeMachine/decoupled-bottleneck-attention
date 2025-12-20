import unittest

try:
    import torch  # type: ignore
except Exception as e:  # pragma: no cover
    raise unittest.SkipTest(f"torch is required for these tests but is not available: {e}")

from production.kvcache_backend import dequantize_nf4, dequantize_q4_0, dequantize_q8_0, make_quantspec, quantize_nf4, quantize_q4_0, quantize_q8_0


class TestQuantScaleSentinels(unittest.TestCase):
    def test_q8_scale_never_underflows_to_zero_in_fp16(self) -> None:
        spec = make_quantspec("q8_0", dim=16, qblock=16)
        x = torch.zeros((2, 3, 16), dtype=torch.float16)
        q, s = quantize_q8_0(x, spec)
        self.assertEqual(s.dtype, torch.float16)
        self.assertGreater(float(s.min().item()), 0.0)
        self.assertGreaterEqual(float(s.min().item()), float(torch.finfo(torch.float16).tiny))
        x2 = dequantize_q8_0(q, s, spec)
        self.assertTrue(bool(torch.allclose(x2, torch.zeros_like(x2), atol=0.0, rtol=0.0)))

    def test_q4_scale_never_underflows_to_zero_in_fp16(self) -> None:
        spec = make_quantspec("q4_0", dim=16, qblock=16)
        x = torch.zeros((2, 3, 16), dtype=torch.float16)
        q, s = quantize_q4_0(x, spec)
        self.assertEqual(s.dtype, torch.float16)
        self.assertGreater(float(s.min().item()), 0.0)
        self.assertGreaterEqual(float(s.min().item()), float(torch.finfo(torch.float16).tiny))
        x2 = dequantize_q4_0(q, s, spec)
        self.assertTrue(bool(torch.allclose(x2, torch.zeros_like(x2), atol=0.0, rtol=0.0)))

    def test_nf4_scale_never_underflows_to_zero_in_fp16(self) -> None:
        spec = make_quantspec("nf4", dim=16, qblock=16)
        x = torch.zeros((2, 3, 16), dtype=torch.float16)
        q, s = quantize_nf4(x, spec)
        self.assertEqual(s.dtype, torch.float16)
        self.assertGreater(float(s.min().item()), 0.0)
        self.assertGreaterEqual(float(s.min().item()), float(torch.finfo(torch.float16).tiny))
        x2 = dequantize_nf4(q, s, spec)
        self.assertTrue(bool(torch.allclose(x2, torch.zeros_like(x2), atol=0.0, rtol=0.0)))


