import unittest

try:
    import torch  # type: ignore
except Exception as e:  # pragma: no cover
    raise unittest.SkipTest(f"torch is required for these tests but is not available: {e}")

from production.runtime_tuning import KVCachePolicy


class TestKVCachePolicyString(unittest.TestCase):
    def test_roundtrip_short_parse(self) -> None:
        p = KVCachePolicy(
            k_sem_kind="q4_0",
            k_geo_kind="q8_0",
            v_kind="q4_0",
            k_sem_qblock=32,
            k_geo_qblock=64,
            v_qblock=32,
            residual_len=128,
        )
        s = p.short()
        p2 = KVCachePolicy.parse(s)
        self.assertEqual(p2.short(), s)

    def test_parse_accepts_residual_len_alias(self) -> None:
        p = KVCachePolicy.parse("ksem=q4_0@32,kgeo=q8_0@32,v=q4_0@32,residual_len=64")
        self.assertEqual(p.short(), "ksem=q4_0@32,kgeo=q8_0@32,v=q4_0@32,resid=64")

    def test_missing_fields_raises(self) -> None:
        with self.assertRaises(ValueError):
            _ = KVCachePolicy.parse("ksem=q4_0@32,kgeo=q8_0@32,resid=128")


