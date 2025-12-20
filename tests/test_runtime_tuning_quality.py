import math
import unittest

try:
    import torch  # type: ignore
except Exception as e:  # pragma: no cover
    raise unittest.SkipTest(f"torch is required for these tests but is not available: {e}")

from production.runtime_tuning import policy_quality_reject_reasons


class TestPolicyQualityRejectReasons(unittest.TestCase):
    def test_rejects_on_max_abs_logit(self) -> None:
        metrics = {"max_abs_logit": 0.6}
        reasons = policy_quality_reject_reasons(
            metrics,
            max_abs_logit_tol=0.5,
            delta_nll_tol=None,
            ppl_ratio_tol=None,
            kl_tol=None,
        )
        self.assertTrue(any("max_abs_logit" in r for r in reasons))

    def test_rejects_on_delta_nll_and_ppl_ratio(self) -> None:
        metrics = {"delta_nll": 0.03, "ppl_ratio": math.exp(0.03)}
        reasons = policy_quality_reject_reasons(
            metrics,
            max_abs_logit_tol=None,
            delta_nll_tol=0.02,
            ppl_ratio_tol=1.02,
            kl_tol=None,
        )
        self.assertTrue(any("ΔNLL" in r for r in reasons))
        self.assertTrue(any("ppl_ratio" in r for r in reasons))

    def test_ignores_nan_metrics(self) -> None:
        metrics = {"delta_nll": float("nan"), "ppl_ratio": float("nan"), "kl_base_cand": float("nan"), "max_abs_logit": float("nan")}
        reasons = policy_quality_reject_reasons(
            metrics,
            max_abs_logit_tol=0.5,
            delta_nll_tol=0.02,
            ppl_ratio_tol=1.02,
            kl_tol=0.1,
        )
        self.assertEqual(reasons, [])


