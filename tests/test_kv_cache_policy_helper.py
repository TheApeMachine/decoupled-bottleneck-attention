import io
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

try:
    import torch  # type: ignore
except Exception as e:  # pragma: no cover
    raise unittest.SkipTest(f"torch is required for these tests but is not available: {e}")

from production.kvcache_backend import KVCacheTensorConfig
from production.model import GPT, ModelConfig
from production.runtime_tuning import KVCachePolicy, KVSelfOptConfig


def _make_small_gpt(*, n_layer: int = 4) -> GPT:
    cfg = ModelConfig(
        vocab_size=64,
        block_size=64,
        n_layer=int(n_layer),
        n_head=2,
        d_model=16,
        d_ff=32,
        embed_dim=16,
        attn_mode="decoupled",
        attn_dim=16,
        sem_dim=8,
        geo_dim=8,
        rope=False,
        learned_temp=False,
        dropout=0.0,
    )
    return GPT(cfg)


class _StubPolicyTuner:
    """Stand-in for KVCachePolicySelfOptimizer that returns a pre-configured policy."""

    policy_to_return: KVCachePolicy

    def __init__(self, *args, **kwargs):
        pass

    def choose_policy(self, *, prompt_len: int) -> KVCachePolicy:
        return type(self).policy_to_return


class TestChooseKVCachePolicyHelper(unittest.TestCase):
    def test_no_selfopt_returns_inputs(self) -> None:
        m = _make_small_gpt()
        prompt = torch.zeros((1, 8), dtype=torch.long)
        k_sem = KVCacheTensorConfig(kind="q4_0", qblock=32, residual_len=128)
        k_geo = KVCacheTensorConfig(kind="q8_0", qblock=32, residual_len=128)
        v_dec = KVCacheTensorConfig(kind="q4_0", qblock=32, residual_len=128)

        out = m._choose_kv_cache_policy(
            model=m,
            self_opt=None,
            device=prompt.device,
            prompt=prompt,
            k_sem_cfg=k_sem,
            k_geo_cfg=k_geo,
            v_dec_cfg=v_dec,
            kv_residual=128,
            kv_decode_block=1024,
            kv_fused="auto",
            max_new_tokens=16,
            is_speculative=False,
        )
        k_sem2, k_geo2, v_dec2, promote, kv_res2 = out
        self.assertEqual((k_sem2.kind, k_sem2.qblock, k_sem2.residual_len), (k_sem.kind, k_sem.qblock, k_sem.residual_len))
        self.assertEqual((k_geo2.kind, k_geo2.qblock, k_geo2.residual_len), (k_geo.kind, k_geo.qblock, k_geo.residual_len))
        self.assertEqual((v_dec2.kind, v_dec2.qblock, v_dec2.residual_len), (v_dec.kind, v_dec.qblock, v_dec.residual_len))
        self.assertIsNone(promote)
        self.assertEqual(kv_res2, 128)

    def test_reject_falls_back_to_base_policy(self) -> None:
        m = _make_small_gpt(n_layer=2)
        prompt = torch.zeros((1, 8), dtype=torch.long)
        k_sem = KVCacheTensorConfig(kind="q4_0", qblock=32, residual_len=128)
        k_geo = KVCacheTensorConfig(kind="q8_0", qblock=32, residual_len=128)
        v_dec = KVCacheTensorConfig(kind="q4_0", qblock=32, residual_len=128)

        base_policy = KVCachePolicy(
            k_sem_kind=k_sem.kind,
            k_geo_kind=k_geo.kind,
            v_kind=v_dec.kind,
            k_sem_qblock=k_sem.qblock,
            k_geo_qblock=k_geo.qblock,
            v_qblock=v_dec.qblock,
            residual_len=128,
        )
        chosen_policy = KVCachePolicy(
            k_sem_kind="fp16",
            k_geo_kind="fp16",
            v_kind="fp16",
            k_sem_qblock=32,
            k_geo_qblock=32,
            v_qblock=32,
            residual_len=0,
        )
        _StubPolicyTuner.policy_to_return = chosen_policy

        self_opt = KVSelfOptConfig(mode="startup", scope="cache", policy_quality=True, layerwise_cache=False)

        with (
            patch("production.model.KVCachePolicySelfOptimizer", _StubPolicyTuner),
            patch.object(m, "_policy_quality_metrics_decoupled", return_value={"max_abs_logit": 999.0}),
            patch("production.model.policy_quality_reject_reasons", return_value=["forced_reject"]),
            patch("production.model.warn_policy_quality_reject") as warn_mock,
        ):
            k_sem2, k_geo2, v_dec2, promote, kv_res2 = m._choose_kv_cache_policy(
                model=m,
                self_opt=self_opt,
                device=prompt.device,
                prompt=prompt,
                k_sem_cfg=k_sem,
                k_geo_cfg=k_geo,
                v_dec_cfg=v_dec,
                kv_residual=128,
                kv_decode_block=1024,
                kv_fused="auto",
                max_new_tokens=16,
                is_speculative=False,
            )

        exp_k_sem, exp_k_geo, exp_v = base_policy.to_tensor_cfgs()
        self.assertEqual((k_sem2.kind, k_sem2.qblock, k_sem2.residual_len), (exp_k_sem.kind, exp_k_sem.qblock, exp_k_sem.residual_len))
        self.assertEqual((k_geo2.kind, k_geo2.qblock, k_geo2.residual_len), (exp_k_geo.kind, exp_k_geo.qblock, exp_k_geo.residual_len))
        self.assertEqual((v_dec2.kind, v_dec2.qblock, v_dec2.residual_len), (exp_v.kind, exp_v.qblock, exp_v.residual_len))
        self.assertIsNone(promote)
        self.assertEqual(kv_res2, 128)
        warn_mock.assert_called()

    def test_layerwise_speculative_print_and_promote(self) -> None:
        m = _make_small_gpt(n_layer=4)
        prompt = torch.zeros((1, 8), dtype=torch.long)
        k_sem = KVCacheTensorConfig(kind="q4_0", qblock=32, residual_len=128)
        k_geo = KVCacheTensorConfig(kind="q8_0", qblock=32, residual_len=128)
        v_dec = KVCacheTensorConfig(kind="q4_0", qblock=32, residual_len=128)

        chosen_policy = KVCachePolicy(
            k_sem_kind="q4_0",
            k_geo_kind="q8_0",
            v_kind="q4_0",
            k_sem_qblock=32,
            k_geo_qblock=32,
            v_qblock=32,
            residual_len=64,
        )
        _StubPolicyTuner.policy_to_return = chosen_policy

        self_opt = KVSelfOptConfig(mode="startup", scope="cache", policy_quality=True, layerwise_cache=True)

        call_count = {"n": 0}

        def _reject_side_effect(*args, **kwargs):
            call_count["n"] += 1
            # First call (base policy) rejects, subsequent (layerwise candidates) accept.
            return ["reject"] if call_count["n"] == 1 else []

        buf = io.StringIO()
        with (
            patch("production.model.KVCachePolicySelfOptimizer", _StubPolicyTuner),
            patch.object(m, "_policy_quality_metrics_decoupled", return_value={"max_abs_logit": 999.0}),
            patch.object(m, "_policy_quality_metrics_decoupled_layerwise", return_value={"max_abs_logit": 0.0}),
            patch("production.model.policy_quality_reject_reasons", side_effect=_reject_side_effect),
            redirect_stdout(buf),
        ):
            k_sem2, k_geo2, v_dec2, promote, kv_res2 = m._choose_kv_cache_policy(
                model=m,
                self_opt=self_opt,
                device=prompt.device,
                prompt=prompt,
                k_sem_cfg=k_sem,
                k_geo_cfg=k_geo,
                v_dec_cfg=v_dec,
                kv_residual=128,
                kv_decode_block=1024,
                kv_fused="auto",
                max_new_tokens=16,
                is_speculative=True,
            )

        out = buf.getvalue()
        self.assertIn("[selfopt] layerwise cache-policy enabled for speculative decode:", out)
        self.assertEqual(promote, 1)
        self.assertEqual(kv_res2, 64)
        exp_k_sem, exp_k_geo, exp_v = chosen_policy.to_tensor_cfgs()
        self.assertEqual((k_sem2.kind, k_sem2.qblock, k_sem2.residual_len), (exp_k_sem.kind, exp_k_sem.qblock, exp_k_sem.residual_len))
        self.assertEqual((k_geo2.kind, k_geo2.qblock, k_geo2.residual_len), (exp_k_geo.kind, exp_k_geo.qblock, exp_k_geo.residual_len))
        self.assertEqual((v_dec2.kind, v_dec2.qblock, v_dec2.residual_len), (exp_v.kind, exp_v.qblock, exp_v.residual_len))


if __name__ == "__main__":
    unittest.main()


