import unittest
from unittest.mock import patch


try:
    import torch  # type: ignore
except Exception as e:  # pragma: no cover
    raise unittest.SkipTest(f"torch is required for these tests but is not available: {e}")


from production.runtime_tuning import KVDecodeSelfOptimizer, KVSelfOptConfig, get_triton_kernel_profiles


class _DummyTensorCfg:
    def __init__(self, kind: str):
        self.kind = kind


class _DummyCache:
    """Minimal cache stub to satisfy KVDecodeSelfOptimizer._allowed_fused_modes and _candidate_plans."""

    def __init__(self, *, k_sem_kind: str, k_geo_kind: str, v_kind: str):
        self.k_sem = _DummyTensorCfg(k_sem_kind)
        self.k_geo = _DummyTensorCfg(k_geo_kind)
        self.v = _DummyTensorCfg(v_kind)


class TestTritonKernelProfiles(unittest.TestCase):
    def test_profiles_off_returns_empty(self) -> None:
        profs = get_triton_kernel_profiles(mode="off", device_sig="cpu", fused="triton1pass", decode_block=1024)
        self.assertEqual(profs, [])

    def test_profiles_small_is_subset_of_auto(self) -> None:
        auto = get_triton_kernel_profiles(mode="auto", device_sig="cpu", fused="triton1pass", decode_block=1024)
        small = get_triton_kernel_profiles(mode="small", device_sig="cpu", fused="triton1pass", decode_block=1024)
        self.assertGreaterEqual(len(auto), 1)
        self.assertGreaterEqual(len(small), 1)
        self.assertLessEqual(len(small), len(auto))


class TestDecodePlanCandidateGeneration(unittest.TestCase):
    def test_profile_mode_generates_smaller_search_space_than_expert(self) -> None:
        # Force allow fused modes even in environments without Triton installed. We never execute kernels here;
        # this only tests plan enumeration.
        cache = _DummyCache(k_sem_kind="q4_0", k_geo_kind="q8_0", v_kind="q4_0")

        cfg_profiles = KVSelfOptConfig(
            mode="startup",
            scope="decode",
            decode_blocks=(256, 512, 1024, 2048),
            block_ns=(64, 128),
            warps=(4, 8),
            stages=(2, 3),
            kernel_profiles="auto",
            expert_launch_space=False,
        )
        cfg_expert = KVSelfOptConfig(
            mode="startup",
            scope="decode",
            decode_blocks=(256, 512, 1024, 2048),
            block_ns=(64, 128),
            warps=(4, 8),
            stages=(2, 3),
            kernel_profiles="auto",
            expert_launch_space=True,
        )

        opt_profiles = KVDecodeSelfOptimizer(cfg_profiles, device=torch.device("cpu"), base_fused="auto", base_decode_block=1024)
        opt_expert = KVDecodeSelfOptimizer(cfg_expert, device=torch.device("cpu"), base_fused="auto", base_decode_block=1024)

        with patch("production.runtime_tuning._triton_decoupled_q4q8q4_available", return_value=True):
            plans_profile = opt_profiles._candidate_plans(cache=cache)
            plans_expert = opt_expert._candidate_plans(cache=cache)

        # In profile mode, the candidate space should be dramatically smaller than the cross-product.
        self.assertGreater(len(plans_profile), 0)
        self.assertGreater(len(plans_expert), len(plans_profile))


