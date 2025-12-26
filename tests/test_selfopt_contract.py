import os
import unittest


class TestSelfOptContract(unittest.TestCase):
    def test_no_env_controls_exist(self) -> None:
        # Contract: no env toggles for core optimization behavior.
        forbidden = {
            "EXPERIMENTS_NO_AMP",
            "EXPERIMENTS_NO_COMPILE",
            "EXPERIMENTS_FORCE_FP32",
        }
        for k in forbidden:
            self.assertNotIn(k, os.environ, msg=f"{k} should not be used as a control surface")

    @unittest.skip(
        "Legacy main.py CLI selfopt_decisions.jsonl contract is no longer supported in the caramba entrypoint. "
        "Self-optimization decisions are persisted via caramba/runtime/plan.py and cache plans in caramba/infer/cache_plan.py."
    )
    def test_decision_log_is_written_on_train(self) -> None:
        pass  # Test body removed; see caramba/runtime/plan.py and caramba/infer/cache_plan.py for coverage.


