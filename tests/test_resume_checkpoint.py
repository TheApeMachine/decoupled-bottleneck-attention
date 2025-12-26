import unittest


class TestResumeCheckpoint(unittest.TestCase):
    @unittest.skip(
        "Legacy main.py CLI checkpoint contract is no longer supported in the caramba entrypoint. "
        "Checkpoint/resume behavior is covered by caramba/trainer/upcycle_checkpoint_test.py."
    )
    def test_resume_advances_opt_step(self) -> None:
        pass  # Test body removed; see caramba/trainer/upcycle_checkpoint_test.py for coverage.


if __name__ == "__main__":
    unittest.main()
