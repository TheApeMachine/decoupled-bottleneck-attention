import tempfile
import unittest
from pathlib import Path


class TestTokenIdInt32Validation(unittest.TestCase):
    def _load_tokens_any(self):
        try:
            from production.data import load_tokens_any  # type: ignore
        except Exception as e:  # pragma: no cover
            self.skipTest(f"production.data (and its deps like torch) not available: {e}")
        return load_tokens_any

    def _np(self):
        try:
            import numpy as np  # type: ignore
        except Exception as e:  # pragma: no cover
            self.skipTest(f"numpy is required for these tests but is not available: {e}")
        return np

    def test_rejects_negative_token_ids(self) -> None:
        np = self._np()
        load_tokens_any = self._load_tokens_any()
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "bad_neg.npy"
            np.save(str(p), np.asarray([0, 1, -1, 2], dtype=np.int64))
            with self.assertRaisesRegex(ValueError, r"Invalid token ids.*min=-1"):
                load_tokens_any(path=p, fmt="npy", data_dtype="int64")

    def test_rejects_token_ids_ge_int32_max_exclusive(self) -> None:
        np = self._np()
        load_tokens_any = self._load_tokens_any()
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "bad_big.npy"
            np.save(str(p), np.asarray([0, 1, 2_147_483_647], dtype=np.int64))
            with self.assertRaisesRegex(ValueError, r"Invalid token ids.*max=2147483647"):
                load_tokens_any(path=p, fmt="npy", data_dtype="int64")


if __name__ == "__main__":
    unittest.main()


