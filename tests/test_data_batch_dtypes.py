import unittest


try:
    import torch  # type: ignore
except Exception as e:  # pragma: no cover
    raise unittest.SkipTest(f"torch is required for these tests but is not available: {e}")


from production.data import TokenView, _TOKENS_INT32_SAFE, get_batch_any


class TestGetBatchAnyDtypes(unittest.TestCase):
    def setUp(self) -> None:
        # Ensure we don't leak global state across tests.
        self._orig = dict(_TOKENS_INT32_SAFE)

    def tearDown(self) -> None:
        _TOKENS_INT32_SAFE.clear()
        _TOKENS_INT32_SAFE.update(self._orig)

    def test_safe_tokens_cast_to_int32(self) -> None:
        toks = torch.arange(0, 512, dtype=torch.long)
        _TOKENS_INT32_SAFE[id(toks)] = True
        view = TokenView(toks, 0, int(toks.numel()))
        x, y = get_batch_any(view, batch_size=2, block_size=16, device=torch.device("cpu"))
        self.assertEqual(x.dtype, torch.int32)
        self.assertEqual(y.dtype, torch.long)
        self.assertEqual(tuple(x.shape), (2, 16))
        self.assertEqual(tuple(y.shape), (2, 16))

    def test_unsafe_tokens_keep_int64(self) -> None:
        toks = torch.arange(0, 512, dtype=torch.long)
        _TOKENS_INT32_SAFE[id(toks)] = False
        view = TokenView(toks, 0, int(toks.numel()))
        x, _y = get_batch_any(view, batch_size=2, block_size=16, device=torch.device("cpu"))
        self.assertEqual(x.dtype, torch.long)

    def test_unknown_safety_infers_from_batch(self) -> None:
        toks = torch.arange(0, 512, dtype=torch.long)
        _TOKENS_INT32_SAFE.pop(id(toks), None)
        view = TokenView(toks, 0, int(toks.numel()))
        x, _y = get_batch_any(view, batch_size=2, block_size=16, device=torch.device("cpu"))
        self.assertEqual(x.dtype, torch.int32)


