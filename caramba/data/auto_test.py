from __future__ import annotations

from pathlib import Path

import torch

from caramba.data.auto import build_token_dataset
from caramba.data.text_tokens import TextTokensDataset


def test_build_token_dataset_tokens_file(tmp_path: Path) -> None:
    p = tmp_path / "tiny.tokens"
    p.write_text("1 2 3 4 5 6 7 8 9 10", encoding="utf-8")
    ds = build_token_dataset(path=p, block_size=4)
    assert isinstance(ds, TextTokensDataset)
    x, y = ds[0]
    assert torch.equal(x, torch.tensor([1, 2, 3, 4], dtype=torch.long))
    assert torch.equal(y, torch.tensor([2, 3, 4, 5], dtype=torch.long))

