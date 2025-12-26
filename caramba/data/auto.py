"""Automatic dataset selection based on token file format.

Why this exists:
- Caramba experiments may point at `.npy` (preferred) or legacy `.tokens` files.
- Call sites should not have to special-case each format.
"""

from __future__ import annotations

import logging
from pathlib import Path

from torch import Tensor
from torch.utils.data import Dataset

from caramba.data.npy import NpyDataset
from caramba.data.text_tokens import TextTokensDataset

logger = logging.getLogger(__name__)


def build_token_dataset(*, path: str | Path, block_size: int) -> Dataset[tuple[Tensor, Tensor]]:
    """Build a dataset appropriate for the given path."""
    p = Path(path)
    suf = p.suffix.lower()
    if suf == ".npy":
        return NpyDataset(str(p), block_size=int(block_size))
    if suf in (".tokens", ".txt"):
        return TextTokensDataset(str(p), block_size=int(block_size))
    # Default to NPY behavior for unknown suffixes (common in manifests).
    logger.warning("Unexpected file suffix '%s' for %s, defaulting to NpyDataset", suf, p)
    return NpyDataset(str(p), block_size=int(block_size))

