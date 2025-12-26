"""NPY dataset for preprocessed token data.

Training on raw text is slow because tokenization happens on-the-fly. For
efficiency, we preprocess text into token IDs and save them as .npy files.
This dataset loads those files and serves fixed-length token blocks for
next-token prediction training.
"""
from __future__ import annotations

import importlib
import importlib.util
from typing import cast

import torch
from torch import Tensor
from torch.utils.data import Dataset
from typing_extensions import override


def _require_numpy() -> object:
    """Import numpy, raising a clear error if missing."""
    if importlib.util.find_spec("numpy") is None:
        raise ImportError("numpy is required for NpyDataset")
    return importlib.import_module("numpy")


def _np_attr(np_mod: object, name: str) -> object:
    """Get an attribute from the numpy module."""
    return getattr(np_mod, name, None)


def _np_call(np_mod: object, name: str, *args: object) -> object:
    """Call a numpy function by name."""
    fn = _np_attr(np_mod, name)
    if not callable(fn):
        raise AttributeError(f"numpy.{name} is not callable")
    return fn(*args)


class NpyDataset(Dataset[tuple[Tensor, Tensor]]):
    """Dataset that loads preprocessed tokens from a .npy file.

    The file should contain a 1D array of token IDs. The dataset serves
    fixed-length blocks where each sample is (x, y) with y being the
    next-token shift of x—the standard format for language modeling.
    """

    def __init__(self, path: str, *, block_size: int) -> None:
        """Load tokens from a .npy file.

        Args:
            path: Path to the .npy file containing token IDs
            block_size: Length of token sequences to serve (context length)

        The file is memory-mapped for efficiency with large datasets.
        """
        if block_size <= 0:
            raise ValueError(f"block_size must be > 0, got {block_size}")

        np_mod = _require_numpy()
        arr_obj = _np_call(np_mod, "load", str(path))
        ndarray_t = _np_attr(np_mod, "ndarray")
        if not isinstance(ndarray_t, type) or not isinstance(arr_obj, ndarray_t):
            raise TypeError("Expected numpy.load to return a numpy ndarray/memmap")

        reshape = getattr(arr_obj, "reshape", None)
        if not callable(reshape):
            raise TypeError("Expected numpy array to support .reshape(...)")
        arr = reshape(-1)
        t = cast(Tensor, torch.from_numpy(arr)).to(dtype=torch.long)

        # Clone if the array is read-only (memory-mapped)
        flags = getattr(arr, "flags", None)
        writeable = bool(getattr(flags, "writeable", True))
        if not writeable:
            t = t.clone()

        if len(t) <= int(block_size):
            raise ValueError(
                f"block_size must be smaller than data length, got "
                f"block_size={block_size}, len={len(t)}"
            )

        self.tokens = t
        self.block_size = int(block_size)

    def __len__(self) -> int:
        """Return the number of possible starting positions.

        We can start a block at any position from 0 to len(tokens) - block_size - 1.
        """
        return len(self.tokens) - self.block_size

    @override
    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        """Get a (input, target) pair for language modeling.

        Returns x[0:block_size] and y[1:block_size+1] where y is the
        next-token target for each position in x.
        """
        block = self.tokens[idx : idx + self.block_size + 1]
        x = block[:-1]
        y = block[1:]
        return x, y
