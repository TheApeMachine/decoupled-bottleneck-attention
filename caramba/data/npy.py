"""NPY dataset for preprocessed token data.

Training on raw text is slow because tokenization happens on-the-fly. For
efficiency, we preprocess text into token IDs and save them as .npy files.
This dataset loads those files and serves fixed-length token blocks for
next-token prediction training.
"""
from __future__ import annotations

import importlib
import importlib.util
import warnings
from typing import cast

import torch
from torch import Tensor
from torch.utils.data import Dataset
from typing_extensions import override


_INT32_MAX = 2**31 - 1


def _validate_tokens_int32_safe(t: Tensor) -> None:
    """Validate that token IDs are non-negative and fit in int32.

    Why this exists:
    - We store tokens as int64 in PyTorch, but many kernels/tools assume int32-safe IDs.
    - Scanning multi-billion-token arrays is expensive, so we sample.
    """

    if t.numel() <= 0:
        raise ValueError("Token tensor is empty.")

    sample_n = min(int(t.numel()), 100_000)
    if sample_n == int(t.numel()):
        sample = t
    else:
        idx = torch.randint(0, int(t.numel()), (sample_n,), device=t.device)
        sample = t.view(-1).index_select(0, idx)

    mn = int(sample.min().item())
    mx = int(sample.max().item())
    if mn < 0:
        raise ValueError(f"Token IDs must be non-negative, found min={mn}")
    if mx > _INT32_MAX:
        raise ValueError(f"Token IDs must fit in int32, found max={mx}")


def _require_numpy() -> object:
    """Import numpy, raising a clear error if missing."""
    if importlib.util.find_spec("numpy") is None:
        raise ImportError("numpy is required for NpyDataset")
    return importlib.import_module("numpy")


def _np_attr(np_mod: object, name: str) -> object:
    """Get an attribute from the numpy module."""
    return getattr(np_mod, name, None)


def _np_call(np_mod: object, name: str, *args: object, **kwargs: object) -> object:
    """Call a numpy function by name."""
    fn = _np_attr(np_mod, name)
    if not callable(fn):
        raise AttributeError(f"numpy.{name} is not callable")
    return fn(*args, **kwargs)


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
        # Use mmap_mode="r" to avoid materializing large arrays into RAM.
        arr_obj = _np_call(np_mod, "load", str(path), mmap_mode="r")
        ndarray_t = _np_attr(np_mod, "ndarray")
        if not isinstance(ndarray_t, type) or not isinstance(arr_obj, ndarray_t):
            raise TypeError("Expected numpy.load to return a numpy ndarray/memmap")

        reshape = getattr(arr_obj, "reshape", None)
        if not callable(reshape):
            raise TypeError("Expected numpy array to support .reshape(...)")
        arr = reshape(-1)

        # Suppress PyTorch's warning about non-writable tensors. The warning is
        # about potential undefined behavior if the tensor is written to, but
        # this dataset is read-only (we only slice in __getitem__). Copying
        # would defeat memory-mapping for datasets that don't fit in RAM.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=".*non-writable.*",
                category=UserWarning,
            )
            t = cast(Tensor, torch.from_numpy(arr)).to(dtype=torch.long)

        if len(t) <= int(block_size):
            raise ValueError(
                f"block_size must be smaller than data length, got "
                f"block_size={block_size}, len={len(t)}"
            )

        _validate_tokens_int32_safe(t)

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
