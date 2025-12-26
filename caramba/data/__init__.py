"""Dataset utilities for training data loading.

Training requires feeding the model sequences of tokens. This package
provides Dataset implementations that load preprocessed token data
and serve it in the (input, target) pairs needed for language modeling.
"""
from __future__ import annotations

from caramba.data.npy import NpyDataset

__all__ = ["NpyDataset"]
