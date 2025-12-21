"""Dataset helpers (legacy; prefer `production.data` for training pipelines)."""

from __future__ import annotations

from production.dataset.base import Dataset
from production.dataset.npy import NPYDataset

__all__ = [
    "Dataset",
    "NPYDataset",
]
