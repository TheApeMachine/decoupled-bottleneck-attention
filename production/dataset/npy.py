"""
npy holds the dataset for npy files.
"""

from __future__ import annotations

from typing import cast

import numpy as np

from production.dataset.base import Dataset


class NPYDataset(Dataset):
    """
    NPYDataset is a dataset for npy files.
    """

    def __init__(self, path: str, block_size: int):
        data = cast(np.ndarray, np.load(path, mmap_mode="r"))
        super().__init__(data, block_size)
