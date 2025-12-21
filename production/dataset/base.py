"""
base holds the base class for all datasets.
"""

from typing import Any

import torch


class Dataset(torch.utils.data.Dataset):
    """
    Dataset is a base class that provides a base implementation for a dataset.
    """

    def __init__(self, data: Any, block_size: int):
        self.data = data
        self.block_size = block_size

    def __len__(self) -> int:
        return len(self.data) - self.block_size + 1

    def __getitem__(self, idx: int) -> Any:
        return self.data[idx : idx + self.block_size]
