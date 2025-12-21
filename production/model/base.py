"""
base provides the core interface for all neural models.
"""
from __future__ import annotations
import torch.nn as nn

class BaseModel(nn.Module):
    """
    BaseModel defines the contract for all transformer models in the system.
    """
    def __init__(self) -> None:
        """
        init the base model.
        """
        super().__init__()
