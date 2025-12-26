"""Neural network layers: the building blocks of transformer models.

Each layer type (attention, MLP, normalization, etc.) is defined here as a
configurable nn.Module. The model's topology is composed from these layers
based on the YAML configuration, allowing flexible architecture experiments.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from caramba.config.layer import LayerConfig


class Layer(nn.Module):
    """Base class for all layer modules.

    Provides a common interface with config storage and optional mask support.
    Subclasses implement the actual forward logic.
    """

    def __init__(self, config: LayerConfig) -> None:
        super().__init__()
        self.config = config

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """Forward pass, to be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement forward pass.")

    def cross_attention(
        self,
        Q: Tensor,
        K: Tensor,
        V: Tensor,
        mask: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor]:
        """Compute cross-attention between queries and key-value pairs.

        This is a utility method for layers that need attention computation
        without the full AttentionLayer machinery.
        """
        d_k = Q.size(-1)
        scale = math.sqrt(float(d_k))
        scores = torch.matmul(Q, K.transpose(-2, -1)) / scale

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attention_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, V)
        return output, attention_weights
