"""layer provides layer modules."""

from __future__ import annotations

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from caramba.config.layer import LayerConfig


class Layer(nn.Module):
    """Layer provides a base class for all layer modules."""
    def __init__(self, config: LayerConfig) -> None:
        super().__init__()
        self.config: LayerConfig = config

    def forward(self, x: Tensor, mask=None) -> Tensor:
        """Forward pass for the layer."""
        raise NotImplementedError("Subclasses must implement forward pass.")

    def cross_attention(self, Q, K, V, mask=None) -> tuple[Tensor, Tensor]:
        # Compute the dot products between Q and K, then scale
        d_k = Q.size(-1)
        scores = torch.matmul(
            Q, K.transpose(-2, -1)
        ) / torch.sqrt(
            torch.tensor(d_k, dtype=torch.float32)
        )

        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # Softmax to normalize scores and get attention weights
        attention_weights = F.softmax(scores, dim=-1)

        # Weighted sum of values
        output = torch.matmul(attention_weights, V)
        return output, attention_weights