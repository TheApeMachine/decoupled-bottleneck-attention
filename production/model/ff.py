"""
ff defines the feed-forward MLP for transformer blocks.
"""
from __future__ import annotations
from typing import Protocol, cast
import torch
import torch.nn as nn
from typing_extensions import override

class FeedForwardCfg(Protocol):
    """Structural config required by FeedForward (keeps typing local and strict)."""
    mlp: str
    d_model: int
    d_ff: int
    dropout: float

class FeedForward(nn.Module):
    """Feed-forward MLP used inside a transformer block."""
    def __init__(self, cfg: FeedForwardCfg) -> None:
        """
        init the feed forward layer.
        """
        super().__init__()  # pyright: ignore[reportUnknownMemberType]
        self.cfg: FeedForwardCfg = cfg
        self.drop: nn.Dropout = nn.Dropout(cfg.dropout)
        self.fc1: nn.Linear
        self.fc2: nn.Linear | None
        self.proj: nn.Linear
        self.act: nn.Module

        match cfg.mlp:
            case "swiglu":
                # SWiGLU: proj(silu(fc1(x)) * fc2(x))
                self.make_linear(cfg.d_model, cfg.d_ff, False, nn.SiLU())
            case "gelu":
                # GELU: proj(gelu(fc1(x)))
                self.make_linear(cfg.d_model, cfg.d_ff, False, nn.GELU())
            case _:
                raise ValueError(f"Unknown MLP type: {cfg.mlp}")

    def make_linear(
        self, left_dim: int, right_dim: int, bias: bool, act: nn.Module
    ) -> None:
        """Make a linear layer with the given dimensions and bias."""
        self.fc1 = nn.Linear(left_dim, right_dim, bias=bias)

        if act == nn.SiLU():
            self.fc2 = nn.Linear(left_dim, right_dim, bias=bias)
        else:
            self.fc2 = None

        self.proj = nn.Linear(right_dim, left_dim, bias=bias)
        self.act = act

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the MLP."""
        h = cast(torch.Tensor, self.act(cast(torch.Tensor, self.fc1(x))))
        if self.fc2 is not None:
            h = h * cast(torch.Tensor, self.fc2(x))
        return cast(torch.Tensor, self.drop(cast(torch.Tensor, self.proj(h))))
