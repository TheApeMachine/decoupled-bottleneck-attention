"""
state_reader provides state_dict access helpers.
"""
from __future__ import annotations

from torch import Tensor, nn


class StateReader:
    """
    StateReader provides validated state_dict access.
    """
    def __init__(self, state_dict: dict[str, Tensor]) -> None:
        """
        __init__ initializes the state reader.
        """
        self.state_dict: dict[str, Tensor] = state_dict

    def key(self, *parts: str) -> str:
        """
        key joins path parts with dots.
        """
        return ".".join(p for p in parts if p)

    def get(self, key: str) -> Tensor:
        """
        get returns a tensor for a required key.
        """
        if key not in self.state_dict:
            raise ValueError(f"Missing state_dict key: {key}")
        value = self.state_dict[key]
        if not isinstance(value, Tensor):
            raise ValueError(
                f"Expected tensor for key {key}, got {type(value)!r}"
            )
        return value

    def get_optional(self, key: str) -> Tensor | None:
        """
        get_optional returns a tensor for an optional key.
        """
        if key not in self.state_dict:
            return None
        value = self.state_dict[key]
        if not isinstance(value, Tensor):
            raise ValueError(
                f"Expected tensor for key {key}, got {type(value)!r}"
            )
        return value

    def copy_dense(
        self,
        dst: nn.Module,
        *,
        weight: Tensor,
        bias: Tensor | None,
    ) -> None:
        """
        copy_dense copies weights into a DenseWeight-like module.
        """
        if not hasattr(dst, "weight"):
            raise ValueError(
                f"Expected DenseWeight-like dst, got {type(dst)!r}"
            )
        dst_weight = getattr(dst, "weight")
        if not isinstance(dst_weight, Tensor):
            raise ValueError(
                f"Expected tensor weight, got {type(dst_weight)!r}"
            )
        if dst_weight.shape != weight.shape:
            raise ValueError(
                f"Weight shape mismatch: {dst_weight.shape} vs {weight.shape}"
            )
        dst_weight.data.copy_(weight)

        dst_bias = getattr(dst, "bias", None)
        if (dst_bias is None) != (bias is None):
            raise ValueError("Bias presence mismatch between dst and src")
        if dst_bias is not None:
            if not isinstance(dst_bias, Tensor):
                raise ValueError(
                    f"Expected tensor bias, got {type(dst_bias)!r}"
                )
            if bias is None:
                raise ValueError("Expected source bias to be present")
            if dst_bias.shape != bias.shape:
                raise ValueError(
                    f"Bias shape mismatch: {dst_bias.shape} vs {bias.shape}"
                )
            dst_bias.data.copy_(bias)
