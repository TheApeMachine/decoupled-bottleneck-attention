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

    def _require_tensor(self, key: str, required: bool) -> Tensor | None:
        """
        _require_tensor retrieves a tensor, optionally raising if missing.
        """
        if key not in self.state_dict:
            if required:
                raise ValueError(f"Missing state_dict key: {key}")
            return None
        value = self.state_dict[key]
        if not isinstance(value, Tensor):
            raise ValueError(
                f"Expected tensor for key {key}, got {type(value)!r}"
            )
        return value

    def get(self, key: str) -> Tensor:
        """
        get returns a tensor for a required key.
        """
        result = self._require_tensor(key, required=True)
        # _require_tensor with required=True always returns Tensor or raises
        assert result is not None
        return result

    def get_optional(self, key: str) -> Tensor | None:
        """
        get_optional returns a tensor for an optional key.
        """
        return self._require_tensor(key, required=False)

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
            # bias is guaranteed non-None here by the earlier mismatch check
            assert bias is not None
            if dst_bias.shape != bias.shape:
                raise ValueError(
                    f"Bias shape mismatch: {dst_bias.shape} vs {bias.shape}"
                )
            dst_bias.data.copy_(bias)
