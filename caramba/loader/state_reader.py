"""Safe accessors for state_dict contents.

When loading checkpoints, a lot can go wrong: missing keys, wrong tensor
shapes, type mismatches. StateReader provides validated access methods
that fail with clear error messages instead of cryptic PyTorch exceptions.
"""
from __future__ import annotations

from torch import Tensor, nn


class StateReader:
    """Validated access to checkpoint state_dict contents.

    Wraps a raw state_dict with methods that check for existence and type,
    providing clear error messages when something is wrong.
    """

    def __init__(self, state_dict: dict[str, Tensor]) -> None:
        """Wrap a state_dict for validated access."""
        self.state_dict = state_dict

    def key(self, *parts: str) -> str:
        """Join path parts into a dot-separated key.

        Example: key("model", "layers", "0", "weight") → "model.layers.0.weight"
        """
        return ".".join(p for p in parts if p)

    def _require_tensor(self, key: str, required: bool) -> Tensor | None:
        """Get a tensor, optionally raising if missing."""
        if key not in self.state_dict:
            if required:
                raise ValueError(f"Missing state_dict key: {key}")
            return None
        value = self.state_dict[key]
        if not isinstance(value, Tensor):
            raise ValueError(f"Expected tensor for key {key}, got {type(value)!r}")
        return value

    def get(self, key: str) -> Tensor:
        """Get a required tensor, raising ValueError if missing."""
        result = self._require_tensor(key, required=True)
        assert result is not None
        return result

    def get_optional(self, key: str) -> Tensor | None:
        """Get an optional tensor, returning None if missing."""
        return self._require_tensor(key, required=False)

    def copy_dense(
        self,
        dst: nn.Module,
        *,
        weight: Tensor,
        bias: Tensor | None,
    ) -> None:
        """Copy weight and optional bias into a dense layer.

        Validates shapes match before copying.
        """
        if not hasattr(dst, "weight"):
            raise ValueError(f"Expected DenseWeight-like dst, got {type(dst)!r}")
        dst_weight = getattr(dst, "weight")
        if not isinstance(dst_weight, Tensor):
            raise ValueError(f"Expected tensor weight, got {type(dst_weight)!r}")
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
                raise ValueError(f"Expected tensor bias, got {type(dst_bias)!r}")
            assert bias is not None
            if dst_bias.shape != bias.shape:
                raise ValueError(
                    f"Bias shape mismatch: {dst_bias.shape} vs {bias.shape}"
                )
            dst_bias.data.copy_(bias)
