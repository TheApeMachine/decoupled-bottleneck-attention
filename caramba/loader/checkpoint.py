"""
checkpoint provides weight-loading utilities for PyTorch checkpoints.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import torch
from safetensors.torch import load_file
from torch import Tensor, nn


class CheckpointLoader:
    """Loads state dicts from various checkpoint formats."""

    def load(self, path: Path) -> dict[str, Tensor]:
        """Load a state_dict from torch, safetensors, or sharded index files."""
        path = Path(path)
        if path.name.endswith(".index.json"):
            return self.load_sharded(path)
        if path.suffix == ".safetensors":
            return self.load_safetensors(path)
        return torch.load(path, map_location="cpu")

    def load_mapped(
        self,
        model: nn.Module,
        state_dict: dict[str, Tensor],
        mapping: dict[str, str],
        *,
        strict: bool = True,
    ) -> None:
        """Load a state_dict into a model using key remapping."""
        if not mapping:
            raise ValueError("mapping must be non-empty")

        mapped = {dst: state_dict[src] for src, dst in mapping.items()}
        missing, unexpected = model.load_state_dict(mapped, strict=strict)
        if missing or unexpected:
            raise ValueError(f"load failed: missing={missing}, unexpected={unexpected}")

    def load_sharded(self, index_path: Path) -> dict[str, Tensor]:
        """Load from a sharded checkpoint index."""
        data = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = data.get("weight_map")
        if not isinstance(weight_map, dict):
            raise ValueError("Invalid index file: missing weight_map")

        out: dict[str, Tensor] = {}
        for shard in sorted(set(weight_map.values())):
            shard_path = index_path.parent / shard
            for key, value in self.load(shard_path).items():
                if key in out:
                    raise ValueError(f"Duplicate key in shards: {key}")
                out[key] = value
        return out

    def load_safetensors(self, path: Path) -> dict[str, Tensor]:
        """Load from safetensors format."""
        return load_file(str(path), device="cpu")


class AttentionLoader:
    """Loads attention weights from teacher Q/K/V/O tensors."""

    def load(
        self,
        student: nn.Module,
        *,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        o: Tensor,
    ) -> None:
        """Load Q/K/V/O weights into student attention module.

        Supports both old attribute names (query/key/value/out) and
        new attribute names (q_proj/k_proj/v_proj/out_proj).
        """
        # Try new attribute names first
        q_proj = getattr(student, "q_proj", None)
        k_proj = getattr(student, "k_proj", None)
        v_proj = getattr(student, "v_proj", None)
        out_proj = getattr(student, "out_proj", None)

        if q_proj is not None and k_proj is not None:
            self.copy(cast(nn.Linear, q_proj), q)
            self.copy(cast(nn.Linear, k_proj), k)
            self.copy(cast(nn.Linear, v_proj), v)
            self.copy(cast(nn.Linear, out_proj), o)
        else:
            # Fallback to old attribute names
            self.copy(cast(nn.Linear, student.query), q)  # type: ignore[attr-defined]
            self.copy(cast(nn.Linear, student.key), k)  # type: ignore[attr-defined]
            self.copy(cast(nn.Linear, student.value), v)  # type: ignore[attr-defined]
            self.copy(cast(nn.Linear, student.out), o)  # type: ignore[attr-defined]

    def copy(self, dst: nn.Linear, src: Tensor) -> None:
        """Copy weight tensor to destination module."""
        dst.weight.data.copy_(src)
