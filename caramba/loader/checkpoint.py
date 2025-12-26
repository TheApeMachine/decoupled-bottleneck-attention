"""Generic checkpoint loading for PyTorch and safetensors files.

Checkpoints store trained model weights. This module handles the mechanics
of loading them from various formats (PyTorch .pt/.bin, safetensors, sharded)
and optionally remapping keys when the architecture differs from the source.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import torch
from safetensors.torch import load_file
from torch import Tensor, nn


def _get_torch_version() -> tuple[int, int]:
    """Parse PyTorch version into (major, minor) tuple."""
    version_str = torch.__version__.split("+")[0]
    parts = version_str.split(".")
    return int(parts[0]), int(parts[1])


def _safe_torch_load(path: Path) -> dict[str, Tensor]:
    """Load a checkpoint safely.

    Uses weights_only=True when available (PyTorch ≥2.4) for security—this
    prevents pickle-based arbitrary code execution from malicious checkpoints.
    """
    major, minor = _get_torch_version()

    if (major, minor) >= (2, 4):
        try:
            return torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            return torch.load(path, map_location="cpu")
    else:
        return torch.load(path, map_location="cpu")


class CheckpointLoader:
    """Loads state dictionaries from various checkpoint formats.

    Supports:
    - Single-file PyTorch checkpoints (.pt, .bin)
    - Single-file safetensors (.safetensors)
    - Sharded checkpoints with index files (.index.json)
    """

    def load(self, path: Path) -> dict[str, Tensor]:
        """Load a state_dict, auto-detecting the format from file extension.

        For sharded checkpoints, pass the .index.json file and this will
        load all shards and merge them.
        """
        path = Path(path)
        if path.name.endswith(".index.json"):
            return self.load_sharded(path)
        if path.suffix == ".safetensors":
            return self.load_safetensors(path)
        return _safe_torch_load(path)

    def load_mapped(
        self,
        model: nn.Module,
        state_dict: dict[str, Tensor],
        mapping: dict[str, str],
        *,
        strict: bool = True,
    ) -> None:
        """Load a state_dict with key remapping.

        Use this when the checkpoint uses different parameter names than
        your model. The mapping is {source_key: dest_key}.
        """
        if not mapping:
            raise ValueError("mapping must be non-empty")

        mapped = {dst: state_dict[src] for src, dst in mapping.items()}
        try:
            result = model.load_state_dict(mapped, strict=strict)
        except RuntimeError as e:
            raise ValueError(f"load failed: {e}") from e

        if strict and result is not None:
            missing, unexpected = result
            if missing or unexpected:
                raise ValueError(
                    f"load failed: missing={missing}, unexpected={unexpected}"
                )

    def load_sharded(self, index_path: Path) -> dict[str, Tensor]:
        """Load a sharded checkpoint from its index file.

        Sharded checkpoints split weights across multiple files for large
        models. The index file maps weight names to shard files.
        """
        data = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = data.get("weight_map")
        if not isinstance(weight_map, dict):
            raise ValueError("Invalid index file: missing weight_map")

        out: dict[str, Tensor] = {}
        for shard in sorted(set(weight_map.values())):
            shard_path = index_path.parent / shard
            if shard_path.name.endswith(".index.json"):
                raise ValueError(
                    f"Shard {shard} is an index file, expected tensor file"
                )
            for key, value in self.load(shard_path).items():
                if key in out:
                    raise ValueError(f"Duplicate key in shards: {key}")
                out[key] = value
        return out

    def load_safetensors(self, path: Path) -> dict[str, Tensor]:
        """Load a safetensors file.

        Safetensors is a fast, safe format for storing tensors without
        the security risks of Python pickle.
        """
        return load_file(str(path), device="cpu")


class AttentionLoader:
    """Loads Q/K/V/O weights into attention modules.

    Handles the common pattern of loading attention weights from a
    checkpoint into a model, supporting both old-style (query/key/value/out)
    and new-style (q_proj/k_proj/v_proj/out_proj) attribute names.
    """

    def load(
        self,
        student: nn.Module,
        *,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        o: Tensor,
    ) -> None:
        """Load Q/K/V/O weights into the attention module.

        Auto-detects which attribute names the module uses.
        """
        q_proj = getattr(student, "q_proj", None)
        k_proj = getattr(student, "k_proj", None)
        v_proj = getattr(student, "v_proj", None)
        out_proj = getattr(student, "out_proj", None)

        student_class_name = type(student).__name__

        # New-style names
        if all(p is not None for p in [q_proj, k_proj, v_proj, out_proj]):
            self.copy(cast(nn.Linear, q_proj), q)
            self.copy(cast(nn.Linear, k_proj), k)
            self.copy(cast(nn.Linear, v_proj), v)
            self.copy(cast(nn.Linear, out_proj), o)
        else:
            # Fall back to old-style names
            self._load_with_fallback(student, "q_proj", "query", q, "Q", student_class_name)
            self._load_with_fallback(student, "k_proj", "key", k, "K", student_class_name)
            self._load_with_fallback(student, "v_proj", "value", v, "V", student_class_name)
            self._load_with_fallback(student, "out_proj", "out", o, "Out", student_class_name)

    def _load_with_fallback(
        self,
        student: nn.Module,
        new_name: str,
        old_name: str,
        weight: Tensor,
        proj_label: str,
        class_name: str,
    ) -> None:
        """Try new-style attribute name, fall back to old-style."""
        proj = getattr(student, new_name, None)
        if proj is not None:
            self.copy(cast(nn.Linear, proj), weight)
            return

        proj = getattr(student, old_name, None)
        if proj is not None:
            self.copy(cast(nn.Linear, proj), weight)
            return

        raise ValueError(
            f"Could not find {proj_label} projection in {class_name}. "
            f"Expected '{new_name}' or '{old_name}' attribute."
        )

    def copy(self, dst: nn.Linear, src: Tensor) -> None:
        """Copy weight tensor into a Linear module."""
        dst.weight.data.copy_(src)
