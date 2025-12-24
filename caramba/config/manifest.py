"""Manifest: control interface for the training loop."""
from __future__ import annotations

import json
from pathlib import Path

import yaml
from pydantic import BaseModel

from caramba.config.defaults import Defaults
from caramba.config.group import Group
from caramba.config.model import ModelConfig
from caramba.config.resolve import Resolver


class Manifest(BaseModel):
    """
    Manifest is the control structure parsed from manifest.json/.yaml.
    """
    version: int
    name: str | None = None
    notes: str
    defaults: Defaults
    model: ModelConfig
    groups: list[Group]

    @classmethod
    def from_path(cls, path: Path) -> Manifest:
        """
        from_path loads and validates a manifest payload.
        """
        text = path.read_text(encoding="utf-8")
        match path.suffix.lower():
            case ".json":          payload = json.loads(text)
            case ".yml" | ".yaml": payload = yaml.safe_load(text)
            case s: raise ValueError(f"Unsupported format '{s}'")
        if payload is None:
            raise ValueError("Manifest payload is empty.")
        if not isinstance(payload, dict):
            raise ValueError(
                f"Manifest payload must be a dict, got {type(payload)!r}"
            )

        vars_payload = payload.pop("vars", None)
        if vars_payload is not None:
            if not isinstance(vars_payload, dict):
                raise ValueError(
                    f"Manifest vars must be a dict, got {type(vars_payload)!r}"
                )
            payload = Resolver(vars_payload).resolve(payload)
        return cls.model_validate(payload)
