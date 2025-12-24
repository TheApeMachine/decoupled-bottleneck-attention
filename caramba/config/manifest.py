"""Manifest: control interface for the training loop."""
from __future__ import annotations

import json
from pathlib import Path

import yaml
from pydantic import BaseModel

from caramba.config.defaults import Defaults
from caramba.config.group import Group
from caramba.config.model import ModelConfig


class Manifest(BaseModel):
    """Control structure parsed from manifest.json/.yaml."""
    version: int
    name: str | None = None
    notes: str
    defaults: Defaults
    model: ModelConfig
    groups: list[Group]

    @classmethod
    def from_path(cls, path: Path) -> Manifest:
        text = path.read_text(encoding="utf-8")
        match path.suffix.lower():
            case ".json":          payload = json.loads(text)
            case ".yml" | ".yaml": payload = yaml.safe_load(text)
            case s: raise ValueError(f"Unsupported format '{s}'")
        return cls.model_validate(payload)