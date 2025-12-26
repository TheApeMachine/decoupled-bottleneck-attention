"""Manifest: the top-level experiment configuration file.

A manifest defines everything needed for an experiment: model architecture,
training settings, data paths, and benchmarks. It's loaded from YAML and
supports variable substitution for reusable templates.
"""
from __future__ import annotations

import json
from pathlib import Path

import yaml
from pydantic import BaseModel

from caramba.config import PositiveInt
from caramba.config.defaults import Defaults
from caramba.config.group import Group
from caramba.config.model import ModelConfig
from caramba.config.paper import PaperConfig
from caramba.config.resolve import Resolver, normalize_type_names
from caramba.paper.review import ReviewConfig


class Manifest(BaseModel):
    """The complete experiment specification loaded from YAML.

    Contains model architecture, training runs, and benchmark definitions.
    Variable substitution allows reusing common values throughout the config.

    Optionally includes paper configuration for AI-assisted paper drafting.
    """

    version: PositiveInt
    name: str | None = None
    notes: str
    defaults: Defaults
    model: ModelConfig
    groups: list[Group]
    paper: PaperConfig | None = None
    review: ReviewConfig | None = None

    @classmethod
    def from_path(cls, path: Path) -> "Manifest":
        """Load and validate a manifest from a JSON or YAML file.

        Supports variable substitution via a `vars` section at the top level.
        Variables can be referenced as `$var_name` throughout the config.
        """
        text = path.read_text(encoding="utf-8")
        match path.suffix.lower():
            case ".json":
                payload = json.loads(text)
            case ".yml" | ".yaml":
                payload = yaml.safe_load(text)
            case s:
                raise ValueError(f"Unsupported format '{s}'")

        if payload is None:
            raise ValueError("Manifest payload is empty.")
        if not isinstance(payload, dict):
            raise ValueError(f"Manifest payload must be a dict, got {type(payload)!r}")

        # Process variable substitution
        vars_payload = payload.pop("vars", None)
        if vars_payload is not None:
            if not isinstance(vars_payload, dict):
                raise ValueError(
                    f"Manifest vars must be a dict, got {type(vars_payload)!r}"
                )
            payload = Resolver(vars_payload).resolve(payload)

        # Normalize shorthand type names (e.g., 'nested' → 'NestedTopology')
        payload = normalize_type_names(payload)

        return cls.model_validate(payload)
