"""
resolve provides manifest variable interpolation and type normalization.
"""
from __future__ import annotations

import re
from collections.abc import Mapping


# Maps legacy/shorthand type names to canonical class names
# This enables backward compatibility with older presets
TYPE_ALIASES: dict[str, str] = {
    # Model types
    "transformer": "TransformerModel",
    "gpt": "GPTModel",
    "vit": "ViTModel",
    "mlp": "MLPModel",
    # Topology types
    "branching": "BranchingTopology",
    "cyclic": "CyclicTopology",
    "nested": "NestedTopology",
    "parallel": "ParallelTopology",
    "recurrent": "RecurrentTopology",
    "residual": "ResidualTopology",
    "sequential": "SequentialTopology",
    "stacked": "StackedTopology",
    # Layer types
    "layer_norm": "LayerNormLayer",
    "rms_norm": "RMSNormLayer",
    "linear": "LinearLayer",
    "dropout": "DropoutLayer",
    "attention": "AttentionLayer",
    "swiglu": "SwiGLULayer",
}


def normalize_type_names(payload: object) -> object:
    """
    Recursively normalize legacy type names to canonical class names.

    This allows presets to use shorthand names like 'rms_norm' or 'nested'
    while internally converting them to the expected class names like
    'RMSNormLayer' or 'NestedTopology'.
    """
    if isinstance(payload, Mapping):
        result: dict[str, object] = {}
        for k, v in payload.items():
            if k == "type" and isinstance(v, str):
                # Normalize the type value
                result[k] = TYPE_ALIASES.get(v, v)
            else:
                result[k] = normalize_type_names(v)
        return result
    if isinstance(payload, list):
        return [normalize_type_names(v) for v in payload]
    if isinstance(payload, tuple):
        return tuple(normalize_type_names(v) for v in payload)
    return payload


class Resolver:
    """
    Resolver expands ${var} references in manifest payloads.
    """
    def __init__(self, vars: Mapping[str, object]) -> None:
        """
        __init__ initializes the variable resolver.
        """
        super().__init__()
        self._vars: dict[str, object] = dict(vars)
        self._cache: dict[str, object] = {}
        self._resolving: set[str] = set()
        self._pattern = re.compile(r"\$\{([A-Za-z0-9_]+)\}")

    def resolve(self, value: object) -> object:
        """
        resolve applies variable interpolation to a payload node.
        """
        if isinstance(value, Mapping):
            return {k: self.resolve(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self.resolve(v) for v in value]
        if isinstance(value, tuple):
            return tuple(self.resolve(v) for v in value)
        if isinstance(value, str):
            return self._resolve_str(value)
        return value

    def _resolve_str(self, value: str) -> object:
        """
        _resolve_str resolves ${var} placeholders inside strings.
        """
        matches = list(self._pattern.finditer(value))
        if not matches:
            return value
        if len(matches) == 1 and matches[0].span() == (0, len(value)):
            name = matches[0].group(1)
            return self._resolve_var(name)

        def _replace(match: re.Match[str]) -> str:
            name = match.group(1)
            return str(self._resolve_var(name))

        return self._pattern.sub(_replace, value)

    def _resolve_var(self, name: str) -> object:
        """
        _resolve_var resolves a single variable by name.
        """
        if name in self._cache:
            return self._cache[name]
        if name in self._resolving:
            raise ValueError(f"Cycle detected in manifest vars: {name}")
        if name not in self._vars:
            raise ValueError(f"Unknown manifest variable: {name}")

        self._resolving.add(name)
        resolved = self.resolve(self._vars[name])
        self._resolving.remove(name)
        self._cache[name] = resolved
        return resolved
