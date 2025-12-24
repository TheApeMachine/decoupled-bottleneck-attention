"""
resolve provides manifest variable interpolation.
"""
from __future__ import annotations

import re
from collections.abc import Mapping


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
