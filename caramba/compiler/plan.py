"""Plan printer: human-readable IR for debugging.

After compilation, you may want to inspect what the config looks like.
The planner renders the lowered manifest as a structured text format,
making it easy to verify that repeats expanded correctly and layers
are configured as expected.
"""
from __future__ import annotations

from typing import Iterable

from caramba.config.layer import LayerConfig
from caramba.config.manifest import Manifest
from caramba.config.topology import NodeConfig, TopologyConfig


class Planner:
    """Renders human-readable execution plans from lowered manifests.

    Useful for debugging config issues by showing the fully expanded
    layer structure.
    """

    def format(self, manifest: Manifest) -> str:
        """Render a human-readable plan for a lowered manifest."""
        out: list[str] = []
        out.append(f"manifest.version={manifest.version}")
        out.append(f"model.type={manifest.model.type.value}")
        out.append("model.topology:")
        out.extend(
            self.format_topology(manifest.model.topology, indent=2, path="model")
        )
        return "\n".join(out)

    def is_topology(self, node: NodeConfig) -> bool:
        """Check if node is a topology (has layers attribute)."""
        return hasattr(node, "layers")

    def format_topology(
        self, config: TopologyConfig, *, indent: int, path: str
    ) -> Iterable[str]:
        """Format a topology node with its children."""
        pad = " " * indent
        yield (
            f"{pad}- topology={config.type.value} "
            f"repeat={getattr(config, 'repeat', None)} "
            f"path={path}.topology"
        )

        for i, node in enumerate(config.layers):
            yield from self.format_node(
                node, indent=indent + 2, path=f"{path}.topology.layers[{i}]"
            )

    def format_node(
        self, config: NodeConfig, *, indent: int, path: str
    ) -> Iterable[str]:
        """Format a topology or layer node."""
        if self.is_topology(config):
            yield from self.format_topology(config, indent=indent, path=path)  # type: ignore[arg-type]
        else:
            yield from self.format_layer(config, indent=indent, path=path)  # type: ignore[arg-type]

    def format_layer(
        self, config: LayerConfig, *, indent: int, path: str
    ) -> Iterable[str]:
        """Format a single layer node."""
        pad = " " * indent
        yield f"{pad}- layer={config.type.value} path={path}"
