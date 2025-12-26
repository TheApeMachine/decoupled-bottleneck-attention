"""
lower provides a small lowering pass from user configs to canonical configs.
"""
from __future__ import annotations

from caramba.config.manifest import Manifest
from caramba.config.model import ModelConfig
from caramba.config.topology import NodeConfig, TopologyConfig


class Lowerer:
    """Lowerer expands repeat at compile-time."""

    def lower_manifest(self, manifest: Manifest) -> Manifest:
        """Lower a manifest into canonical form."""
        lowered_model = self.lower_model(manifest.model)
        return manifest.model_copy(update={"model": lowered_model})

    def lower_model(self, model: ModelConfig) -> ModelConfig:
        """Lower a model config into canonical form."""
        lowered_topology = self.lower_topology(model.topology)
        return model.model_copy(update={"topology": lowered_topology})

    def lower_topology(self, config: TopologyConfig) -> TopologyConfig:
        """Expand topology-level repeat at compile-time."""
        lowered = self.lower_nodes(list(config.layers))
        layers = self.repeat_nodes(lowered, repeat=int(config.repeat))
        return config.model_copy(update={"layers": layers, "repeat": 1})

    def lower_nodes(self, nodes: list[NodeConfig]) -> list[NodeConfig]:
        """Lower nested topology nodes."""
        return [
            self.lower_topology(node) if self.is_topology(node) else node  # type: ignore[arg-type]
            for node in nodes
        ]

    def repeat_nodes(self, nodes: list[NodeConfig], *, repeat: int) -> list[NodeConfig]:
        """Repeat nodes by deep-copying each element."""
        return [
            node.model_copy(deep=True)
            for _ in range(repeat)
            for node in nodes
        ]

    def is_topology(self, node: NodeConfig) -> bool:
        """Check if node is a topology (has layers attribute that is a list)."""
        if not hasattr(node, "layers"):
            return False
        layers = getattr(node, "layers")
        return isinstance(layers, list)

