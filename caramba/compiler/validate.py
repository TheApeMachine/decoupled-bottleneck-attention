"""Validation pass: check shape invariants and constraints.

After lowering, we validate that the config makes sense:
- Layer IO dimensions are compatible (output of layer N matches input of layer N+1)
- Residual connections preserve shape
- Attention layers have valid head/dim configurations
- Parallel branches have consistent outputs

Catching these errors early prevents cryptic runtime failures.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from caramba.config.layer import (
    AttentionLayerConfig,
    AttentionMode,
    DropoutLayerConfig,
    LayerConfig,
    LinearLayerConfig,
)
from caramba.config.manifest import Manifest
from caramba.config.topology import NodeConfig, TopologyConfig


@dataclass(frozen=True, slots=True)
class _IO:
    """Input/output dimension pair for a node."""

    d_in: int | None
    d_out: int | None


class Validator:
    """Checks cross-layer shape invariants.

    Validates that layers connect properly and attention configs are valid.
    """

    def validate_manifest(self, manifest: Manifest) -> None:
        """Validate a full manifest including attention layer constraints."""
        self.validate_topology(manifest.model.topology, path="model.topology")
        self._validate_all_attention_layers(manifest)

    def validate(self, manifest: Manifest) -> None:
        """Alias for validate_manifest (backwards compatibility)."""
        return self.validate_manifest(manifest)

    def validate_topology(
        self, config: TopologyConfig, *, path: str = "model.topology"
    ) -> None:
        """Validate a topology config by inferring IO dimensions."""
        self.infer_topology_io(config, path=path)

    def is_topology(self, node: NodeConfig) -> bool:
        """Check if node is a topology (has layers attribute)."""
        return hasattr(node, "layers")

    def infer_node_io(self, node: NodeConfig, *, path: str) -> _IO:
        """Infer IO dimensions for any node type."""
        if self.is_topology(node):
            return self.infer_topology_io(node, path=path)  # type: ignore[arg-type]
        return self.infer_layer_io(node)  # type: ignore[arg-type]

    def infer_layer_io(self, config: LayerConfig) -> _IO:
        """Infer IO dimensions for a layer.

        - Linear: explicit d_in/d_out
        - Dropout: shape-transparent (None/None)
        - Others: preserve d_model
        """
        if isinstance(config, LinearLayerConfig):
            return _IO(d_in=int(config.d_in), d_out=int(config.d_out))

        if isinstance(config, DropoutLayerConfig):
            return _IO(d_in=None, d_out=None)

        d_model = getattr(config, "d_model", None)
        if d_model is not None:
            d = int(d_model)
            return _IO(d_in=d, d_out=d)

        raise ValueError(f"Unsupported layer config: {type(config)!r}")

    def infer_topology_io(self, config: TopologyConfig, *, path: str) -> _IO:
        """Infer IO dimensions for a topology.

        Different topology types have different constraints:
        - Parallel/Branching: all branches must have same output
        - Residual: all nodes must preserve shape
        - Sequential-like: chain IO through nodes
        """
        if config.type.value in ("ParallelTopology", "BranchingTopology"):
            outs = self.collect_branch_outs(list(config.layers), path=path)
            return self.require_single_out(outs, path=path, kind=config.type.value)

        if config.type.value == "ResidualTopology":
            io = self.infer_seq_io(list(config.layers), path=path)
            if io.d_out is not None:
                self.require_shape_preserving(list(config.layers), path=path)
            return io

        return self.infer_seq_io(list(config.layers), path=path)

    def infer_seq_io(self, nodes: list[NodeConfig], *, path: str) -> _IO:
        """Infer IO for sequential nodes, checking dimension compatibility."""
        cur: int | None = None
        for i, node in enumerate(nodes):
            node_path = f"{path}.layers[{i}]"
            io = self.infer_node_io(node, path=node_path)

            if io.d_in is not None:
                self.require_match(cur, io.d_in, path=node_path)
                cur = io.d_in

            if io.d_out is not None:
                cur = io.d_out

        return _IO(d_in=None, d_out=cur)

    def require_match(self, cur: int | None, want: int, *, path: str) -> None:
        """Require dimension match between consecutive layers."""
        if cur is not None and cur != want:
            raise ValueError(
                f"{path}: expected d_in={cur}, got d_in={want}. "
                "Fix: make this node's input dim match the previous node's output dim."
            )

    def collect_branch_outs(self, nodes: list[NodeConfig], *, path: str) -> set[int]:
        """Collect output dimensions from parallel branches."""
        outs: set[int] = set()
        for i, node in enumerate(nodes):
            node_path = f"{path}.layers[{i}]"
            io = self.infer_node_io(node, path=node_path)
            if io.d_out is not None:
                outs.add(io.d_out)
        return outs

    def require_single_out(self, outs: set[int], *, path: str, kind: str) -> _IO:
        """Require all branches have the same output dimension."""
        if len(outs) > 1:
            raise ValueError(
                f"{path}: {kind} requires consistent d_out, got {sorted(outs)}. "
                f"Fix: ensure all {kind} branches produce the same d_out."
            )
        return _IO(d_in=None, d_out=next(iter(outs)) if outs else None)

    def require_shape_preserving(self, nodes: list[NodeConfig], *, path: str) -> None:
        """Require all nodes preserve shape (for residual connections)."""
        for i, node in enumerate(nodes):
            node_path = f"{path}.layers[{i}]"
            io = self.infer_node_io(node, path=node_path)
            if io.d_in is not None and io.d_out is not None and io.d_in != io.d_out:
                raise ValueError(
                    f"{node_path}: residual requires shape-preserving nodes, got "
                    f"d_in={io.d_in}, d_out={io.d_out}. "
                    "Fix: ensure all nodes inside residual preserve d_model."
                )

    def iter_layers(
        self, topology: TopologyConfig, *, path: str
    ) -> Iterable[tuple[LayerConfig, str]]:
        """Iterate over all layers in a topology recursively."""
        for i, node in enumerate(topology.layers):
            node_path = f"{path}.layers[{i}]"
            if self.is_topology(node):
                yield from self.iter_layers(node, path=node_path)  # type: ignore[arg-type]
            else:
                yield node, node_path  # type: ignore[misc]

    def _validate_all_attention_layers(self, manifest: Manifest) -> None:
        """Validate all attention layers in the manifest."""
        for layer, path in self.iter_layers(
            manifest.model.topology, path="model.topology"
        ):
            if isinstance(layer, AttentionLayerConfig):
                self.validate_attention(layer, path=path)

    def validate_attention(self, config: AttentionLayerConfig, *, path: str) -> None:
        """Validate attention layer configuration.

        Checks:
        - n_heads divisible by n_kv_heads (GQA constraint)
        - Decoupled: sem_dim, geo_dim, v_dim divisible by n_heads
        - Decoupled + RoPE: geo_head_dim must be even
        - Standard: attn_dim == n_heads * head_dim
        """
        d_model = config.d_model
        n_heads = config.n_heads
        n_kv_heads = config.kv_heads

        if n_heads % n_kv_heads != 0:
            raise ValueError(
                f"{path}: expected n_heads % n_kv_heads == 0, got "
                f"n_heads={n_heads}, n_kv_heads={n_kv_heads}. "
                "Fix: choose n_kv_heads that divides n_heads."
            )

        if config.mode == AttentionMode.DECOUPLED:
            if config.sem_dim is None or config.geo_dim is None:
                raise ValueError(
                    f"{path}: decoupled mode requires sem_dim and geo_dim. "
                    "Fix: set both sem_dim and geo_dim in the config."
                )

            sem_dim = config.sem_dim
            geo_dim = config.geo_dim
            v_dim = config.v_dim

            if sem_dim % n_heads != 0:
                raise ValueError(
                    f"{path}: sem_dim must be divisible by n_heads, got "
                    f"sem_dim={sem_dim}, n_heads={n_heads}. "
                    "Fix: choose sem_dim that is divisible by n_heads."
                )

            if geo_dim % n_heads != 0:
                raise ValueError(
                    f"{path}: geo_dim must be divisible by n_heads, got "
                    f"geo_dim={geo_dim}, n_heads={n_heads}. "
                    "Fix: choose geo_dim that is divisible by n_heads."
                )

            if v_dim % n_heads != 0:
                raise ValueError(
                    f"{path}: v_dim must be divisible by n_heads, got "
                    f"v_dim={v_dim}, n_heads={n_heads}. "
                    "Fix: choose v_dim (or attn_dim) that is divisible by n_heads."
                )

            geo_head_dim = geo_dim // n_heads
            if config.rope_enabled and geo_head_dim % 2 != 0:
                raise ValueError(
                    f"{path}: decoupled mode with RoPE requires even geo_head_dim, got "
                    f"geo_head_dim={geo_head_dim}. "
                    "Fix: adjust geo_dim so geo_dim / n_heads is even."
                )
        else:
            head_dim = config.head_dim
            attn_dim = config.attn_dim if config.attn_dim is not None else d_model

            if attn_dim != n_heads * head_dim:
                raise ValueError(
                    f"{path}: expected attn_dim == n_heads * head_dim, got "
                    f"attn_dim={attn_dim}, n_heads={n_heads}, head_dim={head_dim}. "
                    "Fix: ensure attn_dim = n_heads * head_dim."
                )
