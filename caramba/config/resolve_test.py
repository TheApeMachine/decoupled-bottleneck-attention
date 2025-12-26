"""
Unit tests for the resolve module (variable interpolation and type normalization).
"""
from __future__ import annotations

import unittest
from typing import Any, cast

from caramba.config.resolve import (
    Resolver,
    TYPE_ALIASES,
    normalize_type_names,
)


class TestResolver(unittest.TestCase):
    """Tests for Resolver variable interpolation."""

    def test_resolve_simple_string_var(self) -> None:
        """Resolves a simple string variable."""
        resolver = Resolver({"name": "test"})
        result = resolver.resolve("${name}")
        self.assertEqual(result, "test")

    def test_resolve_int_var(self) -> None:
        """Resolves an integer variable."""
        resolver = Resolver({"count": 42})
        result = resolver.resolve("${count}")
        self.assertEqual(result, 42)

    def test_resolve_float_var(self) -> None:
        """Resolves a float variable."""
        resolver = Resolver({"rate": 0.001})
        result = resolver.resolve("${rate}")
        self.assertEqual(result, 0.001)

    def test_resolve_embedded_var(self) -> None:
        """Resolves embedded variables in a string."""
        resolver = Resolver({"name": "model", "version": "v1"})
        result = resolver.resolve("${name}_${version}.pt")
        self.assertEqual(result, "model_v1.pt")

    def test_resolve_nested_dict(self) -> None:
        """Resolves variables in nested dictionaries."""
        resolver = Resolver({"d_model": 512, "n_heads": 8})
        result = cast(dict[str, Any], resolver.resolve({
            "model": {
                "hidden_size": "${d_model}",
                "attention": {
                    "heads": "${n_heads}",
                }
            }
        }))
        self.assertEqual(result["model"]["hidden_size"], 512)
        self.assertEqual(result["model"]["attention"]["heads"], 8)

    def test_resolve_list(self) -> None:
        """Resolves variables in lists."""
        resolver = Resolver({"a": 1, "b": 2, "c": 3})
        result = resolver.resolve(["${a}", "${b}", "${c}"])
        self.assertEqual(result, [1, 2, 3])

    def test_resolve_tuple(self) -> None:
        """Resolves variables in tuples."""
        resolver = Resolver({"x": 10, "y": 20})
        result = resolver.resolve(("${x}", "${y}"))
        self.assertEqual(result, (10, 20))

    def test_resolve_no_change(self) -> None:
        """Leaves non-variable strings unchanged."""
        resolver = Resolver({})
        result = resolver.resolve("hello world")
        self.assertEqual(result, "hello world")

    def test_resolve_passthrough_types(self) -> None:
        """Non-string primitives are passed through unchanged."""
        resolver = Resolver({})
        self.assertEqual(resolver.resolve(42), 42)
        self.assertEqual(resolver.resolve(3.14), 3.14)
        self.assertEqual(resolver.resolve(True), True)
        self.assertEqual(resolver.resolve(None), None)

    def test_resolve_unknown_var_raises(self) -> None:
        """Unknown variables raise ValueError."""
        resolver = Resolver({"a": 1})
        with self.assertRaises(ValueError) as ctx:
            resolver.resolve("${unknown}")
        self.assertIn("Unknown manifest variable", str(ctx.exception))

    def test_resolve_cycle_raises(self) -> None:
        """Circular variable references raise ValueError."""
        resolver = Resolver({
            "a": "${b}",
            "b": "${a}",
        })
        with self.assertRaises(ValueError) as ctx:
            resolver.resolve("${a}")
        self.assertIn("Cycle detected", str(ctx.exception))

    def test_resolve_transitive_vars(self) -> None:
        """Transitive variable references are resolved."""
        resolver = Resolver({
            "base": 64,
            "doubled": "${base}",  # Note: This will return 64, not 128
        })
        result = resolver.resolve("${doubled}")
        self.assertEqual(result, 64)


class TestNormalizeTypeNames(unittest.TestCase):
    """Tests for normalize_type_names function."""

    def test_normalize_simple_type(self) -> None:
        """Normalizes a simple type field."""
        payload = {"type": "transformer"}
        result = cast(dict[str, Any], normalize_type_names(payload))
        self.assertEqual(result["type"], "TransformerModel")

    def test_normalize_nested_types(self) -> None:
        """Normalizes types in nested structures."""
        payload = {
            "model": {
                "type": "transformer",
                "topology": {
                    "type": "stacked",
                }
            }
        }
        result = cast(dict[str, Any], normalize_type_names(payload))
        self.assertEqual(result["model"]["type"], "TransformerModel")
        self.assertEqual(result["model"]["topology"]["type"], "StackedTopology")

    def test_normalize_layer_types(self) -> None:
        """Normalizes layer type names."""
        payload = {
            "layers": [
                {"type": "rms_norm", "eps": 1e-5},
                {"type": "attention", "n_heads": 8},
                {"type": "swiglu", "hidden_dim": 256},
                {"type": "linear", "d_in": 128, "d_out": 128},
            ]
        }
        result = cast(dict[str, Any], normalize_type_names(payload))
        self.assertEqual(result["layers"][0]["type"], "RMSNormLayer")
        self.assertEqual(result["layers"][1]["type"], "AttentionLayer")
        self.assertEqual(result["layers"][2]["type"], "SwiGLULayer")
        self.assertEqual(result["layers"][3]["type"], "LinearLayer")

    def test_normalize_topology_types(self) -> None:
        """Normalizes topology type names."""
        for legacy, canonical in [
            ("branching", "BranchingTopology"),
            ("cyclic", "CyclicTopology"),
            ("nested", "NestedTopology"),
            ("parallel", "ParallelTopology"),
            ("recurrent", "RecurrentTopology"),
            ("residual", "ResidualTopology"),
            ("sequential", "SequentialTopology"),
            ("stacked", "StackedTopology"),
        ]:
            payload = {"type": legacy}
            result = cast(dict[str, Any], normalize_type_names(payload))
            self.assertEqual(result["type"], canonical, f"Failed for {legacy}")

    def test_normalize_preserves_canonical_names(self) -> None:
        """Already-canonical names are preserved."""
        payload = {
            "model": {
                "type": "TransformerModel",
                "topology": {
                    "type": "StackedTopology",
                    "layers": [
                        {"type": "RMSNormLayer"},
                        {"type": "AttentionLayer"},
                    ]
                }
            }
        }
        result = cast(dict[str, Any], normalize_type_names(payload))
        self.assertEqual(result["model"]["type"], "TransformerModel")
        self.assertEqual(result["model"]["topology"]["type"], "StackedTopology")
        self.assertEqual(result["model"]["topology"]["layers"][0]["type"], "RMSNormLayer")

    def test_normalize_preserves_non_type_fields(self) -> None:
        """Non-type fields are preserved unchanged."""
        payload = {
            "type": "transformer",
            "name": "my_model",
            "d_model": 512,
            "layers": [1, 2, 3],
            "config": {
                "type": "nested",
                "value": "unchanged",
            }
        }
        result = cast(dict[str, Any], normalize_type_names(payload))
        self.assertEqual(result["name"], "my_model")
        self.assertEqual(result["d_model"], 512)
        self.assertEqual(result["layers"], [1, 2, 3])
        self.assertEqual(result["config"]["value"], "unchanged")

    def test_normalize_handles_tuple(self) -> None:
        """Normalizes types within tuples."""
        payload = ({"type": "linear"}, {"type": "attention"})
        result = cast(tuple[dict[str, Any], ...], normalize_type_names(payload))
        self.assertEqual(result[0]["type"], "LinearLayer")
        self.assertEqual(result[1]["type"], "AttentionLayer")

    def test_normalize_handles_primitives(self) -> None:
        """Primitive values are returned unchanged."""
        self.assertEqual(normalize_type_names("string"), "string")
        self.assertEqual(normalize_type_names(42), 42)
        self.assertEqual(normalize_type_names(3.14), 3.14)
        self.assertEqual(normalize_type_names(True), True)
        self.assertEqual(normalize_type_names(None), None)

    def test_normalize_handles_empty_structures(self) -> None:
        """Empty structures are handled correctly."""
        self.assertEqual(normalize_type_names({}), {})
        self.assertEqual(normalize_type_names([]), [])
        self.assertEqual(normalize_type_names(()), ())

    def test_normalize_unknown_type_preserved(self) -> None:
        """Unknown type values are preserved."""
        payload = {"type": "custom_layer"}
        result = cast(dict[str, Any], normalize_type_names(payload))
        self.assertEqual(result["type"], "custom_layer")


class TestTypeAliases(unittest.TestCase):
    """Tests for TYPE_ALIASES constant."""

    def test_aliases_exist(self) -> None:
        """All expected aliases are defined."""
        expected = [
            "transformer", "gpt", "vit", "mlp",  # Models
            "branching", "cyclic", "nested", "parallel",  # Topologies
            "recurrent", "residual", "sequential", "stacked",
            "layer_norm", "rms_norm", "linear", "dropout",  # Layers
            "attention", "swiglu",
        ]
        for alias in expected:
            self.assertIn(alias, TYPE_ALIASES, f"Missing alias: {alias}")

    def test_aliases_map_to_correct_types(self) -> None:
        """Aliases map to correctly-formatted canonical names."""
        for alias, canonical in TYPE_ALIASES.items():
            # Canonical names should be PascalCase
            self.assertTrue(
                canonical[0].isupper(),
                f"Canonical name should be PascalCase: {canonical}"
            )
            # Canonical names should end with Model, Topology, or Layer
            valid_suffix = (
                canonical.endswith("Model") or
                canonical.endswith("Topology") or
                canonical.endswith("Layer")
            )
            self.assertTrue(
                valid_suffix,
                f"Canonical name should end with Model/Topology/Layer: {canonical}"
            )


if __name__ == "__main__":
    unittest.main()
