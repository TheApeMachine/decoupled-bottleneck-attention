"""
manifest_test provides tests for JSON/YAML manifest loading.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from pydantic import ValidationError

from caramba.config.layer import LayerType, LinearLayerConfig
from caramba.config.manifest import Manifest


class ManifestTest(unittest.TestCase):
    """
    ManifestTest provides tests for the Manifest class.
    """
    def test_load_yaml_manifest(self) -> None:
        """
        test loading a YAML manifest.
        """
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "m.yml"
            path.write_text(
                "\n".join(
                    [
                        "version: 1",
                        "name: test",
                        'notes: "x"',
                        "defaults:",
                        "  wandb: false",
                        "  wandb_project: \"\"",
                        "  wandb_entity: \"\"",
                        "model:",
                        "  type: transformer",
                        "  topology:",
                        "    type: stacked",
                        "    layers:",
                        "      - type: linear",
                        "        operation:",
                        "          type: matmul",
                        "        weight:",
                        "          type: dense",
                        "          d_in: 128",
                        "          d_out: 128",
                        "          bias: true",
                        "groups:",
                        "  - name: g",
                        "    description: d",
                        "    data: ''",
                        "    runs:",
                        "      - id: r",
                        "        mode: train",
                        "        exp: e",
                        "        seed: 1",
                        "        steps: 2",
                        "        expected: {}",
                    ]
                ),
                encoding="utf-8",
            )

            m = Manifest.from_path(path)
            self.assertEqual(m.model.topology.layers[0].type, LayerType.LINEAR)

    def test_load_json_manifest(self) -> None:
        """
        test loading a JSON manifest.
        """
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "m.json"
            payload = {
                "version": 1,
                "name": "test",
                "notes": "x",
                "defaults": {
                    "wandb": False,
                    "wandb_project": "",
                    "wandb_entity": "",
                },
                "model": {
                    "type": "transformer",
                    "topology": {
                        "type": "stacked",
                        "layers": [
                            {
                                "type": "linear",
                                "operation": {"type": "matmul"},
                                "weight": {
                                    "type": "dense",
                                    "d_in": 128,
                                    "d_out": 128,
                                    "bias": True,
                                },
                            }
                        ],
                    },
                },
                "groups": [
                    {
                        "name": "g",
                        "description": "d",
                        "data": "",
                        "runs": [
                            {
                                "id": "r",
                                "mode": "train",
                                "exp": "e",
                                "seed": 1,
                                "steps": 2,
                                "expected": {},
                            }
                        ],
                    }
                ],
            }
            path.write_text(json.dumps(payload), encoding="utf-8")

            m = Manifest.from_path(path)
            self.assertEqual(m.model.topology.type.value, "stacked")

    def test_resolves_vars(self) -> None:
        """
        test resolving vars in a manifest payload.
        """
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "m.yml"
            path.write_text(
                "\n".join(
                    [
                        "version: 1",
                        "name: test",
                        'notes: "x"',
                        "vars:",
                        "  d_in: 16",
                        "  d_out: 32",
                        "defaults:",
                        "  wandb: false",
                        "  wandb_project: \"\"",
                        "  wandb_entity: \"\"",
                        "model:",
                        "  type: transformer",
                        "  topology:",
                        "    type: stacked",
                        "    layers:",
                        "      - type: linear",
                        "        operation:",
                        "          type: matmul",
                        "        weight:",
                        "          type: dense",
                        "          d_in: \"${d_in}\"",
                        "          d_out: \"${d_out}\"",
                        "groups: []",
                    ]
                ),
                encoding="utf-8",
            )

            m = Manifest.from_path(path)
            layer = m.model.topology.layers[0]
            self.assertEqual(layer.type, LayerType.LINEAR)
            assert isinstance(layer, LinearLayerConfig)
            self.assertEqual(layer.weight.d_in, 16)
            self.assertEqual(layer.weight.d_out, 32)

    def test_rejects_invalid_layer_shape(self) -> None:
        """
        test rejecting an invalid layer shape.
        """
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.yml"
            # Old preset style (layer params under `config:`) should fail validation.
            path.write_text(
                "\n".join(
                    [
                        "version: 1",
                        "notes: x",
                        "defaults:",
                        "  wandb: false",
                        "  wandb_project: \"\"",
                        "  wandb_entity: \"\"",
                        "model:",
                        "  type: transformer",
                        "  topology:",
                        "    type: stacked",
                        "    layers:",
                        "      - type: linear",
                        "        config:",
                        "          d_in: 128",
                        "          d_out: 128",
                        "groups: []",
                    ]
                ),
                encoding="utf-8",
            )

            with self.assertRaises(ValidationError):
                _ = Manifest.from_path(path)
