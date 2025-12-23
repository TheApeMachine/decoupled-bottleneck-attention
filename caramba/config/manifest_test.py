"""
manifest_test provides tests for JSON/YAML manifest loading.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from pydantic import ValidationError

from caramba.config.layer import LayerType
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
                        "model:",
                        "  type: transformer",
                        "  network:",
                        "    type: stacked",
                        "    layers:",
                        "      - type: linear",
                        "        d_in: 128",
                        "        d_out: 128",
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
                "defaults": {"wandb": False},
                "model": {
                    "type": "transformer",
                    "network": {
                        "type": "stacked",
                        "layers": [{"type": "linear", "d_in": 128, "d_out": 128}],
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
                        "defaults: {wandb: false}",
                        "model:",
                        "  type: transformer",
                        "  network:",
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
