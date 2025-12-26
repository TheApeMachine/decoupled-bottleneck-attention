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
                        "  type: TransformerModel",
                        "  topology:",
                        "    type: StackedTopology",
                        "    layers:",
                        "      - type: LinearLayer",
                        "        d_in: 128",
                        "        d_out: 128",
                        "        bias: true",
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

    def test_load_yaml_manifest_with_compare_verify(self) -> None:
        """
        test loading a YAML manifest with a typed compare verify block.
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
                        "  type: TransformerModel",
                        "  topology:",
                        "    type: StackedTopology",
                        "    layers:",
                        "      - type: LinearLayer",
                        "        d_in: 128",
                        "        d_out: 128",
                        "        bias: true",
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
                        "        verify:",
                        "          type: compare",
                        "          batches: 1",
                        "          logits:",
                        "            max_mean_l1: 1.0",
                        "            max_max_l1: 2.0",
                    ]
                ),
                encoding="utf-8",
            )

            m = Manifest.from_path(path)
            run = m.groups[0].runs[0]
            self.assertIsNotNone(run.verify)

    def test_rejects_compare_verify_without_metrics(self) -> None:
        """
        test rejecting compare verify blocks without attention/logits.
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
                        "  type: TransformerModel",
                        "  topology:",
                        "    type: StackedTopology",
                        "    layers:",
                        "      - type: LinearLayer",
                        "        d_in: 128",
                        "        d_out: 128",
                        "        bias: true",
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
                        "        verify:",
                        "          type: compare",
                        "          batches: 1",
                    ]
                ),
                encoding="utf-8",
            )

            with self.assertRaises(ValidationError):
                _ = Manifest.from_path(path)

    def test_load_yaml_manifest_with_eval_verify(self) -> None:
        """
        test loading a YAML manifest with a typed eval verify block.
        """
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "m.yml"
            path.write_text(
                "\n".join(
                    [
                        "version: 1",
                        "name: test",
                        'notes: \"x\"',
                        "defaults:",
                        "  wandb: false",
                        "  wandb_project: \"\"",
                        "  wandb_entity: \"\"",
                        "model:",
                        "  type: TransformerModel",
                        "  topology:",
                        "    type: StackedTopology",
                        "    layers:",
                        "      - type: LinearLayer",
                        "        d_in: 128",
                        "        d_out: 128",
                        "        bias: true",
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
                        "        verify:",
                        "          type: eval",
                        "          tokenizer:",
                        "            type: tiktoken",
                        "            encoding: gpt2",
                        "          max_new_tokens: 4",
                        "          cases:",
                        "            - id: strawberry_r",
                        "              prompt: \"How many times do we find the letter r in the word strawberry?\"",
                        "              kind: choice_logprob",
                        "              choices: [\"1\", \"2\", \"3\", \"4\"]",
                        "              answer: \"3\"",
                        "          thresholds:",
                        "            min_student_accuracy: 0.0",
                        "            max_accuracy_drop: 1.0",
                    ]
                ),
                encoding="utf-8",
            )

            m = Manifest.from_path(path)
            run = m.groups[0].runs[0]
            self.assertIsNotNone(run.verify)

    def test_load_yaml_manifest_with_kvcache_verify(self) -> None:
        """
        test loading a YAML manifest with a typed kvcache verify block.
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
                        "  type: TransformerModel",
                        "  topology:",
                        "    type: StackedTopology",
                        "    layers:",
                        "      - type: LinearLayer",
                        "        d_in: 128",
                        "        d_out: 128",
                        "        bias: true",
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
                        "        verify:",
                        "          type: kvcache",
                        "          n_layers: 1",
                        "          batch_size: 1",
                        "          max_seq_len: 8",
                        "          teacher:",
                        "            k: { kind: fp16, qblock: 32, residual_len: 0 }",
                        "            v: { kind: fp16, qblock: 32, residual_len: 0 }",
                        "          student:",
                        "            k_sem: { kind: q4_0, qblock: 32, residual_len: 0 }",
                        "            k_geo: { kind: q8_0, qblock: 32, residual_len: 0 }",
                        "            v: { kind: q4_0, qblock: 32, residual_len: 0 }",
                    ]
                ),
                encoding="utf-8",
            )

            m = Manifest.from_path(path)
            run = m.groups[0].runs[0]
            self.assertIsNotNone(run.verify)

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
                    "type": "TransformerModel",
                    "topology": {
                        "type": "StackedTopology",
                        "layers": [
                            {
                                "type": "LinearLayer",
                                "d_in": 128,
                                "d_out": 128,
                                "bias": True,
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
            self.assertEqual(m.model.topology.type.value, "StackedTopology")

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
                        "  type: TransformerModel",
                        "  topology:",
                        "    type: StackedTopology",
                        "    layers:",
                        "      - type: LinearLayer",
                        "        d_in: \"${d_in}\"",
                        "        d_out: \"${d_out}\"",
                        "groups: []",
                    ]
                ),
                encoding="utf-8",
            )

            m = Manifest.from_path(path)
            layer = m.model.topology.layers[0]
            self.assertEqual(layer.type, LayerType.LINEAR)
            assert isinstance(layer, LinearLayerConfig)
            self.assertEqual(layer.d_in, 16)
            self.assertEqual(layer.d_out, 32)

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
                        "  type: TransformerModel",
                        "  topology:",
                        "    type: StackedTopology",
                        "    layers:",
                        "      - type: LinearLayer",
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
