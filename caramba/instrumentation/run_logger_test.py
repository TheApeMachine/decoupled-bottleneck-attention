from __future__ import annotations

import json
from pathlib import Path

from caramba.instrumentation.run_logger import RunLogger


def test_run_logger_writes_jsonl(tmp_path: Path) -> None:
    rl = RunLogger(tmp_path, filename="train.jsonl", enabled=True)
    rl.log_event(type="phase_start", run_id="r1", phase="blockwise", step=0, data={"x": 1})
    rl.log_metrics(run_id="r1", phase="blockwise", step=1, metrics={"loss": 0.1234})
    rl.close()

    p = tmp_path / "train.jsonl"
    lines = p.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    ev0 = json.loads(lines[0])
    ev1 = json.loads(lines[1])

    assert ev0["type"] == "phase_start"
    assert ev0["run_id"] == "r1"
    assert ev0["phase"] == "blockwise"
    assert ev0["step"] == 0

    assert ev1["type"] == "metrics"
    assert ev1["data"]["metrics"]["loss"] == 0.1234


def test_run_logger_best_effort_disabled(tmp_path: Path) -> None:
    rl = RunLogger(tmp_path, enabled=False)
    rl.log_event(type="x", data={"a": 1})
    rl.close()
    assert not (tmp_path / "train.jsonl").exists()

