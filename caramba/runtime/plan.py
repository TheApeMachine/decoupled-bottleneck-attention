"""Runtime planning and persistence for reuse.

Some runtime decisions depend on the device (dtype, AMP dtype, compile usage)
and are worth caching so repeated runs don't re-decide heuristics every time.

The plan is derived from:
- The manifest/model/train config (minus volatile fields)
- The device and torch version

It is written to disk as JSON and can be reused across runs.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RuntimePlan:
    """Resolved runtime decisions for a specific (device, config) signature."""

    key: str
    device: str
    torch_version: str
    dtype: str
    use_amp: bool
    amp_dtype: str
    batch_size: int
    compile: bool
    compile_mode: str


def _stable_json(obj: object) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def make_plan_key(payload: dict[str, Any]) -> str:
    """Make a stable key for a plan payload."""

    h = hashlib.sha1(_stable_json(payload).encode("utf-8"))
    return h.hexdigest()[:16]


def save_plan(path: Path, plan: RuntimePlan, *, payload: dict[str, Any]) -> None:
    """Persist a plan + the payload that produced it."""

    path.parent.mkdir(parents=True, exist_ok=True)
    blob = {
        "payload": payload,
        "plan": {
            "key": plan.key,
            "device": plan.device,
            "torch_version": plan.torch_version,
            "dtype": plan.dtype,
            "use_amp": plan.use_amp,
            "amp_dtype": plan.amp_dtype,
            "batch_size": plan.batch_size,
            "compile": plan.compile,
            "compile_mode": plan.compile_mode,
        },
    }
    path.write_text(_stable_json(blob) + "\n", encoding="utf-8")


def load_plan(path: Path) -> RuntimePlan | None:
    """Load a plan from disk."""

    try:
        blob = json.loads(path.read_text(encoding="utf-8"))
    except OSError as e:
        logger.debug("Failed to read plan file %s: %s", path, e)
        return None
    except json.JSONDecodeError as e:
        logger.debug("Failed to parse JSON from plan file %s: %s", path, e)
        return None
    if not isinstance(blob, dict):
        logger.debug("Plan file %s does not contain a dict at top level", path)
        return None
    plan = blob.get("plan", None)
    if not isinstance(plan, dict):
        logger.debug("Plan file %s missing 'plan' dict key", path)
        return None
    try:
        return RuntimePlan(
            key=str(plan["key"]),
            device=str(plan["device"]),
            torch_version=str(plan["torch_version"]),
            dtype=str(plan["dtype"]),
            use_amp=bool(plan["use_amp"]),
            amp_dtype=str(plan["amp_dtype"]),
            batch_size=int(plan["batch_size"]),
            compile=bool(plan["compile"]),
            compile_mode=str(plan["compile_mode"]),
        )
    except Exception as e:
        logger.debug("Failed to construct RuntimePlan from %s: %s", path, e)
        return None

