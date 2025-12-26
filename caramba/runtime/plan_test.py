from __future__ import annotations

from pathlib import Path

from caramba.runtime.plan import RuntimePlan, load_plan, make_plan_key, save_plan


def test_plan_roundtrip(tmp_path: Path) -> None:
    payload = {"a": 1, "b": {"c": 2}}
    key = make_plan_key(payload)
    plan = RuntimePlan(
        key=key,
        device="cpu",
        torch_version="x",
        dtype="float32",
        use_amp=False,
        amp_dtype="bfloat16",
        batch_size=4,
        compile=False,
        compile_mode="reduce-overhead",
    )
    path = tmp_path / "plan.json"
    save_plan(path, plan, payload=payload)
    loaded = load_plan(path)
    assert loaded is not None
    # Assert all fields are correctly serialized/deserialized.
    assert loaded.key == plan.key
    assert loaded.device == plan.device
    assert loaded.torch_version == plan.torch_version
    assert loaded.dtype == plan.dtype
    assert loaded.use_amp == plan.use_amp
    assert loaded.amp_dtype == plan.amp_dtype
    assert loaded.batch_size == plan.batch_size
    assert loaded.compile == plan.compile
    assert loaded.compile_mode == plan.compile_mode
