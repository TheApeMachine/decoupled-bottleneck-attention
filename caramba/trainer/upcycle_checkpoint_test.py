from __future__ import annotations

import pytest

from caramba.trainer.upcycle import Upcycle


def test_validate_checkpoint_state_rejects_missing_keys() -> None:
    bad: dict[str, object] = {"run_id": "r", "phase": "global", "step": 1}
    with pytest.raises(ValueError):
        Upcycle._validate_checkpoint_state(bad)


def test_validate_checkpoint_state_accepts_valid_state() -> None:
    """Test that a complete, valid checkpoint state dict is accepted."""
    # Construct a dict with all required keys expected by the validator.
    valid_state: dict[str, object] = {
        "run_id": "test_run",
        "phase": "global",
        "step": 100,
        "model_state_dict": {},
        "optimizer_state_dict": {},
    }
    # Should not raise; if it does, the test fails.
    try:
        Upcycle._validate_checkpoint_state(valid_state)
    except ValueError as e:
        pytest.fail(f"Valid state should be accepted, but got ValueError: {e}")

