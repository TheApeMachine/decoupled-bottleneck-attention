from __future__ import annotations

import pytest

from caramba.instrumentation.live_plotter import LivePlotter


def test_live_plotter_best_effort_no_crash() -> None:
    lp = LivePlotter(enabled=True, title="test", plot_every=1)
    # Even if matplotlib isn't installed, this should not raise.
    lp.update(step=1, scalars={"loss": 1.0})
    lp.close()


def test_live_plotter_with_matplotlib() -> None:
    """Test that plotting works when matplotlib is available."""
    pytest.importorskip("matplotlib")
    lp = LivePlotter(enabled=True, title="test_mpl", plot_every=1)
    # If matplotlib is available, enabled should remain True after init.
    if lp.enabled:
        # Perform a few updates to trigger plotting.
        lp.update(step=1, scalars={"loss": 1.0, "acc": 0.5})
        lp.update(step=2, scalars={"loss": 0.8, "acc": 0.6})
        lp.update(step=3, scalars={"loss": 0.6, "acc": 0.7})
        # Verify internal state is consistent.
        assert lp._fig is not None
        assert lp._ax is not None
        assert len(lp._steps) == 3
    lp.close()
    # After close, internal state should be cleared.
    assert lp._fig is None
    assert lp._ax is None

