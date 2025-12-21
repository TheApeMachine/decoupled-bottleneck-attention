from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, Optional


def append_jsonl(path: Optional[str], record: Dict[str, Any]) -> None:
    """Append a single JSON record to a .jsonl file (best-effort)."""
    if not path:
        return
    try:
        rec = dict(record)
        rec.setdefault("ts", float(time.time()))
        os.makedirs(os.path.dirname(str(path)) or ".", exist_ok=True)
        with open(str(path), "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, sort_keys=True, default=str) + "\n")
    except Exception:
        # Logging must never break training.
        return


class SelfOptLogger:
    """Unified event logger used by the runner and self-optimization components.

    Goals:
    - One callsite API for emitting structured events
    - Optional JSONL persistence (best-effort)
    - Optional forwarding to `RunLogger` (TensorBoard/W&B/plots)
    - Optional human-readable echo (best-effort)
    """

    def __init__(
        self,
        *,
        jsonl_path: Optional[str] = None,
        run_logger: Optional[Any] = None,
        echo: bool = True,
    ) -> None:
        self.jsonl_path = str(jsonl_path) if jsonl_path else None
        self.run_logger = run_logger
        self.echo = bool(echo)

    def log(self, event: Dict[str, Any], *, msg: Optional[str] = None, echo: Optional[bool] = None) -> None:
        """Log an event, optionally echoing a human-readable message."""
        try:
            # Always forward structured events first (so a failing print doesn't drop metrics).
            if self.run_logger is not None:
                try:
                    self.run_logger.log(event)
                except Exception:
                    pass
            append_jsonl(self.jsonl_path, event)
        finally:
            do_echo = self.echo if echo is None else bool(echo)
            if do_echo and msg:
                try:
                    print(str(msg), flush=True)
                except Exception:
                    pass

    def finalize(self, **kwargs: Any) -> None:
        if self.run_logger is None:
            return
        try:
            self.run_logger.finalize(**kwargs)
        except Exception:
            pass

    def close(self) -> None:
        if self.run_logger is None:
            return
        try:
            self.run_logger.close()
        except Exception:
            pass


