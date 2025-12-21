"""Time helpers.

Why this exists:
- Many logs/artifacts want an ISO timestamp, but we avoid sprinkling datetime
  formatting everywhere.
"""

from __future__ import annotations

import datetime


def now_iso() -> str:
    """Why: consistent, human-friendly timestamps for logs and metadata."""
    return datetime.datetime.now(datetime.timezone.utc).astimezone().isoformat(timespec="seconds")


