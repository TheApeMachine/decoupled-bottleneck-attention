"""Legacy config helpers (implementation).

Why this exists:
- `production/config.py` is a stable public import path used across runners/CLI.
- Splitting the implementation keeps each concern small (device selection,
  dataset/layer inference, intent derivation, experiment presets).
"""

from __future__ import annotations


