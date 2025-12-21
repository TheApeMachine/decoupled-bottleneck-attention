"""Back-compat wrapper (migrated).

The decode-plan self-optimizer now lives in `production.optimizer.tuner.decode_optimizer`.
This module remains to preserve import paths while we migrate callers.
"""

from __future__ import annotations

from production.optimizer.tuner.decode_optimizer import KVDecodeSelfOptimizer

__all__ = ["KVDecodeSelfOptimizer"]


