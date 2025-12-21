"""Back-compat wrapper (migrated).

The cache-policy self-optimizer now lives in `production.optimizer.tuner.policy_optimizer`.
This module remains to preserve import paths while we migrate callers.
"""

from __future__ import annotations

from production.optimizer.tuner.policy_optimizer import KVCachePolicySelfOptimizer

__all__ = ["KVCachePolicySelfOptimizer"]


