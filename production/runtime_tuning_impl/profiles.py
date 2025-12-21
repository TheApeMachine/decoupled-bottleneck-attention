"""Back-compat wrapper (migrated).

Prefer importing from `production.optimizer.tuner.profiles`.
"""

from __future__ import annotations

from production.optimizer.tuner.profiles import (
    TritonKernelProfile,
    parse_cc_from_device_sig as _parse_cc_from_device_sig,
    get_triton_kernel_profiles,
)

__all__ = [
    "TritonKernelProfile",
    "_parse_cc_from_device_sig",
    "get_triton_kernel_profiles",
]

