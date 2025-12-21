"""Decoupled bottleneck attention (public wrapper).

Why this exists:
- `production/attention.py` (and callers) import from this stable path.
- The implementation is split into `decoupled_attention_impl/` to keep files small.
"""

from __future__ import annotations

from production.attention_impl.decoupled_attention_impl.public import (
    DecoupledBottleneckAttention,
    TRITON_AVAILABLE,
    _decoupled_qk_cat,
    _decoupled_scores_f32,
    _triton_decoupled_q4q8q4_available,
    neg_inf,
)

__all__ = [
    "DecoupledBottleneckAttention",
    "TRITON_AVAILABLE",
    "_decoupled_qk_cat",
    "_decoupled_scores_f32",
    "_triton_decoupled_q4q8q4_available",
    "neg_inf",
]


