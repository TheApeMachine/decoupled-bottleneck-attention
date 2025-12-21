"""Decoupled attention implementation (internal package).

Why this exists:
- `production.attention_impl.decoupled_attention` is a stable import path used by
  `production/attention.py`.
- The implementation is large (PyTorch + optional Triton kernels), so we split it
  into small modules here while keeping the public wrapper tiny.
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


