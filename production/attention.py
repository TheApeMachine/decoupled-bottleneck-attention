"""Attention layers (public API).

Implementation lives in `production/attention_impl/` to keep this module small and decoupled.
"""

from __future__ import annotations

from production.attention_impl.decoupled_attention import (
    DecoupledBottleneckAttention,
    TRITON_AVAILABLE,
    _decoupled_qk_cat,
    _decoupled_scores_f32,
    _triton_decoupled_q4q8q4_available,
    neg_inf,
)

__all__ = [
    "TRITON_AVAILABLE",
    "DecoupledBottleneckAttention",
    "_decoupled_qk_cat",
    "_decoupled_scores_f32",
    "_triton_decoupled_q4q8q4_available",
    "neg_inf",
]
