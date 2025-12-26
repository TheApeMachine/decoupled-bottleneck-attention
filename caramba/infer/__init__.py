"""Inference utilities with KV-cache management.

Inference is where all the training pays off. This package provides:
- KV-cache creation and management for efficient autoregressive generation
- Standard greedy/sampling generation loops
- Speculative decoding for faster inference with a draft model
- Support for both standard attention and DBA caches
"""
from caramba.infer.context import InferContext, causal_mask
from caramba.infer.generate import (
    GenerateConfig,
    Generator,
    create_caches,
    generate,
    sample_next_token,
)
from caramba.infer.speculative import (
    SpeculativeConfig,
    SpeculativeGenerator,
    speculative_generate,
)

__all__ = [
    "InferContext",
    "causal_mask",
    "Generator",
    "GenerateConfig",
    "generate",
    "create_caches",
    "sample_next_token",
    "SpeculativeConfig",
    "SpeculativeGenerator",
    "speculative_generate",
]
