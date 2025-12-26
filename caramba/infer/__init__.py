"""
caramba.infer provides inference utilities including KV-cache management.
"""
from caramba.infer.context import InferContext, causal_mask
from caramba.infer.generate import (
    Generator,
    GenerateConfig,
    generate,
    create_caches,
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
