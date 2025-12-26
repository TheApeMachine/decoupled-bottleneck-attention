"""Shared utility functions for benchmark modules.

Common helpers that multiple benchmark modules need, like extracting
vocabulary size from different model types.
"""
from __future__ import annotations

from torch import nn


def get_model_vocab_size(model: nn.Module, default: int = 32000) -> int:
    """Extract vocabulary size from a model.

    Tries multiple common patterns:
    1. model.vocab_size (simple models, test dummies)
    2. model.config.vocab_size (HuggingFace style)
    3. model.get_input_embeddings().num_embeddings
    4. Falls back to default

    Args:
        model: The model to inspect
        default: Fallback value if vocab size can't be determined

    Returns:
        The vocabulary size as an integer
    """
    # Direct vocab_size attribute
    if hasattr(model, "vocab_size"):
        vocab_size = getattr(model, "vocab_size")
        if isinstance(vocab_size, int) and vocab_size > 0:
            return vocab_size

    # HuggingFace-style config
    if hasattr(model, "config") and hasattr(model.config, "vocab_size"):  # type: ignore[union-attr]
        return int(model.config.vocab_size)  # type: ignore[union-attr]

    # Embedding layer
    if hasattr(model, "get_input_embeddings"):
        embedding = model.get_input_embeddings()  # type: ignore[operator]
        if embedding is not None and hasattr(embedding, "num_embeddings"):
            return int(embedding.num_embeddings)  # type: ignore[union-attr]

    return default
