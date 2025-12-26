"""
utils provides shared utility functions for benchmark modules.
"""
from __future__ import annotations

from torch import nn


def get_model_vocab_size(model: nn.Module, default: int = 32000) -> int:
    """Get vocab size from model, with fallback to default.

    Attempts to extract the vocabulary size from a model by checking:
    1. model.vocab_size attribute (common for simple models)
    2. model.config.vocab_size attribute (HuggingFace style)
    3. model.get_input_embeddings().num_embeddings attribute
    4. Falls back to the provided default value

    Args:
        model: The model to extract vocab size from
        default: Fallback value if vocab size cannot be determined (default: 32000)

    Returns:
        The vocabulary size as an integer
    """
    # Try direct vocab_size attribute (common for simple models and test dummies)
    if hasattr(model, "vocab_size"):
        vocab_size = getattr(model, "vocab_size")
        if isinstance(vocab_size, int) and vocab_size > 0:
            return vocab_size

    # Try common config attributes (HuggingFace style)
    if hasattr(model, "config") and hasattr(model.config, "vocab_size"):  # type: ignore[union-attr]
        return int(model.config.vocab_size)  # type: ignore[union-attr]

    # Try getting from embedding layer
    if hasattr(model, "get_input_embeddings"):
        embedding = model.get_input_embeddings()  # type: ignore[operator]
        if embedding is not None and hasattr(embedding, "num_embeddings"):
            return int(embedding.num_embeddings)  # type: ignore[union-attr]

    # Fallback to a reasonable default
    return default
