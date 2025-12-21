"""RNG seeding helpers.

Why this exists:
- Runners want deterministic behavior across torch+cuda where possible.
- We keep imports local to avoid importing torch on module import.
"""

from __future__ import annotations


def set_seed(seed: int) -> None:
    """Why: make training/sampling reproducible across Python and torch RNGs."""
    import random
    import torch

    random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


