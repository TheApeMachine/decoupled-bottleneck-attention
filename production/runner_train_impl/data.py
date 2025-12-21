"""Dataset loading for training runner.

Why this exists:
- Training wants views + a stable batching function; the underlying storage can vary.
- Keeping dataset I/O separate reduces noise in the training loop and helps with profiling.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import cast

import torch

from production.run_config import TrainConfig


@dataclass(frozen=True)
class DatasetState:
    """Why: bundle together the minimal dataset objects the runner needs."""

    train_view: object
    val_view: object
    vocab_size: int
    n_total_tokens: int
    fmt: str


def load_dataset(run_cfg: TrainConfig) -> DatasetState:
    """Why: resolve format, load tokens (mmap when possible), split, and infer vocab if needed."""
    import production.data as data_mod  # pylint: disable=import-outside-toplevel

    data_path = Path(str(run_cfg.data))
    fmt = data_mod.infer_data_format(data_path, str(run_cfg.data_format))
    tokens_any = data_mod.load_tokens_any(path=data_path, fmt=fmt, data_dtype=str(run_cfg.data_dtype))

    vocab = run_cfg.vocab_size
    if vocab is None:
        vocab = data_mod.determine_vocab_size(tokens_any=tokens_any, vocab_size=None, tokenizer=str(run_cfg.tokenizer))
    vocab_i = int(cast(int, vocab))

    n_total = int(tokens_any.numel()) if isinstance(tokens_any, torch.Tensor) else int(len(tokens_any))
    train_view, val_view = data_mod.split_train_val(tokens_any, val_frac=float(run_cfg.val_frac))

    return DatasetState(
        train_view=train_view,
        val_view=val_view,
        vocab_size=vocab_i,
        n_total_tokens=int(n_total),
        fmt=str(fmt),
    )


