"""Perplexity benchmark: measuring language modeling quality.

Perplexity is exp(average cross-entropy loss) over tokens. Lower is better.
For upcycling, we want the student's perplexity to be close to the teacher's,
proving we've preserved model quality while changing the architecture.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader

from caramba.config.benchmark import PerplexityBenchmarkConfig
from caramba.data.npy import NpyDataset


@dataclass
class PerplexityResult:
    """Results from a perplexity benchmark run."""

    model_name: str
    perplexity: float
    loss: float
    num_tokens: int
    num_batches: int


class PerplexityBenchmark:
    """Measures perplexity on a token dataset.

    Computes cross-entropy loss over the dataset and converts to perplexity.
    The dataset is cached so multiple models can be evaluated without
    reloading the data.
    """

    def __init__(
        self, config: PerplexityBenchmarkConfig, device: torch.device
    ) -> None:
        """Set up the benchmark with config and target device."""
        self.config = config
        self.device = device
        self._dataset: NpyDataset | None = None
        self._loader: DataLoader[tuple[Tensor, Tensor]] | None = None

    def _get_loader(self) -> DataLoader[tuple[Tensor, Tensor]]:
        """Lazily initialize and cache the dataset and dataloader."""
        if self._dataset is None:
            self._dataset = NpyDataset(
                self.config.dataset, block_size=self.config.block_size
            )
        if self._loader is None:
            self._loader = DataLoader(
                self._dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                drop_last=True,
            )
        return self._loader

    def run(self, model: nn.Module, model_name: str) -> PerplexityResult:
        """Run the perplexity benchmark on a model.

        Iterates through the dataset, computing cross-entropy loss for each
        batch, then converts total loss to perplexity.
        """
        model.eval()
        loader = self._get_loader()

        total_loss = 0.0
        total_tokens = 0
        num_batches = 0

        with torch.no_grad():
            for x, y in loader:
                x = x.to(self.device)
                y = y.to(self.device)

                logits = model(x)

                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    y.view(-1),
                    reduction="sum",
                )

                batch_tokens = y.numel()
                total_loss += float(loss)
                total_tokens += batch_tokens
                num_batches += 1

                if self.config.num_batches and num_batches >= self.config.num_batches:
                    break

        avg_loss = total_loss / total_tokens if total_tokens > 0 else float("inf")
        perplexity = math.exp(avg_loss) if avg_loss < 20 else float("inf")

        return PerplexityResult(
            model_name=model_name,
            perplexity=perplexity,
            loss=avg_loss,
            num_tokens=total_tokens,
            num_batches=num_batches,
        )
