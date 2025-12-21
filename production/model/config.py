"""
Model architecture configuration (self-optimizing).

Allows the model to auto-fit the task, which makes it very easy to
run on different kinds of hardware, without having to manage all
the configuration. This helps a lot while running experiments.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal

import torch


class Mode(Enum):
    """
    Mode is equivalent to the attention layout of the model. This allows us to
    compare different attention architectures without duplicating code.
    """
    BASELINE = "baseline"
    GQA = "gqa"
    BOTTLENECK = "bottleneck"
    DECOUPLED = "decoupled"


@dataclass
class ModelConfig:
    """
    ModelConfig acts as the main entry point for the model configuration, while at the
    same time grouping some of the easier self-optimization steps.
    """

    def __init__(self, device: torch.device | None = None) -> None:
        self.device: torch.device = device if device is not None else torch.device("cpu")
        self.vocab_size: int = 0
        self.block_size: int = 0

        self.n_layer: int = 0
        self.n_head: int = 0
        self.kv_head: int | None = None  # for GQA: number of KV heads (defaults to n_head)
        self.d_model: int = 0
        self.dim_multiplier: int = 0
        self.d_ff: int = 0

        self.embed_dim: int = 0  # lexical bottleneck if < d_model

        self.attn_mode: Mode = Mode.BOTTLENECK
        self.attn_dim: int = 0
        self.head_dim: int = 0
        self.sem_dim: int = 0
        self.geo_dim: int = 0

        self.decoupled_gate: bool = True

        self.rope: bool = True
        self.rope_base: float = 10000.0

        self.tie_qk: bool = False
        self.null_attn: bool = False
        self.learned_temp: bool = True

        self.mlp: Literal["swiglu", "gelu"] = "swiglu"
        self.dropout: float = 0.0

    def optimize(self) -> None:
        """
        optimize the main model dimensions based on the task entropy and model capacity.
        """
        self.fit_core_dims()
        self.fit_n_layer()
        self.fit_dimensions()

    def fit_n_layer(self) -> None:
        """
        fit_n_layer decides the number of layers for the model based on the task
        entropy and the model capacity.
        """
        # More depth for width + semantic variety; less depth for long contexts.
        self.n_layer = int(
            (self.d_model / self.n_head) +
            (self.vocab_size.bit_length() // 2) -
            (self.block_size.bit_length() // 2)
        ) + 2

    def derive_n_head(self, d_model: int) -> int:
        """
        attention heads must evenly partition the model width; we choose the
        divisor that best matches an entropy-driven target head resolution.
        """
        target = (
            self.vocab_size.bit_length() + self.block_size.bit_length()
        ) * self.dim_multiplier
        
        return min(
            [i for i in range(1, d_model + 1) if d_model % i == 0],
            key=lambda i: abs(d_model // i - target)
        )

    def fit_core_dims(self) -> None:
        """
        set width, head count, MLP expansion, and embedding bottleneck from
        task entropy + model capacity so we avoid brittle, static ratios.
        """
        v_bits, b_bits = self.vocab_size.bit_length(), self.block_size.bit_length()

        # 1. Model Width: Scales with total entropy, boosted by semantic density.
        self.d_model = (v_bits + b_bits) << (v_bits // 4)

        # 2. Resolution multiplier scales with the model's total bit-depth.
        self.dim_multiplier = self.d_model.bit_length() // 2
        self.n_head = self.derive_n_head(self.d_model)

        # 3. MLP Expansion: Scales with the 'Meaning Bits' (semantic complexity) of the task.
        self.d_ff = self.d_model * max(2, v_bits // 4)

        # 4. Lexical Bottleneck: Scales bitwise with the information gap between model and vocab.
        self.embed_dim = self.d_model >> max(0, self.d_model.bit_length() - v_bits)

    def fit_dimensions(self) -> None:
        """
        attention width and its semantic/geometric split depend on the chosen
        attention architecture (baseline/bottleneck/decoupled/GQA).
        """
        self.attn_dim = self.attn_dim_for_mode()
        self.kv_head = self.kv_head_for_mode()
        self.head_dim = self.attn_dim // max(1, self.n_head)
        self.geo_dim = self.geo_dim_for_mode()
        self.sem_dim = self.attn_dim - self.geo_dim

    def attn_dim_for_mode(self) -> int:
        """bottleneck tightens as context cost increases relative to semantic variety."""
        v, b = self.vocab_size.bit_length(), self.block_size.bit_length()
        return self.d_model * v // b // self.n_head * self.n_head if self.attn_mode in (
            Mode.BOTTLENECK, Mode.DECOUPLED
        ) else self.d_model

    def kv_head_for_mode(self) -> int | None:
        """GQA reduces KV bandwidth by sharing KV across query heads when it is safe."""
        return self.n_head // max(
            1, self.block_size.bit_length() // self.n_layer
        ) if self.attn_mode == Mode.GQA else None

    def geo_dim_for_mode(self) -> int:
        """decoupled attention allocates some capacity to position/geometry; RoPE prefers even dims."""
        if self.attn_mode != Mode.DECOUPLED:
            return 0
        v, b = self.vocab_size.bit_length(), self.block_size.bit_length()
        geo = (self.attn_dim // self.n_head) * b // (v + b)
        return (geo & ~1 if self.rope else geo) * self.n_head
