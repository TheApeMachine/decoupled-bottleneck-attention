"""
Diffusion head configuration for the caramba framework.

This module provides Pydantic-based configuration for the diffusion
next-token head adapter, supporting serialization to/from YAML manifests.
"""

from __future__ import annotations

import enum
from typing import Literal
from pydantic import Field

from caramba.config import Config, PositiveFloat, PositiveInt, Probability


class DiffusionScheduler(str, enum.Enum):
    """Available diffusion schedulers from 🤗 diffusers."""
    DDPM = "ddpm"
    DDIM = "ddim"
    DPM = "dpm"


class DiffusionHeadConfig(Config):
    """Configuration for the diffusion-based next-token head.

    This head learns to denoise the target token embedding conditioned
    on transformer hidden states. It's a lightweight adapter that can
    be trained while keeping the transformer backbone frozen.

    Attributes:
        enabled: Whether the diffusion head is active
        num_train_timesteps: Total DDPM training timesteps (default 1000)
        num_infer_steps: Inference steps with fast schedulers (default 12)
        time_embed_dim: Dimension of sinusoidal time embedding
        mlp_mult: Hidden layer size = mlp_mult * embed_dim
        cfg_dropout_p: Conditioning dropout for classifier-free guidance
        cfg_guidance_scale: Inference guidance scale (1.0 = no guidance)
        scheduler: Which diffusers scheduler to use
        loss_weight: Weight when combining with cross-entropy loss

    Example YAML:
        diffusion_head:
          enabled: true
          scheduler: ddim
          num_infer_steps: 12
          cfg_guidance_scale: 1.5
    """
    enabled: bool = False
    num_train_timesteps: PositiveInt = 1000
    num_infer_steps: PositiveInt = 12
    time_embed_dim: PositiveInt = 128
    mlp_mult: PositiveInt = 4
    cfg_dropout_p: Probability = 0.10
    cfg_guidance_scale: PositiveFloat = 1.5
    scheduler: DiffusionScheduler = DiffusionScheduler.DDIM
    loss_weight: PositiveFloat = 0.10

    def to_runtime_config(self) -> object:
        """Convert to the runtime dataclass used by the layer.

        Returns:
            The runtime DiffusionHeadConfig dataclass from layer module
        """
        from caramba.layer.diffusion_head import (
            DiffusionHeadConfig as RuntimeConfig,
        )
        return RuntimeConfig(
            enabled=self.enabled,
            num_train_timesteps=self.num_train_timesteps,
            num_infer_steps=self.num_infer_steps,
            time_embed_dim=self.time_embed_dim,
            mlp_mult=self.mlp_mult,
            cfg_dropout_p=self.cfg_dropout_p,
            cfg_guidance_scale=self.cfg_guidance_scale,
            scheduler=self.scheduler.value,
            loss_weight=self.loss_weight,
        )


def NoDiffusionHeadConfig() -> DiffusionHeadConfig:
    """Factory function that returns a disabled DiffusionHeadConfig.

    Used as a default_factory in ModelConfig to provide a disabled
    diffusion head config without requiring any arguments.
    """
    return DiffusionHeadConfig(enabled=False)
