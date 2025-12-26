"""Diffusion head configuration for hybrid token generation.

The diffusion head is an optional adapter that adds denoising-based
generation on top of a standard autoregressive transformer. It can
be trained while keeping the backbone frozen, making it cheap to
experiment with on laptops.
"""
from __future__ import annotations

import enum
from typing import Literal

from pydantic import Field

from caramba.config import Config, PositiveFloat, PositiveInt, Probability


class DiffusionScheduler(str, enum.Enum):
    """Available diffusion schedulers from 🤗 diffusers.

    DDPM: Original denoising diffusion (slow but simple)
    DDIM: Faster sampling with deterministic steps
    DPM: Even faster with DPM-Solver++
    """

    DDPM = "ddpm"
    DDIM = "ddim"
    DPM = "dpm"


class DiffusionHeadConfig(Config):
    """Configuration for the diffusion-based next-token head.

    When enabled, the model learns to denoise target embeddings conditioned
    on transformer features. This provides an alternative generation path
    that can be more controllable than pure autoregressive sampling.
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

        The layer module uses a frozen dataclass for performance;
        this method bridges from the Pydantic config.
        """
        from caramba.layer.diffusion_head import DiffusionHeadConfig as RuntimeConfig

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


def no_diffusion_head_config() -> DiffusionHeadConfig:
    """Factory for a disabled diffusion head config."""
    return DiffusionHeadConfig(enabled=False)
