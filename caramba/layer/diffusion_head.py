"""
Diffusion-based next-token head (adapter) conditioned on transformer features.

This is an embedding-space diffusion head: we diffuse the target token embedding
and denoise it conditioned on the transformer's pre-logit features.
Inspired by Diffusion-LM but used as an adapter/head rather than a full standalone generator.

Key design points:
- Uses 🤗 diffusers schedulers for add_noise(...) and step(...) sampling
- Can be trained while freezing the transformer backbone (cheap experiments on laptops)
- Supports classifier-free guidance for improved generation quality
- Dependency-gated: works without diffusers (just can't instantiate the head)
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import importlib.util
import math
from typing import cast

from typing_extensions import override

import torch
from torch import nn, Tensor
import torch.nn.functional as F


def _spec_exists(name: str) -> bool:
    """Check if a module spec exists without importing."""
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ModuleNotFoundError):
        return False


DIFFUSERS_AVAILABLE: bool = _spec_exists("diffusers")


@dataclass(frozen=True)
class DiffusionHeadConfig:
    """Configuration for the diffusion next-token head.

    Attributes:
        enabled: Whether the diffusion head is active
        num_train_timesteps: Total diffusion timesteps for training
        num_infer_steps: Inference steps (fewer = faster, can use DDIM/DPM)
        time_embed_dim: Dimension of sinusoidal time embedding
        mlp_mult: Hidden layer multiplier for denoiser MLP
        cfg_dropout_p: Classifier-free guidance dropout probability during training
        cfg_guidance_scale: Guidance scale for inference (1.0 = no guidance)
        scheduler: Which diffusers scheduler to use ("ddpm", "ddim", "dpm")
        loss_weight: Weight for diffusion loss when combined with CE loss
    """
    enabled: bool = False
    num_train_timesteps: int = 1000
    num_infer_steps: int = 12
    time_embed_dim: int = 128
    mlp_mult: int = 4
    cfg_dropout_p: float = 0.10
    cfg_guidance_scale: float = 1.5
    scheduler: str = "ddim"  # "ddpm" | "ddim" | "dpm"
    loss_weight: float = 0.10


@dataclass(frozen=True)
class StepOutput:
    """Output from a scheduler step."""
    prev_sample: Tensor


class DiffusersSchedulerAdapter:
    """Type-safe wrapper around diffusers scheduler objects.

    This provides explicit typing for the diffusers scheduler API,
    avoiding dynamic attribute access throughout the codebase.
    """

    def __init__(self, inner: object) -> None:
        self._inner: object = inner

    @property
    def timesteps(self) -> Tensor:
        """Get the inference timesteps tensor."""
        ts = getattr(self._inner, "timesteps", None)
        if not isinstance(ts, Tensor):
            raise TypeError("diffusers scheduler timesteps must be a torch.Tensor")
        return ts

    def add_noise(
        self,
        original_samples: Tensor,
        noise: Tensor,
        timesteps: Tensor,
    ) -> Tensor:
        """Add noise to samples at the given timesteps."""
        fn = getattr(self._inner, "add_noise", None)
        if not callable(fn):
            raise TypeError("diffusers scheduler missing add_noise")
        out = fn(original_samples, noise, timesteps)
        if not isinstance(out, Tensor):
            raise TypeError("diffusers scheduler add_noise must return Tensor")
        return out

    def set_timesteps(self, num_inference_steps: int, *, device: torch.device) -> None:
        """Configure the inference timestep schedule."""
        fn = getattr(self._inner, "set_timesteps", None)
        if not callable(fn):
            raise TypeError("diffusers scheduler missing set_timesteps")
        _ = fn(int(num_inference_steps), device=device)

    def step(
        self,
        model_output: Tensor,
        timestep: object,
        sample: Tensor,
    ) -> StepOutput:
        """Perform a single reverse diffusion step."""
        fn = getattr(self._inner, "step", None)
        if not callable(fn):
            raise TypeError("diffusers scheduler missing step")
        out = fn(model_output, timestep, sample)
        prev = getattr(out, "prev_sample", None)
        if not isinstance(prev, Tensor):
            raise TypeError("diffusers scheduler step output missing prev_sample")
        return StepOutput(prev_sample=prev)


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal positional embedding for diffusion timesteps.

    Uses the same formulation as the original Transformer paper
    but applied to scalar timesteps rather than sequence positions.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim: int = int(dim)

    @override
    def forward(self, t: Tensor) -> Tensor:
        """Embed timesteps.

        Args:
            t: Timestep tensor (B,) or scalar

        Returns:
            Embeddings (B, dim)
        """
        if t.dim() != 1:
            t = t.view(-1)
        half = int(self.dim // 2)
        device = t.device
        dtype = torch.float32
        denom = float(max(1, half))
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(0, half, device=device, dtype=dtype) / denom
        )
        args = t.to(dtype=dtype).unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class PerTokenDenoiser(nn.Module):
    """Per-token denoiser network.

    A compact MLP that predicts noise given:
    - x_t: The noisy embedding at timestep t
    - cond: Conditioning from transformer hidden states
    - t_embed: Time embedding

    No attention here - the transformer has already done the sequence mixing.
    This keeps the adapter cheap for laptop training.
    """

    def __init__(
        self,
        *,
        embed_dim: int,
        time_embed_dim: int,
        mlp_mult: int,
    ) -> None:
        super().__init__()
        self.embed_dim: int = int(embed_dim)

        in_dim = 2 * int(embed_dim) + int(time_embed_dim)
        hid = int(max(64, int(mlp_mult) * int(embed_dim)))

        self.in_ln: nn.LayerNorm = nn.LayerNorm(in_dim)
        self.fc1: nn.Linear = nn.Linear(in_dim, hid)
        self.fc2: nn.Linear = nn.Linear(hid, hid)
        self.fc3: nn.Linear = nn.Linear(hid, int(embed_dim))

    @override
    def forward(
        self,
        *,
        x_t: Tensor,
        cond: Tensor,
        t_embed: Tensor,
    ) -> Tensor:
        """Predict noise.

        Args:
            x_t: Noisy embeddings (B, T, D)
            cond: Conditioning (B, T, D)
            t_embed: Time embeddings (B, T, Dt)

        Returns:
            Predicted noise (B, T, D)
        """
        h = torch.cat([x_t, cond, t_embed], dim=-1)
        h = cast(Tensor, self.in_ln(h))
        h = F.silu(cast(Tensor, self.fc1(h)))
        h = F.silu(cast(Tensor, self.fc2(h)))
        return cast(Tensor, self.fc3(h))


class DiffusionNextTokenHead(nn.Module):
    """Diffusion-based next-token head.

    This module learns to denoise the target token embedding conditioned
    on the transformer's pre-logit hidden states. During training, it
    uses a standard DDPM denoising objective. During inference, it uses
    a configured scheduler (DDIM/DPM) for fast sampling.

    The key insight is that we can add diffusion-based generation on top
    of an existing transformer without retraining the backbone. The
    diffusion head can be trained while keeping the transformer frozen.
    """

    def __init__(self, *, embed_dim: int, cfg: DiffusionHeadConfig) -> None:
        super().__init__()
        if not DIFFUSERS_AVAILABLE:
            raise RuntimeError(
                "DiffusionNextTokenHead requires `diffusers` but it is not installed. "
                "Install with: pip install diffusers"
            )

        self.embed_dim: int = int(embed_dim)
        self.cfg: DiffusionHeadConfig = cfg

        self.time_embed: SinusoidalTimeEmbedding = SinusoidalTimeEmbedding(
            int(cfg.time_embed_dim)
        )
        self.time_mlp: nn.Sequential = nn.Sequential(
            nn.Linear(int(cfg.time_embed_dim), int(cfg.time_embed_dim)),
            nn.SiLU(),
            nn.Linear(int(cfg.time_embed_dim), int(cfg.time_embed_dim)),
        )
        self.denoiser: PerTokenDenoiser = PerTokenDenoiser(
            embed_dim=self.embed_dim,
            time_embed_dim=int(cfg.time_embed_dim),
            mlp_mult=int(cfg.mlp_mult),
        )

        self._sched: DiffusersSchedulerAdapter = self._make_scheduler(cfg)

    @staticmethod
    def _make_scheduler(cfg: DiffusionHeadConfig) -> DiffusersSchedulerAdapter:
        """Instantiate the appropriate diffusers scheduler."""
        mod = importlib.import_module("diffusers")
        sched = str(cfg.scheduler or "ddim").strip().lower()

        if sched == "ddpm":
            cls_obj: object | None = getattr(mod, "DDPMScheduler", None)
        elif sched == "dpm":
            cls_obj = getattr(mod, "DPMSolverMultistepScheduler", None)
        else:
            cls_obj = getattr(mod, "DDIMScheduler", None)

        if cls_obj is None or (not callable(cls_obj)):
            raise RuntimeError(
                f"Could not resolve diffusers scheduler class for scheduler={sched!r}"
            )

        inner = cls_obj(num_train_timesteps=int(cfg.num_train_timesteps))
        return DiffusersSchedulerAdapter(inner=inner)

    def _maybe_drop_cond(self, cond: Tensor) -> Tensor:
        """Optionally drop conditioning for classifier-free guidance training.

        During training, we randomly zero out conditioning to teach the model
        to generate both conditionally and unconditionally. At inference time,
        we can interpolate between the two for improved quality.
        """
        p = float(self.cfg.cfg_dropout_p)
        if p <= 0.0 or (not self.training):
            return cond

        B = int(cond.size(0))
        # Drop entire conditioning per sample (broadcast across T, D)
        mask = (torch.rand((B, 1, 1), device=cond.device) >= p).to(dtype=cond.dtype)
        return cond * mask

    def diffusion_loss(
        self,
        *,
        cond: Tensor,
        target_emb: Tensor,
    ) -> Tensor:
        """Compute the denoising training loss.

        One-step DDPM objective:
        1. Sample t ~ U[0..T)
        2. Sample noise
        3. x_t = add_noise(x0, noise, t)
        4. Predict noise with denoiser
        5. Return MSE(predicted_noise, actual_noise)

        Args:
            cond: Conditioning from transformer (B, T, D)
            target_emb: Target token embeddings (B, T, D)

        Returns:
            Scalar MSE loss
        """
        if target_emb.shape != cond.shape:
            raise ValueError(
                f"shape mismatch: cond={tuple(cond.shape)} target_emb={tuple(target_emb.shape)}"
            )

        B = int(cond.size(0))
        device = cond.device

        # One timestep per sample (broadcast across tokens)
        t = torch.randint(
            0, int(self.cfg.num_train_timesteps), (B,), device=device, dtype=torch.int64
        )

        noise = torch.randn_like(target_emb)
        x_t = self._sched.add_noise(target_emb, noise, t)
        cond2 = self._maybe_drop_cond(cond)

        t_emb = cast(Tensor, self.time_mlp(self.time_embed(t))).to(dtype=cond.dtype)
        t_emb = t_emb.unsqueeze(1).expand(-1, target_emb.size(1), -1)

        eps_hat = cast(Tensor, self.denoiser(x_t=x_t, cond=cond2, t_embed=t_emb))
        return F.mse_loss(eps_hat, noise)

    @torch.no_grad()
    def sample_next_logits(
        self,
        *,
        cond_last: Tensor,
        tok_emb_weight_t: Tensor,
        temperature: float = 1.0,
        guidance_scale: float | None = None,
    ) -> Tensor:
        """Sample the next token via reverse diffusion.

        Args:
            cond_last: Transformer features at last position (B, 1, D)
            tok_emb_weight_t: Transposed embedding matrix (D, V)
            temperature: Sampling temperature
            guidance_scale: CFG scale (None = use config default)

        Returns:
            Logits (B, V)
        """
        if cond_last.dim() != 3 or cond_last.size(1) != 1:
            raise ValueError(f"cond_last must be (B, 1, D); got {tuple(cond_last.shape)}")

        gs = float(
            self.cfg.cfg_guidance_scale if guidance_scale is None else guidance_scale
        )
        dev = cond_last.device
        dtype = cond_last.dtype
        B = int(cond_last.size(0))

        # Configure inference schedule
        self._sched.set_timesteps(int(self.cfg.num_infer_steps), device=dev)

        # Start from pure noise
        x = torch.randn((B, 1, self.embed_dim), device=dev, dtype=dtype)

        # Prepare unconditional conditioning for CFG
        cond_uncond = torch.zeros_like(cond_last)

        # Reverse diffusion loop
        for t in self._sched.timesteps:
            # Make per-sample timestep tensor
            if isinstance(t, Tensor):
                t_b = t.expand(B).to(device=dev)
            else:
                t_b = torch.full((B,), int(t), device=dev, dtype=torch.int64)

            t_emb = cast(Tensor, self.time_mlp(self.time_embed(t_b))).to(dtype=dtype)
            t_emb = t_emb.unsqueeze(1)  # (B, 1, Dt)

            eps_cond = self.denoiser(x_t=x, cond=cond_last, t_embed=t_emb)

            if gs != 1.0:
                eps_uncond = self.denoiser(x_t=x, cond=cond_uncond, t_embed=t_emb)
                eps = eps_uncond + gs * (eps_cond - eps_uncond)
            else:
                eps = eps_cond

            step_out = self._sched.step(eps, t, x)
            x = step_out.prev_sample

        # Decode embedding -> logits via tied embedding matrix
        logits = (x @ tok_emb_weight_t.unsqueeze(0)).squeeze(1)

        if temperature != 1.0:
            logits = logits / float(max(1e-6, temperature))

        return logits
