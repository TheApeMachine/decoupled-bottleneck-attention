"""Diffusion-based next-token head for hybrid generation.

This is an experimental adapter that adds diffusion-based token prediction
on top of a standard autoregressive transformer. Instead of directly sampling
from logits, we run a denoising diffusion process conditioned on the
transformer's hidden states to generate a clean token embedding.

Key benefits:
- Can be trained while the backbone is frozen (cheap laptop experiments)
- Supports classifier-free guidance for improved generation quality
- Uses 🤗 diffusers schedulers for efficient sampling (DDIM, DPM-Solver)

Inspired by Diffusion-LM but used as a lightweight head rather than a
full standalone generator.
"""
from __future__ import annotations

import importlib
import importlib.util
import math
from dataclasses import dataclass
from typing import cast

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from typing_extensions import override


def _spec_exists(name: str) -> bool:
    """Check if a module is available without importing it."""
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ModuleNotFoundError):
        return False


DIFFUSERS_AVAILABLE: bool = _spec_exists("diffusers")


@dataclass(frozen=True)
class DiffusionHeadConfig:
    """Configuration for the diffusion next-token head.

    These settings control the diffusion process: how many timesteps,
    which scheduler to use, classifier-free guidance strength, etc.
    """

    enabled: bool = False
    num_train_timesteps: int = 1000
    num_infer_steps: int = 12
    time_embed_dim: int = 128
    mlp_mult: int = 4
    cfg_dropout_p: float = 0.10
    cfg_guidance_scale: float = 1.5
    scheduler: str = "ddim"
    loss_weight: float = 0.10


@dataclass(frozen=True)
class StepOutput:
    """Output from a diffusion scheduler step."""

    prev_sample: Tensor


class DiffusersSchedulerAdapter:
    """Type-safe wrapper around diffusers scheduler objects.

    Provides explicit typing for the diffusers scheduler API, avoiding
    dynamic attribute access and making the code more maintainable.
    """

    def __init__(self, inner: object) -> None:
        self._inner = inner

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

    The same formulation as the original Transformer paper, but applied
    to scalar timesteps rather than sequence positions.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = int(dim)

    @override
    def forward(self, t: Tensor) -> Tensor:
        """Embed timesteps into continuous vectors.

        Args:
            t: Timestep tensor (B,) with integer timestep values

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
            -math.log(10000.0)
            * torch.arange(0, half, device=device, dtype=dtype)
            / denom
        )
        args = t.to(dtype=dtype).unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class PerTokenDenoiser(nn.Module):
    """Compact MLP that predicts noise from noisy embeddings.

    No attention here—the transformer has already done the sequence mixing.
    This keeps the adapter cheap enough for laptop training.
    """

    def __init__(
        self,
        *,
        embed_dim: int,
        time_embed_dim: int,
        mlp_mult: int,
    ) -> None:
        """Set up the denoiser MLP.

        Input is [noisy_emb, conditioning, time_emb] concatenated.
        Output is predicted noise of same dimension as embeddings.
        """
        super().__init__()
        self.embed_dim = int(embed_dim)

        in_dim = 2 * int(embed_dim) + int(time_embed_dim)
        hid = int(max(64, int(mlp_mult) * int(embed_dim)))

        self.in_ln = nn.LayerNorm(in_dim)
        self.fc1 = nn.Linear(in_dim, hid)
        self.fc2 = nn.Linear(hid, hid)
        self.fc3 = nn.Linear(hid, int(embed_dim))

    @override
    def forward(
        self,
        *,
        x_t: Tensor,
        cond: Tensor,
        t_embed: Tensor,
    ) -> Tensor:
        """Predict noise given noisy embedding and conditioning.

        Args:
            x_t: Noisy embeddings (B, T, D)
            cond: Conditioning from transformer (B, T, D)
            t_embed: Time embeddings (B, T, time_dim)

        Returns:
            Predicted noise (B, T, D)
        """
        h = torch.cat([x_t, cond, t_embed], dim=-1)
        h = cast(Tensor, self.in_ln(h))
        h = F.silu(cast(Tensor, self.fc1(h)))
        h = F.silu(cast(Tensor, self.fc2(h)))
        return cast(Tensor, self.fc3(h))


class DiffusionNextTokenHead(nn.Module):
    """Diffusion-based next-token prediction head.

    During training, learns to denoise target embeddings conditioned on
    transformer hidden states (standard DDPM objective).

    During inference, runs reverse diffusion starting from noise,
    conditioned on the last position's features, to generate a clean
    embedding that's then projected to vocabulary logits.
    """

    def __init__(self, *, embed_dim: int, cfg: DiffusionHeadConfig) -> None:
        """Initialize the diffusion head.

        Args:
            embed_dim: Dimension of token embeddings (d_model)
            cfg: Diffusion configuration (timesteps, scheduler, etc.)
        """
        super().__init__()
        if not DIFFUSERS_AVAILABLE:
            raise RuntimeError(
                "DiffusionNextTokenHead requires `diffusers` but it is not installed. "
                "Install with: pip install diffusers"
            )

        self.embed_dim = int(embed_dim)
        self.cfg = cfg

        self.time_embed = SinusoidalTimeEmbedding(int(cfg.time_embed_dim))
        self.time_mlp = nn.Sequential(
            nn.Linear(int(cfg.time_embed_dim), int(cfg.time_embed_dim)),
            nn.SiLU(),
            nn.Linear(int(cfg.time_embed_dim), int(cfg.time_embed_dim)),
        )
        self.denoiser = PerTokenDenoiser(
            embed_dim=self.embed_dim,
            time_embed_dim=int(cfg.time_embed_dim),
            mlp_mult=int(cfg.mlp_mult),
        )

        self._sched = self._make_scheduler(cfg)

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
        """Randomly drop conditioning for classifier-free guidance training.

        During training, we sometimes zero out the conditioning so the model
        learns to generate both with and without it. At inference, we can
        blend conditioned and unconditioned predictions for better quality.
        """
        p = float(self.cfg.cfg_dropout_p)
        if p <= 0.0 or (not self.training):
            return cond

        B = int(cond.size(0))
        mask = (torch.rand((B, 1, 1), device=cond.device) >= p).to(dtype=cond.dtype)
        return cond * mask

    def diffusion_loss(
        self,
        *,
        cond: Tensor,
        target_emb: Tensor,
    ) -> Tensor:
        """Compute the DDPM training loss.

        One training step:
        1. Sample random timestep t
        2. Add noise to target embedding at timestep t
        3. Predict the noise with the denoiser
        4. Return MSE between predicted and actual noise

        Args:
            cond: Transformer features (B, T, D)
            target_emb: Target token embeddings (B, T, D)

        Returns:
            Scalar MSE loss
        """
        if target_emb.shape != cond.shape:
            raise ValueError(
                f"shape mismatch: cond={tuple(cond.shape)} "
                f"target_emb={tuple(target_emb.shape)}"
            )

        B = int(cond.size(0))
        device = cond.device

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
        """Sample next-token logits via reverse diffusion.

        Starts from noise and iteratively denoises conditioned on the
        transformer's last position features. The final clean embedding
        is projected to vocabulary logits.

        Args:
            cond_last: Transformer features at last position (B, 1, D)
            tok_emb_weight_t: Transposed embedding matrix (D, V)
            temperature: Sampling temperature for final logits
            guidance_scale: CFG scale (None = use config default)

        Returns:
            Logits (B, V)
        """
        if cond_last.dim() != 3 or cond_last.size(1) != 1:
            raise ValueError(
                f"cond_last must be (B, 1, D); got {tuple(cond_last.shape)}"
            )

        gs = float(
            self.cfg.cfg_guidance_scale if guidance_scale is None else guidance_scale
        )
        dev = cond_last.device
        dtype = cond_last.dtype
        B = int(cond_last.size(0))

        self._sched.set_timesteps(int(self.cfg.num_infer_steps), device=dev)

        x = torch.randn((B, 1, self.embed_dim), device=dev, dtype=dtype)
        cond_uncond = torch.zeros_like(cond_last)

        for t in self._sched.timesteps:
            if isinstance(t, Tensor):
                t_b = t.expand(B).to(device=dev)
            else:
                t_b = torch.full((B,), int(t), device=dev, dtype=torch.int64)

            t_emb = cast(Tensor, self.time_mlp(self.time_embed(t_b))).to(dtype=dtype)
            t_emb = t_emb.unsqueeze(1)

            eps_cond = self.denoiser(x_t=x, cond=cond_last, t_embed=t_emb)

            if gs != 1.0:
                eps_uncond = self.denoiser(x_t=x, cond=cond_uncond, t_embed=t_emb)
                eps = eps_uncond + gs * (eps_cond - eps_uncond)
            else:
                eps = eps_cond

            step_out = self._sched.step(eps, t, x)
            x = step_out.prev_sample

        logits = (x @ tok_emb_weight_t.unsqueeze(0)).squeeze(1)

        if temperature != 1.0:
            logits = logits / float(max(1e-6, temperature))

        return logits
