"""
Tests for the diffusion next-token head.
"""

from __future__ import annotations

import pytest
import torch
from torch import Tensor

from caramba.layer.diffusion_head import (
    DIFFUSERS_AVAILABLE,
    DiffusionHeadConfig,
    DiffusionNextTokenHead,
    PerTokenDenoiser,
    SinusoidalTimeEmbedding,
)


class TestSinusoidalTimeEmbedding:
    """Tests for sinusoidal time embedding."""

    def test_forward_shape(self) -> None:
        """Test that embedding has correct output shape."""
        dim = 128
        batch_size = 4
        emb = SinusoidalTimeEmbedding(dim)
        t = torch.randint(0, 1000, (batch_size,))
        out = emb(t)
        assert out.shape == (batch_size, dim)

    def test_forward_odd_dim(self) -> None:
        """Test that odd dimensions work correctly."""
        dim = 127
        batch_size = 4
        emb = SinusoidalTimeEmbedding(dim)
        t = torch.randint(0, 1000, (batch_size,))
        out = emb(t)
        assert out.shape == (batch_size, dim)

    def test_different_timesteps_produce_different_embeddings(self) -> None:
        """Test that different timesteps produce different embeddings."""
        dim = 64
        emb = SinusoidalTimeEmbedding(dim)
        t1 = torch.tensor([0])
        t2 = torch.tensor([500])
        out1 = emb(t1)
        out2 = emb(t2)
        assert not torch.allclose(out1, out2)


class TestPerTokenDenoiser:
    """Tests for the per-token denoiser network."""

    def test_forward_shape(self) -> None:
        """Test that denoiser has correct output shape."""
        embed_dim = 256
        time_embed_dim = 128
        batch_size = 2
        seq_len = 16

        denoiser = PerTokenDenoiser(
            embed_dim=embed_dim,
            time_embed_dim=time_embed_dim,
            mlp_mult=4,
        )

        x_t = torch.randn(batch_size, seq_len, embed_dim)
        cond = torch.randn(batch_size, seq_len, embed_dim)
        t_embed = torch.randn(batch_size, seq_len, time_embed_dim)

        out = denoiser(x_t=x_t, cond=cond, t_embed=t_embed)
        assert out.shape == (batch_size, seq_len, embed_dim)

    def test_gradients_flow(self) -> None:
        """Test that gradients flow through the denoiser."""
        embed_dim = 64
        time_embed_dim = 32

        denoiser = PerTokenDenoiser(
            embed_dim=embed_dim,
            time_embed_dim=time_embed_dim,
            mlp_mult=2,
        )

        x_t = torch.randn(1, 4, embed_dim, requires_grad=True)
        cond = torch.randn(1, 4, embed_dim)
        t_embed = torch.randn(1, 4, time_embed_dim)

        out = denoiser(x_t=x_t, cond=cond, t_embed=t_embed)
        loss = out.sum()
        loss.backward()

        assert x_t.grad is not None
        assert x_t.grad.shape == x_t.shape


@pytest.mark.skipif(not DIFFUSERS_AVAILABLE, reason="diffusers not installed")
class TestDiffusionNextTokenHead:
    """Tests for the full diffusion head (requires diffusers)."""

    @pytest.fixture
    def head(self) -> DiffusionNextTokenHead:
        """Create a diffusion head for testing."""
        cfg = DiffusionHeadConfig(
            enabled=True,
            num_train_timesteps=100,  # Smaller for faster tests
            num_infer_steps=4,
            time_embed_dim=64,
            mlp_mult=2,
            cfg_dropout_p=0.1,
            cfg_guidance_scale=1.5,
            scheduler="ddim",
        )
        return DiffusionNextTokenHead(embed_dim=128, cfg=cfg)

    def test_diffusion_loss(self, head: DiffusionNextTokenHead) -> None:
        """Test that diffusion loss computes correctly."""
        batch_size = 2
        seq_len = 8
        embed_dim = 128

        cond = torch.randn(batch_size, seq_len, embed_dim)
        target_emb = torch.randn(batch_size, seq_len, embed_dim)

        head.train()
        loss = head.diffusion_loss(cond=cond, target_emb=target_emb)

        assert loss.shape == ()
        assert loss.item() > 0

    def test_diffusion_loss_shape_mismatch(self, head: DiffusionNextTokenHead) -> None:
        """Test that shape mismatch raises error."""
        cond = torch.randn(2, 8, 128)
        target_emb = torch.randn(2, 16, 128)  # Different seq_len

        with pytest.raises(ValueError, match="shape mismatch"):
            head.diffusion_loss(cond=cond, target_emb=target_emb)

    def test_sample_next_logits(self, head: DiffusionNextTokenHead) -> None:
        """Test that sampling produces correct shape logits."""
        batch_size = 2
        embed_dim = 128
        vocab_size = 1000

        cond_last = torch.randn(batch_size, 1, embed_dim)
        tok_emb_weight_t = torch.randn(embed_dim, vocab_size)

        head.eval()
        logits = head.sample_next_logits(
            cond_last=cond_last,
            tok_emb_weight_t=tok_emb_weight_t,
            temperature=1.0,
        )

        assert logits.shape == (batch_size, vocab_size)

    def test_sample_next_logits_wrong_shape(self, head: DiffusionNextTokenHead) -> None:
        """Test that wrong cond_last shape raises error."""
        cond_last = torch.randn(2, 4, 128)  # Not (B, 1, D)
        tok_emb_weight_t = torch.randn(128, 1000)

        with pytest.raises(ValueError, match="must be"):
            head.sample_next_logits(
                cond_last=cond_last,
                tok_emb_weight_t=tok_emb_weight_t,
            )

    def test_cfg_dropout_during_training(self, head: DiffusionNextTokenHead) -> None:
        """Test that CFG dropout is applied during training."""
        batch_size = 100  # Large batch to statistically see dropout
        seq_len = 4
        embed_dim = 128

        cond = torch.ones(batch_size, seq_len, embed_dim)
        head.train()

        # Check that _maybe_drop_cond drops some conditioning
        dropped = head._maybe_drop_cond(cond)

        # Some samples should be zeroed (with p=0.1, expect ~10 zeros)
        zero_samples = (dropped.sum(dim=(1, 2)) == 0).sum()
        assert zero_samples > 0, "Expected some samples to be dropped"
        assert zero_samples < batch_size, "Expected some samples to remain"

    def test_cfg_dropout_not_applied_during_eval(
        self, head: DiffusionNextTokenHead
    ) -> None:
        """Test that CFG dropout is NOT applied during eval."""
        batch_size = 10
        seq_len = 4
        embed_dim = 128

        cond = torch.ones(batch_size, seq_len, embed_dim)
        head.eval()

        dropped = head._maybe_drop_cond(cond)

        # All samples should be preserved during eval
        assert torch.allclose(cond, dropped)


@pytest.mark.skipif(not DIFFUSERS_AVAILABLE, reason="diffusers not installed")
class TestDiffusionHeadSchedulers:
    """Test different scheduler options."""

    @pytest.mark.parametrize("scheduler", ["ddpm", "ddim", "dpm"])
    def test_scheduler_creation(self, scheduler: str) -> None:
        """Test that all supported schedulers can be created."""
        cfg = DiffusionHeadConfig(
            enabled=True,
            num_train_timesteps=100,
            scheduler=scheduler,
        )
        head = DiffusionNextTokenHead(embed_dim=64, cfg=cfg)
        assert head._sched is not None


class TestDiffusionHeadConfig:
    """Tests for diffusion head config."""

    def test_default_values(self) -> None:
        """Test that default config values are sensible."""
        cfg = DiffusionHeadConfig()
        assert cfg.enabled is False
        assert cfg.num_train_timesteps == 1000
        assert cfg.num_infer_steps == 12
        assert cfg.scheduler == "ddim"
        assert 0 < cfg.cfg_dropout_p < 1
        assert cfg.cfg_guidance_scale >= 1.0

    def test_config_is_frozen(self) -> None:
        """Test that config is immutable."""
        cfg = DiffusionHeadConfig()
        with pytest.raises(Exception):  # FrozenInstanceError
            cfg.enabled = True  # type: ignore[misc]
