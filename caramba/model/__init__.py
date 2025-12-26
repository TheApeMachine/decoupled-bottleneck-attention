"""
caramba.model contains model components.
"""

from __future__ import annotations

from torch import Tensor, nn
from typing_extensions import override

from caramba.model.embedder import Embedder
from caramba.config.model import ModelConfig
from caramba.config.diffusion import DiffusionHeadConfig
from caramba.layer.diffusion_head import (
    DiffusionNextTokenHead,
    DIFFUSERS_AVAILABLE,
    DiffusionHeadConfig as RuntimeDiffusionConfig,
)


class Model(nn.Module):
    """
    Model composes the embedder, network topology, and optional diffusion head.

    The diffusion head is a lightweight adapter that learns to denoise target
    token embeddings conditioned on transformer features. It can be trained
    while the backbone is frozen.
    """

    def __init__(self, config: ModelConfig) -> None:
        """
        Initialize the model.

        Args:
            config: Model configuration including embedder, topology, and
                    optional diffusion head settings.
        """
        super().__init__()
        self.config: ModelConfig = config
        self.embedder: Embedder = Embedder(config.embedder)
        self.topology: nn.Module = config.topology.build()

        # Optional diffusion head
        self.diffusion_head: DiffusionNextTokenHead | None = None
        if config.diffusion_head.enabled:
            if not DIFFUSERS_AVAILABLE:
                raise RuntimeError(
                    "diffusion_head.enabled=True but `diffusers` is not installed. "
                    "Install with: pip install diffusers"
                )
            # Convert pydantic config to runtime dataclass
            runtime_cfg_obj = config.diffusion_head.to_runtime_config()
            # Cast to the expected type
            from typing import cast
            runtime_cfg = cast(RuntimeDiffusionConfig, runtime_cfg_obj)
            # Get embed_dim from embedder config
            embed_dim = self._get_embed_dim()
            self.diffusion_head = DiffusionNextTokenHead(
                embed_dim=embed_dim,
                cfg=runtime_cfg,
            )

    def _get_embed_dim(self) -> int:
        """Extract embedding dimension from embedder config."""
        from caramba.config.embedder import TokenEmbedderConfig
        if isinstance(self.config.embedder, TokenEmbedderConfig):
            return self.config.embedder.d_model
        raise ValueError(
            "Cannot determine embed_dim for diffusion head: "
            "embedder must be TokenEmbedderConfig"
        )

    @override
    def forward(
        self,
        x: Tensor,
        *,
        return_features: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """
        Forward pass through embedder and network.

        Args:
            x: Input tensor (token ids if using token embedder)
            return_features: If True, return (features, logits) instead of just logits.
                           Features are the pre-logit hidden states useful for
                           diffusion head training.

        Returns:
            If return_features=False: logits (B, T, vocab_size)
            If return_features=True: (features, logits) where features is (B, T, d_model)
        """
        x = self.embedder(x)
        features = self.topology(x)

        if return_features:
            # Compute logits from features (tied embeddings)
            logits = self._features_to_logits(features)
            return features, logits

        return features

    def _features_to_logits(self, features: Tensor) -> Tensor:
        """Convert hidden features to vocabulary logits via tied embeddings.

        Args:
            features: Hidden states (B, T, d_model)

        Returns:
            Logits (B, T, vocab_size)
        """
        if self.embedder.token_embedding is None:
            # No tied embedding - features are already logits
            return features
        # Tied embedding: logits = features @ embedding.weight.T
        return features @ self.embedder.token_embedding.weight.t()

    def diffusion_loss(
        self,
        features: Tensor,
        target_ids: Tensor,
    ) -> Tensor:
        """Compute diffusion head loss.

        Args:
            features: Transformer features (B, T, d_model)
            target_ids: Target token ids (B, T)

        Returns:
            Scalar MSE loss

        Raises:
            RuntimeError: If diffusion head is not enabled
        """
        if self.diffusion_head is None:
            raise RuntimeError("diffusion_loss called but diffusion_head is not enabled")
        if self.embedder.token_embedding is None:
            raise RuntimeError("diffusion_loss requires token embeddings")

        # Get target embeddings
        target_emb = self.embedder.token_embedding(target_ids)
        return self.diffusion_head.diffusion_loss(cond=features, target_emb=target_emb)

    def sample_with_diffusion(
        self,
        features_last: Tensor,
        *,
        temperature: float = 1.0,
        guidance_scale: float | None = None,
    ) -> Tensor:
        """Sample next token logits using the diffusion head.

        Args:
            features_last: Features at the last position (B, 1, d_model)
            temperature: Sampling temperature
            guidance_scale: CFG scale (None = use config default)

        Returns:
            Logits (B, vocab_size)
        """
        if self.diffusion_head is None:
            raise RuntimeError("sample_with_diffusion called but diffusion_head is not enabled")
        if self.embedder.token_embedding is None:
            raise RuntimeError("sample_with_diffusion requires token embeddings")

        return self.diffusion_head.sample_next_logits(
            cond_last=features_last,
            tok_emb_weight_t=self.embedder.token_embedding.weight.t(),
            temperature=temperature,
            guidance_scale=guidance_scale,
        )