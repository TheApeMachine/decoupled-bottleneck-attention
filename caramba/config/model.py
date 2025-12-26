"""Model configuration: the complete specification for a model.

A model config ties together:
- Embedder: how to convert tokens to vectors
- Topology: the layer structure
- Optional diffusion head for hybrid generation
"""
from __future__ import annotations

import enum

from pydantic import Field

from caramba.config import Config
from caramba.config.diffusion import DiffusionHeadConfig
from caramba.config.embedder import EmbedderConfig, NoEmbedderConfig
from caramba.config.topology import TopologyConfig


class ModelType(str, enum.Enum):
    """Type of model architecture."""

    TRANSFORMER = "TransformerModel"
    GPT = "GPTModel"
    VIT = "ViTModel"
    MLP = "MLPModel"

    @classmethod
    def from_str(cls, s: str) -> "ModelType":
        """Convert a string to a ModelType."""
        return cls(s)

    @staticmethod
    def module_name() -> str:
        """Return the Python module containing model implementations."""
        return "caramba.model"


class ModelConfig(Config):
    """Complete specification for a model architecture.

    Combines embedder, topology, and optional diffusion head into a
    single config that can build the full model.
    """

    type: ModelType
    embedder: EmbedderConfig = Field(default_factory=NoEmbedderConfig)
    topology: TopologyConfig
    diffusion_head: DiffusionHeadConfig = Field(default_factory=DiffusionHeadConfig)
    tied_embeddings: bool = True
