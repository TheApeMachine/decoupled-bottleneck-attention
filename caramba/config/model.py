"""
model provides the model configuration.
"""
from __future__ import annotations

import enum
from pydantic import BaseModel, Field
from caramba.config import Config
from caramba.config.embedder import EmbedderConfig, NoEmbedderConfig
from caramba.config.topology import TopologyConfig
from caramba.config.diffusion import DiffusionHeadConfig


class ModelType(str, enum.Enum):
    """
    ModelType provides the model type.
    """
    TRANSFORMER = "TransformerModel"
    GPT = "GPTModel"
    VIT = "ViTModel"
    MLP = "MLPModel"

    @classmethod
    def from_str(cls, s: str) -> ModelType:
        """Converts a string to a ModelType."""
        return cls(s)

    @staticmethod
    def module_name() -> str:
        """Returns the module name for the model type."""
        return "caramba.model"


class ModelConfig(Config):
    """
    ModelConfig provides the model configuration.

    Attributes:
        type: The model type (transformer, gpt, etc.)
        embedder: Embedding layer configuration
        topology: Network topology configuration
        diffusion_head: Optional diffusion next-token head configuration
    """
    type: ModelType
    embedder: EmbedderConfig = Field(default_factory=NoEmbedderConfig)
    topology: TopologyConfig
    diffusion_head: DiffusionHeadConfig = Field(default_factory=DiffusionHeadConfig)