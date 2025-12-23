"""
model provides the model configuration.
"""
from __future__ import annotations

import enum
from pydantic import BaseModel, Field
from caramba.config.embedder import EmbedderConfig, NoEmbedderConfig
from caramba.config.topology import TopologyConfig


class ModelType(str, enum.Enum):
    """
    ModelType provides the model type.
    """
    TRANSFORMER = "transformer"
    GPT = "gpt"
    VIT = "vit"
    MLP = "mlp"

class ModelConfig(BaseModel):
    """
    ModelConfig provides the model configuration.
    """
    type: ModelType
    embedder: EmbedderConfig = Field(default_factory=NoEmbedderConfig)
    topology: TopologyConfig
